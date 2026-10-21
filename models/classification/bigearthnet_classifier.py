import os
import threading
import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.models.base import BaseModel

logger = logging.getLogger("satquery.models.bigearthnet")

# -------------------------------------------------------------------------
# Environment Variables & Configurable Defaults
# -------------------------------------------------------------------------
DEFAULT_MODEL_ID = os.getenv("BIGEARTHNET_MODEL_ID", "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0")
DEFAULT_REVISION = os.getenv("BIGEARTHNET_MODEL_REVISION", "main")
DEFAULT_CACHE_DIR = os.getenv("BIGEARTHNET_CACHE_DIR", None)
DEFAULT_DEVICE = os.getenv("BIGEARTHNET_DEVICE", None)
DEFAULT_THRESHOLD = float(os.getenv("BIGEARTHNET_THRESHOLD", "0.5"))
DEFAULT_CHECKPOINT_PATH = os.getenv("BIGEARTHNET_CHECKPOINT_PATH", None)

# -------------------------------------------------------------------------
# Band Definitions & Orderings
# -------------------------------------------------------------------------
# Official ESA Sentinel-2 technical documentation order (conformed to by v0.2.0):
V020_BAND_NAMES: List[str] = [
    "B02",  # Blue (10m)
    "B03",  # Green (10m)
    "B04",  # Red (10m)
    "B05",  # Vegetation Red Edge 1 (20m)
    "B06",  # Vegetation Red Edge 2 (20m)
    "B07",  # Vegetation Red Edge 3 (20m)
    "B08",  # NIR (10m)
    "B8A",  # Narrow NIR (20m)
    "B11",  # SWIR-1 (20m)
    "B12",  # SWIR-2 (20m)
]

# Non-standard 10m-grouped order (used in earlier v0.1.1 release):
V011_BAND_NAMES: List[str] = [
    "B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A"
]

# Default export for backwards compatibility
BAND_NAMES = V020_BAND_NAMES

BAND_DESCRIPTIONS: Dict[str, str] = {
    "B02": "Blue (10m, 490 nm)",
    "B03": "Green (10m, 560 nm)",
    "B04": "Red (10m, 665 nm)",
    "B05": "Vegetation Red Edge 1 (20m, 705 nm)",
    "B06": "Vegetation Red Edge 2 (20m, 740 nm)",
    "B07": "Vegetation Red Edge 3 (20m, 783 nm)",
    "B08": "NIR (10m, 842 nm)",
    "B8A": "Narrow NIR (20m, 865 nm)",
    "B11": "SWIR-1 (20m, 1610 nm)",
    "B12": "SWIR-2 (20m, 2190 nm)",
}

# -------------------------------------------------------------------------
# Channel Normalization Statistics
# Computed across official training split at 120x120 resolution (reBEN benchmark)
# -------------------------------------------------------------------------
# For standard ESA band order (v0.2.0):
V020_CHANNEL_MEANS: np.ndarray = np.array(
    [438.3721, 614.0557, 588.4096, 942.8433, 1769.9316, 2049.5515, 2193.2920, 2235.5566, 1568.2268, 997.7325],
    dtype=np.float32,
)

V020_CHANNEL_STDS: np.ndarray = np.array(
    [607.0269, 603.2968, 684.5688, 738.4327, 1100.4561, 1275.8054, 1369.3717, 1356.5441, 1070.1613, 813.5276],
    dtype=np.float32,
)

# For v0.1.1 (permuted to [B02, B03, B04, B08, B05, B06, B07, B11, B12, B8A]):
_V011_PERM = [0, 1, 2, 6, 3, 4, 5, 8, 9, 7]
V011_CHANNEL_MEANS: np.ndarray = V020_CHANNEL_MEANS[_V011_PERM]
V011_CHANNEL_STDS: np.ndarray = V020_CHANNEL_STDS[_V011_PERM]

# Defaults for module exports
CHANNEL_MEANS = V020_CHANNEL_MEANS
CHANNEL_STDS = V020_CHANNEL_STDS

# -------------------------------------------------------------------------
# CORINE Land Cover (CLC) 19-Class Nomenclature
# Lexicographically sorted, matching exact 19 multi-label output logits
# -------------------------------------------------------------------------
BIGEARTHNET_19_CLASSES: List[str] = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]

TARGET_SPATIAL_SIZE = (120, 120)


def build_resnet50_s2_architecture() -> nn.Module:
    """
    Constructs the exact ResNet-50 architecture matching official BigEarthNetv2 checkpoints:
    - 10-channel Conv2d input layer (kernel=7, stride=2, padding=3, bias=False)
    - ResNet-50 convolutional backbone (Bottleneck layers 1-4)
    - 19-class linear head (Linear(2048, 19))
    """
    import torchvision.models as tv_models

    model = tv_models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, 19)
    return model


class BigEarthNetResNet50Classifier(BaseModel):
    """
    Specialist Land-Cover Classification Model:
    BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0 (and compatible with v0.1.1)

    This model is a ResNet-50 multi-label image classifier pre-trained on the refined
    BigEarthNet v2.0 (reBEN) dataset using 10 Sentinel-2 spectral bands.

    Key Specifications:
    - Task: Multi-label land-cover classification (19 CORINE Land Cover classes).
    - Architecture: ResNet-50 backbone with 10-channel conv1 and 19-class output linear layer.
    - Required input bands: 10 Sentinel-2 bands in ESA/reBEN order (or mapped via metadata).
    - Input spatial dimensions: 120 x 120 pixels (bilinearly resized if different).
    - Normalization: Per-channel standardization using official reBEN dataset statistics.
    - Output activation: Sigmoid multi-label probabilities in range [0.0, 1.0].
    - Scientific integrity: Enforces strictly verified weights; rejects visual RGB screenshots.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        threshold: Optional[float] = None,
    ):
        resolved_model_id = model_id or DEFAULT_MODEL_ID
        is_v011 = "v0.1.1" in resolved_model_id
        version_str = "0.1.1" if is_v011 else "0.2.0"

        super().__init__(
            model_name=resolved_model_id,
            task_type="land_cover_classification",
            implementation_status="pretrained_model",
            is_trained=False,
            is_remote_sensing_adapted=True,
            license="MIT",
            version=version_str,
        )
        self.model_id = resolved_model_id
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        self.revision = revision or DEFAULT_REVISION
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.default_threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        self._device = device or DEFAULT_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()

        # Resolve band names and normalization statistics based on checkpoint version
        if is_v011:
            self.band_names = V011_BAND_NAMES
            self.channel_means = V011_CHANNEL_MEANS
            self.channel_stds = V011_CHANNEL_STDS
        else:
            self.band_names = V020_BAND_NAMES
            self.channel_means = V020_CHANNEL_MEANS
            self.channel_stds = V020_CHANNEL_STDS

    @property
    def status(self) -> str:
        if self._is_loaded:
            return "loaded"
        if self._load_error or self.implementation_status == "model_loading_error":
            return "model_loading_error"
        return "unavailable"

    def _resolve_checkpoint_path(self) -> Optional[str]:
        """
        Resolves the checkpoint path using the following hierarchy:
        1. Explicitly passed instance checkpoint_path
        2. Dynamic BIGEARTHNET_CHECKPOINT_PATH environment variable (stripped of quotes/whitespace)
        3. Module default DEFAULT_CHECKPOINT_PATH
        4. Local Hugging Face Hub cache without network calls (local_files_only=True)
        5. Direct inspection of default ~/.cache/huggingface/hub snapshots directory
        """
        raw_path = self.checkpoint_path or os.getenv("BIGEARTHNET_CHECKPOINT_PATH") or DEFAULT_CHECKPOINT_PATH
        if raw_path:
            cleaned = str(raw_path).strip().strip("'\"")
            return cleaned

        # Try Hugging Face Hub cache locally without network calls
        try:
            from huggingface_hub import hf_hub_download
            local_cached = hf_hub_download(
                repo_id=self.model_id,
                filename="model.safetensors",
                revision=self.revision,
                cache_dir=self.cache_dir,
                local_files_only=True,
            )
            if local_cached and os.path.isfile(local_cached):
                logger.info(f"Discovered local HuggingFace cache weights for '{self.model_id}' at: {local_cached}")
                return local_cached
        except Exception:
            pass

        # Direct scan of standard Hugging Face Hub snapshots directory
        try:
            hf_cache_home = os.getenv("HF_HOME") or os.path.expanduser("~/.cache/huggingface/hub")
            model_dir_name = "models--" + self.model_id.replace("/", "--")
            snapshots_dir = os.path.join(hf_cache_home, model_dir_name, "snapshots")
            if os.path.isdir(snapshots_dir):
                for snap in sorted(os.listdir(snapshots_dir), reverse=True):
                    candidate = os.path.join(snapshots_dir, snap, "model.safetensors")
                    if os.path.isfile(candidate):
                        logger.info(f"Discovered HuggingFace snapshot weights for '{self.model_id}' at: {candidate}")
                        return candidate
        except Exception:
            pass

        return None

    def load(self, force_reload: bool = False) -> None:
        """
        Loads official pretrained weights into the ResNet-50 architecture.
        Attempts to resolve weights from:
        1. Explicitly configured checkpoint_path (or dynamic BIGEARTHNET_CHECKPOINT_PATH)
        2. Local Hugging Face Hub cache
        3. Hugging Face Hub download from 'BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0' (or v0.1.1)

        Thread-safe and idempotent: repeated concurrent calls safely reuse loaded weights.
        Never silently falls back to random weights or mock predictions.
        """
        with self._lock:
            if self._is_loaded and not force_reload:
                return

            logger.info(f"Loading BigEarthNet ResNet-50 model '{self.model_id}' (rev: {self.revision}) on device '{self._device}'...")
            self._load_error = None
            self._is_loaded = False

            try:
                # 1. Resolve weight file path
                resolved_weight_path = self._resolve_checkpoint_path()

                if resolved_weight_path:
                    if not os.path.isfile(resolved_weight_path):
                        err_msg = (
                            f"Specified checkpoint file for '{self.model_id}' was not found at '{resolved_weight_path}'. "
                            "SatQuery AI enforces scientific integrity and will never silently substitute random weights."
                        )
                        self._load_error = err_msg
                        self.implementation_status = "model_loading_error"
                        logger.error(err_msg)
                        raise FileNotFoundError(err_msg)
                else:
                    try:
                        from huggingface_hub import hf_hub_download
                        resolved_weight_path = hf_hub_download(
                            repo_id=self.model_id,
                            filename="model.safetensors",
                            revision=self.revision,
                            cache_dir=self.cache_dir,
                        )
                    except Exception as dl_err:
                        err_msg = (
                            f"Failed to download or locate model weights for '{self.model_id}': {dl_err}. "
                            "SatQuery AI enforces scientific integrity and will never silently substitute "
                            "random weights or fabricate classification predictions."
                        )
                        self._load_error = err_msg
                        self.implementation_status = "model_loading_error"
                        logger.error(err_msg)
                        raise RuntimeError(err_msg) from dl_err

                if not resolved_weight_path or not os.path.isfile(resolved_weight_path):
                    err_msg = (
                        f"Model weights file for '{self.model_id}' was not found at '{resolved_weight_path}'. "
                        "SatQuery AI will not run uninitialized random weights."
                    )
                    self._load_error = err_msg
                    self.implementation_status = "model_loading_error"
                    logger.error(err_msg)
                    raise FileNotFoundError(err_msg)

                # 2. Build exact architecture
                model = build_resnet50_s2_architecture()

                # 3. Load state dict from safetensors or PyTorch checkpoint
                if resolved_weight_path.endswith(".safetensors"):
                    from safetensors import safe_open

                    state_dict = {}
                    with safe_open(resolved_weight_path, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            clean_key = k.replace("model.vision_encoder.", "").replace("vision_encoder.", "")
                            state_dict[clean_key] = f.get_tensor(k)
                else:
                    raw_state = torch.load(resolved_weight_path, map_location="cpu")
                    if "state_dict" in raw_state:
                        raw_state = raw_state["state_dict"]
                    state_dict = {
                        k.replace("model.vision_encoder.", "").replace("vision_encoder.", ""): v
                        for k, v in raw_state.items()
                    }

                # 4. Strict load into architecture
                try:
                    model.load_state_dict(state_dict, strict=True)
                except RuntimeError as align_err:
                    err_msg = f"Weight alignment mismatch loading '{self.model_id}': {align_err}"
                    self._load_error = err_msg
                    self.implementation_status = "model_loading_error"
                    logger.error(err_msg)
                    raise RuntimeError(err_msg) from align_err

                model.to(self._device)
                model.eval()

                self.model = model
                self._device = str(next(model.parameters()).device)
                self._is_loaded = True
                self.is_trained = True
                self.implementation_status = "pretrained_model"
                self._load_error = None
                logger.info(f"Successfully loaded '{self.model_id}' weights from '{resolved_weight_path}' ({len(state_dict)} tensors) on {self._device}.")

            except Exception as e:
                self._is_loaded = False
                self.is_trained = False
                self.implementation_status = "model_loading_error"
                if not self._load_error:
                    self._load_error = str(e)
                logger.error(f"Error initializing '{self.model_id}': {e}")
                raise

    def preprocess(
        self,
        image: Union[np.ndarray, torch.Tensor],
        band_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Prepares raw satellite imagery for model forward pass:
        1. Formats input into tensor (1, 10, H, W)
        2. Validates that exactly 10 bands are provided (or reorders them if band_names supplied)
        3. Detects unit scale: scales reflectance [0.0, 1.0] to native Sentinel-2 DN scale [0, 10000]
        4. Bilinearly resizes spatial dimensions to (120, 120)
        5. Standardizes each channel using the official reBEN channel means and standard deviations
        """
        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image.copy()).float()
        elif isinstance(image, torch.Tensor):
            tensor = image.clone().float()
        else:
            raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(image)}")

        # Handle dimensions
        if tensor.ndim == 3:
            # Check if channel-first (10, H, W) or channel-last (H, W, 10)
            if tensor.shape[0] == 10:
                tensor = tensor.unsqueeze(0)  # (1, 10, H, W)
            elif tensor.shape[2] == 10:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 10, H, W)
            else:
                channels_found = tensor.shape[2] if tensor.shape[2] in (1, 3, 4) else tensor.shape[0]
                raise ValueError(
                    f"BigEarthNet ResNet-50 requires exactly 10 Sentinel-2 bands {self.band_names}. "
                    f"Received tensor with shape {list(tensor.shape)} ({channels_found} channel(s) detected). "
                    "Rendered RGB browse images or Copernicus Browser screenshots lack the required multispectral bands "
                    "(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12). Please provide raw Sentinel-2 L2A multispectral data (GeoTIFF)."
                )
        elif tensor.ndim == 4:
            if tensor.shape[1] != 10:
                if tensor.shape[3] == 10:
                    tensor = tensor.permute(0, 3, 1, 2)
                else:
                    channels_found = tensor.shape[3] if tensor.shape[3] in (1, 3, 4) else tensor.shape[1]
                    raise ValueError(
                        f"BigEarthNet ResNet-50 requires exactly 10 Sentinel-2 bands {self.band_names}. "
                        f"Received batch with shape {list(tensor.shape)} ({channels_found} channel(s) detected)."
                    )
        else:
            raise ValueError(f"Invalid input tensor dimensionality: {tensor.ndim}. Expected 3 or 4 dimensions.")

        # Reorder bands if custom band_names provided
        if band_names is not None:
            if len(band_names) != 10:
                raise ValueError(
                    f"Provided band_names list must contain exactly 10 bands {self.band_names}. "
                    f"Got {band_names}."
                )
            band_name_upper = [b.upper() for b in band_names]
            for required_band in self.band_names:
                if required_band not in band_name_upper:
                    raise ValueError(
                        f"Missing required Sentinel-2 band '{required_band}'. "
                        f"Required bands: {self.band_names}. Provided: {band_names}."
                    )
            reorder_indices = [band_name_upper.index(b) for b in self.band_names]
            tensor = tensor[:, reorder_indices, :, :]

        # Detect scale: If values are float reflectance in range [0, 2], scale to native Sentinel-2 DN [0, 10000]
        max_val = float(tensor.max())
        if max_val <= 2.0 and max_val > 0.0:
            tensor = tensor * 10000.0

        # Bilinearly resize to target spatial dimensions (120, 120)
        if tensor.shape[2:] != TARGET_SPATIAL_SIZE:
            tensor = F.interpolate(tensor, size=TARGET_SPATIAL_SIZE, mode="bilinear", align_corners=False)

        # Standardize per channel: (x - mean) / std
        means_t = torch.from_numpy(self.channel_means).view(1, 10, 1, 1).to(tensor.device)
        stds_t = torch.from_numpy(self.channel_stds).view(1, 10, 1, 1).to(tensor.device)
        tensor = (tensor - means_t) / stds_t

        return tensor

    def predict(
        self,
        images: List[np.ndarray],
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes multi-label land-cover classification on 10-band Sentinel-2 imagery.

        Returns:
        - answer: comprehensive human-readable summary of detected classes
        - confidence: top class sigmoid probability (uncalibrated)
        - confidence_label: explicit scientific disclosure
        - predicted_classes: list of classes exceeding the probability threshold
        - class_probabilities: dictionary of all 19 CORINE classes with probabilities [0.0, 1.0]
        - top_predictions: ranked list of {label, probability}
        - evidence: structured VisualEvidence artifact item
        - metadata: provenance recording model ID, bands, spatial size, and normalization
        """
        if not self._is_loaded or self.model is None:
            err_msg = (
                f"Model '{self.model_id}' is not loaded. Cannot run inference without authentic weights. "
                f"Call load() first. (Last error: {self._load_error})"
            )
            raise RuntimeError(err_msg)

        if not images:
            raise ValueError("No input image arrays provided for prediction.")

        raw_image = images[0]
        thresh = threshold if threshold is not None else self.default_threshold
        band_names = metadata.get("band_names") if metadata else None

        input_tensor = self.preprocess(raw_image, band_names=band_names).to(self._device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        class_probs = {cls_name: float(probs[i]) for i, cls_name in enumerate(BIGEARTHNET_19_CLASSES)}
        sorted_preds = sorted(class_probs.items(), key=lambda kv: kv[1], reverse=True)

        predicted_classes = [c for c, p in sorted_preds if p >= thresh]

        top_prob = float(sorted_preds[0][1]) if sorted_preds else 0.0
        top_class = sorted_preds[0][0] if sorted_preds else "None"

        # Construct comprehensive answer text
        if predicted_classes:
            classes_str = ", ".join(f"'{c}' ({class_probs[c]:.1%})" for c in predicted_classes)
            answer_text = (
                f"Multi-label land-cover classification using BigEarthNetv2 ({self.model_id.split('/')[-1]}) "
                f"detected {len(predicted_classes)} CORINE Land Cover (CLC) class(es) exceeding decision threshold {thresh:.2f}: "
                f"{classes_str}. Top detected category is '{top_class}' with sigmoid output {top_prob:.2%}."
            )
        else:
            answer_text = (
                f"Multi-label land-cover classification using BigEarthNetv2 ({self.model_id.split('/')[-1]}) "
                f"detected 0 classes exceeding decision threshold {thresh:.2f}. "
                f"Highest category was '{top_class}' with sigmoid output {top_prob:.2%}."
            )

        evidence_artifact = {
            "id": "ev_bigearthnet_classification",
            "type": "classification",
            "title": f"BigEarthNetv2 Land-Cover Classification ({self.model_id.split('/')[-1]})",
            "description": f"CORINE Land Cover (CLC) 19-class multi-label sigmoid probabilities from 10-band Sentinel-2 multispectral data.",
            "statistics": {
                "model_id": self.model_id,
                "predicted_classes": predicted_classes,
                "top_predictions": [{"label": c, "probability": round(float(p), 4)} for c, p in sorted_preds],
                "threshold": thresh,
                "class_probabilities": class_probs,
                "total_classes": 19,
                "bands_used": self.band_names,
            }
        }

        classification_meta = {
            "source": f"https://huggingface.co/{self.model_id}",
            "backbone": "ResNet-50",
            "dataset": "BigEarthNet v2.0 (reBEN)",
            "band_order": self.band_names,
            "normalization": "reBEN official channel means and stds",
            "weights_status": "authentic_official_checkpoint",
            "license": "MIT",
            "scientific_limitations": [
                "Multi-label sigmoid probabilities are uncalibrated and represent relative class presence.",
                "Input patch spatial resolution is 120x120 pixels (~1.2 km x 1.2 km ground extent).",
                "Visual RGB screenshots without 10 multispectral bands are invalid.",
            ],
            **(metadata or {})
        }

        return {
            "model_name": self.model_name,
            "primary_model": self.model_name,
            "actual_model_used": self.model_name,
            "fallback_used": False,
            "model_status": self.status,
            "task": self.task_type,
            "answer": answer_text,
            "confidence": round(top_prob, 4),
            "confidence_label": "Multi-label Sigmoid Output (Uncalibrated)",
            "implementation_status": self.implementation_status,
            "is_remote_sensing_adapted": True,
            "bands_used": self.band_names,
            "input_shape": list(input_tensor.shape),
            "target_spatial_size": list(TARGET_SPATIAL_SIZE),
            "threshold": thresh,
            "predicted_classes": predicted_classes,
            "class_probabilities": class_probs,
            "top_predictions": [{"label": c, "probability": round(float(p), 6)} for c, p in sorted_preds],
            "evidence": [evidence_artifact],
            "models": [self.model_name],
            "metadata": classification_meta,
        }
