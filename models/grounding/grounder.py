import logging
import os
import re
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image

from app.models.base import BaseModel
from app.remote_sensing.preprocessing import (
    convert_array_to_base64_png, 
    normalize_to_uint8,
    robust_remote_sensing_preprocess
)

logger = logging.getLogger("satquery.grounding")

def extract_grounding_targets(query: str) -> List[str]:
    """
    Extracts open-vocabulary grounding target phrases from user text query.
    Normalizes remote sensing entities into representative query tokens for OWL-ViT.
    """
    q = (query or "").lower().strip()
    
    # Priority semantic maps for remote-sensing domain targets
    domain_map = [
        (["water", "river", "lake", "ocean", "pond", "reservoir"], ["water body", "river", "lake"]),
        (["building", "built-up", "urban", "house", "structure", "settlement"], ["buildings", "roof", "urban structure"]),
        (["ship", "boat", "vessel"], ["ships", "boat", "vessel"]),
        (["road", "highway", "runway", "pavement"], ["road", "highway", "runway"]),
        (["vegetation", "crop", "forest", "tree", "plant", "agriculture"], ["vegetation", "forest", "field"]),
        (["vehicle", "car", "truck"], ["car", "vehicle", "truck"]),
        (["bridge"], ["bridge", "overpass"]),
        (["airport", "airplane", "aircraft"], ["airplane", "aircraft", "runway"])
    ]
    
    for keywords, targets in domain_map:
        if any(kw in q for kw in keywords):
            return targets

    # Generic extraction: remove prompt action verbs and prefixes
    cleaned = re.sub(
        r"^(find|locate|highlight|show|detect|where is|where are|outline|box|segment|identify)\s+(all\s+|the\s+|any\s+)?",
        "",
        q
    ).strip()
    cleaned = re.sub(r"[?.!]", "", cleaned).strip()
    if cleaned:
        return [cleaned]
    return ["object of interest"]


class BaseGrounder(BaseModel):
    """Abstract base class for visual grounding models."""
    def __init__(self, model_name: str, implementation_status: str = "baseline"):
        super().__init__(
            model_name=model_name,
            task_type="grounding",
            implementation_status=implementation_status,
            is_trained=False,
            is_remote_sensing_adapted=False
        )


class ClassicalBaselineGrounder(BaseGrounder):
    """
    Classical spectral thresholding baseline for remote sensing feature localization.
    Transparently labeled: NOT a deep neural vision-language grounding model.
    Extracts candidate regions via spectral heuristics and OpenCV / SciPy connected components.
    Preserved as the verified deterministic fallback.
    """
    def __init__(self):
        super().__init__(
            model_name="ClassicalBaselineGrounder",
            implementation_status="baseline"
        )

    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not images:
            raise ValueError("Grounding task requires an input satellite image.")

        img = images[0]
        meta = metadata or {}
        q = (query or "water body").lower().strip()
        h, w = img.shape[:2]
        img_uint8, meta_proc = robust_remote_sensing_preprocess(img, meta)

        # Convert to grayscale for thresholding
        if img_uint8.ndim == 3 and img_uint8.shape[2] >= 3:
            gray = (0.299 * img_uint8[:, :, 0] + 0.587 * img_uint8[:, :, 1] + 0.114 * img_uint8[:, :, 2]).astype(np.uint8)
        else:
            gray = img_uint8 if img_uint8.ndim == 2 else img_uint8[:, :, 0]

        boxes = []
        mask = np.zeros((h, w), dtype=np.uint8)

        if any(w_kw in q for w_kw in ["water", "river", "lake", "ocean", "pond"]):
            mask = np.where(gray < 70, 255, 0).astype(np.uint8)
            label = "Low Reflectance / Water Candidate"
        elif any(v_kw in q for v_kw in ["vegetation", "crop", "forest", "tree", "plant", "agriculture"]):
            if img_uint8.ndim == 3 and img_uint8.shape[2] >= 3:
                g = img_uint8[:, :, 1].astype(float)
                r = img_uint8[:, :, 0].astype(float)
                mask = np.where((g - r) > 8, 255, 0).astype(np.uint8)
            else:
                mask = np.where(gray > 120, 255, 0).astype(np.uint8)
            label = "High Green Band Differential / Vegetation"
        else:
            mask = np.where((gray > 80) & (gray < 220), 255, 0).astype(np.uint8)
            label = "Mid-Tone Texture / Structure Candidate"

        # Extract bounding boxes from connected components / contours
        try:
            import cv2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > (h * w * 0.005):
                    x, y, bw, bh = cv2.boundingRect(c)
                    boxes.append({
                        "xmin": float(x),
                        "ymin": float(y),
                        "xmax": float(x + bw),
                        "ymax": float(y + bh),
                        "label": label,
                        "confidence": None
                    })
        except ImportError:
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(mask)
            slices = ndimage.find_objects(labeled_array)
            for sl in slices:
                y_slice, x_slice = sl
                area = (y_slice.stop - y_slice.start) * (x_slice.stop - x_slice.start)
                if area > (h * w * 0.005):
                    boxes.append({
                        "xmin": float(x_slice.start),
                        "ymin": float(y_slice.start),
                        "xmax": float(x_slice.stop),
                        "ymax": float(y_slice.stop),
                        "label": label,
                        "confidence": None
                    })

        # Render overlay image
        overlay = img_uint8.copy()
        if overlay.ndim == 2:
            overlay = np.stack([overlay]*3, axis=-1)
        elif overlay.shape[2] > 3:
            overlay = overlay[:, :, :3]
        
        mask_overlay = overlay.copy()
        mask_overlay[:, :, 0] = np.clip(mask_overlay[:, :, 0].astype(int) + mask.astype(int), 0, 255)
        overlay = (0.7 * overlay + 0.3 * mask_overlay).astype(np.uint8)

        overlay_b64 = convert_array_to_base64_png(overlay)
        mask_b64 = convert_array_to_base64_png(mask)
        orig_b64 = convert_array_to_base64_png(img_uint8)

        grounded_pixels = int(np.count_nonzero(mask))
        grounded_pct = (grounded_pixels / (h * w)) * 100.0 if (h * w) > 0 else 0.0

        answer = (
            f"Classical heuristic baseline localized {len(boxes)} candidate region(s) "
            f"for query criteria '{q}'. (Method: spectral threshold + morphological contouring). "
            f"Candidate coverage: {grounded_pct:.2f}% of scene area ({grounded_pixels:,} pixels)."
        )

        evidence = [
            {
                "id": "ev_grounding_overlay",
                "type": "overlay",
                "title": f"Heuristic Candidate Overlay — {label}",
                "description": f"Threshold-derived spatial region markers for target: '{q}'",
                "data_base64": overlay_b64,
                "boxes": boxes
            },
            {
                "id": "ev_grounding_mask",
                "type": "mask",
                "title": f"Spectral Threshold Mask — {label}",
                "description": "Binary pixel mask generated via classical spectral thresholding baseline.",
                "data_base64": mask_b64,
                "statistics": {
                    "grounded_pixels": grounded_pixels,
                    "total_pixels": h * w,
                    "grounded_area_percent": f"{grounded_pct:.2f}%",
                    "detection_method": "classical_spectral_thresholding"
                }
            }
        ]

        return {
            "answer": answer,
            "boxes": boxes,
            "model_type": "classical_baseline",
            "model_name": self.model_name,
            "primary_model": self.model_name,
            "actual_model_used": self.model_name,
            "fallback_used": False,
            "model_status": "baseline",
            "implementation_status": "baseline",
            "confidence": None,
            "confidence_label": "Not available (Classical Threshold Baseline)",
            "models": [self.model_name],
            "evidence": evidence,
            "model_provenance": {
                "primary_model": self.model_name,
                "actual_model_used": self.model_name,
                "checkpoint_status": "baseline",
                "fallback_used": False,
                "device": "cpu",
                "preprocessing": "Classical spectral band thresholding + contour extraction",
                "target_query": q
            }
        }


class OWLViTGroundingProvider(BaseGrounder):
    """
    Open-Vocabulary Visual Grounding Provider powered by google/owlvit-base-patch32.
    Accepts arbitrary natural-language target prompts and localizes bounding boxes.
    Pretrained General VLM: NOT remote-sensing fine-tuned (honestly declared).
    Lazy-loaded: initializes model only upon inference request when checkpoint is available.
    """
    @staticmethod
    def _find_checkpoint_path(checkpoint_path: Optional[str] = None) -> str:
        """
        Dynamically resolves the OWL-ViT checkpoint directory path.
        Supports:
        1. Explicitly supplied checkpoint path (takes top priority, even if pointing to a test/missing path)
        2. GROUNDING_MODEL_CHECKPOINT_PATH / RS_GROUNDING_CHECKPOINT_PATH (if set in environment)
        3. Repository-root models/weights/owlvit_base (regardless of current working directory)
        """
        # 1. Explicitly supplied checkpoint path has top priority
        if checkpoint_path is not None:
            return checkpoint_path

        # 2. Environment variable has next priority
        env_path = os.getenv("GROUNDING_MODEL_CHECKPOINT_PATH") or os.getenv("RS_GROUNDING_CHECKPOINT_PATH")
        if env_path is not None:
            return env_path

        # 3. Dynamic repository-root candidate lookup anchored from this file
        current_file = os.path.abspath(__file__)
        repo_root = os.path.abspath(os.path.join(current_file, "..", "..", "..", "..", ".."))

        candidates = [
            os.path.join(repo_root, "models", "weights", "owlvit_base"),
            os.path.join(os.getcwd(), "models", "weights", "owlvit_base"),
            os.path.join(os.getcwd(), "..", "models", "weights", "owlvit_base"),
            "models/weights/owlvit_base"
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return os.path.abspath(c)

        return os.path.join(repo_root, "models", "weights", "owlvit_base")

    def __init__(
        self,
        model_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None
    ):
        model_id = model_name or os.getenv("GROUNDING_MODEL_ID", "google/owlvit-base-patch32")
        super().__init__(
            model_name=model_id,
            implementation_status="unavailable"
        )
        self.checkpoint_path = self._find_checkpoint_path(checkpoint_path)
        self.configured_device = device or os.getenv("GROUNDING_MODEL_DEVICE", "cpu")
        self.configured_dtype = dtype or os.getenv("GROUNDING_MODEL_DTYPE", "float32")
        self._processor = None
        self._model = None
        self._is_loaded = False
        self.is_remote_sensing_adapted = False
        self.adaptation_status = "pretrained_general_grounding"

    @property
    def is_available(self) -> bool:
        """Returns True if and only if local checkpoint directory exists on disk."""
        return bool(self.checkpoint_path and os.path.exists(self.checkpoint_path))

    def _resolve_device(self):
        import torch
        if self.configured_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.configured_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for OWL-ViT but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(self.configured_device)

    def load(self) -> bool:
        """
        Lazily loads OWL-ViT processor and model into memory.
        Validates checkpoint existence; never downloads weights during request execution.
        """
        if self._is_loaded:
            return True

        if not self.is_available:
            logger.info(
                f"OWL-ViT grounding checkpoint not found at '{self.checkpoint_path}'. "
                "Implementation status remains 'unavailable' (fallback active)."
            )
            self._is_loaded = False
            self.implementation_status = "unavailable"
            return False

        try:
            import torch
            from transformers import OwlViTProcessor, OwlViTForObjectDetection

            device = self._resolve_device()
            logger.info(f"Loading OWL-ViT from checkpoint '{self.checkpoint_path}' onto {device}...")

            torch_dtype = torch.float16 if self.configured_dtype == "float16" and device.type == "cuda" else torch.float32

            self._processor = OwlViTProcessor.from_pretrained(self.checkpoint_path)
            self._model = OwlViTForObjectDetection.from_pretrained(
                self.checkpoint_path,
                torch_dtype=torch_dtype
            )
            self._model.to(device)
            self._model.eval()

            self._is_loaded = True
            self.implementation_status = "real_model"
            self.is_remote_sensing_adapted = False
            logger.info("OWL-ViT visual grounding model successfully loaded.")
            return True

        except Exception as e:
            logger.error(f"Failed to load OWL-ViT checkpoint from '{self.checkpoint_path}': {e}")
            self._processor = None
            self._model = None
            self._is_loaded = False
            self.implementation_status = "unavailable"
            return False

    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes genuine OWL-ViT open-vocabulary visual grounding inference.
        """
        if not images:
            raise ValueError("Grounding task requires an input satellite image.")

        if not self._is_loaded and not self.load():
            raise RuntimeError(
                f"OWL-ViT neural checkpoint unavailable at '{self.checkpoint_path}'. "
                "Fallback to ClassicalBaselineGrounder."
            )

        import torch
        img = images[0]
        meta = metadata or {}
        raw_query = query or "water body"
        targets = extract_grounding_targets(raw_query)

        h, w = img.shape[:2]
        img_uint8, meta_proc = robust_remote_sensing_preprocess(img, meta)

        # Convert to 3-channel RGB PIL Image
        if img_uint8.ndim == 2:
            rgb_arr = np.stack([img_uint8] * 3, axis=-1)
            band_mapping = [1, 1, 1]
        elif img_uint8.shape[2] == 1:
            rgb_arr = np.repeat(img_uint8, 3, axis=-1)
            band_mapping = [1, 1, 1]
        elif img_uint8.shape[2] >= 3:
            rgb_arr = img_uint8[:, :, :3]
            band_mapping = [1, 2, 3]
        else:
            rgb_arr = np.pad(img_uint8, ((0, 0), (0, 0), (0, 3 - img_uint8.shape[2])), mode="edge")
            band_mapping = [1, img_uint8.shape[2], img_uint8.shape[2]]

        pil_image = Image.fromarray(rgb_arr)
        device = next(self._model.parameters()).device

        # Tokenize image and text query for OWL-ViT
        inputs = self._processor(text=[targets], images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Neural forward pass
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process bounding boxes with detection threshold
        target_sizes = torch.tensor([[h, w]], device=device)
        threshold = float(os.getenv("GROUNDING_DETECTION_THRESHOLD", "0.10"))
        if hasattr(self._processor, "post_process_grounded_object_detection"):
            results = self._processor.post_process_grounded_object_detection(
                outputs=outputs,
                threshold=threshold,
                target_sizes=target_sizes,
                text_labels=[targets]
            )[0]
        else:
            results = self._processor.post_process_object_detection(
                outputs=outputs,
                threshold=threshold,
                target_sizes=target_sizes
            )[0]

        pred_boxes = results["boxes"].detach().cpu().numpy() if hasattr(results["boxes"], "detach") else np.array(results["boxes"])
        pred_scores = results["scores"].detach().cpu().numpy() if hasattr(results["scores"], "detach") else np.array(results["scores"])
        pred_labels = results["labels"].detach().cpu().numpy() if hasattr(results["labels"], "detach") else np.array(results["labels"])
        text_labels = results.get("text_labels") or []

        boxes = []
        mask = np.zeros((h, w), dtype=np.uint8)

        for idx, (box, score, label_idx) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            xmin, ymin, xmax, ymax = box
            xmin = max(0.0, min(float(xmin), float(w)))
            ymin = max(0.0, min(float(ymin), float(h)))
            xmax = max(0.0, min(float(xmax), float(w)))
            ymax = max(0.0, min(float(ymax), float(h)))

            if xmax > xmin and ymax > ymin:
                if idx < len(text_labels) and text_labels[idx]:
                    target_label = text_labels[idx]
                elif int(label_idx) < len(targets):
                    target_label = targets[int(label_idx)]
                else:
                    target_label = targets[0]
                conf_score = round(float(score), 4)
                boxes.append({
                    "xmin": round(xmin, 1),
                    "ymin": round(ymin, 1),
                    "xmax": round(xmax, 1),
                    "ymax": round(ymax, 1),
                    "label": target_label,
                    "confidence": conf_score
                })
                # Populate mask
                mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 255

        # Render overlay image with bounding box highlights
        overlay = rgb_arr.copy()
        mask_overlay = overlay.copy()
        mask_overlay[:, :, 0] = np.clip(mask_overlay[:, :, 0].astype(int) + mask.astype(int), 0, 255)
        overlay = (0.75 * overlay + 0.25 * mask_overlay).astype(np.uint8)

        overlay_b64 = convert_array_to_base64_png(overlay)
        mask_b64 = convert_array_to_base64_png(mask)
        orig_b64 = convert_array_to_base64_png(rgb_arr)

        grounded_pixels = int(np.count_nonzero(mask))
        grounded_pct = (grounded_pixels / (h * w)) * 100.0 if (h * w) > 0 else 0.0

        primary_target_str = ", ".join(targets)
        answer = (
            f"OWL-ViT open-vocabulary neural detector localized {len(boxes)} candidate region(s) "
            f"for target '{primary_target_str}' across {w}x{h} scene. "
            f"Total grounded coverage: {grounded_pct:.2f}% ({grounded_pixels:,} pixels). "
            f"Note: Model is a general-purpose vision foundation detector (not remote-sensing fine-tuned)."
        )

        evidence = [
            {
                "id": "ev_grounding_overlay",
                "type": "overlay",
                "title": f"OWL-ViT Neural Bounding Box Overlay — {primary_target_str}",
                "description": f"Open-vocabulary localized bounding boxes and labels for: '{raw_query}'",
                "data_base64": overlay_b64,
                "boxes": boxes
            },
            {
                "id": "ev_grounding_mask",
                "type": "mask",
                "title": f"Neural Grounding Mask — {primary_target_str}",
                "description": "Binary localization mask derived from predicted bounding box bounds.",
                "data_base64": mask_b64,
                "statistics": {
                    "grounded_pixels": grounded_pixels,
                    "total_pixels": h * w,
                    "grounded_area_percent": f"{grounded_pct:.2f}%",
                    "detection_count": len(boxes),
                    "model_name": self.model_name,
                    "device": str(device)
                }
            },
            {
                "id": "ev_grounding_original",
                "type": "original",
                "title": "Preprocessed Input Scene",
                "description": f"Preprocessed optical raster ({w}x{h} px, CRS: {meta.get('crs_display', 'CRS unavailable')}).",
                "data_base64": orig_b64,
                "statistics": {
                    "dimensions": f"{w}x{h}",
                    "crs": meta.get("crs_display", "CRS unavailable"),
                    "modality": meta.get("modality", "OPTICAL")
                }
            }
        ]

        return {
            "answer": answer,
            "boxes": boxes,
            "model_type": "neural_model",
            "model_name": self.model_name,
            "primary_model": self.model_name,
            "actual_model_used": self.model_name,
            "fallback_used": False,
            "model_status": "loaded",
            "implementation_status": "real_model",
            "remote_sensing_adapted": False,
            "is_remote_sensing_adapted": False,
            "adaptation_status": "pretrained_general_grounding",
            "confidence": None,
            "confidence_label": "Uncalibrated OWL-ViT detector scores",
            "models": [self.model_name],
            "evidence": evidence,
            "model_provenance": {
                "primary_model": self.model_name,
                "actual_model_used": self.model_name,
                "checkpoint_status": "loaded",
                "fallback_used": False,
                "device": str(device),
                "preprocessing": "Multiband robust min-max normalization + RGB projection",
                "input_band_mapping": band_mapping,
                "target_queries": targets,
                "target_query": raw_query,
                "raw_query": raw_query
            }
        }


class GroundingProvider(BaseModel):
    """
    Visual Grounding Orchestrator.
    Manages open-vocabulary neural grounding (google/owlvit-base-patch32)
    with verified fallback to ClassicalBaselineGrounder.
    Transparently reports primary model, actual model used, fallback status, and provenance.
    """
    def __init__(self):
        super().__init__(
            model_name="GroundingProvider",
            task_type="grounding",
            implementation_status="baseline",
            is_trained=False,
            is_remote_sensing_adapted=False
        )
        self.baseline = ClassicalBaselineGrounder()
        self.deep_grounder = OWLViTGroundingProvider()

    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        primary_name = self.deep_grounder.model_name  # "google/owlvit-base-patch32"

        # Attempt neural execution if checkpoint is present
        if self.deep_grounder.is_available:
            try:
                res = self.deep_grounder.predict(images, query, metadata)
                res["models"] = [self.model_name, self.deep_grounder.model_name]
                return res
            except Exception as e:
                logger.warning(
                    f"Neural grounding detector '{primary_name}' failed during inference ({e}). "
                    "Dispatching to deterministic baseline fallback."
                )

        # Fallback to verified baseline
        logger.info(
            f"Neural grounding checkpoint '{self.deep_grounder.checkpoint_path}' unavailable. "
            f"Using baseline fallback: '{self.baseline.model_name}'."
        )
        res = self.baseline.predict(images, query, metadata)
        res["primary_model"] = primary_name
        res["actual_model_used"] = self.baseline.model_name
        res["fallback_used"] = True
        res["model_status"] = "checkpoint_not_found" if not self.deep_grounder.is_available else "inference_failed"
        res["implementation_status"] = "baseline"
        res["models"] = [self.model_name, primary_name, self.baseline.model_name]
        res["model_provenance"] = {
            "primary_model": primary_name,
            "actual_model_used": self.baseline.model_name,
            "checkpoint_status": "checkpoint_not_found" if not self.deep_grounder.is_available else "inference_failed",
            "fallback_used": True,
            "device": "cpu",
            "preprocessing": "Classical spectral band thresholding + contour extraction",
            "target_query": query
        }
        return res


# Backwards compatibility alias
RemoteSensingGrounder = GroundingProvider
