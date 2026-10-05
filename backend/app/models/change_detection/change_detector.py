import logging
import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod

from app.models.base import BaseModel
from app.remote_sensing.registration import align_image_pair
from app.remote_sensing.preprocessing import (
    convert_array_to_base64_png, 
    normalize_to_uint8,
    robust_remote_sensing_preprocess
)

logger = logging.getLogger("satquery.change_detection")

class BaseChangeDetector(BaseModel):
    """Abstract base adapter for bi-temporal remote sensing change detection."""
    def __init__(self, model_name: str, implementation_status: str = "baseline"):
        super().__init__(
            model_name=model_name,
            task_type="change_detection",
            implementation_status=implementation_status,
            is_trained=False,
            is_remote_sensing_adapted=False
        )

    @abstractmethod
    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        pass

class PixelDifferenceChangeBaseline(BaseChangeDetector):
    """
    Classical pixel-difference change detection baseline.
    Computes absolute spectral reflectance delta across co-registered or rescaled rasters.
    Preserved as a reliable, honest baseline for algorithmic comparisons and fallbacks.
    """
    def __init__(self, difference_threshold: float = 35.0):
        super().__init__(
            model_name="PixelDifferenceChangeBaseline",
            implementation_status="baseline"
        )
        self.threshold = difference_threshold

    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if len(images) < 2:
            raise ValueError("Bi-temporal change detection requires at least 2 image inputs (Earlier & Later).")

        meta = metadata or {}
        meta_a = meta.get("primary", {})
        meta_b = meta.get("secondary", {})

        img_a, img_b = images[0], images[1]
        img_a, img_b_aligned, reg_info = align_image_pair(img_a, img_b, meta_a, meta_b)

        h, w = img_a.shape[:2]
        img_a_u8, meta_a_proc = robust_remote_sensing_preprocess(img_a, meta_a)
        img_b_u8, meta_b_proc = robust_remote_sensing_preprocess(img_b_aligned, meta_b)

        # Convert to grayscale for spectral delta calculation
        if img_a_u8.ndim == 3 and img_a_u8.shape[2] >= 3:
            gray_a = (0.299 * img_a_u8[:, :, 0] + 0.587 * img_a_u8[:, :, 1] + 0.114 * img_a_u8[:, :, 2]).astype(np.uint8)
        else:
            gray_a = img_a_u8 if img_a_u8.ndim == 2 else img_a_u8[:, :, 0]

        if img_b_u8.ndim == 3 and img_b_u8.shape[2] >= 3:
            gray_b = (0.299 * img_b_u8[:, :, 0] + 0.587 * img_b_u8[:, :, 1] + 0.114 * img_b_u8[:, :, 2]).astype(np.uint8)
        else:
            gray_b = img_b_u8 if img_b_u8.ndim == 2 else img_b_u8[:, :, 0]

        # Compute absolute difference with thresholding
        diff = np.abs(gray_a.astype(float) - gray_b.astype(float)).astype(np.uint8)
        change_mask = np.where(diff > self.threshold, 255, 0).astype(np.uint8)

        # Compute change metrics
        changed_pixels = int(np.count_nonzero(change_mask))
        total_pixels = h * w
        change_ratio = (changed_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        change_detected = bool(change_ratio > 0.5)

        # Colorized change map (Red indicates changed pixels)
        change_map = np.zeros((h, w, 3), dtype=np.uint8)
        change_map[:, :, 0] = change_mask
        change_map[:, :, 1] = (change_mask.astype(float) * 0.2).astype(np.uint8)
        change_map[:, :, 2] = (change_mask.astype(float) * 0.2).astype(np.uint8)

        # Overlay on secondary (later) image
        overlay = img_b_u8.copy()
        if overlay.ndim == 2:
            overlay = np.stack([overlay]*3, axis=-1)
        elif overlay.shape[2] > 3:
            overlay = overlay[:, :, :3]
            
        overlay = (0.60 * overlay + 0.40 * change_map).astype(np.uint8)

        # Extract major change cluster bounding boxes
        boxes = []
        try:
            import cv2
            contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > (total_pixels * 0.003):
                    x, y, bw, bh = cv2.boundingRect(c)
                    boxes.append({
                        "xmin": float(x),
                        "ymin": float(y),
                        "xmax": float(x + bw),
                        "ymax": float(y + bh),
                        "label": "Change Cluster",
                        "confidence": None
                    })
        except ImportError:
            from scipy import ndimage
            labeled_arr, _ = ndimage.label(change_mask)
            slices = ndimage.find_objects(labeled_arr)
            for sl in slices:
                y_sl, x_sl = sl
                if (y_sl.stop - y_sl.start) * (x_sl.stop - x_sl.start) > (total_pixels * 0.003):
                    boxes.append({
                        "xmin": float(x_sl.start),
                        "ymin": float(y_sl.start),
                        "xmax": float(x_sl.stop),
                        "ymax": float(y_sl.stop),
                        "label": "Change Cluster",
                        "confidence": None
                    })

        # Generate base64 representations for all 4 visual evidence products
        t1_b64 = convert_array_to_base64_png(img_a_u8)
        t2_b64 = convert_array_to_base64_png(img_b_u8)
        change_map_b64 = convert_array_to_base64_png(change_map)
        overlay_b64 = convert_array_to_base64_png(overlay)

        georef_note = ""
        if not reg_info.get("georeferenced", True):
            georef_note = " [Geospatial Notice]: Input imagery is not georeferenced (CRS unavailable). Genuine bi-temporal change detection requires georeferenced GeoTIFFs."
        elif not reg_info.get("co_registered", False):
            georef_note = " [Geospatial Notice]: Input imagery lacks verified spatial co-registration."

        answer = (
            f"Bi-temporal spectral difference analysis completed across {w}x{h} scene. "
            f"Observed pixel-level spectral reflectance delta > {self.threshold} across "
            f"{change_ratio:.2f}% of the scene area ({changed_pixels:,} of {total_pixels:,} pixels). "
            f"Change detected: {'Yes' if change_detected else 'No'}. "
            f"Identified {len(boxes)} primary change cluster regions. "
            f"Spatial co-registration status: {reg_info.get('co_registered')} ({reg_info.get('registration_method')})."
            f"{georef_note}"
        )

        evidence = [
            {
                "id": "ev_change_map",
                "type": "change_map",
                "title": "Bi-Temporal Change Map",
                "description": f"Binary difference magnitude mask (threshold > {self.threshold}). Changed pixels highlighted in red.",
                "data_base64": change_map_b64,
                "statistics": {
                    "changed_area_percent": f"{change_ratio:.2f}%",
                    "changed_pixel_count": changed_pixels,
                    "total_scene_pixels": total_pixels,
                    "difference_threshold": self.threshold,
                    "change_detected": change_detected,
                    "change_cluster_count": len(boxes),
                    "georeferenced": reg_info.get("georeferenced", False),
                    "co_registered": reg_info.get("co_registered"),
                    "registration_method": reg_info.get("registration_method")
                }
            },
            {
                "id": "ev_change_overlay",
                "type": "overlay",
                "title": "Change Cluster Spatial Overlay",
                "description": "Secondary scene overlaid with detected change clusters and bounding regions.",
                "data_base64": overlay_b64,
                "boxes": boxes
            },
            {
                "id": "ev_t1_before",
                "type": "original",
                "title": "T1: Earlier Scene (Before)",
                "description": f"Preprocessed baseline scene T1 ({w}x{h} px, CRS: {meta_a.get('crs_display', 'CRS unavailable')}).",
                "data_base64": t1_b64,
                "statistics": {
                    "dimensions": f"{w}x{h}",
                    "crs": meta_a.get("crs_display", "CRS unavailable"),
                    "modality": meta_a.get("modality", "OPTICAL")
                }
            },
            {
                "id": "ev_t2_after",
                "type": "original",
                "title": "T2: Later Scene (After)",
                "description": f"Preprocessed baseline scene T2 aligned to T1 grid.",
                "data_base64": t2_b64,
                "statistics": {
                    "dimensions": f"{w}x{h}",
                    "crs": meta_b.get("crs_display", "CRS unavailable"),
                    "modality": meta_b.get("modality", "OPTICAL")
                }
            }
        ]

        return {
            "answer": answer,
            "change_detected": change_detected,
            "change_area_percentage": round(change_ratio, 2),
            "changed_area_percent": round(change_ratio, 2),
            "change_mask": change_map_b64,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "model_type": "baseline",
            "model_name": self.model_name,
            "primary_model": self.model_name,
            "actual_model_used": self.model_name,
            "fallback_used": False,
            "model_status": "baseline",
            "implementation_status": "baseline",
            "confidence": None,
            "confidence_label": "Not available (Spectral Difference Baseline)",
            "models": [self.model_name],
            "registration_info": reg_info,
            "evidence": evidence,
            "model_provenance": {
                "primary_model": self.model_name,
                "actual_model_used": self.model_name,
                "checkpoint_status": "baseline",
                "fallback_used": False,
                "device": "cpu",
                "preprocessing": "Grayscale spectral difference with thresholding"
            }
        }


class SiameseRSChangeDetector(BaseChangeDetector):
    """
    Bitemporal Image Transformer (BIT-CD) Remote Sensing Change Detection Provider.
    Deep learning neural network for bi-temporal optical satellite change detection.
    Pretrained on remote sensing change datasets (LEVIR-CD, WHU-CD).
    Lazy-loaded: loads checkpoint on demand, never during boot or request loops.
    """
    @staticmethod
    def _find_checkpoint_path(checkpoint_path: Optional[str] = None) -> str:
        """
        Dynamically resolves the BIT-CD checkpoint path.
        Supports:
        1. Explicitly supplied checkpoint path (takes top priority, even if pointing to a test/missing path)
        2. CHANGE_MODEL_CHECKPOINT_PATH / RS_CHANGE_CHECKPOINT_PATH (if set in environment)
        3. Repository-root models/weights/bit_cd.pth (regardless of current working directory)
        """
        # 1. Explicitly supplied checkpoint path has top priority
        if checkpoint_path is not None:
            return checkpoint_path

        # Dynamic repository-root candidate lookup anchored from this file
        current_file = os.path.abspath(__file__)
        repo_root = os.path.abspath(os.path.join(current_file, "..", "..", "..", "..", ".."))

        # 2. Environment variable has next priority
        env_path = os.getenv("CHANGE_MODEL_CHECKPOINT_PATH") or os.getenv("RS_CHANGE_CHECKPOINT_PATH")
        if env_path is not None:
            if not os.path.isabs(env_path):
                return os.path.abspath(os.path.join(repo_root, env_path))
            return env_path

        candidates = [
            os.path.join(repo_root, "models", "weights", "bit_cd.pth"),
            os.path.join(os.getcwd(), "models", "weights", "bit_cd.pth"),
            os.path.join(os.getcwd(), "..", "models", "weights", "bit_cd.pth"),
            "models/weights/bit_cd.pth"
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return os.path.abspath(c)

        return os.path.join(repo_root, "models", "weights", "bit_cd.pth")

    def __init__(
        self, 
        model_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None
    ):
        model_id = model_name or os.getenv("CHANGE_MODEL_ID", "BIT-CD")
        super().__init__(
            model_name=model_id,
            implementation_status="unavailable"
        )
        self.checkpoint_path = self._find_checkpoint_path(checkpoint_path)
        self.configured_device = device or os.getenv("CHANGE_MODEL_DEVICE", "cpu")
        self.configured_dtype = dtype or os.getenv("CHANGE_MODEL_DTYPE", "float32")
        self._model = None
        self._is_loaded = False
        self.is_remote_sensing_adapted = True

    @property
    def is_available(self) -> bool:
        """Returns True if and only if the physical checkpoint file exists on disk."""
        return bool(self.checkpoint_path and os.path.exists(self.checkpoint_path))

    def _resolve_device(self):
        import torch
        if self.configured_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.configured_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for BIT-CD but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(self.configured_device)

    def load(self) -> bool:
        """
        Lazily loads the BIT-CD neural checkpoint into memory.
        Validates checkpoint existence; never downloads weights during request execution.
        """
        if self._is_loaded:
            return True

        if not self.is_available:
            logger.info(
                f"BIT-CD checkpoint not found at '{self.checkpoint_path}'. "
                "Implementation status remains 'unavailable' (fallback active)."
            )
            self._is_loaded = False
            self.implementation_status = "unavailable"
            return False

        try:
            import torch
            from app.models.change_detection.bit_model import BitemporalImageTransformer

            device = self._resolve_device()
            logger.info(f"Loading BIT-CD neural change detector from {self.checkpoint_path} onto {device}...")

            model = BitemporalImageTransformer(
                in_channels=3,
                num_classes=2,
                token_len=4,
                token_dim=32,
                num_decoder_layers=8,
                decoder_dim_head=8,
                enc_depth=1,
                resnet_stages_num=4
            )

            # Load checkpoint safely with weights_only=False
            ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model_G_state_dict" in ckpt:
                state_dict = ckpt["model_G_state_dict"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif isinstance(ckpt, dict):
                state_dict = ckpt
            else:
                state_dict = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt

            # Clean any 'module.' prefixes from DataParallel/DistributedDataParallel
            cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned_state, strict=True)
            model.to(device)
            model.eval()

            self._model = model
            self._is_loaded = True
            self.implementation_status = "real_model"
            self.is_remote_sensing_adapted = True
            logger.info("BIT-CD neural change detection model successfully loaded.")
            return True

        except Exception as e:
            logger.error(f"Failed to load BIT-CD checkpoint from '{self.checkpoint_path}': {e}")
            self._model = None
            self._is_loaded = False
            self.implementation_status = "unavailable"
            return False

    def _prepare_tensor(self, arr_u8: np.ndarray, device) -> Tuple[Any, Dict[str, Any]]:
        """
        Converts a preprocessed remote sensing uint8/float array to normalized PyTorch tensor [1, 3, H, W].
        Uses standard remote sensing ImageNet RGB mean & std normalization.
        """
        import torch

        # Ensure 3-channel RGB representation
        if arr_u8.ndim == 2:
            rgb = np.stack([arr_u8] * 3, axis=-1)
            band_mapping = [1, 1, 1]
        elif arr_u8.shape[2] == 1:
            rgb = np.repeat(arr_u8, 3, axis=-1)
            band_mapping = [1, 1, 1]
        elif arr_u8.shape[2] >= 3:
            rgb = arr_u8[:, :, :3]
            band_mapping = [1, 2, 3]
        else:
            rgb = np.pad(arr_u8, ((0, 0), (0, 0), (0, 3 - arr_u8.shape[2])), mode="edge")
            band_mapping = [1, arr_u8.shape[2], arr_u8.shape[2]]

        # Normalize to [0.0, 1.0] float32
        tensor = torch.from_numpy(rgb).float() / 255.0
        # Permute from [H, W, C] to [C, H, W]
        tensor = tensor.permute(2, 0, 1)

        # ImageNet standardization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # Add batch dimension [1, 3, H, W]
        tensor = tensor.unsqueeze(0).to(device)
        return tensor, {"input_band_mapping": band_mapping, "normalized_shape": list(tensor.shape)}

    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Performs genuine neural bi-temporal change detection tensor forward pass.
        """
        if len(images) < 2:
            raise ValueError("Bi-temporal change detection requires at least 2 image inputs (Earlier & Later).")

        if not self._is_loaded and not self.load():
            raise RuntimeError(
                f"BIT-CD neural checkpoint unavailable at '{self.checkpoint_path}'. "
                "Trigger fallback to PixelDifferenceChangeBaseline."
            )

        import torch

        meta = metadata or {}
        meta_a = meta.get("primary", {})
        meta_b = meta.get("secondary", {})

        img_a, img_b = images[0], images[1]
        # Align images using remote sensing spatial registration
        img_a, img_b_aligned, reg_info = align_image_pair(img_a, img_b, meta_a, meta_b)

        h, w = img_a.shape[:2]
        img_a_u8, meta_a_proc = robust_remote_sensing_preprocess(img_a, meta_a)
        img_b_u8, meta_b_proc = robust_remote_sensing_preprocess(img_b_aligned, meta_b)

        device = next(self._model.parameters()).device
        t1_tensor, meta_t1 = self._prepare_tensor(img_a_u8, device)
        t2_tensor, meta_t2 = self._prepare_tensor(img_b_u8, device)

        # Genuine PyTorch neural forward pass with no gradients
        with torch.no_grad():
            logits = self._model(t1_tensor, t2_tensor)  # [1, 2, H, W]
            probs = torch.softmax(logits, dim=1)         # [1, 2, H, W]
            # Channel 1 represents change probability
            change_prob = probs[0, 1].cpu().numpy()      # [H, W], range [0.0, 1.0]

        # Binary thresholding at 0.50 probability
        change_mask = (change_prob > 0.50).astype(np.uint8) * 255

        # Compute spatial statistics
        changed_pixels = int(np.count_nonzero(change_mask))
        total_pixels = h * w
        change_ratio = (changed_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        change_detected = bool(change_ratio > 0.5)

        # Colorized change probability map (Red gradient for change confidence)
        change_map = np.zeros((h, w, 3), dtype=np.uint8)
        change_map[:, :, 0] = (change_prob * 255.0).astype(np.uint8)
        change_map[:, :, 1] = (change_mask.astype(float) * 0.15).astype(np.uint8)
        change_map[:, :, 2] = (change_mask.astype(float) * 0.15).astype(np.uint8)

        # Spatial overlay on later scene
        overlay = img_b_u8.copy()
        if overlay.ndim == 2:
            overlay = np.stack([overlay] * 3, axis=-1)
        elif overlay.shape[2] > 3:
            overlay = overlay[:, :, :3]
        overlay = (0.55 * overlay + 0.45 * change_map).astype(np.uint8)

        # Extract change cluster bounding boxes
        boxes = []
        try:
            import cv2
            contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > (total_pixels * 0.003):
                    x, y, bw, bh = cv2.boundingRect(c)
                    boxes.append({
                        "xmin": float(x),
                        "ymin": float(y),
                        "xmax": float(x + bw),
                        "ymax": float(y + bh),
                        "label": "Neural Change Cluster",
                        "confidence": None
                    })
        except ImportError:
            from scipy import ndimage
            labeled_arr, _ = ndimage.label(change_mask)
            slices = ndimage.find_objects(labeled_arr)
            for sl in slices:
                y_sl, x_sl = sl
                if (y_sl.stop - y_sl.start) * (x_sl.stop - x_sl.start) > (total_pixels * 0.003):
                    boxes.append({
                        "xmin": float(x_sl.start),
                        "ymin": float(y_sl.start),
                        "xmax": float(x_sl.stop),
                        "ymax": float(y_sl.stop),
                        "label": "Neural Change Cluster",
                        "confidence": None
                    })

        t1_b64 = convert_array_to_base64_png(img_a_u8)
        t2_b64 = convert_array_to_base64_png(img_b_u8)
        change_map_b64 = convert_array_to_base64_png(change_map)
        overlay_b64 = convert_array_to_base64_png(overlay)

        georef_note = ""
        if not reg_info.get("georeferenced", True):
            georef_note = " [Geospatial Notice]: Input imagery is not georeferenced (CRS unavailable). Genuine bi-temporal change detection requires georeferenced GeoTIFFs."
        elif not reg_info.get("co_registered", False):
            georef_note = " [Geospatial Notice]: Input imagery lacks verified spatial co-registration."

        answer = (
            f"Neural bi-temporal change analysis using {self.model_name} completed across {w}x{h} scene. "
            f"Bitemporal Transformer detected spatial surface changes across {change_ratio:.2f}% of the scene "
            f"({changed_pixels:,} of {total_pixels:,} pixels with change probability > 0.50). "
            f"Change detected: {'Yes' if change_detected else 'No'}. "
            f"Identified {len(boxes)} primary change cluster regions. "
            f"Spatial co-registration status: {reg_info.get('co_registered')} ({reg_info.get('registration_method')}). "
            f"Note: Specific semantic attribution requires auxiliary ground truth or multi-band spectral classification."
            f"{georef_note}"
        )

        evidence = [
            {
                "id": "ev_change_map",
                "type": "change_map",
                "title": "BIT-CD Neural Change Map",
                "description": f"Neural change probability map thresholded at p > 0.50. Change regions highlighted in red.",
                "data_base64": change_map_b64,
                "statistics": {
                    "changed_area_percent": f"{change_ratio:.2f}%",
                    "changed_pixel_count": changed_pixels,
                    "total_scene_pixels": total_pixels,
                    "model_name": self.model_name,
                    "change_detected": change_detected,
                    "change_cluster_count": len(boxes),
                    "georeferenced": reg_info.get("georeferenced", False),
                    "co_registered": reg_info.get("co_registered"),
                    "registration_method": reg_info.get("registration_method"),
                    "device": str(device)
                }
            },
            {
                "id": "ev_change_overlay",
                "type": "overlay",
                "title": "Neural Change Cluster Spatial Overlay",
                "description": "Secondary scene overlaid with Transformer-detected change clusters and spatial bounding boxes.",
                "data_base64": overlay_b64,
                "boxes": boxes
            },
            {
                "id": "ev_t1_before",
                "type": "original",
                "title": "T1: Earlier Scene (Before)",
                "description": f"Preprocessed baseline scene T1 ({w}x{h} px, CRS: {meta_a.get('crs_display', 'CRS unavailable')}).",
                "data_base64": t1_b64,
                "statistics": {
                    "dimensions": f"{w}x{h}",
                    "crs": meta_a.get("crs_display", "CRS unavailable"),
                    "modality": meta_a.get("modality", "OPTICAL")
                }
            },
            {
                "id": "ev_t2_after",
                "type": "original",
                "title": "T2: Later Scene (After)",
                "description": f"Preprocessed baseline scene T2 aligned to T1 grid.",
                "data_base64": t2_b64,
                "statistics": {
                    "dimensions": f"{w}x{h}",
                    "crs": meta_b.get("crs_display", "CRS unavailable"),
                    "modality": meta_b.get("modality", "OPTICAL")
                }
            }
        ]

        return {
            "answer": answer,
            "change_detected": change_detected,
            "change_area_percentage": round(change_ratio, 2),
            "changed_area_percent": round(change_ratio, 2),
            "change_mask": change_map_b64,
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "model_type": "neural_model",
            "model_name": self.model_name,
            "primary_model": self.model_name,
            "actual_model_used": self.model_name,
            "fallback_used": False,
            "model_status": "loaded",
            "implementation_status": "real_model",
            "confidence": None,
            "confidence_label": "Not available (uncalibrated neural change logits)",
            "models": [self.model_name],
            "registration_info": reg_info,
            "evidence": evidence,
            "model_provenance": {
                "primary_model": self.model_name,
                "actual_model_used": self.model_name,
                "checkpoint_status": "loaded",
                "fallback_used": False,
                "device": str(device),
                "preprocessing": "Multiband robust min-max normalization + ImageNet standardization",
                "input_band_mapping": {
                    "T1": meta_t1.get("input_band_mapping"),
                    "T2": meta_t2.get("input_band_mapping")
                },
                "inference_resolution": f"{w}x{h}"
            }
        }


class ChangeDetectionProvider(BaseModel):
    """
    Change Detection Orchestrator.
    Manages neural model selection (BIT-CD) with verified fallback to PixelDifferenceChangeBaseline.
    Transparently reports primary model, actual model used, fallback status, and provenance.
    """
    def __init__(self):
        super().__init__(
            model_name="ChangeDetectionProvider",
            task_type="change_detection",
            implementation_status="baseline",
            is_trained=False,
            is_remote_sensing_adapted=False
        )
        self.baseline = PixelDifferenceChangeBaseline()
        self.deep_detector = SiameseRSChangeDetector()

    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        primary_name = self.deep_detector.model_name  # "BIT-CD"

        # Attempt neural execution if checkpoint is present
        if self.deep_detector.is_available:
            try:
                res = self.deep_detector.predict(images, query, metadata)
                res["models"] = [self.model_name, self.deep_detector.model_name]
                return res
            except Exception as e:
                logger.warning(
                    f"Neural change detector '{primary_name}' failed during inference ({e}). "
                    "Dispatching to deterministic baseline fallback."
                )

        # Fallback to verified baseline
        logger.info(
            f"Neural change detector checkpoint '{self.deep_detector.checkpoint_path}' unavailable. "
            f"Using baseline fallback: '{self.baseline.model_name}'."
        )
        res = self.baseline.predict(images, query, metadata)
        res["primary_model"] = primary_name
        res["actual_model_used"] = self.baseline.model_name
        res["fallback_used"] = True
        res["model_status"] = "checkpoint_not_found" if not self.deep_detector.is_available else "inference_failed"
        res["implementation_status"] = "baseline"
        res["models"] = [self.model_name, primary_name, self.baseline.model_name]
        res["model_provenance"] = {
            "primary_model": primary_name,
            "actual_model_used": self.baseline.model_name,
            "checkpoint_status": "checkpoint_not_found" if not self.deep_detector.is_available else "inference_failed",
            "fallback_used": True,
            "device": "cpu",
            "preprocessing": "Grayscale spectral difference with thresholding"
        }
        return res


# Backwards compatibility alias
ChangeDetector = PixelDifferenceChangeBaseline
