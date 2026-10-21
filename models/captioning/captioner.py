import logging
import os
from typing import Dict, Any, Optional, List
import numpy as np
from app.models.base import BaseModel
from app.remote_sensing.preprocessing import normalize_to_uint8

logger = logging.getLogger("satquery.captioning")

class BaseCaptioner(BaseModel):
    """Abstract base class for remote sensing captioners."""
    def __init__(self, model_name: str, implementation_status: str = "baseline"):
        super().__init__(
            model_name=model_name,
            task_type="captioning",
            implementation_status=implementation_status,
            is_trained=False,
            is_remote_sensing_adapted=False
        )

class ClassicalSpectralCaptioner(BaseCaptioner):
    """
    Truthful baseline captioner.
    Inspects actual image dimensions, band statistics, dynamic range, and CRS metadata
    to generate an accurate physical description of the raster scene.
    Does NOT assert specific land cover classes unless backed by spectral analysis.
    """
    def __init__(self):
        super().__init__(
            model_name="ClassicalSpectralCaptionerBaseline",
            implementation_status="baseline"
        )

    def predict(
        self, 
        images: List[np.ndarray], 
        query: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not images:
            raise ValueError("Captioning task requires an input satellite raster.")

        img = images[0]
        meta = metadata or {}
        h, w = img.shape[:2]
        bands = meta.get("bands", img.shape[2] if img.ndim > 2 else 1)
        crs = meta.get("crs_display") or meta.get("crs") or "CRS unavailable"
        modality = meta.get("modality", "OPTICAL")

        # Derive factual spectral statistics from actual image pixels
        img_u8 = normalize_to_uint8(img)
        mean_val = float(np.mean(img_u8))
        std_val = float(np.std(img_u8))

        # Check for vegetation/water tendencies via band math if 3-band optical
        characteristics = []
        if modality == "OPTICAL" and img_u8.ndim == 3 and img_u8.shape[2] >= 3:
            r = img_u8[:, :, 0].astype(float)
            g = img_u8[:, :, 1].astype(float)
            b = img_u8[:, :, 2].astype(float)
            greenness = float(np.mean(g - r))
            blueness = float(np.mean(b - (r + g) / 2.0))
            if greenness > 8.0:
                characteristics.append("predominantly vegetated surface albedo")
            if blueness > 15.0:
                characteristics.append("high shortwave/water-absorptive reflectance")
            if std_val > 45.0:
                characteristics.append("high spatial texture and edge contrast")
            else:
                characteristics.append("uniform spectral albedo")
        elif modality == "SAR":
            characteristics.append(f"microwave backscatter intensity (mean: {mean_val:.1f}, std: {std_val:.1f})")

        char_str = ", ".join(characteristics) if characteristics else "moderate spectral albedo variations"

        caption = (
            f"Satellite {modality} raster scene ({w}x{h} px, {bands} band(s), {crs}). "
            f"Scene statistics show {char_str} with dynamic range spread std={std_val:.1f}."
        )

        return {
            "caption": caption,
            "answer": caption,
            "model_type": "baseline",
            "implementation_status": "baseline",
            "status": "demo",
            "confidence": None,
            "confidence_label": "Not available (Heuristic Spectral Baseline)",
            "models": [self.model_name],
            "evidence": []
        }

# Backwards compatibility alias clearly marked as baseline
RemoteSensingCaptioner = ClassicalSpectralCaptioner
