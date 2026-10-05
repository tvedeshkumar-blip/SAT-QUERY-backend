import numpy as np
from typing import Dict, Any, Optional

def detect_modality_structured(
    arr: np.ndarray, 
    metadata: Optional[Dict[str, Any]] = None, 
    hint: str = "primary",
    user_selection: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detects satellite image modality with provenance tracking.
    Never claims a heuristic rule is a trained classifier.
    
    Returns:
    {
      "modality": "OPTICAL" | "SAR" | "MULTISPECTRAL",
      "source": "user" | "metadata" | "filename" | "heuristic",
      "confidence": None
    }
    """
    meta = metadata or {}

    # 1. User explicit selection takes highest priority
    if user_selection and user_selection.upper() in ("OPTICAL", "SAR", "MULTISPECTRAL"):
        return {
            "modality": user_selection.upper(),
            "source": "user",
            "confidence": None
        }

    # 2. Metadata sensor or modality tag
    meta_mod = (meta.get("modality") or "").upper()
    if meta_mod in ("OPTICAL", "SAR", "MULTISPECTRAL"):
        return {
            "modality": meta_mod,
            "source": "metadata",
            "confidence": None
        }

    sensor_tag = (meta.get("sensor") or meta.get("tags", {}).get("SENSOR") or "").lower()
    if "risat" in sensor_tag or "sentinel-1" in sensor_tag or "c-band" in sensor_tag or "sar" in sensor_tag:
        return {
            "modality": "SAR",
            "source": "metadata",
            "confidence": None
        }

    # 3. Explicit input role hint from API payload
    hint_lower = (hint or "").lower()
    if hint_lower == "sar":
        return {
            "modality": "SAR",
            "source": "user",
            "confidence": None
        }
    if hint_lower in ("optical", "rgb"):
        return {
            "modality": "OPTICAL",
            "source": "user",
            "confidence": None
        }

    # 4. Filename cues
    filename = (meta.get("filename") or "").lower()
    if any(k in filename for k in ["sar", "risat", "sentinel1", "s1_"]):
        return {
            "modality": "SAR",
            "source": "filename",
            "confidence": None
        }

    # 5. Channel count heuristic
    bands = meta.get("bands", arr.shape[2] if arr.ndim == 3 else 1)
    if bands > 3:
        return {
            "modality": "MULTISPECTRAL",
            "source": "heuristic",
            "confidence": None
        }

    # 6. Single channel / 2D array heuristic
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        return {
            "modality": "SAR",
            "source": "heuristic",
            "confidence": None
        }

    # 7. Check if all 3 RGB channels are identical (grayscale encoded as RGB)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        if np.array_equal(r, g) and np.array_equal(g, b):
            variance = float(np.var(r))
            if variance > 1000.0:
                return {
                    "modality": "SAR",
                    "source": "heuristic",
                    "confidence": None
                }

    # Default to optical reflectance
    return {
        "modality": "OPTICAL",
        "source": "heuristic",
        "confidence": None
    }

def detect_image_modality(arr: np.ndarray, metadata: Dict[str, Any], hint: str = "primary") -> str:
    """
    Backwards-compatible convenience function returning only the modality string.
    """
    result = detect_modality_structured(arr, metadata, hint)
    return result["modality"]
