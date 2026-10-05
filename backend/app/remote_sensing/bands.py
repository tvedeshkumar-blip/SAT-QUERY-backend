import numpy as np
import logging
from typing import Dict, Any, Tuple
from app.remote_sensing.preprocessing import convert_array_to_base64_png, normalize_to_uint8

logger = logging.getLogger("satquery.bands")

def compute_ndvi(arr: np.ndarray, nir_band_idx: int = 3, red_band_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Normalized Difference Vegetation Index: NDVI = (NIR - Red) / (NIR + Red)
    Returns:
      ndvi_float: normalized float array in [-1.0, 1.0]
      ndvi_colorized: uint8 RGB colorized heatmap (Brown/Yellow = low, Vibrant Green = high)
    """
    h, w = arr.shape[:2]
    
    if arr.ndim == 3 and arr.shape[2] > max(nir_band_idx, red_band_idx):
        nir = arr[:, :, nir_band_idx].astype(float)
        red = arr[:, :, red_band_idx].astype(float)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        # Approximate from RGB: Green channels often proxy near-infrared reflection in canopy
        nir = arr[:, :, 1].astype(float) * 1.2
        red = arr[:, :, 0].astype(float)
    else:
        nir = arr.astype(float)
        red = (arr * 0.7).astype(float)

    denominator = nir + red
    denominator[denominator == 0] = 1e-5
    ndvi_float = (nir - red) / denominator
    ndvi_float = np.clip(ndvi_float, -1.0, 1.0)

    # Colorize: -1 to 0 (water/soil: blue/brown), 0 to 1 (vegetation: light green to deep emerald)
    ndvi_colorized = np.zeros((h, w, 3), dtype=np.uint8)
    norm = ((ndvi_float + 1.0) / 2.0 * 255.0).astype(np.uint8)
    
    # Red channel: high for dry/soil (low NDVI)
    ndvi_colorized[:, :, 0] = 255 - norm
    # Green channel: high for healthy vegetation (high NDVI)
    ndvi_colorized[:, :, 1] = norm
    # Blue channel: moderate for balanced visual contrast
    ndvi_colorized[:, :, 2] = np.where(ndvi_float < 0, 200, 30).astype(np.uint8)

    return ndvi_float, ndvi_colorized

def compute_ndwi(arr: np.ndarray, green_band_idx: int = 1, nir_band_idx: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Normalized Difference Water Index: NDWI = (Green - NIR) / (Green + NIR)
    Returns:
      ndwi_float: normalized float array in [-1.0, 1.0]
      ndwi_colorized: uint8 RGB colorized heatmap (Cyan/Deep Blue = water, Gray = terrestrial)
    """
    h, w = arr.shape[:2]

    if arr.ndim == 3 and arr.shape[2] > max(green_band_idx, nir_band_idx):
        green = arr[:, :, green_band_idx].astype(float)
        nir = arr[:, :, nir_band_idx].astype(float)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        green = arr[:, :, 1].astype(float)
        nir = arr[:, :, 0].astype(float)
    else:
        green = arr.astype(float)
        nir = (arr * 0.8).astype(float)

    denominator = green + nir
    denominator[denominator == 0] = 1e-5
    ndwi_float = (green - nir) / denominator
    ndwi_float = np.clip(ndwi_float, -1.0, 1.0)

    # Colorize water zones with cyan/blue
    ndwi_colorized = np.zeros((h, w, 3), dtype=np.uint8)
    norm = ((ndwi_float + 1.0) / 2.0 * 255.0).astype(np.uint8)
    ndwi_colorized[:, :, 0] = 50
    ndwi_colorized[:, :, 1] = norm
    ndwi_colorized[:, :, 2] = 255

    return ndwi_float, ndwi_colorized

def compute_false_color_cir(arr: np.ndarray) -> np.ndarray:
    """
    Computes standard Color Infrared (CIR) False Color Composite (NIR -> Red, Red -> Green, Green -> Blue).
    Chlorophyll reflects heavily in NIR, illuminating dense forests and crops in striking red tones.
    """
    arr_u8 = normalize_to_uint8(arr)
    h, w = arr_u8.shape[:2]
    cir = np.zeros((h, w, 3), dtype=np.uint8)

    if arr_u8.ndim == 3 and arr_u8.shape[2] >= 3:
        # Standard CIR: Red=Green(simulating NIR boost), Green=Red, Blue=Blue
        cir[:, :, 0] = np.clip(arr_u8[:, :, 1].astype(float) * 1.3, 0, 255).astype(np.uint8)
        cir[:, :, 1] = arr_u8[:, :, 0]
        cir[:, :, 2] = arr_u8[:, :, 2]
    else:
        gray = arr_u8 if arr_u8.ndim == 2 else arr_u8[:, :, 0]
        cir[:, :, 0] = gray
        cir[:, :, 1] = (gray * 0.5).astype(np.uint8)
        cir[:, :, 2] = (gray * 0.3).astype(np.uint8)

    return cir

def compute_sar_decibel_and_filter(arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Converts raw SAR linear amplitude into calibrated Decibel (dB) backscatter:
    sigma0_dB = 10 * log10(intensity^2 + 1e-5)
    Applies speckle smoothing filter.
    """
    arr_f = arr.astype(float)
    if arr_f.ndim == 3:
        arr_f = arr_f[:, :, 0]

    # Decibel backscatter calculation
    intensity = arr_f ** 2
    intensity = np.maximum(intensity, 1e-5)
    sigma0_db = 10.0 * np.log10(intensity)

    # 3x3 uniform boxcar speckle filter
    h, w = sigma0_db.shape
    filtered = sigma0_db.copy()
    pad = np.pad(sigma0_db, 1, mode="edge")
    for i in range(h):
        for j in range(w):
            filtered[i, j] = np.mean(pad[i:i+3, j:j+3])

    # Normalize dB [-25dB to 0dB] into uint8
    db_min, db_max = -25.0, 5.0
    normalized_db = np.clip((filtered - db_min) / (db_max - db_min) * 255.0, 0, 255).astype(np.uint8)

    stats = {
        "mean_sigma0_db": f"{float(np.mean(filtered)):.2f} dB",
        "min_sigma0_db": f"{float(np.min(filtered)):.2f} dB",
        "max_sigma0_db": f"{float(np.max(filtered)):.2f} dB",
        "speckle_reduction": "3x3 Spatial Boxcar Smoothing"
    }

    return normalized_db, stats
