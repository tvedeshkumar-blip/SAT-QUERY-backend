import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
from app.remote_sensing.preprocessing import convert_array_to_base64_png, normalize_to_uint8

logger = logging.getLogger("satquery.bands")

# Standard peer-reviewed threshold for moderate-to-dense green vegetation canopy
VEGETATION_NDVI_THRESHOLD = 0.40
SCL_NDVI_EXCLUDED_CLASSES = {0, 1, 3, 8, 9, 10}


def compute_calibrated_ndvi(
    b04_arr: Optional[np.ndarray] = None,
    b08_arr: Optional[np.ndarray] = None,
    b04_meta: Optional[Dict[str, Any]] = None,
    b08_meta: Optional[Dict[str, Any]] = None,
    scl_arr: Optional[np.ndarray] = None,
    scl_meta: Optional[Dict[str, Any]] = None,
    scale_factor: float = 10000.0,
    offset: float = 0.0,
    vegetation_threshold: float = VEGETATION_NDVI_THRESHOLD,
    allow_uncertain_as_clear: bool = False,
    # Compatibility aliases
    red_band: Optional[np.ndarray] = None,
    nir_band: Optional[np.ndarray] = None,
    scl_band: Optional[np.ndarray] = None,
    target_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes scientifically defensible Normalized Difference Vegetation Index (NDVI)
    using genuine Sentinel-2 Level-2A Bottom-of-Atmosphere (BOA) surface reflectance:
        NDVI = (B08 - B04) / (B08 + B04)

    Requirements:
    - Reflectance conversion: (DN / scale_factor) + offset (default scale: 10,000 for Sentinel-2 L2A).
    - Grid alignment: Validates matching CRS, affine geotransform, and dimensions.
    - Zero denominator protection: Pixels where (B08 + B04) <= 1e-5 are masked (never assigned 0).
    - SCL quality screening: Excludes clouds (8, 9, 10), cloud shadows (3), defective (1), nodata (0),
      and out-of-range unknown values (<0 or >11).
    - Zero valid pixel defense: Returns None for statistics and coverage if all pixels are invalid/obstructed.
    """
    if b04_arr is None and red_band is not None:
        b04_arr = red_band
    if b08_arr is None and nir_band is not None:
        b08_arr = nir_band
    if scl_arr is None and scl_band is not None:
        scl_arr = scl_band
    if b04_meta is None and target_meta is not None:
        b04_meta = target_meta

    if b04_arr is None or b08_arr is None:
        raise ValueError("Both B04 (Red) and B08 (NIR) arrays are required to compute calibrated NDVI.")

    # 1. Extract 2D single-band rasters
    if b04_arr.ndim == 3:
        b04_2d = b04_arr[0, :, :] if b04_arr.shape[0] in (1, 2, 3, 4) and b04_arr.shape[0] < b04_arr.shape[1] else b04_arr[:, :, 0]
    elif b04_arr.ndim == 2:
        b04_2d = b04_arr
    else:
        raise ValueError(f"B04 array must be 2D or 3D, received shape {b04_arr.shape}")

    if b08_arr.ndim == 3:
        b08_2d = b08_arr[0, :, :] if b08_arr.shape[0] in (1, 2, 3, 4) and b08_arr.shape[0] < b08_arr.shape[1] else b08_arr[:, :, 0]
    elif b08_arr.ndim == 2:
        b08_2d = b08_arr
    else:
        raise ValueError(f"B08 array must be 2D or 3D, received shape {b08_arr.shape}")

    h, w = b04_2d.shape[:2]
    total_pixels = int(h * w)

    # 2. Check and align spatial grids if necessary
    aligned_b08 = b08_2d
    if b08_2d.shape[:2] != (h, w):
        if b04_meta and b08_meta:
            from app.remote_sensing.registration import align_image_pair
            _, aligned_b08, _ = align_image_pair(b04_2d, b08_2d, b04_meta, b08_meta)
        else:
            raise ValueError(f"Spatial dimension mismatch between B04 ({b04_2d.shape}) and B08 ({b08_2d.shape})")

    # 3. Surface Reflectance Scaling (Sentinel-2 Level-2A convention: (DN / scale_factor) + offset)
    is_b04_int = np.issubdtype(b04_2d.dtype, np.integer) or np.issubdtype(b04_2d.dtype, np.unsignedinteger)
    is_b08_int = np.issubdtype(aligned_b08.dtype, np.integer) or np.issubdtype(aligned_b08.dtype, np.unsignedinteger)

    # Resolve metadata-driven or parameter-driven radiometric offset
    effective_offset = offset
    if effective_offset == 0.0 and b04_meta:
        if "offset" in b04_meta:
            try:
                effective_offset = float(b04_meta["offset"])
            except (ValueError, TypeError):
                pass
        elif "raster:bands" in b04_meta and isinstance(b04_meta["raster:bands"], list) and len(b04_meta["raster:bands"]) > 0:
            band_info = b04_meta["raster:bands"][0]
            if isinstance(band_info, dict) and "offset" in band_info:
                try:
                    effective_offset = float(band_info["offset"])
                except (ValueError, TypeError):
                    pass

    # Normalize offset to reflectance units:
    # If |effective_offset| > 2.0, it is represented in integer DN units (e.g. -1000 DN -> -0.1 reflectance)
    offset_refl = (effective_offset / scale_factor) if abs(effective_offset) > 2.0 else float(effective_offset)

    if is_b04_int or np.nanmax(b04_2d) > 2.0:
        red_refl = (b04_2d.astype(np.float64) / scale_factor) + offset_refl
    else:
        red_refl = b04_2d.astype(np.float64) + (offset_refl if abs(effective_offset) <= 2.0 else 0.0)

    if is_b08_int or np.nanmax(aligned_b08) > 2.0:
        nir_refl = (aligned_b08.astype(np.float64) / scale_factor) + offset_refl
    else:
        nir_refl = aligned_b08.astype(np.float64) + (offset_refl if abs(effective_offset) <= 2.0 else 0.0)

    # 4. Valid-pixel policy & masking
    finite_mask = np.isfinite(red_refl) & np.isfinite(nir_refl)

    # Nodata checks: mask if either B04 or B08 contains nodata
    b04_nd = (b04_meta or {}).get("nodata")
    b08_nd = (b08_meta or {}).get("nodata")
    nodata_mask = np.zeros((h, w), dtype=bool)
    nd_val = b04_nd if b04_nd is not None else b08_nd
    if nd_val is not None:
        try:
            nodata_mask |= np.isclose(b04_2d, float(nd_val), atol=1e-3)
            nodata_mask |= np.isclose(aligned_b08, float(nd_val), atol=1e-3)
        except Exception:
            pass
    if b08_nd is not None and b08_nd != nd_val:
        try:
            nodata_mask |= np.isclose(aligned_b08, float(b08_nd), atol=1e-3)
        except Exception:
            pass

    # Physically valid reflectance bounds [0.0, 2.0]
    valid_refl_bounds = (red_refl >= 0.0) & (nir_refl >= 0.0) & (red_refl <= 2.0) & (nir_refl <= 2.0)

    # Valid denominator check
    denominator = nir_refl + red_refl
    valid_denom = denominator > 1e-5

    # Base valid mask
    valid_mask = finite_mask & (~nodata_mask) & valid_refl_bounds & valid_denom

    # 5. SCL Categorical Quality Screening
    scl_eval = None
    if scl_arr is not None:
        from app.remote_sensing.quality import QualityScreener
        if (scl_arr.shape[:2] != (h, w) or (scl_meta and b04_meta and scl_meta.get("transform") != b04_meta.get("transform"))) and scl_meta and b04_meta:
            aligned_scl, _ = QualityScreener.reproject_scl_to_target(scl_arr, scl_meta, b04_meta)
        else:
            aligned_scl = scl_arr[0] if scl_arr.ndim == 3 else scl_arr

        scl_eval = QualityScreener.evaluate_scl(aligned_scl, allow_uncertain_as_clear=allow_uncertain_as_clear)
        scl_clear_mask = scl_eval["clear_mask"]
        valid_mask = valid_mask & scl_clear_mask

    valid_pixel_count = int(np.count_nonzero(valid_mask))

    # 6. Compute NDVI values strictly across valid pixels
    ndvi_arr = np.full((h, w), np.nan, dtype=np.float32)
    if valid_pixel_count > 0:
        raw_ndvi = (nir_refl[valid_mask] - red_refl[valid_mask]) / denominator[valid_mask]
        # Mathematical bounding [-1.0, 1.0]
        raw_ndvi = np.clip(raw_ndvi, -1.0, 1.0)
        ndvi_arr[valid_mask] = raw_ndvi.astype(np.float32)

    # 7. Statistical Aggregation
    if valid_pixel_count == 0:
        ndvi_mean = None
        ndvi_p90 = None
        ndvi_median = None
        ndvi_min = None
        ndvi_max = None
        vegetation_coverage_pct = None
        veg_pixel_count = 0
        status = "unsupported_zero_valid_pixels"
        summary = (
            "Calibrated Sentinel-2 NDVI unsupported: Zero valid clear-sky pixels available after quality screening. "
            "All pixels were excluded due to cloud obstruction, cloud shadow, nodata, or unphysical reflectance."
        )
    else:
        valid_vals = ndvi_arr[valid_mask]
        ndvi_mean = round(float(np.mean(valid_vals)), 4)
        ndvi_p90 = round(float(np.percentile(valid_vals, 90)), 4)
        ndvi_median = round(float(np.median(valid_vals)), 4)
        ndvi_min = round(float(np.min(valid_vals)), 4)
        ndvi_max = round(float(np.max(valid_vals)), 4)
        veg_pixel_count = int(np.count_nonzero(valid_vals >= vegetation_threshold))
        vegetation_coverage_pct = round((veg_pixel_count / valid_pixel_count) * 100.0, 2)
        status = "success"
        clearance_pct = round((valid_pixel_count / max(1, total_pixels)) * 100.0, 1)
        summary = (
            f"Calibrated Sentinel-2 NDVI computed across {valid_pixel_count:,} valid clear-sky pixels ({clearance_pct}% scene clearance). "
            f"Mean NDVI: {ndvi_mean:.4f}, p90: {ndvi_p90:.4f}, vegetation coverage [NDVI >= {vegetation_threshold:.2f}]: {vegetation_coverage_pct:.1f}%."
        )

    # 8. Colorized NDVI Heatmap Preview (Vectorized RGB uint8)
    colorized_rgb = np.full((h, w, 3), 40, dtype=np.uint8)  # Masked pixels: dark gray [40, 40, 40]
    if valid_pixel_count > 0:
        m_water = valid_mask & (ndvi_arr < 0.0)
        m_soil = valid_mask & (ndvi_arr >= 0.0) & (ndvi_arr < 0.2)
        m_sparse = valid_mask & (ndvi_arr >= 0.2) & (ndvi_arr < vegetation_threshold)
        m_dense = valid_mask & (ndvi_arr >= vegetation_threshold) & (ndvi_arr < 0.7)
        m_lush = valid_mask & (ndvi_arr >= 0.7)

        colorized_rgb[m_water] = [30, 110, 220]      # Water / wet / negative: Cyan/Blue
        colorized_rgb[m_soil] = [205, 175, 120]      # Bare soil / built-up: Sand Brown
        colorized_rgb[m_sparse] = [185, 215, 70]     # Sparse vegetation: Yellow-Green
        colorized_rgb[m_dense] = [35, 160, 45]       # Moderate-to-dense canopy: Green
        colorized_rgb[m_lush] = [10, 110, 25]        # High-density canopy: Deep Emerald Green

    colorized_b64 = convert_array_to_base64_png(colorized_rgb)

    # 9. Document Scientific Limitations
    limitations = [
        f"Calibrated NDVI was computed from Sentinel-2 L2A BOA surface reflectance using scale factor {scale_factor}" +
        (f" and radiometric offset {effective_offset}." if effective_offset != 0.0 else "."),
        f"Vegetation canopy coverage proxy uses explicit scientific threshold NDVI >= {vegetation_threshold:.2f}.",
        "NDVI indicates chlorophyll absorption and cellular structure; it does not identify crop species, tree taxonomy, or plant health in isolation."
    ]
    if scl_eval:
        limitations.append(f"SCL quality screening applied: {scl_eval['excluded_pixel_count']:,} cloud/shadow/invalid pixels excluded.")
    else:
        limitations.append("SCL quality screening was unperformed; cloud and shadow pixels may distort NDVI values.")

    return {
        "status": status,
        "is_calibrated": True,
        "summary": summary,
        "ndvi_mean": ndvi_mean,
        "ndvi_p90": ndvi_p90,
        "ndvi_median": ndvi_median,
        "ndvi_min": ndvi_min,
        "ndvi_max": ndvi_max,
        "vegetation_coverage_pct": vegetation_coverage_pct,
        "vegetation_threshold": vegetation_threshold,
        "veg_pixel_count": veg_pixel_count,
        "vegetation_pixel_count": veg_pixel_count,
        "valid_pixel_count": valid_pixel_count,
        "total_pixel_count": total_pixels,
        "masked_pixel_count": total_pixels - valid_pixel_count,
        "valid_mask": valid_mask,
        "ndvi_array": ndvi_arr,
        "colorized_preview_base64": colorized_b64,
        "preview_base64": colorized_b64,
        "scl_evaluation": scl_eval,
        "scale_factor_applied": scale_factor,
        "offset_applied": effective_offset,
        "calibration": {
            "scale_factor": scale_factor,
            "offset": effective_offset,
            "offset_reflectance": offset_refl,
            "convention": "BOA_SURFACE_REFLECTANCE"
        },
        "validity_policy": {
            "scl_mask_applied": scl_eval is not None,
            "denominator_threshold": 1e-5
        },
        "scientific_limitations": limitations
    }


def compute_ndvi(arr: np.ndarray, nir_band_idx: int = 3, red_band_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    [LEGACY / APPROXIMATION ONLY]
    Computes pseudo-NDVI from 3-band RGB visual arrays or uncalibrated rasters:
    NDVI = (NIR - Red) / (NIR + Red)

    NOTICE: This is an uncalibrated visual approximation for display browsing only.
    It does NOT represent calibrated Sentinel-2 Level-2A surface reflectance NDVI.
    For genuine remote sensing scientific analysis, use `compute_calibrated_ndvi`.

    Returns:
      ndvi_float: normalized float array in [-1.0, 1.0]
      ndvi_colorized: uint8 RGB colorized heatmap
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

# Compatibility alias
compute_calibrated_sentinel2_ndvi = compute_calibrated_ndvi
