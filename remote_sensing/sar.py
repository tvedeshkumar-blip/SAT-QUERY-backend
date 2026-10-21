"""
Stage 13: Sentinel-1 Radiometrically Terrain-Corrected (RTC) Dual-Polarization SAR Processor.

Implements peer-reviewed polarimetric and radiometric analysis for Sentinel-1 C-band SAR:
1. Cross-Polarization Ratio:
       q = gamma0_VH / gamma0_VV  (linear power ratio)
       q_dB = 10 * log10(q) = VH_dB - VV_dB  (decibels)

2. Dual-Polarimetric Radar Vegetation Index (DpRVI for intensity / RTC products):
       DpRVI = (q * (q + 3)) / ((q + 1) ** 2)  (Mandal et al., 2020)
   Valid physical domain:
       - q >= 0: DpRVI in [0.0, 1.0]
       - Low values (~0.0 - 0.2): smooth bare soil, water, specular surface scattering
       - Moderate values (~0.2 - 0.45): sparse vegetation, early crop growth
       - High values (~0.45 - 1.0): dense vegetation canopy, mature crops, forest volume scattering

3. Radiometric Calibration & Logarithmic Averages:
       - Linear backscatter power: gamma0 (m^2/m^2)
       - Decibel average: 10 * log10(mean(gamma0_linear)) (physically correct radiometric power average)
       - Pixel-wise dB average: mean(10 * log10(gamma0_linear)) (distinguished from mean power in dB)

4. Rigorous Screening:
       - Screens nodata sentinels (e.g. -32768.0, 0, or metadata nodata)
       - Masks non-positive linear power (<= 0.0) before logarithmic conversion
       - Masks non-finite floats (NaN, Inf)
       - Transparently reports provider-level terrain shadow mask limitations
       - Zero-valid-pixel defense: Returns None for statistics when valid pixels are absent
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from app.remote_sensing.preprocessing import convert_array_to_base64_png

logger = logging.getLogger("satquery.sar")

DPRVI_FORMULA_REFERENCE = "DpRVI = (q * (q + 3)) / ((q + 1) ** 2), where q = VH / VV (Mandal et al., 2020)"
SAR_RTC_DEFAULT_NODATA = -32768.0


def compute_calibrated_dual_pol_sar(
    vv_arr: Optional[np.ndarray] = None,
    vh_arr: Optional[np.ndarray] = None,
    vv_meta: Optional[Dict[str, Any]] = None,
    vh_meta: Optional[Dict[str, Any]] = None,
    mask_arr: Optional[np.ndarray] = None,
    mask_meta: Optional[Dict[str, Any]] = None,
    input_is_db: bool = False,
    nodata_value: Optional[float] = None,
    dprvi_vegetation_threshold: float = 0.45,
    # Compatibility aliases
    vv_band: Optional[np.ndarray] = None,
    vh_band: Optional[np.ndarray] = None,
    target_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes scientifically valid Sentinel-1 RTC dual-polarization backscatter metrics,
    cross-polarization ratio, and DpRVI (Dual-Polarimetric Radar Vegetation Index).

    Args:
        vv_arr: 2D or 3D array of VV backscatter.
        vh_arr: 2D or 3D array of VH backscatter.
        vv_meta: GeoTIFF metadata for VV (CRS, affine transform, nodata).
        vh_meta: GeoTIFF metadata for VH.
        mask_arr: Optional auxiliary boolean or categorical validity mask.
        mask_meta: Metadata for mask array.
        input_is_db: True if input rasters are already in decibels (dB), False if linear power.
        nodata_value: Explicit nodata sentinel value to mask (e.g. -32768.0).
        dprvi_vegetation_threshold: Canopy threshold for DpRVI volume scattering proxy (default: 0.45).
    """
    if vv_arr is None and vv_band is not None:
        vv_arr = vv_band
    if vh_arr is None and vh_band is not None:
        vh_arr = vh_band
    if vv_meta is None and target_meta is not None:
        vv_meta = target_meta

    if vv_arr is None or vh_arr is None:
        raise ValueError("Both VV and VH arrays are required for Sentinel-1 dual-polarization analysis.")

    # 1. Extract 2D single-band rasters
    if vv_arr.ndim == 3:
        vv_2d = vv_arr[0, :, :] if vv_arr.shape[0] in (1, 2, 3, 4) and vv_arr.shape[0] < vv_arr.shape[1] else vv_arr[:, :, 0]
    elif vv_arr.ndim == 2:
        vv_2d = vv_arr
    else:
        raise ValueError(f"VV array must be 2D or 3D, received shape {vv_arr.shape}")

    if vh_arr.ndim == 3:
        vh_2d = vh_arr[0, :, :] if vh_arr.shape[0] in (1, 2, 3, 4) and vh_arr.shape[0] < vh_arr.shape[1] else vh_arr[:, :, 0]
    elif vh_arr.ndim == 2:
        vh_2d = vh_arr
    else:
        raise ValueError(f"VH array must be 2D or 3D, received shape {vh_arr.shape}")

    h, w = vv_2d.shape[:2]
    total_pixels = int(h * w)

    # 2. Check and align spatial grids if necessary
    aligned_vh = vh_2d
    if vh_2d.shape[:2] != (h, w):
        if vv_meta and vh_meta:
            from app.remote_sensing.registration import align_image_pair
            _, aligned_vh, _ = align_image_pair(vv_2d, vh_2d, vv_meta, vh_meta)
        else:
            raise ValueError(f"Spatial dimension mismatch between VV ({vv_2d.shape}) and VH ({vh_2d.shape})")

    # 3. Handle decibel to linear power conversion if input is dB
    vv_raw = vv_2d.astype(np.float64)
    vh_raw = aligned_vh.astype(np.float64)

    # Detect if metadata indicates dB
    calib_str = str((vv_meta or {}).get("tags", {})).lower() + " " + str(vv_meta or {}).lower()
    is_db_unit = input_is_db or ("db" in calib_str and "linear" not in calib_str)

    # 4. Nodata & Non-finite masking
    finite_mask = np.isfinite(vv_raw) & np.isfinite(vh_raw)

    # Standard Sentinel-1 RTC nodata values
    nodata_mask = np.zeros((h, w), dtype=bool)
    sentinel_nodatas = [-32768.0, 32767.0, -9999.0]
    if nodata_value is not None:
        sentinel_nodatas.append(float(nodata_value))
    for meta in (vv_meta, vh_meta):
        nd = (meta or {}).get("nodata")
        if nd is not None:
            try:
                sentinel_nodatas.append(float(nd))
            except (ValueError, TypeError):
                pass

    for nd in set(sentinel_nodatas):
        nodata_mask |= np.isclose(vv_raw, nd, atol=1e-2)
        nodata_mask |= np.isclose(vh_raw, nd, atol=1e-2)

    # 5. Linear power derivation & positive domain enforcement
    if is_db_unit:
        # Input is dB: valid range typically [-45 dB, +15 dB]
        valid_db_bounds = (vv_raw > -60.0) & (vh_raw > -60.0) & (vv_raw < 30.0) & (vh_raw < 30.0)
        valid_input_mask = finite_mask & (~nodata_mask) & valid_db_bounds
        vv_linear = np.full((h, w), np.nan, dtype=np.float64)
        vh_linear = np.full((h, w), np.nan, dtype=np.float64)
        vv_linear[valid_input_mask] = np.power(10.0, vv_raw[valid_input_mask] / 10.0)
        vh_linear[valid_input_mask] = np.power(10.0, vh_raw[valid_input_mask] / 10.0)
    else:
        # Input is linear backscatter power (gamma0 / sigma0 in m^2/m^2)
        # Strictly positive power required for valid radar physics
        positive_power = (vv_raw > 1e-7) & (vh_raw > 1e-7)
        # Bound extreme unphysical radar power values (> 50.0 linear power = +17 dB)
        physical_bounds = (vv_raw <= 50.0) & (vh_raw <= 50.0)
        valid_input_mask = finite_mask & (~nodata_mask) & positive_power & physical_bounds
        vv_linear = np.full((h, w), np.nan, dtype=np.float64)
        vh_linear = np.full((h, w), np.nan, dtype=np.float64)
        vv_linear[valid_input_mask] = vv_raw[valid_input_mask]
        vh_linear[valid_input_mask] = vh_raw[valid_input_mask]

    # Combine with optional external mask
    valid_mask = valid_input_mask
    if mask_arr is not None:
        if mask_arr.shape[:2] == (h, w):
            m_2d = mask_arr if mask_arr.ndim == 2 else mask_arr[:, :, 0]
            valid_mask = valid_mask & (m_2d.astype(bool))

    valid_pixel_count = int(np.count_nonzero(valid_mask))

    # 6. Dual-Polarization Metrics Computation
    dprvi_arr = np.full((h, w), np.nan, dtype=np.float32)
    cross_pol_ratio_arr = np.full((h, w), np.nan, dtype=np.float32)

    if valid_pixel_count > 0:
        vv_valid = vv_linear[valid_mask]
        vh_valid = vh_linear[valid_mask]

        # Cross-polarization ratio q = VH / VV (linear power)
        q = vh_valid / vv_valid
        cross_pol_ratio_arr[valid_mask] = q.astype(np.float32)

        # Track physical assumption violations where q > 1.0 (VH > VV)
        # Mandal et al. (2020) assumes co-polarized return dominates cross-polarized return (lambda1 = VV >= VH = lambda2).
        # For targets with q > 1.0, the formula exceeds 1.0 prior to clipping; these must be explicitly flagged.
        assumption_violation_mask = q > 1.0
        assumption_violation_count = int(np.count_nonzero(assumption_violation_mask))
        assumption_violation_pct = round((assumption_violation_count / max(1, valid_pixel_count)) * 100.0, 2)

        # DpRVI = (q * (q + 3)) / ((q + 1) ** 2)  (Mandal et al., 2020)
        dprvi_vals = (q * (q + 3.0)) / np.square(q + 1.0)
        # Strictly bound to [0.0, 1.0]
        dprvi_vals = np.clip(dprvi_vals, 0.0, 1.0)
        dprvi_arr[valid_mask] = dprvi_vals.astype(np.float32)
    else:
        assumption_violation_count = 0
        assumption_violation_pct = 0.0

    # 7. Statistical Aggregation
    if valid_pixel_count == 0:
        dprvi_mean = None
        dprvi_p90 = None
        dprvi_median = None
        dprvi_min = None
        dprvi_max = None
        cross_pol_ratio_linear = None
        cross_pol_ratio_db = None
        vv_mean_linear = None
        vh_mean_linear = None
        vv_mean_db = None
        vh_mean_db = None
        vv_pixel_db_mean = None
        vh_pixel_db_mean = None
        vegetation_proxy_coverage_pct = None
        veg_pixel_count = 0
        status = "unsupported_zero_valid_pixels"
        summary = (
            "Calibrated Sentinel-1 RTC dual-polarization analysis unsupported: Zero valid radar pixels available. "
            "All pixels were excluded due to nodata sentinels, non-positive linear power, or out-of-bounds radiometry."
        )
    else:
        dprvi_valid = dprvi_arr[valid_mask]
        q_valid = cross_pol_ratio_arr[valid_mask]
        vv_valid = vv_linear[valid_mask]
        vh_valid = vh_linear[valid_mask]

        dprvi_mean = round(float(np.mean(dprvi_valid)), 4)
        dprvi_p90 = round(float(np.percentile(dprvi_valid, 90)), 4)
        dprvi_median = round(float(np.median(dprvi_valid)), 4)
        dprvi_min = round(float(np.min(dprvi_valid)), 4)
        dprvi_max = round(float(np.max(dprvi_valid)), 4)

        cross_pol_ratio_linear = round(float(np.mean(q_valid)), 4)
        cross_pol_ratio_db = round(float(10.0 * np.log10(max(1e-7, cross_pol_ratio_linear))), 2)

        # Radiometrically correct mean linear power
        vv_mean_linear = round(float(np.mean(vv_valid)), 6)
        vh_mean_linear = round(float(np.mean(vh_valid)), 6)

        # Decibel conversions of mean linear power (true radiometric average)
        vv_mean_db = round(float(10.0 * np.log10(max(1e-7, vv_mean_linear))), 2)
        vh_mean_db = round(float(10.0 * np.log10(max(1e-7, vh_mean_linear))), 2)

        # Mean of pixel-wise dB (explicitly distinguished from mean power in dB)
        vv_pixel_db_mean = round(float(np.mean(10.0 * np.log10(np.maximum(1e-7, vv_valid)))), 2)
        vh_pixel_db_mean = round(float(np.mean(10.0 * np.log10(np.maximum(1e-7, vh_valid)))), 2)

        # Vegetation volume scattering proxy coverage
        veg_pixel_count = int(np.count_nonzero(dprvi_valid >= dprvi_vegetation_threshold))
        vegetation_proxy_coverage_pct = round((veg_pixel_count / valid_pixel_count) * 100.0, 2)
        status = "success"
        clearance_pct = round((valid_pixel_count / max(1, total_pixels)) * 100.0, 1)
        summary = (
            f"Sentinel-1 RTC dual-pol analysis computed across {valid_pixel_count:,} valid radar pixels ({clearance_pct}% scene clearance). "
            f"Mean DpRVI: {dprvi_mean:.4f}, p90: {dprvi_p90:.4f}, cross-pol ratio: {cross_pol_ratio_db:.1f} dB ({cross_pol_ratio_linear:.4f} linear), "
            f"VV: {vv_mean_db:.1f} dB, VH: {vh_mean_db:.1f} dB."
        )

    # 8. Colorized DpRVI Heatmap Preview (Vectorized RGB uint8)
    colorized_rgb = np.full((h, w, 3), 40, dtype=np.uint8)  # Masked pixels: dark gray [40, 40, 40]
    if valid_pixel_count > 0:
        m_specular = valid_mask & (dprvi_arr < 0.20)
        m_moderate = valid_mask & (dprvi_arr >= 0.20) & (dprvi_arr < dprvi_vegetation_threshold)
        m_dense = valid_mask & (dprvi_arr >= dprvi_vegetation_threshold) & (dprvi_arr < 0.70)
        m_volume = valid_mask & (dprvi_arr >= 0.70)

        colorized_rgb[m_specular] = [30, 110, 220]      # Water / smooth bare soil (specular): Cyan/Blue
        colorized_rgb[m_moderate] = [205, 175, 120]     # Sparse vegetation / rough soil: Sand Yellow/Brown
        colorized_rgb[m_dense] = [35, 160, 45]          # Moderate-to-dense crop canopy: Green
        colorized_rgb[m_volume] = [10, 100, 25]         # Dense canopy / forest volume scattering: Emerald Green

    colorized_b64 = convert_array_to_base64_png(colorized_rgb)

    # 9. Document Scientific Limitations
    limitations = [
        "Sentinel-1 RTC backscatter represents radiometrically terrain-corrected gamma0 backscatter power (m^2/m^2).",
        f"DpRVI is calculated as {DPRVI_FORMULA_REFERENCE}.",
        f"Vegetation volume scattering proxy uses explicit scientific threshold DpRVI >= {dprvi_vegetation_threshold:.2f}.",
        "Terrain layover and shadow masks are unavailable from the STAC provider; topographic relief shadows are not masked.",
        "DpRVI characterizes radar volume scattering randomness; it does not replace optical chlorophyll NDVI or identify crop taxonomy in isolation."
    ]
    if assumption_violation_count > 0:
        limitations.append(
            f"Physical assumption notice: {assumption_violation_count:,} valid radar pixels ({assumption_violation_pct}%) had cross-polarization ratio q > 1.0 (VH > VV), violating the dominant co-polarized power assumption of DpRVI (Mandal et al., 2020). These values were clipped to 1.0 and flagged as uncertified volume scattering approximations."
        )

    return {
        "status": status,
        "is_calibrated": True,
        "summary": summary,
        "dprvi_mean": dprvi_mean,
        "dprvi_p90": dprvi_p90,
        "dprvi_median": dprvi_median,
        "dprvi_min": dprvi_min,
        "dprvi_max": dprvi_max,
        "dprvi_assumption_violations": {
            "count": assumption_violation_count,
            "assumption_violation_count": assumption_violation_count,
            "percentage": assumption_violation_pct,
            "assumption_violation_pct": assumption_violation_pct,
            "has_violations": assumption_violation_count > 0,
            "flagged_unphysical_q": assumption_violation_count > 0,
            "warning": "Cross-polarization ratio exceeded 1.0 (VH > VV)" if assumption_violation_count > 0 else None
        },
        "cross_pol_ratio_linear": cross_pol_ratio_linear,
        "cross_pol_ratio_db": cross_pol_ratio_db,
        "vv_mean_linear": vv_mean_linear,
        "vh_mean_linear": vh_mean_linear,
        "vv_mean_db": vv_mean_db,
        "vh_mean_db": vh_mean_db,
        "vv_pixel_db_mean": vv_pixel_db_mean,
        "vh_pixel_db_mean": vh_pixel_db_mean,
        "vegetation_proxy_coverage_pct": vegetation_proxy_coverage_pct,
        "vegetation_threshold": dprvi_vegetation_threshold,
        "veg_pixel_count": veg_pixel_count,
        "valid_pixel_count": valid_pixel_count,
        "total_pixel_count": total_pixels,
        "masked_pixel_count": total_pixels - valid_pixel_count,
        "valid_mask": valid_mask,
        "dprvi_array": dprvi_arr,
        "cross_pol_array": cross_pol_ratio_arr,
        "colorized_preview_base64": colorized_b64,
        "preview_base64": colorized_b64,
        "polarization_mode": "DUAL_VV_VH",
        "calibration": {
            "quantity": "gamma0",
            "convention": "RTC_GAMMA0_LINEAR_POWER",
            "unit": "linear_power_m2_per_m2",
            "input_was_db": is_db_unit
        },
        "validity_policy": {
            "nodata_sentinels_excluded": [-32768.0],
            "non_positive_power_masked": True,
            "sar_terrain_layover_shadow_mask": "unavailable_from_stac_provider"
        },
        "scientific_limitations": limitations
    }

# Compatibility alias
compute_dprvi_and_backscatter = compute_calibrated_dual_pol_sar
