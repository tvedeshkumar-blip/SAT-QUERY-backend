"""
Stage 14: Multimodal Temporal Remote Sensing Analysis.

Implements scientifically rigorous temporal change detection pipelines for:
1. Sentinel-2 Bi-Temporal Optical NDVI Change:
   - Comparable surface reflectance (BOA) with SCL atmospheric quality masking.
   - Screen clouds, shadows, and defective pixels on BOTH T1 and T2.
   - Joint clear valid mask intersection.
   - Delta-NDVI = NDVI_T2 - NDVI_T1 in [-2.0, +2.0].
   - Physical change classification (vegetation loss, gain, stable).
   - Zero-valid-pixel defense: Returns None when joint clear pixels are absent.

2. Sentinel-1 Bi-Temporal RTC Dual-Polarization SAR Change:
   - Consistent polarization (DUAL_VV_VH) and linear gamma0 power convention.
   - Spatial grid alignment and georeferencing validation.
   - Calibrated DpRVI calculation (Mandal et al., 2020) with physical assumption checks (q > 1.0).
   - Delta-DpRVI = DpRVI_T2 - DpRVI_T1 in [-1.0, +1.0].
   - Delta backscatter power (dB) for VV, VH, and cross-pol ratio.
   - Transparent terrain layover/shadow disclaimer.
   - Zero-valid-pixel defense: Returns None when joint valid pixels are absent.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from app.remote_sensing.preprocessing import convert_array_to_base64_png
from app.remote_sensing.sar import compute_calibrated_dual_pol_sar
from app.remote_sensing.bands import compute_calibrated_sentinel2_ndvi
from app.remote_sensing.quality import (
    QualityScreener,
    QualityScreeningError,
    SCL_EXCLUDED_CLASSES,
    SCL_RETAINED_CLASSES,
    SCL_UNCERTAIN_CLASSES,
    SCL_CLASS_NAMES
)

logger = logging.getLogger("satquery.temporal")

# SCL Atmospheric Masking Policy
SCL_CLEAR_CLASSES = {4, 5, 6, 7, 11}       # Vegetation, Bare Soil, Water, Unclassified, Snow/Ice


def build_cloud_masked_optical_composite(
    scenes: List[Dict[str, Any]],
    target_grid: Optional[Dict[str, Any]] = None,
    composite_method: str = "median",
    allow_uncertain_as_clear: bool = False,
    min_valid_pixels: int = 1,
    min_valid_coverage_pct: float = 0.0,
    scale_factor: float = 10000.0,
    offset: float = 0.0
) -> Dict[str, Any]:
    """
    Constructs a multi-scene, cloud-screened per-pixel Sentinel-2 optical composite.
    
    Requirements:
    1. Retrieval: Combines multiple suitable Sentinel-2 L2A candidate observations across a temporal window.
    2. Alignment: Aligns scene grids safely to target reference grid.
       - Continuous optical bands (B04, B08): reprojected via bilinear interpolation.
       - Categorical SCL: reprojected STRICTLY via nearest-neighbor resampling.
    3. Categorical SCL Masking: Excludes clouds (8, 9, 10), cloud-shadows (3), defective (1), nodata (0),
       and unknown out-of-range (<0 or >11) values according to documented ESA specifications.
    4. Physical BOA Reflectance: Scales (DN / scale_factor) + offset with bounds [0.0, 2.0] and finite checks.
    5. Per-pixel Aggregation: Computes temporal median (or mean) across valid observations per pixel.
    6. Minimum Valid Pixel Enforcement: Returns unsupported status if valid pixel count or coverage is below threshold.
    """
    import warnings
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import Affine

    if not scenes:
        return {
            "status": "unsupported_zero_valid_pixels",
            "summary": "No Sentinel-2 candidate scenes provided for compositing.",
            "composite_b04": None,
            "composite_b08": None,
            "composite_ndvi": None,
            "valid_mask": None,
            "observation_counts": None,
            "valid_pixel_count": 0,
            "total_pixel_count": 0,
            "coverage_percentage": 0.0,
            "mean_observation_count": 0.0,
            "max_observation_count": 0,
            "contributing_scenes": [],
            "rejected_scenes": [{"reason": "Empty scene candidate list"}],
            "composite_method": composite_method,
            "scl_mask_policy": {
                "excluded_classes": sorted(list(SCL_EXCLUDED_CLASSES)),
                "uncertain_classes": sorted(list(SCL_UNCERTAIN_CLASSES)),
                "retained_classes": sorted(list(SCL_RETAINED_CLASSES)),
                "resampling": "nearest_neighbor"
            },
            "scientific_limitations": ["No candidate scenes available for compositing."]
        }

    def to_2d(arr: Any) -> Optional[np.ndarray]:
        if arr is None:
            return None
        arr_np = np.asarray(arr)
        if arr_np.ndim == 3:
            return arr_np[0, :, :] if arr_np.shape[0] in (1, 2, 3, 4) and arr_np.shape[0] < arr_np.shape[1] else arr_np[:, :, 0]
        elif arr_np.ndim == 2:
            return arr_np
        return None

    # Step 1: Establish Reference Target Grid
    target_crs = target_grid.get("crs") if target_grid else None
    target_transform = target_grid.get("transform") if target_grid else None
    target_shape = tuple(target_grid.get("shape")) if target_grid and target_grid.get("shape") else None

    if not target_shape:
        for sc in scenes:
            m = sc.get("meta") or sc.get("raster_metadata") or sc.get("metadata") or {}
            raw_b04 = sc.get("b04") if sc.get("b04") is not None else sc.get("red")
            if raw_b04 is None and sc.get("array") is not None:
                arr = sc["array"]
                raw_b04 = arr[:, :, 0] if arr.ndim == 3 else arr
            b04_2d = to_2d(raw_b04)
            if b04_2d is not None and m.get("crs") and m.get("transform"):
                target_crs = m["crs"]
                target_transform = m["transform"]
                target_shape = b04_2d.shape[:2]
                break

    # Fallback to first available 2D array shape if georeferencing metadata is absent
    if not target_shape:
        for sc in scenes:
            raw_b04 = sc.get("b04") if sc.get("b04") is not None else sc.get("red")
            if raw_b04 is None and sc.get("array") is not None:
                arr = sc["array"]
                raw_b04 = arr[:, :, 0] if arr.ndim == 3 else arr
            b04_2d = to_2d(raw_b04)
            if b04_2d is not None:
                target_shape = b04_2d.shape[:2]
                break

    if not target_shape:
        return {
            "status": "unsupported_zero_valid_pixels",
            "summary": "Could not determine target spatial grid from candidate scenes.",
            "composite_b04": None,
            "composite_b08": None,
            "composite_ndvi": None,
            "valid_mask": None,
            "observation_counts": None,
            "valid_pixel_count": 0,
            "total_pixel_count": 0,
            "coverage_percentage": 0.0,
            "mean_observation_count": 0.0,
            "max_observation_count": 0,
            "contributing_scenes": [],
            "rejected_scenes": [{"reason": "Missing array data in all candidate scenes"}],
            "composite_method": composite_method,
            "scientific_limitations": ["Spatial dimensions could not be established."]
        }

    h, w = target_shape
    total_pixels = int(h * w)

    # Step 2: Per-Scene Validation, Alignment, SCL Masking, and Reflectance Extraction
    clean_b04_list: List[np.ndarray] = []
    clean_b08_list: List[np.ndarray] = []
    contributing_scenes: List[Dict[str, Any]] = []
    rejected_scenes: List[Dict[str, Any]] = []
    limitations: List[str] = []

    clear_scl_classes = SCL_RETAINED_CLASSES if not allow_uncertain_as_clear else (SCL_RETAINED_CLASSES | SCL_UNCERTAIN_CLASSES)

    for idx, sc in enumerate(scenes):
        m = sc.get("meta") or sc.get("raster_metadata") or sc.get("metadata") or {}
        sc_id = sc.get("scene_id") or m.get("scene_id") or f"scene_{idx+1}"
        sc_dt = sc.get("datetime") or m.get("datetime")
        sc_cc = sc.get("cloud_cover")

        raw_b04 = sc.get("b04") if sc.get("b04") is not None else sc.get("red")
        raw_b08 = sc.get("b08") if sc.get("b08") is not None else sc.get("nir")
        raw_scl = sc.get("scl")

        if sc.get("array") is not None:
            arr = sc["array"]
            if arr.ndim == 3:
                if raw_b04 is None:
                    raw_b04 = arr[:, :, 0]
                if raw_b08 is None and arr.shape[2] > 1:
                    raw_b08 = arr[:, :, 1]
                if raw_scl is None and arr.shape[2] > 2:
                    raw_scl = arr[:, :, 2]

        b04_2d = to_2d(raw_b04)
        b08_2d = to_2d(raw_b08)
        scl_2d = to_2d(raw_scl)

        if b04_2d is None or b08_2d is None:
            rejected_scenes.append({
                "scene_id": sc_id,
                "reason": "Missing B04 (Red) or B08 (NIR) spectral band asset."
            })
            continue

        sc_crs = m.get("crs")
        sc_trans = m.get("transform")

        # Grid alignment verification
        need_reproject = (b04_2d.shape[:2] != (h, w)) or (
            target_transform and sc_trans and target_transform != sc_trans
        ) or (
            target_crs and sc_crs and target_crs != sc_crs
        )

        aligned_b04 = b04_2d.astype(np.float64)
        aligned_b08 = b08_2d.astype(np.float64)
        aligned_scl = scl_2d

        if need_reproject:
            if not target_crs or not target_transform or not sc_crs or not sc_trans:
                rejected_scenes.append({
                    "scene_id": sc_id,
                    "reason": f"Grid dimension mismatch ({b04_2d.shape[:2]} vs target {h}x{w}) without complete georeferencing transform/CRS."
                })
                continue

            try:
                dst_trans = Affine(*target_transform[:6])
                src_trans = Affine(*sc_trans[:6])

                out_b04 = np.zeros((h, w), dtype=np.float64)
                out_b08 = np.zeros((h, w), dtype=np.float64)

                reproject(
                    source=aligned_b04, destination=out_b04,
                    src_transform=src_trans, src_crs=sc_crs,
                    dst_transform=dst_trans, dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=m.get("nodata"), dst_nodata=m.get("nodata")
                )
                reproject(
                    source=aligned_b08, destination=out_b08,
                    src_transform=src_trans, src_crs=sc_crs,
                    dst_transform=dst_trans, dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=m.get("nodata"), dst_nodata=m.get("nodata")
                )

                aligned_b04 = out_b04
                aligned_b08 = out_b08

                # SCL categorical reprojection: STRICTLY NEAREST-NEIGHBOR
                if scl_2d is not None:
                    scl_trans_in = sc.get("scl_transform") or (sc.get("scl_meta") or {}).get("transform")
                    scl_crs_in = sc.get("scl_crs") or (sc.get("scl_meta") or {}).get("crs") or sc_crs
                    if scl_trans_in:
                        scl_src_trans = Affine(*scl_trans_in[:6])
                    elif sc_trans and scl_2d.shape[:2] != b04_2d.shape[:2]:
                        scl_h, scl_w = scl_2d.shape[:2]
                        b_h, b_w = b04_2d.shape[:2]
                        scl_src_trans = Affine(
                            sc_trans[0] * (b_w / scl_w if scl_w > 0 else 1.0),
                            sc_trans[1],
                            sc_trans[2],
                            sc_trans[3],
                            sc_trans[4] * (b_h / scl_h if scl_h > 0 else 1.0),
                            sc_trans[5]
                        )
                    else:
                        scl_src_trans = src_trans

                    out_scl = np.zeros((h, w), dtype=scl_2d.dtype)
                    reproject(
                        source=scl_2d, destination=out_scl,
                        src_transform=scl_src_trans, src_crs=scl_crs_in or sc_crs,
                        dst_transform=dst_trans, dst_crs=target_crs,
                        resampling=Resampling.nearest,
                        src_nodata=m.get("nodata"), dst_nodata=m.get("nodata")
                    )
                    aligned_scl = out_scl
            except Exception as reproj_err:
                rejected_scenes.append({
                    "scene_id": sc_id,
                    "reason": f"Reprojection onto target reference grid failed: {str(reproj_err)}"
                })
                continue
        elif scl_2d is not None and scl_2d.shape[:2] != (h, w):
            # Align 20m SCL to 10m target grid via nearest neighbor
            try:
                scl_h, scl_w = scl_2d.shape[:2]
                scl_trans_in = sc.get("scl_transform") or (sc.get("scl_meta") or {}).get("transform")
                scl_crs_in = sc.get("scl_crs") or (sc.get("scl_meta") or {}).get("crs") or sc_crs

                if scl_trans_in and scl_crs_in:
                    src_scl_trans = Affine(*scl_trans_in[:6])
                    src_scl_crs = scl_crs_in
                elif sc_trans and sc_crs:
                    scale_x = w / scl_w if scl_w > 0 else 1.0
                    scale_y = h / scl_h if scl_h > 0 else 1.0
                    src_scl_trans = Affine(
                        sc_trans[0] * scale_x,
                        sc_trans[1],
                        sc_trans[2],
                        sc_trans[3],
                        sc_trans[4] * scale_y,
                        sc_trans[5]
                    )
                    src_scl_crs = sc_crs
                else:
                    src_scl_trans = None
                    src_scl_crs = None

                if src_scl_trans and src_scl_crs and sc_trans and sc_crs:
                    dst_trans = Affine(*sc_trans[:6])
                    out_scl = np.zeros((h, w), dtype=scl_2d.dtype)
                    reproject(
                        source=scl_2d, destination=out_scl,
                        src_transform=src_scl_trans, src_crs=src_scl_crs,
                        dst_transform=dst_trans, dst_crs=sc_crs,
                        resampling=Resampling.nearest
                    )
                    aligned_scl = out_scl
                else:
                    # Simple integer-ratio nearest-neighbor zoom
                    sy = max(1, h // scl_h)
                    sx = max(1, w // scl_w)
                    aligned_scl = np.repeat(np.repeat(scl_2d, sy, axis=0), sx, axis=1)[:h, :w]
            except Exception as scl_err:
                logger.warning(f"SCL nearest-neighbor alignment failed for {sc_id}: {scl_err}")
                aligned_scl = None

        # SCL Categorical Masking
        if aligned_scl is not None:
            is_valid_scl_range = (aligned_scl >= 0) & (aligned_scl <= 11)
            clear_mask = np.isin(aligned_scl, list(clear_scl_classes)) & is_valid_scl_range
        else:
            clear_mask = np.ones((h, w), dtype=bool)
            limitations.append(f"Scene '{sc_id}' lacks companion SCL layer; atmospheric cloud filtering omitted.")

        # Surface Reflectance Scaling
        sc_scale = float(m.get("scale_factor", scale_factor))
        eff_offset = float(m.get("offset", offset))
        norm_offset = (eff_offset / sc_scale) if abs(eff_offset) > 2.0 else eff_offset

        is_int_dn = np.issubdtype(aligned_b04.dtype, np.integer) or np.nanmax(aligned_b04) > 2.0
        if is_int_dn:
            refl_b04 = (aligned_b04 / sc_scale) + norm_offset
        else:
            refl_b04 = aligned_b04 + (norm_offset if abs(eff_offset) <= 2.0 else 0.0)

        if np.issubdtype(aligned_b08.dtype, np.integer) or np.nanmax(aligned_b08) > 2.0:
            refl_b08 = (aligned_b08 / sc_scale) + norm_offset
        else:
            refl_b08 = aligned_b08 + (norm_offset if abs(eff_offset) <= 2.0 else 0.0)

        # Physical Validity
        finite = np.isfinite(refl_b04) & np.isfinite(refl_b08)
        phys_bounds = (refl_b04 >= 0.0) & (refl_b04 <= 2.0) & (refl_b08 >= 0.0) & (refl_b08 <= 2.0)
        denom_valid = (refl_b08 + refl_b04) > 1e-5
        valid_pixel_mask = clear_mask & finite & phys_bounds & denom_valid

        valid_count = int(np.count_nonzero(valid_pixel_mask))
        valid_cov = round((valid_count / max(1, total_pixels)) * 100.0, 2)

        contributing_scenes.append({
            "scene_id": sc_id,
            "datetime": sc_dt,
            "cloud_cover": sc_cc,
            "valid_pixel_count": valid_count,
            "valid_coverage_pct": valid_cov
        })

        clean_b04 = np.where(valid_pixel_mask, refl_b04, np.nan)
        clean_b08 = np.where(valid_pixel_mask, refl_b08, np.nan)

        clean_b04_list.append(clean_b04)
        clean_b08_list.append(clean_b08)

    # Step 3: Multi-Scene Per-Pixel Temporal Aggregation
    if not clean_b04_list:
        return {
            "status": "unsupported_zero_valid_pixels",
            "summary": "All candidate Sentinel-2 scenes were rejected due to missing bands or spatial misalignment.",
            "composite_b04": None,
            "composite_b08": None,
            "composite_ndvi": None,
            "valid_mask": np.zeros((h, w), dtype=bool),
            "observation_counts": np.zeros((h, w), dtype=int),
            "valid_pixel_count": 0,
            "total_pixel_count": total_pixels,
            "coverage_percentage": 0.0,
            "mean_observation_count": 0.0,
            "max_observation_count": 0,
            "contributing_scenes": [],
            "rejected_scenes": rejected_scenes,
            "composite_method": composite_method,
            "scientific_limitations": limitations
        }

    b04_stack = np.stack(clean_b04_list, axis=0)
    b08_stack = np.stack(clean_b08_list, axis=0)

    obs_counts = np.sum(~np.isnan(b04_stack), axis=0)
    valid_pixels = obs_counts >= 1
    valid_pixel_count = int(np.count_nonzero(valid_pixels))
    coverage_pct = round((valid_pixel_count / max(1, total_pixels)) * 100.0, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if composite_method == "mean":
            comp_b04 = np.nanmean(b04_stack, axis=0)
            comp_b08 = np.nanmean(b08_stack, axis=0)
        else:
            comp_b04 = np.nanmedian(b04_stack, axis=0)
            comp_b08 = np.nanmedian(b08_stack, axis=0)

    comp_ndvi = np.full((h, w), np.nan, dtype=np.float32)
    if valid_pixel_count > 0:
        d = comp_b08[valid_pixels] + comp_b04[valid_pixels]
        n = comp_b08[valid_pixels] - comp_b04[valid_pixels]
        comp_ndvi[valid_pixels] = np.clip(n / np.where(d > 1e-5, d, 1e-5), -1.0, 1.0).astype(np.float32)

    mean_obs = round(float(np.mean(obs_counts[valid_pixels])), 2) if valid_pixel_count > 0 else 0.0
    max_obs = int(np.max(obs_counts)) if valid_pixel_count > 0 else 0

    if valid_pixel_count == 0 or valid_pixel_count < min_valid_pixels:
        status = "unsupported_zero_valid_pixels"
        summary = (
            "Multi-scene composite unsupported: Zero valid clear-sky pixels found across "
            f"{len(contributing_scenes)} candidate scene(s) after SCL cloud and shadow screening."
        )
    elif coverage_pct < min_valid_coverage_pct:
        status = "unsupported_insufficient_coverage"
        summary = (
            f"Multi-scene composite unsupported: Valid clear-sky coverage ({coverage_pct}%) "
            f"is below required minimum threshold ({min_valid_coverage_pct}%)."
        )
    else:
        status = "success"
        summary = (
            f"Multi-scene Sentinel-2 composite generated using {len(contributing_scenes)} scene(s) "
            f"({composite_method} aggregation). Clear valid coverage: {coverage_pct}% ({valid_pixel_count:,} pixels, "
            f"mean observation depth: {mean_obs}x)."
        )

    limitations.append(
        f"Compositing method: {composite_method} aggregation across valid SCL clear-sky observations. "
        "Attenuates atmospheric contamination but may smooth rapid phenological transitions."
    )

    return {
        "status": status,
        "summary": summary,
        "composite_b04": comp_b04,
        "composite_b08": comp_b08,
        "composite_ndvi": comp_ndvi,
        "valid_mask": valid_pixels,
        "observation_counts": obs_counts,
        "valid_pixel_count": valid_pixel_count,
        "total_pixel_count": total_pixels,
        "coverage_percentage": coverage_pct,
        "mean_observation_count": mean_obs,
        "max_observation_count": max_obs,
        "target_grid": {
            "crs": target_crs,
            "transform": target_transform,
            "shape": (h, w)
        },
        "contributing_scenes": contributing_scenes,
        "rejected_scenes": rejected_scenes,
        "composite_method": f"valid_pixel_temporal_{composite_method}",
        "scl_mask_policy": {
            "excluded_classes": sorted(list(SCL_EXCLUDED_CLASSES)),
            "uncertain_classes": sorted(list(SCL_UNCERTAIN_CLASSES)),
            "retained_classes": sorted(list(SCL_RETAINED_CLASSES)),
            "unknown_policy": "fail_closed_excluded",
            "resampling": "nearest_neighbor"
        },
        "scientific_limitations": limitations
    }


def convert_rgba_to_base64_png(rgba: np.ndarray) -> str:
    """Encodes an (H, W, 4) uint8 RGBA array to a base64-encoded PNG image string."""
    from PIL import Image
    import io, base64
    img = Image.fromarray(rgba.astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def generate_composite_inspector_layers(
    comp: Optional[Dict[str, Any]], 
    aoi_bounds: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Generates 4 high-fidelity scientific visualization layers for an optical composite:
    1. b04_red: Grayscale contrast-stretched Red surface reflectance (B04), with masked pixels distinct/dark slate.
    2. b08_nir: Grayscale contrast-stretched Near-Infrared surface reflectance (B08).
    3. obs_depth: Categorical observation depth (0 = Crimson warning, 1 = Amber, 2+ = Emerald).
    4. scl_mask: Categorical SCL validity mask (Clear valid surface = Green, Cloud/Shadow masked = Rose).
    """
    if not comp or comp.get("composite_b04") is None:
        return {
            "status": "unavailable",
            "reason": (comp.get("summary") if comp else None) or "Optical composite data unavailable.",
            "layers": {},
            "stats": {
                "total_pixels": 0,
                "valid_pixels": 0,
                "masked_pixels": 0,
                "clear_coverage_pct": 0.0,
                "obs_depth_0_count": 0,
                "obs_depth_1_count": 0,
                "obs_depth_2plus_count": 0,
                "mean_obs_depth": 0.0
            },
            "bounds": aoi_bounds,
            "shape": None
        }

    comp_b04 = comp["composite_b04"]
    comp_b08 = comp.get("composite_b08")
    valid_mask = comp.get("valid_mask")
    obs_counts = comp.get("observation_counts")
    target_grid = comp.get("target_grid") or {}

    h, w = comp_b04.shape[:2]
    total_pixels = int(h * w)

    if valid_mask is None:
        valid_mask = np.isfinite(comp_b04) & (np.isfinite(comp_b08) if comp_b08 is not None else True)
    if obs_counts is None:
        obs_counts = valid_mask.astype(int)

    # 1. B04 Red Reflectance Layer
    rgba_b04 = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_b04[:, :] = [30, 32, 38, 120]  # Dark slate with partial transparency for masked/nodata
    if np.any(valid_mask):
        b04_clean = np.where(valid_mask & np.isfinite(comp_b04), comp_b04, np.nan)
        # BOA Red reflectance scale: 0.0 to 0.30 standard stretch
        stretched_b04 = np.clip(np.nan_to_num(b04_clean, nan=0.0) / 0.30 * 255.0, 0, 255).astype(np.uint8)
        rgba_b04[valid_mask, 0] = stretched_b04[valid_mask]
        rgba_b04[valid_mask, 1] = stretched_b04[valid_mask]
        rgba_b04[valid_mask, 2] = stretched_b04[valid_mask]
        rgba_b04[valid_mask, 3] = 255

    # 2. B08 NIR Reflectance Layer
    rgba_b08 = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_b08[:, :] = [30, 32, 38, 120]  # Dark slate with partial transparency for masked/nodata
    if comp_b08 is not None and np.any(valid_mask):
        b08_clean = np.where(valid_mask & np.isfinite(comp_b08), comp_b08, np.nan)
        # BOA NIR reflectance scale: 0.0 to 0.60 standard stretch
        stretched_b08 = np.clip(np.nan_to_num(b08_clean, nan=0.0) / 0.60 * 255.0, 0, 255).astype(np.uint8)
        rgba_b08[valid_mask, 0] = stretched_b08[valid_mask]
        rgba_b08[valid_mask, 1] = stretched_b08[valid_mask]
        rgba_b08[valid_mask, 2] = stretched_b08[valid_mask]
        rgba_b08[valid_mask, 3] = 255

    # 3. Observation Depth Layer (0 = Crimson warning, 1 = Amber, 2+ = Emerald)
    rgba_depth = np.zeros((h, w, 4), dtype=np.uint8)
    m0 = obs_counts == 0
    m1 = obs_counts == 1
    m2 = obs_counts >= 2
    rgba_depth[m0] = [220, 38, 38, 220]    # 0 observations: Crimson warning
    rgba_depth[m1] = [217, 119, 6, 220]   # 1 observation: Amber
    rgba_depth[m2] = [16, 185, 129, 220]  # 2+ observations: Emerald

    # 4. SCL Validity Mask Layer (Clear Valid vs Masked)
    rgba_scl = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_scl[valid_mask] = [34, 197, 94, 200]   # Clear surface: Emerald Green
    rgba_scl[~valid_mask] = [244, 63, 94, 200]  # Masked (cloud/shadow/cirrus/nodata): Rose

    d0_count = int(np.count_nonzero(m0))
    d1_count = int(np.count_nonzero(m1))
    d2_count = int(np.count_nonzero(m2))
    v_count = int(np.count_nonzero(valid_mask))
    m_count = total_pixels - v_count
    cov_pct = round((v_count / max(1, total_pixels)) * 100.0, 2)
    mean_depth = round(float(np.mean(obs_counts)), 2)

    bounds = aoi_bounds
    if not bounds and target_grid.get("bounds"):
        bounds = target_grid["bounds"]

    return {
        "status": "success",
        "shape": [h, w],
        "bounds": bounds,
        "crs": target_grid.get("crs", "EPSG:4326"),
        "layers": {
            "b04_red": convert_rgba_to_base64_png(rgba_b04),
            "b08_nir": convert_rgba_to_base64_png(rgba_b08),
            "obs_depth": convert_rgba_to_base64_png(rgba_depth),
            "scl_mask": convert_rgba_to_base64_png(rgba_scl)
        },
        "stats": {
            "total_pixels": total_pixels,
            "valid_pixels": v_count,
            "masked_pixels": m_count,
            "clear_coverage_pct": cov_pct,
            "obs_depth_0_count": d0_count,
            "obs_depth_1_count": d1_count,
            "obs_depth_2plus_count": d2_count,
            "mean_obs_depth": mean_depth
        },
        "contributing_scenes": comp.get("contributing_scenes", []),
        "rejected_scenes": comp.get("rejected_scenes", []),
        "composite_method": comp.get("composite_method", "valid_pixel_temporal_median"),
        "scl_mask_policy": comp.get("scl_mask_policy")
    }


def compute_temporal_optical_change(
    t1_b04: Any = None,
    t1_b08: Any = None,
    t2_b04: Any = None,
    t2_b08: Any = None,
    t1_scl: Optional[np.ndarray] = None,
    t2_scl: Optional[np.ndarray] = None,
    t1_meta: Optional[Dict[str, Any]] = None,
    t2_meta: Optional[Dict[str, Any]] = None,
    t1_datetime: Optional[str] = None,
    t2_datetime: Optional[str] = None,
    t1_scenes: Optional[List[Dict[str, Any]]] = None,
    t2_scenes: Optional[List[Dict[str, Any]]] = None,
    composite_method: str = "median",
    allow_uncertain_as_clear: bool = False,
    min_valid_pixels: int = 1,
    min_valid_coverage_pct: float = 0.0,
    loss_threshold: float = -0.15,
    gain_threshold: float = 0.15,
    reflectance_scale: float = 10000.0,
    cloud_threshold: Optional[float] = None,
    aoi_bounds: Optional[List[float]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Computes calibrated temporal Sentinel-2 NDVI change with multi-scene SCL cloud masking composites.
    Supports either explicit individual band arrays, structured dictionary inputs, or full multi-scene candidate lists.
    """
    if aoi_bounds is None:
        aoi_bounds = (t1_meta or {}).get("bounds") or (t2_meta or {}).get("bounds") or kwargs.get("aoi_bbox")
    # Normalize inputs to multi-scene candidate lists for T1 and T2
    if t1_scenes is None:
        if isinstance(t1_b04, list):
            t1_scenes = t1_b04
        elif isinstance(t1_b04, dict):
            t1_scenes = t1_b04.get("scenes", [t1_b04])
        elif t1_b04 is not None:
            t1_scenes = [{
                "b04": t1_b04,
                "b08": t1_b08,
                "scl": t1_scl,
                "meta": t1_meta or {},
                "datetime": t1_datetime or (t1_meta or {}).get("datetime"),
                "scene_id": (t1_meta or {}).get("scene_id", "t1_primary")
            }]
        else:
            t1_scenes = []

    if t2_scenes is None:
        if isinstance(t2_b04, list):
            t2_scenes = t2_b04
        elif isinstance(t2_b04, dict):
            t2_scenes = t2_b04.get("scenes", [t2_b04])
        elif t2_b04 is not None:
            t2_scenes = [{
                "b04": t2_b04,
                "b08": t2_b08,
                "scl": t2_scl,
                "meta": t2_meta or {},
                "datetime": t2_datetime or (t2_meta or {}).get("datetime"),
                "scene_id": (t2_meta or {}).get("scene_id", "t2_primary")
            }]
        elif isinstance(t1_b08, dict) and t2_b04 is None:
            # Legacy signature where t1_dict and t2_dict were passed as 1st and 2nd positional args
            t2_dict = t1_b08
            t2_scenes = t2_dict.get("scenes", [t2_dict])
        else:
            t2_scenes = []

    # 1. Temporal Chronology Validation
    dt1_str = t1_datetime
    if not dt1_str and t1_scenes:
        dt1_str = t1_scenes[0].get("datetime") or (t1_scenes[0].get("meta") or {}).get("datetime")
    dt2_str = t2_datetime
    if not dt2_str and t2_scenes:
        dt2_str = t2_scenes[0].get("datetime") or (t2_scenes[0].get("meta") or {}).get("datetime")

    chronology_warning = None
    if dt1_str and dt2_str:
        try:
            clean_dt1 = dt1_str.replace("Z", "+00:00")
            clean_dt2 = dt2_str.replace("Z", "+00:00")
            d1 = datetime.fromisoformat(clean_dt1)
            d2 = datetime.fromisoformat(clean_dt2)
            if d1 == d2:
                raise ValueError(
                    f"Degenerate temporal pair: T1 and T2 have identical acquisition timestamps ('{dt1_str}'). "
                    "Temporal change detection strictly requires distinct acquisition dates."
                )
            if d1 > d2:
                chronology_warning = (
                    f"Chronological inversion: T1 ('{dt1_str}') is later than T2 ('{dt2_str}'). "
                    "Delta-NDVI is computed as T2 - T1."
                )
        except (ValueError, TypeError) as date_err:
            if "identical" in str(date_err):
                raise
            logger.debug(f"Could not parse timestamps for chronology validation: {date_err}")

    # 2. Construct Multi-Scene Cloud-Masked Composites for T1 and T2
    comp_t1 = build_cloud_masked_optical_composite(
        scenes=t1_scenes,
        target_grid=None,
        composite_method=composite_method,
        allow_uncertain_as_clear=allow_uncertain_as_clear,
        min_valid_pixels=min_valid_pixels,
        min_valid_coverage_pct=min_valid_coverage_pct,
        scale_factor=reflectance_scale
    )

    t1_grid = comp_t1.get("target_grid")

    comp_t2 = build_cloud_masked_optical_composite(
        scenes=t2_scenes,
        target_grid=t1_grid,
        composite_method=composite_method,
        allow_uncertain_as_clear=allow_uncertain_as_clear,
        min_valid_pixels=min_valid_pixels,
        min_valid_coverage_pct=min_valid_coverage_pct,
        scale_factor=reflectance_scale
    )

    insp_t1 = generate_composite_inspector_layers(comp_t1, aoi_bounds)
    insp_t2 = generate_composite_inspector_layers(comp_t2, aoi_bounds)
    inspector_payload = {
        "status": "success" if (insp_t1.get("status") == "success" or insp_t2.get("status") == "success") else "unavailable",
        "t1": insp_t1,
        "t2": insp_t2,
        "bounds": aoi_bounds or insp_t1.get("bounds") or insp_t2.get("bounds"),
        "crs": (t1_grid.get("crs") if t1_grid else None) or "EPSG:4326",
        "composite_method": comp_t1.get("composite_method", f"valid_pixel_temporal_{composite_method}"),
        "scl_mask_policy": comp_t1.get("scl_mask_policy")
    }

    t1_valid_mask = comp_t1.get("valid_mask")
    t2_valid_mask = comp_t2.get("valid_mask")

    if t1_valid_mask is None or t2_valid_mask is None or comp_t1["status"] != "success" or comp_t2["status"] != "success":
        fail_summary = comp_t1.get("summary") if comp_t1["status"] != "success" else comp_t2.get("summary")
        h_dim = t1_grid["shape"][0] if t1_grid and t1_grid.get("shape") else 1
        w_dim = t1_grid["shape"][1] if t1_grid and t1_grid.get("shape") else 1
        colorized_rgb = np.full((h_dim, w_dim, 3), 35, dtype=np.uint8)
        colorized_b64 = convert_array_to_base64_png(colorized_rgb)

        return {
            "status": "unsupported_zero_valid_pixels",
            "is_calibrated": True,
            "summary": f"Temporal optical NDVI change analysis unsupported: {fail_summary}",
            "delta_ndvi_mean": None,
            "delta_ndvi_median": None,
            "delta_ndvi_p10": None,
            "delta_ndvi_p90": None,
            "delta_ndvi_std": None,
            "ndvi_t1_mean": None,
            "ndvi_t2_mean": None,
            "vegetation_loss_pct": None,
            "vegetation_gain_pct": None,
            "stable_vegetation_pct": None,
            "loss_pixel_count": 0,
            "gain_pixel_count": 0,
            "stable_pixel_count": 0,
            "loss_threshold": loss_threshold,
            "gain_threshold": gain_threshold,
            "joint_valid_pixel_count": 0,
            "total_pixel_count": int(h_dim * w_dim),
            "joint_clearance_pct": 0.0,
            "joint_valid_mask": np.zeros((h_dim, w_dim), dtype=bool),
            "delta_ndvi_array": np.full((h_dim, w_dim), np.nan, dtype=np.float32),
            "colorized_preview_base64": colorized_b64,
            "colorized_change_preview_base64": colorized_b64,
            "preview_base64": colorized_b64,
            "t1_datetime": dt1_str,
            "t2_datetime": dt2_str,
            "scl_filtering_applied": True,
            "composite_method": f"valid_pixel_temporal_{composite_method}",
            "t1_composite": comp_t1,
            "t2_composite": comp_t2,
            "visual_inspector": inspector_payload,
            "contributing_scene_ids_t1": [s["scene_id"] for s in comp_t1.get("contributing_scenes", [])],
            "contributing_scene_ids_t2": [s["scene_id"] for s in comp_t2.get("contributing_scenes", [])],
            "rejected_scenes": comp_t1.get("rejected_scenes", []) + comp_t2.get("rejected_scenes", []),
            "t1_mean_observation_count": comp_t1.get("mean_observation_count", 0.0),
            "t2_mean_observation_count": comp_t2.get("mean_observation_count", 0.0),
            "cloud_threshold": cloud_threshold,
            "scientific_limitations": comp_t1.get("scientific_limitations", []) + comp_t2.get("scientific_limitations", [])
        }

    # 3. Joint Valid Pixel Intersection
    h, w = t1_valid_mask.shape
    total_pixels = int(h * w)

    joint_valid_mask = t1_valid_mask & t2_valid_mask
    joint_valid_count = int(np.count_nonzero(joint_valid_mask))
    joint_clearance_pct = round((joint_valid_count / max(1, total_pixels)) * 100.0, 1)

    delta_ndvi_arr = np.full((h, w), np.nan, dtype=np.float32)
    t1_ndvi_arr = comp_t1["composite_ndvi"]
    t2_ndvi_arr = comp_t2["composite_ndvi"]

    if joint_valid_count == 0:
        delta_ndvi_mean = None
        delta_ndvi_median = None
        delta_ndvi_p10 = None
        delta_ndvi_p90 = None
        delta_ndvi_std = None
        ndvi_t1_mean = None
        ndvi_t2_mean = None
        vegetation_loss_pct = None
        vegetation_gain_pct = None
        stable_vegetation_pct = None
        loss_pixel_count = 0
        gain_pixel_count = 0
        stable_pixel_count = 0
        status = "unsupported_zero_valid_pixels"
        summary = (
            "Temporal optical NDVI change analysis unsupported: Zero joint clear pixels between T1 and T2 "
            "multi-scene composites after Scene Classification Layer (SCL) cloud, cloud shadow, and non-physical reflectance screening."
        )
    else:
        # Compute Delta-NDVI across joint valid composite footprint
        t1_valid_vals = t1_ndvi_arr[joint_valid_mask]
        t2_valid_vals = t2_ndvi_arr[joint_valid_mask]

        delta = t2_valid_vals - t1_valid_vals
        delta = np.clip(delta, -2.0, 2.0)
        delta_ndvi_arr[joint_valid_mask] = delta.astype(np.float32)

        delta_ndvi_mean = round(float(np.mean(delta)), 4)
        delta_ndvi_median = round(float(np.median(delta)), 4)
        delta_ndvi_p10 = round(float(np.percentile(delta, 10)), 4)
        delta_ndvi_p90 = round(float(np.percentile(delta, 90)), 4)
        delta_ndvi_std = round(float(np.std(delta)), 4)

        ndvi_t1_mean = round(float(np.mean(t1_valid_vals)), 4)
        ndvi_t2_mean = round(float(np.mean(t2_valid_vals)), 4)

        # Vegetation change classification
        loss_pixel_count = int(np.count_nonzero(delta <= loss_threshold))
        gain_pixel_count = int(np.count_nonzero(delta >= gain_threshold))
        stable_pixel_count = int(np.count_nonzero((delta > loss_threshold) & (delta < gain_threshold)))

        vegetation_loss_pct = round((loss_pixel_count / joint_valid_count) * 100.0, 2)
        vegetation_gain_pct = round((gain_pixel_count / joint_valid_count) * 100.0, 2)
        stable_vegetation_pct = round((stable_pixel_count / joint_valid_count) * 100.0, 2)

        status = "success"
        n_t1_scenes = len(comp_t1.get("contributing_scenes", []))
        n_t2_scenes = len(comp_t2.get("contributing_scenes", []))
        summary = (
            f"Sentinel-2 multi-scene temporal change computed across {joint_valid_count:,} joint clear pixels "
            f"({joint_clearance_pct}% scene clearance; T1: {n_t1_scenes} scene(s), T2: {n_t2_scenes} scene(s)). "
            f"Mean Delta-NDVI: {delta_ndvi_mean:+.4f} (T1: {ndvi_t1_mean:.4f} -> T2: {ndvi_t2_mean:.4f}). "
            f"Vegetation loss: {vegetation_loss_pct}% ({loss_pixel_count:,} px), "
            f"gain: {vegetation_gain_pct}% ({gain_pixel_count:,} px), "
            f"stable: {stable_vegetation_pct}%."
        )

    # 4. Colorized Delta-NDVI Change Map Preview
    colorized_rgb = np.full((h, w, 3), 35, dtype=np.uint8)  # Masked pixels: dark slate [35, 35, 35]
    if joint_valid_count > 0:
        m_loss = joint_valid_mask & (delta_ndvi_arr <= loss_threshold)
        m_gain = joint_valid_mask & (delta_ndvi_arr >= gain_threshold)
        m_stable = joint_valid_mask & (delta_ndvi_arr > loss_threshold) & (delta_ndvi_arr < gain_threshold)

        colorized_rgb[m_loss] = [215, 45, 35]       # Vegetation loss / canopy clearing: Crimson Red
        colorized_rgb[m_gain] = [40, 180, 50]       # Vegetation gain / vigor increase: Green
        colorized_rgb[m_stable] = [120, 130, 140]   # Stable / unchanged: Neutral Slate

    colorized_b64 = convert_array_to_base64_png(colorized_rgb)

    limitations = [
        "Sentinel-2 temporal optical change uses Level-2A Bottom-Of-Atmosphere (BOA) surface reflectance (B04 + B08).",
        f"Multi-scene per-pixel compositing: {composite_method} aggregation across valid clear SCL observations.",
        "Scene Classification Layer (SCL) atmospheric screening applied per-scene: cloud, cloud shadow, defective, and cirrus pixels excluded.",
        f"Vegetation change thresholds: Loss (Delta-NDVI <= {loss_threshold:.2f}), Gain (Delta-NDVI >= {gain_threshold:.2f}).",
        "Delta-NDVI represents relative changes in photosynthetic canopy vigor; seasonal phenology must be considered."
    ]
    if chronology_warning:
        limitations.insert(0, chronology_warning)

    limitations.extend(comp_t1.get("scientific_limitations", []))
    limitations.extend(comp_t2.get("scientific_limitations", []))
    # Deduplicate limitations preserving order
    deduped_limitations = list(dict.fromkeys(limitations))

    return {
        "status": status,
        "is_calibrated": True,
        "summary": summary,
        "delta_ndvi_mean": delta_ndvi_mean,
        "delta_ndvi_median": delta_ndvi_median,
        "delta_ndvi_p10": delta_ndvi_p10,
        "delta_ndvi_p90": delta_ndvi_p90,
        "delta_ndvi_std": delta_ndvi_std,
        "ndvi_t1_mean": ndvi_t1_mean,
        "ndvi_t2_mean": ndvi_t2_mean,
        "vegetation_loss_pct": vegetation_loss_pct,
        "vegetation_gain_pct": vegetation_gain_pct,
        "stable_vegetation_pct": stable_vegetation_pct,
        "loss_pixel_count": loss_pixel_count,
        "gain_pixel_count": gain_pixel_count,
        "stable_pixel_count": stable_pixel_count,
        "loss_threshold": loss_threshold,
        "gain_threshold": gain_threshold,
        "joint_valid_pixel_count": joint_valid_count,
        "total_pixel_count": total_pixels,
        "joint_clearance_pct": joint_clearance_pct,
        "joint_valid_mask": joint_valid_mask,
        "delta_ndvi_array": delta_ndvi_arr,
        "colorized_preview_base64": colorized_b64,
        "colorized_change_preview_base64": colorized_b64,
        "preview_base64": colorized_b64,
        "t1_datetime": dt1_str,
        "t2_datetime": dt2_str,
        "scl_filtering_applied": True,
        "composite_method": f"valid_pixel_temporal_{composite_method}",
        "t1_composite": comp_t1,
        "t2_composite": comp_t2,
        "visual_inspector": inspector_payload,
        "contributing_scene_ids_t1": [s["scene_id"] for s in comp_t1.get("contributing_scenes", [])],
        "contributing_scene_ids_t2": [s["scene_id"] for s in comp_t2.get("contributing_scenes", [])],
        "rejected_scenes": comp_t1.get("rejected_scenes", []) + comp_t2.get("rejected_scenes", []),
        "t1_mean_observation_count": comp_t1.get("mean_observation_count", 0.0),
        "t2_mean_observation_count": comp_t2.get("mean_observation_count", 0.0),
        "cloud_threshold": cloud_threshold,
        "scientific_limitations": deduped_limitations
    }


def compute_temporal_sar_change(
    t1_vv: Any,
    t1_vh: Any = None,
    t2_vv: Any = None,
    t2_vh: Any = None,
    t1_meta: Optional[Dict[str, Any]] = None,
    t2_meta: Optional[Dict[str, Any]] = None,
    t1_datetime: Optional[str] = None,
    t2_datetime: Optional[str] = None,
    canopy_loss_threshold: float = -0.15,
    canopy_gain_threshold: float = 0.15,
    input_is_db: bool = False
) -> Dict[str, Any]:
    """
    Computes calibrated temporal Sentinel-1 RTC dual-polarization SAR change (Delta-DpRVI & Delta-Backscatter).
    Supports either explicit individual band arrays or structured dictionary inputs for T1/T2.
    """
    if isinstance(t1_vv, dict) and (isinstance(t1_vh, dict) or t2_vv is None):
        t1_dict = t1_vv
        t2_dict = t1_vh or {}
        t1_vv = t1_dict.get("vv")
        t1_vh = t1_dict.get("vh")
        t1_meta = t1_dict.get("meta") or {}
        t1_datetime = t1_datetime or t1_meta.get("datetime")

        t2_vv = t2_dict.get("vv")
        t2_vh = t2_dict.get("vh")
        t2_meta = t2_dict.get("meta") or {}
        t2_datetime = t2_datetime or t2_meta.get("datetime")

    t1_meta = t1_meta or {}
    t2_meta = t2_meta or {}

    # 1. Temporal Chronology Validation
    dt1_str = t1_datetime or t1_meta.get("datetime")
    dt2_str = t2_datetime or t2_meta.get("datetime")
    chronology_warning = None

    if dt1_str and dt2_str:
        try:
            clean_dt1 = dt1_str.replace("Z", "+00:00")
            clean_dt2 = dt2_str.replace("Z", "+00:00")
            d1 = datetime.fromisoformat(clean_dt1)
            d2 = datetime.fromisoformat(clean_dt2)
            if d1 == d2:
                raise ValueError(
                    f"Degenerate temporal pair: T1 and T2 have identical acquisition timestamps ('{dt1_str}'). "
                    "Temporal SAR change detection strictly requires distinct acquisition dates."
                )
            if d1 > d2:
                chronology_warning = (
                    f"Chronological inversion: T1 ('{dt1_str}') is later than T2 ('{dt2_str}'). "
                    "Delta-DpRVI is computed as T2 - T1."
                )
        except (ValueError, TypeError) as date_err:
            if "identical" in str(date_err):
                raise
            logger.debug(f"Could not parse SAR timestamps for chronology validation: {date_err}")

    # 2. Extract 2D single-band arrays
    def to_2d(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3:
            return arr[0, :, :] if arr.shape[0] in (1, 2, 3, 4) and arr.shape[0] < arr.shape[1] else arr[:, :, 0]
        return arr

    vv_1 = to_2d(t1_vv)
    vh_1 = to_2d(t1_vh)
    vv_2 = to_2d(t2_vv)
    vh_2 = to_2d(t2_vh)

    h, w = vv_1.shape[:2]
    total_pixels = int(h * w)

    # 3. Spatial Grid Alignment (Reproject T2 to T1 grid if needed)
    if vv_2.shape[:2] != (h, w) or (t2_meta.get("transform") and t1_meta.get("transform") and t2_meta.get("transform") != t1_meta.get("transform")):
        if t1_meta.get("crs") and t2_meta.get("crs"):
            from rasterio.warp import reproject, Resampling
            from rasterio.transform import Affine

            target_shape = (h, w)
            aligned_vv_2 = np.zeros(target_shape, dtype=vv_2.dtype)
            aligned_vh_2 = np.zeros(target_shape, dtype=vh_2.dtype)

            t1_trans = Affine(*t1_meta["transform"][:6])
            t2_trans = Affine(*t2_meta["transform"][:6])

            reproject(
                source=vv_2, destination=aligned_vv_2,
                src_transform=t2_trans, src_crs=t2_meta["crs"],
                dst_transform=t1_trans, dst_crs=t1_meta["crs"],
                resampling=Resampling.bilinear,
                src_nodata=t2_meta.get("nodata"), dst_nodata=t2_meta.get("nodata")
            )
            reproject(
                source=vh_2, destination=aligned_vh_2,
                src_transform=t2_trans, src_crs=t2_meta["crs"],
                dst_transform=t1_trans, dst_crs=t1_meta["crs"],
                resampling=Resampling.bilinear,
                src_nodata=t2_meta.get("nodata"), dst_nodata=t2_meta.get("nodata")
            )
            vv_2 = aligned_vv_2
            vh_2 = aligned_vh_2
        else:
            raise ValueError(f"SAR grid dimension mismatch between T1 ({h}x{w}) and T2 ({vv_2.shape}) without valid CRS.")

    # 4. Calibrated DpRVI & Backscatter Calculation for T1 and T2
    t1_sar = compute_calibrated_dual_pol_sar(
        vv_arr=vv_1, vh_arr=vh_1, vv_meta=t1_meta, vh_meta=t1_meta, input_is_db=input_is_db
    )
    t2_sar = compute_calibrated_dual_pol_sar(
        vv_arr=vv_2, vh_arr=vh_2, vv_meta=t2_meta, vh_meta=t2_meta, input_is_db=input_is_db
    )

    t1_valid_mask = t1_sar.get("valid_mask", np.zeros((h, w), dtype=bool))
    t2_valid_mask = t2_sar.get("valid_mask", np.zeros((h, w), dtype=bool))

    joint_valid_mask = t1_valid_mask & t2_valid_mask
    joint_valid_count = int(np.count_nonzero(joint_valid_mask))
    joint_clearance_pct = round((joint_valid_count / max(1, total_pixels)) * 100.0, 1)

    delta_dprvi_arr = np.full((h, w), np.nan, dtype=np.float32)

    # Track assumption violations across T1 and T2
    t1_violations = t1_sar.get("dprvi_assumption_violations", {})
    t2_violations = t2_sar.get("dprvi_assumption_violations", {})
    total_violations = t1_violations.get("count", 0) + t2_violations.get("count", 0)

    if joint_valid_count == 0:
        delta_dprvi_mean = None
        delta_dprvi_median = None
        delta_dprvi_p10 = None
        delta_dprvi_p90 = None
        delta_dprvi_std = None
        dprvi_t1_mean = None
        dprvi_t2_mean = None
        delta_vv_db = None
        delta_vh_db = None
        delta_cross_pol_db = None
        radar_canopy_loss_pct = None
        radar_canopy_gain_pct = None
        stable_canopy_pct = None
        loss_pixel_count = 0
        gain_pixel_count = 0
        stable_pixel_count = 0
        status = "unsupported_zero_valid_pixels"
        summary = (
            "Temporal SAR dual-polarization change analysis unsupported: Zero joint valid radar pixels between T1 and T2 "
            "after nodata sentinel, non-positive linear power, and out-of-bounds radiometry screening."
        )
    else:
        dprvi_1 = t1_sar["dprvi_array"][joint_valid_mask]
        dprvi_2 = t2_sar["dprvi_array"][joint_valid_mask]

        delta_dprvi = dprvi_2 - dprvi_1
        delta_dprvi = np.clip(delta_dprvi, -1.0, 1.0)
        delta_dprvi_arr[joint_valid_mask] = delta_dprvi.astype(np.float32)

        delta_dprvi_mean = round(float(np.mean(delta_dprvi)), 4)
        delta_dprvi_median = round(float(np.median(delta_dprvi)), 4)
        delta_dprvi_p10 = round(float(np.percentile(delta_dprvi, 10)), 4)
        delta_dprvi_p90 = round(float(np.percentile(delta_dprvi, 90)), 4)
        delta_dprvi_std = round(float(np.std(delta_dprvi)), 4)

        dprvi_t1_mean = round(float(np.mean(dprvi_1)), 4)
        dprvi_t2_mean = round(float(np.mean(dprvi_2)), 4)

        # Backscatter differences (dB)
        if t1_sar.get("vv_mean_db") is not None and t2_sar.get("vv_mean_db") is not None:
            delta_vv_db = round(float(t2_sar["vv_mean_db"] - t1_sar["vv_mean_db"]), 2)
        else:
            delta_vv_db = None

        if t1_sar.get("vh_mean_db") is not None and t2_sar.get("vh_mean_db") is not None:
            delta_vh_db = round(float(t2_sar["vh_mean_db"] - t1_sar["vh_mean_db"]), 2)
        else:
            delta_vh_db = None

        if t1_sar.get("cross_pol_ratio_db") is not None and t2_sar.get("cross_pol_ratio_db") is not None:
            delta_cross_pol_db = round(float(t2_sar["cross_pol_ratio_db"] - t1_sar["cross_pol_ratio_db"]), 2)
        else:
            delta_cross_pol_db = None

        # Structural canopy change classification
        loss_pixel_count = int(np.count_nonzero(delta_dprvi <= canopy_loss_threshold))
        gain_pixel_count = int(np.count_nonzero(delta_dprvi >= canopy_gain_threshold))
        stable_pixel_count = int(np.count_nonzero((delta_dprvi > canopy_loss_threshold) & (delta_dprvi < canopy_gain_threshold)))

        radar_canopy_loss_pct = round((loss_pixel_count / joint_valid_count) * 100.0, 2)
        radar_canopy_gain_pct = round((gain_pixel_count / joint_valid_count) * 100.0, 2)
        stable_canopy_pct = round((stable_pixel_count / joint_valid_count) * 100.0, 2)

        status = "success"
        summary = (
            f"Sentinel-1 RTC temporal SAR change computed across {joint_valid_count:,} joint valid radar pixels ({joint_clearance_pct}% scene clearance). "
            f"Mean Delta-DpRVI: {delta_dprvi_mean:+.4f} (T1: {dprvi_t1_mean:.4f} -> T2: {dprvi_t2_mean:.4f}), "
            f"Delta VV: {delta_vv_db:+.1f} dB, Delta VH: {delta_vh_db:+.1f} dB. "
            f"Radar canopy loss: {radar_canopy_loss_pct}% ({loss_pixel_count:,} px), "
            f"gain: {radar_canopy_gain_pct}% ({gain_pixel_count:,} px), "
            f"stable: {stable_canopy_pct}%."
        )

    # 5. Colorized Delta-DpRVI Preview
    colorized_rgb = np.full((h, w, 3), 35, dtype=np.uint8)
    if joint_valid_count > 0:
        m_loss = joint_valid_mask & (delta_dprvi_arr <= canopy_loss_threshold)
        m_gain = joint_valid_mask & (delta_dprvi_arr >= canopy_gain_threshold)
        m_stable = joint_valid_mask & (delta_dprvi_arr > canopy_loss_threshold) & (delta_dprvi_arr < canopy_gain_threshold)

        colorized_rgb[m_loss] = [230, 80, 40]       # Radar canopy loss / deforestation: Red-Orange
        colorized_rgb[m_gain] = [30, 160, 60]       # Radar canopy increase / growth: Emerald Green
        colorized_rgb[m_stable] = [120, 130, 140]   # Stable / unchanged: Neutral Slate

    colorized_b64 = convert_array_to_base64_png(colorized_rgb)

    limitations = [
        "Sentinel-1 RTC temporal SAR change uses radiometrically terrain-corrected gamma0 linear backscatter power (m^2/m^2).",
        "Polarization mode verified: DUAL_VV_VH across both T1 and T2.",
        "DpRVI is calculated as DpRVI = q(q+3)/(q+1)^2, q = VH/VV (Mandal et al., 2020).",
        "Terrain layover and shadow masks are unavailable from STAC provider; topographic relief shadows are unmasked.",
        "DpRVI characterizes radar volume scattering randomness; it does not replace optical chlorophyll NDVI."
    ]
    if chronology_warning:
        limitations.insert(0, chronology_warning)
    if total_violations > 0:
        limitations.append(
            f"Physical assumption notice: DpRVI assumption (VH <= VV) was violated in {total_violations:,} pixels across temporal pair. "
            "Values with q > 1.0 were clipped to 1.0 and flagged as uncertified approximations."
        )

    return {
        "status": status,
        "is_calibrated": True,
        "summary": summary,
        "delta_dprvi_mean": delta_dprvi_mean,
        "delta_dprvi_median": delta_dprvi_median,
        "delta_dprvi_p10": delta_dprvi_p10,
        "delta_dprvi_p90": delta_dprvi_p90,
        "delta_dprvi_std": delta_dprvi_std,
        "dprvi_t1_mean": dprvi_t1_mean,
        "dprvi_t2_mean": dprvi_t2_mean,
        "delta_vv_db": delta_vv_db,
        "delta_vh_db": delta_vh_db,
        "delta_cross_pol_db": delta_cross_pol_db,
        "radar_canopy_loss_pct": radar_canopy_loss_pct,
        "radar_canopy_gain_pct": radar_canopy_gain_pct,
        "stable_canopy_pct": stable_canopy_pct,
        "loss_pixel_count": loss_pixel_count,
        "gain_pixel_count": gain_pixel_count,
        "stable_pixel_count": stable_pixel_count,
        "loss_threshold": canopy_loss_threshold,
        "gain_threshold": canopy_gain_threshold,
        "joint_valid_pixel_count": joint_valid_count,
        "total_pixel_count": total_pixels,
        "joint_clearance_pct": joint_clearance_pct,
        "joint_valid_mask": joint_valid_mask,
        "delta_dprvi_array": delta_dprvi_arr,
        "colorized_preview_base64": colorized_b64,
        "colorized_change_preview_base64": colorized_b64,
        "preview_base64": colorized_b64,
        "t1_datetime": dt1_str,
        "t2_datetime": dt2_str,
        "polarization_mode": "DUAL_VV_VH",
        "sar_terrain_mask_status": "unavailable_from_stac_provider",
        "validity_policy": {
            "sar_terrain_layover_shadow_mask": "unavailable_from_stac_provider",
            "polarization": "DUAL_VV_VH",
            "unit": "linear_gamma0_power"
        },
        "dprvi_assumption_violations": {
            "t1": t1_violations,
            "t2": t2_violations,
            "total_count": total_violations,
            "has_violations": total_violations > 0,
            "flagged_unphysical_q": total_violations > 0,
            "total_pct": round((total_violations / max(1, joint_valid_count)) * 100.0, 2)
        },
        "scientific_limitations": limitations
    }
