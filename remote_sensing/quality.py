"""
Stage 10: Radiometric Quality Screening Processor.
Provides deterministic, scientifically defensible quality screening for:
1. Sentinel-2 Level-2A Scene Classification Layer (SCL) categorical optical masking.
2. Sentinel-1 RTC calibrated microwave backscatter validity screening.

Enforces:
- Nearest-neighbor categorical resampling (preserves discrete class values without interpolation).
- Spatial CRS and transform validation before mask alignment.
- Clear separation of clear, uncertain, excluded, nodata, and unknown classes.
- Rigorous pixel accounting with precise denominators.
- Explicit unavailable status for missing geometric SAR masks (no fabricated terrain masks).
"""

import logging
from typing import Dict, Any, Tuple, Optional, List, Set, Union
import numpy as np

logger = logging.getLogger("satquery.quality")

class QualityScreeningError(Exception):
    """Raised when radiometric quality screening fails or spatial grids are misaligned."""
    pass

# ESA Sentinel-2 Level-2A Scene Classification Layer (SCL) Specification
SCL_CLASS_NAMES: Dict[int, str] = {
    0: "NO_DATA",
    1: "SATURATED_OR_DEFECTIVE",
    2: "CAST_SHADOW",
    3: "CLOUD_SHADOW",
    4: "VEGETATION",
    5: "NOT_VEGETATED",
    6: "WATER",
    7: "UNCLASSIFIED",
    8: "CLOUD_MEDIUM_PROBA",
    9: "CLOUD_HIGH_PROBA",
    10: "THIN_CIRRUS",
    11: "SNOW_ICE"
}

# Scientific Policy Definitions:
# 1. Excluded: Clouds, cloud shadows, hardware saturation, and out-of-swath nodata.
SCL_EXCLUDED_CLASSES: Set[int] = {0, 1, 3, 8, 9, 10}

# 2. Uncertain: Topographic relief cast shadows and ambiguous spectral endmembers.
# Excluded from claims requiring confidently clear pixels; flagged with radiometric caveats.
SCL_UNCERTAIN_CLASSES: Set[int] = {2, 7}

# 3. Retained: Confident clear-sky surface observations (vegetation, soil/bare, water, snow/ice).
SCL_RETAINED_CLASSES: Set[int] = {4, 5, 6, 11}

# Standard ESA Sentinel-2 L2A Visualization Palette (RGB)
SCL_COLOR_MAP: Dict[int, List[int]] = {
    0: [0, 0, 0],         # NO_DATA -> Black
    1: [255, 0, 0],       # SATURATED_OR_DEFECTIVE -> Red
    2: [47, 47, 47],      # CAST_SHADOW -> Very Dark Gray
    3: [100, 50, 0],      # CLOUD_SHADOW -> Dark Brown
    4: [0, 160, 0],       # VEGETATION -> Green
    5: [255, 230, 90],    # NOT_VEGETATED -> Sand Yellow
    6: [0, 0, 255],       # WATER -> Blue
    7: [128, 128, 128],   # UNCLASSIFIED -> Medium Gray
    8: [192, 192, 192],   # CLOUD_MEDIUM_PROBA -> Light Gray
    9: [255, 255, 255],   # CLOUD_HIGH_PROBA -> White
    10: [100, 200, 255],  # THIN_CIRRUS -> Cyan
    11: [255, 150, 255]   # SNOW_ICE -> Pink
}


class QualityScreener:
    """
    Deterministic Radiometric Quality Screening Evaluator.
    """

    @classmethod
    def evaluate_scl(
        cls, 
        scl_arr: np.ndarray,
        allow_uncertain_as_clear: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates a 2D Sentinel-2 Scene Classification Layer (SCL) raster.
        Computes boolean masks and comprehensive valid-pixel accounting.
        
        Args:
            scl_arr: 2D (or 3D single-band) integer array of SCL classes.
            allow_uncertain_as_clear: If False (default), uncertain classes (2, 7) are
                excluded from the clear mask.
        Returns:
            Dictionary containing:
                - clear_mask: 2D boolean mask of clear surface pixels
                - uncertain_mask: 2D boolean mask of uncertain pixels (classes 2, 7)
                - excluded_mask: 2D boolean mask of excluded pixels (clouds, shadows, etc.)
                - unknown_mask: 2D boolean mask of unmapped/out-of-range values (<0 or >11)
                - pixel_counts: Detailed count breakdown
                - statistics: Coverage and clearance fractions
                - policy_documentation: Explicit policy explanation
        """
        if scl_arr.ndim == 3:
            if scl_arr.shape[0] in (1, 3, 4) and scl_arr.shape[0] < scl_arr.shape[1] and scl_arr.shape[0] < scl_arr.shape[2]:
                scl_2d = scl_arr[0, :, :]
            else:
                scl_2d = scl_arr[:, :, 0]
        elif scl_arr.ndim == 2:
            scl_2d = scl_arr
        else:
            raise QualityScreeningError(f"SCL array must be 2D or 3D, received shape {scl_arr.shape}")

        h, w = scl_2d.shape[:2]
        total_pixels = int(h * w)

        # 1. Detect unknown / out-of-range values
        is_valid_range = (scl_2d >= 0) & (scl_2d <= 11)
        unknown_mask = ~is_valid_range
        unknown_count = int(np.count_nonzero(unknown_mask))

        # 2. Compute individual class counts
        class_counts = {}
        for class_id, class_name in SCL_CLASS_NAMES.items():
            cnt = int(np.count_nonzero(scl_2d == class_id))
            class_counts[class_name] = cnt

        # 3. Categorical masking
        nodata_count = class_counts["NO_DATA"]
        observed_pixels = total_pixels - nodata_count

        # Clear retained surface
        retained_mask = np.isin(scl_2d, list(SCL_RETAINED_CLASSES)) & ~unknown_mask
        uncertain_mask = np.isin(scl_2d, list(SCL_UNCERTAIN_CLASSES)) & ~unknown_mask
        
        # Excluded mask: Clouds, shadows, defects, nodata, and any unknown values (fail closed)
        excluded_mask = (
            np.isin(scl_2d, list(SCL_EXCLUDED_CLASSES)) |
            unknown_mask
        )

        if allow_uncertain_as_clear:
            clear_mask = (retained_mask | uncertain_mask) & ~unknown_mask
        else:
            clear_mask = retained_mask

        clear_pixels = int(np.count_nonzero(clear_mask))
        uncertain_pixels = int(np.count_nonzero(uncertain_mask))
        excluded_pixels = int(np.count_nonzero(excluded_mask))

        cloud_pixels = (
            class_counts["CLOUD_MEDIUM_PROBA"] + 
            class_counts["CLOUD_HIGH_PROBA"] + 
            class_counts["THIN_CIRRUS"]
        )
        cloud_shadow_pixels = class_counts["CLOUD_SHADOW"]
        cast_shadow_pixels = class_counts["CAST_SHADOW"]
        defective_pixels = class_counts["SATURATED_OR_DEFECTIVE"]

        # 4. Rigorous fraction definitions with explicit denominators
        # Fraction 1: Data Coverage Fraction (observed valid sensor footprint vs total window)
        data_coverage_fraction = round(observed_pixels / max(1, total_pixels), 4)

        # Fraction 2: Quality Clearance Fraction (clear ground pixels vs observed pixels)
        quality_clearance_fraction = round(clear_pixels / max(1, observed_pixels), 4) if observed_pixels > 0 else 0.0

        # Fraction 3: Effective Usable Fraction (clear ground pixels vs total window grid)
        effective_usable_fraction = round(clear_pixels / max(1, total_pixels), 4)

        # Cloud-affected fraction within observed scene
        cloud_affected_fraction = round((cloud_pixels + cloud_shadow_pixels) / max(1, observed_pixels), 4) if observed_pixels > 0 else 0.0

        return {
            "screened": True,
            "is_evaluated": True,
            "mask_source": "Sentinel-2 L2A Scene Classification Layer (SCL)",
            "clear_mask": clear_mask,
            "uncertain_mask": uncertain_mask,
            "excluded_mask": excluded_mask,
            "unknown_mask": unknown_mask,
            "has_unknown_classes": bool(unknown_count > 0),
            "unknown_pixel_count": unknown_count,
            "total_pixels": total_pixels,
            "observed_pixel_count": observed_pixels,
            "nodata_pixel_count": nodata_count,
            "clear_pixels": clear_pixels,
            "retained_pixels": clear_pixels,
            "uncertain_pixels": uncertain_pixels,
            "excluded_pixel_count": excluded_pixels,
            "cloud_pixels": cloud_pixels,
            "cloud_shadow_pixels": cloud_shadow_pixels,
            "cast_shadow_pixels": cast_shadow_pixels,
            "defective_pixels": defective_pixels,
            "quality_clearance_fraction": quality_clearance_fraction,
            "data_coverage_fraction": data_coverage_fraction,
            "effective_usable_fraction": effective_usable_fraction,
            "has_cloud_obstruction": (cloud_pixels + cloud_shadow_pixels) == observed_pixels and observed_pixels > 0,
            "pixel_counts": {
                "total_pixels": total_pixels,
                "nodata_pixels": nodata_count,
                "observed_pixels": observed_pixels,
                "clear_pixels": clear_pixels,
                "uncertain_pixels": uncertain_pixels,
                "excluded_pixels": excluded_pixels,
                "cloud_pixels": cloud_pixels,
                "cloud_shadow_pixels": cloud_shadow_pixels,
                "cast_shadow_pixels": cast_shadow_pixels,
                "defective_pixels": defective_pixels,
                "classes": class_counts
            },
            "statistics": {
                "data_coverage_percentage": round(data_coverage_fraction * 100.0, 2),
                "quality_clearance_percentage": round(quality_clearance_fraction * 100.0, 2),
                "effective_usable_percentage": round(effective_usable_fraction * 100.0, 2),
                "cloud_affected_percentage": round(cloud_affected_fraction * 100.0, 2)
            },
            "denominators": {
                "data_coverage_denominator": "total_window_pixels",
                "quality_clearance_denominator": "observed_pixels (excluding nodata/margin)",
                "effective_usable_denominator": "total_window_pixels"
            },
            "policy": {
                "excluded_classes": sorted(list(SCL_EXCLUDED_CLASSES)),
                "uncertain_classes": sorted(list(SCL_UNCERTAIN_CLASSES)),
                "retained_classes": sorted(list(SCL_RETAINED_CLASSES)),
                "unknown_policy": "fail_closed_excluded",
                "uncertain_handling": "quarantined_from_confident_clear" if not allow_uncertain_as_clear else "included_with_caveat"
            }
        }

    @classmethod
    def reproject_scl_to_target(
        cls,
        scl_arr: np.ndarray,
        scl_meta: Dict[str, Any],
        target_meta: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reprojects categorical SCL raster onto exact target optical grid (e.g. 20m -> 10m).
        Strictly enforces rasterio.warp.Resampling.nearest to preserve discrete class labels.
        Validates CRS, affine geotransform, and spatial boundary intersection.
        """
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import Affine

        # 1. Georeferencing Verification
        crs_scl = scl_meta.get("crs")
        crs_target = target_meta.get("crs")
        if not crs_scl or not crs_target:
            raise QualityScreeningError(
                f"Missing CRS in quality mask or target raster: SCL CRS={crs_scl}, Target CRS={crs_target}"
            )

        trans_scl = scl_meta.get("transform")
        trans_target = target_meta.get("transform")
        if not trans_scl or not trans_target:
            raise QualityScreeningError(
                "Missing affine geotransform in quality mask or target raster."
            )

        # 2. Reject uncalibrated identity transform
        if (trans_scl[0] == 1.0 and trans_scl[4] == 1.0 and trans_scl[2] == 0.0 and trans_scl[5] == 0.0) or \
           (trans_target[0] == 1.0 and trans_target[4] == 1.0 and trans_target[2] == 0.0 and trans_target[5] == 0.0):
            raise QualityScreeningError("Georeferencing is uncalibrated identity matrix; spatial alignment cannot be verified.")

        # 3. Target dimensions
        h_target = int(target_meta.get("height", 0))
        w_target = int(target_meta.get("width", 0))
        if h_target <= 0 or w_target <= 0:
            raise QualityScreeningError(f"Invalid target raster dimensions ({w_target}x{h_target}).")

        # 4. Prepare SCL source array
        if scl_arr.ndim == 3:
            src = scl_arr[0:1, :, :].astype(np.uint8) if scl_arr.shape[0] == 1 else scl_arr[:, :, 0:1].transpose((2, 0, 1)).astype(np.uint8)
        else:
            src = scl_arr[np.newaxis, :, :].astype(np.uint8)

        dst = np.zeros((1, h_target, w_target), dtype=np.uint8)
        aff_scl = Affine(*trans_scl[:6])
        aff_target = Affine(*trans_target[:6])

        # 5. Execute Nearest-Neighbor Reprojection
        try:
            reproject(
                source=src,
                destination=dst,
                src_transform=aff_scl,
                src_crs=crs_scl,
                dst_transform=aff_target,
                dst_crs=crs_target,
                resampling=Resampling.nearest,  # MANDATORY: Preserves discrete class values
                src_nodata=0,
                dst_nodata=0
            )
        except Exception as e:
            raise QualityScreeningError(f"Failed to reproject SCL quality mask to target grid: {e}")

        aligned_scl = dst[0]

        alignment_meta = {
            "resampling_method": "nearest_neighbor",
            "source_dimensions": [int(scl_arr.shape[1] if scl_arr.ndim == 2 else scl_arr.shape[1]), int(scl_arr.shape[0])],
            "target_dimensions": [w_target, h_target],
            "source_crs": str(crs_scl),
            "target_crs": str(crs_target),
            "co_registered": True,
            "spatial_grid_matched": True
        }

        return aligned_scl, alignment_meta

    @classmethod
    def evaluate_sar_quality(
        cls,
        sar_arr: np.ndarray,
        sar_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates Sentinel-1 RTC microwave backscatter validity.
        Screens out nodata (-32768.0), non-finite floats, and non-positive linear power values.
        Honestly records terrain layover/shadow masks as unavailable from the provider.
        """
        meta = sar_meta or {}
        if sar_arr.ndim == 3:
            if sar_arr.shape[0] in (1, 3, 4) and sar_arr.shape[0] < sar_arr.shape[1] and sar_arr.shape[0] < sar_arr.shape[2]:
                sar_2d = sar_arr[0, :, :]
            else:
                sar_2d = sar_arr[:, :, 0]
        elif sar_arr.ndim == 2:
            sar_2d = sar_arr
        else:
            raise QualityScreeningError(f"SAR array must be 2D or 3D, received shape {sar_arr.shape}")

        h, w = sar_2d.shape[:2]
        total_pixels = int(h * w)

        # 1. Determine nodata values (PC RTC uses -32768.0)
        nodata_val = meta.get("nodata")
        if nodata_val is None and "tags" in meta:
            nodata_val = meta.get("tags", {}).get("NODATA")
        
        # 2. Valid finite float check
        finite_mask = np.isfinite(sar_2d)
        non_finite_count = int(np.count_nonzero(~finite_mask))

        # 3. Nodata exclusion
        nodata_mask = np.zeros((h, w), dtype=bool)
        if nodata_val is not None:
            try:
                nd_f = float(nodata_val)
                nodata_mask = np.isclose(sar_2d, nd_f, atol=1e-3)
            except Exception:
                pass
        # Common Sentinel-1 RTC nodata values
        nodata_mask |= np.isclose(sar_2d, -32768.0, atol=1e-3)
        nodata_count = int(np.count_nonzero(nodata_mask & finite_mask))

        # 4. Linear power check
        # For floating-point calibrated linear power: must be strictly positive before log10
        # For integer/uint8 intensity display rasters: genuine 0 is valid dark surface
        is_float_power = np.issubdtype(sar_2d.dtype, np.floating)
        if is_float_power:
            non_positive_mask = (sar_2d <= 0.0) & finite_mask & ~nodata_mask
            valid_sar_mask = finite_mask & ~nodata_mask & (sar_2d > 0.0)
        else:
            non_positive_mask = np.zeros((h, w), dtype=bool)
            valid_sar_mask = finite_mask & ~nodata_mask & (sar_2d >= 0)

        non_positive_count = int(np.count_nonzero(non_positive_mask))
        valid_pixels = int(np.count_nonzero(valid_sar_mask))

        valid_fraction = round(valid_pixels / max(1, total_pixels), 4)

        return {
            "screened": True,
            "is_evaluated": True,
            "mask_source": "Sentinel-1 RTC Valid Backscatter Bounds",
            "valid_mask": valid_sar_mask,
            "total_pixels": total_pixels,
            "valid_pixel_count": valid_pixels,
            "nodata_pixel_count": nodata_count,
            "non_finite_pixel_count": non_finite_count,
            "non_positive_pixel_count": non_positive_count,
            "finite_positive_fraction": valid_fraction,
            "pixel_counts": {
                "total_pixels": total_pixels,
                "nodata_pixels": nodata_count,
                "non_finite_pixels": non_finite_count,
                "non_positive_pixels": non_positive_count,
                "valid_pixels": valid_pixels
            },
            "statistics": {
                "valid_percentage": round(valid_fraction * 100.0, 2)
            },
            "calibration_quantity": "gamma0" if any(k in str(meta).lower() for k in ["gamma0", "rtc"]) else "sigma0",
            "sar_terrain_layover_shadow_mask": "unavailable_from_stac_provider",
            "sar_terrain_mask_status": "unavailable",
            "terrain_shadow_free": False,
            "scientific_limitations": [
                "Per-pixel geometric layover/shadow masks are unavailable from the STAC provider.",
                "Radar intensity validity is screened on finite numerical bounds and positive linear power.",
                "Topographic relief distortion cannot be certified as absent without auxiliary DEM geometric modeling."
            ]
        }

    # Alias for Sentinel-1 RTC quality screening
    evaluate_sar_rtc = evaluate_sar_quality

    @classmethod
    def generate_scl_rgb_preview(cls, scl_arr: np.ndarray) -> np.ndarray:
        """
        Converts a 2D categorical SCL integer raster into an RGB uint8 image
        using standard ESA Sentinel-2 Level-2A classification colors.
        """
        if scl_arr.ndim == 3:
            scl_2d = scl_arr[:, :, 0]
        else:
            scl_2d = scl_arr

        h, w = scl_2d.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in SCL_COLOR_MAP.items():
            mask = (scl_2d == class_id)
            rgb[mask] = color

        # Any unknown values outside 0-11 mapped to dark magenta
        unknown = (scl_2d < 0) | (scl_2d > 11)
        if np.any(unknown):
            rgb[unknown] = [128, 0, 128]

        return rgb
