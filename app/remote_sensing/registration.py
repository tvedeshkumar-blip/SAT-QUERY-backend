import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from PIL import Image

logger = logging.getLogger("satquery.registration")

def compute_bounds_overlap(bounds_a: List[float], bounds_b: List[float]) -> float:
    """
    Computes percentage of spatial overlap between two bounding boxes [left, bottom, right, top].
    Returns overlap percentage (0.0 to 100.0).
    """
    left = max(bounds_a[0], bounds_b[0])
    bottom = max(bounds_a[1], bounds_b[1])
    right = min(bounds_a[2], bounds_b[2])
    top = min(bounds_a[3], bounds_b[3])

    if right <= left or top <= bottom:
        return 0.0

    inter_area = (right - left) * (top - bottom)
    area_a = (bounds_a[2] - bounds_a[0]) * (bounds_a[3] - bounds_a[1])
    area_b = (bounds_b[2] - bounds_b[0]) * (bounds_b[3] - bounds_b[1])

    min_area = min(area_a, area_b)
    if min_area <= 0:
        return 0.0

    return float(round((inter_area / min_area) * 100.0, 2))

def compute_source_valid_mask(
    arr: np.ndarray,
    nodata: Optional[Any] = None,
    layout: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Computes a 2D boolean mask (True = valid data, False = nodata / NaN / Inf).
    A pixel is invalid if ANY channel contains the declared NoData value, NaN, or infinity.
    Preserves legitimate zero-valued pixels unless nodata is explicitly set to 0.

    Array layout resolution:
    - 2D arrays: (H, W), unambiguous.
    - 3D arrays:
      1. Explicit `layout` parameter: 'HWC' (channel-last) or 'CHW' (channel-first).
      2. Trusted `metadata`: 'layout', 'channel_axis', 'bands' / 'count', 'width', 'height'.
      3. Unambiguous shape deduction: exactly one boundary axis is a standard channel
         count (<= 16) and the other is a spatial dimension (> 16), or spatial dimensions
         form a clear square grid with standard channels.
      4. Explicit failure on genuinely ambiguous shapes: if boundary axes are equal
         (e.g. (3, 3, 3)), both <= 4, or cannot be resolved without a hint, raises ValueError.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if arr.ndim == 2:
        h, w = arr.shape
        channel_axis = None
    elif arr.ndim == 3:
        d0, d1, d2 = arr.shape
        channel_axis = None
        h, w = 0, 0

        # 1. Explicit layout parameter
        if layout is not None:
            layout_norm = str(layout).strip().upper()
            if layout_norm in ("HWC", "CHANNELS_LAST", "CHANNEL_LAST", "LAST"):
                channel_axis = -1
                h, w = d0, d1
            elif layout_norm in ("CHW", "CHANNELS_FIRST", "CHANNEL_FIRST", "FIRST"):
                channel_axis = 0
                h, w = d1, d2
            elif layout_norm in ("HW", "2D"):
                if d0 == 1:
                    channel_axis = 0
                    h, w = d1, d2
                elif d2 == 1:
                    channel_axis = -1
                    h, w = d0, d1
                else:
                    raise ValueError(
                        f"Layout '{layout}' specified 2D/HW, but 3D array has shape {arr.shape} with no singleton channel."
                    )
            elif layout_norm not in ("AUTO", "NONE", ""):
                raise ValueError(f"Unsupported layout '{layout}'. Expected 'HWC' or 'CHW'.")

        # 2. Trusted metadata inspection
        if channel_axis is None and metadata is not None and isinstance(metadata, dict):
            meta_layout = metadata.get("layout") or metadata.get("array_layout")
            if meta_layout is not None:
                ml_norm = str(meta_layout).strip().upper()
                if ml_norm in ("HWC", "CHANNELS_LAST", "CHANNEL_LAST", "LAST"):
                    channel_axis = -1
                    h, w = d0, d1
                elif ml_norm in ("CHW", "CHANNELS_FIRST", "CHANNEL_FIRST", "FIRST"):
                    channel_axis = 0
                    h, w = d1, d2
                elif ml_norm not in ("AUTO", "NONE", ""):
                    raise ValueError(f"Unsupported metadata layout '{meta_layout}'. Expected 'HWC' or 'CHW'.")

            if channel_axis is None and "channel_axis" in metadata:
                meta_axis = metadata["channel_axis"]
                if meta_axis in (0, "0", "first"):
                    channel_axis = 0
                    h, w = d1, d2
                elif meta_axis in (-1, 2, "-1", "2", "last"):
                    channel_axis = -1
                    h, w = d0, d1

            if channel_axis is None:
                meta_bands = metadata.get("bands") or metadata.get("count")
                if meta_bands is not None:
                    try:
                        b_int = int(meta_bands)
                        if b_int == d0 and b_int != d2:
                            channel_axis = 0
                            h, w = d1, d2
                        elif b_int == d2 and b_int != d0:
                            channel_axis = -1
                            h, w = d0, d1
                    except (ValueError, TypeError):
                        pass

            if channel_axis is None:
                meta_w = metadata.get("width")
                meta_h = metadata.get("height")
                if meta_w is not None and meta_h is not None:
                    try:
                        exp_w = int(meta_w)
                        exp_h = int(meta_h)
                        is_chw = (d1 == exp_h and d2 == exp_w)
                        is_hwc = (d0 == exp_h and d1 == exp_w)
                        if is_chw and not is_hwc:
                            channel_axis = 0
                            h, w = d1, d2
                        elif is_hwc and not is_chw:
                            channel_axis = -1
                            h, w = d0, d1
                    except (ValueError, TypeError):
                        pass

        # 3. Shape deduction when unambiguous; explicit error when ambiguous
        if channel_axis is None:
            MAX_CHANNELS = 16
            # Ambiguous: identical boundary dimensions (e.g. (3, 3, 3), (4, 100, 4))
            if d0 == d2:
                raise ValueError(
                    f"Ambiguous 3D array shape {arr.shape}: first and last dimensions are equal ({d0}). "
                    "Cannot distinguish (C, H, W) from (H, W, C) without explicit layout hint or metadata. "
                    "Specify layout='HWC' or layout='CHW'."
                )

            # Square spatial grid with standard channels (e.g. (10, 10, 3) or (3, 10, 10))
            if d0 == d1 and d2 <= 4 and d0 > d2:
                channel_axis = -1
                h, w = d0, d1
            elif d1 == d2 and d0 <= 4 and d2 > d0:
                channel_axis = 0
                h, w = d1, d2
            # Ambiguous: both boundary dimensions are common channel counts (e.g. (3, 10, 4), (4, 5, 3))
            elif d0 <= 4 and d2 <= 4:
                raise ValueError(
                    f"Ambiguous 3D array shape {arr.shape}: both first ({d0}) and last ({d2}) dimensions "
                    "are common channel counts (<= 4). Cannot determine layout without explicit layout hint. "
                    "Specify layout='HWC' or layout='CHW'."
                )
            # Ambiguous: both boundary dimensions are plausible bands (e.g. (8, 256, 12))
            elif d0 <= MAX_CHANNELS and d2 <= MAX_CHANNELS:
                raise ValueError(
                    f"Ambiguous 3D array shape {arr.shape}: both first ({d0}) and last ({d2}) dimensions "
                    f"are plausible channel counts (<= {MAX_CHANNELS}). "
                    "Cannot determine layout without explicit layout hint or metadata. "
                    "Specify layout='HWC' or layout='CHW'."
                )
            # Ambiguous: neither dimension is within standard channel range (e.g. (64, 64, 64))
            elif d0 > MAX_CHANNELS and d2 > MAX_CHANNELS:
                raise ValueError(
                    f"Ambiguous 3D array shape {arr.shape}: neither first ({d0}) nor last ({d2}) dimension "
                    f"is a standard channel count (<= {MAX_CHANNELS}). "
                    "Cannot determine layout without explicit layout hint or metadata. "
                    "Specify layout='HWC' or layout='CHW'."
                )
            # Unambiguous: one is <= MAX_CHANNELS and the other is > MAX_CHANNELS
            elif d0 <= MAX_CHANNELS and d2 > MAX_CHANNELS:
                channel_axis = 0
                h, w = d1, d2
            elif d2 <= MAX_CHANNELS and d0 > MAX_CHANNELS:
                channel_axis = -1
                h, w = d0, d1
    else:
        raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D array with shape {arr.shape}.")

    mask = np.ones((h, w), dtype=bool)

    # 1. Floating-point NaN / Inf checks
    if np.issubdtype(arr.dtype, np.floating):
        invalid_floats = np.isnan(arr) | np.isinf(arr)
        if channel_axis is None:
            mask &= ~invalid_floats
        else:
            mask &= ~np.any(invalid_floats, axis=channel_axis)

    # 2. Source nodata exclusions
    if nodata is not None:
        try:
            nd_val = float(nodata)
            if not np.isnan(nd_val):
                if channel_axis is None:
                    mask &= (arr != nd_val)
                else:
                    # Pixel is invalid if ANY required channel equals nodata
                    mask &= ~np.any(arr == nd_val, axis=channel_axis)
        except Exception:
            pass

    return mask

def align_image_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    meta_a: Optional[Dict[str, Any]] = None,
    meta_b: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Performs rigorous geospatial compatibility check, CRS alignment, and resampling
    between two satellite scenes (Bi-temporal or Optical + SAR).

    Strict Scientific Rules:
    1. Zero-padding and margins: Rasterio reprojection tracks valid data footprints explicitly;
       zero-padded margins and non-overlapping pixels are never treated as observed data.
    2. Valid-data mask: Produces an explicit shared boolean mask combining source nodata,
       mutual footprint intersection, and validity across both modalities.
    3. Missing georeferencing: When affine transforms or adequate geospatial metadata are missing,
       does NOT claim geographic alignment. Resizing is labeled strictly as display-only.
    4. Genuine zeros: Legitimate zero measurements within valid footprints are strictly preserved.
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    meta_a = meta_a or {}
    meta_b = meta_b or {}

    crs_a = meta_a.get("crs")
    crs_b = meta_b.get("crs")
    bounds_a = meta_a.get("bounds")
    bounds_b = meta_b.get("bounds")
    transform_a = meta_a.get("transform")
    transform_b = meta_b.get("transform")
    if not transform_a and crs_a and bounds_a and len(bounds_a) == 4 and w_a > 0 and h_a > 0:
        try:
            from rasterio.transform import from_bounds
            transform_a = list(from_bounds(*bounds_a, w_a, h_a))[:6]
        except Exception:
            pass
    if not transform_b and crs_b and bounds_b and len(bounds_b) == 4 and w_b > 0 and h_b > 0:
        try:
            from rasterio.transform import from_bounds
            transform_b = list(from_bounds(*bounds_b, w_b, h_b))[:6]
        except Exception:
            pass
    nodata_a = meta_a.get("nodata")
    nodata_b = meta_b.get("nodata")

    # Compute source validity masks with explicit layout and metadata hints
    layout_a = meta_a.get("layout") or ("HWC" if img_a.ndim == 3 else None)
    layout_b = meta_b.get("layout") or ("HWC" if img_b.ndim == 3 else None)
    src_valid_a = compute_source_valid_mask(img_a, nodata_a, layout=layout_a, metadata=meta_a)
    src_valid_b = compute_source_valid_mask(img_b, nodata_b, layout=layout_b, metadata=meta_b)

    # Stage 10: Integrate radiometric quality masks if present
    q_mask_a = meta_a.get("quality_mask")
    if q_mask_a is None:
        q_mask_a = meta_a.get("scl_mask")
    if q_mask_a is None and isinstance(meta_a.get("quality_screening"), dict):
        q_mask_a = meta_a["quality_screening"].get("clear_mask")
    if q_mask_a is not None and isinstance(q_mask_a, np.ndarray) and q_mask_a.shape[:2] == (h_a, w_a):
        src_valid_a = src_valid_a & q_mask_a

    q_mask_b = meta_b.get("quality_mask")
    if q_mask_b is None:
        q_mask_b = meta_b.get("sar_valid_mask")
    if q_mask_b is None and isinstance(meta_b.get("quality_screening"), dict):
        q_mask_b = meta_b["quality_screening"].get("valid_mask")
    if q_mask_b is not None and isinstance(q_mask_b, np.ndarray) and q_mask_b.shape[:2] == (h_b, w_b):
        src_valid_b = src_valid_b & q_mask_b

    # Case 1: Both images have authentic CRS, bounds, and affine transforms
    if crs_a and crs_b and bounds_a and bounds_b and transform_a and transform_b:
        crs_matched = (crs_a == crs_b)
        overlap_pct = 0.0

        # Transform bounds_b to crs_a if coordinate systems differ to compute true geospatial overlap
        bounds_b_in_crs_a = bounds_b
        if not crs_matched:
            try:
                from rasterio.warp import transform_bounds
                bounds_b_in_crs_a = list(transform_bounds(crs_b, crs_a, *bounds_b))
            except Exception as tb_err:
                logger.warning(f"Could not transform bounds from {crs_b} to {crs_a}: {tb_err}")
                bounds_b_in_crs_a = None

        if bounds_b_in_crs_a:
            overlap_pct = compute_bounds_overlap(bounds_a, bounds_b_in_crs_a)

        # 1a. Explicit rejection if no mutual geospatial overlap exists
        if overlap_pct <= 0.0:
            logger.warning(
                f"Zero spatial overlap detected between scenes: primary={crs_a}, secondary={crs_b}."
            )
            try:
                import cv2
                img_b_aligned = cv2.resize(img_b, (w_a, h_a), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                pil_b = Image.fromarray(img_b)
                img_b_aligned = np.array(pil_b.resize((w_a, h_a), Image.Resampling.BILINEAR))

            registration_info = {
                "georeferenced": True,
                "co_registered": False,
                "alignment_valid": False,
                "is_display_only": True,
                "crs_matched": crs_matched,
                "primary_crs": crs_a,
                "secondary_crs": crs_b,
                "spatial_overlap_percent": 0.0,
                "primary_dimensions": [w_a, h_a],
                "secondary_original_dimensions": [w_b, h_b],
                "registration_method": "no_spatial_overlap",
                "valid_mask": None,
                "valid_pixel_count": 0,
                "total_pixel_count": int(w_a * h_a),
                "notes": (
                    "Zero spatial overlap between scene footprints. "
                    "Rasters do not intersect geographically and cannot be co-registered or analyzed jointly."
                )
            }
            return img_a, img_b_aligned, registration_info

        # 1b. Check if grids are already identical
        same_dims = (h_a, w_a) == (h_b, w_b)
        same_bounds = False
        if bounds_a and bounds_b and len(bounds_a) == 4 and len(bounds_b) == 4:
            same_bounds = all(abs(float(a) - float(b)) < 1e-3 for a, b in zip(bounds_a, bounds_b))
        same_transforms = False
        if transform_a and transform_b and len(transform_a) >= 6 and len(transform_b) >= 6:
            same_transforms = all(abs(float(a) - float(b)) < 1e-3 for a, b in zip(transform_a[:6], transform_b[:6]))

        if crs_matched and same_dims and same_transforms and same_bounds:
            img_b_aligned = img_b.copy()
            shared_valid_mask = src_valid_a & src_valid_b
            valid_px = int(np.count_nonzero(shared_valid_mask))
            is_co_reg = bool(overlap_pct > 0.0 and valid_px > 0)
            registration_info = {
                "georeferenced": True,
                "co_registered": is_co_reg,
                "alignment_valid": True,
                "is_display_only": False,
                "crs_matched": True,
                "primary_crs": crs_a,
                "secondary_crs": crs_b,
                "target_crs": crs_a,
                "target_transform": transform_a,
                "target_shape": [h_a, w_a],
                "spatial_overlap_percent": overlap_pct,
                "primary_dimensions": [w_a, h_a],
                "secondary_original_dimensions": [w_b, h_b],
                "registration_method": "identical_grid_co_registered",
                "valid_mask": shared_valid_mask,
                "valid_pixel_count": valid_px,
                "total_pixel_count": int(w_a * h_a),
                "grid_checks": {
                    "crs_matched": True,
                    "transform_matched": True,
                    "bounds_matched": True,
                    "dimensions_matched": True,
                    "pixel_size_meters": [abs(float(transform_a[0])), abs(float(transform_a[4]))] if len(transform_a) >= 5 else [10.0, 10.0]
                },
                "notes": (
                    f"Verified identical spatial reference ({crs_a}), matching affine geotransform, "
                    f"matching bounding coordinates, and identical {w_a}x{h_a} pixel grid dimensions. "
                    f"Full geospatial co-registration verified directly from raster headers without reprojection."
                )
            }
            return img_a, img_b_aligned, registration_info

        # 1c. Geospatial reprojection onto primary target grid
        try:
            import rasterio
            from rasterio.warp import reproject, Resampling
            from rasterio.transform import Affine

            aff_a = Affine(*transform_a[:6])
            aff_b = Affine(*transform_b[:6])

            # Determine appropriate resampling method:
            # Nearest for integer/categorical masks, bilinear for continuous observations
            is_int_dtype = issubclass(img_b.dtype.type, (np.integer, np.unsignedinteger, np.bool_))
            data_resampling = Resampling.nearest if is_int_dtype else Resampling.bilinear

            # 1. Reproject data
            if img_b.ndim == 2:
                src = img_b[np.newaxis, :, :]
                dst = np.zeros((1, h_a, w_a), dtype=img_b.dtype)
            else:
                src = np.transpose(img_b, (2, 0, 1))
                dst = np.zeros((img_b.shape[2], h_a, w_a), dtype=img_b.dtype)

            reproject(
                source=src,
                destination=dst,
                src_transform=aff_b,
                src_crs=crs_b,
                dst_transform=aff_a,
                dst_crs=crs_a,
                resampling=data_resampling,
                src_nodata=nodata_b,
                dst_nodata=nodata_b
            )

            # 2. Reproject validity mask of secondary image
            src_mask_u8 = (src_valid_b.astype(np.uint8) * 255)[np.newaxis, :, :]
            dst_mask_u8 = np.zeros((1, h_a, w_a), dtype=np.uint8)

            reproject(
                source=src_mask_u8,
                destination=dst_mask_u8,
                src_transform=aff_b,
                src_crs=crs_b,
                dst_transform=aff_a,
                dst_crs=crs_a,
                resampling=Resampling.nearest,
                src_nodata=0,
                dst_nodata=0
            )

            dst_valid_b = (dst_mask_u8[0] == 255)
            shared_valid_mask = src_valid_a & dst_valid_b

            if img_b.ndim == 2:
                img_b_aligned = dst[0]
            else:
                img_b_aligned = np.transpose(dst, (1, 2, 0))

            valid_px = int(np.count_nonzero(shared_valid_mask))
            is_co_reg = bool(overlap_pct > 0.0 and valid_px > 0)

            registration_info = {
                "georeferenced": True,
                "co_registered": is_co_reg,
                "alignment_valid": True,
                "is_display_only": False,
                "crs_matched": crs_matched,
                "primary_crs": crs_a,
                "secondary_crs": crs_b,
                "target_crs": crs_a,
                "target_transform": transform_a,
                "target_shape": [h_a, w_a],
                "spatial_overlap_percent": overlap_pct,
                "primary_dimensions": [w_a, h_a],
                "secondary_original_dimensions": [w_b, h_b],
                "registration_method": "geospatial_rasterio_reprojected",
                "valid_mask": shared_valid_mask,
                "valid_pixel_count": valid_px,
                "total_pixel_count": int(w_a * h_a),
                "notes": (
                    f"Rasterio reprojection succeeded onto {crs_a} target grid. Overlap: {overlap_pct}%, "
                    f"Valid intersecting pixels: {valid_px:,} of {w_a * h_a:,}."
                )
            }
            return img_a, img_b_aligned, registration_info
        except Exception as e:
            logger.warning(f"Rasterio reprojection failed ({e}).")

        # Reprojection could not be performed (reprojection error)
        try:
            import cv2
            img_b_aligned = cv2.resize(img_b, (w_a, h_a), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            pil_b = Image.fromarray(img_b)
            img_b_aligned = np.array(pil_b.resize((w_a, h_a), Image.Resampling.BILINEAR))

        notes_reason = "Reprojection failed or affine transform incompatible"
        registration_info = {
            "georeferenced": True,
            "co_registered": False,
            "alignment_valid": False,
            "is_display_only": True,
            "crs_matched": crs_matched,
            "primary_crs": crs_a,
            "secondary_crs": crs_b,
            "spatial_overlap_percent": overlap_pct,
            "primary_dimensions": [w_a, h_a],
            "secondary_original_dimensions": [w_b, h_b],
            "registration_method": "unreferenced_visual_resize_only",
            "valid_mask": None,
            "valid_pixel_count": 0,
            "total_pixel_count": int(w_a * h_a),
            "notes": (
                f"{notes_reason}. Resizing is strictly display-only. "
                "Safe geospatial alignment was NOT established and cannot be used for spatial masks or area calculations."
            )
        }
        return img_a, img_b_aligned, registration_info

    # Case 2: One or both images lack georeferencing
    # Perform pixel array rescaling only for display preview
    if (h_a, w_a) != (h_b, w_b):
        try:
            import cv2
            img_b_aligned = cv2.resize(img_b, (w_a, h_a), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            pil_b = Image.fromarray(img_b)
            img_b_aligned = np.array(pil_b.resize((w_a, h_a), Image.Resampling.BILINEAR))
        align_method = "unreferenced_visual_resize_only"
    else:
        # Same dimension unreferenced arrays
        img_b_aligned = img_b.copy()
        align_method = "unreferenced_same_dimensions_display_only"

    notes = (
        "CRS, affine geotransforms, or spatial bounds unavailable for one or both inputs. "
        "Rasters were scaled for visual display preview only; true physical geospatial co-registration was NOT established. "
        "Display-only resizing cannot be used for spatial masks, area calculations, or geospatial conclusions."
    )

    registration_info = {
        "georeferenced": False,
        "co_registered": False,
        "alignment_valid": False,
        "is_display_only": True,
        "crs_matched": False,
        "primary_crs": crs_a or "CRS unavailable",
        "secondary_crs": crs_b or "CRS unavailable",
        "spatial_overlap_percent": None,
        "primary_dimensions": [w_a, h_a],
        "secondary_original_dimensions": [w_b, h_b],
        "registration_method": align_method,
        "valid_mask": None,
        "valid_pixel_count": 0,
        "total_pixel_count": int(w_a * h_a),
        "notes": notes
    }
    return img_a, img_b_aligned, registration_info
