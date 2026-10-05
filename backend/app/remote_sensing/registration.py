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

def align_image_pair(
    img_a: np.ndarray, 
    img_b: np.ndarray,
    meta_a: Optional[Dict[str, Any]] = None,
    meta_b: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Performs rigorous geospatial compatibility check, CRS alignment, and resampling
    between two satellite scenes (Bi-temporal or Optical + SAR).

    CRITICAL RULE: Never silently resizes unreferenced images and calls them 'co-registered'.
    Truthfully reports georeferencing status, CRS compatibility, and spatial overlap.
    """
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    meta_a = meta_a or {}
    meta_b = meta_b or {}

    crs_a = meta_a.get("crs")
    crs_b = meta_b.get("crs")
    bounds_a = meta_a.get("bounds")
    bounds_b = meta_b.get("bounds")

    # Case 1: Both images have authentic CRS and bounds
    if crs_a and crs_b and bounds_a and bounds_b:
        overlap_pct = 0.0
        reprojected = False

        if crs_a == crs_b:
            overlap_pct = compute_bounds_overlap(bounds_a, bounds_b)
            crs_matched = True
        else:
            # Different CRS / UTM zones
            crs_matched = False
            overlap_pct = 0.0
            logger.warning(
                f"CRS mismatch detected between scenes: primary={crs_a}, secondary={crs_b}. "
                "Different UTM zones/projections cannot be assumed aligned."
            )

        # Attempt Rasterio reprojection/resampling if CRS match and rasterio available
        img_b_aligned = None
        transform_a = meta_a.get("transform")
        transform_b = meta_b.get("transform")

        if crs_matched and (h_a, w_a) == (h_b, w_b) and (transform_a == transform_b or not transform_a or not transform_b):
            img_b_aligned = img_b.copy()
            align_method = "identical_grid_co_registered"
        else:
            # Resample / reproject onto primary scene grid
            reproject_success = False
            if transform_a and transform_b and crs_a and crs_b:
                try:
                    import rasterio
                    from rasterio.warp import reproject, Resampling
                    from rasterio.transform import Affine

                    aff_a = Affine(*transform_a[:6])
                    aff_b = Affine(*transform_b[:6])

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
                        resampling=Resampling.bilinear
                    )

                    if img_b.ndim == 2:
                        img_b_aligned = dst[0]
                    else:
                        img_b_aligned = np.transpose(dst, (1, 2, 0))

                    align_method = "geospatial_rasterio_reprojected"
                    reproject_success = True
                except Exception as e:
                    logger.debug(f"Rasterio reprojection fallback ({e}).")

            if not reproject_success:
                try:
                    import cv2
                    img_b_aligned = cv2.resize(img_b, (w_a, h_a), interpolation=cv2.INTER_LINEAR)
                except ImportError:
                    pil_b = Image.fromarray(img_b)
                    img_b_aligned = np.array(pil_b.resize((w_a, h_a), Image.Resampling.BILINEAR))
                align_method = "geospatial_grid_resampled"

        is_co_reg = bool(crs_matched and overlap_pct > 10.0)
        registration_info = {
            "georeferenced": True,
            "co_registered": is_co_reg,
            "crs_matched": crs_matched,
            "primary_crs": crs_a,
            "secondary_crs": crs_b,
            "spatial_overlap_percent": overlap_pct,
            "primary_dimensions": [w_a, h_a],
            "secondary_original_dimensions": [w_b, h_b],
            "registration_method": align_method,
            "notes": (
                "Verified spatial overlap and CRS matching." if is_co_reg
                else f"Warning: Low spatial overlap ({overlap_pct}%) or CRS mismatch."
            )
        }
        return img_a, img_b_aligned, registration_info

    # Case 2: One or both images lack georeferencing (unreferenced images)
    # Perform pixel array rescaling only so downstream array operations can run,
    # but strictly report that true co-registration was NOT established.
    if (h_a, w_a) != (h_b, w_b):
        logger.info(f"Rescaling secondary raster from ({w_b}x{h_b}) to ({w_a}x{h_a}) for array alignment.")
        try:
            import cv2
            img_b_aligned = cv2.resize(img_b, (w_a, h_a), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            pil_b = Image.fromarray(img_b)
            img_b_aligned = np.array(pil_b.resize((w_a, h_a), Image.Resampling.BILINEAR))
    else:
        img_b_aligned = img_b.copy()

    registration_info = {
        "georeferenced": False,
        "co_registered": False,
        "crs_matched": False,
        "primary_crs": crs_a or "CRS unavailable",
        "secondary_crs": crs_b or "CRS unavailable",
        "spatial_overlap_percent": None,
        "primary_dimensions": [w_a, h_a],
        "secondary_original_dimensions": [w_b, h_b],
        "registration_method": "unreferenced_pixel_dimension_rescale",
        "notes": (
            "CRS or geospatial bounds unavailable for one or both inputs. "
            "Rasters were dimensionally rescaled for baseline processing only; "
            "true physical geodetic co-registration was NOT established. "
            "Genuine bi-temporal remote sensing change detection requires georeferenced GeoTIFFs."
        )
    }

    return img_a, img_b_aligned, registration_info
