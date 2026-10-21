import base64
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger("satquery.validation")

class ValidationError(ValueError):
    """Custom exception for geospatial and image validation errors."""
    pass

class InputValidator:
    """
    Geospatial Input Validation Layer.
    Validates file formats, MIME types, payload size, dimensions, CRS compatibility,
    spatial overlap, and dual-image comparability.
    """

    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit
    ALLOWED_MIME_TYPES = {
        "image/png", "image/jpeg", "image/jpg",
        "image/tiff", "image/tif", "image/geotiff",
        "application/octet-stream"
    }

    @classmethod
    def validate_image_payload(cls, data_b64: str, mime_type: str, filename: Optional[str] = None) -> int:
        """
        Validates Base64 data string, size limit, and MIME type.
        Returns size in bytes.
        """
        if not data_b64 or not data_b64.strip():
            raise ValidationError("Image payload is empty or invalid.")

        raw_str = data_b64.strip()
        if raw_str.startswith(("http://", "https://")):
            import httpx
            try:
                resp = httpx.get(raw_str, timeout=10.0, follow_redirects=True)
                if resp.status_code != 200:
                    raise ValidationError(f"Failed to fetch image from URL '{raw_str}': HTTP {resp.status_code}")
                raw_bytes = resp.content
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(f"Failed to fetch image from URL: {e}")
        else:
            clean_b64 = raw_str.split(",")[-1]
            try:
                raw_bytes = base64.b64decode(clean_b64, validate=True)
            except Exception:
                raise ValidationError("Failed to decode base64 image data: malformed encoding.")

        size = len(raw_bytes)
        if size == 0:
            raise ValidationError("Uploaded file contains 0 bytes.")

        if size > cls.MAX_FILE_SIZE_BYTES:
            raise ValidationError(
                f"File size ({size / (1024*1024):.1f}MB) exceeds the maximum allowed limit of 50MB."
            )

        norm_mime = mime_type.lower().strip()
        if norm_mime not in cls.ALLOWED_MIME_TYPES:
            fname = (filename or "").lower()
            if not (fname.endswith(".png") or fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".tif") or fname.endswith(".tiff")):
                raise ValidationError(
                    f"Unsupported file format '{mime_type}'. Supported formats: GeoTIFF (.tif, .tiff), PNG, JPEG."
                )

        return size

    @classmethod
    def validate_classification_raster(cls, array: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates suitability of input raster for BigEarthNetv2 10-band multispectral classification.
        Rejects visual screenshots, single-band rasters, and missing spectral channels.
        """
        bands = meta.get("bands")
        if bands is None:
            if array.ndim == 3:
                bands = array.shape[2] if array.shape[2] in (1, 3, 4, 10, 12, 13) else array.shape[0]
            else:
                bands = 1

        is_preview = (
            meta.get("asset_category") == "visual_preview"
            or (meta.get("filename") or "").lower().endswith((".jpg", ".jpeg"))
            or (bands in (1, 3, 4) and not meta.get("is_geotiff"))
        )

        if is_preview or bands < 10:
            return {
                "valid": False,
                "reason": (
                    f"Asset has {bands} channel(s). BigEarthNetv2 is a 10-band Sentinel-2 multispectral model "
                    "requiring bands (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12). "
                    "Visual browse screenshots or RGB renderings lack the required spectral bands. "
                    "Please provide raw Sentinel-2 L2A multispectral data (GeoTIFF) or select single-image VQA/captioning."
                ),
                "bands_detected": bands,
                "is_screenshot_or_preview": is_preview,
            }

        return {
            "valid": True,
            "bands_detected": bands,
            "is_screenshot_or_preview": False,
        }

    @classmethod
    def _compute_ground_resolution_meters(
        cls,
        meta: Dict[str, Any]
    ) -> Optional[Tuple[float, float]]:
        """
        Computes effective ground pixel resolution (res_x_m, res_y_m) in meters.
        Accounts for projected vs geographic CRS and central latitude scale on WGS84 ellipsoid.
        """
        import math
        from rasterio.crs import CRS

        crs_val = meta.get("crs")
        if not crs_val:
            return None

        try:
            crs_obj = CRS.from_user_input(crs_val)
        except Exception:
            return None

        res = meta.get("resolution")
        if not res or len(res) < 2 or res[0] <= 0 or res[1] <= 0:
            transform = meta.get("transform")
            if transform and len(transform) >= 6:
                try:
                    res = [abs(float(transform[0])), abs(float(transform[4]))]
                except (ValueError, TypeError):
                    res = None
            elif meta.get("bounds") and meta.get("width") and meta.get("height"):
                try:
                    b = meta["bounds"]
                    w = float(meta["width"])
                    h = float(meta["height"])
                    if w > 0 and h > 0:
                        res = [abs((b[2] - b[0]) / w), abs((b[3] - b[1]) / h)]
                    else:
                        res = None
                except Exception:
                    res = None
            else:
                return None

        if res[0] <= 0 or res[1] <= 0:
            return None

        if crs_obj.is_projected:
            factor = 1.0
            if crs_obj.linear_units_factor:
                try:
                    factor = float(crs_obj.linear_units_factor[1])
                except (IndexError, TypeError, ValueError):
                    factor = 1.0
            return (float(res[0]) * factor, float(res[1]) * factor)

        elif crs_obj.is_geographic:
            lat = 0.0
            if meta.get("bounds") and len(meta["bounds"]) == 4:
                try:
                    lat = (float(meta["bounds"][1]) + float(meta["bounds"][3])) / 2.0
                except (ValueError, TypeError):
                    lat = 0.0

            lat = max(-89.0, min(89.0, lat))
            lat_rad = math.radians(lat)

            a = 6378137.0
            b = 6356752.314245
            e2 = 1.0 - (b * b) / (a * a)

            denom = math.sqrt(1.0 - e2 * (math.sin(lat_rad) ** 2))
            M = a * (1.0 - e2) / (denom ** 3)
            N = a / denom

            m_per_deg_lat = (math.pi / 180.0) * M
            m_per_deg_lon = (math.pi / 180.0) * N * math.cos(lat_rad)

            res_x_m = float(res[0]) * m_per_deg_lon
            res_y_m = float(res[1]) * m_per_deg_lat
            return (res_x_m, res_y_m)

        return None

    @classmethod
    def validate_dual_scenes(
        cls,
        meta_a: Dict[str, Any],
        meta_b: Dict[str, Any],
        mode: str = "bitemporal"
    ) -> Dict[str, Any]:
        """
        Validates spatial and temporal compatibility between two satellite scenes.
        Never claims co-registration without CRS and spatial evidence.
        """
        w_a, h_a = meta_a.get("width", 0), meta_a.get("height", 0)
        w_b, h_b = meta_b.get("width", 0), meta_b.get("height", 0)

        if w_a == 0 or h_a == 0 or w_b == 0 or h_b == 0:
            raise ValidationError(
                "Unable to determine spatial dimensions of one or both scenes."
            )

        crs_a = meta_a.get("crs")
        crs_b = meta_b.get("crs")
        bounds_a = meta_a.get("bounds")
        bounds_b = meta_b.get("bounds")

        report = {
            "valid": False,
            "georeferenced": bool(crs_a and crs_b),
            "co_registered": False,
            "crs_compatible": False,
            "overlap_detected": False,
            "spatial_overlap_pct": None,
            "resolution_compatible": True,
            "warnings": [],
        }

        # CRS must be present and equal.
        if crs_a and crs_b:
            report["crs_compatible"] = crs_a == crs_b
            if not report["crs_compatible"]:
                report["warnings"].append(
                    f"CRS mismatch: Primary={crs_a}, Secondary={crs_b}."
                )
        else:
            report["warnings"].append(
                "CRS unavailable for one or both scenes; "
                "genuine bi-temporal remote sensing change detection requires georeferenced GeoTIFFs."
            )

        # Calculate spatial overlap only when both scenes have authentic CRS and bounds.
        if crs_a and crs_b and bounds_a and bounds_b:
            norm_bounds_b = bounds_b
            if crs_a != crs_b:
                try:
                    from rasterio.warp import transform_bounds
                    norm_bounds_b = list(transform_bounds(crs_b, crs_a, *bounds_b))
                except Exception:
                    norm_bounds_b = None

            if norm_bounds_b is not None:
                left = max(bounds_a[0], norm_bounds_b[0])
                bottom = max(bounds_a[1], norm_bounds_b[1])
                right = min(bounds_a[2], norm_bounds_b[2])
                top = min(bounds_a[3], norm_bounds_b[3])

                if right > left and top > bottom:
                    intersection_area = (right - left) * (top - bottom)

                    area_a = (
                        (bounds_a[2] - bounds_a[0])
                        * (bounds_a[3] - bounds_a[1])
                    )
                    area_b = (
                        (norm_bounds_b[2] - norm_bounds_b[0])
                        * (norm_bounds_b[3] - norm_bounds_b[1])
                    )

                    min_area = min(area_a, area_b)

                    if min_area > 0:
                        report["spatial_overlap_pct"] = round(
                            (intersection_area / min_area) * 100.0, 2
                        )
                        report["overlap_detected"] = (
                            report["spatial_overlap_pct"] > 0
                        )
                else:
                    report["spatial_overlap_pct"] = 0.0
            else:
                report["spatial_overlap_pct"] = 0.0
        elif crs_a and crs_b:
            report["warnings"].append(
                "Spatial bounds unavailable for one or both scenes."
            )

        if crs_a and crs_b and not report["overlap_detected"]:
            report["warnings"].append(
                "No positive spatial overlap established between scenes."
            )

        # Resolution compatibility (units-aware ground resolution).
        ground_res_a = cls._compute_ground_resolution_meters(meta_a)
        ground_res_b = cls._compute_ground_resolution_meters(meta_b)

        if ground_res_a and ground_res_b:
            try:
                ratio_x = max(ground_res_a[0], ground_res_b[0]) / max(
                    1e-6, min(ground_res_a[0], ground_res_b[0])
                )
                ratio_y = max(ground_res_a[1], ground_res_b[1]) / max(
                    1e-6, min(ground_res_a[1], ground_res_b[1])
                )
                ratio = max(ratio_x, ratio_y)

                if ratio > 10.0:
                    report["resolution_compatible"] = False
                    report["warnings"].append(
                        f"Large ground resolution disparity ({ratio:.1f}x)."
                    )
            except (TypeError, ValueError, IndexError):
                report["resolution_compatible"] = False
        elif not meta_a.get("crs") and not meta_b.get("crs") and meta_a.get("resolution") and meta_b.get("resolution"):
            try:
                res_a = meta_a["resolution"]
                res_b = meta_b["resolution"]
                ratio_x = max(res_a[0], res_b[0]) / max(1e-6, min(res_a[0], res_b[0]))
                ratio_y = max(res_a[1], res_b[1]) / max(1e-6, min(res_a[1], res_b[1]))
                ratio = max(ratio_x, ratio_y)
                if ratio > 10.0:
                    report["resolution_compatible"] = False
                    report["warnings"].append(
                        f"Large resolution disparity ({ratio:.1f}x)."
                    )
            except (TypeError, ValueError, IndexError):
                report["resolution_compatible"] = False
        elif meta_a.get("crs") and meta_b.get("crs"):
            report["resolution_compatible"] = False
            report["warnings"].append("Unable to determine ground resolution for one or both scenes.")

        # Explicit co-registration claim.
        report["co_registered"] = bool(
            report["crs_compatible"]
            and report["overlap_detected"]
            and report["spatial_overlap_pct"] is not None
            and report["spatial_overlap_pct"] >= 90.0
            and report["resolution_compatible"]
        )

        report["valid"] = bool(
            report["crs_compatible"]
            and report["overlap_detected"]
        )

        return report
