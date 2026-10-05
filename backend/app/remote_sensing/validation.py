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

        clean_b64 = data_b64.split(",")[-1]
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
            left = max(bounds_a[0], bounds_b[0])
            bottom = max(bounds_a[1], bounds_b[1])
            right = min(bounds_a[2], bounds_b[2])
            top = min(bounds_a[3], bounds_b[3])

            if right > left and top > bottom:
                intersection_area = (right - left) * (top - bottom)

                area_a = (
                    (bounds_a[2] - bounds_a[0])
                    * (bounds_a[3] - bounds_a[1])
                )
                area_b = (
                    (bounds_b[2] - bounds_b[0])
                    * (bounds_b[3] - bounds_b[1])
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
        elif crs_a and crs_b:
            report["warnings"].append(
                "Spatial bounds unavailable for one or both scenes."
            )

        if crs_a and crs_b and not report["overlap_detected"]:
            report["warnings"].append(
                "No positive spatial overlap established between scenes."
            )

        # Resolution compatibility.
        res_a = meta_a.get("resolution")
        res_b = meta_b.get("resolution")

        if res_a and res_b:
            try:
                ratio_x = max(res_a[0], res_b[0]) / max(
                    1e-6, min(res_a[0], res_b[0])
                )
                ratio_y = max(res_a[1], res_b[1]) / max(
                    1e-6, min(res_a[1], res_b[1])
                )
                ratio = max(ratio_x, ratio_y)

                if ratio > 10.0:
                    report["resolution_compatible"] = False
                    report["warnings"].append(
                        f"Large resolution disparity ({ratio:.1f}x)."
                    )
            except (TypeError, ValueError, IndexError):
                report["resolution_compatible"] = False

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
