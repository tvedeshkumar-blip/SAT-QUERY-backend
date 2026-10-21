import base64
import io
import os
import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("satquery.acquisition.validator")

class AssetValidator:
    """
    Physical Satellite Asset Validator.
    Verifies actual retrieved file binaries on disk or in memory rather than trusting catalog labels.
    Inspects format readability, raster band count, dimensions, data types, nodata values,
    CRS, affine transform, spatial bounds, and analysis task compatibility.
    """

    @classmethod
    def validate_asset(
        cls,
        data_or_path: Union[str, bytes],
        filename: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
        reported_category: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates the actual asset bytes or file.
        Returns a structured validation report including physical metadata,
        asset category, scientific status, and task compatibility.
        """
        raw_bytes: bytes
        target_filename = filename or "satellite_asset"

        # Determine whether input is file path, base64 data URI, or raw bytes
        if isinstance(data_or_path, str):
            if data_or_path.startswith("data:") or "," in data_or_path:
                clean_b64 = data_or_path.split(",")[-1]
                raw_bytes = base64.b64decode(clean_b64)
            elif os.path.isfile(data_or_path):
                target_filename = os.path.basename(data_or_path)
                with open(data_or_path, "rb") as f:
                    raw_bytes = f.read()
            else:
                try:
                    raw_bytes = base64.b64decode(data_or_path)
                except Exception:
                    raw_bytes = data_or_path.encode("utf-8", errors="ignore")
        elif isinstance(data_or_path, bytes):
            raw_bytes = data_or_path
        else:
            return cls._generate_unsupported_report(
                target_filename, 
                "Unsupported input type provided to asset validator.", 
                provenance
            )

        if not raw_bytes or len(raw_bytes) == 0:
            return cls._generate_unsupported_report(
                target_filename, 
                "Asset payload is empty (0 bytes).", 
                provenance
            )

        # Inspect binary format using Rasterio first
        rasterio_result = cls._inspect_with_rasterio(raw_bytes, target_filename)
        if rasterio_result["is_valid"]:
            return cls._build_validation_report(
                rasterio_result, 
                target_filename, 
                provenance, 
                reported_category
            )

        # Fallback inspection with PIL for standard image formats (JPEG, PNG)
        pil_result = cls._inspect_with_pil(raw_bytes, target_filename)
        if pil_result["is_valid"]:
            return cls._build_validation_report(
                pil_result, 
                target_filename, 
                provenance, 
                reported_category
            )

        return cls._generate_unsupported_report(
            target_filename, 
            f"Unable to decode file binary as either a GeoTIFF raster or valid image ({pil_result.get('error', 'unsupported format')}).",
            provenance
        )

    @classmethod
    def _inspect_with_rasterio(cls, raw_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Inspects raster structure via Rasterio MemoryFile."""
        try:
            import rasterio
            from rasterio.io import MemoryFile

            with MemoryFile(raw_bytes) as memfile:
                with memfile.open() as dataset:
                    is_geotiff = dataset.driver == "GTiff" or filename.lower().endswith((".tif", ".tiff"))
                    crs_str = str(dataset.crs) if dataset.crs else None
                    bounds_list = [float(b) for b in dataset.bounds] if dataset.bounds else None
                    transform_list = [float(x) for x in dataset.transform[:6]] if dataset.transform else None
                    resolution_list = [float(dataset.res[0]), float(dataset.res[1])] if dataset.res else None

                    dtypes_str = str(dataset.dtypes[0]) if dataset.dtypes else "unknown"

                    return {
                        "is_valid": True,
                        "driver": dataset.driver or "GTiff",
                        "is_geotiff": is_geotiff,
                        "georeferenced": bool(crs_str),
                        "width": int(dataset.width),
                        "height": int(dataset.height),
                        "bands": int(dataset.count),
                        "dtype": dtypes_str,
                        "nodata": dataset.nodata,
                        "crs": crs_str,
                        "crs_display": crs_str if crs_str else "CRS unavailable",
                        "bounds": bounds_list,
                        "transform": transform_list,
                        "resolution": resolution_list,
                        "tags": dict(dataset.tags()) if hasattr(dataset, "tags") else {},
                        "file_size_bytes": len(raw_bytes)
                    }
        except Exception as e:
            return {"is_valid": False, "error": str(e)}

    @classmethod
    def _inspect_with_pil(cls, raw_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Inspects standard image structure via PIL."""
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            img_format = img.format or ("JPEG" if filename.lower().endswith((".jpg", ".jpeg")) else "PNG")
            bands = len(img.getbands()) if hasattr(img, "getbands") else (1 if img.mode == "L" else 3)
            
            return {
                "is_valid": True,
                "driver": img_format,
                "is_geotiff": False,
                "georeferenced": False,
                "width": int(img.width),
                "height": int(img.height),
                "bands": int(bands),
                "dtype": "uint8",
                "nodata": None,
                "crs": None,
                "crs_display": "CRS unavailable (Unprojected Browse Image)",
                "bounds": None,
                "transform": None,
                "resolution": None,
                "tags": {},
                "file_size_bytes": len(raw_bytes)
            }
        except Exception as e:
            return {"is_valid": False, "error": str(e)}

    @classmethod
    def _build_validation_report(
        cls,
        meta: Dict[str, Any],
        filename: str,
        provenance: Optional[Dict[str, Any]],
        reported_category: Optional[str]
    ) -> Dict[str, Any]:
        """
        Synthesizes validated physical raster properties into scientific asset categories
        and per-task compatibility statements.
        """
        bands = meta["bands"]
        is_geotiff = meta["is_geotiff"]
        georeferenced = meta["georeferenced"]
        fname_lower = filename.lower()
        prov = provenance or {}
        collection = (prov.get("collection") or "").lower()
        is_sar = "sentinel-1" in collection or "sar" in collection or "risat" in fname_lower

        # 1. Determine physical asset category
        if not is_geotiff or not georeferenced:
            asset_category = "visual_preview"
            file_format = meta.get("driver", "JPEG")
            scientific_status = "visual_display_only"
        elif bands == 1:
            asset_category = "single_band_spectral"
            file_format = "GeoTIFF"
            scientific_status = "calibrated_surface_reflectance" if not is_sar else "calibrated_sar_backscatter"
        elif bands == 3:
            asset_category = "rgb_visual_raster"
            file_format = "GeoTIFF"
            scientific_status = "visual_display_only" if not is_sar else "calibrated_sar_backscatter"
        elif bands >= 4:
            asset_category = "multispectral_cube"
            file_format = "GeoTIFF"
            scientific_status = "calibrated_surface_reflectance"
        else:
            asset_category = "unsupported"
            file_format = meta.get("driver", "Unknown")
            scientific_status = "unverified"

        # 2. Formulate explicit scientific limitations
        limitations: List[str] = []
        if asset_category == "visual_preview":
            limitations.append("Visual preview browse asset (8-bit compressed RGB).")
            limitations.append("Lacks calibrated multispectral surface reflectance or raw SAR complex numbers.")
            limitations.append("Strictly prohibited for quantitative band mathematics (NDVI/NDWI) and SAR dB backscatter.")
        elif asset_category == "single_band_spectral":
            limitations.append(f"Single-band raster asset ({bands} channel, {meta['dtype']}).")
            limitations.append("Cannot perform bi-band ratioing (e.g. NDVI requires both Red and NIR bands).")
            if not is_sar:
                limitations.append("Display rendered as grayscale single-band reflectance.")
        elif asset_category == "rgb_visual_raster":
            limitations.append("3-band RGB visual raster (true-color rendered).")
            limitations.append("Does not contain Near-Infrared (Band 8); vegetative canopy analysis is limited to visible proxies.")
        elif asset_category == "multispectral_cube":
            limitations.append(f"Multispectral raster cube with {bands} bands ({meta['dtype']}).")
            limitations.append(f"Georeferenced to {meta['crs_display']}.")

        # 3. Task compatibility evaluation
        compatibility: Dict[str, Dict[str, str]] = {}

        # VQA
        if asset_category in ("visual_preview", "rgb_visual_raster", "multispectral_cube"):
            compatibility["vqa"] = {
                "status": "supported" if asset_category != "visual_preview" else "supported_with_limitations",
                "reason": "VLM and spectral baseline can inspect visible features in the scene." if asset_category != "visual_preview" else "Visual questions supported for observable browse features; uncalibrated."
            }
        elif asset_category == "single_band_spectral":
            compatibility["vqa"] = {
                "status": "supported_with_limitations",
                "reason": "Single-band grayscale raster; questions must consider single-channel reflectance/backscatter."
            }
        else:
            compatibility["vqa"] = {"status": "rejected", "reason": "Asset format unsupported for visual VQA."}

        # Captioning
        if asset_category in ("visual_preview", "rgb_visual_raster", "multispectral_cube", "single_band_spectral"):
            compatibility["captioning"] = {
                "status": "supported" if asset_category in ("rgb_visual_raster", "multispectral_cube") else "supported_with_limitations",
                "reason": "Descriptive summary derived from observable raster statistics."
            }
        else:
            compatibility["captioning"] = {"status": "rejected", "reason": "Asset format unsupported for captioning."}

        # Grounding
        if asset_category in ("visual_preview", "rgb_visual_raster", "multispectral_cube"):
            compatibility["grounding"] = {
                "status": "supported" if georeferenced else "supported_with_limitations",
                "reason": "Object localization operable on visual channels." if georeferenced else "Operable in image pixel space; unprojected coordinates."
            }
        elif asset_category == "single_band_spectral":
            compatibility["grounding"] = {
                "status": "supported_with_limitations",
                "reason": "Grayscale single-channel detection; chromatic targets may have lower detection salience."
            }
        else:
            compatibility["grounding"] = {"status": "rejected", "reason": "Asset format unsupported for visual grounding."}

        # Spectral Indices (NDVI / NDWI)
        if asset_category == "visual_preview":
            compatibility["spectral_indices"] = {
                "status": "rejected",
                "reason": "Visual preview browse asset is an 8-bit RGB/JPEG image. Quantitative NDVI/NDWI calculation is rejected because calibrated multispectral surface reflectance bands (Red Band 4 and NIR Band 8) are absent."
            }
        elif asset_category == "single_band_spectral":
            compatibility["spectral_indices"] = {
                "status": "rejected",
                "reason": "Asset contains only 1 spectral channel. NDVI requires both Red (Band 4) and NIR (Band 8) bands."
            }
        elif asset_category == "rgb_visual_raster":
            compatibility["spectral_indices"] = {
                "status": "rejected",
                "reason": "True-color visual RGB raster lacks calibrated NIR (Band 8) channel required for physical NDVI calculation."
            }
        elif asset_category == "multispectral_cube":
            compatibility["spectral_indices"] = {
                "status": "supported",
                "reason": "Multispectral raster contains required surface reflectance bands."
            }
        else:
            compatibility["spectral_indices"] = {"status": "rejected", "reason": "Unsupported raster format."}

        # SAR Decibel Backscatter
        if is_sar and is_geotiff:
            compatibility["sar_backscatter"] = {
                "status": "supported",
                "reason": "Calibrated microwave SAR raster suitable for sigma-0 backscatter decibel modeling."
            }
        else:
            compatibility["sar_backscatter"] = {
                "status": "rejected",
                "reason": "Asset is not a microwave SAR raster (requires Sentinel-1 or RISAT linear amplitude GeoTIFF)."
            }

        return {
            "status": "valid",
            "filename": filename,
            "file_format": file_format,
            "asset_category": asset_category,
            "raster_metadata": meta,
            "scientific_status": scientific_status,
            "analysis_compatibility": compatibility,
            "scientific_limitations": limitations,
            "provenance": prov,
            "error": None
        }

    @classmethod
    def _generate_unsupported_report(
        cls, 
        filename: str, 
        error_msg: str, 
        provenance: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generates a structured rejection report for corrupted or unreadable files."""
        return {
            "status": "invalid",
            "filename": filename,
            "file_format": "UNKNOWN",
            "asset_category": "unsupported",
            "raster_metadata": {
                "width": 0,
                "height": 0,
                "bands": 0,
                "dtype": "unknown",
                "crs": None,
                "crs_display": "CRS unavailable",
                "is_geotiff": False,
                "georeferenced": False
            },
            "scientific_status": "unverified",
            "analysis_compatibility": {
                "vqa": {"status": "rejected", "reason": error_msg},
                "captioning": {"status": "rejected", "reason": error_msg},
                "grounding": {"status": "rejected", "reason": error_msg},
                "spectral_indices": {"status": "rejected", "reason": error_msg},
                "sar_backscatter": {"status": "rejected", "reason": error_msg}
            },
            "scientific_limitations": [f"File unreadable: {error_msg}"],
            "provenance": provenance or {},
            "error": error_msg
        }
