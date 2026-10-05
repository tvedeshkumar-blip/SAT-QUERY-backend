import base64
import io
import logging
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("satquery.geotiff")

def parse_geotiff_or_image(image_base64: str, filename: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Parses an incoming Base64 image payload. Attempts rasterio GeoTIFF parsing first.
    If rasterio is unavailable or input is standard PNG/JPEG, falls back gracefully to PIL.
    Preserves spatial metadata (CRS, bounds, transform, band count, nodata, resolution)
    wherever genuinely available.

    CRITICAL INTEGRITY RULE: Never guesses CRS (e.g. EPSG:32644 fallback is forbidden).
    If CRS is not in metadata or tags, metadata['crs'] = None.
    """
    clean_b64 = image_base64.split(",")[-1] if "," in image_base64 else image_base64
    image_bytes = base64.b64decode(clean_b64)

    fname = (filename or "uploaded_raster.png").lower()
    is_tiff_name = fname.endswith(".tif") or fname.endswith(".tiff")

    metadata: Dict[str, Any] = {
        "filename": filename or "uploaded_raster.png",
        "crs": None,
        "crs_display": "CRS unavailable",
        "bounds": None,
        "transform": None,
        "resolution": None,
        "width": 0,
        "height": 0,
        "bands": 3,
        "dtype": "uint8",
        "nodata": None,
        "modality": "OPTICAL",
        "is_geotiff": False,
        "georeferenced": False,
        "tags": {}
    }

    # Attempt Rasterio extraction
    try:
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(image_bytes) as memfile:
            with memfile.open() as dataset:
                is_tiff = (dataset.driver == "GTiff") or is_tiff_name
                metadata["is_geotiff"] = is_tiff
                metadata["width"] = int(dataset.width)
                metadata["height"] = int(dataset.height)
                metadata["bands"] = int(dataset.count)
                metadata["dtype"] = str(dataset.dtypes[0])
                metadata["nodata"] = dataset.nodata
                metadata["tags"] = dict(dataset.tags()) if hasattr(dataset, "tags") else {}

                # Strict CRS handling
                if dataset.crs:
                    metadata["crs"] = str(dataset.crs)
                    metadata["crs_display"] = str(dataset.crs)
                    metadata["georeferenced"] = True
                elif is_tiff:
                    # Check for explicit PIL GeoAsciiParamsTag (34737)
                    try:
                        pil_img = Image.open(io.BytesIO(image_bytes))
                        if hasattr(pil_img, "tag_v2"):
                            ascii_params = pil_img.tag_v2.get(34737)
                            if ascii_params:
                                ascii_str = ascii_params.decode("utf-8", errors="ignore") if isinstance(ascii_params, bytes) else str(ascii_params)
                                cleaned = ascii_str.strip().strip("|")
                                if "EPSG:" in cleaned:
                                    for part in cleaned.split("|"):
                                        if "EPSG:" in part:
                                            metadata["crs"] = part.strip()
                                            metadata["crs_display"] = metadata["crs"]
                                            metadata["georeferenced"] = True
                                            break
                                elif "WGS" in cleaned or "UTM" in cleaned:
                                    metadata["crs"] = cleaned
                                    metadata["crs_display"] = cleaned
                                    metadata["georeferenced"] = True
                    except Exception:
                        pass

                if not metadata["crs"]:
                    metadata["crs"] = None
                    metadata["crs_display"] = "CRS unavailable"
                    metadata["georeferenced"] = False

                if metadata.get("georeferenced") and dataset.bounds:
                    b = dataset.bounds
                    metadata["bounds"] = [float(b.left), float(b.bottom), float(b.right), float(b.top)]

                if metadata.get("georeferenced") and dataset.transform:
                    metadata["transform"] = [float(x) for x in dataset.transform[:6]]
                    metadata["resolution"] = [float(dataset.res[0]), float(dataset.res[1])]

                # Read array
                arr = dataset.read()
                # Transpose from (bands, height, width) to (height, width, bands)
                if arr.ndim == 3:
                    arr = np.transpose(arr, (1, 2, 0))

                logger.info(f"Read GeoTIFF with Rasterio: shape={arr.shape}, CRS={metadata['crs']}")
                return arr, metadata
    except Exception as e:
        logger.debug(f"Rasterio parse not applicable or failed ({e}); falling back to PIL.")

    # Fallback: PIL standard decode
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        is_tiff = pil_img.format in ("TIFF", "TIF") or is_tiff_name

        if is_tiff and hasattr(pil_img, "tag_v2"):
            tags = dict(pil_img.tag_v2)
            metadata["is_geotiff"] = True
            
            pixel_scale = tags.get(33550)
            tie_points = tags.get(33922)

            if pixel_scale and tie_points and len(tie_points) >= 6:
                x0 = float(tie_points[3])
                y0 = float(tie_points[4])
                dx = float(pixel_scale[0])
                dy = float(pixel_scale[1])
                w = pil_img.width
                h = pil_img.height
                left = x0
                top = y0
                right = x0 + (w * dx)
                bottom = y0 - (h * dy)
                metadata["bounds"] = [round(left, 4), round(bottom, 4), round(right, 4), round(top, 4)]
                metadata["resolution"] = [dx, dy]
                metadata["transform"] = [x0, dx, 0.0, y0, 0.0, -dy]

            ascii_params = tags.get(34737)
            if ascii_params and isinstance(ascii_params, (str, bytes)):
                ascii_str = ascii_params.decode("utf-8", errors="ignore") if isinstance(ascii_params, bytes) else ascii_params
                cleaned_ascii = ascii_str.strip().strip("|")
                if "EPSG:" in cleaned_ascii:
                    for part in cleaned_ascii.split("|"):
                        if "EPSG:" in part:
                            metadata["crs"] = part.strip()
                            metadata["crs_display"] = metadata["crs"]
                            metadata["georeferenced"] = True
                            break
                elif "WGS" in cleaned_ascii or "UTM" in cleaned_ascii:
                    metadata["crs"] = cleaned_ascii
                    metadata["crs_display"] = cleaned_ascii
                    metadata["georeferenced"] = True

            if not metadata["crs"]:
                metadata["crs"] = None
                metadata["crs_display"] = "CRS unavailable"
                metadata["georeferenced"] = False
                metadata["bounds"] = None
                metadata["transform"] = None
                metadata["resolution"] = None

        # Convert to numpy array
        if pil_img.mode not in ("RGB", "L", "RGBA"):
            pil_img = pil_img.convert("RGB")

        arr = np.array(pil_img)
        metadata["width"] = int(pil_img.width)
        metadata["height"] = int(pil_img.height)
        metadata["bands"] = int(arr.shape[2] if arr.ndim == 3 else 1)
        metadata["dtype"] = str(arr.dtype)

        if not metadata["crs"]:
            metadata["crs"] = None
            metadata["crs_display"] = "CRS unavailable"

        return arr, metadata
    except Exception as err:
        logger.error(f"Failed to decode image payload: {err}")
        raise ValueError(f"Invalid satellite image payload or unsupported file format: {err}")
