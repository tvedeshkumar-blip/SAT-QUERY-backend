import numpy as np
from PIL import Image
import io
import base64
from typing import Dict, Any, Tuple, Optional

def normalize_to_uint8(arr: np.ndarray, clip_percentiles: Tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
    """
    Normalizes multi-bit satellite rasters (uint16, int16, float32) into 8-bit uint8 using percentile clipping.
    Safely handles NaNs, infinities, and flat constant arrays.
    """
    if arr.dtype == np.uint8 and not np.isnan(arr).any():
        return arr

    arr_clean = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    
    # Compute percentiles on non-zero pixels if possible to avoid nodata bias
    non_zero = arr_clean[arr_clean != 0]
    if len(non_zero) > 100:
        p_low, p_high = np.percentile(non_zero, clip_percentiles)
    else:
        p_low, p_high = np.percentile(arr_clean, clip_percentiles)
    
    if p_high > p_low:
        normalized = np.clip((arr_clean - p_low) / (p_high - p_low) * 255.0, 0, 255)
    else:
        # Uniform or low contrast
        max_val = np.max(arr_clean)
        min_val = np.min(arr_clean)
        if max_val > min_val:
            normalized = ((arr_clean - min_val) / (max_val - min_val) * 255.0)
        else:
            normalized = np.zeros_like(arr_clean)
        
    return normalized.astype(np.uint8)

def robust_remote_sensing_preprocess(
    arr: np.ndarray, 
    metadata: Optional[Dict[str, Any]] = None,
    target_modality: Optional[str] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Scientific remote sensing preprocessing pipeline.
    Preserves radiometric integrity, handles nodata/NaNs, records explicit transformation provenance.
    Never silently converts multispectral/SAR data without metadata records.
    """
    meta = (metadata or {}).copy()
    provenance: Dict[str, Any] = {
        "source_format": meta.get("driver") or ("GeoTIFF" if meta.get("is_geotiff") else "Standard Raster"),
        "source_dimensions": [int(arr.shape[0]), int(arr.shape[1])],
        "source_dtype": str(arr.dtype),
        "nodata_value": meta.get("nodata"),
        "nan_detected": bool(np.isnan(arr).any()),
    }

    # Clean NaNs and Infs
    arr_clean = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)

    # Determine band count and layout
    if arr_clean.ndim == 2:
        band_count = 1
    elif arr_clean.ndim == 3:
        band_count = arr_clean.shape[2]
    else:
        raise ValueError(f"Unsupported raster dimension: {arr_clean.ndim}")

    provenance["source_bands"] = band_count
    modality = (target_modality or meta.get("modality") or "OPTICAL").upper()
    provenance["detected_modality"] = modality

    # Modality-aware transformation
    if modality == "SAR":
        # SAR single-band microwave backscatter
        if band_count > 1 and arr_clean.ndim == 3:
            sar_band = arr_clean[:, :, 0]
            provenance["preprocessing"] = "extracted_polarization_band_0"
        else:
            sar_band = arr_clean if arr_clean.ndim == 2 else arr_clean[:, :, 0]
            provenance["preprocessing"] = "single_band_microwave_intensity"

        # Apply log/dB transformation if values appear to be power/amplitude
        max_val = float(np.max(sar_band))
        if max_val > 10.0:
            # Linear power or amplitude to dB proxy: 10 * log10(val + 1e-4)
            db_band = 10.0 * np.log10(np.clip(sar_band, 1e-4, None))
            norm_sar = normalize_to_uint8(db_band)
            provenance["normalization"] = "decibel_db_scale_percentile_2_98"
        else:
            norm_sar = normalize_to_uint8(sar_band)
            provenance["normalization"] = "linear_percentile_2_98"

        output_arr = norm_sar
    else:
        # Optical / Multispectral
        if band_count == 1:
            output_arr = normalize_to_uint8(arr_clean)
            provenance["preprocessing"] = "single_band_panchromatic_to_grayscale"
            provenance["normalization"] = "percentile_2_98"
        elif band_count == 3:
            output_arr = normalize_to_uint8(arr_clean)
            provenance["preprocessing"] = "standard_rgb_pass_through"
            provenance["normalization"] = "percentile_2_98"
        elif band_count == 4 and (meta.get("color_mode") == "RGBA" or meta.get("has_alpha") or not meta.get("is_geotiff")):
            # RGBA image: handle alpha channel properly
            rgb_bands = arr_clean[:, :, :3]
            alpha_band = arr_clean[:, :, 3]
            if np.any(alpha_band < 254.0):
                alpha_norm = np.clip(alpha_band / 255.0, 0.0, 1.0)[:, :, None]
                blended = rgb_bands * alpha_norm + 255.0 * (1.0 - alpha_norm)
                output_arr = normalize_to_uint8(blended)
                provenance["preprocessing"] = "rgba_alpha_blended_to_rgb"
                provenance["alpha_blended"] = True
            else:
                output_arr = normalize_to_uint8(rgb_bands)
                provenance["preprocessing"] = "rgba_alpha_opaque_to_rgb"
            provenance["is_visual_preview"] = True
            provenance["nir_preserved"] = False
            provenance["normalization"] = "percentile_2_98"
        elif band_count >= 4:
            # Multispectral (e.g. B, G, R, NIR) -> Extract natural color RGB (bands 1,2,3)
            # and preserve NIR band in metadata
            rgb_bands = arr_clean[:, :, :3]
            output_arr = normalize_to_uint8(rgb_bands)
            provenance["preprocessing"] = "multispectral_bands_1_2_3_to_rgb"
            provenance["nir_preserved"] = True
            provenance["normalization"] = "percentile_2_98"
        else:
            output_arr = normalize_to_uint8(arr_clean)
            provenance["preprocessing"] = "generic_band_normalization"
            provenance["normalization"] = "percentile_2_98"

    meta["provenance"] = provenance
    meta["preprocessing"] = provenance["preprocessing"]
    meta["normalization"] = provenance["normalization"]
    return output_arr, meta

def convert_array_to_base64_png(arr: np.ndarray) -> str:
    """
    Converts a NumPy array (RGB or Grayscale uint8) to a Base64-encoded PNG image string.
    Safely handles rasterio channel-first (C, H, W) and standard (H, W, C) arrays.
    """
    # 1. Channel-first transpose if rasterio layout (C, H, W)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and (arr.shape[0] < arr.shape[1] or arr.shape[2] > 4):
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[0] in (3, 4):
            arr = np.transpose(arr[:3], (1, 2, 0))
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]

    arr_uint8 = normalize_to_uint8(arr)
    if arr_uint8.ndim == 2:
        img = Image.fromarray(arr_uint8, mode="L")
    elif arr_uint8.ndim == 3 and arr_uint8.shape[2] == 3:
        img = Image.fromarray(arr_uint8, mode="RGB")
    else:
        sq = arr_uint8.squeeze()
        if sq.ndim == 2:
            img = Image.fromarray(sq, mode="L")
        elif sq.ndim == 3 and sq.shape[2] == 3:
            img = Image.fromarray(sq, mode="RGB")
        else:
            img = Image.fromarray(sq[:, :, 0] if sq.ndim == 3 else sq, mode="L")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def make_json_safe(obj: Any) -> Any:
    """
    Recursively converts NumPy arrays, NumPy scalars, and nested structures
    into JSON-serializable Python types.
    - 2D/3D numpy arrays (including boolean masks) are converted to base64-encoded PNG image strings.
    - 1D numpy arrays are converted to Python lists.
    - NumPy scalars (int, float, bool) are converted to native Python primitives.
    - NaNs and Infs are converted to None.
    - Preserves nested dicts and lists.
    """
    if hasattr(obj, "model_dump"):
        return make_json_safe(obj.model_dump())
    if isinstance(obj, np.ndarray):
        if obj.ndim in (2, 3):
            try:
                if obj.dtype == bool:
                    return convert_array_to_base64_png(obj.astype(np.uint8) * 255)
                return convert_array_to_base64_png(obj)
            except Exception:
                return None
        elif obj.ndim == 1:
            return [make_json_safe(x) for x in obj.tolist()]
        elif obj.ndim == 0:
            return make_json_safe(obj.item())
        return None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj

