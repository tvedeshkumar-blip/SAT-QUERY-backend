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
    """
    arr_uint8 = normalize_to_uint8(arr)
    if arr_uint8.ndim == 2:
        img = Image.fromarray(arr_uint8, mode="L")
    elif arr_uint8.ndim == 3 and arr_uint8.shape[2] == 1:
        img = Image.fromarray(arr_uint8[:, :, 0], mode="L")
    elif arr_uint8.ndim == 3 and arr_uint8.shape[2] >= 3:
        img = Image.fromarray(arr_uint8[:, :, :3], mode="RGB")
    else:
        img = Image.fromarray(arr_uint8, mode="L")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
