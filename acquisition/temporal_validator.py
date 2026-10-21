import os
import re
import base64
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

from app.acquisition.asset_validator import AssetValidator
from app.remote_sensing.registration import compute_bounds_overlap

logger = logging.getLogger("satquery.acquisition.temporal_validator")

class TemporalPairValidator:
    """
    Physical Satellite Temporal Pair Validator.
    Validates bi-temporal asset pairs before inference.
    Checks:
    1. Readability and raster data suitability (rejects browse-only thumbnails for change analysis).
    2. Modality compatibility (strictly rejects uncalibrated cross-sensor optical-SAR dynamic fusion in Stage 5).
    3. Chronological validity and acquisition timestamps (rejects degenerate identical timestamps).
    4. Georeferencing and CRS compatibility (rejects unprojected or mismatched CRS).
    5. Spatial overlap within the active AOI (rejects zero or insufficient overlap).
    6. Confounders (cloud cover delta, seasonal phenology, resolution disparity).
    7. Model readiness (reports truthful BIT-CD vs PixelDifferenceChangeBaseline fallback).
    """

    @classmethod
    def validate_pair(
        cls,
        t1_input: Optional[Union[str, bytes]] = None,
        t2_input: Optional[Union[str, bytes]] = None,
        t1_filename: Optional[str] = None,
        t2_filename: Optional[str] = None,
        t1_provenance: Optional[Dict[str, Any]] = None,
        t2_provenance: Optional[Dict[str, Any]] = None,
        t1_metadata: Optional[Dict[str, Any]] = None,
        t2_metadata: Optional[Dict[str, Any]] = None,
        aoi_bbox: Optional[List[float]] = None,
        retrieval_dir: Optional[str] = None,
        samples_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates a bi-temporal scene pair.
        Returns a structured validation response matching STACTemporalPairValidateResponse.
        """
        t1_prov = t1_provenance or {}
        t2_prov = t2_provenance or {}
        t1_meta_in = t1_metadata or {}
        t2_meta_in = t2_metadata or {}

        # Resolve paths/payloads if scene IDs or names were passed
        t1_data, fname_1 = cls._resolve_payload(t1_input, t1_filename, t1_prov, retrieval_dir, samples_dir)
        t2_data, fname_2 = cls._resolve_payload(t2_input, t2_filename, t2_prov, retrieval_dir, samples_dir)

        warnings: List[str] = []
        limitations: List[str] = []

        # 1. Missing Input Checks
        if t1_data is None or t2_data is None:
            missing_part = "T1 (pre-event)" if t1_data is None else "T2 (post-event)"
            if t1_data is None and t2_data is None:
                missing_part = "both T1 and T2"
            return cls._reject(
                reason=f"Missing temporal scene: {missing_part} asset payload is not available. Both scenes must be retrieved before validation.",
                t1_val=None,
                t2_val=None,
                warnings=["Select and retrieve both candidate scenes before running change detection."],
                limitations=["Bi-temporal analysis requires two distinct scene assets."]
            )

        # 2. Physical Asset Validation for Each Scene
        val_1 = AssetValidator.validate_asset(t1_data, filename=fname_1, provenance=t1_prov)
        val_2 = AssetValidator.validate_asset(t2_data, filename=fname_2, provenance=t2_prov)

        if val_1.get("status") != "valid":
            err = val_1.get("error") or "Unreadable file binary"
            return cls._reject(
                reason=f"T1 asset validation failed: {err}",
                t1_val=val_1,
                t2_val=val_2,
                warnings=[f"T1 file error: {err}"],
                limitations=["T1 asset could not be decoded as a valid raster."]
            )

        if val_2.get("status") != "valid":
            err = val_2.get("error") or "Unreadable file binary"
            return cls._reject(
                reason=f"T2 asset validation failed: {err}",
                t1_val=val_1,
                t2_val=val_2,
                warnings=[f"T2 file error: {err}"],
                limitations=["T2 asset could not be decoded as a valid raster."]
            )

        # 3. Reject Preview-Only Browse Assets (Thumbnails)
        is_t1_preview = (
            val_1.get("asset_category") == "visual_preview"
            or t1_prov.get("asset_key") in ("thumbnail", "preview")
            or fname_1.lower().endswith((".jpg", ".jpeg"))
            or not val_1.get("raster_metadata", {}).get("is_geotiff", False)
        )
        is_t2_preview = (
            val_2.get("asset_category") == "visual_preview"
            or t2_prov.get("asset_key") in ("thumbnail", "preview")
            or fname_2.lower().endswith((".jpg", ".jpeg"))
            or not val_2.get("raster_metadata", {}).get("is_geotiff", False)
        )

        if is_t1_preview or is_t2_preview:
            culprit = "T1" if (is_t1_preview and not is_t2_preview) else ("T2" if is_t2_preview and not is_t1_preview else "both T1 and T2")
            return cls._reject(
                reason=(
                    f"Browse-only preview asset detected in {culprit}. "
                    "Unprojected 8-bit visual previews (JPEG/PNG browse images) are strictly rejected for scientific bi-temporal change detection. "
                    "Calibrated, georeferenced GeoTIFF rasters with surface reflectance or calibrated backscatter are required."
                ),
                t1_val=val_1,
                t2_val=val_2,
                warnings=["Select a genuine GeoTIFF raster asset (e.g. 'visual' or multispectral band) rather than 'thumbnail'."],
                limitations=["Browse previews lack calibrated surface reflectance and geometric projection."]
            )

        # 4. Modality Compatibility Check
        mod_1 = (t1_prov.get("modality") or t1_meta_in.get("modality") or ("SAR" if "sar" in fname_1.lower() else "OPTICAL")).upper()
        mod_2 = (t2_prov.get("modality") or t2_meta_in.get("modality") or ("SAR" if "sar" in fname_2.lower() else "OPTICAL")).upper()

        if mod_1 != mod_2:
            return cls._reject(
                reason=(
                    f"Incompatible sensor modalities: T1 is {mod_1} and T2 is {mod_2}. "
                    "Multi-sensor Optical-to-SAR change detection is not supported for dynamically acquired scenes in Stage 5. "
                    "Both acquisitions must share the same sensor modality."
                ),
                t1_val=val_1,
                t2_val=val_2,
                warnings=[f"Modality disparity: {mod_1} vs {mod_2}"],
                limitations=["Cross-modality optical-SAR change detection requires specialized domain mapping."]
            )

        # 5. Timestamp and Chronology Validation
        dt1_str = t1_prov.get("datetime") or t1_meta_in.get("datetime")
        dt2_str = t2_prov.get("datetime") or t2_meta_in.get("datetime")
        chronology_status = "unknown"
        delta_days: Optional[float] = None

        if dt1_str and dt2_str:
            dt1 = cls._parse_iso_datetime(dt1_str)
            dt2 = cls._parse_iso_datetime(dt2_str)

            if dt1 and dt2:
                # Degenerate pair check: identical timestamps
                if dt1 == dt2:
                    return cls._reject(
                        reason=(
                            f"Degenerate temporal pair: T1 and T2 have identical acquisition timestamps ({dt1_str}). "
                            "Bi-temporal change detection requires distinct pre-event and post-event observations."
                        ),
                        t1_val=val_1,
                        t2_val=val_2,
                        t1_datetime=dt1_str,
                        t2_datetime=dt2_str,
                        temporal_delta_days=0.0,
                        chronology_status="identical",
                        warnings=["Acquisition dates are identical."],
                        limitations=["Zero temporal separation between scenes."]
                    )

                diff_seconds = (dt2 - dt1).total_seconds()
                delta_days = round(diff_seconds / 86400.0, 2)

                if delta_days < 0:
                    chronology_status = "inverted"
                    warnings.append(
                        f"Temporal inversion: T1 ({dt1_str}) was acquired after T2 ({dt2_str}) "
                        f"(Delta: {delta_days} days). Ensure T1 represents the pre-event baseline."
                    )
                else:
                    chronology_status = "chronological"

                # Seasonal phenology confounder
                if abs(delta_days) > 120 and mod_1 == "OPTICAL":
                    limitations.append(
                        f"Seasonal phenology caution: {abs(delta_days):.1f} days elapsed between acquisitions. "
                        "Observed spectral differences may be dominated by seasonal vegetation phenology or sun angle rather than structural land cover changes."
                    )
            else:
                warnings.append("Could not parse ISO datetime strings for accurate temporal baseline calculation.")
        else:
            warnings.append("Acquisition timestamp metadata missing for one or both scenes; temporal baseline duration is unconfirmed.")

        # 6. Coordinate Reference System (CRS) & Georeferencing
        meta_1 = val_1.get("raster_metadata", {})
        meta_2 = val_2.get("raster_metadata", {})

        crs_1 = meta_1.get("crs")
        crs_2 = meta_2.get("crs")
        georef_1 = meta_1.get("georeferenced", False)
        georef_2 = meta_2.get("georeferenced", False)

        if not georef_1 or not georef_2 or not crs_1 or not crs_2:
            return cls._reject(
                reason=(
                    "Georeferencing missing: One or both scenes lack coordinate reference system (CRS) tags. "
                    "Accurate spatial co-registration cannot be verified."
                ),
                t1_val=val_1,
                t2_val=val_2,
                t1_datetime=dt1_str,
                t2_datetime=dt2_str,
                temporal_delta_days=delta_days,
                chronology_status=chronology_status,
                warnings=["Inputs lack geodetic CRS coordinates."],
                limitations=["Unprojected scenes cannot be spatially co-registered."]
            )

        if crs_1 != crs_2:
            return cls._reject(
                reason=(
                    f"CRS mismatch: T1 projection is '{crs_1}' while T2 projection is '{crs_2}'. "
                    "Automatic reprojection between differing coordinate systems without verified ground control points is disallowed to prevent geometric distortion."
                ),
                t1_val=val_1,
                t2_val=val_2,
                t1_datetime=dt1_str,
                t2_datetime=dt2_str,
                temporal_delta_days=delta_days,
                chronology_status=chronology_status,
                warnings=[f"CRS mismatch: {crs_1} vs {crs_2}"],
                limitations=["Scenes exist in different spatial coordinate systems."]
            )

        # 7. Spatial Footprint Overlap Calculation
        bounds_1 = meta_1.get("bounds")
        bounds_2 = meta_2.get("bounds")
        overlap_pct = 0.0

        if bounds_1 and bounds_2:
            overlap_pct = compute_bounds_overlap(bounds_1, bounds_2)

            if overlap_pct <= 0.0:
                return cls._reject(
                    reason=(
                        "Zero spatial overlap: T1 footprint and T2 footprint do not intersect. "
                        "Bi-temporal change detection requires overlapping geographical coverage."
                    ),
                    t1_val=val_1,
                    t2_val=val_2,
                    t1_datetime=dt1_str,
                    t2_datetime=dt2_str,
                    temporal_delta_days=delta_days,
                    chronology_status=chronology_status,
                    spatial_overlap_pct=0.0,
                    warnings=["No common spatial footprint exists between these two scenes."],
                    limitations=["Disjoint scene footprints."]
                )

            if overlap_pct < 10.0:
                return cls._reject(
                    reason=(
                        f"Insufficient spatial overlap: Scenes share only {overlap_pct:.1f}% mutual footprint "
                        "(minimum required is 10.0%)."
                    ),
                    t1_val=val_1,
                    t2_val=val_2,
                    t1_datetime=dt1_str,
                    t2_datetime=dt2_str,
                    temporal_delta_days=delta_days,
                    chronology_status=chronology_status,
                    spatial_overlap_pct=overlap_pct,
                    warnings=[f"Marginal spatial overlap ({overlap_pct:.1f}%)."],
                    limitations=["Overlap area is too small for dependable change analysis."]
                )
        else:
            warnings.append("Spatial boundary vectors unavailable; overlap percentage could not be geometrically calculated.")

        # 8. Resolution & Dimension Compatibility
        res_1 = meta_1.get("resolution")
        res_2 = meta_2.get("resolution")
        res_compatible = True

        if res_1 and res_2:
            try:
                ratio_x = max(res_1[0], res_2[0]) / max(1e-6, min(res_1[0], res_2[0]))
                ratio_y = max(res_1[1], res_2[1]) / max(1e-6, min(res_1[1], res_2[1]))
                ratio = max(ratio_x, ratio_y)
                if ratio > 5.0:
                    res_compatible = False
                    warnings.append(
                        f"Significant resolution disparity ({ratio:.1f}x) between T1 and T2. "
                        "Resampling will smooth fine-scale change features."
                    )
            except Exception:
                res_compatible = True

        w1, h1 = meta_1.get("width", 0), meta_1.get("height", 0)
        w2, h2 = meta_2.get("width", 0), meta_2.get("height", 0)
        if (w1, h1) != (w2, h2):
            warnings.append(
                f"Dimension discrepancy: T1 ({w1}x{h1}) vs T2 ({w2}x{h2}). "
                "Secondary raster will be resampled to match primary raster grid."
            )

        # 9. Confounder Analysis: Cloud Cover
        cloud_1 = t1_prov.get("cloud_cover")
        cloud_2 = t2_prov.get("cloud_cover")
        if cloud_1 is not None and cloud_2 is not None and mod_1 == "OPTICAL":
            if cloud_1 > 30.0 or cloud_2 > 30.0:
                warnings.append(
                    f"Cloud cover advisory: T1 has {cloud_1}%, T2 has {cloud_2}%. "
                    "Cloud formations and shadow occlusions will induce apparent spectral differences."
                )
                limitations.append(
                    "Cloud contamination: Visible differences may be dominated by cloud masks rather than ground changes."
                )

        # 10. Scientific Caveat and Provenance
        limitations.append(
            "Scientific distinction: Detected differences represent raw pixel reflectance deltas and model candidate clusters, "
            "NOT confirmed real-world land alterations. Solar elevation, atmospheric turbidity, and co-registration precision affect observed delta."
        )

        # 11. Model & Provider Readiness
        candidate_providers, primary_prov, fallback_stat = cls._assess_provider_readiness()

        co_registered = bool(crs_1 == crs_2 and overlap_pct >= 90.0 and res_compatible)
        status = "compatible" if not warnings else "compatible_with_limitations"

        return {
            "status": status,
            "is_compatible": True,
            "rejection_reason": None,
            "t1_validation": val_1,
            "t2_validation": val_2,
            "t1_datetime": dt1_str,
            "t2_datetime": dt2_str,
            "temporal_delta_days": delta_days,
            "chronology_status": chronology_status,
            "spatial_overlap_pct": overlap_pct,
            "overlap_detected": bool(overlap_pct > 0.0),
            "crs_compatible": True,
            "resolution_compatible": res_compatible,
            "modality_compatible": True,
            "co_registered": co_registered,
            "warnings": warnings,
            "scientific_limitations": limitations,
            "candidate_providers": candidate_providers,
            "primary_provider": primary_prov,
            "fallback_status": fallback_stat,
            "recommended_action": (
                "Temporal pair is compatible. Proceed with bi-temporal change detection."
                if status == "compatible"
                else "Temporal pair is compatible with scientific caveats. Review warnings before executing inference."
            )
        }

    @classmethod
    def _assess_provider_readiness(cls) -> Tuple[List[str], str, str]:
        """Checks if BIT-CD neural model checkpoint is available on disk."""
        try:
            from app.models.change_detection.change_detector import SiameseRSChangeDetector
            deep_detector = SiameseRSChangeDetector()
            if deep_detector.is_available:
                return (
                    ["BIT-CD (Bitemporal Transformer)", "PixelDifferenceChangeBaseline"],
                    "BIT-CD",
                    "Loaded and ready"
                )
        except Exception:
            pass

        return (
            ["PixelDifferenceChangeBaseline"],
            "PixelDifferenceChangeBaseline",
            "Active fallback (BIT-CD weights unmounted)"
        )

    @classmethod
    def _resolve_payload(
        cls,
        payload_input: Optional[Union[str, bytes]],
        filename: Optional[str],
        prov: Dict[str, Any],
        retrieval_dir: Optional[str],
        samples_dir: Optional[str]
    ) -> Tuple[Optional[Union[str, bytes]], str]:
        """Resolves raw bytes or local file path from scene identifiers or payload."""
        target_fname = filename or prov.get("filename") or "scene.tif"

        if payload_input is not None:
            return payload_input, target_fname

        # Check local sample ID
        scene_id = prov.get("scene_id") or prov.get("id") or ""
        if scene_id.startswith("local_sample_"):
            clean_name = scene_id.replace("local_sample_", "") + ".tif"
            base_dir = samples_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "samples"))
            sample_p = os.path.join(base_dir, clean_name)
            if os.path.exists(sample_p):
                return sample_p, clean_name

        # Check retrieval cache
        if scene_id and retrieval_dir and os.path.exists(retrieval_dir):
            for f in os.listdir(retrieval_dir):
                if f.startswith(scene_id):
                    return os.path.join(retrieval_dir, f), f

        return None, target_fname

    @classmethod
    def _parse_iso_datetime(cls, dt_str: str) -> Optional[datetime]:
        """Parses ISO 8601 datetime string."""
        try:
            clean = dt_str.replace("Z", "+00:00")
            return datetime.fromisoformat(clean)
        except Exception:
            try:
                # Basic YYYY-MM-DD
                return datetime.strptime(dt_str[:10], "%Y-%m-%d")
            except Exception:
                return None

    @classmethod
    def _reject(
        cls,
        reason: str,
        t1_val: Optional[Dict[str, Any]],
        t2_val: Optional[Dict[str, Any]],
        t1_datetime: Optional[str] = None,
        t2_datetime: Optional[str] = None,
        temporal_delta_days: Optional[float] = None,
        chronology_status: str = "unknown",
        spatial_overlap_pct: Optional[float] = None,
        warnings: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Builds a structured rejection response."""
        cand, prim, fb = cls._assess_provider_readiness()
        all_warnings = warnings or []
        all_warnings.append(reason)

        return {
            "status": "rejected",
            "is_compatible": False,
            "rejection_reason": reason,
            "t1_validation": t1_val,
            "t2_validation": t2_val,
            "t1_datetime": t1_datetime,
            "t2_datetime": t2_datetime,
            "temporal_delta_days": temporal_delta_days,
            "chronology_status": chronology_status,
            "spatial_overlap_pct": spatial_overlap_pct,
            "overlap_detected": bool(spatial_overlap_pct is not None and spatial_overlap_pct > 0),
            "crs_compatible": False,
            "resolution_compatible": False,
            "modality_compatible": False,
            "co_registered": False,
            "warnings": all_warnings,
            "scientific_limitations": limitations or [reason],
            "candidate_providers": cand,
            "primary_provider": prim,
            "fallback_status": fb,
            "recommended_action": f"Rejection: {reason}. Choose alternative acquisitions before proceeding."
        }
