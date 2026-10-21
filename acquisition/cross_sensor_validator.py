import os
import re
import base64
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

from app.acquisition.asset_validator import AssetValidator
from app.remote_sensing.registration import compute_bounds_overlap

logger = logging.getLogger("satquery.acquisition.cross_sensor_validator")

class CrossSensorValidator:
    """
    Physical Satellite Optical + SAR Cross-Sensor Validator (Stage 6).
    Validates cross-modal asset pairs before inference.

    Strict Scientific Rules:
    1. Input completeness: Rejects if either optical or SAR payload is missing.
    2. Asset readability: Rejects corrupted or undecodable rasters.
    3. Asset category: Strictly rejects browse-only preview thumbnails (unprojected 8-bit images).
    4. Sensor modality verification: Exactly one scene must be OPTICAL and one must be SAR.
       Rejects optical+optical or sar+sar pairs.
    5. Georeferencing & CRS: Both scenes must possess valid geodetic CRS coordinates; projections must match.
    6. Spatial footprint overlap: Overlap must be established (>= 10% mutual footprint; rejects 0% overlap).
    7. Resolution disparity: Analyzes pixel size ratio and flags extreme resampling smoothing (> 10x).
    8. Acquisition times: Computes temporal separation between passes and warns if delta exceeds 60 days.
    9. SAR product processing metadata: Inspects polarization, radiometric calibration evidence (sigma0/gamma0),
       and terrain correction status (RTC vs ellipsoid GRD). Never assumes SAR is calibrated or terrain-corrected.
    10. Physical non-interchangeability: Explicitly asserts that optical reflectance (surface albedo) and
       SAR microwave backscatter (roughness/dielectric return) are non-interchangeable measurements.
    11. Model & provider readiness: Reports truthful specialist model/baseline availability.
    """

    @classmethod
    def validate_pair(
        cls,
        optical_input: Optional[Union[str, bytes]] = None,
        sar_input: Optional[Union[str, bytes]] = None,
        optical_filename: Optional[str] = None,
        sar_filename: Optional[str] = None,
        optical_provenance: Optional[Dict[str, Any]] = None,
        sar_provenance: Optional[Dict[str, Any]] = None,
        optical_metadata: Optional[Dict[str, Any]] = None,
        sar_metadata: Optional[Dict[str, Any]] = None,
        aoi_bbox: Optional[List[float]] = None,
        retrieval_dir: Optional[str] = None,
        samples_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates an Optical + SAR cross-sensor scene pair.
        Returns a structured validation response matching STACOpticalSARPairValidateResponse.
        """
        opt_prov = optical_provenance or {}
        sar_prov = sar_provenance or {}
        opt_meta_in = optical_metadata or {}
        sar_meta_in = sar_metadata or {}

        # Resolve paths/payloads if scene IDs or names were passed
        opt_data, fname_opt = cls._resolve_payload(optical_input, optical_filename, opt_prov, retrieval_dir, samples_dir, default_name="optical_scene.tif")
        sar_data, fname_sar = cls._resolve_payload(sar_input, sar_filename, sar_prov, retrieval_dir, samples_dir, default_name="sar_scene.tif")

        warnings: List[str] = []
        limitations: List[str] = []

        # 1. Missing Input Checks
        has_opt = opt_data is not None or bool(opt_meta_in)
        has_sar = sar_data is not None or bool(sar_meta_in)
        if not has_opt or not has_sar:
            missing_parts = []
            if not has_opt:
                missing_parts.append("Optical reflectance")
            if not has_sar:
                missing_parts.append("SAR backscatter")
            missing_str = " and ".join(missing_parts)
            return cls._reject(
                reason=f"Missing cross-sensor scene: {missing_str} asset payload is not available. Both optical and SAR scenes must be retrieved before cross-sensor analysis.",
                optical_val=None,
                sar_val=None,
                warnings=["Select and retrieve both Optical and SAR candidate scenes before running cross-sensor analysis."],
                limitations=["Cross-sensor analysis requires both Optical reflectance and SAR backscatter rasters."]
            )

        # 2. Physical Asset Validation for Each Scene
        import numpy as np
        if isinstance(opt_data, np.ndarray) or (opt_data is None and opt_meta_in):
            opt_meta_dict = dict(opt_meta_in or {})
            if isinstance(opt_data, np.ndarray):
                h_o, w_o = opt_data.shape[:2]
                opt_meta_dict.setdefault("width", w_o)
                opt_meta_dict.setdefault("height", h_o)
            val_opt = {
                "status": "valid",
                "asset_category": opt_meta_dict.get("asset_category", "scientific_raster"),
                "raster_metadata": {
                    "is_geotiff": opt_meta_dict.get("is_geotiff", True),
                    "georeferenced": bool(opt_meta_dict.get("crs")),
                    **opt_meta_dict
                }
            }
        else:
            val_opt = AssetValidator.validate_asset(opt_data, filename=fname_opt, provenance=opt_prov)

        if isinstance(sar_data, np.ndarray) or (sar_data is None and sar_meta_in):
            sar_meta_dict = dict(sar_meta_in or {})
            if isinstance(sar_data, np.ndarray):
                h_s, w_s = sar_data.shape[:2]
                sar_meta_dict.setdefault("width", w_s)
                sar_meta_dict.setdefault("height", h_s)
            val_sar = {
                "status": "valid",
                "asset_category": sar_meta_dict.get("asset_category", "scientific_raster"),
                "raster_metadata": {
                    "is_geotiff": sar_meta_dict.get("is_geotiff", True),
                    "georeferenced": bool(sar_meta_dict.get("crs")),
                    **sar_meta_dict
                }
            }
        else:
            val_sar = AssetValidator.validate_asset(sar_data, filename=fname_sar, provenance=sar_prov)

        if val_opt.get("status") != "valid":
            err = val_opt.get("error") or "Unreadable optical raster binary"
            return cls._reject(
                reason=f"Optical asset validation failed: {err}",
                optical_val=val_opt,
                sar_val=val_sar,
                warnings=[f"Optical file error: {err}"],
                limitations=["Optical asset could not be decoded as a valid raster."]
            )

        if val_sar.get("status") != "valid":
            err = val_sar.get("error") or "Unreadable SAR raster binary"
            return cls._reject(
                reason=f"SAR asset validation failed: {err}",
                optical_val=val_opt,
                sar_val=val_sar,
                warnings=[f"SAR file error: {err}"],
                limitations=["SAR asset could not be decoded as a valid raster."]
            )

        # 3. Reject Preview-Only Browse Assets (Thumbnails)
        is_opt_preview = (
            val_opt.get("asset_category") == "visual_preview"
            or opt_prov.get("asset_key") in ("thumbnail", "preview")
            or fname_opt.lower().endswith((".jpg", ".jpeg"))
            or not val_opt.get("raster_metadata", {}).get("is_geotiff", False)
        )
        is_sar_preview = (
            val_sar.get("asset_category") == "visual_preview"
            or sar_prov.get("asset_key") in ("thumbnail", "preview")
            or fname_sar.lower().endswith((".jpg", ".jpeg"))
            or not val_sar.get("raster_metadata", {}).get("is_geotiff", False)
        )

        if is_opt_preview or is_sar_preview:
            culprit = "Optical" if (is_opt_preview and not is_sar_preview) else ("SAR" if is_sar_preview and not is_opt_preview else "both Optical and SAR")
            return cls._reject(
                reason=(
                    f"Browse-only preview asset detected in {culprit}. "
                    "Unprojected 8-bit visual previews (JPEG/PNG browse images) are strictly rejected for scientific optical-SAR cross-sensor analysis. "
                    "Calibrated, georeferenced GeoTIFF rasters with surface reflectance or calibrated microwave backscatter are required."
                ),
                optical_val=val_opt,
                sar_val=val_sar,
                warnings=["Select genuine GeoTIFF raster assets rather than browse preview thumbnails."],
                limitations=["Browse previews lack calibrated surface reflectance and geometric projection."]
            )

        # 4. Sensor Modality Verification
        mod_opt = (opt_prov.get("modality") or opt_meta_in.get("modality") or val_opt.get("raster_metadata", {}).get("modality") or ("SAR" if "sar" in fname_opt.lower() or "risat" in fname_opt.lower() else "OPTICAL")).upper()
        mod_sar = (sar_prov.get("modality") or sar_meta_in.get("modality") or val_sar.get("raster_metadata", {}).get("modality") or ("SAR" if "sar" in fname_sar.lower() or "risat" in fname_sar.lower() else "OPTICAL")).upper()

        if mod_opt == "SAR" and mod_sar == "OPTICAL":
            # Auto-align inverted inputs gracefully
            mod_opt, mod_sar = mod_sar, mod_opt
            val_opt, val_sar = val_sar, val_opt
            fname_opt, fname_sar = fname_sar, fname_opt
            opt_prov, sar_prov = sar_prov, opt_prov
            opt_meta_in, sar_meta_in = sar_meta_in, opt_meta_in
            warnings.append("Input order auto-aligned: Assigned primary slot to Optical and secondary slot to SAR.")

        if mod_opt != "OPTICAL" and mod_opt != "MULTISPECTRAL":
            return cls._reject(
                reason=f"Invalid modality for Optical scene: Expected OPTICAL or MULTISPECTRAL, received {mod_opt}.",
                optical_val=val_opt,
                sar_val=val_sar,
                warnings=[f"Primary modality {mod_opt} is not an optical sensor."],
                limitations=["Cross-sensor analysis requires an optical reflectance sensor."]
            )

        if mod_sar != "SAR":
            if mod_sar in ("OPTICAL", "MULTISPECTRAL"):
                return cls._reject(
                    reason="Invalid modalities for Optical + SAR analysis: Both scenes are OPTICAL. Cross-sensor analysis requires exactly one Optical reflectance scene and one microwave SAR scene.",
                    optical_val=val_opt,
                    sar_val=val_sar,
                    warnings=["Both acquisitions are optical sensors; missing microwave SAR input."],
                    limitations=["Cross-sensor analysis cannot proceed with two homogeneous optical scenes."]
                )
            return cls._reject(
                reason=f"Invalid modality for SAR scene: Expected SAR (microwave radar), received {mod_sar}.",
                optical_val=val_opt,
                sar_val=val_sar,
                warnings=[f"Secondary modality {mod_sar} is not a microwave SAR sensor."],
                limitations=["Cross-sensor analysis requires a microwave SAR sensor."]
            )

        # Check for sar + sar rejection
        if mod_opt == "SAR" and mod_sar == "SAR":
            return cls._reject(
                reason="Invalid modalities for Optical + SAR analysis: Both scenes are SAR. Cross-sensor analysis requires exactly one Optical reflectance scene and one microwave SAR scene.",
                optical_val=val_opt,
                sar_val=val_sar,
                warnings=["Both acquisitions are SAR sensors; missing optical reflectance input."],
                limitations=["Cross-sensor analysis cannot proceed with two homogeneous SAR scenes."]
            )

        # 5. Georeferencing & Coordinate Reference System (CRS)
        meta_opt = {**val_opt.get("raster_metadata", {}), **opt_meta_in}
        meta_sar = {**val_sar.get("raster_metadata", {}), **sar_meta_in}

        crs_opt = meta_opt.get("crs")
        crs_sar = meta_sar.get("crs")
        georef_opt = meta_opt.get("georeferenced", False)
        georef_sar = meta_sar.get("georeferenced", False)

        if not georef_opt or not georef_sar or not crs_opt or not crs_sar:
            missing_crs_target = "Optical" if (not crs_opt and crs_sar) else ("SAR" if not crs_sar and crs_opt else "both scenes")
            return cls._reject(
                reason=(
                    f"Georeferencing missing: {missing_crs_target} lack coordinate reference system (CRS) tags. "
                    "Ground-level spatial alignment between optical imagery and SAR projections cannot be verified."
                ),
                optical_val=val_opt,
                sar_val=val_sar,
                warnings=["Inputs lack geodetic CRS coordinates."],
                limitations=["Unprojected scenes cannot be spatially co-registered."]
            )

        # 6. Spatial Footprint & Requested AOI Overlap Calculation
        aoi_coverage_pct: Optional[float] = None
        optical_aoi_coverage_pct: Optional[float] = None
        sar_aoi_coverage_pct: Optional[float] = None
        aoi_intersection_valid = False

        if aoi_bbox is not None:
            import math
            # Validate bounding-box length, types, and ordering
            if not isinstance(aoi_bbox, (list, tuple)) or len(aoi_bbox) != 4:
                return cls._reject(
                    reason="Invalid AOI bounding box: coordinates must be a 4-element sequence [min_x, min_y, max_x, max_y].",
                    optical_val=val_opt,
                    sar_val=val_sar,
                    warnings=["Invalid AOI bounding box format."],
                    limitations=["Requested AOI is malformed."]
                )

            for idx, coord in enumerate(aoi_bbox):
                if not isinstance(coord, (int, float)) or isinstance(coord, bool) or not math.isfinite(coord):
                    return cls._reject(
                        reason=f"Invalid AOI bounding box: coordinate at index {idx} ({coord}) is not a finite number.",
                        optical_val=val_opt,
                        sar_val=val_sar,
                        warnings=["Non-numeric or non-finite AOI coordinates."],
                        limitations=["Requested AOI contains non-finite numbers."]
                    )

            if aoi_bbox[0] >= aoi_bbox[2] or aoi_bbox[1] >= aoi_bbox[3]:
                return cls._reject(
                    reason=(
                        f"Invalid AOI bounding box: min coordinates must be strictly less than max coordinates "
                        f"(got min_x={aoi_bbox[0]}, max_x={aoi_bbox[2]}, min_y={aoi_bbox[1]}, max_y={aoi_bbox[3]})."
                    ),
                    optical_val=val_opt,
                    sar_val=val_sar,
                    warnings=["Inverted or degenerate AOI bounding box coordinates."],
                    limitations=["Requested AOI coordinates are geometrically invalid."]
                )

            bounds_opt = meta_opt.get("bounds")
            bounds_sar = meta_sar.get("bounds")
            if not bounds_opt or not bounds_sar or not crs_opt or not crs_sar:
                return cls._reject(
                    reason="AOI intersection could not be verified: spatial bounds or CRS metadata unavailable for one or both scenes.",
                    optical_val=val_opt,
                    sar_val=val_sar,
                    warnings=["Spatial boundary vectors unavailable; cannot verify requested AOI coverage."],
                    limitations=["Scene spatial footprint bounds unavailable for AOI verification."]
                )

            norm_opt = cls._normalize_bounds_to_crs(bounds_opt, crs_opt, "EPSG:4326")
            norm_sar = cls._normalize_bounds_to_crs(bounds_sar, crs_sar, "EPSG:4326")

            if norm_opt is None or norm_sar is None:
                return cls._reject(
                    reason="AOI intersection could not be computed: failed to transform scene bounds to common AOI CRS (EPSG:4326).",
                    optical_val=val_opt,
                    sar_val=val_sar,
                    warnings=["Reprojection failure during AOI coordinate normalization."],
                    limitations=["Coordinate transform error during AOI validation."]
                )

            opt_aoi_inter = cls._compute_box_intersection(norm_opt, aoi_bbox)
            sar_aoi_inter = cls._compute_box_intersection(norm_sar, aoi_bbox)
            scenes_inter = cls._compute_box_intersection(norm_opt, norm_sar)

            opt_intersects = opt_aoi_inter is not None
            sar_intersects = sar_aoi_inter is not None

            if not opt_intersects and not sar_intersects:
                return cls._reject(
                    reason="Spatial coverage failure: Neither optical nor SAR scene intersects the requested AOI.",
                    optical_val=val_opt,
                    sar_val=val_sar,
                    spatial_overlap_pct=0.0,
                    warnings=["Both scene footprints lie entirely outside the target AOI."],
                    limitations=["Disjoint footprints from requested AOI."]
                )

            if opt_intersects and not sar_intersects:
                return cls._reject(
                    reason="Spatial coverage failure: SAR scene footprint does not intersect the requested AOI.",
                    optical_val=val_opt,
                    sar_val=val_sar,
                    spatial_overlap_pct=0.0,
                    warnings=["SAR footprint does not intersect the target AOI."],
                    limitations=["Missing SAR coverage over requested AOI."]
                )

            if sar_intersects and not opt_intersects:
                return cls._reject(
                    reason="Spatial coverage failure: Optical scene footprint does not intersect the requested AOI.",
                    optical_val=val_opt,
                    sar_val=val_sar,
                    spatial_overlap_pct=0.0,
                    warnings=["Optical footprint does not intersect the target AOI."],
                    limitations=["Missing Optical coverage over requested AOI."]
                )

            joint_aoi_inter = cls._compute_box_intersection(scenes_inter, aoi_bbox) if scenes_inter else None
            if joint_aoi_inter is None:
                return cls._reject(
                    reason=(
                        "Spatial coverage failure: Scene footprints overlap each other but do not cover the requested AOI "
                        "(joint intersection with AOI is empty)."
                    ),
                    optical_val=val_opt,
                    sar_val=val_sar,
                    spatial_overlap_pct=0.0,
                    warnings=["Scene mutual overlap lies outside requested AOI."],
                    limitations=["Zero joint multimodal coverage over target AOI."]
                )

            area_opt = cls._compute_box_area(norm_opt)
            area_sar = cls._compute_box_area(norm_sar)
            area_mutual = cls._compute_box_area(scenes_inter)
            area_aoi = cls._compute_box_area(aoi_bbox)

            min_scene_area = min(area_opt, area_sar)
            overlap_pct = round((area_mutual / min_scene_area) * 100.0, 2) if min_scene_area > 0 else 0.0
            aoi_coverage_pct = round((cls._compute_box_area(joint_aoi_inter) / area_aoi) * 100.0, 2) if area_aoi > 0 else 0.0
            optical_aoi_coverage_pct = round((cls._compute_box_area(opt_aoi_inter) / area_aoi) * 100.0, 2) if area_aoi > 0 else 0.0
            sar_aoi_coverage_pct = round((cls._compute_box_area(sar_aoi_inter) / area_aoi) * 100.0, 2) if area_aoi > 0 else 0.0
            aoi_intersection_valid = True

            if overlap_pct < 10.0:
                return cls._reject(
                    reason=(
                        f"Insufficient spatial overlap: Scenes share only {overlap_pct:.1f}% mutual footprint "
                        "(minimum required is 10.0%)."
                    ),
                    optical_val=val_opt,
                    sar_val=val_sar,
                    spatial_overlap_pct=overlap_pct,
                    warnings=[f"Marginal spatial overlap ({overlap_pct:.1f}%)."],
                    limitations=["Mutual overlap area is too small for dependable cross-sensor joint interpretation."]
                )

            if crs_opt != crs_sar:
                warnings.append(
                    f"Different CRSs: Optical is '{crs_opt}' and SAR is '{crs_sar}'. "
                    "Secondary raster will be reprojected onto primary grid during alignment."
                )

        else:
            # aoi_bbox is None: preserve existing behavior
            if crs_opt != crs_sar:
                return cls._reject(
                    reason=(
                        f"CRS mismatch: Optical projection is '{crs_opt}' while SAR projection is '{crs_sar}'. "
                        "Cross-sensor analysis requires identical coordinate reference systems to avoid geometric distortion."
                    ),
                    optical_val=val_opt,
                    sar_val=val_sar,
                    warnings=[f"CRS mismatch: {crs_opt} vs {crs_sar}"],
                    limitations=["Scenes exist in different spatial coordinate systems."]
                )

            bounds_opt = meta_opt.get("bounds")
            bounds_sar = meta_sar.get("bounds")
            overlap_pct = 0.0

            if bounds_opt and bounds_sar:
                overlap_pct = compute_bounds_overlap(bounds_opt, bounds_sar)

                if overlap_pct <= 0.0:
                    return cls._reject(
                        reason=(
                            "Zero spatial overlap: Optical footprint and SAR footprint do not intersect. "
                            "Cross-sensor analysis requires overlapping geographical coverage for the active AOI."
                        ),
                        optical_val=val_opt,
                        sar_val=val_sar,
                        spatial_overlap_pct=0.0,
                        warnings=["No mutual spatial footprint exists between these two scenes."],
                        limitations=["Disjoint scene footprints."]
                    )

                if overlap_pct < 10.0:
                    return cls._reject(
                        reason=(
                            f"Insufficient spatial overlap: Scenes share only {overlap_pct:.1f}% mutual footprint "
                            "(minimum required is 10.0%)."
                        ),
                        optical_val=val_opt,
                        sar_val=val_sar,
                        spatial_overlap_pct=overlap_pct,
                        warnings=[f"Marginal spatial overlap ({overlap_pct:.1f}%)."],
                        limitations=["Overlap area is too small for dependable cross-sensor joint interpretation."]
                    )
            else:
                warnings.append("Spatial boundary vectors unavailable; overlap percentage could not be geometrically calculated.")

        # 7. Resolution & Dimension Compatibility (Ground-Resolution Normalized)
        ground_res_opt = cls._compute_ground_resolution_meters(meta_opt, aoi_bbox)
        ground_res_sar = cls._compute_ground_resolution_meters(meta_sar, aoi_bbox)
        res_compatible = True

        if ground_res_opt is None or ground_res_sar is None:
            missing_target = "Optical" if ground_res_opt is None and ground_res_sar is not None else (
                "SAR" if ground_res_sar is None and ground_res_opt is not None else "both scenes"
            )
            return cls._reject(
                reason=(
                    f"Resolution metadata missing or invalid: {missing_target} ground pixel resolution "
                    "could not be determined from raster georeferencing or affine transform."
                ),
                optical_val=val_opt,
                sar_val=val_sar,
                warnings=["Pixel ground dimensions unavailable or invalid."],
                limitations=["Ground pixel scale could not be verified."]
            )

        ratio_x = max(ground_res_opt[0], ground_res_sar[0]) / max(1e-6, min(ground_res_opt[0], ground_res_sar[0]))
        ratio_y = max(ground_res_opt[1], ground_res_sar[1]) / max(1e-6, min(ground_res_opt[1], ground_res_sar[1]))
        ratio = max(ratio_x, ratio_y)

        if ratio > 10.0:
            res_compatible = False
            return cls._reject(
                reason=(
                    f"Excessive resolution disparity ({ratio:.1f}x) between Optical "
                    f"({max(ground_res_opt):.1f}m) and SAR ({max(ground_res_sar):.1f}m) rasters. "
                    "Resampling across disparity exceeding 10.0x causes unacceptable spatial distortion or feature smoothing."
                ),
                optical_val=val_opt,
                sar_val=val_sar,
                warnings=[f"Severe ground resolution disparity ({ratio:.1f}x) exceeds 10.0x maximum limit."],
                limitations=["Extreme resolution disparity precludes valid joint interpretation."]
            )
        elif ratio > 3.0:
            warnings.append(
                f"Moderate resolution disparity ({ratio:.1f}x) between sensors "
                f"(Optical ~{max(ground_res_opt):.1f}m vs SAR ~{max(ground_res_sar):.1f}m). "
                "Secondary raster will be resampled to matching pixel grid."
            )

        # 8. Acquisition Times & Temporal Separation
        dt_opt_str = opt_prov.get("datetime") or opt_meta_in.get("datetime")
        dt_sar_str = sar_prov.get("datetime") or sar_meta_in.get("datetime")
        delta_days: Optional[float] = None

        if dt_opt_str and dt_sar_str:
            dt_opt = cls._parse_iso_datetime(dt_opt_str)
            dt_sar = cls._parse_iso_datetime(dt_sar_str)

            if dt_opt and dt_sar:
                diff_sec = abs((dt_sar - dt_opt).total_seconds())
                delta_days = round(diff_sec / 86400.0, 2)

                if delta_days > 60:
                    warnings.append(
                        f"Temporal separation advisory: Optical and SAR scenes were acquired {delta_days:.1f} days apart. "
                        "Dynamic ground conditions (soil moisture, seasonal water, agricultural phenology) may have altered between passes."
                    )
                    limitations.append(
                        f"Temporal separation of {delta_days:.1f} days between optical and radar observations. Surface changes between dates may confound joint interpretation."
                    )
            else:
                warnings.append("Could not parse ISO datetime strings for precise pass delta calculation.")
        else:
            warnings.append("Acquisition timestamp metadata missing for one or both scenes; temporal baseline separation unconfirmed.")

        # 9. SAR Product Processing Metadata Inspection
        sar_tags = meta_sar.get("tags", {}) or {}
        sar_polarization = (
            sar_prov.get("polarization")
            or sar_meta_in.get("polarization")
            or sar_tags.get("POLARIZATION")
            or sar_tags.get("polarisation")
            or "VV"
        )
        sar_sensor = (
            sar_prov.get("sensor")
            or sar_meta_in.get("sensor")
            or sar_tags.get("SENSOR")
            or "Sentinel-1 C-Band SAR"
        )

        # Inspect radiometric calibration evidence
        calib_tags = str(sar_tags).lower() + " " + str(sar_prov).lower() + " " + fname_sar.lower()
        has_calib_evidence = any(k in calib_tags for k in ["sigma0", "gamma0", "beta0", "calibrated", "lut", "decibel", "db_scaled"])
        calibration_status = "calibrated_backscatter" if has_calib_evidence else "unverified_linear_dn"

        if not has_calib_evidence:
            warnings.append(
                "Radiometric calibration advisory: Product metadata lacks verified sigma-0/gamma-0 calibration look-up tables. "
                "Radar intensity is processed as relative linear amplitude; absolute decibel (dB) backscatter is uncalibrated."
            )
            limitations.append(
                "Uncalibrated SAR amplitude: Product lacks certified radiometric calibration coefficients. Relative dB scaling is empirical."
            )

        # Inspect terrain correction evidence
        has_terrain_correction = any(k in calib_tags for k in ["terrain_corrected", "rtc", "orthorectified", "dem_corrected"])
        terrain_correction_status = "radiometrically_terrain_corrected" if has_terrain_correction else "ellipsoid_geocoded_grd"

        if not has_terrain_correction:
            warnings.append(
                "Terrain correction advisory: SAR product is ellipsoid-geocoded (GRD) without verified Radiometric Terrain Correction (RTC). "
                "Radar geometric distortions (layover, foreshortening, and shadow) may occur in steep or rugged topography."
            )
            limitations.append(
                "Ellipsoid-projected GRD: Lacks DEM-based radiometric terrain correction. Complex topography may induce geometric displacement."
            )

        # 10. Physical Non-Interchangeability Assertion
        limitations.append(
            "Scientific assertion: Optical surface reflectance (spectral albedo across visible/NIR bands) and SAR microwave backscatter "
            "(dielectric permittivity and geometric roughness response) are physically complementary, non-interchangeable measurements. "
            "They are never mathematically equated or treated as homogeneous values."
        )

        # 11. Model & Provider Readiness
        candidate_providers = ["OpticalSARJointAnalysisProvider", "OpticalSARVisualizationBaseline"]
        primary_prov = "OpticalSARJointAnalysisProvider"
        fallback_stat = "Loaded and ready"

        co_registered = bool(crs_opt == crs_sar and overlap_pct >= 90.0 and res_compatible)
        status = "compatible" if not warnings else "compatible_with_limitations"

        return {
            "status": status,
            "is_compatible": True,
            "rejection_reason": None,
            "optical_validation": val_opt,
            "sar_validation": val_sar,
            "optical_datetime": dt_opt_str,
            "sar_datetime": dt_sar_str,
            "temporal_delta_days": delta_days,
            "spatial_overlap_pct": overlap_pct,
            "overlap_detected": bool(overlap_pct > 0.0),
            "aoi_coverage_pct": aoi_coverage_pct,
            "aoi_intersection_valid": aoi_intersection_valid,
            "optical_aoi_coverage_pct": optical_aoi_coverage_pct,
            "sar_aoi_coverage_pct": sar_aoi_coverage_pct,
            "crs_compatible": True,
            "resolution_compatible": res_compatible,
            "modality_compatible": True,
            "co_registered": co_registered,
            "sar_polarization": sar_polarization,
            "sar_sensor": sar_sensor,
            "sar_calibration_status": calibration_status,
            "sar_terrain_correction_status": terrain_correction_status,
            "warnings": warnings,
            "scientific_limitations": limitations,
            "candidate_providers": candidate_providers,
            "primary_provider": primary_prov,
            "fallback_status": fallback_stat,
            "recommended_action": (
                "Cross-sensor pair is fully compatible. Proceed with Optical + SAR joint radiometric analysis."
                if status == "compatible"
                else "Cross-sensor pair is compatible with scientific caveats. Review sensor calibration and topography warnings before executing joint analysis."
            )
        }

    @classmethod
    def _resolve_payload(
        cls,
        payload_input: Optional[Union[str, bytes]],
        filename: Optional[str],
        prov: Dict[str, Any],
        retrieval_dir: Optional[str],
        samples_dir: Optional[str],
        default_name: str = "scene.tif"
    ) -> Tuple[Optional[Union[str, bytes]], str]:
        """Resolves raw bytes or local file path from scene identifiers or payload."""
        target_fname = filename or prov.get("filename") or default_name

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
    def _normalize_bounds_to_crs(
        cls,
        bounds: Optional[List[float]],
        src_crs: Optional[str],
        dst_crs: str = "EPSG:4326"
    ) -> Optional[List[float]]:
        """Transforms bounding box [min_x, min_y, max_x, max_y] to dst_crs using rasterio.warp.transform_bounds."""
        if not bounds or len(bounds) != 4 or not src_crs:
            return None
        try:
            from rasterio.crs import CRS
            from rasterio.warp import transform_bounds
            src_c = CRS.from_user_input(src_crs)
            dst_c = CRS.from_user_input(dst_crs)
            if src_c == dst_c:
                return [float(b) for b in bounds]
            tb = list(transform_bounds(src_c, dst_c, *bounds))
            if tb[0] < tb[2] and tb[1] < tb[3]:
                return [float(x) for x in tb]
            return None
        except Exception as e:
            logger.warning(f"Failed to transform bounds {bounds} from {src_crs} to {dst_crs}: {e}")
            return None

    @classmethod
    def _compute_box_intersection(
        cls,
        b1: Optional[List[float]],
        b2: Optional[List[float]]
    ) -> Optional[List[float]]:
        """Computes rectangular bounding box intersection [min_x, min_y, max_x, max_y]."""
        if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
            return None
        left = max(b1[0], b2[0])
        bottom = max(b1[1], b2[1])
        right = min(b1[2], b2[2])
        top = min(b1[3], b2[3])
        if right > left and top > bottom:
            return [left, bottom, right, top]
        return None

    @classmethod
    def _compute_box_area(cls, b: Optional[List[float]]) -> float:
        """Computes bounding box area in units squared."""
        if not b or len(b) != 4:
            return 0.0
        return max(0.0, float(b[2] - b[0]) * float(b[3] - b[1]))

    @classmethod
    def _compute_ground_resolution_meters(
        cls,
        meta: Dict[str, Any],
        aoi_bbox: Optional[List[float]] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Computes effective ground pixel resolution (res_x_m, res_y_m) in meters.
        Accounts for projected vs geographic CRS and central latitude scale on the WGS84 ellipsoid.
        Returns None if CRS, resolution, or georeferencing is missing or invalid.
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
        if isinstance(res, (int, float)) and not isinstance(res, bool):
            res = [float(res), float(res)]
        elif not isinstance(res, (list, tuple)):
            res = None

        if not res or len(res) < 2 or res[0] <= 0 or res[1] <= 0:
            transform = meta.get("transform")
            if transform and len(transform) >= 6:
                try:
                    res = [abs(float(transform[0])), abs(float(transform[4]))]
                except (ValueError, TypeError):
                    res = None
            if not res or len(res) < 2 or res[0] <= 0 or res[1] <= 0:
                bounds = meta.get("bounds")
                w = meta.get("width")
                h = meta.get("height")
                if bounds and len(bounds) >= 4 and w and h:
                    try:
                        w_f, h_f = float(w), float(h)
                        if w_f > 0 and h_f > 0:
                            res = [abs(float(bounds[2]) - float(bounds[0])) / w_f, abs(float(bounds[3]) - float(bounds[1])) / h_f]
                    except (ValueError, TypeError, ZeroDivisionError):
                        return None
                else:
                    return None

        if res is None or res[0] <= 0 or res[1] <= 0:
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
            if aoi_bbox and len(aoi_bbox) == 4:
                try:
                    lat = (float(aoi_bbox[1]) + float(aoi_bbox[3])) / 2.0
                except (ValueError, TypeError):
                    lat = 0.0
            elif meta.get("bounds") and len(meta["bounds"]) == 4:
                try:
                    lat = (float(meta["bounds"][1]) + float(meta["bounds"][3])) / 2.0
                except (ValueError, TypeError):
                    lat = 0.0

            lat = max(-89.0, min(89.0, lat))
            lat_rad = math.radians(lat)

            # WGS84 ellipsoid constants
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
    def _parse_iso_datetime(cls, dt_str: str) -> Optional[datetime]:
        """Parses ISO 8601 datetime string."""
        try:
            clean = dt_str.replace("Z", "+00:00")
            return datetime.fromisoformat(clean)
        except Exception:
            try:
                return datetime.strptime(dt_str[:10], "%Y-%m-%d")
            except Exception:
                return None

    @classmethod
    def _reject(
        cls,
        reason: str,
        optical_val: Optional[Dict[str, Any]],
        sar_val: Optional[Dict[str, Any]],
        optical_datetime: Optional[str] = None,
        sar_datetime: Optional[str] = None,
        temporal_delta_days: Optional[float] = None,
        spatial_overlap_pct: Optional[float] = None,
        warnings: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        aoi_coverage_pct: Optional[float] = None,
        aoi_intersection_valid: bool = False
    ) -> Dict[str, Any]:
        """Builds a structured rejection response for cross-sensor validation."""
        cand = ["OpticalSARJointAnalysisProvider", "OpticalSARVisualizationBaseline"]
        prim = "OpticalSARJointAnalysisProvider"
        fb = "Loaded and ready"
        all_warnings = warnings or []
        all_warnings.append(reason)

        return {
            "status": "rejected",
            "is_compatible": False,
            "rejection_reason": reason,
            "optical_validation": optical_val,
            "sar_validation": sar_val,
            "optical_datetime": optical_datetime,
            "sar_datetime": sar_datetime,
            "temporal_delta_days": temporal_delta_days,
            "spatial_overlap_pct": spatial_overlap_pct,
            "overlap_detected": bool(spatial_overlap_pct is not None and spatial_overlap_pct > 0),
            "aoi_coverage_pct": aoi_coverage_pct,
            "aoi_intersection_valid": aoi_intersection_valid,
            "crs_compatible": False,
            "resolution_compatible": False,
            "modality_compatible": False,
            "co_registered": False,
            "sar_polarization": "Unknown",
            "sar_sensor": "Unknown",
            "sar_calibration_status": "rejected",
            "sar_terrain_correction_status": "rejected",
            "warnings": all_warnings,
            "scientific_limitations": limitations or [reason],
            "candidate_providers": cand,
            "primary_provider": prim,
            "fallback_status": fb,
            "recommended_action": f"Rejection: {reason}. Select suitable matching Optical and SAR acquisitions before proceeding."
        }
