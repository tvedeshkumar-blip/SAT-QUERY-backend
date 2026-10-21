"""
Stage 8: Scientific Analysis Quality and Evidence Reporting.
Builds structured, reproducible, schema-validated analysis reports
strictly preserving physical measurements, unit scales, verification outcomes,
and honest representation of unavailable provenance.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from app.schemas.report import (
    PhysicalMeasurementReportItem,
    ReportEvidenceArtifactItem,
    ReportProvenanceSchema,
    ReportReproducibilitySchema,
    ReportVerificationSummarySchema,
    ScientificReportSchema,
)
from app.schemas.verification import (
    EvidencePackageSchema,
    PhysicalMeasurementItem,
    DisplayStatisticItem,
    HeuristicIndicatorItem,
    VerificationResponseSchema,
)
# EvidencePackageBuilder imported locally in build_report if needed

logger = logging.getLogger("satquery.reports.scientific")


class ScientificReportBuilder:
    """
    Constructs an audit-grade, schema-validated ScientificReportSchema from
    an analysis response and optional evidence package.
    """

    @staticmethod
    def build_report(
        analysis_response: Union[Dict[str, Any], Any],
        evidence_package: Optional[EvidencePackageSchema] = None,
    ) -> ScientificReportSchema:
        """
        Builds a reproducible ScientificReportSchema.
        Deterministic for identical input data.
        """
        # Convert response to dictionary safely
        if hasattr(analysis_response, "model_dump"):
            resp_data = analysis_response.model_dump()
        elif hasattr(analysis_response, "dict"):
            resp_data = analysis_response.dict()
        elif isinstance(analysis_response, dict):
            resp_data = analysis_response
        else:
            resp_data = getattr(analysis_response, "__dict__", {})

        analysis_id = str(resp_data.get("id") or "analysis_unknown")
        task = str(resp_data.get("task") or "general")
        query = str(resp_data.get("query") or (resp_data.get("answer", "")[:120]))
        mode = str(resp_data.get("mode") or "auto")
        created_at = str(resp_data.get("created_at") or (datetime.utcnow().isoformat() + "Z"))
        answer_text = str(resp_data.get("answer") or "No analysis answer text recorded.")
        metadata = resp_data.get("metadata") or {}

        # 1. Build or retrieve EvidencePackageSchema
        if evidence_package is None:
            try:
                from app.agent.verifier import EvidencePackageBuilder
                evidence_package = EvidencePackageBuilder.from_analysis_response(resp_data, raw_metadata=metadata)
            except Exception as e:
                logger.warning(f"Could not build evidence package from response: {e}")
                evidence_package = None

        # Track explicitly unavailable or missing fields
        unavailable_fields: List[str] = []

        # 2. Extract Provenance
        sensor_identities = []
        source_asset_ids = []
        acquisition_timestamps = []
        crs = None
        spatial_resolution = None
        dimensions = None
        registration_status: Dict[str, Any] = {}
        valid_pixel_count = None
        total_pixel_count = None
        valid_pixel_percentage = None

        if evidence_package:
            sensor_identities = list(evidence_package.sensor_identities)
            source_asset_ids = [str(a.get("id", "asset")) for a in evidence_package.source_assets]
            acquisition_timestamps = list(evidence_package.acquisition_timestamps)
            crs = evidence_package.crs
            spatial_resolution = evidence_package.resolution
            dimensions = evidence_package.dimensions
            registration_status = dict(evidence_package.registration_status)
            valid_pixel_count = evidence_package.valid_pixel_count
            total_pixel_count = evidence_package.total_pixel_count
            valid_pixel_percentage = evidence_package.valid_pixel_percentage
        else:
            if metadata.get("sensor"):
                sensor_identities.append(str(metadata["sensor"]))
            if metadata.get("id"):
                source_asset_ids.append(str(metadata["id"]))
            if metadata.get("datetime"):
                acquisition_timestamps.append(str(metadata["datetime"]))
            crs = metadata.get("crs")
            spatial_resolution = metadata.get("resolution")
            dimensions = f"{metadata.get('width', '')}x{metadata.get('height', '')}" if metadata.get("width") else None

        # Check for unavailable fields honestly
        if not sensor_identities:
            unavailable_fields.append("sensor_identities: not provided in source metadata")
        if not source_asset_ids:
            unavailable_fields.append("source_asset_ids: unverified catalog assets")
        if not acquisition_timestamps:
            unavailable_fields.append("acquisition_timestamps: unprovided in source metadata")
        if not crs:
            unavailable_fields.append("crs: missing or unprojected local image coordinates")
        if spatial_resolution is None:
            unavailable_fields.append("spatial_resolution_m: uncalibrated ground sample distance")
        if dimensions is None or dimensions == "x":
            unavailable_fields.append("dimensions: raster dimensions not recorded")
        if not registration_status:
            unavailable_fields.append("registration_status: multi-sensor co-registration unperformed or unrecorded")

        # Quality screening extraction
        quality_screening = None
        if evidence_package and getattr(evidence_package, "quality_screening", None):
            quality_screening = evidence_package.quality_screening
        elif metadata.get("quality_screening"):
            quality_screening = metadata["quality_screening"]
        elif metadata.get("optical_sar_joint", {}).get("quality_screening"):
            quality_screening = metadata["optical_sar_joint"]["quality_screening"]

        observed_pixel_count = None
        excluded_quality_pixel_count = None
        quality_clearance_percentage = None

        if quality_screening and isinstance(quality_screening, dict):
            opt_q = quality_screening.get("optical")
            if isinstance(opt_q, dict):
                observed_pixel_count = opt_q.get("observed_pixel_count")
                excluded_quality_pixel_count = opt_q.get("excluded_pixel_count")
                clr_frac = opt_q.get("quality_clearance_fraction")
                if clr_frac is not None:
                    quality_clearance_percentage = round(float(clr_frac) * 100.0, 2)

        # Provider info
        provider_name = str(resp_data.get("primary_model") or resp_data.get("actual_model_used") or "SatQueryAgent")
        actual_model_used = resp_data.get("actual_model_used")
        fallback_used = bool(resp_data.get("fallback_used", False))
        model_status = resp_data.get("model_status")
        is_trained_model = False

        if evidence_package and evidence_package.provider_info:
            prov_info = evidence_package.provider_info
            provider_name = prov_info.provider_name or provider_name
            actual_model_used = prov_info.actual_model_used or actual_model_used
            fallback_used = prov_info.fallback_used
            is_trained_model = prov_info.is_trained_model
            model_status = prov_info.model_status or model_status

        # Check for SAR geometric layover/shadow mask availability
        has_sar = any("sar" in str(s).lower() or "sentinel-1" in str(s).lower() for s in sensor_identities)
        if not has_sar and evidence_package:
            has_sar = any("sar" in pm.name.lower() for pm in evidence_package.physical_measurements)
        if has_sar:
            unavailable_fields.append("sar_terrain_layover_shadow_mask: unavailable from STAC provider (no geometric layover/shadow mask asset)")

        has_optical = any("optical" in str(s).lower() or "sentinel-2" in str(s).lower() for s in sensor_identities)
        if not has_optical and evidence_package:
            has_optical = any("optical" in pm.name.lower() for pm in evidence_package.physical_measurements)
        opt_screen_done = bool(quality_screening and isinstance(quality_screening.get("optical"), dict) and quality_screening["optical"].get("screened"))
        if has_optical and not opt_screen_done:
            unavailable_fields.append("optical_radiometric_quality_screening: SCL asset not acquired or screening unperformed")

        # Extract calibrated Sentinel-2 B04+B08 NDVI provenance
        ndvi_res = metadata.get("ndvi_result") or metadata.get("ndvi") or (metadata.get("optical_sar_joint", {}).get("ndvi") if isinstance(metadata.get("optical_sar_joint"), dict) else None)
        ndvi_mean = None
        ndvi_p90 = None
        veg_pct = None
        veg_thresh = None
        scale_fac = None
        scl_policy = None
        b04_id = None
        b08_id = None
        scl_id = None
        ndvi_status = None

        if isinstance(ndvi_res, dict):
            ndvi_mean = ndvi_res.get("ndvi_mean")
            ndvi_p90 = ndvi_res.get("ndvi_p90")
            veg_pct = ndvi_res.get("vegetation_coverage_pct")
            veg_thresh = ndvi_res.get("vegetation_threshold", 0.40)
            scale_fac = ndvi_res.get("calibration", {}).get("scale_factor", 10000.0)
            scl_policy = ndvi_res.get("validity_policy", {}).get("scl_mask_applied", "SCL screening applied: classes 0, 1, 3, 8, 9, 10 excluded")
            b04_id = ndvi_res.get("provenance", {}).get("b04_asset_id") or "B04"
            b08_id = ndvi_res.get("provenance", {}).get("b08_asset_id") or "B08"
            scl_id = ndvi_res.get("provenance", {}).get("scl_asset_id") or "SCL"
            ndvi_status = ndvi_res.get("status")
        elif evidence_package:
            for pm in evidence_package.physical_measurements:
                if pm.name == "optical_mean_ndvi" and pm.value is not None:
                    try:
                        ndvi_mean = float(pm.value)
                    except (ValueError, TypeError):
                        pass
                if pm.name == "optical_p90_ndvi" and pm.value is not None:
                    try:
                        ndvi_p90 = float(pm.value)
                    except (ValueError, TypeError):
                        pass
            for hi in evidence_package.heuristic_indicators:
                if hi.name == "calibrated_vegetation_coverage":
                    veg_pct = hi.percentage
                    veg_thresh = 0.40

        provenance = ReportProvenanceSchema(
            analysis_id=analysis_id,
            task=task,
            query=query,
            mode=mode,
            created_at=created_at,
            sensor_identities=sensor_identities,
            source_asset_ids=source_asset_ids,
            acquisition_timestamps=acquisition_timestamps,
            crs=crs,
            spatial_resolution_m=spatial_resolution,
            dimensions=dimensions,
            registration_status=registration_status,
            valid_pixel_count=valid_pixel_count,
            total_pixel_count=total_pixel_count,
            valid_pixel_percentage=valid_pixel_percentage,
            quality_screening=quality_screening,
            observed_pixel_count=observed_pixel_count,
            excluded_quality_pixel_count=excluded_quality_pixel_count,
            quality_clearance_percentage=quality_clearance_percentage,
            provider_name=provider_name,
            actual_model_used=actual_model_used,
            fallback_used=fallback_used,
            is_trained_model=is_trained_model,
            model_status=model_status,
            ndvi_mean=ndvi_mean,
            ndvi_p90=ndvi_p90,
            vegetation_coverage_pct=veg_pct,
            ndvi_vegetation_threshold=veg_thresh,
            reflectance_scale_factor=scale_fac,
            scl_masking_policy=str(scl_policy) if scl_policy else None,
            b04_asset_id=b04_id,
            b08_asset_id=b08_id,
            scl_asset_id=scl_id,
            ndvi_status=ndvi_status,
            aoi_containment_status=metadata.get("aoi_containment_status"),
            actual_fixture_bounds_wgs84=metadata.get("actual_fixture_bounds_wgs84"),
        )

        # Extract calibrated Sentinel-1 RTC dual-pol DpRVI provenance
        dprvi_res = metadata.get("dprvi_result") or metadata.get("dprvi") or (metadata.get("optical_sar_joint", {}).get("dprvi") if isinstance(metadata.get("optical_sar_joint"), dict) else None)
        dprvi_mean = None
        dprvi_p90 = None
        q_lin = None
        q_db = None
        vv_mean_db = None
        vh_mean_db = None
        sar_pol_mode = None
        sar_veg_pct = None
        dprvi_veg_thresh = None
        vv_id = None
        vh_id = None
        dprvi_status = None
        sar_terrain_status = None

        if isinstance(dprvi_res, dict):
            dprvi_mean = dprvi_res.get("dprvi_mean")
            dprvi_p90 = dprvi_res.get("dprvi_p90")
            q_lin = dprvi_res.get("cross_pol_ratio_linear")
            q_db = dprvi_res.get("cross_pol_ratio_db")
            vv_mean_db = dprvi_res.get("vv_mean_db")
            vh_mean_db = dprvi_res.get("vh_mean_db")
            sar_pol_mode = dprvi_res.get("polarization_mode", "DUAL_VV_VH")
            sar_veg_pct = dprvi_res.get("vegetation_proxy_coverage_pct")
            dprvi_veg_thresh = dprvi_res.get("vegetation_threshold", 0.45)
            vv_id = dprvi_res.get("provenance", {}).get("vv_asset_id") or "VV"
            vh_id = dprvi_res.get("provenance", {}).get("vh_asset_id") or "VH"
            dprvi_status = dprvi_res.get("status")
            sar_terrain_status = dprvi_res.get("validity_policy", {}).get("sar_terrain_layover_shadow_mask", "unavailable_from_stac_provider")
        elif evidence_package:
            for pm in evidence_package.physical_measurements:
                if pm.name == "sar_mean_dprvi" and pm.value is not None:
                    try:
                        dprvi_mean = float(pm.value)
                    except (ValueError, TypeError):
                        pass
                if pm.name == "sar_p90_dprvi" and pm.value is not None:
                    try:
                        dprvi_p90 = float(pm.value)
                    except (ValueError, TypeError):
                        pass
                if pm.name == "sar_cross_pol_ratio_linear" and pm.value is not None:
                    try:
                        q_lin = float(pm.value)
                    except (ValueError, TypeError):
                        pass
                if pm.name == "sar_cross_pol_ratio_db" and pm.value is not None:
                    try:
                        q_db = float(pm.value)
                    except (ValueError, TypeError):
                        pass
                if pm.name == "sar_vv_mean_to_db" and pm.value is not None:
                    try:
                        vv_mean_db = float(pm.value)
                    except (ValueError, TypeError):
                        pass
                if pm.name == "sar_vh_mean_to_db" and pm.value is not None:
                    try:
                        vh_mean_db = float(pm.value)
                    except (ValueError, TypeError):
                        pass
            for hi in evidence_package.heuristic_indicators:
                if hi.name == "radar_vegetation_volume_coverage":
                    sar_veg_pct = hi.percentage
                    dprvi_veg_thresh = 0.45

        provenance.dprvi_mean = dprvi_mean
        provenance.dprvi_p90 = dprvi_p90
        provenance.cross_pol_ratio_linear = q_lin
        provenance.cross_pol_ratio_db = q_db
        provenance.sar_vv_mean_db = vv_mean_db
        provenance.sar_vh_mean_db = vh_mean_db
        provenance.sar_polarization_mode = sar_pol_mode
        provenance.sar_vegetation_proxy_coverage_pct = sar_veg_pct
        provenance.dprvi_vegetation_threshold = dprvi_veg_thresh
        provenance.vv_asset_id = vv_id
        provenance.vh_asset_id = vh_id
        provenance.dprvi_status = dprvi_status
        provenance.sar_terrain_mask_status = sar_terrain_status

        # Extract calibrated bi-temporal change provenance
        opt_chg = metadata.get("temporal_optical_change") or (metadata.get("optical_sar_joint", {}).get("temporal_optical_change") if isinstance(metadata.get("optical_sar_joint"), dict) else None)
        if isinstance(opt_chg, dict):
            provenance.delta_ndvi_mean = opt_chg.get("delta_ndvi_mean")
            provenance.vegetation_loss_pct = opt_chg.get("vegetation_loss_pct")
            provenance.vegetation_gain_pct = opt_chg.get("vegetation_gain_pct")
            provenance.temporal_t1_datetime = opt_chg.get("t1_datetime")
            provenance.temporal_t2_datetime = opt_chg.get("t2_datetime")
            provenance.joint_valid_pixel_count = opt_chg.get("joint_valid_pixel_count")

        sar_chg = metadata.get("temporal_sar_change") or (metadata.get("optical_sar_joint", {}).get("temporal_sar_change") if isinstance(metadata.get("optical_sar_joint"), dict) else None)
        if isinstance(sar_chg, dict):
            provenance.delta_dprvi_mean = sar_chg.get("delta_dprvi_mean")
            provenance.radar_canopy_loss_pct = sar_chg.get("radar_canopy_loss_pct")
            provenance.radar_canopy_gain_pct = sar_chg.get("radar_canopy_gain_pct")
            if not provenance.temporal_t1_datetime:
                provenance.temporal_t1_datetime = sar_chg.get("t1_datetime")
            if not provenance.temporal_t2_datetime:
                provenance.temporal_t2_datetime = sar_chg.get("t2_datetime")
            if not provenance.joint_valid_pixel_count:
                provenance.joint_valid_pixel_count = sar_chg.get("joint_valid_pixel_count")
            provenance.dprvi_assumption_violations_count = sar_chg.get("dprvi_assumption_violations", {}).get("total_count", 0)

        temp_prov = metadata.get("temporal_provenance")
        if isinstance(temp_prov, dict):
            if not provenance.temporal_t1_datetime:
                provenance.temporal_t1_datetime = temp_prov.get("t1", {}).get("datetime")
            if not provenance.temporal_t2_datetime:
                provenance.temporal_t2_datetime = temp_prov.get("t2", {}).get("datetime")

        if not provenance.temporal_t1_datetime and acquisition_timestamps:
            provenance.temporal_t1_datetime = acquisition_timestamps[0]
            if len(acquisition_timestamps) > 1:
                provenance.temporal_t2_datetime = acquisition_timestamps[1]

        # 3. Physical Measurements
        physical_measurements: List[PhysicalMeasurementReportItem] = []
        valid_artifact_ids = set()

        if evidence_package:
            valid_artifact_ids = {a.artifact_id for a in evidence_package.evidence_artifacts}
            for pm in evidence_package.physical_measurements:
                # Map source artifact ID based on measurement modality
                matched_art_ids = []
                name_l = pm.name.lower()
                cal_l = str(pm.calibration_quantity or "").lower()
                if "optical" in name_l or "reflectance" in name_l or "boa" in cal_l:
                    if "optical_input" in valid_artifact_ids:
                        matched_art_ids.append("optical_input")
                    elif "ev_t1_before" in valid_artifact_ids:
                        matched_art_ids.append("ev_t1_before")
                if "sar" in name_l or "power" in name_l or "gamma" in cal_l or "sigma" in cal_l:
                    if "sar_input" in valid_artifact_ids:
                        matched_art_ids.append("sar_input")
                if "change" in name_l or "bitemporal" in name_l or "delta" in name_l:
                    if "ev_change_map" in valid_artifact_ids:
                        matched_art_ids.append("ev_change_map")
                    elif "ev_temporal_optical_change_map" in valid_artifact_ids:
                        matched_art_ids.append("ev_temporal_optical_change_map")
                    elif "ev_temporal_sar_change_map" in valid_artifact_ids:
                        matched_art_ids.append("ev_temporal_sar_change_map")
                if not matched_art_ids and valid_artifact_ids:
                    # If specific modality not matched, leave empty or link joint mask if applicable
                    if "mask" in pm.mask_applied.lower() and pm.mask_applied in valid_artifact_ids:
                        matched_art_ids.append(pm.mask_applied)
                    elif "ev_change_map" in valid_artifact_ids:
                        matched_art_ids.append("ev_change_map")

                physical_measurements.append(PhysicalMeasurementReportItem(
                    name=pm.name,
                    value=pm.value,
                    unit=pm.unit,
                    statistic_type=pm.statistic_type,
                    calibration_quantity=pm.calibration_quantity,
                    mask_applied=pm.mask_applied,
                    source_artifact_ids=matched_art_ids
                ))

        # 4. Display Statistics (Visualization Only)
        display_statistics: List[DisplayStatisticItem] = []
        if evidence_package:
            display_statistics = list(evidence_package.display_statistics)

        # 5. Heuristic Indicators (Never certified classifications)
        heuristic_indicators: List[HeuristicIndicatorItem] = []
        if evidence_package:
            for hi in evidence_package.heuristic_indicators:
                # Strictly enforce is_certified_classification is False
                heuristic_indicators.append(HeuristicIndicatorItem(
                    name=hi.name,
                    percentage=hi.percentage,
                    pixel_count=hi.pixel_count,
                    definition=hi.definition,
                    is_certified_classification=False,
                    limitations=hi.limitations
                ))

        # 6. Evidence Artifacts
        evidence_artifacts: List[ReportEvidenceArtifactItem] = []
        if evidence_package and evidence_package.evidence_artifacts:
            for ea in evidence_package.evidence_artifacts:
                evidence_artifacts.append(ReportEvidenceArtifactItem(
                    artifact_id=ea.artifact_id,
                    artifact_type=ea.artifact_type,
                    title=ea.title,
                    description=ea.description,
                    statistics=ea.statistics or {}
                ))
        elif resp_data.get("evidence"):
            for ev in resp_data["evidence"]:
                evidence_artifacts.append(ReportEvidenceArtifactItem(
                    artifact_id=ev.get("id", "artifact"),
                    artifact_type=ev.get("type", "processed"),
                    title=ev.get("title", "Evidence Artifact"),
                    description=ev.get("description"),
                    statistics=ev.get("statistics") or {}
                ))

        # 7. Verification Summary
        verification_summary: Optional[ReportVerificationSummarySchema] = None
        verif_raw = resp_data.get("verification")
        if isinstance(verif_raw, dict):
            try:
                v_obj = VerificationResponseSchema(**verif_raw)
            except Exception:
                v_obj = None
        elif isinstance(verif_raw, VerificationResponseSchema):
            v_obj = verif_raw
        else:
            v_obj = None

        if v_obj:
            verification_summary = ReportVerificationSummarySchema(
                verification_status=v_obj.verification_status,
                deterministic_passed=v_obj.deterministic_passed,
                deterministic_failures=list(v_obj.deterministic_failures),
                claims=list(v_obj.claims),
                contradictions=list(v_obj.contradictions),
                unsupported_claims=list(v_obj.unsupported_claims),
                missing_evidence=list(v_obj.missing_evidence),
                scientific_limitations=list(v_obj.scientific_limitations),
                recommended_checks=list(v_obj.recommended_checks),
                is_interpretive_only=v_obj.is_interpretive_only,
                interpretive_statement=v_obj.interpretive_statement,
                summary_explanation=v_obj.summary_explanation,
            )

        # 8. Reproducibility
        trace_data = resp_data.get("trace") or {}
        trace_steps_raw = trace_data.get("steps") or []
        pipeline_steps = []
        for step in trace_steps_raw:
            if isinstance(step, dict) and step.get("step"):
                pipeline_steps.append(str(step["step"]))
            elif hasattr(step, "step"):
                pipeline_steps.append(str(step.step))

        if quality_screening and "RADIOMETRIC_QUALITY_SCREENING_APPLIED" not in pipeline_steps:
            pipeline_steps.append("RADIOMETRIC_QUALITY_SCREENING_APPLIED")

        if (getattr(provenance, "ndvi_mean", None) is not None or getattr(provenance, "ndvi_status", None)) and "SENTINEL2_CALIBRATED_NDVI_COMPUTED" not in pipeline_steps:
            pipeline_steps.append("SENTINEL2_CALIBRATED_NDVI_COMPUTED")

        if (getattr(provenance, "dprvi_mean", None) is not None or getattr(provenance, "dprvi_status", None)) and "SENTINEL1_RTC_DUAL_POL_COMPUTED" not in pipeline_steps:
            pipeline_steps.append("SENTINEL1_RTC_DUAL_POL_COMPUTED")

        reproducibility = ReportReproducibilitySchema(
            software_version="SatQuery-AI v1.0.0 (ISRO PS 26167 Stage 8 Scientific Reporting)",
            provider_config={
                "provider_name": provider_name,
                "fallback_activated": fallback_used,
                "trained_weights_present": is_trained_model,
                "mode": mode,
            },
            pipeline_steps=pipeline_steps,
            unavailable_fields=unavailable_fields,
        )

        return ScientificReportSchema(
            report_id=f"report_{analysis_id}",
            report_version="1.0.0",
            generated_at=created_at,
            provenance=provenance,
            physical_measurements=physical_measurements,
            display_statistics=display_statistics,
            heuristic_indicators=heuristic_indicators,
            evidence_artifacts=evidence_artifacts,
            verification_summary=verification_summary,
            reproducibility=reproducibility,
            executive_summary=answer_text,
        )


def generate_scientific_markdown(report: ScientificReportSchema) -> str:
    """
    Renders a validated ScientificReportSchema as a formal Markdown document.
    """
    prov = report.provenance
    verif = report.verification_summary

    lines: List[str] = []
    lines.append(f"# SAT-QUERY-AI — Scientific Analysis & Verification Report")
    lines.append("")
    lines.append(f"**Report ID:** `{report.report_id}` | **Generated:** `{report.generated_at}`")
    lines.append(f"**Software:** {report.reproducibility.software_version}")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append(f"> {report.integrity_notice}")
    lines.append("")

    # Provenance
    lines.append("## 1. Analysis Provenance & Geometry")
    lines.append("")
    lines.append("| Parameter | Value | Parameter | Value |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(f"| **Analysis ID** | `{prov.analysis_id}` | **Task Classification** | `{prov.task}` |")
    lines.append(f"| **Operational Mode** | `{prov.mode}` | **Coordinate System (CRS)** | `{prov.crs or 'Not Provided'}` |")
    lines.append(f"| **Spatial Resolution** | `{prov.spatial_resolution_m or 'Uncalibrated'}` | **Raster Dimensions** | `{prov.dimensions or 'Not Specified'}` |")
    reg_val = "True (Verified)" if prov.registration_status.get("co_registered") else "False / Unperformed"
    lines.append(f"| **Co-Registration** | `{reg_val}` | **Valid Pixel Footprint** | `{prov.valid_pixel_percentage or 'N/A'}%` (`{prov.valid_pixel_count or 'N/A'}` px) |")
    if prov.quality_screening:
        lines.append(f"| **Quality Screening** | `Applied` | **Clear Quality Pixels** | `{prov.quality_clearance_percentage or 'N/A'}%` (Excluded: `{prov.excluded_quality_pixel_count or 0}` px) |")
    lines.append(f"| **Provider Engine** | `{prov.provider_name}` | **Model Status** | `{prov.model_status or 'baseline'}` (Fallback: `{prov.fallback_used}`) |")
    lines.append(f"| **Sensor Identities** | `{', '.join(prov.sensor_identities) or 'None'}` | **Acquisition Datetimes** | `{', '.join(prov.acquisition_timestamps) or 'None'}` |")
    if prov.aoi_containment_status == "out_of_bounds_clamped":
        lines.append(f"| **AOI Containment** | `Out of Bounds (Clamped Demonstration Fixture)` | **Actual Fixture Bounds (WGS84)** | `{prov.actual_fixture_bounds_wgs84 or 'N/A'}` |")

    if prov.ndvi_mean is not None or prov.ndvi_status:
        lines.append("")
        lines.append("### Calibrated Sentinel-2 B04+B08 NDVI Provenance")
        lines.append("")
        lines.append("| Metric / Parameter | Value | Description / Standard |")
        lines.append("| :--- | :--- | :--- |")
        if prov.ndvi_mean is not None:
            lines.append(f"| **Mean NDVI** | `{prov.ndvi_mean}` | Unitless ratio [-1.0, 1.0] across valid screened pixels |")
        if prov.ndvi_p90 is not None:
            lines.append(f"| **90th-Percentile NDVI (p90)** | `{prov.ndvi_p90}` | Unitless ratio [-1.0, 1.0] representing peak vegetation vitality |")
        if prov.vegetation_coverage_pct is not None:
            lines.append(f"| **Vegetation Coverage** | `{prov.vegetation_coverage_pct}%` | Valid pixels exceeding calibrated NDVI threshold `{prov.ndvi_vegetation_threshold or 0.40}` |")
        if prov.reflectance_scale_factor:
            lines.append(f"| **Calibration Scale** | `{prov.reflectance_scale_factor}` | Integer reflectance divisor (BOA surface reflectance DN / 10000.0) |")
        if prov.scl_masking_policy:
            lines.append(f"| **Quality Mask Policy** | `{prov.scl_masking_policy}` | Classes 0, 1, 3, 8, 9, 10 excluded; invalid denominator masked |")
        if prov.ndvi_status == "unsupported_zero_valid_pixels":
            lines.append(f"| **Calculation Status** | `unsupported_zero_valid_pixels` | Zero valid clear pixels observed; numerical NDVI statistics suppressed |")

    if prov.dprvi_mean is not None or prov.dprvi_status:
        lines.append("")
        lines.append("### Sentinel-1 RTC Dual-Polarization (VV + VH) & DpRVI Provenance")
        lines.append("")
        lines.append("| Metric / Parameter | Value | Description / Standard |")
        lines.append("| :--- | :--- | :--- |")
        if prov.dprvi_mean is not None:
            lines.append(f"| **Mean DpRVI** | `{prov.dprvi_mean}` | Dual-Polarimetric Radar Vegetation Index [0.0, 1.0] (Mandal et al., 2020) |")
        if prov.dprvi_p90 is not None:
            lines.append(f"| **90th-Percentile DpRVI (p90)** | `{prov.dprvi_p90}` | Peak radar volume scattering proxy [0.0, 1.0] |")
        if prov.cross_pol_ratio_linear is not None:
            lines.append(f"| **Cross-Pol Ratio (linear)** | `{prov.cross_pol_ratio_linear}` | Mean linear power ratio q = VH / VV across valid radar pixels |")
        if prov.cross_pol_ratio_db is not None:
            lines.append(f"| **Cross-Pol Ratio (dB)** | `{prov.cross_pol_ratio_db} dB` | 10*log10(mean cross-pol ratio) |")
        if prov.sar_vv_mean_db is not None:
            lines.append(f"| **Mean VV Backscatter** | `{prov.sar_vv_mean_db} dB` | Radiometric average: 10*log10(mean linear gamma0) |")
        if prov.sar_vh_mean_db is not None:
            lines.append(f"| **Mean VH Backscatter** | `{prov.sar_vh_mean_db} dB` | Radiometric average: 10*log10(mean linear gamma0) |")
        if prov.sar_vegetation_proxy_coverage_pct is not None:
            lines.append(f"| **Radar Vegetation Proxy** | `{prov.sar_vegetation_proxy_coverage_pct}%` | Valid pixels exceeding DpRVI canopy threshold `{prov.dprvi_vegetation_threshold or 0.45}` |")
        lines.append(f"| **Terrain Layover/Shadow Mask** | `{prov.sar_terrain_mask_status or 'unavailable_from_stac_provider'}` | Provider geometric mask unavailable; terrain relief shadows not masked |")
        if prov.dprvi_status == "unsupported_zero_valid_pixels":
            lines.append(f"| **Calculation Status** | `unsupported_zero_valid_pixels` | Zero valid radar pixels observed; numerical dual-pol statistics suppressed |")

    if prov.temporal_t1_datetime and prov.temporal_t2_datetime and prov.vegetation_loss_pct is None and prov.radar_canopy_loss_pct is None:
        lines.append("")
        lines.append("### Bi-Temporal Optical Acquisition & Registration Provenance")
        lines.append("")
        lines.append("| Metric / Parameter | Value | Description / Standard |")
        lines.append("| :--- | :--- | :--- |")
        lines.append(f"| **T1 Acquisition (Pre-Event)** | `{prov.temporal_t1_datetime}` | Baseline optical acquisition timestamp |")
        lines.append(f"| **T2 Acquisition (Post-Event)** | `{prov.temporal_t2_datetime}` | Comparison optical acquisition timestamp |")
        co_reg = prov.registration_status.get("co_registered", False) if isinstance(prov.registration_status, dict) else False
        reg_m = prov.registration_status.get("registration_method", "N/A") if isinstance(prov.registration_status, dict) else "N/A"
        lines.append(f"| **Co-Registration Status** | `{co_reg} ({reg_m})` | Spatial alignment verification |")
        lines.append("| **Scientific Classification** | `Candidate Spectral Delta (Unconfirmed)` | Raw pixel differences require independent ground verification |")

    if prov.delta_ndvi_mean is not None or (prov.temporal_t1_datetime and prov.vegetation_loss_pct is not None):
        lines.append("")
        lines.append("### Sentinel-2 Temporal Optical Change & SCL Screening Provenance")
        lines.append("")
        lines.append("| Metric / Parameter | Value | Description / Standard |")
        lines.append("| :--- | :--- | :--- |")
        if prov.temporal_t1_datetime:
            lines.append(f"| **T1 Acquisition** | `{prov.temporal_t1_datetime}` | Baseline optical acquisition timestamp |")
        if prov.temporal_t2_datetime:
            lines.append(f"| **T2 Acquisition** | `{prov.temporal_t2_datetime}` | Comparison optical acquisition timestamp |")
        if prov.delta_ndvi_mean is not None:
            lines.append(f"| **Mean ΔNDVI** | `{prov.delta_ndvi_mean}` | Temporal difference (NDVI_T2 - NDVI_T1) [-2.0, 2.0] across joint valid pixels |")
        if prov.vegetation_loss_pct is not None:
            lines.append(f"| **Vegetation Loss** | `{prov.vegetation_loss_pct}%` | Joint valid pixels with ΔNDVI <= -0.15 |")
        if prov.vegetation_gain_pct is not None:
            lines.append(f"| **Vegetation Gain** | `{prov.vegetation_gain_pct}%` | Joint valid pixels with ΔNDVI >= +0.15 |")
        if prov.joint_valid_pixel_count is not None:
            lines.append(f"| **Joint Clear Pixels** | `{prov.joint_valid_pixel_count}` px | Pixels clear of cloud/shadow in both T1 and T2 SCL masks |")

    if prov.delta_dprvi_mean is not None or (prov.temporal_t1_datetime and prov.radar_canopy_loss_pct is not None):
        lines.append("")
        lines.append("### Sentinel-1 RTC Temporal SAR Change & DpRVI Provenance")
        lines.append("")
        lines.append("| Metric / Parameter | Value | Description / Standard |")
        lines.append("| :--- | :--- | :--- |")
        if prov.temporal_t1_datetime:
            lines.append(f"| **T1 Acquisition** | `{prov.temporal_t1_datetime}` | Baseline SAR acquisition timestamp |")
        if prov.temporal_t2_datetime:
            lines.append(f"| **T2 Acquisition** | `{prov.temporal_t2_datetime}` | Comparison SAR acquisition timestamp |")
        if prov.delta_dprvi_mean is not None:
            lines.append(f"| **Mean ΔDpRVI** | `{prov.delta_dprvi_mean}` | Dual-pol radar vegetation index change (DpRVI_T2 - DpRVI_T1) [-1.0, 1.0] |")
        if prov.radar_canopy_loss_pct is not None:
            lines.append(f"| **Radar Canopy Loss** | `{prov.radar_canopy_loss_pct}%` | Joint valid pixels with ΔDpRVI <= -0.15 |")
        if prov.radar_canopy_gain_pct is not None:
            lines.append(f"| **Radar Canopy Gain** | `{prov.radar_canopy_gain_pct}%` | Joint valid pixels with ΔDpRVI >= +0.15 |")
        if prov.dprvi_assumption_violations_count:
            lines.append(f"| **DpRVI Assumption Violations** | `{prov.dprvi_assumption_violations_count}` px | Observations where q = VH/VV > 1.0 violating Mandal et al. (2020) |")
        lines.append(f"| **Terrain Layover/Shadow Mask** | `{prov.sar_terrain_mask_status or 'unavailable_from_stac_provider'}` | Geometric layover/shadow mask unavailable; relief shadows unmasked |")

    lines.append("")

    # Executive Synthesis
    lines.append("## 2. Executive Synthesis & Findings")
    lines.append("")
    lines.append(report.executive_summary)
    lines.append("")

    # Calibrated Physical Measurements
    lines.append("## 3. Calibrated Physical Measurements")
    lines.append("")
    if report.physical_measurements:
        lines.append("| Measurement | Value | Unit | Statistic Type | Mask Applied | Supporting Artifacts |")
        lines.append("| :--- | :---: | :--- | :--- | :--- | :--- |")
        for pm in report.physical_measurements:
            arts = ", ".join(f"`{a}`" for a in pm.source_artifact_ids) if pm.source_artifact_ids else "None"
            lines.append(f"| **{pm.name}** | `{pm.value}` | `{pm.unit}` | {pm.statistic_type} | `{pm.mask_applied}` | {arts} |")
    else:
        lines.append("*No calibrated physical measurements available in this analysis package.*")
    lines.append("")

    # Display Statistics (Visualization Only)
    lines.append("## 4. Normalized Display Statistics (Visualization Only)")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> Display statistics are 8-bit normalized or percentile-stretched arrays generated exclusively for screen rendering. They MUST NOT be conflated with calibrated physical radiometry.")
    lines.append("")
    if report.display_statistics:
        lines.append("| Display Statistic | Render Value | Scale | Purpose |")
        lines.append("| :--- | :---: | :--- | :--- |")
        for ds in report.display_statistics:
            lines.append(f"| **{ds.name}** | `{ds.value}` | {ds.scale} | `{ds.purpose}` |")
    else:
        lines.append("*No separate display statistics recorded.*")
    lines.append("")

    # Heuristic Indicators
    lines.append("## 5. Preliminary Heuristic Indicators")
    lines.append("")
    lines.append("> [!WARNING]")
    lines.append("> Heuristic indicators are uncalibrated empirical threshold proxies. They are NOT certified land-cover classifications or ground truth.")
    lines.append("")
    if report.heuristic_indicators:
        lines.append("| Indicator | Extent (%) | Pixel Count | Empirical Threshold | Certified Classification? |")
        lines.append("| :--- | :---: | :---: | :--- | :---: |")
        for hi in report.heuristic_indicators:
            cert_str = "YES" if hi.is_certified_classification else "**NO (Proxy Only)**"
            lines.append(f"| **{hi.name}** | `{hi.percentage}%` | `{hi.pixel_count}` | `{hi.definition}` | {cert_str} |")
    else:
        lines.append("*No heuristic indicators computed for this query.*")
    lines.append("")

    # Evidence Artifacts
    lines.append("## 6. Generated Evidence Artifacts")
    lines.append("")
    if report.evidence_artifacts:
        lines.append("| Artifact ID | Type | Title | Summary Statistics |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for ea in report.evidence_artifacts:
            stats_str = ", ".join(f"{k}: {v}" for k, v in ea.statistics.items()) if ea.statistics else "Visual"
            lines.append(f"| `{ea.artifact_id}` | `{ea.artifact_type}` | {ea.title} | {stats_str} |")
    else:
        lines.append("*No visual evidence artifacts generated.*")
    lines.append("")

    # Scientific Verification Summary
    lines.append("## 7. Evidence-Grounded Scientific Verification")
    lines.append("")
    if verif:
        status_badge = verif.verification_status.upper()
        lines.append(f"**Verification Status:** `{status_badge}` | **Deterministic Checks Passed:** `{verif.deterministic_passed}`")
        lines.append("")
        lines.append(f"**Summary:** {verif.summary_explanation}")
        lines.append("")

        if verif.claims:
            lines.append("### Evaluated Scientific Claims")
            lines.append("")
            lines.append("| Claim | Outcome | Cited Artifacts | Justification / Contradiction Reason |")
            lines.append("| :--- | :---: | :--- | :--- |")
            for c in verif.claims:
                cited = ", ".join(f"`{a}`" for a in c.cited_artifact_ids) if c.cited_artifact_ids else "None"
                reason_clean = (c.reason or "Verified against evidence package.").replace("\n", " ")
                lines.append(f"| {c.claim_text} | **`{c.status.upper()}`** | {cited} | {reason_clean} |")
            lines.append("")

        if verif.contradictions:
            lines.append("### Contradictions Identified")
            lines.append("")
            for con in verif.contradictions:
                lines.append(f"- ❌ {con}")
            lines.append("")

        if verif.scientific_limitations:
            lines.append("### Scientific Caveats & Limitations")
            lines.append("")
            for lim in verif.scientific_limitations:
                lines.append(f"- ⚠️ {lim}")
            lines.append("")
    else:
        lines.append("*Stage 7 verification was not executed for this run.*")
    lines.append("")

    # Reproducibility & Traceability
    lines.append("## 8. Reproducibility & Pipeline Traceability")
    lines.append("")
    rep = report.reproducibility
    if rep.pipeline_steps:
        lines.append(f"**Execution Pipeline Steps:** `{ ' → '.join(rep.pipeline_steps) }`")
        lines.append("")
    if rep.unavailable_fields:
        lines.append("### Unavailable / Unprovided Metadata Fields")
        lines.append("")
        for uf in rep.unavailable_fields:
            lines.append(f"- `{uf}`")
        lines.append("")
    else:
        lines.append("*All required pipeline metadata fields were observed and verified.*")
        lines.append("")

    lines.append("---")
    lines.append("*Report generated autonomously by SatQuery AI Multimodal Architecture (ISRO PS 26167).*")
    return "\n".join(lines)
