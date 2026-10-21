"""
Unit and integration test suite for Stage 8: Scientific Analysis Quality and Evidence Reporting.
Tests all 15 required verification criteria:
1. Complete report generated from valid evidence package
2. Missing provenance fields represented honestly
3. Physical measurements retaining correct units and statistic types
4. Display values remaining separate from physical measurements
5. Gamma-naught and dB statistic names remaining correct
6. Heuristic indicators retaining preliminary/non-certified status
7. Claim verification statuses and evidence references preserved
8. Unknown or fabricated artifact references rejected
9. Unsupported and contradicted claims remaining unchanged
10. Invalid report data failing schema validation
11. Report-generation failure preserving underlying analysis response
12. Deterministic report generation for identical inputs
13. API compatibility with existing clients
14. Markdown report export and schema endpoint behavior
15. Safe handling of filenames and exported content (no secrets, path traversal prevention)
"""

import copy
import re
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas.report import (
    ScientificReportSchema,
    ReportProvenanceSchema,
    PhysicalMeasurementReportItem,
    ReportEvidenceArtifactItem,
    ReportVerificationSummarySchema,
    ReportReproducibilitySchema,
)
from app.schemas.verification import (
    EvidencePackageSchema,
    PhysicalMeasurementItem,
    DisplayStatisticItem,
    HeuristicIndicatorItem,
    EvidenceArtifactItem,
    ProviderInfoItem,
    ClaimVerificationItem,
    VerificationResponseSchema,
)
from app.schemas.analysis import AnalysisRequest, AnalysisResponseSchema
from app.reports.scientific_report import (
    ScientificReportBuilder,
    generate_scientific_markdown,
)
from app.api.routes_reports import sanitize_filename
from app.agent.controller import agent_controller

client = TestClient(app)


def build_full_valid_evidence_package() -> EvidencePackageSchema:
    """Helper to build a complete, verified multimodal evidence package."""
    meas = [
        PhysicalMeasurementItem(
            name="optical_mean_boa_reflectance",
            value=0.1298,
            unit="unitless ratio [0, 1]",
            statistic_type="mean surface reflectance",
            mask_applied="joint_valid_mask",
            calibration_quantity="boa_surface_reflectance",
        ),
        PhysicalMeasurementItem(
            name="sar_mean_linear_power",
            value=0.7417,
            unit="linear power ratio",
            statistic_type="mean linear gamma-naught power",
            mask_applied="joint_valid_mask",
            calibration_quantity="gamma0",
        ),
        PhysicalMeasurementItem(
            name="sar_mean_pixel_db",
            value=-4.98,
            unit="dB",
            statistic_type="mean pixel-wise gamma-naught in decibels",
            mask_applied="joint_valid_mask",
            calibration_quantity="gamma0_db",
        ),
    ]

    disp = [
        DisplayStatisticItem(
            name="optical_display_mean_dn",
            value=92.8,
            scale="8-bit normalized [0, 255]",
            purpose="Screen visualization brightness",
        ),
        DisplayStatisticItem(
            name="sar_display_mean_dn",
            value=105.8,
            scale="8-bit normalized [0, 255]",
            purpose="Screen visualization intensity",
        ),
    ]

    heuristics = [
        HeuristicIndicatorItem(
            name="heuristic_surface_water_proxy",
            percentage=4.88,
            pixel_count=192,
            definition="Display optical <= 25 and SAR <= 40",
            is_certified_classification=False,
            limitations="Preliminary scene-normalized empirical heuristic proxy; not validated ground truth.",
        ),
        HeuristicIndicatorItem(
            name="heuristic_structural_proxy",
            percentage=6.88,
            pixel_count=271,
            definition="Display optical >= 110 and SAR >= 140",
            is_certified_classification=False,
            limitations="Preliminary scene-normalized empirical heuristic proxy; not validated ground truth.",
        ),
    ]

    artifacts = [
        EvidenceArtifactItem(
            artifact_id="optical_input",
            artifact_type="source",
            title="Calibrated Sentinel-2 B04 Surface Reflectance Window",
            description="10m BOA surface reflectance red band.",
            statistics={"mean_reflectance": 0.1298},
        ),
        EvidenceArtifactItem(
            artifact_id="sar_input",
            artifact_type="source",
            title="Calibrated Sentinel-1 RTC VV Window",
            description="10m Radiometrically Terrain Corrected VV polarization.",
            statistics={"mean_linear_power": 0.7417, "mean_db": -4.98},
        ),
        EvidenceArtifactItem(
            artifact_id="joint_valid_mask",
            artifact_type="mask",
            title="Joint Optical-SAR Valid Pixel Footprint",
            description="Binary mask excluding nodata and non-overlapping pixels.",
            statistics={"valid_pixels": 245000, "coverage_pct": 93.46},
        ),
    ]

    return EvidencePackageSchema(
        task_id="optical_sar",
        query="Quantify surface reflectance and SAR linear power over the study area.",
        sensor_identities=["Sentinel-2", "Sentinel-1"],
        source_assets=[
            {"id": "S2A_MSIL2A_20240315", "type": "optical"},
            {"id": "S1A_IW_GRDH_20240315", "type": "sar"},
        ],
        acquisition_timestamps=["2024-03-15T05:30:00Z", "2024-03-15T00:15:00Z"],
        crs="EPSG:32643",
        resolution=10.0,
        dimensions="512x512",
        registration_status={"co_registered": True, "method": "rasterio_reproject_nearest"},
        valid_pixel_count=245000,
        total_pixel_count=262144,
        valid_pixel_percentage=93.46,
        physical_measurements=meas,
        display_statistics=disp,
        heuristic_indicators=heuristics,
        evidence_artifacts=artifacts,
        provider_info=ProviderInfoItem(
            provider_name="DeterministicOpticalSARProvider",
            actual_model_used="DeterministicOpticalSARProvider",
            fallback_used=False,
            is_trained_model=False,
            model_status="baseline_real_raster",
        ),
    )


def test_01_complete_report_generated_from_valid_evidence_package():
    """Test 1: Complete report generated from a valid evidence package."""
    pkg = build_full_valid_evidence_package()
    proto = {
        "id": "analysis_test_01",
        "task": "optical_sar",
        "query": "Quantify calibrated surface reflectance and radar backscatter.",
        "mode": "optical_sar",
        "answer": "Surface reflectance averaged 0.1298; SAR linear power averaged 0.7417.",
        "created_at": "2026-10-11T00:00:00Z",
        "verification": {
            "verification_status": "supported",
            "deterministic_passed": True,
            "deterministic_failures": [],
            "claims": [
                {
                    "claim_text": "Surface reflectance averaged 0.1298",
                    "claim_type": "physical_measurement",
                    "status": "supported",
                    "cited_artifact_ids": ["optical_input"],
                    "reason": "Matches calibrated optical_input.",
                }
            ],
            "contradictions": [],
            "unsupported_claims": [],
            "missing_evidence": [],
            "scientific_limitations": ["Heuristic proxies are uncalibrated."],
            "recommended_checks": [],
            "is_interpretive_only": True,
            "interpretive_statement": "Interpretive verification only.",
            "summary_explanation": "All claims deterministic cross-checks verified.",
        },
    }

    report = ScientificReportBuilder.build_report(proto, evidence_package=pkg)

    assert isinstance(report, ScientificReportSchema)
    assert report.report_id == "report_analysis_test_01"
    assert report.provenance.crs == "EPSG:32643"
    assert report.provenance.spatial_resolution_m == 10.0
    assert report.provenance.valid_pixel_percentage == 93.46
    assert report.provenance.is_trained_model is False
    assert len(report.physical_measurements) == 3
    assert len(report.display_statistics) == 2
    assert len(report.heuristic_indicators) == 2
    assert len(report.evidence_artifacts) == 3
    assert report.verification_summary is not None
    assert report.verification_summary.verification_status == "supported"

    md = generate_scientific_markdown(report)
    assert "# SAT-QUERY-AI — Scientific Analysis & Verification Report" in md
    assert "EPSG:32643" in md
    assert "optical_mean_boa_reflectance" in md
    assert "heuristic_surface_water_proxy" in md


def test_02_missing_provenance_fields_represented_honestly():
    """Test 2: Missing provenance fields represented honestly in unavailable_fields."""
    proto = {
        "id": "analysis_test_missing_prov",
        "task": "vqa",
        "answer": "Scene contains agricultural fields.",
        "metadata": {},  # completely empty metadata
    }

    report = ScientificReportBuilder.build_report(proto, evidence_package=None)

    assert report.provenance.crs is None
    assert report.provenance.spatial_resolution_m is None
    assert report.provenance.sensor_identities == []
    assert report.provenance.acquisition_timestamps == []

    unavail = report.reproducibility.unavailable_fields
    assert any("crs:" in f for f in unavail)
    assert any("sensor_identities:" in f for f in unavail)
    assert any("acquisition_timestamps:" in f for f in unavail)
    assert any("registration_status:" in f for f in unavail)


def test_03_physical_measurements_retaining_correct_units_and_statistic_types():
    """Test 3: Physical measurements retaining correct units and statistic types."""
    pkg = build_full_valid_evidence_package()
    report = ScientificReportBuilder.build_report({"id": "a1", "answer": "ok"}, evidence_package=pkg)

    by_name = {pm.name: pm for pm in report.physical_measurements}
    assert "optical_mean_boa_reflectance" in by_name
    assert by_name["optical_mean_boa_reflectance"].unit == "unitless ratio [0, 1]"
    assert by_name["optical_mean_boa_reflectance"].statistic_type == "mean surface reflectance"
    assert by_name["optical_mean_boa_reflectance"].value == 0.1298

    assert "sar_mean_linear_power" in by_name
    assert by_name["sar_mean_linear_power"].unit == "linear power ratio"
    assert by_name["sar_mean_linear_power"].statistic_type == "mean linear gamma-naught power"
    assert by_name["sar_mean_linear_power"].value == 0.7417

    assert "sar_mean_pixel_db" in by_name
    assert by_name["sar_mean_pixel_db"].unit == "dB"
    assert by_name["sar_mean_pixel_db"].statistic_type == "mean pixel-wise gamma-naught in decibels"
    assert by_name["sar_mean_pixel_db"].value == -4.98


def test_04_display_values_remaining_separate_from_physical_measurements():
    """Test 4: Display values remaining separate from physical measurements."""
    pkg = build_full_valid_evidence_package()
    report = ScientificReportBuilder.build_report({"id": "a2", "answer": "ok"}, evidence_package=pkg)

    disp_vals = [d.value for d in report.display_statistics]
    assert 92.8 in disp_vals
    assert 105.8 in disp_vals

    phys_vals = [p.value for p in report.physical_measurements]
    assert 92.8 not in phys_vals
    assert 105.8 not in phys_vals

    md = generate_scientific_markdown(report)
    assert "Normalized Display Statistics (Visualization Only)" in md
    assert "MUST NOT be conflated with calibrated physical radiometry" in md


def test_05_gamma_naught_and_db_statistic_names_remaining_correct():
    """Test 5: Gamma-naught and dB statistic names remaining correct."""
    pkg = build_full_valid_evidence_package()
    report = ScientificReportBuilder.build_report({"id": "a3", "answer": "ok"}, evidence_package=pkg)

    sar_linear = next(p for p in report.physical_measurements if p.name == "sar_mean_linear_power")
    sar_db = next(p for p in report.physical_measurements if p.name == "sar_mean_pixel_db")

    assert sar_linear.calibration_quantity == "gamma0"
    assert "linear" in sar_linear.statistic_type.lower()
    assert sar_linear.unit != "dB"

    assert sar_db.calibration_quantity == "gamma0_db"
    assert sar_db.unit == "dB"
    assert "decibel" in sar_db.statistic_type.lower() or "db" in sar_db.statistic_type.lower()


def test_06_heuristic_indicators_retaining_preliminary_status():
    """Test 6: Heuristic indicators retaining their preliminary/non-certified status."""
    pkg = build_full_valid_evidence_package()
    report = ScientificReportBuilder.build_report({"id": "a4", "answer": "ok"}, evidence_package=pkg)

    for hi in report.heuristic_indicators:
        assert hi.is_certified_classification is False
        assert len(hi.limitations) > 0

    # Test that setting is_certified_classification=True raises ValidationError
    with pytest.raises(ValidationError) as exc_info:
        bad_rep = copy.deepcopy(report)
        bad_hi = HeuristicIndicatorItem(
            name="fake_water",
            percentage=10.0,
            pixel_count=100,
            definition="test",
            is_certified_classification=True,  # Disallowed
            limitations="none",
        )
        ScientificReportSchema(
            report_id="bad_rep",
            generated_at="2026-10-11T00:00:00Z",
            provenance=bad_rep.provenance,
            physical_measurements=[],
            display_statistics=[],
            heuristic_indicators=[bad_hi],
            evidence_artifacts=[],
            reproducibility=bad_rep.reproducibility,
            executive_summary="test",
        )
    assert "Scientific integrity violation" in str(exc_info.value)


def test_07_claim_verification_statuses_and_evidence_references_preserved():
    """Test 7: Claim verification statuses and evidence references being preserved."""
    pkg = build_full_valid_evidence_package()
    proto = {
        "id": "a5",
        "answer": "ok",
        "verification": {
            "verification_status": "partially_supported",
            "deterministic_passed": True,
            "deterministic_failures": [],
            "claims": [
                {
                    "claim_text": "Reflectance was 0.1298",
                    "claim_type": "physical_measurement",
                    "status": "supported",
                    "cited_artifact_ids": ["optical_input"],
                    "reason": "Verified",
                },
                {
                    "claim_text": "Unverified cloud percentage is 0.0%",
                    "claim_type": "heuristic",
                    "status": "unsupported",
                    "cited_artifact_ids": [],
                    "reason": "Missing cloud mask",
                },
            ],
            "contradictions": [],
            "unsupported_claims": ["Unverified cloud percentage is 0.0%"],
            "missing_evidence": ["cloud_mask"],
            "scientific_limitations": ["Cloud mask missing"],
            "recommended_checks": [],
            "is_interpretive_only": True,
            "interpretive_statement": "Interpretive note",
            "summary_explanation": "One supported, one unsupported claim.",
        },
    }

    report = ScientificReportBuilder.build_report(proto, evidence_package=pkg)
    verif = report.verification_summary
    assert verif is not None
    assert verif.verification_status == "partially_supported"
    assert len(verif.claims) == 2
    assert verif.claims[0].status == "supported"
    assert verif.claims[0].cited_artifact_ids == ["optical_input"]
    assert verif.claims[1].status == "unsupported"
    assert "cloud_mask" in verif.missing_evidence


def test_08_unknown_or_fabricated_artifact_references_rejected():
    """Test 8: Unknown or fabricated artifact references being rejected by schema validation."""
    pkg = build_full_valid_evidence_package()
    report = ScientificReportBuilder.build_report({"id": "a6", "answer": "ok"}, evidence_package=pkg)

    # 8a: Measurement cites unknown artifact
    with pytest.raises(ValidationError) as exc1:
        bad_meas = copy.deepcopy(report.physical_measurements)
        bad_meas[0].source_artifact_ids = ["fabricated_asset_xyz"]
        ScientificReportSchema(
            report_id="test_bad_art",
            generated_at="2026-10-11T00:00:00Z",
            provenance=report.provenance,
            physical_measurements=bad_meas,
            display_statistics=report.display_statistics,
            heuristic_indicators=report.heuristic_indicators,
            evidence_artifacts=report.evidence_artifacts,
            reproducibility=report.reproducibility,
            executive_summary="test",
        )
    assert "fabricated_asset_xyz" in str(exc1.value)

    # 8b: Verification claim cites unknown artifact
    with pytest.raises(ValidationError) as exc2:
        bad_verif = ReportVerificationSummarySchema(
            verification_status="supported",
            deterministic_passed=True,
            claims=[
                ClaimVerificationItem(
                    claim_text="Fabricated claim",
                    claim_type="physical",
                    status="supported",
                    cited_artifact_ids=["ghost_artifact_99"],
                )
            ],
            summary_explanation="Test verification"
        )
        ScientificReportSchema(
            report_id="test_bad_art2",
            generated_at="2026-10-11T00:00:00Z",
            provenance=report.provenance,
            physical_measurements=report.physical_measurements,
            display_statistics=report.display_statistics,
            heuristic_indicators=report.heuristic_indicators,
            evidence_artifacts=report.evidence_artifacts,
            verification_summary=bad_verif,
            reproducibility=report.reproducibility,
            executive_summary="test",
        )
    assert "ghost_artifact_99" in str(exc2.value)



def test_09_unsupported_and_contradicted_claims_remaining_unchanged():
    """Test 9: Unsupported and contradicted claims remaining unchanged in reports."""
    pkg = build_full_valid_evidence_package()
    proto = {
        "id": "a7",
        "answer": "ok",
        "verification": {
            "verification_status": "unsupported",
            "deterministic_passed": False,
            "deterministic_failures": ["Rule violation"],
            "claims": [
                {
                    "claim_text": "SAR linear power was 5.0",
                    "claim_type": "physical_measurement",
                    "status": "contradicted",
                    "cited_artifact_ids": ["sar_input"],
                    "reason": "SAR linear power is 0.7417, not 5.0",
                },
                {
                    "claim_text": "Thermal band observed 300K",
                    "claim_type": "physical_measurement",
                    "status": "unsupported",
                    "cited_artifact_ids": [],
                    "reason": "No thermal sensor in package",
                },
            ],
            "contradictions": ["SAR linear power is 0.7417, not 5.0"],
            "unsupported_claims": ["Thermal band observed 300K"],
            "missing_evidence": ["thermal_input"],
            "scientific_limitations": [],
            "recommended_checks": [],
            "is_interpretive_only": True,
            "interpretive_statement": "Interpretive note",
            "summary_explanation": "Contradictions found.",
        },
    }

    report = ScientificReportBuilder.build_report(proto, evidence_package=pkg)
    assert report.verification_summary.verification_status == "unsupported"
    statuses = [c.status for c in report.verification_summary.claims]
    assert "contradicted" in statuses
    assert "unsupported" in statuses
    assert "supported" not in statuses


def test_10_invalid_report_data_failing_schema_validation():
    """Test 10: Invalid report data failing schema validation."""
    pkg = build_full_valid_evidence_package()
    report = ScientificReportBuilder.build_report({"id": "a8", "answer": "ok"}, evidence_package=pkg)

    # 10a: Unit contradiction: linear power with unit dB
    with pytest.raises(ValidationError) as exc:
        bad_meas = copy.deepcopy(report.physical_measurements)
        bad_meas[0].statistic_type = "mean linear power"
        bad_meas[0].unit = "dB"
        ScientificReportSchema(
            report_id="test_unit_fail",
            generated_at="2026-10-11T00:00:00Z",
            provenance=report.provenance,
            physical_measurements=bad_meas,
            display_statistics=report.display_statistics,
            heuristic_indicators=report.heuristic_indicators,
            evidence_artifacts=report.evidence_artifacts,
            reproducibility=report.reproducibility,
            executive_summary="test",
        )
    assert "Unit contradiction" in str(exc.value)


def test_11_report_generation_failure_preserving_underlying_analysis_response():
    """Test 11: Report-generation failure preserving the underlying analysis response."""
    import numpy as np
    from app.remote_sensing.preprocessing import convert_array_to_base64_png
    from app.schemas.analysis import ImageInput

    img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    b64_img = convert_array_to_base64_png(img)

    # When ScientificReportBuilder raises an exception, controller handles it safely
    with patch("app.agent.controller.ScientificReportBuilder.build_report", side_effect=RuntimeError("Synthetic crash")):
        req = AnalysisRequest(
            query="What is the surface feature?",
            mode="single",
            images=[ImageInput(data=b64_img, filename="test.png", role="primary")],
        )
        resp = agent_controller.process_request(req)
        assert isinstance(resp, AnalysisResponseSchema)
        assert resp.answer != ""
        assert resp.reproducible_report is None  # Safely fell back to None without crashing response!


def test_12_deterministic_report_generation_for_identical_inputs():
    """Test 12: Deterministic report generation for identical evidence inputs."""
    pkg = build_full_valid_evidence_package()
    proto = {
        "id": "analysis_det_12",
        "task": "optical_sar",
        "query": "Quantify measurements",
        "answer": "Identical answer text",
        "created_at": "2026-10-11T00:00:00Z",
    }

    rep1 = ScientificReportBuilder.build_report(proto, evidence_package=pkg)
    rep2 = ScientificReportBuilder.build_report(proto, evidence_package=pkg)

    assert rep1.model_dump() == rep2.model_dump()

    md1 = generate_scientific_markdown(rep1)
    md2 = generate_scientific_markdown(rep2)
    assert md1 == md2


def test_13_api_compatibility_with_existing_clients():
    """Test 13: API compatibility with existing clients and standard endpoints."""
    # 13a: Health check
    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200

    # 13b: Existing PDF and JSON report endpoints
    payload = {
        "id": "test_export_13",
        "task": "vqa",
        "answer": "Analysis findings",
        "models": ["SatQueryAgent"],
        "execution_time_ms": 100.0,
    }
    res_json = client.post("/api/v1/reports/json", json=payload)
    assert res_json.status_code == 200
    assert "satquery_report_test_export_13.json" in res_json.headers["Content-Disposition"]

    res_pdf = client.post("/api/v1/reports/pdf", json=payload)
    assert res_pdf.status_code == 200
    assert "satquery_report_test_export_13.pdf" in res_pdf.headers["Content-Disposition"]


def test_14_markdown_export_and_scientific_report_endpoints():
    """Test 14: Frontend rendering and export endpoints (Markdown & Scientific Schema)."""
    pkg = build_full_valid_evidence_package()
    proto = {
        "id": "analysis_export_14",
        "task": "optical_sar",
        "answer": "Reflectance 0.1298; SAR power 0.7417.",
        "created_at": "2026-10-11T00:00:00Z",
    }
    report = ScientificReportBuilder.build_report(proto, evidence_package=pkg)

    payload = {
        "id": "analysis_export_14",
        "task": "optical_sar",
        "answer": "Reflectance 0.1298; SAR power 0.7417.",
        "reproducible_report": report.model_dump(),
    }

    # Test Markdown endpoint
    res_md = client.post("/api/v1/reports/markdown", json=payload)
    assert res_md.status_code == 200
    assert "text/markdown" in res_md.headers["Content-Type"]
    assert "satquery_report_analysis_export_14.md" in res_md.headers["Content-Disposition"]
    assert "# SAT-QUERY-AI — Scientific Analysis & Verification Report" in res_md.text

    # Test dedicated scientific schema endpoint
    res_sci = client.post("/api/v1/reports/scientific", json=payload)
    assert res_sci.status_code == 200
    sci_data = res_sci.json()
    assert sci_data["report_id"] == "report_analysis_export_14"
    assert len(sci_data["physical_measurements"]) == 3


def test_15_safe_handling_of_filenames_and_exported_content():
    """Test 15: Safe handling of filenames and exported content (path traversal prevention, no secrets)."""
    # 15a: Filename sanitization against path traversal and dangerous characters
    assert sanitize_filename("../../../etc/passwd", "md") == "satquery_report_etc_passwd.md"
    assert sanitize_filename("..\\..\\windows\\system32", "pdf") == "satquery_report_windows_system32.pdf"
    assert sanitize_filename("test; rm -rf /", "json") == "satquery_report_test__rm_-rf.json"
    assert sanitize_filename("safe_id_123", "md") == "satquery_report_safe_id_123.md"
    assert sanitize_filename("", "md") == "satquery_report_export.md"
    assert sanitize_filename(None, "md") == "satquery_report_export.md"

    # Filename length cap check
    long_name = "a" * 200
    safe_long = sanitize_filename(long_name, "md")
    assert len(safe_long) <= 64

    # 15b: Exported markdown contains no internal passwords, API keys, or absolute file paths
    pkg = build_full_valid_evidence_package()
    report = ScientificReportBuilder.build_report({"id": "sec_test", "answer": "Report text"}, evidence_package=pkg)
    md_text = generate_scientific_markdown(report)

    assert "api_key" not in md_text.lower()
    assert "secret" not in md_text.lower()
    assert "password" not in md_text.lower()
    assert "c:\\users" not in md_text.lower()
    assert "/home/" not in md_text.lower()
