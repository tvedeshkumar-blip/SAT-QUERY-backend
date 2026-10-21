"""
Comprehensive regression test suite for Stage 7: Evidence-Grounded LLM Scientific Verification.
Tests deterministic integrity checks, bounded evidence contracts, LLM schema parsing,
non-overridable failure gates, error handling, and API integration.
All LLM interactions are strictly mocked; no live network calls are made.
"""

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.verification import (
    ClaimVerificationItem,
    DisplayStatisticItem,
    EvidenceArtifactItem,
    EvidencePackageSchema,
    HeuristicIndicatorItem,
    PhysicalMeasurementItem,
    ProviderInfoItem,
    VerificationResponseSchema,
)
from app.agent.verifier import (
    DeterministicVerifier,
    EvidencePackageBuilder,
    LLMScientificVerifier,
)
from app.chatbot.lmstudio_client import LMStudioConnectionError, LMStudioAPIError

client = TestClient(app)


def build_sample_evidence_package(
    crs="EPSG:32643",
    co_registered=True,
    is_trained=False,
    include_units=True,
    sar_quantity="gamma0",
    has_artifacts=True
) -> EvidencePackageSchema:
    """Helper to build a controlled test evidence package."""
    meas = []
    if include_units:
        meas.append(PhysicalMeasurementItem(
            name="optical_mean_boa_reflectance",
            value=0.1298,
            unit="unitless (surface reflectance BOA)",
            statistic_type="mean",
            mask_applied="joint_valid_mask",
            calibration_quantity="boa_surface_reflectance"
        ))
        meas.append(PhysicalMeasurementItem(
            name="sar_mean_linear_power",
            value=0.7417,
            unit="linear power (m^2/m^2)",
            statistic_type="mean",
            mask_applied="joint_valid_mask",
            calibration_quantity=sar_quantity
        ))
        meas.append(PhysicalMeasurementItem(
            name="sar_mean_pixel_db",
            value=-4.98,
            unit="dB",
            statistic_type="mean of pixel-wise dB",
            mask_applied="joint_valid_mask",
            calibration_quantity=sar_quantity
        ))
    else:
        # Intentionally missing units and statistic type
        meas.append(PhysicalMeasurementItem(
            name="optical_mean_boa_reflectance",
            value=0.1298,
            unit="",
            statistic_type="",
            mask_applied="",
            calibration_quantity="boa_surface_reflectance"
        ))

    disp = [
        DisplayStatisticItem(
            name="optical_albedo_display_mean",
            value=92.8,
            scale="8-bit normalized [0, 255]",
            purpose="visualization_only"
        ),
        DisplayStatisticItem(
            name="sar_intensity_display_mean",
            value=105.8,
            scale="8-bit normalized [0, 255]",
            purpose="visualization_only"
        )
    ]

    heuristics = [
        HeuristicIndicatorItem(
            name="heuristic_surface_water_proxy",
            percentage=4.88,
            pixel_count=192,
            definition="Display optical <= 25 and SAR <= 40",
            is_certified_classification=False,
            limitations="Preliminary scene-normalized empirical heuristic proxy; not validated ground truth."
        ),
        HeuristicIndicatorItem(
            name="heuristic_structural_proxy",
            percentage=6.88,
            pixel_count=271,
            definition="Display optical >= 110 and SAR >= 140",
            is_certified_classification=False,
            limitations="Preliminary scene-normalized empirical heuristic proxy; not validated ground truth."
        )
    ]

    artifacts = []
    if has_artifacts:
        artifacts = [
            EvidenceArtifactItem(
                artifact_id="optical_input",
                artifact_type="original",
                title="Sentinel-2 BOA Visual",
                statistics={"mean_boa": 0.1298}
            ),
            EvidenceArtifactItem(
                artifact_id="sar_input",
                artifact_type="original",
                title="Sentinel-1 RTC Linear Power",
                statistics={"mean_linear": 0.7417}
            ),
            EvidenceArtifactItem(
                artifact_id="heuristic_water_mask",
                artifact_type="mask",
                title="Surface Water Empirical Proxy",
                statistics={"percentage": 4.88}
            )
        ]

    return EvidencePackageSchema(
        task_id="test_task_001",
        query="Verify cross-sensor surface reflectance and radar backscatter",
        sensor_identities=["Sentinel-2A", "Sentinel-1A"],
        acquisition_timestamps=["2025-05-18T05:27:01Z", "2025-05-18T01:14:22Z"],
        source_assets=[
            {"id": "S2A_B04", "collection": "sentinel-2-l2a", "calibration": "BOA_reflectance"},
            {"id": "S1A_RTC_VV", "collection": "sentinel-1-rtc", "calibration": "gamma0"}
        ],
        crs=crs,
        resolution="10.0m",
        dimensions="128x128",
        registration_status={"co_registered": co_registered, "overlap_pct": 100.0, "valid_pixels": 3936},
        valid_pixel_count=3936,
        total_pixel_count=4096,
        valid_pixel_percentage=96.09,
        physical_measurements=meas,
        display_statistics=disp,
        heuristic_indicators=heuristics,
        evidence_artifacts=artifacts,
        provider_info=ProviderInfoItem(
            provider_name="OpticalSARJointAnalysisProvider",
            actual_model_used="OpticalSARCrossSensorBaseline",
            fallback_used=False,
            is_trained_model=is_trained
        ),
        known_limitations=["Atmospheric optical shadow artifacts may persist."]
    )


# Test 1: Supported claims with valid evidence
def test_supported_claims_with_valid_evidence():
    pkg = build_sample_evidence_package()
    claims = [
        ClaimVerificationItem(
            claim_text="Sentinel-2 BOA surface reflectance mean is 0.1298 within the joint valid mask.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"],
            evidence_found="optical_mean_boa_reflectance=0.1298 unitless"
        )
    ]
    det_passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=claims)
    assert det_passed is True
    assert len(failures) == 0
    assert eval_claims[0].status == "supported"


# Test 2: Unsupported claims with no supporting artifact or missing evidence
def test_unsupported_claims_with_no_supporting_artifact():
    pkg = build_sample_evidence_package()
    claims = [
        ClaimVerificationItem(
            claim_text="Cloud coverage was exactly 0.00% across the entire scene.",
            claim_type="general",
            status="unsupported",
            cited_artifact_ids=[],
            reason="No cloud-cover metric or quality flag exists in the evidence package."
        )
    ]
    det_passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=claims)
    assert det_passed is True
    assert eval_claims[0].status == "unsupported"


# Test 3: Contradictory claims versus deterministic metadata
def test_contradictory_claims_versus_deterministic_metadata():
    pkg = build_sample_evidence_package(co_registered=False)
    claims = [
        ClaimVerificationItem(
            claim_text="The optical and SAR rasters are sub-pixel co-registered and geometrically aligned.",
            claim_type="registration",
            status="supported",
            cited_artifact_ids=["optical_input", "sar_input"]
        )
    ]
    det_passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=claims)
    assert det_passed is False
    assert any("co-registration" in f.lower() for f in failures)
    assert eval_claims[0].status == "contradicted"


# Test 4: Missing units and missing provenance
def test_missing_units_and_provenance_detected():
    pkg = build_sample_evidence_package(include_units=False)
    det_passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg)
    assert det_passed is False
    assert any("lacks a declared physical unit" in f for f in failures)


# Test 5: Physical reflectance versus normalized display values
def test_physical_reflectance_versus_normalized_display_conflation():
    pkg = build_sample_evidence_package()
    # Claim asserting display 92.8 is the BOA surface reflectance
    claims = [
        ClaimVerificationItem(
            claim_text="The scene has a mean surface reflectance of 92.8.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    det_passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=claims)
    assert det_passed is False
    assert any("conflated with physical boa surface reflectance" in f.lower() for f in failures)
    assert eval_claims[0].status == "contradicted"


# Test 6: Gamma-naught versus sigma-naught labeling
def test_gamma_naught_versus_sigma_naught_labeling():
    pkg = build_sample_evidence_package(sar_quantity="gamma0")
    # Claim describing RTC gamma-0 asset as sigma-naught
    claims = [
        ClaimVerificationItem(
            claim_text="The SAR sensor recorded mean calibrated sigma-0 backscatter of -4.98 dB.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["sar_input"]
        )
    ]
    det_passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=claims)
    assert det_passed is False
    assert any("sigma-naught" in f.lower() for f in failures)
    assert eval_claims[0].status == "contradicted"


# Test 7: Mean linear power versus mean pixel-wise dB naming
def test_mean_linear_power_versus_decibel_naming():
    pkg = build_sample_evidence_package()
    claims = [
        ClaimVerificationItem(
            claim_text="The radar mean linear power is 0.7417 dB.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["sar_input"]
        )
    ]
    det_passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=claims)
    assert det_passed is False
    assert any("linear power" in f.lower() and "db" in f.lower() for f in failures)
    assert eval_claims[0].status == "contradicted"


# Test 8: Heuristic indicators incorrectly described as certified classifications
def test_heuristic_indicator_incorrectly_described_as_certified():
    pkg = build_sample_evidence_package()
    claims = [
        ClaimVerificationItem(
            claim_text="The system detected certified surface water ground truth of 4.88%.",
            claim_type="heuristic",
            status="supported",
            cited_artifact_ids=["heuristic_water_mask"]
        )
    ]
    det_passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=claims)
    assert det_passed is False
    assert any("certified" in f.lower() for f in failures)
    assert eval_claims[0].status == "contradicted"


# Test 9: False trained-model claims
def test_false_trained_model_claim():
    pkg = build_sample_evidence_package(is_trained=False)
    claims = [
        ClaimVerificationItem(
            claim_text="A deep learning neural network model was used for cross-sensor inference.",
            claim_type="model",
            status="supported"
        )
    ]
    det_passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=claims)
    assert det_passed is False
    assert any("trained model" in f.lower() for f in failures)
    assert eval_claims[0].status == "contradicted"


# Test 10: Unknown artifact references
def test_unknown_artifact_references():
    pkg = build_sample_evidence_package()
    claims = [
        ClaimVerificationItem(
            claim_text="Observed in artifact non_existent_artifact_999.",
            claim_type="general",
            status="supported",
            cited_artifact_ids=["non_existent_artifact_999"]
        )
    ]
    det_passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=claims)
    assert det_passed is False
    assert any("unknown artifact id" in f.lower() for f in failures)
    assert eval_claims[0].status == "unsupported"


# Test 11: Non-overridable deterministic gate (LLM claims supported, deterministic overrides to unsupported)
def test_llm_cannot_override_deterministic_failures():
    pkg = build_sample_evidence_package(co_registered=False)
    mock_client = MagicMock()
    mock_client.is_reachable.return_value = True
    
    # LLM hallucinates that verification passed with supported status!
    hallucinated_llm_json = {
        "verification_status": "supported",
        "claims": [
            {
                "claim_text": "Scenes are pixel-aligned and co-registered.",
                "claim_type": "registration",
                "status": "supported",
                "cited_artifact_ids": ["optical_input"],
                "reason": "LLM erroneously thinks it is fine."
            }
        ],
        "contradictions": [],
        "unsupported_claims": [],
        "missing_evidence": [],
        "scientific_limitations": [],
        "recommended_checks": [],
        "summary_explanation": "LLM claims everything is valid."
    }
    mock_client.chat_completion.return_value = {
        "content": json.dumps(hallucinated_llm_json),
        "model": "mock-llm-model"
    }

    verifier = LLMScientificVerifier(client=mock_client)
    res = verifier.verify(
        evidence_package=pkg,
        candidate_claims=[
            ClaimVerificationItem(
                claim_text="Scenes are pixel-aligned and co-registered.",
                claim_type="registration",
                status="supported"
            )
        ]
    )

    # Verification status MUST NOT be 'supported' due to deterministic failure!
    assert res.deterministic_passed is False
    assert res.verification_status in ("unsupported", "partially_supported")
    assert any("co-registration" in c.lower() for c in res.contradictions)
    assert res.is_interpretive_only is True


# Test 12: Malformed LLM output gracefully handled
def test_malformed_llm_output_handling():
    pkg = build_sample_evidence_package()
    mock_client = MagicMock()
    mock_client.is_reachable.return_value = True
    mock_client.chat_completion.return_value = {
        "content": "Not valid JSON at all! Random LLM text...",
        "model": "mock-llm-model"
    }

    verifier = LLMScientificVerifier(client=mock_client)
    res = verifier.verify(evidence_package=pkg)

    # Must be handled gracefully without raising, falling back to safe deterministic result
    assert res.verification_status in ("not_run", "supported")
    assert res.deterministic_passed is True
    assert res.llm_run is False
    assert "schema" in res.summary_explanation.lower() or "not run" in res.summary_explanation.lower()


# Test 13: LLM unavailable / offline path
def test_llm_unavailable_offline_path():
    pkg = build_sample_evidence_package()
    mock_client = MagicMock()
    mock_client.is_reachable.return_value = False

    verifier = LLMScientificVerifier(client=mock_client)
    res = verifier.verify(evidence_package=pkg)

    assert res.verification_status == "not_run"
    assert res.deterministic_passed is True
    assert res.llm_run is False
    assert "offline" in res.summary_explanation.lower() or "unreachable" in res.summary_explanation.lower()


# Test 14: LLM timeout path
def test_llm_timeout_path():
    pkg = build_sample_evidence_package()
    mock_client = MagicMock()
    mock_client.is_reachable.return_value = True
    mock_client.chat_completion.side_effect = LMStudioConnectionError("LM Studio request timed out after 30s.")

    verifier = LLMScientificVerifier(client=mock_client)
    res = verifier.verify(evidence_package=pkg)

    assert res.verification_status == "not_run"
    assert res.deterministic_passed is True
    assert res.llm_run is False
    assert "timed out" in res.summary_explanation.lower() or "could not be completed" in res.summary_explanation.lower()


# Test 15: Preservation of existing analysis response fields and endpoint integration
def test_preservation_of_analysis_response_fields():
    from app.tests.test_optical_sar_cross_sensor import create_geotiff_b64
    opt_b64 = create_geotiff_b64(bounds=(77.5, 12.9, 77.6, 13.0))
    sar_b64 = create_geotiff_b64(bounds=(77.5, 12.9, 77.6, 13.0), tags={"POLARIZATION": "VV"})

    with patch("app.chatbot.lmstudio_client.LMStudioClient.is_reachable", return_value=False):
        payload = {
            "images": [
                {
                    "data": opt_b64,
                    "mimeType": "image/tiff",
                    "filename": "s2.tif",
                    "role": "optical",
                    "provenance": {
                        "modality": "OPTICAL",
                        "sensor": "Sentinel-2 L2A",
                        "datetime": "2024-02-01T10:00:00Z"
                    }
                },
                {
                    "data": sar_b64,
                    "mimeType": "image/tiff",
                    "filename": "s1.tif",
                    "role": "sar",
                    "provenance": {
                        "modality": "SAR",
                        "sensor": "Sentinel-1 GRD",
                        "datetime": "2024-02-05T10:00:00Z"
                    }
                }
            ],
            "query": "Joint optical and SAR assessment",
            "mode": "optical_sar"
        }
        resp = client.post("/api/v1/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Check existing fields preserved
        for field in [
            "id", "task", "mode", "answer", "confidence", "confidence_label",
            "models", "implementation_status", "evidence", "trace",
            "metadata", "confidence_breakdown", "conflict_info", "execution_time_ms", "created_at"
        ]:
            assert field in data, f"Missing field: {field}"

        # Check additive Stage 7 verification field
        assert "verification" in data
        verif = data["verification"]
        assert verif is not None
        assert verif["verification_status"] in ("not_run", "supported", "unsupported", "partially_supported")
        assert "deterministic_passed" in verif
        assert "is_interpretive_only" in verif
        assert verif["is_interpretive_only"] is True


# Test 16: Dedicated /analyze/verify endpoint
def test_dedicated_verify_endpoint():
    pkg = build_sample_evidence_package()
    with patch("app.chatbot.lmstudio_client.LMStudioClient.is_reachable", return_value=False):
        pkg_dict = pkg.model_dump() if hasattr(pkg, "model_dump") else pkg.dict()
        resp = client.post("/api/v1/analyze/verify", json={"evidence_package": pkg_dict})
        assert resp.status_code == 200
        data = resp.json()
        assert data["verification_status"] == "not_run"
        assert data["deterministic_passed"] is True
        assert data["is_interpretive_only"] is True
        assert len(data["claims"]) >= 0


# Test 17: Dedicated /analyze/verify endpoint rejecting empty request
def test_dedicated_verify_endpoint_rejects_empty():
    resp = client.post("/api/v1/analyze/verify", json={})
    assert resp.status_code == 400
    assert "Either 'evidence_package' or 'analysis_response' must be provided." in resp.json()["detail"]
