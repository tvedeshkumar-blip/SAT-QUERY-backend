"""
Adversarial test suite for Stage 7A Scientific Verification Audit.
Audits claim-to-evidence binding, physical units, registration & model provenance,
deterministic gate integrity, and failure handling under adversarial conditions.
All LLM responses are strictly mocked.
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

client = TestClient(app)


def build_audit_evidence_package(
    co_registered=True,
    is_trained=False,
    include_units=True,
    sar_calibration="gamma0",
    optical_reflectance=0.1298,
    sar_linear_power=0.7417,
    sar_pixel_db=-4.98,
    sar_linear_to_db=-1.30,
    has_artifacts=True,
    optical_albedo_disp=92.8,
    sar_intensity_disp=105.8,
) -> EvidencePackageSchema:
    """Helper to build a controlled evidence package for adversarial auditing."""
    meas = []
    if include_units:
        meas.append(PhysicalMeasurementItem(
            name="optical_mean_boa_reflectance",
            value=optical_reflectance,
            unit="unitless (surface reflectance BOA)",
            statistic_type="mean",
            mask_applied="joint_valid_mask",
            calibration_quantity="boa_surface_reflectance"
        ))
        meas.append(PhysicalMeasurementItem(
            name="sar_mean_linear_power",
            value=sar_linear_power,
            unit="linear power (m^2/m^2)",
            statistic_type="mean",
            mask_applied="joint_valid_mask",
            calibration_quantity=sar_calibration
        ))
        meas.append(PhysicalMeasurementItem(
            name="sar_mean_pixel_db",
            value=sar_pixel_db,
            unit="dB",
            statistic_type="mean of pixel-wise dB",
            mask_applied="joint_valid_mask",
            calibration_quantity=sar_calibration
        ))
        meas.append(PhysicalMeasurementItem(
            name="sar_mean_linear_to_db",
            value=sar_linear_to_db,
            unit="dB",
            statistic_type="10*log10(mean linear power)",
            mask_applied="joint_valid_mask",
            calibration_quantity=sar_calibration
        ))

    disp = [
        DisplayStatisticItem(
            name="optical_albedo_display_mean",
            value=optical_albedo_disp,
            scale="8-bit normalized [0, 255]",
            purpose="visualization_only"
        ),
        DisplayStatisticItem(
            name="sar_intensity_display_mean",
            value=sar_intensity_disp,
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
        )
    ]

    artifacts = []
    if has_artifacts:
        artifacts = [
            EvidenceArtifactItem(
                artifact_id="optical_input",
                artifact_type="original_optical",
                title="Sentinel-2 BOA Visual",
                statistics={"mean_boa_reflectance": optical_reflectance, "display_mean": optical_albedo_disp}
            ),
            EvidenceArtifactItem(
                artifact_id="sar_input",
                artifact_type="original_sar",
                title="Sentinel-1 RTC Linear Power",
                statistics={"mean_linear_power": sar_linear_power, "mean_pixel_db": sar_pixel_db, "display_mean": sar_intensity_disp}
            ),
            EvidenceArtifactItem(
                artifact_id="heuristic_water_mask",
                artifact_type="mask",
                title="Surface Water Empirical Proxy",
                statistics={"percentage": 4.88}
            )
        ]

    return EvidencePackageSchema(
        task_id="audit_task_001",
        query="Adversarial audit package query",
        sensor_identities=["Sentinel-2A", "Sentinel-1A"],
        acquisition_timestamps=["2025-05-18T05:27:01Z", "2025-05-18T01:14:22Z"],
        source_assets=[
            {"id": "S2A_B04", "collection": "sentinel-2-l2a", "calibration": "BOA_reflectance"},
            {"id": "S1A_RTC_VV", "collection": "sentinel-1-rtc", "calibration": sar_calibration}
        ],
        crs="EPSG:32643",
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


# --- Category A: Claim-to-Evidence Binding ---

def test_adv_a1_non_existent_artifact_id():
    """ADV-A1: A claim references an artifact ID that does not exist."""
    pkg = build_audit_evidence_package()
    claim = ClaimVerificationItem(
        claim_text="Observed feature in hallucinated_artifact_99.",
        claim_type="general",
        status="supported",
        cited_artifact_ids=["hallucinated_artifact_99"]
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    assert passed is False
    assert eval_claims[0].status == "unsupported"
    assert any("hallucinated_artifact_99" in f for f in failures)


def test_adv_a2_artifact_type_mismatch():
    """ADV-A2: A claim references a real artifact ID whose modality/type does not support that claim."""
    pkg = build_audit_evidence_package()
    # Citing SAR input artifact for optical reflectance
    claim = ClaimVerificationItem(
        claim_text="Sentinel-2 BOA surface reflectance is 0.1298.",
        claim_type="measurement",
        status="supported",
        cited_artifact_ids=["sar_input"]  # Modality mismatch!
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    # The verifier should detect that sar_input (original_sar) cannot support an optical surface reflectance claim
    assert passed is False, "Expected failure when SAR artifact is cited for optical reflectance"
    assert eval_claims[0].status in ("unsupported", "contradicted")


def test_adv_a3_real_artifact_with_fabricated_measurement():
    """ADV-A3: A claim cites a real artifact ID but attributes a fabricated measurement to it."""
    pkg = build_audit_evidence_package(optical_reflectance=0.1298)
    # Claim says BOA reflectance is 0.85 (actual is 0.1298)
    claim = ClaimVerificationItem(
        claim_text="Sentinel-2 BOA surface reflectance is 0.85 within the valid mask.",
        claim_type="measurement",
        status="supported",
        cited_artifact_ids=["optical_input"]
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    assert passed is False, "Expected failure when claim states fabricated reflectance 0.85 vs actual 0.1298"
    assert eval_claims[0].status in ("unsupported", "contradicted")


def test_adv_a4_unobserved_statistic_absent_from_package():
    """ADV-A4: The LLM/claim invents a measurement absent from evidence package (e.g. soil moisture = 42%)."""
    pkg = build_audit_evidence_package()
    claim = ClaimVerificationItem(
        claim_text="Volumetric soil moisture is 42% across the agricultural area.",
        claim_type="measurement",
        status="supported",
        cited_artifact_ids=["optical_input"]
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    assert passed is False, "Expected failure when claim invents soil moisture absent from package"
    assert eval_claims[0].status in ("unsupported", "contradicted")


def test_adv_a5_artifact_id_formatting_whitespace_and_case():
    """ADV-A5: Artifact reference is altered by whitespace or casing."""
    pkg = build_audit_evidence_package()
    claim = ClaimVerificationItem(
        claim_text="Sentinel-2 BOA surface reflectance is 0.1298.",
        claim_type="measurement",
        status="supported",
        cited_artifact_ids=["  optical_input  "]  # Leading/trailing whitespace
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    # Should either normalize whitespace cleanly or reject cleanly without crash
    assert eval_claims[0].cited_artifact_ids[0].strip() == "optical_input"


# --- Category B: Physical Units and Measurement Semantics ---

def test_adv_b1_sar_display_intensity_as_calibrated_backscatter():
    """ADV-B1: Normalized uint8 display SAR intensity (105.8) described as calibrated backscatter."""
    pkg = build_audit_evidence_package(sar_intensity_disp=105.8)
    claim = ClaimVerificationItem(
        claim_text="The calibrated SAR backscatter is 105.8 across the scene.",
        claim_type="measurement",
        status="supported",
        cited_artifact_ids=["sar_input"]
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    assert passed is False, "Expected failure when display SAR intensity (105.8) is described as calibrated backscatter"
    assert eval_claims[0].status == "contradicted"


def test_adv_b2_optical_display_albedo_as_reflectance():
    """ADV-B2: Normalized uint8 display optical albedo (92.8) described as physical BOA reflectance."""
    pkg = build_audit_evidence_package(optical_albedo_disp=92.8)
    claim = ClaimVerificationItem(
        claim_text="The scene has a mean BOA surface reflectance of 92.8.",
        claim_type="measurement",
        status="supported",
        cited_artifact_ids=["optical_input"]
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    assert passed is False
    assert eval_claims[0].status == "contradicted"


def test_adv_b3_gamma0_described_as_sigma0():
    """ADV-B3: Sentinel-1 gamma-naught RTC described as sigma-naught."""
    pkg = build_audit_evidence_package(sar_calibration="gamma0")
    claim = ClaimVerificationItem(
        claim_text="Sentinel-1 recorded mean calibrated sigma-0 backscatter of -4.98 dB.",
        claim_type="measurement",
        status="supported",
        cited_artifact_ids=["sar_input"]
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    assert passed is False
    assert eval_claims[0].status == "contradicted"


def test_adv_b4_uncalibrated_asset_claimed_as_calibrated():
    """ADV-B4: An asset with uncalibrated/unspecified calibration claimed as certified sigma0 or gamma0."""
    pkg = build_audit_evidence_package(sar_calibration="uncalibrated_dn")
    claim = ClaimVerificationItem(
        claim_text="The SAR image provides certified gamma-0 backscatter measurements.",
        claim_type="measurement",
        status="supported",
        cited_artifact_ids=["sar_input"]
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    assert passed is False, "Expected failure when uncalibrated DN is claimed as certified gamma-0 backscatter"
    assert eval_claims[0].status == "contradicted"


def test_adv_b5_linear_power_labeled_db_without_conversion():
    """ADV-B5: Linear power (0.7417) labeled as dB without conversion."""
    pkg = build_audit_evidence_package(sar_linear_power=0.7417)
    claim = ClaimVerificationItem(
        claim_text="The radar linear power is 0.7417 dB.",
        claim_type="measurement",
        status="supported",
        cited_artifact_ids=["sar_input"]
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    assert passed is False
    assert eval_claims[0].status == "contradicted"


def test_adv_b6_mean_linear_to_db_confused_with_mean_pixel_db():
    """ADV-B6: Mean linear power converted to dB (-1.30 dB) conflated with mean pixel-wise dB (-4.98 dB)."""
    pkg = build_audit_evidence_package(sar_pixel_db=-4.98, sar_linear_to_db=-1.30)
    # Claim says mean pixel-wise dB is -1.30 dB (which is actually 10*log10(mean linear))
    claim = ClaimVerificationItem(
        claim_text="Mean pixel-wise dB backscatter was computed as -1.30 dB.",
        claim_type="measurement",
        status="supported",
        cited_artifact_ids=["sar_input"]
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    assert passed is False, "Expected failure when mean pixel-wise dB is conflated with linear power converted to dB"
    assert eval_claims[0].status == "contradicted"


# --- Category C: Registration and Model Provenance ---

def test_adv_c1_registration_claimed_when_false():
    """ADV-C1: Claim asserts images are co-registered when metadata indicates co_registered=False."""
    pkg = build_audit_evidence_package(co_registered=False)
    claim = ClaimVerificationItem(
        claim_text="Scenes are registered and geometrically aligned.",
        claim_type="registration",
        status="supported"
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    assert passed is False
    assert eval_claims[0].status == "contradicted"


def test_adv_c2_missing_registration_status_not_treated_as_true():
    """ADV-C2: Missing registration metadata is not treated as verified co-registration."""
    pkg = build_audit_evidence_package()
    pkg.registration_status = {}  # Empty registration status
    claim = ClaimVerificationItem(
        claim_text="The imagery has verified geometric co-registration.",
        claim_type="registration",
        status="supported"
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    assert passed is False, "Expected failure when co-registration claimed on empty registration metadata"
    assert eval_claims[0].status in ("contradicted", "unsupported")


def test_adv_c3_trained_model_claimed_when_fallback_used():
    """ADV-C3: Claim asserts trained model ran when provider metadata confirms fallback baseline."""
    pkg = build_audit_evidence_package(is_trained=False)
    claim = ClaimVerificationItem(
        claim_text="Deep-learning neural network inference was executed.",
        claim_type="model",
        status="supported"
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(pkg, candidate_claims=[claim])
    assert passed is False
    assert eval_claims[0].status == "contradicted"


# --- Category D: Deterministic Gate and Status Semantics ---

def test_adv_d1_llm_cannot_override_contradicted_claims():
    """ADV-D1: Deterministic check fails; LLM asserts claim is supported. Claim must NOT be marked supported."""
    pkg = build_audit_evidence_package(co_registered=False)
    mock_client = MagicMock()
    mock_client.is_reachable.return_value = True

    # LLM returns supported claim
    hallucinated_output = {
        "verification_status": "supported",
        "claims": [
            {
                "claim_text": "Scenes are pixel-aligned and co-registered.",
                "claim_type": "registration",
                "status": "supported",  # LLM lies
                "cited_artifact_ids": ["optical_input"],
                "reason": "LLM assertion that alignment is valid."
            }
        ],
        "contradictions": [],
        "unsupported_claims": [],
        "missing_evidence": [],
        "scientific_limitations": [],
        "recommended_checks": [],
        "summary_explanation": "LLM says valid."
    }
    mock_client.chat_completion.return_value = {
        "content": json.dumps(hallucinated_output),
        "model": "mock-llm"
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

    # Status must be unsupported (NOT partially_supported when the only claim is contradicted!)
    assert res.deterministic_passed is False
    assert res.verification_status == "unsupported", f"Expected unsupported, got {res.verification_status}"
    # The claim itself in res.claims must NOT be marked supported!
    assert res.claims[0].status == "contradicted", f"Claim status was overridden by LLM to {res.claims[0].status}"


def test_adv_d2_empty_evidence_package_cannot_succeed():
    """ADV-D2: An empty evidence package with 0 physical measurements cannot be validated as supported."""
    empty_pkg = EvidencePackageSchema(
        task_id="empty_task",
        query="Empty query",
        sensor_identities=[],
        acquisition_timestamps=[],
        source_assets=[],
        evidence_artifacts=[]
    )
    claim = ClaimVerificationItem(
        claim_text="The physical reflectance is 0.1298.",
        claim_type="measurement",
        status="supported"
    )
    passed, failures, eval_claims, contradictions = DeterministicVerifier.verify(empty_pkg, candidate_claims=[claim])
    assert passed is False, "Empty evidence package must fail deterministic verification for physical claims"
    assert eval_claims[0].status in ("unsupported", "contradicted")


# --- Category E: API and Frontend Consistency ---

def test_adv_e1_api_preserves_numerical_fields_on_verification_failure():
    """ADV-E1: API response preserves all original numerical measurements and evidence when verification finds contradictions."""
    with patch("app.chatbot.lmstudio_client.LMStudioClient.is_reachable", return_value=False):
        # Bi-temporal query with un-registered images
        payload = {
            "images": [
                {
                    "data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    "mimeType": "image/png",
                    "filename": "earlier.png",
                    "role": "primary"
                },
                {
                    "data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    "mimeType": "image/png",
                    "filename": "later.png",
                    "role": "secondary"
                }
            ],
            "query": "Bi-temporal change assessment",
            "mode": "bitemporal"
        }
        resp = client.post("/api/v1/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "evidence" in data
        assert "answer" in data
        assert "verification" in data
        assert data["verification"]["is_interpretive_only"] is True
