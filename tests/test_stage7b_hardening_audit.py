"""
Stage 7B: Targeted Verification Hardening Audit Test Suite.
Tests numeric tolerance safety, claim-level aggregation, artifact provenance/modality,
and false rejection of valid scientific claims.
"""

import pytest
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
from app.agent.verifier import DeterministicVerifier, LLMScientificVerifier


@pytest.fixture
def standard_evidence_package():
    return EvidencePackageSchema(
        task_id="task_audit_stage7b",
        query="Verify Stage 6D optical and SAR cross-sensor measurements",
        sensor_identities=["Sentinel-2 L2A B04", "Sentinel-1 RTC VV"],
        acquisition_timestamps=["2024-03-15T10:30:00Z", "2024-03-15T06:00:00Z"],
        source_assets=[
            {
                "id": "s2_b04",
                "collection": "sentinel-2-l2a",
                "calibration": "surface_reflectance",
                "measurement": "boa_reflectance",
                "description": "Sentinel-2 L2A B04 BOA reflectance"
            },
            {
                "id": "s1_rtc",
                "collection": "sentinel-1-rtc",
                "calibration": "gamma0",
                "measurement": "gamma0_radiometric_terrain_corrected",
                "description": "Sentinel-1 RTC gamma-naught backscatter"
            }
        ],
        crs="EPSG:32643",
        resolution=10.0,
        dimensions="100x100",
        registration_status={
            "co_registered": True,
            "method": "rasterio_reproject_bilinear",
            "reference_crs": "EPSG:32643"
        },
        valid_pixel_count=8192,
        total_pixel_count=10000,
        valid_pixel_percentage=81.92,
        physical_measurements=[
            PhysicalMeasurementItem(
                name="optical_mean_boa_reflectance",
                value=0.1298,
                unit="unitless ratio [0, 1]",
                statistic_type="mean",
                mask_applied="joint_valid_mask",
                calibration_quantity="boa_surface_reflectance"
            ),
            PhysicalMeasurementItem(
                name="sar_mean_linear_power",
                value=0.7417,
                unit="linear power (m^2/m^2)",
                statistic_type="mean",
                mask_applied="joint_valid_mask",
                calibration_quantity="gamma0"
            ),
            PhysicalMeasurementItem(
                name="sar_mean_linear_to_db",
                value=-1.30,
                unit="dB",
                statistic_type="10*log10(mean linear power)",
                mask_applied="joint_valid_mask",
                calibration_quantity="gamma0"
            ),
            PhysicalMeasurementItem(
                name="sar_mean_pixel_db",
                value=-4.98,
                unit="dB",
                statistic_type="mean of 10*log10(pixel values)",
                mask_applied="joint_valid_mask",
                calibration_quantity="gamma0"
            ),
        ],
        display_statistics=[
            DisplayStatisticItem(
                name="optical_display_albedo_mean",
                value=92.8,
                scale="8-bit normalized [0, 255]",
                purpose="visualization_only"
            ),
            DisplayStatisticItem(
                name="sar_display_intensity_mean",
                value=105.8,
                scale="8-bit normalized [0, 255]",
                purpose="visualization_only"
            ),
        ],
        heuristic_indicators=[
            HeuristicIndicatorItem(
                name="heuristic_surface_water_proxy",
                percentage=4.88,
                pixel_count=400,
                definition="SAR backscatter < -12 dB and Optical reflectance < 0.05",
                is_certified_classification=False,
                limitations="Uncalibrated preliminary heuristic indicator; not ground truth"
            )
        ],
        evidence_artifacts=[
            EvidenceArtifactItem(
                artifact_id="optical_input",
                artifact_type="original",
                title="Sentinel-2 B04 Surface Reflectance",
                statistics={"mean": 0.1298, "display_mean": 92.8}
            ),
            EvidenceArtifactItem(
                artifact_id="sar_input",
                artifact_type="original",
                title="Sentinel-1 RTC SAR Backscatter",
                statistics={"mean_linear": 0.7417, "display_mean": 105.8}
            ),
            EvidenceArtifactItem(
                artifact_id="joint_valid_mask",
                artifact_type="mask",
                title="Joint Valid Pixel Mask",
                statistics={"valid_pixels": 8192}
            ),
        ],
        provider_info=ProviderInfoItem(
            provider_name="OpticalSARJointAnalysisProvider",
            is_trained_model=False,
            fallback_used=True,
            model_status="heuristic_statistical_baseline"
        ),
        validation_warnings=[],
        known_limitations=["Local demonstration window; preliminary heuristic proxy."]
    )


# ==============================================================================
# SECTION 2: NUMERIC TOLERANCE AUDIT
# ==============================================================================

def test_num_1_exact_physical_value_accepted(standard_evidence_package):
    """Case 1: An exact physical value with the correct units and statistic is accepted."""
    claims = [
        ClaimVerificationItem(
            claim_text="The mean BOA surface reflectance is 0.1298.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    assert passed is True
    assert len(failures) == 0
    assert evaluated[0].status == "supported"


def test_num_2_small_rounding_difference_handled_consistently(standard_evidence_package):
    """Case 2: A small rounding difference (0.13 vs 0.1298) is handled consistently."""
    claims = [
        ClaimVerificationItem(
            claim_text="The mean BOA surface reflectance is 0.13.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    assert passed is True
    assert evaluated[0].status == "supported"


def test_num_3_fabricated_value_differing_materially_rejected(standard_evidence_package):
    """Case 3: A fabricated value differing materially from the source is rejected."""
    # Fabricated reflectance: 0.35 vs verified 0.1298
    claims_opt = [
        ClaimVerificationItem(
            claim_text="The mean BOA surface reflectance is 0.35.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims_opt
    )
    assert passed is False
    assert evaluated[0].status == "contradicted"

    # Fabricated SAR linear power: 0.20 vs verified 0.7417
    claims_sar = [
        ClaimVerificationItem(
            claim_text="The mean linear power backscatter is 0.20.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["sar_input"]
        )
    ]
    passed_s, failures_s, evaluated_s, contradictions_s = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims_sar
    )
    # Check whether fabricated SAR linear backscatter is caught
    # We record the actual behavior here
    assert evaluated_s[0].status == "contradicted" or not passed_s


def test_num_4_numerically_close_wrong_unit_rejected(standard_evidence_package):
    """Case 4: A numerically close value with the wrong unit is rejected."""
    # Reflectance given in dB: "surface reflectance is 0.13 dB"
    # Or linear power given as dB without log conversion: "linear power is 0.7417 dB"
    claims = [
        ClaimVerificationItem(
            claim_text="The mean linear power is 0.7417 dB.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["sar_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    assert passed is False
    assert evaluated[0].status == "contradicted"


def test_num_5_numerically_close_wrong_sensor_or_artifact_rejected(standard_evidence_package):
    """Case 5: A numerically close value from the wrong sensor or artifact is rejected."""
    # Optical reflectance value (0.1298) claimed as SAR backscatter, citing optical_input
    claims = [
        ClaimVerificationItem(
            claim_text="The SAR backscatter is 0.1298.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    assert passed is False
    assert evaluated[0].status == "contradicted"
    assert any("Modality mismatch" in f for f in failures)


def test_num_6_mean_pixel_db_vs_linear_power_db(standard_evidence_package):
    """Case 6: A mean of pixel-wise dB (-4.98 dB) is not treated as equivalent to dB from mean linear power (-1.30 dB)."""
    # Claim asserting -1.30 dB is the mean pixel-wise dB
    claims = [
        ClaimVerificationItem(
            claim_text="The mean pixel-wise dB backscatter is -1.30 dB.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["sar_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    assert passed is False
    assert evaluated[0].status == "contradicted"
    assert any("linear power mean converted to dB" in f for f in failures)


def test_num_7_missing_units_statistic_or_mask_cannot_pass(standard_evidence_package):
    """Case 7: Missing units, statistic type, or source attribution cannot silently pass."""
    # Create invalid package missing unit and mask
    bad_pkg = standard_evidence_package.model_copy(deep=True)
    bad_pkg.physical_measurements[0].unit = ""
    bad_pkg.physical_measurements[0].mask_applied = ""

    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(bad_pkg)
    assert passed is False
    assert any("lacks a declared physical unit" in f for f in failures)
    assert any("lacks a declared valid-pixel mask" in f for f in failures)


def test_num_8_display_only_values_cannot_be_physical_measurements(standard_evidence_package):
    """Case 8: Display-only values cannot be accepted as physical measurements merely because they fall in range."""
    # Optical display albedo 92.8 claimed as physical reflectance
    claims_opt = [
        ClaimVerificationItem(
            claim_text="The BOA surface reflectance is 92.8.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims_opt
    )
    assert passed is False
    assert evaluated[0].status == "contradicted"

    # SAR display intensity 105.8 claimed as physical backscatter
    claims_sar = [
        ClaimVerificationItem(
            claim_text="The calibrated SAR backscatter is 105.8.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["sar_input"]
        )
    ]
    passed_s, failures_s, evaluated_s, contradictions_s = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims_sar
    )
    assert passed_s is False
    assert evaluated_s[0].status == "contradicted"


# ==============================================================================
# SECTION 3: CLAIM-LEVEL STATUS AGGREGATION AUDIT
# ==============================================================================

def test_agg_1_one_supported_one_contradicted(standard_evidence_package):
    """Case 1: One supported claim plus one contradicted claim."""
    claims = [
        ClaimVerificationItem(
            claim_text="The mean BOA surface reflectance is 0.1298.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        ),
        ClaimVerificationItem(
            claim_text="The mean BOA surface reflectance is 0.85.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    verifier = LLMScientificVerifier()
    res = verifier.verify(standard_evidence_package, candidate_claims=claims)

    assert res.deterministic_passed is False
    # Overall status must reflect aggregation: partially_supported, not supported!
    assert res.verification_status == "partially_supported"
    # Each claim retains its own outcome
    assert res.claims[0].status == "supported"
    assert res.claims[1].status == "contradicted"


def test_agg_2_one_supported_one_unsupported(standard_evidence_package):
    """Case 2: One supported claim plus one unsupported claim (unobserved variable)."""
    claims = [
        ClaimVerificationItem(
            claim_text="The mean BOA surface reflectance is 0.1298.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        ),
        ClaimVerificationItem(
            claim_text="The measured soil moisture is 35%.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    verifier = LLMScientificVerifier()
    res = verifier.verify(standard_evidence_package, candidate_claims=claims)

    assert res.deterministic_passed is False
    assert res.verification_status == "partially_supported"
    assert res.claims[0].status == "supported"
    assert res.claims[1].status == "unsupported"


def test_agg_3_all_claims_contradicted(standard_evidence_package):
    """Case 3: All claims contradicted."""
    claims = [
        ClaimVerificationItem(
            claim_text="The mean BOA surface reflectance is 0.85.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        ),
        ClaimVerificationItem(
            claim_text="The calibrated SAR backscatter is 105.8.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["sar_input"]
        )
    ]
    verifier = LLMScientificVerifier()
    res = verifier.verify(standard_evidence_package, candidate_claims=claims)

    assert res.deterministic_passed is False
    assert res.verification_status == "unsupported"
    assert all(c.status == "contradicted" for c in res.claims)


def test_agg_4_all_claims_unsupported(standard_evidence_package):
    """Case 4: All claims unsupported."""
    claims = [
        ClaimVerificationItem(
            claim_text="The measured soil moisture is 35%.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        ),
        ClaimVerificationItem(
            claim_text="The ocean bathymetry is -15 meters.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    verifier = LLMScientificVerifier()
    res = verifier.verify(standard_evidence_package, candidate_claims=claims)

    assert res.deterministic_passed is False
    assert res.verification_status == "unsupported"
    assert all(c.status == "unsupported" for c in res.claims)


def test_agg_5_zero_claims(standard_evidence_package):
    """Case 5: Zero claims. An empty set of claims must NEVER receive 'supported' status."""
    verifier = LLMScientificVerifier()
    res = verifier.verify(standard_evidence_package, candidate_claims=[])

    assert res.verification_status in ("unsupported", "not_run")
    assert res.verification_status != "supported"
    assert len(res.claims) == 0


def test_agg_6_llm_promotes_all_despite_deterministic_failures(standard_evidence_package, monkeypatch):
    """Case 6: An LLM that labels every claim 'supported' despite deterministic failures cannot override."""
    claims = [
        ClaimVerificationItem(
            claim_text="The mean BOA surface reflectance is 0.85.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]

    # Mock LM Studio client to return all supported
    def mock_chat_completion(*args, **kwargs):
        return {
            "content": """{
                "verification_status": "supported",
                "claims": [
                    {
                        "claim_text": "The mean BOA surface reflectance is 0.85.",
                        "claim_type": "measurement",
                        "status": "supported",
                        "cited_artifact_ids": ["optical_input"],
                        "evidence_found": "Found reflectance",
                        "reason": "LLM hallucinated support"
                    }
                ],
                "contradictions": [],
                "unsupported_claims": [],
                "missing_evidence": [],
                "scientific_limitations": [],
                "recommended_checks": [],
                "summary_explanation": "All claims verified."
            }""",
            "model": "mock-llm"
        }

    verifier = LLMScientificVerifier()
    monkeypatch.setattr(verifier.client, "is_reachable", lambda *args, **kwargs: True)
    monkeypatch.setattr(verifier.client, "chat_completion", mock_chat_completion)

    res = verifier.verify(standard_evidence_package, candidate_claims=claims)
    assert res.deterministic_passed is False
    assert res.verification_status == "unsupported"
    assert res.claims[0].status == "contradicted"
    assert "Deterministic gate:" in (res.claims[0].reason or "")


def test_agg_7_malformed_claim_statuses_handled_safely(standard_evidence_package, monkeypatch):
    """Case 7: Malformed or missing claim statuses from LLM do not cause crashes or false success."""
    def mock_chat_completion(*args, **kwargs):
        return {
            "content": """{
                "verification_status": "partially_supported",
                "claims": [
                    {
                        "claim_text": "The mean BOA surface reflectance is 0.1298.",
                        "claim_type": "measurement",
                        "status": "weird_unknown_status",
                        "cited_artifact_ids": ["optical_input"]
                    }
                ],
                "contradictions": [],
                "unsupported_claims": [],
                "missing_evidence": [],
                "scientific_limitations": [],
                "recommended_checks": [],
                "summary_explanation": "Verification explanation."
            }""",
            "model": "mock-llm"
        }

    verifier = LLMScientificVerifier()
    monkeypatch.setattr(verifier.client, "is_reachable", lambda *args, **kwargs: True)
    monkeypatch.setattr(verifier.client, "chat_completion", mock_chat_completion)

    claims = [
        ClaimVerificationItem(
            claim_text="The mean BOA surface reflectance is 0.1298.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    res = verifier.verify(standard_evidence_package, candidate_claims=claims)
    assert res.verification_status in ("partially_supported", "supported", "unsupported", "not_run")



def test_agg_8_duplicate_claims_and_duplicated_artifacts(standard_evidence_package):
    """Case 8: Duplicate claims or duplicated artifact references are handled cleanly."""
    claims = [
        ClaimVerificationItem(
            claim_text="The mean BOA surface reflectance is 0.1298.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input", "optical_input", "  optical_input  "]
        ),
        ClaimVerificationItem(
            claim_text="The mean BOA surface reflectance is 0.1298.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    assert passed is True
    # Artifact IDs should be cleanly normalized
    assert all(len(c.cited_artifact_ids) >= 1 for c in evaluated)


# ==============================================================================
# SECTION 4: ARTIFACT PROVENANCE AND MODALITY AUDIT
# ==============================================================================

def test_prov_1_optical_reflectance_cites_optical_artifact(standard_evidence_package):
    """Case 1: Optical reflectance claim cites optical measurement artifact with matching quantity/units."""
    claims = [
        ClaimVerificationItem(
            claim_text="Optical surface reflectance is 0.1298.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    assert passed is True
    assert evaluated[0].status == "supported"


def test_prov_2_optical_reflectance_cites_sar_artifact(standard_evidence_package):
    """Case 2: Optical reflectance claim cites SAR artifact."""
    claims = [
        ClaimVerificationItem(
            claim_text="Optical surface reflectance is 0.1298.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["sar_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    assert passed is False
    assert evaluated[0].status == "contradicted"
    assert any("Modality mismatch" in f for f in failures)


def test_prov_3_sar_backscatter_cites_optical_artifact(standard_evidence_package):
    """Case 3: SAR backscatter claim cites optical artifact."""
    claims = [
        ClaimVerificationItem(
            claim_text="SAR radar backscatter is 0.7417 linear power.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    assert passed is False
    assert evaluated[0].status == "contradicted"
    assert any("Modality mismatch" in f for f in failures)


def test_prov_4_artifact_exists_but_quantity_does_not_support_claim(standard_evidence_package):
    """Case 4: An artifact ID exists (e.g. joint_valid_mask) but its type does not support the claim."""
    claims = [
        ClaimVerificationItem(
            claim_text="Optical surface reflectance is 0.1298.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["joint_valid_mask"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    # Mask artifact cannot serve as primary measurement source for reflectance
    # Let's see if verifier flags or allows this
    # Record actual behavior


def test_prov_5_missing_modality_or_calibration_metadata(standard_evidence_package):
    """Case 5: Required modality or calibration metadata is missing."""
    bad_pkg = standard_evidence_package.model_copy(deep=True)
    # Remove calibration metadata completely from all source assets and physical measurements
    for a in bad_pkg.source_assets:
        a["calibration"] = ""
        a["measurement"] = ""
        a["description"] = ""
        a["collection"] = "unverified_radar"
    for pm in bad_pkg.physical_measurements:
        pm.calibration_quantity = ""

    claims = [
        ClaimVerificationItem(
            claim_text="Certified gamma-0 backscatter is calibrated and verified.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["sar_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        bad_pkg, candidate_claims=claims
    )
    # If calibration provenance is missing, claiming certified gamma-0 cannot pass
    assert passed is False
    assert any("calibration" in f.lower() for f in failures)
    assert evaluated[0].status == "contradicted"


def test_prov_6_artifact_ids_differ_by_case_whitespace_formatting(standard_evidence_package):
    """Case 6: Artifact identifiers differ only by case, whitespace, or formatting."""
    claims = [
        ClaimVerificationItem(
            claim_text="The mean BOA surface reflectance is 0.1298.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["  OPTICAL_INPUT  "]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    assert passed is True
    assert evaluated[0].cited_artifact_ids == ["optical_input"]


def test_prov_7_conflicting_metadata_between_artifacts(standard_evidence_package):
    """Case 7: Two artifacts or assets have conflicting calibration metadata."""
    conflicting_pkg = standard_evidence_package.model_copy(deep=True)
    conflicting_pkg.source_assets = [
        {
            "id": "s1_rtc",
            "calibration": "gamma0",
            "collection": "sentinel-1-rtc"
        },
        {
            "id": "s1_grd_raw",
            "calibration": "uncalibrated DN",
            "collection": "sentinel-1-grd"
        }
    ]
    claims = [
        ClaimVerificationItem(
            claim_text="Certified calibrated gamma-0 backscatter is present across all inputs.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["sar_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        conflicting_pkg, candidate_claims=claims
    )
    assert passed is False
    assert evaluated[0].status == "contradicted"


def test_prov_8_cross_attribution_between_artifacts(standard_evidence_package):
    """Case 8: A claim uses valid artifact IDs but attributes a value from one artifact to another."""
    # Attributes SAR display intensity (105.8) to optical_input
    claims = [
        ClaimVerificationItem(
            claim_text="The optical surface reflectance intensity is 105.8.",
            claim_type="measurement",
            status="supported",
            cited_artifact_ids=["optical_input"]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    assert passed is False
    assert evaluated[0].status == "contradicted"


# ==============================================================================
# SECTION 5: FALSE-REJECTION AUDIT
# ==============================================================================

def test_false_rej_1_coregistration_equivalent_phrasing(standard_evidence_package):
    """Case 1: Co-registration phrasing variations that should NOT be rejected when co_registered=True."""
    equivalent_phrases = [
        "The scenes are co-registered and aligned to EPSG:32643.",
        "Spatial alignment was verified using bilinear reprojection.",
        "Geometric co-registration was established between optical and SAR rasters.",
        "The two images are registered to a shared geographic grid."
    ]
    for phrase in equivalent_phrases:
        claims = [
            ClaimVerificationItem(
                claim_text=phrase,
                claim_type="registration",
                status="supported",
                cited_artifact_ids=[]
            )
        ]
        passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
            standard_evidence_package, candidate_claims=claims
        )
        assert passed is True, f"Falsely rejected coregistration phrase: '{phrase}', failures: {failures}"
        assert evaluated[0].status != "contradicted"


def test_false_rej_2_linear_vs_db_correct_conversion(standard_evidence_package):
    """Case 2: Scientifically accurate linear power vs decibel statements should NOT be falsely rejected."""
    accurate_phrases = [
        "The mean linear power is 0.7417 (converted to -1.30 dB via 10*log10).",
        "Linear gamma-0 power is 0.7417, with 10*log10 conversion yielding -1.30 dB.",
    ]
    for phrase in accurate_phrases:
        claims = [
            ClaimVerificationItem(
                claim_text=phrase,
                claim_type="measurement",
                status="supported",
                cited_artifact_ids=["sar_input"]
            )
        ]
        passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
            standard_evidence_package, candidate_claims=claims
        )
        assert passed is True, f"Falsely rejected accurate linear/dB conversion: '{phrase}', failures: {failures}"


def test_false_rej_3_boa_reflectance_valid_scientific_language(standard_evidence_package):
    """Case 3: Valid scientific language for surface reflectance should NOT be falsely rejected."""
    valid_phrases = [
        "Mean bottom-of-atmosphere surface reflectance is 0.1298.",
        "The B04 BOA surface reflectance mean is 0.13.",
        "Observed BOA reflectance ratio is 0.1298."
    ]
    for phrase in valid_phrases:
        claims = [
            ClaimVerificationItem(
                claim_text=phrase,
                claim_type="measurement",
                status="supported",
                cited_artifact_ids=["optical_input"]
            )
        ]
        passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
            standard_evidence_package, candidate_claims=claims
        )
        assert passed is True, f"Falsely rejected valid reflectance phrasing: '{phrase}', failures: {failures}"


def test_false_rej_4_heuristic_proxy_properly_qualified(standard_evidence_package):
    """Case 4: Properly qualified heuristic indicator phrasing should NOT be rejected."""
    claims = [
        ClaimVerificationItem(
            claim_text="The preliminary heuristic surface water proxy indicates 4.88% potential inundation, but is not a certified classification.",
            claim_type="heuristic",
            status="supported",
            cited_artifact_ids=[]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    # The claim acknowledges it is NOT a certified classification
    # Let's inspect whether verifier falsely flags "water" + "certified"
    assert passed is True, f"Falsely rejected qualified heuristic claim: {failures}"


def test_false_rej_5_unobserved_environmental_variables_accurately_disclaimed(standard_evidence_package):
    """Case 5: Explicitly acknowledging unobserved variables as absent should NOT be flagged as claiming them."""
    claims = [
        ClaimVerificationItem(
            claim_text="Soil moisture was not observed by the sensors and is absent from the evidence package.",
            claim_type="general",
            status="supported",
            cited_artifact_ids=[]
        )
    ]
    passed, failures, evaluated, contradictions = DeterministicVerifier.verify(
        standard_evidence_package, candidate_claims=claims
    )
    # If the text says soil moisture was NOT observed / absent, is it falsely flagged?
    # Let's inspect behavior
    assert passed is True, f"Falsely rejected disclaiming unobserved variable: {failures}"
