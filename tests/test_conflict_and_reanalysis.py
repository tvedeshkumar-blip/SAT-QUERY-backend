import pytest
import numpy as np
from app.agent.conflict_detector import ConflictDetector, ConflictCheckResult

def test_grounding_zero_detections_conflict():
    primary_result = {
        "answer": "Detected 0 features.",
        "evidence": [{"id": "ev_box", "type": "boxes", "boxes": []}]
    }
    eval_res = ConflictDetector.evaluate(
        task="grounding",
        query="Locate urban building structures.",
        primary_result=primary_result,
        metadata={}
    )
    assert eval_res.conflict_detected is True
    assert eval_res.conflict_type == "GROUNDING_ZERO_DETECTIONS"
    assert eval_res.reanalysis_tool == "adaptive_spectral_thresholding"

def test_change_zero_delta_conflict():
    primary_result = {
        "answer": "Negligible change detected.",
        "change_area_percentage": 0.0,
        "evidence": []
    }
    eval_res = ConflictDetector.evaluate(
        task="change_detection",
        query="What is the massive expansion in this region?",
        primary_result=primary_result,
        metadata={}
    )
    assert eval_res.conflict_detected is True
    assert eval_res.conflict_type == "CHANGE_ZERO_DELTA"
    assert eval_res.reanalysis_tool == "multi_threshold_spectral_difference"

def test_no_conflict_when_consistent():
    primary_result = {
        "answer": "Identified lake feature.",
        "evidence": [{"id": "ev_box", "type": "boxes", "boxes": [{"xmin": 10, "ymin": 10, "xmax": 30, "ymax": 30}]}]
    }
    eval_res = ConflictDetector.evaluate(
        task="grounding",
        query="Locate lake.",
        primary_result=primary_result,
        metadata={}
    )
    assert eval_res.conflict_detected is False


# ============================================================================
# Optical-SAR Conflict & Contradiction Detection Tests (Step 4C)
# ============================================================================

def test_optical_sar_water_specular_conflict():
    """
    Assert that when an optical observation claims an open water body,
    but co-registered SAR exhibits high diffuse backscatter (> 180) and 0% specular water,
    a physical conflict is detected (OPTICAL_SAR_SPECULAR_WATER_CONTRADICTION).
    """
    primary_result = {
        "answer": "Identified open water reservoir in optical imagery.",
        "water_pct": 0.0,
        "mean_sar_display": 215.0,
        "registration_info": {
            "co_registered": True,
            "alignment_valid": True,
            "is_display_only": False
        }
    }
    eval_res = ConflictDetector.evaluate(
        task="optical_sar",
        query="Locate open water body and reservoir",
        primary_result=primary_result,
        metadata={}
    )
    assert eval_res.conflict_detected is True
    assert eval_res.conflict_type == "OPTICAL_SAR_SPECULAR_WATER_CONTRADICTION"
    assert eval_res.reanalysis_tool == "cross_sensor_polarimetric_decomposition"
    assert "inconsistent with specular reflection" in eval_res.reanalysis_reason


def test_optical_sar_urban_backscatter_conflict():
    """
    Assert that when a claim asserts dense urban buildings,
    but co-registered SAR and optical evidence show > 75% specular water,
    an urban double-bounce contradiction is detected.
    """
    primary_result = {
        "answer": "Dense urban commercial center detected.",
        "water_pct": 85.0,
        "mean_sar_display": 30.0,
        "registration_info": {
            "co_registered": True,
            "alignment_valid": True,
            "is_display_only": False
        }
    }
    eval_res = ConflictDetector.evaluate(
        task="optical_sar",
        query="Detect urban buildings and commercial structures",
        primary_result=primary_result,
        metadata={}
    )
    assert eval_res.conflict_detected is True
    assert eval_res.conflict_type == "OPTICAL_SAR_URBAN_BACKSCATTER_CONTRADICTION"
    assert eval_res.reanalysis_tool == "urban_coherence_analysis"


def test_optical_sar_modality_non_interchangeability_conflict():
    """
    Assert that claims attempting to equate optical reflectance with radar backscatter
    trigger MODALITY_NON_INTERCHANGEABILITY_VIOLATION.
    """
    primary_result = {
        "answer": "Optical reflectance equals SAR backscatter across all bands.",
        "water_pct": 40.0,
        "mean_sar_display": 100.0,
        "registration_info": {
            "co_registered": True,
            "alignment_valid": True,
            "is_display_only": False
        }
    }
    eval_res = ConflictDetector.evaluate(
        task="optical_sar",
        query="Analyze cross-sensor alignment",
        primary_result=primary_result,
        metadata={}
    )
    assert eval_res.conflict_detected is True
    assert eval_res.conflict_type == "MODALITY_NON_INTERCHANGEABILITY_VIOLATION"
    assert eval_res.reanalysis_tool == "physical_modality_separation"


def test_optical_sar_no_conflict_when_physically_consistent():
    """
    Assert that when optical and SAR physical measurements are consistent
    (open water claim with low specular radar backscatter and high water pct),
    no conflict is detected.
    """
    primary_result = {
        "answer": "Verified open water reservoir.",
        "water_pct": 65.0,
        "mean_sar_display": 35.0,
        "registration_info": {
            "co_registered": True,
            "alignment_valid": True,
            "is_display_only": False
        }
    }
    eval_res = ConflictDetector.evaluate(
        task="optical_sar",
        query="Locate open water lake",
        primary_result=primary_result,
        metadata={}
    )
    assert eval_res.conflict_detected is False
    assert eval_res.status == "no_conflict"


def test_optical_sar_insufficient_evidence_unaligned():
    """
    Assert that rasters lacking authenticated geospatial co-registration (display-only/unaligned)
    do NOT trigger false conflicts; they return status='unsupported' with uncertainty reason.
    """
    primary_result = {
        "answer": "Display-only preview composite.",
        "water_pct": 0.0,
        "mean_sar_display": 220.0,
        "registration_info": {
            "co_registered": False,
            "alignment_valid": False,
            "is_display_only": True
        }
    }
    eval_res = ConflictDetector.evaluate(
        task="optical_sar",
        query="Locate open water body and reservoir",
        primary_result=primary_result,
        metadata={}
    )
    assert eval_res.conflict_detected is False
    assert eval_res.status == "unsupported"
    assert "lack authenticated geospatial co-registration" in eval_res.uncertainty_reason


def test_optical_sar_insufficient_evidence_zero_valid_pixels():
    """
    Assert that zero valid clear-sky pixels produce status='insufficient_evidence'
    rather than a false conflict or verification.
    """
    primary_result = {
        "answer": "Zero valid pixels after cloud masking.",
        "valid_pixel_count": 0,
        "registration_info": {
            "co_registered": True,
            "alignment_valid": True,
            "is_display_only": False
        }
    }
    eval_res = ConflictDetector.evaluate(
        task="optical_sar",
        query="Locate open water lake",
        primary_result=primary_result,
        metadata={}
    )
    assert eval_res.conflict_detected is False
    assert eval_res.status == "insufficient_evidence"
    assert "Zero valid clear-sky pixels" in eval_res.uncertainty_reason


def test_optical_sar_unsupported_propagation():
    """
    Assert that when underlying fusion status is unsupported, ConflictDetector
    propagates status='unsupported' without claiming conflict or success.
    """
    primary_result = {
        "status": "unsupported",
        "answer": "Optical + SAR joint analysis rejected: excessive resolution disparity.",
        "registration_info": {
            "co_registered": True,
            "alignment_valid": True,
            "is_display_only": False
        }
    }
    eval_res = ConflictDetector.evaluate(
        task="optical_sar",
        query="Fuse optical and SAR",
        primary_result=primary_result,
        metadata={}
    )
    assert eval_res.conflict_detected is False
    assert eval_res.status == "unsupported"
    assert "excessive resolution disparity" in eval_res.uncertainty_reason


def test_candidate_claim_contradicted_by_evidence():
    """
    Assert that a candidate claim flagged as 'contradicted' triggers
    CLAIM_CONTRADICTED_BY_EVIDENCE in ConflictDetector.
    """
    primary_result = {
        "answer": "Optical analysis completed.",
        "evidence": []
    }
    claims = [
        {
            "claim_text": "The observed Delta-NDVI is +3.5.",
            "status": "contradicted",
            "reason": "Delta-NDVI +3.5 exceeds mathematical limits [-2.0, 2.0]."
        }
    ]
    eval_res = ConflictDetector.evaluate(
        task="temporal_optical_change",
        query="Compute NDVI change",
        primary_result=primary_result,
        metadata={},
        candidate_claims=claims
    )
    assert eval_res.conflict_detected is True
    assert eval_res.conflict_type == "CLAIM_CONTRADICTED_BY_EVIDENCE"
    assert "exceeds mathematical limits" in eval_res.reanalysis_reason
