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
