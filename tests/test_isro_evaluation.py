import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.evaluation.isro_evaluator import ISROEvaluator
from app.evaluation.synthetic_validator import SyntheticPipelineValidator
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_mask_iou_computation():
    mask_a = np.zeros((50, 50), dtype=np.uint8)
    mask_b = np.zeros((50, 50), dtype=np.uint8)
    
    mask_a[10:40, 10:40] = 255
    mask_b[10:40, 10:40] = 255
    # Perfect overlap -> IoU = 1.0
    iou = ISROEvaluator.compute_mask_iou(mask_a, mask_b)
    assert iou == 1.0

def test_box_iou_computation():
    box1 = [10.0, 10.0, 50.0, 50.0]
    box2 = [10.0, 10.0, 50.0, 50.0]
    iou = ISROEvaluator.compute_box_iou(box1, box2)
    assert iou == 1.0

    # Non-overlapping
    box3 = [60.0, 60.0, 100.0, 100.0]
    iou_zero = ISROEvaluator.compute_box_iou(box1, box3)
    assert iou_zero == 0.0

def test_cross_modal_correlation():
    arr1 = np.ones((20, 20), dtype=np.uint8) * 50
    arr2 = np.ones((20, 20), dtype=np.uint8) * 100
    ncc = ISROEvaluator.compute_cross_modal_correlation(arr1, arr2)
    assert isinstance(ncc, float)

def test_synthetic_validation_endpoint():
    response = client.post("/api/v1/evaluation/synthetic-validation", json={})
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert data["is_official_isro_evaluation"] is False
    assert "SYNTHETIC" in data["disclaimer"].upper()

def test_official_evaluation_endpoint_unmounted():
    response = client.post("/api/v1/evaluation/official-run", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_mounted"
    assert data["is_official_evaluation"] is True
