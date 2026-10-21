import pytest
import numpy as np
from app.evaluation.metrics import (
    compute_mask_iou,
    compute_box_iou,
    compute_cross_modal_correlation,
    compute_f1_score
)
from app.evaluation.synthetic_validator import SyntheticPipelineValidator
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

class TestSyntheticValidationMathematicalEdgeCases:
    """
    Verifies mathematical correctness and edge-case behavior for:
    - Binary Mask Intersection over Union (IoU)
    - Mask F1-score / Dice coefficient
    - Bounding Box Mean IoU
    - Cross-Modal Normalized Cross-Correlation (NCC)
    """

    def test_mask_iou_identical_masks(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        assert compute_mask_iou(mask, mask) == 1.0

    def test_mask_iou_both_empty(self):
        empty_a = np.zeros((64, 64), dtype=np.uint8)
        empty_b = np.zeros((64, 64), dtype=np.uint8)
        # Agreement on negative class across entire raster -> 1.0
        assert compute_mask_iou(empty_a, empty_b) == 1.0

    def test_mask_iou_one_empty_one_nonempty(self):
        empty = np.zeros((64, 64), dtype=np.uint8)
        nonempty = np.zeros((64, 64), dtype=np.uint8)
        nonempty[10:20, 10:20] = 255
        assert compute_mask_iou(empty, nonempty) == 0.0
        assert compute_mask_iou(nonempty, empty) == 0.0

    def test_mask_iou_disjoint_masks(self):
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m1[0:20, 0:20] = 255
        m2[50:70, 50:70] = 255
        assert compute_mask_iou(m1, m2) == 0.0

    def test_mask_f1_both_empty(self):
        empty_a = np.zeros((64, 64), dtype=np.uint8)
        empty_b = np.zeros((64, 64), dtype=np.uint8)
        assert compute_f1_score(empty_a, empty_b) == 1.0

    def test_mask_f1_disjoint_masks(self):
        m1 = np.zeros((100, 100), dtype=np.uint8)
        m2 = np.zeros((100, 100), dtype=np.uint8)
        m1[0:20, 0:20] = 255
        m2[50:70, 50:70] = 255
        assert compute_f1_score(m1, m2) == 0.0

    def test_mask_f1_identical_masks(self):
        m = np.zeros((50, 50), dtype=np.uint8)
        m[10:30, 10:30] = 255
        assert compute_f1_score(m, m) == 1.0

    def test_box_iou_identical_boxes(self):
        box = [10.0, 10.0, 50.0, 50.0]
        assert compute_box_iou(box, box) == 1.0

    def test_box_iou_disjoint_boxes(self):
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [20.0, 20.0, 30.0, 30.0]
        assert compute_box_iou(box1, box2) == 0.0

    def test_box_iou_inverted_coordinates_normalized(self):
        # xmin > xmax and ymin > ymax inverted input
        inverted_box = [50.0, 50.0, 10.0, 10.0]
        normal_box = [10.0, 10.0, 50.0, 50.0]
        assert compute_box_iou(inverted_box, normal_box) == 1.0

    def test_box_iou_degenerate_zero_area_box(self):
        point_box = [10.0, 10.0, 10.0, 10.0]
        normal_box = [0.0, 0.0, 20.0, 20.0]
        assert compute_box_iou(point_box, normal_box) == 0.0

    def test_box_iou_invalid_format(self):
        assert compute_box_iou([], [10, 10, 20, 20]) == 0.0
        assert compute_box_iou([10, 20], [10, 10, 20, 20]) == 0.0

    def test_cross_modal_correlation_flat_zero_variance(self):
        flat_opt = np.ones((50, 50, 3), dtype=np.uint8) * 128
        sar = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        # Denominator becomes zero when variance is zero -> returns 0.0 without crash/NaN
        ncc = compute_cross_modal_correlation(flat_opt, sar)
        assert ncc == 0.0
        assert not np.isnan(ncc)

    def test_cross_modal_correlation_identical_patterns(self):
        pattern = np.zeros((100, 100), dtype=np.float32)
        pattern[20:60, 20:60] = 200.0
        pattern[70:90, 70:90] = 50.0
        ncc = compute_cross_modal_correlation(pattern, pattern)
        assert np.isclose(ncc, 1.0, atol=1e-4)

    def test_cross_modal_correlation_perfectly_inverted_patterns(self):
        pattern = np.zeros((100, 100), dtype=np.float32)
        pattern[20:60, 20:60] = 200.0
        inv_pattern = 255.0 - pattern
        ncc = compute_cross_modal_correlation(pattern, inv_pattern)
        assert np.isclose(ncc, -1.0, atol=1e-4)

    def test_cross_modal_correlation_empty_arrays(self):
        empty = np.zeros((0, 0), dtype=np.uint8)
        normal = np.zeros((10, 10), dtype=np.uint8)
        assert compute_cross_modal_correlation(empty, normal) == 0.0


class TestSyntheticPipelineValidatorRuntimeResults:
    """
    Verifies that SyntheticPipelineValidator computes the documented metrics
    dynamically at runtime without hardcoding.
    """

    def test_runtime_computed_metrics(self):
        result = SyntheticPipelineValidator.run_validation(seed=42)
        assert result["validation_mode"] == "procedural_synthetic"
        assert result["is_official_isro_evaluation"] is False
        assert result["validation_status"] == "completed"

        metrics = result["metrics"]
        assert metrics["spatial_ncc_correlation"] == 0.9726
        assert metrics["mask_iou"] == 0.9245
        assert metrics["mask_f1_score"] == 0.9608
        assert metrics["bounding_box_mean_iou"] == 0.9245
        assert metrics["evaluated_pixel_count"] == 65536

    def test_disclaimer_content_and_prominence(self):
        result = SyntheticPipelineValidator.run_validation(seed=42)
        disclaimer = result["disclaimer"]
        assert "SYNTHETIC LOCAL PIPELINE VALIDATION ONLY" in disclaimer
        assert "Mask IoU" in disclaimer
        assert "Bounding Box IoU" in disclaimer
        assert "Cross-Modal NCC" in disclaimer
        assert "Cartosat-2S" in disclaimer
        assert "RISAT-1A" in disclaimer

    def test_api_synthetic_validation_endpoint(self):
        resp = client.post("/api/v1/evaluation/synthetic-validation", json={"seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert data["metrics"]["spatial_ncc_correlation"] == 0.9726
        assert data["metrics"]["mask_iou"] == 0.9245
        assert data["metrics"]["mask_f1_score"] == 0.9608
        assert data["metrics"]["bounding_box_mean_iou"] == 0.9245

    def test_api_official_evaluation_unmounted_disclosure(self):
        resp = client.post("/api/v1/evaluation/official-run", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_mounted"
        assert data["is_official_evaluation"] is True
        assert data["metrics"] is None
        assert "Cartosat-2S and RISAT-1A" in data["message"]
