import os
import pytest
import numpy as np
import torch
from unittest.mock import patch

from app.models.change_detection.change_detector import (
    ChangeDetectionProvider, 
    PixelDifferenceChangeBaseline,
    SiameseRSChangeDetector
)
from app.models.change_detection.bit_model import BitemporalImageTransformer
from app.remote_sensing.registration import align_image_pair
from app.remote_sensing.validation import InputValidator


def test_1_provider_initialization_without_weights():
    """Test 1: Provider initializes correctly without weights mounted."""
    siamese = SiameseRSChangeDetector(checkpoint_path="non_existent_weights.pth")
    assert siamese.model_name == "BIT-CD"
    assert siamese._is_loaded is False
    assert siamese.is_remote_sensing_adapted is True
    
    provider = ChangeDetectionProvider()
    assert provider.baseline.model_name == "PixelDifferenceChangeBaseline"
    assert provider.deep_detector.model_name == "BIT-CD"


def test_2_missing_checkpoint_reports_unavailable():
    """Test 2: Missing checkpoint correctly reports unavailable without crashing."""
    siamese = SiameseRSChangeDetector(checkpoint_path="definitely_missing_bit.pth")
    assert siamese.is_available is False
    loaded = siamese.load()
    assert loaded is False
    assert siamese.implementation_status == "unavailable"


def test_3_missing_checkpoint_triggers_fallback():
    """Test 3: Missing neural checkpoint cleanly triggers fallback to deterministic baseline."""
    provider = ChangeDetectionProvider()
    # Force checkpoint to missing path
    provider.deep_detector.checkpoint_path = "non_existent_bit.pth"
    
    t1 = np.full((64, 64, 3), 100, dtype=np.uint8)
    t2 = np.full((64, 64, 3), 100, dtype=np.uint8)
    t2[10:30, 10:30] = 220  # Artificial change block

    res = provider.predict([t1, t2], query="Detect change between T1 and T2.")
    assert res["fallback_used"] is True
    assert res["primary_model"] == "BIT-CD"
    assert res["actual_model_used"] == "PixelDifferenceChangeBaseline"
    assert res["model_status"] == "checkpoint_not_found"
    assert res["implementation_status"] == "baseline"
    assert res["change_detected"] is True


def test_4_no_fabricated_confidence():
    """Test 4: No fabricated probability or confidence score is produced."""
    provider = ChangeDetectionProvider()
    t1 = np.full((50, 50, 3), 80, dtype=np.uint8)
    t2 = np.full((50, 50, 3), 80, dtype=np.uint8)
    res = provider.predict([t1, t2])
    
    # Confidence must strictly be None for uncalibrated outputs
    assert res.get("confidence") is None
    assert "Not available" in res.get("confidence_label", "")


def test_5_model_provenance_identifies_actual_model():
    """Test 5: Model provenance correctly identifies primary and actual models used in both neural and fallback states."""
    provider = ChangeDetectionProvider()
    t1 = np.full((50, 50, 3), 80, dtype=np.uint8)
    t2 = np.full((50, 50, 3), 80, dtype=np.uint8)
    res = provider.predict([t1, t2])

    assert "model_provenance" in res
    prov = res["model_provenance"]
    assert prov["primary_model"] == "BIT-CD"
    assert "preprocessing" in prov

    if provider.deep_detector.is_available and not res["fallback_used"]:
        # Physical checkpoint is mounted and genuinely executed
        assert prov["actual_model_used"] == "BIT-CD"
        assert prov["fallback_used"] is False
        assert prov["checkpoint_status"] == "loaded"
        assert res["model_status"] == "loaded"
        assert res["implementation_status"] == "real_model"
        assert res.get("confidence") is None  # Uncalibrated neural logits
    else:
        # Deterministic baseline fallback state
        assert prov["actual_model_used"] == "PixelDifferenceChangeBaseline"
        assert prov["fallback_used"] is True
        assert prov["checkpoint_status"] in ["checkpoint_not_found", "baseline"]
        assert res["model_status"] in ["checkpoint_not_found", "baseline"]
        assert res["implementation_status"] == "baseline"
        assert res.get("confidence") is None

    # Explicitly test deterministic fallback provenance when weights are forced absent
    fb_provider = ChangeDetectionProvider()
    fb_provider.deep_detector.checkpoint_path = "non_existent_bit_weights.pth"
    fb_res = fb_provider.predict([t1, t2])
    fb_prov = fb_res["model_provenance"]
    assert fb_prov["primary_model"] == "BIT-CD"
    assert fb_prov["actual_model_used"] == "PixelDifferenceChangeBaseline"
    assert fb_prov["fallback_used"] is True
    assert fb_prov["checkpoint_status"] == "checkpoint_not_found"
    assert fb_res["model_status"] == "checkpoint_not_found"
    assert fb_res["implementation_status"] == "baseline"
    assert fb_res.get("confidence") is None


def test_6_t1_t2_preprocessing_produces_valid_tensors():
    """Test 6: T1/T2 preprocessing produces valid 4D PyTorch float32 tensors."""
    siamese = SiameseRSChangeDetector()
    dev = torch.device("cpu")

    # Case A: Standard 3-band RGB
    arr_3band = np.full((64, 64, 3), 128, dtype=np.uint8)
    tensor_a, meta_a = siamese._prepare_tensor(arr_3band, dev)
    assert tensor_a.shape == (1, 3, 64, 64)
    assert tensor_a.dtype == torch.float32
    assert meta_a["input_band_mapping"] == [1, 2, 3]

    # Case B: 1-band grayscale/SAR raster
    arr_1band = np.full((40, 40), 90, dtype=np.uint8)
    tensor_b, meta_b = siamese._prepare_tensor(arr_1band, dev)
    assert tensor_b.shape == (1, 3, 40, 40)
    assert tensor_b.dtype == torch.float32
    assert meta_b["input_band_mapping"] == [1, 1, 1]


def test_7_invalid_spatial_compatibility_handling():
    """Test 7: Spatial validation handles mismatched CRS and bounds without crashing."""
    meta_a = {
        "crs": "EPSG:32644",
        "crs_display": "EPSG:32644",
        "bounds": [400000.0, 1400000.0, 401000.0, 1401000.0],
        "width": 64,
        "height": 64
    }
    meta_b = {
        "crs": "EPSG:32643",
        "crs_display": "EPSG:32643",
        "bounds": [200000.0, 2000000.0, 201000.0, 2001000.0],
        "width": 64,
        "height": 64
    }
    validation = InputValidator.validate_dual_scenes(meta_a, meta_b, mode="bitemporal")
    assert validation["crs_compatible"] is False
    assert validation["overlap_detected"] is False

    # Ensure baseline processes without unhandled exception while recording registration status
    baseline = PixelDifferenceChangeBaseline()
    t1 = np.full((64, 64, 3), 50, dtype=np.uint8)
    t2 = np.full((64, 64, 3), 50, dtype=np.uint8)
    res = baseline.predict([t1, t2], metadata={"primary": meta_a, "secondary": meta_b})
    assert res["registration_info"]["co_registered"] is False


def test_8_synthetic_change_statistics():
    """Test 8: Synthetic change array computes accurate spatial metrics."""
    baseline = PixelDifferenceChangeBaseline(difference_threshold=30.0)
    h, w = 100, 100
    t1 = np.full((h, w, 3), 50, dtype=np.uint8)
    t2 = np.full((h, w, 3), 50, dtype=np.uint8)
    # 20x20 block = 400 pixels = 4.00% of 10,000 pixels
    t2[10:30, 10:30] = 200

    res = baseline.predict([t1, t2])
    assert res["change_detected"] is True
    assert res["changed_pixels"] == 400
    assert res["total_pixels"] == 10000
    assert abs(res["changed_area_percent"] - 4.00) < 0.01


def test_9_neural_bit_forward_pass_execution():
    """Test 9: Real BIT-CD neural forward pass executes with valid PyTorch weights."""
    detector = SiameseRSChangeDetector()
    # Instantiate real BIT-CD model directly in eval mode
    bit_net = BitemporalImageTransformer(
        in_channels=3, 
        num_classes=2, 
        token_len=4, 
        token_dim=32, 
        num_decoder_layers=2
    )
    bit_net.eval()
    detector._model = bit_net
    detector._is_loaded = True
    detector.implementation_status = "real_model"

    t1 = np.full((64, 64, 3), 70, dtype=np.uint8)
    t2 = np.full((64, 64, 3), 70, dtype=np.uint8)
    t2[15:35, 15:35] = 210

    res = detector.predict([t1, t2], query="Detect change with neural BIT.")
    assert res["model_type"] == "neural_model"
    assert res["primary_model"] == "BIT-CD"
    assert res["actual_model_used"] == "BIT-CD"
    assert res["fallback_used"] is False
    assert res["model_status"] == "loaded"
    assert res["implementation_status"] == "real_model"
    assert res["confidence"] is None  # Zero uncalibrated fake confidence
    assert len(res["evidence"]) == 4
    assert res["evidence"][0]["type"] == "change_map"
    assert res["evidence"][1]["type"] == "overlay"
    assert "input_band_mapping" in res["model_provenance"]


def test_10_api_schema_backward_compatibility():
    """Test 10: API response maintains full schema backward compatibility."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.remote_sensing.preprocessing import convert_array_to_base64_png

    client = TestClient(app)
    t1 = np.full((64, 64, 3), 60, dtype=np.uint8)
    t2 = np.full((64, 64, 3), 60, dtype=np.uint8)
    t2[10:30, 10:30] = 200
    b64_t1 = convert_array_to_base64_png(t1)
    b64_t2 = convert_array_to_base64_png(t2)

    payload = {
        "query": "Detect and map all changed structures between date 1 and date 2.",
        "mode": "bitemporal",
        "images": [
            {"data": b64_t1, "filename": "t1.png", "role": "primary"},
            {"data": b64_t2, "filename": "t2.png", "role": "secondary"}
        ]
    }
    resp = client.post("/api/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["task"] == "change_detection"
    assert "primary_model" in data
    assert "actual_model_used" in data
    assert "fallback_used" in data
    assert "model_status" in data
    assert "model_provenance" in data
    assert "trace" in data
    assert len(data["evidence"]) >= 3


def test_11_physical_checkpoint_conditional():
    """Test 11: Real physical checkpoint integration test (runs only when checkpoint exists on disk)."""
    ckpt_path = os.getenv("CHANGE_MODEL_CHECKPOINT_PATH") or os.getenv("RS_CHANGE_CHECKPOINT_PATH")
    if not ckpt_path or not os.path.exists(ckpt_path):
        pytest.skip("Physical BIT-CD checkpoint not mounted; skipping physical weights test.")

    detector = SiameseRSChangeDetector(checkpoint_path=ckpt_path)
    assert detector.load() is True
    t1 = np.full((64, 64, 3), 100, dtype=np.uint8)
    t2 = np.full((64, 64, 3), 100, dtype=np.uint8)
    res = detector.predict([t1, t2])
    assert res["actual_model_used"] == "BIT-CD"
    assert res["fallback_used"] is False
