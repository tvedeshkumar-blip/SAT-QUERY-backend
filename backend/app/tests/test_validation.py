import pytest
import base64
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.remote_sensing.validation import InputValidator, ValidationError

def test_valid_image_payload():
    dummy_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    b64 = base64.b64encode(dummy_bytes).decode("utf-8")
    size = InputValidator.validate_image_payload(b64, "image/png", "test.png")
    assert size == len(dummy_bytes)

def test_empty_image_payload():
    with pytest.raises(ValidationError, match="empty or invalid"):
        InputValidator.validate_image_payload("", "image/png")

def test_malformed_base64():
    with pytest.raises(ValidationError, match="malformed"):
        InputValidator.validate_image_payload("!!!not_base64!!!", "image/png")

def test_unsupported_mime_type():
    dummy_bytes = b"sample content"
    b64 = base64.b64encode(dummy_bytes).decode("utf-8")
    with pytest.raises(ValidationError, match="Unsupported file format"):
        InputValidator.validate_image_payload(b64, "application/pdf", "document.pdf")

def test_oversized_file():
    # Test oversized file threshold check logic
    with pytest.raises(ValidationError, match="exceeds the maximum allowed limit"):
        # Create a mock that bypasses memory allocation but tests validation logic
        large_bytes = b"0" * 100
        b64 = base64.b64encode(large_bytes).decode("utf-8")
        orig_max = InputValidator.MAX_FILE_SIZE_BYTES
        try:
            InputValidator.MAX_FILE_SIZE_BYTES = 50  # Lower limit for testing
            InputValidator.validate_image_payload(b64, "image/png", "oversized.png")
        finally:
            InputValidator.MAX_FILE_SIZE_BYTES = orig_max

def test_dual_scenes_validation_valid():
    meta_a = {"width": 512, "height": 512, "crs": "EPSG:4326", "bounds": [0, 0, 10, 10], "modality": "OPTICAL"}
    meta_b = {"width": 512, "height": 512, "crs": "EPSG:4326", "bounds": [5, 5, 15, 15], "modality": "OPTICAL"}
    rep = InputValidator.validate_dual_scenes(meta_a, meta_b, mode="bitemporal")
    assert rep["valid"] is True
    assert rep["crs_compatible"] is True
    assert rep["overlap_detected"] is True

def test_dual_scenes_validation_crs_mismatch():
    meta_a = {"width": 512, "height": 512, "crs": "EPSG:32644", "modality": "OPTICAL"}
    meta_b = {"width": 512, "height": 512, "crs": "EPSG:32643", "modality": "OPTICAL"}
    rep = InputValidator.validate_dual_scenes(meta_a, meta_b, mode="bitemporal")
    assert rep["crs_compatible"] is False
    assert len(rep["warnings"]) > 0

def test_dual_scenes_validation_zero_overlap():
    meta_a = {"width": 512, "height": 512, "crs": "EPSG:32644", "bounds": [0, 0, 10, 10]}
    meta_b = {"width": 512, "height": 512, "crs": "EPSG:32644", "bounds": [100, 100, 200, 200]}
    rep = InputValidator.validate_dual_scenes(meta_a, meta_b, mode="bitemporal")
    assert rep["overlap_detected"] is False

def test_dual_scenes_resolution_disparity():
    meta_a = {"width": 512, "height": 512, "resolution": [0.5, 0.5]}
    meta_b = {"width": 512, "height": 512, "resolution": [30.0, 30.0]}
    rep = InputValidator.validate_dual_scenes(meta_a, meta_b)
    assert any("resolution disparity" in w for w in rep["warnings"])
