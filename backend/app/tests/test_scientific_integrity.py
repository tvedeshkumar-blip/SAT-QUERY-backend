import pytest
import os
import io
import base64
import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2
from fastapi.testclient import TestClient

from app.main import app
from app.remote_sensing.geotiff import parse_geotiff_or_image
from app.agent.registry import model_registry
from app.evaluation.synthetic_validator import SyntheticPipelineValidator

client = TestClient(app)

def create_raw_tif_b64(with_crs_tag: bool = False):
    img = Image.new("RGB", (32, 32), color=(50, 100, 150))
    ifd = ImageFileDirectory_v2()
    ifd[33550] = (2.5, 2.5, 0.0) # ModelPixelScaleTag
    ifd[33922] = (0.0, 0.0, 0.0, 432000.0, 1420000.0, 0.0) # ModelTiepointTag
    if with_crs_tag:
        ifd[34737] = b"WGS 84 / UTM zone 44N|EPSG:32644|"
    buf = io.BytesIO()
    img.save(buf, format="TIFF", tiffinfo=ifd)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def test_1_no_hardcoded_benchmark_scores():
    """Rule 1: GET /api/v1/evaluation must not return fabricated scores."""
    res = client.get("/api/v1/evaluation")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("not_evaluated", "completed")
    
    # If not evaluated, results must be empty or contain completed status only
    forbidden_fabricated_scores = ["84.6%", "62.4%", "79.2%", "88.1%"]
    for item in data.get("results", []):
        assert item.get("evaluation_status") != "not_evaluated" or item.get("score") is None
        assert item.get("score") not in forbidden_fabricated_scores

def test_2_no_fake_confidence_values():
    """Rule 2: Models without calibrated confidence must report confidence = None."""
    # Test VQA
    vqa_model = model_registry.get_model("vqa")
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    res = vqa_model.predict([dummy_img], query="Test query")
    assert res["confidence"] is None
    assert "not" in res.get("confidence_label", "").lower()

    # Test Captioning
    cap_model = model_registry.get_model("captioning")
    res_cap = cap_model.predict([dummy_img])
    assert res_cap["confidence"] is None

def test_3_no_guessed_crs():
    """Rule 3: Missing CRS must return None and 'CRS unavailable', never guessing EPSG:32644."""
    # A. Raster with pixel scale & tiepoint but missing CRS tag
    b64_no_crs = create_raw_tif_b64(with_crs_tag=False)
    arr, meta = parse_geotiff_or_image(b64_no_crs, "test_nocrs.tif")
    assert meta["crs"] is None
    assert meta["crs_display"] == "CRS unavailable"

    # B. Plain PNG without any spatial tags
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    arr_png, meta_png = parse_geotiff_or_image(png_b64, "scene.png")
    assert meta_png["crs"] is None
    assert meta_png["crs_display"] == "CRS unavailable"

def test_4_synthetic_evaluation_is_labelled_synthetic():
    """Rule 4: Synthetic validation must explicitly declare itself as synthetic."""
    res = client.post("/api/v1/evaluation/synthetic-validation", json={})
    assert res.status_code == 200
    data = res.json()
    assert data["is_official_isro_evaluation"] is False
    assert "SYNTHETIC" in data["disclaimer"].upper()
    assert data["validation_mode"] in ("procedural_synthetic", "user_provided_pair")

def test_5_unloaded_models_not_reported_as_loaded():
    """Rule 5: Models without initialized tensor weights must NOT report status = 'loaded'."""
    detailed_models = model_registry.list_models_detailed()
    for m in detailed_models:
        if m["implementation_status"] == "baseline":
            assert m["status"] == "baseline"
        elif m["implementation_status"] == "unavailable":
            assert m["status"] == "unavailable"
        elif m["implementation_status"] == "demo":
            assert m["status"] in ("demo", "baseline")

def test_6_missing_weights_do_not_produce_trained_claims():
    """Rule 6: Untrained heuristic models must not claim is_trained=True or is_remote_sensing_adapted=True."""
    detailed_models = model_registry.list_models_detailed()
    for m in detailed_models:
        if m["implementation_status"] in ("baseline", "demo", "unavailable"):
            assert m["is_trained"] is False
            assert m["is_remote_sensing_adapted"] is False
