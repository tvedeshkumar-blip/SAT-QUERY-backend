import sys
import os
import io
import base64
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.main import app

client = TestClient(app)

def create_dummy_base64_image(width=100, height=100, color=(100, 150, 200)):
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "models" in data
    for m in data["models"]:
        assert m["status"] in ("baseline", "loaded", "unavailable", "demo")

def test_models_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 6

def test_single_image_vqa():
    img_b64 = create_dummy_base64_image()
    payload = {
        "images": [{"data": img_b64, "mimeType": "image/png", "filename": "scene.png"}],
        "query": "What type of land cover dominates this image?",
        "mode": "single"
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["task"] in ["vqa", "captioning"]
    assert "answer" in data
    assert "trace" in data
    assert data["implementation_status"] in ("baseline", "demo", "production_model")

def test_bitemporal_change_detection():
    img_a = create_dummy_base64_image(color=(50, 100, 150))
    img_b = create_dummy_base64_image(color=(200, 100, 50))
    payload = {
        "images": [
            {"data": img_a, "mimeType": "image/png", "filename": "earlier.png", "role": "primary"},
            {"data": img_b, "mimeType": "image/png", "filename": "later.png", "role": "secondary"}
        ],
        "query": "What changed between these two dates?",
        "mode": "bitemporal"
    }
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["task"] in ["change_vqa", "change_detection"]
    assert len(data["evidence"]) >= 1

def test_evaluation_endpoint():
    response = client.get("/api/v1/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ("not_evaluated", "completed")
