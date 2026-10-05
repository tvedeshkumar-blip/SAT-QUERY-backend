import pytest
import numpy as np
import io
import base64
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.remote_sensing.bands import (
    compute_ndvi, 
    compute_ndwi, 
    compute_false_color_cir, 
    compute_sar_decibel_and_filter
)
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_compute_ndvi():
    arr = np.zeros((50, 50, 4), dtype=np.uint8)
    arr[:, :, 3] = 200  # High NIR
    arr[:, :, 0] = 50   # Low Red
    ndvi_f, ndvi_col = compute_ndvi(arr, nir_band_idx=3, red_band_idx=0)
    assert ndvi_f.shape == (50, 50)
    assert np.all(ndvi_f > 0.5)  # High vegetation index
    assert ndvi_col.shape == (50, 50, 3)

def test_compute_ndwi():
    arr = np.zeros((50, 50, 4), dtype=np.uint8)
    arr[:, :, 1] = 220  # High Green
    arr[:, :, 3] = 40   # Low NIR
    ndwi_f, ndwi_col = compute_ndwi(arr, green_band_idx=1, nir_band_idx=3)
    assert ndwi_f.shape == (50, 50)
    assert np.all(ndwi_f > 0.5)  # High water index
    assert ndwi_col.shape == (50, 50, 3)

def test_compute_cir():
    arr = np.full((50, 50, 3), 100, dtype=np.uint8)
    cir = compute_false_color_cir(arr)
    assert cir.shape == (50, 50, 3)

def test_compute_sar_db():
    arr = np.random.randint(10, 200, (64, 64), dtype=np.uint8)
    norm_db, stats = compute_sar_decibel_and_filter(arr)
    assert norm_db.shape == (64, 64)
    assert "mean_sigma0_db" in stats

def test_spectral_indices_endpoint():
    img = Image.new("RGB", (64, 64), (100, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    response = client.post("/api/v1/analyze/spectral-indices", json={
        "images": [{"data": b64, "mimeType": "image/png", "filename": "crop.png"}],
        "query": "Compute indices",
        "mode": "single"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "ndvi" in data["indices"]
    assert "ndwi" in data["indices"]
    assert "cir" in data["indices"]
    assert "sar_db" in data["indices"]
