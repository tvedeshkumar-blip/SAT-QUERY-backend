import os
import sys
import io
import json
import base64

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.acquisition.temporal_validator import TemporalPairValidator
from app.acquisition.stac_service import stac_service
from app.models.change_detection.change_detector import (
    ChangeDetectionProvider, 
    PixelDifferenceChangeBaseline, 
    SiameseRSChangeDetector
)
from app.remote_sensing.preprocessing import convert_array_to_base64_png
from app.remote_sensing.validation import ValidationError

client = TestClient(app)

SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "samples"))
T1_PATH = os.path.join(SAMPLES_DIR, "temporal_t1_pre_monsoon.tif")
T2_PATH = os.path.join(SAMPLES_DIR, "temporal_t2_post_monsoon.tif")

def encode_file_b64(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/tiff;base64," + base64.b64encode(f.read()).decode("utf-8")

def create_preview_jpeg_b64(w: int = 128, h: int = 128) -> str:
    img = Image.new("RGB", (w, h), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# 1. Missing T1 or T2
def test_1_missing_t1_or_t2():
    res = TemporalPairValidator.validate_pair(
        t1_input=None,
        t2_input=encode_file_b64(T2_PATH),
        t1_filename="missing_t1.tif",
        t2_filename="t2.tif"
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "Missing temporal scene" in res["rejection_reason"]


# 2. Identical / Degenerate Timestamps
def test_2_identical_timestamps_rejection():
    t1_b64 = encode_file_b64(T1_PATH)
    t2_b64 = encode_file_b64(T2_PATH)
    res = TemporalPairValidator.validate_pair(
        t1_input=t1_b64,
        t2_input=t2_b64,
        t1_filename="t1.tif",
        t2_filename="t2.tif",
        t1_provenance={"datetime": "2023-05-15T00:00:00Z", "modality": "OPTICAL"},
        t2_provenance={"datetime": "2023-05-15T00:00:00Z", "modality": "OPTICAL"}
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "Degenerate temporal pair" in res["rejection_reason"]
    assert res["chronology_status"] == "identical"


# 3. Empty Catalog Results & Retrieval Failures
def test_3_empty_catalog_results_and_retrieval_failure():
    # Empty catalog search for remote ocean coordinates
    resp = client.post("/api/v1/acquisition/search-temporal-pair", json={
        "bbox": [-179.0, -80.0, -178.0, -79.0],
        "t1_start_date": "2020-01-01",
        "t1_end_date": "2020-01-10",
        "t2_start_date": "2020-06-01",
        "t2_end_date": "2020-06-10",
        "collection": "sentinel-2-l2a"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("zero_results", "success", "partial")

    # Retrieval failure for unknown scene ID
    ret_resp = client.post("/api/v1/acquisition/retrieve", json={
        "scene_id": "non_existent_scene_xyz",
        "asset_key": "visual"
    })
    assert ret_resp.status_code == 400


# 4. Non-Overlapping Footprints
def test_4_non_overlapping_footprints():
    # Construct synthetic rasters with disjoint bounds
    import rasterio
    from rasterio.transform import from_bounds
    
    buf1 = io.BytesIO()
    with rasterio.open(
        buf1, 'w', driver='GTiff', height=64, width=64, count=3, dtype='uint8',
        crs='EPSG:32644', transform=from_bounds(100, 100, 200, 200, 64, 64)
    ) as ds:
        ds.write(np.full((3, 64, 64), 50, dtype=np.uint8))
        
    buf2 = io.BytesIO()
    with rasterio.open(
        buf2, 'w', driver='GTiff', height=64, width=64, count=3, dtype='uint8',
        crs='EPSG:32644', transform=from_bounds(500, 500, 600, 600, 64, 64)
    ) as ds:
        ds.write(np.full((3, 64, 64), 50, dtype=np.uint8))

    res = TemporalPairValidator.validate_pair(
        t1_input=buf1.getvalue(),
        t2_input=buf2.getvalue(),
        t1_filename="scene_a.tif",
        t2_filename="scene_b.tif",
        t1_provenance={"datetime": "2023-01-01T00:00:00Z", "modality": "OPTICAL"},
        t2_provenance={"datetime": "2023-06-01T00:00:00Z", "modality": "OPTICAL"}
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "Zero spatial overlap" in res["rejection_reason"]


# 5. CRS Incompatibility
def test_5_crs_incompatibility():
    import rasterio
    from rasterio.transform import from_bounds

    buf1 = io.BytesIO()
    with rasterio.open(
        buf1, 'w', driver='GTiff', height=64, width=64, count=3, dtype='uint8',
        crs='EPSG:32643', transform=from_bounds(100, 100, 200, 200, 64, 64)
    ) as ds:
        ds.write(np.full((3, 64, 64), 50, dtype=np.uint8))

    buf2 = io.BytesIO()
    with rasterio.open(
        buf2, 'w', driver='GTiff', height=64, width=64, count=3, dtype='uint8',
        crs='EPSG:32644', transform=from_bounds(100, 100, 200, 200, 64, 64)
    ) as ds:
        ds.write(np.full((3, 64, 64), 50, dtype=np.uint8))

    res = TemporalPairValidator.validate_pair(
        t1_input=buf1.getvalue(),
        t2_input=buf2.getvalue(),
        t1_filename="utm43.tif",
        t2_filename="utm44.tif",
        t1_provenance={"datetime": "2023-01-01T00:00:00Z", "modality": "OPTICAL"},
        t2_provenance={"datetime": "2023-06-01T00:00:00Z", "modality": "OPTICAL"}
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "CRS mismatch" in res["rejection_reason"]


# 6. Preview-Only Inputs Rejection
def test_6_preview_only_inputs_rejection():
    preview_jpeg = create_preview_jpeg_b64()
    t2_geotiff = encode_file_b64(T2_PATH)

    res = TemporalPairValidator.validate_pair(
        t1_input=preview_jpeg,
        t2_input=t2_geotiff,
        t1_filename="thumbnail_preview.jpg",
        t2_filename="temporal_t2.tif",
        t1_provenance={"asset_key": "thumbnail", "datetime": "2023-01-01T00:00:00Z"},
        t2_provenance={"asset_key": "visual", "datetime": "2023-06-01T00:00:00Z"}
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "Browse-only preview" in res["rejection_reason"]


# 7. Missing Neural Checkpoints & Truthful Fallback Reporting
def test_7_truthful_fallback_reporting():
    provider = ChangeDetectionProvider()
    arr1 = np.full((64, 64, 3), 60, dtype=np.uint8)
    arr2 = np.full((64, 64, 3), 60, dtype=np.uint8)
    arr2[20:40, 20:40] = 220

    res = provider.predict([arr1, arr2], query="Detect change")
    assert res["primary_model"] == "BIT-CD"
    assert res["actual_model_used"] == "PixelDifferenceChangeBaseline"
    assert res["fallback_used"] is True
    assert "Scientific Caveat" in res["answer"]


# 8. Provenance Preservation Across Pipeline
def test_8_provenance_preservation_in_analysis():
    t1_b64 = encode_file_b64(T1_PATH)
    t2_b64 = encode_file_b64(T2_PATH)

    prov_t1 = {
        "scene_id": "S2A_MSIL2A_20230410T053000_TEST",
        "collection": "sentinel-2-l2a",
        "datetime": "2023-04-10T05:30:00Z",
        "asset_key": "visual",
        "modality": "OPTICAL"
    }
    prov_t2 = {
        "scene_id": "S2A_MSIL2A_20231105T053000_TEST",
        "collection": "sentinel-2-l2a",
        "datetime": "2023-11-05T05:30:00Z",
        "asset_key": "visual",
        "modality": "OPTICAL"
    }

    payload = {
        "query": "Detect and map structural changes between T1 and T2.",
        "mode": "bitemporal",
        "images": [
            {"data": t1_b64, "filename": "temporal_t1_pre_monsoon.tif", "role": "primary", "provenance": prov_t1},
            {"data": t2_b64, "filename": "temporal_t2_post_monsoon.tif", "role": "secondary", "provenance": prov_t2}
        ]
    }

    resp = client.post("/api/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["task"] == "change_detection"
    meta = data["metadata"]
    assert "temporal_provenance" in meta
    t_prov = meta["temporal_provenance"]
    assert t_prov["t1"]["catalog_id"] == "S2A_MSIL2A_20230410T053000_TEST"
    assert t_prov["t2"]["catalog_id"] == "S2A_MSIL2A_20231105T053000_TEST"
    assert t_prov["t1"]["datetime"] == "2023-04-10T05:30:00Z"
    assert t_prov["t2"]["datetime"] == "2023-11-05T05:30:00Z"

    # Verify explicit distinction between pixel difference, predicted change, and confirmed change
    dist = t_prov["scientific_distinction"]
    assert "detected_pixel_differences" in dist
    assert "model_predicted_change" in dist
    assert "confirmed_real_world_change" in dist


# 9. Defensible Area Calculations vs Unprojected Geographic CRS
def test_9_defensible_area_calculations():
    # Test projected GeoTIFF (EPSG:32644 UTM) computes hectares
    detector = PixelDifferenceChangeBaseline()
    arr1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    arr2 = np.full((100, 100, 3), 50, dtype=np.uint8)
    arr2[10:30, 10:30] = 200  # 400 changed pixels

    # Scenario A: Projected UTM grid (10m x 10m resolution)
    meta_projected = {
        "crs": "EPSG:32644",
        "resolution": [10.0, 10.0],
        "bounds": [1000, 1000, 2000, 2000]
    }
    res_proj = detector.predict([arr1, arr2], query="Change test", metadata={
        "primary": meta_projected,
        "secondary": meta_projected
    })
    ev_stat_proj = res_proj["evidence"][0]["statistics"]
    assert ev_stat_proj["changed_area_hectares"] is not None
    assert ev_stat_proj["changed_area_hectares"] == 4.0  # 400 * 100m2 = 40,000 m2 = 4.0 ha

    # Scenario B: Unprojected Geographic CRS (EPSG:4326 in degrees)
    meta_geographic = {
        "crs": "EPSG:4326",
        "resolution": [0.0001, 0.0001],
        "bounds": [77.0, 12.0, 78.0, 13.0]
    }
    res_geo = detector.predict([arr1, arr2], query="Change test", metadata={
        "primary": meta_geographic,
        "secondary": meta_geographic
    })
    ev_stat_geo = res_geo["evidence"][0]["statistics"]
    assert ev_stat_geo["changed_area_hectares"] is None
    assert "omitted" in ev_stat_geo["area_hectares_status"]


# 10. End-to-End API Endpoint: Validate Temporal Pair
def test_10_api_validate_temporal_pair():
    t1_b64 = encode_file_b64(T1_PATH)
    t2_b64 = encode_file_b64(T2_PATH)

    resp = client.post("/api/v1/acquisition/validate-temporal-pair", json={
        "t1_data_base64": t1_b64,
        "t2_data_base64": t2_b64,
        "t1_filename": "temporal_t1_pre_monsoon.tif",
        "t2_filename": "temporal_t2_post_monsoon.tif",
        "t1_provenance": {
            "scene_id": "test_t1",
            "datetime": "2023-04-10T05:30:00Z",
            "modality": "OPTICAL",
            "cloud_cover": 1.2
        },
        "t2_provenance": {
            "scene_id": "test_t2",
            "datetime": "2023-11-05T05:30:00Z",
            "modality": "OPTICAL",
            "cloud_cover": 2.5
        }
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_compatible"] is True
    assert data["status"] in ("compatible", "compatible_with_limitations")
    assert data["crs_compatible"] is True
    assert data["spatial_overlap_pct"] is not None
    assert data["spatial_overlap_pct"] > 90.0
    assert data["temporal_delta_days"] is not None
    assert abs(data["temporal_delta_days"] - 209.0) < 1.0
