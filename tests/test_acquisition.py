import os
import sys
from unittest.mock import patch, MagicMock

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.acquisition import STACSearchRequest, STACRetrieveRequest
from app.acquisition.stac_service import STACAcquisitionService, stac_service

client = TestClient(app)

MOCK_STAC_RESPONSE = {
    "type": "FeatureCollection",
    "features": [
        {
            "id": "S2A_43PGQ_20240328_0_L2A",
            "collection": "sentinel-2-l2a",
            "bbox": [77.45, 12.85, 77.75, 13.15],
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[77.45, 12.85], [77.75, 12.85], [77.75, 13.15], [77.45, 13.15], [77.45, 12.85]]]
            },
            "properties": {
                "datetime": "2024-03-28T05:25:20Z",
                "eo:cloud_cover": 0.35,
                "platform": "Sentinel-2A"
            },
            "assets": {
                "thumbnail": {
                    "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/S2A/thumbnail.jpg",
                    "type": "image/jpeg",
                    "roles": ["thumbnail"]
                },
                "visual": {
                    "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/S2A/TCI.tif",
                    "type": "image/tiff; application=geotiff",
                    "roles": ["visual"]
                },
                "red": {
                    "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/S2A/B04.tif",
                    "type": "image/tiff; application=geotiff",
                    "roles": ["data"]
                }
            }
        }
    ]
}

def test_stac_search_endpoint_with_mock():
    """Verify STAC search API endpoint with mocked Earth Search response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_STAC_RESPONSE

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.__exit__.return_value = None

    with patch("app.acquisition.stac_service.httpx.Client", return_value=mock_client_instance):
        payload = {
            "bbox": [77.45, 12.85, 77.75, 13.15],
            "start_date": "2024-03-01",
            "end_date": "2024-03-31",
            "collection": "sentinel-2-l2a",
            "max_cloud_cover": 20.0,
            "limit": 5
        }
        res = client.post("/api/v1/acquisition/search", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["status"] == "success"
        assert data["total_results"] == 1
        assert len(data["scenes"]) == 1

        scene = data["scenes"][0]
        assert scene["id"] == "S2A_43PGQ_20240328_0_L2A"
        assert scene["modality"] == "OPTICAL"
        assert scene["cloud_cover"] == 0.35
        assert "thumbnail" in scene["assets"]
        assert scene["assets"]["thumbnail"]["category"] == "visual_preview"
        assert scene["assets"]["visual"]["category"] == "scientific_raster"

def test_stac_search_invalid_coordinates():
    """Verify error reporting on inverted latitude coordinates."""
    payload = {
        "bbox": [77.45, 13.15, 77.75, 12.85],  # min_lat > max_lat
        "collection": "sentinel-2-l2a"
    }
    res = client.post("/api/v1/acquisition/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "error"
    assert "Invalid bounding box" in data["error"]

def test_stac_search_empty_results():
    """Verify clean handling of zero results without data fabrication."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"type": "FeatureCollection", "features": []}

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.__exit__.return_value = None

    with patch("app.acquisition.stac_service.httpx.Client", return_value=mock_client_instance):
        payload = {
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "collection": "sentinel-2-l2a"
        }
        res = client.post("/api/v1/acquisition/search", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "zero_results"
        assert data["total_results"] == 0
        assert len(data["scenes"]) == 0

def test_stac_search_network_fallback():
    """Verify graceful fallback to authentic local sample catalog when network is unreachable."""
    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = Exception("Connection refused")
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.__exit__.return_value = None

    with patch("app.acquisition.stac_service.httpx.Client", return_value=mock_client_instance):
        service = STACAcquisitionService()
        req = STACSearchRequest(
            bbox=[77.45, 12.85, 77.75, 13.15],
            collection="sentinel-2-l2a"
        )
        resp = service.search(req)
        assert resp.status == "success"
        assert "Local In-Repository" in resp.provider
        assert resp.total_results >= 1

def test_stac_retrieve_local_sample():
    """Verify safe retrieval and inspection of authentic local GeoTIFF sample."""
    service = STACAcquisitionService()
    # Trigger fallback to populate cache
    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = Exception("Offline")
    mock_client_instance.__enter__.return_value = mock_client_instance
    mock_client_instance.__exit__.return_value = None

    with patch("app.acquisition.stac_service.httpx.Client", return_value=mock_client_instance):
        service.search(STACSearchRequest(bbox=[77.45, 12.85, 77.75, 13.15]))

    req = STACRetrieveRequest(
        scene_id="local_sample_cartosat_optical_bengaluru",
        asset_key="visual"
    )
    resp = service.retrieve(req)
    assert resp.status == "success"
    assert resp.category == "scientific_raster"
    assert resp.size_bytes > 0
    assert resp.data_base64.startswith("data:image/tiff;base64,")
    assert resp.metadata.get("is_geotiff") is True
    assert "EPSG:32644" in (resp.metadata.get("crs") or "")

def test_stac_retrieve_untrusted_host_blocked():
    """Verify that arbitrary external hosts not on the whitelist are strictly rejected."""
    service = STACAcquisitionService()
    service._scene_cache["evil_scene"] = {
        "raw_assets": {
            "thumbnail": {
                "href": "https://malicious-external-site.com/evil.jpg",
                "type": "image/jpeg"
            }
        }
    }
    req = STACRetrieveRequest(scene_id="evil_scene", asset_key="thumbnail")
    resp = service.retrieve(req)
    assert resp.status == "error"
    assert "not in the whitelist" in resp.error

def test_demo_bitemporal_pair_endpoint():
    """Verify Demo 3 authentic Sentinel-2 GeoTIFF pair endpoint."""
    res = client.get("/api/v1/acquisition/demo-bitemporal-pair")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert data["mode"] == "bitemporal"
    assert data["query"] == "What changed between these two dates?"
    assert data["aoi_bbox"] == [77.4491, 13.0597, 77.5444, 13.1531]
    assert len(data["files"]) == 2

    t1 = data["files"][0]
    t2 = data["files"][1]

    # Authentic filenames and roles
    assert t1["filename"] == "S2B_43PGQ_20230528_0_L2A_visual_window_1792262748.tif"
    assert t1["role"] == "primary"
    assert t1["isGeoTIFF"] is True
    assert t1["size"] > 3000000
    assert t1["dataUrl"].startswith("data:image/tiff;base64,")

    assert t2["filename"] == "S2A_43PGQ_20231119_0_L2A_visual_window_1792262818.tif"
    assert t2["role"] == "secondary"
    assert t2["isGeoTIFF"] is True
    assert t2["size"] > 3000000
    assert t2["dataUrl"].startswith("data:image/tiff;base64,")

    # Authentic STAC provenance
    assert t1["stacProvenance"]["scene_id"] == "S2B_43PGQ_20230528_0_L2A"
    assert t1["stacProvenance"]["tile"] == "43PGQ"
    assert t1["stacProvenance"]["datetime"] == "2023-05-28T05:26:49Z"
    assert t1["stacProvenance"]["sensor"] == "Sentinel-2B MSI Level-2A"

    assert t2["stacProvenance"]["scene_id"] == "S2A_43PGQ_20231119_0_L2A"
    assert t2["stacProvenance"]["tile"] == "43PGQ"
    assert t2["stacProvenance"]["datetime"] == "2023-11-19T05:27:01Z"
    assert t2["stacProvenance"]["sensor"] == "Sentinel-2A MSI Level-2A"

    # Authentic raster validation & CRS
    assert t1["validation"]["status"] == "valid"
    assert t1["validation"]["raster_metadata"]["crs"] == "EPSG:32643"
    assert t1["validation"]["raster_metadata"]["width"] == 1024
    assert t1["validation"]["raster_metadata"]["height"] == 1024
    assert t1["validation"]["raster_metadata"]["bands"] == 3

    assert t2["validation"]["status"] == "valid"
    assert t2["validation"]["raster_metadata"]["crs"] == "EPSG:32643"
    assert t2["validation"]["raster_metadata"]["width"] == 1024
    assert t2["validation"]["raster_metadata"]["height"] == 1024
    assert t2["validation"]["raster_metadata"]["bands"] == 3

    # Honest scientific limitations on RGB vs calibrated NDVI
    assert any("Band 8" in lim for lim in t1["scientificLimitations"])
    assert any("NDVI" in lim for lim in t1["scientificLimitations"])

    # Authentic preview rendering paths for browser display
    assert t1["previewDataUrl"].startswith("data:image/")
    assert t2["previewDataUrl"].startswith("data:image/")
    assert t1["previewUrl"] == "/api/v1/acquisition/demo-bitemporal-preview/t1"
    assert t2["previewUrl"] == "/api/v1/acquisition/demo-bitemporal-preview/t2"

def test_demo_bitemporal_preview_image_endpoints():
    """Verify authentic JPEG/PNG preview image endpoints for Demo 3 browser rendering."""
    res_t1 = client.get("/api/v1/acquisition/demo-bitemporal-preview/t1")
    assert res_t1.status_code == 200
    assert "image/jpeg" in res_t1.headers["content-type"]
    assert len(res_t1.content) > 50000

    res_t2 = client.get("/api/v1/acquisition/demo-bitemporal-preview/t2")
    assert res_t2.status_code == 200
    assert "image/jpeg" in res_t2.headers["content-type"]
    assert len(res_t2.content) > 50000

    res_invalid = client.get("/api/v1/acquisition/demo-bitemporal-preview/invalid_scene")
    assert res_invalid.status_code == 404

def test_demo_bitemporal_pair_analysis_execution():
    """Verify full remote sensing change detection execution using the authentic Demo 3 pair."""
    pair_res = client.get("/api/v1/acquisition/demo-bitemporal-pair")
    assert pair_res.status_code == 200
    pair = pair_res.json()

    payload = {
        "images": [
            {
                "data": f["dataUrl"],
                "mimeType": "image/tiff",
                "filename": f["filename"],
                "role": f["role"],
                "asset_category": f["assetCategory"],
                "provenance": f["stacProvenance"],
                "scientific_limitations": f["scientificLimitations"]
            }
            for f in pair["files"]
        ],
        "query": pair["query"],
        "mode": pair["mode"],
        "aoi_bbox": pair["aoi_bbox"]
    }

    res = client.post("/api/v1/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    meta = data.get("metadata", {})
    assert meta.get("georeferenced") is True
    assert meta.get("crs") == "EPSG:32643"
    assert meta.get("bounds") == [765590.0, 1445120.0, 775830.0, 1455360.0]

    # Verify co-registration assertion and authentic provenance in result
    assert "Inputs are verified co-registered" in data.get("answer", "")
    temporal_prov = meta.get("temporal_provenance", {})
    assert temporal_prov.get("t1", {}).get("catalog_id") == "S2B_43PGQ_20230528_0_L2A"
    assert temporal_prov.get("t2", {}).get("catalog_id") == "S2A_43PGQ_20231119_0_L2A"

