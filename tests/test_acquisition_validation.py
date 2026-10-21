import os
import sys
import io
import base64

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
import numpy as np
from PIL import Image
from starlette.testclient import TestClient

from app.main import app

from app.acquisition.asset_validator import AssetValidator
from app.acquisition.stac_service import STACAcquisitionService, stac_service
from app.schemas.acquisition import STACRetrieveRequest, STACValidateAssetRequest
from app.schemas.analysis import AnalysisRequest, ImageInput

client = TestClient(app)

def test_validate_asset_geotiff_sample():
    """Verify physical validation of a genuine in-repo GeoTIFF raster."""
    sample_path = os.path.join(stac_service.samples_dir, "cartosat_optical_bengaluru.tif")
    assert os.path.exists(sample_path)

    report = AssetValidator.validate_asset(
        sample_path,
        filename="cartosat_optical_bengaluru.tif",
        provenance={"scene_id": "test_cartosat", "collection": "cartosat-optical"}
    )

    assert report["status"] == "valid"
    assert report["file_format"] == "GeoTIFF"
    assert report["asset_category"] == "rgb_visual_raster"
    assert report["raster_metadata"]["bands"] == 3
    assert report["raster_metadata"]["width"] == 256
    assert report["raster_metadata"]["height"] == 256
    assert "32644" in report["raster_metadata"]["crs"]
    assert report["raster_metadata"]["georeferenced"] is True

    # Check task compatibility
    compat = report["analysis_compatibility"]
    assert compat["vqa"]["status"] == "supported"
    assert compat["captioning"]["status"] == "supported"
    assert compat["grounding"]["status"] == "supported"
    assert compat["spectral_indices"]["status"] == "rejected"  # Lacks calibrated NIR band
    assert "NIR" in compat["spectral_indices"]["reason"]

def test_validate_asset_visual_preview():
    """Verify physical validation of an 8-bit JPEG browse preview."""
    # Create synthetic 8-bit JPEG
    img = Image.new("RGB", (100, 100), color=(120, 150, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()
    b64_str = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("utf-8")

    report = AssetValidator.validate_asset(
        b64_str,
        filename="preview.jpg",
        provenance={"scene_id": "S2A_TEST_PREVIEW", "asset_key": "thumbnail"}
    )

    assert report["status"] == "valid"
    assert report["file_format"] == "JPEG"
    assert report["asset_category"] == "visual_preview"
    assert report["scientific_status"] == "visual_display_only"
    assert report["raster_metadata"]["georeferenced"] is False
    assert report["raster_metadata"]["bands"] == 3

    # Task compatibility
    compat = report["analysis_compatibility"]
    assert compat["vqa"]["status"] == "supported_with_limitations"
    assert compat["captioning"]["status"] == "supported_with_limitations"
    assert compat["spectral_indices"]["status"] == "rejected"
    assert "Quantitative NDVI/NDWI calculation is rejected" in compat["spectral_indices"]["reason"]

def test_validate_asset_single_band_spectral():
    """Verify validation of a single-band raster."""
    arr = np.zeros((64, 64), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    report = AssetValidator.validate_asset(
        png_bytes,
        filename="B04_red.png",
        provenance={"scene_id": "S2A_TEST", "asset_key": "red"}
    )

    assert report["status"] == "valid"
    assert report["raster_metadata"]["bands"] == 1
    # PNG unprojected is visual preview
    assert report["asset_category"] == "visual_preview"
    assert report["analysis_compatibility"]["spectral_indices"]["status"] == "rejected"

def test_validate_asset_corrupted_payload():
    """Verify safe rejection of unreadable corrupted data."""
    corrupted_bytes = b"NOT_A_VALID_IMAGE_OR_TIFF_BINARY_DATA"
    report = AssetValidator.validate_asset(corrupted_bytes, filename="corrupt.tif")

    assert report["status"] == "invalid"
    assert report["asset_category"] == "unsupported"
    assert report["analysis_compatibility"]["vqa"]["status"] == "rejected"
    assert "Unable to decode" in report["error"]

def test_api_validate_endpoint():
    """Test POST /api/v1/acquisition/validate endpoint."""
    img = Image.new("RGB", (64, 64), color=(50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "filename": "test_asset.jpg",
        "data_base64": b64_str,
        "provenance": {"scene_id": "test_scene_123", "provider": "Element84"}
    }
    resp = client.post("/api/v1/acquisition/validate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "valid"
    assert data["asset_category"] == "visual_preview"
    assert data["analysis_compatibility"]["spectral_indices"]["status"] == "rejected"

def test_analysis_rejects_spectral_indices_on_preview():
    """Verify that POST /api/v1/analyze/spectral-indices rejects visual previews."""
    img = Image.new("RGB", (64, 64), color=(80, 120, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "images": [{
            "data": b64_str,
            "mimeType": "image/jpeg",
            "filename": "browse_thumbnail.jpg",
            "role": "primary",
            "asset_category": "visual_preview"
        }],
        "query": "Calculate NDVI",
        "mode": "single"
    }

    resp = client.post("/api/v1/analyze/spectral-indices", json=payload)
    assert resp.status_code == 400
    assert "Quantitative spectral index calculation is strictly forbidden on visual preview browse imagery" in resp.json()["detail"]

def test_analysis_accepts_vqa_on_imported_scene():
    """Verify that single-scene VQA executes and preserves STAC provenance and trace steps."""
    img = Image.new("RGB", (64, 64), color=(60, 100, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    stac_prov = {
        "provider": "Element84 Earth Search (AWS)",
        "collection": "sentinel-2-l2a",
        "scene_id": "S2A_43PGQ_20240328_0_L2A",
        "datetime": "2024-03-28T05:27:14Z",
        "asset_key": "thumbnail"
    }

    payload = {
        "images": [{
            "data": b64_str,
            "mimeType": "image/png",
            "filename": "S2A_43PGQ_20240328_0_L2A_thumbnail.png",
            "role": "primary",
            "asset_category": "visual_preview",
            "provenance": stac_prov,
            "scientific_limitations": ["Visual browse asset only."]
        }],
        "query": "What observable water features exist in this scene?",
        "mode": "single"
    }

    resp = client.post("/api/v1/analyze/vqa", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["task"] in ("vqa", "single")
    assert len(data["answer"]) > 0

    # Verify trace contains ASSET_VALIDATION step
    trace_steps = [s["step"] for s in data["trace"]["steps"]]
    assert "ASSET_VALIDATION" in trace_steps

    # Verify metadata preserves STAC provenance
    assert "provenance" in data["metadata"]
    assert data["metadata"]["provenance"]["scene_id"] == "S2A_43PGQ_20240328_0_L2A"
    assert data["metadata"]["provenance"]["collection"] == "sentinel-2-l2a"

def test_retrieval_size_limit_rejection():
    """Verify that an asset exceeding MAX_RETRIEVAL_BYTES is rejected without silent substitution."""
    service = STACAcquisitionService()
    # Artificially set a low limit of 500 bytes
    service.MAX_DOWNLOAD_BYTES = 500

    service._scene_cache["test_large_scene"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "large_cog": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif",
                "type": "image/tiff",
                "file:size": 50 * 1024 * 1024  # 50 MB
            }
        }
    }

    req = STACRetrieveRequest(
        scene_id="test_large_scene",
        asset_key="large_cog",
        collection="sentinel-2-l2a"
    )

    resp = service.retrieve(req)
    assert resp.status == "error"
    assert "exceeds the configured retrieval safety limit" in resp.error
    assert resp.asset_key == "large_cog"  # Not substituted with thumbnail!

def test_retrieval_rejects_missing_asset_key_without_silent_thumbnail():
    """Verify that requesting an absent asset key fails truthfully rather than substituting a thumbnail."""
    service = STACAcquisitionService()
    service._scene_cache["test_scene_strict"] = {
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-20T00:00:00Z",
        "raw_assets": {
            "thumbnail": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/thumb.jpg",
                "type": "image/jpeg"
            }
        }
    }

    req = STACRetrieveRequest(
        scene_id="test_scene_strict",
        asset_key="nir_b08",  # Not in raw_assets!
        collection="sentinel-2-l2a"
    )

    resp = service.retrieve(req)
    assert resp.status == "error"
    assert "Requested asset key 'nir_b08' is not available" in resp.error
    assert resp.asset_key == "nir_b08"  # Did NOT substitute thumbnail!
