"""
Unit and integration test suite for Stage 9: Interactive Windowed Acquisition & Workspace Ingestion Subsystem.
Tests all requirements:
1. Windowed retrieval request schema validation (bounding box geometry, latitudes, longitudes).
2. Protocol & host security allowlists (HTTPS enforcement, untrusted host rejection).
3. HTTP 206 Partial Content verification and non-206 rejection.
4. Safe window dimension clamping (<= 1024x1024 px).
5. In-memory GeoTIFF serialization fidelity (CRS, affine transform, nodata, dtypes).
6. REST API endpoint POST /api/v1/acquisition/retrieve-window behavior & error handling.
7. Local sample windowed retrieval for offline continuity.
8. End-to-end integration: windowed acquisition, cross-sensor validation, agent analysis, verification, and Stage 8 reporting.
"""

import base64
import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from pydantic import ValidationError
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.schemas.acquisition import (
    STACWindowedRetrieveRequest,
    STACRetrieveResponse
)
from app.acquisition.windowed_reader import SafeWindowedCOGReader, WindowedRetrievalError
from app.acquisition.stac_service import stac_service, STACAcquisitionService
from app.acquisition.asset_validator import AssetValidator
from app.acquisition.cross_sensor_validator import CrossSensorValidator
from app.agent.controller import agent_controller
from app.schemas.analysis import AnalysisRequest, ImageInput


client = TestClient(app)


def test_01_windowed_request_schema_validation():
    """Validates bounding box format, latitude/longitude bounds, and length requirements."""
    # Valid bounding box in EPSG:4326 [min_lon, min_lat, max_lon, max_lat]
    req = STACWindowedRetrieveRequest(
        scene_id="S2A_TEST_01",
        asset_key="B04",
        aoi_bbox=[77.58, 12.96, 77.60, 12.98],
        collection="sentinel-2-l2a"
    )
    assert req.aoi_bbox == [77.58, 12.96, 77.60, 12.98]
    assert req.scene_id == "S2A_TEST_01"

    # Inverted latitude: min_lat > max_lat
    with pytest.raises(ValidationError):
        STACWindowedRetrieveRequest(
            scene_id="S2A_TEST_01",
            asset_key="B04",
            aoi_bbox=[77.58, 13.00, 77.60, 12.90]
        )

    # Invalid latitude out of [-90, 90]
    with pytest.raises(ValidationError):
        STACWindowedRetrieveRequest(
            scene_id="S2A_TEST_01",
            asset_key="B04",
            aoi_bbox=[77.58, -95.0, 77.60, 12.90]
        )

    # Invalid longitude out of [-180, 180]
    with pytest.raises(ValidationError):
        STACWindowedRetrieveRequest(
            scene_id="S2A_TEST_01",
            asset_key="B04",
            aoi_bbox=[190.0, 12.0, 195.0, 13.0]
        )

    # Incorrect element count (3 elements instead of 4)
    with pytest.raises(ValidationError):
        STACWindowedRetrieveRequest(
            scene_id="S2A_TEST_01",
            asset_key="B04",
            aoi_bbox=[77.58, 12.0, 77.60]
        )


def test_02_untrusted_host_rejected_by_windowed_reader():
    """Security check: assets residing on untrusted hosts are rejected immediately."""
    untrusted_url = "https://unauthorized-storage.example.org/rasters/test.tif"
    with pytest.raises(WindowedRetrievalError, match="not in the trusted satellite host allowlist"):
        SafeWindowedCOGReader.read_cog_window(
            url=untrusted_url,
            bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
        )


def test_03_insecure_http_protocol_rejected():
    """Security check: insecure unencrypted HTTP protocol URLs are rejected."""
    insecure_url = "http://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif"
    with pytest.raises(WindowedRetrievalError, match="Insecure protocol: only HTTPS URLs are allowed"):
        SafeWindowedCOGReader.read_cog_window(
            url=insecure_url,
            bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
        )


def test_04_missing_or_failed_206_range_rejected():
    """Ensures servers that return HTTP 200 instead of HTTP 206 Partial Content are aborted."""
    test_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif"
    
    mock_head = MagicMock()
    mock_head.status_code = 200
    mock_head.headers = {"content-length": "50000000", "accept-ranges": "bytes"}

    mock_range = MagicMock()
    mock_range.status_code = 200  # BAD: Server attempted full object transfer instead of 206

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_head
    mock_client.stream.return_value.__enter__.return_value = mock_range

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(WindowedRetrievalError, match="instead of 206 Partial Content"):
            SafeWindowedCOGReader.read_cog_window(
                url=test_url,
                bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
            )


def test_05_in_memory_geotiff_serialization_and_metadata_fidelity():
    """Verifies array serialization to GeoTIFF preserves CRS, transform, nodata, and passes asset validation."""
    arr = np.random.uniform(0.01, 1.5, size=(64, 64)).astype(np.float32)
    meta = {
        "crs": "EPSG:32643",
        "transform": [10.0, 0.0, 779880.0, 0.0, -10.0, 1436370.0],
        "nodata": -32768.0,
        "bands": 1,
        "tags": {
            "POLARIZATION": "VV",
            "SENSOR": "Sentinel-1 C-SAR IW RTC",
            "CALIBRATION": "gamma0"
        }
    }

    raw_bytes = STACAcquisitionService._array_to_geotiff_bytes(arr, meta)
    assert len(raw_bytes) > 0

    # Read back through rasterio to confirm structural integrity
    with MemoryFile(raw_bytes) as mem:
        with mem.open() as src:
            assert str(src.crs) == "EPSG:32643"
            assert src.shape == (64, 64)
            assert src.dtypes[0] == "float32"
            assert src.nodata == -32768.0
            assert src.tags().get("POLARIZATION") == "VV"
            read_data = src.read(1)
            np.testing.assert_allclose(read_data, arr, rtol=1e-5)

    # Validate through AssetValidator
    val_report = AssetValidator.validate_asset(
        raw_bytes,
        filename="test_serialized_window.tif",
        provenance={"sensor": "Sentinel-1 C-SAR IW RTC", "modality": "SAR"}
    )
    assert val_report["status"] == "valid"
    assert val_report["asset_category"] in ("scientific_raster", "single_band_spectral")
    assert val_report["raster_metadata"]["crs"] == "EPSG:32643"
    assert val_report["raster_metadata"]["width"] == 64
    assert val_report["raster_metadata"]["height"] == 64


def test_06_local_sample_windowed_retrieval():
    """Tests windowed retrieval for local demonstration sample rasters."""
    # 1. Valid spatial intersection with cartosat footprint (80.374..80.380, 12.847..12.853)
    req = STACWindowedRetrieveRequest(
        scene_id="local_sample_cartosat_optical_bengaluru",
        asset_key="visual",
        aoi_bbox=[80.375, 12.848, 80.378, 12.852]
    )

    resp = stac_service.retrieve_window(req)
    assert resp.status == "success"
    assert resp.category == "scientific_raster"
    assert resp.data_base64.startswith("data:image/tiff;base64,")
    assert resp.provenance.get("windowed_crop") is True
    assert resp.metadata.get("crs") is not None
    assert resp.validation is not None
    assert resp.validation.status == "valid"

    # 2. Empty spatial intersection rejection check (disjoint bounding box)
    disjoint_req = STACWindowedRetrieveRequest(
        scene_id="local_sample_cartosat_optical_bengaluru",
        asset_key="visual",
        aoi_bbox=[77.58, 12.96, 77.60, 12.98]
    )
    disjoint_resp = stac_service.retrieve_window(disjoint_req)
    assert disjoint_resp.status == "error"
    assert "empty spatial intersection" in disjoint_resp.error.lower()


def test_07_api_endpoint_windowed_retrieval_success_and_error_handling():
    """Tests POST /api/v1/acquisition/retrieve-window HTTP interface."""
    # 1. Valid local sample window retrieval
    valid_payload = {
        "scene_id": "local_sample_cartosat_optical_bengaluru",
        "asset_key": "visual",
        "aoi_bbox": [80.375, 12.848, 80.378, 12.852]
    }
    r = client.post("/api/v1/acquisition/retrieve-window", json=valid_payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "success"
    assert data["category"] == "scientific_raster"
    assert "data_base64" in data
    assert data["data_base64"].startswith("data:image/tiff;base64,")
    assert data["metadata"]["width"] > 0
    assert data["metadata"]["height"] > 0

    # 2. Invalid AOI bounding box format returns 422 Unprocessable Entity
    invalid_bbox_payload = {
        "scene_id": "local_sample_cartosat_optical_bengaluru",
        "asset_key": "visual",
        "aoi_bbox": [80.375, 14.00, 80.378, 12.00]  # min_lat > max_lat
    }
    r_err = client.post("/api/v1/acquisition/retrieve-window", json=invalid_bbox_payload)
    assert r_err.status_code == 422

    # 3. Non-existent scene returns 400 Bad Request
    missing_scene_payload = {
        "scene_id": "non_existent_scene_xyz_999",
        "asset_key": "B04",
        "aoi_bbox": [80.375, 12.848, 80.378, 12.852]
    }
    r_miss = client.post("/api/v1/acquisition/retrieve-window", json=missing_scene_payload)
    assert r_miss.status_code == 400
    assert "not found in active STAC search session" in r_miss.json()["detail"]


def test_08_end_to_end_windowed_acquisition_and_cross_sensor_analysis():
    """
    End-to-end integration:
    Generates two windowed scientific GeoTIFFs (Optical surface reflectance & SAR backscatter),
    validates the pair via CrossSensorValidator, executes joint analysis in AgentController,
    and confirms deterministic verification passes with Stage 8 reproducible report generation.
    """
    # 1. Generate Optical Window GeoTIFF (Sentinel-2 BOA Red Reflectance, 32x32, EPSG:32643)
    opt_data = np.full((32, 32), 1298, dtype=np.uint16)  # 1298 DN = 0.1298 BOA reflectance
    opt_meta = {
        "crs": "EPSG:32643",
        "transform": [10.0, 0.0, 779880.0, 0.0, -10.0, 1436370.0],
        "nodata": 0,
        "bands": 1,
        "tags": {"SENSOR": "Sentinel-2 MSI Level-2A", "DATA_TYPE": "Surface Reflectance BOA"}
    }
    opt_bytes = STACAcquisitionService._array_to_geotiff_bytes(opt_data, opt_meta)
    opt_b64 = "data:image/tiff;base64," + base64.b64encode(opt_bytes).decode("utf-8")

    # 2. Generate SAR Window GeoTIFF (Sentinel-1 RTC VV Backscatter, 32x32, EPSG:32643)
    sar_data = np.full((32, 32), 0.7417, dtype=np.float32)  # 0.7417 linear gamma-0 power
    sar_meta = {
        "crs": "EPSG:32643",
        "transform": [10.0, 0.0, 779880.0, 0.0, -10.0, 1436370.0],
        "nodata": -32768.0,
        "bands": 1,
        "tags": {
            "SENSOR": "Sentinel-1 C-SAR IW RTC",
            "POLARIZATION": "VV",
            "CALIBRATION": "gamma0",
            "TERRAIN_CORRECTION": "radiometrically_terrain_corrected"
        }
    }
    sar_bytes = STACAcquisitionService._array_to_geotiff_bytes(sar_data, sar_meta)
    sar_b64 = "data:image/tiff;base64," + base64.b64encode(sar_bytes).decode("utf-8")

    # 3. Cross-Sensor Pair Pre-Validation
    pair_report = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_filename="s2_b04_window.tif",
        sar_filename="s1_vv_window.tif",
        optical_provenance={"sensor": "Sentinel-2", "modality": "OPTICAL"},
        sar_provenance={
            "sensor": "Sentinel-1",
            "modality": "SAR",
            "polarization": "VV",
            "calibration_status": "calibrated_backscatter",
            "terrain_correction_status": "radiometrically_terrain_corrected"
        }
    )
    assert pair_report["status"] in ("compatible", "compatible_with_limitations")
    assert pair_report["is_compatible"] is True
    assert pair_report["crs_compatible"] is True
    assert pair_report["modality_compatible"] is True

    # 4. Ingest into AgentController via AnalysisRequest
    req = AnalysisRequest(
        query="Perform joint optical and SAR analysis of surface reflectance and radar roughness.",
        mode="optical_sar",
        images=[
            ImageInput(
                data=opt_b64,
                mimeType="image/tiff",
                filename="s2_b04_window.tif",
                role="optical",
                asset_category="scientific_raster",
                provenance={
                    "sensor": "Sentinel-2 MSI",
                    "modality": "OPTICAL",
                    "datetime": "2024-03-28T05:27:01Z"
                }
            ),
            ImageInput(
                data=sar_b64,
                mimeType="image/tiff",
                filename="s1_vv_window.tif",
                role="sar",
                asset_category="scientific_raster",
                provenance={
                    "sensor": "Sentinel-1 C-SAR",
                    "modality": "SAR",
                    "polarization": "VV",
                    "calibration_status": "calibrated_backscatter",
                    "terrain_correction_status": "radiometrically_terrain_corrected",
                    "datetime": "2024-03-22T00:40:42Z"
                }
            )
        ]
    )

    response = agent_controller.process_request(req)

    # 5. Verify analysis execution, verification, and Stage 8 report integrity
    assert response.actual_model_used == "OpticalSARJointAnalysisProvider"
    assert response.fallback_used is False
    assert response.verification is not None
    assert response.verification.is_interpretive_only is True
    assert response.reproducible_report is not None
    assert response.reproducible_report.report_id.startswith("report_")

    # Verify visual and quantitative evidence artifacts produced
    assert len(response.evidence) == 4
    artifact_ids = [ev.id for ev in response.evidence]
    assert "ev_optical_reflectance" in artifact_ids
    assert "ev_sar_backscatter" in artifact_ids
    assert "ev_joint_fused" in artifact_ids
    assert "ev_sar_specular_mask" in artifact_ids

    # Verify Stage 8 report captures all 4 evidence artifacts
    assert len(response.reproducible_report.evidence_artifacts) == 4


def test_09_redirect_to_untrusted_host_rejected():
    """Security check: HTTP redirects pointing to untrusted hosts are intercepted and rejected."""
    initial_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/valid_path.tif"
    untrusted_redirect_url = "https://malicious-satellite-cache.example.com/exploit.tif"

    mock_redirect_resp = MagicMock()
    mock_redirect_resp.status_code = 302
    mock_redirect_resp.headers = {"Location": untrusted_redirect_url}

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_redirect_resp

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(WindowedRetrievalError, match="not in the trusted satellite host allowlist"):
            SafeWindowedCOGReader.read_cog_window(
                url=initial_url,
                bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
            )


def test_10_redirect_to_insecure_http_protocol_rejected():
    """Security check: HTTP redirects downgrading to unencrypted HTTP protocol are intercepted and rejected."""
    initial_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/valid_path.tif"
    insecure_redirect_url = "http://sentinel-cogs.s3.us-west-2.amazonaws.com/downgraded.tif"

    mock_redirect_resp = MagicMock()
    mock_redirect_resp.status_code = 301
    mock_redirect_resp.headers = {"Location": insecure_redirect_url}

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_redirect_resp

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(WindowedRetrievalError, match="Insecure protocol"):
            SafeWindowedCOGReader.read_cog_window(
                url=initial_url,
                bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
            )


def test_11_cumulative_transfer_limit_enforced():
    """Ensures cumulative transfer limit across probe and raster reading is strictly enforced."""
    test_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif"

    # 1. max_bytes validator rejects values exceeding 20 MB cap or non-positive
    with pytest.raises(ValidationError):
        STACWindowedRetrieveRequest(
            scene_id="S2A_TEST_01",
            asset_key="B04",
            aoi_bbox=[77.58, 12.96, 77.60, 12.98],
            max_bytes=25 * 1024 * 1024  # Exceeds 20 MB
        )

    with pytest.raises(ValidationError):
        STACWindowedRetrieveRequest(
            scene_id="S2A_TEST_01",
            asset_key="B04",
            aoi_bbox=[77.58, 12.96, 77.60, 12.98],
            max_bytes=-100
        )

    # 2. SafeWindowedCOGReader rejects when transfer budget is exceeded during initial probe
    mock_head = MagicMock()
    mock_head.status_code = 200
    mock_head.headers = {"content-length": "50000000", "accept-ranges": "bytes"}

    mock_range = MagicMock()
    mock_range.status_code = 206
    mock_range.headers = {"content-range": "bytes 0-2047/50000000"}
    mock_range.iter_bytes.return_value = [b"0" * 2048]

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_head
    mock_client.stream.return_value.__enter__.return_value = mock_range

    with patch("httpx.Client", return_value=mock_client):
        with pytest.raises(WindowedRetrievalError, match="Cumulative transfer limit of 1024 bytes exceeded"):
            SafeWindowedCOGReader.read_cog_window(
                url=test_url,
                bbox_wgs84=[77.58, 12.96, 77.60, 12.98],
                max_bytes=1024  # Budget smaller than probe payload
            )


def test_12_empty_spatial_intersection_remote_rejected():
    """Verifies that an AOI completely disjoint from the remote raster extent is rejected immediately."""
    test_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif"

    mock_head = MagicMock()
    mock_head.status_code = 200
    mock_head.headers = {"content-length": "50000000", "accept-ranges": "bytes"}

    mock_range = MagicMock()
    mock_range.status_code = 206
    mock_range.headers = {"content-range": "bytes 0-1023/50000000"}
    mock_range.iter_bytes.return_value = [b"0" * 1024]

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_head
    mock_client.stream.return_value.__enter__.return_value = mock_range

    # Mock rasterio DatasetReader with bounds covering Bengaluru [77.5, 12.9, 77.7, 13.1]
    mock_src = MagicMock()
    mock_src.__enter__.return_value = mock_src
    mock_src.crs = "EPSG:4326"
    mock_src.width = 1000
    mock_src.height = 1000
    mock_src.count = 1
    mock_src.dtypes = ["uint16"]
    mock_src.bounds = MagicMock(left=77.5, bottom=12.9, right=77.7, top=13.1)
    mock_src.transform = rasterio.Affine(0.0002, 0.0, 77.5, 0.0, -0.0002, 13.1)

    with patch("httpx.Client", return_value=mock_client):
        with patch("rasterio.open", return_value=mock_src):
            with pytest.raises(WindowedRetrievalError, match="empty spatial intersection"):
                SafeWindowedCOGReader.read_cog_window(
                    url=test_url,
                    bbox_wgs84=[-122.5, 37.7, -122.3, 37.9]  # San Francisco (0% intersection)
                )


def test_13_missing_or_identity_crs_rejected():
    """Ensures missing CRS or uncalibrated identity transform is rejected as geospatial evidence."""
    # 1. GeoTIFF serialization rejects missing CRS
    with pytest.raises(ValueError, match="Cannot serialize raster without valid Coordinate Reference System"):
        STACAcquisitionService._array_to_geotiff_bytes(
            np.ones((16, 16), dtype=np.uint16),
            {"crs": None, "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0]}
        )

    # 2. Windowed COG Reader rejects missing CRS
    test_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif"
    mock_head = MagicMock(status_code=200, headers={"content-length": "1000", "accept-ranges": "bytes"})
    mock_range = MagicMock(status_code=206, headers={"content-range": "bytes 0-1023/1000"})
    mock_range.iter_bytes.return_value = [b"0" * 1024]
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.head.return_value = mock_head
    mock_client.stream.return_value.__enter__.return_value = mock_range

    mock_src_nocrs = MagicMock()
    mock_src_nocrs.__enter__.return_value = mock_src_nocrs
    mock_src_nocrs.crs = None
    mock_src_nocrs.transform = rasterio.Affine(0.0002, 0.0, 77.5, 0.0, -0.0002, 13.1)
    with patch("httpx.Client", return_value=mock_client):
        with patch("rasterio.open", return_value=mock_src_nocrs):
            with pytest.raises(WindowedRetrievalError, match="Asset lacks a valid Coordinate Reference System"):
                SafeWindowedCOGReader.read_cog_window(
                    url=test_url,
                    bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
                )

    # 3. Windowed COG Reader rejects identity affine transform
    mock_src_identity = MagicMock()
    mock_src_identity.__enter__.return_value = mock_src_identity
    mock_src_identity.crs = "EPSG:4326"
    mock_src_identity.transform = rasterio.Affine.identity()
    with patch("httpx.Client", return_value=mock_client):
        with patch("rasterio.open", return_value=mock_src_identity):
            with pytest.raises(WindowedRetrievalError, match="identity affine transform without geospatial calibration"):
                SafeWindowedCOGReader.read_cog_window(
                    url=test_url,
                    bbox_wgs84=[77.58, 12.96, 77.60, 12.98]
                )


def test_14_existing_acquisition_endpoints_compatible():
    """Confirms existing search and full retrieval endpoints remain fully backwards-compatible."""
    # 1. Search endpoint
    search_payload = {
        "bbox": [77.5, 12.9, 77.7, 13.1],
        "collection": "sentinel-2-l2a",
        "limit": 2
    }
    r_search = client.post("/api/v1/acquisition/search", json=search_payload)
    assert r_search.status_code == 200
    search_data = r_search.json()
    assert search_data["status"] in ("success", "zero_results")

    # 2. Retrieve endpoint (for local sample)
    retrieve_payload = {
        "scene_id": "local_sample_cartosat_optical_bengaluru",
        "asset_key": "visual"
    }
    r_ret = client.post("/api/v1/acquisition/retrieve", json=retrieve_payload)
    assert r_ret.status_code == 200
    ret_data = r_ret.json()
    assert ret_data["status"] == "success"
    assert ret_data["category"] == "scientific_raster"
    assert "data_base64" in ret_data


def test_15_retrieve_window_endpoint_with_screen_quality_serialization():
    """
    Regression Test for HTTP 500 / PydanticSerializationError on POST /api/v1/acquisition/retrieve-window:
    Verifies that when screen_quality=True is requested (as sent by the frontend UI when clicking
    'Acquire Window (AOI Crop)'), all NumPy arrays, boolean masks, and nested evaluations
    are converted to JSON-serializable representations (base64 PNG data URIs and Python primitives)
    without raising pydantic_core.PydanticSerializationError.
    """
    # 1. Local sample regression check with screen_quality=True
    local_payload = {
        "scene_id": "local_sample_cartosat_optical_bengaluru",
        "asset_key": "visual",
        "aoi_bbox": [80.375, 12.848, 80.378, 12.852],
        "screen_quality": True
    }
    r_local = client.post("/api/v1/acquisition/retrieve-window", json=local_payload)
    assert r_local.status_code == 200, f"Expected 200 but got {r_local.status_code}: {r_local.text}"
    local_data = r_local.json()
    assert local_data["status"] == "success"
    assert local_data["quality_screening"] is not None
    assert "optical" in local_data["quality_screening"]
    assert local_data["quality_screening"]["optical"]["screened"] is True
    assert local_data["quality_screening"]["clear_mask"].startswith("data:image/png;base64,")
    assert local_data["metadata"]["scl_mask"].startswith("data:image/png;base64,")

    # 2. Simulated online Sentinel-2 scene with companion SCL asset
    simulated_scene_id = "S2A_SIMULATED_SERIALIZATION_001"
    stac_service._scene_cache[simulated_scene_id] = {
        "provider": "Element84 Earth Search (AWS)",
        "catalog_url": "https://earth-search.aws.element84.com/v1",
        "collection": "sentinel-2-l2a",
        "datetime": "2024-03-28T05:27:01Z",
        "raw_assets": {
            "visual": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test_visual.tif",
                "type": "image/tiff"
            },
            "scl": {
                "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test_scl.tif",
                "type": "image/tiff"
            }
        }
    }

    sim_opt_data = np.full((32, 32, 3), 120, dtype=np.uint8)
    sim_scl_data = np.full((32, 32), 4, dtype=np.uint8)  # Class 4 = Vegetation
    sim_meta = {
        "crs": "EPSG:32643",
        "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0],
        "nodata": 0,
        "bands": 3,
        "width": 32,
        "height": 32,
        "tags": {"SENSOR": "Sentinel-2A"}
    }
    sim_scl_meta = {
        "crs": "EPSG:32643",
        "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0],
        "nodata": 0,
        "bands": 1,
        "width": 32,
        "height": 32
    }

    def mock_read_window(url, bbox_wgs84, max_bytes=None):
        if "scl" in url.lower():
            return sim_scl_data, sim_scl_meta
        return sim_opt_data, sim_meta

    with patch.object(SafeWindowedCOGReader, "read_cog_window", side_effect=mock_read_window):
        online_payload = {
            "scene_id": simulated_scene_id,
            "asset_key": "visual",
            "aoi_bbox": [80.375, 12.848, 80.378, 12.852],
            "screen_quality": True
        }
        r_online = client.post("/api/v1/acquisition/retrieve-window", json=online_payload)
        assert r_online.status_code == 200, f"Expected 200 but got {r_online.status_code}: {r_online.text}"
        online_data = r_online.json()
        assert online_data["status"] == "success"
        assert online_data["quality_screening"] is not None
        assert online_data["quality_screening"]["optical"]["screened"] is True
        assert online_data["quality_screening"]["clear_mask"].startswith("data:image/png;base64,")
        assert online_data["quality_screening"]["clear_mask_base64"].startswith("data:image/png;base64,")
        assert online_data["metadata"]["scl_mask"].startswith("data:image/png;base64,")
        assert online_data["metadata"]["width"] == 32
        assert online_data["metadata"]["height"] == 32
        assert online_data["metadata"]["crs"] == "EPSG:32643"
        # Validate that validation.raster_metadata also contains valid string masks
        assert online_data["validation"]["raster_metadata"]["scl_mask"].startswith("data:image/png;base64,")

    # 3. Simulated online Sentinel-1 SAR scene with RTC quality screening
    simulated_sar_id = "S1A_SIMULATED_SERIALIZATION_001"
    stac_service._scene_cache[simulated_sar_id] = {
        "provider": "Microsoft Planetary Computer",
        "catalog_url": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "collection": "sentinel-1-rtc",
        "datetime": "2024-03-28T00:00:00Z",
        "raw_assets": {
            "vv": {
                "href": "https://sentinel1euwestrtc.blob.core.windows.net/test_vv.tif",
                "type": "image/tiff"
            }
        }
    }

    sim_sar_data = np.full((32, 32), 0.05, dtype=np.float32)
    sim_sar_meta = {
        "crs": "EPSG:32643",
        "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0],
        "nodata": -32768.0,
        "bands": 1,
        "width": 32,
        "height": 32,
        "tags": {"SENSOR": "Sentinel-1A"}
    }

    with patch.object(SafeWindowedCOGReader, "read_cog_window", return_value=(sim_sar_data, sim_sar_meta)):
        sar_payload = {
            "scene_id": simulated_sar_id,
            "asset_key": "vv",
            "aoi_bbox": [80.375, 12.848, 80.378, 12.852],
            "screen_quality": True
        }
        r_sar = client.post("/api/v1/acquisition/retrieve-window", json=sar_payload)
        assert r_sar.status_code == 200, f"Expected 200 but got {r_sar.status_code}: {r_sar.text}"
        sar_data = r_sar.json()
        assert sar_data["status"] == "success"
        assert sar_data["quality_screening"] is not None
        assert sar_data["quality_screening"]["sar"]["screened"] is True
        assert sar_data["quality_screening"]["valid_mask"].startswith("data:image/png;base64,")
        assert sar_data["metadata"]["sar_mask"].startswith("data:image/png;base64,")

