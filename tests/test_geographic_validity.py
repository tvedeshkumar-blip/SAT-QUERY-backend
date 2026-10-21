"""Regression Tests for Step 6B: Geographic Validity Guard.

Verifies:
1. When a local fixture does not intersect the requested AOI and the offline fallback clamps
   the raster window, provenance records aoi_containment_status == "out_of_bounds_clamped"
   and actual_fixture_bounds_wgs84.
2. Scientific limitations explicitly state that the local fixture does not intersect the requested
   AOI and is used only for demonstration purposes.
3. Strict rejection behavior for remote COG rasters with empty AOI intersection is preserved.
4. Demo 4 end-to-end orchestration exposes the containment disclosure without fabricating
   authentic geographic coverage.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.acquisition.stac_service import (
    STACAcquisitionService,
    STACCompositeRetrieveRequest
)
from app.acquisition.windowed_reader import (
    SafeWindowedCOGReader,
    WindowedRetrievalError
)

client = TestClient(app)


def test_local_sample_out_of_bounds_aoi_records_containment_status():
    """Verify local fixture out-of-bounds AOI clamps window and records containment disclosure."""
    service = STACAcquisitionService()

    # Query Mumbai AOI [72.75, 18.88, 73.05, 19.30] against temporal_t1 fixture (in Karnataka)
    mumbai_bbox = [72.75, 18.88, 73.05, 19.30]
    req = STACCompositeRetrieveRequest(
        scene_id="local_sample_temporal_t1_pre_monsoon",
        aoi_bbox=mumbai_bbox,
        collection="sentinel-2-l2a",
        bands=["b04", "b08", "scl"],
        compute_ndvi=True
    )

    resp = service.retrieve_composite_window(req)
    assert resp.status == "success"

    # 1. Provenance records explicit out_of_bounds_clamped status
    prov = resp.provenance
    assert prov.get("aoi_containment_status") == "out_of_bounds_clamped"
    assert "actual_fixture_bounds_wgs84" in prov
    bounds = prov["actual_fixture_bounds_wgs84"]
    assert len(bounds) == 4
    # Actual fixture is in Karnataka (lon ~77.06 to 77.08, lat ~16.70 to 16.72)
    assert 76.5 < bounds[0] < 77.5
    assert 16.0 < bounds[1] < 17.5

    # 2. Scientific limitations disclose geographic mismatch
    lims = resp.scientific_limitations
    assert any("Local demonstration fixture does not intersect requested AOI" in l for l in lims)


def test_local_sample_in_bounds_aoi_records_contained_status():
    """Verify local fixture intersecting requested AOI records contained status without mismatch limitation."""
    service = STACAcquisitionService()

    # risat_sar_mumbai.tif native WGS84 bounds are approx (72.915, 18.983, 72.922, 18.990)
    inside_bbox = [72.916, 18.984, 72.920, 18.988]
    req = STACCompositeRetrieveRequest(
        scene_id="local_sample_risat_sar_mumbai",
        aoi_bbox=inside_bbox,
        collection="sentinel-1-rtc",
        bands=["vv", "vh"],
        compute_dprvi=True
    )

    resp = service.retrieve_composite_window(req)
    assert resp.status == "success"

    prov = resp.provenance
    assert prov.get("aoi_containment_status") == "contained"
    assert not any("does not intersect requested AOI" in l for l in resp.scientific_limitations)


def test_remote_cog_empty_aoi_intersection_strictly_rejected():
    """Verify SafeWindowedCOGReader strictly preserves hard rejection for remote rasters with empty intersection."""
    # Use a mock dataset whose native extent does not intersect the requested AOI
    from affine import Affine
    mock_dataset = MagicMock()
    mock_dataset.crs = "EPSG:4326"
    mock_dataset.transform = Affine(0.001, 0.0, 77.0, 0.0, -0.001, 28.5)
    # Raster covers Delhi: [77.0, 28.0, 77.5, 28.5]
    mock_dataset.bounds = MagicMock(left=77.0, bottom=28.0, right=77.5, top=28.5)
    mock_dataset.__enter__.return_value = mock_dataset

    with patch("rasterio.open", return_value=mock_dataset), \
         patch("httpx.Client.head") as mock_head, \
         patch("httpx.Client.stream") as mock_stream:

        mock_head.return_value = MagicMock(status_code=200, headers={"accept-ranges": "bytes", "content-length": "100000"})
        mock_stream_resp = MagicMock(status_code=206, headers={"content-range": "bytes 0-1023/100000", "content-length": "1024"})
        mock_stream.return_value.__enter__.return_value = mock_stream_resp

        # Request Mumbai AOI [72.75, 18.88, 73.05, 19.30] against Delhi raster -> empty intersection
        with pytest.raises(WindowedRetrievalError, match="empty spatial intersection"):
            SafeWindowedCOGReader.read_cog_window(
                url="https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif",
                bbox_wgs84=[72.75, 18.88, 73.05, 19.30]
            )


def test_demo4_e2e_exposes_aoi_containment_disclosure():
    """Verify Demo 4 end-to-end orchestration exposes out-of-bounds containment disclosure."""
    payload = {
        "images": [],
        "query": "Perform joint optical and radar fusion for Mumbai [72.75, 18.88, 73.05, 19.30]",
        "mode": "orchestration"
    }

    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()

    # 1. Metadata records containment status
    meta = data.get("metadata", {})
    assert meta.get("aoi_containment_status") == "out_of_bounds_clamped"
    assert "actual_fixture_bounds_wgs84" in meta

    # 2. Execution trace contains explicit AOI advisory
    trace = data.get("trace", {})
    trace_steps = trace.get("steps", [])
    assert any(s.get("step") == "AOI_CONTAINMENT_ADVISORY" for s in trace_steps)

    # 3. Scientific report includes limitation
    verif = data.get("verification", {})
    limitations = verif.get("scientific_limitations", [])
    assert any("Local demonstration fixture does not intersect requested AOI" in l for l in limitations)

    # 4. Reproducible report provenance reflects containment status
    rep = data.get("reproducible_report", {})
    rep_prov = rep.get("provenance", {})
    assert rep_prov.get("aoi_containment_status") == "out_of_bounds_clamped"


def test_cross_modal_sample_simulation_rejected():
    """Step 6D: Verify optical raster cannot be used to simulate SAR backscatter."""
    service = STACAcquisitionService()
    req = STACCompositeRetrieveRequest(
        scene_id="local_sample_temporal_t1_pre_monsoon",
        aoi_bbox=[77.06, 16.69, 77.08, 16.72],
        collection="sentinel-1-rtc",
        bands=["vv", "vh"],
        compute_dprvi=True
    )
    resp = service.retrieve_composite_window(req)
    assert resp.status == "error"
    assert "Modality mismatch" in resp.error
    assert "optical raster and cannot be used as a microwave SAR observation" in resp.error


def test_sar_fixture_requested_as_optical_rejected():
    """Step 6D: Verify SAR raster cannot be used to simulate optical multispectral bands."""
    service = STACAcquisitionService()
    req = STACCompositeRetrieveRequest(
        scene_id="local_sample_risat_sar_mumbai",
        aoi_bbox=[72.915, 18.983, 72.922, 18.989],
        collection="sentinel-2-l2a",
        bands=["b04", "b08", "scl"],
        compute_ndvi=True
    )
    resp = service.retrieve_composite_window(req)
    assert resp.status == "error"
    assert "Modality mismatch" in resp.error
    assert "SAR raster and cannot be used as an optical multispectral observation" in resp.error


def test_demo4_e2e_disentangled_fixtures_cross_sensor_validation_failure():
    """Step 6D: Verify Demo 4 end-to-end exposes separate optical/SAR fixtures and validation rejection."""
    payload = {
        "images": [],
        "query": "Perform joint optical and radar fusion for Mumbai [72.75, 18.88, 73.05, 19.30]",
        "mode": "orchestration"
    }

    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Disentangled optical and SAR fixtures recorded in metadata
    meta = data.get("metadata", {})
    cs_prov = meta.get("optical_sar_provenance", {})
    assert cs_prov.get("optical", {}).get("catalog_id") == "local_sample_cartosat_optical_bengaluru"
    assert cs_prov.get("sar", {}).get("catalog_id") == "local_sample_risat_sar_mumbai"

    # Cross-sensor verification accurately reports spatial mismatch and 0% overlap
    cs_verif = cs_prov.get("cross_sensor_verification", {})
    assert cs_verif.get("spatial_overlap_pct") == 0.0
    assert cs_verif.get("crs_matching") is False

    # Execution trace records rejected fusion step
    trace_steps = data.get("trace", {}).get("steps", [])
    assert any("rejected optical-SAR pair" in s.get("detail", "") for s in trace_steps)

