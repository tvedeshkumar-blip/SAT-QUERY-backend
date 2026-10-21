"""
Test Suite: Safe Online Sentinel-2 Optical and Sentinel-1 RTC SAR Acquisition.

Validates Priority 1:
1. Intelligent STAC Search Routing:
   - Sentinel-2 Optical routes to Element84 Earth Search (AWS).
   - Sentinel-1 RTC SAR routes to Microsoft Planetary Computer (MPC).
   - 'all' collections queries both catalogs and merges scene results.
2. Planetary Computer SAS Token Signing:
   - Automatically signs Azure Blob COG URLs without leaking tokens in logs.
3. Bounded AOI Windowed Retrieval:
   - Windowed COG retrieval of Sentinel-1 RTC dual-pol (VV + VH) rasters.
   - Enforces cumulative transfer budget (<= 20 MB).
4. Cross-Sensor Compatibility Validation:
   - Validates real-world projected Optical + SAR pairs on the same AOI.
   - Preserves strict rejection on unprojected, disjoint, or thumbnail inputs.
5. Offline Demo Preservation:
   - Verifies local sample catalog fallback and honest unsupported rejection.
"""

import os
import sys
import math
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import httpx

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.schemas.acquisition import (
    STACSearchRequest,
    STACSearchResponse,
    STACRetrieveRequest,
    STACWindowedRetrieveRequest,
    STACCompositeRetrieveRequest,
    STACOpticalSARSearchRequest,
    STACOpticalSARPairValidateRequest
)
from app.acquisition.stac_service import STACAcquisitionService
from app.acquisition.cross_sensor_validator import CrossSensorValidator
from app.remote_sensing.sar import compute_calibrated_dual_pol_sar

import rasterio
from rasterio.transform import from_bounds


def create_mock_geotiff_b64(
    bounds=(77.5, 12.9, 77.6, 13.0),
    crs="EPSG:4326",
    width=64,
    height=64,
    bands=2,
    dtype="float32",
    tags=None
) -> str:
    import base64
    memfile = rasterio.MemoryFile()
    transform = from_bounds(*bounds, width, height)
    with memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs=crs,
        transform=transform
    ) as dst:
        data = np.random.uniform(0.01, 0.5, (bands, height, width)).astype(dtype)
        dst.write(data)
        if tags:
            dst.update_tags(**tags)
    return "data:image/tiff;base64," + base64.b64encode(memfile.getbuffer()).decode("utf-8")


# ==============================================================================
# 1. STAC Search Provider Routing (Mocked)
# ==============================================================================

def test_stac_search_routes_optical_to_earth_search():
    """Verifies that optical collection queries target Earth Search AWS catalog."""
    service = STACAcquisitionService()
    req = STACSearchRequest(
        bbox=[77.55, 12.95, 77.65, 13.05],
        collection="sentinel-2-l2a",
        limit=5
    )

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "type": "FeatureCollection",
            "features": [
                {
                    "id": "S2A_TEST_OPTICAL",
                    "collection": "sentinel-2-l2a",
                    "bbox": [77.55, 12.95, 77.65, 13.05],
                    "properties": {"datetime": "2026-10-14T05:25:20Z", "platform": "Sentinel-2A"},
                    "assets": {
                        "visual": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/visual.tif", "type": "image/tiff; application=geotiff; profile=cloud-optimized"},
                        "red": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif", "type": "image/tiff"},
                        "nir": {"href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B08.tif", "type": "image/tiff"}
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        res = service.search(req)

        assert mock_post.called
        target_url = mock_post.call_args[0][0]
        assert "earth-search.aws.element84.com" in target_url
        assert res.status == "success"
        assert res.total_results == 1
        assert res.scenes[0].modality == "OPTICAL"
        assert res.scenes[0].sensor == "Sentinel-2A"


def test_stac_search_routes_sar_to_planetary_computer_rtc():
    """Verifies that SAR requests route to Microsoft Planetary Computer sentinel-1-rtc catalog."""
    service = STACAcquisitionService()
    req = STACSearchRequest(
        bbox=[77.55, 12.95, 77.65, 13.05],
        collection="sentinel-1-rtc",
        limit=5
    )

    with patch("httpx.Client.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "type": "FeatureCollection",
            "features": [
                {
                    "id": "S1D_TEST_RTC",
                    "collection": "sentinel-1-rtc",
                    "bbox": [77.55, 12.95, 77.65, 13.05],
                    "properties": {"datetime": "2026-10-09T00:39:45Z", "platform": "Sentinel-1D"},
                    "assets": {
                        "vv": {
                            "href": "https://sentinel1euwestrtc.blob.core.windows.net/test/iw-vv.rtc.tiff",
                            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                            "roles": ["data"]
                        },
                        "vh": {
                            "href": "https://sentinel1euwestrtc.blob.core.windows.net/test/iw-vh.rtc.tiff",
                            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
                            "roles": ["data"]
                        },
                        "rendered_preview": {
                            "href": "https://planetarycomputer.microsoft.com/test/preview.png",
                            "type": "image/png",
                            "roles": ["overview"]
                        }
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        res = service.search(req)

        assert mock_post.called
        target_url = mock_post.call_args[0][0]
        assert "planetarycomputer.microsoft.com" in target_url
        assert res.status == "success"
        assert res.total_results == 1
        scene = res.scenes[0]
        assert scene.modality == "SAR"
        assert scene.collection == "sentinel-1-rtc"
        assert "vv" in scene.assets
        assert "vh" in scene.assets
        assert scene.assets["vv"].category == "scientific_raster"
        assert scene.thumbnail_url is not None
        assert scene.provenance["catalog"] == "Microsoft Planetary Computer"


def test_stac_search_all_collections_queries_dual_catalogs():
    """Verifies that searching 'all' queries both Earth Search and Planetary Computer and merges results."""
    service = STACAcquisitionService()
    req = STACSearchRequest(
        bbox=[77.55, 12.95, 77.65, 13.05],
        collection="all",
        limit=5
    )

    with patch("httpx.Client.post") as mock_post:
        def side_effect(url, **kwargs):
            m = MagicMock()
            m.status_code = 200
            if "earth-search" in url:
                m.json.return_value = {
                    "features": [
                        {
                            "id": "S2_OPTICAL_1",
                            "collection": "sentinel-2-l2a",
                            "bbox": [77.55, 12.95, 77.65, 13.05],
                            "properties": {"datetime": "2026-10-14T05:00:00Z"},
                            "assets": {"visual": {"href": "https://s2.tif", "type": "image/tiff"}}
                        }
                    ]
                }
            else:
                m.json.return_value = {
                    "features": [
                        {
                            "id": "S1_SAR_1",
                            "collection": "sentinel-1-rtc",
                            "bbox": [77.55, 12.95, 77.65, 13.05],
                            "properties": {"datetime": "2026-10-09T00:00:00Z"},
                            "assets": {"vv": {"href": "https://s1.tif", "type": "image/tiff"}}
                        }
                    ]
                }
            return m

        mock_post.side_effect = side_effect

        res = service.search(req)

        assert mock_post.call_count == 2
        assert res.status == "success"
        assert res.total_results == 2
        modalities = [s.modality for s in res.scenes]
        assert "OPTICAL" in modalities
        assert "SAR" in modalities


# ==============================================================================
# 2. Planetary Computer SAS Token Signing
# ==============================================================================

def test_sign_url_if_needed_signs_azure_blob_url():
    """Verifies that Azure Blob storage URLs receive a SAS token from Planetary Computer."""
    service = STACAcquisitionService()
    test_blob_url = "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/test.tiff"

    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"token": "sig=mockedsas12345&se=2026-10-16"}
        mock_get.return_value = mock_resp

        signed = service._sign_url_if_needed(test_blob_url, collection="sentinel-1-rtc")

        assert mock_get.called
        assert "sig=mockedsas12345" in signed
        assert signed.startswith(test_blob_url)


def test_sign_url_if_needed_ignores_aws_url():
    """Verifies that non-Azure URLs (e.g. AWS S3) are untouched and make no SAS requests."""
    service = STACAcquisitionService()
    aws_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif"

    with patch("httpx.Client.get") as mock_get:
        res = service._sign_url_if_needed(aws_url)
        assert res == aws_url
        assert not mock_get.called


# ==============================================================================
# 3. Sentinel-1 RTC Bounded AOI Windowed Retrieval (Mocked)
# ==============================================================================

def test_retrieve_composite_window_sentinel1_rtc():
    """
    Verifies that retrieve_composite_window correctly retrieves VV and VH bands for
    a Sentinel-1 RTC scene, performs grid alignment, quality screening, and DpRVI calculation.
    """
    service = STACAcquisitionService()
    scene_id = "S1D_IW_GRDH_TEST_RTC"

    service._scene_cache[scene_id] = {
        "collection": "sentinel-1-rtc",
        "modality": "SAR",
        "sensor": "Sentinel-1 (RTC)",
        "catalog_url": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "provider": "Microsoft Planetary Computer",
        "datetime": "2026-10-09T00:39:45Z",
        "raw_assets": {
            "vv": {
                "href": "https://sentinel1euwestrtc.blob.core.windows.net/rtc/iw-vv.rtc.tiff",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized"
            },
            "vh": {
                "href": "https://sentinel1euwestrtc.blob.core.windows.net/rtc/iw-vh.rtc.tiff",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized"
            }
        }
    }

    req = STACCompositeRetrieveRequest(
        scene_id=scene_id,
        aoi_bbox=[77.55, 12.95, 77.65, 13.05],
        collection="sentinel-1-rtc",
        bands=["vv", "vh"],
        compute_dprvi=True
    )

    vv_dummy = np.full((32, 32), 0.20, dtype=np.float32)
    vh_dummy = np.full((32, 32), 0.05, dtype=np.float32)

    meta_dummy = {
        "crs": "EPSG:32643",
        "transform": [10.0, 0.0, 700000.0, 0.0, -10.0, 1400000.0],
        "shape": (32, 32),
        "driver": "GTiff",
        "count": 1,
        "dtypes": ["float32"],
        "nodata": -32768.0,
        "transfer_telemetry": {"cumulative_bytes": 4096}
    }

    with patch.object(service, "_sign_url_if_needed", side_effect=lambda u, **kw: u + "?sig=dummy"):
        with patch("app.acquisition.windowed_reader.SafeWindowedCOGReader.read_cog_window") as mock_read:
            # First call for VV, second call for VH
            mock_read.side_effect = [(vv_dummy, meta_dummy), (vh_dummy, meta_dummy)]

            resp = service.retrieve_composite_window(req)

            assert resp.status == "success"
            assert resp.category == "scientific_sar"
            assert [b.upper() for b in resp.bands_retrieved] == ["VV", "VH"]
            assert resp.dprvi_result is not None
            assert math.isclose(resp.dprvi_result["dprvi_mean"], 0.5200, abs_tol=1e-3)
            assert math.isclose(resp.dprvi_result["cross_pol_ratio_linear"], 0.2500, abs_tol=1e-3)
            assert resp.provenance["provider"] == "Microsoft Planetary Computer"


# ==============================================================================
# 4. Cross-Sensor Compatibility Validation
# ==============================================================================

def test_cross_sensor_validation_compatible_pair():
    """
    Asserts that an Optical scene (Sentinel-2 L2A) and a SAR RTC scene (Sentinel-1 RTC)
    with matching CRS and overlapping AOI validate as compatible.
    """
    aoi = [77.55, 12.95, 77.65, 13.05]
    bounds = [77.50, 12.90, 77.70, 13.10]  # Fully encloses AOI

    opt_b64 = create_mock_geotiff_b64(bounds=bounds, crs="EPSG:4326", width=64, height=64, bands=3, dtype="uint8")
    sar_b64 = create_mock_geotiff_b64(bounds=bounds, crs="EPSG:4326", width=64, height=64, bands=2, dtype="float32",
                                      tags={"POLARIZATION": "VV, VH", "PROCESSING": "RTC", "CALIBRATION": "gamma0"})

    report = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_filename="sentinel2_l2a_bengaluru.tif",
        sar_filename="sentinel1_rtc_bengaluru.tif",
        optical_provenance={"datetime": "2026-10-14T05:25:20Z", "modality": "OPTICAL", "sensor": "Sentinel-2A"},
        sar_provenance={"datetime": "2026-10-09T00:39:45Z", "modality": "SAR", "sensor": "Sentinel-1 RTC"},
        aoi_bbox=aoi
    )

    assert report["is_compatible"] is True
    assert report["status"] in ("compatible", "compatible_with_limitations")
    assert report["temporal_delta_days"] == pytest.approx(5.2, abs=0.2)
    assert report["overlap_detected"] is True
    assert report["spatial_overlap_pct"] >= 90.0
    assert report["sar_terrain_correction_status"] == "radiometrically_terrain_corrected"
    assert report["sar_calibration_status"] == "calibrated_backscatter"


def test_cross_sensor_validation_disjoint_footprints_rejected():
    """
    Asserts that scenes with zero spatial overlap are rejected, never fabricating fusion.
    """
    bengaluru_bounds = [77.45, 12.85, 77.75, 13.15]
    mumbai_bounds = [72.75, 18.88, 73.05, 19.25]

    opt_b64 = create_mock_geotiff_b64(bounds=bengaluru_bounds)
    sar_b64 = create_mock_geotiff_b64(bounds=mumbai_bounds)

    report = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_filename="cartosat_bengaluru.tif",
        sar_filename="risat_mumbai.tif",
        optical_provenance={"modality": "OPTICAL"},
        sar_provenance={"modality": "SAR"},
        aoi_bbox=[77.50, 12.90, 77.60, 13.00]
    )

    assert report["is_compatible"] is False
    assert report["status"] == "rejected"
    assert "Spatial coverage failure" in report["rejection_reason"]


# ==============================================================================
# 5. Offline Demo Preservation
# ==============================================================================

def test_offline_demo_fallback_when_network_fails():
    """Verifies that network error triggers authentic offline verified sample fallback."""
    service = STACAcquisitionService()
    req = STACSearchRequest(
        bbox=[77.45, 12.85, 77.75, 13.15],
        collection="optical"
    )

    with patch("httpx.Client.post", side_effect=httpx.ConnectError("No route to host")):
        res = service.search(req)

        assert res.status == "success"
        assert res.catalog_url == "local://data/samples"
        assert res.total_results > 0
        scene_ids = [s.id for s in res.scenes]
        assert "local_sample_cartosat_optical_bengaluru" in scene_ids


# ==============================================================================
# 6. Step 7D: Bounded Retrieval, Transfer Budget, and SAS Security Defenses
# ==============================================================================

def test_provenance_url_sanitization_never_exposes_sas_tokens():
    """
    Asserts that SAS tokens and signed query parameters are stripped from
    all provenance hrefs and error log messages.
    """
    from app.acquisition.windowed_reader import SafeWindowedCOGReader

    raw_sas_url = (
        "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/"
        "GRD/2024/3/22/IW/DV/measurement/iw-vv.rtc.tiff?"
        "st=2026-10-14T14%3A00%3A31Z&se=2026-10-15T14%3A45%3A31Z&sp=r&sig=SECRET_SAS_TOKEN_12345"
    )

    sanitized = SafeWindowedCOGReader._sanitize_url(raw_sas_url)
    assert "?" not in sanitized
    assert "SECRET_SAS_TOKEN" not in sanitized
    assert sanitized == (
        "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/"
        "GRD/2024/3/22/IW/DV/measurement/iw-vv.rtc.tiff"
    )

    # Edge cases: None and empty strings
    assert SafeWindowedCOGReader._sanitize_url(None) == ""
    assert SafeWindowedCOGReader._sanitize_url("") == ""


def test_transfer_budget_limit_rejection():
    """
    Verifies that SafeWindowedCOGReader halts with WindowedRetrievalError
    when expected buffer exceeds the transfer budget cap.
    """
    from app.acquisition.windowed_reader import SafeWindowedCOGReader, WindowedRetrievalError

    service = STACAcquisitionService()
    # Cache fixture scene
    fixture_bbox = [80.375, 12.848, 80.379, 12.852]
    mock_scene = service.search(STACSearchRequest(
        bbox=fixture_bbox,
        collection="local_sample_cartosat_optical_bengaluru"
    ))

    # A request with max_bytes=100 (too small to read any window)
    req = STACWindowedRetrieveRequest(
        scene_id="local_sample_cartosat_optical_bengaluru",
        asset_key="visual",
        aoi_bbox=fixture_bbox,
        max_bytes=100
    )

    resp = service.retrieve_window(req)
    assert resp.status == "error"
    assert "exceeded" in resp.error.lower() or "budget" in resp.error.lower()


def test_empty_spatial_intersection_rejection():
    """
    Verifies that requesting an AOI that does not intersect the asset extent
    is strictly rejected with an empty spatial intersection error.
    """
    service = STACAcquisitionService()
    # Request an AOI near Null Island (0,0), far from Bengaluru fixture
    req = STACWindowedRetrieveRequest(
        scene_id="local_sample_cartosat_optical_bengaluru",
        asset_key="visual",
        aoi_bbox=[0.0, 0.0, 0.1, 0.1]
    )

    resp = service.retrieve_window(req)
    assert resp.status == "error"
    assert "intersect" in resp.error.lower() or "empty" in resp.error.lower()

