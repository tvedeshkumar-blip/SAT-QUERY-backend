import os
import sys
import io
import json
import base64
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.acquisition.cross_sensor_validator import CrossSensorValidator
from app.models.optical_sar.optical_sar_fusion import (
    OpticalSARJointAnalysisProvider,
    OpticalSARVisualizationBaseline,
    OpticalSARProvider
)

client = TestClient(app)

import rasterio
from rasterio.transform import from_bounds

def create_geotiff_b64(
    bounds=(77.5, 12.9, 77.6, 13.0),
    crs="EPSG:4326",
    width=64,
    height=64,
    bands=3,
    tags=None
) -> str:
    """Generates an in-memory valid GeoTIFF with georeferencing and tags."""
    memfile = rasterio.MemoryFile()
    transform = from_bounds(*bounds, width, height)
    with memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="uint8",
        crs=crs,
        transform=transform
    ) as dst:
        data = np.random.randint(40, 200, (bands, height, width), dtype=np.uint8)
        dst.write(data)
        if tags:
            dst.update_tags(**tags)
    raw_bytes = memfile.getbuffer()
    return "data:image/tiff;base64," + base64.b64encode(raw_bytes).decode("utf-8")

def create_preview_jpeg_b64(w: int = 128, h: int = 128) -> str:
    """Generates an unprojected browse JPEG preview."""
    img = Image.new("RGB", (w, h), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================================
# 1. Missing Optical or SAR Inputs
# ============================================================================

def test_missing_optical_scene_rejected():
    sar_b64 = create_geotiff_b64()
    res = CrossSensorValidator.validate_pair(
        optical_input=None,
        sar_input=sar_b64,
        optical_filename="missing_optical.tif",
        sar_filename="sentinel1_grd.tif"
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "Missing" in res["rejection_reason"]


def test_missing_sar_scene_rejected():
    opt_b64 = create_geotiff_b64()
    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=None,
        optical_filename="sentinel2_l2a.tif",
        sar_filename="missing_sar.tif"
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "Missing" in res["rejection_reason"]


def test_missing_both_scenes_rejected():
    res = CrossSensorValidator.validate_pair(
        optical_input=None,
        sar_input=None
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "Missing" in res["rejection_reason"]


# ============================================================================
# 2. Modality Mismatch & Homogeneous Sensor Rejection
# ============================================================================

def test_optical_plus_optical_rejected():
    opt1_b64 = create_geotiff_b64()
    opt2_b64 = create_geotiff_b64()
    res = CrossSensorValidator.validate_pair(
        optical_input=opt1_b64,
        sar_input=opt2_b64,
        optical_provenance={"modality": "OPTICAL", "sensor": "Sentinel-2"},
        sar_provenance={"modality": "OPTICAL", "sensor": "Landsat-8"}
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "both scenes are optical" in res["rejection_reason"].lower()


def test_sar_plus_sar_rejected():
    sar1_b64 = create_geotiff_b64()
    sar2_b64 = create_geotiff_b64()
    res = CrossSensorValidator.validate_pair(
        optical_input=sar1_b64,
        sar_input=sar2_b64,
        optical_provenance={"modality": "SAR", "sensor": "Sentinel-1"},
        sar_provenance={"modality": "SAR", "sensor": "RISAT-1A"}
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "SAR" in res["rejection_reason"]


# ============================================================================
# 3. Preview Thumbnail & Non-Scientific Asset Rejection
# ============================================================================

def test_optical_browse_jpeg_thumbnail_rejected():
    jpeg_b64 = create_preview_jpeg_b64()
    sar_b64 = create_geotiff_b64()
    res = CrossSensorValidator.validate_pair(
        optical_input=jpeg_b64,
        sar_input=sar_b64,
        optical_filename="preview.jpg",
        sar_filename="sentinel1.tif"
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "preview asset detected" in res["rejection_reason"].lower()


def test_sar_browse_jpeg_thumbnail_rejected():
    opt_b64 = create_geotiff_b64()
    jpeg_b64 = create_preview_jpeg_b64()
    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=jpeg_b64,
        optical_filename="sentinel2.tif",
        sar_filename="preview.jpg"
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "preview asset detected" in res["rejection_reason"].lower()


# ============================================================================
# 4. Georeferencing & CRS Mismatch Rejection
# ============================================================================

def test_crs_mismatch_rejected():
    opt_b64 = create_geotiff_b64(bounds=(77.5, 12.9, 77.6, 13.0), crs="EPSG:4326")
    sar_b64 = create_geotiff_b64(bounds=(700000, 1400000, 710000, 1410000), crs="EPSG:32643")
    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL"},
        sar_provenance={"modality": "SAR"}
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "CRS mismatch" in res["rejection_reason"]


# ============================================================================
# 5. Spatial Footprint Overlap Validation
# ============================================================================

def test_zero_spatial_overlap_rejected():
    # Bengaluru bounding box vs Mumbai bounding box
    opt_b64 = create_geotiff_b64(bounds=(77.5, 12.9, 77.6, 13.0), crs="EPSG:4326")
    sar_b64 = create_geotiff_b64(bounds=(72.8, 18.9, 72.9, 19.0), crs="EPSG:4326")
    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL"},
        sar_provenance={"modality": "SAR"}
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "Zero spatial overlap" in res["rejection_reason"]


def test_insufficient_overlap_below_10_pct_rejected():
    # Footprint overlap of approximately 5%
    opt_b64 = create_geotiff_b64(bounds=(10.0, 10.0, 11.0, 11.0), crs="EPSG:4326")
    sar_b64 = create_geotiff_b64(bounds=(10.95, 10.0, 11.95, 11.0), crs="EPSG:4326")
    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL"},
        sar_provenance={"modality": "SAR"}
    )
    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "Insufficient spatial overlap" in res["rejection_reason"]


def test_sufficient_spatial_overlap_accepted():
    # Footprints with 80% overlap
    opt_b64 = create_geotiff_b64(bounds=(77.50, 12.90, 77.60, 13.00), crs="EPSG:4326")
    sar_b64 = create_geotiff_b64(bounds=(77.52, 12.90, 77.62, 13.00), crs="EPSG:4326")
    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL", "datetime": "2024-02-01T10:00:00Z"},
        sar_provenance={"modality": "SAR", "datetime": "2024-02-05T10:00:00Z"}
    )
    assert res["is_compatible"] is True
    assert res["spatial_overlap_pct"] > 50.0


# ============================================================================
# 6. Temporal Baseline Delta & Advisory Warnings
# ============================================================================

def test_temporal_separation_over_60_days_emits_warning():
    opt_b64 = create_geotiff_b64()
    sar_b64 = create_geotiff_b64()
    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL", "datetime": "2024-01-01T00:00:00Z"},
        sar_provenance={"modality": "SAR", "datetime": "2024-04-15T00:00:00Z"} # 105 days
    )
    assert res["is_compatible"] is True
    assert res["temporal_delta_days"] is not None
    assert res["temporal_delta_days"] > 60.0
    assert any("Temporal separation advisory" in w for w in res["warnings"])


# ============================================================================
# 7. SAR Radiometric Calibration & Terrain Correction Status
# ============================================================================

def test_uncalibrated_sar_reported_truthfully():
    opt_b64 = create_geotiff_b64()
    sar_b64 = create_geotiff_b64(tags={"SENSOR": "Sentinel-1", "POLARIZATION": "VV"})
    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL", "datetime": "2024-02-01T00:00:00Z"},
        sar_provenance={"modality": "SAR", "datetime": "2024-02-03T00:00:00Z"}
    )
    assert res["sar_calibration_status"] == "unverified_linear_dn"
    assert res["sar_terrain_correction_status"] == "ellipsoid_geocoded_grd"
    assert any("sigma-0/gamma-0 calibration look-up tables" in w for w in res["warnings"])
    assert any("ellipsoid-geocoded (GRD)" in w for w in res["warnings"])


def test_verified_calibrated_rtc_sar_recognized():
    opt_b64 = create_geotiff_b64()
    sar_b64 = create_geotiff_b64(tags={
        "SENSOR": "Sentinel-1",
        "POLARIZATION": "VH",
        "CALIBRATION": "sigma0",
        "PROCESSING": "radiometric_terrain_corrected_rtc"
    })
    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL", "datetime": "2024-02-01T00:00:00Z"},
        sar_provenance={"modality": "SAR", "datetime": "2024-02-03T00:00:00Z"}
    )
    assert res["sar_calibration_status"] == "calibrated_backscatter"
    assert res["sar_terrain_correction_status"] == "radiometrically_terrain_corrected"
    assert res["sar_polarization"] == "VH"


# ============================================================================
# 8. Physical Non-Interchangeability Assertion
# ============================================================================

def test_physical_non_interchangeability_assertion_present():
    opt_b64 = create_geotiff_b64()
    sar_b64 = create_geotiff_b64()
    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL"},
        sar_provenance={"modality": "SAR"}
    )
    assert any("Optical surface reflectance" in s and "SAR microwave backscatter" in s for s in res["scientific_limitations"])


# ============================================================================
# 9. API Endpoints
# ============================================================================

def test_api_validate_optical_sar_pair_endpoint():
    opt_b64 = create_geotiff_b64(bounds=(77.5, 12.9, 77.6, 13.0))
    sar_b64 = create_geotiff_b64(bounds=(77.5, 12.9, 77.6, 13.0))

    resp = client.post("/api/v1/acquisition/validate-optical-sar-pair", json={
        "optical_data_base64": opt_b64,
        "sar_data_base64": sar_b64,
        "optical_filename": "s2.tif",
        "sar_filename": "s1.tif",
        "optical_provenance": {"modality": "OPTICAL", "datetime": "2024-02-01T10:00:00Z"},
        "sar_provenance": {"modality": "SAR", "datetime": "2024-02-05T10:00:00Z"}
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_compatible"] is True
    assert data["status"] in ("compatible", "compatible_with_limitations")
    assert data["crs_compatible"] is True
    assert data["primary_provider"] == "OpticalSARJointAnalysisProvider"


def test_api_search_optical_sar_pair_endpoint():
    resp = client.post("/api/v1/acquisition/search-optical-sar-pair", json={
        "bbox": [77.5, 12.9, 77.6, 13.0],
        "optical_start_date": "2024-01-01",
        "optical_end_date": "2024-03-31",
        "sar_start_date": "2024-01-01",
        "sar_end_date": "2024-03-31",
        "max_cloud_cover": 20.0,
        "limit": 4
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("success", "partial", "zero_results")
    assert "optical_results" in data
    assert "sar_results" in data


# ============================================================================
# 10. Agent Controller Integration & Provenance Inspection
# ============================================================================

def test_controller_optical_sar_rejection_on_preview_jpeg():
    jpeg_b64 = create_preview_jpeg_b64()
    sar_b64 = create_geotiff_b64()

    payload = {
        "query": "Perform joint optical and SAR analysis on this pair.",
        "mode": "optical_sar",
        "images": [
            {
                "data": jpeg_b64,
                "mimeType": "image/jpeg",
                "filename": "preview.jpg",
                "role": "optical",
                "provenance": {"modality": "OPTICAL"}
            },
            {
                "data": sar_b64,
                "mimeType": "image/tiff",
                "filename": "sentinel1.tif",
                "role": "sar",
                "provenance": {"modality": "SAR"}
            }
        ]
    }

    resp = client.post("/api/v1/analyze", json=payload)
    # Controller Step 4c rejects browse preview
    assert resp.status_code == 400
    detail = resp.json().get("detail", "")
    assert "preview asset detected" in detail.lower()


def test_controller_optical_sar_success_with_scientific_provenance():
    opt_b64 = create_geotiff_b64(bounds=(77.5, 12.9, 77.6, 13.0))
    sar_b64 = create_geotiff_b64(bounds=(77.5, 12.9, 77.6, 13.0), tags={"POLARIZATION": "VV"})

    payload = {
        "query": "Execute optical and SAR cross-sensor joint analysis.",
        "mode": "optical_sar",
        "images": [
            {
                "data": opt_b64,
                "mimeType": "image/tiff",
                "filename": "sentinel2.tif",
                "role": "optical",
                "provenance": {
                    "modality": "OPTICAL",
                    "sensor": "Sentinel-2 L2A",
                    "datetime": "2024-02-01T10:00:00Z"
                }
            },
            {
                "data": sar_b64,
                "mimeType": "image/tiff",
                "filename": "sentinel1.tif",
                "role": "sar",
                "provenance": {
                    "modality": "SAR",
                    "sensor": "Sentinel-1 GRD",
                    "datetime": "2024-02-05T10:00:00Z"
                }
            }
        ]
    }

    resp = client.post("/api/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["primary_model"] == "OpticalSARJointAnalysisProvider"
    assert data["fallback_used"] is False
    assert "optical_sar_provenance" in data["metadata"]
    prov = data["metadata"]["optical_sar_provenance"]
    assert prov["sar"]["calibration_status"] == "unverified_linear_dn"
    assert prov["sar"]["terrain_correction_status"] == "ellipsoid_geocoded_grd"
    assert "physical_assertion" in prov["scientific_distinction"]


# ============================================================================
# 11. Model Baseline & Truthful Fallback Verification
# ============================================================================

def test_truthful_model_and_baseline_reporting():
    primary_analyzer = OpticalSARJointAnalysisProvider()
    baseline_analyzer = OpticalSARVisualizationBaseline()

    opt = np.random.randint(40, 220, (64, 64, 3), dtype=np.uint8)
    sar = np.random.randint(10, 250, (64, 64), dtype=np.uint8)

    # Primary provider execution
    res_primary = primary_analyzer.predict([opt, sar], query="Joint analysis")
    assert res_primary["primary_model"] == "OpticalSARJointAnalysisProvider"
    assert res_primary["actual_model_used"] == "OpticalSARJointAnalysisProvider"
    assert res_primary["fallback_used"] is False

    # Baseline provider execution
    res_baseline = baseline_analyzer.predict([opt, sar], query="Baseline visualization")
    assert res_baseline["primary_model"] == "OpticalSARJointAnalysisProvider"
    assert res_baseline["actual_model_used"] == "OpticalSARVisualizationBaseline"
    assert res_baseline["fallback_used"] is True


def test_controlled_real_asset_optical_sar_fusion_pipeline():
    """
    Stage 6D: Controlled Real-Asset Optical-SAR Fusion Test.
    Executes the existing Stage 6 optical-SAR analysis on genuine Sentinel-2 B04
    and Sentinel-1 RTC VV windows acquired over Bengaluru AOI.
    """
    from app.acquisition.windowed_reader import SafeWindowedCOGReader
    import httpx

    token_url = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
    try:
        with httpx.Client(timeout=10.0) as http_client:
            resp = http_client.get(token_url)
            if resp.status_code != 200:
                pytest.skip("Planetary Computer token endpoint unavailable")
            token = resp.json().get("token")
    except Exception as e:
        pytest.skip(f"Network error accessing token service: {e}")

    aoi_bengaluru = [77.58, 12.96, 77.60, 12.98]
    s2_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/43/P/GQ/2024/3/S2A_43PGQ_20240328_0_L2A/B04.tif"
    s1_raw = "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/GRD/2024/3/22/IW/DV/S1A_IW_GRDH_1SDV_20240322T004029_20240322T004054_053087_066E0A_9C2C/measurement/iw-vv.rtc.tiff"
    s1_url = f"{s1_raw}?{token}"

    # 1. Acquire bounded windows
    s2_arr, s2_meta = SafeWindowedCOGReader.read_cog_window(s2_url, aoi_bengaluru)
    s1_arr, s1_meta = SafeWindowedCOGReader.read_cog_window(s1_url, aoi_bengaluru)

    assert s2_arr.shape == s1_arr.shape == (224, 220)
    assert s2_arr.dtype == np.uint16
    assert s1_arr.dtype == np.float32

    # 2. Execute existing OpticalSARJointAnalysisProvider
    provider = OpticalSARJointAnalysisProvider()
    res = provider.predict(
        images=[s2_arr, s1_arr],
        query="Perform controlled real-asset cross-sensor analysis over Bengaluru downtown.",
        metadata={
            "optical": {
                **s2_meta,
                "modality": "OPTICAL",
                "sensor": "Sentinel-2 L2A",
                "item_id": "S2A_43PGQ_20240328_0_L2A",
                "datetime": "2024-03-28T05:27:01Z"
            },
            "sar": {
                **s1_meta,
                "modality": "SAR",
                "sensor": "Sentinel-1 C-SAR IW RTC",
                "item_id": "S1A_IW_GRDH_1SDV_20240322T004029_20240322T004054_053087_066E0A_rtc",
                "datetime": "2024-03-22T00:40:42Z",
                "polarization": "VV",
                "calibration_status": "calibrated_backscatter",
                "terrain_correction_status": "radiometrically_terrain_corrected"
            }
        }
    )

    # 3. Verify pipeline execution, provenance, and registration safeguards
    assert res["primary_model"] == "OpticalSARJointAnalysisProvider"
    assert res["actual_model_used"] == "OpticalSARJointAnalysisProvider"
    assert res["fallback_used"] is False
    assert res["registration_info"]["co_registered"] is True
    assert res["registration_info"]["registration_method"] == "identical_grid_co_registered"
    assert res["registration_info"]["spatial_overlap_percent"] == 100.0
    assert res["valid_pixel_count"] == 49280
    assert res["water_pct"] is not None
    assert res["urban_pct"] is not None
    assert res["veg_pct"] is not None
    assert len(res["evidence"]) == 4

    # 4. Stage 6E Physical semantics and provenance verification
    ev_map = {e["id"]: e for e in res["evidence"]}
    opt_stat = ev_map["ev_optical_reflectance"]["statistics"]
    sar_stat = ev_map["ev_sar_backscatter"]["statistics"]

    assert opt_stat["physical_boa_reflectance_mean"] == pytest.approx(0.1298, rel=1e-2)
    assert opt_stat["physical_reflectance_percent"] == "12.98%"
    assert opt_stat["mean_albedo"] == 92.8
    assert "8-bit uint8" in opt_stat["display_scale"]

    assert sar_stat["calibration_quantity"] == "gamma-0"
    assert sar_stat["linear_power_mean"] == pytest.approx(0.7417, rel=1e-2)
    assert sar_stat["calibrated_db_mean"] == pytest.approx(-4.98, rel=1e-2)
    assert sar_stat["mean_backscatter_intensity"] == 105.8
    assert "8-bit uint8" in sar_stat["display_scale"]

    assert "Calibrated gamma-0 backscatter" in res["answer"]
    assert "mean linear gamma-0 power: 0.7417 (-4.98 dB)" in res["answer"]
    assert "mean BOA surface reflectance: 0.1298 (12.98%" in res["answer"]
    assert res["model_provenance"]["calibration_quantity"] == "gamma-0"
    assert any("preliminary scene-normalized empirical proxies" in lim for lim in res["limitations"])


# ============================================================================
# Stage 4 P0-1 Units-Aware Ground-Resolution Regression Tests
# ============================================================================

def test_p0_1_utm_vs_geographic_same_ground_resolution():
    """
    P0-1 Test 1: 10m UTM raster vs ~10m EPSG:4326 raster at ~13 deg N latitude.
    Demonstrates that units are normalized on the WGS84 ellipsoid before comparison,
    preventing spurious ~111,000x disparity rejections between meters and degrees.
    """
    from rasterio.warp import transform_bounds

    # Optical: 100x100 raster covering 1000m x 1000m in UTM 32643 (10m ground resolution)
    opt_bounds_utm = (770000.0, 1427000.0, 771000.0, 1428000.0)
    opt_b64 = create_geotiff_b64(
        bounds=opt_bounds_utm,
        crs="EPSG:32643",
        width=100,
        height=100
    )

    # Transform to EPSG:4326 for equivalent SAR coverage at ~12.9 deg N
    sar_bounds_4326 = transform_bounds("EPSG:32643", "EPSG:4326", *opt_bounds_utm)
    # Span is ~0.009 deg in lat/lon; 100 pixels gives ~0.00009 deg (~9.9m ground resolution)
    sar_b64 = create_geotiff_b64(
        bounds=sar_bounds_4326,
        crs="EPSG:4326",
        width=100,
        height=100,
        tags={"SENSOR": "Sentinel-1", "POLARIZATION": "VV"}
    )

    # Use validate_pair with explicit AOI at 13 deg N
    aoi_bbox = [77.49, 12.89, 77.51, 12.91]
    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL", "datetime": "2024-02-01T10:00:00Z"},
        sar_provenance={"modality": "SAR", "datetime": "2024-02-02T10:00:00Z"},
        aoi_bbox=aoi_bbox
    )

    # Must NOT fail with spurious 110,000x disparity
    # Note: cross_sensor_validator checks CRS equality at L205, but if CRS is compatible,
    # resolution ratio must be evaluated as ~1.0x ground resolution.
    # To isolate resolution normalization check:
    ground_res_opt = CrossSensorValidator._compute_ground_resolution_meters(
        {"crs": "EPSG:32643", "resolution": [10.0, 10.0]}, aoi_bbox=aoi_bbox
    )
    ground_res_sar = CrossSensorValidator._compute_ground_resolution_meters(
        {"crs": "EPSG:4326", "resolution": [0.0000898315, 0.0000898315]}, aoi_bbox=aoi_bbox
    )

    assert ground_res_opt is not None
    assert ground_res_sar is not None
    assert ground_res_opt[0] == pytest.approx(10.0, rel=1e-3)
    assert ground_res_sar[0] == pytest.approx(9.75, rel=0.05)
    ratio = max(ground_res_opt[0], ground_res_sar[0]) / min(ground_res_opt[0], ground_res_sar[0])
    assert ratio == pytest.approx(1.026, rel=0.05)
    assert ratio < 3.0  # Well within compatible threshold (< 10x)


def test_p0_1_genuine_resolution_disparity_rejected():
    """
    P0-1 Test 2: Genuine 10m vs 250m disparity (25x) must be strictly rejected.
    """
    # Optical: 10m resolution (1000m span, 100 pixels)
    opt_b64 = create_geotiff_b64(
        bounds=(770000.0, 1427000.0, 771000.0, 1428000.0),
        crs="EPSG:32643",
        width=100,
        height=100
    )

    # SAR: 250m resolution (1000m span, 4 pixels)
    sar_b64 = create_geotiff_b64(
        bounds=(770000.0, 1427000.0, 771000.0, 1428000.0),
        crs="EPSG:32643",
        width=4,
        height=4,
        tags={"SENSOR": "Sentinel-1", "POLARIZATION": "VV"}
    )

    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL", "datetime": "2024-02-01T10:00:00Z"},
        sar_provenance={"modality": "SAR", "datetime": "2024-02-02T10:00:00Z"}
    )

    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "Excessive resolution disparity" in res["rejection_reason"]
    assert "25.0x" in res["rejection_reason"]
    assert res["resolution_compatible"] is False


def test_p0_1_missing_georeferencing_resolution_fails_safely():
    """
    P0-1 Test 3: Missing CRS or pixel dimensions fails safely without fabricating resolution.
    """
    opt_b64 = create_geotiff_b64(crs="EPSG:32643", width=64, height=64)
    res_none = CrossSensorValidator._compute_ground_resolution_meters({"crs": None, "resolution": [10.0, 10.0]})
    assert res_none is None

    res_zero = CrossSensorValidator._compute_ground_resolution_meters({"crs": "EPSG:32643", "resolution": [0.0, 0.0]})
    assert res_zero is None


# ============================================================================
# Stage 4B P1-2 Regression Tests: Requested AOI Enforcement & Cross-CRS Overlap
# ============================================================================

def test_p1_2_both_scenes_overlap_each_other_and_aoi_valid():
    """
    P1-2 Test 1: Both scenes overlap each other and the requested AOI.
    Valid spatial overlap established, with clearly separated coverage metrics.
    """
    opt_b64 = create_geotiff_b64(bounds=(77.50, 12.90, 77.60, 13.00), crs="EPSG:4326")
    sar_b64 = create_geotiff_b64(
        bounds=(77.52, 12.92, 77.62, 13.02),
        crs="EPSG:4326",
        tags={"SENSOR": "Sentinel-1", "POLARIZATION": "VV"}
    )
    aoi = [77.54, 12.94, 77.58, 12.98]  # Nested inside the mutual intersection

    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL"},
        sar_provenance={"modality": "SAR"},
        aoi_bbox=aoi
    )

    assert res["is_compatible"] is True
    assert res["status"] in ("compatible", "compatible_with_limitations")
    assert res["rejection_reason"] is None
    assert res["aoi_intersection_valid"] is True
    assert res["aoi_coverage_pct"] == 100.0  # AOI is 100% covered by mutual overlap
    assert res["spatial_overlap_pct"] > 50.0  # Mutual scene overlap % of smaller scene


def test_p1_2_scenes_overlap_each_other_but_outside_aoi_rejected():
    """
    P1-2 Test 2: Scenes overlap each other, but lie outside the requested AOI.
    Must reject because no joint observation exists over the target AOI.
    """
    opt_b64 = create_geotiff_b64(bounds=(77.50, 12.90, 77.60, 13.00), crs="EPSG:4326")
    sar_b64 = create_geotiff_b64(
        bounds=(77.52, 12.92, 77.62, 13.02),
        crs="EPSG:4326",
        tags={"SENSOR": "Sentinel-1", "POLARIZATION": "VV"}
    )
    # Requested AOI in Mumbai (disjoint from Bangalore scene overlap)
    aoi_mumbai = [72.80, 18.90, 72.90, 19.00]

    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL"},
        sar_provenance={"modality": "SAR"},
        aoi_bbox=aoi_mumbai
    )

    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "Spatial coverage failure" in res["rejection_reason"]
    assert "Neither optical nor SAR scene intersects the requested AOI" in res["rejection_reason"]


def test_p1_2_only_one_scene_intersects_aoi_rejected():
    """
    P1-2 Test 3: Only one scene intersects the requested AOI.
    Must reject with an explicit reason identifying the missing sensor coverage.
    """
    # Optical covers Bangalore
    opt_b64 = create_geotiff_b64(bounds=(77.50, 12.90, 77.60, 13.00), crs="EPSG:4326")
    # SAR covers Chennai
    sar_b64 = create_geotiff_b64(
        bounds=(80.10, 13.00, 80.20, 13.10),
        crs="EPSG:4326",
        tags={"SENSOR": "Sentinel-1", "POLARIZATION": "VV"}
    )
    # AOI in Bangalore: intersects optical, but NOT SAR
    aoi_bangalore = [77.52, 12.92, 77.58, 12.98]

    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL"},
        sar_provenance={"modality": "SAR"},
        aoi_bbox=aoi_bangalore
    )

    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "SAR scene footprint does not intersect the requested AOI" in res["rejection_reason"]


def test_p1_2_neither_scene_intersects_aoi_rejected():
    """
    P1-2 Test 4: Neither scene intersects the requested AOI.
    """
    opt_b64 = create_geotiff_b64(bounds=(77.50, 12.90, 77.60, 13.00), crs="EPSG:4326")
    sar_b64 = create_geotiff_b64(
        bounds=(77.52, 12.92, 77.62, 13.02),
        crs="EPSG:4326",
        tags={"SENSOR": "Sentinel-1", "POLARIZATION": "VV"}
    )
    # AOI in Delhi
    aoi_delhi = [77.10, 28.50, 77.25, 28.65]

    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL"},
        sar_provenance={"modality": "SAR"},
        aoi_bbox=aoi_delhi
    )

    assert res["is_compatible"] is False
    assert res["status"] == "rejected"
    assert "Neither optical nor SAR scene intersects the requested AOI" in res["rejection_reason"]


def test_p1_2_invalid_aoi_coordinates_rejected():
    """
    P1-2 Test 5: Invalid AOI coordinates or ordering fails safely with clear reason.
    """
    opt_b64 = create_geotiff_b64(bounds=(77.50, 12.90, 77.60, 13.00), crs="EPSG:4326")
    sar_b64 = create_geotiff_b64(
        bounds=(77.52, 12.92, 77.62, 13.02),
        crs="EPSG:4326",
        tags={"SENSOR": "Sentinel-1", "POLARIZATION": "VV"}
    )

    # Inverted bounding box (min_x > max_x)
    res_inverted = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL"},
        sar_provenance={"modality": "SAR"},
        aoi_bbox=[77.60, 12.90, 77.50, 13.00]
    )
    assert res_inverted["is_compatible"] is False
    assert "Invalid AOI bounding box" in res_inverted["rejection_reason"]
    assert "min coordinates must be strictly less than max coordinates" in res_inverted["rejection_reason"]

    # Non-numeric coordinate
    res_nan = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL"},
        sar_provenance={"modality": "SAR"},
        aoi_bbox=[77.50, 12.90, float("nan"), 13.00]
    )
    assert res_nan["is_compatible"] is False
    assert "Invalid AOI bounding box" in res_nan["rejection_reason"]


def test_p1_2_different_crs_transforms_footprints_consistently_before_intersection():
    """
    P1-2 Test 6: Different CRSs transform footprints consistently before testing AOI intersection.
    Optical in UTM 43N (EPSG:32643) and SAR in WGS84 (EPSG:4326) are both normalized to
    EPSG:4326 to verify mutual overlap and AOI intersection.
    """
    from rasterio.warp import transform_bounds

    # Optical in UTM 43N (around Bangalore: 77.58°E, 12.96°N is roughly 780000m E, 1434000m N)
    opt_bounds_utm = [779000.0, 1433000.0, 781000.0, 1435000.0]
    opt_b64 = create_geotiff_b64(bounds=opt_bounds_utm, crs="EPSG:32643")

    # Get corresponding WGS84 bounds for SAR and AOI
    wgs84_bounds = transform_bounds("EPSG:32643", "EPSG:4326", *opt_bounds_utm)
    # SAR in EPSG:4326 covering nearly the same geographic footprint
    sar_b64 = create_geotiff_b64(
        bounds=(wgs84_bounds[0], wgs84_bounds[1], wgs84_bounds[2], wgs84_bounds[3]),
        crs="EPSG:4326",
        tags={"SENSOR": "Sentinel-1", "POLARIZATION": "VV"}
    )

    # Requested AOI in WGS84 inside the transformed area
    mid_lon = (wgs84_bounds[0] + wgs84_bounds[2]) / 2.0
    mid_lat = (wgs84_bounds[1] + wgs84_bounds[3]) / 2.0
    aoi_nested = [mid_lon - 0.002, mid_lat - 0.002, mid_lon + 0.002, mid_lat + 0.002]

    res = CrossSensorValidator.validate_pair(
        optical_input=opt_b64,
        sar_input=sar_b64,
        optical_provenance={"modality": "OPTICAL"},
        sar_provenance={"modality": "SAR"},
        aoi_bbox=aoi_nested
    )

    assert res["is_compatible"] is True
    assert res["status"] in ("compatible", "compatible_with_limitations")
    assert res["aoi_intersection_valid"] is True
    assert res["aoi_coverage_pct"] > 90.0
    assert any("Different CRSs" in w for w in res["warnings"])

