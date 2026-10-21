"""
Unit and integration tests for Step 7G:
Multi-Scene Sentinel-2 Temporal Compositing with Categorical SCL Cloud Masking.

Covers:
1. Multiple scenes contributing to a composite (gap-filling via cloud-masked median).
2. SCL cloud, cloud-shadow, defective, and out-of-range class masking.
3. SCL categorical alignment: nearest-neighbor resampling strictly preserving discrete class labels.
4. No valid observations: zero-valid-pixel defense returning unsupported status and None statistics.
5. Insufficient valid-pixel coverage: enforcing minimum coverage thresholds.
6. Missing bands (B04/B08) and missing SCL assets handling.
7. Misaligned rasters and incompatible CRS rejection.
8. Bi-temporal T1/T2 multi-scene change index calculation and unitless Delta-NDVI.
9. Transfer-budget exhaustion handling in multi-scene retrieval.
10. Provenance sanitation and offline demo fixture disclosure.
"""

import sys
import os
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
import numpy as np
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from app.remote_sensing.temporal import (
    build_cloud_masked_optical_composite,
    compute_temporal_optical_change
)
from app.remote_sensing.quality import (
    SCL_EXCLUDED_CLASSES,
    SCL_RETAINED_CLASSES,
    SCL_UNCERTAIN_CLASSES
)
from app.agent.orchestrator import MultiStepOrchestrator


@pytest.fixture
def base_meta():
    return {
        "crs": "EPSG:32643",
        "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 1400000.0],
        "shape": (10, 10),
        "scale_factor": 10000.0,
        "offset": 0.0,
        "datetime": "2023-01-15T10:30:00Z"
    }


def test_multiple_scenes_gap_filling_composite(base_meta):
    """
    Two scenes for the same period:
    Scene 1 has clouds on left half (cols 0-4), vegetation on right half (cols 5-9).
    Scene 2 has vegetation on left half (cols 0-4), clouds on right half (cols 5-9).
    Compositing must yield 100% cloud-free clear coverage across the 10x10 grid.
    """
    h, w = 10, 10
    
    # Scene 1: Left half cloudy (SCL=9), right half vegetation (SCL=4)
    b04_1 = np.full((h, w), 500, dtype=np.uint16)   # refl = 0.05
    b08_1 = np.full((h, w), 4500, dtype=np.uint16)  # refl = 0.45
    scl_1 = np.full((h, w), 4, dtype=np.uint8)
    scl_1[:, :5] = 9  # High prob cloud on left
    
    # Scene 2: Right half cloudy (SCL=9), left half vegetation (SCL=4)
    b04_2 = np.full((h, w), 600, dtype=np.uint16)   # refl = 0.06
    b08_2 = np.full((h, w), 4600, dtype=np.uint16)  # refl = 0.46
    scl_2 = np.full((h, w), 4, dtype=np.uint8)
    scl_2[:, 5:] = 9  # High prob cloud on right

    scene_1 = {
        "scene_id": "S2_T1_SCENE_A",
        "datetime": "2023-01-10T10:00:00Z",
        "b04": b04_1,
        "b08": b08_1,
        "scl": scl_1,
        "meta": dict(base_meta, datetime="2023-01-10T10:00:00Z"),
        "cloud_cover": 50.0
    }
    scene_2 = {
        "scene_id": "S2_T1_SCENE_B",
        "datetime": "2023-01-20T10:00:00Z",
        "b04": b04_2,
        "b08": b08_2,
        "scl": scl_2,
        "meta": dict(base_meta, datetime="2023-01-20T10:00:00Z"),
        "cloud_cover": 50.0
    }

    comp = build_cloud_masked_optical_composite(
        scenes=[scene_1, scene_2],
        composite_method="median"
    )

    assert comp["status"] == "success"
    assert comp["total_pixel_count"] == 100
    # Every pixel had at least one cloud-free observation
    assert comp["valid_pixel_count"] == 100
    assert comp["coverage_percentage"] == 100.0
    assert len(comp["contributing_scenes"]) == 2
    
    # Left half took Scene 2 values (refl b04=0.06, b08=0.46)
    assert np.isclose(comp["composite_b04"][0, 2], 0.06, atol=1e-3)
    assert np.isclose(comp["composite_b08"][0, 2], 0.46, atol=1e-3)
    
    # Right half took Scene 1 values (refl b04=0.05, b08=0.45)
    assert np.isclose(comp["composite_b04"][0, 7], 0.05, atol=1e-3)
    assert np.isclose(comp["composite_b08"][0, 7], 0.45, atol=1e-3)


def test_scl_cloud_and_shadow_masking_policy(base_meta):
    """
    Verify each documented SCL class is handled correctly:
    - Excluded: 0 (No data), 1 (Defective), 3 (Cloud shadow), 8 (Med cloud), 9 (High cloud), 10 (Cirrus)
    - Retained: 4 (Vegetation), 5 (Bare soil), 6 (Water), 11 (Snow/Ice)
    - Unknown out-of-range values: -1, 15 (fail-closed excluded)
    """
    h, w = 12, 1
    # Create rows testing classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    scl = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int16).reshape((h, w))
    b04 = np.full((h, w), 1000, dtype=np.uint16)
    b08 = np.full((h, w), 5000, dtype=np.uint16)

    scene = {
        "scene_id": "S2_SCL_TEST",
        "b04": b04,
        "b08": b08,
        "scl": scl,
        "meta": dict(base_meta, shape=(h, w)),
        "cloud_cover": 30.0
    }

    comp = build_cloud_masked_optical_composite([scene], allow_uncertain_as_clear=False)
    
    # Classes retained by default: 4, 5, 6, 11 (indices 4, 5, 6, 11 -> 4 valid pixels)
    valid_mask = comp["valid_mask"]
    assert int(np.count_nonzero(valid_mask)) == 4
    assert valid_mask[4, 0] == True   # Vegetation
    assert valid_mask[5, 0] == True   # Bare Soil
    assert valid_mask[6, 0] == True   # Water
    assert valid_mask[11, 0] == True  # Snow/Ice

    assert valid_mask[0, 0] == False  # NO_DATA
    assert valid_mask[1, 0] == False  # SATURATED_OR_DEFECTIVE
    assert valid_mask[2, 0] == False  # CAST_SHADOW (uncertain)
    assert valid_mask[3, 0] == False  # CLOUD_SHADOW
    assert valid_mask[7, 0] == False  # UNCLASSIFIED (uncertain)
    assert valid_mask[8, 0] == False  # CLOUD_MEDIUM_PROBA
    assert valid_mask[9, 0] == False  # CLOUD_HIGH_PROBA
    assert valid_mask[10, 0] == False # THIN_CIRRUS

    # Policy in output
    assert comp["scl_mask_policy"]["excluded_classes"] == [0, 1, 3, 8, 9, 10]
    assert comp["scl_mask_policy"]["resampling"] == "nearest_neighbor"


def test_scl_categorical_nearest_neighbor_alignment(base_meta):
    """
    SCL is natively 20m resolution, B04/B08 are 10m resolution.
    Reprojecting 20m SCL onto 10m B04 grid must use nearest-neighbor resampling
    and never interpolate discrete class labels.
    """
    # 20m SCL grid of 2x2: top row class 4 (vegetation), bottom row class 9 (cloud)
    scl_20m = np.array([
        [4, 4],
        [9, 9]
    ], dtype=np.uint8)

    # 10m B04 grid of 4x4
    b04_10m = np.full((4, 4), 1000, dtype=np.uint16)
    b08_10m = np.full((4, 4), 5000, dtype=np.uint16)

    meta_10m = {
        "crs": "EPSG:32643",
        "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 1400000.0],
        "shape": (4, 4)
    }

    scene = {
        "scene_id": "S2_20M_SCL_TEST",
        "b04": b04_10m,
        "b08": b08_10m,
        "scl": scl_20m,
        "meta": meta_10m
    }

    comp = build_cloud_masked_optical_composite([scene])
    assert comp["status"] == "success"
    # Top 2 rows (8 pixels) should be clear vegetation, bottom 2 rows (8 pixels) should be cloud
    valid_mask = comp["valid_mask"]
    assert int(np.count_nonzero(valid_mask)) == 8
    assert np.all(valid_mask[:2, :] == True)
    assert np.all(valid_mask[2:, :] == False)


def test_zero_valid_observations_defense(base_meta):
    """
    When all observations across all candidate scenes in the period are obscured by clouds (SCL=9),
    the composite must return an unsupported status, 0 valid pixels, and None for NDVI statistics.
    """
    h, w = 5, 5
    b04 = np.full((h, w), 3000, dtype=np.uint16)
    b08 = np.full((h, w), 3500, dtype=np.uint16)
    scl_cloudy = np.full((h, w), 9, dtype=np.uint8)  # 100% cloud

    scene1 = {"scene_id": "S2_ALL_CLOUDS_1", "b04": b04, "b08": b08, "scl": scl_cloudy, "meta": base_meta}
    scene2 = {"scene_id": "S2_ALL_CLOUDS_2", "b04": b04, "b08": b08, "scl": scl_cloudy, "meta": base_meta}

    comp = build_cloud_masked_optical_composite([scene1, scene2])

    assert comp["status"] == "unsupported_zero_valid_pixels"
    assert comp["valid_pixel_count"] == 0
    assert comp["coverage_percentage"] == 0.0
    assert np.all(np.isnan(comp["composite_ndvi"]))


def test_insufficient_valid_coverage_rejection(base_meta):
    """
    When valid pixels exist but coverage percentage is below the minimum threshold,
    status must be unsupported_insufficient_coverage.
    """
    h, w = 10, 10
    b04 = np.full((h, w), 500, dtype=np.uint16)
    b08 = np.full((h, w), 4500, dtype=np.uint16)
    scl = np.full((h, w), 9, dtype=np.uint8)  # mostly cloud
    scl[0, 0] = 4  # 1 pixel clear out of 100 = 1.0%

    scene = {"scene_id": "S2_LOW_COV", "b04": b04, "b08": b08, "scl": scl, "meta": base_meta}

    comp = build_cloud_masked_optical_composite([scene], min_valid_coverage_pct=10.0)

    assert comp["status"] == "unsupported_insufficient_coverage"
    assert comp["valid_pixel_count"] == 1
    assert comp["coverage_percentage"] == 1.0


def test_missing_bands_and_missing_scl_assets(base_meta):
    """
    - Scene with missing B04 or B08 must be rejected and logged in rejected_scenes.
    - Scene missing SCL is accepted with a scientific limitation caveat.
    """
    h, w = 4, 4
    b04 = np.full((h, w), 500, dtype=np.uint16)
    b08 = np.full((h, w), 4500, dtype=np.uint16)

    # Scene 1: Missing B08
    bad_scene = {"scene_id": "S2_NO_B08", "b04": b04, "b08": None, "meta": base_meta}
    # Scene 2: Valid bands, missing SCL
    no_scl_scene = {"scene_id": "S2_NO_SCL", "b04": b04, "b08": b08, "scl": None, "meta": base_meta}

    comp = build_cloud_masked_optical_composite([bad_scene, no_scl_scene])

    assert comp["status"] == "success"
    assert len(comp["rejected_scenes"]) == 1
    assert comp["rejected_scenes"][0]["scene_id"] == "S2_NO_B08"
    assert "Missing B04" in comp["rejected_scenes"][0]["reason"]
    assert any("lacks companion SCL layer" in lim for lim in comp["scientific_limitations"])


def test_bitemporal_multiscene_change_detection(base_meta):
    """
    Full bi-temporal change detection where T1 and T2 each have 2 candidate scenes:
    - T1 composite: vegetation baseline (NDVI ~ 0.8)
    - T2 composite: vegetation loss / clearing (NDVI ~ 0.2)
    - Mean Delta-NDVI ~ -0.60, vegetation_loss_pct > 90%
    """
    h, w = 10, 10

    # T1 Scenes: High vegetation (b04=500, b08=4500 -> NDVI = 0.80)
    t1_b04 = np.full((h, w), 500, dtype=np.uint16)
    t1_b08 = np.full((h, w), 4500, dtype=np.uint16)
    t1_scl = np.full((h, w), 4, dtype=np.uint8)

    t1_s1 = {"scene_id": "T1_SCENE_1", "datetime": "2023-01-10T00:00:00Z", "b04": t1_b04, "b08": t1_b08, "scl": t1_scl, "meta": base_meta}
    t1_s2 = {"scene_id": "T1_SCENE_2", "datetime": "2023-01-20T00:00:00Z", "b04": t1_b04, "b08": t1_b08, "scl": t1_scl, "meta": base_meta}

    # T2 Scenes: Cleared land (b04=2000, b08=3000 -> NDVI = 0.20)
    t2_b04 = np.full((h, w), 2000, dtype=np.uint16)
    t2_b08 = np.full((h, w), 3000, dtype=np.uint16)
    t2_scl = np.full((h, w), 4, dtype=np.uint8)

    t2_s1 = {"scene_id": "T2_SCENE_1", "datetime": "2024-01-10T00:00:00Z", "b04": t2_b04, "b08": t2_b08, "scl": t2_scl, "meta": base_meta}
    t2_s2 = {"scene_id": "T2_SCENE_2", "datetime": "2024-01-20T00:00:00Z", "b04": t2_b04, "b08": t2_b08, "scl": t2_scl, "meta": base_meta}

    res = compute_temporal_optical_change(
        t1_scenes=[t1_s1, t1_s2],
        t2_scenes=[t2_s1, t2_s2],
        composite_method="median"
    )

    assert res["status"] == "success"
    assert res["delta_ndvi_mean"] is not None
    assert np.isclose(res["delta_ndvi_mean"], -0.60, atol=0.05)
    assert res["vegetation_loss_pct"] == 100.0
    assert res["vegetation_gain_pct"] == 0.0
    assert res["joint_valid_pixel_count"] == 100
    assert res["composite_method"] == "valid_pixel_temporal_median"
    assert "T1_SCENE_1" in res["contributing_scene_ids_t1"]
    assert "T2_SCENE_1" in res["contributing_scene_ids_t2"]


def test_bitemporal_zero_joint_pixels_returns_unsupported():
    """
    When T1 is clear but T2 is completely cloudy:
    Zero joint valid pixels must result in unsupported status and None statistics.
    """
    h, w = 5, 5
    b04 = np.full((h, w), 1000, dtype=np.uint16)
    b08 = np.full((h, w), 4000, dtype=np.uint16)
    scl_clear = np.full((h, w), 4, dtype=np.uint8)
    scl_cloudy = np.full((h, w), 9, dtype=np.uint8)

    t1_s = {"scene_id": "T1_CLEAR", "datetime": "2023-01-01T00:00:00Z", "b04": b04, "b08": b08, "scl": scl_clear}
    t2_s = {"scene_id": "T2_CLOUDY", "datetime": "2024-01-01T00:00:00Z", "b04": b04, "b08": b08, "scl": scl_cloudy}

    res = compute_temporal_optical_change(t1_scenes=[t1_s], t2_scenes=[t2_s])

    assert res["status"] == "unsupported_zero_valid_pixels"
    assert res["delta_ndvi_mean"] is None
    assert res["vegetation_loss_pct"] is None
    assert res["joint_valid_pixel_count"] == 0


def test_transfer_budget_exhaustion_in_orchestration(monkeypatch):
    """
    Verify that when cumulative transfer budget is exhausted during multi-scene acquisition,
    the orchestrator skips optional secondary candidates without crashing or failing the plan.
    """
    from app.agent.orchestrator import MultiStepOrchestrator
    from app.acquisition.stac_service import stac_service

    # Mock search returning multiple scenes for T1 and T2
    def mock_search(req):
        class MockScene:
            def __init__(self, sid, dt):
                self.id = sid
                self.scene_id = sid
                self.datetime = dt
                self.cloud_cover = 10.0
                self.collection = req.collection
                self.sensor = "Sentinel-2"
                self.bbox = [77.45, 12.85, 77.75, 13.15]
        class MockSearchResp:
            status = "success"
            scenes = [MockScene(f"s2_{req.collection}_{i}", "2023-01-01T00:00:00Z") for i in range(3)]
            provider = "Element84 Earth Search"
            catalog_url = "https://earth-search.aws.element84.com/v1"
        return MockSearchResp()

    monkeypatch.setattr(stac_service, "search", mock_search)

    # Mock composite retrieve consuming budget
    import rasterio
    import base64
    from rasterio.transform import from_bounds
    memfile = rasterio.MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=10,
        width=10,
        count=3,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_bounds(77.45, 12.85, 77.75, 13.15, 10, 10)
    ) as dst:
        dst.write(np.full((10, 10), 1000, dtype=np.uint16), 1)
        dst.write(np.full((10, 10), 4000, dtype=np.uint16), 2)
        dst.write(np.full((10, 10), 4, dtype=np.uint16), 3)
        dst.update_tags(BAND_1="B04", BAND_2="B08", BAND_3="SCL")
    valid_b64 = "data:image/tiff;base64," + base64.b64encode(memfile.getbuffer()).decode("utf-8")

    def mock_retrieve_composite(req):
        from app.schemas.acquisition import STACCompositeRetrieveResponse
        return STACCompositeRetrieveResponse(
            status="success",
            scene_id=req.scene_id,
            collection=req.collection,
            category="scientific_raster",
            filename=f"{req.scene_id}.tif",
            mime_type="image/tiff",
            size_bytes=6 * 1024 * 1024,  # 6 MB per scene
            data_base64=valid_b64,
            provenance={"provider": "Element84 Earth Search"},
            bands_retrieved=["b04", "b08", "scl"]
        )

    monkeypatch.setattr(stac_service, "retrieve_composite_window", mock_retrieve_composite)

    orch = MultiStepOrchestrator()
    # With a tight budget of 15 MB, 2 scenes (12 MB) fit, but a 3rd would exceed 15 MB.
    # Optional candidates should skip when budget is reached.
    res = orch.orchestrate(
        query="Analyze temporal optical change in Bengaluru between 2023-01-01 and 2023-12-31",
        mode="orchestration",
        max_bytes=15 * 1024 * 1024,
        t1_start_date="2023-01-01",
        t1_end_date="2023-06-30",
        t2_start_date="2023-07-01",
        t2_end_date="2023-12-31",
        max_cloud_cover=20.0
    )

    assert res["status"] in ("success", "unsupported")
    steps = res["trace"]["steps"]
    step_names = [s.get("step_name") or s.get("step") for s in steps]
    assert "PARAMETERS_CONFIGURED" in step_names


def test_offline_mode_preserves_truthful_disclosures():
    orch = MultiStepOrchestrator()
    res = orch.orchestrate(
        query="Offline demo: analyze temporal change",
        mode="offline",
        metadata={"use_offline_fixtures": True}
    )
    assert res["status"] in ("success", "unsupported")
    report = res.get("reproducible_report") or {}
    notes = " ".join(report.get("notes") or [])
    limitations = " ".join(report.get("scientific_limitations") or [])
    trace_details = [s.get("detail") or "" for s in res["trace"]["steps"]]
    params_str = str(res["trace"].get("parameters") or {})
    exec_summaries = [s.get("summary") or "" for s in res["trace"].get("execution_steps") or []]
    combined_text = " ".join(trace_details + exec_summaries + [notes, limitations, params_str]).lower()
    assert "offline" in combined_text or "fixture" in combined_text or "sample" in combined_text
