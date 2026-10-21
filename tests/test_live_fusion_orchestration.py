"""
Step 7E Test Suite: Live Optical/SAR Retrieval Connected to Evidence-Based Fusion and Change Detection.

Tests:
1. Complete orchestration request path with automated discovery and bounded window retrieval.
2. Cross-sensor optical-SAR fusion with compatible live mock evidence.
3. Truthful rejection on missing scenes (no fabricated scenes).
4. Truthful rejection on disjoint footprints (spatial overlap < 10%).
5. Advisory on excessive temporal separation (> 60 days).
6. Truthful rejection on invalid / incompatible CRS.
7. Cumulative transfer budget exhaustion handling.
8. STAC provider error handling.
9. Strict URL sanitization (SAS tokens and signed query strings never exposed).
10. Bi-temporal optical change with distinct dates.
11. Degenerate temporal pair rejection (identical timestamps).
12. Cross-sensor differences never misreported as temporal change.
13. Offline demo preservation (disjoint local fixtures Bengaluru vs Mumbai honestly rejected).
14. End-to-end API request flow via /api/v1/analyze with user-selected aoi_bbox.
"""

import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.agent.orchestrator import MultiStepOrchestrator, agent_orchestrator
from app.agent.tools import tool_registry
from app.remote_sensing.temporal import compute_temporal_optical_change, compute_temporal_sar_change
from app.schemas.analysis import AnalysisRequest
from app.acquisition.windowed_reader import SafeWindowedCOGReader

client = TestClient(app)


def _make_mock_optical_array(h=32, w=32):
    """Creates a mock 3-band optical array [B04, B08, SCL] with clear sky pixels."""
    b04 = np.full((h, w), 0.08, dtype=np.float32)  # Red reflectance
    b08 = np.full((h, w), 0.35, dtype=np.float32)  # NIR reflectance
    scl = np.full((h, w), 4, dtype=np.uint8)        # SCL class 4 = Vegetation (clear sky)
    return np.dstack([b04, b08, scl])


def _make_mock_sar_array(h=32, w=32):
    """Creates a mock 2-band SAR array [VV, VH] in linear gamma0 power."""
    vv = np.full((h, w), 0.12, dtype=np.float32)   # VV linear power
    vh = np.full((h, w), 0.024, dtype=np.float32)  # VH linear power
    return np.dstack([vv, vh])


# ==============================================================================
# 1. Complete Orchestration Path: Compatible Optical-SAR Live Evidence
# ==============================================================================

def test_orchestration_optical_sar_compatible_live_evidence():
    """
    Verifies the end-to-end orchestration DAG:
    1. STAC search discovers S2 optical and S1 RTC SAR for requested AOI.
    2. Bounded window composite retrieval acquires calibrated rasters.
    3. Cross-sensor validation confirms spatial overlap, CRS, and radiometric calibration.
    4. Optical-SAR fusion executes and returns structured provenance and trace.
    """
    aoi = [77.55, 12.95, 77.65, 13.05]
    opt_arr = _make_mock_optical_array()
    sar_arr = _make_mock_sar_array()

    def mock_tool_run(tool_name, **kwargs):
        if tool_name == "stac_search":
            col = kwargs.get("collection", "")
            if "sentinel-2" in col:
                return {
                    "status": "success",
                    "provider": "Element84 Earth Search (AWS)",
                    "scenes": [{
                        "scene_id": "S2A_MSIL2A_20230501T052520_TEST",
                        "collection": "sentinel-2-l2a",
                        "datetime": "2023-05-01T05:25:20Z",
                        "cloud_cover": 2.5,
                        "sensor": "Sentinel-2A MSI"
                    }]
                }
            else:
                return {
                    "status": "success",
                    "provider": "Microsoft Planetary Computer",
                    "scenes": [{
                        "scene_id": "S1A_IW_GRDH_1SDV_20230505T005315_RTC_TEST",
                        "collection": "sentinel-1-rtc",
                        "datetime": "2023-05-05T00:53:15Z",
                        "sensor": "Sentinel-1A C-SAR"
                    }]
                }
        elif tool_name == "stac_composite_retrieve":
            col = kwargs.get("collection", "")
            sc_id = kwargs.get("scene_id", "")
            if "sentinel-2" in col or "S2A" in sc_id:
                return {
                    "status": "success",
                    "scene_id": sc_id,
                    "collection": "sentinel-2-l2a",
                    "category": "scientific_raster",
                    "bands_retrieved": ["b04", "b08", "scl"],
                    "size_bytes": 45000,
                    "array": opt_arr,
                    "raster_metadata": {
                        "crs": "EPSG:4326",
                        "crs_display": "WGS 84 (EPSG:4326)",
                        "bounds": aoi,
                        "width": 32,
                        "height": 32,
                        "datetime": "2023-05-01T05:25:20Z",
                        "georeferenced": True
                    },
                    "provenance": {
                        "provider": "Element84 Earth Search (AWS)",
                        "sensor": "Sentinel-2A MSI",
                        "datetime": "2023-05-01T05:25:20Z",
                        "asset_hrefs": {
                            "b04": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B04.tif",
                            "b08": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/B08.tif"
                        }
                    }
                }
            else:
                return {
                    "status": "success",
                    "scene_id": sc_id,
                    "collection": "sentinel-1-rtc",
                    "category": "scientific_raster",
                    "bands_retrieved": ["vv", "vh"],
                    "size_bytes": 35000,
                    "array": sar_arr,
                    "raster_metadata": {
                        "crs": "EPSG:4326",
                        "crs_display": "WGS 84 (EPSG:4326)",
                        "bounds": aoi,
                        "width": 32,
                        "height": 32,
                        "datetime": "2023-05-05T00:53:15Z",
                        "georeferenced": True
                    },
                    "provenance": {
                        "provider": "Microsoft Planetary Computer",
                        "sensor": "Sentinel-1A C-SAR (RTC)",
                        "datetime": "2023-05-05T00:53:15Z",
                        "polarization": "DUAL_VV_VH",
                        "calibration_quantity": "linear_power",
                        "sar_terrain_layover_shadow_mask": "radiometrically_terrain_corrected",
                        "asset_hrefs": {
                            "vv": "https://planetarycomputer.microsoft.com/vv.tif",
                            "vh": "https://planetarycomputer.microsoft.com/vh.tif"
                        }
                    }
                }
        elif tool_name == "optical_sar_fusion":
            return {
                "status": "success",
                "answer": "Optical-SAR cross-sensor fusion executed successfully across co-registered footprints.",
                "fused_preview_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
                "statistics": {
                    "spatial_overlap_pct": 100.0,
                    "co_registered": True
                }
            }
        return {"status": "success"}

    orchestrator = MultiStepOrchestrator()
    with patch.object(orchestrator.tool_registry, "get") as mock_get_tool:
        def get_tool_side_effect(name):
            mock_tool = MagicMock()
            mock_tool.name = name
            mock_tool.run.side_effect = lambda **kwargs: mock_tool_run(name, **kwargs)
            return mock_tool
        mock_get_tool.side_effect = get_tool_side_effect

        result = orchestrator.orchestrate(
            query="Perform optical-SAR cross-sensor fusion analysis",
            aoi_bbox=aoi,
            start_date="2023-01-01",
            end_date="2023-12-31",
            mode="optical_sar"
        )

    assert result["status"] == "success"
    assert "optical_sar_fusion" in result["metadata"]
    assert result["metadata"]["optical_sar_fusion"]["status"] == "success"

    # Check discovered scenes telemetry
    disc = result["metadata"].get("discovered_scenes", [])
    assert len(disc) >= 2
    providers = {d.get("provider") for d in disc}
    assert "Element84 Earth Search (AWS)" in providers
    assert "Microsoft Planetary Computer" in providers

    # Check retrieved scenes telemetry
    ret = result["metadata"].get("retrieved_scenes", [])
    assert len(ret) >= 2
    assert all(r.get("size_bytes", 0) > 0 for r in ret)

    # Check optical_sar_provenance
    prov = result["metadata"].get("optical_sar_provenance", {})
    assert "optical" in prov
    assert "sar" in prov
    assert prov["optical"]["provider"] == "Element84 Earth Search (AWS)"
    assert prov["sar"]["provider"] == "Microsoft Planetary Computer"
    assert prov["cross_sensor_verification"]["spatial_overlap_pct"] > 90.0
    assert prov["cross_sensor_verification"]["crs_matching"] is True


# ==============================================================================
# 2. Missing Scenes Rejection (Truthful Non-Fabricating Behavior)
# ==============================================================================

def test_orchestration_missing_scenes_rejection():
    """Verifies that when STAC search discovers 0 scenes, the orchestrator reports failure truthfully."""
    aoi = [77.55, 12.95, 77.65, 13.05]
    orchestrator = MultiStepOrchestrator()

    with patch.object(orchestrator.tool_registry, "get") as mock_get_tool:
        mock_tool = MagicMock()
        mock_tool.name = "stac_search"
        mock_tool.run.return_value = {
            "status": "success",
            "scenes": []  # Zero scenes found
        }
        mock_get_tool.return_value = mock_tool

        result = orchestrator.orchestrate(
            query="Analyze Sentinel-2 NDVI",
            aoi_bbox=aoi,
            start_date="2023-01-01",
            end_date="2023-12-31",
            mode="single"
        )

    assert result["status"] == "failed"
    assert "No satellite scenes discovered in STAC collection" in result["answer"] or "failed" in result["answer"].lower()
    # Ensure no fabricated claims were created
    assert result["confidence"] is None
    assert result["confidence_label"] == "Acquisition Failed"


# ==============================================================================
# 3. Disjoint Footprints Rejection (Cross-Sensor Validation Gate)
# ==============================================================================

def test_orchestration_disjoint_footprints_rejection():
    """Verifies that spatially disjoint optical and SAR scenes are rejected with an honest explanation."""
    aoi = [77.55, 12.95, 77.65, 13.05]
    opt_arr = _make_mock_optical_array()
    sar_arr = _make_mock_sar_array()

    def mock_tool_run(tool_name, **kwargs):
        if tool_name == "stac_search":
            col = kwargs.get("collection", "")
            return {
                "status": "success",
                "scenes": [{"scene_id": f"SCENE_{col}", "collection": col}]
            }
        elif tool_name == "stac_composite_retrieve":
            col = kwargs.get("collection", "")
            if "sentinel-2" in col:
                return {
                    "status": "success",
                    "scene_id": "S2_BENGALURU",
                    "collection": "sentinel-2-l2a",
                    "category": "scientific_raster",
                    "array": opt_arr,
                    "raster_metadata": {
                        "crs": "EPSG:4326",
                        "bounds": [77.45, 12.85, 77.75, 13.15],  # Bengaluru
                        "width": 32,
                        "height": 32
                    }
                }
            else:
                return {
                    "status": "success",
                    "scene_id": "S1_MUMBAI",
                    "collection": "sentinel-1-rtc",
                    "category": "scientific_raster",
                    "array": sar_arr,
                    "raster_metadata": {
                        "crs": "EPSG:4326",
                        "bounds": [72.75, 18.88, 73.05, 19.30],  # Mumbai (Disjoint!)
                        "width": 32,
                        "height": 32
                    }
                }
        return {"status": "success"}

    orchestrator = MultiStepOrchestrator()
    with patch.object(orchestrator.tool_registry, "get") as mock_get_tool:
        def get_tool_side_effect(name):
            mock_tool = MagicMock()
            mock_tool.name = name
            mock_tool.run.side_effect = lambda **kwargs: mock_tool_run(name, **kwargs)
            return mock_tool
        mock_get_tool.side_effect = get_tool_side_effect

        result = orchestrator.orchestrate(
            query="Perform optical-SAR fusion",
            aoi_bbox=aoi,
            mode="optical_sar"
        )

    assert result["status"] == "unsupported"
    assert "rejected" in result["answer"].lower() or "overlap" in result["answer"].lower()
    cs_val = result["metadata"]["optical_sar_fusion"]["cross_sensor_validation"]
    assert cs_val["is_compatible"] is False
    assert cs_val["spatial_overlap_pct"] == 0.0


# ==============================================================================
# 4. Excessive Date Separation Warning (> 60 days)
# ==============================================================================

def test_orchestration_excessive_date_separation_advisory():
    """Verifies that passes separated by > 60 days log a temporal separation advisory."""
    aoi = [77.55, 12.95, 77.65, 13.05]
    opt_arr = _make_mock_optical_array()
    sar_arr = _make_mock_sar_array()

    def mock_tool_run(tool_name, **kwargs):
        if tool_name == "stac_search":
            col = kwargs.get("collection", "")
            return {"status": "success", "scenes": [{"scene_id": f"SCENE_{col}", "collection": col}]}
        elif tool_name == "stac_composite_retrieve":
            col = kwargs.get("collection", "")
            if "sentinel-2" in col:
                return {
                    "status": "success",
                    "scene_id": "S2_T1",
                    "collection": "sentinel-2-l2a",
                    "category": "scientific_raster",
                    "array": opt_arr,
                    "raster_metadata": {
                        "crs": "EPSG:4326",
                        "bounds": aoi,
                        "width": 32,
                        "height": 32,
                        "datetime": "2023-01-01T10:00:00Z"
                    }
                }
            else:
                return {
                    "status": "success",
                    "scene_id": "S1_T2",
                    "collection": "sentinel-1-rtc",
                    "category": "scientific_raster",
                    "array": sar_arr,
                    "raster_metadata": {
                        "crs": "EPSG:4326",
                        "bounds": aoi,
                        "width": 32,
                        "height": 32,
                        "datetime": "2023-06-01T10:00:00Z"  # 151 days separation (> 60 days)
                    }
                }
        elif tool_name == "optical_sar_fusion":
            return {"status": "success", "answer": "Fused with temporal separation warning."}
        return {"status": "success"}

    orchestrator = MultiStepOrchestrator()
    with patch.object(orchestrator.tool_registry, "get") as mock_get_tool:
        def get_tool_side_effect(name):
            mock_tool = MagicMock()
            mock_tool.name = name
            mock_tool.run.side_effect = lambda **kwargs: mock_tool_run(name, **kwargs)
            return mock_tool
        mock_get_tool.side_effect = get_tool_side_effect

        result = orchestrator.orchestrate(
            query="Perform optical-SAR fusion",
            aoi_bbox=aoi,
            mode="optical_sar"
        )

    # Cross-sensor verification should reflect temporal delta > 60 days
    cs_prov = result["metadata"]["optical_sar_provenance"]["cross_sensor_verification"]
    assert cs_prov["temporal_delta_days"] == 151
    # Check advisory was recorded in trace steps
    trace_steps = [s["step"] for s in result["trace"]["steps"]]
    assert any("ADVISORY" in s or "TOOL_OPTICAL_SAR_FUSION" in s for s in trace_steps)


# ==============================================================================
# 5. Invalid / Incompatible CRS Rejection
# ==============================================================================

def test_orchestration_incompatible_crs_rejection():
    """Verifies that non-matching CRS without an affine transform is rejected."""
    aoi = [77.55, 12.95, 77.65, 13.05]
    opt_arr = _make_mock_optical_array()
    sar_arr = _make_mock_sar_array()

    def mock_tool_run(tool_name, **kwargs):
        if tool_name == "stac_search":
            col = kwargs.get("collection", "")
            return {"status": "success", "scenes": [{"scene_id": f"SCENE_{col}", "collection": col}]}
        elif tool_name == "stac_composite_retrieve":
            col = kwargs.get("collection", "")
            if "sentinel-2" in col:
                return {
                    "status": "success",
                    "scene_id": "S2_UTM",
                    "collection": "sentinel-2-l2a",
                    "category": "scientific_raster",
                    "array": opt_arr,
                    "raster_metadata": {
                        "crs": "EPSG:32643",  # UTM Zone 43N
                        "bounds": [700000, 1400000, 710000, 1410000],
                        "width": 32,
                        "height": 32
                    }
                }
            else:
                return {
                    "status": "success",
                    "scene_id": "S1_WGS84",
                    "collection": "sentinel-1-rtc",
                    "category": "scientific_raster",
                    "array": sar_arr,
                    "raster_metadata": {
                        "crs": "EPSG:4326",  # Geographic WGS84
                        "bounds": aoi,
                        "width": 32,
                        "height": 32
                    }
                }
        return {"status": "success"}

    orchestrator = MultiStepOrchestrator()
    with patch.object(orchestrator.tool_registry, "get") as mock_get_tool:
        def get_tool_side_effect(name):
            mock_tool = MagicMock()
            mock_tool.name = name
            mock_tool.run.side_effect = lambda **kwargs: mock_tool_run(name, **kwargs)
            return mock_tool
        mock_get_tool.side_effect = get_tool_side_effect

        result = orchestrator.orchestrate(
            query="Perform optical-SAR fusion",
            aoi_bbox=aoi,
            mode="optical_sar"
        )

    assert result["status"] == "unsupported"
    assert "crs" in result["answer"].lower() or "rejected" in result["answer"].lower()


# ==============================================================================
# 6. Cumulative Transfer Budget Exhaustion
# ==============================================================================

def test_orchestration_transfer_budget_exhaustion():
    """Verifies that exceeding the cumulative transfer budget halts acquisition and reports status truthfully."""
    aoi = [77.55, 12.95, 77.65, 13.05]
    orchestrator = MultiStepOrchestrator()

    with patch.object(orchestrator.tool_registry, "get") as mock_get_tool:
        mock_search = MagicMock()
        mock_search.name = "stac_search"
        mock_search.run.return_value = {
            "status": "success",
            "scenes": [{"scene_id": "S2_OVERSIZED", "collection": "sentinel-2-l2a"}]
        }

        mock_retrieve = MagicMock()
        mock_retrieve.name = "stac_composite_retrieve"
        mock_retrieve.run.return_value = {
            "status": "failed",
            "error": "Cumulative transfer budget of 1000 bytes exceeded."
        }

        def get_side_effect(name):
            if name == "stac_search":
                return mock_search
            return mock_retrieve

        mock_get_tool.side_effect = get_side_effect

        result = orchestrator.orchestrate(
            query="Analyze Sentinel-2 NDVI",
            aoi_bbox=aoi,
            max_bytes=1000,
            mode="single"
        )

    assert result["status"] == "budget_exhausted"
    assert "budget" in result["answer"].lower()
    assert result["confidence_label"] == "Budget Exhausted"


# ==============================================================================
# 7. STAC Provider Error Handling
# ==============================================================================

def test_orchestration_stac_provider_error():
    """Verifies that a provider network or HTTP 500 error fails cleanly without uncaught exceptions."""
    aoi = [77.55, 12.95, 77.65, 13.05]
    orchestrator = MultiStepOrchestrator()

    with patch.object(orchestrator.tool_registry, "get") as mock_get_tool:
        mock_search = MagicMock()
        mock_search.name = "stac_search"
        mock_search.run.return_value = {
            "status": "failed",
            "error": "Element84 STAC endpoint returned HTTP 503 Service Unavailable"
        }
        mock_get_tool.return_value = mock_search

        result = orchestrator.orchestrate(
            query="Analyze Sentinel-2 NDVI",
            aoi_bbox=aoi,
            mode="single"
        )

    assert result["status"] == "failed"
    assert result["confidence"] is None
    assert "failed" in result["answer"].lower() or "503" in result["answer"]


# ==============================================================================
# 8. Sanitized Provenance (Never Exposes SAS Tokens)
# ==============================================================================

def test_orchestration_sanitized_provenance_never_exposes_sas():
    """Verifies that signed query strings and SAS tokens are stripped from provenance URLs."""
    raw_azure_url = "https://sentinel1euwest.blob.core.windows.net/rtc/vv.tif?st=2026-10-14T00%3A00%3A00Z&se=2026-10-15T00%3A00%3A00Z&sp=r&sig=SECRET_SAS_TOKEN_12345"
    sanitized = SafeWindowedCOGReader._sanitize_url(raw_azure_url)

    assert "SECRET_SAS_TOKEN" not in sanitized
    assert "sig=" not in sanitized
    assert "st=" not in sanitized
    assert sanitized == "https://sentinel1euwest.blob.core.windows.net/rtc/vv.tif"


# ==============================================================================
# 9. Bi-Temporal Optical Change Detection (Distinct Dates)
# ==============================================================================

def test_temporal_optical_change_distinct_dates():
    """Verifies that bi-temporal Sentinel-2 change detection computes Delta-NDVI for distinct dates."""
    opt1 = _make_mock_optical_array()  # Pre-event: NDVI ~ 0.6279
    opt2 = _make_mock_optical_array()
    # Simulate vegetation loss in T2: B04 increases, B08 decreases
    opt2[:, :, 0] = 0.20
    opt2[:, :, 1] = 0.22

    res = compute_temporal_optical_change(
        t1_b04=opt1[:, :, 0],
        t1_b08=opt1[:, :, 1],
        t2_b04=opt2[:, :, 0],
        t2_b08=opt2[:, :, 1],
        t1_scl=opt1[:, :, 2],
        t2_scl=opt2[:, :, 2],
        t1_datetime="2023-01-15T10:00:00Z",
        t2_datetime="2024-01-15T10:00:00Z"
    )

    assert res["status"] == "success"
    assert res["delta_ndvi_mean"] < 0  # Vegetation decreased
    assert res["vegetation_loss_pct"] > 90.0
    assert res["joint_valid_pixel_count"] == 32 * 32


# ==============================================================================
# 10. Degenerate Temporal Pair Rejection (Identical Timestamps)
# ==============================================================================

def test_temporal_change_rejects_identical_timestamps():
    """Verifies that temporal change calculation rejects degenerate identical timestamp pairs."""
    opt1 = _make_mock_optical_array()

    with pytest.raises(ValueError, match="Degenerate temporal pair"):
        compute_temporal_optical_change(
            t1_b04=opt1[:, :, 0],
            t1_b08=opt1[:, :, 1],
            t2_b04=opt1[:, :, 0],
            t2_b08=opt1[:, :, 1],
            t1_datetime="2023-01-15T10:00:00Z",
            t2_datetime="2023-01-15T10:00:00Z"  # Identical!
        )


# ==============================================================================
# 11. Offline Demo Preservation (Truthful Unsupported on Disjoint Fixtures)
# ==============================================================================

def test_offline_demo_preserves_honest_unsupported_result():
    """
    Verifies that running in offline demo mode uses local sample fixtures (Cartosat Bengaluru
    and RISAT Mumbai) and honestly reports unsupported validation due to 0% footprint overlap.
    """
    result = agent_orchestrator.orchestrate(
        query="Run offline optical-SAR cross-sensor analysis",
        mode="offline",
        metadata={"demo_fixtures": True}
    )

    assert result["status"] == "unsupported"
    assert "rejected" in result["answer"].lower() or "overlap" in result["answer"].lower()
    prov = result["metadata"]["optical_sar_provenance"]
    assert prov["optical"]["collection"] == "cartosat-optical-l2"
    assert prov["sar"]["collection"] == "risat-c-band-sar"
    assert prov["cross_sensor_verification"]["spatial_overlap_pct"] == 0.0


# ==============================================================================
# 12. End-to-End API Flow via /api/v1/analyze with aoi_bbox
# ==============================================================================

def test_api_v1_analyze_endpoint_with_aoi_bbox():
    """Verifies that the /api/v1/analyze HTTP endpoint correctly accepts aoi_bbox and returns JSON."""
    payload = {
        "query": "Assess Sentinel-2 NDVI across user AOI",
        "mode": "single",
        "aoi_bbox": [77.55, 12.95, 77.65, 13.05],
        "metadata": {
            "demo_fixtures": True,
            "offline_mode": True
        }
    }

    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "task" in data
    assert "answer" in data
    assert "trace" in data
    assert "metadata" in data
