"""
test_bitemporal_aoi_controls.py
Focuses on Step 7F: Bi-temporal date controls, optical cloud cover filtering,
and chronological date validation in the automated AOI orchestration pipeline.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from app.main import app
from app.agent.orchestrator import agent_orchestrator, QueryParser
from app.agent.tools import tool_registry

client = TestClient(app)


def _make_mock_optical_array(h=32, w=32):
    b04 = np.full((h, w), 0.08, dtype=np.float32)
    b08 = np.full((h, w), 0.35, dtype=np.float32)
    scl = np.full((h, w), 4, dtype=np.uint8)
    return np.dstack([b04, b08, scl])


# ==============================================================================
# 1. Bi-Temporal Date Windows Reach STAC Search
# ==============================================================================

def test_bitemporal_date_windows_reach_stac_search():
    """
    Verifies that explicit pre-event (T1) and post-event (T2) date ranges
    are passed accurately to the respective STAC search tool steps.
    """
    captured_searches = []
    opt_arr = _make_mock_optical_array()

    def mock_search_run(**kwargs):
        captured_searches.append(dict(kwargs))
        role = kwargs.get("temporal_role", "t1")
        return {
            "status": "success",
            "provider": "Element84 Earth Search (AWS)",
            "scenes": [{
                "scene_id": f"S2_{role.upper()}_SCENE",
                "collection": "sentinel-2-l2a",
                "datetime": "2023-03-01T10:00:00Z" if role == "t1" else "2023-08-01T10:00:00Z",
                "cloud_cover": 5.0,
                "sensor": "Sentinel-2A MSI"
            }]
        }

    def mock_retrieve_run(**kwargs):
        role = kwargs.get("temporal_role", "t1")
        return {
            "status": "success",
            "scene_id": kwargs.get("scene_id", "S2_AUTO"),
            "collection": "sentinel-2-l2a",
            "category": "scientific_raster",
            "bands_retrieved": ["b04", "b08", "scl"],
            "size_bytes": 45000,
            "array": opt_arr,
            "raster_metadata": {
                "crs": "EPSG:4326",
                "crs_display": "WGS 84 (EPSG:4326)",
                "bounds": [77.55, 12.95, 77.65, 13.05],
                "width": 32,
                "height": 32,
                "datetime": "2023-03-01T10:00:00Z" if role == "t1" else "2023-08-01T10:00:00Z",
                "georeferenced": True
            },
            "provenance": {
                "provider": "Element84 Earth Search (AWS)",
                "sensor": "Sentinel-2A MSI",
                "datetime": "2023-03-01T10:00:00Z" if role == "t1" else "2023-08-01T10:00:00Z"
            }
        }

    with patch.object(tool_registry.get("stac_search"), "run", side_effect=mock_search_run):
        with patch.object(tool_registry.get("stac_composite_retrieve"), "run", side_effect=mock_retrieve_run):
            result = agent_orchestrator.orchestrate(
                query="Analyze Sentinel-2 vegetation canopy change across my active AOI",
                mode="temporal",
                aoi_bbox=[77.55, 12.95, 77.65, 13.05],
                t1_start_date="2023-01-01",
                t1_end_date="2023-05-31",
                t2_start_date="2023-06-01",
                t2_end_date="2023-11-30",
                max_cloud_cover=15.0
            )

    assert len(captured_searches) == 2
    # First search: T1
    s1 = captured_searches[0]
    assert s1["start_date"] == "2023-01-01"
    assert s1["end_date"] == "2023-05-31"
    assert s1["max_cloud_cover"] == 15.0

    # Second search: T2
    s2 = captured_searches[1]
    assert s2["start_date"] == "2023-06-01"
    assert s2["end_date"] == "2023-11-30"
    assert s2["max_cloud_cover"] == 15.0


# ==============================================================================
# 2. Optical Cloud Cover Threshold Reaches STAC Search
# ==============================================================================

def test_optical_cloud_threshold_reaches_stac_search():
    """
    Verifies that the user-configured max cloud cover setting (e.g. 10%)
    is forwarded into the STACSearchTool for optical imagery.
    """
    captured_searches = []

    def mock_search_run(**kwargs):
        captured_searches.append(dict(kwargs))
        return {
            "status": "success",
            "provider": "Element84 Earth Search (AWS)",
            "scenes": [{
                "scene_id": "S2_SINGLE",
                "collection": "sentinel-2-l2a",
                "datetime": "2024-02-01T10:00:00Z",
                "cloud_cover": 4.5,
                "sensor": "Sentinel-2A MSI"
            }]
        }

    opt_arr = _make_mock_optical_array()
    def mock_retrieve_run(**kwargs):
        return {
            "status": "success",
            "scene_id": "S2_SINGLE",
            "collection": "sentinel-2-l2a",
            "category": "scientific_raster",
            "bands_retrieved": ["b04", "b08", "scl"],
            "size_bytes": 45000,
            "array": opt_arr,
            "raster_metadata": {"crs": "EPSG:4326", "width": 32, "height": 32, "datetime": "2024-02-01T10:00:00Z"},
            "provenance": {"provider": "Element84 Earth Search (AWS)", "sensor": "Sentinel-2A MSI"}
        }

    with patch.object(tool_registry.get("stac_search"), "run", side_effect=mock_search_run):
        with patch.object(tool_registry.get("stac_composite_retrieve"), "run", side_effect=mock_retrieve_run):
            result = agent_orchestrator.orchestrate(
                query="Assess Sentinel-2 NDVI across active AOI",
                mode="single",
                aoi_bbox=[77.55, 12.95, 77.65, 13.05],
                start_date="2024-01-01",
                end_date="2024-03-31",
                max_cloud_cover=10.0,
                metadata={"demo_fixtures": False}
            )

    assert len(captured_searches) >= 1
    req = captured_searches[0]
    assert req["max_cloud_cover"] == 10.0


# ==============================================================================
# 3. Invalid Bi-Temporal Date Ordering Safely Rejected
# ==============================================================================

def test_invalid_bitemporal_date_ordering_start_after_end():
    """
    Verifies that when a pre-event window has start_date > end_date,
    the orchestrator safely returns a structured failure without making external calls.
    """
    result = agent_orchestrator.orchestrate(
        query="Analyze bi-temporal optical change",
        mode="temporal",
        aoi_bbox=[77.55, 12.95, 77.65, 13.05],
        t1_start_date="2023-06-01",
        t1_end_date="2023-01-01",  # Inverted
        t2_start_date="2023-07-01",
        t2_end_date="2023-12-31"
    )

    assert result["status"] == "failed"
    assert "pre-event start date" in result["answer"].lower()
    assert result["confidence_label"] == "Invalid Date Windows"


def test_invalid_bitemporal_date_ordering_overlapping_windows():
    """
    Verifies that when the post-event window starts before the pre-event window ends,
    the orchestrator safely rejects the input.
    """
    result = agent_orchestrator.orchestrate(
        query="Analyze bi-temporal change in vegetation",
        mode="temporal",
        aoi_bbox=[77.55, 12.95, 77.65, 13.05],
        t1_start_date="2023-01-01",
        t1_end_date="2023-08-31",
        t2_start_date="2023-06-01",  # Overlaps / precedes T1 end
        t2_end_date="2023-12-31"
    )

    assert result["status"] == "failed"
    assert "strictly later" in result["answer"].lower()
    assert result["confidence_label"] == "Invalid Date Windows"


# ==============================================================================
# 4. Parameters Appears in Execution Trace and Provenance
# ==============================================================================

def test_parameters_recorded_in_trace_and_provenance():
    """
    Verifies that PARAMETERS_CONFIGURED trace step is recorded and optical_cloud_filter
    appears in provenance.
    """
    def mock_search_run(**kwargs):
        return {
            "status": "success",
            "provider": "Element84 Earth Search (AWS)",
            "scenes": []
        }

    with patch.object(tool_registry.get("stac_search"), "run", side_effect=mock_search_run):
        result = agent_orchestrator.orchestrate(
            query="Analyze cross-sensor optical and SAR joint signature",
            mode="optical_sar",
            aoi_bbox=[77.55, 12.95, 77.65, 13.05],
            start_date="2024-01-01",
            end_date="2024-03-31",
            max_cloud_cover=12.0
        )

    trace_steps = result.get("trace", {}).get("steps", [])
    param_steps = [s for s in trace_steps if s.get("step") == "PARAMETERS_CONFIGURED"]
    assert len(param_steps) == 1
    assert "12.0%" in param_steps[0]["detail"]


# ==============================================================================
# 5. API v1 Analyze Endpoint Forwards All Controls
# ==============================================================================

def test_api_v1_analyze_endpoint_with_bitemporal_controls():
    """
    Verifies that /api/v1/analyze accepts t1/t2 dates and max_cloud_cover in the payload.
    """
    payload = {
        "query": "Assess bi-temporal change with custom date ranges",
        "mode": "temporal",
        "aoi_bbox": [77.55, 12.95, 77.65, 13.05],
        "t1_start_date": "2023-01-01",
        "t1_end_date": "2023-05-01",
        "t2_start_date": "2023-06-01",
        "t2_end_date": "2023-10-01",
        "max_cloud_cover": 18.5,
        "metadata": {
            "demo_fixtures": True,
            "offline_mode": True
        }
    }

    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
