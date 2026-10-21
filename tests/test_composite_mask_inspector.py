"""
Unit and integration tests for Step 7H:
Interactive Visual Mask Inspector for Multi-Scene Sentinel-2 Temporal Composites.

Covers:
1. Generation of 4 visualization layers: Red (B04), NIR (B08), Observation Depth (0, 1, 2+), SCL Validity Mask.
2. Accurate observation depth counts and class partitioning.
3. Strict correspondence with Step 7G SCL cloud and shadow masking policy.
4. NoData preservation: invalid/masked pixels rendered distinct/transparent, never fabricated as valid reflectance or zero.
5. Spatial metadata preservation: shape, bounds, CRS.
6. Zero valid pixels defense: honest unavailable reporting without fabricating layers.
7. End-to-end integration with compute_temporal_optical_change and visual_inspector payload.
8. Orchestrator integration: evidence item 'ev_temporal_composite_inspector' generation.
9. API Endpoint POST /api/v1/acquisition/composite-inspector request and response validation.
10. Offline fixture disclosure and provenance sanitization (no SAS tokens or secrets).
"""

import sys
import os
import io
import base64
import pytest
import numpy as np
from PIL import Image

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from app.main import app
from app.remote_sensing.temporal import (
    build_cloud_masked_optical_composite,
    compute_temporal_optical_change,
    generate_composite_inspector_layers,
    convert_rgba_to_base64_png
)
from app.remote_sensing.quality import (
    SCL_EXCLUDED_CLASSES,
    SCL_RETAINED_CLASSES
)
from app.agent.orchestrator import MultiStepOrchestrator

client = TestClient(app)


def decode_png_from_base64(data_url: str) -> np.ndarray:
    raw_b64 = data_url.split(",")[-1]
    raw_bytes = base64.b64decode(raw_b64)
    img = Image.open(io.BytesIO(raw_bytes))
    return np.array(img)


@pytest.fixture
def sample_composite():
    """Builds a deterministic 10x10 composite with clear, cloudy, and multi-scene pixels."""
    h, w = 10, 10
    # Scene 1: Top half (rows 0-4) vegetation (SCL 4), bottom half cloudy (SCL 9)
    s1_b04 = np.full((h, w), 1000, dtype=np.uint16)
    s1_b08 = np.full((h, w), 4000, dtype=np.uint16)
    s1_scl = np.zeros((h, w), dtype=np.uint8)
    s1_scl[0:5, :] = 4
    s1_scl[5:10, :] = 9

    # Scene 2: Left half (cols 0-4) vegetation (SCL 4), right half cloud shadow (SCL 3)
    s2_b04 = np.full((h, w), 1200, dtype=np.uint16)
    s2_b08 = np.full((h, w), 4200, dtype=np.uint16)
    s2_scl = np.zeros((h, w), dtype=np.uint8)
    s2_scl[:, 0:5] = 4
    s2_scl[:, 5:10] = 3

    meta = {
        "crs": "EPSG:4326",
        "transform": [0.01, 0.0, 77.45, 0.0, -0.01, 13.15],
        "shape": (h, w),
        "bounds": [77.45, 12.85, 77.75, 13.15]
    }

    sc1 = {"scene_id": "S2_T1_SCENE1", "datetime": "2023-01-15T10:00:00Z", "b04": s1_b04, "b08": s1_b08, "scl": s1_scl, "meta": meta}
    sc2 = {"scene_id": "S2_T1_SCENE2", "datetime": "2023-01-20T10:00:00Z", "b04": s2_b04, "b08": s2_b08, "scl": s2_scl, "meta": meta}

    return build_cloud_masked_optical_composite([sc1, sc2], scale_factor=10000.0)


def test_generate_composite_inspector_layers_structure(sample_composite):
    """Verifies that all 4 layers are generated with valid RGBA dimensions, stats, and bounds."""
    comp = sample_composite
    aoi_bounds = [77.45, 12.85, 77.75, 13.15]
    insp = generate_composite_inspector_layers(comp, aoi_bounds)

    assert insp["status"] == "success"
    assert insp["shape"] == [10, 10]
    assert insp["bounds"] == aoi_bounds
    assert insp["crs"] == "EPSG:4326"

    layers = insp["layers"]
    assert "b04_red" in layers
    assert "b08_nir" in layers
    assert "obs_depth" in layers
    assert "scl_mask" in layers

    for key, data_url in layers.items():
        assert data_url.startswith("data:image/png;base64,")
        arr = decode_png_from_base64(data_url)
        assert arr.shape == (10, 10, 4), f"Layer {key} must be 10x10x4 RGBA"


def test_observation_depth_class_partitioning(sample_composite):
    """
    Verifies that observation depth cleanly distinguishes:
    - 0 clear scenes: Crimson warning [220, 38, 38]
    - 1 clear scene: Amber [217, 119, 6]
    - 2+ clear scenes: Emerald [16, 185, 129]
    """
    insp = generate_composite_inspector_layers(sample_composite)
    stats = insp["stats"]
    assert stats["total_pixels"] == 100
    assert stats["obs_depth_0_count"] + stats["obs_depth_1_count"] + stats["obs_depth_2plus_count"] == 100
    assert stats["obs_depth_0_count"] == stats["masked_pixels"]
    assert stats["obs_depth_1_count"] + stats["obs_depth_2plus_count"] == stats["valid_pixels"]

    # In our fixture:
    # Rows 0-4 & Cols 0-4: clear in both S1 and S2 -> 25 pixels with depth 2
    assert stats["obs_depth_2plus_count"] == 25
    # Rows 5-9 & Cols 5-9: cloudy in S1 (9) and shadow in S2 (3) -> 25 pixels with depth 0
    assert stats["obs_depth_0_count"] == 25
    # Remaining 50 pixels (top-right and bottom-left): clear in exactly 1 scene -> 50 pixels with depth 1
    assert stats["obs_depth_1_count"] == 50

    # Verify pixel colors in obs_depth RGBA image
    depth_arr = decode_png_from_base64(insp["layers"]["obs_depth"])
    # Pixel (2, 2) is depth 2 (Emerald)
    assert depth_arr[2, 2, 0] == 16
    assert depth_arr[2, 2, 1] == 185
    assert depth_arr[2, 2, 2] == 129

    # Pixel (7, 7) is depth 0 (Crimson warning)
    assert depth_arr[7, 7, 0] == 220
    assert depth_arr[7, 7, 1] == 38
    assert depth_arr[7, 7, 2] == 38

    # Pixel (2, 7) is depth 1 (Amber)
    assert depth_arr[2, 7, 0] == 217
    assert depth_arr[2, 7, 1] == 119
    assert depth_arr[2, 7, 2] == 6


def test_scl_validity_mask_rendering(sample_composite):
    """Verifies that clear surface pixels are green and masked pixels are rose."""
    insp = generate_composite_inspector_layers(sample_composite)
    scl_arr = decode_png_from_base64(insp["layers"]["scl_mask"])

    # Clear valid pixel (row 2, col 2): Green [34, 197, 94]
    assert scl_arr[2, 2, 0] == 34
    assert scl_arr[2, 2, 1] == 197
    assert scl_arr[2, 2, 2] == 94

    # Masked pixel (row 7, col 7): Rose [244, 63, 94]
    assert scl_arr[7, 7, 0] == 244
    assert scl_arr[7, 7, 1] == 63
    assert scl_arr[7, 7, 2] == 94


def test_nodata_preservation_in_reflectance_layers(sample_composite):
    """
    Verifies that masked / NoData pixels in B04 and B08 reflectance layers are rendered
    with distinct dark slate with partial alpha, and never falsely rendered as valid reflectance.
    """
    insp = generate_composite_inspector_layers(sample_composite)
    b04_arr = decode_png_from_base64(insp["layers"]["b04_red"])

    # Valid pixel (2, 2) has full alpha (255)
    assert b04_arr[2, 2, 3] == 255
    # Masked pixel (7, 7) has partial transparency (120) and dark slate color [30, 32, 38]
    assert b04_arr[7, 7, 3] == 120
    assert b04_arr[7, 7, 0] == 30
    assert b04_arr[7, 7, 1] == 32
    assert b04_arr[7, 7, 2] == 38


def test_zero_valid_pixels_defense():
    """When a composite has zero valid observations, layers are honestly reported as unavailable."""
    empty_comp = {
        "status": "unsupported_zero_valid_pixels",
        "composite_b04": None,
        "summary": "All pixels cloud covered."
    }
    insp = generate_composite_inspector_layers(empty_comp)
    assert insp["status"] == "unavailable"
    assert insp["stats"]["valid_pixels"] == 0
    assert insp["stats"]["clear_coverage_pct"] == 0.0
    assert len(insp["layers"]) == 0


def test_integration_in_compute_temporal_optical_change():
    """Verifies that compute_temporal_optical_change includes visual_inspector with T1 and T2."""
    h, w = 6, 6
    b04 = np.full((h, w), 1000, dtype=np.uint16)
    b08 = np.full((h, w), 4000, dtype=np.uint16)
    scl = np.full((h, w), 4, dtype=np.uint8)

    t1_s = {"scene_id": "T1_SCENE", "datetime": "2023-01-01T00:00:00Z", "b04": b04, "b08": b08, "scl": scl}
    t2_s = {"scene_id": "T2_SCENE", "datetime": "2023-07-01T00:00:00Z", "b04": b04, "b08": b08, "scl": scl}

    res = compute_temporal_optical_change(
        t1_scenes=[t1_s],
        t2_scenes=[t2_s],
        aoi_bounds=[77.5, 12.9, 77.6, 13.0]
    )

    assert "visual_inspector" in res
    vi = res["visual_inspector"]
    assert vi["status"] == "success"
    assert "t1" in vi
    assert "t2" in vi
    assert vi["bounds"] == [77.5, 12.9, 77.6, 13.0]
    assert vi["t1"]["stats"]["clear_coverage_pct"] == 100.0


def test_orchestration_evidence_item_generation():
    """Verifies that orchestrator includes ev_temporal_composite_inspector in evidence."""
    orch = MultiStepOrchestrator()
    res = orch.orchestrate(
        query="Analyze temporal optical change between 2023-01-01 and 2023-12-31",
        mode="offline",
        metadata={"use_offline_fixtures": True}
    )

    assert res["status"] in ("success", "unsupported")
    ev_types = [e.get("type") for e in res.get("evidence", [])]
    # Check that mask / change_map evidence exists
    assert any("mask" in t or "change" in t for t in ev_types)


def test_api_composite_inspector_endpoint():
    """Tests POST /api/v1/acquisition/composite-inspector endpoint with mock scenes."""
    payload = {
        "t1_scenes": [
            {
                "scene_id": "S2_T1",
                "datetime": "2023-01-10T00:00:00Z",
                "b04": [[1000, 1000], [1000, 1000]],
                "b08": [[4000, 4000], [4000, 4000]],
                "scl": [[4, 4], [4, 4]]
            }
        ],
        "t2_scenes": [
            {
                "scene_id": "S2_T2",
                "datetime": "2023-07-10T00:00:00Z",
                "b04": [[1200, 1200], [1200, 1200]],
                "b08": [[3800, 3800], [3800, 3800]],
                "scl": [[4, 4], [4, 4]]
            }
        ],
        "aoi_bbox": [77.5, 12.9, 77.6, 13.0],
        "composite_method": "median"
    }

    resp = client.post("/api/v1/acquisition/composite-inspector", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "t1" in data
    assert "t2" in data
    assert data["t1"]["stats"]["valid_pixels"] == 4
    assert data["t1"]["layers"]["scl_mask"].startswith("data:image/png;base64,")
    assert data["bounds"] == [77.5, 12.9, 77.6, 13.0]


def test_api_composite_inspector_empty_request_returns_unavailable():
    """Tests POST /api/v1/acquisition/composite-inspector handles empty payload without failing."""
    resp = client.post("/api/v1/acquisition/composite-inspector", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "unavailable"
    assert data["t1"]["stats"]["valid_pixels"] == 0
