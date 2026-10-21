"""Genuine End-to-End Test for Demo 4 (Optical + SAR Joint Analysis).

Tests the full end-to-end pipeline from the exact request payload emitted by Demo 4
through FastAPI router /api/v1/analyze, MultiStepOrchestrator, STAC composite retrieval
of authentic local offline fixtures, CrossSensorValidator, radiometric fusion, deterministic
verifier, and response serialization.
"""
import pytest
from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_demo4_e2e_disentangled_offline_behavior():
    """Verify Demo 4 offline request retrieves separate fixtures and honestly reports unsupported fusion."""
    payload = {
        "images": [],
        "query": "Perform joint optical and radar fusion for Mumbai [72.75, 18.88, 73.05, 19.30]",
        "mode": "orchestration"
    }

    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()

    # Core task, mode, and honest unsupported status
    assert data["task"] == "optical_sar"
    assert data["mode"] == "orchestration"
    assert data["confidence"] is None
    assert "Unsupported" in data["confidence_label"]
    assert "rejected" in data["answer"].lower() or "spatial coverage" in data["answer"].lower()
    assert data["verification"]["verification_status"] == "unsupported"

    # Multi-step execution trace DAG: separate optical and SAR retrieval, rejected fusion step
    trace = data.get("trace", {})
    exec_steps = trace.get("execution_steps", [])
    assert len(exec_steps) == 3

    assert exec_steps[0]["tool"] == "stac_composite_retrieve"
    assert exec_steps[0]["status"] == "completed"
    assert "optical" in exec_steps[0]["summary"].lower()

    assert exec_steps[1]["tool"] == "stac_composite_retrieve"
    assert exec_steps[1]["status"] == "completed"
    assert "sar" in exec_steps[1]["summary"].lower()

    assert exec_steps[2]["tool"] == "optical_sar_fusion"
    assert exec_steps[2]["status"] == "unsupported"

    # Evidence artifacts: no fabricated fused composite raster
    evidence = data.get("evidence", [])
    assert not any(e.get("type") == "fused" for e in evidence)

    # Disentangled source asset IDs in report provenance
    rep = data.get("reproducible_report", {})
    rep_prov = rep.get("provenance", {})
    source_ids = rep_prov.get("source_asset_ids", [])
    assert "local_sample_cartosat_optical_bengaluru" in source_ids
    assert "local_sample_risat_sar_mumbai" in source_ids
    assert rep_prov.get("aoi_containment_status") == "out_of_bounds_clamped"

    # Deterministic verifier skipped for unsupported status
    verification = data.get("verification", {})
    assert verification.get("verification_status") == "unsupported"
    assert verification.get("deterministic_passed") is False


def test_demo4_e2e_honesty_on_ambiguous_request():
    """Verify orchestrator returns honest clarification rather than fabricated success on ambiguity."""
    payload = {
        "images": [],
        "query": "Perform joint optical and radar fusion",
        "mode": "orchestration"
    }

    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Must honestly report clarification required due to missing spatial coordinates
    assert data.get("is_clarification_required") is True
    assert "spatial_location" in data.get("missing_parameters", [])
    assert data.get("confidence") is None
