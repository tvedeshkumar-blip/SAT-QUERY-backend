import os
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from app.models.grounding.grounder import (
    GroundingProvider,
    ClassicalBaselineGrounder,
    OWLViTGroundingProvider,
    extract_grounding_targets
)
from app.agent.router import TaskRouter
from app.agent.planner import AgentPlanner
from app.agent.registry import ModelRegistry


def test_1_provider_initialization_without_weights():
    """Test 1: Provider initializes cleanly without weights mounted."""
    owlvit = OWLViTGroundingProvider(checkpoint_path="non_existent_owlvit/")
    assert owlvit.model_name == "google/owlvit-base-patch32"
    assert owlvit._is_loaded is False
    assert owlvit.is_remote_sensing_adapted is False
    assert owlvit.adaptation_status == "pretrained_general_grounding"

    provider = GroundingProvider()
    assert provider.baseline.model_name == "ClassicalBaselineGrounder"
    assert provider.deep_grounder.model_name == "google/owlvit-base-patch32"


def test_2_missing_checkpoint_reports_unavailable():
    """Test 2: Missing checkpoint reports unavailable without throwing exceptions."""
    owlvit = OWLViTGroundingProvider(checkpoint_path="invalid_path_to_owlvit/")
    assert owlvit.is_available is False
    loaded = owlvit.load()
    assert loaded is False
    assert owlvit.implementation_status == "unavailable"


def test_3_missing_checkpoint_triggers_fallback():
    """Test 3: Missing checkpoint cleanly executes deterministic ClassicalBaselineGrounder fallback."""
    provider = GroundingProvider()
    provider.deep_grounder.checkpoint_path = "non_existent_dir_12345/"

    img = np.full((80, 80, 3), 40, dtype=np.uint8)
    res = provider.predict([img], query="Highlight the water body.")

    assert res["fallback_used"] is True
    assert res["primary_model"] == "google/owlvit-base-patch32"
    assert res["actual_model_used"] == "ClassicalBaselineGrounder"
    assert res["model_status"] == "checkpoint_not_found"
    assert res["implementation_status"] == "baseline"
    assert "boxes" in res
    assert len(res["evidence"]) >= 2


def test_4_target_query_extraction():
    """Test 4: Target query extractor accurately parses natural language prompts."""
    # Common remote sensing entities
    assert "water body" in extract_grounding_targets("Highlight the water body.")
    assert "buildings" in extract_grounding_targets("Find all the buildings.")
    assert "ships" in extract_grounding_targets("Locate ships near the port.")
    assert "road" in extract_grounding_targets("Where are the roads?")
    assert "vegetation" in extract_grounding_targets("Show me the vegetation cover.")

    # Arbitrary targets
    custom_targets = extract_grounding_targets("Find the solar panels.")
    assert "solar panels" in custom_targets


def test_5_no_fabricated_confidence_on_baseline():
    """Test 5: Baseline fallback does not invent confidence probabilities."""
    baseline = ClassicalBaselineGrounder()
    img = np.full((60, 60, 3), 150, dtype=np.uint8)
    res = baseline.predict([img], query="Locate vegetation.")

    assert res["confidence"] is None
    assert "Not available" in res["confidence_label"]
    for box in res.get("boxes", []):
        assert box.get("confidence") is None


def test_6_model_provenance_structure():
    """Test 6: Model provenance correctly identifies actual model and target queries in both neural and fallback states."""
    provider = GroundingProvider()
    img = np.full((60, 60, 3), 150, dtype=np.uint8)
    res = provider.predict([img], query="Find the buildings.")

    assert "model_provenance" in res
    prov = res["model_provenance"]
    assert prov["primary_model"] == "google/owlvit-base-patch32"
    assert "target_query" in prov

    if provider.deep_grounder.is_available and not res["fallback_used"]:
        # Physical checkpoint is mounted and genuinely executed
        assert prov["actual_model_used"] == "google/owlvit-base-patch32"
        assert prov["fallback_used"] is False
        assert prov["checkpoint_status"] == "loaded"
        assert res["model_status"] == "loaded"
        assert res["implementation_status"] == "real_model"
        assert res.get("confidence") is None  # Uncalibrated detector scores
    else:
        # Deterministic baseline fallback state
        assert prov["actual_model_used"] == "ClassicalBaselineGrounder"
        assert prov["fallback_used"] is True
        assert prov["checkpoint_status"] in ["checkpoint_not_found", "baseline"]
        assert res["model_status"] in ["checkpoint_not_found", "baseline"]
        assert res["implementation_status"] == "baseline"

    # Explicitly test deterministic fallback provenance when weights are forced absent
    fb_provider = GroundingProvider()
    fb_provider.deep_grounder.checkpoint_path = "non_existent_owlvit_weights/"
    fb_res = fb_provider.predict([img], query="Find the buildings.")
    fb_prov = fb_res["model_provenance"]
    assert fb_prov["primary_model"] == "google/owlvit-base-patch32"
    assert fb_prov["actual_model_used"] == "ClassicalBaselineGrounder"
    assert fb_prov["fallback_used"] is True
    assert fb_prov["checkpoint_status"] == "checkpoint_not_found"
    assert fb_res["model_status"] == "checkpoint_not_found"
    assert fb_res["implementation_status"] == "baseline"


def test_7_geotiff_multiband_and_empty_handling():
    """Test 7: Multiband rasters and edge-case inputs are processed cleanly."""
    baseline = ClassicalBaselineGrounder()

    # Case A: 4-band satellite image (e.g., RGB + NIR)
    img_4band = np.full((64, 64, 4), 100, dtype=np.uint8)
    res_4band = baseline.predict([img_4band], query="Highlight the water body.")
    assert "boxes" in res_4band
    assert len(res_4band["evidence"]) >= 2

    # Case B: Empty image list should raise ValueError
    with pytest.raises(ValueError):
        baseline.predict([])


def test_8_agent_routing_and_planning_for_grounding():
    """Test 8: Agent Router, Planner, and Registry coordinate grounding queries."""
    queries = [
        "Find the buildings.",
        "Highlight the water body.",
        "Locate the road.",
        "Where are the ships?",
        "Pinpoint the solar panels.",
        "Show me the vegetation."
    ]
    for q in queries:
        decision = TaskRouter.route(query=q, image_count=1, modalities=["OPTICAL"])
        assert decision.task == "grounding", f"Query '{q}' routed to '{decision.task}' instead of 'grounding'"

    # Planner verification
    plan = AgentPlanner.create_plan("grounding", ["OPTICAL"], [{}])
    assert "GroundingProvider" in plan["selected_models"]
    assert "ClassicalBaselineGrounder" in plan["fallback_models"]

    # Registry verification
    registry = ModelRegistry()
    adapter = registry.get_model("grounding")
    assert adapter.model_name == "GroundingProvider"


def test_9_mocked_owlvit_neural_inference_pass():
    """Test 9: Real OWL-ViT provider executes neural inference when model is loaded."""
    owlvit = OWLViTGroundingProvider()

    # Mock processor and model
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    # Mock outputs
    mock_outputs = MagicMock()
    mock_model.return_value = mock_outputs

    # Mock post-processed results: 1 detection box
    mock_boxes = [{
        "boxes": torch.tensor([[10.0, 15.0, 45.0, 55.0]]),
        "scores": torch.tensor([0.785]),
        "labels": torch.tensor([0]),
        "text_labels": ["water body"]
    }]
    mock_processor.return_value = {"pixel_values": torch.zeros((1, 3, 64, 64))}
    mock_processor.post_process_grounded_object_detection.return_value = mock_boxes
    mock_processor.post_process_object_detection.return_value = mock_boxes

    owlvit._processor = mock_processor
    owlvit._model = mock_model
    owlvit._is_loaded = True
    owlvit.implementation_status = "real_model"

    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    res = owlvit.predict([img], query="Highlight the water body.")

    assert res["model_type"] == "neural_model"
    assert res["primary_model"] == "google/owlvit-base-patch32"
    assert res["actual_model_used"] == "google/owlvit-base-patch32"
    assert res["fallback_used"] is False
    assert res["model_status"] == "loaded"
    assert res["is_remote_sensing_adapted"] is False
    assert len(res["boxes"]) == 1
    assert res["boxes"][0]["label"] == "water body"
    assert res["boxes"][0]["confidence"] == 0.785
    assert len(res["evidence"]) == 3
    assert res["evidence"][0]["type"] == "overlay"
    assert res["evidence"][1]["type"] == "mask"


def test_10_api_schema_backward_compatibility():
    """Test 10: API response maintains full schema backward compatibility for grounding."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.remote_sensing.preprocessing import convert_array_to_base64_png

    client = TestClient(app)
    img = np.full((64, 64, 3), 80, dtype=np.uint8)
    b64_img = convert_array_to_base64_png(img)

    payload = {
        "query": "Highlight the water body in this scene.",
        "mode": "auto",
        "images": [
            {"data": b64_img, "filename": "test.png", "role": "primary"}
        ]
    }
    resp = client.post("/api/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["task"] == "grounding"
    assert "primary_model" in data
    assert "actual_model_used" in data
    assert "fallback_used" in data
    assert "model_status" in data
    assert "model_provenance" in data
    assert "trace" in data
    assert len(data["evidence"]) >= 2


def test_11_physical_checkpoint_conditional():
    """Test 11: Real physical checkpoint test (runs only when checkpoint exists on disk)."""
    owlvit = OWLViTGroundingProvider()
    if not owlvit.is_available:
        pytest.skip("Physical OWL-ViT checkpoint not mounted; skipping physical weights test.")

    assert owlvit.load() is True
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    res = owlvit.predict([img], query="Find the buildings.")
    assert res["actual_model_used"] == "google/owlvit-base-patch32"
    assert res["fallback_used"] is False
    assert res["model_status"] == "loaded"
    assert res["implementation_status"] == "real_model"
    assert "boxes" in res
    assert len(res["evidence"]) >= 3
