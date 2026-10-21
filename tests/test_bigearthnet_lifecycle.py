import os
import io
import base64
import threading
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.models.classification.bigearthnet_classifier import (
    BigEarthNetResNet50Classifier,
    build_resnet50_s2_architecture,
    BIGEARTHNET_19_CLASSES,
    V020_BAND_NAMES,
)
from app.agent.registry import ModelRegistry, model_registry
from app.agent.tools import tool_registry


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def synthetic_valid_checkpoint(tmp_path):
    """
    Creates a genuine, valid 10-channel ResNet-50 safetensors checkpoint for deterministic testing.
    """
    from safetensors.torch import save_file

    model = build_resnet50_s2_architecture()
    state_dict = model.state_dict()
    ckpt_path = tmp_path / "model.safetensors"
    save_file(state_dict, str(ckpt_path))
    return str(ckpt_path)


class TestBigEarthNetLoadingLifecycle:
    """
    Validates the end-to-end loading lifecycle, thread-safety, lazy loading,
    and diagnostic integrity for BigEarthNet ResNet-50.
    """

    def test_configured_valid_checkpoint_loads_successfully(self, synthetic_valid_checkpoint):
        """1. A configured valid checkpoint loads successfully into memory."""
        clf = BigEarthNetResNet50Classifier(checkpoint_path=synthetic_valid_checkpoint)
        assert clf.is_loaded is False
        assert clf.is_trained is False
        assert clf.status == "unavailable"

        clf.load()

        assert clf.is_loaded is True
        assert clf.is_trained is True
        assert clf.status == "loaded"
        assert clf.implementation_status == "pretrained_model"
        assert clf.model is not None

        # Verify inference forward pass works with loaded weights
        dummy_10band = np.random.uniform(500, 3000, (120, 120, 10)).astype(np.float32)
        res = clf.predict([dummy_10band])
        assert res["task"] == "land_cover_classification"
        assert res["model_status"] == "loaded"
        assert len(res["class_probabilities"]) == 19
        assert res["fallback_used"] is False

    def test_missing_or_invalid_checkpoint_produces_honest_failure_state(self, tmp_path):
        """2. A missing or invalid checkpoint produces a useful, honest failure state without mock fallback."""
        nonexistent_path = str(tmp_path / "nonexistent_model.safetensors")
        clf = BigEarthNetResNet50Classifier(checkpoint_path=nonexistent_path)

        with pytest.raises(FileNotFoundError) as exc_info:
            clf.load()

        assert "was not found at" in str(exc_info.value)
        assert "never silently substitute random weights" in str(exc_info.value)
        assert clf.is_loaded is False
        assert clf.is_trained is False
        assert clf.status == "model_loading_error"
        assert clf.implementation_status == "model_loading_error"

    def test_classification_request_triggers_lazy_loading(self, synthetic_valid_checkpoint, monkeypatch):
        """3. A classification request triggers loading when lazy loading is intended."""
        monkeypatch.setenv("BIGEARTHNET_CHECKPOINT_PATH", synthetic_valid_checkpoint)

        # Fresh registry instance with lazy-loaded classifier
        registry = ModelRegistry()
        clf = registry.get_model("classification")
        # get_model should have triggered loading
        assert clf.is_loaded is True
        assert clf.is_trained is True
        assert clf.status == "loaded"

    def test_models_endpoint_reports_actual_state_before_and_after_loading(self, synthetic_valid_checkpoint, monkeypatch):
        """4. /api/v1/models reports the actual model state before and after loading."""
        test_client = TestClient(app)

        # Temporarily create a fresh classifier in an unloaded state
        unloaded_clf = BigEarthNetResNet50Classifier(checkpoint_path=synthetic_valid_checkpoint)
        orig_clf = model_registry._models["classification"]

        try:
            model_registry._models["classification"] = unloaded_clf
            model_registry._models["land_cover_classification"] = unloaded_clf

            # Before loading: API must report unavailable and is_trained=False
            resp_before = test_client.get("/api/v1/models")
            assert resp_before.status_code == 200
            models_before = resp_before.json()["models"]
            clf_before = [m for m in models_before if m["task"] == "classification"][0]
            assert clf_before["status"] == "unavailable"
            assert clf_before["is_trained"] is False

            # Trigger loading via get_model
            model_registry.get_model("classification")

            # After loading: API must report loaded and is_trained=True
            resp_after = test_client.get("/api/v1/models")
            assert resp_after.status_code == 200
            models_after = resp_after.json()["models"]
            clf_after = [m for m in models_after if m["task"] == "classification"][0]
            assert clf_after["status"] == "loaded"
            assert clf_after["is_trained"] is True

        finally:
            model_registry._models["classification"] = orig_clf
            model_registry._models["land_cover_classification"] = orig_clf

    def test_both_classification_task_names_report_consistent_state(self, synthetic_valid_checkpoint):
        """5. Both classification task names ('classification' and 'land_cover_classification') report consistent state."""
        registry = ModelRegistry()
        clf1 = registry.get_model("classification")
        clf2 = registry.get_model("land_cover_classification")

        assert clf1 is clf2
        assert clf1.status == clf2.status
        assert clf1.is_trained == clf2.is_trained

        detailed = registry.list_models_detailed()
        c_entry = [m for m in detailed if m["task"] == "classification"][0]
        lcc_entry = [m for m in detailed if m["task"] == "land_cover_classification"][0]

        assert c_entry["status"] == lcc_entry["status"]
        assert c_entry["is_trained"] == lcc_entry["is_trained"]
        assert c_entry["implementation_status"] == lcc_entry["implementation_status"]

    def test_repeated_and_concurrent_requests_do_not_reload_weights(self, synthetic_valid_checkpoint):
        """6. Repeated and concurrent requests do not reload weights unnecessarily (thread-safe)."""
        clf = BigEarthNetResNet50Classifier(checkpoint_path=synthetic_valid_checkpoint)
        load_count = 0
        original_build = build_resnet50_s2_architecture

        def counting_build():
            nonlocal load_count
            load_count += 1
            return original_build()

        clf_module = __import__("app.models.classification.bigearthnet_classifier", fromlist=["build_resnet50_s2_architecture"])
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(clf_module, "build_resnet50_s2_architecture", counting_build)

        threads = []
        errors = []

        def worker():
            try:
                clf.load()
            except Exception as e:
                errors.append(e)

        for _ in range(8):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        monkeypatch.undo()

        assert len(errors) == 0
        assert clf.is_loaded is True
        assert load_count == 1  # Exactly one architecture construction occurred across all threads

    def test_existing_non_classification_routes_continue_working(self, test_client):
        """7. Existing non-classification routes continue working."""
        # Health check
        resp_health = test_client.get("/health")
        assert resp_health.status_code == 200
        assert resp_health.json()["status"] == "healthy"

        # Models route
        resp_models = test_client.get("/api/v1/models")
        assert resp_models.status_code == 200
        assert "models" in resp_models.json()

        # VQA analysis route with valid visual preview image
        tiny_png = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAFElEQVR42mNk+M9Qz8DAwM"
            "DAAAAZAwH+T5r1qAAAAABJRU5ErkJggg=="
        )
        vqa_payload = {
            "query": "Is there cloud cover in this scene?",
            "mode": "single",
            "images": [{"filename": "preview.png", "data": tiny_png, "mimeType": "image/png"}]
        }
        resp_vqa = test_client.post("/api/v1/analyze/vqa", json=vqa_payload)
        assert resp_vqa.status_code == 200

    def test_no_mock_predictions_used_as_fallback(self):
        """8. No mock predictions are used as a fallback when weights are missing."""
        clf = BigEarthNetResNet50Classifier()
        clf._is_loaded = False
        clf.model = None

        dummy_img = np.random.uniform(500, 3000, (120, 120, 10)).astype(np.float32)
        with pytest.raises(RuntimeError) as exc_info:
            clf.predict([dummy_img])

        assert "Cannot run inference without authentic weights" in str(exc_info.value)

    def test_invalid_checkpoint_produces_safe_api_diagnostic(self, tmp_path, test_client, monkeypatch):
        """Verifies that an invalid checkpoint produces a visible diagnostic API state rather than an unhandled crash."""
        bad_checkpoint = str(tmp_path / "bad_checkpoint.safetensors")
        monkeypatch.setenv("BIGEARTHNET_CHECKPOINT_PATH", bad_checkpoint)

        # Force un-load in registry to test diagnostic response on next request
        broken_clf = BigEarthNetResNet50Classifier(checkpoint_path=bad_checkpoint)
        orig_clf = model_registry._models["land_cover_classification"]

        try:
            model_registry._models["land_cover_classification"] = broken_clf
            model_registry._models["classification"] = broken_clf

            # Send classify request with 10-band synthetic raster
            dummy_arr = np.random.uniform(500, 3000, (120, 120, 10)).astype(np.float32)
            # Create a simple TIFF payload
            import rasterio
            from rasterio.transform import from_origin

            bio = io.BytesIO()
            transform = from_origin(500000.0, 2000000.0, 10.0, 10.0)
            with rasterio.open(
                bio,
                "w",
                driver="GTiff",
                height=120,
                width=120,
                count=10,
                dtype=rasterio.float32,
                crs="EPSG:32643",
                transform=transform,
            ) as dst:
                for b in range(10):
                    dst.write(dummy_arr[:, :, b], b + 1)

            b64_tif = base64.b64encode(bio.getvalue()).decode("utf-8")

            payload = {
                "query": "Classify land cover",
                "mode": "classification",
                "images": [
                    {
                        "data": b64_tif,
                        "filename": "test_s2_10band.tif",
                        "mimeType": "image/tiff",
                        "isGeoTIFF": True,
                    }
                ]
            }

            resp = test_client.post("/api/v1/analyze/classify", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["implementation_status"] == "model_loading_error"
            assert data["model_status"] == "model_loading_error"
            assert "Model loading failed" in data["answer"] or "not found" in data["answer"]

        finally:
            model_registry._models["land_cover_classification"] = orig_clf
            model_registry._models["classification"] = orig_clf
