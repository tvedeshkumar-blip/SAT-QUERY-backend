import os
import sys
import base64
import numpy as np
import pytest
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.agent.controller import agent_controller
from app.agent.router import TaskRouter
from app.schemas.analysis import AnalysisRequest, ImageInput
from app.remote_sensing.preprocessing import convert_array_to_base64_png
from app.remote_sensing.validation import ValidationError

test_client = TestClient(app)

SUGGESTED_TEST_QUESTION = (
    "Describe the visible land cover in this satellite image. "
    "Identify apparent vegetation, buildings, roads, and water where visible. "
    "Distinguish observations from uncertain interpretations. "
    "Do not claim to have performed multispectral analysis."
)


def _create_rgb_base64(width=128, height=128):
    """Generates synthetic RGB satellite preview image in base64."""
    img_arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Green canopy patch
    img_arr[10:60, 10:60] = [30, 140, 40]
    # Water/shadow patch
    img_arr[70:110, 70:110] = [10, 25, 45]
    # Bright built-up patch
    img_arr[10:40, 80:110] = [180, 185, 190]
    # Mixed ground
    img_arr[70:110, 10:50] = [110, 95, 70]

    pil_img = Image.fromarray(img_arr, mode="RGB")
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _create_rgba_base64(width=128, height=128):
    """Generates synthetic RGBA satellite preview image with transparent alpha channel."""
    img_arr = np.zeros((height, width, 4), dtype=np.uint8)
    # Green canopy patch with full opacity
    img_arr[10:60, 10:60] = [30, 140, 40, 255]
    # Water patch with full opacity
    img_arr[70:110, 70:110] = [10, 25, 45, 255]
    # Transparent margin/corner (alpha = 0)
    img_arr[:10, :] = [0, 0, 0, 0]
    img_arr[:, :10] = [0, 0, 0, 0]
    # Semi-transparent overlay (alpha = 128)
    img_arr[60:70, 60:70] = [200, 200, 200, 128]
    # Remaining pixels fully opaque
    img_arr[img_arr[:, :, 3] == 0] = [0, 0, 0, 0]

    pil_img = Image.fromarray(img_arr, mode="RGBA")
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


class TestSingleImageVQARepair:
    """End-to-end tests for the Single-Image VQA repair and verification."""

    def test_successful_vqa_request_with_valid_rgb_image(self):
        """1. Successful VQA request with a valid RGB image using the suggested test question."""
        rgb_b64 = _create_rgb_base64()
        req = AnalysisRequest(
            query=SUGGESTED_TEST_QUESTION,
            mode="single",
            images=[ImageInput(data=rgb_b64, filename="preview_rgb.jpg", mimeType="image/jpeg")]
        )
        resp = agent_controller.process_request(req)

        assert resp.task == "vqa", f"Expected task 'vqa' but got '{resp.task}'"
        assert "BigEarthNetResNet50Classifier" not in resp.models, "BigEarthNet must NOT be invoked for VQA"
        assert resp.actual_model_used in ("gemini-3.8-flash", "SpectralStatisticsBaseline")
        assert resp.fallback_used is True
        assert resp.confidence is None
        assert len(resp.evidence) >= 1
        assert resp.evidence[0].type == "processed"

        # Answer must ground visible features and distinguish observations from uncertainties/interpretations
        ans = resp.answer.lower()
        assert "vegetation" in ans or "canopy" in ans
        assert "uncertain" in ans or "limitations" in ans or "interpretation" in ans or "not observable" in ans
        assert "multispectral" in ans

    def test_successful_vqa_request_with_rgba_image(self):
        """2. Successful VQA request with an RGBA image containing transparent alpha channel."""
        rgba_b64 = _create_rgba_base64()
        req = AnalysisRequest(
            query=SUGGESTED_TEST_QUESTION,
            mode="vqa",
            images=[ImageInput(data=rgba_b64, filename="preview_rgba.png", mimeType="image/png")]
        )
        resp = agent_controller.process_request(req)

        assert resp.task == "vqa"
        assert "BigEarthNetResNet50Classifier" not in resp.models
        assert resp.actual_model_used in ("gemini-3.8-flash", "SpectralStatisticsBaseline")
        assert len(resp.evidence) >= 1
        assert resp.evidence[0].data_base64.startswith("data:image/png;base64,")

    def test_missing_image_rejection_for_vqa(self):
        """3. Missing image request is rejected with a clean informative error."""
        # Via controller directly
        req = AnalysisRequest(
            query=SUGGESTED_TEST_QUESTION,
            mode="vqa",
            images=[]
        )
        with pytest.raises(ValidationError) as exc_info:
            agent_controller.process_request(req)
        assert "No image provided" in str(exc_info.value)
        assert "Single-Image VQA requires an input satellite image" in str(exc_info.value)

        # Via FastAPI /api/v1/analyze/vqa endpoint
        api_resp = test_client.post(
            "/api/v1/analyze/vqa",
            json={"query": SUGGESTED_TEST_QUESTION, "images": [], "mode": "vqa"}
        )
        assert api_resp.status_code == 400
        assert "No image provided" in api_resp.json()["detail"]

    def test_malformed_base64_data_rejection(self):
        """4. Malformed base64 image data is rejected with a validation error."""
        req = AnalysisRequest(
            query=SUGGESTED_TEST_QUESTION,
            mode="single",
            images=[ImageInput(data="not_valid_base64_content!!!", filename="bad.png", mimeType="image/png")]
        )
        with pytest.raises(ValidationError) as exc_info:
            agent_controller.process_request(req)
        assert "Failed to decode base64" in str(exc_info.value) or "malformed" in str(exc_info.value)

    def test_unsupported_mime_type_rejection(self):
        """5. Unsupported MIME type is rejected with a clear error."""
        req = AnalysisRequest(
            query=SUGGESTED_TEST_QUESTION,
            mode="single",
            images=[ImageInput(data=_create_rgb_base64(), filename="document.pdf", mimeType="application/pdf")]
        )
        with pytest.raises(ValidationError) as exc_info:
            agent_controller.process_request(req)
        assert "Unsupported file format" in str(exc_info.value)

    def test_explicit_vqa_mode_routing_does_not_invoke_bigearthnet(self):
        """6 & 7. Explicit VQA mode routing routes to VQA and proves BigEarthNet is not invoked."""
        queries_mentioning_land_cover = [
            SUGGESTED_TEST_QUESTION,
            "Describe the visible land cover in this satellite image.",
            "What type of land cover is visible?",
            "Identify apparent land cover features including vegetation and buildings.",
        ]

        for q in queries_mentioning_land_cover:
            # Mode set to 'vqa'
            decision_vqa = TaskRouter.route(q, image_count=1, modalities=["OPTICAL"], mode_hint="vqa")
            assert decision_vqa.task == "vqa", f"Failed for query '{q}' in mode 'vqa'"
            assert "BigEarthNetResNet50Classifier" not in decision_vqa.selected_models
            assert "GenericVLMOrchestrator" in decision_vqa.selected_models

            # Mode set to 'single'
            decision_single = TaskRouter.route(q, image_count=1, modalities=["OPTICAL"], mode_hint="single")
            assert decision_single.task == "vqa", f"Failed for query '{q}' in mode 'single'"
            assert "BigEarthNetResNet50Classifier" not in decision_single.selected_models

    def test_classification_mode_still_rejects_four_channel_preview(self):
        """8. Proof that classification mode still strictly rejects 4-channel visual preview images."""
        rgba_b64 = _create_rgba_base64()
        req = AnalysisRequest(
            query="Classify land cover",
            mode="classification",
            images=[ImageInput(data=rgba_b64, filename="preview_4channel.png", mimeType="image/png")]
        )
        resp = agent_controller.process_request(req)

        assert resp.task == "land_cover_classification"
        assert resp.implementation_status == "invalid_input"
        assert resp.confidence is None
        assert resp.confidence_label == "Invalid Input (Multispectral 10-Band Required)"
        assert "Asset has 3 channel(s)" in resp.answer or "Asset has 4 channel(s)" in resp.answer or "10-band" in resp.answer
        assert "BigEarthNetv2 is a 10-band Sentinel-2 multispectral model" in resp.answer

    def test_accurate_model_selected_vs_model_used_metadata(self):
        """9 & 10. Accurate model-selected versus model-used metadata reporting."""
        rgb_b64 = _create_rgb_base64()
        req = AnalysisRequest(
            query=SUGGESTED_TEST_QUESTION,
            mode="vqa",
            images=[ImageInput(data=rgb_b64, filename="preview.jpg", mimeType="image/jpeg")]
        )
        resp = agent_controller.process_request(req)

        # Primary model is the intended vision-language model
        assert resp.primary_model in ("Florence-2-base", "qwen2.5-vl-3b-instruct")
        # Actual model used is the truthful fallback executed
        assert resp.actual_model_used in ("gemini-3.8-flash", "SpectralStatisticsBaseline")
        assert resp.fallback_used is True

        step_names = [s.step for s in resp.trace.steps]
        assert "MODEL_EXECUTED" in step_names
        assert "RESPONSE_GENERATED" in step_names

    def test_live_inference_on_existing_preview_image_file(self):
        """11. Live smoke test using the existing preview satellite image from data/retrieved."""
        preview_path = os.path.join(backend_dir, "..", "data", "retrieved", "S2A_43PGQ_20231119_preview.jpg")
        preview_path = os.path.normpath(preview_path)

        if not os.path.exists(preview_path):
            pytest.skip(f"Preview image '{preview_path}' not found on disk")

        with open(preview_path, "rb") as f:
            b64_data = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

        req = AnalysisRequest(
            query=SUGGESTED_TEST_QUESTION,
            mode="single",
            images=[ImageInput(
                data=b64_data,
                filename="S2A_43PGQ_20231119_preview.jpg",
                mimeType="image/jpeg",
                asset_category="visual_preview"
            )]
        )

        resp = agent_controller.process_request(req)
        assert resp.task == "vqa"
        assert resp.actual_model_used in ("gemini-3.8-flash", "SpectralStatisticsBaseline")
        assert resp.fallback_used is True
        assert len(resp.evidence) >= 1
        assert "vegetation" in resp.answer.lower() or "land cover" in resp.answer.lower()

    def test_fastapi_vqa_endpoint_end_to_end(self):
        """12. Full HTTP round-trip through FastAPI /api/v1/analyze/vqa endpoint."""
        rgb_b64 = _create_rgb_base64()
        payload = {
            "query": SUGGESTED_TEST_QUESTION,
            "mode": "vqa",
            "images": [
                {
                    "data": rgb_b64,
                    "mimeType": "image/jpeg",
                    "filename": "smoke_test_preview.jpg"
                }
            ]
        }
        res = test_client.post("/api/v1/analyze/vqa", json=payload)
        assert res.status_code == 200, f"HTTP error {res.status_code}: {res.text}"
        data = res.json()

        assert data["task"] == "vqa"
        assert data["actual_model_used"] in ("gemini-3.8-flash", "SpectralStatisticsBaseline")
        assert data["fallback_used"] is True
        assert len(data["evidence"]) >= 1
