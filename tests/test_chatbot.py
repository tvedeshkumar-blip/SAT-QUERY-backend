"""
Automated Test Suite for SatQuery AI Chatbot Subsystem (Hugging Face Spaces Qwen-VL).
Covers:
- Root & general health endpoints
- Chat health endpoint under online, standby, and offline states
- Successful inference with mocked Gradio Client
- Correct argument and endpoint mapping (question, api_name="/predict")
- Empty and oversized message validation
- Timeout, queue delay, and connection failure handling
- Unexpected upstream response formats
- Multi-turn conversation history preservation
- Grounding & scientific integrity rules (BIT-CD, OWL-ViT, SAR, scene metadata)
- CORS headers and methods
- Frontend response schema compatibility
"""

import json
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.chatbot.schemas import ChatMessage, ChatRequest, ChatResponse
from app.chatbot.hf_client import (
    HuggingFaceClient,
    HFSpaceUnavailableError,
    HFSpaceStandbyError,
    HFSpaceTimeoutError,
    HFSpaceRateLimitError,
    HFSpaceInferenceError
)
from app.chatbot.satquery_chatbot import SatQueryChatbot, satquery_chatbot
from app.chatbot.prompts import (
    SATQUERY_SYSTEM_PROMPT,
    format_evidence_context,
    assemble_qwen_prompt
)

client = TestClient(app)


# =========================================================================
# 1. Scientific Integrity Prompt & Grounding Constraints
# =========================================================================

def test_1_scientific_integrity_system_prompt():
    """Verify that the SatQuery system prompt strictly enforces all required integrity rules."""
    prompt = SATQUERY_SYSTEM_PROMPT
    assert "SatQuery AI, a satellite imagery analysis assistant" in prompt
    assert "Never invent satellite observations" in prompt
    assert "Never fabricate measurements" in prompt
    assert "Never claim a model ran when it did not" in prompt
    assert "Distinguish model observations from interpretation" in prompt
    assert "Preserve model provenance" in prompt
    assert "Treat neural confidence scores as uncalibrated" in prompt
    assert "If evidence is unavailable, say so" in prompt
    assert "Do not pretend to have access to imagery that was not supplied" in prompt
    assert "Do not fabricate coordinates, CRS, sensor information, dates, or detected objects" in prompt
    assert "BIT-CD" in prompt
    assert "OWL-ViT" in prompt
    assert "SAR" in prompt


# =========================================================================
# 2. Successful Inference with Mocked Gradio Client & Argument Mapping
# =========================================================================

def test_2_successful_huggingface_inference():
    """Verify successful end-to-end chat completion when HF Space responds cleanly."""
    mock_hf_client = HuggingFaceClient(
        space_id="Astro406/satquery-qwen-vl",
        api_name="/predict",
        request_timeout=65.0
    )
    bot = SatQueryChatbot(client=mock_hf_client)

    captured_kwargs = {}

    def mock_predict(question, api_name=None):
        captured_kwargs["question"] = question
        captured_kwargs["api_name"] = api_name
        return "Optical reflectance analysis reveals extensive coastal mangrove canopy."

    mock_gradio = MagicMock()
    mock_gradio.predict.side_effect = mock_predict

    with patch.object(mock_hf_client, "_get_or_create_client", return_value=mock_gradio):
        with patch.object(mock_hf_client, "probe_status", return_value={"status": "online", "stage": "RUNNING"}):
            res = bot.chat(
                message="What land cover dominates the coastal zone?",
                conversation_id="conv-101"
            )

            assert isinstance(res, ChatResponse)
            assert res.status == "success"
            assert res.provider == "Hugging Face Spaces"
            assert res.model == "Qwen/Qwen2.5-VL-3B-Instruct"
            assert "mangrove canopy" in res.response
            assert res.conversation_id == "conv-101"
            assert res.error is None

            # Verify exact Gradio API contract
            assert captured_kwargs["api_name"] == "/predict"
            assert "What land cover dominates the coastal zone?" in captured_kwargs["question"]
            assert "SatQuery AI" in captured_kwargs["question"]


# =========================================================================
# 3. Conversation History & Context Preservation
# =========================================================================

def test_3_conversation_history_assembly():
    """Verify multi-turn conversation history is formatted and included in prompt."""
    history = [
        ChatMessage(role="user", content="Did the lake surface area change?"),
        ChatMessage(role="assistant", content="Bi-temporal analysis was initiated on the temporal pair.")
    ]

    assembled = assemble_qwen_prompt(
        message="Give me the precise area delta.",
        history=history,
        evidence={"delta_km2": 4.2}
    )

    assert "Did the lake surface area change?" in assembled
    assert "Bi-temporal analysis was initiated" in assembled
    assert "Give me the precise area delta." in assembled
    assert "delta_km2" in assembled
    assert "RECENT CONVERSATION HISTORY" in assembled


# =========================================================================
# 4. Specialist Evidence Injection & Prompt Grounding
# =========================================================================

def test_4_evidence_injection():
    """Verify that structured specialist evidence is injected with strict grounding instructions."""
    evidence_data = {
        "model": "BIT-CD",
        "status": "loaded",
        "change_area_percent": 21.88,
        "fallback_used": False,
        "detections": [
            {"model": "google/owlvit-base-patch32", "label": "ship", "confidence": 0.89}
        ]
    }

    assembled = assemble_qwen_prompt(
        message="Summarize the change detection findings.",
        evidence=evidence_data
    )

    assert "VERIFIED SATQUERY SPECIALIST EVIDENCE" in assembled
    assert "BIT-CD" in assembled
    assert "21.88" in assembled
    assert "owlvit-base-patch32" in assembled
    assert "DO NOT extrapolate, fabricate additional detections" in assembled


# =========================================================================
# 5. Empty and Oversized Message Validation
# =========================================================================

def test_5_empty_and_whitespace_message_validation():
    """Verify empty or whitespace-only messages are rejected cleanly."""
    bot = SatQueryChatbot()

    # Empty string
    res_empty = bot.chat(message="")
    assert res_empty.status == "error"
    assert "non-empty" in res_empty.response

    # Whitespace only
    res_ws = bot.chat(message="    \n\t  ")
    assert res_ws.status == "error"
    assert "non-empty" in res_ws.response

    # API Route level validation via TestClient
    resp_empty_api = client.post("/api/v1/chat", json={"message": "   "})
    assert resp_empty_api.status_code == 422  # Pydantic validation error


def test_6_oversized_message_validation():
    """Verify excessively large messages are rejected to prevent upstream overload."""
    bot = SatQueryChatbot()
    huge_msg = "X" * 12000

    res = bot.chat(message=huge_msg)
    assert res.status == "error"
    assert "exceeds the maximum allowable length" in res.response

    # API Route level validation
    resp_api = client.post("/api/v1/chat", json={"message": huge_msg})
    assert resp_api.status_code == 422


# =========================================================================
# 6. Chat Health States (Online, Standby, Offline)
# =========================================================================

def test_7_chat_health_online():
    """Verify /api/v1/chat/health reports online when Space runtime is RUNNING."""
    mock_probe = {
        "status": "online",
        "available": True,
        "provider": "Hugging Face Spaces",
        "space_id": "Astro406/satquery-qwen-vl",
        "configured_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "stage": "RUNNING",
        "base_url": "https://astro406-satquery-qwen-vl.hf.space",
        "hardware": "zero-a10g"
    }

    with patch.object(satquery_chatbot.client, "probe_status", return_value=mock_probe):
        resp = client.get("/api/v1/chat/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "online"
        assert data["available"] is True
        assert data["provider"] == "Hugging Face Spaces"
        assert data["space_id"] == "Astro406/satquery-qwen-vl"
        assert data["stage"] == "RUNNING"


def test_8_chat_health_standby():
    """Verify /api/v1/chat/health reports standby when Space runtime is SLEEPING or BUILDING."""
    mock_probe = {
        "status": "standby",
        "available": False,
        "provider": "Hugging Face Spaces",
        "space_id": "Astro406/satquery-qwen-vl",
        "configured_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "stage": "SLEEPING",
        "base_url": "https://astro406-satquery-qwen-vl.hf.space",
        "hardware": "zero-a10g"
    }

    with patch.object(satquery_chatbot.client, "probe_status", return_value=mock_probe):
        resp = client.get("/api/v1/chat/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "standby"
        assert data["available"] is False
        assert data["stage"] == "SLEEPING"


def test_9_chat_health_offline():
    """Verify /api/v1/chat/health reports offline when Space is STOPPED or unreachable."""
    mock_probe = {
        "status": "offline",
        "available": False,
        "provider": "Hugging Face Spaces",
        "space_id": "Astro406/satquery-qwen-vl",
        "configured_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "stage": "STOPPED",
        "base_url": "https://astro406-satquery-qwen-vl.hf.space",
        "hardware": "unknown"
    }

    with patch.object(satquery_chatbot.client, "probe_status", return_value=mock_probe):
        resp = client.get("/api/v1/chat/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "offline"
        assert data["available"] is False
        assert data["stage"] == "STOPPED"


# =========================================================================
# 7. Error Handling: Timeouts, Queue Delays, and Upstream Failures
# =========================================================================

def test_10_timeout_handling():
    """Verify graceful handling when inference exceeds the request timeout."""
    mock_hf_client = HuggingFaceClient()
    bot = SatQueryChatbot(client=mock_hf_client)

    with patch.object(mock_hf_client, "probe_status", return_value={"status": "online", "stage": "RUNNING"}):
        with patch.object(mock_hf_client, "predict", side_effect=HFSpaceTimeoutError("Timeout after 65s")):
            res = bot.chat(message="Are any ships detected in the harbor?")
            assert res.status == "unavailable"
            assert "timed out" in res.response.lower()
            assert res.error is not None
            # Do NOT hallucinate an answer
            assert "ships" not in res.response


def test_11_standby_and_queue_delay_handling():
    """Verify informative response when Space is waking up from standby."""
    mock_hf_client = HuggingFaceClient()
    bot = SatQueryChatbot(client=mock_hf_client)

    with patch.object(mock_hf_client, "probe_status", return_value={"status": "standby", "stage": "SLEEPING"}):
        with patch.object(mock_hf_client, "predict", side_effect=HFSpaceStandbyError("Space is sleeping")):
            res = bot.chat(message="Compute NDVI for this scene.")
            assert res.status == "standby"
            assert "waking up" in res.response.lower()


def test_12_upstream_connection_failure():
    """Verify safe fallback when Space cannot be reached over the network."""
    mock_hf_client = HuggingFaceClient()
    bot = SatQueryChatbot(client=mock_hf_client)

    with patch.object(mock_hf_client, "probe_status", return_value={"status": "offline", "stage": "UNREACHABLE"}):
        res = bot.chat(message="Identify water bodies.")
        assert res.status == "unavailable"
        assert "offline or unreachable" in res.response


# =========================================================================
# 8. Unexpected Upstream Response Format Handling
# =========================================================================

def test_13_unexpected_response_formats():
    """Verify that list, tuple, and raw responses from Gradio are handled cleanly."""
    mock_hf_client = HuggingFaceClient()
    bot = SatQueryChatbot(client=mock_hf_client)

    # Case A: Gradio returns a single-item list
    mock_gradio_list = MagicMock()
    mock_gradio_list.predict.return_value = ["List-wrapped answer from Qwen."]
    with patch.object(mock_hf_client, "_get_or_create_client", return_value=mock_gradio_list):
        with patch.object(mock_hf_client, "probe_status", return_value={"status": "online", "stage": "RUNNING"}):
            res_list = bot.chat(message="Describe this feature.")
            assert res_list.status == "success"
            assert res_list.response == "List-wrapped answer from Qwen."

    # Case B: Gradio returns a raw string
    mock_gradio_str = MagicMock()
    mock_gradio_str.predict.return_value = "Direct string answer."
    with patch.object(mock_hf_client, "_get_or_create_client", return_value=mock_gradio_str):
        with patch.object(mock_hf_client, "probe_status", return_value={"status": "online", "stage": "RUNNING"}):
            res_str = bot.chat(message="Describe this feature again.")
            assert res_str.status == "success"
            assert res_str.response == "Direct string answer."


# =========================================================================
# 9. Safeguards for Workspace Scenes Without Analysis Evidence
# =========================================================================

def test_14_imported_workspace_scenes_without_analysis_evidence():
    """
    Verify prompt grounding when T1/T2 scenes are imported in workspace but perception pipeline has not run.
    Ensures model is strictly instructed not to fabricate pixel changes and to direct user to 'Execute Pipeline'.
    """
    workspace_ctx = {
        "mode": "bitemporal",
        "scene_count": 2,
        "scenes": [
            {
                "role": "primary",
                "scene_id": "S2A_MSIL2A_20230601T052651_N0509_R105_T43PGQ",
                "sensor": "MSI",
                "platform": "Sentinel-2",
                "crs": "EPSG:32643"
            },
            {
                "role": "secondary",
                "scene_id": "S2B_MSIL2A_20231201T052709_N0510_R105_T43PGQ",
                "sensor": "MSI",
                "platform": "Sentinel-2",
                "crs": "EPSG:32643"
            }
        ]
    }

    assembled = assemble_qwen_prompt(
        message="What scenes are currently imported in my workspace?",
        workspace_context=workspace_ctx,
        evidence=None
    )

    assert "ACTIVE WORKSPACE SCENE METADATA (PIPELINE NOT YET EXECUTED)" in assembled
    assert "S2A_MSIL2A_20230601T052651_N0509_R105_T43PGQ" in assembled
    assert "EPSG:32643" in assembled
    assert "Execute Pipeline" in assembled
    assert "NEVER fabricate, hallucinate, or guess pixel changes" in assembled


# =========================================================================
# 10. Root, Health, and CORS Verification
# =========================================================================

def test_15_root_and_general_health_endpoints():
    """Verify backend root (/) and general health (/health) endpoints."""
    # Root
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    data_root = resp_root.json()
    assert data_root["status"] == "online"
    assert "SatQuery AI" in data_root["title"]

    # General Backend Health
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    data_health = resp_health.json()
    assert data_health["status"] == "healthy"
    assert data_health["service"] == "SatQuery AI Backend"


def test_16_cors_configuration():
    """Verify that CORS preflight and origin validation are properly configured."""
    response = client.options(
        "/api/v1/chat",
        headers={
            "Origin": "https://sat-query-gamma.vercel.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://sat-query-gamma.vercel.app"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_17_no_gemini_dependency_in_chatbot():
    """Verify that the SatQuery chatbot subsystem has zero dependency on Gemini or Google GenAI."""
    import importlib
    module_names = [
        "app.chatbot.satquery_chatbot",
        "app.chatbot.hf_client",
        "app.chatbot.prompts",
        "app.chatbot.schemas",
        "app.api.routes_chat"
    ]

    for mod_name in module_names:
        mod = importlib.import_module(mod_name)
        module_source = open(mod.__file__, "r", encoding="utf-8").read()
        assert "google.genai" not in module_source, f"google.genai found in {mod.__file__}"
        assert "from google import genai" not in module_source, f"genai import found in {mod.__file__}"
        assert "GenericVLMSynthesisAdapter" not in module_source, f"Gemini adapter found in {mod.__file__}"
        assert "gemini-2.5-flash" not in module_source, f"Gemini model ID found in {mod.__file__}"


def test_18_zerogpu_quota_rate_limit_handling():
    """Verify that ZeroGPU quota limits return an informative unavailable response with instructions."""
    mock_hf_client = HuggingFaceClient()
    bot = SatQueryChatbot(client=mock_hf_client)

    with patch.object(mock_hf_client, "probe_status", return_value={"status": "online", "stage": "RUNNING"}):
        with patch.object(
            mock_hf_client,
            "predict",
            side_effect=HFSpaceRateLimitError("ZeroGPU quota reached: Exceeded runs limit")
        ):
            res = bot.chat(message="Detect ships in the satellite image.")
            assert res.status == "unavailable"
            assert "ZeroGPU" in res.response
            assert "HF_TOKEN" in res.response
            assert res.error is not None
