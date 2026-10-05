import json
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.chatbot.schemas import ChatMessage, ChatRequest, ChatResponse
from app.chatbot.lmstudio_client import LMStudioClient, LMStudioConnectionError, LMStudioAPIError
from app.chatbot.satquery_chatbot import SatQueryChatbot, satquery_chatbot
from app.chatbot.prompts import SATQUERY_SYSTEM_PROMPT, format_evidence_context

client = TestClient(app)

# Helper mock for httpx responses
def make_mock_response(status_code: int, json_data: dict):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.text = json.dumps(json_data)
    return mock_resp


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


def test_2_successful_chatbot_response():
    """Verify successful end-to-end chat completion when LM Studio is reachable."""
    lm_client = LMStudioClient(base_url="http://127.0.0.1:1234/v1", default_model="qwen-rs-model")
    bot = SatQueryChatbot(client=lm_client)

    mock_completion_payload = {
        "id": "chatcmpl-test-001",
        "object": "chat.completion",
        "model": "qwen-rs-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Optical reflectance analysis reveals extensive coastal mangrove canopy."
            },
            "finish_reason": "stop"
        }],
        "usage": {"total_tokens": 42}
    }

    with patch.object(lm_client, "is_reachable", return_value=True):
        with patch("httpx.Client.post", return_value=make_mock_response(200, mock_completion_payload)):
            res = bot.chat(
                message="What land cover dominates the coastal zone?",
                conversation_id="conv-101"
            )

            assert isinstance(res, ChatResponse)
            assert res.status == "success"
            assert res.provider == "LM Studio"
            assert res.model == "qwen-rs-model"
            assert "mangrove canopy" in res.response
            assert res.conversation_id == "conv-101"
            assert res.error is None


def test_3_conversation_history():
    """Verify that multi-turn dialogue history is correctly formatted and passed to LM Studio."""
    lm_client = LMStudioClient(base_url="http://127.0.0.1:1234/v1", default_model="rs-llm")
    bot = SatQueryChatbot(client=lm_client)

    captured_payloads = []

    def mock_post(url, json=None, **kwargs):
        captured_payloads.append(json)
        return make_mock_response(200, {
            "model": "rs-llm",
            "choices": [{"message": {"role": "assistant", "content": "Water extent increased by 4.2%."}}]
        })

    history = [
        ChatMessage(role="user", content="Did the lake change size?"),
        ChatMessage(role="assistant", content="Bi-temporal analysis was initiated on the temporal pair.")
    ]

    with patch.object(lm_client, "is_reachable", return_value=True):
        with patch("httpx.Client.post", side_effect=mock_post):
            res = bot.chat(
                message="Give me the precise area delta.",
                history=history
            )

            assert res.status == "success"
            assert len(captured_payloads) == 1
            sent_messages = captured_payloads[0]["messages"]
            
            # Message 0 is system prompt
            assert sent_messages[0]["role"] == "system"
            # Message 1 & 2 are history
            assert sent_messages[1]["role"] == "user"
            assert sent_messages[1]["content"] == "Did the lake change size?"
            assert sent_messages[2]["role"] == "assistant"
            assert sent_messages[2]["content"] == "Bi-temporal analysis was initiated on the temporal pair."
            # Message 3 is current user query
            assert sent_messages[3]["role"] == "user"
            assert sent_messages[3]["content"] == "Give me the precise area delta."


def test_4_evidence_injection():
    """Verify that structured specialist evidence is injected with strict grounding instructions."""
    lm_client = LMStudioClient(base_url="http://127.0.0.1:1234/v1", default_model="rs-llm")
    bot = SatQueryChatbot(client=lm_client)

    captured_payloads = []

    def mock_post(url, json=None, **kwargs):
        captured_payloads.append(json)
        return make_mock_response(200, {
            "model": "rs-llm",
            "choices": [{"message": {"role": "assistant", "content": "Verified BIT-CD change area is 21.88%."}}]
        })

    evidence_data = {
        "model": "BIT-CD",
        "status": "loaded",
        "change_area_percent": 21.88,
        "fallback_used": False,
        "detections": [
            {"model": "google/owlvit-base-patch32", "label": "ship", "confidence": 0.89}
        ]
    }

    with patch.object(lm_client, "is_reachable", return_value=True):
        with patch("httpx.Client.post", side_effect=mock_post):
            res = bot.chat(
                message="Summarize the change detection findings.",
                evidence=evidence_data
            )

            assert res.status == "success"
            sent_messages = captured_payloads[0]["messages"]
            system_msg = sent_messages[0]["content"]

            assert "VERIFIED SATQUERY SPECIALIST EVIDENCE" in system_msg
            assert "BIT-CD" in system_msg
            assert "21.88" in system_msg
            assert "owlvit-base-patch32" in system_msg
            assert "DO NOT extrapolate, fabricate additional detections" in system_msg


def test_5_lmstudio_unavailable():
    """Verify graceful handling when LM Studio is not running (offline)."""
    lm_client = LMStudioClient(base_url="http://127.0.0.1:1234/v1", default_model="local-model")
    bot = SatQueryChatbot(client=lm_client)

    # Force is_reachable to False (server offline)
    with patch.object(lm_client, "is_reachable", return_value=False):
        res = bot.chat("Analyze this satellite raster.")

        assert res.status == "unavailable"
        assert res.provider == "LM Studio"
        assert "offline or unreachable" in res.response
        assert "http://127.0.0.1:1234/v1" in res.response
        assert res.error is not None
        # Must not fabricate an LLM answer
        assert "Analysis reveals" not in res.response


def test_6_api_endpoint_schema():
    """Verify POST /api/v1/chat and GET /api/v1/chat/health against FastAPI TestClient."""
    # Test 6a: Health check when offline
    with patch.object(satquery_chatbot, "is_available", return_value=False):
        resp_health = client.get("/api/v1/chat/health")
        assert resp_health.status_code == 200
        data_health = resp_health.json()
        assert data_health["status"] == "offline"
        assert data_health["provider"] == "LM Studio"

    # Test 6b: Chat endpoint when offline
    with patch.object(satquery_chatbot, "is_available", return_value=False):
        chat_req = {
            "message": "Where are the urban clusters?",
            "conversation_id": "test-sess-1",
            "history": []
        }
        resp = client.post("/api/v1/chat", json=chat_req)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unavailable"
        assert data["provider"] == "LM Studio"
        assert data["conversation_id"] == "test-sess-1"
        assert "offline" in data["response"]

    # Test 6c: Chat endpoint when online (mocked LM Studio)
    with patch.object(satquery_chatbot, "is_available", return_value=True):
        with patch.object(satquery_chatbot.client, "chat_completion", return_value={
            "content": "Identified 3 urban clusters in southern quadrant.",
            "model": "meta-llama-3-8b-instruct",
            "usage": {"total_tokens": 30}
        }):
            chat_req = {
                "message": "Where are the urban clusters?",
                "conversation_id": "test-sess-2",
                "evidence": {"urban_clusters": 3}
            }
            resp = client.post("/api/v1/chat", json=chat_req)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert data["provider"] == "LM Studio"
            assert data["model"] == "meta-llama-3-8b-instruct"
            assert "3 urban clusters" in data["response"]


def test_7_no_gemini_dependency():
    """Verify that the SatQuery chatbot subsystem has zero dependency on Gemini or Google GenAI."""
    import importlib
    module_names = [
        "app.chatbot.satquery_chatbot",
        "app.chatbot.lmstudio_client",
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
