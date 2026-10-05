import logging
from fastapi import APIRouter
from app.chatbot.schemas import ChatRequest, ChatResponse
from app.chatbot.satquery_chatbot import satquery_chatbot

logger = logging.getLogger("satquery.api.chat")
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def satquery_chat(request: ChatRequest):
    """
    Dedicated SatQuery AI Chatbot endpoint powered by a local LLM via LM Studio.
    Accepts user message, multi-turn history, and optional SatQuery specialist evidence.
    Gracefully handles offline LM Studio server without fabricating responses.
    """
    return satquery_chatbot.chat(
        message=request.message,
        history=request.history,
        evidence=request.evidence,
        model=request.model,
        conversation_id=request.conversation_id,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

@router.get("/chat/health")
def satquery_chat_health():
    """
    Status probe for the local LM Studio chatbot backend.
    """
    available = satquery_chatbot.is_available()
    return {
        "status": "online" if available else "offline",
        "available": available,
        "provider": "LM Studio",
        "base_url": satquery_chatbot.client.base_url,
        "configured_model": satquery_chatbot.client.default_model or "dynamic_lookup"
    }
