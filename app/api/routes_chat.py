import logging
from fastapi import APIRouter
from app.chatbot.schemas import ChatRequest, ChatResponse
from app.chatbot.satquery_chatbot import satquery_chatbot

logger = logging.getLogger("satquery.api.chat")
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def satquery_chat(request: ChatRequest):
    """
    Dedicated SatQuery AI Chatbot endpoint powered by Qwen/Qwen2.5-VL-3B-Instruct
    hosted on Hugging Face Spaces (Astro406/satquery-qwen-vl).
    Accepts user message, multi-turn history, and optional SatQuery specialist evidence.
    Gracefully handles offline, standby, or queued Spaces without fabricating responses.
    """
    return satquery_chatbot.chat(
        message=request.message,
        history=request.history,
        evidence=request.evidence,
        workspace_context=request.workspace_context,
        model=request.model,
        conversation_id=request.conversation_id,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

@router.get("/chat/health")
def satquery_chat_health():
    """
    Status probe for the Hugging Face Spaces chatbot backend.
    Performs a lightweight metadata inspection without triggering model inference or GPU allocation.
    Returns:
    - status: 'online', 'standby', or 'offline'
    - available: boolean
    - provider: 'Hugging Face Spaces'
    - space_id: Hugging Face repository ID
    - configured_model: Model name
    - stage: Current Space runtime stage (e.g. RUNNING, SLEEPING)
    - base_url: Host endpoint URL
    """
    return satquery_chatbot.get_health_status()
