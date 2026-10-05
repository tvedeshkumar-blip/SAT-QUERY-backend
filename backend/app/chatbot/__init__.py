"""
SatQuery AI Local Chatbot Subsystem.
Powered by a dedicated local/custom LLM through LM Studio's OpenAI-compatible API.
Independent subsystem completely decoupled from hosted APIs (e.g. Gemini).
"""

from app.chatbot.schemas import ChatMessage, ChatRequest, ChatResponse
from app.chatbot.lmstudio_client import LMStudioClient, LMStudioClientError, LMStudioConnectionError
from app.chatbot.satquery_chatbot import SatQueryChatbot, satquery_chatbot
from app.chatbot.prompts import SATQUERY_SYSTEM_PROMPT, format_evidence_context

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "LMStudioClient",
    "LMStudioClientError",
    "LMStudioConnectionError",
    "SatQueryChatbot",
    "satquery_chatbot",
    "SATQUERY_SYSTEM_PROMPT",
    "format_evidence_context",
]
