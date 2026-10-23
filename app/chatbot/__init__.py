"""
SatQuery AI Chatbot Subsystem.
Powered by Qwen/Qwen2.5-VL-3B-Instruct hosted on Hugging Face Spaces (Astro406/satquery-qwen-vl).
Independent subsystem completely decoupled from hosted APIs (e.g. Gemini).
"""

from app.chatbot.schemas import ChatMessage, ChatRequest, ChatResponse
from app.chatbot.hf_client import (
    HuggingFaceClient,
    HFClientError,
    HFSpaceUnavailableError,
    HFSpaceStandbyError,
    HFSpaceTimeoutError,
    HFSpaceRateLimitError,
    HFSpaceInferenceError
)
from app.chatbot.lmstudio_client import (
    LMStudioClient,
    LMStudioClientError,
    LMStudioConnectionError,
    LMStudioAPIError
)
from app.chatbot.satquery_chatbot import SatQueryChatbot, satquery_chatbot
from app.chatbot.prompts import (
    SATQUERY_SYSTEM_PROMPT,
    format_evidence_context,
    assemble_qwen_prompt
)

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "HuggingFaceClient",
    "HFClientError",
    "HFSpaceUnavailableError",
    "HFSpaceStandbyError",
    "HFSpaceTimeoutError",
    "HFSpaceRateLimitError",
    "HFSpaceInferenceError",
    "LMStudioClient",
    "LMStudioClientError",
    "LMStudioConnectionError",
    "LMStudioAPIError",
    "SatQueryChatbot",
    "satquery_chatbot",
    "SATQUERY_SYSTEM_PROMPT",
    "format_evidence_context",
    "assemble_qwen_prompt",
]
