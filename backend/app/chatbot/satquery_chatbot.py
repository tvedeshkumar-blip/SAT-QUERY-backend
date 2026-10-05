"""
SatQuery AI Dedicated Local Chatbot Engine.
Orchestrates prompt assembly, conversation history, evidence injection,
and dispatch to the local LM Studio OpenAI-compatible endpoint.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from app.chatbot.schemas import ChatMessage, ChatResponse
from app.chatbot.lmstudio_client import (
    LMStudioClient, 
    LMStudioConnectionError, 
    LMStudioAPIError, 
    LMStudioClientError
)
from app.chatbot.prompts import SATQUERY_SYSTEM_PROMPT, format_evidence_context

logger = logging.getLogger("satquery.chatbot")

class SatQueryChatbot:
    """
    Dedicated SatQuery AI Chatbot powered by a local/custom LLM via LM Studio.
    Completely separate subsystem from hosted foundation APIs (such as Gemini).
    """

    def __init__(self, client: Optional[LMStudioClient] = None):
        self.client = client or LMStudioClient()

    def is_available(self) -> bool:
        """
        Returns True if the local LM Studio instance is actively running and reachable.
        """
        return self.client.is_reachable()

    def chat(
        self,
        message: str,
        history: Optional[List[Union[ChatMessage, Dict[str, str]]]] = None,
        evidence: Optional[Union[Dict[str, Any], List[Any]]] = None,
        model: Optional[str] = None,
        conversation_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> ChatResponse:
        """
        Executes a conversational turn with the local LLM.
        
        Args:
            message: User query or follow-up question.
            history: Prior conversation turns.
            evidence: Optional structured SatQuery telemetry (e.g. BIT-CD, OWL-ViT, indices).
            model: Optional model override.
            conversation_id: Session identifier.
            temperature: Sampling temperature.
            max_tokens: Token generation limit.
            
        Returns:
            ChatResponse with natural-language text and execution provenance.
        """
        target_model = model or self.client.default_model or "local-model"

        # Check connectivity before attempting execution
        if not self.is_available():
            logger.info("LM Studio reachability probe returned False. Returning offline status.")
            return ChatResponse(
                response=(
                    f"LM Studio local server is offline or unreachable at {self.client.base_url}. "
                    "To enable natural-language satellite chat, start LM Studio and ensure a local LLM is loaded."
                ),
                model=target_model if target_model != "local-model" else "unavailable",
                provider="LM Studio",
                status="unavailable",
                conversation_id=conversation_id,
                error="LM Studio local endpoint is offline (connection refused or timed out)."
            )

        # 1. Assemble messages array
        messages: List[Dict[str, str]] = []

        # System prompt with evidence injection
        full_system_prompt = SATQUERY_SYSTEM_PROMPT + format_evidence_context(evidence)
        messages.append({
            "role": "system",
            "content": full_system_prompt
        })

        # Multi-turn conversation history
        if history:
            for item in history:
                if isinstance(item, ChatMessage):
                    messages.append({"role": item.role, "content": item.content})
                elif isinstance(item, dict) and "role" in item and "content" in item:
                    messages.append({"role": str(item["role"]), "content": str(item["content"])})

        # Current user query
        messages.append({
            "role": "user",
            "content": message
        })

        # 2. Dispatch to LM Studio client
        temp = temperature if temperature is not None else 0.2
        tokens = max_tokens if max_tokens is not None else 1024

        try:
            result = self.client.chat_completion(
                messages=messages,
                model=model,
                temperature=temp,
                max_tokens=tokens
            )
            return ChatResponse(
                response=result["content"],
                model=result["model"],
                provider="LM Studio",
                status="success",
                conversation_id=conversation_id
            )
        except LMStudioConnectionError as ce:
            logger.warning(f"LM Studio connection error during chat execution: {ce}")
            return ChatResponse(
                response=(
                    f"Connection to LM Studio was lost during generation ({self.client.base_url}). "
                    "Ensure LM Studio server is running and the model has sufficient memory."
                ),
                model=target_model,
                provider="LM Studio",
                status="unavailable",
                conversation_id=conversation_id,
                error=str(ce)
            )
        except LMStudioAPIError as ae:
            logger.error(f"LM Studio API returned error: {ae}")
            return ChatResponse(
                response=f"LM Studio returned error HTTP {ae.status_code}. Please verify the model configuration.",
                model=target_model,
                provider="LM Studio",
                status="error",
                conversation_id=conversation_id,
                error=str(ae)
            )
        except Exception as e:
            logger.error(f"Unexpected error in SatQueryChatbot: {e}")
            return ChatResponse(
                response="An unexpected internal error occurred while communicating with the local LLM.",
                model=target_model,
                provider="LM Studio",
                status="error",
                conversation_id=conversation_id,
                error=str(e)
            )

# Global default chatbot instance
satquery_chatbot = SatQueryChatbot()
