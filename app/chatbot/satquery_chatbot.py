"""
SatQuery AI Dedicated Chatbot Engine.
Orchestrates prompt assembly, conversation history, evidence injection,
and dispatch to the Hugging Face Spaces Qwen-VL endpoint (or fallback local client).
"""

import logging
from typing import Any, Dict, List, Optional, Union

from app.chatbot.schemas import ChatMessage, ChatResponse
from app.chatbot.hf_client import (
    HuggingFaceClient,
    HFSpaceUnavailableError,
    HFSpaceStandbyError,
    HFSpaceTimeoutError,
    HFSpaceRateLimitError,
    HFSpaceInferenceError
)
from app.chatbot.prompts import (
    SATQUERY_SYSTEM_PROMPT,
    format_evidence_context,
    assemble_qwen_prompt
)

logger = logging.getLogger("satquery.chatbot")

class SatQueryChatbot:
    """
    Dedicated SatQuery AI Chatbot powered by Qwen/Qwen2.5-VL-3B-Instruct hosted on
    Hugging Face Spaces (Astro406/satquery-qwen-vl).
    Completely decoupled from locally hosted LM Studio instances and hosted foundation APIs (such as Gemini).
    """

    def __init__(self, client: Optional[Any] = None):
        self.client = client or HuggingFaceClient()

    def is_available(self, timeout: float = 5.0) -> bool:
        """
        Returns True strictly when the underlying inference provider is online and ready.
        """
        if hasattr(self.client, "is_reachable"):
            return self.client.is_reachable(timeout=timeout)
        return True

    def get_health_status(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Probes provider status without triggering model inference.
        Returns 'online', 'standby', or 'offline'.
        """
        if hasattr(self.client, "probe_status"):
            return self.client.probe_status(timeout=timeout)
        
        # Fallback for custom or legacy clients
        online = self.is_available(timeout=timeout)
        return {
            "status": "online" if online else "offline",
            "available": online,
            "provider": getattr(self.client, "provider_name", "Custom"),
            "space_id": getattr(self.client, "space_id", None),
            "configured_model": getattr(self.client, "default_model", "dynamic_lookup"),
            "stage": "RUNNING" if online else "UNREACHABLE",
            "base_url": getattr(self.client, "base_url", "")
        }

    def chat(
        self,
        message: str,
        history: Optional[List[Union[ChatMessage, Dict[str, str]]]] = None,
        evidence: Optional[Union[Dict[str, Any], List[Any]]] = None,
        workspace_context: Optional[Union[Dict[str, Any], List[Any]]] = None,
        model: Optional[str] = None,
        conversation_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> ChatResponse:
        """
        Executes a conversational inference turn.

        Args:
            message: User query or follow-up question.
            history: Prior conversation turns.
            evidence: Optional structured SatQuery telemetry (e.g. BIT-CD, OWL-ViT, indices).
            workspace_context: Optional structured metadata summary of scenes imported into the workspace.
            model: Optional model override.
            conversation_id: Session identifier.
            temperature: Sampling temperature.
            max_tokens: Token generation limit.

        Returns:
            ChatResponse with natural-language text and execution provenance.
        """
        target_model = model or getattr(self.client, "default_model", "Qwen/Qwen2.5-VL-3B-Instruct")
        provider = getattr(self.client, "provider_name", "Hugging Face Spaces")
        space_id = getattr(self.client, "space_id", "Astro406/satquery-qwen-vl")

        # 1. Validate and normalize user input
        cleaned_msg = (message or "").strip()
        if not cleaned_msg:
            return ChatResponse(
                response="Please provide a non-empty question or satellite intelligence query.",
                model=target_model,
                provider=provider,
                status="error",
                conversation_id=conversation_id,
                error="Empty or whitespace-only message rejected."
            )

        if len(cleaned_msg) > 10000:
            return ChatResponse(
                response="Your message exceeds the maximum allowable length of 10,000 characters. Please provide a more concise question.",
                model=target_model,
                provider=provider,
                status="error",
                conversation_id=conversation_id,
                error="Oversized message rejected."
            )

        # 2. Check provider reachability via metadata probe (avoid cold inference if definitely offline)
        if hasattr(self.client, "probe_status"):
            probe = self.client.probe_status(timeout=4.0)
            if probe.get("status") == "offline":
                logger.info(f"Hugging Face Space probe reported offline/unreachable: {probe.get('stage')}")
                return ChatResponse(
                    response=(
                        f"The SatQuery AI inference service on Hugging Face Space ({space_id}) is currently "
                        "offline or unreachable. Please try again in a few moments."
                    ),
                    model=target_model,
                    provider=provider,
                    status="unavailable",
                    conversation_id=conversation_id,
                    space_id=space_id,
                    error=f"Hugging Face Space is offline ({probe.get('stage', 'UNREACHABLE')})."
                )

        # 3. Handle Hugging Face Gradio client prediction path
        if hasattr(self.client, "predict"):
            prompt = assemble_qwen_prompt(
                message=cleaned_msg,
                history=history,
                evidence=evidence,
                workspace_context=workspace_context
            )

            try:
                result = self.client.predict(question=prompt)
                return ChatResponse(
                    response=result["content"],
                    model=result.get("model", target_model),
                    provider=result.get("provider", provider),
                    status="success",
                    conversation_id=conversation_id,
                    space_id=result.get("space_id", space_id),
                    duration_sec=result.get("duration_sec")
                )
            except HFSpaceTimeoutError as te:
                logger.warning(f"Inference timeout on Hugging Face Space {space_id}: {te}")
                return ChatResponse(
                    response=(
                        "Inference request timed out waiting for Hugging Face Space. "
                        "The model may be warming up or experiencing queue delays. Please retry in a few moments."
                    ),
                    model=target_model,
                    provider=provider,
                    status="unavailable",
                    conversation_id=conversation_id,
                    space_id=space_id,
                    error=str(te)
                )
            except HFSpaceStandbyError as se:
                logger.info(f"Hugging Face Space {space_id} is in standby/waking up: {se}")
                return ChatResponse(
                    response=(
                        "The Hugging Face Space is currently waking up from standby. "
                        "Please allow approximately 30-60 seconds and submit your query again."
                    ),
                    model=target_model,
                    provider=provider,
                    status="standby",
                    conversation_id=conversation_id,
                    space_id=space_id,
                    error=str(se)
                )
            except HFSpaceRateLimitError as rle:
                logger.warning(f"ZeroGPU rate/quota limit on Space {space_id}: {rle}")
                return ChatResponse(
                    response=(
                        "The Hugging Face Space has reached its anonymous ZeroGPU compute limit. "
                        "To grant higher quota, configure an HF_TOKEN in the Render environment variables, "
                        "or retry once the ZeroGPU hourly quota refreshes."
                    ),
                    model=target_model,
                    provider=provider,
                    status="unavailable",
                    conversation_id=conversation_id,
                    space_id=space_id,
                    error=str(rle)
                )
            except HFSpaceUnavailableError as ue:
                logger.warning(f"Hugging Face Space {space_id} is unavailable: {ue}")
                return ChatResponse(
                    response=(
                        f"Connection to Hugging Face Space {space_id} could not be established. "
                        "Please verify the Space status or retry shortly."
                    ),
                    model=target_model,
                    provider=provider,
                    status="unavailable",
                    conversation_id=conversation_id,
                    space_id=space_id,
                    error=str(ue)
                )
            except HFSpaceInferenceError as ie:
                logger.error(f"Inference error on Hugging Face Space {space_id}: {ie}")
                return ChatResponse(
                    response=(
                        "An error occurred while generating a response from the satellite intelligence model. "
                        "Please check your input or retry."
                    ),
                    model=target_model,
                    provider=provider,
                    status="error",
                    conversation_id=conversation_id,
                    space_id=space_id,
                    error=str(ie)
                )
            except Exception as e:
                logger.error(f"Unexpected error in SatQueryChatbot HF dispatch: {e}")
                return ChatResponse(
                    response="An unexpected internal error occurred while communicating with the satellite model.",
                    model=target_model,
                    provider=provider,
                    status="error",
                    conversation_id=conversation_id,
                    space_id=space_id,
                    error=str(e)
                )

        # 4. Fallback for legacy OpenAI-compatible clients (e.g. LM Studio if injected)
        elif hasattr(self.client, "chat_completion"):
            messages: List[Dict[str, str]] = []
            full_system_prompt = SATQUERY_SYSTEM_PROMPT + format_evidence_context(
                evidence=evidence,
                workspace_context=workspace_context
            )
            messages.append({"role": "system", "content": full_system_prompt})

            if history:
                for item in history:
                    if isinstance(item, ChatMessage):
                        messages.append({"role": item.role, "content": item.content})
                    elif isinstance(item, dict) and "role" in item and "content" in item:
                        messages.append({"role": str(item["role"]), "content": str(item["content"])})

            messages.append({"role": "user", "content": cleaned_msg})

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
                    provider=getattr(self.client, "provider_name", "LM Studio"),
                    status="success",
                    conversation_id=conversation_id
                )
            except Exception as e:
                logger.error(f"Legacy client error in SatQueryChatbot: {e}")
                return ChatResponse(
                    response="An error occurred communicating with the local inference client.",
                    model=target_model,
                    provider=getattr(self.client, "provider_name", "LM Studio"),
                    status="error",
                    conversation_id=conversation_id,
                    error=str(e)
                )

        else:
            return ChatResponse(
                response="No supported inference client configured.",
                model=target_model,
                provider=provider,
                status="error",
                conversation_id=conversation_id,
                error="Configured client has neither predict() nor chat_completion()."
            )

# Global default chatbot instance
satquery_chatbot = SatQueryChatbot()
