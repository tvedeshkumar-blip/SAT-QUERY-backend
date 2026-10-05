"""
LM Studio OpenAI-Compatible Local API Client for SatQuery AI.
Communicates strictly with the local LM Studio instance over HTTP (e.g. http://127.0.0.1:1234/v1).
Does NOT invoke Gemini, Google GenAI, or any cloud API.
"""

import os
import logging
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger("satquery.chatbot.lmstudio")

class LMStudioClientError(Exception):
    """Base exception for LM Studio client failures."""
    pass

class LMStudioConnectionError(LMStudioClientError):
    """Raised when LM Studio local endpoint cannot be reached."""
    pass

class LMStudioAPIError(LMStudioClientError):
    """Raised when LM Studio returns an HTTP error code."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"LM Studio API returned HTTP {status_code}: {message}")

class LMStudioClient:
    """
    Lightweight, robust client for LM Studio's local OpenAI-compatible API.
    Configured via environment variables:
    - LMSTUDIO_BASE_URL (default: 'http://127.0.0.1:1234/v1')
    - LMSTUDIO_MODEL (default: '' - dynamic lookup or model fallback)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: float = 120.0,
        connect_timeout: float = 2.0
    ):
        raw_url = base_url or os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        self.base_url = raw_url.rstrip("/")
        self.default_model = default_model if default_model is not None else os.getenv("LMSTUDIO_MODEL", "")
        self.timeout = float(os.getenv("LMSTUDIO_TIMEOUT", str(timeout)))
        self.connect_timeout = float(os.getenv("LMSTUDIO_CONNECT_TIMEOUT", str(connect_timeout)))

    def is_reachable(self, timeout: Optional[float] = None) -> bool:
        """
        Non-blocking health probe to check if LM Studio local server is active.
        """
        chk_timeout = timeout if timeout is not None else self.connect_timeout
        url = f"{self.base_url}/models"
        try:
            with httpx.Client(timeout=httpx.Timeout(chk_timeout, connect=chk_timeout)) as client:
                resp = client.get(url)
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError):
            return False
        except Exception as e:
            logger.debug(f"LM Studio reachability check failed: {e}")
            return False

    def get_available_models(self, timeout: Optional[float] = None) -> List[str]:
        """
        Queries /v1/models from LM Studio to discover loaded/available local models.
        """
        chk_timeout = timeout if timeout is not None else self.connect_timeout
        url = f"{self.base_url}/models"
        try:
            with httpx.Client(timeout=httpx.Timeout(chk_timeout, connect=chk_timeout)) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    models_data = data.get("data", [])
                    return [m.get("id") for m in models_data if isinstance(m, dict) and m.get("id")]
                return []
        except Exception as e:
            logger.debug(f"Failed to query LM Studio models: {e}")
            return []

    def resolve_model(self, requested_model: Optional[str] = None) -> str:
        """
        Resolves which model identifier to submit to LM Studio.
        1. Explicit requested_model takes top priority
        2. Configured LMSTUDIO_MODEL environment variable
        3. First model discovered via /v1/models
        4. Fallback placeholder string 'local-model'
        """
        if requested_model and requested_model.strip():
            return requested_model.strip()

        if self.default_model and self.default_model.strip():
            return self.default_model.strip()

        available = self.get_available_models()
        if available:
            return available[0]

        return "local-model"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Sends a standard OpenAI-compatible chat completion payload to LM Studio.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            model: Optional model name override.
            temperature: Generation temperature.
            max_tokens: Maximum token length.
            timeout: Optional call-specific timeout.
            
        Returns:
            Dict containing 'content', 'model', and raw response telemetry.
        """
        target_model = self.resolve_model(model)
        url = f"{self.base_url}/chat/completions"
        call_timeout = timeout if timeout is not None else self.timeout

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            with httpx.Client(timeout=httpx.Timeout(call_timeout, connect=self.connect_timeout)) as client:
                resp = client.post(url, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as ce:
            logger.warning(f"Connection refused to LM Studio at {self.base_url}: {ce}")
            raise LMStudioConnectionError(
                f"Could not connect to LM Studio at {self.base_url}. Ensure LM Studio local server is running."
            ) from ce
        except httpx.TimeoutException as te:
            logger.warning(f"Request to LM Studio timed out after {call_timeout}s: {te}")
            raise LMStudioConnectionError(
                f"LM Studio request timed out after {call_timeout}s."
            ) from te
        except Exception as e:
            logger.error(f"Unexpected network error communicating with LM Studio: {e}")
            raise LMStudioConnectionError(f"Network error communicating with LM Studio: {e}") from e

        if resp.status_code != 200:
            error_body = resp.text
            logger.error(f"LM Studio error HTTP {resp.status_code}: {error_body}")
            raise LMStudioAPIError(resp.status_code, error_body)

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise LMStudioAPIError(resp.status_code, "LM Studio response contained no choices.")

        msg = choices[0].get("message", {})
        content = msg.get("content", "").strip()
        actual_model = data.get("model", target_model)

        return {
            "content": content,
            "model": actual_model,
            "usage": data.get("usage", {}),
            "raw": data
        }
