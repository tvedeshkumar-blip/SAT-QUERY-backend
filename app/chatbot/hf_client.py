"""
Hugging Face Gradio Client for SatQuery AI.
Communicates with the public Hugging Face Space 'Astro406/satquery-qwen-vl'
hosting 'Qwen/Qwen2.5-VL-3B-Instruct' via the Gradio API contract (/predict).

Provides:
- Lightweight metadata probing (no GPU allocation, non-blocking)
- Distinct status determination: online, standby, offline
- Safe client reuse and thread-safe singleton initialization
- Dynamic version inspection for gradio_client.Client authentication arguments
- Strict execution timeouts and queue delay handling
- Clean exception hierarchy without credential leakage
"""

import os
import re
import time
import inspect
import logging
import threading
import concurrent.futures
from typing import Any, Dict, Optional, Union
import httpx
from gradio_client import Client

logger = logging.getLogger("satquery.chatbot.hf")


def redact_credentials(text: Any, token: Optional[str] = None) -> str:
    """
    Sanitizes string and exception diagnostics to prevent credential leakage in logs or API responses.
    Redacts specific instance token as well as general Hugging Face token patterns and Bearer tokens.
    """
    if text is None:
        return ""
    s = str(text)
    if token and str(token).strip():
        s = s.replace(str(token).strip(), "[REDACTED_HF_TOKEN]")
    # Hugging Face personal access tokens typically start with 'hf_' followed by alphanumeric characters
    s = re.sub(r"hf_[a-zA-Z0-9]{16,}", "[REDACTED_HF_TOKEN]", s)
    # Bearer tokens in headers or exception dumps
    s = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]{8,}", r"\1[REDACTED_TOKEN]", s, flags=re.IGNORECASE)
    # URL query parameter tokens
    s = re.sub(r"([?&]token=)[^&\s'\"]+", r"\1[REDACTED_TOKEN]", s, flags=re.IGNORECASE)
    return s


def get_supported_auth_arg(client_cls: Any = None) -> Optional[str]:
    """
    Determines whether the installed gradio_client.Client expects 'token' or 'hf_token'.
    Inspects gradio_client.Client constructor signature dynamically rather than guessing.

    Returns:
        'token' if supported (gradio_client >= 1.0)
        'hf_token' if supported (gradio_client < 1.0)
        None if neither is supported as a named parameter.
    """
    cls_to_check = client_cls if client_cls is not None else Client
    try:
        sig = inspect.signature(cls_to_check)
        params = sig.parameters
        if "token" in params:
            return "token"
        if "hf_token" in params:
            return "hf_token"
        # If client_cls is a generic Mock or wrapper with **kwargs, check gradio_client.Client directly
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            import gradio_client
            real_sig = inspect.signature(gradio_client.Client)
            if "token" in real_sig.parameters:
                return "token"
            if "hf_token" in real_sig.parameters:
                return "hf_token"
    except Exception as exc:
        logger.debug(f"Signature inspection error for gradio_client.Client: {exc}")
        return "token"
    return None


class HFClientError(Exception):
    """Base exception for Hugging Face client failures."""
    pass

class HFSpaceUnavailableError(HFClientError):
    """Raised when the Hugging Face Space is unreachable, stopped, or offline."""
    pass

class HFSpaceStandbyError(HFClientError):
    """Raised when the Hugging Face Space is sleeping, paused, or warming up."""
    pass

class HFSpaceTimeoutError(HFClientError):
    """Raised when inference or Gradio queue waiting exceeds the timeout limit."""
    pass

class HFSpaceRateLimitError(HFClientError):
    """Raised when ZeroGPU runs quota or Hugging Face rate limit is reached."""
    pass

class HFSpaceInferenceError(HFClientError):
    """Raised when the Space returns an unexpected error during generation."""
    pass


class HuggingFaceClient:
    """
    Production-ready client wrapper for Hugging Face Spaces Gradio API.
    Reuses gradio_client.Client safely across concurrent requests and enforces
    bounded execution timeouts.
    """

    def __init__(
        self,
        space_id: Optional[str] = None,
        api_name: Optional[str] = None,
        request_timeout: Optional[float] = None,
        token: Optional[str] = None,
        default_model: Optional[str] = None
    ):
        self.space_id = space_id or os.getenv("HF_SPACE_ID", "Astro406/satquery-qwen-vl")
        self.api_name = api_name or os.getenv("HF_API_NAME", "/predict")
        self.request_timeout = float(request_timeout or os.getenv("HF_REQUEST_TIMEOUT", "65.0"))
        self.token = token or os.getenv("HF_TOKEN") or None
        self.default_model = default_model or os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.provider_name = "Hugging Face Spaces"

        # Derived subdomain base URL
        subdomain = self.space_id.replace("/", "-").lower()
        self.base_url = f"https://{subdomain}.hf.space"

        self._client: Optional[Client] = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        token_configured = bool(self.token)
        return (
            f"<HuggingFaceClient space_id='{self.space_id}' "
            f"api_name='{self.api_name}' "
            f"auth_token_set={token_configured}>"
        )

    def get_space_metadata(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Queries the Hugging Face public Spaces REST API for lightweight metadata.
        Does NOT trigger GPU allocation or model warm-up.
        """
        meta_url = f"https://huggingface.co/api/spaces/{self.space_id}"
        try:
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"

            with httpx.Client(timeout=timeout) as http_client:
                resp = http_client.get(meta_url, headers=headers)

            if resp.status_code == 200:
                data = resp.json()
                runtime = data.get("runtime", {})
                stage = runtime.get("stage", "UNKNOWN").upper()
                host = data.get("host") or self.base_url
                hardware = runtime.get("hardware", {}).get("current", "unknown")
                sdk = data.get("sdk", "gradio")
                models = data.get("models", [self.default_model])

                return {
                    "stage": stage,
                    "host": host,
                    "hardware": hardware,
                    "sdk": sdk,
                    "models": models,
                    "raw": data
                }
            else:
                logger.warning(
                    f"Hugging Face metadata probe returned HTTP {resp.status_code} for {self.space_id}"
                )
                return {
                    "stage": "UNREACHABLE",
                    "error": f"HTTP {resp.status_code}",
                    "host": self.base_url,
                    "hardware": "unknown"
                }
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError) as exc:
            clean_diag = redact_credentials(exc, self.token)
            logger.debug(f"Hugging Face metadata probe network failure for {self.space_id}: {clean_diag}")
            return {
                "stage": "UNREACHABLE",
                "error": clean_diag,
                "host": self.base_url,
                "hardware": "unknown"
            }
        except Exception as exc:
            clean_diag = redact_credentials(exc, self.token)
            logger.warning(f"Unexpected error probing Hugging Face Space {self.space_id}: {clean_diag}")
            return {
                "stage": "UNREACHABLE",
                "error": clean_diag,
                "host": self.base_url,
                "hardware": "unknown"
            }

    def probe_status(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Determines the operational status of the Space:
        - 'online': Space is running and ready to accept inference requests.
        - 'standby': Space is sleeping, paused, building, or warming up.
        - 'offline': Space is stopped, in error, or unreachable.
        """
        meta = self.get_space_metadata(timeout=timeout)
        stage = meta.get("stage", "UNREACHABLE")
        host = meta.get("host", self.base_url)
        hardware = meta.get("hardware", "unknown")

        if stage == "RUNNING":
            status = "online"
            available = True
        elif stage in (
            "SLEEPING", "PAUSED", "STARTING", "BUILDING",
            "APP_STARTING", "RESTARTING", "INITIALIZING"
        ):
            status = "standby"
            available = False
        else:
            status = "offline"
            available = False

        return {
            "status": status,
            "available": available,
            "provider": self.provider_name,
            "space_id": self.space_id,
            "configured_model": self.default_model,
            "stage": stage,
            "base_url": host,
            "hardware": hardware
        }

    def is_reachable(self, timeout: float = 5.0) -> bool:
        """
        Quick boolean check: returns True strictly when the Space is online.
        """
        probe = self.probe_status(timeout=timeout)
        return probe.get("status") == "online"

    def _resolve_client_kwargs(self, client_cls: Any = None) -> Dict[str, Any]:
        """
        Determines the supported authentication kwargs for gradio_client.Client
        without guessing or passing unsupported keyword arguments.
        """
        if not self.token or not str(self.token).strip():
            return {}

        clean_token = str(self.token).strip()
        target_cls = client_cls if client_cls is not None else Client
        auth_arg = get_supported_auth_arg(target_cls)
        if auth_arg == "token":
            return {"token": clean_token}
        elif auth_arg == "hf_token":
            return {"hf_token": clean_token}
        else:
            logger.warning(
                "Installed gradio_client.Client does not declare 'token' or 'hf_token' parameters. "
                "Omitting authentication parameter to avoid unexpected keyword argument errors."
            )
            return {}

    def _get_or_create_client(self) -> Client:
        """
        Thread-safe singleton retrieval for gradio_client.Client.
        Reuses existing connection to avoid per-request handshake overhead.
        """
        if self._client is not None:
            return self._client

        with self._lock:
            if self._client is not None:
                return self._client

            logger.info(f"Initializing Gradio Client connection to Space: {self.space_id}")
            try:
                client_kwargs = self._resolve_client_kwargs()
                client = Client(self.space_id, **client_kwargs)
                self._client = client
                return self._client
            except Exception as exc:
                clean_diag = redact_credentials(exc, self.token)
                logger.error(f"Failed to connect Gradio Client to {self.space_id}: {clean_diag}")
                raise HFSpaceUnavailableError(
                    f"Unable to establish connection to Hugging Face Space '{self.space_id}': {clean_diag}"
                ) from exc

    def reset_client(self) -> None:
        """Resets the cached Gradio client instance, forcing re-initialization on next call."""
        with self._lock:
            self._client = None
            logger.info(f"Reset cached Gradio client for Space {self.space_id}")

    def predict(
        self,
        question: str,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Dispatches inference query to the Hugging Face Space /predict endpoint.
        Uses a bounded thread pool executor to strictly enforce the configured request timeout.

        Args:
            question: Formatted natural-language text prompt.
            timeout: Optional call-specific timeout override in seconds.

        Returns:
            Dict containing 'content', 'model', 'provider', and execution provenance.
        """
        cleaned_question = (question or "").strip()
        if not cleaned_question:
            raise ValueError("Prediction question cannot be empty or whitespace-only.")

        call_timeout = timeout if timeout is not None else self.request_timeout
        client = self._get_or_create_client()

        start_time = time.time()
        logger.info(
            f"Dispatching query ({len(cleaned_question)} chars) to {self.space_id}{self.api_name} [timeout={call_timeout}s]"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                client.predict,
                question=cleaned_question,
                api_name=self.api_name
            )
            try:
                raw_response = future.result(timeout=call_timeout)
            except concurrent.futures.TimeoutError as te:
                duration = time.time() - start_time
                logger.warning(
                    f"Hugging Face Space '{self.space_id}' request timed out after {duration:.2f}s (budget: {call_timeout}s)"
                )
                raise HFSpaceTimeoutError(
                    f"Hugging Face Space request timed out after {call_timeout:.0f} seconds."
                ) from te
            except Exception as exc:
                duration = time.time() - start_time
                clean_diag = redact_credentials(exc, self.token)
                err_str = clean_diag.lower()
                logger.error(
                    f"Inference error on Space {self.space_id} after {duration:.2f}s: {clean_diag}"
                )

                if "sleeping" in err_str or "paused" in err_str:
                    raise HFSpaceStandbyError(
                        f"Hugging Face Space '{self.space_id}' is currently sleeping or warming up."
                    ) from exc
                elif "zerogpu" in err_str or "limit" in err_str or "quota" in err_str or "rate limit" in err_str:
                    raise HFSpaceRateLimitError(
                        f"ZeroGPU quota reached on Space '{self.space_id}': {clean_diag}"
                    ) from exc
                elif "unreachable" in err_str or "connection" in err_str or "could not connect" in err_str:
                    self.reset_client()
                    raise HFSpaceUnavailableError(
                        f"Hugging Face Space '{self.space_id}' is currently offline or unreachable: {clean_diag}"
                    ) from exc
                else:
                    raise HFSpaceInferenceError(
                        f"Hugging Face Space inference failed: {clean_diag}"
                    ) from exc

        duration = time.time() - start_time
        logger.info(f"Inference completed in {duration:.2f}s from {self.space_id}")

        # Parse output: /predict returns a single string response
        if isinstance(raw_response, (list, tuple)):
            content = str(raw_response[0]).strip() if raw_response else ""
        elif isinstance(raw_response, str):
            content = raw_response.strip()
        else:
            content = str(raw_response).strip()

        return {
            "content": content,
            "model": self.default_model,
            "provider": self.provider_name,
            "duration_sec": round(duration, 3),
            "space_id": self.space_id
        }
