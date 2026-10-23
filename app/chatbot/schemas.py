from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator

class ChatMessage(BaseModel):
    """
    Standard chat message schema for conversational multi-turn dialogue.
    """
    role: Literal["system", "user", "assistant"] = Field(
        ..., 
        description="Role of the message sender: system, user, or assistant"
    )
    content: str = Field(
        ..., 
        description="Text content of the message"
    )

class ChatRequest(BaseModel):
    """
    Incoming request payload for the SatQuery Qwen-VL chatbot.
    """
    message: str = Field(
        ..., 
        description="The user's natural-language query or follow-up question"
    )
    conversation_id: Optional[str] = Field(
        default=None, 
        description="Optional session / conversation tracking identifier"
    )
    history: Optional[List[ChatMessage]] = Field(
        default_factory=list, 
        description="Chronological conversation history for multi-turn context"
    )
    evidence: Optional[Union[Dict[str, Any], List[Any]]] = Field(
        default=None, 
        description="Optional structured SatQuery specialist evidence (e.g. BIT-CD, OWL-ViT, spectral indices, metadata)"
    )
    workspace_context: Optional[Union[Dict[str, Any], List[Any]]] = Field(
        default=None,
        description="Optional structured summary of scenes and metadata loaded in the active workspace"
    )
    model: Optional[str] = Field(
        default=None, 
        description="Optional model identifier override"
    )
    temperature: Optional[float] = Field(
        default=0.2, 
        ge=0.0, 
        le=2.0, 
        description="Sampling temperature for analytical precision (default 0.2)"
    )
    max_tokens: Optional[int] = Field(
        default=1024, 
        ge=1, 
        le=8192, 
        description="Maximum token budget for assistant generation"
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if not cleaned:
            raise ValueError("Message cannot be empty or whitespace-only.")
        if len(cleaned) > 10000:
            raise ValueError("Message exceeds maximum allowable length of 10,000 characters.")
        return cleaned

class ChatResponse(BaseModel):
    """
    Structured response payload returned by the SatQuery chatbot.
    """
    response: str = Field(
        ..., 
        description="Assistant natural-language response"
    )
    model: str = Field(
        default="Qwen/Qwen2.5-VL-3B-Instruct", 
        description="The model name or identifier used for generation"
    )
    provider: str = Field(
        default="Hugging Face Spaces", 
        description="Designation of the inference engine"
    )
    status: str = Field(
        ..., 
        description="Execution status: 'success', 'standby', 'unavailable', or 'error'"
    )
    conversation_id: Optional[str] = Field(
        default=None, 
        description="Conversation session identifier"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error detail message if status is not 'success'"
    )
    space_id: Optional[str] = Field(
        default=None,
        description="Hugging Face Space identifier providing the inference service"
    )
    duration_sec: Optional[float] = Field(
        default=None,
        description="Elapsed inference duration in seconds"
    )
