from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class RAGDocument(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

class RAGQuery(BaseModel):
    query_text: str
    top_k: int = 2
    threshold: float = 0.20

class RAGRetrievalResult(BaseModel):
    query: str
    retrieval_used: bool
    retrieved_documents: List[RAGDocument] = Field(default_factory=list)
    context_text: str = ""
    reason: str = ""
