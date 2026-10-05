import pytest
from app.rag.rag_service import rag_service
from app.rag.vector_store import InMemoryVectorStore
from app.rag.schemas import RAGQuery

def test_rag_retrieval_for_cartosat():
    res = rag_service.retrieve_context(
        query="What are the optical sensor bands for Cartosat-2S?",
        metadata={"sensor": "Cartosat-2S", "modality": "OPTICAL"}
    )
    assert res.retrieval_used is True
    assert len(res.retrieved_documents) >= 1
    assert "Cartosat-2S" in res.context_text

def test_rag_skipped_for_generic_query():
    res = rag_service.retrieve_context(
        query="What is the pixel count?",
        metadata={"sensor": None, "modality": "OPTICAL"}
    )
    assert res.retrieval_used is False
    assert res.context_text == ""
