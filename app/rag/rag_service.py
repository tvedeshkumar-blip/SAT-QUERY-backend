import logging
from typing import Dict, Any, Optional
from app.rag.schemas import RAGQuery, RAGRetrievalResult
from app.rag.vector_store import InMemoryVectorStore
from app.rag.retriever import GeospatialRetriever

logger = logging.getLogger("satquery.rag")

class RAGService:
    """
    Geospatial Earth Context RAG Service.
    Answers: 'Does this image analysis require contextual geospatial information?'
    If yes: query -> embedding -> retrieval -> context.
    If not: cleanly skips RAG without fake retrieval.
    """
    def __init__(self):
        self.vector_store = InMemoryVectorStore()
        self.retriever = GeospatialRetriever(self.vector_store)

    def should_retrieve(self, query: str, metadata: Dict[str, Any]) -> bool:
        """
        Determines whether contextual Earth/sensor knowledge is relevant for the current request.
        """
        q_lower = (query or "").lower()
        context_triggers = [
            "sensor", "cartosat", "risat", "c-band", "polarization", "specular",
            "bengaluru", "mumbai", "bellandur", "wetland", "mangrove", "mudflat",
            "double-bounce", "resolution", "wavelength", "band"
        ]
        
        # Check query keywords
        if any(trigger in q_lower for trigger in context_triggers):
            return True

        # Check metadata attributes
        meta_sensor = (metadata.get("sensor") or "").lower()
        meta_filename = (metadata.get("filename") or "").lower()
        if any(s in meta_sensor or s in meta_filename for s in ["cartosat", "risat", "bengaluru", "mumbai"]):
            return True

        return False

    def retrieve_context(self, query: str, metadata: Dict[str, Any]) -> RAGRetrievalResult:
        if not self.should_retrieve(query, metadata):
            return RAGRetrievalResult(
                query=query,
                retrieval_used=False,
                reason="Query and imagery did not trigger geospatial/sensor context retrieval criteria."
            )

        # Build retrieval query combining natural language query and sensor metadata
        search_terms = [query]
        if metadata.get("sensor"):
            search_terms.append(metadata["sensor"])
        if metadata.get("modality"):
            search_terms.append(metadata["modality"])
            
        full_query = " ".join(search_terms)
        rag_query = RAGQuery(query_text=full_query, top_k=2, threshold=0.15)
        matches = self.retriever.retrieve(rag_query)

        if not matches:
            return RAGRetrievalResult(
                query=query,
                retrieval_used=False,
                reason="Context search triggered but similarity threshold (< 0.15) was not satisfied."
            )

        docs = [doc for doc, _ in matches]
        context_paragraphs = [f"[{doc.title}]: {doc.content}" for doc in docs]
        context_text = "\n\n".join(context_paragraphs)

        logger.info(f"Earth Context RAG retrieved {len(docs)} document(s) for query: '{query}'")

        return RAGRetrievalResult(
            query=query,
            retrieval_used=True,
            retrieved_documents=docs,
            context_text=context_text,
            reason=f"Retrieved {len(docs)} relevant domain knowledge article(s)."
        )

# Global RAG instance
rag_service = RAGService()
