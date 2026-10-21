from typing import List, Tuple
from app.rag.schemas import RAGDocument, RAGQuery
from app.rag.vector_store import InMemoryVectorStore

class GeospatialRetriever:
    """
    Retriever that queries the geospatial knowledge index using semantic query matching.
    """
    def __init__(self, vector_store: InMemoryVectorStore):
        self.store = vector_store

    def retrieve(self, query: RAGQuery) -> List[Tuple[RAGDocument, float]]:
        return self.store.search(
            query_text=query.query_text,
            top_k=query.top_k,
            threshold=query.threshold
        )
