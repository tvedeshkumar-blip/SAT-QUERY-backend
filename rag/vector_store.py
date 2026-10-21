from typing import List, Tuple
import numpy as np
from app.rag.schemas import RAGDocument
from app.rag.embedding_provider import TFIDFEmbeddingProvider

# Curated documented geospatial knowledge base
CURATED_GEOSPATIAL_DOCUMENTS = [
    RAGDocument(
        id="geo_kb_001",
        title="ISRO Cartosat-2S Optical Sensor Specifications",
        content=(
            "ISRO Cartosat-2S provides high-resolution optical imagery with a 0.65-meter panchromatic "
            "band and a 1.6-meter 4-band multispectral instrument (B1: Blue 0.45-0.52 µm, B2: Green 0.52-0.59 µm, "
            "B3: Red 0.62-0.68 µm, B4: NIR 0.77-0.86 µm). In urban environments, high contrast in the red and NIR "
            "bands facilitates delineation of building rooftops and paved road corridors."
        ),
        metadata={"sensor": "Cartosat-2S", "modality": "OPTICAL", "agency": "ISRO"}
    ),
    RAGDocument(
        id="geo_kb_002",
        title="ISRO RISAT-1A / EOS-04 C-Band SAR Properties",
        content=(
            "ISRO RISAT-1A (EOS-04) carries an active C-band Synthetic Aperture Radar operating at 5.35 GHz. "
            "It supports circular and linear polarizations (HH, HV, VV, VH). Calm open water exhibits specular reflection, "
            "appearing very dark (typically < -18 dB backscatter). Built structures with orthogonal ground-wall junctions "
            "produce dihedral double-bounce reflections, appearing extremely bright in both co-pol and cross-pol channels."
        ),
        metadata={"sensor": "RISAT-1A", "modality": "SAR", "agency": "ISRO"}
    ),
    RAGDocument(
        id="geo_kb_003",
        title="Bengaluru Urban Wetland & Lake Hydrology",
        content=(
            "The Bengaluru metropolitan region features interconnected lake cascades (e.g., Bellandur, Varthur, Hebbal). "
            "These shallow urban water bodies frequently experience severe eutrophication, leading to dense water hyacinth mats "
            "and seasonal algal blooms. On optical imagery, weed-covered lake surfaces display high NIR reflectance resembling "
            "vegetation rather than clear water, requiring cross-modal SAR or NDWI temporal validation."
        ),
        metadata={"region": "Bengaluru", "feature": "Wetland", "application": "Water body monitoring"}
    ),
    RAGDocument(
        id="geo_kb_004",
        title="Mumbai Coastal Plain & Tidal Mudflat Signatures",
        content=(
            "The Mumbai coastal corridor exhibits high semi-diurnal tidal fluctuations impacting mudflats, creeks (Thane Creek), "
            "and mangrove forests. Mangrove stands produce complex volumetric radar scattering in C-band SAR. At low tide, "
            "exposed intertidal mudflats exhibit smooth, moist surfaces that mimic water bodies on radar but display soil reflectance on optical rasters."
        ),
        metadata={"region": "Mumbai", "feature": "Coastline/Mudflats", "application": "Intertidal classification"}
    )
]

class InMemoryVectorStore:
    """
    In-memory vector store for authentic geospatial and remote sensing knowledge retrieval.
    """
    def __init__(self):
        self.documents: List[RAGDocument] = CURATED_GEOSPATIAL_DOCUMENTS
        self.embedder = TFIDFEmbeddingProvider()
        self.embeddings: List[np.ndarray] = []
        self._initialize_index()

    def _initialize_index(self):
        corpus = [f"{doc.title} {doc.content}" for doc in self.documents]
        self.embedder.fit(corpus)
        self.embeddings = [self.embedder.transform(text) for text in corpus]

    def search(self, query_text: str, top_k: int = 2, threshold: float = 0.15) -> List[Tuple[RAGDocument, float]]:
        q_vec = self.embedder.transform(query_text)
        if np.linalg.norm(q_vec) == 0:
            return []

        results = []
        for doc, doc_vec in zip(self.documents, self.embeddings):
            sim = float(np.dot(q_vec, doc_vec))
            if sim >= threshold:
                doc_copy = doc.model_copy()
                doc_copy.score = round(sim, 3)
                results.append((doc_copy, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
