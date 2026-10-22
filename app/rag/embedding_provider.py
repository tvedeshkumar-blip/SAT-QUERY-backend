import re
import math
from typing import List, Dict
import numpy as np

class TFIDFEmbeddingProvider:
    """
    Lightweight deterministic TF-IDF embedding provider for geospatial RAG.
    Zero external dependencies, fast CPU execution, fully reproducible.
    """
    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text)]

    def fit(self, corpus: List[str]):
        doc_count = len(corpus)
        df: Dict[str, int] = {}

        for doc in corpus:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        self.vocabulary = {term: idx for idx, term in enumerate(sorted(df.keys()))}
        self.idf = {
            term: math.log((doc_count + 1.0) / (freq + 1.0)) + 1.0 
            for term, freq in df.items()
        }

    def transform(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vec = np.zeros(len(self.vocabulary), dtype=np.float32)
        if not tokens or not self.vocabulary:
            return vec

        tf: Dict[str, int] = {}
        for t in tokens:
            if t in self.vocabulary:
                tf[t] = tf.get(t, 0) + 1

        for term, count in tf.items():
            idx = self.vocabulary[term]
            vec[idx] = (count / len(tokens)) * self.idf.get(term, 1.0)

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
