from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...

    def encode(self, texts: list[str]) -> np.ndarray | sparse.csr_matrix: ...


class TfidfEmbedder:
    """Offline sparse embeddings. No model download required."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        corpus = texts or [""]
        self._vectorizer.fit(corpus)
        self._fitted = True

    def encode(self, texts: list[str]) -> sparse.csr_matrix:
        if not self._fitted:
            self.fit(texts)
        return self._vectorizer.transform(texts).astype(np.float32)


class DenseEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def create_embedder(model_name: str = "") -> Embedder:
    if model_name.strip():
        return DenseEmbedder(model_name.strip())
    return TfidfEmbedder()


def cosine_scores(
    query: np.ndarray | sparse.csr_matrix, matrix: np.ndarray | sparse.csr_matrix
) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.array([], dtype=np.float32)
    q = query.reshape(1, -1)
    return cosine_similarity(q, matrix)[0].astype(np.float32)
