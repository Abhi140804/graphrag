from __future__ import annotations

import numpy as np

from graphrag.embeddings import Embedder, cosine_scores, create_embedder
from graphrag.models import Chunk


class VectorStore:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or create_embedder()
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        texts = [c.text for c in self._chunks]
        self.embedder.fit(texts)
        self._matrix = self.embedder.encode(texts)

    def search(self, query: str, top_k: int = 8) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []
        q = self.embedder.encode([query])
        scores = cosine_scores(q[0], self._matrix)
        order = np.argsort(-scores)[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in order]

    def by_entity_ids(self, entity_ids: set[str]) -> list[Chunk]:
        return [c for c in self._chunks if entity_ids.intersection(c.entity_ids)]

    def get(self, chunk_id: str) -> Chunk | None:
        for chunk in self._chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    @property
    def size(self) -> int:
        return len(self._chunks)
