from __future__ import annotations

from graphrag.config import Settings, get_settings
from graphrag.graph.store import GraphStore
from graphrag.models import GraphPath, RetrievedEvidence
from graphrag.reasoning import is_relational_query, rank_paths, seed_entities
from graphrag.vector_store import VectorStore


def reciprocal_rank_fusion(
    ranked_ids: list[list[str]], k: int = 60
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in ranked_ids:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """Fuse vector similarity with graph neighborhood / multi-hop expansion."""

    def __init__(
        self,
        graph: GraphStore,
        vectors: VectorStore,
        settings: Settings | None = None,
    ) -> None:
        self.graph = graph
        self.vectors = vectors
        self.settings = settings or get_settings()

    def retrieve(self, query: str) -> tuple[list[RetrievedEvidence], list[GraphPath]]:
        top_k = self.settings.vector_top_k
        hops = self.settings.graph_hops
        vector_hits = self.vectors.search(query, top_k=top_k)
        seeds = seed_entities(self.graph, query)
        seed_ids = [e.id for e in seeds]
        graph_entities, relations, paths = self.graph.expand(seed_ids, hops=hops)

        if len(seeds) >= 2:
            extra = self.graph.shortest_paths(seeds[0].id, seeds[1].id, max_hops=hops)
            paths = extra + paths
        paths = rank_paths(query, paths, limit=12)

        graph_entity_ids = {e.id for e in graph_entities} | set(seed_ids)
        graph_chunks = self.vectors.by_entity_ids(graph_entity_ids)

        vector_rank = [chunk.id for chunk, _ in vector_hits]
        graph_rank = [chunk.id for chunk in graph_chunks]
        if is_relational_query(query):
            # Graph evidence leads for multi-hop / relational questions.
            fused = reciprocal_rank_fusion([graph_rank, vector_rank])
        else:
            fused = reciprocal_rank_fusion([vector_rank, graph_rank])

        vector_scores = {chunk.id: score for chunk, score in vector_hits}
        graph_boost = {chunk.id: 1.0 / (1.0 + i) for i, chunk in enumerate(graph_chunks)}

        evidence_map: dict[str, RetrievedEvidence] = {}
        for chunk, score in vector_hits:
            evidence_map[chunk.id] = RetrievedEvidence(
                chunk=chunk,
                vector_score=score,
                graph_score=graph_boost.get(chunk.id, 0.0),
                source="vector",
            )
        for chunk in graph_chunks:
            current = evidence_map.get(chunk.id)
            gscore = graph_boost.get(chunk.id, 0.0)
            if current:
                current.graph_score = gscore
                current.source = "hybrid"
            else:
                evidence_map[chunk.id] = RetrievedEvidence(
                    chunk=chunk,
                    vector_score=0.0,
                    graph_score=gscore,
                    source="graph",
                )

        vw = self.settings.fusion_vector_weight
        gw = self.settings.fusion_graph_weight
        results: list[RetrievedEvidence] = []
        for item in evidence_map.values():
            rrf = fused.get(item.chunk.id, 0.0)
            item.fused_score = vw * item.vector_score + gw * item.graph_score + rrf
            results.append(item)
        results.sort(key=lambda e: e.fused_score, reverse=True)
        return results[:top_k], paths[:12]
