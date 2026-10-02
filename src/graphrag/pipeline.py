from __future__ import annotations

from pathlib import Path

from graphrag.config import Settings, get_settings
from graphrag.embeddings import create_embedder
from graphrag.extract import chunk_text, extract_from_text
from graphrag.graph.factory import create_graph_store
from graphrag.graph.store import GraphStore
from graphrag.llm import maybe_generate
from graphrag.models import Document, QueryResult
from graphrag.reasoning import reason_over_paths, seed_entities
from graphrag.retrieval import HybridRetriever
from graphrag.vector_store import VectorStore


class GraphRAG:
    def __init__(
        self,
        settings: Settings | None = None,
        graph: GraphStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.graph = graph or create_graph_store(self.settings)
        self.vectors = VectorStore(create_embedder(self.settings.embedding_model))
        self.retriever = HybridRetriever(self.graph, self.vectors, self.settings)

    def ingest_documents(self, documents: list[Document]) -> dict[str, int]:
        all_chunks = []
        rel_count = 0
        for doc in documents:
            for chunk in chunk_text(doc.text, doc.id):
                extracted = extract_from_text(chunk.text, doc.id, chunk.id)
                chunk.entity_ids = [e.id for e in extracted.entities]
                for entity in extracted.entities:
                    self.graph.upsert_entity(entity)
                for relation in extracted.relations:
                    self.graph.upsert_relation(relation)
                    rel_count += 1
                all_chunks.append(chunk)
        if all_chunks:
            self.vectors.add(all_chunks)
        stats = self.graph.stats()
        return {
            "documents": len(documents),
            "chunks": len(all_chunks),
            "relations": rel_count,
            "nodes": int(stats.get("nodes") or 0),
        }

    def ingest_file(self, path: str | Path) -> dict[str, int]:
        import json

        raw = json.loads(Path(path).read_text())
        docs = [Document.model_validate(item) for item in raw]
        return self.ingest_documents(docs)

    def query(self, question: str) -> QueryResult:
        evidence, paths = self.retriever.retrieve(question)
        seeds = seed_entities(self.graph, question)
        graph_answer = reason_over_paths(question, paths, seeds)
        passages = "\n\n".join(
            f"[{item.chunk.id} | {item.source} | {item.fused_score:.3f}]\n{item.chunk.text}"
            for item in evidence[:5]
        )
        prompt = (
            f"Question: {question}\n\n{graph_answer}\n\nPassages:\n{passages}\n\n"
            "Write a concise answer and cite connecting entities."
        )
        llm_answer = maybe_generate(prompt, self.settings)
        answer = llm_answer or self._compose_answer(question, graph_answer, evidence)
        return QueryResult(
            query=question,
            answer=answer,
            evidence=evidence,
            paths=paths,
            seed_entities=seeds,
            backend=self.graph.name,
        )

    def _compose_answer(self, question: str, graph_answer: str, evidence) -> str:
        if evidence:
            support = evidence[0].chunk.text
            return f"{graph_answer}\n\nSupporting passage: {support}"
        return graph_answer or f"No indexed evidence for: {question}"

    def close(self) -> None:
        self.graph.close()
