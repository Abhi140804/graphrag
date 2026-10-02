from __future__ import annotations

from pydantic import BaseModel, Field


class Entity(BaseModel):
    id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, str] = Field(default_factory=dict)


class Relation(BaseModel):
    source_id: str
    target_id: str
    type: str
    evidence: str = ""
    confidence: float = 1.0
    doc_id: str = ""
    chunk_id: str = ""


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    entity_ids: list[str] = Field(default_factory=list)


class Document(BaseModel):
    id: str
    title: str = ""
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class GraphPath(BaseModel):
    nodes: list[Entity]
    edges: list[Relation]
    hops: int
    score: float = 0.0


class RetrievedEvidence(BaseModel):
    chunk: Chunk
    vector_score: float = 0.0
    graph_score: float = 0.0
    fused_score: float = 0.0
    source: str = "hybrid"


class QueryResult(BaseModel):
    query: str
    answer: str
    evidence: list[RetrievedEvidence]
    paths: list[GraphPath]
    seed_entities: list[Entity]
    backend: str
