from __future__ import annotations

import re

from graphrag.graph.store import GraphStore
from graphrag.models import Entity, GraphPath

_RELATIONAL_CUES = re.compile(
    r"\b(who|which company|acquired|founder|ceo|parent|subsidiary|based|headquarter|"
    r"partner|developed|works at|relationship|connected|between)\b",
    re.I,
)
_CUE_RELATIONS = (
    ("acquired", "ACQUIRED"),
    ("ceo", "CEO_OF"),
    ("founder", "FOUNDED"),
    ("founded", "FOUNDED"),
    ("parent", "SUBSIDIARY_OF"),
    ("subsidiary", "SUBSIDIARY_OF"),
    ("headquarter", "LOCATED_IN"),
    ("based", "LOCATED_IN"),
    ("where", "LOCATED_IN"),
    ("partner", "PARTNERED_WITH"),
    ("developed", "DEVELOPED"),
)


def is_relational_query(query: str) -> bool:
    return bool(_RELATIONAL_CUES.search(query))


def seed_entities(graph: GraphStore, query: str, limit: int = 6) -> list[Entity]:
    seeds = graph.search_entities(query, limit=limit)
    # Prefer longer/more specific names that actually appear in the query.
    q = query.lower()
    ranked = []
    for entity in seeds:
        names = [entity.name, *entity.aliases]
        hit = max((len(n) for n in names if n.lower() in q), default=0)
        ranked.append((hit, entity))
    ranked.sort(key=lambda item: -item[0])
    return [e for hit, e in ranked if hit] or seeds[:3]


def format_path(path: GraphPath) -> str:
    if not path.nodes:
        return ""
    parts = [path.nodes[0].name]
    for i, edge in enumerate(path.edges):
        nxt = path.nodes[i + 1].name if i + 1 < len(path.nodes) else "?"
        forward = edge.source_id == path.nodes[i].id
        arrow = f"-[{edge.type}]->" if forward else f"<-[{edge.type}]-"
        parts.append(f"{arrow} {nxt}")
    return " ".join(parts)


def path_relevance(query: str, path: GraphPath) -> float:
    q = query.lower()
    score = path.score
    for node in path.nodes:
        lowered = node.name.lower()
        if lowered in q:
            score += 1.5
        if "where" in q and node.type == "LOCATION":
            score += 4
        if "who" in q and node.type == "PERSON":
            score += 2
    for cue, rel in _CUE_RELATIONS:
        if cue in q and any(edge.type == rel for edge in path.edges):
            score += 3
    score -= 0.15 * path.hops
    return score


def rank_paths(query: str, paths: list[GraphPath], limit: int = 5) -> list[GraphPath]:
    return sorted(paths, key=lambda p: path_relevance(query, p), reverse=True)[:limit]


def reason_over_paths(query: str, paths: list[GraphPath], seeds: list[Entity]) -> str:
    if not paths:
        if seeds:
            names = ", ".join(e.name for e in seeds[:5])
            return f"Linked entities for the query: {names}."
        return "No graph paths were found for this query."

    best = rank_paths(query, paths, limit=3)
    lines = ["Multi-hop graph reasoning:"]
    for path in best:
        lines.append(f"- {format_path(path)}")
        if path.edges:
            evidence = next((e.evidence for e in path.edges if e.evidence), "")
            if evidence:
                lines.append(f"  evidence: {evidence}")
    if is_relational_query(query) and best:
        tail = _answer_node(query, best[0])
        lines.append(f"Likely answer entity: {tail.name} ({tail.type}).")
    return "\n".join(lines)


def _answer_node(query: str, path: GraphPath):
    q = query.lower()
    if "where" in q:
        for node in reversed(path.nodes):
            if node.type == "LOCATION":
                return node
    if "who" in q:
        for node in reversed(path.nodes):
            if node.type == "PERSON":
                return node
    return path.nodes[-1]
