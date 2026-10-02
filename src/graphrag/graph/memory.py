from __future__ import annotations

from collections import deque

import networkx as nx

from graphrag.graph.store import GraphStore
from graphrag.models import Entity, GraphPath, Relation


class MemoryGraphStore(GraphStore):
    """Undirected-for-search, directed-for-semantics NetworkX fallback."""

    name = "memory"

    def __init__(self) -> None:
        self._g = nx.MultiDiGraph()
        self._entities: dict[str, Entity] = {}

    def upsert_entity(self, entity: Entity) -> None:
        existing = self._entities.get(entity.id)
        if existing:
            aliases = sorted(set(existing.aliases + entity.aliases + [entity.name, existing.name]))
            props = {**existing.properties, **entity.properties}
            entity = Entity(
                id=entity.id,
                name=existing.name or entity.name,
                type=existing.type if existing.type != "UNKNOWN" else entity.type,
                aliases=aliases,
                properties=props,
            )
        self._entities[entity.id] = entity
        self._g.add_node(entity.id, **entity.model_dump())

    def upsert_relation(self, relation: Relation) -> None:
        if relation.source_id not in self._g:
            self.upsert_entity(
                Entity(id=relation.source_id, name=relation.source_id, type="UNKNOWN")
            )
        if relation.target_id not in self._g:
            self.upsert_entity(
                Entity(id=relation.target_id, name=relation.target_id, type="UNKNOWN")
            )
        for _, _, data in self._g.edges(relation.source_id, data=True):
            if (
                data.get("target_id") == relation.target_id
                and data.get("type") == relation.type
            ):
                return
        self._g.add_edge(
            relation.source_id,
            relation.target_id,
            **relation.model_dump(),
        )

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def search_entities(self, query: str, limit: int = 8) -> list[Entity]:
        q = query.lower()
        scored: list[tuple[int, Entity]] = []
        for entity in self._entities.values():
            names = [entity.name, entity.id, *entity.aliases]
            score = 0
            for name in names:
                n = name.lower()
                if not n:
                    continue
                if n == q:
                    score = max(score, 100)
                elif n in q or q in n:
                    score = max(score, 80)
                elif any(tok and tok in n for tok in q.split() if len(tok) > 2):
                    score = max(score, 40)
            if score:
                scored.append((score, entity))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [entity for _, entity in scored[:limit]]

    def neighbors(self, entity_id: str, hops: int = 1) -> tuple[list[Entity], list[Relation]]:
        entities, relations, _ = self.expand([entity_id], hops=hops)
        return entities, relations

    def shortest_paths(
        self, source_id: str, target_id: str, max_hops: int = 4
    ) -> list[GraphPath]:
        if source_id not in self._g or target_id not in self._g:
            return []
        undirected = self._g.to_undirected()
        try:
            raw_paths = nx.all_simple_paths(undirected, source_id, target_id, cutoff=max_hops)
        except nx.NetworkXError:
            return []
        results: list[GraphPath] = []
        for node_ids in raw_paths:
            path = self._path_from_nodes(node_ids)
            if path:
                results.append(path)
            if len(results) >= 8:
                break
        results.sort(key=lambda p: (p.hops, -p.score))
        return results

    def expand(
        self, seed_ids: list[str], hops: int = 3, limit: int = 50
    ) -> tuple[list[Entity], list[Relation], list[GraphPath]]:
        seen: set[str] = set()
        relations: dict[tuple[str, str, str], Relation] = {}
        paths: list[GraphPath] = []
        queue: deque[tuple[str, list[str]]] = deque()
        for seed in seed_ids:
            if seed in self._g:
                queue.append((seed, [seed]))
                seen.add(seed)

        while queue and len(seen) < limit:
            node, trail = queue.popleft()
            hop = len(trail) - 1
            if hop >= hops:
                continue
            for nbr, edge_data in self._incident(node):
                rel = Relation(**{k: v for k, v in edge_data.items() if k in Relation.model_fields})
                relations[(rel.source_id, rel.target_id, rel.type)] = rel
                if nbr not in seen:
                    seen.add(nbr)
                    new_trail = trail + [nbr]
                    queue.append((nbr, new_trail))
                    path = self._path_from_nodes(new_trail)
                    if path:
                        paths.append(path)

        entities = [self._entities[i] for i in seen if i in self._entities]
        return entities, list(relations.values()), paths[:40]

    def stats(self) -> dict[str, int | str]:
        return {
            "backend": self.name,
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
        }

    def snapshot(self) -> dict[str, list]:
        nodes = [entity.model_dump() for entity in self._entities.values()]
        edges = []
        for _src, _tgt, data in self._g.edges(data=True):
            edges.append({k: data[k] for k in Relation.model_fields if k in data})
        return {"nodes": nodes, "edges": edges}

    def close(self) -> None:
        return None

    def _incident(self, node: str) -> list[tuple[str, dict]]:
        items: list[tuple[str, dict]] = []
        for _, tgt, data in self._g.out_edges(node, data=True):
            items.append((tgt, data))
        for src, _, data in self._g.in_edges(node, data=True):
            items.append((src, data))
        return items

    def _path_from_nodes(self, node_ids: list[str]) -> GraphPath | None:
        if len(node_ids) < 2:
            return None
        nodes = [self._entities[i] for i in node_ids if i in self._entities]
        edges: list[Relation] = []
        for a, b in zip(node_ids, node_ids[1:]):
            rel = self._best_edge(a, b)
            if rel:
                edges.append(rel)
        hops = max(len(node_ids) - 1, 0)
        score = 1.0 / (1.0 + hops)
        return GraphPath(nodes=nodes, edges=edges, hops=hops, score=score)

    def _best_edge(self, a: str, b: str) -> Relation | None:
        candidates: list[Relation] = []
        for src, tgt in ((a, b), (b, a)):
            if not self._g.has_edge(src, tgt):
                continue
            for data in self._g.get_edge_data(src, tgt).values():
                candidates.append(Relation(**{k: v for k, v in data.items() if k in Relation.model_fields}))
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.confidence)
