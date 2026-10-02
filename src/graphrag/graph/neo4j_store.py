from __future__ import annotations

from neo4j import GraphDatabase

from graphrag.graph.store import GraphStore
from graphrag.models import Entity, GraphPath, Relation


class Neo4jGraphStore(GraphStore):
    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
            )

    def upsert_entity(self, entity: Entity) -> None:
        aliases = list({*entity.aliases, entity.name})
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {id: $id})
                SET e.name = coalesce(e.name, $name),
                    e.type = CASE WHEN e.type IS NULL OR e.type = 'UNKNOWN' THEN $type ELSE e.type END,
                    e.aliases = $aliases
                """,
                id=entity.id,
                name=entity.name,
                type=entity.type,
                aliases=aliases,
            )

    def upsert_relation(self, relation: Relation) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (s:Entity {id: $source_id})
                MERGE (t:Entity {id: $target_id})
                MERGE (s)-[r:RELATED {type: $type}]->(t)
                SET r.evidence = $evidence,
                    r.confidence = $confidence,
                    r.doc_id = $doc_id,
                    r.chunk_id = $chunk_id
                """,
                **relation.model_dump(),
            )

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._driver.session() as session:
            rec = session.run("MATCH (e:Entity {id: $id}) RETURN e", id=entity_id).single()
        if not rec:
            return None
        return self._entity_from_node(rec["e"])

    def search_entities(self, query: str, limit: int = 8) -> list[Entity]:
        tokens = [t for t in query.lower().split() if len(t) > 2]
        with self._driver.session() as session:
            records = session.run(
                """
                MATCH (e:Entity)
                WITH e, toLower(e.name) AS n, [a IN coalesce(e.aliases, []) | toLower(a)] AS aliases
                WHERE n CONTAINS toLower($q)
                   OR toLower($q) CONTAINS n
                   OR any(a IN aliases WHERE a CONTAINS toLower($q) OR toLower($q) CONTAINS a)
                   OR any(tok IN $tokens WHERE n CONTAINS tok OR any(a IN aliases WHERE a CONTAINS tok))
                RETURN e,
                       CASE
                         WHEN n = toLower($q) THEN 100
                         WHEN n CONTAINS toLower($q) OR toLower($q) CONTAINS n THEN 80
                         ELSE 40
                       END AS score
                ORDER BY score DESC
                LIMIT $limit
                """,
                q=query,
                tokens=tokens,
                limit=limit,
            )
            return [self._entity_from_node(r["e"]) for r in records]

    def neighbors(self, entity_id: str, hops: int = 1) -> tuple[list[Entity], list[Relation]]:
        entities, relations, _ = self.expand([entity_id], hops=hops)
        return entities, relations

    def shortest_paths(
        self, source_id: str, target_id: str, max_hops: int = 4
    ) -> list[GraphPath]:
        hops = max(1, min(max_hops, 6))
        query = (
            "MATCH (s:Entity {id: $source}), (t:Entity {id: $target}) "
            f"MATCH p = shortestPath((s)-[*..{hops}]-(t)) "
            "RETURN p LIMIT 8"
        )
        with self._driver.session() as session:
            records = session.run(query, source=source_id, target=target_id)
            return [path for rec in records if (path := self._path_from_neo4j(rec["p"]))]

    def expand(
        self, seed_ids: list[str], hops: int = 3, limit: int = 50
    ) -> tuple[list[Entity], list[Relation], list[GraphPath]]:
        hops = max(1, min(hops, 6))
        query = (
            "MATCH (s:Entity) WHERE s.id IN $seeds "
            f"MATCH p = (s)-[*1..{hops}]-(n:Entity) "
            "WITH p LIMIT $limit RETURN p"
        )
        with self._driver.session() as session:
            records = list(session.run(query, seeds=seed_ids, limit=limit))
        entities: dict[str, Entity] = {}
        relations: dict[tuple[str, str, str], Relation] = {}
        paths: list[GraphPath] = []
        for rec in records:
            path = self._path_from_neo4j(rec["p"])
            if not path:
                continue
            paths.append(path)
            for node in path.nodes:
                entities[node.id] = node
            for rel in path.edges:
                relations[(rel.source_id, rel.target_id, rel.type)] = rel
        return list(entities.values()), list(relations.values()), paths[:40]

    def stats(self) -> dict[str, int | str]:
        with self._driver.session() as session:
            rec = session.run(
                "MATCH (e:Entity) OPTIONAL MATCH (e)-[r:RELATED]->() "
                "RETURN count(DISTINCT e) AS nodes, count(r) AS edges"
            ).single()
        return {
            "backend": self.name,
            "nodes": rec["nodes"] if rec else 0,
            "edges": rec["edges"] if rec else 0,
        }

    def snapshot(self) -> dict[str, list]:
        with self._driver.session() as session:
            node_rows = session.run("MATCH (e:Entity) RETURN e")
            nodes = [self._entity_from_node(r["e"]).model_dump() for r in node_rows]
            edge_rows = session.run(
                """
                MATCH (s:Entity)-[r:RELATED]->(t:Entity)
                RETURN s.id AS source_id, t.id AS target_id, r
                """
            )
            edges = []
            for row in edge_rows:
                rel = row["r"]
                edges.append(
                    {
                        "source_id": row["source_id"],
                        "target_id": row["target_id"],
                        "type": rel.get("type") or rel.type,
                        "evidence": rel.get("evidence") or "",
                        "confidence": float(rel.get("confidence") or 1.0),
                        "doc_id": rel.get("doc_id") or "",
                        "chunk_id": rel.get("chunk_id") or "",
                    }
                )
        return {"nodes": nodes, "edges": edges}

    def close(self) -> None:
        self._driver.close()

    def _entity_from_node(self, node) -> Entity:
        data = dict(node)
        return Entity(
            id=data.get("id", ""),
            name=data.get("name", data.get("id", "")),
            type=data.get("type", "UNKNOWN"),
            aliases=list(data.get("aliases") or []),
            properties={
                k: str(v)
                for k, v in data.items()
                if k not in {"id", "name", "type", "aliases"}
            },
        )

    def _path_from_neo4j(self, path) -> GraphPath | None:
        nodes = [self._entity_from_node(n) for n in path.nodes]
        edges: list[Relation] = []
        for rel in path.relationships:
            edges.append(
                Relation(
                    source_id=rel.start_node.get("id"),
                    target_id=rel.end_node.get("id"),
                    type=rel.get("type") or rel.type,
                    evidence=rel.get("evidence") or "",
                    confidence=float(rel.get("confidence") or 1.0),
                    doc_id=rel.get("doc_id") or "",
                    chunk_id=rel.get("chunk_id") or "",
                )
            )
        hops = max(len(nodes) - 1, 0)
        return GraphPath(nodes=nodes, edges=edges, hops=hops, score=1.0 / (1.0 + hops))
