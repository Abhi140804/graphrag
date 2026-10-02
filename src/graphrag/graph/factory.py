from __future__ import annotations

from graphrag.config import Settings, get_settings
from graphrag.graph.memory import MemoryGraphStore
from graphrag.graph.store import GraphStore


def create_graph_store(settings: Settings | None = None) -> GraphStore:
    settings = settings or get_settings()
    if settings.use_neo4j:
        from graphrag.graph.neo4j_store import Neo4jGraphStore

        try:
            return Neo4jGraphStore(
                settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
            )
        except Exception:
            return MemoryGraphStore()
    return MemoryGraphStore()
