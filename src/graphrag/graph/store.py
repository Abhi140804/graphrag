from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag.models import Entity, GraphPath, Relation


class GraphStore(ABC):
    """Knowledge graph backend used for multi-hop expansion."""

    name: str = "abstract"

    @abstractmethod
    def upsert_entity(self, entity: Entity) -> None: ...

    @abstractmethod
    def upsert_relation(self, relation: Relation) -> None: ...

    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity | None: ...

    @abstractmethod
    def search_entities(self, query: str, limit: int = 8) -> list[Entity]: ...

    @abstractmethod
    def neighbors(self, entity_id: str, hops: int = 1) -> tuple[list[Entity], list[Relation]]: ...

    @abstractmethod
    def shortest_paths(
        self, source_id: str, target_id: str, max_hops: int = 4
    ) -> list[GraphPath]: ...

    @abstractmethod
    def expand(
        self, seed_ids: list[str], hops: int = 3, limit: int = 50
    ) -> tuple[list[Entity], list[Relation], list[GraphPath]]: ...

    @abstractmethod
    def stats(self) -> dict[str, int | str]: ...

    @abstractmethod
    def snapshot(self) -> dict[str, list]: ...

    @abstractmethod
    def close(self) -> None: ...
