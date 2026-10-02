from graphrag.graph.memory import MemoryGraphStore
from graphrag.models import Entity, Relation


def test_multihop_path_from_acquisition_to_ceo():
    g = MemoryGraphStore()
    for name, typ in (
        ("DeepMind", "ORG"),
        ("Google", "ORG"),
        ("Sundar Pichai", "PERSON"),
        ("Alphabet", "ORG"),
        ("Mountain View", "LOCATION"),
    ):
        g.upsert_entity(Entity(id=name.lower().replace(" ", "_"), name=name, type=typ))
    g.upsert_relation(Relation(source_id="google", target_id="deepmind", type="ACQUIRED"))
    g.upsert_relation(Relation(source_id="sundar_pichai", target_id="google", type="CEO_OF"))
    g.upsert_relation(Relation(source_id="google", target_id="alphabet", type="SUBSIDIARY_OF"))
    g.upsert_relation(Relation(source_id="alphabet", target_id="mountain_view", type="LOCATED_IN"))

    paths = g.shortest_paths("deepmind", "sundar_pichai", max_hops=3)
    assert paths
    names = [n.id for n in paths[0].nodes]
    assert "deepmind" in names and "sundar_pichai" in names
    assert paths[0].hops == 2

    entities, _, expanded = g.expand(["deepmind"], hops=3)
    entity_ids = {e.id for e in entities}
    assert "mountain_view" in entity_ids
    assert expanded

    snap = g.snapshot()
    assert {n["id"] for n in snap["nodes"]} >= {"deepmind", "google", "sundar_pichai"}
    assert any(e["type"] == "ACQUIRED" for e in snap["edges"])
