from graphrag.retrieval import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_prefers_shared_hits():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]])
    assert fused["a"] > fused["b"]
    assert fused["c"] > fused["b"]


def test_hybrid_retrieval_answers_relational_query(rag_from_sample):
    result = rag_from_sample.query("Who is the CEO of the company that acquired DeepMind?")
    blob = (result.answer + " " + " ".join(e.chunk.text for e in result.evidence)).lower()
    assert "sundar pichai" in blob
    assert result.paths
    node_names = {n.name.lower() for path in result.paths for n in path.nodes}
    assert "deepmind" in node_names
    assert "google" in node_names
    sources = {item.source for item in result.evidence}
    assert sources & {"hybrid", "graph", "vector"}


def test_hybrid_retrieval_uses_location_hop(rag_from_sample):
    result = rag_from_sample.query("Where is the parent company of Google headquartered?")
    blob = result.answer.lower()
    assert "mountain view" in blob
    assert result.backend == "memory"
