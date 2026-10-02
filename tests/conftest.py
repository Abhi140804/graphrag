from pathlib import Path

import pytest

from graphrag.graph.memory import MemoryGraphStore
from graphrag.models import Document
from graphrag.pipeline import GraphRAG

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_corpus.json"


@pytest.fixture
def rag_from_sample() -> GraphRAG:
    rag = GraphRAG(graph=MemoryGraphStore())
    rag.ingest_file(SAMPLE)
    return rag
