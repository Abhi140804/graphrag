# Graph-Enhanced RAG (GraphRAG)

Hybrid **vector + knowledge-graph** retrieval for questions that need entity relationships, not just similar text.

## What it does

- Maps entities and relations out of unstructured text (acquisitions, founders, CEOs, locations, products).
- Stores them in **Neo4j** when `NEO4J_URI` is set, otherwise an **in-memory NetworkX** graph.
- Retrieves with **reciprocal rank fusion**: dense/sparse vector hits plus multi-hop graph expansion.
- Returns answer text plus the connecting paths (multi-hop reasoning).

A relational question such as *“Who is the CEO of the company that acquired DeepMind?”* walks `DeepMind ← ACQUIRED ← Google ← CEO_OF ← Sundar Pichai` instead of hoping a single chunk contains the full chain.

## Setup

```bash
cd ~/Projects/graphrag
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Neo4j is optional. Leave `NEO4J_URI` empty to use the in-memory graph. Optional `EMBEDDING_MODEL` switches TF-IDF for sentence-transformers (`pip install -e ".[transformers]"`). Optional `OPENAI_API_KEY` lets answers be written by an LLM.

## Run

```bash
# ingest sample corpus and ask a multi-hop question
python -m graphrag.cli query "Who is the CEO of the company that acquired DeepMind?"

python -m graphrag.cli ingest --file data/sample_corpus.json
python -m graphrag.cli serve
```

The server loads the sample corpus, opens **http://127.0.0.1:8000/** (the dashboard), and still exposes OpenAPI at **http://127.0.0.1:8000/docs**.

```bash
curl -s localhost:8000/ingest -H 'content-type: application/json' -d @data/sample_corpus.json
# POST {"documents":[...]} is the API shape; wrap the file or use /docs
```

```python
from graphrag import GraphRAG

rag = GraphRAG()
rag.ingest_file("data/sample_corpus.json")
print(rag.query("Where is the parent company of Google headquartered?").answer)
```

## Tests

```bash
pytest -q
```

## Layout

- `src/graphrag/extract.py` — sentence-level entity/relation mapping
- `src/graphrag/graph/` — `GraphStore` with memory + Neo4j backends
- `src/graphrag/retrieval.py` — hybrid RRF ranking
- `src/graphrag/reasoning.py` — multi-hop path explanations
- `src/graphrag/api.py` — FastAPI dashboard, `/ingest`, `/query`, `/graph`, `/health`
- `src/graphrag/static/` — GraphRAG Console UI
