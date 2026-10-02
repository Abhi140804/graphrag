from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from graphrag.models import Document
from graphrag.pipeline import GraphRAG

STATIC = Path(__file__).resolve().parent / "static"
SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample_corpus.json"

rag: GraphRAG | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global rag
    rag = GraphRAG()
    if rag.graph.stats().get("nodes", 0) == 0 and SAMPLE.exists():
        rag.ingest_file(SAMPLE)
    yield
    if rag:
        rag.close()


app = FastAPI(
    title="Graph-Enhanced RAG",
    description="Hybrid vector-graph retrieval with multi-hop entity reasoning.",
    lifespan=lifespan,
)


class IngestRequest(BaseModel):
    documents: list[Document]


class IngestTextRequest(BaseModel):
    title: str = "untitled"
    text: str = Field(min_length=8)
    doc_id: str = ""


class QueryRequest(BaseModel):
    question: str = Field(min_length=2)


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    assert rag
    return {"ok": True, **rag.graph.stats(), "chunks": rag.vectors.size}


@app.get("/graph")
def graph_snapshot() -> dict:
    assert rag
    snap = rag.graph.snapshot()
    return {"backend": rag.graph.name, "chunks": rag.vectors.size, **snap}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    assert rag
    return rag.ingest_documents(req.documents)


@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest) -> dict:
    assert rag
    doc_id = req.doc_id.strip() or req.title.lower().replace(" ", "-")[:40]
    return rag.ingest_documents(
        [Document(id=doc_id, title=req.title, text=req.text)]
    )


@app.post("/ingest/sample")
def ingest_sample() -> dict:
    assert rag
    if not SAMPLE.exists():
        raise HTTPException(status_code=404, detail="Sample corpus not found")
    return rag.ingest_file(SAMPLE)


@app.post("/query")
def query(req: QueryRequest) -> dict:
    assert rag
    return rag.query(req.question).model_dump()


app.mount("/static", StaticFiles(directory=STATIC), name="static")
