from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphrag.pipeline import GraphRAG

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample_corpus.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-Enhanced RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ingest = sub.add_parser("ingest", help="Ingest a JSON corpus")
    ingest.add_argument("--file", default=str(SAMPLE))
    ask = sub.add_parser("query", help="Ask a hybrid vector-graph question")
    ask.add_argument("question")
    ask.add_argument("--file", default=str(SAMPLE), help="Corpus to ingest first")
    sub.add_parser("serve", help="Run the FastAPI server")
    args = parser.parse_args()

    if args.cmd == "serve":
        import threading
        import time
        import webbrowser

        import uvicorn

        def _open_dashboard() -> None:
            time.sleep(0.9)
            webbrowser.open("http://127.0.0.1:8000/")

        threading.Thread(target=_open_dashboard, daemon=True).start()
        uvicorn.run("graphrag.api:app", host="127.0.0.1", port=8000, reload=False)
        return

    rag = GraphRAG()
    try:
        stats = rag.ingest_file(args.file)
        if args.cmd == "ingest":
            print(json.dumps(stats, indent=2))
            return
        result = rag.query(args.question)
        print(result.answer)
        print("\n-- paths --")
        for path in result.paths[:5]:
            names = " -> ".join(n.name for n in path.nodes)
            types = ", ".join(e.type for e in path.edges)
            print(f"[{path.hops} hop | {types}] {names}")
    finally:
        rag.close()


if __name__ == "__main__":
    main()
