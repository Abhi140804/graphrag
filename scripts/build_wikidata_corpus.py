"""Build a large company-knowledge corpus from Wikidata.

Each company becomes one document of short factual sentences (CEO, founders,
headquarters, parent company, developed products) that match the patterns in
graphrag.extract, so the rule-based extractor can build a real multi-hop graph.

Usage: python scripts/build_wikidata_corpus.py --min-sitelinks 10 --out data/wikidata_companies.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "graphrag-corpus-builder/0.1 (https://github.com/Abhi140804/graphrag)"
COMPANY_TYPES = "wd:Q4830453 wd:Q783794 wd:Q891723 wd:Q6881511 wd:Q1058914 wd:Q210167 wd:Q2085381"

COMPANIES = """
SELECT DISTINCT ?co WHERE {{
  VALUES ?type {{ {types} }}
  ?co wdt:P31 ?type; wikibase:sitelinks ?links.
  FILTER(?links >= {min_links})
}}
"""

RELATION = """
SELECT ?co ?coLabel ?obj ?objLabel WHERE {{
  VALUES ?co {{ {ids} }}
  {pattern}
  ?co rdfs:label ?coLabel. FILTER(LANG(?coLabel) = "en")
  ?obj rdfs:label ?objLabel. FILTER(LANG(?objLabel) = "en")
}}
"""

PATTERNS = {
    "ceo": "?co wdt:P169 ?obj.",
    "founder": "?co wdt:P112 ?obj.",
    "hq": "?co wdt:P159 ?obj.",
    "parent": "?co wdt:P749 ?obj.",
    "product": "?obj wdt:P178 ?co; wikibase:sitelinks ?pl. FILTER(?pl >= 15)",
}

TEMPLATES = {
    "ceo": "{obj} is the CEO of {co}.",
    "founder": "{co} was founded by {obj}.",
    "hq": "{co} is headquartered in {obj}.",
    "parent": "{co} is a subsidiary of {obj}.",
    "product": "{co} developed {obj}.",
}


def sparql(query: str) -> list[dict]:
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=data,
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)["results"]["bindings"]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    return []


def clean(label: str) -> str:
    return " ".join(label.replace(".", " ").split())


def usable(label: str) -> bool:
    # Very short or lowercase names (e.g. "C", "ed") substring-match unrelated questions.
    return len(label) >= 3 and (label[0].isupper() or label[0].isdigit()) and not re.fullmatch(r"Q\d+", label)


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def build(min_links: int, batch: int) -> list[dict]:
    rows = sparql(COMPANIES.format(types=COMPANY_TYPES, min_links=min_links))
    company_ids = sorted({qid(r["co"]["value"]) for r in rows})
    print(f"{len(company_ids)} companies")

    labels: dict[str, str] = {}
    facts: dict[str, list[str]] = defaultdict(list)
    for kind, pattern in PATTERNS.items():
        count = 0
        for i in range(0, len(company_ids), batch):
            ids = " ".join(f"wd:{c}" for c in company_ids[i : i + batch])
            for r in sparql(RELATION.format(ids=ids, pattern=pattern)):
                co, obj = clean(r["coLabel"]["value"]), clean(r["objLabel"]["value"])
                if not usable(co) or not usable(obj) or co == obj:
                    continue
                cid = qid(r["co"]["value"])
                labels[cid] = co
                sentence = TEMPLATES[kind].format(co=co, obj=obj)
                if sentence not in facts[cid]:
                    facts[cid].append(sentence)
                    count += 1
        print(f"{kind}: {count} facts")

    return [
        {"id": f"wd-{cid}", "title": labels[cid], "text": " ".join(sentences)}
        for cid, sentences in sorted(facts.items(), key=lambda kv: labels[kv[0]])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-sitelinks", type=int, default=10)
    parser.add_argument("--batch", type=int, default=300)
    parser.add_argument("--out", default="data/wikidata_companies.json")
    args = parser.parse_args()
    docs = build(args.min_sitelinks, args.batch)
    Path(args.out).write_text(json.dumps(docs, indent=1, ensure_ascii=False))
    print(f"wrote {len(docs)} documents to {args.out}")


if __name__ == "__main__":
    main()
