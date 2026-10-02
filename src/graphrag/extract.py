from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from graphrag.models import Chunk, Entity, Relation

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_YEAR = re.compile(r"\s+in\s+\d{4}$")
_STOP = {"the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for"}

_ORG_HINTS = (
    "inc",
    "corp",
    "llc",
    "lab",
    "labs",
    "university",
    "openai",
    "google",
    "alphabet",
    "deepmind",
    "microsoft",
    "amazon",
    "anthropic",
    "gpt",
    "claude",
    "alphago",
)
_PLACE_HINTS = (
    "city",
    "valley",
    "view",
    "francisco",
    "london",
    "york",
    "seattle",
    "redmond",
    "mountain",
)


@dataclass
class Extraction:
    entities: list[Entity]
    relations: list[Relation]


def entity_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or hashlib.sha1(name.encode()).hexdigest()[:12]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def chunk_text(text: str, doc_id: str, max_chars: int = 420) -> list[Chunk]:
    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    idx = 0
    for sentence in sentences:
        if size + len(sentence) > max_chars and buf:
            chunks.append(Chunk(id=f"{doc_id}:{idx}", doc_id=doc_id, text=" ".join(buf)))
            idx += 1
            buf, size = [], 0
        buf.append(sentence)
        size += len(sentence)
    if buf:
        chunks.append(Chunk(id=f"{doc_id}:{idx}", doc_id=doc_id, text=" ".join(buf)))
    return chunks


def guess_type(name: str) -> str:
    lowered = name.lower()
    if any(h in lowered.split() or h == lowered for h in _PLACE_HINTS):
        return "LOCATION"
    if any(h in lowered for h in _ORG_HINTS):
        return "ORG"
    tokens = name.split()
    if len(tokens) >= 2 and all(t[:1].isupper() for t in tokens):
        return "PERSON"
    return "ORG" if name[:1].isupper() else "CONCEPT"


def _clean_span(span: str) -> str:
    span = span.strip(" .;,:")
    span = re.sub(r"^(the|a|an)\s+", "", span, flags=re.I)
    span = _YEAR.sub("", span)
    span = re.split(r"\b(?:which|that|who|where)\b", span, maxsplit=1)[0]
    return span.strip(" .;,:")


def _entity(name: str) -> Entity:
    return Entity(id=entity_id(name), name=name, type=guess_type(name), aliases=[name])


def extract_from_text(text: str, doc_id: str = "", chunk_id: str = "") -> Extraction:
    entities: dict[str, Entity] = {}
    relations: list[Relation] = []

    def add(src: Entity, tgt: Entity, rel_type: str, evidence: str, confidence: float = 0.9) -> None:
        entities[src.id] = src
        entities[tgt.id] = tgt
        relations.append(
            Relation(
                source_id=src.id,
                target_id=tgt.id,
                type=rel_type,
                evidence=evidence,
                confidence=confidence,
                doc_id=doc_id,
                chunk_id=chunk_id,
            )
        )

    rules: list[tuple[re.Pattern[str], str, str, str]] = [
        (re.compile(r"^(?P<t>.+?) was acquired by (?P<s>.+)$", re.I), "s", "t", "ACQUIRED"),
        (re.compile(r"^(?P<s>.+?) acquired (?P<t>.+)$", re.I), "s", "t", "ACQUIRED"),
        (re.compile(r"^(?P<s>.+?) (?:is|was) (?:the )?CEO of (?P<t>.+)$", re.I), "s", "t", "CEO_OF"),
        (
            re.compile(r"^(?P<t>.+?) (?:was|were) (?:founded|co-founded) by (?P<s>.+?)(?: in (?P<loc>.+))?$", re.I),
            "s",
            "t",
            "FOUNDED",
        ),
        (re.compile(r"^(?P<s>.+?) (?:founded|co-founded) (?P<t>.+)$", re.I), "s", "t", "FOUNDED"),
        (re.compile(r"^(?P<s>.+?) (?:is|was) (?:a |an )?subsidiary of (?P<t>.+)$", re.I), "s", "t", "SUBSIDIARY_OF"),
        (re.compile(r"^(?P<s>.+?) (?:is|was) (?:the )?parent (?:company )?of (?P<t>.+)$", re.I), "s", "t", "PARENT_OF"),
        (re.compile(r"^(?P<s>.+?) (?:is|was) (?:headquartered|based) in (?P<t>.+)$", re.I), "s", "t", "LOCATED_IN"),
        (re.compile(r"^(?P<s>.+?) (?:developed|created|built|released) (?P<t>.+)$", re.I), "s", "t", "DEVELOPED"),
        (re.compile(r"^(?P<s>.+?) (?:works|worked) at (?P<t>.+)$", re.I), "s", "t", "WORKS_AT"),
        (re.compile(r"^(?P<s>.+?) (?:partnered with|partners with) (?P<t>.+)$", re.I), "s", "t", "PARTNERED_WITH"),
    ]

    for sentence in split_sentences(text):
        stripped = sentence.rstrip(".")
        matched = False
        for pattern, src_key, tgt_key, rel_type in rules:
            match = pattern.match(stripped)
            if not match:
                continue
            src = _entity(_clean_span(match.group(src_key)))
            tgt = _entity(_clean_span(match.group(tgt_key)))
            if src.name and tgt.name and src.id != tgt.id:
                add(src, tgt, rel_type, sentence)
                loc = match.groupdict().get("loc")
                if loc:
                    place = _entity(_clean_span(loc))
                    add(tgt, place, "LOCATED_IN", sentence, 0.8)
                matched = True
                break
        if not matched:
            for name in re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,4})\b", sentence):
                if name.lower() not in _STOP:
                    ent = _entity(name)
                    entities.setdefault(ent.id, ent)

    return Extraction(entities=list(entities.values()), relations=relations)
