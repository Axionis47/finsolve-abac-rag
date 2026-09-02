"""The LLM graph: (subject, relation, object) triples read out of each chunk.

Where this sits next to `build_structural.py`: that build knows the corpus
skeleton (which document, which department, which quarter). This one is meant to
add what the skeleton cannot see, the links inside the prose: which entity owns a
metric, which number belongs to which quarter, which cause is claimed for which
effect. Those are the links a multi-hop question needs.

TRUST BOUNDARY. The model's answer is untrusted text, so nothing it writes ever
reaches a query string or a permission decision. It contributes node labels and a
relation name, and that is all. The chunk it was extracted from is what decides
who may walk the edge, and that comes from our loop, not from the model.

FAILING IS NORMAL. A model asked for JSON will sometimes answer with prose, a
half array, or a triple with a missing field. When the answer for a chunk cannot
be read at all, that chunk is skipped and counted; the build carries on and
reports the fallback rate, so a bad prompt or a bad model shows up as a number
instead of as a mysteriously thin graph.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finrag.graph.schema import (
    Edge,
    LAYER_ENTITY,
    Node,
    chunk_node_id,
    dedupe_edges,
    dedupe_nodes,
    make_chunk_node,
    make_entity_node,
    normalize_entity,
    normalize_relation,
)
from finrag.graph.store import GraphCounts, GraphStore
from finrag.llm.base import LLMBackend
from finrag.models import Chunk

SYSTEM_PROMPT = (
    "You read internal company documents and pull out the facts they state as "
    "subject-relation-object triples. You answer with JSON and nothing else."
)

PROMPT_TEMPLATE = """List the facts this passage states, as triples.

Rules:
- Answer with a JSON array only. No prose, no code fence.
- Each item is {{"subject": "...", "relation": "...", "object": "..."}}
- Cover named entities, metrics, numbers, and any cause the passage claims.
- relation is a short snake_case verb phrase, such as reported_revenue,
  owned_by, caused_by, mitigated_by.
- Copy names and numbers exactly as the passage writes them.
- At most {max_triples} items. Answer [] if the passage states no facts.

Passage heading: {section_path}
Passage:
{text}
"""

# How many triples we ask for per chunk. Enough for a dense paragraph, small
# enough that one chunk cannot flood the graph with near-duplicates.
DEFAULT_MAX_TRIPLES = 12

# Bump this when the extraction instructions or parsing contract change. It is
# part of every cache key, so an old answer is never reused under a new prompt.
PROMPT_VERSION = "1"

# Longest entity name or relation we accept. A model that answers with a whole
# paragraph in the "object" field is malfunctioning, and a node labelled with a
# paragraph would match every query term in the seed scan.
MAX_NAME_LENGTH = 120

# A chunk node links to every entity a triple from it named, so the entity
# subgraph is reachable from the chunks and the chunks are reachable from it.
RELATION_MENTIONS_ENTITY = "mentions_entity"


@dataclass(frozen=True)
class Triple:
    """One extracted fact. Subject and object become entity nodes."""

    subject: str
    relation: str
    object: str


@dataclass(frozen=True)
class LLMBuildStats:
    """What the build did, so a bad extraction is visible as numbers."""

    n_chunks: int
    n_parsed: int
    n_skipped: int
    n_triples: int
    n_dropped_items: int
    n_cache_hits: int = 0
    n_model_calls: int = 0
    model: str = "unknown"
    prompt_version: str = PROMPT_VERSION

    @property
    def fallback_rate(self) -> float:
        """Share of chunks whose answer could not be read at all."""
        if self.n_chunks == 0:
            return 0.0
        return self.n_skipped / self.n_chunks


def _model_name(llm: LLMBackend, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return str(getattr(llm, "model", None) or getattr(llm, "name", "unknown"))


def _input_hash(chunk: Chunk) -> str:
    """Hash every chunk field that reaches the extraction prompt."""
    material = f"{chunk.section_path}\0{chunk.text}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _read_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    """Usable cache rows by chunk id; malformed derived rows are ignored."""
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with open(cache_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("chunk_id"), str):
                rows[row["chunk_id"]] = row
    return rows


def _cached_answer(
    row: dict[str, Any] | None,
    chunk: Chunk,
    model: str,
    max_triples: int,
) -> ParsedAnswer | None:
    """Read a current usable cache row, or return None to call the model."""
    if row is None:
        return None
    expected = {
        "input_hash": _input_hash(chunk),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "max_triples": max_triples,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            return None
    triples_raw = row.get("triples")
    if not isinstance(triples_raw, list):
        return None
    parsed = parse_triples(json.dumps(triples_raw))
    if parsed.triples is None:
        return None
    dropped = row.get("n_dropped", 0)
    if not isinstance(dropped, int) or isinstance(dropped, bool) or dropped < 0:
        dropped = 0
    return ParsedAnswer(triples=parsed.triples, n_dropped=dropped)


def _cache_row(
    chunk: Chunk,
    model: str,
    max_triples: int,
    parsed: ParsedAnswer,
) -> dict[str, Any]:
    triples = parsed.triples or []
    return {
        "chunk_id": chunk.chunk_id,
        "input_hash": _input_hash(chunk),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "max_triples": max_triples,
        "n_dropped": parsed.n_dropped,
        "triples": [
            {"subject": item.subject, "relation": item.relation, "object": item.object}
            for item in triples
        ],
    }


def _write_cache(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Replace the local derived cache with deterministic JSONL."""
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda item: str(item["chunk_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(cache_path)


def build_prompt(chunk: Chunk, max_triples: int = DEFAULT_MAX_TRIPLES) -> str:
    """The extraction prompt for one chunk."""
    return PROMPT_TEMPLATE.format(
        max_triples=max_triples,
        section_path=chunk.section_path,
        text=chunk.text,
    )


def find_json_array(raw: str) -> str | None:
    """The outermost `[ ... ]` in the answer, or None if there is none.

    Models wrap JSON in code fences and apologies, and refusing those answers
    would throw away work that is otherwise fine. Slicing between the first "["
    and the last "]" handles every wrapper we have seen without a parser.
    """
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    return raw[start : end + 1]


@dataclass(frozen=True)
class ParsedAnswer:
    """One model answer, read. `triples` is None when the answer was unusable."""

    triples: list[Triple] | None
    n_dropped: int = 0

    @property
    def usable(self) -> bool:
        return self.triples is not None


def parse_triples(raw: str) -> ParsedAnswer:
    """Read one model answer.

    `triples is None` means "skip this chunk and count it": no array found, or
    the JSON does not parse, or it is not a list. A list that parses but holds
    some malformed items keeps the good ones and counts the rest in `n_dropped`,
    because one bad item is not a failed chunk.
    """
    candidate = find_json_array(raw)
    if candidate is None:
        return ParsedAnswer(triples=None)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return ParsedAnswer(triples=None)
    if not isinstance(parsed, list):
        return ParsedAnswer(triples=None)

    triples: list[Triple] = []
    n_dropped = 0
    for item in parsed:
        triple = _read_triple(item)
        if triple is None:
            n_dropped += 1
        else:
            triples.append(triple)
    return ParsedAnswer(triples=triples, n_dropped=n_dropped)


def _read_triple(item: Any) -> Triple | None:
    """One triple from one list item, or None if it is not usable."""
    if not isinstance(item, dict):
        return None

    subject = _read_name(item.get("subject"))
    relation = _read_name(item.get("relation"))
    object_ = _read_name(item.get("object"))
    if subject is None or relation is None or object_ is None:
        return None

    folded_relation = normalize_relation(relation)
    if not folded_relation:
        return None
    if not normalize_entity(subject) or not normalize_entity(object_):
        return None
    return Triple(subject=subject, relation=folded_relation, object=object_)


def _read_name(value: Any) -> str | None:
    """A short non-empty string, or None. Numbers are accepted as their text."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed or len(trimmed) > MAX_NAME_LENGTH:
        return None
    return trimmed


def triples_to_graph(chunk: Chunk, triples: Sequence[Triple]) -> tuple[list[Node], list[Edge]]:
    """Nodes and edges for one chunk's triples.

    Three edges come out of one triple: subject -relation-> object, and a
    mentions_entity edge from the chunk to each end. Without those two the entity
    graph would float free of the corpus and a traversal could never get from a
    query term to a chunk to return.
    """
    nodes: list[Node] = [make_chunk_node(chunk)]
    edges: list[Edge] = []
    this_chunk = chunk_node_id(chunk.chunk_id)

    for triple in triples:
        subject = make_entity_node(triple.subject)
        object_ = make_entity_node(triple.object)
        if subject.id == object_.id:
            # A self-loop says nothing and would inflate the path counts the
            # traversal ranks by.
            continue
        nodes.append(subject)
        nodes.append(object_)
        edges.append(
            Edge(
                source=subject.id,
                target=object_.id,
                relation=triple.relation,
                layer=LAYER_ENTITY,
                source_chunk_id=chunk.chunk_id,
            )
        )
        for entity in (subject, object_):
            edges.append(
                Edge(
                    source=this_chunk,
                    target=entity.id,
                    relation=RELATION_MENTIONS_ENTITY,
                    layer=LAYER_ENTITY,
                    source_chunk_id=chunk.chunk_id,
                )
            )

    return nodes, edges


def extract(
    llm: LLMBackend,
    chunks: Sequence[Chunk],
    max_triples: int = DEFAULT_MAX_TRIPLES,
    cache_path: str | Path | None = None,
    model: str | None = None,
) -> tuple[list[Node], list[Edge], LLMBuildStats]:
    """Ask the model for triples chunk by chunk and collect one graph.

    One call per chunk, on purpose: a batched prompt saves calls but one bad
    answer then costs the whole batch, and the fallback rate stops meaning
    anything per chunk.
    """
    nodes: list[Node] = []
    edges: list[Edge] = []
    n_parsed = 0
    n_skipped = 0
    n_triples = 0
    n_dropped_items = 0
    n_cache_hits = 0
    n_model_calls = 0
    resolved_model = _model_name(llm, model)
    cached = _read_cache(cache_path) if cache_path is not None else {}
    cache_rows: list[dict[str, Any]] = []

    for chunk in chunks:
        parsed = _cached_answer(cached.get(chunk.chunk_id), chunk, resolved_model, max_triples)
        if parsed is None:
            n_model_calls += 1
            answer = llm.generate(build_prompt(chunk, max_triples), system=SYSTEM_PROMPT)
            parsed = parse_triples(answer)
        else:
            n_cache_hits += 1
        if parsed.triples is None:
            n_skipped += 1
            continue

        if cache_path is not None:
            cache_rows.append(_cache_row(chunk, resolved_model, max_triples, parsed))

        n_parsed += 1
        n_triples += len(parsed.triples)
        n_dropped_items += parsed.n_dropped
        chunk_nodes, chunk_edges = triples_to_graph(chunk, parsed.triples)
        nodes.extend(chunk_nodes)
        edges.extend(chunk_edges)

    stats = LLMBuildStats(
        n_chunks=len(chunks),
        n_parsed=n_parsed,
        n_skipped=n_skipped,
        n_triples=n_triples,
        n_dropped_items=n_dropped_items,
        n_cache_hits=n_cache_hits,
        n_model_calls=n_model_calls,
        model=resolved_model,
    )
    if cache_path is not None:
        _write_cache(cache_path, cache_rows)
    return dedupe_nodes(nodes), dedupe_edges(edges), stats


def build_and_push(
    llm: LLMBackend,
    chunks: Sequence[Chunk],
    store: GraphStore,
    max_triples: int = DEFAULT_MAX_TRIPLES,
    cache_path: str | Path | None = None,
    model: str | None = None,
) -> tuple[GraphCounts, LLMBuildStats]:
    """Extract the LLM graph and write it on its own, replacing the namespace.

    For the real build the caller usually wants the structural lists added to
    these before one write, so the two graphs live together; see
    docs/architecture.md.
    Any object with `generate(prompt, system)` works here, so the Vertex backend
    goes in the same slot the FakeLLM occupies in the tests.
    """
    nodes, edges, stats = extract(
        llm,
        chunks,
        max_triples,
        cache_path=cache_path,
        model=model,
    )
    return store.write_graph(nodes, edges), stats
