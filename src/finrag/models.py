"""Core data contracts shared by ingest, policy, retrieval, and eval.

Everything downstream depends on these shapes, so they are frozen and boring:
plain dataclasses, JSON-serialisable field types, no behaviour beyond
serialisation helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

ROLES: tuple[str, ...] = (
    "finance",
    "marketing",
    "hr",
    "engineering",
    "c_level",
    "employee",
)

DOC_TYPES: tuple[str, ...] = ("md", "csv_row", "aggregate")


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of the corpus.

    `text` is the searchable body (breadcrumb prepended by the chunker).
    `payload` holds structured extras such as the full HR row, which is served
    only when the policy engine permits `view_row`; it is never searchable.
    `allowed_roles` is computed at ingest by evaluating the policy engine, so
    it is a cache of decisions, not a parallel permission table.
    """

    chunk_id: str
    doc_id: str
    section_path: str
    section_slug: str
    text: str
    owner_dept: str
    doc_type: str
    sensitivity: str
    allowed_roles: tuple[str, ...] = ()
    quarter: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON view; `allowed_roles` becomes a list."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section_path": self.section_path,
            "section_slug": self.section_slug,
            "text": self.text,
            "owner_dept": self.owner_dept,
            "doc_type": self.doc_type,
            "sensitivity": self.sensitivity,
            "allowed_roles": list(self.allowed_roles),
            "quarter": self.quarter,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            section_path=data["section_path"],
            section_slug=data["section_slug"],
            text=data["text"],
            owner_dept=data["owner_dept"],
            doc_type=data["doc_type"],
            sensitivity=data["sensitivity"],
            allowed_roles=tuple(data.get("allowed_roles") or ()),
            quarter=data.get("quarter"),
            payload=dict(data.get("payload") or {}),
        )

    def to_json(self) -> str:
        """One chunks.jsonl line: compact, no newlines inside."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> Chunk:
        return cls.from_dict(json.loads(line))


@dataclass(frozen=True)
class Hit:
    """One ranked retrieval result.

    Carries every attribute the policy replay and the doc/section metrics need,
    so no consumer has to look chunks up again. `debug` holds per-signal scores,
    fusion ranks, or graph paths, depending on the retriever.
    """

    chunk_id: str
    doc_id: str
    section_slug: str
    rank: int
    score: float
    retriever: str
    owner_dept: str
    doc_type: str
    sensitivity: str
    allowed_roles: tuple[str, ...] = ()
    text: str = ""
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "section_slug": self.section_slug,
            "rank": self.rank,
            "score": self.score,
            "retriever": self.retriever,
            "owner_dept": self.owner_dept,
            "doc_type": self.doc_type,
            "sensitivity": self.sensitivity,
            "allowed_roles": list(self.allowed_roles),
            "text": self.text,
            "debug": dict(self.debug),
        }


@dataclass(frozen=True)
class ResourceAttrs:
    """The resource side of an ABAC decision."""

    owner_dept: str
    doc_type: str
    sensitivity: str


@dataclass(frozen=True)
class Decision:
    """Outcome of one policy evaluation.

    `effect` is "permit" or "deny"; `rule_name` names the matching rule, or is
    None when nothing matched and the default deny applied.
    """

    effect: str
    rule_name: str | None = None

    @property
    def permitted(self) -> bool:
        return self.effect == "permit"
