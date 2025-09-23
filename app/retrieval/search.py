from __future__ import annotations
from typing import Dict, List, Optional
import re

from app.retrieval.filtering import prefilter_by_allowed_roles

_token_re = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _token_re.findall(text or "")]


def score(text: str, q_tokens: List[str]) -> int:
    tokens = tokenize(text)
    if not tokens:
        return 0
    q_set = set(q_tokens)
    return sum(1 for t in tokens if t in q_set)


def search_index(index: List[Dict], query: str, role: str, top_k: int = 5, pdp: Optional[object] = None) -> List[Dict]:
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    # prefilter by allowed_roles
    candidates = prefilter_by_allowed_roles(index, role)

    # optional PDP post-check for each candidate
    out: List[Dict] = []
    for it in candidates:
        if pdp is not None:
            decision = pdp.evaluate(
                subject={"role": role},
                resource={
                    "owner_dept": it.get("owner_dept"),
                    "doc_type": it.get("doc_type"),
                    "sensitivity": it.get("sensitivity"),
                },
                action="retrieve_chunk",
            )
            if decision.effect != "permit":
                continue
        s = score(it.get("text", ""), q_tokens)
        if s > 0:
            out.append({**it, "_score": s})
    out.sort(key=lambda x: x.get("_score", 0), reverse=True)
    return out[: max(1, int(top_k))]

