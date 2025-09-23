from __future__ import annotations
from typing import List, Dict, Any


def make_key(item: Dict[str, Any]) -> str:
    return f"{item.get('source_path','')}#{item.get('section_path','')}"


def rrf_fuse(dense: List[Dict[str, Any]], sparse: List[Dict[str, Any]], k: int = 60, top_k: int = 5) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    index: Dict[str, Dict[str, Any]] = {}

    for rank, it in enumerate(dense, start=1):
        key = make_key(it)
        index[key] = it
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    for rank, it in enumerate(sparse, start=1):
        key = make_key(it)
        index[key] = it
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    fused_keys = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: List[Dict[str, Any]] = []
    for key, _ in fused_keys[:top_k]:
        out.append(index[key])
    return out

