from __future__ import annotations
from math import log
from typing import List, Dict, Any
import re

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")] 


class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(d) for d in docs]
        self.doc_lens = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        # term frequencies per doc
        self.tfs: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for doc in self.docs:
            tf: Dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            self.tfs.append(tf)
            for t in tf.keys():
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.N = len(self.docs)
        # idf with log smoothing
        self.idf: Dict[str, float] = {}
        for t, d in self.df.items():
            self.idf[t] = log(1 + (self.N - d + 0.5) / (d + 0.5))

    def score(self, q_tokens: List[str], doc_index: int) -> float:
        tf = self.tfs[doc_index]
        dl = self.doc_lens[doc_index]
        score = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            idf = self.idf.get(t, 0.0)
            freq = tf[t]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
            score += idf * (freq * (self.k1 + 1)) / denom
        return score

    def topn(self, query: str, n: int) -> List[int]:
        q = tokenize(query)
        scores = [(i, self.score(q, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, s in scores[:n] if s > 0]


def bm25_search(items: List[Dict[str, Any]], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    texts = [it.get("text", "") for it in items]
    bm = BM25(texts)
    idxs = bm.topn(query, top_k * 5)
    return [items[i] for i in idxs[:top_k]]

