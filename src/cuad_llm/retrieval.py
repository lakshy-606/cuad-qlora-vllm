"""Hybrid BM25 + dense retrieval over one contract's passages, fused with reciprocal rank fusion.

Used by the RAG baseline: instead of asking the model about every chunk, retrieve the few passages most
likely to hold a clause category and ask once per (contract, category). Needs the `rag` dependency group.
"""

from __future__ import annotations

import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from cuad_llm.chunking import Chunk

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    def __init__(self, passages: list[str], passage_vecs: np.ndarray):
        """`passage_vecs` are the passages' embeddings, one row each, in the same order."""
        self.n = len(passages)
        self.bm25 = BM25Okapi([tokenize(p) or [""] for p in passages])
        vecs = np.ascontiguousarray(passage_vecs, dtype=np.float32)
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query: str, query_vec: np.ndarray, k: int, rrf_k: int = 60) -> list[int]:
        """Indices of the top-k passages by reciprocal rank fusion of BM25 and cosine rankings."""
        bm25_scores = self.bm25.get_scores(tokenize(query))
        # Passages with no query term get no BM25 credit; their tie order is arbitrary.
        bm25_rank = [i for i in np.argsort(-bm25_scores, kind="stable") if bm25_scores[i] > 0]
        q = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        _, dense_rank = self.index.search(q, self.n)
        scores = np.zeros(self.n)
        for ranking in (bm25_rank, dense_rank[0]):
            scores[ranking] += 1.0 / (rrf_k + np.arange(1, len(ranking) + 1))
        return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


def merge_ranges(chunks: list[Chunk]) -> list[tuple[int, int]]:
    """Merge overlapping or touching character ranges, in document order."""
    merged: list[list[int]] = []
    for c in sorted(chunks, key=lambda c: c.start):
        if merged and c.start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], c.end)
        else:
            merged.append([c.start, c.end])
    return [(s, e) for s, e in merged]
