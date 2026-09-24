import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")

from cuad_llm.chunking import Chunk
from cuad_llm.retrieval import HybridRetriever, merge_ranges

PASSAGES = [
    "This agreement shall be governed by the laws of the State of New York.",
    "The licensee shall pay royalties quarterly.",
    "Neither party may assign this agreement without prior written consent.",
    "The term of this agreement is five years.",
]


def test_hybrid_search_ranks_lexical_and_dense_matches_first():
    vecs = np.eye(4, dtype=np.float32)
    r = HybridRetriever(PASSAGES, vecs)
    # Lexical and dense signals agree on passage 2.
    assert r.search("assign consent", np.array([0, 0, 1, 0]), k=1) == [2]
    # With no lexical overlap, the dense signal decides.
    assert r.search("zzz", np.array([0, 0, 0, 1]), k=2)[0] == 3


def test_merge_ranges():
    chunks = [Chunk(50, 80, ""), Chunk(0, 30, ""), Chunk(20, 40, ""), Chunk(80, 90, "")]
    assert merge_ranges(chunks) == [(0, 40), (50, 90)]
