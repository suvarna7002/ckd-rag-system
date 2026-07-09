"""
reranker.py

Cross-encoder reranking layer — the final Phase 2 step. Takes the fused
RRF candidates from HybridEngine and rescoring them with a cross-encoder
that jointly attends over (query, chunk) pairs, rather than comparing
independently-computed vectors like the dense retriever does.

Usage:
    from reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, hybrid_results, top_n=5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sentence_transformers import CrossEncoder

# TYPE_CHECKING-only import avoids a circular import at runtime: hybrid_engine's
# main() imports CrossEncoderReranker from this module, so this module must NOT
# import anything from hybrid_engine at import time. The HybridResult type is
# only needed for static type checkers/IDEs, not at runtime, and `from __future__
# import annotations` (above) makes all annotations lazy strings anyway, so this
# is safe.
if TYPE_CHECKING:
    from hybrid_engine import HybridResult

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"
DEFAULT_TOP_N = 5


@dataclass
class RerankedResult:
    """A single reranked hit. Keeps the original RRF signal (rrf_score,
    dense_rank, sparse_rank) alongside the new cross-encoder score so you
    can compare "where fusion put this" vs "where the cross-encoder put
    this" directly — that comparison is the whole point of the demo."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    rerank_score: float  # cross-encoder relevance score, higher is better
    rank: int  # 0-indexed final position after reranking
    rrf_score: float  # carried over from the HybridResult that fed this
    dense_rank: int | None  # carried over from the HybridResult that fed this
    sparse_rank: int | None  # carried over from the HybridResult that fed this


class CrossEncoderReranker:
    """Wraps sentence_transformers.CrossEncoder to rescore RRF candidates
    by joint query-document relevance instead of independent rank
    position. `BAAI/bge-reranker-base` is a solid default: lightweight
    enough to run on CPU in a demo, strong enough to be a legitimate
    quality signal (unlike a toy MiniLM reranker).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        # Loaded once at construction — reused across queries. Loading a
        # cross-encoder per-query would make the API painfully slow.
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list["HybridResult"],
        top_n: int = DEFAULT_TOP_N,
    ) -> list[RerankedResult]:
        """Score every candidate against the query with the cross-encoder
        and return the top_n, re-sorted by cross-encoder relevance.

        Args:
            query: the raw user question.
            candidates: HybridResult objects from HybridEngine.query()
                (typically the top ~20 fused candidates, not just top 5 —
                reranking only helps if it has a wide enough pool to
                reorder within).
            top_n: number of results to keep after reranking.
        """
        if not candidates:
            return []

        # Cross-encoders score (query, document) PAIRS jointly — this is
        # the key difference from dense retrieval, which embeds query and
        # document SEPARATELY and compares vectors after the fact. Joint
        # attention lets the model actually check whether the specific
        # chunk answers the specific question, not just whether they're
        # "semantically nearby" in embedding space.
        pairs = [(query, candidate.text) for candidate in candidates]
        raw_scores = self.model.predict(pairs)

        # Zip candidates with their cross-encoder scores, sort descending.
        scored = list(zip(candidates, raw_scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)

        reranked_results: list[RerankedResult] = []
        for rank, (candidate, score) in enumerate(scored[:top_n]):
            reranked_results.append(
                RerankedResult(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    metadata=candidate.metadata,
                    rerank_score=float(score),
                    rank=rank,
                    rrf_score=candidate.rrf_score,
                    dense_rank=candidate.dense_rank,
                    sparse_rank=candidate.sparse_rank,
                )
            )
        return reranked_results