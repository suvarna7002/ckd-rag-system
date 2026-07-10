"""
hybrid_engine.py

Reciprocal Rank Fusion (RRF) layer that combines DenseRetriever (cosine
similarity over embeddings) and SparseRetriever (BM25 keyword matching)
into a single ranked list.

RRF fuses by RANK, not raw score, because cosine similarity (bounded
[0,1]) and BM25 score (unbounded, corpus-dependent) live on incompatible
scales — you cannot weighted-average them directly. Rank position is the
only thing both retrievers can be fairly compared on.

Usage:
    python src/retrieval/hybrid_engine.py
"""

from __future__ import annotations
from src.retrieval.bm25_index import BM25Corpus
from dataclasses import dataclass
from typing import Any

from src.retrieval.dense_retriever import DenseRetriever, DenseResult
from src.retrieval.sparse_retriever import SparseRetriever, SparseResult

# Balanced production weights
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_SPARSE_WEIGHT = 0.5
DEFAULT_RRF_K = 60  # standard damping constant

# Production settings: Cast a wide net (100) to capture tabular exact matches,
# and pass all 100 to the cross-encoder to guarantee a successful rescue.
DEFAULT_RETRIEVAL_K = 100
DEFAULT_TOP_N = 100


@dataclass
class HybridResult:
    """A single fused hit. Carries both retrievers' original ranks
    (1-indexed, None if the chunk didn't appear in that list) for
    transparency — this is what lets you show "this chunk surfaced
    because of keyword precision" vs "because of semantic similarity"
    in the metadata panel."""

    chunk_id: str
    text: str
    rrf_score: float
    dense_rank: int | None  # 1-indexed rank in the dense list, None if absent
    sparse_rank: int | None  # 1-indexed rank in the sparse list, None if absent
    metadata: dict[str, Any]


class HybridEngine:
    """Combines DenseRetriever and SparseRetriever via Reciprocal Rank
    Fusion. Weights are configurable at construction time (and overridable
    per-query) so dense/sparse contribution can be tuned and the tradeoff
    demonstrated directly — e.g. 0.5/0.5 provides balanced retrieval.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or SparseRetriever()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def query(
        self,
        question: str,
        top_n: int = DEFAULT_TOP_N,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> list[HybridResult]:
        """Run both retrievers, fuse via RRF, and return the top_n results."""
        weight_dense = dense_weight if dense_weight is not None else self.dense_weight
        weight_sparse = sparse_weight if sparse_weight is not None else self.sparse_weight

        dense_results = self.dense_retriever.query(question, top_k=retrieval_k)
        sparse_results = self.sparse_retriever.query(question, top_k=retrieval_k)

        fused = self._fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=weight_dense,
            sparse_weight=weight_sparse,
        )

        return fused[:top_n]

    def _fuse(
        self,
        dense_results: list[DenseResult],
        sparse_results: list[SparseResult],
        dense_weight: float,
        sparse_weight: float,
    ) -> list[HybridResult]:
        """Core RRF math."""
        candidates: dict[str, dict[str, Any]] = {}

        for dense_result in dense_results:
            dense_rank_1indexed = dense_result.rank + 1
            candidates[dense_result.chunk_id] = {
                "text": dense_result.text,
                "metadata": dense_result.metadata,
                "dense_rank": dense_rank_1indexed,
                "sparse_rank": None,
            }

        for sparse_result in sparse_results:
            sparse_rank_1indexed = sparse_result.rank + 1
            if sparse_result.chunk_id in candidates:
                candidates[sparse_result.chunk_id]["sparse_rank"] = sparse_rank_1indexed
            else:
                candidates[sparse_result.chunk_id] = {
                    "text": sparse_result.text,
                    "metadata": sparse_result.metadata,
                    "dense_rank": None,
                    "sparse_rank": sparse_rank_1indexed,
                }

        fused_results: list[HybridResult] = []
        for chunk_id, info in candidates.items():
            dense_term = (
                dense_weight / (self.rrf_k + info["dense_rank"])
                if info["dense_rank"] is not None
                else 0.0
            )
            sparse_term = (
                sparse_weight / (self.rrf_k + info["sparse_rank"])
                if info["sparse_rank"] is not None
                else 0.0
            )
            rrf_score = dense_term + sparse_term

            fused_results.append(
                HybridResult(
                    chunk_id=chunk_id,
                    text=info["text"],
                    rrf_score=rrf_score,
                    dense_rank=info["dense_rank"],
                    sparse_rank=info["sparse_rank"],
                    metadata=info["metadata"],
                )
            )

        fused_results.sort(key=lambda r: r.rrf_score, reverse=True)
        return fused_results


def main() -> None:
    from src.retrieval.reranker import CrossEncoderReranker
 
    engine = HybridEngine()
    reranker = CrossEncoderReranker()
 
    sample_query = "What is the eGFR threshold for stage G3b?"
    print(f"Query: {sample_query!r}")
    print(f"RRF weights: dense={engine.dense_weight}, sparse={engine.sparse_weight}\n")
 
    # Uses the robust 100/100 default parameters automatically
    fused_candidates = engine.query(sample_query)
    if not fused_candidates:
        print("No fused candidates returned.")
        return
 
    print(f"--- RRF fused candidates (top {len(fused_candidates)}) ---")
    for i, r in enumerate(fused_candidates[:50]): # print top 50 for readability
        print(f"[rrf #{i}] rrf_score={r.rrf_score:.5f} dense_rank={r.dense_rank} sparse_rank={r.sparse_rank} chunk_id={r.chunk_id}")
 
    # Step 2: Rerank down to final 5 results
    reranked = reranker.rerank(sample_query, fused_candidates, top_n=5)
 
    print("\n--- Cross-encoder reranked leaderboard (top 5) ---")
    for i, r in enumerate(reranked):
        try:
            print(f"[reranked #{i}] score={r.rerank_score:.5f} chunk_id={r.chunk_id}")
            print(f"  originally: rrf_score={r.rrf_score:.5f}  dense_rank={r.dense_rank}  sparse_rank={r.sparse_rank}")
            print(f"  source:  {r.metadata.get('source')}")
            print(f"  section: {r.metadata.get('section')}")
            print(f"  text:    {r.text[:200]}...")
        except AttributeError:
            hr = r.hybrid_result
            print(f"[reranked #{i}] score={r.rerank_score:.5f} chunk_id={hr.chunk_id}")
            print(f"  originally: rrf_score={hr.rrf_score:.5f}  dense_rank={hr.dense_rank}  sparse_rank={hr.sparse_rank}")
            print(f"  source:  {hr.metadata.get('source')}")
            print(f"  section: {hr.metadata.get('section')}")
            print(f"  text:    {hr.text[:200]}...")
        print()
 
 
if __name__ == "__main__":
    main()