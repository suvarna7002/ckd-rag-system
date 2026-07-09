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

DEFAULT_DENSE_WEIGHT = 0.6
DEFAULT_SPARSE_WEIGHT = 0.4
DEFAULT_RRF_K = 60  # standard damping constant; higher = less weight on top ranks

# How many candidates to pull from EACH retriever before fusing. This must
# be >= the final top_n you want returned, and ideally larger, so a chunk
# that ranks, say, #15 in dense but #2 in sparse still gets counted in the
# dense list rather than being treated as "absent" (rank -> infinity) just
# because we didn't ask dense for enough candidates.
DEFAULT_RETRIEVAL_K = 20
DEFAULT_TOP_N = 10


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
    demonstrated directly — e.g. 0.7/0.3 favors semantic recall, 0.3/0.7
    favors exact clinical terminology.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        # Allow injecting pre-built retrievers (useful for tests / reusing
        # a warm connection); default to constructing fresh ones otherwise.
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
        """Run both retrievers, fuse via RRF, and return the top_n results.

        Args:
            question: raw user question.
            top_n: number of fused results to return.
            retrieval_k: how many candidates to pull from EACH retriever
                before fusing (see DEFAULT_RETRIEVAL_K comment above).
            dense_weight / sparse_weight: per-query override of the
                weights set at construction time, for quick tuning
                experiments without re-instantiating the engine.
        """
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
        """Core RRF math:

            rrf_score(chunk) = dense_weight / (k + dense_rank)
                              + sparse_weight / (k + sparse_rank)

        Ranks are 1-indexed here (rank 1 = best), which is the standard
        RRF convention — it keeps the denominator away from a possible
        "+0" edge case and matches how RRF is described in the literature.
        A chunk absent from a given list contributes 0 for that term
        (equivalent to treating its rank as infinity).
        """
        # candidates: chunk_id -> accumulated fusion state
        candidates: dict[str, dict[str, Any]] = {}

        for dense_result in dense_results:
            # DenseResult.rank is 0-indexed; convert to 1-indexed for RRF.
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
                # Already seen via dense — just attach the sparse rank.
                candidates[sparse_result.chunk_id]["sparse_rank"] = sparse_rank_1indexed
            else:
                # Sparse-only hit — dense never surfaced this chunk at all.
                candidates[sparse_result.chunk_id] = {
                    "text": sparse_result.text,
                    "metadata": sparse_result.metadata,
                    "dense_rank": None,
                    "sparse_rank": sparse_rank_1indexed,
                }

        # Compute RRF score for every candidate chunk.
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

        # Sort by fused score, descending (best match first).
        fused_results.sort(key=lambda r: r.rrf_score, reverse=True)
        return fused_results


def main() -> None:
    # Local import so cross-encoder isn't forced unless __main__ actually runs.
    from src.retrieval.reranker import CrossEncoderReranker
 
    engine = HybridEngine(dense_weight=0.6, sparse_weight=0.4)
    reranker = CrossEncoderReranker()
 
    sample_query = "What is the eGFR threshold for stage G3b?"
    print(f"Query: {sample_query!r}")
    print(
        f"RRF weights: dense={engine.dense_weight}, sparse={engine.sparse_weight}, "
        f"k={engine.rrf_k}\n"
    )
 
    # Step 1: fuse a wide candidate pool (20)
    fused_candidates = engine.query(sample_query, top_n=20)
 
    if not fused_candidates:
        print("No fused candidates returned — check that both indexes are built.")
        return
 
    print(f"--- RRF fused candidates (top {len(fused_candidates)}) ---")
    for i, r in enumerate(fused_candidates):
        print(
            f"[rrf #{i}] rrf_score={r.rrf_score:.5f} "
            f"dense_rank={r.dense_rank} sparse_rank={r.sparse_rank} "
            f"chunk_id={r.chunk_id}"
        )
 
    # Step 2: rerank the fused pool down to the final top 5
    reranked = reranker.rerank(sample_query, fused_candidates, top_n=5)
 
    print("\n--- Cross-encoder reranked leaderboard (top 5) ---")
    for i, r in enumerate(reranked):
        # Accessing properties assuming Claude's RerankedResult structure
        # If r exposes properties differently, adjust attributes (e.g., r.hybrid_result)
        try:
            print(f"[reranked #{i}] score={r.rerank_score:.5f} chunk_id={r.chunk_id}")
            print(f"  originally: rrf_score={r.rrf_score:.5f}  dense_rank={r.dense_rank}  sparse_rank={r.sparse_rank}")
            print(f"  source:  {r.metadata.get('source')}")
            print(f"  section: {r.metadata.get('section')}")
            print(f"  text:    {r.text[:200]}...")
        except AttributeError:
            # Fallback if Claude wrapped it inside an inner object called hybrid_result
            hr = r.hybrid_result
            print(f"[reranked #{i}] score={r.rerank_score:.5f} chunk_id={hr.chunk_id}")
            print(f"  originally: rrf_score={hr.rrf_score:.5f}  dense_rank={hr.dense_rank}  sparse_rank={hr.sparse_rank}")
            print(f"  source:  {hr.metadata.get('source')}")
            print(f"  section: {hr.metadata.get('section')}")
            print(f"  text:    {hr.text[:200]}...")
        print()
 
 
if __name__ == "__main__":
    main()