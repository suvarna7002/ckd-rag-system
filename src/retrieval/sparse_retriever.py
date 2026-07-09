"""
sparse_retriever.py

Sparse (BM25) retrieval over the pre-built index from bm25_index.py.
Loads the serialized BM25Corpus, tokenizes a live query with the exact
same function used at index-build time, and returns a structured,
ranked list of results that mirrors DenseResult's shape — so the
Phase 2 Step 3 RRF fusion layer can treat both retrievers uniformly.

Usage:
    python src/retrieval/sparse_retriever.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.retrieval.bm25_index import BM25Corpus, load_bm25_corpus, tokenize

# ---------------------------------------------------------------------------
# Paths — resolved the same way as chroma_store.py / dense_retriever.py so
# this module works regardless of the caller's working directory.
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
BM25_INDEX_PATH = Path(os.path.join(ROOT_DIR, "data", "processed", "bm25_index.pkl"))

DEFAULT_TOP_K = 10


@dataclass
class SparseResult:
    """A single BM25 hit. Field names/types intentionally mirror
    DenseResult (chunk_id, text, score/similarity, rank, metadata) so
    the fusion layer can zip dense + sparse rankings without special-
    casing either side."""

    chunk_id: str
    text: str
    score: float  # raw BM25 score, higher is better (no fixed upper bound)
    rank: int  # 0-indexed position in this retriever's ranking
    metadata: dict[str, Any] = field(default_factory=dict)


class SparseRetriever:
    """Thin wrapper around a pre-built BM25Corpus for sparse (keyword)
    retrieval. Loads the index once at construction time — no rebuilding
    per query — and reuses bm25_index.py's tokenize() so the query is
    preprocessed identically to how the corpus was tokenized at ingestion.

    This is the keyword-precision anchor for terms dense embeddings tend
    to blur, e.g. "stage G3b" vs "stage G3a", or "uACR" vs "uPCR".
    """

    def __init__(self, index_path: Path = BM25_INDEX_PATH) -> None:
        # The pickled BM25Corpus already carries chunk_ids, metadatas, and
        # texts in lockstep with the BM25Okapi index (see bm25_index.py),
        # so there's no separate chunks.json load here — re-reading that
        # file independently would risk the two data sources drifting out
        # of positional sync if chunks.json is ever regenerated without
        # rebuilding the BM25 index to match.
        self.corpus: BM25Corpus = load_bm25_corpus(index_path)

    def query(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[SparseResult]:
        """Tokenize the question and return the top_k highest-scoring
        chunks by BM25 score, descending.

        Args:
            question: raw user question, e.g. "What is the eGFR threshold
                for stage G3b?"
            top_k: number of results to return.
        """
        # CRITICAL: use the identical tokenize() function from bm25_index.py
        # that built the corpus. Any drift here (different lowercasing,
        # different punctuation handling) silently breaks keyword matching
        # without raising an error — scores would just come back near zero.
        query_tokens = tokenize(question)
        scores = self.corpus.bm25.get_scores(query_tokens)

        # Rank all chunk indices by score, descending, then take top_k.
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return self._to_sparse_results(ranked_indices, scores)

    def _to_sparse_results(
        self, ranked_indices: list[int], scores: Any
    ) -> list[SparseResult]:
        """Map ranked BM25 corpus positions back to chunk_id/text/metadata
        using the aligned arrays stored in BM25Corpus."""
        results: list[SparseResult] = []
        for rank, idx in enumerate(ranked_indices):
            results.append(
                SparseResult(
                    chunk_id=self.corpus.chunk_ids[idx],
                    text=self.corpus.texts[idx],
                    score=float(scores[idx]),
                    rank=rank,
                    metadata=self.corpus.metadatas[idx],
                )
            )
        return results


def main() -> None:
    retriever = SparseRetriever()

    sample_query = "What is the eGFR threshold for stage G3b?"
    print(f"Query: {sample_query!r}\n")

    results = retriever.query(sample_query, top_k=DEFAULT_TOP_K)

    if not results:
        print("No results returned — is the BM25 index built and non-empty?")
        return

    for r in results:
        print(f"[rank {r.rank}] score={r.score:.4f} chunk_id={r.chunk_id}")
        print(f"  source:  {r.metadata.get('source')}")
        print(f"  section: {r.metadata.get('section')}")
        print(f"  page:    {r.metadata.get('page_start')}-{r.metadata.get('page_end')}")
        print(f"  text:    {r.text[:200]}...")
        print()


if __name__ == "__main__":
    main()