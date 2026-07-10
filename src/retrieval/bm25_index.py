"""
bm25_index.py

Builds a BM25 (Okapi) sparse retrieval index over the same chunk set used
in the ChromaDB dense index, so hybrid retrieval can fuse both rankings
over identical chunk_ids.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Config — adjust paths if your project layout differs
# ---------------------------------------------------------------------------
CHUNKS_PATH = Path("data/processed/chunks.json")
BM25_INDEX_PATH = Path("data/processed/bm25_index.pkl")

# Clinical tokenization: keep alphanumeric tokens together so codes like
# "g3a", "uacr", "egfr", "kdoqi" survive as single tokens instead of being
# split apart by punctuation. This regex grabs runs of letters/digits and
# drops everything else (commas, parens, slashes, etc.).
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class BM25Corpus:
    """Container bundling the BM25 index with the chunk metadata needed
    to map BM25 results back to the same chunk_ids used in ChromaDB."""

    bm25: BM25Okapi
    chunk_ids: list[str]
    metadatas: list[dict[str, Any]]
    texts: list[str]


# ---------------------------------------------------------------------------
# Namespace Interceptor for Pickle
# ---------------------------------------------------------------------------
class BM25CustomUnpickler(pickle.Unpickler):
    """Custom unpickler to transparently resolve the '__main__' namespace trap.
    
    When running this file directly with `python -m src.retrieval.bm25_index`, 
    pickle serializes BM25Corpus under the namespace '__main__.BM25Corpus'.
    This interceptor ensures that downstream scripts running from separate execution 
    contexts (like test_pipeline.py) redirect lookups cleanly to this module.
    """
    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "BM25Corpus":
            return BM25Corpus
        return super().find_class(module, name)


def load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Load the chunk records produced by the chunking pipeline.

    Each record is expected to have the shape:
        {"text": str, "metadata": {"chunk_id": str, ...}}
    """
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Could not find chunks file at {chunks_path.resolve()}. "
            "Run the chunking pipeline first."
        )

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks: list[dict[str, Any]] = json.load(f)

    if not chunks:
        raise ValueError(f"{chunks_path} loaded but contained zero chunks.")

    return chunks


def tokenize(text: str) -> list[str]:
    """Tokenize a chunk of text for BM25.

    Steps:
      1. Lowercase (BM25 is case-sensitive by default; we want
         "eGFR" and "egfr" to match the same token).
      2. Extract alphanumeric runs via regex — this strips punctuation
         (commas, periods, parens, slashes) while keeping clinical
         shorthand like "g3a" or "uacr" intact as single tokens, since
         they contain no internal punctuation once lowercased.
      3. Return the resulting token list (no stemming/stopword removal —
         clinical guideline text is dense with exact terminology we don't
         want to blur; BM25's own term-frequency weighting handles common
         words reasonably well at this corpus size).
    """
    lowered = text.lower()
    tokens = TOKEN_PATTERN.findall(lowered)
    return tokens


def build_bm25_corpus(chunks: list[dict[str, Any]]) -> BM25Corpus:
    """Extract text/metadata from chunks, tokenize, and build the BM25 index.

    Preserves the exact iteration order of `chunks`, so index position i
    in the BM25 corpus corresponds to chunk_ids[i] / metadatas[i] / texts[i].
    This positional alignment is what lets you map a BM25 score back to
    the correct chunk_id after scoring/ranking.
    """
    chunk_ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    texts: list[str] = []
    tokenized_corpus: list[list[str]] = []

    for chunk in chunks:
        metadata = chunk["metadata"]
        chunk_id = metadata["chunk_id"]
        text = chunk["text"]

        chunk_ids.append(chunk_id)
        metadatas.append(metadata)
        texts.append(text)
        tokenized_corpus.append(tokenize(text))

    bm25 = BM25Okapi(tokenized_corpus)

    return BM25Corpus(
        bm25=bm25,
        chunk_ids=chunk_ids,
        metadatas=metadatas,
        texts=texts,
    )


def save_bm25_corpus(corpus: BM25Corpus, output_path: Path) -> None:
    """Serialize the BM25 index + aligned chunk_ids/metadata/texts to disk.

    We pickle the whole BM25Corpus object (not just the BM25Okapi instance)
    so that chunk_ids and metadatas travel with the index — without this,
    a BM25 score by itself is just a position in a list with no way to
    trace it back to a chunk_id.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(corpus, f)


def load_bm25_corpus(index_path: Path) -> BM25Corpus:
    """Load a previously built BM25Corpus from disk (used at query time
    by the hybrid retrieval layer, so BM25 doesn't rebuild every call)."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"No BM25 index found at {index_path.resolve()}. "
            "Run this script to build one first."
        )
    with index_path.open("rb") as f:
        # Use custom unpickler instead of standard pickle.load to prevent namespace errors
        corpus: BM25Corpus = BM25CustomUnpickler(f).load()
    return corpus


def query_bm25(corpus: BM25Corpus, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Convenience query helper — tokenizes the query the same way the
    corpus was tokenized, scores all chunks, and returns the top_k results
    with their chunk_id, score, and metadata (used later by the RRF
    fusion layer, and useful here for a quick sanity check via __main__).
    """
    query_tokens = tokenize(query)
    scores = corpus.bm25.get_scores(query_tokens)

    # Pair each score with its index, sort descending, take top_k
    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices):
        results.append(
            {
                "rank": rank,
                "chunk_id": corpus.chunk_ids[idx],
                "score": float(scores[idx]),
                "metadata": corpus.metadatas[idx],
                "text_preview": corpus.texts[idx][:200],
            }
        )
    return results


def main() -> None:
    print(f"Loading chunks from {CHUNKS_PATH} ...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks.")

    print("Tokenizing and building BM25Okapi index ...")
    corpus = build_bm25_corpus(chunks)

    print(f"Saving BM25 index to {BM25_INDEX_PATH} ...")
    save_bm25_corpus(corpus, BM25_INDEX_PATH)
    print("Done.")

    # Quick sanity check so a bad build fails loudly instead of silently
    sample_query = "eGFR threshold for stage G3b"
    print(f"\nSanity check — query: {sample_query!r}")
    for result in query_bm25(corpus, sample_query, top_k=3)[:3]:
        print(
            f"  rank={result['rank']} score={result['score']:.3f} "
            f"chunk_id={result['chunk_id']} "
            f"source={result['metadata'].get('source')}"
        )


if __name__ == "__main__":
    main()