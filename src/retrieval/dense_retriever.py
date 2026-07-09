"""
Dense retrieval over the ChromaDB collection built by chroma_store.py.
Embeds a user question with the same embedder used at ingestion time,
queries the persistent "ckd_guidelines" collection, and returns a
structured, ranked list of results by cosine similarity.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from src.retrieval.embedder import embed_text

# ---------------------------------------------------------------------------
# Paths — mirrors chroma_store.py exactly so both modules resolve to the
# same on-disk collection regardless of the caller's working directory.
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
CHROMA_DIR = os.path.join(ROOT_DIR, "chroma_db")
COLLECTION_NAME = "ckd_guidelines"

DEFAULT_TOP_K = 10


@dataclass
class DenseResult:
    """A single dense-retrieval hit, structured for downstream RRF fusion
    and for the FastAPI response schema in Phase 3."""

    chunk_id: str
    text: str
    similarity: float  # cosine similarity, higher is better (1.0 = identical)
    distance: float  # raw ChromaDB cosine distance (1 - similarity)
    rank: int  # 0-indexed position in this retriever's ranking
    metadata: dict[str, Any] = field(default_factory=dict)


class DenseRetriever:
    """Thin wrapper around a persistent ChromaDB collection for dense
    (embedding-based) retrieval. Reuses the exact embedding function and
    collection name/path from chroma_store.py so it queries the same
    index that was built during ingestion — no re-embedding of the corpus.
    """

    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.collection: Collection = self._load_collection(collection_name)

    def _load_collection(self, collection_name: str) -> Collection:
        """Fetch the existing collection. Fails loudly rather than silently
        creating an empty one — an empty collection here almost always means
        chroma_store.py hasn't been run yet, and that's a setup bug worth
        surfacing immediately, not masking with an auto-create.
        """
        try:
            return self.chroma_client.get_collection(name=collection_name)
        except Exception as e:
            raise RuntimeError(
                f"Collection '{collection_name}' not found at {self.chroma_client}. "
                "Run chroma_store.py to build the vector store first."
            ) from e

    def query(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[DenseResult]:
        """Embed the question and return the top_k most similar chunks.

        Args:
            question: raw user question, e.g. "What is the eGFR threshold
                for stage G3b?"
            top_k: number of results to return, ranked by cosine similarity
                descending (best match first).
        """
        # Embed the query with the identical embedding function used at
        # ingestion time — query and corpus vectors must live in the same
        # embedding space for cosine similarity to be meaningful.
        query_embedding = embed_text(question)

        raw_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return self._to_dense_results(raw_results)

    @staticmethod
    def _to_dense_results(raw_results: dict[str, Any]) -> list[DenseResult]:
        """Convert ChromaDB's parallel-list response format into a clean,
        ranked list of DenseResult objects. ChromaDB already returns hits
        sorted by distance ascending (closest first), so list order here
        doubles as rank order.
        """
        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        results: list[DenseResult] = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            # ChromaDB collection was created with {"hnsw:space": "cosine"},
            # so "distance" here is cosine distance: similarity = 1 - distance.
            similarity = 1.0 - distance
            results.append(
                DenseResult(
                    chunk_id=chunk_id,
                    text=text,
                    similarity=similarity,
                    distance=distance,
                    rank=rank,
                    metadata=metadata,
                )
            )
        return results


def main() -> None:
    retriever = DenseRetriever()

    sample_query = "What is the eGFR threshold for stage G3b?"
    print(f"Query: {sample_query!r}\n")

    results = retriever.query(sample_query, top_k=DEFAULT_TOP_K)

    if not results:
        print("No results returned — is the collection populated?")
        return

    for r in results:
        print(f"[rank {r.rank}] similarity={r.similarity:.4f} chunk_id={r.chunk_id}")
        print(f"  source:  {r.metadata.get('source')}")
        print(f"  section: {r.metadata.get('section')}")
        print(f"  page:    {r.metadata.get('page_start')}-{r.metadata.get('page_end')}")
        print(f"  text:    {r.text[:200]}...")
        print()


if __name__ == "__main__":
    main()