"""
src/api/main.py

FastAPI application for the Clinical RAG Pipeline.
Exposes /v1/ask, /v1/documents, /v1/ingest. All pipeline components
(HybridEngine, CrossEncoderReranker, ClinicalGenerator, CitationVerifier,
ResponseBuilder) are reused as-is from their existing modules — this file
only orchestrates them and owns the web-layer concerns (routing, request
validation, config, logging, error handling).
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

#import chromadb
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.generation.citation_verifier import CitationVerifier, VerificationError
from src.generation.generator import ClinicalGenerator, ClinicalResponseSchema, GenerationError
from src.generation.response_builder import ResponseBuilder, StructuredRAGResponse
from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_loader import process_directory
from src.retrieval.bm25_index import (
    build_bm25_corpus,
    load_chunks,
    save_bm25_corpus,
)
from src.retrieval.embedder import embed_text
from src.retrieval.hybrid_engine import HybridEngine
from src.retrieval.reranker import CrossEncoderReranker

# ---------------------------------------------------------------------------
# Logging — replaces print(). Configured once at import time; uvicorn's
# own logger config governs formatting/handlers when run via `uvicorn`.
# ---------------------------------------------------------------------------
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ckd_rag_api")

# ---------------------------------------------------------------------------
# Environment-based configuration — no hardcoded paths. Every value has a
# sensible default derived from the project root, but can be overridden
# via env vars for deployment (e.g. a mounted volume in a container).
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", str(ROOT_DIR / "chroma_db")))
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "ckd_guidelines")
CHUNKS_PATH = Path(os.environ.get("CHUNKS_PATH", str(ROOT_DIR / "data" / "processed" / "chunks.json")))
BM25_INDEX_PATH = Path(os.environ.get("BM25_INDEX_PATH", str(ROOT_DIR / "data" / "processed" / "bm25_index.pkl")))
RAW_DATA_DIR = Path(os.environ.get("RAW_DATA_DIR", str(ROOT_DIR / "data" / "raw")))

DEFAULT_RETRIEVAL_K = int(os.environ.get("DEFAULT_RETRIEVAL_K", "100"))
DEFAULT_RERANK_TOP_N = int(os.environ.get("DEFAULT_RERANK_TOP_N", "5"))
DEDUP_SIMILARITY_THRESHOLD = float(os.environ.get("DEDUP_SIMILARITY_THRESHOLD", "0.95"))


# ---------------------------------------------------------------------------
# Lifespan — replaces @app.on_event("startup"). Every heavyweight resource
# (embedding model inside the retrievers, cross-encoder, Anthropic clients,
# ChromaDB connection) is constructed exactly once here and stored on
# app.state, not re-created per request.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing RAG pipeline components...")

    app.state.hybrid_engine = HybridEngine()
    app.state.reranker = CrossEncoderReranker()
    app.state.generator = ClinicalGenerator()
    app.state.verifier = CitationVerifier()
    app.state.builder = ResponseBuilder()

    app.state.chroma_collection = app.state.hybrid_engine.dense_retriever.collection

    logger.info("Pipeline initialized successfully.")
    yield
    logger.info("Shutting down RAG pipeline.")


app = FastAPI(
    title="Clinical RAG API",
    description="Backend API for Clinical Guidelines Retrieval-Augmented Generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency accessors — read pipeline objects off app.state so endpoint
# signatures stay declarative and each dependency is swappable in tests.
# ---------------------------------------------------------------------------


def get_hybrid_engine(request: Request) -> HybridEngine:
    return request.app.state.hybrid_engine


def get_reranker(request: Request) -> CrossEncoderReranker:
    return request.app.state.reranker


def get_generator(request: Request) -> ClinicalGenerator:
    return request.app.state.generator


def get_verifier(request: Request) -> CitationVerifier:
    return request.app.state.verifier


def get_builder(request: Request) -> ResponseBuilder:
    return request.app.state.builder


def get_chroma_collection(request: Request):
    return request.app.state.chroma_collection


# ---------------------------------------------------------------------------
# API Schemas
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The clinical question to answer.")
    top_n: int = Field(default=DEFAULT_RERANK_TOP_N, ge=1, le=20, description="Number of reranked chunks to feed the LLM.")
    retrieval_k: int = Field(default=DEFAULT_RETRIEVAL_K, ge=1, le=200, description="Number of chunks to fetch per retriever.")


class DocumentSummary(BaseModel):
    source: str
    sections: list[str]
    chunk_count: int


class DocumentsResponse(BaseModel):
    total_chunks: int
    unique_documents: list[str]
    documents: list[DocumentSummary]


class IngestRequest(BaseModel):
    directory: Optional[str] = Field(
        default=None, description="Path to raw PDFs to ingest. Defaults to RAW_DATA_DIR if omitted."
    )


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_added: int
    total_chunks_in_corpus: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/ask", response_model=StructuredRAGResponse)
async def ask_question(
    request: AskRequest,
    hybrid_engine: HybridEngine = Depends(get_hybrid_engine),
    reranker: CrossEncoderReranker = Depends(get_reranker),
    generator: ClinicalGenerator = Depends(get_generator),
    verifier: CitationVerifier = Depends(get_verifier),
    builder: ResponseBuilder = Depends(get_builder),
) -> StructuredRAGResponse:
    """
    Data flow: HybridEngine.query() -> CrossEncoderReranker.rerank()
    -> ClinicalGenerator.generate_answer() -> CitationVerifier.verify_response()
    -> ResponseBuilder.build_response().

    All four pipeline calls are synchronous (Anthropic SDK + sentence-
    transformers are both sync under the hood), so the whole chain runs in
    a threadpool via run_in_threadpool to avoid blocking the event loop
    while a request is in flight.
    """

    def _run_pipeline() -> StructuredRAGResponse:
        fused_candidates = hybrid_engine.query(
            question=request.query,
            top_n=request.retrieval_k,
            retrieval_k=request.retrieval_k,
        )
        if not fused_candidates:
            raise HTTPException(status_code=404, detail="No relevant context found for this query.")

        reranked_results = reranker.rerank(
            query=request.query,
            candidates=fused_candidates,
            top_n=request.top_n,
        )

        generated_response: ClinicalResponseSchema = generator.generate_answer(
            query=request.query,
            reranked_results=reranked_results,
        )
        verification_result = verifier.verify_response(
            response=generated_response,
            reranked_results=reranked_results,
        )
        return builder.build_response(
            generated_response=generated_response,
            verification_result=verification_result,
        )

    try:
        return await run_in_threadpool(_run_pipeline)
    except HTTPException:
        raise
    except GenerationError as e:
        logger.error("Generation failed for query=%r: %s", request.query, e)
        raise HTTPException(status_code=502, detail=f"Generation failed: {e}") from e
    except VerificationError as e:
        logger.error("Citation verification failed for query=%r: %s", request.query, e)
        raise HTTPException(status_code=502, detail=f"Citation verification failed: {e}") from e


@app.get("/v1/documents", response_model=DocumentsResponse)
async def list_documents(
    chroma_collection=Depends(get_chroma_collection),
) -> DocumentsResponse:
    """Reads document/section metadata directly from ChromaDB — the source
    of truth for what's currently queryable by /v1/ask."""

    def _list() -> DocumentsResponse:
        result = chroma_collection.get(include=["metadatas"])
        metadatas: list[dict[str, Any]] = result.get("metadatas") or []

        grouped: dict[str, dict[str, Any]] = {}
        for meta in metadatas:
            if not meta or "source" not in meta:
                continue
            source = meta["source"]
            entry = grouped.setdefault(source, {"sections": set(), "chunk_count": 0})
            if meta.get("section"):
                entry["sections"].add(meta["section"])
            entry["chunk_count"] += 1

        documents = [
            DocumentSummary(source=source, sections=sorted(info["sections"]), chunk_count=info["chunk_count"])
            for source, info in sorted(grouped.items())
        ]

        return DocumentsResponse(
            total_chunks=chroma_collection.count(),
            unique_documents=sorted(grouped.keys()),
            documents=documents,
        )

    try:
        return await run_in_threadpool(_list)
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        raise HTTPException(status_code=503, detail=f"Could not read document index: {e}") from e


@app.post("/v1/ingest", response_model=dict)
def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    """
    Runs the full ingestion pipeline as a background task and returns
    immediately (202-style acknowledgement) so the request doesn't time
    out on large directories. Poll /v1/documents afterward to confirm
    completion.

    Full pipeline (no placeholders):
        process_directory()  -> extract text+metadata from every PDF
        chunk_pages()        -> section-aware chunking
        chunks.json write    -> persist the full, updated chunk corpus
        embed_text() + Chroma.add()  -> vector index, with the same
                                         cosine-similarity dedup logic
                                         used by the original index build
        build_bm25_corpus() + save_bm25_corpus() -> full BM25 rebuild,
                                                     kept in sync with Chroma
    """

    def run_ingestion(target_dir: Optional[str]) -> None:
        collection = app.state.hybrid_engine.dense_retriever.collection
        try:
            raw_dir = Path(target_dir) if target_dir else RAW_DATA_DIR
            logger.info("Starting ingestion for: %s", raw_dir)

            # 1. Extract + chunk every PDF in the target directory.
            extracted_pages, pdf_count = process_directory(raw_dir)
            new_chunks = chunk_pages(extracted_pages)
            logger.info("Extracted %d pages from %d PDFs -> %d chunks.", len(extracted_pages), pdf_count, len(new_chunks))

            # 2. Merge with existing corpus and persist chunks.json.
            existing_chunks = load_chunks(CHUNKS_PATH) if CHUNKS_PATH.exists() else []
            all_chunks = existing_chunks + new_chunks
            CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CHUNKS_PATH.open("w", encoding="utf-8") as f:
                json.dump(all_chunks, f, indent=2)

            # 3. Embed + add new chunks to ChromaDB, deduplicating against
            # existing vectors the same way the original build script does.
            collection = app.state.hybrid_engine.dense_retriever.collection
            inserted = 0
            for chunk in new_chunks:
                chunk_id = chunk["metadata"]["chunk_id"]
                vector = embed_text(chunk["text"])

                is_duplicate = False
                if collection.count() > 0:
                    results = collection.query(query_embeddings=[vector], n_results=1)
                    distances = results.get("distances") or [[]]
                    if distances and distances[0]:
                        similarity = 1.0 - distances[0][0]
                        is_duplicate = similarity > DEDUP_SIMILARITY_THRESHOLD

                if not is_duplicate:
                    collection.add(
                        ids=[chunk_id],
                        embeddings=[vector],
                        metadatas=[chunk["metadata"]],
                        documents=[chunk["text"]],
                    )
                    inserted += 1

            # 4. Rebuild BM25 over the full, updated corpus (rank_bm25 has
            # no incremental-update API, so this is a full rebuild — fine
            # at this project's ~15-25 document scale).
            new_bm25_corpus = build_bm25_corpus(all_chunks)
            save_bm25_corpus(new_bm25_corpus, BM25_INDEX_PATH)

            logger.info(
                "Ingestion complete: %d PDFs, %d new chunks (%d inserted after dedup), %d total chunks.",
                pdf_count, len(new_chunks), inserted, len(all_chunks),
            )
        except Exception as e:
            logger.error("Ingestion failed: %s", e)

    background_tasks.add_task(run_ingestion, request.directory)
    return {"status": "accepted", "message": "Ingestion started in the background. Poll /v1/documents to confirm."}


# ---------------------------------------------------------------------------
# Global exception handlers — clean HTTP responses instead of raw
# tracebacks leaking to the client.
# ---------------------------------------------------------------------------


@app.exception_handler(GenerationError)
async def generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
    logger.error("GenerationError: %s", exc)
    return JSONResponse(status_code=502, content={"error": "generation_failed", "detail": str(exc)})


@app.exception_handler(VerificationError)
async def verification_error_handler(request: Request, exc: VerificationError) -> JSONResponse:
    logger.error("VerificationError: %s", exc)
    return JSONResponse(status_code=502, content={"error": "verification_failed", "detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": "An unexpected error occurred. Please try again."},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}