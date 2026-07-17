"""
api_client.py

All communication with the FastAPI backend lives here, isolated from
Streamlit UI code. Each function maps to one backend endpoint and returns
either parsed JSON (dict) or raises a clear, catchable exception — UI code
decides how to display failures, this module never calls st.error() etc.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

# Backend URL is never hardcoded — overridable via env var so the same
# frontend build can point at local, staging, or production backends
# without a code change.
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

DEFAULT_TIMEOUT_SECONDS = 60


class APIError(Exception):
    """Raised for any failure talking to the backend — network error,
    non-2xx response, or malformed JSON. UI code catches this one
    exception type rather than handling requests.exceptions.* directly."""


def ask_question(query: str, top_n: int = 5, retrieval_k: int = 100) -> dict[str, Any]:
    """POST /v1/ask — runs the full RAG pipeline for a clinical question.

    Returns the StructuredRAGResponse payload as a dict:
        answer, sources_used, confidence, retrieved_documents,
        verified_citations, total_citations, unsupported_claims,
        retriever_scores.
    """
    payload = {"query": query, "top_n": top_n, "retrieval_k": retrieval_k}
    try:
        response = requests.post(
            f"{FASTAPI_URL}/v1/ask", json=payload, timeout=DEFAULT_TIMEOUT_SECONDS
        )
    except requests.exceptions.ConnectionError as e:
        raise APIError(
            f"Could not reach the backend at {FASTAPI_URL}. Is the FastAPI server running?"
        ) from e
    except requests.exceptions.Timeout as e:
        raise APIError("The request timed out. The backend may be under heavy load.") from e

    if response.status_code != 200:
        detail = _extract_error_detail(response)
        raise APIError(f"Backend returned {response.status_code}: {detail}")

    return response.json()


def get_documents() -> dict[str, Any]:
    """GET /v1/documents — returns indexed document/chunk statistics.

    Returns a dict with: total_chunks, unique_documents, documents
    (list of {source, sections, chunk_count}).
    """
    try:
        response = requests.get(f"{FASTAPI_URL}/v1/documents", timeout=DEFAULT_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as e:
        raise APIError(
            f"Could not reach the backend at {FASTAPI_URL}. Is the FastAPI server running?"
        ) from e
    except requests.exceptions.Timeout as e:
        raise APIError("The request timed out.") from e

    if response.status_code != 200:
        detail = _extract_error_detail(response)
        raise APIError(f"Backend returned {response.status_code}: {detail}")

    return response.json()


def trigger_ingest(directory: Optional[str] = None) -> dict[str, Any]:
    """POST /v1/ingest — kicks off background ingestion of new PDFs.

    Returns immediately with an acknowledgement (ingestion runs async on
    the backend); does not wait for ingestion to finish.
    """
    payload = {"directory": directory} if directory else {}
    try:
        response = requests.post(
            f"{FASTAPI_URL}/v1/ingest", json=payload, timeout=DEFAULT_TIMEOUT_SECONDS
        )
    except requests.exceptions.ConnectionError as e:
        raise APIError(
            f"Could not reach the backend at {FASTAPI_URL}. Is the FastAPI server running?"
        ) from e

    if response.status_code != 200:
        detail = _extract_error_detail(response)
        raise APIError(f"Backend returned {response.status_code}: {detail}")

    return response.json()


def _extract_error_detail(response: requests.Response) -> str:
    """Pull a human-readable detail message out of a FastAPI error
    response, falling back to raw text if the body isn't the expected
    {"detail": ...} or {"error": ..., "detail": ...} shape."""
    try:
        body = response.json()
        return body.get("detail") or body.get("error") or str(body)
    except ValueError:
        return response.text or "No error detail provided."