"""
components/documents.py

Renders the "Knowledge Base" explorer view: a table of all indexed
documents plus expandable per-document section lists. Reuses whatever is
already cached in st.session_state["doc_stats"] (populated by the
sidebar) rather than re-fetching, so opening this page doesn't trigger an
extra API call.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from services.api_client import APIError, get_documents


def render_documents_page() -> None:
    st.markdown("# Knowledge Base")
    st.caption("All documents currently indexed for retrieval.")

    if st.button("Reload from backend"):
        _refresh_stats()

    stats = st.session_state.get("doc_stats")
    error = st.session_state.get("doc_stats_error")

    if error and not stats:
        st.error(f"Could not load document index: {error}")
        return

    if not stats:
        _refresh_stats()
        stats = st.session_state.get("doc_stats")
        if not stats:
            st.error("No document data available.")
            return

    col1, col2 = st.columns(2)
    col1.metric("Indexed documents", len(stats.get("unique_documents", [])))
    col2.metric("Total chunks", stats.get("total_chunks", 0))

    st.markdown("---")

    documents: list[dict[str, Any]] = stats.get("documents", [])
    if not documents:
        st.caption("No documents indexed yet.")
        return

    # Summary table: one row per document.
    table_rows = [
        {
            "Document": doc.get("source", "Unknown"),
            "Chunks": doc.get("chunk_count", 0),
            "Sections": len(doc.get("sections", [])),
        }
        for doc in documents
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.markdown("### Section detail")
    for doc in documents:
        source = doc.get("source", "Unknown document")
        sections = doc.get("sections", [])
        chunk_count = doc.get("chunk_count", 0)

        with st.expander(f"{source}  ·  {chunk_count} chunks"):
            if sections:
                for section in sections:
                    st.markdown(f"- {section}")
            else:
                st.caption("No section labels detected for this document.")


def _refresh_stats() -> None:
    try:
        st.session_state["doc_stats"] = get_documents()
        st.session_state["doc_stats_error"] = None
    except APIError as e:
        st.session_state["doc_stats_error"] = str(e)