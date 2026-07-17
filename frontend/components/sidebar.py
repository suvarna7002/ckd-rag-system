"""
components/sidebar.py

Renders the sidebar: knowledge-base statistics pulled from GET
/v1/documents, plus a manual refresh control. Statistics are cached in
st.session_state so every rerun of the main page doesn't re-hit the
backend — only an explicit refresh (or app startup) does.
"""

from __future__ import annotations

import streamlit as st

from services.api_client import APIError, get_documents


def _load_document_stats() -> None:
    """Fetch fresh stats from the backend and store them in session state.
    Any failure is stored as an error message rather than raised, so a
    backend outage doesn't crash sidebar rendering."""
    try:
        stats = get_documents()
        st.session_state["doc_stats"] = stats
        st.session_state["doc_stats_error"] = None
    except APIError as e:
        st.session_state["doc_stats_error"] = str(e)


def render_sidebar() -> None:
    """Render the full sidebar: title, knowledge-base stats, pipeline
    description, and a refresh button."""
    with st.sidebar:
        st.markdown("### System Information")

        # Load stats once per session on first render; subsequent renders
        # reuse the cached value until the user clicks Refresh.
        if "doc_stats" not in st.session_state:
            _load_document_stats()

        stats = st.session_state.get("doc_stats")
        error = st.session_state.get("doc_stats_error")

        if error:
            st.warning(f"Could not load knowledge base stats: {error}")
        elif stats:
            st.markdown("**Knowledge Base**")
            st.metric("Indexed documents", len(stats.get("unique_documents", [])))
            st.metric("Total chunks", stats.get("total_chunks", 0))

        st.markdown("---")
        st.markdown("**Pipeline**")
        st.markdown(
            "- Hybrid Retrieval (BM25 + dense)\n"
            "- Reciprocal Rank Fusion\n"
            "- Cross-Encoder Reranking\n"
            "- LLM Generation\n"
            "- Citation Verification"
        )

        st.markdown("---")
        if st.button("Refresh Document Statistics", use_container_width=True):
            _load_document_stats()
            st.rerun()