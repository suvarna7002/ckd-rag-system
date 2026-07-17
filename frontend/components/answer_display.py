"""
components/answer_display.py

Renders a StructuredRAGResponse (from POST /v1/ask) as three sections:
Clinical Answer, Evidence Sources, and Citation Verification. Pure
presentation — takes the already-fetched response dict, no API calls here.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_answer(response: dict[str, Any]) -> None:
    """Render the full answer block for a single /v1/ask response."""
    _render_clinical_answer(response)
    st.markdown("---")
    _render_evidence_sources(response)
    st.markdown("---")
    _render_citation_verification(response)


def _render_clinical_answer(response: dict[str, Any]) -> None:
    st.markdown("## Clinical Answer")
    answer = response.get("answer", "")
    if answer.strip().lower().startswith("i do not have enough information"):
        st.info(answer)
    else:
        st.markdown(answer)


def _render_evidence_sources(response: dict[str, Any]) -> None:
    st.markdown("## Evidence Sources")
    sources = response.get("sources_used", [])

    if not sources:
        st.caption("No sources were used for this answer.")
        return

    for i, source in enumerate(sources, start=1):
        document = source.get("document", "Unknown document")
        section = source.get("section", "Unknown section")
        chunk_id = source.get("chunk_id", "—")

        with st.expander(f"[{i}] {document}"):
            st.markdown(f"**Section:** {section}")
            st.markdown(f"**Chunk ID:** `{chunk_id}`")


def _render_citation_verification(response: dict[str, Any]) -> None:
    st.markdown("## Citation Verification")

    confidence = response.get("confidence", 0.0)
    verified = response.get("verified_citations", 0)
    total = response.get("total_citations", 0)
    unsupported = response.get("unsupported_claims", [])

    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{confidence * 100:.0f}%")
    col2.metric("Verified Citations", f"{verified}/{total}" if total else "—")
    col3.metric(
        "Status",
        "✅ Verified" if total and verified == total else ("⚠️ Partial" if total else "—"),
    )

    if unsupported:
        st.warning(f"{len(unsupported)} claim(s) could not be verified against retrieved evidence:")
        for claim in unsupported:
            st.markdown(f"- ❌ {claim}")
    elif total > 0:
        st.success("All cited claims were verified against retrieved evidence.")