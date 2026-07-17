"""
app.py

Entry point for the Clinical CKD RAG Assistant frontend. Two views,
switched via a sidebar radio (kept as a single-file "page switcher"
rather than Streamlit's native multipage folder convention, so the
sidebar's knowledge-base stats render identically on both views):

    - "Ask a Question"  -> question input, calls POST /v1/ask, renders
                            the answer via components/answer_display.py
    - "Knowledge Base"   -> document explorer, calls GET /v1/documents
                            via components/documents.py

All backend calls go through api_client.py; this file only handles
Streamlit layout/state and delegates rendering to components/.
"""

from __future__ import annotations

import streamlit as st

from services.api_client import APIError, ask_question
from components.answer_display import render_answer
from components.documents import render_documents_page
from components.sidebar import render_sidebar

st.set_page_config(
    page_title="Clinical CKD RAG Assistant",
    page_icon="🩺",
    layout="wide",
)

EXAMPLE_QUESTIONS = [
    "What is the eGFR threshold for CKD stage G3b?",
    "When should nephrology referral be considered?",
    "What are KDIGO recommendations for SGLT2 inhibitors?",
]


def render_ask_page() -> None:
    st.markdown("# Clinical CKD RAG Assistant")
    st.caption("Evidence-grounded answers from KDIGO guidelines and nephrology literature.")

    with st.expander("Example questions"):
        for example in EXAMPLE_QUESTIONS:
            st.markdown(f"- {example}")

    query = st.text_area(
        "Ask a clinical question...",
        height=100,
        placeholder="e.g. What is the eGFR threshold for CKD stage G3b?",
        key="question_input",
    )

    submitted = st.button("Generate Answer", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Please enter a question before submitting.")
            return

        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            try:
                response = ask_question(query.strip())
                st.session_state["last_response"] = response
                st.session_state["last_query"] = query.strip()
            except APIError as e:
                st.error(f"Could not generate an answer: {e}")
                return

    # Persist the last answer across reruns (e.g. when the sidebar refresh
    # button is clicked) so the user doesn't lose their result.
    if "last_response" in st.session_state:
        st.markdown("---")
        st.caption(f"Question: {st.session_state.get('last_query', '')}")
        render_answer(st.session_state["last_response"])


def main() -> None:
    render_sidebar()

    page = st.sidebar.radio("Navigate", ["Ask a Question", "Knowledge Base"], label_visibility="collapsed")

    if page == "Ask a Question":
        render_ask_page()
    else:
        render_documents_page()


if __name__ == "__main__":
    main()