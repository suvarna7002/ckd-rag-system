"""
generator.py

Phase 3, Step 1: Grounded Generation and Structured Output Delivery.

Takes the reranked chunks produced by the Phase 2 hybrid retrieval engine,
wraps them as numbered context documents, and asks Claude to answer strictly
from that context — using the Claude API's native structured-outputs feature
(client.messages.parse(..., output_format=PydanticModel)) so the response is
guaranteed to validate against ClinicalResponseSchema without manual JSON
parsing or retry logic.

Citation integrity note: the `sources_used` list returned to the caller is
NOT trusted from the model's own output. It is rebuilt deterministically from
the actual upstream reranked_results (the same objects used to build the
context block), indexed the same way the model was told to cite them. This
guarantees every citation a user sees maps to a real chunk_id/source/section
that was actually retrieved — the model cannot hallucinate a document name
or section that doesn't exist, because the code overwrites whatever it
returns for that field. This is intentionally stricter than what was asked
for verbatim, and is called out here rather than silently changed.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import anthropic
from anthropic import Anthropic
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class SourceUsed(BaseModel):
    """A single source reference backing a cited claim in the answer."""

    document: str
    section: str
    chunk_id: str


class RetrieverScores(BaseModel):
    """Per-retriever scores for the chunks that made it into the final
    context window, aligned by position with the numbered documents.

    Note: dense/sparse raw scores are only populated if the upstream
    result objects expose them. The RerankedResult objects produced by
    reranker.py in this pipeline carry `dense_rank`/`sparse_rank` (integer
    ranks) rather than raw cosine similarity / BM25 scores, since those
    raw values aren't threaded through fusion. Where a raw score isn't
    available, the corresponding list is left empty rather than
    fabricated — see `_build_retriever_scores` below.
    """

    dense: list[float] = Field(default_factory=list)
    sparse: list[float] = Field(default_factory=list)
    fused: list[float] = Field(default_factory=list)


class ClinicalResponseSchema(BaseModel):
    """Structured, citation-tracked answer returned by ClinicalGenerator."""

    answer: str = Field(
        description="The clinical answer string containing inline bracketed citations like [1], [2]"
    )
    sources_used: list[SourceUsed] = Field(
        description="List of raw source references that match the cited index positions."
    )
    confidence: float = Field(
        default=0.0, description="Confidence placeholder for citation verification math."
    )
    retrieved_documents: int = Field(default=0)
    verified_citations: int = Field(default=0)
    total_citations: int = Field(default=0)
    unsupported_claims: list[str] = Field(
        default_factory=list, description="To be populated by step 2 validator."
    )
    retriever_scores: RetrieverScores


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GenerationError(Exception):
    """Raised when grounded generation fails — wraps SDK errors and
    validation failures so callers get one exception type to handle."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Single named default — never inlined elsewhere in the class logic, so
# swapping models means changing this one constant (or passing `model=`
# at construction time).
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are a clinical guideline assistant specializing in chronic kidney disease (CKD) and nephrology.

You will be given a set of numbered <document> blocks inside a <context> block, followed by a clinical question.

Follow these rules with no exceptions:

1. ABSOLUTE GROUNDING: Only answer using information directly stated in the provided <context> documents. Never use outside medical knowledge, never extrapolate beyond what is explicitly written, and never fill gaps with clinical reasoning that isn't backed by the text.

2. INLINE CITATIONS: Every clinical assertion in your answer must be followed by a bracketed citation number (e.g., [1], [2]) that maps directly to the `id` attribute of the <document> it came from. If a single sentence draws on multiple documents, cite all of them (e.g., [1][3]).

3. INSUFFICIENT INFORMATION CLAUSE: If the provided context documents do not contain explicit facts, numbers, or tables that safely answer the question, your `answer` field MUST be EXACTLY the following string, with no additions, no partial answer, and no synthesis:
"I do not have enough information in the provided guidelines to answer this question safely."

Do not soften rule 3, do not hedge around it, and do not attempt to partially answer if the explicit fact is not present."""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ClinicalGenerator:
    """Orchestrates grounded generation over reranked retrieval results.

    Wraps the Anthropic client and the Claude API's native structured
    outputs feature (`client.messages.parse(..., output_format=Model)`),
    which constrains decoding so the response is guaranteed to validate
    against ClinicalResponseSchema — no manual JSON parsing, no retry
    logic for malformed output.
    """

    def __init__(
        self,
        client: Optional[Anthropic] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """
        Args:
            client: an already-instantiated anthropic.Anthropic client.
                If omitted, one is constructed using default credential
                resolution (ANTHROPIC_API_KEY environment variable).
            model: model string to use for generation. Never hardcoded
                inline elsewhere in this class — always read from here.
            max_tokens: max output tokens for the generation call.
        """
        self.client = client or Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    # -- public orchestration entrypoint -----------------------------------

    def generate_answer(self, query: str, reranked_results: list[Any]) -> ClinicalResponseSchema:
        """Generate a grounded, structured, citation-tracked clinical answer.

        Args:
            query: the user's clinical question.
            reranked_results: ordered list of reranked chunk objects from
                the Phase 2 hybrid engine (RerankedResult instances or
                anything exposing chunk_id/text/metadata and, optionally,
                rrf_score/rerank_score/dense_rank/sparse_rank).

        Returns:
            A validated ClinicalResponseSchema instance.

        Raises:
            GenerationError: if the API call fails or the response cannot
                be validated against the schema.
        """
        context_block = self._build_context_block(reranked_results)
        user_message = self._build_user_message(query, context_block)

        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                output_format=ClinicalResponseSchema,
            )
        except anthropic.APIError as e:
            raise GenerationError(f"Anthropic API error during generation: {e}") from e
        except anthropic.APIConnectionError as e:
            raise GenerationError(f"Connection error reaching Anthropic API: {e}") from e
        except ValidationError as e:
            raise GenerationError(f"Model output failed schema validation: {e}") from e

        parsed = response.parsed_output
        if parsed is None:
            # Structured outputs can still fail to populate parsed_output on
            # refusal (stop_reason == "refusal") or max_tokens truncation —
            # both are documented edge cases where schema compliance isn't
            # guaranteed. Surface this clearly rather than returning a
            # half-built schema.
            stop_reason = getattr(response, "stop_reason", "unknown")
            raise GenerationError(
                f"No parsed structured output returned (stop_reason={stop_reason}). "
                "This can happen on refusal or max_tokens truncation."
            )

        return self._finalize_response(parsed, reranked_results)

    # -- context construction -----------------------------------------------

    def _build_context_block(self, reranked_results: list[Any]) -> str:
        """Wrap each reranked chunk in a numbered <document> tag, forming
        the <context> payload the model is instructed to answer from.

        Document metadata (source, section) is embedded in each tag so the
        model can produce human-readable citations, even though the final
        `sources_used` list is rebuilt deterministically afterward rather
        than trusted from the model's own output (see module docstring).
        """
        document_blocks = []
        for i, result in enumerate(reranked_results, start=1):
            source = self._get_metadata_field(result, "source", default="Unknown source")
            section = self._get_metadata_field(result, "section", default="Unknown section")
            text = self._get_text(result)

            document_blocks.append(
                f'<document id="{i}" source="{source}" section="{section}">\n'
                f"{text}\n"
                f"</document>"
            )

        return "<context>\n" + "\n".join(document_blocks) + "\n</context>"

    def _build_user_message(self, query: str, context_block: str) -> str:
        """Compose the final user-turn message: context followed by the
        clinical question, kept clearly separated for the model."""
        return f"{context_block}\n\n<question>\n{query}\n</question>"

    # -- post-processing ------------------------------------------------

    def _finalize_response(
        self, parsed: ClinicalResponseSchema, reranked_results: list[Any]
    ) -> ClinicalResponseSchema:
        """Overwrite the model-generated bookkeeping fields with
        deterministic, code-computed values where accuracy matters more
        than letting the model self-report:

        - sources_used: rebuilt from the real upstream metadata (see
          module docstring — never trust the model's own citation text).
        - retrieved_documents: exact count of chunks actually provided.
        - total_citations: counted directly from the answer text via regex,
          not self-reported by the model.
        - retriever_scores: rebuilt from whatever numeric fields the
          upstream result objects actually expose.

        confidence, verified_citations, and unsupported_claims are left
        as returned (they are placeholders for the Phase 3 Step 2 citation
        verifier, which hasn't run yet).
        """
        parsed.sources_used = self._build_sources_used(reranked_results)
        parsed.retrieved_documents = len(reranked_results)
        parsed.total_citations = self._count_citations(parsed.answer)
        parsed.retriever_scores = self._build_retriever_scores(reranked_results)
        return parsed

    def _build_sources_used(self, reranked_results: list[Any]) -> list[SourceUsed]:
        """Deterministically build the source list from the same objects
        used to construct the context block, in the same order (so index
        i+1 here matches citation [i+1] the model was told to use)."""
        sources: list[SourceUsed] = []
        for result in reranked_results:
            sources.append(
                SourceUsed(
                    document=self._get_metadata_field(result, "source", default="Unknown source"),
                    section=self._get_metadata_field(result, "section", default="Unknown section"),
                    chunk_id=self._get_chunk_id(result),
                )
            )
        return sources

    def _build_retriever_scores(self, reranked_results: list[Any]) -> RetrieverScores:
        """Extract whatever numeric retriever scores are actually present
        on the upstream result objects.

        `fused` is populated from `rrf_score` (the pre-rerank fusion
        score) when present. `dense`/`sparse` are only populated if the
        upstream objects happen to expose raw numeric scores for those
        retrievers (e.g. a future `dense_score`/`sparse_score` attribute);
        the current RerankedResult only carries `dense_rank`/`sparse_rank`
        (integer positions, not scores), so those lists are left empty by
        default rather than filled with a fabricated number.
        """
        dense_scores: list[float] = []
        sparse_scores: list[float] = []
        fused_scores: list[float] = []

        for result in reranked_results:
            dense_score = getattr(result, "dense_score", None)
            if dense_score is not None:
                dense_scores.append(float(dense_score))

            sparse_score = getattr(result, "sparse_score", None)
            if sparse_score is not None:
                sparse_scores.append(float(sparse_score))

            fused_score = getattr(result, "rrf_score", None)
            if fused_score is not None:
                fused_scores.append(float(fused_score))

        return RetrieverScores(dense=dense_scores, sparse=sparse_scores, fused=fused_scores)

    @staticmethod
    def _count_citations(answer: str) -> int:
        """Count bracketed citation markers like [1], [2] in the answer
        text. Counts occurrences, not unique numbers, since a repeated
        citation of the same source still reflects a distinct claim
        being backed by it."""
        return len(re.findall(r"\[\d+\]", answer))

    # -- shared attribute/metadata access -----------------------------------

    @staticmethod
    def _get_text(result: Any) -> str:
        """Extract chunk text from a result object, tolerating either a
        `.text` attribute or a dict-shaped result."""
        if hasattr(result, "text"):
            return result.text
        if isinstance(result, dict):
            return result.get("text", "")
        raise GenerationError(f"Result object has no accessible text field: {result!r}")

    @staticmethod
    def _get_chunk_id(result: Any) -> str:
        """Extract chunk_id from a result object, tolerating either a
        `.chunk_id` attribute or a dict-shaped result."""
        if hasattr(result, "chunk_id"):
            return result.chunk_id
        if isinstance(result, dict):
            return result.get("chunk_id", "")
        raise GenerationError(f"Result object has no accessible chunk_id field: {result!r}")

    @staticmethod
    def _get_metadata_field(result: Any, field: str, default: str = "") -> str:
        """Extract a field from a result object's metadata dict, tolerating
        either a `.metadata` attribute or a dict-shaped result with a
        nested "metadata" key. Falls back to `default` if missing."""
        metadata = None
        if hasattr(result, "metadata"):
            metadata = result.metadata
        elif isinstance(result, dict):
            metadata = result.get("metadata", {})

        if not metadata:
            return default

        return metadata.get(field, default)