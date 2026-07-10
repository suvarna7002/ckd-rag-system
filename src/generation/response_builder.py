"""
response_builder.py

Phase 3, Step 3: Structured Response Schema and Metadata Integration.

Combines the output of generator.py (grounded answer + deterministic source
metadata) with the output of citation_verifier.py (per-citation faithfulness
verdicts) into a single frontend-ready object. No string parsing is required
downstream — the frontend renders directly from StructuredRAGResponse.

Field ownership is intentionally split between the two upstream stages and
never re-derived here:
- answer, sources_used, retrieved_documents, retriever_scores: owned by
  generator.py's ClinicalResponseSchema (already deterministic there — see
  generator.py's module docstring for why sources_used isn't trusted from
  the model itself).
- confidence, verified_citations, total_citations, unsupported_claims:
  owned by citation_verifier.py's VerificationResult. These fields exist as
  placeholders on ClinicalResponseSchema (all zeroed/empty) precisely so
  this stage can overwrite them with the real verification outcome rather
  than two stages disagreeing about who computed what.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from src.generation.citation_verifier import VerificationResult
from src.generation.generator import ClinicalResponseSchema, RetrieverScores, SourceUsed

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class StructuredRAGResponse(BaseModel):
    """Final, frontend-ready response for a single clinical query.

    Serializes cleanly via `.model_dump()` (or `.model_dump_json()`) with
    no further processing required — every field is already in its final
    display form.
    """

    answer: str
    sources_used: list[SourceUsed]
    confidence: float
    retrieved_documents: int
    verified_citations: int
    total_citations: int
    unsupported_claims: list[str]
    retriever_scores: RetrieverScores


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class ResponseBuilder:
    """Merges a ClinicalResponseSchema (generation output) with a
    VerificationResult (citation verification output) into a single
    StructuredRAGResponse.

    Contains no generation or verification logic itself — purely an
    integration/assembly layer, so it has no external dependencies beyond
    the two upstream schemas.
    """

    def build_response(
        self,
        generated_response: ClinicalResponseSchema,
        verification_result: Optional[VerificationResult],
    ) -> StructuredRAGResponse:
        """Combine generation output and verification results into a final
        response object.

        Args:
            generated_response: the ClinicalResponseSchema produced by
                ClinicalGenerator.generate_answer(). Its answer,
                sources_used, retrieved_documents, and retriever_scores
                fields are carried through unchanged — they're already
                deterministic (see generator.py).
            verification_result: the VerificationResult produced by
                CitationVerifier.verify_response() for this same answer.
                May be None if verification was skipped or failed at the
                pipeline level (e.g. the caller chose not to run
                verification for this query) — in that case, verification
                fields fall back to zero/empty rather than fabricating a
                confidence score (see `_empty_verification_fields`).

        Returns:
            A StructuredRAGResponse ready for direct frontend rendering.
        """
        verified_citations, total_citations, unsupported_claims, confidence = (
            self._extract_verification_fields(verification_result)
        )

        return StructuredRAGResponse(
            answer=generated_response.answer,
            sources_used=generated_response.sources_used,
            confidence=confidence,
            retrieved_documents=generated_response.retrieved_documents,
            verified_citations=verified_citations,
            total_citations=total_citations,
            unsupported_claims=unsupported_claims,
            retriever_scores=generated_response.retriever_scores,
        )

    # -- internal helpers ---------------------------------------------------

    def _extract_verification_fields(
        self, verification_result: Optional[VerificationResult]
    ) -> tuple[int, int, list[str], float]:
        """Pull the four verification-owned fields out of a
        VerificationResult, or fall back to safe zero/empty defaults if no
        verification result is available.

        Returns:
            (verified_citations, total_citations, unsupported_claims, confidence)

        Missing verification is treated the same as "zero citations
        verified" rather than raising — a query where verification wasn't
        run (or failed at the pipeline level, upstream of
        CitationVerifier's own internal fail-safes) should still produce a
        renderable response, just one that honestly reports it has no
        verification confidence to offer. No confidence value is ever
        fabricated in this branch.
        """
        if verification_result is None:
            return self._empty_verification_fields()

        return (
            verification_result.verified_citations,
            verification_result.total_citations,
            verification_result.unsupported_claims,
            verification_result.confidence,
        )

    @staticmethod
    def _empty_verification_fields() -> tuple[int, int, list[str], float]:
        """Safe defaults when no verification result exists: zero
        citations verified, zero total, no unsupported claims recorded,
        and confidence 0.0 — never fabricated, per the same rule
        CitationVerifier itself follows (zero citations -> confidence 0.0)."""
        return (0, 0, [], 0.0)