"""
citation_verifier.py

Phase 3, Step 2: LLM-as-a-Judge Citation Verification.

Takes a generated ClinicalResponseSchema answer (from generator.py) plus the
same reranked_results that were used to produce it, and checks each inline
citation independently: does the cited chunk actually support the specific
claim it's attached to?

Design note (consistent with generator.py's citation-integrity approach):
the judge model is NEVER asked to echo back the citation number or claim
text. It only returns a boolean verdict + explanation for a claim/evidence
pair we already know the identity of. This avoids a second failure mode on
top of the one generator.py already guards against — a judge that mangles
or re-numbers the claim it's grading would silently corrupt the verification
results themselves. citation_number and claim are always populated by code
from the parsed citation, never from the judge's own output.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import anthropic
from anthropic import Anthropic
from pydantic import BaseModel, Field, ValidationError

from src.generation.generator import ClinicalResponseSchema, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class CitationVerification(BaseModel):
    """Verdict for a single citation occurrence in the generated answer."""

    citation_number: int
    claim: str
    supported: bool
    explanation: str


class VerificationResult(BaseModel):
    """Aggregated verification outcome across every citation occurrence
    found in a ClinicalResponseSchema's answer text."""

    verified_citations: int
    total_citations: int
    unsupported_claims: list[str]
    confidence: float
    citation_results: list[CitationVerification]


class _JudgeVerdict(BaseModel):
    """Internal-only schema for the raw judge call. Deliberately excludes
    citation_number/claim — those are supplied by code, not requested from
    the model, per the design note above."""

    supported: bool
    explanation: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class VerificationError(Exception):
    """Raised when the verification pipeline itself fails in a way that
    prevents producing a VerificationResult at all (as opposed to an
    individual claim failing verification, which is a normal, expected
    outcome and not an error)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Reuses generator.py's model default so both stages of Phase 3 stay in
# sync unless a caller explicitly overrides one or the other.
DEFAULT_VERIFIER_MODEL = DEFAULT_MODEL
DEFAULT_MAX_TOKENS = 512

JUDGE_SYSTEM_PROMPT = """You are evaluating a clinical RAG system.

Your task is NOT to answer the medical question.
Your task is ONLY to determine whether the provided evidence supports the generated claim.

Rules:
1. If the evidence explicitly supports the claim: return SUPPORTED.
2. If the claim introduces information, numbers, recommendations, or conclusions not present in the evidence: return UNSUPPORTED.
3. Do not use outside medical knowledge.
4. Be strict. Prefer UNSUPPORTED over guessing.

Return structured output only."""


# Matches a bracketed citation marker like [1], [12], etc.
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")

# Splits on sentence-ending punctuation followed by whitespace, keeping any
# trailing citation markers (e.g. "...G3b [1][2]. Next sentence...") attached
# to the sentence that precedes them.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class CitationVerifier:
    """LLM-as-judge citation verification layer.

    For every inline citation in a generated answer, isolates the sentence
    (claim) it's attached to, looks up the retrieved chunk it points to, and
    asks Claude — acting purely as a grader, never as an answerer — whether
    that specific chunk actually supports that specific claim.
    """

    def __init__(
        self,
        client: Optional[Anthropic] = None,
        model: str = DEFAULT_VERIFIER_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """
        Args:
            client: an already-instantiated anthropic.Anthropic client.
                If omitted, one is constructed using default credential
                resolution (ANTHROPIC_API_KEY environment variable).
            model: model string used for judge calls. Never hardcoded
                inline elsewhere in this class — always read from here.
            max_tokens: max output tokens per judge call (verdicts are
                short, so this can stay small).
        """
        self.client = client or Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    # -- public orchestration entrypoint -----------------------------------

    def verify_response(
        self, response: ClinicalResponseSchema, reranked_results: list[Any]
    ) -> VerificationResult:
        """Verify every inline citation in `response.answer` against the
        chunk it cites in `reranked_results`.

        Args:
            response: the ClinicalResponseSchema produced by
                ClinicalGenerator.generate_answer(). Citation numbers in
                `response.answer` (e.g. [1], [2]) are assumed to be
                1-indexed positions into `reranked_results`, matching how
                generator.py built the numbered <document> context blocks.
            reranked_results: the same ordered list of reranked chunk
                objects that was passed to ClinicalGenerator.generate_answer()
                for this response — needed to look up the evidence text
                behind each citation number.

        Returns:
            A VerificationResult aggregating a verdict for every citation
            occurrence found in the answer text.

        Raises:
            VerificationError: if extraction finds no citations to check,
                or if every judge call fails outright (as opposed to
                returning individual UNSUPPORTED verdicts, which is a
                normal result, not an error).
        """
        claim_citations = self._extract_citations_with_claims(response.answer)

        if not claim_citations:
            # No citations at all is a valid state (e.g. the insufficient-
            # information clause response), not a pipeline failure.
            return VerificationResult(
                verified_citations=0,
                total_citations=0,
                unsupported_claims=[],
                confidence=0.0,
                citation_results=[],
            )

        citation_results: list[CitationVerification] = []
        for citation_number, claim in claim_citations:
            evidence_text = self._get_evidence_text(citation_number, reranked_results)
            verification = self._verify_single_claim(
                citation_number=citation_number,
                claim=claim,
                evidence_text=evidence_text,
            )
            citation_results.append(verification)

        return self._aggregate_results(citation_results)

    # -- claim extraction -----------------------------------------------

    def _extract_citations_with_claims(self, answer: str) -> list[tuple[int, str]]:
        """Split the answer into sentences, and for every citation marker
        found in a sentence, emit an (citation_number, claim_text) pair.

        A sentence with multiple citations (e.g. "...normal [1][2].")
        produces one entry per citation number, since each citation is an
        independent claim of support that must be checked on its own —
        one citation being supported doesn't imply the other is.

        The claim text has citation markers stripped out before being
        sent to the judge, so the judge evaluates clean prose rather than
        text cluttered with bracket numbers.
        """
        sentences = _SENTENCE_SPLIT_PATTERN.split(answer.strip())

        claim_citations: list[tuple[int, str]] = []
        for sentence in sentences:
            citation_numbers = _CITATION_PATTERN.findall(sentence)
            if not citation_numbers:
                continue

            clean_claim = _CITATION_PATTERN.sub("", sentence).strip()
            if not clean_claim:
                continue

            for number_str in citation_numbers:
                claim_citations.append((int(number_str), clean_claim))

        return claim_citations

    def _get_evidence_text(self, citation_number: int, reranked_results: list[Any]) -> Optional[str]:
        """Map a 1-indexed citation number back to the chunk text it refers
        to. Returns None if the citation number is out of range (e.g. the
        model hallucinated a citation index beyond what was retrieved) —
        callers must treat this as automatically UNSUPPORTED rather than
        querying the judge with no evidence to check against."""
        index = citation_number - 1
        if index < 0 or index >= len(reranked_results):
            return None

        result = reranked_results[index]
        if hasattr(result, "text"):
            return result.text
        if isinstance(result, dict):
            return result.get("text")
        return None

    # -- judge call -----------------------------------------------------

    def _verify_single_claim(
        self, citation_number: int, claim: str, evidence_text: Optional[str]
    ) -> CitationVerification:
        """Send one (claim, evidence) pair to the judge model and return a
        CitationVerification. citation_number and claim are always taken
        from the arguments passed in, never from the model's response.
        """
        if evidence_text is None:
            # Citation points outside the retrieved document set entirely —
            # there's no evidence to check, so this is unsupported by
            # definition. No API call needed; this is a deterministic fact,
            # not a judgment call.
            return CitationVerification(
                citation_number=citation_number,
                claim=claim,
                supported=False,
                explanation=(
                    f"Citation [{citation_number}] does not correspond to any "
                    "retrieved document — no evidence exists to verify this claim against."
                ),
            )

        user_message = (
            f"<evidence>\n{evidence_text}\n</evidence>\n\n"
            f"<claim>\n{claim}\n</claim>\n\n"
            "Does the evidence explicitly support the claim? Return SUPPORTED or UNSUPPORTED "
            "with a brief explanation."
        )

        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                output_format=_JudgeVerdict,
            )
        except anthropic.APIError as e:
            return self._verification_failure(
                citation_number, claim, f"Anthropic API error during verification: {e}"
            )
        except anthropic.APIConnectionError as e:
            return self._verification_failure(
                citation_number, claim, f"Connection error reaching Anthropic API: {e}"
            )
        except ValidationError as e:
            return self._verification_failure(
                citation_number, claim, f"Judge output failed schema validation: {e}"
            )

        verdict = response.parsed_output
        if verdict is None:
            stop_reason = getattr(response, "stop_reason", "unknown")
            return self._verification_failure(
                citation_number,
                claim,
                f"No parsed verdict returned (stop_reason={stop_reason}).",
            )

        return CitationVerification(
            citation_number=citation_number,
            claim=claim,
            supported=verdict.supported,
            explanation=verdict.explanation,
        )

    @staticmethod
    def _verification_failure(
        citation_number: int, claim: str, reason: str
    ) -> CitationVerification:
        """Build a fail-safe CitationVerification for cases where the judge
        call itself couldn't be completed. Per the judge's own rule 4
        ("prefer UNSUPPORTED over guessing"), an inability to verify is
        treated as unsupported rather than silently passing the claim."""
        return CitationVerification(
            citation_number=citation_number,
            claim=claim,
            supported=False,
            explanation=f"Verification could not be completed: {reason}",
        )

    # -- aggregation ------------------------------------------------------

    def _aggregate_results(
        self, citation_results: list[CitationVerification]
    ) -> VerificationResult:
        """Roll up individual citation verdicts into a VerificationResult.

        confidence = verified_citations / total_citations, or 0.0 if there
        are no citations to check (guarded upstream in verify_response, but
        kept here too since this method could be reused independently).
        """
        total_citations = len(citation_results)
        verified_citations = sum(1 for r in citation_results if r.supported)
        unsupported_claims = [r.claim for r in citation_results if not r.supported]

        confidence = (
            verified_citations / total_citations if total_citations > 0 else 0.0
        )

        return VerificationResult(
            verified_citations=verified_citations,
            total_citations=total_citations,
            unsupported_claims=unsupported_claims,
            confidence=confidence,
            citation_results=citation_results,
        )