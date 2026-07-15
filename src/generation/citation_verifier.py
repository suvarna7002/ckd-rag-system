"""
citation_verifier.py

Phase 3, Step 2: LLM-as-a-Judge Citation Verification.

Takes a generated ClinicalResponseSchema answer (from generator.py) plus the
same reranked_results that were used to produce it, and checks each inline
citation independently: does the cited chunk actually support the specific
claim it's attached to?

Design note (consistent with generator.py's citation-integrity approach):
the judge model is NEVER asked to echo back the citation number or claim
text. It only returns a boolean verdict + explanation for each claim/evidence
pair, in the same order those pairs were sent. citation_number and claim are
always populated by code from the parsed citation (by matching response
position, never by trusting any identifier the model returns), so a judge
that mangles or re-numbers an item can't silently corrupt which claim a
verdict gets attached to.

Cost note: all citation checks for a single generated answer are batched
into ONE API call (one prompt containing every claim/evidence pair, one
structured response containing a verdict per pair) rather than issuing a
separate API call per citation. A single multi-citation answer previously
cost one call per citation occurrence; it now costs exactly one call,
regardless of how many citations it contains.
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


class _BatchJudgeItem(BaseModel):
    """Internal-only schema for a single verdict within a batched judge
    call. Deliberately excludes citation_number/claim — those are supplied
    by code via positional matching, not requested from the model, per the
    design note above."""

    supported: bool
    explanation: str


class _BatchJudgeVerdict(BaseModel):
    """Internal-only schema for the raw batched judge call. `items` must
    be returned in the same order the claim/evidence pairs were presented
    in the prompt — verified by length check before being trusted."""

    items: list[_BatchJudgeItem]


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
# Batched calls return one verdict per citation instead of one per call,
# so this needs more headroom than the old single-item 512 default.
DEFAULT_MAX_TOKENS = 3072

JUDGE_SYSTEM_PROMPT = """You are evaluating a clinical RAG system.

Your task is NOT to answer the medical question.
Your task is ONLY to determine, for EACH numbered claim/evidence pair below,
whether the provided evidence supports the generated claim.

Rules:
1. If the evidence explicitly supports the claim: mark it SUPPORTED.
2. If the claim introduces information, numbers, recommendations, or conclusions not present in the evidence: mark it UNSUPPORTED.
3. Do not use outside medical knowledge.
4. Be strict. Prefer UNSUPPORTED over guessing.
5. You will be given N claim/evidence pairs. You MUST return exactly N verdicts, in the SAME ORDER as the pairs were given. Do not skip, merge, or reorder any pair.

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
    (claim) it's attached to, looks up the retrieved chunk(s) it points to,
    and asks Claude — acting purely as a grader, never as an answerer —
    whether that specific evidence actually supports that specific claim.
    All claim/evidence pairs for a single answer are verified in one
    batched API call rather than one call per citation.
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
            max_tokens: max output tokens for the batched judge call.
        """
        self.client = client or Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    # -- public orchestration entrypoint -----------------------------------

    def verify_response(
        self, response: ClinicalResponseSchema, reranked_results: list[Any]
    ) -> VerificationResult:
        """Verify every inline citation in `response.answer` against the
        chunk(s) it cites in `reranked_results`, using a single batched
        API call for the whole answer.

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
        """
        sentence_citations = self._extract_sentence_citations(response.answer)
        if not sentence_citations:
            # No citations at all is a valid state (e.g. the insufficient-
            # information clause response), not a pipeline failure.
            return VerificationResult(
                verified_citations=0,
                total_citations=0,
                unsupported_claims=[],
                confidence=0.0,
                citation_results=[],
            )

        # Flatten (claim, citation_numbers) pairs into one item per
        # individual citation occurrence, since each citation is an
        # independent claim of support that must be checked on its own —
        # one citation being supported doesn't imply another is.
        flat_items: list[tuple[int, str, Optional[str]]] = []
        for claim, citation_numbers in sentence_citations:
            combined_evidence = self._get_combined_evidence_text(citation_numbers, reranked_results)
            for citation_number in citation_numbers:
                flat_items.append((citation_number, claim, combined_evidence))

        citation_results = self._verify_batch(flat_items)
        return self._aggregate_results(citation_results)

    # -- claim extraction -----------------------------------------------

    def _extract_sentence_citations(self, answer: str) -> list[tuple[str, list[int]]]:
        """Split the answer into sentences, keeping all citation numbers in
        a sentence grouped together so they can be verified against their
        combined evidence rather than one chunk at a time."""
        sentences = _SENTENCE_SPLIT_PATTERN.split(answer.strip())
        sentence_citations: list[tuple[str, list[int]]] = []
        for sentence in sentences:
            citation_numbers = [int(n) for n in _CITATION_PATTERN.findall(sentence)]
            if not citation_numbers:
                continue

            clean_claim = _CITATION_PATTERN.sub("", sentence).strip()
            if not clean_claim:
                continue

            sentence_citations.append((clean_claim, citation_numbers))
        return sentence_citations

    def _get_combined_evidence_text(
        self, citation_numbers: list[int], reranked_results: list[Any]
    ) -> Optional[str]:
        """Concatenate evidence for every citation attached to a sentence,
        so a claim that legitimately draws on multiple chunks is graded
        against all of them together, not one at a time."""
        texts = []
        for n in citation_numbers:
            t = self._get_evidence_text(n, reranked_results)
            if t:
                texts.append(f"[{n}] {t}")
        return "\n\n".join(texts) if texts else None

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

    def _verify_batch(
        self, flat_items: list[tuple[int, str, Optional[str]]]
    ) -> list[CitationVerification]:
        """Verify every (citation_number, claim, evidence) triple for a
        response in a single API call.

        Items with no evidence (citation pointed outside the retrieved set)
        are resolved deterministically without consuming a slot in the
        batched prompt — no evidence means automatically unsupported,
        which is a fact, not a judgment call. Only items that actually have
        evidence are sent to the judge, and results are recombined with the
        deterministic ones afterward, preserving original order.
        """
        results: list[Optional[CitationVerification]] = [None] * len(flat_items)

        judgeable_indices: list[int] = []
        prompt_sections: list[str] = []
        for i, (citation_number, claim, evidence_text) in enumerate(flat_items):
            if evidence_text is None:
                results[i] = CitationVerification(
                    citation_number=citation_number,
                    claim=claim,
                    supported=False,
                    explanation=(
                        f"Citation [{citation_number}] does not correspond to any "
                        "retrieved document — no evidence exists to verify this claim against."
                    ),
                )
                continue

            judgeable_indices.append(i)
            item_num = len(judgeable_indices)
            prompt_sections.append(
                f"--- Pair {item_num} ---\n"
                f"<evidence>\n{evidence_text}\n</evidence>\n"
                f"<claim>\n{claim}\n</claim>"
            )

        if not judgeable_indices:
            # Every citation was out-of-range; nothing to send to the judge.
            return [r for r in results if r is not None]

        user_message = (
            f"You will evaluate {len(judgeable_indices)} claim/evidence pairs below. "
            f"Return exactly {len(judgeable_indices)} verdicts in `items`, in the same order.\n\n"
            + "\n\n".join(prompt_sections)
        )

        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                output_format=_BatchJudgeVerdict,
            )
            verdict = response.parsed_output

            if verdict is None:
                stop_reason = getattr(response, "stop_reason", "unknown")
                raise VerificationError(
                    f"No parsed verdict returned (stop_reason={stop_reason})."
                )
            if len(verdict.items) != len(judgeable_indices):
                raise VerificationError(
                    f"Judge returned {len(verdict.items)} verdicts for "
                    f"{len(judgeable_indices)} claim/evidence pairs — cannot "
                    "reliably match verdicts to claims by position."
                )

            for position, flat_index in enumerate(judgeable_indices):
                citation_number, claim, _ = flat_items[flat_index]
                item = verdict.items[position]
                results[flat_index] = CitationVerification(
                    citation_number=citation_number,
                    claim=claim,
                    supported=item.supported,
                    explanation=item.explanation,
                )

        except (anthropic.APIError, anthropic.APIConnectionError, ValidationError, VerificationError):
            try:
                response = self.client.messages.parse(
                    model=self.model, max_tokens=self.max_tokens, system=JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}], output_format=_BatchJudgeVerdict,
                )
                verdict = response.parsed_output
                if verdict and len(verdict.items) == len(judgeable_indices):
                    for position, flat_index in enumerate(judgeable_indices):
                        citation_number, claim, _ = flat_items[flat_index]
                        item = verdict.items[position]
                        results[flat_index] = CitationVerification(
                            citation_number=citation_number, claim=claim,
                            supported=item.supported, explanation=item.explanation,
                        )
                    return [r for r in results if r is not None]
            except Exception:
                pass
        except Exception as e:
            reason = f"Batched verification call failed: {e}"
            for flat_index in judgeable_indices:
                if results[flat_index] is None:
                    citation_number, claim, _ = flat_items[flat_index]
                    results[flat_index] = self._verification_failure(citation_number, claim, reason)

        return [r for r in results if r is not None]

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