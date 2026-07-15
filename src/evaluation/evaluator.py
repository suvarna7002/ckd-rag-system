"""
evaluator.py

Phase 4: Automated Evaluation Framework.
Runs the golden Q&A dataset through the entire RAG pipeline and applies 
an LLM-as-a-judge pattern to score correctness, faithfulness, and citation accuracy.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import anthropic

# Pipeline imports
from src.retrieval.hybrid_engine import HybridEngine
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.generator import ClinicalGenerator
from src.generation.citation_verifier import CitationVerifier
from src.generation.response_builder import ResponseBuilder

# Load environment configs
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
GOLDEN_DATASET_PATH = Path("data/processed/golden_dataset.json")
EVAL_REPORT_PATH = Path("data/processed/eval_report.json")

# Centralized tuning parameters matching retrieval configurations
RETRIEVAL_TOP_N = 100
RETRIEVAL_K = 100
RERANK_TOP_N = 8

JUDGE_SYSTEM_PROMPT = """You are an expert clinical validation judge. 
Your job is to compare a RAG system's 'Generated Answer' against a fact-checked 'Golden Answer'.
Evaluate if the Generated Answer is clinically accurate and covers the core factual requirements of the Golden Answer.

Provide your assessment in the strict structured format required."""

class JudgeSchema(BaseModel):
    is_correct: bool = Field(description="True if the generated answer matches the semantic facts of the golden answer without clinical contradictions.")
    explanation: str = Field(description="Brief explanation of why the answer is correct or what factual contradiction/omission occurred.")

class RAGEvaluator:
    def __init__(self) -> None:
        print("Initializing evaluation pipeline pipelines...")
        self.engine = HybridEngine(dense_weight=0.5, sparse_weight=0.5)
        self.reranker = CrossEncoderReranker()
        self.generator = ClinicalGenerator()
        self.verifier = CitationVerifier()
        self.builder = ResponseBuilder()
        self.client = anthropic.Anthropic()

    def run_eval_suite(self) -> None:
        if not GOLDEN_DATASET_PATH.exists():
            raise FileNotFoundError(f"Missing golden dataset at {GOLDEN_DATASET_PATH}. Save it there first.")

        with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
            test_cases: List[Dict[str, Any]] = json.load(f)

        print(f"Loaded {len(test_cases)} evaluation test cases. Commencing run...\n")
        results = []

        total_cases = len(test_cases)
        correct_counts = 0
        faithful_counts = 0
        perfect_citation_counts = 0
        retrieval_hit_counts = 0

        for idx, case in enumerate(test_cases, start=1):
            q_id = case.get("id", f"Q{idx}")
            question = case["question"]
            golden_answer = case.get("golden_answer") or case.get("gold_answer")
            category = case.get("category", "general")

            print(f"[{idx}/{total_cases}] Processing {q_id} ({category})...")

            try:
                # 1. Pipeline Execution using Centralized Named Constants
                fused = self.engine.query(question, top_n=RETRIEVAL_TOP_N, retrieval_k=RETRIEVAL_K)
                reranked = self.reranker.rerank(question, fused, top_n=RERANK_TOP_N)
                
                generated = self.generator.generate_answer(question, reranked)
                verification = self.verifier.verify_response(generated, reranked)
                final_response = self.builder.build_response(generated, verification)
                
                # 2. Fix 1: Calculate Retrieval Hit Rate
                retrieved_docs = set()
                for r in reranked:
                    if hasattr(r, "metadata") and hasattr(r.metadata, "get"):
                        retrieved_docs.add(r.metadata.get("source"))
                    elif isinstance(r, dict):
                        retrieved_docs.add(r.get("metadata", {}).get("source"))
                
                expected_docs = set(case.get("expected_documents", []))
                retrieval_hit = bool(expected_docs & retrieved_docs) if expected_docs else True

                # 3. LLM-as-a-Judge Correctness Call
                judge_result = self.judge_correctness(question, final_response.answer, golden_answer)

                # 4. Compute Metrics
                is_correct = judge_result.is_correct if judge_result is not None else False
                is_faithful = len(final_response.unsupported_claims) == 0
                
                total_cites = final_response.total_citations
                verified_cites = final_response.verified_citations
                has_perfect_citations = total_cites == verified_cites if total_cites > 0 else True

                if is_correct: correct_counts += 1
                if is_faithful: faithful_counts += 1
                if has_perfect_citations: perfect_citation_counts += 1
                if retrieval_hit: retrieval_hit_counts += 1

                results.append({
                    "id": q_id,
                    "category": category,
                    "question": question,
                    "golden_answer": golden_answer,
                    "generated_answer": final_response.answer,
                    "metrics": {
                        "retrieval_hit": retrieval_hit,
                        "correctness": is_correct,
                        "faithfulness": is_faithful,
                        "citation_accuracy": has_perfect_citations,
                        "total_citations": total_cites,
                        "verified_citations": verified_cites
                    },
                    "judge_explanation": judge_result.explanation if judge_result is not None else "LLM-as-a-judge formatting error occurred."
                })

            except Exception as e:
                print(f"❌ Error processing test case {q_id}: {e}")
                # Fix 3: Keep schema perfectly uniform on code exceptions
                results.append({
                    "id": q_id,
                    "category": category,
                    "question": question,
                    "golden_answer": golden_answer,
                    "generated_answer": None,
                    "metrics": {
                        "retrieval_hit": False,
                        "correctness": False,
                        "faithfulness": False,
                        "citation_accuracy": False,
                        "total_citations": 0,
                        "verified_citations": 0
                    },
                    "judge_explanation": f"Pipeline Error Execution Branch Failure: {str(e)}",
                    "error": str(e)
                })

        # Summary calculations
        summary = {
            "total_evaluated": total_cases,
            "headline_retrieval_hit_rate": (retrieval_hit_counts / total_cases) * 100,
            "headline_correctness_rate": (correct_counts / total_cases) * 100,
            "headline_faithfulness_rate": (faithful_counts / total_cases) * 100,
            "headline_citation_accuracy": (perfect_citation_counts / total_cases) * 100,
            "category_breakdown": {}
        }

        # Fix 2: Dynamically aggregate per-category breakdowns
        category_stats = {}
        for r in results:
            cat = r["category"]
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "correct": 0, "faithful": 0, "citation": 0, "retrieval": 0}
            
            m = r.get("metrics", {})
            category_stats[cat]["total"] += 1
            if m.get("correctness"): category_stats[cat]["correct"] += 1
            if m.get("faithfulness"): category_stats[cat]["faithful"] += 1
            if m.get("citation_accuracy"): category_stats[cat]["citation"] += 1
            if m.get("retrieval_hit"): category_stats[cat]["retrieval"] += 1

        for cat, counts in category_stats.items():
            tot = counts["total"]
            summary["category_breakdown"][cat] = {
                "total_cases": tot,
                "retrieval_hit_rate": (counts["retrieval"] / tot) * 100,
                "correctness_rate": (counts["correct"] / tot) * 100,
                "faithfulness_rate": (counts["faithful"] / tot) * 100,
                "citation_accuracy": (counts["citation"] / tot) * 100
            }

        output_data = {
            "summary": summary,
            "detailed_results": results
        }

        with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as out:
            json.dump(output_data, out, indent=2)

        self.log_summary(summary)

    def judge_correctness(self, question: str, generated: str, golden: str) -> Any:
        """Executes structured LLM evaluation comparing answers."""
        prompt = (
            f"Clinical Question: {question}\n\n"
            f"Expected Golden Answer:\n{golden}\n\n"
            f"RAG Generated Answer:\n{generated}\n\n"
            "Compare the facts. Does the RAG answer preserve the accuracy of the Golden Answer?"
        )
        try:
            response = self.client.messages.parse(
                model="claude-sonnet-5",
                max_tokens=1500,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=JudgeSchema
            )
            return response.parsed_output
        except Exception as e:
            print(f"⚠️ LLM-as-a-judge parse/API failure: {e}")
            return None

    @staticmethod
    def log_summary(summary: dict) -> None:
        print("\n" + "="*60)
        print("📊 PHASE 4 EVALUATION FRAMEWORK METRICS REPORT")
        print("="*60)
        print(f"Total Clinical Test Cases: {summary['total_evaluated']}")
        print(f"🔍 Retrieval Hit Rate:     {summary['headline_retrieval_hit_rate']:.1f}%")
        print(f"🎯 Answer Correctness:     {summary['headline_correctness_rate']:.1f}%")
        print(f"🛡️  Context Faithfulness:  {summary['headline_faithfulness_rate']:.1f}%")
        print(f"📜 Citation Accuracy:     {summary['headline_citation_accuracy']:.1f}%")
        
        print("\n📂 PER-CATEGORY BREAKDOWN:")
        for cat, metrics in summary["category_breakdown"].items():
            print(f"  ▪️ [{cat.upper()}] (n={metrics['total_cases']})")
            print(f"      Retrieval Hit:  {metrics['retrieval_hit_rate']:.1f}%")
            print(f"      Correctness:    {metrics['correctness_rate']:.1f}%")
            print(f"      Faithfulness:   {metrics['faithfulness_rate']:.1f}%")
            print(f"      Citation Acc:   {metrics['citation_accuracy']:.1f}%")
        print("="*60)
        print(f"Full structural output written to: {EVAL_REPORT_PATH}\n")

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.run_eval_suite()