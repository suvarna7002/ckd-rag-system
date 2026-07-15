# src/generation/test_pipeline.py

#import sys
#sys.path.insert(0, ".")  # run this from project root

from dotenv import load_dotenv
load_dotenv()


from src.retrieval.hybrid_engine import HybridEngine
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.generator import ClinicalGenerator
from src.generation.citation_verifier import CitationVerifier
from src.generation.response_builder import ResponseBuilder

QUERY = "What is the dialysis protocol for pediatric G5 patients on ECMO?"

# 1. Retrieve + fuse + rerank (Phase 2)
engine = HybridEngine(dense_weight=0.5, sparse_weight=0.5)
reranker = CrossEncoderReranker()

fused = engine.query(QUERY, top_n=100, retrieval_k=100)
reranked_results = reranker.rerank(QUERY, fused, top_n=5)

print(f"Retrieved {len(reranked_results)} chunks for generation.\n")

# 2. Generate grounded answer (Phase 3 Step 1)
generator = ClinicalGenerator()
generated = generator.generate_answer(QUERY, reranked_results)
print("--- Generated Answer ---")
print(generated.answer)
print()

# 3. Verify citations (Phase 3 Step 2)
verifier = CitationVerifier()
verification = verifier.verify_response(generated, reranked_results)
print("--- Verification ---")
print(f"verified: {verification.verified_citations}/{verification.total_citations}")
print(f"confidence: {verification.confidence:.2f}")
for r in verification.citation_results:
    print(f"  [{r.citation_number}] supported={r.supported} — {r.explanation[:100]}")
print()

# 4. Build final structured response (Phase 3 Step 3)
builder = ResponseBuilder()
final = builder.build_response(generated, verification)
print("--- Final Structured Response ---")
import json
print(json.dumps(final.model_dump(), indent=2))