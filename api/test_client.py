import json
import requests

BASE_URL = "http://127.0.0.1:8000"


def test_documents():
    print("\n=== Fetching Document Summary ===")
    response = requests.get(f"{BASE_URL}/v1/documents")
    print(json.dumps(response.json(), indent=2))


def test_ask(query: str):
    print(f"\n=== Querying: {query!r} ===")
    payload = {
        "query": query,
        "top_n": 5,
        "retrieval_k": 100
    }
    response = requests.post(f"{BASE_URL}/v1/ask", json=payload)
    if response.status_code != 200:
        print(f"Error ({response.status_code}): {response.text}")
        return

    data = response.json()
    print("\n--- GENERATED ANSWER ---")
    print(data["answer"])

    # SourceUsed only carries document/section/chunk_id — there is no
    # citation_number or page_number field anywhere in the pipeline
    # (generator.py's SourceUsed model). Printing what actually exists.
    print("\n--- SOURCES USED ---")
    for s in data["sources_used"]:
        print(f"- {s['document']} | section: {s['section']} | chunk_id: {s['chunk_id']}")

    print("\n--- CITATION FAITHFULNESS VERDICTS ---")
    print(f"Confidence: {data['confidence'] * 100:.1f}%")
    print(f"Verified Citations: {data['verified_citations']}/{data['total_citations']}")
    if data["unsupported_claims"]:
        print("\n[WARNING] Unsupported claims found:")
        for claim in data["unsupported_claims"]:
            print(f"  ❌ {claim}")
    else:
        print("  ✅ All claims matched and verified by LLM-as-a-judge.")


if __name__ == "__main__":
    try:
        test_documents()
    except Exception as e:
        print(f"Could not connect to API server. Did you start Uvicorn? Error: {e}")
        exit(1)

    test_ask("What is the eGFR threshold for stage G3b?")