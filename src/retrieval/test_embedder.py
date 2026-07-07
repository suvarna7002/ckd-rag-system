import json
from embedder import embed_text

# Load chunks
with open("/Users/suvarna2007/ckd-rag-system/data/processed/chunks.json", "r") as f:
    chunks = json.load(f)

# Test first chunk only
text = chunks[0]["text"]

embedding = embed_text(text)

print("Embedding dimension:", len(embedding))
print("First 5 values:", embedding[:5])

print("\nMetadata:")
print(chunks[0]["metadata"])