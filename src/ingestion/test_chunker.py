import json
from chunker import chunk_pages


with open("/Users/suvarna2007/ckd-rag-system/data/processed/extracted_pages.json", "r") as f:
    pages = json.load(f)


chunks = chunk_pages(pages)

# Save chunks
with open("/Users/suvarna2007/ckd-rag-system/data/processed/chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)


print("Total chunks:", len(chunks))
print("Saved to data/processed/chunks.json")

print("\nExample chunk:")
print(chunks[0]["text"][:500])

print("\nMetadata:")
print(chunks[0]["metadata"])