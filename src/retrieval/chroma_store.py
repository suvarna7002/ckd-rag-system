import os
import json
import chromadb
import time
from embedder import embed_text

# 1. Dynamically calculate absolute paths based on your folder structure
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
CHUNKS_PATH = os.path.join(ROOT_DIR, "data", "processed", "chunks.json")
CHROMA_DIR = os.path.join(ROOT_DIR, "chroma_db")

def build_vector_store():
    # 2. Load your pre-processed chunks
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Could not find chunks.json at {CHUNKS_PATH}. Did you run your chunker?")
        
    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)
        
    print(f"Loaded {len(chunks)} chunks from chunks.json. Starting indexing pipeline...")

    # 3. Initialize Persistent ChromaDB Client
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Clean slate: Wipe the collection if it already exists so we don't duplicate data on re-runs
    collection_name = "ckd_guidelines"
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"Resetting existing collection: '{collection_name}'")
    except Exception:
        pass # Collection didn't exist yet, which is perfect

    # Create a fresh collection using Cosine Similarity for clinical text geometry
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Trackers for the end summary
    total_processed = len(chunks)
    inserted_count = 0
    skipped_count = 0

    # 4. Loop through every chunk, embed it, deduplicate, and add to ChromaDB
    for i, chunk in enumerate(chunks):
        text_content = chunk["text"]
        metadata = chunk["metadata"]
        
        # Pull out your sequential unique ID
        chunk_id = metadata["chunk_id"]
        
        print(f"[{i+1}/{total_processed}] Embedding chunk: {chunk_id}")
        vector_embedding = embed_text(text_content)
        
        # --- DEDUPLICATION LOGIC ---
        is_duplicate = False
        
        # Only query if the collection isn't empty
        if collection.count() > 0:
            results = collection.query(
                query_embeddings=[vector_embedding],
                n_results=1
            )
            
            # Extract distance and convert to similarity
            if results["distances"] and results["distances"][0]:
                distance = results["distances"][0][0]
                similarity = 1.0 - distance
                
                # Check against the 0.95 threshold
                if similarity > 0.95:
                    is_duplicate = True
                    matched_id = results["ids"][0][0]
                    print(f"  -> Skipped! Duplicate found. Similarity: {similarity:.4f} (Matches: {matched_id})")
                    skipped_count += 1
        
        # Insert if it's not a duplicate
        if not is_duplicate:
            collection.add(
                ids=[chunk_id],
                embeddings=[vector_embedding],
                metadatas=[metadata],
                documents=[text_content]
            )
            inserted_count += 1

        # Rate Limiting
        if (i + 1) % 5 == 0:
            time.sleep(1)

    # Final summary printout
    print(f"\n--- Indexing Complete ---")
    print(f"Total chunks processed: {total_processed}")
    print(f"Chunks inserted: {inserted_count}")
    print(f"Chunks skipped as duplicates: {skipped_count}")

if __name__ == "__main__":
    build_vector_store()