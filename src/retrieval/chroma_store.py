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
    # This automatically targets your local ./chroma_db directory
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

    # 4. Loop through every chunk, embed it, and add it to ChromaDB
    for i, chunk in enumerate(chunks):
        text_content = chunk["text"]
        metadata = chunk["metadata"]
        
        # Pull out your sequential unique ID (e.g., "KDIGO-Blood-Pressure-CKD-2021_RENAAL_304")
        chunk_id = metadata["chunk_id"]
        
        # Generate the 1,536-dimensional vector using your embedder file
        print(f"[{i+1}/{len(chunks)}] Embedding chunk: {chunk_id}")
        vector_embedding = embed_text(text_content)
        
        # Save straight to ChromaDB
        collection.add(
            ids=[chunk_id],
            embeddings=[vector_embedding],
            metadatas=[metadata],
            documents=[text_content]
        )

        if (i + 1) % 5 == 0:
            time.sleep(1) # Takes a 1-second break every 5 chunks to keep TPM low

    print(f"\n Success! Successfully indexed {collection.count()} chunks into ChromaDB.")

if __name__ == "__main__":
    build_vector_store()