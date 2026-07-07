from typing import List, Dict, Any
import re


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation using word count.
    ~1 token ≈ 0.75 words for English text.
    """
    return int(len(text.split()) / 0.75)


def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving readability.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text: str, max_chars: int = 4000, overlap_chars: int = 500) -> List[str]:
    """
    Splits text into chunks while trying to preserve sentence boundaries.
    """
    chunks = []

    while len(text) > max_chars:
        split_point = text[:max_chars].rfind(". ")

        # FIX 1: Prevent infinite loop by forcing hard-split if boundary is missing or too early
        if split_point == -1 or split_point <= overlap_chars:
            split_point = max_chars

        chunk = text[:split_point + 1].strip()
        chunks.append(chunk)

        text = text[max(0, split_point + 1 - overlap_chars):]

    if text.strip():
        chunks.append(text.strip())

    return chunks


def create_chunk_metadata(
    page_group: List[Dict[str, Any]],
    section: str,
    chunk_index: int
) -> Dict[str, Any]:
    """
    Creates metadata for each chunk.
    """
    return {
        "source": page_group[0]["metadata"]["source"],
        "document_id": page_group[0]["metadata"].get("document_id"),
        "section": section,
        "page_start": page_group[0]["metadata"]["page"],
        "page_end": page_group[-1]["metadata"]["page"],
        "chunk_id": f"{page_group[0]['metadata']['document_id']}_{section}_{chunk_index}",
        "chunk_index": chunk_index,
    }


def chunk_pages(
    pages: List[Dict[str, Any]],
    max_chars: int = 4000,
    overlap_chars: int = 500
) -> List[Dict[str, Any]]:
    """
    Converts extracted PDF pages into section-aware chunks.
    """
    chunks = []
    current_section = "Unknown Section"
    current_pages = []
    chunk_counter = 0

    for page in pages:
        section = page["metadata"].get("section")

        # FIX 2: Evaluate section change BEFORE updating tracking variables
        if current_pages and section and section != current_section:
            new_chunks = build_chunks(
                current_pages,
                current_section,
                chunk_counter,
                max_chars,
                overlap_chars
            )
            chunks.extend(new_chunks)
            # FIX 3: Increment by chunk counts, not page counts
            chunk_counter += len(new_chunks)
            current_pages = []

        if section:
            current_section = section

        current_pages.append(page)

    # Process remaining pages
    if current_pages:
        new_chunks = build_chunks(
            current_pages,
            current_section,
            chunk_counter,
            max_chars,
            overlap_chars
        )
        chunks.extend(new_chunks)

    return chunks


def build_chunks(
    pages: List[Dict[str, Any]],
    section: str,
    start_index: int,
    max_chars: int,
    overlap_chars: int
) -> List[Dict[str, Any]]:
    """
    Builds chunks from pages belonging to the same section.
    """
    combined_text = " ".join(
        clean_text(page["text"])
        for page in pages
    )

    text_chunks = split_text(
        combined_text,
        max_chars,
        overlap_chars
    )

    output = []
    for i, text in enumerate(text_chunks):
        metadata = create_chunk_metadata(
            pages,
            section,
            start_index + i
        )
        metadata["char_count"] = len(text)
        output.append(
            {
                "text": text,
                "metadata": metadata
            }
        )

    return output