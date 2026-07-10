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
    Normalize whitespace while preserving readability and structural newlines.
    """
    # **NEW FIX**: Collapse horizontal spaces but preserve newlines for list-splitting
    text = re.sub(r"[ \t]+", " ", text) 
    # Normalize excessive consecutive newlines down to a max of two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, max_chars: int = 4000, overlap_chars: int = 500) -> List[str]:
    """
    Splits text into chunks while trying to preserve sentence and list boundaries.
    """
    chunks = []

    while len(text) > max_chars:
        # 1. Primary Strategy: Try splitting on a sentence boundary
        split_point = text[:max_chars].rfind(". ")

        # **NEW FIX**: 2. Secondary Strategy: Fallback to newline splitting for lists/tables
        if split_point == -1 or split_point <= overlap_chars:
            split_point = text[:max_chars].rfind("\n")

        # 3. Tertiary Strategy: Force hard character split if all else fails
        if split_point == -1 or split_point <= overlap_chars:
            split_point = max_chars
        else:
            # Include the split character (period or newline) in the chunk
            split_point += 1

        chunk = text[:split_point].strip()
        chunks.append(chunk)

        # Calculate raw overlap start
        overlap_start = max(0, split_point - overlap_chars)
        
        # **NEW FIX**: Move forward to the next space so the overlap chunk starts with a whole word
        next_space = text[overlap_start:].find(" ")
        if next_space != -1 and (overlap_start + next_space) < split_point:
            overlap_start += next_space + 1

        text = text[overlap_start:]

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

        # FIX:
        # Now we flush on ANY section change, including into or out of an
        # undetected heading — a page with no detected heading still closes
        # out whatever section preceded it, bounding each group at the page
        # level instead of merging indefinitely.
        if current_pages and section != current_section:
            new_chunks = build_chunks(
                current_pages,
                current_section,
                chunk_counter,
                max_chars,
                overlap_chars
            )
            chunks.extend(new_chunks)
            # Increment by chunk counts, not page counts
            chunk_counter += len(new_chunks)
            current_pages = []

        # FIX: Now an undetected
        # heading gets its own distinct, page-tagged label so it neither
        # merges backward nor collides with a genuinely different undetected
        # page elsewhere in the document.
        current_section = section if section else f"Unlabeled (p.{page['metadata']['page']})"

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
    # **NEW FIX**: Join pages with a newline instead of a space to preserve structure
    combined_text = "\n".join(
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