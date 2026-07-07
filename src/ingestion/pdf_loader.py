import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz


def normalize_text(text: str) -> str:
    """
    Normalizes whitespace in the extracted text.
    Replaces newlines with spaces, collapses multiple spaces,
    and strips leading/trailing whitespace.
    """
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def detect_section_heading(raw_text: str) -> Optional[str]:
    """
    Attempts to detect a section heading from the raw page text using simple heuristics.
    Looks for numbered headings (e.g., '2.3 Blood Pressure') or large uppercase titles.
    """
    lines = raw_text.split('\n')

    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue

        # Heuristic 1: Numbered section headings
        if re.match(r'^\d+\.\d+(?:\.\d+)?\s*[:\-]?\s+[A-Z]', line):
            return line

        # Heuristic 2: All uppercase titles
        if re.match(r'^[A-Z][A-Z\s\-]{3,}$', line):
            return line

    return None


def process_page(
    page: fitz.Page,
    filename: str,
    document_id: str,
    page_num: int
) -> Optional[Dict[str, Any]]:
    """
    Extracts text from a single PDF page, detects headings, normalizes the text,
    and formats the output dictionary. Skips pages with meaningless text.
    """
    raw_text = page.get_text()

    if not raw_text or len(raw_text.strip()) < 30:
        return None

    section = detect_section_heading(raw_text)
    clean_text = normalize_text(raw_text)

    if not clean_text:
        return None

    return {
        "text": clean_text,
        "metadata": {
            "source": filename,
            "document_id": document_id,
            "page": page_num,
            "section": section
        }
    }


def load_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """
    Opens a single PDF file and iterates through its pages to extract text and metadata.
    """
    pages_data = []

    try:
        with fitz.open(file_path) as doc:
            for i, page in enumerate(doc):
                page_data = process_page(
                    page,
                    file_path.name,
                    file_path.stem,
                    i + 1
                )

                if page_data:
                    pages_data.append(page_data)

    except Exception as e:
        raise RuntimeError(f"Failed processing {file_path}") from e

    return pages_data


def process_directory(directory_path: Path) -> tuple[List[Dict[str, Any]], int]:
    """
    Recursively finds and processes all PDFs in the given directory.
    """
    all_extracted_pages = []
    pdf_files = list(directory_path.rglob("*.pdf"))

    for pdf_path in pdf_files:
        document_pages = load_pdf(pdf_path)
        all_extracted_pages.extend(document_pages)

    return all_extracted_pages, len(pdf_files)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_data_dir = base_dir / "data" / "raw"

    if not raw_data_dir.exists():
        print(f"Directory {raw_data_dir} not found. Creating dummy directory for demo...")
        raw_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for PDFs in: {raw_data_dir}")

    extracted_data, pdf_count = process_directory(raw_data_dir)

    print("\n--- Extraction Summary ---")
    print(f"Total PDFs processed: {pdf_count}")
    print(f"Total pages extracted: {len(extracted_data)}")

    if extracted_data:
        print("\n--- First Extracted Record ---")
        import json
        print(json.dumps(extracted_data[0], indent=4))
    else:
        print("\nNo text extracted. (Ensure there are valid PDFs in data/raw/)")