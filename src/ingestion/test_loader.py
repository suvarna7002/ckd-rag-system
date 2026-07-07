import json
from pathlib import Path
from pdf_loader import process_directory

base_dir = Path(__file__).resolve().parent.parent.parent
raw_data_dir = base_dir / "data" / "raw"
processed_file = base_dir / "data" / "processed" / "extracted_pages.json"

documents, pdf_count = process_directory(raw_data_dir)

with open(processed_file, "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

print(f"Saved {len(documents)} pages to {processed_file}")