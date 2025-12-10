from pathlib import Path
import json
from src.ingest import extract_pdf
from src.chunker import page_to_chunks
from src.embed_index import build_index

ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = ROOT / "data"
PDF_PATH = DATA_DIR / "qatar_test_doc.pdf"

OUT_PROCESSED = DATA_DIR / "processed"
OUT_PROCESSED.mkdir(exist_ok=True, parents=True)

print(f"Using PDF: {PDF_PATH}")
if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

print("Extracting PDF...")
pages = extract_pdf(PDF_PATH)

all_chunks = []
for p in pages:
    chunks = page_to_chunks(p)
    all_chunks.extend(chunks)

print(f"Created {len(all_chunks)} chunks.")

CHUNKS_PATH = OUT_PROCESSED / "chunks.jsonl"
with open(CHUNKS_PATH, "w") as f:
    for c in all_chunks:
        f.write(json.dumps(c) + "\n")

print(f"Saved chunks to: {CHUNKS_PATH}")

loaded_chunks = []
with open(CHUNKS_PATH, "r") as f:
    for line in f:
        loaded_chunks.append(json.loads(line))

print(f"Loaded {len(loaded_chunks)} chunks for indexing...")


build_index(loaded_chunks)
print("Index built successfully.")
