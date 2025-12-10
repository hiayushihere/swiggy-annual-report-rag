# src/embed_index.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(TEXT_EMBED_MODEL)
    return _model

def build_index(chunks, index_path=INDEX_DIR / "faiss.index", meta_path=INDEX_DIR / "meta.pkl"):
    """
    chunks: list of {'id','text','meta'}
    Builds FAISS index (Inner-product on normalized vectors).
    """
    model = get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # normalize
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved index ({len(chunks)} vectors) to {index_path}")

def load_index(index_path=INDEX_DIR / "faiss.index", meta_path=INDEX_DIR / "meta.pkl"):
    index = faiss.read_index(str(index_path))
    chunks = pickle.load(open(meta_path, "rb"))
    return index, chunks

def search(query, k=5):
    model = get_model()
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    index, chunks = load_index()
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        hits = chunks[idx]
        results.append({"score": float(score), **hits})
    return results
