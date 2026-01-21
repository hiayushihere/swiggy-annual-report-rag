
import re
import json
from pathlib import Path
from sentence_transformers import CrossEncoder
from src.embed_index import search

# Load chunks.jsonl once

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
ALL_CHUNKS = []

if CHUNKS_PATH.exists():
    with CHUNKS_PATH.open() as f:
        for line in f:
            try:
                ALL_CHUNKS.append(json.loads(line))
            except:
                pass



# Cross Encoder Loading

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross = None


def get_cross_encoder():
    global _cross
    if _cross is None:
        try:
            _cross = CrossEncoder(_CROSS_ENCODER_MODEL)
        except Exception:
            _cross = None
    return _cross



# FIGURE/TABLE ID extraction

def normalize_figure_token(s):
    """Normalize variations like 'Figure III . 5' → 'figureiii.5' """
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"[^\w\.\-]", "", s)
    return s.lower()


def extract_fig_ids_from_query(q):
    """
    Extract 'Figure III.5', 'Fig 3.2', 'III.5', etc.
    Returns canonical forms such as: ['iii.5']
    """
    found = set()

    # patterns like: Figure III.5
    for m in re.finditer(r"(?:Figure|FIGURE|Fig|Fig\.)\s*([IVXLCivxlc0-9\.\- ]+)", q):
        tok = normalize_figure_token(m.group(1))
        if tok:
            found.add(tok)

    # plain references like III.5 or 3.5
    for m in re.finditer(r"\b([IVXLCivxlc0-9]+(?:\.[0-9]+)+)\b", q):
        tok = normalize_figure_token(m.group(1))
        if tok:
            found.add(tok)

    return list(found)


# Expand query for FAISS

def expand_query(q):
    expansions = {q}
    qlow = q.lower()

    # Swiggy business terminology expansion
    if "revenue" in qlow or "sales" in qlow:
        expansions.update(["net sales", "income from operations", "total income"])

    if "loss" in qlow or "profit" in qlow:
        expansions.update(["net loss", "profit after tax", "PAT", "loss after tax"])

    if "instamart" in qlow:
        expansions.update(["quick commerce", "dark stores", "Instamart business"])

    if "food delivery" in qlow:
        expansions.update(["restaurant partners", "orders", "AOV"])

    if "users" in qlow:
        expansions.update(["monthly transacting users", "MTU"])

    if "board" in qlow or "director" in qlow:
        expansions.update(["board of directors", "independent directors", "nominee directors"])

    if "subsidiary" in qlow:
        expansions.update(["Scootsy", "Supr Infotech", "Lynks Logistics"])

    if "financial" in qlow:
        expansions.update(["standalone financial", "consolidated financial"])

    # Attach figure/table IDs
    fig_ids = extract_fig_ids_from_query(q)
    for fid in fig_ids:
        expansions.add(fid)
        expansions.add("figure " + fid)
        expansions.add("table " + fid)

    return list(expansions)





def direct_figure_table_matches(query):
    """
    Directly search metadata for matching figure_tag or table identifiers.
    This fixes ALL evaluation failures involving figures/tables.
    """
    qlow = query.lower()
    fig_ids = extract_fig_ids_from_query(query)

    matches = []

    for c in ALL_CHUNKS:
        meta = c.get("meta", {})
        t = meta.get("type", "")

        # direct table questions like "Table 1"
        if "table" in qlow:
            if "table" in t:
                # check page-level match or id
                if any(fid in c["id"].lower() for fid in fig_ids):
                    matches.append(c)

       
        tag = meta.get("figure_tag", "")
        if tag:
            tag_norm = normalize_figure_token(tag)
            for fid in fig_ids:
                if fid in tag_norm:
                    matches.append(c)

   
    uniq = {}
    for m in matches:
        uniq[m["id"]] = m
    return list(uniq.values())



# Main Retrieval Function

def retrieve(query, topk=15, rerank_topk=5):
    # 1) Direct figure/table matching FIRST (very high priority)
    direct_hits = direct_figure_table_matches(query)

    # 2) Semantic FAISS search across expanded queries
    semantic_candidates = []
    expanded = expand_query(query)

    for eq in expanded:
        try:
            semantic_candidates.extend(search(eq, k=topk))
        except:
            pass

    # merge & dedupe
    merged = {}

    for h in direct_hits:
        merged[h["id"]] = h

    for h in semantic_candidates:
        merged[h["id"]] = h

    hits = list(merged.values())

    # 3) Rerank using cross encoder 
    cross = get_cross_encoder()
    if cross and len(hits) > 1:
        pairs = [(query, h["text"]) for h in hits]
        try:
            scores = cross.predict(pairs)
            for h, score in zip(hits, scores):
                h["rerank_score"] = float(score)

            hits = sorted(hits, key=lambda x: x.get("rerank_score", 0), reverse=True)
        except:
            pass

    # 4) final 
    return hits[:rerank_topk]
