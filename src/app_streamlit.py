import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.retriever import retrieve
from src.generator_ollama import generate_answer



DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="Multi-Modal RAG System",
    layout="wide",
)

CUSTOM_CSS = """
<style>

body {
    font-family: "Inter", sans-serif;
    background-color: #fafafa;
}

h1, h2, h3 {
    font-weight: 600;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.stButton>button {
    border-radius: 6px;
    padding: 0.5rem 1rem;
    background-color: #2c3e50;
    color: white;
    border: 1px solid #2c3e50;
}

.stButton>button:hover {
    background-color: #3d5166;
    border-color: #3d5166;
}

.dataframe {
    font-size: 14px;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("Swiggy Annual Report AI Assistant")
st.subheader("RAG-based Question Answering on Swiggy Annual Report FY 2023–24")

def build_context_preview(chunks, limit=300):
    rows = []
    images = []

    for h in chunks:
        meta = h.get("meta", {})
        page = meta.get("page", "?")
        typ = meta.get("type", "text")

        preview = h["text"][:limit].replace("\n", " ")
        rows.append({"Page": page, "Type": typ, "Preview": preview})

        img_path = meta.get("path")
        if img_path and Path(img_path).exists():
            images.append(img_path)

    return rows, images

tab1 = st.container()
with tab1:

    st.header("Ask Questions About the Document")

    question = st.text_input(
        "Enter your question",
        placeholder="Example: What was Swiggy's consolidated loss in FY24?"
    )

    run = st.button("Generate Answer")

    if run and question.strip():
        with st.spinner("Retrieving context and generating answer..."):

            hits = retrieve(question, topk=15, rerank_topk=5)
            answer = generate_answer(question, hits)
            rows, images = build_context_preview(hits)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Chunks")
        st.dataframe(rows, use_container_width=True)

        if images:
            st.subheader("Retrieved Images")
            cols = st.columns(3)
            for idx, img_path in enumerate(images):
                cols[idx % 3].image(str(img_path), caption=Path(img_path).name)

