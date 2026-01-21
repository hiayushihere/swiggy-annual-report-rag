## Swiggy Annual Report RAG System

A Retrieval-Augmented Generation (RAG) based AI system for answering questions from the Swiggy Annual Report FY 2023–24 using text, tables, and figures.

### Overview

This project implements a fully local, multimodal RAG pipeline capable of answering natural language questions strictly from the Swiggy Annual Report.

## Dataset / Document Source

Document Name: Swiggy Annual Report FY 2023–24  

Format: PDF  

Source (Publicly Available Official Website): 

[https://www.swiggy.com/Annual-Report-FY-2023-24.pdf/](https://www.swiggy.com/corporate/wp-content/uploads/2024/10/Annual-Report-FY-2023-24-1.pdf)

The report was accessed from Swiggy’s official website and used solely for academic and evaluation purposes in this assignment.


Unlike standard text-only RAG systems, this pipeline handles:

Text extraction

Financial Table parsing

Image extraction

Vector-rendered figure OCR

Chunking and indexing

Hybrid retrieval (semantic + figure/table-aware)

Local LLM answering (Ollama – Llama 3.1)

Streamlit user interface

### Features

1. PDF Ingestion

Extracts text, tables, images, and vector graphics

Renders pages with no embedded images for OCR

OCR on all figures → makes charts searchable


2. Smart Chunking

Sliding-window text chunking

Semantic table summaries

OCR-based figure chunks with auto-detected figure IDs


3. Hybrid Retrieval

Dense embeddings (MiniLM) + FAISS

Direct matching for figure/table queries

Cross-encoder reranking


4. Local LLM Answering

Uses Llama 3.1 via Ollama

Strict context-grounded answers with page citations


5. Streamlit Dashboard

Clean, minimal UI

Ask questions

Preview retrieved chunks

See extracted images/tables

### Installation & Setup

1. Clone the Repository

git clone `https://github.com/hiayushihere/swiggy-annual-report-rag.git`

`cd swiggy-annual-report-rag`

3. Create a Virtual Environment

`python3 -m venv .venv`

`source .venv/bin/activate`

4. Install Requirements

`pip install -r requirements.txt`

5. Install Ollama

Download from:

➡ https://ollama.com/download

After installation, pull the model:

`ollama pull llama3.1`


5. Add Your PDF

Place your PDF inside:
data/Annual-Report-FY-2023-24.pdf
(or replace filename in run_pipeline.py)

6. Run the Ingestion + Indexing Pipeline

This extracts text, tables, images → creates FAISS index.
`python run_pipeline.py`
Outputs appear in:

data/ingested/

data/processed/chunks.jsonl

data/index/faiss.index

#### Run the Streamlit Application
`streamlit run src/app_streamlit.py`

A browser window will open with:

A question box

Retrieved context table

Image previews

LLM answers with page citations
<img width="1453" height="818" alt="Screenshot 2026-01-21 at 11 08 48 AM" src="https://github.com/user-attachments/assets/157657e9-882c-4647-ae1d-654d88c976e5" />
<img width="1428" height="779" alt="Screenshot 2026-01-21 at 11 09 22 AM" src="https://github.com/user-attachments/assets/e8647f4b-baeb-403e-b9ba-0765b3a76324" />





