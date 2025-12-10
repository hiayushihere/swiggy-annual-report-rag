# src/generator_ollama.py
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1"  

PROMPT_TEMPLATE = """
You are a strict document-reading assistant. Use ONLY the provided context to answer.

Rules:
- NEVER use outside knowledge.
- Provide a detailed, well-structured explanation.
- Include all relevant details from the context.
- Cite page numbers in parentheses, e.g. (page 4).
- If the answer is not in the context, reply exactly: "I don't know".

Context:
{context}

Question:
{question}

Answer (with citations and full detail):
"""


def call_ollama(prompt, timeout=60):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "num_predict": 2048,       
        "temperature": 0.2,        
        "top_p": 0.9,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict):
        if "response" in data:
            return data["response"]
        if "text" in data:
            return data["text"]
        return str(data)
    return str(data)

def generate_answer(question, retrieved_chunks):
    ctx_parts = []
    for c in retrieved_chunks:
        txt = c["text"]
        if len(txt) > 1200:
            txt = txt[:1200] + " ... "
        page = c.get("meta", {}).get("page", "?")
        ctx_parts.append(f"[page {page}] {txt}")
    context = "\n\n".join(ctx_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return call_ollama(prompt)
