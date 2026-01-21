
import re

ROMAN = r"(?:M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))" 

FIG_PATTERNS = [

    rf"(Figure|FIGURE|Fig|Fig\.)\s*{ROMAN}\s*[\.\-\s]?\s*\d+(\.\d+)?",
    r"(Figure|FIGURE|Fig|Fig\.)\s*\d+(\.\d+)?",
    r"(Figure\s+[A-Za-z]+\.\s*[0-9]+)",
    r"Figure\s+[A-Za-z0-9\.\-]+"
]
FIG_REGEX = re.compile("|".join(FIG_PATTERNS), flags=re.IGNORECASE)

def semantic_table_text(table, page, idx):
    rows_as_text = [
        "\t".join([str(c) if c is not None else "" for c in row])
        for row in table
    ]
    header_row = table[0] if table else []
    header_labels = []
    for h in header_row:
        if h is None:
            continue
        header_labels.append(str(h).strip())
    semantic_desc = ""
    if header_labels:
        semantic_desc = "This table includes fields: " + ", ".join(header_labels) + ". "
    tbl_text = f"TABLE (page {page}, table {idx}): {semantic_desc}\n" + "\n".join(rows_as_text)
    return tbl_text

def extract_figure_tag(ocr_text, surrounding_text=""):
    text_to_scan = ""
    if ocr_text:
        text_to_scan += ocr_text + "\n"
    if surrounding_text:
        text_to_scan += surrounding_text + "\n"
    if not text_to_scan.strip():
        return ""
    m = FIG_REGEX.search(text_to_scan)
    if not m:
        simple = re.search(r"Figure\s+[IVXLCivxlc0-9]+\s*[^\w\d]\s*\d+", text_to_scan)
        if simple:
            return simple.group(0)
        return ""
    return m.group(0).strip()

def chunk_text(text, page, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    cid = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        ctext = " ".join(chunk_words)
        chunks.append({
            "id": f"{page}-txt-{cid}",
            "text": ctext,
            "meta": {"page": page, "type": "text"}
        })
        cid += 1
        i += chunk_size - overlap
    return chunks


def page_to_chunks(page_obj, chunk_size=300, overlap=50):
    """
    Input:
        page_obj with keys:
        'page', 'text', 'tables', 'images'
        images: list of {'ocr','path'}

    Returns:
        list of chunks with enriched metadata for Swiggy Annual Report RAG
    """

    chunks = []
    page = page_obj.get("page")
    page_text = page_obj.get("text", "") or ""

    SOURCE_NAME = "Swiggy Annual Report FY 2023-24"

   
    # TEXT CHUNKS
 
    if page_text.strip():
        text_chunks = chunk_text(page_text, page, chunk_size, overlap)

        for c in text_chunks:
            fig_tag = extract_figure_tag("", c["text"])

            c["meta"] = {
                "page": page,
                "type": "text",
                "source": SOURCE_NAME
            }

            if fig_tag:
                c["meta"]["figure_tag"] = fig_tag

            chunks.append(c)

    # TABLE CHUNKS

    for t_i, table in enumerate(page_obj.get("tables", []) or []):
        try:
            tbl_text = semantic_table_text(table, page, t_i)

            fig_tag = extract_figure_tag("", tbl_text)

            meta = {
                "page": page,
                "type": "table",
                "source": SOURCE_NAME
            }

            if fig_tag:
                meta["figure_tag"] = fig_tag

            chunks.append({
                "id": f"{page}-tbl-{t_i}",
                "text": tbl_text,
                "meta": meta
            })

        except Exception:
            continue


    # IMAGE OCR CHUNKS

    images = page_obj.get("images", []) or []

    for img_i, img in enumerate(images):
        ocr_text = img.get("ocr", "") or ""
        fig_tag = extract_figure_tag(ocr_text, page_text)

        if ocr_text.strip():
            txt = (
                f"{fig_tag} IMAGE_OCR (page {page}): {ocr_text}"
                if fig_tag else
                f"IMAGE_OCR (page {page}): {ocr_text}"
            )

            meta = {
                "page": page,
                "type": "image",
                "path": img.get("path"),
                "source": SOURCE_NAME
            }

            if fig_tag:
                meta["figure_tag"] = fig_tag

            chunks.append({
                "id": f"{page}-imgocr-{img_i}",
                "text": txt,
                "meta": meta
            })

        else:
            meta = {
                "page": page,
                "type": "image",
                "path": img.get("path"),
                "source": SOURCE_NAME
            }

            chunks.append({
                "id": f"{page}-img-{img_i}",
                "text": f"IMAGE (page {page}) - no OCR text",
                "meta": meta
            })

    return chunks
