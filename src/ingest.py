
import pdfplumber
import fitz
from pathlib import Path
import pytesseract
from PIL import Image
import json

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = DATA_DIR / "ingested"
OUT_DIR.mkdir(exist_ok=True, parents=True)

IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(exist_ok=True, parents=True)


def extract_pdf(pdf_path):
    pages_output = []

    pdf = pdfplumber.open(pdf_path)

    for page_idx, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        tables = page.extract_tables() or []

        pages_output.append({
            "page": page_idx,
            "text": text,
            "tables": tables,
            "images": []      
        })

    pdf.close()

    doc = fitz.open(str(pdf_path))

    for p in range(doc.page_count):
        page = doc[p]
        image_list = page.get_images(full=True)

        if len(image_list) > 0:
            for img_idx, img_data in enumerate(image_list):
                xref = img_data[0]
                pix = fitz.Pixmap(doc, xref)

                img_path = IMG_DIR / f"page{p+1}_img{xref}.png"
                pix.save(img_path)

                try:
                    ocr_text = pytesseract.image_to_string(Image.open(img_path))
                except Exception:
                    ocr_text = ""

                pages_output[p]["images"].append({
                    "ocr": ocr_text,
                    "path": str(img_path)
                })


        if len(image_list) == 0:
            render_path = IMG_DIR / f"page{p+1}_render.png"
            pix = page.get_pixmap(dpi=200)
            pix.save(render_path)

            try:
                ocr_text = pytesseract.image_to_string(Image.open(render_path))
            except Exception:
                ocr_text = ""

            pages_output[p]["images"].append({
                "ocr": ocr_text,
                "path": str(render_path),
                "rendered": True
            })

    return pages_output


if __name__ == "__main__":
    pdf_path = DATA_DIR / "Annual-Report-FY-2023-24.pdf"
    pages_output = extract_pdf(pdf_path)

    out_file = OUT_DIR / "pages.json"
    with open(out_file, "w") as f:
        json.dump(pages_output, f, indent=2)

    print(f"Extracted {len(pages_output)} pages with text, tables, and images.")
