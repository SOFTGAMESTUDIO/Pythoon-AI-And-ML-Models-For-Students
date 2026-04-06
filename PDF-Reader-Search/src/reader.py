import fitz

def read_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page in doc:
        text = page.get_text()
        pages.append(text)

    return pages