"""Extract PDF text without storing uploaded documents on disk."""

import pymupdf


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    """Return nonempty pages with their original, one-based page numbers.

    Raises ValueError for unreadable or password-protected PDFs.
    Scanned pages require OCR, which this first stage does not provide.
    """
    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported.")
            pages = []
            for page_number, page in enumerate(document, start=1):
                text = page.get_text().strip()
                if text:
                    pages.append({"page_number": page_number, "text": text})
            return pages
    except RuntimeError as exc:
        raise ValueError("The uploaded file could not be read as a PDF.") from exc

