import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_pdf(texts):
    with pymupdf.open() as doc:
        for text in texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        return doc.tobytes()


def upload(data, filename="synthetic.pdf", content_type="application/pdf"):
    return client.post("/documents/extract", files={"file": (filename, data, content_type)})


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_extract_preserves_original_page_numbers():
    response = upload(make_pdf(["Fictional hospital policy", "", "Synthetic discharge workflow"]))
    assert response.status_code == 200
    assert response.json() == {
        "filename": "synthetic.pdf",
        "extracted_page_count": 2,
        "pages": [
            {"page_number": 1, "text": "Fictional hospital policy"},
            {"page_number": 3, "text": "Synthetic discharge workflow"},
        ],
    }


@pytest.mark.parametrize("data,filename,mime,status", [
    (b"", "empty.pdf", "application/pdf", 400),
    (b"hello", "fake.pdf", "application/pdf", 400),
    (b"%PDF-1.7\ninvalid", "broken.pdf", "application/pdf", 400),
    (b"hello", "text.txt", "text/plain", 400),
    (b"hello", "fake.pdf", "image/png", 400),
    (b"%PDF-" + b"x" * (10 * 1024 * 1024), "large.pdf", "application/pdf", 413),
])
def test_invalid_upload(data, filename, mime, status):
    assert upload(data, filename, mime).status_code == status


def test_blank_pdf():
    assert upload(make_pdf([""])).status_code == 422


def test_password_protected_pdf():
    with pymupdf.open(stream=make_pdf(["Synthetic policy"]), filetype="pdf") as doc:
        data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="example")
    assert upload(data).status_code == 400


def test_missing_upload():
    assert client.post("/documents/extract").status_code == 422
