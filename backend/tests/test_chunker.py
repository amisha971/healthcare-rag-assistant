import pytest

from app.services.chunker import chunk_pages
from test_documents import client, make_pdf


def test_overlap_and_final_partial_chunk():
    words = "one two three four five six seven eight nine ten eleven".split()
    chunks = chunk_pages([{"page_number": 3, "text": " ".join(words)}], "policy.pdf", 5, 2)
    assert [c["text"].split() for c in chunks] == [words[:5], words[3:8], words[6:11]]
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2]
    assert all(c["page_number"] == 3 and c["filename"] == "policy.pdf" for c in chunks)
    assert all(c["word_count"] == 5 for c in chunks)


def test_page_boundaries_and_short_pages():
    chunks = chunk_pages([
        {"page_number": 1, "text": "one two three four five six"},
        {"page_number": 3, "text": "seven\n eight"},
        {"page_number": 4, "text": "  "},
    ], "policy.pdf", 5, 0)
    assert [(c["page_number"], c["text"], c["word_count"]) for c in chunks] == [
        (1, "one two three four five", 5), (1, "six", 1), (3, "seven\n eight", 2),
    ]
    assert chunk_pages([], "empty.pdf") == []


def test_heading_and_paragraph_breaks_survive_overlapping_windows():
    chunks = chunk_pages([{"page_number": 3, "text": "Equipment Workflow\n\nReport missing entries now"}], "policy.pdf", 4, 1)
    assert chunks[0]["text"] == "Equipment Workflow\n\nReport missing"
    assert chunks[1]["text"] == "missing entries now"
    assert [c["word_count"] for c in chunks] == [4, 3]


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)])
def test_invalid_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([], "policy.pdf", size, overlap)


def test_chunks_endpoint():
    response = client.post("/documents/chunks?chunk_size=3&overlap=1", files={
        "file": ("policy.pdf", make_pdf(["one two three four five", "", "six seven"]), "application/pdf")
    })
    assert response.status_code == 200
    data = response.json()
    assert data["extracted_page_count"] == 2
    assert data["chunk_count"] == 3
    assert [c["text"] for c in data["chunks"]] == ["one two three", "three four five", "six seven"]
    assert [c["page_number"] for c in data["chunks"]] == [1, 1, 3]


@pytest.mark.parametrize("query", ["chunk_size=0", "chunk_size=2001", "overlap=-1", "chunk_size=5&overlap=5"])
def test_endpoint_rejects_invalid_settings(query):
    response = client.post(f"/documents/chunks?{query}", files={
        "file": ("policy.pdf", make_pdf(["synthetic policy"]), "application/pdf")
    })
    assert response.status_code == 422


def test_chunks_rejects_invalid_pdf():
    response = client.post("/documents/chunks", files={"file": ("fake.pdf", b"hello", "application/pdf")})
    assert response.status_code == 400
