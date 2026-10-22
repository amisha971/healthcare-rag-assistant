import json
import subprocess
import sys
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from app.main import app
from app.api import documents
from app.services.embeddings import EmbeddingError
from app.services.vector_store import StorageError, VectorStore, get_vector_store
from test_documents import client, make_pdf


def embedded(text="Synthetic inspection policy"):
    return [{"chunk_index": 0, "filename": "same.pdf", "page_number": 3,
             "word_count": len(text.split()), "text": text, "embedding": [1.0] + [0.0] * 767}]


def test_real_storage_isolates_same_filename_uploads(tmp_path):
    store = VectorStore(tmp_path / "chroma_data")
    first = store.save_document(embedded())
    second = store.save_document(embedded("Synthetic workflow"))
    assert first["document_id"] != second["document_id"]
    result = store.get_document(first["document_id"])
    assert result["chunk_count"] == 1
    assert result["chunks"][0]["page_number"] == 3
    assert result["chunks"][0]["text"] == "Synthetic inspection policy"
    assert store.get_document(second["document_id"])["chunks"][0]["text"] == "Synthetic workflow"
    assert store.get_document(str(uuid4()))["chunks"] == []


def test_persists_across_processes(tmp_path):
    path = str(tmp_path / "chroma_data")
    write = """
import json, sys
from pathlib import Path
from app.services.vector_store import VectorStore
chunks = json.loads(sys.argv[2])
print(json.dumps(VectorStore(Path(sys.argv[1])).save_document(chunks)))
"""
    saved = subprocess.run([sys.executable, "-c", write, path, json.dumps(embedded())],
                           check=True, capture_output=True, text=True)
    document_id = json.loads(saved.stdout)["document_id"]
    read = """
import json, sys
from pathlib import Path
from app.services.vector_store import VectorStore
print(json.dumps(VectorStore(Path(sys.argv[1])).get_document(sys.argv[2])))
"""
    loaded = subprocess.run([sys.executable, "-c", read, path, document_id],
                            check=True, capture_output=True, text=True)
    assert json.loads(loaded.stdout)["chunks"][0]["text"] == "Synthetic inspection policy"


@pytest.fixture
def local_store(tmp_path):
    store = VectorStore(tmp_path / "chroma_data")
    app.dependency_overrides[get_vector_store] = lambda: store
    yield store
    app.dependency_overrides.pop(get_vector_store, None)


def test_ingest_then_get(local_store, monkeypatch):
    async def fake_embed(chunks):
        return [{**chunk, "embedding": [1.0] + [0.0] * 767} for chunk in chunks]
    monkeypatch.setattr(documents, "embed_chunks", fake_embed)
    response = client.post("/documents/ingest", files={
        "file": ("policy.pdf", make_pdf(["Synthetic policy", "", "Synthetic workflow"]), "application/pdf")
    })
    assert response.status_code == 201
    data = response.json()
    assert data["chunk_count"] == 2
    result = client.get(f"/documents/{data['document_id']}/chunks")
    assert result.status_code == 200
    assert [c["page_number"] for c in result.json()["chunks"]] == [1, 3]
    assert "embedding" not in result.json()["chunks"][0]


def test_missing_document_and_invalid_id(local_store):
    assert client.get(f"/documents/{uuid4()}/chunks").status_code == 404
    assert client.get("/documents/invalid/chunks").status_code == 422


def test_provider_failure_stores_nothing(local_store, monkeypatch):
    monkeypatch.setattr(documents, "embed_chunks", AsyncMock(side_effect=EmbeddingError(429, "Rate limit")))
    response = client.post("/documents/ingest", files={
        "file": ("policy.pdf", make_pdf(["Synthetic policy"]), "application/pdf")
    })
    assert response.status_code == 429
    assert local_store.collection.count() == 0


def test_list_empty_database(local_store):
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_list_groups_chunks_and_keeps_same_filename_uploads_distinct(local_store):
    chunks = embedded()
    chunks.append({**chunks[0], "chunk_index": 1, "page_number": 4})
    first = local_store.save_document(chunks)
    second = local_store.save_document(embedded("Another synthetic upload"))
    response = client.get("/documents")
    assert response.status_code == 200
    expected = [
        {"document_id": first["document_id"], "filename": "same.pdf", "chunk_count": 2},
        {"document_id": second["document_id"], "filename": "same.pdf", "chunk_count": 1},
    ]
    assert response.json() == sorted(expected, key=lambda item: (item["filename"], item["document_id"]))
    assert client.get(f"/documents/{first['document_id']}/chunks").status_code == 200


def test_list_storage_failure_is_safe(local_store, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("private database details")
    monkeypatch.setattr(local_store.collection, "get", fail)
    response = client.get("/documents")
    assert response.status_code == 503
    assert response.json() == {"detail": "Could not list documents from the local vector database."}
