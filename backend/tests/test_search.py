import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from app.api import documents
from app.main import app
from app.services import embeddings
from app.services.vector_store import StorageError, VectorStore, get_vector_store
from test_documents import client


def chunk(index, text, vector):
    return {"chunk_index": index, "filename": "same.pdf", "page_number": index + 1,
            "word_count": len(text.split()), "text": text, "embedding": vector}


@pytest.fixture
def searchable(tmp_path, monkeypatch):
    store = VectorStore(tmp_path / "chroma_data")
    x = [1.0, 0.0] + [0.0] * 766
    y = [0.0, 1.0] + [0.0] * 766
    selected = store.save_document([
        chunk(0, "Synthetic inspection checklist", x),
        chunk(1, "Synthetic missing entry escalation", y),
    ])["document_id"]
    other = store.save_document([chunk(0, "Other document", y)])["document_id"]
    app.dependency_overrides[get_vector_store] = lambda: store
    query = AsyncMock(return_value=y)
    monkeypatch.setattr(documents, "embed_query", query)
    yield store, selected, other, query
    app.dependency_overrides.pop(get_vector_store, None)


def test_search_ranks_scoped_passages_and_metadata(searchable):
    _, selected, other, query = searchable
    response = client.post(f"/documents/{selected}/search", json={"question": "  Who handles missing entries?  ", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["match_count"] == 2
    matches = data["matches"]
    assert matches[0]["text"] == "Synthetic missing entry escalation"
    assert matches[0]["page_number"] == 2
    assert matches[0]["filename"] == "same.pdf"
    assert matches[0]["distance"] == pytest.approx(0, abs=1e-6)
    assert matches[0]["distance"] <= matches[1]["distance"]
    assert all(m["document_id"] == selected and m["document_id"] != other for m in matches)
    assert matches[0]["chunk_id"] == f"{selected}:1"
    query.assert_awaited_once_with("Who handles missing entries?")


def test_top_k_is_respected(searchable):
    _, selected, _, _ = searchable
    response = client.post(f"/documents/{selected}/search", json={"question": "missing entries", "top_k": 1})
    assert response.status_code == 200
    assert response.json()["match_count"] == 1


def test_unknown_document_skips_provider(searchable):
    _, _, _, query = searchable
    response = client.post(f"/documents/{uuid4()}/search", json={"question": "policy"})
    assert response.status_code == 404
    query.assert_not_awaited()


@pytest.mark.parametrize("body", [
    {}, {"question": "  "}, {"question": "x" * 2001},
    {"question": "policy", "top_k": 0}, {"question": "policy", "top_k": 11},
    {"question": "policy", "top_k": True}, {"question": "policy", "extra": "ignored?"},
])
def test_invalid_search_inputs(searchable, body):
    _, selected, _, query = searchable
    assert client.post(f"/documents/{selected}/search", json=body).status_code == 422
    query.assert_not_awaited()


def test_search_provider_error(searchable):
    _, selected, _, query = searchable
    query.side_effect = embeddings.EmbeddingError(429, "Rate limit")
    assert client.post(f"/documents/{selected}/search", json={"question": "policy"}).status_code == 429


def test_search_storage_error(searchable, monkeypatch):
    store, selected, _, _ = searchable
    def fail(*args):
        raise StorageError("Could not search the local vector database.")
    monkeypatch.setattr(store, "search", fail)
    assert client.post(f"/documents/{selected}/search", json={"question": "policy"}).status_code == 503


def test_query_embedding_uses_query_task(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(embeddings, "load_dotenv", lambda *args, **kwargs: None)
    post = AsyncMock(return_value=httpx.Response(200, json={"embeddings": [{"values": [2.0] + [0.0] * 767}]}))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    vector = asyncio.run(embeddings.embed_query("Synthetic question"))
    assert vector == [1.0] + [0.0] * 767
    request = post.call_args.kwargs["json"]["requests"][0]
    assert request["taskType"] == "RETRIEVAL_QUERY"
    assert request["outputDimensionality"] == 768
