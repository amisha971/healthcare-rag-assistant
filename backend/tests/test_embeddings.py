import asyncio
import json
import math
from unittest.mock import AsyncMock

import httpx
import pytest

from app.services import embeddings
from test_documents import client, make_pdf

CHUNKS = [
    {"text": "Synthetic policy", "filename": "policy.pdf", "page_number": 1, "chunk_index": 0},
    {"text": "Synthetic workflow", "filename": "policy.pdf", "page_number": 3, "chunk_index": 1},
]


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(embeddings, "load_dotenv", lambda *args, **kwargs: None)
    post = AsyncMock(return_value=httpx.Response(200, json={
        "embeddings": [{"values": [1.0] * 768}, {"values": [2.0] + [0.0] * 767}]
    }))
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    return post


def test_embedding_metadata_order_and_normalization(provider):
    result = asyncio.run(embeddings.embed_chunks(CHUNKS))
    assert [{k: v for k, v in item.items() if k != "embedding"} for item in result] == CHUNKS
    assert all(math.isclose(sum(v*v for v in item["embedding"]), 1.0) for item in result)
    assert result[1]["embedding"][0] == 1.0
    kwargs = provider.call_args.kwargs
    assert kwargs["headers"] == {"x-goog-api-key": "test-key"}
    assert [r["content"]["parts"][0]["text"] for r in kwargs["json"]["requests"]] == [c["text"] for c in CHUNKS]
    for request in kwargs["json"]["requests"]:
        assert request["taskType"] == "RETRIEVAL_DOCUMENT"
        assert request["outputDimensionality"] == 768
        assert "embedContentConfig" not in request


@pytest.mark.parametrize("status,expected", [(429, 429), (401, 503), (403, 503), (400, 502), (500, 502)])
def test_provider_errors_are_sanitized(provider, status, expected):
    provider.return_value = httpx.Response(status, text="private provider detail test-key")
    with pytest.raises(embeddings.EmbeddingError) as caught:
        asyncio.run(embeddings.embed_chunks(CHUNKS))
    assert caught.value.status_code == expected
    assert "test-key" not in str(caught.value)


@pytest.mark.parametrize("error,expected", [(httpx.ReadTimeout("secret"), 504), (httpx.ConnectError("secret"), 502)])
def test_network_errors(provider, error, expected):
    provider.side_effect = error
    with pytest.raises(embeddings.EmbeddingError) as caught:
        asyncio.run(embeddings.embed_chunks(CHUNKS))
    assert caught.value.status_code == expected


@pytest.mark.parametrize("body", [
    {}, {"embeddings": []},
    {"embeddings": [{"values": [0] * 768}] * 2},
    {"embeddings": [{"values": ["bad"] * 768}] * 2},
])
def test_invalid_vectors(provider, body):
    provider.return_value = httpx.Response(200, json=body)
    with pytest.raises(embeddings.EmbeddingError, match="invalid vectors"):
        asyncio.run(embeddings.embed_chunks(CHUNKS))


@pytest.mark.parametrize("length", [1, 3072])
def test_dimension_mismatch_has_safe_diagnostic(provider, length):
    provider.return_value = httpx.Response(200, json={
        "embeddings": [{"values": [1.0] * length}] * 2
    })
    with pytest.raises(embeddings.EmbeddingError, match=f"expected 768, received {length}") as caught:
        asyncio.run(embeddings.embed_chunks(CHUNKS))
    assert caught.value.status_code == 502


def test_missing_key_does_not_call_provider(provider, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(embeddings.EmbeddingError) as caught:
        asyncio.run(embeddings.embed_chunks(CHUNKS))
    assert caught.value.status_code == 503
    provider.assert_not_called()


def test_chunk_cap_does_not_call_provider(provider):
    with pytest.raises(embeddings.EmbeddingError) as caught:
        asyncio.run(embeddings.embed_chunks(CHUNKS * 51))
    assert caught.value.status_code == 422
    provider.assert_not_called()


def test_embed_endpoint(provider):
    response = client.post("/documents/embed", files={
        "file": ("policy.pdf", make_pdf(["Synthetic policy", "", "Synthetic workflow"]), "application/pdf")
    })
    assert response.status_code == 200
    assert response.json()["chunk_count"] == 2
    assert response.json()["dimensions"] == 768
    assert [c["page_number"] for c in response.json()["chunks"]] == [1, 3]


def test_endpoint_reports_rate_limit(provider):
    provider.return_value = httpx.Response(429, text="private provider detail")
    response = client.post("/documents/embed", files={
        "file": ("policy.pdf", make_pdf(["Synthetic policy"]), "application/pdf")
    })
    assert response.status_code == 429
    assert "private" not in json.dumps(response.json())
