"""Generate document embeddings through the Gemini REST API."""

import math
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

MODEL = "gemini-embedding-001"
DIMENSIONS = 768
MAX_CHUNKS = 100
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:batchEmbedContents"


class EmbeddingError(Exception):
    """A safe message and HTTP status, without provider details or secrets."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


async def embed_chunks(chunks: list[dict]) -> list[dict]:
    return await _embed_items(chunks, "RETRIEVAL_DOCUMENT")


async def embed_query(question: str) -> list[float]:
    embedded = await _embed_items([{"text": question}], "RETRIEVAL_QUERY")
    return embedded[0]["embedding"]


async def _embed_items(chunks: list[dict], task_type: str) -> list[dict]:
    # Resolve backend/.env independently of the server's working directory.
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or key == "replace_with_your_key":
        raise EmbeddingError(503, "Set GEMINI_API_KEY in backend/.env and restart the server.")
    if not chunks:
        return []
    if len(chunks) > MAX_CHUNKS:
        raise EmbeddingError(422, "This stage supports at most 100 chunks per embedding request.")

    payload = {"requests": [
        {
            "model": f"models/{MODEL}",
            "content": {"parts": [{"text": chunk["text"]}]},
            "taskType": task_type,
            "outputDimensionality": DIMENSIONS,
        }
        for chunk in chunks
    ]}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(URL, headers={"x-goog-api-key": key}, json=payload)
    except httpx.TimeoutException:
        raise EmbeddingError(504, "Embedding service timed out. Try again later.") from None
    except httpx.RequestError:
        raise EmbeddingError(502, "Could not reach the embedding service.") from None

    if response.status_code == 429:
        raise EmbeddingError(429, "Embedding API quota or rate limit reached. Check your quota before retrying.")
    if response.status_code in {401, 403}:
        raise EmbeddingError(503, "Embedding API access denied. Check your API key and permissions.")
    if not response.is_success:
        raise EmbeddingError(502, "Embedding service rejected the request. Check model access and input limits.")
    try:
        vectors = [entry["values"] for entry in response.json()["embeddings"]]
        if len(vectors) != len(chunks):
            raise ValueError("Wrong vector count")
        normalized = []
        for vector in vectors:
            if isinstance(vector, list) and len(vector) != DIMENSIONS:
                raise EmbeddingError(
                    502,
                    f"Embedding dimension mismatch: expected {DIMENSIONS}, received {len(vector)}. "
                    "Check the embedding request configuration.",
                )
            if not isinstance(vector, list) or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in vector
            ):
                raise ValueError("Invalid vector")
            norm = math.sqrt(sum(v * v for v in vector))
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("Invalid norm")
            normalized.append([v / norm for v in vector])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise EmbeddingError(502, "Embedding service returned invalid vectors.") from None
    return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, normalized)]
