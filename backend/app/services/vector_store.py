"""Persistent local storage for already embedded chunks."""

from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import chromadb
from chromadb.config import Settings

from app.services.embeddings import DIMENSIONS, MODEL

DB_PATH = Path(__file__).resolve().parents[2] / "chroma_data"
COLLECTION_NAME = "healthcare_gemini_001_768_v1"


class StorageError(Exception):
    pass


class VectorStore:
    def __init__(self, path: Path):
        self.client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        # Embeddings come from Gemini; Chroma must not generate its own.
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": MODEL, "dimensions": DIMENSIONS},
        )

    def save_document(self, chunks: list[dict]) -> dict:
        if not chunks:
            raise ValueError("Cannot store an empty document.")
        document_id = str(uuid4())
        try:
            self.collection.add(
                ids=[f"{document_id}:{chunk['chunk_index']}" for chunk in chunks],
                embeddings=[chunk["embedding"] for chunk in chunks],
                documents=[chunk["text"] for chunk in chunks],
                metadatas=[{
                    "document_id": document_id,
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "word_count": chunk["word_count"],
                } for chunk in chunks],
            )
        except Exception:
            raise StorageError("Could not save document to the local vector database.") from None
        return {
            "document_id": document_id,
            "filename": chunks[0]["filename"],
            "chunk_count": len(chunks),
            "model": MODEL,
            "dimensions": DIMENSIONS,
        }

    def list_documents(self) -> list[dict]:
        try:
            # Text and vectors are unnecessary for the upload list.
            result = self.collection.get(include=["metadatas"])
        except Exception:
            raise StorageError("Could not list documents from the local vector database.") from None
        documents = {}
        for metadata in result["metadatas"]:
            document_id = metadata["document_id"]
            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "filename": metadata["filename"],
                    "chunk_count": 0,
                }
            documents[document_id]["chunk_count"] += 1
        return sorted(documents.values(), key=lambda item: (item["filename"], item["document_id"]))

    def get_document(self, document_id: str) -> dict:
        try:
            result = self.collection.get(
                where={"document_id": document_id}, include=["documents", "metadatas"]
            )
        except Exception:
            raise StorageError("Could not read the local vector database.") from None
        chunks = [
            {**metadata, "text": text}
            for metadata, text in zip(result["metadatas"], result["documents"])
        ]
        chunks.sort(key=lambda chunk: chunk["chunk_index"])
        return {"document_id": document_id, "chunk_count": len(chunks), "chunks": chunks}


    def search(self, document_id: str, query_embedding: list[float], top_k: int) -> list[dict]:
        try:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                where={"document_id": document_id},
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            raise StorageError("Could not search the local vector database.") from None
        return [
            {**metadata, "chunk_id": chunk_id, "text": text, "distance": distance}
            for chunk_id, metadata, text, distance in zip(
                result["ids"][0], result["metadatas"][0],
                result["documents"][0], result["distances"][0],
            )
        ]


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore(DB_PATH)
