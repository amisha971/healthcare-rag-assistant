"""PDF upload and extraction endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.services.pdf_extractor import extract_pdf_pages
from app.services.chunker import chunk_pages
from app.services.embeddings import DIMENSIONS, MODEL, EmbeddingError, embed_chunks, embed_query
from app.services.vector_store import StorageError, VectorStore, get_vector_store
from app.schemas.search import SearchRequest
from app.services.answer_generator import GenerationError, generate_answer

router = APIRouter(prefix="/documents", tags=["documents"])
MAX_PDF_BYTES = 10 * 1024 * 1024


@router.get("", description="List saved uploads from the local database. No Gemini call.")
async def list_documents(store: VectorStore = Depends(get_vector_store)):
    try:
        return await run_in_threadpool(store.list_documents)
    except StorageError as exc:
        raise HTTPException(503, str(exc)) from None


@router.post("/extract")
async def extract_document(file: UploadFile = File(...)):
    return await read_pdf_upload(file)


@router.post("/chunks")
async def chunk_document(
    file: UploadFile = File(...),
    chunk_size: int = Query(200, ge=1, le=2000),
    overlap: int = Query(40, ge=0),
):
    if overlap >= chunk_size:
        await file.close()
        raise HTTPException(422, "overlap must be smaller than chunk_size.")
    document = await read_pdf_upload(file)
    chunks = await run_in_threadpool(
        chunk_pages, document["pages"], document["filename"], chunk_size, overlap
    )
    return {
        "filename": document["filename"],
        "extracted_page_count": document["extracted_page_count"],
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


@router.post("/embed", description="Sends uploaded text to Gemini. Use synthetic PDFs only.")
async def embed_document(file: UploadFile = File(...)):
    document = await read_pdf_upload(file)
    chunks = await run_in_threadpool(chunk_pages, document["pages"], document["filename"])
    try:
        embedded = await embed_chunks(chunks)
    except EmbeddingError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    return {
        "filename": document["filename"],
        "chunk_count": len(embedded),
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "chunks": embedded,
    }


@router.post("/ingest", status_code=201, description="Embed and store a synthetic PDF. Sends text to Gemini.")
async def ingest_document(
    file: UploadFile = File(...), store: VectorStore = Depends(get_vector_store)
):
    document = await read_pdf_upload(file)
    chunks = await run_in_threadpool(chunk_pages, document["pages"], document["filename"])
    try:
        embedded = await embed_chunks(chunks)
        return await run_in_threadpool(store.save_document, embedded)
    except EmbeddingError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    except StorageError as exc:
        raise HTTPException(503, str(exc)) from None


@router.get("/{document_id}/chunks")
async def get_stored_chunks(document_id: UUID, store: VectorStore = Depends(get_vector_store)):
    try:
        document = await run_in_threadpool(store.get_document, str(document_id))
    except StorageError as exc:
        raise HTTPException(503, str(exc)) from None
    if not document["chunks"]:
        raise HTTPException(404, "Document not found.")
    return document


@router.post("/{document_id}/search", description="Search one document. Sends the question to Gemini.")
async def search_document(
    document_id: UUID, request: SearchRequest, store: VectorStore = Depends(get_vector_store)
):
    try:
        # Check existence before spending embedding quota on the question.
        document = await run_in_threadpool(store.get_document, str(document_id))
        if not document["chunks"]:
            raise HTTPException(404, "Document not found.")
        vector = await embed_query(request.question)
        matches = await run_in_threadpool(
            store.search, str(document_id), vector, min(request.top_k, document["chunk_count"])
        )
    except EmbeddingError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    except StorageError as exc:
        raise HTTPException(503, str(exc)) from None
    return {"document_id": str(document_id), "match_count": len(matches), "matches": matches}


@router.post("/{document_id}/ask", description="Answer from retrieved passages. Sends question and evidence to Gemini.")
async def ask_document(
    document_id: UUID, request: SearchRequest, store: VectorStore = Depends(get_vector_store)
):
    retrieved = await search_document(document_id, request, store)
    try:
        answer = await generate_answer(request.question, retrieved["matches"])
    except GenerationError as exc:
        raise HTTPException(exc.status_code, str(exc)) from None
    return {"document_id": str(document_id), **answer}


async def read_pdf_upload(file: UploadFile):
    """Share upload validation and extraction between both endpoints."""
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Upload a file with a .pdf extension.")
        if file.content_type not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(400, "Upload a PDF file.")

        # Read at most one byte beyond the limit to detect oversized uploads.
        pdf_bytes = await file.read(MAX_PDF_BYTES + 1)
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise HTTPException(413, "PDF exceeds the 10 MB limit.")
        if not pdf_bytes:
            raise HTTPException(400, "The uploaded file is empty.")
        if not pdf_bytes.startswith(b"%PDF-"):
            raise HTTPException(400, "The file does not have a valid PDF header.")
        try:
            pages = await run_in_threadpool(extract_pdf_pages, pdf_bytes)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not pages:
            raise HTTPException(422, "No extractable text found. Scanned PDFs require OCR.")
        return {
            "filename": file.filename,
            "extracted_page_count": len(pages),
            "pages": pages,
        }
    finally:
        await file.close()
