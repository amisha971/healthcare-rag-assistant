# Healthcare RAG Document Assistant

Stages 1–9: PDF extraction, chunking, Gemini embeddings, persistent storage, semantic retrieval, cited answers, saved-document discovery, an evaluation runner, and a React interface. Use synthetic documents only.
This is a document retrieval and summarization project, not a diagnostic or treatment tool.

## Run on macOS

Start in the backend directory:

```bash
cd /Users/amisha/healthcare-rag-assistant/backend
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/create_sample_pdf.py
uvicorn app.main:app --reload
```

For a fresh checkout, run `python3 -m venv .venv` before activating the environment.

Open http://127.0.0.1:8000/health and expect `{"status":"ok"}`.
Open http://127.0.0.1:8000/docs, expand `POST /documents/extract`, choose
**Try it out**, select `backend/samples/synthetic-policy.pdf`, and click **Execute**.

Expect HTTP 200 with `filename`, `extracted_page_count: 2`, and a `pages` list
containing page numbers 1 and 3. Page 2 is blank; page 3 keeps its original number
so later citations can point to the correct page.

## How the files interact

`app/main.py` creates the FastAPI application and registers the documents router.
`app/api/documents.py` validates the filename, content type, size, and PDF header.
It passes the bytes to `app/services/pdf_extractor.py`, which opens the PDF in
memory and extracts text page by page. No uploaded PDF is saved by the endpoint.

`enumerate(document, start=1)` uses human-readable page numbering.
`if text` skips pages without selectable text. The `with` block closes the PDF
even when extraction fails. `run_in_threadpool` runs synchronous PDF extraction
outside FastAPI's async event loop.

The 10 MB limit bounds what the endpoint reads into memory after multipart parsing;
it is not a server-level network request limit. A deployed service will also need
request limits at the server or proxy layer.

## Test

In a second terminal, from the backend directory:

```bash
cd /Users/amisha/healthcare-rag-assistant/backend
source .venv/bin/activate
python -m pytest -q
```

Verified locally: 96 tests passed (embedding provider calls are mocked). Six dependency
warnings appeared: five from PyMuPDF and one about urllib3/LibreSSL; no test failed. Tests use generated synthetic PDFs and cover original
page numbering, invalid uploads, the size limit, blank PDFs, password protection,
missing uploads, and the health endpoint.

HTTP 400 means an empty, invalid, unsupported, or password-protected upload.
HTTP 413 means the file exceeds 10 MB. HTTP 422 means a missing upload or no
extractable text; scanned PDFs need OCR, which is not implemented in this stage.
Review the JSON `detail` field for the specific explanation.

`POST /documents/embed` sends text to Gemini and requires `GEMINI_API_KEY`.
Extraction and chunking do not require API keys. `POST /documents/ingest` embeds
and stores chunks locally; `GET /documents/{document_id}/chunks` reads them back.
`POST /documents/{document_id}/search` retrieves matching passages within one document.
`POST /documents/{document_id}/ask` generates cited answers or abstains from retrieved evidence.
The React frontend lives in `frontend/`; see the stage-nine guide for local setup. Generation provider calls are also mocked in tests. Local environment files, PDFs, virtual
environments, and generated database directories are excluded from Git.

Complete stage-one file contents are included in [the walkthrough](docs/stage-1.md).
That walkthrough records the stage-one snapshot. See the [stage-two walkthrough](docs/stage-2.md)
for current router code, complete chunking code, and Swagger testing instructions
for `POST /documents/chunks` (default: 200 words, 40-word overlap).
See the [stage-three walkthrough](docs/stage-3.md) for API key configuration,
embedding concepts, full current router code, and live Swagger verification.

The [stage-four walkthrough](docs/stage-4.md) covers persistent storage, current
router code, document IDs, and verification across server restarts.

The [stage-five walkthrough](docs/stage-5.md) explains semantic search, top-k,
cosine distance, current source code, and Swagger verification.

The [stage-six walkthrough](docs/stage-6.md) explains grounded answer generation,
citation checks, abstention, prompt injection limitations, complete code, and live tests.

`GET /documents` lists saved uploads and their IDs without a cloud call.
See the [stage-seven walkthrough](docs/stage-7.md) for complete code and testing.

The [stage-eight walkthrough](docs/stage-8.md) provides a six-case synthetic
evaluation dataset, run commands, metric definitions, limitations, and human review
instructions. Live aggregate results are pending; automated tests are not model
accuracy measurements. Token usage is explicitly unmeasured in current reports.

The [evaluation follow-up](docs/stage-8-follow-up.md) preserves line breaks in new
chunks and versions a refined citation heuristic. Re-ingest to apply formatting
changes; old uploads and evaluation reports remain unchanged.

The [stage-nine walkthrough](docs/stage-9.md) covers the React interface, Node
setup, frontend/backend startup, full source, manual verification, and build limitations.
Frontend build verified; visual/browser verification remains pending.
