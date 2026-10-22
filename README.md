# Healthcare RAG Document Assistant

A full-stack application for asking questions about synthetic healthcare PDFs.
Upload a document, select it from your library, and receive answers grounded in
retrieved passages, with filenames, original page numbers, and supporting quotes.
The assistant abstains when retrieved evidence is insufficient.

Built with **Python, FastAPI, PyMuPDF, Gemini, ChromaDB, React, and Vite**.
This is a document retrieval and summarization project, not a diagnostic or
treatment recommendation system. Use synthetic documents only.

## Features

- PDF validation and text extraction with original page numbers.
- Page-based chunks with a 200-word size, 40-word overlap, and preserved line breaks.
- Normalized 768-dimensional Gemini embeddings stored in persistent local ChromaDB.
- Document-scoped semantic retrieval and a saved-document library.
- Generated answers with source citations or explicit abstention.
- A responsive React interface for uploads, questions, and source inspection.
- Automated tests and a small synthetic evaluation dataset.

## Architecture

```text
Upload:   PDF → extract pages → chunk text → Gemini embeddings → ChromaDB
Question: question → query embedding → document-scoped retrieval
          → Gemini generation → citation validation → answer or abstention
```

The backend supplies retrieved passages to Gemini at question time; it does not
fine-tune the model. Citation metadata comes from stored records. The backend
checks source IDs and quote presence, but these checks do not prove that a quote
logically supports every generated claim.

## Local setup

Prerequisites: Git, Python with venv support, Node.js 22, npm, and a Gemini API key
with embedding and generation access. Verified locally with Python 3.9 and Node 22.

### Backend

From the repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Create `backend/.env` in your editor with:

```dotenv
GEMINI_API_KEY=your_api_key_here
GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite
```

If `.env` already exists, update it without replacing other settings. The generation
model above was used in successful local evaluations; choose a model available to
your account. Environment variables override `.env`. Restart FastAPI after changes.
Never commit your actual API key.

From `backend/`, generate the fictional sample and start the server:

```bash
python scripts/create_sample_pdf.py
uvicorn app.main:app --reload
```

- Health: http://127.0.0.1:8000/health
- Swagger UI: http://127.0.0.1:8000/docs

### Frontend

In a second terminal, from the repository root:

```bash
cd frontend
npm ci
npm run dev
```

Open **http://127.0.0.1:5173**. The Vite development proxy forwards `/api` requests
to FastAPI on port 8000. API keys stay on the backend.

## Try it

1. Upload `backend/samples/synthetic-policy.pdf` through the interface.
2. Ask: **Who should staff notify about missing inspection entries?**
3. Check the shift coordinator answer and its supporting quote from page 3.
4. Ask: **What time does the cafeteria open?** Expect abstention and no citations.
5. Refresh the page. Saved uploads reload from ChromaDB.

The sample has text on pages 1 and 3; page 2 is blank. Original page numbers remain
intact. Each upload receives a unique document ID, even when filenames match.
`GET /documents` recovers saved IDs. Uploaded original PDFs are not retained by the
endpoint; extracted text, vectors, and metadata persist in `backend/chroma_data/`.

Uploads and questions send synthetic text to Google and consume API quota; charges
may apply. No real patient information is needed for this project.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check backend health |
| GET | `/documents` | List saved uploads and IDs |
| POST | `/documents/extract` | Inspect extracted pages |
| POST | `/documents/chunks` | Inspect chunks |
| POST | `/documents/embed` | Generate embeddings without saving |
| POST | `/documents/ingest` | Embed and store an upload |
| GET | `/documents/{id}/chunks` | Read stored passages |
| POST | `/documents/{id}/search` | Retrieve relevant passages |
| POST | `/documents/{id}/ask` | Generate a cited answer or abstain |

Search and ask accept a JSON body such as:

```json
{
  "question": "Who receives reports of missing inspection entries?",
  "top_k": 2
}
```

## Verification

From `backend/`, with the virtual environment activated:

```bash
python -m pytest -q
```

The last verified run passed **96 tests**. Tests cover extraction, upload limits,
chunking, embedding requests, real temporary Chroma storage, persistence across
processes, document isolation, retrieval, citations, errors, and evaluation scoring.
Cloud responses are mocked, so passing tests do not measure live-model accuracy.
Six dependency warnings appeared: five from PyMuPDF and one about urllib3/LibreSSL.

From `frontend/`:

```bash
npm run build
```

The production build passed. The frontend upload, citation, and abstention flows
were manually confirmed; automated browser verification was unavailable.
Edit `frontend/src/` and `frontend/index.html`, not generated `frontend/dist/` files.

## Measured evaluation

The dataset in `backend/evaluation/policy_cases.json` contains four supported
questions and two unsupported questions against a single synthetic PDF with only
**two chunks**. These are smoke checks, not broad healthcare benchmarks.

| Observation | Run 2 | Run 3 |
|---|---:|---:|
| Completed cases / attempted | 6/6 | 5/6 |
| API errors | 0 | 1 timeout (504) |
| Macro Recall@2 over supported completed cases | 100% (4 cases) | 100% (4 cases) |
| Answer phrase match | 4/4 | 4/4 |
| Unsupported-question abstention | 2/2 | 1/1 completed |
| Citation location/quote heuristic | 2/5, original rule | 5/5, revised rule |
| Mean search round trip | 0.750 seconds | 0.699 seconds |
| Mean ask round trip | 11.111 seconds | 11.803 seconds |

Both runs used `gemini-3.5-flash-lite`. An earlier run failed all six generation
requests. Run 3's timed-out question abstained correctly in a separate manual retry;
that retry is not included in the saved run's metrics.

Retrieving both available chunks makes Recall@2 an easy check. Phrase matching is
not semantic answer correctness. The citation heuristic changed between runs to
accept longer real quotes containing the expected passage, so its scores are not
directly comparable. Human correctness and faithfulness labels remain unset.
Token usage and costs are not measured. Latency excludes failed cases.

To run an evaluation, keep the backend running. From `backend/`:

```bash
python -m evaluation.run --document-id YOUR_DOCUMENT_UUID --top-k 2 --output evaluation/results/new-run.json
```

Replace the UUID using `GET /documents`. The runner verifies the expected passages,
uses live cloud quota, and writes measured results without overwriting existing
reports. Generated reports remain local and are excluded from Git.

## Limitations

- No authentication or ownership checks. Document IDs scope retrieval but do not authorize users.
- No OCR for scanned PDFs, document deletion, hybrid search, reranking, or persistent chat history.
- Citation presence checks and prompt boundaries do not eliminate hallucinations or guarantee prompt-injection resistance.
- The 10 MB extraction read limit applies after multipart parsing; deployment needs server-level request and resource limits.
- The document library scans chunk metadata and needs a catalog/pagination for larger datasets.
- Provider rate limits, model access, timeouts, and availability can interrupt requests. Automatic retries are not implemented.
- The Vite proxy is for development. Production deployment needs backend hosting, durable storage, secrets, access control, and API routing.
- No production-readiness or healthcare compliance claim is made.

HTTP 400 indicates invalid uploads; 413 indicates an oversized PDF; 422 indicates
invalid inputs or no extractable text; 429 indicates provider quota/rate limits;
502/503/504 indicate provider, storage, availability, or timeout errors. Inspect
the safe response `detail`. Never share API keys while debugging.

## Repository structure

```text
backend/
  app/api/          FastAPI routes
  app/services/     PDF extraction, chunking, embeddings, storage, generation
  app/schemas/      Request validation
  evaluation/       Synthetic dataset and evaluation runner
  scripts/          Synthetic PDF generator
  tests/            Automated tests
frontend/
  src/              React application and responsive CSS
  vite.config.js    Development API proxy
  package-lock.json Reproducible frontend dependencies
```

Environment files, PDFs, Chroma data, generated evaluation reports, virtual
environments, dependencies, and build output are excluded from Git.
