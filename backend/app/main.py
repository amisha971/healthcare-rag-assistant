from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.documents import router

app = FastAPI(title="Healthcare Document Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
