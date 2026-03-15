"""
FastAPI app for the GitHub RAG service (local; deployment in phase 3).

Run locally:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

- http://localhost:8000 — Chat UI
- http://localhost:8000/docs — Swagger UI
- http://localhost:8000/health — Service health (Milvus, OpenAI)
"""
from contextlib import asynccontextmanager
from pathlib import Path

import config
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import rag


class ChatRequest(BaseModel):
    message: str


def _check_env_on_startup():
    missing = config.check_env()
    if missing:
        raise RuntimeError(f"Missing required env: {', '.join(missing)}. Set them in .env.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_env_on_startup()
    yield


app = FastAPI(
    title="GitHub RAG API",
    description="Search over ingested GitHub repo content; chat uses only RAG context.",
    version="0.1.0",
    lifespan=lifespan,
)

# Project root for template path
ROOT = Path(__file__).resolve().parent


@app.get("/", include_in_schema=False)
def index():
    """Serve the chat UI."""
    return FileResponse(ROOT / "templates" / "index.html", media_type="text/html")


@app.get("/health")
def health():
    """
    Health check for services used by the app.
    Returns status of Milvus (and collection) and OpenAI.
    """
    milvus = rag.check_milvus_health()
    openai_status = rag.check_openai_health()
    overall = "ok" if (milvus["status"] == "ok" and openai_status["status"] == "ok") else "degraded"
    return {
        "status": overall,
        "services": {
            "milvus": milvus,
            "openai": openai_status,
        },
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """
    RAG chat: search Milvus with the user message, then answer with OpenAI using only retrieved context.
    Body: { "message": "your question" }
    Returns: { "answer": "...", "sources": [ {"repo": "owner/repo", "url": "https://github.com/owner/repo"}, ... ] }
    """
    message = (req.message or "").strip()
    if not message:
        return {"answer": "Please provide a non-empty question.", "sources": []}
    try:
        chunks = rag.search_dense(message)
        answer, sources = rag.answer_with_rag(message, chunks)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        return {"answer": f"Sorry, an error occurred: {e!s}", "sources": []}


@app.get("/api")
def api_info():
    """JSON entrypoint for API users (docs, health)."""
    return {"service": "GitHub RAG", "docs": "/docs", "health": "/health"}
