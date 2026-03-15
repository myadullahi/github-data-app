"""
FastAPI app for the GitHub RAG service (local; deployment in phase 3).

Run locally:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

- http://localhost:8000 — Chat UI
- http://localhost:8000/docs — Swagger UI
- http://localhost:8000/health — Service health (Milvus, OpenAI)

Serverless (Vercel): rag (and pymilvus) are lazy-loaded so the function can start even if
pymilvus is large or fails to load; /health and /chat then return a clear error instead of crashing.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import config
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Lazy-load rag so pymilvus is not imported at app startup (avoids serverless crash from heavy/binary deps)
_rag = None
_rag_error = None


def _get_rag():
    """Import rag on first use; cache result or error so we don't crash the process."""
    global _rag, _rag_error
    if _rag is not None:
        return _rag
    if _rag_error is not None:
        raise _rag_error
    try:
        import rag as _rag_module
        _rag = _rag_module
        return _rag
    except Exception as e:
        _rag_error = e
        raise


class ChatRequest(BaseModel):
    message: str


def _check_env_on_startup():
    missing = config.check_env()
    if missing:
        raise RuntimeError(f"Missing required env: {', '.join(missing)}. Set them in .env.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Don't crash on missing env in serverless (e.g. Vercel); /health and /chat will show errors
    try:
        _check_env_on_startup()
    except RuntimeError as e:
        logger.warning("Startup env check failed (app will run degraded): %s", e)
    yield


app = FastAPI(
    title="GitHub RAG API",
    description="Search over ingested GitHub repo content; chat uses only RAG context.",
    version="0.1.0",
    lifespan=lifespan,
)

# Project root for template path
ROOT = Path(__file__).resolve().parent
# Cap user message length to avoid context flooding and token overflow (chat path)
MAX_CHAT_MESSAGE_CHARS = 8000


def _index_path() -> Path:
    """Path to index.html; try project root (main.py dir) and cwd for serverless."""
    for base in (ROOT, Path.cwd(), Path.cwd() / ".."):
        p = (base if base.is_absolute() else base.resolve()) / "templates" / "index.html"
        if p.is_file():
            return p
    return ROOT / "templates" / "index.html"  # let it fail with clear error if missing


@app.get("/", include_in_schema=False)
def index():
    """Serve the chat UI."""
    path = _index_path()
    if not path.is_file():
        return HTMLResponse(
            status_code=503,
            content="""<!DOCTYPE html><html><head><title>GitHub RAG</title></head><body style="font-family:sans-serif;padding:2rem;max-width:600px;">
<h1>Configuration needed</h1>
<p>Either <code>templates/index.html</code> is missing in the deployment, or required env vars are not set.</p>
<p>Set in Vercel: <strong>OPENAI_API_KEY</strong>, <strong>MILVUS_URI</strong>, <strong>MILVUS_TOKEN</strong>.</p>
<p><a href="/health">/health</a> — check service status.</p>
</body></html>""",
        )
    return FileResponse(path, media_type="text/html")


@app.get("/health")
def health():
    """
    Health check for services used by the app.
    Returns status of Milvus (and collection) and OpenAI.
    """
    try:
        rag = _get_rag()
    except Exception as e:
        return {
            "status": "degraded",
            "services": {
                "milvus": {"status": "error", "message": f"RAG module failed to load: {e!s}"},
                "openai": {"status": "error", "message": "RAG module not loaded"},
            },
        }
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
        return {"answer": "Please provide a non-empty question.", "sources": [], "reranked_chunks": []}
    if len(message) > MAX_CHAT_MESSAGE_CHARS:
        message = message[:MAX_CHAT_MESSAGE_CHARS]
    try:
        rag = _get_rag()
    except Exception as e:
        return {
            "answer": f"RAG is not available (module failed to load: {e!s}). On serverless (e.g. Vercel), pymilvus may be too large or incompatible—consider a different host (Railway, Render, Fly.io) for full RAG.",
            "sources": [],
            "reranked_chunks": [],
        }
    try:
        # Use more chunks (10) so answer-rich content is more likely included; LLM gets full content
        chunks = rag.hybrid_search(message, rerank_top=10)
        answer, sources = rag.answer_with_rag(message, chunks)
        # Expose full reranked chunk content so you can verify exactly what the LLM received
        reranked_chunks = [
            {"repo": c.get("repo") or "", "source": c.get("source") or "", "content": (c.get("content") or "").strip()}
            for c in chunks
        ]
        return {"answer": answer, "sources": sources, "reranked_chunks": reranked_chunks}
    except Exception as e:
        return {"answer": f"Sorry, an error occurred: {e!s}", "sources": [], "reranked_chunks": []}


@app.get("/api")
def api_info():
    """JSON entrypoint for API users (docs, health)."""
    return {"service": "GitHub RAG", "docs": "/docs", "health": "/health"}
