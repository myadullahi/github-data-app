"""
RAG: search Milvus (dense) and answer with OpenAI using only retrieved context.
"""
from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from pymilvus import MilvusClient

import config
from load_data import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    embed_texts,
)

logger = logging.getLogger(__name__)


def _repo_to_github_url(repo: str) -> str:
    """Turn owner/repo into https://github.com/owner/repo."""
    repo = (repo or "").strip()
    if not repo:
        return ""
    return f"https://github.com/{repo}"


# How many chunks to retrieve for context (before reranker in phase 2)
RAG_TOP_K = 10
# Model for chat completion (answer from context)
CHAT_MODEL = "gpt-4o-mini"


def get_milvus_client() -> MilvusClient:
    return MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


def search_dense(
    query: str,
    *,
    top_k: int = RAG_TOP_K,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, Any]]:
    """
    Embed query, search Milvus, return list of {content, source, repo}.
    """
    openai_client = get_openai_client()
    milvus = get_milvus_client()
    if not query or not query.strip():
        return []
    vectors = embed_texts(openai_client, [query.strip()], model=EMBEDDING_MODEL)
    if not vectors:
        return []
    search_params = {"metric_type": "COSINE", "params": {}}
    res = milvus.search(
        collection_name=collection_name,
        data=vectors,
        limit=top_k,
        output_fields=["content", "source", "repo"],
        search_params=search_params,
    )
    # res is List[List[dict]]; one query -> one list of hits
    hits = res[0] if res else []
    out = []
    for h in hits:
        entity = h.get("entity") or {}
        content = entity.get("content") or ""
        source = entity.get("source") or ""
        repo = entity.get("repo") or ""
        if content:
            out.append({"content": content, "source": source, "repo": repo})
    return out


def answer_with_rag(query: str, chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    """
    Build a prompt from retrieved chunks and call OpenAI to answer using only that context.
    Returns (answer_text, sources) where sources is a list of {"repo": "owner/repo", "url": "https://github.com/owner/repo"}.
    """
    empty_answer = (
        "I couldn't find any relevant content in the indexed repositories for your question. "
        "Try rephrasing or asking about topics that might be in the loaded GitHub repos."
    )
    if not chunks:
        return empty_answer, []

    client = get_openai_client()
    context_parts = []
    for i, c in enumerate(chunks, 1):
        repo = c.get("repo") or ""
        source = c.get("source") or ""
        content = (c.get("content") or "").strip()
        if not content:
            continue
        url = _repo_to_github_url(repo)
        context_parts.append(
            f"[{i}] repo: {repo} | GitHub: {url}\nsource: {source}\n{content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a helpful assistant that answers questions using ONLY the provided context from GitHub repository code and documentation. "
        "If the answer cannot be found in the context, say so clearly. Do not use external knowledge. "
        "When citing sources, always use the full GitHub repo URL (e.g. https://github.com/owner/repo). "
        "Do not use generic references like [1] or source: [2]; use the actual URL. Keep answers concise."
    )
    user = (
        "Context from the knowledge base:\n\n"
        f"{context}\n\n"
        "---\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above. Cite sources with the full GitHub URL."
    )
    try:
        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
        )
        msg = r.choices[0].message
        answer = (msg.content or "").strip()
    except Exception as e:
        logger.exception("OpenAI chat failed: %s", e)
        answer = f"Sorry, an error occurred while generating the answer: {e!s}"

    # Build sources list with URLs (dedupe by repo)
    seen: set[str] = set()
    sources = []
    for c in chunks:
        repo = (c.get("repo") or "").strip()
        if not repo or repo in seen:
            continue
        seen.add(repo)
        sources.append({"repo": repo, "url": _repo_to_github_url(repo)})
    return answer, sources


def check_milvus_health() -> dict[str, Any]:
    """Return { status: 'ok'|'error', message?, collection_exists?, row_count? }."""
    out: dict[str, Any] = {"status": "error", "message": ""}
    try:
        client = get_milvus_client()
        if not client.has_collection(COLLECTION_NAME):
            out["message"] = f"Collection '{COLLECTION_NAME}' does not exist."
            out["collection_exists"] = False
            return out
        out["collection_exists"] = True
        # Optional: get row count (may be slow on huge collections)
        try:
            stats = client.get_collection_stats(COLLECTION_NAME)
            out["row_count"] = stats.get("row_count")
        except Exception:
            out["row_count"] = None
        out["status"] = "ok"
        out["message"] = "Connected"
        return out
    except Exception as e:
        out["message"] = str(e)
        return out


def check_openai_health() -> dict[str, Any]:
    """Return { status: 'ok'|'error', message? }."""
    out: dict[str, Any] = {"status": "error", "message": ""}
    if not config.OPENAI_API_KEY:
        out["message"] = "OPENAI_API_KEY not set"
        return out
    try:
        client = get_openai_client()
        # Minimal check: one short embedding (fast)
        client.embeddings.create(input=["ok"], model=EMBEDDING_MODEL)
        out["status"] = "ok"
        out["message"] = "API key valid"
        return out
    except Exception as e:
        out["message"] = str(e)
        return out
