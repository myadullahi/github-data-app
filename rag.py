"""
RAG: hybrid search (dense + sparse), rerank to top-5, then answer with OpenAI using only retrieved context.
"""
from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI
from pymilvus import MilvusClient

import config
from load_data import (
    COLLECTION_NAME,
    COLLECTION_NAME_SPARSE,
    EMBEDDING_MODEL,
    embed_texts,
)
from sparse_utils import text_to_sparse_vector

logger = logging.getLogger(__name__)


def _repo_to_github_url(repo: str) -> str:
    """Turn owner/repo into https://github.com/owner/repo."""
    repo = (repo or "").strip()
    if not repo:
        return ""
    return f"https://github.com/{repo}"


# Hybrid: fetch this many from dense and from sparse each, then RRF merge (larger pool so OBenner-style Q&A can appear)
HYBRID_TOP_K = 50
# After RRF merge, rerank and keep this many for the answer (more chunks = better chance of including Q&A content)
RERANK_TOP_K = 10
# Max chunks from a single repo in the final reranked list (reduces single-repo bias)
MAX_CHUNKS_PER_REPO = 3
# RRF constant (reciprocal rank fusion)
RRF_K = 60
# Model for chat completion (answer from context)
CHAT_MODEL = "gpt-4o-mini"


def get_milvus_client() -> MilvusClient:
    return MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _chunk_key(c: dict[str, Any]) -> tuple[str, str, str]:
    """Stable key for deduping chunks across dense and sparse results."""
    return (
        (c.get("content") or "").strip(),
        (c.get("source") or "").strip(),
        (c.get("repo") or "").strip(),
    )


def search_dense(
    query: str,
    *,
    top_k: int = HYBRID_TOP_K,
    collection_name: str = COLLECTION_NAME,
) -> list[dict[str, Any]]:
    """
    Embed query, search Milvus dense index, return list of {content, source, repo}.
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


def search_sparse(
    query: str,
    *,
    top_k: int = HYBRID_TOP_K,
    collection_name: str = COLLECTION_NAME_SPARSE,
) -> list[dict[str, Any]]:
    """
    Build sparse vector from query, search Milvus sparse index, return list of {content, source, repo}.
    Returns [] if sparse collection does not exist or search fails.
    """
    milvus = get_milvus_client()
    if not query or not query.strip():
        return []
    try:
        if not milvus.has_collection(collection_name):
            return []
    except Exception:
        return []
    q_sparse = text_to_sparse_vector(query.strip())
    if not q_sparse:
        return []
    # Milvus sparse search expects list of sparse vectors (each dict {index: value})
    search_params = {"metric_type": "IP", "params": {}}
    try:
        res = milvus.search(
            collection_name=collection_name,
            data=[q_sparse],
            limit=top_k,
            output_fields=["content", "source", "repo"],
            search_params=search_params,
            anns_field="sparse_vector",
        )
    except Exception as e:
        logger.debug("Sparse search failed (e.g. not supported): %s", e)
        return []
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


def _rrf_merge(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Merge dense and sparse result lists using reciprocal rank fusion; dedupe by chunk key."""
    scores: dict[tuple[str, str, str], float] = {}
    chunks_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rank, c in enumerate(dense):
        key = _chunk_key(c)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        chunks_by_key[key] = c
    for rank, c in enumerate(sparse):
        key = _chunk_key(c)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        chunks_by_key[key] = c
    # Sort by score descending, return chunks in order
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    return [chunks_by_key[key] for key, _ in ordered]


def _apply_repo_diversity(chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Limit how many chunks from each repo can appear in the top_k to reduce single-repo bias."""
    if not chunks or top_k <= 0:
        return []
    out: list[dict[str, Any]] = []
    repo_count: dict[str, int] = {}
    for c in chunks:
        if len(out) >= top_k:
            break
        repo = (c.get("repo") or "").strip()
        if repo_count.get(repo, 0) >= MAX_CHUNKS_PER_REPO:
            continue
        out.append(c)
        repo_count[repo] = repo_count.get(repo, 0) + 1
    # If we have room and skipped some, fill with remaining chunks (no cap) so we still return top_k when possible
    if len(out) < top_k:
        seen_keys = {_chunk_key(c) for c in out}
        for c in chunks:
            if len(out) >= top_k:
                break
            if _chunk_key(c) in seen_keys:
                continue
            seen_keys.add(_chunk_key(c))
            out.append(c)
    return out


def _content_richness_score(content: str) -> float:
    """
    Score 0..1 for how much the chunk looks like substantive Q&A (not just a short question list).
    Longer chunks and chunks with markdown sections / explanatory text get a boost so they
    outrank short 'here are 3 questions' lists when the user asks for answers.
    """
    s = (content or "").strip()
    if not s:
        return 0.0
    # Length: Q&A content (e.g. OBenner spark.md) tends to be long; question-only lists are short
    length_norm = min(1.0, len(s) / 3000)
    # Markdown sections (## ) often indicate structured Q&A with answers
    section_count = s.count("\n## ")
    section_bonus = min(0.3, section_count * 0.05)
    return 0.7 * length_norm + section_bonus


def _rerank_with_embeddings(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = RERANK_TOP_K,
) -> list[dict[str, Any]]:
    """Rerank chunks by embedding similarity to query, with a boost for answer-rich (long, structured) content."""
    if not chunks or top_k <= 0:
        return []
    if len(chunks) <= top_k:
        return chunks
    openai_client = get_openai_client()
    texts = [(c.get("content") or "").strip() for c in chunks]
    if not any(texts):
        return chunks[:top_k]
    try:
        query_emb = embed_texts(openai_client, [query], model=EMBEDDING_MODEL)
        if not query_emb:
            return chunks[:top_k]
        chunk_embs = embed_texts(openai_client, texts, model=EMBEDDING_MODEL)
        if len(chunk_embs) != len(chunks):
            return chunks[:top_k]
    except Exception as e:
        logger.warning("Rerank embedding failed: %s", e)
        return chunks[:top_k]
    q = query_emb[0]
    # Cosine similarity + boost for content richness (so Q&A chunks outrank short question lists)
    RICHNESS_WEIGHT = 0.15  # blend: 85% embedding, 15% richness
    sims = []
    for i, emb in enumerate(chunk_embs):
        dot = sum(a * b for a, b in zip(q, emb))
        norm_q = sum(x * x for x in q) ** 0.5
        norm_e = sum(x * x for x in emb) ** 0.5
        cos = (dot / (norm_q * norm_e)) if (norm_q and norm_e) else 0.0
        richness = _content_richness_score(texts[i])
        combined = (1.0 - RICHNESS_WEIGHT) * cos + RICHNESS_WEIGHT * richness
        sims.append((i, combined))
    sims.sort(key=lambda x: -x[1])
    ordered = [chunks[i] for i, _ in sims]
    return _apply_repo_diversity(ordered, top_k)


def hybrid_search(
    query: str,
    *,
    top_k_dense: int = HYBRID_TOP_K,
    top_k_sparse: int = HYBRID_TOP_K,
    rerank_top: int = RERANK_TOP_K,
) -> list[dict[str, Any]]:
    """
    Run dense + sparse search, merge with RRF, rerank with embeddings, return top rerank_top chunks (default 5).
    If sparse collection is missing, uses dense-only then reranks.
    """
    if not query or not query.strip():
        return []
    dense_hits = search_dense(query, top_k=top_k_dense)
    sparse_hits = search_sparse(query, top_k=top_k_sparse)
    if dense_hits and sparse_hits:
        merged = _rrf_merge(dense_hits, sparse_hits)
    elif dense_hits:
        merged = dense_hits
    elif sparse_hits:
        merged = sparse_hits
    else:
        return []
    reranked = _rerank_with_embeddings(query, merged, top_k=rerank_top)
    # Log top reranked chunks (server console)
    for i, c in enumerate(reranked, 1):
        repo = c.get("repo") or ""
        source = c.get("source") or ""
        raw = (c.get("content") or "").strip()
        preview = (raw[:200] + "…") if len(raw) > 200 else raw
        logger.info("Reranked chunk %d/%d: repo=%s source=%s preview=%s", i, len(reranked), repo, source, preview[:100])
    return reranked


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
        "Do not use generic references like [1] or source: [2]; use the actual URL. Keep answers concise.\n\n"
        "If the user's question is ambiguous or could refer to multiple things in the context (e.g. 'what can this app do?' when context describes several different apps or repos), "
        "ask for clarification before answering—for example which repo, which tool, or which feature they mean—so you do not assume and answer about the wrong one.\n\n"
        "Cite the source directly with each answer: for every question-answer pair or list item, put the source URL right after that answer (e.g. 'Source: https://github.com/owner/repo' or 'Source: https://github.com/owner/repo#anchor'). "
        "Do not only list sources in a separate section at the end—each answer should have its source immediately after it.\n\n"
        "Important: When the user asks for 'answers' to interview questions (or 'give me the answers', 'explain those', etc.), "
        "you MUST provide the actual answers from the context when the context contains them. Many context chunks are in Q&A format (question followed by answer). "
        "Do NOT refuse with 'I am unable to provide answers' or only point to repositories when the context already includes the answer text. "
        "Only say the answer is not in the context when it truly is missing."
    )
    user = (
        "Context from the knowledge base:\n\n"
        f"{context}\n\n"
        "---\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above. If the user asked for answers to specific questions, provide those answers from the context when present. "
        "For each question or list item, cite the source URL directly after that answer (e.g. Source: https://github.com/owner/repo)."
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
