"""
Tests for RAG and API, aligned with BENCHMARK.md (5 tests).
Run with: pytest test_rag.py -v
CI runs these without real Milvus/OpenAI (mocks used where needed).
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock

# Set env before importing main so lifespan check passes in CI
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("MILVUS_URI", "http://test")
os.environ.setdefault("MILVUS_TOKEN", "test-token")

import rag


# ----- Unit tests (no mocks for these) -----


def test_chunk_key():
    """Stable key for deduping chunks."""
    c = {"content": "hello", "source": "github:a:b", "repo": "owner/repo"}
    key = rag._chunk_key(c)
    assert key == ("hello", "github:a:b", "owner/repo")
    assert rag._chunk_key({}) == ("", "", "")


def test_rrf_merge_dedupes_and_orders():
    """RRF merge combines dense + sparse, dedupes by chunk key, orders by score."""
    dense = [
        {"content": "a", "source": "s1", "repo": "r1"},
        {"content": "b", "source": "s2", "repo": "r2"},
    ]
    sparse = [
        {"content": "b", "source": "s2", "repo": "r2"},  # duplicate
        {"content": "c", "source": "s3", "repo": "r3"},
    ]
    merged = rag._rrf_merge(dense, sparse, k=60)
    # Should be 3 unique chunks; b appears in both so gets higher RRF score
    assert len(merged) == 3
    keys = [rag._chunk_key(c) for c in merged]
    assert keys[0] == ("b", "s2", "r2")  # b ranked in both lists
    assert set(keys) == {("a", "s1", "r1"), ("b", "s2", "r2"), ("c", "s3", "r3")}


def test_apply_repo_diversity_caps_per_repo():
    """First pass caps at MAX_CHUNKS_PER_REPO per repo; multiple repos appear (BENCHMARK: diversity)."""
    # Mixed repos: first pass takes up to MAX_CHUNKS_PER_REPO per repo, then fill to top_k
    mixed = (
        [{"content": f"a{i}", "source": "s", "repo": "repo/a"} for i in range(4)]
        + [{"content": f"b{i}", "source": "s", "repo": "repo/b"} for i in range(4)]
        + [{"content": "c", "source": "s", "repo": "repo/c"}]
    )
    mixed = list(mixed)
    top = rag._apply_repo_diversity(mixed, top_k=10)
    assert len(top) == min(10, len(mixed))  # 9 chunks in -> 9 out
    repos = [c["repo"] for c in top]
    # Should have at least 2 repos (diversity)
    assert len(set(repos)) >= 2
    # No repo should dominate the first 9 (cap is 3 per repo in first pass)
    from collections import Counter
    counts = Counter(repos)
    assert max(counts.values()) <= 4  # at most 4 from one repo after fill


def test_content_richness_score_prefers_long_and_structured():
    """Longer content and markdown sections get higher richness score."""
    short = "Just a short list."
    long_text = "x" * 4000
    with_sections = "## Q1\nanswer\n\n## Q2\nanswer\n\n## Q3\nanswer"
    s_short = rag._content_richness_score(short)
    s_long = rag._content_richness_score(long_text)
    s_sections = rag._content_richness_score(with_sections)
    assert s_short < s_long
    assert s_sections > s_short
    assert 0 <= s_short <= 1 and 0 <= s_long <= 1


def test_repo_to_github_url():
    """Sources use full GitHub URL (BENCHMARK: no [1]/[2])."""
    # _repo_to_github_url is used inside answer_with_rag; test via public behavior
    url = rag._repo_to_github_url("OBenner/data-engineering-interview-questions")
    assert url == "https://github.com/OBenner/data-engineering-interview-questions"
    assert rag._repo_to_github_url("") == ""


# ----- API tests (mock RAG so no real Milvus/OpenAI) -----


@patch("rag.check_openai_health")
@patch("rag.check_milvus_health")
def test_chat_response_shape(mock_milvus, mock_openai):
    """POST /chat returns answer, sources, reranked_chunks (Test 3 structure)."""
    mock_milvus.return_value = {"status": "ok", "message": "ok", "row_count": 0}
    mock_openai.return_value = {"status": "ok", "message": "ok"}

    from fastapi.testclient import TestClient
    import main

    with patch("rag.hybrid_search") as mock_search, patch("rag.answer_with_rag") as mock_answer:
        mock_search.return_value = [
            {"content": "Some context.", "source": "github:o/r:path", "repo": "o/r"}
        ]
        mock_answer.return_value = (
            "Here is the answer. Source: https://github.com/o/r",
            [{"repo": "o/r", "url": "https://github.com/o/r"}],
        )
        client = TestClient(main.app)
        r = client.post("/chat", json={"message": "test question"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    assert "reranked_chunks" in data
    assert isinstance(data["sources"], list)
    assert isinstance(data["reranked_chunks"], list)
    if data["sources"]:
        assert "repo" in data["sources"][0]
        assert "url" in data["sources"][0]
        assert data["sources"][0]["url"].startswith("https://github.com/")


@patch("rag.check_openai_health")
@patch("rag.check_milvus_health")
def test_sources_have_full_github_urls(mock_milvus, mock_openai):
    """Citations are full GitHub URLs, not [1] or [2] (BENCHMARK requirement)."""
    mock_milvus.return_value = {"status": "ok"}
    mock_openai.return_value = {"status": "ok"}

    from fastapi.testclient import TestClient
    import main

    with patch("rag.hybrid_search") as mock_search, patch("rag.answer_with_rag") as mock_answer:
        mock_search.return_value = [
            {"content": "Context from OBenner.", "source": "github:OBenner/repo:content/spark.md", "repo": "OBenner/repo"}
        ]
        mock_answer.return_value = (
            "Answer here. Source: https://github.com/OBenner/repo",
            [{"repo": "OBenner/repo", "url": "https://github.com/OBenner/repo"}],
        )
        client = TestClient(main.app)
        r = client.post("/chat", json={"message": "Spark questions"})
    assert r.status_code == 200
    data = r.json()
    for s in data["sources"]:
        assert s["url"].startswith("https://github.com/"), "sources must be full GitHub URLs"
    # Answer text should not contain generic refs (model is prompted not to)
    assert "source: [1]" not in data["answer"].lower()
    assert "source: [2]" not in data["answer"].lower()


@patch("rag.check_openai_health")
@patch("rag.check_milvus_health")
def test_chat_empty_message(mock_milvus, mock_openai):
    """Empty message returns 200 with placeholder answer and empty sources."""
    mock_milvus.return_value = {"status": "ok"}
    mock_openai.return_value = {"status": "ok"}

    from fastapi.testclient import TestClient
    import main

    client = TestClient(main.app)
    r = client.post("/chat", json={"message": "   "})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert data.get("sources") == []
    assert data.get("reranked_chunks", []) == []


@patch("rag.check_openai_health")
@patch("rag.check_milvus_health")
def test_health_returns_services(mock_milvus, mock_openai):
    """GET /health returns status and services (milvus, openai)."""
    mock_milvus.return_value = {"status": "ok", "message": "Connected", "row_count": 100}
    mock_openai.return_value = {"status": "ok", "message": "API key valid"}

    from fastapi.testclient import TestClient
    import main

    client = TestClient(main.app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "services" in data
    assert "milvus" in data["services"]
    assert "openai" in data["services"]
