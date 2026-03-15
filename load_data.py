"""
Ingest public GitHub repo data into Milvus for RAG.

- Fetches file contents from GitHub (repos you have legal access to: public or token-authorized).
- Skips noise (node_modules, vendor, .git, lock files, minified assets).
- Chunking: small files = one chunk; large files = overlapping line windows (code-friendly).
- Embeds with OpenAI, inserts into a Milvus collection (partitioned by repo).

Usage:
  Set OPENAI_API_KEY, MILVUS_URI, MILVUS_TOKEN in .env. Optional: GITHUB_TOKEN for rate limits.
  Repos by hand:
    python load_data.py --repos owner/repo1 owner/repo2
  Repos by topic (discover then load):
    python discover_repos.py --topic machine-learning --language Python --output repos.txt
    python load_data.py --repos-file repos.txt
"""
from __future__ import annotations

import argparse
import ast
import base64
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tiktoken
from github import Auth, Github
from github.GithubException import GithubException
from openai import OpenAI
from pymilvus import DataType, MilvusClient
from tqdm import tqdm

import config
from repo_filters import is_text_file, should_skip_path
from sparse_utils import text_to_sparse_vector

# Defaults (override via env or args)
COLLECTION_NAME = "github_rag"
COLLECTION_NAME_SPARSE = "github_rag_sparse"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # text-embedding-3-small
# Meaningful code chunking: files longer than this (lines) are split into overlapping windows
LARGE_FILE_LINES = 400
CHUNK_LINES = 350
CHUNK_OVERLAP_LINES = 50
# Chunk strategies: file (one chunk per small file / windows for large), function (Python ast), markdown (by ##), lines (always windows)
CHUNK_STRATEGIES = ("file", "function", "markdown", "lines")
# Concurrency and batching (tune for speed vs rate limits; reduce if you hit GitHub/OpenAI limits)
GITHUB_FETCH_WORKERS = 10   # parallel blob fetches per repo
# OpenAI embedding: 300k tokens per request total; each item ≤8k tokens → batch ≤37
EMBED_BATCH_SIZE = 37
INSERT_BATCH_SIZE = 200    # Milvus insert batch size
# Complete-repos only: we only load repos with ≤ this many eligible files (and load all of them).
# Repos with more are skipped so the RAG never has partial knowledge.
MAX_FILES_PER_REPO_DEFAULT = 200
# OpenAI embedding API limit 8192 tokens – truncate by token count (code can be ~2 chars/token)
MAX_EMBED_TOKENS = 8000

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_github_client():
    """GitHub client; uses GITHUB_TOKEN if set. retry=None so 403/404 fail fast and we skip that repo."""
    token = config.GITHUB_TOKEN
    if token:
        return Github(auth=Auth.Token(token), retry=None)
    return Github(retry=None)  # unauthenticated: public only; no retry = no long backoff on 403


def _chunk_by_file(path: str, content: str, repo_id: str) -> list[tuple[str, str]]:
    """Small files = one chunk; large files = overlapping line windows."""
    lines = content.splitlines()
    if len(lines) <= LARGE_FILE_LINES:
        return [(f"github:{repo_id}:{path}", content)]
    out = []
    step = CHUNK_LINES - CHUNK_OVERLAP_LINES
    for start in range(0, len(lines), step):
        end = min(start + CHUNK_LINES, len(lines))
        chunk_lines = lines[start:end]
        chunk_content = "\n".join(chunk_lines)
        line_spec = f"#L{start + 1}-L{end}"
        out.append((f"github:{repo_id}:{path}{line_spec}", chunk_content))
        if end >= len(lines):
            break
    return out


def _chunk_by_lines(path: str, content: str, repo_id: str) -> list[tuple[str, str]]:
    """Always split into overlapping line windows (same as large-file behavior)."""
    lines = content.splitlines()
    if not lines:
        return [(f"github:{repo_id}:{path}", content or " ")]
    out = []
    step = CHUNK_LINES - CHUNK_OVERLAP_LINES
    for start in range(0, len(lines), step):
        end = min(start + CHUNK_LINES, len(lines))
        chunk_lines = lines[start:end]
        chunk_content = "\n".join(chunk_lines)
        line_spec = f"#L{start + 1}-L{end}"
        out.append((f"github:{repo_id}:{path}{line_spec}", chunk_content))
        if end >= len(lines):
            break
    return out


def _chunk_by_markdown(path: str, content: str, repo_id: str) -> list[tuple[str, str]]:
    """Split by markdown headings (## or ###). Each section becomes one chunk."""
    lines = content.splitlines()
    if not lines:
        return [(f"github:{repo_id}:{path}", content or " ")]
    # Section boundaries: line indices where a heading starts (## or ###); avoid duplicating 0
    section_starts = [0]
    for i, line in enumerate(lines):
        if i > 0 and re.match(r"^#{2,6}\s", line.strip()):
            section_starts.append(i)
    section_starts.append(len(lines))
    out = []
    for j in range(len(section_starts) - 1):
        start, next_start = section_starts[j], section_starts[j + 1]
        chunk_lines = lines[start:next_start]
        chunk_content = "\n".join(chunk_lines).strip()
        if not chunk_content:
            continue
        line_spec = f"#L{start + 1}-L{next_start}"
        out.append((f"github:{repo_id}:{path}{line_spec}", chunk_content))
    if not out:
        return [(f"github:{repo_id}:{path}", content)]
    return out


def _chunk_by_function(path: str, content: str, repo_id: str) -> list[tuple[str, str]]:
    """Python: chunk by top-level function and class (ast). Other files: fall back to file strategy."""
    ext = (Path(path).suffix or "").lower()
    if ext != ".py":
        return _chunk_by_file(path, content, repo_id)
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _chunk_by_file(path, content, repo_id)
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else (start + 1)
            chunk_lines = content.splitlines()[start:end]
            chunk_content = "\n".join(chunk_lines)
            line_spec = f"#L{start + 1}-L{end}"
            out.append((f"github:{repo_id}:{path}{line_spec}", chunk_content))
    if not out:
        return _chunk_by_file(path, content, repo_id)
    return out


def file_to_chunks(
    path: str,
    content: str,
    repo_id: str,
    strategy: str = "file",
) -> list[tuple[str, str]]:
    """
    Turn one file into one or more (source, content) chunks for embedding.
    strategy: "file" (default), "function" (Python ast), "markdown" (by ##), "lines".
    """
    if strategy == "lines":
        return _chunk_by_lines(path, content, repo_id)
    if strategy == "markdown":
        return _chunk_by_markdown(path, content, repo_id)
    if strategy == "function":
        return _chunk_by_function(path, content, repo_id)
    return _chunk_by_file(path, content, repo_id)


def _count_eligible_via_contents_api(repo) -> int:
    """Count eligible files via Contents API (list dirs only, no content fetch)."""
    count = 0
    queue: list[str] = [""]
    try:
        while queue:
            path = queue.pop(0)
            contents = repo.get_contents(path)
            if not isinstance(contents, list):
                contents = [contents]
            for item in contents:
                if item.type == "dir":
                    if not should_skip_path(item.path):
                        queue.append(item.path)
                    continue
                size = getattr(item, "size", 0) or 0
                if is_text_file(item.path, size):
                    count += 1
    except GithubException:
        return 999999  # so we skip this repo (incomplete or inaccessible)
    return count


def _fetch_repo_files_via_contents_api(
    repo,
    *,
    max_files_per_repo: int = MAX_FILES_PER_REPO_DEFAULT,
) -> list[tuple[str, str]]:
    """
    Walk repo via Contents API (get_contents); use when Git Data API returns 403.
    Only loads repo if eligible file count ≤ max_files_per_repo (complete-repos only); then loads all.
    """
    count = _count_eligible_via_contents_api(repo)
    if count > max_files_per_repo:
        return []
    out: list[tuple[str, str]] = []
    queue: list[str] = [""]
    while queue:
        path = queue.pop(0)
        try:
            contents = repo.get_contents(path)
        except GithubException as e:
            if e.status in (403, 404):
                return out
            raise
        if not isinstance(contents, list):
            contents = [contents]
        for item in contents:
            if item.type == "dir":
                if not should_skip_path(item.path):
                    queue.append(item.path)
                continue
            size = getattr(item, "size", 0) or 0
            if not is_text_file(item.path, size):
                continue
            try:
                file_obj = repo.get_contents(item.path)
                raw = file_obj.decoded_content
                content = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            out.append((item.path, content))
    return out


def _fetch_one_blob(repo, path: str, sha: str) -> tuple[str, str] | None:
    """Fetch a single blob; returns (path, content) or None on error (e.g. 403 Forbidden)."""
    try:
        blob = repo.get_git_blob(sha)
        content = base64.b64decode(blob.content).decode("utf-8", errors="replace")
        return (path, content)
    except GithubException as e:
        if e.status == 403:
            logger.warning("Skipping %s (403 Forbidden - repo may restrict API blob access)", path)
        else:
            logger.debug("Skip %s: %s", path, e)
        return None
    except Exception:
        return None


def fetch_repo_files(
    gh: Github,
    owner: str,
    repo_name: str,
    *,
    max_files_per_repo: int = MAX_FILES_PER_REPO_DEFAULT,
) -> list[tuple[str, str]]:
    """
    List and fetch text file contents from default branch (code + docs; excludes noise paths).
    Tries Git Data API first (tree + blobs); on 403, falls back to Contents API. Caps files per repo.
    Returns list of (file_path, content).
    """
    try:
        repo = gh.get_repo(f"{owner}/{repo_name}")
    except GithubException as e:
        if e.status in (403, 404):
            raise
        raise
    default_branch = repo.default_branch
    try:
        tree = repo.get_git_tree(default_branch, recursive=True)
    except GithubException as e:
        if e.status == 403:
            logger.info("Git Data API 403 for %s/%s; trying Contents API ...", owner, repo_name)
            return _fetch_repo_files_via_contents_api(repo, max_files_per_repo=max_files_per_repo)
        raise
    # Collect (path, sha) for eligible blobs
    to_fetch = [
        (item.path, item.sha)
        for item in tree.tree
        if item.type == "blob" and is_text_file(item.path, item.size)
    ]
    if not to_fetch:
        return []
    if len(to_fetch) > max_files_per_repo:
        logger.info(
            "Skipping %s/%s (has %d files; we only load complete repos with ≤%d files to avoid partial knowledge)",
            owner, repo_name, len(to_fetch), max_files_per_repo,
        )
        return []
    out = []
    workers = min(GITHUB_FETCH_WORKERS, len(to_fetch))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one_blob, repo, path, sha): path for path, sha in to_fetch}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                out.append(result)
            else:
                path = futures[future]
                logger.debug("Skip %s (fetch failed)", path)
    # If most blobs returned 403, try Contents API for the repo
    if len(out) < len(to_fetch) // 2 and len(to_fetch) > 5:
        logger.info("Many 403s for %s/%s; trying Contents API ...", owner, repo_name)
        contents_out = _fetch_repo_files_via_contents_api(repo, max_files_per_repo=max_files_per_repo)
        if len(contents_out) > len(out):
            return contents_out
    return out


def iter_repos(
    gh: Github,
    repo_specs: list[str],
    *,
    max_files_per_repo: int = MAX_FILES_PER_REPO_DEFAULT,
):
    """
    Yields (owner, repo_name, [(file_path, content), ...]) for each repo.
    Only includes repos with ≤ max_files_per_repo eligible files (complete repos only).
    """
    for spec in repo_specs:
        spec = spec.strip()
        if not spec or spec.startswith("#"):
            continue
        # Accept owner/repo or owner.repo
        normalized = spec.replace(".", "/", 1) if "/" not in spec else spec
        parts = normalized.split("/", 1)
        if len(parts) != 2:
            logger.warning("Invalid repo spec: %s (use owner/repo)", spec)
            continue
        owner, repo_name = parts
        try:
            logger.info("Fetching %s/%s (complete repos with ≤%d files only) ...", owner, repo_name, max_files_per_repo)
            files = fetch_repo_files(
                gh, owner, repo_name,
                max_files_per_repo=max_files_per_repo,
            )
            yield owner, repo_name, files
        except GithubException as e:
            if e.status in (403, 404):
                logger.warning("Skipping %s (%s %s - repo may restrict API or not exist)", spec, e.status, "Forbidden" if e.status == 403 else "Not Found")
            else:
                logger.warning("Failed to fetch %s: %s", spec, e)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", spec, e)


def _truncate_to_tokens(text: str, max_tokens: int = MAX_EMBED_TOKENS, encoding=None) -> str:
    """Truncate text to max_tokens (embedding API limit 8192). Uses cl100k_base (same as text-embedding-3-small)."""
    if encoding is None:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])


def embed_texts(client: OpenAI, texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Batch embed with OpenAI. Truncates each input to MAX_EMBED_TOKENS; empty after truncation → single space (API rejects empty)."""
    enc = tiktoken.get_encoding("cl100k_base")
    out = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = [_truncate_to_tokens(t, MAX_EMBED_TOKENS, enc) for t in texts[i : i + EMBED_BATCH_SIZE]]
        batch = [s if (s and s.strip()) else " " for s in batch]
        r = client.embeddings.create(input=batch, model=model)
        for e in r.data:
            out.append(e.embedding)
    return out


# Partition key: repo (owner/repo). Enables "search only in this repo" and better perf when filtering.
NUM_PARTITIONS = 64
# Demo repo used when all given repos fail (403); known to allow API access.
DEMO_REPO = "octocat/Hello-World"


def ensure_collection(client: MilvusClient, collection_name: str, dim: int):
    """Create collection if not exists (custom schema: id auto, vector, content, source, repo partition key)."""
    if client.has_collection(collection_name):
        logger.info("Collection %s already exists; will insert into it.", collection_name)
        return
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(
        field_name="repo",
        datatype=DataType.VARCHAR,
        max_length=512,
        is_partition_key=True,
    )
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="id", index_type="STL_SORT")
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        num_partitions=NUM_PARTITIONS,
    )
    logger.info("Created collection %s (dim=%s, partition_key=repo, num_partitions=%s).", collection_name, dim, NUM_PARTITIONS)


def _sparse_supported() -> bool:
    """True if this pymilvus supports SPARSE_FLOAT_VECTOR (not supported on Zilliz Cloud as of 2024)."""
    return getattr(DataType, "SPARSE_FLOAT_VECTOR", None) is not None


def ensure_sparse_collection(client: MilvusClient, collection_name: str = COLLECTION_NAME_SPARSE) -> bool:
    """
    Create sparse-only collection if not exists (for hybrid search). Same partition key as main collection.
    Returns True if sparse collection is available and was created or already exists; False if sparse not supported.
    """
    if not _sparse_supported():
        logger.info("Sparse vectors not supported (e.g. Zilliz Cloud); skipping sparse collection.")
        return False
    if client.has_collection(collection_name):
        logger.info("Sparse collection %s already exists.", collection_name)
        return True
    try:
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(
            field_name="repo",
            datatype=DataType.VARCHAR,
            max_length=512,
            is_partition_key=True,
        )
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="id", index_type="STL_SORT")
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            num_partitions=NUM_PARTITIONS,
        )
        logger.info("Created sparse collection %s (partition_key=repo).", collection_name)
        return True
    except Exception as e:
        logger.warning("Could not create sparse collection (server may not support sparse): %s", e)
        return False


def collection_has_repo_field(client: MilvusClient, collection_name: str) -> bool:
    """True if the collection schema includes the partition key field 'repo'."""
    try:
        desc = client.describe_collection(collection_name)
        fields = desc.get("fields") or []
        return any(f.get("name") == "repo" for f in fields)
    except Exception:
        return False


def _check_github_access(gh: Github) -> None:
    """Probe a public demo repo; exit with clear message if 403 (token/access issue)."""
    try:
        gh.get_repo(DEMO_REPO)
        return
    except GithubException as e:
        if e.status != 403:
            return
    logger.error(
        "GitHub returned 403 for public repo %s. This is an access/token issue, not the repo.\n"
        "  • If you use GITHUB_TOKEN in .env:\n"
        "    - Classic token: needs scope 'public_repo' (or 'repo' for private).\n"
        "    - Fine-grained token: grant 'Contents: Read' and 'Metadata: Read', and set repository access to 'All public repositories' (or add the repos you need).\n"
        "  • Try removing GITHUB_TOKEN from .env to use unauthenticated access (60 req/hr) and see if 403 goes away.\n"
        "  • Check for corporate proxy or firewall blocking api.github.com.",
        DEMO_REPO,
    )
    sys.exit(1)


def load_repos_to_milvus(
    repos: list[str],
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
    embedding_dim: int = EMBEDDING_DIM,
    max_files_per_repo: int = MAX_FILES_PER_REPO_DEFAULT,
    chunk_strategy: str = "file",
) -> int:
    """Fetch GitHub files, embed, insert into Milvus. Returns number of rows inserted."""
    missing = config.check_env()
    if missing:
        logger.error("Missing env: %s. Set them in .env (see .env.example).", ", ".join(missing))
        sys.exit(1)

    gh = get_github_client()
    _check_github_access(gh)
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    milvus = MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)

    ensure_collection(milvus, collection_name, embedding_dim)
    use_sparse = ensure_sparse_collection(milvus, COLLECTION_NAME_SPARSE)
    use_repo_partition = collection_has_repo_field(milvus, collection_name)
    if not use_repo_partition:
        logger.warning("Collection %s has no 'repo' field (created before partitioning). Inserting without repo. Drop and re-run to use partitioning.", collection_name)

    total_inserted = 0
    for repo_idx, (owner, repo_name, files) in enumerate(
        iter_repos(gh, repos, max_files_per_repo=max_files_per_repo),
        start=1,
    ):
        repo_id = f"{owner}/{repo_name}"
        logger.info("Processing %s (%d/%d repos) ...", repo_id, repo_idx, len(repos))
        if not files:
            logger.info("No eligible files (text/code) in %s/%s", owner, repo_name)
            continue
        # Remove existing rows for this repo so re-runs don't duplicate
        if use_repo_partition:
            try:
                safe = repo_id.replace("\\", "\\\\").replace('"', '\\"')
                milvus.delete(collection_name, filter=f'repo == "{safe}"')
                if use_sparse:
                    milvus.delete(COLLECTION_NAME_SPARSE, filter=f'repo == "{safe}"')
            except Exception as e:
                logger.debug("Delete existing repo rows: %s", e)
        logger.info("Fetched %d files from %s; building chunks (strategy=%s) ...", len(files), repo_id, chunk_strategy)
        sources = []
        contents = []
        for path, content in files:
            for source, chunk_content in file_to_chunks(path, content, repo_id, strategy=chunk_strategy):
                sources.append(source)
                contents.append(chunk_content[:65535])
        logger.info("Embedding %d chunks for %s (this may take a while) ...", len(contents), repo_id)
        vectors = embed_texts(openai_client, contents, model=embedding_model)
        # Insert in batches (id is auto-generated; repo is partition key when schema has it)
        for i in tqdm(range(0, len(contents), INSERT_BATCH_SIZE), desc=f"{owner}/{repo_name}"):
            batch = []
            for v, c, s in zip(
                vectors[i : i + INSERT_BATCH_SIZE],
                contents[i : i + INSERT_BATCH_SIZE],
                sources[i : i + INSERT_BATCH_SIZE],
            ):
                row = {"vector": v, "content": c, "source": s}
                if use_repo_partition:
                    row["repo"] = repo_id
                batch.append(row)
            milvus.insert(collection_name, batch)
            if use_sparse:
                sparse_batch = []
                for c, s in zip(contents[i : i + INSERT_BATCH_SIZE], sources[i : i + INSERT_BATCH_SIZE]):
                    sp = text_to_sparse_vector(c)
                    sparse_row = {"sparse_vector": sp, "content": c[:65535], "source": s}
                    if use_repo_partition:
                        sparse_row["repo"] = repo_id
                    sparse_batch.append(sparse_row)
                milvus.insert(COLLECTION_NAME_SPARSE, sparse_batch)
            total_inserted += len(batch)
        logger.info("Inserted %s rows from %s/%s", len(contents), owner, repo_name)
    # If nothing was inserted (e.g. all repos 403), try demo repo so the pipeline isn't empty
    effective = [s.strip() for s in repos if s.strip() and not s.strip().startswith("#")]
    if total_inserted == 0 and effective and DEMO_REPO not in effective:
        logger.warning("No rows from your repo list (many orgs restrict API). Loading demo repo %s so you have data.", DEMO_REPO)
        for _owner, _repo_name, _files in iter_repos(gh, [DEMO_REPO], max_files_per_repo=max_files_per_repo):
            if not _files:
                break
            _repo_id = f"{_owner}/{_repo_name}"
            if use_repo_partition:
                try:
                    milvus.delete(collection_name, filter=f'repo == "{_repo_id}"')
                    if use_sparse:
                        milvus.delete(COLLECTION_NAME_SPARSE, filter=f'repo == "{_repo_id}"')
                except Exception:
                    pass
            _sources, _contents = [], []
            for _path, _content in _files:
                for _src, _chunk in file_to_chunks(_path, _content, _repo_id, strategy=chunk_strategy):
                    _sources.append(_src)
                    _contents.append(_chunk[:65535])
            _vectors = embed_texts(openai_client, _contents, model=embedding_model)
            for i in range(0, len(_contents), INSERT_BATCH_SIZE):
                batch = []
                for v, c, s in zip(_vectors[i:i + INSERT_BATCH_SIZE], _contents[i:i + INSERT_BATCH_SIZE], _sources[i:i + INSERT_BATCH_SIZE]):
                    row = {"vector": v, "content": c, "source": s}
                    if use_repo_partition:
                        row["repo"] = _repo_id
                    batch.append(row)
                milvus.insert(collection_name, batch)
                if use_sparse:
                    sparse_batch = []
                    for c, s in zip(_contents[i:i + INSERT_BATCH_SIZE], _sources[i:i + INSERT_BATCH_SIZE]):
                        sp = text_to_sparse_vector(c)
                        sparse_row = {"sparse_vector": sp, "content": c[:65535], "source": s, "repo": _repo_id}
                        sparse_batch.append(sparse_row)
                    milvus.insert(COLLECTION_NAME_SPARSE, sparse_batch)
                total_inserted += len(batch)
            logger.info("Inserted %s rows from demo repo %s", len(_contents), DEMO_REPO)
            break
    return total_inserted


def main():
    ap = argparse.ArgumentParser(description="Load GitHub repo files into Milvus.")
    ap.add_argument("--repos", nargs="*", help="List of owner/repo")
    ap.add_argument("--repos-file", type=Path, help="Path to file with owner/repo per line")
    ap.add_argument("--collection", default=COLLECTION_NAME, help="Milvus collection name")
    ap.add_argument(
        "--chunk-strategy",
        choices=CHUNK_STRATEGIES,
        default="file",
        help="Chunking strategy: file (one chunk per small file / windows for large), function (Python ast), markdown (by ##), lines (always windows). Default: file.",
    )
    ap.add_argument(
        "--max-files-per-repo",
        type=int,
        default=MAX_FILES_PER_REPO_DEFAULT,
        help="Only load repos with this many eligible files or fewer (complete repos only; default %d)." % MAX_FILES_PER_REPO_DEFAULT,
    )
    args = ap.parse_args()

    repos = list(args.repos or [])
    if args.repos_file and args.repos_file.exists():
        repos.extend(args.repos_file.read_text().strip().splitlines())
    if not repos:
        print("Provide --repos owner/repo ... or --repos-file path", file=sys.stderr)
        sys.exit(1)

    n = load_repos_to_milvus(
        repos,
        collection_name=args.collection,
        max_files_per_repo=args.max_files_per_repo,
        chunk_strategy=args.chunk_strategy,
    )
    logger.info("Done. Total rows inserted: %s", n)


if __name__ == "__main__":
    main()
