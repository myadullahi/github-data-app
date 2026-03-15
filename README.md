# GitHub RAG app

RAG app that ingests public GitHub repo data into Milvus and lets you query it with OpenAI. Built in steps: this repo covers **requirements, env, loading data** (query/LLM step is next).

---

## Setup

1. **Python env**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment**
   - Copy `.env.example` to `.env` and set:
     - `OPENAI_API_KEY` – required (embeddings + future LLM)
     - `MILVUS_URI` – Zilliz Cloud cluster endpoint
     - `MILVUS_TOKEN` – API key or `username:password`
     - `GITHUB_TOKEN` – optional; higher rate limits and access to private repos you’re allowed to use (scope: `public_repo` or `repo` for private)
   - Never commit `.env`.

**If you get 403 for every repo (including the demo):** The script will now exit immediately with a message. Common causes:
- **Classic token:** Use scope `public_repo` (or `repo` for private). Create at GitHub → Settings → Developer settings → Personal access tokens (classic).
- **Fine-grained token:** Grant **Contents: Read** and **Metadata: Read**; under "Repository access" choose **All public repositories** (or add each repo you need). Creating access for "Only select repositories" without adding any gives 403 for everything.
- **Quick test:** Remove `GITHUB_TOKEN` from `.env` and run again. Unauthenticated access (60 req/hr) often works for public repos; if 403 goes away, the issue was the token.

---

## Milvus / Zilliz Cloud

- **Credentials:** Set `MILVUS_URI` and `MILVUS_TOKEN` in `.env`. Get them from Zilliz Cloud (Cluster → Connect, API Keys).
- **Collection:** `load_data.py` creates the collection if it doesn’t exist. Schema: `id` (auto), `vector` (1536-dim), `content`, `source`, `repo` (partition key). Partitioning is by `repo` (64 partitions) for scoped search.
- **Dense vectors only** (OpenAI embeddings). Sparse/hybrid can be added later if needed.

---

## Loading data

### Quick test (recommended first)

Many large orgs (Apache, AWS, etc.) restrict GitHub API access, so your list may return 403. To confirm the pipeline works:

```bash
python load_data.py --repos octocat/Hello-World
```

If **all** repos in your list return 403, the script will automatically load **octocat/Hello-World** once so you get at least some rows in Milvus.

### Discover repos by topic (filter happens here)

**Discover only outputs repos that are small enough to load in full** (≤ `--max-files-per-repo` eligible files). That way you control how many repos end up in the DB; if the topic has mostly huge repos, discover keeps searching until it finds enough complete ones (or hits `--max-repos`).

```bash
python discover_repos.py --topic data-engineering --language Python --max-repos 50 --max-files-per-repo 200 --output repos.txt
```

- **`--max-files-per-repo`** (default 200): only list repos with this many eligible files or fewer. Bigger repos are skipped so you don’t get “no data” when load_data would skip them anyway.
- **`--max-repos`**: max lines in the output file (default 50).
- **`--min-stars`**: optional; helps surface focused/smaller repos.

### Load into Milvus

```bash
python load_data.py --repos-file repos.txt
# or
python load_data.py --repos owner1/repo1 owner2/repo2
```

- **Git Data API** first; on 403, **Contents API** fallback. Repos that 403 on both are skipped.
- load_data still skips any repo with more than `--max-files-per-repo` files (safety net for hand-edited lists). Default 200.

### Filter an existing list (optional)

To keep only repos that pass a quick API check:

```bash
python filter_repos.py repos.txt -o repos_verified.txt
python load_data.py --repos-file repos_verified.txt
```

---

## What gets loaded

- **Included:** Source code (`.py`, `.js`, etc.), Markdown, config files, within size limit.
- **Excluded:** `node_modules/`, `vendor/`, `.git/`, `dist/`, lock files, minified assets.
- **Complete repos only:** Repos with more than `--max-files-per-repo` eligible files are **skipped**. Every repo in the index is loaded in full so the RAG has complete context.
- **Chunking:** Small files = one chunk; large files = overlapping line windows; `source` includes path and optional line range (e.g. `github:owner/repo:path#L1-L350`).

---

## Scripts

| Script | Purpose |
|--------|--------|
| `load_data.py` | Ingest GitHub repos → embed (OpenAI) → insert into Milvus. Creates collection if needed; demo repo fallback if 0 rows. |
| `discover_repos.py` | Search repos by topic (and language, stars); write `owner/repo` to a file. |
| `filter_repos.py` | Filter a repo list to only those that allow API access (no 403 on probe). |
| `config.py` | Loads `.env` and exposes `OPENAI_API_KEY`, `MILVUS_URI`, `MILVUS_TOKEN`, `GITHUB_TOKEN`. |

---

## End-to-end example

```bash
source venv/bin/activate
# 1) Ensure .env has OPENAI_API_KEY, MILVUS_URI, MILVUS_TOKEN (and optionally GITHUB_TOKEN)
# 2) Test with a known-good repo
python load_data.py --repos octocat/Hello-World
# 3) Or discover and load (may hit 403 on many; demo repo will load if 0 rows)
python discover_repos.py --topic data-engineering --language Python --max-repos 20 --output repos.txt
python load_data.py --repos-file repos.txt
```

Next step: add a query/LLM layer that searches Milvus and uses OpenAI to answer over the loaded data.
