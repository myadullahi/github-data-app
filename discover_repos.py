"""
Discover public GitHub repos by knowledge area (topic) for loading into the RAG collection.

Uses GitHub Search API: topic + optional language and min stars.
By default only outputs repos with ≤ --max-files-per-repo eligible files (complete repos), so you
control how many repos end up in the DB and avoid loading huge repos that would be skipped later.

Usage:
  python discover_repos.py --topic data-engineering --language Python --max-repos 50 --output repos.txt
  python discover_repos.py --topic machine-learning --max-files-per-repo 200 --min-stars 50

Requires GITHUB_TOKEN in .env for reasonable rate limits (search is 30 req/min authenticated).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config
from github import Auth, Github
from github.GithubException import GithubException

from repo_filters import is_text_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MAX_FILES_PER_REPO = 200


def _count_eligible_files(gh: Github, full_name: str) -> int | None:
    """
    Return number of eligible (text/code) files in repo, or None if 403/404.
    Uses same rules as load_data (repo_filters.is_text_file).
    """
    try:
        repo = gh.get_repo(full_name)
        tree = repo.get_git_tree(repo.default_branch, recursive=True)
        return sum(
            1 for item in tree.tree
            if item.type == "blob" and is_text_file(item.path, item.size)
        )
    except GithubException as e:
        if e.status in (403, 404):
            return None
        raise
    except Exception:
        return None


def discover_repos(
    topic: str,
    *,
    language: str | None = None,
    min_stars: int = 0,
    max_repos: int = 100,
    max_files_per_repo: int = DEFAULT_MAX_FILES_PER_REPO,
    verify_api_access: bool = False,
) -> list[str]:
    """
    Search GitHub for public repos by topic. Only returns repos with ≤ max_files_per_repo
    eligible files (complete repos). Returns list of "owner/repo".
    """
    # Build query: topic:X [language:Y] [stars:>N]
    query_parts = [f"topic:{topic}"]
    if language:
        query_parts.append(f"language:{language}")
    if min_stars > 0:
        query_parts.append(f"stars:>{min_stars}")
    query = " ".join(query_parts)

    gh = Github(auth=Auth.Token(config.GITHUB_TOKEN) if config.GITHUB_TOKEN else None, retry=None)
    repos = gh.search_repositories(query=query)
    seen = set()
    out = []
    try:
        for i, repo in enumerate(repos):
            if len(out) >= max_repos:
                break
            full_name = repo.full_name
            if full_name in seen:
                continue
            seen.add(full_name)
            count = _count_eligible_files(gh, full_name)
            if count is None:
                logger.info("Skip %s (403/404 - API restricted or not found)", full_name)
                continue
            if count > max_files_per_repo:
                logger.info("Skip %s (has %d files; only complete repos with ≤%d)", full_name, count, max_files_per_repo)
                continue
            if count == 0:
                logger.info("Skip %s (no eligible text/code files)", full_name)
                continue
            out.append(full_name)
            logger.info("[%d] %s (stars=%s, %d files)", len(out), full_name, repo.stargazers_count, count)
    except Exception as e:
        logger.warning("Search stopped: %s", e)
    return out


def main():
    ap = argparse.ArgumentParser(description="Discover GitHub repos by topic for RAG ingestion (complete repos only).")
    ap.add_argument("--topic", required=True, help="GitHub topic (e.g. machine-learning, react, data-engineering)")
    ap.add_argument("--language", default=None, help="Filter by language (e.g. Python, JavaScript)")
    ap.add_argument("--min-stars", type=int, default=0, help="Minimum star count (default 0)")
    ap.add_argument("--max-repos", type=int, default=50, help="Max repos to write (default 50)")
    ap.add_argument(
        "--max-files-per-repo",
        type=int,
        default=DEFAULT_MAX_FILES_PER_REPO,
        help="Only include repos with this many eligible files or fewer (default %d)" % DEFAULT_MAX_FILES_PER_REPO,
    )
    ap.add_argument("--output", type=Path, default=None, help="Write owner/repo per line (default: stdout)")
    args = ap.parse_args()

    if not config.GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN not set. Search rate limit is strict (10/min). Set it in .env for 30/min.")

    repos = discover_repos(
        args.topic,
        language=args.language,
        min_stars=args.min_stars,
        max_repos=args.max_repos,
        max_files_per_repo=args.max_files_per_repo,
    )
    lines = [r + "\n" for r in repos]
    if args.output:
        args.output.write_text("".join(lines), encoding="utf-8")
        logger.info("Wrote %d repos to %s", len(lines), args.output)
    else:
        sys.stdout.writelines(lines)
    return 0 if repos else 1


if __name__ == "__main__":
    sys.exit(main())
