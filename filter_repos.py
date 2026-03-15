"""
Filter repos.txt to only repos that allow API access (no 403).

Reads owner/repo lines from a file, probes each with GitHub API (get_repo + get_git_tree).
Writes only repos that succeed (no 403/404) so load_data.py won't hit restricted repos.

Usage:
  python filter_repos.py repos.txt -o repos_verified.txt
  python filter_repos.py repos.txt   # writes to stdout
  python filter_repos.py repos.txt -o repos_verified.txt --progress

Requires GITHUB_TOKEN in .env. Uses retry=None so 403 fails immediately.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config
from github import Auth, Github
from github.GithubException import GithubException

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _repo_is_accessible(gh: Github, full_name: str) -> bool:
    """Probe repo with get_repo + get_git_tree; return False if 403/404."""
    try:
        repo = gh.get_repo(full_name)
        repo.get_git_tree(repo.default_branch, recursive=False)
        return True
    except GithubException as e:
        if e.status in (403, 404):
            return False
        raise
    except Exception:
        return False


def filter_repos(repo_specs: list[str], *, show_progress: bool = False) -> list[str]:
    """
    Return only repo specs that allow API access (no 403/404).
    """
    if not config.GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN required. Set it in .env.")
        return []

    gh = Github(auth=Auth.Token(config.GITHUB_TOKEN), retry=None)
    out = []
    for i, spec in enumerate(repo_specs, start=1):
        spec = spec.strip()
        if not spec or spec.startswith("#"):
            continue
        normalized = spec.replace(".", "/", 1) if "/" not in spec else spec
        if "/" not in normalized or normalized.count("/") != 1:
            logger.warning("Invalid spec: %s", spec)
            continue
        if show_progress:
            logger.info("Checking %s (%d/%d) ...", normalized, i, len(repo_specs))
        if _repo_is_accessible(gh, normalized):
            out.append(normalized)
            logger.info("  OK %s", normalized)
        else:
            logger.info("  Skip %s (403/404 - API restricted or not found)", normalized)
    return out


def main():
    ap = argparse.ArgumentParser(description="Filter repo list to only API-accessible repos.")
    ap.add_argument("input", type=Path, help="Input file (owner/repo per line)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output file (default: stdout)")
    ap.add_argument("--progress", action="store_true", help="Log each repo as it is checked")
    args = ap.parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 1

    specs = args.input.read_text(encoding="utf-8").strip().splitlines()
    allowed = filter_repos(specs, show_progress=args.progress)
    lines = [r + "\n" for r in allowed]

    if args.output:
        args.output.write_text("".join(lines), encoding="utf-8")
        logger.info("Wrote %d API-accessible repos to %s (skipped %d)", len(allowed), args.output, len(specs) - len(allowed))
    else:
        sys.stdout.writelines(lines)
    return 0 if allowed else 1


if __name__ == "__main__":
    sys.exit(main())
