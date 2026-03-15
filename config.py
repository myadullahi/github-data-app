"""
Load API keys and settings from environment.
Set these in .env (copy from .env.example) or export in shell.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # optional; for higher rate limits / private access


def check_env() -> list[str]:
    """Return list of missing required env var names."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not MILVUS_URI:
        missing.append("MILVUS_URI")
    if not MILVUS_TOKEN:
        missing.append("MILVUS_TOKEN")
    return missing
