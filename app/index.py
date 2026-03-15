"""Vercel FastAPI entrypoint (app/index.py so it runs as the app, not served as a file)."""
import sys
from pathlib import Path

# Ensure project root is on path when Vercel runs from app/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from main import app
