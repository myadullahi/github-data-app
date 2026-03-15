"""Vercel FastAPI entrypoint. Mount main app at /api so rewrites (.*) -> /api/$1 hit our routes."""
import sys
from pathlib import Path

# Ensure project root is on path when Vercel runs from app/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fastapi import FastAPI
from main import app as main_app

# Mount our app at /api so Vercel rewrite /(.*) -> /api/$1 delivers /, /health, /chat to us
app = FastAPI()
app.mount("/api", main_app)
