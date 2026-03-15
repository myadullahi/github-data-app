"""
Vercel FastAPI entrypoint (zero-config).
Expose the FastAPI app here so Vercel uses ASGI instead of the legacy class handler.
"""
from main import app
