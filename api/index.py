"""
Vercel serverless entrypoint: wrap FastAPI app with Mangum so Vercel can invoke it.
All routes (/, /health, /chat, /docs, etc.) are handled by the FastAPI app.
"""
from mangum import Mangum

from main import app

handler = Mangum(app, lifespan="auto")
