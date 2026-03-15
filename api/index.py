"""
Vercel serverless entrypoint: wrap FastAPI app with Mangum so Vercel can invoke it.
All routes (/, /health, /chat, /docs, etc.) are handled by the FastAPI app.
"""
import traceback

from mangum import Mangum

# Lazy-import app so we can catch import errors (e.g. pymilvus) and return them instead of crashing
_app = None


def _get_app():
    global _app
    if _app is None:
        from main import app
        _app = app
    return _app


def _handler(event, context):
    try:
        app = _get_app()
        return Mangum(app, lifespan="auto")(event, context)
    except Exception as e:
        body = f"Startup/import error:\n{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "text/plain; charset=utf-8"},
            "body": body,
        }


handler = _handler
