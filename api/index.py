"""
Vercel Python serverless: runtime expects a class handler(BaseHTTPRequestHandler).
We delegate to the FastAPI app via Mangum by building an API Gateway-style event.
"""
import asyncio
import json
import traceback
from http.server import BaseHTTPRequestHandler


def _parse_query(qs: str) -> dict:
    if not qs:
        return None
    out = {}
    for part in qs.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k in out:
                if isinstance(out[k], list):
                    out[k].append(v)
                else:
                    out[k] = [out[k], v]
            else:
                out[k] = v
    return {k: (v if isinstance(v, list) else [v]) for k, v in out.items()} if out else None


def _build_event(handler_self) -> dict:
    path = handler_self.path
    if "?" in path:
        path, qs = path.split("?", 1)
    else:
        qs = ""
    # Vercel rewrites /(.*) -> /api/$1, so path is like /api/ or /api/health; strip /api for FastAPI
    if path.startswith("/api"):
        path = path[4:] or "/"
    content_length = int(handler_self.headers.get("Content-Length", 0) or 0)
    body = handler_self.rfile.read(content_length).decode("utf-8", errors="replace") if content_length else None
    headers = {k.lower(): v for k, v in handler_self.headers.items()}
    qparams = _parse_query(qs)
    return {
        "version": "1.0",
        "resource": path,
        "path": path,
        "httpMethod": handler_self.command,
        "headers": headers,
        "multiValueHeaders": {k: [v] for k, v in headers.items()},
        "queryStringParameters": {k: (v[0] if len(v) == 1 else ",".join(v)) for k, v in (qparams or {}).items()},
        "multiValueQueryStringParameters": qparams,
        "requestContext": {"httpMethod": handler_self.command, "path": path, "requestTimeEpoch": 0},
        "body": body,
        "isBase64Encoded": False,
    }


def _handle_request(handler_self):
    try:
        event = _build_event(handler_self)
        context = {}
        from mangum import Mangum
        from main import app
        mangum = Mangum(app, lifespan="auto")
        response = asyncio.run(mangum(event, context))
        status = int(response.get("statusCode", 500))
        headers = response.get("headers") or {}
        body = response.get("body") or ""
        if isinstance(body, dict):
            body = json.dumps(body)
        if not isinstance(body, (bytes, str)):
            body = str(body)
        if isinstance(body, str):
            body = body.encode("utf-8")
        handler_self.send_response(status)
        for k, v in headers.items():
            handler_self.send_header(k, str(v))
        handler_self.send_header("Content-Length", str(len(body)))
        handler_self.end_headers()
        handler_self.wfile.write(body)
    except Exception as e:
        err_body = f"Handler error:\n{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        err_bytes = err_body.encode("utf-8")
        handler_self.send_response(500)
        handler_self.send_header("Content-Type", "text/plain; charset=utf-8")
        handler_self.send_header("Content-Length", str(len(err_bytes)))
        handler_self.end_headers()
        handler_self.wfile.write(err_bytes)


class handler(BaseHTTPRequestHandler):
    """Vercel expects this class name and base class."""

    def do_GET(self):
        _handle_request(self)

    def do_POST(self):
        _handle_request(self)

    def do_PUT(self):
        _handle_request(self)

    def do_DELETE(self):
        _handle_request(self)

    def do_HEAD(self):
        _handle_request(self)

    def do_OPTIONS(self):
        _handle_request(self)

    def log_message(self, format, *args):
        pass
