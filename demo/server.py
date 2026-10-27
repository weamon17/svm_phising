"""Local web demo: python demo/server.py [port]  ->  http://127.0.0.1:8000

Standard library only (no Flask/Streamlit), works offline, listens on localhost only.
GET /            the page (demo/index.html)
GET /api/model   model summary
POST /api/predict  {"url": "..."} -> prediction + explanation
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.demo import DemoModel  # noqa: E402

MODEL = DemoModel()
PAGE = (ROOT / "demo" / "index.html").read_bytes()
MAX_BODY = 10_000


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/model":
            self._json(200, MODEL.summary())
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/predict":
            return self._json(404, {"error": "Not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._json(413, {"error": "Request too large"})
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None
        if not isinstance(payload, dict):
            return self._json(400, {"error": "Body must be a JSON object: {\"url\": \"...\"}"})
        try:
            result = MODEL.predict(str(payload.get("url", "")))
        except ValueError as error:
            return self._json(400, {"error": str(error)})
        self._json(200, result)

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second process bind a port that is already in use,
    # so two demos would silently share it. Fail loudly there instead.
    allow_reuse_address = os.name != "nt"


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    try:
        server = Server(("127.0.0.1", port), Handler)
    except OSError:
        sys.exit(f"Port {port} is already in use. Stop the other server or run: python demo/server.py {port + 1}")
    print(f"SVM phishing demo running at http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
