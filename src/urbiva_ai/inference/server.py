"""Local HTTP endpoint compatible with the app's POST /api/vision.

Without a checkpoint the server answers 503 {status: "not_configured"}, just
like the application does without an API key, so the app falls back to
manual classification. Images are processed in memory and never stored.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from urbiva_ai.inference.contract import MAX_BYTES, ContractError, VisionResult, decode_request

Predict = Callable[[bytes], VisionResult]

RATE_WINDOW_S = 60.0
RATE_MAX = 10


class RateLimiter:
    def __init__(self, max_requests: int = RATE_MAX, window_s: float = RATE_WINDOW_S, clock=time.monotonic):
        self.max, self.window, self.clock = max_requests, window_s, clock
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self.lock:
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.max:
                return False
            q.append(now)
            return True


def handle(body: bytes, predict: Optional[Predict], model_name: str) -> tuple[int, dict]:
    """Pure request handler, used by the HTTP server and by tests."""
    if predict is None:
        return 503, {"status": "not_configured", "message": "No local model is loaded. Classify the barrier manually."}
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 400, {"status": "error", "message": "Request body must be valid JSON."}
    try:
        raw, _media = decode_request(payload)
    except ContractError as err:
        return err.status, {"status": "error", "message": err.message}
    try:
        result = predict(raw)
    except Exception:  # noqa: BLE001 - never leak internals to the client
        return 502, {"status": "error", "message": "The image could not be analysed. Classify it manually."}
    return 200, {"status": "ok", "result": result.to_dict(), "model": model_name}


def make_handler(predict: Optional[Predict], model_name: str, limiter: RateLimiter):
    class Handler(BaseHTTPRequestHandler):
        server_version = "urbiva-ai"

        def _send(self, status: int, obj: dict) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/vision":
                self._send(404, {"status": "error", "message": "Not found."})
                return
            if not limiter.allow(self.client_address[0]):
                self._send(429, {"status": "error", "message": "Too many requests. Try again in a minute."})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BYTES * 2:
                self._send(413, {"status": "error", "message": "The request is too large."})
                return
            status, obj = handle(self.rfile.read(length), predict, model_name)
            self._send(status, obj)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._send(200, {"status": "ok", "model_loaded": predict is not None})
            else:
                self._send(405, {"status": "error", "message": "Use POST /api/vision."})

        def log_message(self, fmt: str, *args) -> None:
            return  # no request logging: requests contain images

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, checkpoint: str | None = None) -> None:
    predict: Optional[Predict] = None
    name = "uva-net-s10"
    if checkpoint:
        from urbiva_ai.inference.predictor import Predictor

        p = Predictor(checkpoint)
        predict, name = p.predict, f"uva-net-s10:{p.cfg.experiment_id}"
    httpd = ThreadingHTTPServer((host, port), make_handler(predict, name, RateLimiter()))
    print(f"urbiva-ai serving on http://{host}:{port}/api/vision (model loaded: {predict is not None})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
