"""A local stand-in for the laptop inference service (inference_service/server.py), for tests.

It binds 127.0.0.1 on a free port and answers the service's endpoints: GET /v1/health (no
token), GET /v1/status and GET /v1/models/<part> (bearer token), and POST /v1/predict (token).
Tests change its attributes to set the reply: the models it lists, the status, a raw body, a
delay. Standard library only.
"""
from __future__ import annotations

import http.server
import json
import threading
import time
from typing import Any, Optional
from urllib.parse import unquote

FAKE_LAPTOP_TOKEN = "fake-laptop-token-for-tests-only-0003"
MODEL_PREFIX = "/v1/models/"


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "FakeLaptop"
    sys_version = ""

    def log_message(self, *args: Any) -> None:  # keep test output quiet
        pass

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client gave up (a timeout test); nothing to answer

    def _authorised(self) -> bool:
        fake = self.server.fake  # type: ignore[attr-defined]
        return self.headers.get("Authorization") == f"Bearer {fake.token}"

    def _listed(self) -> list:
        fake = self.server.fake  # type: ignore[attr-defined]
        return [{"part_number": pn, "version": v} for pn, v in sorted(fake.models.items())]

    def do_GET(self) -> None:  # noqa: N802
        fake = self.server.fake  # type: ignore[attr-defined]
        fake.requests.append(("GET", self.path, dict(self.headers)))
        if fake.delay:
            time.sleep(fake.delay)
        path = self.path.split("?")[0]
        if path == "/v1/health":  # unauthenticated, as in the real service
            if fake.health_raw is not None:
                self._send(fake.health_status, fake.health_raw)
                return
            payload = {"status": "ok", "busy": False, "in_flight": 0}
            self._send(fake.health_status, json.dumps(payload).encode("utf-8"))
            return
        if not self._authorised():
            self._send(401, b'{"error": "unauthorized"}')
            return
        if path == "/v1/status":
            payload = {
                "status": "ok", "version": "1", "models": self._listed(),
                "busy": False, "in_flight": 0, "gpu": {"torch_loaded": False, "cuda_available": None},
            }
            self._send(200, json.dumps(payload).encode("utf-8"))
            return
        if path.startswith(MODEL_PREFIX):
            if fake.model_raw is not None:
                self._send(fake.model_status, fake.model_raw)
                return
            part = unquote(path[len(MODEL_PREFIX):])
            if part in fake.models:
                payload = {"part_number": part, "approved": True, "model_version": fake.models[part]}
                self._send(200, json.dumps(payload).encode("utf-8"))
                return
            payload = {"approved": False, "reason": f"No approved model for part {part}. Ask Quality to approve one."}
            self._send(404, json.dumps(payload).encode("utf-8"))
            return
        self._send(404, b'{"error": "not found"}')

    def do_POST(self) -> None:  # noqa: N802
        fake = self.server.fake  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        fake.requests.append(("POST", self.path, dict(self.headers)))
        if fake.delay:
            time.sleep(fake.delay)
        if not self._authorised():
            self._send(401, b'{"error": "unauthorized"}')
            return
        if self.path.split("?")[0] != "/v1/predict":
            self._send(404, b'{"error": "not found"}')
            return
        if fake.predict_raw is not None:
            self._send(fake.predict_status, fake.predict_raw)
            return
        if fake.predict_status != 200:
            self._send(fake.predict_status, json.dumps({"error": fake.predict_error}).encode("utf-8"))
            return
        part = self.headers.get("Part-Number", "")
        if part not in fake.models:
            self._send(503, json.dumps({"error": f"No approved model for part {part}."}).encode("utf-8"))
            return
        payload = {
            "camera_id": self.headers.get("Camera-Id", ""),
            "side": self.headers.get("Side", ""),
            "score": fake.predict_score,
            "defects": [],
            "model_version": fake.models[part],
        }
        payload.update(fake.predict_fields)
        for key in fake.predict_drop:
            payload.pop(key, None)
        fake.last_predict_body = body
        self._send(200, json.dumps(payload).encode("utf-8"))


class FakeLaptop:
    """Start with FakeLaptop(); read .base for the URL; close() when done."""

    def __init__(self, models: Optional[dict] = None, token: str = FAKE_LAPTOP_TOKEN) -> None:
        self.token = token
        self.models: dict[str, str] = dict(models or {})
        self.health_status = 200
        self.health_raw: Optional[bytes] = None
        self.model_status = 200
        self.model_raw: Optional[bytes] = None  # raw reply for GET /v1/models/<part>
        self.predict_status = 200
        self.predict_error = "GPU out of memory while scoring"
        self.predict_raw: Optional[bytes] = None
        self.predict_score = 0.0
        self.predict_fields: dict = {}
        self.predict_drop: list = []
        self.delay = 0.0  # seconds to wait before answering any request
        self.requests: list = []
        self.last_predict_body: bytes = b""
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.server.fake = self  # type: ignore[attr-defined]
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
