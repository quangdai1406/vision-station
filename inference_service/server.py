"""HTTP server for the laptop inference service. Standard library only at import time.

POST /v1/predict
    headers: Authorization: Bearer <token>, Part-Number, Camera-Id, Side (TOP or BOTTOM),
             Content-Type image/png or image/jpeg
    body: the picture bytes (raw, not multipart)
    200: {camera_id, side, score, defects:[{defect_class, score, box:[x,y,w,h]}], model_version}
GET /v1/health
    200: {status, version, models:[{part_number, version}], busy, in_flight, gpu}

Scores are raw: the station applies t_low and t_img, never this service. The token is read from
the environment variable named by --token-env and is never printed, logged or returned.
"""
from __future__ import annotations

import argparse
import contextlib
import hmac
import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from inference_service.registry import ModelRegistry, ModelUnavailable, free_gpu_cache, is_out_of_memory
from station import modelstore as ms
from station.contracts import Frame, Side, StationError

SERVICE_VERSION = "1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_TOKEN_ENV = "INFERENCE_TOKEN"
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024  # largest picture accepted; an engineering limit, not a quality one
PREDICT_PATH = "/v1/predict"
HEALTH_PATH = "/v1/health"
_CAMERA_RE = re.compile(r"[A-Za-z0-9_-]{1,16}")
_PART_MAX = 128
_CONTENT_TYPES = {"image/png": "png", "image/jpeg": "jpeg"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_predict_headers(headers: Any) -> tuple[str, str, Side]:
    """(part_number, camera_id, side) from the request headers. ValueError names the bad header."""
    part = (headers.get("Part-Number") or "").strip()
    if not part or len(part) > _PART_MAX:
        raise ValueError(f"Part-Number header is required (1 to {_PART_MAX} characters)")
    camera = (headers.get("Camera-Id") or "").strip()
    if not _CAMERA_RE.fullmatch(camera):
        raise ValueError("Camera-Id header must be 1 to 16 letters, digits, '_' or '-'")
    try:
        side = Side((headers.get("Side") or "").strip().upper())
    except ValueError:
        raise ValueError("Side header must be TOP or BOTTOM") from None
    return part, camera, side


def gpu_info() -> dict:
    """GPU state, best effort. Reports only what is already loaded: never imports torch."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {"torch_loaded": False, "cuda_available": None}
    try:
        cuda = bool(torch.cuda.is_available())
        return {"torch_loaded": True, "cuda_available": cuda,
                "device": torch.cuda.get_device_name(0) if cuda else None}
    except Exception:  # noqa: BLE001 - health must not fail because of the GPU
        return {"torch_loaded": True, "cuda_available": None, "error": "unavailable"}


class Service:
    """Shared state for the handler threads: the token, the registry and the busy counter."""

    def __init__(self, registry: ModelRegistry, token: str, max_body_bytes: int) -> None:
        if not isinstance(token, str) or not token:
            raise ValueError("an access token is required")
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be positive")
        self.registry = registry
        self.max_body_bytes = max_body_bytes
        self._token = token.encode("utf-8")
        self._count_lock = threading.Lock()
        self._in_flight = 0

    def authorised(self, header_value: Optional[str]) -> bool:
        scheme, _, candidate = (header_value or "").partition(" ")
        if scheme.lower() != "bearer" or not candidate.strip():
            return False
        return hmac.compare_digest(candidate.strip().encode("utf-8"), self._token)

    @contextlib.contextmanager
    def busy(self) -> Iterator[None]:
        with self._count_lock:
            self._in_flight += 1
        try:
            yield
        finally:
            with self._count_lock:
                self._in_flight -= 1

    def in_flight(self) -> int:
        with self._count_lock:
            return self._in_flight


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    service: Service


class _Handler(BaseHTTPRequestHandler):
    server_version = "InferenceService"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base class signature
        return  # the default would print to stderr with every header; we log in log_request

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        sys.stderr.write(f"{self.command} {self.path.split('?')[0]} {code}\n")

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _authorised(self) -> bool:
        if self.server.service.authorised(self.headers.get("Authorization")):
            return True
        self._json(401, {"error": "unauthorized"})
        return False

    def do_GET(self) -> None:  # noqa: N802 - http.server name
        if not self._authorised():
            return
        if self.path.split("?")[0] != HEALTH_PATH:
            self._json(404, {"error": "not found"})
            return
        service = self.server.service
        self._json(200, {
            "status": "ok",
            "version": SERVICE_VERSION,
            "models": service.registry.loaded(),
            "busy": service.in_flight() > 0,
            "in_flight": service.in_flight(),
            "gpu": gpu_info(),
        })

    def do_POST(self) -> None:  # noqa: N802 - http.server name
        if not self._authorised():
            return
        if self.path.split("?")[0] != PREDICT_PATH:
            self._json(404, {"error": "not found"})
            return
        self._predict()

    def _predict(self) -> None:
        service = self.server.service
        length_text = self.headers.get("Content-Length")
        if length_text is None or not length_text.isdigit():
            self._json(411, {"error": "Content-Length is required"})
            return
        length = int(length_text)
        if length == 0:
            self._json(400, {"error": "the request body is empty"})
            return
        if length > service.max_body_bytes:
            self._json(413, {"error": "the picture is larger than the service accepts"})
            return
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        ext = _CONTENT_TYPES.get(content_type)
        if ext is None:
            self._json(415, {"error": "Content-Type must be image/png or image/jpeg"})
            return
        try:
            part, camera, side = parse_predict_headers(self.headers)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self._json(400, {"error": "the request body was cut short"})
            return
        with service.busy():
            try:
                loaded = service.registry.get(part)
                frame = Frame(camera_id=camera, side=side, seq=0, timestamp_utc=utc_now(), data=body, ext=ext)
                result = loaded.model.predict(frame)
            except ModelUnavailable as exc:
                self._json(503, {"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001 - map every failure to a JSON answer
                if is_out_of_memory(exc):
                    service.registry.release(part)
                    self._json(503, {"error": "GPU out of memory while scoring; the model was unloaded "
                                              "and will be retried on the next request"})
                    return
                if isinstance(exc, StationError):
                    self._json(422, {"error": str(exc)})
                    return
                sys.stderr.write(f"predict failed: {type(exc).__name__}\n")
                self._json(500, {"error": "internal error"})
                return
        self._json(200, {
            "camera_id": result.camera_id,
            "side": Side(result.side).value,
            "score": float(result.score),
            "defects": [
                {"defect_class": d.defect_class, "score": float(d.score), "box": [int(v) for v in d.box]}
                for d in result.defects
            ],
            "model_version": loaded.version,
        })


def build_server(registry: ModelRegistry, *, token: str, host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT, max_body_bytes: int = DEFAULT_MAX_BODY_BYTES) -> _Server:
    """A bound, not yet serving, server. port=0 picks a free port (tests use it)."""
    server = _Server((host, port), _Handler)
    server.service = Service(registry, token, max_body_bytes)
    return server


def _extractor_factory(spec: Optional[dict]) -> Callable[[], Any]:
    def factory() -> Any:
        if spec is None:
            raise StationError("no extractor config: start the service with --extractor-config FILE")
        from station.features_torch import TorchExtractor  # noqa: PLC0415 - needs torch at run time

        return TorchExtractor(spec["arch"], device=spec.get("device", "auto"),
                              input_hw=tuple(spec["input_hw"]), weights_path=spec.get("weights_path"))

    return factory


def default_loader(extractor_spec: Optional[dict]) -> Callable[[ms.ModelInfo], Any]:
    """The production loader: the same model classes and decoder the station uses for PatchCore."""

    def load(info: ms.ModelInfo) -> Any:
        if info.type != "patchcore":
            raise ModelUnavailable(f"this service serves patchcore models, not {info.type}")
        from station.anomaly import cv2_rgb_decoder, register_patchcore_loader  # noqa: PLC0415 - lazy

        decode = cv2_rgb_decoder()  # cv2 is imported here, at load time, not at startup
        register_patchcore_loader(_extractor_factory(extractor_spec))
        return ms.load_runtime_model(info, decode)

    return load


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m inference_service",
                                     description="Score station pictures with the approved part models.")
    parser.add_argument("--models", required=True, type=Path, help="the ModelStore root folder")
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to bind")
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV,
                        help="NAME of the environment variable holding the access token")
    parser.add_argument("--extractor-config", type=Path, default=None,
                        help="JSON file with arch, device, input_hw and weights_path (as config model.extractor)")
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_MAX_BODY_BYTES)
    args = parser.parse_args(argv)

    token = os.environ.get(args.token_env, "")
    if not token:
        sys.stderr.write(f"set the environment variable {args.token_env} to the access token "
                         "(its value is not printed)\n")
        return 2
    if not args.models.is_dir():
        sys.stderr.write(f"models folder not found: {args.models}\n")
        return 2
    spec = None
    if args.extractor_config is not None:
        try:
            spec = json.loads(args.extractor_config.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"could not read --extractor-config: {exc}\n")
            return 2
    registry = ModelRegistry(ms.ModelStore(args.models), default_loader(spec))
    server = build_server(registry, token=token, host=args.host, port=args.port,
                          max_body_bytes=args.max_body_bytes)
    host, port = server.server_address[:2]
    sys.stderr.write(f"inference service listening on {host}:{port}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
