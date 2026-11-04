"""Client for the laptop inference service (inference_service/server.py, plan L1).

RemoteModel implements the InferenceModel protocol: predict(frame) posts the raw picture to
POST /v1/predict and maps the reply to an ImageResult. Standard library only.

Any timeout, HTTP error, bad JSON, missing or malformed field, or a model version that changed
mid-session raises StationError, so the session goes to FAULT and never to GOOD.

The bearer token is read once from the environment variable named in config. It is sent only in
the Authorization header and never appears in an error message. Redirects are not followed, and
environment proxies are bypassed, so the token goes straight to the laptop.
"""
from __future__ import annotations

import http.client
import json
import math
import os
import urllib.error
import urllib.request
from typing import Any

from station.contracts import Defect, Frame, ImageResult, Side, StationError

PREDICT_PATH = "/v1/predict"
HEALTH_PATH = "/v1/health"
_CONTENT_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg"}
_ERROR_TEXT_MAX = 200


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not follow redirects: the token must not be sent to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _scrub(text: str, token: str) -> str:
    return text.replace(token, "[redacted]") if token else text


def _read_quietly(exc: urllib.error.HTTPError) -> bytes:
    try:
        return exc.read(4096)
    except Exception:  # noqa: BLE001 - the body is only used for a short message
        return b""


def _laptop_message(raw: bytes, token: str) -> str:
    """The laptop's own {"error": ...} text, short and without the token; '' when absent."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return ""
    if not isinstance(body, dict) or not isinstance(body.get("error"), str):
        return ""
    return _scrub(body["error"], token)[:_ERROR_TEXT_MAX]


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise StationError(f"laptop reply has no usable number for {what}")
    return float(value)


def _score(value: Any, what: str) -> float:
    number = _number(value, what)
    if not 0.0 <= number <= 1.0:
        raise StationError(f"laptop reply {what} is out of range 0..1")
    return number


def _defect(item: Any) -> Defect:
    if not isinstance(item, dict):
        raise StationError("laptop reply has a malformed defect")
    name = item.get("defect_class")
    if not isinstance(name, str) or not name:
        raise StationError("laptop reply defect has no defect_class")
    box = item.get("box")
    if (not isinstance(box, list) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in box)):
        raise StationError("laptop reply defect box must be four whole non-negative numbers")
    return Defect(defect_class=name, score=_score(item.get("score"), "defect score"),
                  box=(box[0], box[1], box[2], box[3]))


class RemoteModel:
    """InferenceModel that scores pictures on the laptop inference service.

    version is '' until check_approved() (GET /v1/health) or the first successful predict()
    sets it. After that, a reply with another model_version is a StationError.
    """

    def __init__(self, base_url: str, token_env: str, timeout_s: float, part_number: str) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("laptop base URL must start with http:// or https://")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not timeout_s > 0:
            raise ValueError("timeout_s must be a positive number")
        if not isinstance(part_number, str) or not part_number.strip():
            raise ValueError("part number must be a non-empty string")
        token = os.environ.get(token_env, "") if isinstance(token_env, str) else ""
        if not token:
            raise StationError(f"the laptop token is not set: the environment variable {token_env} is empty")
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = float(timeout_s)
        self.part_number = part_number.strip()
        self.version = ""
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)

    def __repr__(self) -> str:  # never show the token
        return f"RemoteModel(part_number={self.part_number!r}, version={self.version!r})"

    # ------------------------------------------------------------ HTTP

    def _request(self, req: urllib.request.Request, what: str) -> bytes:
        try:
            with self._opener.open(req, timeout=self._timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            detail = _laptop_message(_read_quietly(exc), self._token)
            exc.close()
            if exc.code == 401:
                text = "laptop rejected the token (401)"
            elif exc.code == 503:
                text = "laptop inference service unavailable (503)"
            elif exc.code >= 500:
                text = f"laptop inference service error ({exc.code})"
            else:
                text = f"laptop answered {exc.code}"
            raise StationError(f"{what}: {text}" + (f": {detail}" if detail else "")) from None
        except (TimeoutError, OSError, http.client.HTTPException, urllib.error.URLError) as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
                raise StationError(f"{what}: timed out after {self._timeout:g} s") from None
            name = type(reason or exc).__name__
            raise StationError(f"{what}: could not reach the laptop ({name})") from None
        except ValueError:
            raise StationError(f"{what}: invalid request to the laptop") from None

    @staticmethod
    def _json(raw: bytes, what: str) -> dict:
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise StationError(f"{what}: laptop reply is not JSON") from None
        if not isinstance(body, dict):
            raise StationError(f"{what}: laptop reply is not a JSON object")
        return body

    def _headers(self, extra: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        headers.update(extra or {})
        return headers

    # ------------------------------------------------------------ API

    def check_approved(self) -> str:
        """Ask the laptop which approved model it lists for this part (GET /v1/health).

        Sets and returns the version. StationError when the laptop is unreachable, rejects the
        token, or does not list this part (no approved model, or not loaded yet).
        """
        req = urllib.request.Request(self._base + HEALTH_PATH, headers=self._headers(), method="GET")
        body = self._json(self._request(req, "laptop health check"), "laptop health check")
        models = body.get("models")
        if not isinstance(models, list):
            raise StationError("laptop health check: reply has no model list")
        for item in models:
            if isinstance(item, dict) and item.get("part_number") == self.part_number:
                version = item.get("version")
                if not isinstance(version, str) or not version:
                    raise StationError(f"laptop health check: no version for part {self.part_number}")
                self.version = version
                return version
        raise StationError(
            f"the laptop does not list an approved model for part {self.part_number}; "
            "ask Quality to approve one and check that the inference service has it loaded"
        )

    def predict(self, frame: Frame) -> ImageResult:
        content_type = _CONTENT_TYPES.get(str(frame.ext).lower())
        if content_type is None:
            raise StationError(f"picture format {frame.ext!r} is not sent to the laptop (png or jpeg only)")
        side = Side(frame.side)
        headers = self._headers({
            "Content-Type": content_type,
            "Part-Number": self.part_number,
            "Camera-Id": frame.camera_id,
            "Side": side.value,
        })
        req = urllib.request.Request(self._base + PREDICT_PATH, data=bytes(frame.data),
                                     headers=headers, method="POST")
        what = f"laptop predict for {frame.camera_id} {side.value}"
        body = self._json(self._request(req, what), what)

        if body.get("camera_id") != frame.camera_id or body.get("side") != side.value:
            raise StationError(f"{what}: reply is for another picture")
        reply_version = body.get("model_version")
        if not isinstance(reply_version, str) or not reply_version:
            raise StationError(f"{what}: reply has no model_version")
        if self.version and reply_version != self.version:
            raise StationError(
                f"{what}: laptop answered with model version {reply_version}, expected {self.version}; "
                "the approved model changed during this part"
            )
        self.version = reply_version

        score = _score(body.get("score"), "score")
        raw_defects = body.get("defects", [])
        if not isinstance(raw_defects, list):
            raise StationError(f"{what}: defects must be a list")
        return ImageResult(camera_id=frame.camera_id, side=side, score=score,
                           defects=tuple(_defect(d) for d in raw_defects))
