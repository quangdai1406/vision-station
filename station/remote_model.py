from __future__ import annotations

import http.client
import json
import math
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote

from station.contracts import Defect, Frame, ImageResult, Side, StationError

PREDICT_PATH = "/v1/predict"
MODEL_PREFIX = "/v1/models/"
_CONTENT_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg"}
_ERROR_TEXT_MAX = 200


class _NoRedirect(urllib.request.HTTPRedirectHandler):

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _scrub(text: str, token: str) -> str:
    return text.replace(token, "[redacted]") if token else text


def _read_quietly(exc: urllib.error.HTTPError) -> bytes:
    try:
        return exc.read(4096)
    except Exception:
        return b""


def _laptop_message(raw: bytes, token: str) -> str:
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

    def __repr__(self) -> str:
        return f"RemoteModel(part_number={self.part_number!r}, version={self.version!r})"


    def _send(self, req: urllib.request.Request, what: str) -> tuple[int, bytes]:
        try:
            with self._opener.open(req, timeout=self._timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            body = _read_quietly(exc)
            exc.close()
            return exc.code, body
        except (TimeoutError, OSError, http.client.HTTPException, urllib.error.URLError) as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
                raise StationError(f"{what}: timed out after {self._timeout:g} s") from None
            name = type(reason or exc).__name__
            raise StationError(f"{what}: could not reach the laptop ({name})") from None
        except ValueError:
            raise StationError(f"{what}: invalid request to the laptop") from None

    def _status_error(self, status: int, raw: bytes, what: str) -> StationError:
        detail = _laptop_message(raw, self._token)
        if status == 401:
            text = "laptop rejected the token (401)"
        elif status == 503:
            text = "laptop inference service unavailable (503)"
        elif status >= 500:
            text = f"laptop inference service error ({status})"
        else:
            text = f"laptop answered {status}"
        return StationError(f"{what}: {text}" + (f": {detail}" if detail else ""))

    def _request(self, req: urllib.request.Request, what: str) -> bytes:
        status, raw = self._send(req, what)
        if not 200 <= status < 300:
            raise self._status_error(status, raw, what)
        return raw

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


    def check_approved(self) -> str:
        what = "laptop approval check"
        url = self._base + MODEL_PREFIX + quote(self.part_number, safe="")
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        status, raw = self._send(req, what)
        if status == 404:
            raise StationError(
                f"the laptop does not list an approved model for part {self.part_number}; "
                "ask Quality to approve one"
            )
        if not 200 <= status < 300:
            raise self._status_error(status, raw, what)
        body = self._json(raw, what)
        if body.get("approved") is not True or body.get("part_number") != self.part_number:
            raise StationError(f"{what}: reply is not an approval for part {self.part_number}")
        version = body.get("model_version")
        if not isinstance(version, str) or not version:
            raise StationError(f"{what}: reply has no model_version for part {self.part_number}")
        self.version = version
        return version

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
