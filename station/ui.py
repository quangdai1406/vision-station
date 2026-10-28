"""HTTP server for the inspection station: a JSON API plus one HTML page.

Standard library only. The page is the single _PAGE string below and loads no
external assets. `session` is duck-typed (see make_server).
"""
from __future__ import annotations

import base64
import dataclasses
import enum
import http.cookies
import json
import re
import sys
import traceback
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from station.auth import ACTIONS, SessionManager, UserStore, role_allows
from station.contracts import InvalidTransition, StationError, Verdict
from station.scan import ScanError
from station.scanstore import _component as _scan_name_check

SESSION_COOKIE = "station_session"
MAX_BODY_BYTES = 64 * 1024
MAX_REASON_CHARS = 500
PREVIEW_CAMERA_RE = re.compile(r"C[1-4]")
# Taking data (plan section 8, step 4). Input bounds only; no quality limit is set here.
MAX_SCAN_TOY_CHARS = 64  # same as the scan store's name limit
MAX_SCAN_DEFECT_CHARS = 64
MAX_SCAN_DEFECTS = 32
SCAN_ACTIVE_STATES = ("SCAN_SIDE", "WAIT_FLIP", "SAVING", "FAULT")
SCAN_SIDES = ("TOP", "BOTTOM")
# Snapshot keys the browser may see. saved_path is left out: it is a path on the station.
_SCAN_VIEW_KEYS = ("state", "side", "step", "counts", "sides", "scan_id", "part_number", "toy_id",
                   "label_verdict", "defects", "marks", "camera_names", "fault", "press_ready",
                   "flip_wait_s", "prompt")
# Pending scan reviews the browser may see: no picture bytes, no folder paths.
_SCAN_PENDING_KEYS = ("scan_id", "part_number", "toy_id", "verdict", "defects", "labelled_by",
                      "labelled_at", "steps")
_SCAN_IMAGE_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
PART_SEARCH_MAX_CHARS = 100
PART_SEARCH_LIMIT_DEFAULT = 20
PART_SEARCH_LIMIT_MAX = 50
MAX_PART_NUMBER_CHARS = 100
MAX_CRITERION_CHARS = 64  # longer than any MES criterion id or class name we accept
MAX_PART_ID_CHARS = 100
# Only these keys leave the server, so no cost or planning data reaches the browser.
_PART_SEARCH_KEYS = ("part_number", "description", "part_type", "lifecycle")
_PART_KEYS = ("part_number", "model_version", "can_inspect", "reason", "parts_mode", "warnings")
_LABEL_KEYS = ("session_id", "verdict", "criterion_id", "status", "changed")
_LABEL_STATUS_KEYS = ("session_id", "part_id", "part_number", "label_status", "verdict",
                      "criterion_id", "reject_reason")
_PENDING_LABEL_KEYS = ("session_id", "part_id", "part_number", "verdict", "criterion_id",
                       "proposed_by", "proposed_at")
MAX_REJECT_REASON_CHARS = 200  # keep in step with station.storage.MAX_REJECT_REASON_CHARS

_ACTION_PATHS = {
    "/api/press": "press",
    "/api/ack": "ack",
    "/api/resolve": "resolve",
    "/api/override": "override",
    "/api/clear-fault": "clear_fault",
    "/api/label": "label_propose",
}

# Taking data POST routes: the part after /api/scan/ -> _Handler method name.
_SCAN_POST = {
    "start": "_scan_start",
    "press": "_scan_press",
    "done-side": "_scan_done_side",
    "mark": "_scan_mark",
    "finish": "_scan_finish",
    "cancel": "_scan_cancel",
    "clear-fault": "_scan_clear_fault",
    "reset": "_scan_reset",
    "review": "_scan_review",
}

_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src data: blob:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


class _HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _to_jsonable(value: Any) -> Any:
    """Make snapshot values JSON-safe: enums -> value, bytes -> base64, dataclasses -> dict."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (bytes, bytearray)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable({f.name: getattr(value, f.name) for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _cookie_header(token: str) -> str:
    return f"{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict"


def _expired_cookie_header() -> str:
    return f"{SESSION_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict"


def _parse_verdict(data: dict) -> Verdict:
    raw = data.get("verdict")
    if not isinstance(raw, str) or raw not in ("GOOD", "BAD"):
        raise _HttpError(400, "verdict must be GOOD or BAD")
    return Verdict(raw)


def _parse_reason(data: dict) -> str:
    raw = data.get("reason")
    if not isinstance(raw, str) or not raw.strip():
        raise _HttpError(400, "reason is required")
    reason = raw.strip()
    if len(reason) > MAX_REASON_CHARS:
        raise _HttpError(400, "reason is too long")
    return reason


def _parse_criterion(data: dict) -> Optional[str]:
    raw = data.get("criterion_id")
    if raw is None:
        return None
    if not isinstance(raw, str) or len(raw) > MAX_CRITERION_CHARS:
        raise _HttpError(400, f"criterion_id must be a string up to {MAX_CRITERION_CHARS} characters")
    return raw


def _parse_reject_reason(data: dict) -> str:
    raw = data.get("reason")
    if not isinstance(raw, str) or not raw.strip():
        raise _HttpError(400, "reason is required")
    reason = raw.strip()
    if len(reason) > MAX_REJECT_REASON_CHARS:
        raise _HttpError(400, f"reason must be up to {MAX_REJECT_REASON_CHARS} characters")
    return reason


def _parse_part_id(data: dict) -> Optional[str]:
    raw = data.get("part_id")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_PART_ID_CHARS:
        raise _HttpError(400, f"part_id must be a string up to {MAX_PART_ID_CHARS} characters")
    return raw.strip()


def _scan_toy_id(raw: Any) -> str:
    """toy_id: 1 to 64 printable characters, usable as one path component (store rule)."""
    if not isinstance(raw, str):
        raise _HttpError(400, "toy_id is required")
    text = raw.strip()
    if not 1 <= len(text) <= MAX_SCAN_TOY_CHARS:
        raise _HttpError(400, f"toy_id must be 1 to {MAX_SCAN_TOY_CHARS} characters")
    if not text.isprintable():
        raise _HttpError(400, "toy_id must not contain control characters")
    try:
        _scan_name_check(text, "toy_id")
    except ValueError:
        raise _HttpError(400, "toy_id must not contain / or \\ and must not start with a dot") from None
    return text


def _scan_part_number(raw: Any) -> Optional[str]:
    """part_number, or None when empty (the station's current part is used then)."""
    if raw is None:
        return None
    if not isinstance(raw, str) or len(raw) > MAX_PART_NUMBER_CHARS:
        raise _HttpError(400, f"part_number must be a string up to {MAX_PART_NUMBER_CHARS} characters")
    text = raw.strip()
    if not text:
        return None
    if not text.isprintable():
        raise _HttpError(400, "part_number must not contain control characters")
    return text


def _parse_scan_defects(raw: Any) -> list:
    """Defect names as the operator typed them (no list of allowed names is enforced here)."""
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_SCAN_DEFECTS:
        raise _HttpError(400, f"defects must be a list of up to {MAX_SCAN_DEFECTS} names")
    out = []
    for item in raw:
        if not isinstance(item, str):
            raise _HttpError(400, "defect names must be text")
        name = item.strip()
        if not 1 <= len(name) <= MAX_SCAN_DEFECT_CHARS:
            raise _HttpError(400, f"each defect name must be 1 to {MAX_SCAN_DEFECT_CHARS} characters")
        out.append(name)
    return out


def _parse_scan_sides(raw: Any) -> list:
    if not isinstance(raw, list) or not 1 <= len(raw) <= len(SCAN_SIDES) or any(
            s not in SCAN_SIDES for s in raw):
        raise _HttpError(400, "sides must be a list of TOP and/or BOTTOM")
    return list(raw)


def _scan_view(snap: Any) -> dict:
    """The part of a scan snapshot the browser may see."""
    if not isinstance(snap, dict):
        raise RuntimeError("scan snapshot must be a dict")
    return {key: snap[key] for key in _SCAN_VIEW_KEYS if key in snap}


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pick(row: dict, keys: tuple) -> dict:
    """Copy only the listed keys that the row has."""
    return {key: row[key] for key in keys if key in row}


def _call_parts(fn, *args, **kwargs):
    """Call a parts method on the station. A PartsError (matched by class name, so this
    module never imports the MES client) becomes 502 with a fixed message: the exception
    text can name hosts or credentials, so it is not sent to the browser."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        if type(exc).__name__ == "PartsError":
            print(f"parts service error: {type(exc).__name__}", file=sys.stderr)
            raise _HttpError(502, "parts service unavailable") from None
        raise


class _Handler(BaseHTTPRequestHandler):
    # Filled in per server by make_server().
    station: Any = None
    users: UserStore | None = None
    sessions: SessionManager | None = None
    # Taking data: None when disabled. Duck-typed like the session (see make_server).
    scan: Any = None

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """No access log: nothing from a request (credentials included) is printed."""

    def do_GET(self) -> None:
        self._run(self._get)

    def do_POST(self) -> None:
        self._run(self._post)

    def _run(self, handler) -> None:
        try:
            handler()
        except _HttpError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except InvalidTransition as exc:
            self._send_json(409, {"error": str(exc) or "action not allowed in the current state"})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc) or "bad request"})
        except Exception:
            traceback.print_exc(file=sys.stderr)
            self._send_json(500, {"error": "internal error"})

    # ----- routing -------------------------------------------------------

    def _get(self) -> None:
        path, _, query = self.path.partition("?")
        if path == "/":
            return self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8",
                              extra=[("Content-Security-Policy", _CSP)])
        if path == "/api/state":
            return self._state(urllib.parse.parse_qs(query))
        if path == "/api/preview":
            return self._preview(urllib.parse.parse_qs(query))
        if path == "/api/parts/search":
            return self._parts_search(urllib.parse.parse_qs(query))
        if path == "/api/part":
            return self._part_get()
        if path == "/api/label/last":
            return self._label_last()
        if path == "/api/label/pending":
            return self._label_pending()
        if path == "/api/scan/state":
            return self._scan_state()
        if path == "/api/scan/step-picture":
            return self._scan_step_picture(urllib.parse.parse_qs(query))
        if path == "/api/scan/pending":
            return self._scan_pending()
        raise _HttpError(404, "not found")

    def _post(self) -> None:
        path = self.path.partition("?")[0]
        if path == "/api/login":
            return self._login()
        if path == "/api/logout":
            return self._logout()
        if path == "/api/part/select":
            return self._part_select()
        if path in ("/api/label/approve", "/api/label/reject"):
            return self._label_review(path.rsplit("/", 1)[1])
        if path.startswith("/api/scan/"):
            handler = _SCAN_POST.get(path[len("/api/scan/"):])
            if handler is None:
                raise _HttpError(404, "not found")
            return getattr(self, handler)()
        action = _ACTION_PATHS.get(path)
        if action is None:
            raise _HttpError(404, "not found")
        return self._act(action)

    # ----- endpoints -----------------------------------------------------

    def _state(self, query: dict) -> None:
        _, username, role = self._require()
        include_images = query.get("images", ["0"])[-1] in ("1", "true", "yes")
        self.station.tick()
        snap = self.station.snapshot(include_images=include_images)
        if not isinstance(snap, dict):
            raise RuntimeError("snapshot() must return a dict")
        body = dict(snap)
        # "scan" is listed only when Taking data is enabled on this station and the role may use it.
        body["auth"] = {
            "username": username,
            "role": role,
            "actions": [a for a in ACTIONS if role_allows(role, a)
                        and (a != "scan" or self.scan is not None)],
        }
        # Everyone may see that a scan holds the cameras (no scan details, no names).
        body["scan_active"] = self._scan_is_active()
        self._send_json(200, body)

    def _preview(self, query: dict) -> None:
        """Live view frame: read-only. Never calls tick() or any action."""
        self._require()  # any logged-in role may view (no action check)
        camera = query.get("camera", [""])[-1]
        if not PREVIEW_CAMERA_RE.fullmatch(camera):
            raise _HttpError(400, "camera must be C1, C2, C3 or C4")
        try:
            jpeg = self.station.preview(camera)
        except Exception:
            jpeg = None  # a preview error must never break the page
        if not isinstance(jpeg, (bytes, bytearray)) or not jpeg:
            return self._send_empty(204)
        return self._send(200, bytes(jpeg), "image/jpeg")

    def _login(self) -> None:
        data = self._read_json()
        username = data.get("username")
        password = data.get("password")
        if not (isinstance(username, str) and isinstance(password, str) and username and password):
            raise _HttpError(400, "username and password are required")
        role = self.users.authenticate(username, password)
        if role is None:
            raise _HttpError(401, "invalid username or password")
        old = self._cookie_token()
        if old:
            self.sessions.destroy(old)
        token = self.sessions.create(username, role)
        self._send_json(200, {"role": role}, extra=[("Set-Cookie", _cookie_header(token))])

    def _logout(self) -> None:
        token = self._cookie_token()
        if token:
            self.sessions.destroy(token)
        self._send_json(200, {"ok": True}, extra=[("Set-Cookie", _expired_cookie_header())])

    def _parts_search(self, query: dict) -> None:
        """Part-number search: any logged-in role. Returns only the allowed row keys."""
        self._require()
        q = query.get("q", [""])[-1].strip()
        if not 1 <= len(q) <= PART_SEARCH_MAX_CHARS:
            raise _HttpError(400, "q is required (1 to 100 characters)")
        raw_limit = query.get("limit", [""])[-1].strip()
        try:
            limit = int(raw_limit)
        except ValueError:
            limit = PART_SEARCH_LIMIT_DEFAULT
        limit = max(1, min(PART_SEARCH_LIMIT_MAX, limit))
        rows = _call_parts(self.station.search_parts, q, limit=limit)
        if not isinstance(rows, list):
            raise RuntimeError("search_parts() must return a list")
        results = [_pick(row, _PART_SEARCH_KEYS) for row in rows[:limit] if isinstance(row, dict)]
        self._send_json(200, {"results": results})

    def _part_get(self) -> None:
        self._require()
        current = _call_parts(self.station.current_part)
        if not isinstance(current, dict):
            raise RuntimeError("current_part() must return a dict")
        self._send_json(200, _pick(current, _PART_KEYS))

    def _part_select(self) -> None:
        """Select a part. `by` is always the logged-in username, never the body."""
        _, username, _role = self._require("select_part")
        data = self._read_json()
        raw = data.get("part_number")
        if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_PART_NUMBER_CHARS:
            raise _HttpError(400, "part_number is required (up to 100 characters)")
        result = _call_parts(self.station.select_part, part_number=raw.strip(), by=username)
        if not isinstance(result, dict):
            raise RuntimeError("select_part() must return a dict")
        self._send_json(200, _pick(result, _PART_KEYS))

    def _act(self, action: str) -> None:
        _, username, _role = self._require(action)
        data = self._read_json()
        station = self.station
        if action == "press":
            accepted = station.press()
            return self._send_json(200, {"accepted": bool(accepted)})
        if action == "ack":
            station.ack_alarm()
        elif action == "clear_fault":
            station.clear_fault()
        elif action == "resolve":
            verdict = _parse_verdict(data)
            reason = _parse_reason(data)
            station.resolve_review(by=username, new_verdict=verdict, reason=reason)
        elif action == "override":
            verdict = _parse_verdict(data)
            reason = _parse_reason(data)
            station.override(by=username, new_verdict=verdict, reason=reason)
        elif action == "label_propose":
            verdict = _parse_verdict(data)
            criterion = _parse_criterion(data)
            part_id = _parse_part_id(data)
            result = station.label_part(by=username, verdict=verdict, criterion_id=criterion,
                                        part_id=part_id)
            return self._send_json(200, _pick(result, _LABEL_KEYS))
        self._send_json(200, {"ok": True})

    def _label_review(self, decision: str) -> None:
        """Admin only. approve: make the pending proposal official, or set an official label when
        'verdict' is sent. reject: reject the pending proposal with a reason."""
        _, username, _role = self._require("label_approve")
        data = self._read_json()
        part_id = _parse_part_id(data)
        station = self.station
        if decision == "reject":
            result = station.reject_label(by=username, reason=_parse_reject_reason(data),
                                          part_id=part_id)
        elif "verdict" in data:
            result = station.set_official_label(by=username, verdict=_parse_verdict(data),
                                                criterion_id=_parse_criterion(data), part_id=part_id)
        else:
            result = station.approve_label(by=username, part_id=part_id)
        self._send_json(200, _pick(result, _LABEL_KEYS))

    def _label_last(self) -> None:
        self._require()
        row = _call_parts(self.station.last_label_status)
        self._send_json(200, {"last": _pick(row, _LABEL_STATUS_KEYS) if isinstance(row, dict) else None})

    def _label_pending(self) -> None:
        self._require("label_approve")
        rows = _call_parts(self.station.pending_labels)
        if not isinstance(rows, list):
            raise RuntimeError("pending_labels() must return a list")
        self._send_json(200, {"pending": [_pick(r, _PENDING_LABEL_KEYS) for r in rows
                                          if isinstance(r, dict)]})

    # ----- Taking data -----------------------------------------------------
    # Every route: login (401), then the 'scan' role (403), then the feature (404 when
    # scan is disabled). Scan refusals (busy, wrong state, bad input) are 400, or 409 for
    # an InvalidTransition. Pictures are only ever sent by /api/scan/step-picture.

    def _scan_gate(self) -> tuple[str, str]:
        _, username, role = self._require("scan")
        if self.scan is None:
            raise _HttpError(404, "taking data is not enabled on this station")
        return username, role

    def _scan_is_active(self) -> bool:
        scan = self.scan
        if scan is None:
            return False
        return scan.snapshot()["state"] in SCAN_ACTIVE_STATES

    def _scan_call(self, fn, *args, **kwargs):
        """Run one scan action. Refusals become 400 with the station's own message."""
        try:
            return fn(*args, **kwargs)
        except InvalidTransition:
            raise
        except (ScanError, StationError, ValueError) as exc:
            raise _HttpError(400, str(exc) or "scan action refused") from None

    def _scan_reply(self, extra: Optional[dict] = None) -> None:
        body = {"scan": _scan_view(self._scan_call(self.scan.snapshot))}
        if extra:
            body.update(extra)
        self._send_json(200, body)

    def _scan_state(self) -> None:
        self._scan_gate()
        self._scan_reply({"may_scan": True})

    def _scan_start(self) -> None:
        username, role = self._scan_gate()
        data = self._read_json()
        toy_id = _scan_toy_id(data.get("toy_id"))
        part_number = _scan_part_number(data.get("part_number"))
        if part_number is None:
            part_number = self._default_part_number()
        label_in = data.get("label")
        if not isinstance(label_in, dict):
            raise _HttpError(400, "label must be an object with verdict and defects")
        verdict = _parse_verdict(label_in).value
        defects = _parse_scan_defects(label_in.get("defects"))
        # Only the verdict and defects come from the client. The label status and the
        # labeller are set here, from the login: a supervisor's label is proposed, an admin's official.
        label = {
            "verdict": verdict,
            "defects": defects,
            "status": "official" if role == "admin" else "proposed",
            "labelled_by": username,
            "labelled_at": _utc_now_text(),
        }
        kwargs: dict[str, Any] = {}
        if data.get("sides") is not None:
            kwargs["sides"] = _parse_scan_sides(data.get("sides"))
        scan_id = self._scan_call(self.scan.start, part_number, toy_id, label, username, **kwargs)
        self._scan_reply({"scan_id": scan_id})

    def _default_part_number(self) -> str:
        """The station's current part, used when the form leaves part_number empty."""
        try:
            current = _call_parts(self.station.current_part)
        except StationError:
            current = None
        pn = current.get("part_number") if isinstance(current, dict) else None
        if not isinstance(pn, str) or not pn.strip():
            raise _HttpError(400, "part_number is required: the station has no current part")
        return pn.strip()

    def _scan_press(self) -> None:
        self._scan_gate()
        self._read_json()
        accepted = self._scan_call(self.scan.press)
        self._scan_reply({"accepted": bool(accepted)})

    def _scan_done_side(self) -> None:
        self._scan_gate()
        self._read_json()
        self._scan_call(self.scan.done_side)
        self._scan_reply()

    def _scan_mark(self) -> None:
        self._scan_gate()
        data = self._read_json()
        camera = data.get("camera_id")
        if not isinstance(camera, str) or not PREVIEW_CAMERA_RE.fullmatch(camera):
            raise _HttpError(400, "camera_id must be C1, C2, C3 or C4")
        side = data.get("side")
        if side not in SCAN_SIDES:
            raise _HttpError(400, "side must be TOP or BOTTOM")
        step = data.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise _HttpError(400, "step must be a whole number of at least 1")
        visible = data.get("visible")
        if not isinstance(visible, bool):
            raise _HttpError(400, "visible must be true or false")
        self._scan_call(self.scan.mark, camera, side, step, visible)
        self._scan_reply()

    def _scan_finish(self) -> None:
        self._scan_gate()
        self._read_json()
        self._scan_call(self.scan.finish)  # the folder path stays on the station
        self._scan_reply({"saved": True})

    def _scan_cancel(self) -> None:
        self._scan_gate()
        self._read_json()
        self._scan_call(self.scan.cancel)
        self._scan_reply()

    def _scan_clear_fault(self) -> None:
        self._scan_gate()
        self._read_json()
        self._scan_call(self.scan.clear_fault)
        self._scan_reply()

    def _scan_reset(self) -> None:
        self._scan_gate()
        self._read_json()
        self._scan_call(self.scan.reset)
        self._scan_reply()

    # Approval of finished scans (admin): gate is the 'label_approve' action, then the feature.
    def _scan_review_gate(self) -> str:
        _, username, _role = self._require("label_approve")
        if self.scan is None:
            raise _HttpError(404, "taking data is not enabled on this station")
        return username

    def _scan_pending_rows(self) -> list:
        rows = self._scan_call(self.scan.pending_reviews)
        if not isinstance(rows, list):
            raise RuntimeError("pending_reviews() must return a list")
        return [_pick(r, _SCAN_PENDING_KEYS) for r in rows if isinstance(r, dict)]

    def _scan_pending(self) -> None:
        self._scan_review_gate()
        self._send_json(200, {"pending": self._scan_pending_rows()})

    def _scan_review(self) -> None:
        """Approve or reject a proposed scan label. The reviewer is the login, never the body."""
        username = self._scan_review_gate()
        data = self._read_json()
        scan_id = data.get("scan_id")
        if not isinstance(scan_id, str) or not scan_id:
            raise _HttpError(400, "scan_id is required")
        decision = data.get("decision")
        if decision not in ("approve", "reject"):
            raise _HttpError(400, "decision must be approve or reject")
        reason = data.get("reason")
        if decision == "reject":
            if not isinstance(reason, str) or not reason.strip():
                raise _HttpError(400, "a rejection needs a reason")
            if len(reason) > MAX_REJECT_REASON_CHARS:
                raise _HttpError(400, f"reason must be {MAX_REJECT_REASON_CHARS} characters or fewer")
            reason = reason.strip()
        elif reason not in (None, ""):
            raise _HttpError(400, "reason is only for a rejection")
        else:
            reason = None
        label = self._scan_call(self.scan.review, scan_id, decision, username, reason)
        self._send_json(200, {"label": _to_jsonable(label), "pending": self._scan_pending_rows()})

    def _scan_step_picture(self, query: dict) -> None:
        """The picture of the camera for the last saved step. 204 when there is none."""
        self._scan_gate()
        camera = query.get("camera", [""])[-1]
        if not PREVIEW_CAMERA_RE.fullmatch(camera):
            raise _HttpError(400, "camera must be C1, C2, C3 or C4")
        got = self._scan_call(self.scan.step_picture, camera)
        if not got:
            return self._send_empty(204)
        data, ext = got
        if not isinstance(data, (bytes, bytearray)) or not data:
            return self._send_empty(204)
        ctype = _SCAN_IMAGE_TYPES.get(str(ext).lower(), "application/octet-stream")
        return self._send(200, bytes(data), ctype)

    # ----- helpers -------------------------------------------------------

    def _cookie_token(self):
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        jar = http.cookies.SimpleCookie()
        try:
            jar.load(raw)
        except http.cookies.CookieError:
            return None
        morsel = jar.get(SESSION_COOKIE)
        return morsel.value if morsel is not None else None

    def _require(self, action: str | None = None):
        """Return (token, username, role) or raise 401 / 403."""
        token = self._cookie_token()
        got = self.sessions.get(token) if token else None
        if got is None:
            raise _HttpError(401, "login required")
        username, role = got
        if action is not None and not role_allows(role, action):
            raise _HttpError(403, "your role may not perform this action")
        return token, username, role

    def _read_json(self) -> dict:
        if self.headers.get("Transfer-Encoding"):
            raise _HttpError(411, "Content-Length is required")
        raw_len = (self.headers.get("Content-Length") or "0").strip()
        if not raw_len.isdigit():
            raise _HttpError(400, "bad Content-Length")
        length = int(raw_len)
        if length > MAX_BODY_BYTES:
            raise _HttpError(413, "body too large")
        if length == 0:
            return {}
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            raise _HttpError(415, "JSON body required")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise _HttpError(400, "invalid JSON")
        if not isinstance(data, dict):
            raise _HttpError(400, "JSON object required")
        return data

    def _send_json(self, status: int, obj: Any, extra=()) -> None:
        body = json.dumps(_to_jsonable(obj)).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", extra)

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

    def _send(self, status: int, body: bytes, ctype: str, extra=()) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        for name, value in extra:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


def make_server(session, users: UserStore, sessions: SessionManager,
                host: str = "127.0.0.1", port: int = 0, scan=None) -> ThreadingHTTPServer:
    """Build (but do not start) the UI server.

    `session` must provide: snapshot(include_images=False) -> dict, press() -> bool,
    resolve_review(by, new_verdict, reason), override(by, new_verdict, reason),
    ack_alarm(), clear_fault(), tick(), preview(camera_id) -> bytes | None.
    snapshot() may also carry prompt (str), press_ready (bool) and flip_wait_s (float),
    and in part mode parts_mode, camera ({open, last_error}), part_number, part_selected,
    model_version, can_inspect and block_reason. Part picker (used when parts_mode is 'mes'):
    search_parts(q, limit) -> list[dict], select_part(part_number, by) -> dict,
    current_part() -> dict. A PartsError (matched by class name) maps to 502.
    InvalidTransition -> 409, ValueError -> 400.

    `scan` (optional, Taking data): None when disabled, else a ScanSession or a wrapper
    with start, press, done_side, mark, finish, cancel, clear_fault, reset,
    step_picture(camera_id) -> (bytes, ext) | None and snapshot() -> dict. Disabled means
    the scan routes answer 404 (after the login and role checks) and the 'scan' action is
    not listed for anyone.
    """
    handler = type("StationHandler", (_Handler,), {
        "station": session,
        "users": users,
        "sessions": sessions,
        "scan": scan,
    })
    return ThreadingHTTPServer((host, port), handler)


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vision Station</title>
<style>
:root {
  --bg: #f3f4f6; --fg: #1b1d21; --card: #ffffff; --muted: #5f6570; --line: #d5d9df;
  --good: #137a3e; --bad: #b42318; --review: #d89b00; --none: #8b929c;
  --btn: #1f2937; --btn-fg: #ffffff;
  --poc: #f5a524;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #121417; --fg: #e6e8eb; --card: #1c1f24; --muted: #9aa1ab; --line: #333840;
    --good: #2fbf6b; --bad: #ff5a4f; --review: #f5b041; --none: #5b626d;
    --btn: #e6e8eb; --btn-fg: #121417;
    --poc: #f5b041;
  }
}
:root[data-theme="dark"] {
  --bg: #121417; --fg: #e6e8eb; --card: #1c1f24; --muted: #9aa1ab; --line: #333840;
  --good: #2fbf6b; --bad: #ff5a4f; --review: #f5b041; --none: #5b626d;
  --btn: #e6e8eb; --btn-fg: #121417;
  --poc: #f5b041;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin: 0; background: var(--bg); color: var(--fg); background-color: var(--bg);
  font: 16px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
  gap: 8px; padding: 12px 16px; background: var(--card); border-bottom: 1px solid var(--line); }
h1 { font-size: 18px; margin: 0; }
h2 { font-size: 18px; margin: 0 0 12px; }
main { padding: 16px; max-width: 980px; margin: 0 auto; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px; margin-bottom: 16px; }
.muted { color: var(--muted); font-size: 14px; }
.verdict { font-size: 56px; font-weight: 800; text-align: center; letter-spacing: 2px;
  padding: 24px 12px; border-radius: 12px; color: #ffffff; margin-top: 8px; }
.v-GOOD { background: var(--good); }
.v-BAD { background: var(--bad); }
.v-REVIEW { background: var(--review); color: #1b1d21; }
.v-NONE { background: var(--none); }
.v-COLLECTED { background: var(--none); }
.counters { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; text-align: center; }
.counters span { display: block; font-size: 14px; color: var(--muted); }
.counters b { display: block; font-size: 28px; }
.tiles { display: grid; grid-template-columns: repeat(var(--cols-m, 2), 1fr); gap: 8px; }
@media (min-width: 700px) { .tiles { grid-template-columns: repeat(var(--cols, 4), 1fr); } }
.poc { background: var(--poc); color: #1b1d21; font-weight: 700; border-radius: 10px;
  padding: 12px 16px; margin-bottom: 16px; }
.hint { margin: 8px 0 0; }
.prompt { font-size: 28px; font-weight: 700; line-height: 1.25; margin: 0 0 12px; }
.status { min-height: 1.4em; font-weight: 600; margin: 8px 0 0; }
.live-tile { margin: 0; min-width: 0; }
.live-tile img { width: 100%; height: auto; min-height: 120px; display: block; background: #000;
  border-radius: 6px; }
.live-tile figcaption { font-size: 13px; color: var(--muted); margin-top: 4px; }
.side-row + .side-row { margin-top: 16px; }
.side-row h3 { font-size: 14px; color: var(--muted); letter-spacing: 1px; margin: 0 0 8px; }
.tile { margin: 0; min-width: 0; }
.tile canvas { width: 100%; height: auto; display: block; background: #000; border-radius: 6px; }
.tile figcaption { font-size: 13px; color: var(--muted); margin-top: 4px; }
button { font: inherit; background: var(--btn); color: var(--btn-fg); border: 0; border-radius: 8px;
  padding: 12px 16px; min-height: 44px; margin: 4px 8px 4px 0; cursor: pointer; }
button:disabled { opacity: 0.4; cursor: not-allowed; }
button.secondary { background: transparent; color: var(--fg); border: 1px solid var(--line); }
input { font: inherit; width: 100%; padding: 10px; margin: 4px 0 12px; border-radius: 8px;
  border: 1px solid var(--line); background: var(--card); color: var(--fg); }
label { display: block; font-size: 14px; color: var(--muted); }
.error { color: var(--bad); font-weight: 600; min-height: 1.4em; }
.login { max-width: 360px; margin: 40px auto; }
.part-line { font-weight: 600; margin: 0 0 8px; }
.part-block { color: var(--bad); font-weight: 600; margin: 0 0 8px; }
.part-note { font-size: 13px; margin: 8px 0 0; min-height: 1.2em; }
.part-results { list-style: none; padding: 0; margin: 8px 0 0; }
.part-results li + li { margin-top: 6px; }
.part-choice { display: block; width: 100%; margin: 0; text-align: left; }
.part-pn { font-weight: 700; margin-right: 8px; }
.part-desc { display: block; font-size: 14px; color: var(--muted); font-weight: 400; }
.cam-status { color: var(--bad); font-weight: 600; margin: 0 0 8px; }
.scan-busy { font-weight: 600; border-color: var(--review); }
.scan-fields { border: 0; padding: 0; margin: 4px 0 12px; }
.scan-fields legend { font-size: 14px; color: var(--muted); padding: 0; }
label.inline { display: inline-block; color: var(--fg); font-size: 16px; margin: 0 16px 8px 0; }
input[type="radio"], input[type="checkbox"] { width: auto; min-height: 0; margin: 0 6px 0 0; }
.scan-buttons button:disabled { opacity: 0.4; }
.scan-pic { margin: 0; min-width: 0; }
.scan-pic img { width: 100%; height: auto; min-height: 120px; display: block; background: #000;
  border-radius: 6px; }
.scan-pic figcaption { font-size: 13px; color: var(--muted); margin: 4px 0; }
</style>
</head>
<body>
<header>
  <h1>Vision Station</h1>
  <div>
    <span id="who" class="muted"></span>
    <button id="scan-open" class="secondary" hidden>Taking data</button>
    <button id="logout" class="secondary" hidden>Log out</button>
  </div>
</header>
<main>
  <section id="login-box" class="card login" hidden>
    <h2>Log in</h2>
    <form id="login-form">
      <label for="username">Username</label>
      <input id="username" name="username" autocomplete="username" required>
      <label for="password">Password</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required>
      <button type="submit">Log in</button>
    </form>
  </section>

  <section id="app" hidden>
    <div id="poc" class="poc" role="status" hidden></div>
    <div class="card">
      <div id="state" class="muted">State: -</div>
      <div id="verdict" class="verdict v-NONE">-</div>
    </div>
    <div class="card counters">
      <div><span>Total</span><b id="c-total">0</b></div>
      <div><span>Good</span><b id="c-good">0</b></div>
      <div><span>Bad</span><b id="c-bad">0</b></div>
    </div>
    <section id="live" class="card" aria-label="Live view">
      <h3 class="live-title">Live view</h3>
      <p id="cam-status" class="cam-status" role="status" hidden></p>
      <div id="live-grid" class="tiles"></div>
      <p class="muted hint">Live view (not saved)</p>
    </section>
    <div id="tiles" class="card"></div>
    <p id="scan-busy" class="card scan-busy" role="status" hidden></p>
    <section id="scan-panel" class="card" aria-label="Taking data" hidden>
      <h2>Taking data</h2>
      <div id="scan-setup">
        <form id="scan-start-form" autocomplete="off">
          <label for="scan-toy">Toy ID</label>
          <input id="scan-toy" maxlength="64" autocomplete="off">
          <label for="scan-part">Part number (empty: the station's current part)</label>
          <input id="scan-part" maxlength="100" autocomplete="off">
          <fieldset class="scan-fields">
            <legend>Label</legend>
            <label class="inline"><input type="radio" name="scan-verdict" value="GOOD" checked> GOOD</label>
            <label class="inline"><input type="radio" name="scan-verdict" value="BAD"> BAD</label>
          </fieldset>
          <div id="scan-defects-box" hidden>
            <label for="scan-defects">Defect names for BAD (comma-separated)</label>
            <input id="scan-defects" maxlength="2000" autocomplete="off">
            <p class="muted hint">Names are kept as typed. Home trial examples: doh_small, doh_large, doh_edge.</p>
          </div>
          <button type="submit" id="scan-start">Start scan</button>
        </form>
        <p id="scan-start-reason" class="muted hint" hidden></p>
      </div>
      <div id="scan-run" hidden>
        <p id="scan-meta" class="muted"></p>
        <p id="scan-prompt" class="prompt" role="status"></p>
        <p id="scan-counts" class="muted"></p>
        <div class="scan-buttons">
          <button type="button" id="scan-press">Press (Enter)</button>
          <button type="button" id="scan-done">Done with this side</button>
          <button type="button" id="scan-fault" class="secondary" hidden>Clear fault</button>
          <button type="button" id="scan-finish">Finish</button>
          <button type="button" id="scan-cancel" class="secondary">Cancel scan</button>
          <button type="button" id="scan-reset" class="secondary" hidden>Start another scan</button>
        </div>
        <p id="scan-marks" class="muted"></p>
        <p id="scan-pics-note" class="muted hint" hidden>Defect marks apply to the step shown here. Earlier steps keep the marks already given, and cannot be changed from this screen.</p>
        <div id="scan-pics" class="tiles" hidden></div>
        <p id="scan-status" class="status" role="status"></p>
      </div>
      <section id="scan-review" aria-label="Scans waiting for approval" hidden>
        <h3>Scans waiting for approval</h3>
        <p class="muted hint">Only an admin makes a scan label official. Rejecting needs a short reason.</p>
        <p id="scan-review-none" class="muted" hidden>No scans waiting.</p>
        <ul id="scan-review-list" class="part-results"></ul>
      </section>
    </section>
    <div id="inspect-controls">
    <section id="part-panel" class="card" aria-label="Part number" hidden>
      <h2>Part number</h2>
      <p id="part-line" class="part-line" role="status"></p>
      <p id="part-block" class="part-block" hidden></p>
      <form id="part-search-form">
        <label for="part-q">Search AISI MES part numbers</label>
        <input id="part-q" maxlength="100" autocomplete="off">
        <button type="submit" id="part-search" class="secondary">Search</button>
      </form>
      <ul id="part-results" class="part-results"></ul>
      <p id="part-note" class="part-note muted"></p>
      <p id="part-warn" class="part-note muted"></p>
    </section>
    <div class="card">
      <p id="prompt" class="prompt" hidden></p>
      <button data-action="press">Press (Enter)</button>
      <button data-action="ack">Acknowledge alarm</button>
      <button data-action="clear_fault">Clear fault</button>
      <p id="press-hint" class="muted hint">Enter key or click: start TOP scan, flip the part, press again for the BOTTOM scan.</p>
      <p id="press-status" class="status" role="status"></p>
    </div>
    <div class="card">
      <label for="reason">Reason (required for resolve and override)</label>
      <input id="reason" maxlength="500" autocomplete="off">
      <button data-action="resolve" data-verdict="GOOD">Resolve REVIEW: GOOD</button>
      <button data-action="resolve" data-verdict="BAD">Resolve REVIEW: BAD</button>
      <button data-action="override" data-verdict="GOOD">Override BAD to GOOD</button>
    </div>
    <div class="card">
      <label for="criterion">Collected part: criterion for a BAD proposal (optional)</label>
      <input id="criterion" maxlength="64" autocomplete="off">
      <button data-action="label_propose" data-verdict="GOOD">Propose GOOD for collected part</button>
      <button data-action="label_propose" data-verdict="BAD">Propose BAD for collected part</button>
      <p id="label-status" class="part-note muted" role="status"></p>
      <div id="label-review" hidden>
        <h3>Pending label proposals (admin)</h3>
        <p class="muted hint">Only an admin makes a label official. Rejecting needs a short reason.</p>
        <label for="reject-reason">Reject reason (required to reject)</label>
        <input id="reject-reason" maxlength="200" autocomplete="off">
        <ul id="label-pending" class="part-results"></ul>
      </div>
    </div>
    </div>
  </section>
  <p id="error" class="error" role="alert"></p>
</main>
<script>
(function () {
  "use strict";
  var POLL_MS = 1000;
  // Engineering settings: client-side press lockout and live-view frame rate.
  var PRESS_LOCKOUT_MS = 1500;
  var PRESS_MAX_WAIT_MS = 4000;
  var PREVIEW_MS = 250;
  var PREVIEW_RETRY_MS = 1000;
  var DEFAULT_CAMERAS = ["C1", "C2", "C3", "C4"];
  var SIDES = ["TOP", "BOTTOM"];
  var PLACEHOLDER_W = 320;
  var PLACEHOLDER_H = 240;
  var PATHS = {
    press: "/api/press", ack: "/api/ack", clear_fault: "/api/clear-fault",
    resolve: "/api/resolve", override: "/api/override", label_propose: "/api/label"
  };
  var LABEL_MS = 3000;
  var MSG_CAPTURING = "Capturing... keep the part still.";
  var MSG_WAIT_CAPTURE = "Wait: the picture is being taken.";
  var MSG_FLIP_FIRST = "Press ignored: flip the part first, then press Enter.";
  var MSG_NOT_ACCEPTED = "Press was not accepted in the current state.";
  var PART_SEARCH_MIN_MS = 300;

  function $(id) { return document.getElementById(id); }

  var el = {
    login: $("login-box"), app: $("app"), form: $("login-form"), who: $("who"),
    logout: $("logout"), error: $("error"), state: $("state"), verdict: $("verdict"),
    total: $("c-total"), good: $("c-good"), bad: $("c-bad"), tiles: $("tiles"),
    reason: $("reason"), criterion: $("criterion"), poc: $("poc"), prompt: $("prompt"),
    pressStatus: $("press-status"), liveGrid: $("live-grid"),
    partPanel: $("part-panel"), partLine: $("part-line"), partBlock: $("part-block"),
    partForm: $("part-search-form"), partQ: $("part-q"), partSearch: $("part-search"),
    partResults: $("part-results"), partNote: $("part-note"), partWarn: $("part-warn"),
    camStatus: $("cam-status"),
    press: document.querySelector('[data-action="press"]')
  };

  var timer = null;
  var gen = 0;
  // part_id and state of the snapshot that the shown images belong to.
  var lastPart;
  var lastState;
  // Last state seen by a poll, and what the server said about pressing.
  var curState;
  var curFault = false;
  var pressReady = true;
  var flipWait = 0;
  var pressAllowed = false;
  // Part picker state. parts_mode 'fixed' (or no key, as in old snapshots) hides the panel.
  var partsMode = "fixed";
  var partSelected = false;
  var canInspect = true;
  var partBusy = false;
  // Request counter: a search response is shown only if no newer search or selection began.
  var partSearchGen = 0;
  var partSearchLastAt = 0;
  var partSearchTimer = null;

  function setError(msg) { el.error.textContent = msg || ""; }
  function setPressStatus(msg) { el.pressStatus.textContent = msg || ""; }

  function request(method, path, body) {
    var opts = { method: method, credentials: "same-origin", headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(path, opts).then(function (res) {
      return res.json().then(
        function (data) { return { ok: res.ok, status: res.status, data: data }; },
        function () { return { ok: res.ok, status: res.status, data: {} }; }
      );
    }).then(function (r) {
      if (!r.ok) {
        var err = new Error((r.data && r.data.error) || ("Request failed (" + r.status + ")"));
        err.status = r.status;
        throw err;
      }
      return r.data;
    });
  }

  // One tile per (side, camera). The bitmap comes only from snapshots that carry
  // b64; metadata-only entries (score, defects) refresh the tile without touching it.
  function Tile(side, cam) {
    var self = this;
    this.side = side;
    this.cam = cam;
    this.key = "";
    this.failed = false;
    this.image = null;
    this.figure = document.createElement("figure");
    this.figure.className = "tile";
    this.canvas = document.createElement("canvas");
    this.canvas.width = PLACEHOLDER_W;
    this.canvas.height = PLACEHOLDER_H;
    this.caption = document.createElement("figcaption");
    this.figure.appendChild(this.canvas);
    this.figure.appendChild(this.caption);
    this.img = new Image();
    this.img.onload = function () { self.draw(); };
    // A decode failure keeps the key, so the tile still counts as having its
    // image and the page does not re-request ?images=1 on every poll. The tile
    // retries only when a fresh full snapshot arrives (part or state changed).
    this.img.onerror = function () {
      self.failed = true;
      self.setCaption();
      self.draw();
    };
    this.caption.textContent = cam + "  no image";
  }

  Tile.prototype.hasImage = function () { return this.key !== ""; };

  Tile.prototype.setBitmap = function (key, src, retry) {
    if (key === this.key && !(retry && this.failed)) { return; }
    this.key = key;
    this.failed = false;
    if (src) {
      this.img.src = src;
    } else {
      this.img.removeAttribute("src");
    }
  };

  // `fresh` is true only for a full (images=1) snapshot; it lets a tile whose
  // decode failed try once more.
  Tile.prototype.update = function (item, fresh) {
    this.image = item || null;
    if (!item) {
      this.setBitmap("", "", false);
    } else if (typeof item.b64 === "string" && item.b64) {
      var ext = typeof item.ext === "string" ? item.ext : "png";
      if (!/^[a-z0-9]{1,8}$/.test(ext)) { ext = "png"; }
      this.setBitmap(ext + ":" + item.b64, "data:image/" + ext + ";base64," + item.b64, !!fresh);
    }
    this.setCaption();
    this.draw();
  };

  Tile.prototype.setCaption = function () {
    var label = this.cam + "  ";
    if (this.failed) {
      label += "image failed to load";
    } else {
      var item = this.image;
      label += (item && typeof item.score === "number") ? "score " + item.score.toFixed(2) : "no image";
    }
    this.caption.textContent = label;
  };

  Tile.prototype.draw = function () {
    var loaded = this.key !== "" && this.img.complete && this.img.naturalWidth > 0;
    var w = loaded ? this.img.naturalWidth : PLACEHOLDER_W;
    var h = loaded ? this.img.naturalHeight : PLACEHOLDER_H;
    if (this.canvas.width !== w) { this.canvas.width = w; }
    if (this.canvas.height !== h) { this.canvas.height = h; }
    var ctx = this.canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    if (!loaded) {
      ctx.fillStyle = "#444";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#ddd";
      ctx.font = "16px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("no image", w / 2, h / 2);
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
      return;
    }
    ctx.drawImage(this.img, 0, 0, w, h);
    // Defect boxes are in the image's own pixel space; scale them to the canvas.
    var sx = this.canvas.width / this.img.naturalWidth;
    var sy = this.canvas.height / this.img.naturalHeight;
    var defects = this.image && Array.isArray(this.image.defects) ? this.image.defects : [];
    var lw = Math.max(2, Math.round(w / 200));
    ctx.lineWidth = lw;
    ctx.strokeStyle = "#ff3b30";
    ctx.fillStyle = "#ff3b30";
    ctx.font = Math.max(12, Math.round(w / 30)) + "px sans-serif";
    defects.forEach(function (d) {
      var b = d && d.box;
      if (!Array.isArray(b) || b.length !== 4) { return; }
      var x = b[0] * sx;
      var y = b[1] * sy;
      ctx.strokeRect(x, y, b[2] * sx, b[3] * sy);
      var text = (d.defect_class || "defect") + (typeof d.score === "number" ? " " + d.score.toFixed(2) : "");
      ctx.fillText(text, x + lw, Math.max(12, y - lw));
    });
  };

  // One column per camera id of the snapshot (camera_ids). Older snapshots without
  // the key get the default C1..C4.
  var tiles = [];
  var tileKey = null;

  function cameraList(s) {
    if (!Array.isArray(s.camera_ids)) { return DEFAULT_CAMERAS.slice(); }
    var out = [];
    s.camera_ids.forEach(function (c) {
      if (typeof c === "string" && c.length > 0 && c.length <= 32) { out.push(c); }
    });
    return out;
  }

  // Rebuilds the tile grid only when the camera list changes.
  function ensureTiles(cams) {
    var key = cams.join("|");
    if (key === tileKey) { return; }
    tileKey = key;
    tiles = [];
    while (el.tiles.firstChild) { el.tiles.removeChild(el.tiles.firstChild); }
    var n = Math.max(1, cams.length);
    SIDES.forEach(function (side) {
      var row = document.createElement("div");
      var title = document.createElement("h3");
      var grid = document.createElement("div");
      row.className = "side-row";
      title.textContent = side;
      grid.className = "tiles";
      grid.style.setProperty("--cols", String(n));
      grid.style.setProperty("--cols-m", String(Math.min(2, n)));
      row.appendChild(title);
      row.appendChild(grid);
      cams.forEach(function (cam) {
        var t = new Tile(side, cam);
        tiles.push(t);
        grid.appendChild(t.figure);
      });
      el.tiles.appendChild(row);
    });
  }

  // ----- live view (not saved) -----------------------------------------
  // One <img> per camera. Each camera fetches its next frame only after the previous
  // request completes, so requests never overlap. Runs only while the page is
  // visible, the user is logged in, and the station is IDLE or WAIT_FLIP with no fault.
  var liveTiles = [];
  var liveKey = null;
  var liveRunning = false;
  var liveGen = 0;

  function LiveTile(cam) {
    this.cam = cam;
    this.timer = null;
    this.url = null;
    this.figure = document.createElement("figure");
    this.figure.className = "live-tile";
    this.img = document.createElement("img");
    this.img.alt = "Live view " + cam;
    var caption = document.createElement("figcaption");
    caption.textContent = cam;
    this.figure.appendChild(this.img);
    this.figure.appendChild(caption);
  }

  // Shows a new frame and releases the object URL it replaces.
  LiveTile.prototype.show = function (blob) {
    var url = URL.createObjectURL(blob);
    var old = this.url;
    this.url = url;
    this.img.src = url;
    if (old) { URL.revokeObjectURL(old); }
  };

  LiveTile.prototype.release = function () {
    if (this.url) { URL.revokeObjectURL(this.url); this.url = null; }
  };

  // Rebuilds the live grid only when the camera list changes.
  function ensureLive(cams) {
    var key = cams.join("|");
    if (key === liveKey) { return; }
    liveKey = key;
    liveStop();
    liveTiles.forEach(function (t) { t.release(); });
    liveTiles = [];
    while (el.liveGrid.firstChild) { el.liveGrid.removeChild(el.liveGrid.firstChild); }
    var n = Math.max(1, cams.length);
    el.liveGrid.style.setProperty("--cols", String(n));
    el.liveGrid.style.setProperty("--cols-m", String(Math.min(2, n)));
    cams.forEach(function (cam) {
      var t = new LiveTile(cam);
      liveTiles.push(t);
      el.liveGrid.appendChild(t.figure);
    });
  }

  function liveWanted() {
    if (el.app.hidden || document.visibilityState !== "visible") { return false; }
    if (curFault) { return false; }
    // While a scan runs the live view stays on, so the part can be placed and turned
    // by eye. It pauses while a step is being saved or the scan is in a fault.
    if (scanIsActive()) {
      return !scanSnap || scanSnap.state === "SCAN_SIDE" || scanSnap.state === "WAIT_FLIP";
    }
    return curState === "IDLE" || curState === "WAIT_FLIP";
  }

  // Paused views keep their last picture.
  function syncLive() {
    if (liveWanted()) { liveStart(); } else { liveStop(); }
  }

  function liveStart() {
    if (liveRunning) { return; }
    liveRunning = true;
    liveGen += 1;
    var g = liveGen;
    liveTiles.forEach(function (t) { liveSchedule(t, g, 0); });
  }

  function liveStop() {
    if (!liveRunning) { return; }
    liveRunning = false;
    liveGen += 1;
    liveTiles.forEach(function (t) {
      if (t.timer !== null) { clearTimeout(t.timer); }
      t.timer = null;
    });
  }

  function liveSchedule(t, g, ms) {
    t.timer = setTimeout(function () {
      t.timer = null;
      liveFetch(t, g);
    }, ms);
  }

  function liveFetch(t, g) {
    if (g !== liveGen) { return; }
    fetch("/api/preview?camera=" + encodeURIComponent(t.cam), {
      method: "GET", credentials: "same-origin", cache: "no-store"
    }).then(function (res) {
      if (res.status === 401) { return { loggedOut: true }; }
      if (res.status === 204) { return null; }
      if (!res.ok) { throw new Error("preview"); }
      return res.blob();
    }).then(function (result) {
      if (g !== liveGen) { return; }
      if (result && result.loggedOut === true) { showLogin(); return; }
      if (result) { t.show(result); }
      liveSchedule(t, g, PREVIEW_MS);
    }).then(null, function () {
      // Errors show nothing and retry quietly.
      if (g === liveGen) { liveSchedule(t, g, PREVIEW_RETRY_MS); }
    });
  }

  function render(s, fresh) {
    el.login.hidden = true;
    el.app.hidden = false;
    el.logout.hidden = false;

    var auth = s.auth || {};
    el.who.textContent = auth.username ? auth.username + " (" + auth.role + ")" : "";
    var allowed = {};
    (auth.actions || []).forEach(function (a) { allowed[a] = true; });
    Array.prototype.forEach.call(document.querySelectorAll("[data-action]"), function (b) {
      b.disabled = !allowed[b.getAttribute("data-action")];
    });

    // A state that differs from the one seen at press time ends the capture lockout.
    pressAllowed = !!allowed.press;
    labelIsAdmin = !!allowed.label_approve;
    scanAllowed = !!allowed.scan;
    reviewAllowed = !!allowed.label_approve;
    scanActiveNow = s.scan_active === true;
    // Part number prefill for the Taking-data form: once, and only until the operator types.
    if (!scanPartTouched && typeof s.part_number === "string" && s.part_number
        && !$("scan-part").value) {
      $("scan-part").value = s.part_number;
    }
    if (captureLocked && s.state !== lockFromState) { unlockPress(); }
    curState = s.state;
    curFault = !!s.fault;
    pressReady = s.press_ready !== false;
    flipWait = typeof s.flip_wait_s === "number" && s.flip_wait_s > 0 ? s.flip_wait_s : 0;
    el.prompt.textContent = typeof s.prompt === "string" ? s.prompt : "";
    el.prompt.hidden = el.prompt.textContent === "";
    renderPart(s);
    renderCamera(s);
    refreshPress();

    var verdict = (s.verdict === "GOOD" || s.verdict === "BAD" || s.verdict === "REVIEW"
      || s.verdict === "COLLECTED") ? s.verdict : "NONE";
    el.verdict.className = "verdict v-" + verdict;
    // A collected part is saved for evidence only: shown as COLLECTED, never as a judgement.
    el.verdict.textContent = verdict === "NONE" ? "-" : (verdict === "COLLECTED" ? "COLLECTED (not judged)" : verdict);

    // The fault text comes from the station; textContent keeps it inert.
    el.state.textContent = s.fault
      ? "FAULT: " + String(s.fault)
      : "State: " + (s.state || "unknown");

    var c = s.counters || {};
    el.total.textContent = c.total || 0;
    el.good.textContent = c.good || 0;
    el.bad.textContent = c.bad || 0;

    var cams = cameraList(s);
    ensureTiles(cams);
    ensureLive(cams);
    renderScanPanel();
    if (scanAllowed) { refreshScan(); }

    // Proof-of-concept banner: only for a partial station (not the 8-image rule).
    var partial = s.partial_station === true;
    el.poc.hidden = !partial;
    if (partial) {
      el.poc.textContent = "PROOF OF CONCEPT: " + cams.length + " camera(s). "
        + "This station does not meet the 8-image rule. "
        + "Results are not for production decisions.";
    }

    var images = Array.isArray(s.images) ? s.images : [];
    tiles.forEach(function (t) {
      t.update(findImage(images, t), fresh);
    });
    syncLive();
    maybeRefreshLabels(false);
  }

  // ----- Taking data (supervisor and admin) -----------------------------------
  // The browser sees scan_active (everyone, from /api/state) and, for a role that may
  // scan, the snapshot from /api/scan/state. A running scan pauses inspection: the
  // inspection controls step aside and the server refuses inspection presses anyway.
  // Every scan value is shown with textContent.
  var SCAN_ACTIVE = { SCAN_SIDE: true, WAIT_FLIP: true, SAVING: true, FAULT: true };
  var MSG_SCAN_FLIP_WAIT = "Press ignored: wait for the countdown, flip the part, then press Enter.";
  var MSG_SCAN_REFUSED = "Press was not accepted in the current state.";
  var scanOpen = false;
  var scanAllowed = false;
  var scanActiveNow = false;
  var scanSnap = null;
  var scanBusy = false;
  var scanFetching = false;
  var scanPicKey = "";
  var scanPicURLs = [];
  var scanPartTouched = false;

  function setScanStatus(msg) { $("scan-status").textContent = msg || ""; }

  function scanIsActive() {
    return scanActiveNow || !!(scanSnap && SCAN_ACTIVE[scanSnap.state] === true);
  }

  function scanVerdict() {
    var picked = document.querySelector('input[name="scan-verdict"]:checked');
    return picked && picked.value === "BAD" ? "BAD" : "GOOD";
  }

  function scanPressEnabled() {
    var s = scanSnap;
    if (!s || scanBusy || inPressLockout()) { return false; }
    if (s.state !== "SCAN_SIDE" && s.state !== "WAIT_FLIP" && s.state !== "FAULT") { return false; }
    return s.press_ready === true;
  }

  function scanDoneEnabled() {
    var s = scanSnap;
    if (!s || scanBusy || s.state !== "SCAN_SIDE") { return false; }
    return ((s.counts && s.counts[s.side]) || 0) >= 1;
  }

  function clearScanPics() {
    scanPicURLs.forEach(function (u) { URL.revokeObjectURL(u); });
    scanPicURLs = [];
    scanPicKey = "";
    var box = $("scan-pics");
    while (box.firstChild) { box.removeChild(box.firstChild); }
    box.hidden = true;
    $("scan-pics-note").hidden = true;
  }

  // Pictures of the last saved step, BAD scans only. The key (scan, side, step count)
  // decides when they are rebuilt, so a tick on screen survives the 1 s poll.
  function refreshScanPictures(s) {
    var side = s.side;
    var count = side && s.counts ? (s.counts[side] || 0) : 0;
    var live = s.state === "SCAN_SIDE" || s.state === "WAIT_FLIP" || s.state === "SAVING";
    if (count < 1 || !live) { clearScanPics(); return; }
    var isBad = s.label_verdict === "BAD";
    var key = String(s.scan_id) + "|" + side + "|" + count;
    if (key === scanPicKey) { return; }
    clearScanPics();
    scanPicKey = key;
    var box = $("scan-pics");
    box.hidden = false;
    $("scan-pics-note").hidden = !isBad;
    var cams = Array.isArray(s.camera_names) ? s.camera_names : [];
    cams.forEach(function (cam) {
      if (typeof cam !== "string" || cam.length === 0 || cam.length > 32) { return; }
      var base = cam + " " + side + " step " + count;
      var fig = document.createElement("figure");
      fig.className = "scan-pic";
      var img = document.createElement("img");
      img.alt = "Picture " + base;
      var cap = document.createElement("figcaption");
      cap.textContent = base + ", loading";
      fig.appendChild(img);
      fig.appendChild(cap);
      if (isBad) {
        var lab = document.createElement("label");
        lab.className = "inline";
        var tick = document.createElement("input");
        tick.type = "checkbox";
        tick.checked = false;
        lab.appendChild(tick);
        lab.appendChild(document.createTextNode("defect visible"));
        tick.addEventListener("change", function () { markScanPicture(tick, cam, side, count); });
        fig.appendChild(lab);
      }
      box.appendChild(fig);
      loadScanPicture(img, cap, base, cam, key);
    });
  }

  function loadScanPicture(img, cap, base, cam, key) {
    fetch("/api/scan/step-picture?camera=" + encodeURIComponent(cam), {
      method: "GET", credentials: "same-origin", cache: "no-store"
    }).then(function (res) {
      if (res.status === 401) { showLogin(); return null; }
      if (res.status !== 200) { return null; }
      return res.blob();
    }).then(function (blob) {
      if (key !== scanPicKey) { return; }
      if (!blob) { cap.textContent = base + ", no picture"; return; }
      var url = URL.createObjectURL(blob);
      scanPicURLs.push(url);
      img.src = url;
      cap.textContent = base;
    }).then(null, function () {
      if (key === scanPicKey) { cap.textContent = base + ", picture failed to load"; }
    });
  }

  function markScanPicture(tick, cam, side, step) {
    var want = tick.checked;
    tick.disabled = true;
    request("POST", "/api/scan/mark", { camera_id: cam, side: side, step: step, visible: want }).then(function (d) {
      tick.disabled = false;
      if (d && d.scan) { scanSnap = d.scan; renderScanPanel(); }
    }, function (err) {
      tick.disabled = false;
      tick.checked = !want;
      if (err.status === 401) { showLogin(); return; }
      setError(err.message);
    });
  }

  function renderScanRun(s) {
    var defects = Array.isArray(s.defects) ? s.defects.filter(function (d) { return typeof d === "string"; }) : [];
    var meta = "Scan " + String(s.scan_id || "-") + ", toy " + String(s.toy_id || "-")
      + ", part " + String(s.part_number || "-") + ", label " + String(s.label_verdict || "-");
    if (s.label_verdict === "BAD" && defects.length) { meta += " (" + defects.join(", ") + ")"; }
    $("scan-meta").textContent = meta;
    $("scan-prompt").textContent = typeof s.prompt === "string" ? s.prompt : "";
    var counts = s.counts || {};
    $("scan-counts").textContent = (Array.isArray(s.sides) ? s.sides : []).map(function (side) {
      return side + ": " + (counts[side] || 0) + " step(s)";
    }).join(". ");
    var pressWait = s.state === "WAIT_FLIP" && typeof s.flip_wait_s === "number" && s.flip_wait_s > 0;
    var pressBtn = $("scan-press");
    pressBtn.textContent = pressWait ? "Wait " + Math.ceil(s.flip_wait_s) + " s" : "Press (Enter)";
    pressBtn.disabled = !scanPressEnabled();
    $("scan-done").disabled = !scanDoneEnabled();
    $("scan-fault").hidden = s.state !== "FAULT";
    $("scan-fault").disabled = scanBusy;
    $("scan-finish").disabled = scanBusy || s.state !== "SAVING";
    $("scan-cancel").hidden = s.state === "DONE";
    $("scan-cancel").disabled = scanBusy;
    $("scan-reset").hidden = s.state !== "DONE";
    $("scan-reset").disabled = scanBusy;
    var marks = s.marks;
    $("scan-marks").textContent = marks && s.label_verdict === "BAD"
      ? "Defect visible so far: " + marks.visible + " of " + marks.pictures + " pictures (all steps)."
      : "";
    refreshScanPictures(s);
  }

  // Shows the panel, the inspection lock and the Start/Press controls from scanSnap.
  function renderScanPanel() {
    var s = scanSnap;
    var active = scanIsActive();
    $("inspect-controls").hidden = active;
    syncLive();
    $("scan-busy").hidden = !active;
    $("scan-busy").textContent = active
      ? "Taking data is running on this station. Inspection is paused until the scan is saved or cancelled."
      : "";
    $("scan-open").hidden = !scanAllowed;
    $("scan-open").textContent = scanOpen ? "Close Taking data" : "Taking data";
    var show = scanAllowed && (scanOpen || active || (s !== null && s.state === "DONE"));
    $("scan-panel").hidden = !show;
    $("scan-review").hidden = !reviewAllowed;
    if (!show) { return; }
    var runView = s !== null && s.state !== "IDLE";
    $("scan-setup").hidden = runView;
    $("scan-run").hidden = !runView;
    // Start is off while an inspection part is in progress (the server refuses it too).
    var inspectBusy = curState !== undefined && curState !== "IDLE";
    $("scan-defects-box").hidden = scanVerdict() !== "BAD";
    $("scan-start").disabled = scanBusy || inspectBusy || active;
    var reason = inspectBusy ? "Taking data starts only when the station is idle. Finish the part in progress first." : "";
    $("scan-start-reason").textContent = reason;
    $("scan-start-reason").hidden = reason === "";
    if (runView) {
      renderScanRun(s);
    } else {
      clearScanPics();
    }
  }

  function refreshScan() {
    if (!scanAllowed || scanFetching) { return; }
    scanFetching = true;
    var g = gen;
    request("GET", "/api/scan/state").then(function (d) {
      scanFetching = false;
      if (g !== gen) { return; }
      scanSnap = d && d.scan ? d.scan : null;
      renderScanPanel();
    }, function (err) {
      scanFetching = false;
      if (g !== gen) { return; }
      if (err.status === 401) { showLogin(); return; }
      setError(err.message);
    });
  }

  function scanAction(path, body, onDone) {
    if (scanBusy) { return; }
    scanBusy = true;
    setError("");
    renderScanPanel();
    request("POST", path, body || {}).then(function (d) {
      scanBusy = false;
      if (d && d.scan) { scanSnap = d.scan; }
      if (onDone) { onDone(d); }
      renderScanPanel();
      refreshScan();
    }, function (err) {
      scanBusy = false;
      if (err.status === 401) { showLogin(); return; }
      setError(err.message);
      renderScanPanel();
      refreshScan();
    });
  }

  function scanPress() {
    if (!scanPressEnabled()) { return; }
    var wasWait = scanSnap !== null && scanSnap.state === "WAIT_FLIP";
    lastPressAt = Date.now();
    setScanStatus(MSG_CAPTURING);
    scanAction("/api/scan/press", {}, function (d) {
      if (d && d.accepted === false) {
        setScanStatus(wasWait ? MSG_SCAN_FLIP_WAIT : MSG_SCAN_REFUSED);
      } else {
        setScanStatus("");
      }
    });
  }

  $("scan-open").addEventListener("click", function () {
    scanOpen = !scanOpen;
    renderScanPanel();
    if (scanOpen) { refreshScan(); refreshScanReview(); }
  });
  Array.prototype.forEach.call(document.querySelectorAll('input[name="scan-verdict"]'), function (r) {
    r.addEventListener("change", renderScanPanel);
  });
  $("scan-part").addEventListener("input", function () { scanPartTouched = true; });
  $("scan-start-form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    if (scanBusy) { return; }
    var toy = $("scan-toy").value.trim();
    if (!toy) { setError("Enter the toy ID first."); return; }
    var verdict = scanVerdict();
    var label = { verdict: verdict, defects: [] };
    if (verdict === "BAD") {
      label.defects = $("scan-defects").value.split(",").map(function (x) { return x.trim(); })
        .filter(function (x) { return x !== ""; });
      if (!label.defects.length) { setError("A BAD scan needs at least one defect name."); return; }
    }
    var body = { toy_id: toy, label: label };
    var part = $("scan-part").value.trim();
    if (part) { body.part_number = part; }
    scanAction("/api/scan/start", body, function () {
      $("scan-toy").value = "";
      $("scan-defects").value = "";
      setScanStatus("Scan started. Press Enter (or the Press button) for step 1.");
    });
  });
  $("scan-press").addEventListener("click", function () { scanPress(); });
  $("scan-done").addEventListener("click", function () {
    if (scanDoneEnabled()) { scanAction("/api/scan/done-side", {}, function () { setScanStatus(""); }); }
  });
  $("scan-finish").addEventListener("click", function () {
    scanAction("/api/scan/finish", {}, function () {
      setScanStatus("Scan saved. Press Start another scan when ready.");
    });
  });
  $("scan-cancel").addEventListener("click", function () {
    if (!window.confirm("Cancel this scan? Nothing will be saved.")) { return; }
    scanAction("/api/scan/cancel", {}, function () { setScanStatus("Scan cancelled. Nothing was saved."); });
  });
  $("scan-fault").addEventListener("click", function () {
    scanAction("/api/scan/clear-fault", {}, function () { setScanStatus(""); });
  });
  $("scan-reset").addEventListener("click", function () {
    scanAction("/api/scan/reset", {}, function () { clearScanPics(); setScanStatus(""); });
  });

  // ----- Taking data: scans waiting for approval (label_approve) ----------
  // One row per proposed scan. Approve sends at once; Reject opens an inline reason
  // box (no browser dialog). Every value is set with textContent. Errors go to the
  // page error line, which stays visible in both panel views.
  var reviewAllowed = false;
  var reviewBusy = false;
  var reviewFetching = false;
  var REVIEW_REASON_MAX = 200;

  function reviewScanText(row) {
    var defects = Array.isArray(row.defects)
      ? row.defects.filter(function (d) { return typeof d === "string"; }) : [];
    return "Toy " + String(row.toy_id || "-") + " (part " + String(row.part_number || "-") + "): "
      + (row.verdict === "BAD" ? "BAD" : "GOOD")
      + (defects.length ? " [" + defects.join(", ") + "]" : "")
      + ", taken by " + String(row.labelled_by || "-")
      + " at " + String(row.labelled_at || "-") + ".";
  }

  function renderScanReview(rows) {
    var list = $("scan-review-list");
    while (list.firstChild) { list.removeChild(list.firstChild); }
    $("scan-review-none").hidden = rows.length > 0;
    rows.forEach(function (row) {
      var li = document.createElement("li");
      li.appendChild(document.createTextNode(reviewScanText(row) + " "));
      var approve = document.createElement("button");
      approve.type = "button";
      approve.textContent = "Approve";
      approve.setAttribute("data-action", "label_approve");
      approve.addEventListener("click", function () { reviewDecide(row.scan_id, "approve", ""); });
      var reject = document.createElement("button");
      reject.type = "button";
      reject.className = "secondary";
      reject.textContent = "Reject";
      reject.setAttribute("data-action", "label_approve");
      var box = document.createElement("div");
      box.hidden = true;
      var reasonInput = document.createElement("input");
      reasonInput.type = "text";
      reasonInput.maxLength = REVIEW_REASON_MAX;
      reasonInput.autocomplete = "off";
      reasonInput.placeholder = "Reason for rejection (required)";
      reasonInput.setAttribute("aria-label", "Reject reason");
      var confirmBtn = document.createElement("button");
      confirmBtn.type = "button";
      confirmBtn.textContent = "Confirm reject";
      confirmBtn.setAttribute("data-action", "label_approve");
      confirmBtn.addEventListener("click", function () {
        reviewDecide(row.scan_id, "reject", reasonInput.value);
      });
      reject.addEventListener("click", function () {
        box.hidden = !box.hidden;
        if (!box.hidden) { reasonInput.focus(); }
      });
      box.appendChild(reasonInput);
      box.appendChild(confirmBtn);
      li.appendChild(approve);
      li.appendChild(reject);
      li.appendChild(box);
      list.appendChild(li);
    });
  }

  function refreshScanReview() {
    if (!reviewAllowed || reviewFetching) { return; }
    reviewFetching = true;
    var g = gen;
    request("GET", "/api/scan/pending").then(function (d) {
      reviewFetching = false;
      if (g !== gen) { return; }
      renderScanReview(Array.isArray(d.pending) ? d.pending : []);
    }, function (err) {
      reviewFetching = false;
      if (g !== gen) { return; }
      if (err.status === 401) { showLogin(); return; }
      setError(err.message);
    });
  }

  function reviewDecide(scanId, decision, reason) {
    if (reviewBusy) { return; }
    var body = { scan_id: scanId, decision: decision };
    if (decision === "reject") {
      var text = String(reason || "").trim();
      if (!text) { setError("A rejection needs a reason."); return; }
      body.reason = text;
    }
    reviewBusy = true;
    setError("");
    request("POST", "/api/scan/review", body).then(function (d) {
      reviewBusy = false;
      renderScanReview(Array.isArray(d.pending) ? d.pending : []);
    }, function (err) {
      reviewBusy = false;
      if (err.status === 401) { showLogin(); return; }
      setError(err.message);
      refreshScanReview();
    });
  }

  // ----- collected-part labels ---------------------------------------------
  // Shown as text only (textContent), so station data cannot inject markup.
  var labelIsAdmin = false;
  var labelFetchAt = 0;

  function labelText(last) {
    if (!last) { return "No collected part yet."; }
    var head = "Last collected part" + (last.part_number ? " (" + last.part_number + ")" : "") + ": ";
    var crit = last.criterion_id ? " [" + last.criterion_id + "]" : "";
    if (last.label_status === "proposed") {
      return head + "PROPOSED " + last.verdict + crit + ", waiting for admin approval.";
    }
    if (last.label_status === "official") {
      return head + "OFFICIAL " + last.verdict + crit + ".";
    }
    if (last.label_status === "rejected") {
      return head + "REJECTED" + (last.reject_reason ? ": " + last.reject_reason : "") + ".";
    }
    return head + "not labelled yet.";
  }

  function renderPending(rows) {
    var list = $("label-pending");
    while (list.firstChild) { list.removeChild(list.firstChild); }
    if (!rows.length) {
      var none = document.createElement("li");
      none.textContent = "No proposals waiting.";
      list.appendChild(none);
      return;
    }
    rows.forEach(function (row) {
      var li = document.createElement("li");
      var txt = "Part " + (row.part_number || "?") + " (" + (row.part_id || "?") + "): proposed "
        + row.verdict + (row.criterion_id ? " [" + row.criterion_id + "]" : "")
        + " by " + (row.proposed_by || "?") + " ";
      li.appendChild(document.createTextNode(txt));
      [["Approve", "approve"], ["Reject", "reject"]].forEach(function (pair) {
        var b = document.createElement("button");
        b.type = "button";
        b.textContent = pair[0];
        b.setAttribute("data-action", "label_approve");
        b.setAttribute("data-decision", pair[1]);
        if (row.part_id) { b.setAttribute("data-part-id", String(row.part_id)); }
        li.appendChild(b);
      });
      list.appendChild(li);
    });
  }

  // Polled at most every LABEL_MS, and right after a label action.
  function maybeRefreshLabels(force) {
    var now = Date.now();
    if (!force && now - labelFetchAt < LABEL_MS) { return; }
    labelFetchAt = now;
    el.labelReview = el.labelReview || $("label-review");
    el.labelReview.hidden = !labelIsAdmin;
    request("GET", "/api/label/last").then(function (d) {
      $("label-status").textContent = labelText(d.last);
    }, function () {});
    if (labelIsAdmin) {
      request("GET", "/api/label/pending").then(function (d) {
        renderPending(Array.isArray(d.pending) ? d.pending : []);
      }, function () {});
    }
  }

  function findImage(images, t) {
    var pick = null;
    images.forEach(function (im) {
      if (im && im.camera_id === t.cam && im.side === t.side) { pick = im; }
    });
    return pick;
  }

  // Images (with b64) are fetched only when the part or state changed since the
  // last image fetch, or when a listed image has no bitmap on screen yet.
  function needsImages(s) {
    if (s.part_id !== lastPart || s.state !== lastState) { return true; }
    var images = Array.isArray(s.images) ? s.images : [];
    return tiles.some(function (t) {
      return !t.hasImage() && findImage(images, t) !== null;
    });
  }

  function stopPolling() {
    if (timer !== null) { clearTimeout(timer); }
    timer = null;
  }

  function schedule(ms) {
    stopPolling();
    timer = setTimeout(poll, ms);
  }

  function showLogin() {
    gen += 1;
    labelFetchAt = 0;
    stopPolling();
    liveStop();
    unlockPress();
    cancelPartSearch();
    partBusy = false;
    partSearchGen += 1;
    clearPartResults();
    lastPart = undefined;
    lastState = undefined;
    curState = undefined;
    pressAllowed = false;
    el.app.hidden = true;
    el.login.hidden = false;
    el.logout.hidden = true;
    el.who.textContent = "";
    scanOpen = false;
    scanAllowed = false;
    scanActiveNow = false;
    scanSnap = null;
    scanBusy = false;
    scanFetching = false;
    reviewAllowed = false;
    reviewBusy = false;
    clearScanPics();
    renderScanPanel();
  }

  function handleError(g, err) {
    if (g !== gen) { return; }
    if (err.status === 401) { showLogin(); return; }
    unlockPress();
    setError(err.message);
    schedule(POLL_MS);
  }

  function poll() {
    timer = null;
    var g = gen;
    request("GET", "/api/state").then(function (s) {
      if (g !== gen) { return; }
      if (!needsImages(s)) {
        render(s, false);
        schedule(POLL_MS);
        return;
      }
      return request("GET", "/api/state?images=1").then(function (full) {
        if (g !== gen) { return; }
        render(full, true);
        lastPart = full.part_id;
        lastState = full.state;
        schedule(POLL_MS);
      });
    }).then(null, function (err) { handleError(g, err); });
  }

  // ----- part picker (only when parts_mode is 'mes') ---------------------
  // The part may change only between parts: the picker is locked while a press is
  // in flight, while a select is running, or whenever the state is not IDLE.
  function partsLocked() {
    return partBusy || pressing || captureLocked || curState !== "IDLE";
  }

  function refreshPartPanel() {
    var locked = partsLocked();
    el.partQ.disabled = locked;
    el.partSearch.disabled = locked;
    Array.prototype.forEach.call(el.partResults.querySelectorAll("button"), function (b) {
      b.disabled = locked;
    });
  }

  function setPartNote(msg) { el.partNote.textContent = msg || ""; }

  function clearPartResults() {
    while (el.partResults.firstChild) { el.partResults.removeChild(el.partResults.firstChild); }
    setPartNote("");
  }

  function cancelPartSearch() {
    if (partSearchTimer !== null) { clearTimeout(partSearchTimer); }
    partSearchTimer = null;
  }

  function warningsText(data) {
    var list = data && Array.isArray(data.warnings) ? data.warnings : [];
    return list.filter(function (w) { return typeof w === "string" && w !== ""; }).join(" ");
  }

  function renderPart(s) {
    partsMode = s.parts_mode === "mes" ? "mes" : "fixed";
    partSelected = s.part_selected === true;
    canInspect = s.can_inspect !== false;
    el.partPanel.hidden = partsMode !== "mes";
    if (partsMode !== "mes") {
      el.partLine.textContent = "";
      el.partBlock.hidden = true;
      return;
    }
    el.partLine.textContent = partSelected
      ? "Part: " + String(s.part_number || "-") + ", model " + String(s.model_version || "-")
      : "No part selected. Search for a part number, then pick it.";
    var reason = typeof s.block_reason === "string" ? s.block_reason : "";
    if (!reason && partSelected && !canInspect) { reason = "Cannot inspect this part."; }
    el.partBlock.textContent = reason;
    el.partBlock.hidden = reason === "";
  }

  // Camera status in the live view section. Absent camera key counts as open.
  function renderCamera(s) {
    var cam = s.camera && typeof s.camera === "object" ? s.camera : null;
    var open = !(cam && cam.open === false);
    el.camStatus.hidden = open;
    if (open) {
      el.camStatus.textContent = "";
      return;
    }
    var why = cam && typeof cam.last_error === "string" && cam.last_error ? cam.last_error : "unknown error";
    el.camStatus.textContent = "Camera not open: " + why + ". The live view retries every few seconds.";
  }

  function showPartResults(rows) {
    clearPartResults();
    var locked = partsLocked();
    rows.forEach(function (row) {
      if (!row || typeof row.part_number !== "string") { return; }
      var li = document.createElement("li");
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "secondary part-choice";
      btn.disabled = locked;
      var pn = document.createElement("span");
      pn.className = "part-pn";
      pn.textContent = row.part_number;
      var desc = document.createElement("span");
      desc.className = "part-desc";
      desc.textContent = typeof row.description === "string" ? row.description : "";
      btn.appendChild(pn);
      btn.appendChild(desc);
      btn.addEventListener("click", function () { selectPart(row.part_number); });
      li.appendChild(btn);
      el.partResults.appendChild(li);
    });
    if (el.partResults.firstChild === null) { setPartNote("No matching part numbers."); }
  }

  // Debounce: a search request goes out at most once per PART_SEARCH_MIN_MS.
  function requestPartSearch() {
    cancelPartSearch();
    var wait = Math.max(0, PART_SEARCH_MIN_MS - (Date.now() - partSearchLastAt));
    partSearchTimer = setTimeout(runPartSearch, wait);
  }

  function runPartSearch() {
    partSearchTimer = null;
    var q = el.partQ.value.trim();
    partSearchGen += 1;
    var g = partSearchGen;
    partSearchLastAt = Date.now();
    clearPartResults();
    if (!q) { return; }
    request("GET", "/api/parts/search?q=" + encodeURIComponent(q) + "&limit=20").then(function (data) {
      if (g !== partSearchGen) { return; }
      showPartResults(data && Array.isArray(data.results) ? data.results : []);
    }, function (err) {
      if (g !== partSearchGen) { return; }
      if (err.status === 401) { showLogin(); return; }
      setError(err.message);
    });
  }

  function selectPart(partNumber) {
    if (partsLocked()) { return; }
    setError("");
    setPartNote("");
    setPartWarn("");
    partBusy = true;
    partSearchGen += 1; // a pending search must not repopulate the list
    cancelPartSearch();
    refreshPartPanel();
    request("POST", "/api/part/select", { part_number: partNumber }).then(function (data) {
      partBusy = false;
      clearPartResults();
      setPartWarn(warningsText(data));
      refreshPress();
      schedule(0);
    }, function (err) {
      partBusy = false;
      refreshPress();
      if (err.status === 401) { showLogin(); return; }
      setError(err.message);
    });
  }

  function setPartWarn(msg) { el.partWarn.textContent = msg || ""; }

  // Press the part: from the Press button or the Enter key. One request at a time.
  var pressing = false;
  // Client-side lockout: after an accepted press the button stays disabled until a
  // poll shows a different state, a poll errors, or PRESS_MAX_WAIT_MS has passed.
  var captureLocked = false;
  var lockFromState;
  var lockTimer = null;
  var lastPressAt = 0;

  function inPressLockout() {
    return pressing || captureLocked || (Date.now() - lastPressAt < PRESS_LOCKOUT_MS);
  }

  // The Press label and enabled state: capture > not ready (flip wait) > allowed.
  function refreshPress() {
    refreshPartPanel();
    var p = el.press;
    if (!p) { return; }
    if (captureLocked) {
      p.textContent = "Capturing...";
      p.disabled = true;
      return;
    }
    if (!pressReady) {
      p.textContent = flipWait > 0 ? "Wait " + Math.ceil(flipWait) + " s" : "Press (Enter)";
      p.disabled = true;
      return;
    }
    if (partsMode === "mes" && !partSelected) {
      p.textContent = "Choose a part";
      p.disabled = true;
      return;
    }
    if (partsMode === "mes" && !canInspect) {
      p.textContent = "No approved model";
      p.disabled = true;
      return;
    }
    p.textContent = "Press (Enter)";
    p.disabled = !pressAllowed;
  }

  function lockPress() {
    captureLocked = true;
    lockFromState = curState;
    if (lockTimer !== null) { clearTimeout(lockTimer); }
    lockTimer = setTimeout(unlockPress, PRESS_MAX_WAIT_MS);
    refreshPress();
  }

  function unlockPress() {
    if (lockTimer !== null) { clearTimeout(lockTimer); }
    lockTimer = null;
    if (captureLocked) {
      captureLocked = false;
      setPressStatus("");
    }
    refreshPress();
  }

  function pressPart() {
    if (inPressLockout()) { setPressStatus(MSG_WAIT_CAPTURE); return; }
    pressing = true;
    lastPressAt = Date.now();
    setError("");
    lockPress();
    setPressStatus(MSG_CAPTURING);
    request("POST", PATHS.press, {}).then(function (data) {
      pressing = false;
      if (data && data.accepted === false) {
        unlockPress();
        setError(curState === "WAIT_FLIP" ? MSG_FLIP_FIRST : MSG_NOT_ACCEPTED);
      }
      schedule(0);
    }, function (err) {
      pressing = false;
      unlockPress();
      if (err.status === 401) { showLogin(); return; }
      setError(err.message);
    });
  }

  document.addEventListener("click", function (ev) {
    var target = ev.target;
    var btn = target && target.closest ? target.closest("[data-action]") : null;
    if (!btn || btn.disabled) { return; }
    var action = btn.getAttribute("data-action");
    if (action === "press") { pressPart(); return; }
    var body = {};
    if (action === "resolve" || action === "override") {
      var reason = el.reason.value.trim();
      if (!reason) { setError("Enter a reason before resolving or overriding."); return; }
      body = { verdict: btn.getAttribute("data-verdict"), reason: reason };
    }
    var path = PATHS[action];
    if (action === "label_propose") {
      body = { verdict: btn.getAttribute("data-verdict") };
      var criterion = el.criterion.value.trim();
      if (body.verdict === "BAD" && criterion) { body.criterion_id = criterion; }
    }
    if (action === "label_approve") {
      // Admin only: approve or reject the pending proposal of one collected part.
      var partId = btn.getAttribute("data-part-id");
      if (partId) { body.part_id = partId; }
      path = btn.getAttribute("data-decision") === "reject" ? "/api/label/reject" : "/api/label/approve";
      if (path === "/api/label/reject") {
        var why = $("reject-reason").value.trim();
        if (!why) { setError("Enter a reject reason first."); return; }
        body.reason = why;
      }
    }
    setError("");
    request("POST", path, body).then(function () {
      if (action === "resolve" || action === "override") { el.reason.value = ""; }
      if (action === "label_approve") { $("reject-reason").value = ""; }
      if (action === "label_propose" || action === "label_approve") { maybeRefreshLabels(true); }
      schedule(0);
    }, function (err) {
      if (err.status === 401) { showLogin(); return; }
      setError(err.message);
    });
  });

  // Enter presses the part, unless focus is in a field or button (typing a reason,
  // or Enter on a focused button, must not press twice). Other keys do nothing.
  document.addEventListener("keydown", function (ev) {
    if (ev.key !== "Enter" || ev.repeat) { return; }
    if (el.app.hidden || !el.press) { return; }
    var focus = document.activeElement;
    var tag = focus && focus.tagName ? focus.tagName : "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || tag === "BUTTON") { return; }
    // While a scan runs, Enter takes a scan step and never an inspection press.
    if (scanIsActive()) {
      if (inPressLockout()) { setScanStatus(MSG_WAIT_CAPTURE); return; }
      if (scanPressEnabled()) { scanPress(); }
      return;
    }
    if (inPressLockout()) { setPressStatus(MSG_WAIT_CAPTURE); return; }
    if (el.press.disabled) { return; }
    pressPart();
  });

  // Live view follows page visibility.
  document.addEventListener("visibilitychange", syncLive);

  el.form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    var user = $("username");
    var pass = $("password");
    var body = { username: user.value, password: pass.value };
    pass.value = "";
    setError("");
    request("POST", "/api/login", body).then(function () {
      user.value = "";
      schedule(0);
    }, function (err) {
      setError(err.message);
    });
  });

  el.partForm.addEventListener("submit", function (ev) {
    ev.preventDefault();
    if (partsLocked()) { return; }
    requestPartSearch();
  });

  el.logout.addEventListener("click", function () {
    request("POST", "/api/logout", {}).catch(function () {}).then(function () {
      showLogin();
    });
  });

  ensureTiles(DEFAULT_CAMERAS);
  ensureLive(DEFAULT_CAMERAS);
  poll();
})();
</script>
</body>
</html>
"""
