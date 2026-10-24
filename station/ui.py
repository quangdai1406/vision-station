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
import sys
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from station.auth import ACTIONS, SessionManager, UserStore, role_allows
from station.contracts import InvalidTransition, Verdict

SESSION_COOKIE = "station_session"
MAX_BODY_BYTES = 64 * 1024
MAX_REASON_CHARS = 500

_ACTION_PATHS = {
    "/api/press": "press",
    "/api/ack": "ack",
    "/api/resolve": "resolve",
    "/api/override": "override",
    "/api/clear-fault": "clear_fault",
}

_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src data:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
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


class _Handler(BaseHTTPRequestHandler):
    # Filled in per server by make_server().
    station: Any = None
    users: UserStore | None = None
    sessions: SessionManager | None = None

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
        raise _HttpError(404, "not found")

    def _post(self) -> None:
        path = self.path.partition("?")[0]
        if path == "/api/login":
            return self._login()
        if path == "/api/logout":
            return self._logout()
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
        body["auth"] = {
            "username": username,
            "role": role,
            "actions": [a for a in ACTIONS if role_allows(role, a)],
        }
        self._send_json(200, body)

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
        self._send_json(200, {"ok": True})

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
                host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Build (but do not start) the UI server.

    `session` must provide: snapshot(include_images=False) -> dict, press() -> bool,
    resolve_review(by, new_verdict, reason), override(by, new_verdict, reason),
    ack_alarm(), clear_fault(), tick(). InvalidTransition -> 409, ValueError -> 400.
    """
    handler = type("StationHandler", (_Handler,), {
        "station": session,
        "users": users,
        "sessions": sessions,
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
.counters { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; text-align: center; }
.counters span { display: block; font-size: 14px; color: var(--muted); }
.counters b { display: block; font-size: 28px; }
.tiles { display: grid; grid-template-columns: repeat(var(--cols-m, 2), 1fr); gap: 8px; }
@media (min-width: 700px) { .tiles { grid-template-columns: repeat(var(--cols, 4), 1fr); } }
.poc { background: var(--poc); color: #1b1d21; font-weight: 700; border-radius: 10px;
  padding: 12px 16px; margin-bottom: 16px; }
.hint { margin: 8px 0 0; }
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
</style>
</head>
<body>
<header>
  <h1>Vision Station</h1>
  <div>
    <span id="who" class="muted"></span>
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
    <div id="tiles" class="card"></div>
    <div class="card">
      <button data-action="press">Press (Enter)</button>
      <button data-action="ack">Acknowledge alarm</button>
      <button data-action="clear_fault">Clear fault</button>
      <p id="press-hint" class="muted hint">Enter key or click: start TOP scan, flip the part, press again for the BOTTOM scan.</p>
    </div>
    <div class="card">
      <label for="reason">Reason (required for resolve and override)</label>
      <input id="reason" maxlength="500" autocomplete="off">
      <button data-action="resolve" data-verdict="GOOD">Resolve REVIEW: GOOD</button>
      <button data-action="resolve" data-verdict="BAD">Resolve REVIEW: BAD</button>
      <button data-action="override" data-verdict="GOOD">Override BAD to GOOD</button>
    </div>
  </section>
  <p id="error" class="error" role="alert"></p>
</main>
<script>
(function () {
  "use strict";
  var POLL_MS = 1000;
  var DEFAULT_CAMERAS = ["C1", "C2", "C3", "C4"];
  var SIDES = ["TOP", "BOTTOM"];
  var PLACEHOLDER_W = 320;
  var PLACEHOLDER_H = 240;
  var PATHS = {
    press: "/api/press", ack: "/api/ack", clear_fault: "/api/clear-fault",
    resolve: "/api/resolve", override: "/api/override"
  };

  function $(id) { return document.getElementById(id); }

  var el = {
    login: $("login-box"), app: $("app"), form: $("login-form"), who: $("who"),
    logout: $("logout"), error: $("error"), state: $("state"), verdict: $("verdict"),
    total: $("c-total"), good: $("c-good"), bad: $("c-bad"), tiles: $("tiles"),
    reason: $("reason"), poc: $("poc"),
    press: document.querySelector('[data-action="press"]')
  };

  var timer = null;
  var gen = 0;
  // part_id and state of the snapshot that the shown images belong to.
  var lastPart;
  var lastState;

  function setError(msg) { el.error.textContent = msg || ""; }

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

    var verdict = (s.verdict === "GOOD" || s.verdict === "BAD" || s.verdict === "REVIEW") ? s.verdict : "NONE";
    el.verdict.className = "verdict v-" + verdict;
    el.verdict.textContent = verdict === "NONE" ? "-" : verdict;

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
    stopPolling();
    lastPart = undefined;
    lastState = undefined;
    el.app.hidden = true;
    el.login.hidden = false;
    el.logout.hidden = true;
    el.who.textContent = "";
  }

  function handleError(g, err) {
    if (g !== gen) { return; }
    if (err.status === 401) { showLogin(); return; }
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

  // Press the part: from the Press button or the Enter key. One request at a time.
  var pressing = false;

  function pressPart() {
    if (pressing) { return; }
    pressing = true;
    setError("");
    request("POST", PATHS.press, {}).then(function (data) {
      pressing = false;
      if (data && data.accepted === false) {
        setError("Press was not accepted in the current state.");
      }
      schedule(0);
    }, function (err) {
      pressing = false;
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
    setError("");
    request("POST", PATHS[action], body).then(function () {
      if (action === "resolve" || action === "override") { el.reason.value = ""; }
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
    if (el.app.hidden || !el.press || el.press.disabled || pressing) { return; }
    var focus = document.activeElement;
    var tag = focus && focus.tagName ? focus.tagName : "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || tag === "BUTTON") { return; }
    pressPart();
  });

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

  el.logout.addEventListener("click", function () {
    request("POST", "/api/logout", {}).catch(function () {}).then(function () {
      showLogin();
    });
  });

  ensureTiles(DEFAULT_CAMERAS);
  poll();
})();
</script>
</body>
</html>
"""
