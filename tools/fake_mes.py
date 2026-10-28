"""HOME TRIAL fake MES: a local stand-in for the machine API, for testing at home.

This is NOT the real MES. It answers the same routes the station uses (part search, part
lookup, criteria, inspection events) so the station can be exercised without the factory
network. Standard library only.

Run (PowerShell or bash), with the token in an environment variable, never on the command line:
  python3 -m tools.fake_mes --token-env STATION_MES_TOKEN --port 3000 [--host 127.0.0.1]
      [--parts FILE.json] [--criteria FILE.json] [--out DIR]

The token is read only from the environment variable named by --token-env. It is checked
against the Authorization header on every request and is never written to a file or printed.

Routes (all need "Authorization: Bearer <token>", else 401):
  GET  /api/machine/parts/search?q=&limit=     list of part objects (limit clamped 1..50)
  GET  /api/machine/parts/<part_number>        one part, or 404
  GET  /api/machine/criteria/<part_number>     list of criteria, or 404
  POST /api/machine/inspection-events          multipart/form-data; needs Idempotency-Key
       first time for a key: 201 {"id": n}; same key again: 200 {"id": n} (a replay)
       missing Idempotency-Key: 400

--parts FILE.json: a JSON list of part objects with "part_number", or an object keyed by
part number. --criteria FILE.json: an object mapping part number to a list of criteria in the
tools/criteria.py format. Without them, a built-in sample part with two visual criteria is used.

Every accepted event (not replays) is appended as one JSON line to OUT/events.jsonl: the key,
the text fields, and the name and byte size of each photo. Photo bytes and the token are not
written. On restart the existing events.jsonl is read so keys keep their ids.
"""
from __future__ import annotations

import argparse
import datetime
import email.parser
import email.policy
import hmac
import http.server
import json
import os
import sys
import threading
import urllib.parse
from pathlib import Path
from typing import Any

from tools.criteria import CriteriaError, _validate as _validate_criteria

BANNER = "HOME TRIAL fake MES, not the real MES"
EVENTS_PATH = "/api/machine/inspection-events"
SEARCH_PATH = "/api/machine/parts/search"
PARTS_PREFIX = "/api/machine/parts/"
CRITERIA_PREFIX = "/api/machine/criteria/"
SEARCH_DEFAULT_LIMIT = 20
SEARCH_LIMIT_MIN = 1
SEARCH_LIMIT_MAX = 50
MAX_BODY_BYTES = 64 * 1024 * 1024
DEFAULT_OUT_DIR = "fake_mes_out"

SAMPLE_PARTS: dict[str, dict[str, Any]] = {
    "HT-LINER-1": {
        "part_number": "HT-LINER-1",
        "description": "HOME TRIAL sample liner (fake data)",
        "part_type": "FG",
        "lifecycle": "ACTIVE",
    },
}
SAMPLE_CRITERIA: dict[str, list[dict[str, Any]]] = {
    "HT-LINER-1": [
        {"id": "CR-1", "criteria_name": "Scratch", "check_type": "visual", "active": True},
        {"id": "CR-2", "criteria_name": "Dent", "check_type": "visual", "active": True},
    ],
}


class FakeMesError(Exception):
    """A configuration problem with the fake MES inputs. Never contains the token."""


class MultipartError(Exception):
    """The request body is not a multipart/form-data message this fake can read."""


def _read_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise FakeMesError(f"cannot read {path}: {exc.strerror or type(exc).__name__}") from None
    except ValueError:
        raise FakeMesError(f"{path} is not valid JSON") from None


def load_parts(path: str) -> dict[str, dict[str, Any]]:
    """Read a parts file (a list of parts, or an object keyed by part number)."""
    data = _read_json(path)
    if isinstance(data, dict):
        items = []
        for key, item in data.items():
            if not isinstance(item, dict):
                raise FakeMesError(f"{path}: part {key!r} is not an object")
            items.append({"part_number": key, **item})
    elif isinstance(data, list):
        items = data
    else:
        raise FakeMesError(f"{path}: expected a list of parts or an object keyed by part number")
    parts: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise FakeMesError(f"{path}: part entry {i} is not an object")
        pn = item.get("part_number")
        if not isinstance(pn, str) or not pn.strip():
            raise FakeMesError(f"{path}: part entry {i} has no 'part_number'")
        if pn in parts:
            raise FakeMesError(f"{path}: part {pn!r} appears more than once")
        parts[pn] = dict(item)
    if not parts:
        raise FakeMesError(f"{path}: no parts in file")
    return parts


def load_criteria(path: str) -> dict[str, list[dict[str, Any]]]:
    """Read a criteria file: an object mapping part number to a list of criteria."""
    data = _read_json(path)
    if not isinstance(data, dict):
        raise FakeMesError(f"{path}: expected an object mapping part number to criteria")
    try:
        return {str(pn): _validate_criteria(items, f"{path}: part {pn}")
                for pn, items in data.items()}
    except CriteriaError as exc:
        raise FakeMesError(str(exc)) from None


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], list[tuple[str, str, bytes]]]:
    """Return (text fields, photos as (field name, filename, bytes)) from a multipart body."""
    if not content_type.lower().startswith("multipart/form-data"):
        raise MultipartError("Content-Type must be multipart/form-data")
    try:
        raw = (b"Content-Type: " + content_type.encode("latin-1")
               + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
        msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(raw)
    except (ValueError, UnicodeEncodeError):
        raise MultipartError("malformed multipart body") from None
    if not msg.is_multipart():
        raise MultipartError("malformed multipart body")
    fields: dict[str, str] = {}
    photos: list[tuple[str, str, bytes]] = []
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str) or not name:
            raise MultipartError("form part without a name")
        data = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is not None:
            photos.append((name, filename, data))
        else:
            try:
                fields[name] = data.decode("utf-8")
            except UnicodeDecodeError:
                raise MultipartError(f"field {name} is not UTF-8") from None
    return fields, photos


def _parse_limit(text: str | None) -> int:
    try:
        value = int(text) if text is not None else SEARCH_DEFAULT_LIMIT
    except ValueError:
        value = SEARCH_DEFAULT_LIMIT
    return min(max(value, SEARCH_LIMIT_MIN), SEARCH_LIMIT_MAX)


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # quiet: request lines can carry search text; headers are never logged

    def _send_json(self, status: int, payload: Any, extra: dict[str, str] | None = None) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        given = self.headers.get("Authorization", "").encode("utf-8")
        expected = f"Bearer {self.server.token}".encode("utf-8")  # type: ignore[attr-defined]
        return hmac.compare_digest(given, expected)

    def _unauthorized(self) -> None:
        self._send_json(401, {"error": "unauthorized"}, {"WWW-Authenticate": "Bearer"})

    def do_GET(self) -> None:
        if not self._authorized():
            return self._unauthorized()
        srv: FakeMesServer = self.server  # type: ignore[assignment]
        url = urllib.parse.urlsplit(self.path)
        if url.path == SEARCH_PATH:
            query = urllib.parse.parse_qs(url.query)
            q = query.get("q", [""])[-1].strip().lower()
            limit = _parse_limit(query.get("limit", [None])[-1])
            rows = [dict(p) for p in srv.parts.values()
                    if q in p["part_number"].lower() or q in str(p.get("description", "")).lower()]
            return self._send_json(200, rows[:limit])
        if url.path.startswith(PARTS_PREFIX):
            pn = urllib.parse.unquote(url.path[len(PARTS_PREFIX):])
            if pn in srv.parts:
                return self._send_json(200, srv.parts[pn])
            return self._send_json(404, {"error": "not found"})
        if url.path.startswith(CRITERIA_PREFIX):
            pn = urllib.parse.unquote(url.path[len(CRITERIA_PREFIX):])
            if pn in srv.criteria:
                return self._send_json(200, srv.criteria[pn])
            return self._send_json(404, {"error": "not found"})
        return self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        srv: FakeMesServer = self.server  # type: ignore[assignment]
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._send_json(400, {"error": "bad Content-Length"})
        if length < 0 or length > MAX_BODY_BYTES:
            return self._send_json(413, {"error": "request body too large"})
        body = self.rfile.read(length)  # read before answering so the connection stays clean
        if not self._authorized():
            return self._unauthorized()
        if urllib.parse.urlsplit(self.path).path != EVENTS_PATH:
            return self._send_json(404, {"error": "not found"})
        key = (self.headers.get("Idempotency-Key") or "").strip()
        if not key:
            return self._send_json(400, {"error": "Idempotency-Key header is required"})
        try:
            fields, photos = parse_multipart(self.headers.get("Content-Type", ""), body)
        except MultipartError as exc:
            return self._send_json(400, {"error": str(exc)})
        status, payload = srv.accept_event(key, fields, photos)
        self._send_json(status, payload)


class FakeMesServer(http.server.ThreadingHTTPServer):
    """The fake MES. Use port 0 in tests; read the bound port from base_url."""

    daemon_threads = True

    def __init__(self, token: str, *, parts: dict[str, dict[str, Any]] | None = None,
                 criteria: dict[str, list[dict[str, Any]]] | None = None,
                 out_dir: str | os.PathLike | None = None,
                 host: str = "127.0.0.1", port: int = 0) -> None:
        if not token:
            raise FakeMesError("no token supplied")
        self.token = token
        self.parts = {pn: dict(p) for pn, p in (parts or SAMPLE_PARTS).items()}
        self.criteria = {pn: [dict(c) for c in cs]
                         for pn, cs in (criteria if criteria is not None else SAMPLE_CRITERIA).items()}
        self.out_dir = Path(out_dir or DEFAULT_OUT_DIR)
        self.events_path = self.out_dir / "events.jsonl"
        self.lock = threading.Lock()
        self.accepted: dict[str, int] = {}  # Idempotency-Key -> event id
        self.next_id = 0
        self.replays = 0
        self._thread: threading.Thread | None = None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._load_existing()
        super().__init__((host, port), _Handler)

    @property
    def base_url(self) -> str:
        return f"http://{self.server_address[0]}:{self.server_address[1]}"

    def _load_existing(self) -> None:
        if not self.events_path.exists():
            return
        with self.events_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # a partial last line from a crash is ignored
                if isinstance(rec, dict) and isinstance(rec.get("key"), str) and isinstance(rec.get("id"), int):
                    self.accepted[rec["key"]] = rec["id"]
                    self.next_id = max(self.next_id, rec["id"])

    def accept_event(self, key: str, fields: dict[str, str],
                     photos: list[tuple[str, str, bytes]]) -> tuple[int, dict[str, int]]:
        """Store one inspection event. Returns (status, payload) for the HTTP answer."""
        with self.lock:
            if key in self.accepted:
                self.replays += 1
                return 200, {"id": self.accepted[key]}
            event_id = self.next_id + 1
            record = {
                "id": event_id,
                "key": key,
                "received_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "fields": fields,
                "photos": [{"field": f, "filename": fn, "bytes": len(data)} for f, fn, data in photos],
            }
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.next_id = event_id
            self.accepted[key] = event_id
            return 201, {"id": event_id}

    def start(self) -> "FakeMesServer":
        self._thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1},
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(5)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=BANNER, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-env", required=True, metavar="NAME",
                    help="name of the environment variable holding the token (the value is never printed)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=3000, help="port (default 3000; 0 picks a free port)")
    ap.add_argument("--parts", metavar="FILE.json", help="parts to serve (default: built-in sample)")
    ap.add_argument("--criteria", metavar="FILE.json", help="criteria per part (default: built-in sample)")
    ap.add_argument("--out", metavar="DIR", default=DEFAULT_OUT_DIR,
                    help=f"folder for events.jsonl (default {DEFAULT_OUT_DIR})")
    args = ap.parse_args(argv)

    token = os.environ.get(args.token_env, "")
    if not token:
        print(f"error: environment variable {args.token_env} is not set or empty. "
              "Set it to the home-trial token and run again (the token is not printed).",
              file=sys.stderr)
        return 2
    try:
        parts = load_parts(args.parts) if args.parts else None
        criteria = load_criteria(args.criteria) if args.criteria else None
        if parts is not None and criteria is None:
            criteria = {pn: [] for pn in parts}
        server = FakeMesServer(token, parts=parts, criteria=criteria, out_dir=args.out,
                               host=args.host, port=args.port)
    except FakeMesError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot start on {args.host}:{args.port}: {exc.strerror or exc}", file=sys.stderr)
        return 2

    print(BANNER)
    print(f"URL: {server.base_url}  (MES API base; events at {EVENTS_PATH})")
    print(f"events: {server.events_path}")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: bound to a non-loopback address; anyone on this network can reach it")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print(f"stopped. replays answered: {server.replays}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
