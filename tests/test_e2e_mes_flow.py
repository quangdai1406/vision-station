"""End-to-end (plan E1): the MES ticket login, the part flow and the laptop, over real HTTP.

The station runs as in production for this flow: auth.mode 'mes_ticket', parts.mode 'mes',
model.remote (the laptop inference service), evidence.sync into a temporary folder. The UI
server listens on 127.0.0.1. The MES is a fake http.server on 127.0.0.1 with the machine
routes (sessions, parts, criteria, inspection-events). The laptop is tests/fake_laptop.py.
Cameras are FakeCameraSet frames. Nothing leaves 127.0.0.1.

The tokens are made-up strings generated in this file. The laptop token reaches RemoteModel
through an environment variable that this test sets (mock.patch.dict) and never reads back.
The MES token is passed to build_station() directly. No file in this test holds a token.
"""
from __future__ import annotations

import contextlib
import copy
import datetime
import email.parser
import email.policy
import hashlib
import http.client
import http.server
import io
import json
import os
import pathlib
import secrets
import shutil
import sqlite3
import tempfile
import threading
import unittest
import urllib.parse
from unittest import mock

from station.app import build_station
from station.cameras import FakeCameraSet
from station.contracts import Side
from station.evidence_sync import planned_destination
from station.outputs import RecordingOutputs
from tests.fake_laptop import FakeLaptop

LAPTOP_TOKEN_ENV = "VISION_E2E_MES_FLOW_LAPTOP_TOKEN"  # a name only
MES_TOKEN_ENV = "VISION_E2E_MES_FLOW_MES_TOKEN"  # a name only, in the config
ROLE_MAP = {"inspector": "operator", "supervisor": "supervisor"}
EXPIRES = "2026-11-04T18:00:00Z"
PAST_MIN_FLIP = 2.5  # past the default min_flip_s (2.0 s) between the TOP and BOTTOM press

PARTS = {
    "PN-A": {"part_number": "PN-A", "description": "Liner left", "part_type": "FG",
             "lifecycle": "ACTIVE", "unit_cost": 12.5, "bom": ["x"]},
    "PN-B": {"part_number": "PN-B", "description": "Liner right", "part_type": "FG",
             "lifecycle": "ACTIVE", "unit_cost": 13.0, "bom": ["y"]},
}
CRITERIA = {
    "PN-A": [
        {"id": "CR-1", "criteria_name": "Scratch", "criteria_description": "surface scratch",
         "check_type": "visual", "active": True},
        {"id": "CR-2", "criteria_name": "Dent", "criteria_description": "dent in the skin",
         "check_type": "visual", "active": True},
        {"id": "CR-3", "criteria_name": "Old mark", "criteria_description": "retired",
         "check_type": "visual", "active": False},
    ],
}


SCRATCH = {"defect_class": "scratch", "score": 0.9, "box": [1, 1, 4, 4]}  # maps to CR-1


def frame_bytes(camera_id: str, side: Side, n: int) -> bytes:
    """Each picture carries its camera and side, so a test can tell which one was attached."""
    return f"PIC {camera_id} {side.value} {n}".encode("ascii")


def parse_multipart(content_type: str, body: bytes) -> tuple[dict, dict]:
    """Return (text fields, files) where files maps field name -> (filename, bytes)."""
    raw = b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(raw)
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True)
        if part.get_filename() is not None:
            files[name] = (part.get_filename(), data)
        else:
            fields[name] = data.decode("utf-8")
    return fields, files


def sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ fake MES

class _MesHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - keep test output quiet
        pass

    def _send(self, status: int, payload) -> None:
        data = json.dumps(payload).encode("utf-8")
        with self.server.lock:
            self.server.wire.append(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _auth_ok(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {self.server.token}"

    def do_GET(self) -> None:
        srv = self.server
        url = urllib.parse.urlsplit(self.path)
        with srv.lock:
            srv.gets.append(url.path)
        if not self._auth_ok():
            return self._send(401, {"error": "unauthorized"})
        if url.path == "/api/machine/parts/search":
            q = urllib.parse.parse_qs(url.query).get("q", [""])[-1].lower()
            rows = [dict(r) for r in srv.parts.values()
                    if q in r["part_number"].lower() or q in r["description"].lower()]
            return self._send(200, rows)
        if url.path.startswith("/api/machine/parts/"):
            pn = urllib.parse.unquote(url.path[len("/api/machine/parts/"):])
            if pn in srv.parts:
                return self._send(200, srv.parts[pn])
            return self._send(404, {"error": "not found"})
        if url.path.startswith("/api/machine/criteria/"):
            pn = urllib.parse.unquote(url.path[len("/api/machine/criteria/"):])
            if pn in srv.criteria:
                return self._send(200, srv.criteria[pn])
            return self._send(404, {"error": "not found"})
        if url.path.startswith("/api/machine/sessions/"):
            sid = url.path[len("/api/machine/sessions/"):]
            with srv.lock:
                sess = srv.sessions.get(sid)
                if sess is None or not sess["active"]:
                    return self._send(401, {"error": "session not valid"})
                payload = {"active": True, "username": sess["username"],
                           "roles": list(sess["roles"]), "expiresAt": EXPIRES}
            return self._send(200, payload)
        return self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        srv = self.server
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        with srv.lock:
            srv.wire.append(body)
        if not self._auth_ok():
            return self._send(401, {"error": "unauthorized"})
        path = urllib.parse.urlsplit(self.path).path
        if path == "/api/machine/sessions/redeem":
            try:
                ticket = json.loads(body.decode("utf-8")).get("ticket")
            except (ValueError, AttributeError):
                return self._send(400, {"error": "bad body"})
            with srv.lock:
                rec = srv.tickets.get(ticket) if isinstance(ticket, str) else None
                if rec is None:
                    return self._send(401, {"error": "unknown ticket"})
                if rec["used"]:
                    return self._send(409, {"error": "ticket already used"})
                rec["used"] = True
                sid = "sess-" + secrets.token_hex(6)
                srv.sessions[sid] = {"username": rec["username"], "roles": rec["roles"], "active": True}
                payload = {"sessionId": sid, "username": rec["username"],
                           "roles": list(rec["roles"]), "expiresAt": EXPIRES}
            return self._send(200, payload)
        if path.startswith("/api/machine/sessions/") and path.endswith("/end"):
            sid = path[len("/api/machine/sessions/"):-len("/end")]
            with srv.lock:
                sess = srv.sessions.get(sid)
                if sess is None:
                    return self._send(404, {"error": "not found"})
                sess["active"] = False
            return self._send(200, {"ok": True})
        if path == "/api/machine/inspection-events":
            key = self.headers.get("Idempotency-Key")
            station_session = self.headers.get("Station-Session")
            with srv.lock:
                sess = srv.sessions.get(station_session) if station_session else None
                record = {
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "key": key,
                    "station_session": station_session,
                    "ctype": self.headers.get("Content-Type"),
                    "body": body,
                    "status": None,
                }
                srv.posts.append(record)
                if sess is None or not sess["active"]:
                    status, payload = 401, {"error": "unknown or ended station session"}
                elif key in srv.accepted:
                    status, payload = 200, {"id": srv.accepted[key]}
                else:
                    srv.next_id += 1
                    srv.accepted[key] = srv.next_id
                    srv.accepted_by[key] = sess["username"]
                    status, payload = 201, {"id": srv.next_id}
                record["status"] = status
            return self._send(status, payload)
        return self._send(404, {"error": "not found"})


class FakeMes(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, token: str) -> None:
        super().__init__(("127.0.0.1", 0), _MesHandler)
        self.token = token
        self.lock = threading.RLock()  # _send() takes it again while a handler holds it
        self.parts = copy.deepcopy(PARTS)
        self.criteria = copy.deepcopy(CRITERIA)
        self.tickets: dict[str, dict] = {}
        self.sessions: dict[str, dict] = {}  # insertion order = login order
        self.gets: list[str] = []
        self.posts: list[dict] = []
        self.accepted: dict[str, int] = {}
        self.accepted_by: dict[str, str] = {}
        self.next_id = 0
        self.wire: list[bytes] = []  # every request and response body the MES saw or sent
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1},
                                       daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def issue(self, username: str, roles: tuple[str, ...]) -> str:
        """A one-time ticket, as the MES hub card would issue it (made up for the test)."""
        ticket = "tkt-" + secrets.token_urlsafe(16)
        with self.lock:
            self.tickets[ticket] = {"username": username, "roles": tuple(roles), "used": False}
        return ticket

    def latest_session(self, username: str) -> str:
        with self.lock:
            sids = [sid for sid, s in self.sessions.items() if s["username"] == username]
        return sids[-1]

    def end(self, sid: str) -> None:
        """The MES ends the session by itself (idle timeout, logout elsewhere)."""
        with self.lock:
            self.sessions[sid]["active"] = False

    def session_active(self, sid: str) -> bool:
        with self.lock:
            return self.sessions[sid]["active"]

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self.thread.join(5)


# ------------------------------------------------------------------ station fakes

class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class LightSpy(RecordingOutputs):
    """Records the light calls and how many parts were already saved at each one."""

    def __init__(self) -> None:
        super().__init__()
        self.store = None
        self.saved_at: dict[str, list[int]] = {}

    def _mark(self, name: str) -> None:
        if self.store is not None:
            self.saved_at.setdefault(name, []).append(len(self.store.recent_parts(50)))

    def show_good(self) -> None:
        self._mark("show_good")
        super().show_good()

    def show_bad(self) -> None:
        self._mark("show_bad")
        super().show_bad()

    def show_review(self) -> None:
        self._mark("show_review")
        super().show_review()

    def show_fault(self) -> None:
        self._mark("show_fault")
        super().show_fault()


# ------------------------------------------------------------------ tests

class MesFlowE2E(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="mes-flow-e2e-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

        # Stderr of the station (tracebacks, log lines) is captured for the token scan.
        self.err = io.StringIO()
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stderr(self.err))
        self.addCleanup(stack.close)

        self.mes_token = "mes-flow-mes-token-" + secrets.token_hex(8)
        self.laptop_token = "mes-flow-laptop-token-" + secrets.token_hex(8)
        self.mes = FakeMes(self.mes_token)
        self.addCleanup(self.mes.close)

        self.laptop = FakeLaptop(models={"PN-A": "v-a", "PN-B": "v-b"}, token=self.laptop_token)
        self.laptop_open = True
        self.addCleanup(self._close_laptop)
        env = mock.patch.dict(os.environ, {LAPTOP_TOKEN_ENV: self.laptop_token})
        env.start()
        self.addCleanup(env.stop)

        self.data_dir = os.path.join(self.tmp, "data")
        self.models_dir = os.path.join(self.tmp, "models")
        os.makedirs(self.models_dir)  # validate_config requires it; the laptop holds the models
        self.dest_root = os.path.join(self.tmp, "laptop-store")
        os.makedirs(self.dest_root)  # stands for the mounted laptop share; sync never creates it
        self.queue_dir = os.path.join(self.tmp, "sync-queue")
        self.clock = FakeClock()
        self.outputs = LightSpy()
        self.app = None
        self.port = None
        self.wire: list[bytes] = []  # request and response bodies seen on the station's HTTP

    def _close_laptop(self) -> None:
        if self.laptop_open:
            self.laptop.close()
            self.laptop_open = False

    # ---- helpers

    def config(self) -> dict:
        return {
            "site": "SITE1", "line": "L1", "station_id": "VIS-E2E-1",
            "lot": "LOT-E2E", "operator": "config-operator", "supervisor": "config-supervisor",
            "material_handler": "mh1",
            "thresholds": {"t_low": 0.5, "t_img": 0.8},
            "data_dir": self.data_dir,
            "users_file": os.path.join(self.tmp, "no-local-users.json"),  # not read in mes_ticket mode
            "ui": {"host": "127.0.0.1", "port": 0, "session_timeout_s": 900},
            "mes": {
                "base_url": self.mes.base_url,
                "token_env": MES_TOKEN_ENV,
                "criteria_ids": ["CR-1", "CR-2"],
                "class_to_criterion": {"scratch": "CR-1"},
                "default_criterion_id": None,
                "sync_interval_s": 3600,  # the tests call sync_once() and evidence_sync_once()
                "default_bad_disposition": "quarantine",
            },
            "cameras": {"kind": "fake"},
            "model": {"kind": "marker", "version": "remote",
                      "remote": {"base_url": self.laptop.base, "token_env": LAPTOP_TOKEN_ENV,
                                 "timeout_s": 5}},
            "outputs": {"kind": "console"},
            "flip_timeout_s": 60,
            "parts": {"mode": "mes", "models_dir": self.models_dir},
            "auth": {"mode": "mes_ticket", "role_map": dict(ROLE_MAP)},
            "evidence": {"sync": {"dest_root": self.dest_root, "queue_dir": self.queue_dir}},
        }

    def start(self) -> None:
        self.outputs.store = None
        self.app = build_station(
            self.config(),
            cameras=FakeCameraSet(frame_bytes=frame_bytes),
            outputs=self.outputs,
            clock=self.clock,
            token=self.mes_token,
        )
        self.outputs.store = self.app.store
        self.addCleanup(self.app.stop)
        self.app.start()
        self.port = self.app.server.server_address[1]

    def call(self, method: str, path: str, body=None, cookie: str | None = None,
             raw_body: bytes | None = None):
        """(status, parsed JSON or text). Bodies and replies are kept in self.wire."""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {}
        data = raw_body
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        if data is not None:
            headers["Content-Type"] = "application/json"
            self.wire.append(data)
        if cookie:
            headers["Cookie"] = f"station_session={cookie}"
        try:
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            set_cookie = resp.getheader("Set-Cookie")
            ctype = resp.getheader("Content-Type") or ""
        finally:
            conn.close()
        self.wire.append(raw)
        self.last_set_cookie = set_cookie
        if ctype.startswith("application/json"):
            return resp.status, json.loads(raw.decode("utf-8") or "{}")
        return resp.status, raw.decode("utf-8", "replace")

    def enter(self, username: str, roles: tuple[str, ...] = ("inspector",)) -> str:
        """Log in through POST /api/enter with a fresh ticket. Returns the station cookie."""
        ticket = self.mes.issue(username, roles)
        status, body = self.call("POST", "/api/enter", {"ticket": ticket})
        self.assertEqual(status, 200, body)
        cookie = self.last_set_cookie.split(";", 1)[0].split("=", 1)[1]
        return cookie

    def select(self, cookie: str, part_number: str) -> dict:
        status, body = self.call("POST", "/api/part/select", {"part_number": part_number}, cookie)
        self.assertEqual(status, 200, body)
        self.assertIs(body["can_inspect"], True, body)
        return body

    def press(self, cookie: str, gap: float = 1.0) -> bool:
        self.clock.now += gap  # > 0.3 s debounce, and the flip gap when gap is PAST_MIN_FLIP
        status, body = self.call("POST", "/api/press", {}, cookie)
        self.assertEqual(status, 200, body)
        return bool(body["accepted"])

    def press_part(self, cookie: str) -> None:
        self.assertTrue(self.press(cookie, 1.0))  # TOP: now WAIT_FLIP
        self.assertTrue(self.press(cookie, PAST_MIN_FLIP))  # BOTTOM: inspection finishes

    def state(self, cookie: str) -> dict:
        status, body = self.call("GET", "/api/state", cookie=cookie)
        self.assertEqual(status, 200, body)
        return body

    def saved_parts(self) -> list[dict]:
        return list(reversed(self.app.store.recent_parts(50)))  # oldest first

    def posted(self) -> list[dict]:
        out = []
        for rec in self.mes.posts:
            fields, files = parse_multipart(rec["ctype"], rec["body"])
            out.append({"record": rec, "fields": fields, "files": files})
        return out

    def all_station_files(self) -> list[pathlib.Path]:
        return [pathlib.Path(dirpath, name)
                for dirpath, _dirs, names in os.walk(self.tmp) for name in names]

    # ---- tests

    def test_ticket_login_rules(self) -> None:
        self.start()
        ticket = self.mes.issue("insp-1", ("inspector",))

        status, _ = self.call("GET", f"/enter?ticket={ticket}")
        self.assertEqual(status, 400)  # the ticket in the query string is refused

        status, page = self.call("GET", "/enter")
        self.assertEqual(status, 200)
        self.assertIn("location", page)  # the page reads the fragment (location.hash)

        status, body = self.call("POST", "/api/enter", {"ticket": ticket})
        self.assertEqual((status, body), (200, {"role": "operator"}))
        cookie = self.last_set_cookie
        self.assertIn("station_session=", cookie)
        self.assertIn("HttpOnly", cookie)
        op = cookie.split(";", 1)[0].split("=", 1)[1]
        self.assertEqual(self.state(op)["auth"]["username"], "insp-1")

        status, body = self.call("POST", "/api/enter", {"ticket": ticket})  # replay
        self.assertEqual(status, 409, body)

        aud_ticket = self.mes.issue("aud-1", ("auditor",))  # a role that is not mapped
        status, body = self.call("POST", "/api/enter", {"ticket": aud_ticket})
        self.assertEqual(status, 403, body)
        self.assertFalse(self.mes.session_active(self.mes.latest_session("aud-1")))

        status, body = self.call("GET", "/api/state")
        self.assertEqual(status, 401)
        self.assertEqual(self.app.store.recent_parts(5), [])

    def test_bad_part_with_a_proposal_is_saved_then_posted_with_its_photo(self) -> None:
        self.laptop.predict_score = 0.9  # every picture at or above t_img: BAD
        self.laptop.predict_fields = {"defects": [SCRATCH]}  # the model names the defect class
        self.start()
        op = self.enter("insp-1")
        sid = self.mes.latest_session("insp-1")

        status, body = self.call("GET", "/api/parts/search?q=PN", cookie=op)
        self.assertEqual(status, 200)
        self.assertEqual(sorted(r["part_number"] for r in body["results"]), ["PN-A", "PN-B"])

        self.select(op, "PN-A")
        status, body = self.call("GET", "/api/part/criteria", cookie=op)
        self.assertEqual(status, 200)
        self.assertEqual(sorted(c["id"] for c in body["criteria"]), ["CR-1", "CR-2"])  # CR-3 inactive

        self.press_part(op)
        st = self.state(op)
        self.assertEqual((st["state"], st["verdict"]), ("VERDICT", "BAD"))
        self.assertIn("show_bad", self.outputs.history)
        self.assertNotIn("show_good", self.outputs.history)
        self.assertEqual(self.app.store.recent_parts(5), [])  # saved when the inspector acks
        self.assertEqual(self.app.sync_once(), {"sent": 0, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(self.mes.posts, [])

        status, body = self.call("POST", "/api/part/defects",
                                 {"criterion_id": "CR-2", "camera_id": "C2", "side": "BOTTOM",
                                  "note": "dent near the edge"}, op)
        self.assertEqual((status, body), (200, {"ok": True, "replaced": False}))

        status, body = self.call("POST", "/api/ack", {}, op)
        self.assertEqual(status, 200, body)
        st = self.state(op)
        self.assertEqual((st["state"], st["counters"]), ("IDLE", {"total": 1, "good": 0, "bad": 1}))

        parts = self.saved_parts()
        self.assertEqual(len(parts), 1)
        part_dir = pathlib.Path(parts[0]["path"])
        self.assertEqual(self.app.store.verify_part(part_dir), [])
        with open(part_dir / "part.json", encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual([(p["criterion_id"], p["proposed_by"], p["camera_id"], p["side"])
                          for p in doc["defect_proposals"]], [("CR-2", "insp-1", "C2", "BOTTOM")])
        session_id = parts[0]["session_id"]

        # Laptop copy: the part folder reaches the evidence destination with equal hashes.
        planned = planned_destination(part_dir, self.dest_root)
        self.assertEqual(self.app.evidence_sync_once(), {"synced": 1, "failed": 0, "pending": 0})
        self.assertTrue(os.path.isdir(planned))
        for picture in sorted(part_dir.glob("*.png")):
            self.assertEqual(sha256_of(picture), sha256_of(pathlib.Path(planned, picture.name)),
                             picture.name)
        self.assertEqual(len(list(pathlib.Path(planned).glob("*.png"))), 8)

        self.assertEqual(self.app.sync_once(), {"sent": 1, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(len(self.mes.posts), 1)
        post = self.posted()[0]
        rec = post["record"]
        self.assertEqual(rec["key"], session_id)  # Idempotency-Key is the session id
        self.assertEqual(rec["station_session"], sid)  # Station-Session is the MES session
        self.assertEqual(rec["auth"], f"Bearer {self.mes_token}")
        self.assertEqual(rec["status"], 201)
        f = post["fields"]
        self.assertEqual(f["verdict"], "BAD")
        self.assertEqual(f["visionSessionId"], session_id)
        self.assertEqual(f["operators"], "insp-1")
        self.assertEqual(f["modelVersion"], "v-a")
        self.assertEqual(json.loads(f["checklistResults"]), {"CR-1": "fail", "CR-2": "fail"})
        self.assertEqual(f["archiveRef"], planned)
        self.assertIsNotNone(datetime.datetime.fromisoformat(f["capturedAt"].replace("Z", "+00:00")))
        self.assertEqual(sorted(post["files"]), ["defectPhoto_CR-1", "defectPhoto_CR-2"])
        _name, proposed_photo = post["files"]["defectPhoto_CR-2"]
        self.assertTrue(proposed_photo.startswith(b"PIC C2 BOTTOM "), proposed_photo)  # the chosen picture
        _name, model_photo = post["files"]["defectPhoto_CR-1"]
        self.assertTrue(model_photo.startswith(b"PIC "), model_photo)  # a picture the model picked

        # Retrying the queue does not post again (the MES key is already accepted).
        self.assertEqual(self.app.sync_once(), {"sent": 0, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(len(self.mes.posts), 1)
        self.assertEqual(self.mes.accepted_by[session_id], "insp-1")

    def test_bad_by_score_alone_with_a_proposal_is_posted(self) -> None:
        # An anomaly model names no defect class. The inspector's proposal is the failed
        # criterion, so the BAD part is posted with that criterion's photo.
        self.laptop.predict_score = 0.9  # BAD by score, no defect classes named
        self.start()
        op = self.enter("insp-1")
        self.select(op, "PN-A")
        self.press_part(op)
        status, body = self.call("POST", "/api/part/defects",
                                 {"criterion_id": "CR-1", "camera_id": "C2", "side": "BOTTOM"}, op)
        self.assertEqual(status, 200, body)
        status, body = self.call("POST", "/api/ack", {}, op)
        self.assertEqual(status, 200, body)
        self.app.evidence_sync_once()
        self.assertEqual(self.app.sync_once(), {"sent": 1, "failed_permanent": 0, "retry_later": 0})

    def test_good_part_goes_green_after_the_save_and_takes_no_proposal(self) -> None:
        self.laptop.predict_score = 0.0
        self.start()
        op = self.enter("insp-1")
        self.select(op, "PN-A")
        self.press_part(op)

        st = self.state(op)
        self.assertEqual((st["state"], st["verdict"]), ("IDLE", "GOOD"))
        self.assertEqual(st["counters"], {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(self.outputs.saved_at["show_good"], [1])  # saved before the green light
        self.assertTrue(self.outputs.green)

        status, body = self.call("POST", "/api/part/defects",
                                 {"criterion_id": "CR-1", "camera_id": "C1", "side": "TOP"}, op)
        self.assertEqual(status, 400, body)
        self.assertIn("not supported", body["error"])

        self.assertEqual(self.app.evidence_sync_once()["synced"], 1)
        self.assertEqual(self.app.sync_once(), {"sent": 1, "failed_permanent": 0, "retry_later": 0})
        post = self.posted()[0]
        self.assertEqual(post["fields"]["verdict"], "GOOD")
        self.assertEqual(json.loads(post["fields"]["checklistResults"]), {"CR-1": "pass", "CR-2": "pass"})
        self.assertEqual(post["files"], {})  # no defect photo: a GOOD post carries no proposal
        self.assertEqual(post["record"]["station_session"], self.mes.latest_session("insp-1"))

    def test_laptop_down_mid_part_faults_and_nothing_is_posted_or_saved(self) -> None:
        self.laptop.predict_score = 0.0
        self.start()
        op = self.enter("insp-1")
        self.select(op, "PN-A")
        self.assertTrue(self.press(op, 1.0))  # TOP scored while the laptop is up
        self.assertEqual(self.state(op)["state"], "WAIT_FLIP")

        self._close_laptop()
        self.assertTrue(self.press(op, PAST_MIN_FLIP))  # BOTTOM: the laptop is gone

        st = self.state(op)
        self.assertEqual(st["state"], "FAULT")
        self.assertIn("show_fault", self.outputs.history)
        self.assertNotIn("show_good", self.outputs.history)
        self.assertFalse(self.outputs.green)
        self.assertEqual(self.app.store.recent_parts(5), [])
        self.assertEqual(self.app.outbox.pending_count(), 0)
        self.assertEqual(self.app.sync_once(), {"sent": 0, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(self.mes.posts, [])

    def test_session_ended_in_wait_flip_is_refused_and_never_posted_under_another_name(self) -> None:
        self.laptop.predict_score = 0.0
        self.start()
        op = self.enter("insp-1")
        sid1 = self.mes.latest_session("insp-1")
        self.select(op, "PN-A")
        self.assertTrue(self.press(op, 1.0))  # TOP: WAIT_FLIP
        self.assertEqual(self.state(op)["state"], "WAIT_FLIP")

        self.mes.end(sid1)  # the MES ends the session while the part waits for the flip
        status, body = self.call("GET", "/api/state", cookie=op)
        self.assertEqual((status, body.get("reason")), (401, "mes_session_ended"))
        status, body = self.call("POST", "/api/press", {}, op)
        self.assertEqual((status, body.get("reason")), (401, "mes_session_ended"))
        self.assertEqual(self.app.session.state.value, "WAIT_FLIP")  # the BOTTOM press never ran
        self.assertEqual(self.app.sync_once(), {"sent": 0, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(self.mes.posts, [])
        self.assertEqual(self.app.store.recent_parts(5), [])

        # The same inspector logs in again. The station finishes the flip (it does not fault):
        # the part is saved GOOD. The queued post still names the session the part was started
        # in (sid1, ended), so this fake MES refuses it and the item waits in the outbox. Nothing
        # is accepted under any name, and the part is not lost.
        op2 = self.enter("insp-1")
        sid2 = self.mes.latest_session("insp-1")
        self.assertNotEqual(sid1, sid2)
        self.assertEqual(self.state(op2)["state"], "WAIT_FLIP")
        self.assertTrue(self.press(op2, PAST_MIN_FLIP))
        st = self.state(op2)
        self.assertEqual((st["state"], st["verdict"]), ("IDLE", "GOOD"))
        self.assertEqual(len(self.app.store.recent_parts(5)), 1)
        self.assertEqual(self.app.sync_once(), {"sent": 0, "failed_permanent": 0, "retry_later": 1})
        self.assertEqual(self.mes.accepted, {})  # nothing accepted under any identity
        self.assertEqual(self.mes.accepted_by, {})
        self.assertEqual(self.app.outbox.pending_count(), 1)  # queued, not lost
        self.assertTrue({rec["station_session"] for rec in self.mes.posts} <= {sid1, sid2})
        self.assertEqual({rec["status"] for rec in self.mes.posts}, {401})


    def test_different_inspector_at_the_flip_faults_and_nothing_is_saved_or_queued(self) -> None:
        self.laptop.predict_score = 0.0
        self.start()
        op1 = self.enter("insp-1")
        self.select(op1, "PN-A")
        self.assertTrue(self.press(op1, 1.0))  # TOP by insp-1
        op2 = self.enter("insp-2")

        self.assertFalse(self.press(op2, PAST_MIN_FLIP))  # BOTTOM by insp-2
        st = self.state(op2)
        self.assertEqual(st["state"], "FAULT")
        self.assertIn("show_fault", self.outputs.history)
        self.assertNotIn("show_good", self.outputs.history)
        self.assertFalse(self.outputs.green)
        self.assertEqual(self.app.store.recent_parts(5), [])
        self.assertEqual(self.app.outbox.pending_count(), 0)
        self.assertEqual(self.app.sync_once(), {"sent": 0, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(self.mes.posts, [])

    def test_review_is_never_green_or_posted_until_a_supervisor_resolves_it(self) -> None:
        self.laptop.predict_score = 0.6  # between t_low and t_img on every picture: REVIEW
        self.start()
        op = self.enter("insp-1")
        sup = self.enter("sup-1", ("supervisor",))
        self.select(op, "PN-A")
        self.press_part(op)

        st = self.state(op)
        self.assertEqual((st["state"], st["verdict"]), ("SUPERVISOR_REVIEW", "REVIEW"))
        self.assertIn("show_review", self.outputs.history)
        self.assertFalse(self.outputs.green)
        self.assertNotIn("show_good", self.outputs.history)
        self.assertEqual(self.app.store.recent_parts(5), [])
        self.assertEqual(self.app.sync_once(), {"sent": 0, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(self.mes.posts, [])

        status, body = self.call("POST", "/api/resolve", {"verdict": "GOOD", "reason": "by hand"}, op)
        self.assertEqual(status, 403, body)  # an operator may not resolve a REVIEW
        self.assertEqual(self.state(op)["state"], "SUPERVISOR_REVIEW")

        status, body = self.call("POST", "/api/resolve", {"verdict": "GOOD", "reason": "checked by hand"},
                                 sup)
        self.assertEqual(status, 200, body)
        self.assertEqual(self.state(sup)["state"], "IDLE")
        self.assertEqual(self.outputs.saved_at["show_good"], [1])  # saved before the green light
        self.assertEqual(self.app.sync_once(), {"sent": 1, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(self.posted()[0]["fields"]["verdict"], "GOOD")
        self.assertEqual(self.posted()[0]["record"]["station_session"], self.mes.latest_session("insp-1"))

    def test_no_token_reaches_a_request_body_a_reply_the_event_log_or_stderr(self) -> None:
        self.laptop.predict_score = 0.0
        self.start()
        status, body = self.call("POST", "/api/enter", {"ticket": "tkt-not-issued"})
        self.assertEqual(status, 401)

        op = self.enter("insp-1")
        self.select(op, "PN-A")
        self.press_part(op)
        self.app.evidence_sync_once()
        self.assertEqual(self.app.sync_once()["sent"], 1)

        # A laptop failure: its message is sent to the station log and must not carry the token.
        self.laptop.predict_status = 503
        self.laptop.predict_error = "GPU out of memory while scoring"
        self.select(op, "PN-A")
        self.assertTrue(self.press(op, 1.0))
        self.assertEqual(self.state(op)["state"], "FAULT")

        forbidden = [self.mes_token.encode(), self.laptop_token.encode()]
        blobs = list(self.wire) + list(self.mes.wire) + [self.err.getvalue().encode("utf-8")]
        for rec in self.mes.posts:
            blobs.append(rec["body"])
        for path in self.all_station_files():  # data dir, outbox, event log, queue, laptop store
            with open(path, "rb") as fh:
                blobs.append(fh.read())
        with contextlib.closing(sqlite3.connect(str(self.app.store.db_path))) as conn:
            for (payload,) in conn.execute("SELECT payload_json FROM events"):
                blobs.append(payload.encode("utf-8"))
        for blob in blobs:
            for token in forbidden:
                self.assertNotIn(token, blob)
        self.assertGreater(len(self.mes.posts), 0)  # the scan saw real traffic


if __name__ == "__main__":
    unittest.main()
