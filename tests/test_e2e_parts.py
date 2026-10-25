"""End-to-end: the multi-part (parts.mode 'mes') flow over the real UI HTTP server.

The MES is a fake http.server on 127.0.0.1 that answers the part search, part lookup,
criteria and inspection-event routes. Cameras are FakeCameraSet frames. The model is a
real approved golden model in a temporary ModelStore (cv2 decoding is patched out, as in
tests/test_app.py), except the REVIEW test, which registers a marker loader so the score
can be set exactly. Nothing touches the network beyond 127.0.0.1. The tokens are made-up
strings generated inside this file and are passed to build_station() directly: no
environment variable is read or written.
"""
from __future__ import annotations

import contextlib
import copy
import email.parser
import email.policy
import http.client
import http.server
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

import numpy as np

from station import auth as station_auth
from station import modelstore
from station.app import build_station
from station.auth import hash_password
from station.cameras import FakeCameraSet
from station.contracts import Frame, Side
from station.golden import GoldenBuilder
from station.inference import MarkerModel
from station.modelstore import ModelStore, safe_part
from station.outputs import RecordingOutputs

OP = ("op-parts-e2e", "op-parts-pass-1")
SUP = ("sup-parts-e2e", "sup-parts-pass-1")
TOKEN_ENV = "VISION_TEST_PARTS_E2E_TOKEN"  # a name only: the token is passed in, not read
PAST_MIN_FLIP = 2.5  # past the default min_flip_s (2.0 s) between the TOP and BOTTOM press
GOOD_MARKER = b"SCORE:0.05"

PARTS = {
    "PN-A": {"part_number": "PN-A", "description": "Liner left", "part_type": "FG",
             "lifecycle": "ACTIVE", "unit_cost": 12.5, "bom": ["x"], "planning": {"wo": 3}},
    "PN-B": {"part_number": "PN-B", "description": "Liner right", "part_type": "FG",
             "lifecycle": "ACTIVE", "unit_cost": 13.0, "bom": ["y"], "planning": {"wo": 1}},
}
CRITERIA = {
    "PN-A": [
        {"id": "CR-1", "criteria_name": "Scratch", "check_type": "visual", "active": True},
        {"id": "CR-2", "criteria_name": "Dent", "check_type": "visual", "active": True},
    ],
    "PN-B": [
        {"id": "CR-7", "criteria_name": "Crack", "check_type": "visual", "active": True},
    ],
}


# ------------------------------------------------------------------ fake MES

class _FakeMesHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - keep test output quiet
        pass

    def _send(self, status: int, payload) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {self.server.token}"

    def do_GET(self) -> None:
        srv = self.server
        url = urllib.parse.urlsplit(self.path)
        with srv.lock:
            srv.gets.append((url.path, self.headers.get("Authorization")))
        if not self._authorized():
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
        return self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        srv = self.server
        key = self.headers.get("Idempotency-Key")
        with srv.lock:
            record = {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "key": key,
                "ctype": self.headers.get("Content-Type"),
                "body": body,
                "status": None,
            }
            srv.posts.append(record)
            if not self._authorized():
                status, payload = 401, {"error": "unauthorized"}
            elif key in srv.accepted:
                status, payload = 200, {"id": srv.accepted[key]}  # replay of a stored key
            else:
                srv.next_id += 1
                srv.accepted[key] = srv.next_id
                status, payload = 201, {"id": srv.next_id}
            record["status"] = status
        self._send(status, payload)


class FakeMes(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, token: str) -> None:
        super().__init__(("127.0.0.1", 0), _FakeMesHandler)
        self.token = token
        self.lock = threading.Lock()
        self.parts = copy.deepcopy(PARTS)
        self.criteria = copy.deepcopy(CRITERIA)
        self.gets: list[tuple[str, str]] = []
        self.posts: list[dict] = []
        self.accepted: dict[str, int] = {}
        self.next_id = 0
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.1},
                                       daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self.thread.join(5)


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


# ------------------------------------------------------------------ fakes for the station

class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class SpyOutputs(RecordingOutputs):
    """Records how many parts were already saved each time a GOOD light is lit."""

    def __init__(self) -> None:
        super().__init__()
        self.store = None
        self.saved_at_green: list[int] = []

    def show_good(self) -> None:
        if self.store is not None:
            self.saved_at_green.append(len(self.store.recent_parts(50)))
        super().show_good()


class ClosedCameras(FakeCameraSet):
    """A camera set whose device is closed, as OpenCvCameraSet reports after a failed open."""

    @property
    def is_open(self) -> bool:
        return False

    @property
    def last_open_error(self):
        return "C1: could not open device 0"


def good_bytes(camera_id: str, side: Side, n: int) -> bytes:
    return bytes([100])  # the golden decoder reads one grey value: 100 is the good value


def marker_loader(info, decode):
    """Test loader for a 'marker' model type: lets a test set each image's score exactly."""
    return MarkerModel(version=info.version)


# ------------------------------------------------------------------ tests

class PartsFlowE2E(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="station-parts-e2e-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

        self.token = "parts-e2e-token-" + secrets.token_hex(8)  # made up, per test run
        self.mes = FakeMes(self.token)
        self.addCleanup(self.mes.close)

        self.users_file = os.path.join(self.tmp, "users.json")
        with open(self.users_file, "w", encoding="utf-8") as fh:
            json.dump({
                OP[0]: {"password_hash": hash_password(OP[1]), "role": "operator"},
                SUP[0]: {"password_hash": hash_password(SUP[1]), "role": "supervisor"},
            }, fh)

        self.models_dir = os.path.join(self.tmp, "models")
        os.makedirs(self.models_dir)

        # A real golden model file, built from good frames of all four cameras.
        def decode(frame: Frame):
            return np.full((8, 8), float(frame.data[0]))

        patcher = mock.patch("station.golden.cv2_decoder", return_value=decode)
        patcher.start()
        self.addCleanup(patcher.stop)
        builder = GoldenBuilder(decode)
        for camera_id in ("C1", "C2", "C3", "C4"):
            for side in (Side.TOP, Side.BOTTOM):
                for value in (100, 102, 98):
                    builder.add(Frame(camera_id=camera_id, side=side, seq=0,
                                      timestamp_utc="2026-10-24T08:00:00Z", data=bytes([value])))
        self.golden_file = os.path.join(self.tmp, "golden.npz")
        builder.build("file-version").save(self.golden_file)

        self.clock = FakeClock()
        self.outputs = SpyOutputs()
        self.markers: dict[tuple[str, str], bytes] = {}
        self.app = None
        self.port = None

    # ---- helpers

    def approve(self, part_number: str, version: str, model_file: str,
                model_type: str = "golden") -> None:
        store = ModelStore(self.models_dir)
        store.add_candidate(part_number, version, pathlib.Path(model_file),
                            {"type": model_type, "camera_ids": ["C1", "C2", "C3", "C4"]},
                            "2026-10-24T08:00:00Z")
        store.approve(part_number, version, by="quality.engineer", reason="approved for test",
                      now_utc="2026-10-24T08:01:00Z")

    def approve_golden(self, part_number: str, version: str) -> None:
        self.approve(part_number, version, self.golden_file)

    def config(self) -> dict:
        return {
            "site": "SITE1", "line": "L1", "station_id": "VIS-PARTS-1",
            "lot": "LOT-7", "operator": OP[0], "supervisor": SUP[0], "material_handler": "mh1",
            "thresholds": {"t_low": 0.5, "t_img": 0.8},
            "data_dir": os.path.join(self.tmp, "data"),
            "users_file": self.users_file,
            "ui": {"host": "127.0.0.1", "port": 0, "session_timeout_s": 900},
            "mes": {
                "base_url": self.mes.base_url,
                "token_env": TOKEN_ENV,
                "criteria_ids": ["CR-1", "CR-2"],
                "class_to_criterion": {"scratch": "CR-1"},
                "default_criterion_id": None,
                "sync_interval_s": 3600,  # the tests call sync_once() themselves
                "default_bad_disposition": "quarantine",
            },
            "cameras": {"kind": "fake"},
            "model": {"kind": "marker", "version": "unused-in-parts-mode"},
            "outputs": {"kind": "console"},
            "flip_timeout_s": 60,
            "parts": {"mode": "mes", "models_dir": self.models_dir},
        }

    def _frame_bytes(self, camera_id: str, side: Side, n: int) -> bytes:
        marker = self.markers.get((camera_id, side.value), GOOD_MARKER)
        return marker + b" |" + camera_id.encode() + b"|" + side.value.encode() + b"|" + str(n).encode()

    def start(self, cameras=None) -> None:
        """Build the station with the parts config, start the UI server and remember its port."""
        self.outputs.store = None
        self.app = build_station(
            self.config(),
            cameras=cameras if cameras is not None else FakeCameraSet(frame_bytes=self._frame_bytes),
            outputs=self.outputs,
            clock=self.clock,
            token=self.token,
        )
        self.outputs.store = self.app.store
        self.addCleanup(self.app.stop)
        self.app.start()
        self.port = self.app.server.server_address[1]

    def req(self, method: str, path: str, body=None, token=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if token:
            headers["Cookie"] = f"station_session={token}"
        try:
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            cookie = resp.getheader("Set-Cookie")
        finally:
            conn.close()
        return resp.status, json.loads(raw.decode("utf-8") or "{}"), cookie

    def login(self, user) -> str:
        status, body, cookie = self.req("POST", "/api/login", {"username": user[0], "password": user[1]})
        self.assertEqual(status, 200, body)
        return cookie.split(";", 1)[0].split("=", 1)[1]

    def press(self, token: str):
        self.clock.now += 1.0  # past the 0.3 s debounce
        return self.req("POST", "/api/press", token=token)

    def press_part(self, token: str) -> None:
        """TOP press, flip gap, BOTTOM press. Both must be accepted."""
        for gap in (1.0, PAST_MIN_FLIP):
            self.clock.now += gap
            status, body, _ = self.req("POST", "/api/press", token=token)
            self.assertEqual((status, body), (200, {"accepted": True}))

    def state(self, token: str) -> dict:
        status, body, _ = self.req("GET", "/api/state", token=token)
        self.assertEqual(status, 200, body)
        return body

    def events(self, kind: str) -> list[dict]:
        with contextlib.closing(sqlite3.connect(str(self.app.store.db_path))) as conn:
            rows = conn.execute("SELECT payload_json FROM events WHERE kind = ? ORDER BY id",
                                (kind,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def selections(self) -> list[dict]:
        """The app-level 'part_selected' events (they carry 'by'). The session logs its own too."""
        return [e for e in self.events("part_selected") if "by" in e]

    def saved_parts(self) -> list[dict]:
        """Saved parts, oldest first."""
        return list(reversed(self.app.store.recent_parts(50)))

    def posted(self) -> list[dict]:
        out = []
        for rec in self.mes.posts:
            fields, files = parse_multipart(rec["ctype"], rec["body"])
            out.append({"record": rec, "fields": fields, "files": files})
        return out

    def assert_no_token_on_disk(self) -> None:
        for root in (os.path.join(self.tmp, "data"), self.models_dir):
            for dirpath, _dirs, files in os.walk(root):
                for name in files:
                    with open(os.path.join(dirpath, name), "rb") as fh:
                        self.assertNotIn(self.token.encode(), fh.read(), name)

    # ---- tests

    def test_full_flow_two_parts_each_with_its_own_model_and_criteria(self) -> None:
        self.approve_golden("PN-A", "2026-10-24-a")
        self.approve_golden("PN-B", "2026-10-24-b")
        self.start(FakeCameraSet(frame_bytes=good_bytes))
        op = self.login(OP)

        # Identity fields only from the search.
        status, body, _ = self.req("GET", "/api/parts/search?q=PN", token=op)
        self.assertEqual(status, 200)
        self.assertEqual(len(body["results"]), 2)
        for row in body["results"]:
            self.assertEqual(set(row), {"part_number", "description", "part_type", "lifecycle"})
        self.assertNotIn("unit_cost", json.dumps(body))

        # A press with no part is ignored and changes nothing.
        status, body, _ = self.press(op)
        self.assertEqual((status, body), (200, {"accepted": False}))
        self.assertEqual(self.state(op)["state"], "IDLE")
        self.assertEqual(self.app.store.recent_parts(5), [])

        # Select part A: the snapshot shows it inspectable with its approved version.
        status, body, _ = self.req("POST", "/api/part/select", {"part_number": "PN-A"}, token=op)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["model_version"], "2026-10-24-a")
        self.assertIs(body["can_inspect"], True)
        st = self.state(op)
        self.assertIs(st["can_inspect"], True)
        self.assertEqual(st["model_version"], "2026-10-24-a")
        self.assertEqual(st["part_number"], "PN-A")

        # A full GOOD inspection of part A.
        self.press_part(op)
        st = self.state(op)
        self.assertEqual(st["state"], "IDLE")
        self.assertEqual(st["verdict"], "GOOD")
        self.assertEqual(st["counters"], {"total": 1, "good": 1, "bad": 0})

        # The record was on disk before the green light came on.
        self.assertEqual(self.outputs.saved_at_green, [1])
        self.assertTrue(self.outputs.green)

        parts = self.saved_parts()
        self.assertEqual(len(parts), 1)
        part_a_dir = pathlib.Path(parts[0]["path"])
        self.assertEqual(len([p for p in part_a_dir.iterdir() if p.suffix == ".png"]), 8)
        self.assertEqual(self.app.store.verify_part(part_a_dir), [])
        with open(part_a_dir / "part.json", encoding="utf-8") as fh:
            doc_a = json.load(fh)
        self.assertEqual(doc_a["part_number"], "PN-A")
        self.assertEqual(doc_a["model_version"], "2026-10-24-a")
        session_a = parts[0]["session_id"]

        self.assertEqual(self.app.sync_once(), {"sent": 1, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(len(self.mes.posts), 1)
        post_a = self.posted()[0]
        self.assertEqual(post_a["record"]["key"], session_a)
        self.assertEqual(post_a["record"]["auth"], f"Bearer {self.token}")
        self.assertEqual(post_a["record"]["status"], 201)
        self.assertEqual(post_a["fields"]["verdict"], "GOOD")
        self.assertEqual(post_a["fields"]["visionSessionId"], session_a)
        self.assertEqual(json.loads(post_a["fields"]["checklistResults"]),
                         {"CR-1": "pass", "CR-2": "pass"})
        self.assertEqual(post_a["files"], {})

        # Switch to part B (its own approved version) and inspect it.
        status, body, _ = self.req("POST", "/api/part/select", {"part_number": "PN-B"}, token=op)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["model_version"], "2026-10-24-b")
        self.assertIs(body["can_inspect"], True)
        self.press_part(op)
        self.assertEqual(self.state(op)["counters"], {"total": 2, "good": 2, "bad": 0})
        self.assertEqual(self.outputs.saved_at_green, [1, 2])

        parts = self.saved_parts()
        self.assertEqual(len(parts), 2)
        self.assertNotEqual(parts[0]["session_id"], parts[1]["session_id"])
        part_b_dir = pathlib.Path(parts[1]["path"])
        with open(part_b_dir / "part.json", encoding="utf-8") as fh:
            doc_b = json.load(fh)
        self.assertEqual(doc_b["part_number"], "PN-B")
        self.assertEqual(doc_b["model_version"], "2026-10-24-b")

        self.assertEqual(self.app.sync_once(), {"sent": 1, "failed_permanent": 0, "retry_later": 0})
        post_b = self.posted()[1]
        self.assertEqual(post_b["record"]["key"], parts[1]["session_id"])
        self.assertEqual(json.loads(post_b["fields"]["checklistResults"]), {"CR-7": "pass"})

        self.assertEqual([e["part_number"] for e in self.selections()], ["PN-A", "PN-B"])
        self.assertEqual(self.gets_for_criteria(), ["PN-A", "PN-B"])
        self.assert_no_token_on_disk()

    def gets_for_criteria(self) -> list[str]:
        return [urllib.parse.unquote(path[len("/api/machine/criteria/"):])
                for path, _auth in self.mes.gets if path.startswith("/api/machine/criteria/")]

    def test_part_without_an_approved_model_is_blocked_and_press_stays_refused(self) -> None:
        self.start(FakeCameraSet(frame_bytes=good_bytes))  # no model is approved for PN-A
        op = self.login(OP)

        status, body, _ = self.req("POST", "/api/part/select", {"part_number": "PN-A"}, token=op)
        self.assertEqual(status, 200, body)  # a block is not an error
        self.assertIs(body["can_inspect"], False)
        self.assertEqual(body["reason"], "No approved model for part PN-A. Ask Quality to approve one.")
        self.assertNotIn("model_version", body)

        st = self.state(op)
        self.assertIs(st["can_inspect"], False)
        self.assertEqual(st["block_reason"], body["reason"])

        status, body, _ = self.press(op)
        self.assertEqual((status, body), (200, {"accepted": False}))
        self.assertEqual(self.state(op)["state"], "IDLE")
        self.assertEqual(self.state(op)["counters"], {"total": 0, "good": 0, "bad": 0})
        self.assertEqual(self.app.store.recent_parts(5), [])
        self.assertEqual(self.mes.posts, [])

    def test_tampered_approved_model_blocks_the_part_with_no_fallback(self) -> None:
        self.approve_golden("PN-A", "2026-10-24-a")
        self.approve_golden("PN-A", "2026-10-24-b")  # the current approved version
        tampered = os.path.join(self.models_dir, safe_part("PN-A"), "2026-10-24-b", "golden.npz")
        with open(tampered, "ab") as fh:
            fh.write(b"tampered")
        self.start(FakeCameraSet(frame_bytes=good_bytes))
        op = self.login(OP)

        status, body, _ = self.req("POST", "/api/part/select", {"part_number": "PN-A"}, token=op)
        self.assertEqual(status, 200, body)
        self.assertIs(body["can_inspect"], False)
        self.assertIn("is not usable", body["reason"])
        self.assertIn("sha256", body["reason"])
        self.assertNotIn("model_version", body)  # the older 2026-10-24-a is not used instead
        self.assertNotIn("2026-10-24-a", json.dumps(body))

        st = self.state(op)
        self.assertIs(st["can_inspect"], False)
        self.assertFalse(st.get("model_version"))
        self.assertEqual(self.press(op)[1], {"accepted": False})
        self.assertEqual(self.app.store.recent_parts(5), [])

    def test_select_is_refused_with_409_while_a_part_waits_for_the_flip(self) -> None:
        self.approve_golden("PN-A", "2026-10-24-a")
        self.approve_golden("PN-B", "2026-10-24-b")
        self.start(FakeCameraSet(frame_bytes=good_bytes))
        op = self.login(OP)

        status, body, _ = self.req("POST", "/api/part/select", {"part_number": "PN-A"}, token=op)
        self.assertEqual(status, 200, body)
        self.assertEqual(self.press(op)[1], {"accepted": True})  # TOP: now WAIT_FLIP
        self.assertEqual(self.state(op)["state"], "WAIT_FLIP")

        status, body, _ = self.req("POST", "/api/part/select", {"part_number": "PN-B"}, token=op)
        self.assertEqual(status, 409, body)
        self.assertIn("error", body)

        st = self.state(op)
        self.assertEqual(st["state"], "WAIT_FLIP")
        self.assertEqual(st["part_number"], "PN-A")
        self.assertEqual(st["model_version"], "2026-10-24-a")
        self.assertEqual([e["part_number"] for e in self.selections()], ["PN-A"])

    def test_viewer_cannot_select_and_no_cookie_is_401(self) -> None:
        # The station has no 'viewer' role (auth.ROLES is operator, supervisor, admin). A
        # role that may not select is simulated by removing select_part from the operator's
        # policy, as tests/test_ui.py does.
        self.approve_golden("PN-A", "2026-10-24-a")
        self.start(FakeCameraSet(frame_bytes=good_bytes))
        op = self.login(OP)

        with mock.patch.dict(station_auth._POLICY, {"operator": frozenset({"press", "ack"})}):
            status, body, _ = self.req("POST", "/api/part/select", {"part_number": "PN-A"}, token=op)
        self.assertEqual(status, 403, body)
        self.assertEqual(self.app.session.part_number, "")
        self.assertEqual(self.selections(), [])

        # No session cookie: 401 for the part routes and nothing reaches the MES.
        status, _, _ = self.req("POST", "/api/part/select", {"part_number": "PN-A"})
        self.assertEqual(status, 401)
        status, _, _ = self.req("GET", "/api/parts/search?q=PN")
        self.assertEqual(status, 401)
        status, _, _ = self.req("GET", "/api/part")
        self.assertEqual(status, 401)
        self.assertEqual(self.app.session.part_number, "")
        self.assertEqual(self.selections(), [])
        self.assertEqual([path for path, _ in self.mes.gets if "/criteria/" in path], [])

    def test_review_outcome_is_never_green_and_never_posted_in_parts_mode(self) -> None:
        # The marker loader lets this part score 0.6 on one BOTTOM image: between t_low (0.5)
        # and t_img (0.8), so the verdict is REVIEW.
        self.markers[("C3", "BOTTOM")] = b"SCORE:0.6"
        marker_file = os.path.join(self.tmp, "marker.bin")
        with open(marker_file, "wb") as fh:
            fh.write(b"marker model: read by the test loader only")
        with mock.patch.dict(modelstore.LOADERS, {"marker": marker_loader}):
            self.approve("PN-A", "2026-10-24-m", marker_file, model_type="marker")
            self.start(FakeCameraSet(frame_bytes=self._frame_bytes))
            op = self.login(OP)
            status, body, _ = self.req("POST", "/api/part/select", {"part_number": "PN-A"}, token=op)
            self.assertEqual((status, body["can_inspect"]), (200, True), body)

            self.press_part(op)

        st = self.state(op)
        self.assertEqual(st["state"], "SUPERVISOR_REVIEW")
        self.assertEqual(st["verdict"], "REVIEW")
        self.assertEqual(st["counters"]["total"], 0)
        self.assertFalse(self.outputs.green)
        self.assertTrue(self.outputs.amber)
        self.assertNotIn("show_good", self.outputs.history)
        self.assertEqual(self.app.store.recent_parts(5), [])  # nothing saved yet

        self.assertEqual(self.app.sync_once(), {"sent": 0, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(self.app.outbox.pending_count(), 0)
        self.assertEqual(self.mes.posts, [])

    def test_camera_closed_shows_open_false_in_the_snapshot(self) -> None:
        self.start(ClosedCameras(frame_bytes=good_bytes))
        op = self.login(OP)
        st = self.state(op)
        self.assertEqual(st["camera"], {"open": False, "last_error": "C1: could not open device 0"})
        self.assertEqual(st["parts_mode"], "mes")


if __name__ == "__main__":
    unittest.main()
