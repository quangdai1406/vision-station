"""End-to-end: real session, store, outbox and MES client, driven through the UI HTTP API.

The MES is a fake http.server on 127.0.0.1 that records every request and can
answer 201, 500 or 422. Cameras are FakeCameraSet frames whose bytes carry score
markers, read by the real MarkerModel. Nothing here touches the network beyond
127.0.0.1.
"""
from __future__ import annotations

import email.parser
import email.policy
import http.client
import http.server
import json
import os
import pathlib
import shutil
import tempfile
import threading
import time
import unittest

from station.app import build_station
from station.auth import hash_password
from station.cameras import FakeCameraSet
from station.contracts import Side
from station.inference import MarkerModel
from station.outputs import RecordingOutputs

TOKEN_ENV = "VISION_TEST_E2E_TOKEN"
TOKEN = "e2e-test-token-not-a-real-secret"
OPERATOR = ("operator1", "operator-pass-1")
SUPERVISOR = ("supervisor1", "supervisor-pass-1")
GOOD_MARKER = b"SCORE:0.05"


# ------------------------------------------------------------------ fake MES

class _MesHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - no request logging in tests
        pass

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
            srv.requests.append(record)
            if srv.mode == 201 and key in srv.accepted:
                status, payload = 200, {"id": srv.accepted[key]}  # replay of a stored key
            elif srv.mode == 201:
                srv.next_id += 1
                srv.accepted[key] = srv.next_id
                status, payload = 201, {"id": srv.next_id}
            elif srv.mode == 422:
                status, payload = 422, {"error": "verdict disagrees with checklistResults"}
            else:
                status, payload = 500, {"error": "database unavailable"}
            record["status"] = status
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class FakeMes(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _MesHandler)
        self.lock = threading.Lock()
        self.mode = 201
        self.requests: list[dict] = []
        self.accepted: dict[str, int] = {}  # Idempotency-Key -> event id
        self.next_id = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


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


# ------------------------------------------------------------------ UI client

class Ui:
    def __init__(self, app) -> None:
        self.host, self.port = app.server.server_address[:2]

    def req(self, method: str, path: str, body=None, token=None):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
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

    def login(self, username: str, password: str) -> str:
        status, body, cookie = self.req("POST", "/api/login",
                                        {"username": username, "password": password})
        if status != 200 or not cookie:
            raise AssertionError(f"login failed: {status} {body}")
        return cookie.split(";", 1)[0].split("=", 1)[1]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ------------------------------------------------------------------ tests

class StationE2E(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="station-e2e-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

        self.mes = FakeMes()
        threading.Thread(target=self.mes.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
        self.addCleanup(self._stop_mes)

        os.environ[TOKEN_ENV] = TOKEN
        self.addCleanup(os.environ.pop, TOKEN_ENV, None)

        self.users_file = os.path.join(self.tmp, "users.json")
        with open(self.users_file, "w", encoding="utf-8") as fh:
            json.dump({
                OPERATOR[0]: {"password_hash": hash_password(OPERATOR[1]), "role": "operator"},
                SUPERVISOR[0]: {"password_hash": hash_password(SUPERVISOR[1]), "role": "supervisor"},
            }, fh)

        self.markers: dict[tuple[str, str], bytes] = {}
        self.clock = FakeClock()
        self.outputs = RecordingOutputs()
        self.t0 = time.time()
        self.app = self.start_app()
        self.ui = Ui(self.app)
        self.op = self.ui.login(*OPERATOR)
        self.sup = self.ui.login(*SUPERVISOR)

    def _stop_mes(self) -> None:
        self.mes.shutdown()
        self.mes.server_close()

    def config(self) -> dict:
        return {
            "site": "SITE1", "line": "L1", "station_id": "VIS-1", "part_number": "PN-100",
            "lot": "LOT-7", "operator": "operator1", "supervisor": "supervisor1",
            "material_handler": "mh1",
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
            "model": {"kind": "marker", "version": "e2e-model-1"},
            "outputs": {"kind": "console"},
            "flip_timeout_s": 60,
        }

    def _frame_bytes(self, camera_id: str, side: Side, n: int) -> bytes:
        marker = self.markers.get((camera_id, side.value), GOOD_MARKER)
        return marker + b" |" + camera_id.encode() + b"|" + side.value.encode() + b"|" + str(n).encode()

    def start_app(self):
        app = build_station(
            self.config(),
            cameras=FakeCameraSet(frame_bytes=self._frame_bytes),
            model=MarkerModel(version="e2e-model-1"),
            outputs=self.outputs,
            clock=self.clock,
        )
        app.start()
        self.addCleanup(app.stop)
        return app

    # helpers
    def press(self, token=None, ui=None) -> None:
        self.clock.now += 1.0  # past the 0.3 s debounce
        status, body, _ = (ui or self.ui).req("POST", "/api/press", token=token or self.op)
        self.assertEqual((status, body), (200, {"accepted": True}))

    def state(self, token=None) -> dict:
        status, body, _ = self.ui.req("GET", "/api/state", token=token or self.op)
        self.assertEqual(status, 200, body)
        return body

    def saved_part(self, app=None):
        row = (app or self.app).store.recent_parts(1)[0]
        return row["session_id"], pathlib.Path(row["path"])

    def posted(self) -> list[dict]:
        out = []
        for rec in self.mes.requests:
            fields, files = parse_multipart(rec["ctype"], rec["body"])
            out.append({"record": rec, "fields": fields, "files": files})
        return out

    def assert_no_token_on_disk(self) -> None:
        for root, _dirs, files in os.walk(os.path.join(self.tmp, "data")):
            for name in files:
                with open(os.path.join(root, name), "rb") as fh:
                    self.assertNotIn(TOKEN.encode(), fh.read(), name)

    @staticmethod
    def pngs(part_dir: pathlib.Path) -> list[str]:
        return sorted(p.name for p in part_dir.iterdir() if p.suffix == ".png")

    # 1. GOOD
    def test_good_part_is_saved_and_posted_once(self) -> None:
        self.press()
        self.press()

        st = self.state()
        self.assertEqual(st["state"], "IDLE")
        self.assertEqual(st["verdict"], "GOOD")
        self.assertEqual(st["counters"], {"total": 1, "good": 1, "bad": 0})
        self.assertTrue(self.outputs.green)
        self.assertFalse(self.outputs.red)

        session_id, part_dir = self.saved_part()
        self.assertEqual(len(self.pngs(part_dir)), 8)
        self.assertEqual(self.app.store.verify_part(part_dir), [])

        self.assertEqual(self.app.sync_once(), {"sent": 1, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(len(self.mes.requests), 1)
        post = self.posted()[0]
        self.assertEqual(post["record"]["key"], session_id)
        self.assertEqual(post["record"]["auth"], f"Bearer {TOKEN}")
        self.assertEqual(post["record"]["status"], 201)
        self.assertEqual(post["fields"]["verdict"], "GOOD")
        self.assertEqual(post["fields"]["visionSessionId"], session_id)
        self.assertEqual(post["fields"]["archiveRef"], str(part_dir))
        self.assertEqual(json.loads(post["fields"]["checklistResults"]), {"CR-1": "pass", "CR-2": "pass"})
        self.assertEqual(post["files"], {})  # no defect photo for a GOOD part
        self.assert_no_token_on_disk()

    # 2. BAD
    def test_bad_part_waits_for_ack_then_posts_with_one_defect_photo(self) -> None:
        self.markers[("C2", "TOP")] = b"SCORE:0.92;CLASS:scratch;BOX:10,20,30,40"
        self.press()
        self.press()

        st = self.state()
        self.assertEqual(st["state"], "VERDICT")
        self.assertEqual(st["verdict"], "BAD")
        self.assertEqual(st["counters"]["total"], 0)  # not counted until acknowledged
        self.assertTrue(self.outputs.red)
        self.assertTrue(self.outputs.alarm)
        self.assertFalse(self.outputs.green)

        status, body, _ = self.ui.req("POST", "/api/ack", token=self.op)
        self.assertEqual((status, body), (200, {"ok": True}))
        st = self.state()
        self.assertEqual(st["state"], "IDLE")
        self.assertEqual(st["counters"], {"total": 1, "good": 0, "bad": 1})
        self.assertTrue(self.outputs.red)
        self.assertFalse(self.outputs.alarm)

        session_id, part_dir = self.saved_part()
        self.assertEqual(len(self.pngs(part_dir)), 8)
        self.assertEqual(self.app.store.verify_part(part_dir), [])

        self.app.sync_once()
        self.assertEqual(len(self.mes.requests), 1)
        post = self.posted()[0]
        f = post["fields"]
        self.assertEqual(post["record"]["key"], session_id)
        self.assertEqual(f["verdict"], "BAD")
        self.assertEqual(f["rejectedQuantity"], "1")
        self.assertEqual(f["rejectionReason"], "scratch")
        self.assertEqual(f["disposition"], "quarantine")
        self.assertEqual(json.loads(f["checklistResults"]), {"CR-1": "fail", "CR-2": "pass"})
        self.assertEqual(list(post["files"]), ["defectPhoto_CR-1"])
        filename, data = post["files"]["defectPhoto_CR-1"]
        self.assertEqual(filename, "C2_TOP.png")
        self.assertEqual(data, b"SCORE:0.92;CLASS:scratch;BOX:10,20,30,40 |C2|TOP|0")

    # 3. REVIEW
    def test_review_part_is_never_green_until_a_supervisor_resolves_it(self) -> None:
        self.markers[("C3", "BOTTOM")] = b"SCORE:0.6"  # between t_low 0.5 and t_img 0.8
        self.press()
        self.press()

        st = self.state()
        self.assertEqual(st["state"], "SUPERVISOR_REVIEW")
        self.assertEqual(st["verdict"], "REVIEW")
        self.assertEqual(st["counters"]["total"], 0)
        self.assertFalse(self.outputs.green)
        self.assertTrue(self.outputs.amber)
        self.assertNotIn("show_good", self.outputs.history)

        status, body, _ = self.ui.req("POST", "/api/resolve",
                                      {"verdict": "GOOD", "reason": "looks fine"}, token=self.op)
        self.assertEqual(status, 403, body)
        self.assertEqual(self.state()["state"], "SUPERVISOR_REVIEW")
        self.assertFalse(self.outputs.green)

        status, body, _ = self.ui.req("POST", "/api/resolve",
                                      {"verdict": "GOOD", "reason": "checked by hand"}, token=self.sup)
        self.assertEqual((status, body), (200, {"ok": True}))
        st = self.state()
        self.assertEqual(st["state"], "IDLE")
        self.assertEqual(st["verdict"], "GOOD")
        self.assertEqual(st["counters"], {"total": 1, "good": 1, "bad": 0})
        self.assertTrue(self.outputs.green)

        session_id, part_dir = self.saved_part()
        self.assertEqual(len(self.pngs(part_dir)), 8)
        self.assertEqual(self.app.store.verify_part(part_dir), [])

        self.app.sync_once()
        self.assertEqual(len(self.mes.requests), 1)  # posted once, after resolution
        post = self.posted()[0]
        f = post["fields"]
        self.assertEqual(f["verdict"], "GOOD")
        self.assertEqual(f["overrideOriginalVerdict"], "REVIEW")
        self.assertEqual(f["overrideReason"], "checked by hand")
        self.assertEqual(f["overrideBy"], SUPERVISOR[0])
        self.assertEqual(f["disposition"], "released")
        self.assertEqual(post["files"], {})

    # 4. MES down, then back
    def test_mes_down_keeps_the_item_then_sends_it_exactly_once(self) -> None:
        self.mes.mode = 500
        self.press()
        self.press()
        session_id, _ = self.saved_part()

        self.assertEqual(self.app.sync_once(now=self.t0),
                         {"sent": 0, "failed_permanent": 0, "retry_later": 1})
        self.assertEqual(self.app.outbox.pending_count(), 1)
        self.assertEqual([r["status"] for r in self.mes.requests], [500])

        # The station keeps working while MES is down.
        st = self.state()
        self.assertEqual(st["counters"], {"total": 1, "good": 1, "bad": 0})
        self.assertTrue(self.outputs.green)
        self.assertEqual(self.app.store.counters()["total"], 1)

        self.mes.mode = 201
        self.assertEqual(self.app.sync_once(now=self.t0 + 3600),
                         {"sent": 1, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(len(self.mes.accepted), 1)
        self.assertEqual({r["key"] for r in self.mes.requests}, {session_id})
        self.assertEqual([r["status"] for r in self.mes.requests], [500, 201])

        # Nothing is left to send: a repeat sends nothing.
        self.assertEqual(self.app.sync_once(now=self.t0 + 7200),
                         {"sent": 0, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(len(self.mes.requests), 2)
        self.assertEqual(self.app.outbox.list_items()[0]["status"], "sent")

    # 5. 422 is permanent
    def test_422_marks_the_item_failed_and_it_is_not_retried(self) -> None:
        self.mes.mode = 422
        self.press()
        self.press()
        session_id, _ = self.saved_part()

        self.assertEqual(self.app.sync_once(now=self.t0),
                         {"sent": 0, "failed_permanent": 1, "retry_later": 0})
        failed = self.app.outbox.failed_permanent_items()
        self.assertEqual([f["session_id"] for f in failed], [session_id])
        self.assertIn("422", failed[0]["last_error"])
        self.assertEqual([r["status"] for r in self.mes.requests], [422])

        self.mes.mode = 201
        self.assertEqual(self.app.sync_once(now=self.t0 + 7200)["sent"], 0)
        self.assertEqual(self.app.sync_once(now=self.t0 + 14400)["sent"], 0)
        self.assertEqual(len(self.mes.requests), 1)
        self.assertEqual(len(self.mes.accepted), 0)

    # 6. restart keeps counters and the unsent item
    def test_restart_keeps_counters_and_the_unsent_outbox_item(self) -> None:
        self.mes.mode = 500
        self.press()
        self.press()
        session_id, _ = self.saved_part()
        self.app.sync_once(now=self.t0)
        self.app.stop()

        app2 = self.start_app()
        ui2 = Ui(app2)
        op2 = ui2.login(*OPERATOR)
        self.assertEqual(app2.outbox.pending_count(), 1)
        self.assertEqual(app2.session.counters, {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(app2.store.counters()["total"], 1)

        self.mes.mode = 201
        self.assertEqual(app2.sync_once(now=self.t0 + 3600)["sent"], 1)
        self.assertEqual(len(self.mes.accepted), 1)
        self.assertEqual({r["key"] for r in self.mes.requests}, {session_id})

        # The next part after the restart is counted and gets a new id.
        self.press(token=op2, ui=ui2)
        self.press(token=op2, ui=ui2)
        self.assertEqual(ui2.req("GET", "/api/state", token=op2)[1]["counters"]["total"], 2)
        rows = app2.store.recent_parts(2)
        self.assertNotEqual(rows[0]["part_id"], rows[1]["part_id"])
        self.assertNotEqual(rows[0]["session_id"], rows[1]["session_id"])
        self.assertEqual(app2.store.verify_part(pathlib.Path(rows[0]["path"])), [])


if __name__ == "__main__":
    unittest.main()
