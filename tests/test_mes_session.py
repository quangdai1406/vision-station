"""Tests for station/mes_session.py and the Station-Session header on MES posts.

Runs against a local fake MES on 127.0.0.1, port 0. No network.
"""
from __future__ import annotations

import http.server
import json
import pathlib
import sqlite3
import tempfile
import threading
import unittest
import urllib.parse

from station.contracts import Frame, ImageResult, PartRecord, Side, Thresholds, Verdict
from station.mes_client import MesClient, MesConfig
from station.outbox import Outbox
from station.mes_session import MesSessionClient, MesSessionError, SessionInfo

TOKEN = "test-token-only-0001"
REDEEM = "/api/machine/sessions/redeem"
SID = "sess-abc-123"


class _State:
    def __init__(self):
        self.routes = {}  # (method, path) -> (status, body dict or None)
        self.requests = []  # (method, path, headers, body bytes)


class _Handler(http.server.BaseHTTPRequestHandler):
    def _handle(self, method):
        state = self.server.state
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        path = urllib.parse.urlsplit(self.path).path
        state.requests.append((method, self.path, dict(self.headers.items()), body))
        status, payload = state.routes.get((method, path), (404, {"error": "not found"}))
        raw = b"" if payload is None else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def log_message(self, *args):
        pass


class _FakeMesCase(unittest.TestCase):
    def setUp(self):
        self.state = _State()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.state = self.state
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def dead_base(self):
        """A base URL where nothing listens (connection refused)."""
        probe = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        port = probe.server_address[1]
        probe.server_close()
        return f"http://127.0.0.1:{port}"


SESSION_BODY = {"sessionId": SID, "username": "anna.k", "roles": ["inspector"],
                "expiresAt": "2026-11-04T09:00:00Z"}
VALIDATE_BODY = {"active": True, "username": "anna.k", "roles": ["inspector"],
                 "expiresAt": "2026-11-04T09:15:00Z"}


class RedeemTests(_FakeMesCase):
    def test_redeem_success_returns_session_info(self):
        self.state.routes[("POST", REDEEM)] = (200, SESSION_BODY)
        client = MesSessionClient(self.base, TOKEN, timeout_s=5)
        info = client.redeem("ticket-one")
        self.assertEqual(info, SessionInfo(session_id=SID, username="anna.k",
                                           roles=("inspector",),
                                           expires_utc="2026-11-04T09:00:00Z"))
        method, path, headers, body = self.state.requests[0]
        self.assertEqual(json.loads(body.decode("utf-8")), {"ticket": "ticket-one"})
        self.assertEqual(headers.get("Authorization"), "Bearer " + TOKEN)

    def test_second_redeem_of_same_ticket_is_409(self):
        self.state.routes[("POST", REDEEM)] = (409, {"error": "ticket already used"})
        client = MesSessionClient(self.base, TOKEN, timeout_s=5)
        with self.assertRaises(MesSessionError) as ctx:
            client.redeem("ticket-one")
        self.assertEqual(ctx.exception.status, 409)
        self.assertNotIn(TOKEN, str(ctx.exception))
        self.assertNotIn("ticket-one", str(ctx.exception))

    def test_unexpected_format_raises(self):
        self.state.routes[("POST", REDEEM)] = (200, {"sessionId": SID})
        client = MesSessionClient(self.base, TOKEN, timeout_s=5)
        with self.assertRaises(MesSessionError):
            client.redeem("ticket-one")

    def test_empty_ticket_is_refused_before_any_request(self):
        client = MesSessionClient(self.base, TOKEN, timeout_s=5)
        with self.assertRaises(ValueError):
            client.redeem("")
        self.assertEqual(self.state.requests, [])

    def test_network_error_raises_without_token_or_ticket(self):
        client = MesSessionClient(self.dead_base(), TOKEN, timeout_s=2)
        with self.assertRaises(MesSessionError) as ctx:
            client.redeem("ticket-secret-xyz")
        text = str(ctx.exception)
        self.assertNotIn(TOKEN, text)
        self.assertNotIn("ticket-secret-xyz", text)

    def test_redirect_is_not_followed(self):
        self.state.routes[("POST", REDEEM)] = (302, None)
        client = MesSessionClient(self.base, TOKEN, timeout_s=5)
        with self.assertRaises(MesSessionError):
            client.redeem("ticket-one")

    def test_bad_construction_is_refused(self):
        with self.assertRaises(MesSessionError):
            MesSessionClient("ftp://mes", TOKEN)
        with self.assertRaises(MesSessionError):
            MesSessionClient(self.base, "")

    def test_token_not_in_repr(self):
        client = MesSessionClient(self.base, TOKEN, timeout_s=5)
        self.assertNotIn(TOKEN, repr(client))


class ValidateTests(_FakeMesCase):
    def test_active_session_returns_info(self):
        self.state.routes[("GET", f"/api/machine/sessions/{SID}")] = (200, VALIDATE_BODY)
        client = MesSessionClient(self.base, TOKEN, timeout_s=5)
        info = client.validate(SID)
        self.assertIsNotNone(info)
        self.assertEqual(info.username, "anna.k")
        self.assertEqual(info.expires_utc, "2026-11-04T09:15:00Z")

    def test_inactive_session_returns_none(self):
        body = dict(VALIDATE_BODY, active=False)
        self.state.routes[("GET", f"/api/machine/sessions/{SID}")] = (200, body)
        self.assertIsNone(MesSessionClient(self.base, TOKEN, timeout_s=5).validate(SID))

    def test_401_returns_none(self):
        self.state.routes[("GET", f"/api/machine/sessions/{SID}")] = (401, {"error": "no"})
        self.assertIsNone(MesSessionClient(self.base, TOKEN, timeout_s=5).validate(SID))

    def test_validate_network_error_raises(self):
        client = MesSessionClient(self.dead_base(), TOKEN, timeout_s=2)
        with self.assertRaises(MesSessionError) as ctx:
            client.validate(SID)
        self.assertNotIn(TOKEN, str(ctx.exception))

    def test_unsafe_session_id_is_refused_before_any_request(self):
        client = MesSessionClient(self.base, TOKEN, timeout_s=5)
        with self.assertRaises(ValueError):
            client.validate("../admin")
        self.assertEqual(self.state.requests, [])


class EndTests(_FakeMesCase):
    def test_end_posts_to_session_end_path(self):
        self.state.routes[("POST", f"/api/machine/sessions/{SID}/end")] = (200, {"ok": True})
        MesSessionClient(self.base, TOKEN, timeout_s=5).end(SID)
        method, path, headers, _ = self.state.requests[0]
        self.assertEqual((method, path), ("POST", f"/api/machine/sessions/{SID}/end"))
        self.assertEqual(headers.get("Authorization"), "Bearer " + TOKEN)

    def test_end_of_unknown_session_is_not_an_error(self):
        self.state.routes[("POST", f"/api/machine/sessions/{SID}/end")] = (404, {"error": "gone"})
        MesSessionClient(self.base, TOKEN, timeout_s=5).end(SID)

    def test_end_network_error_raises_without_token(self):
        client = MesSessionClient(self.dead_base(), TOKEN, timeout_s=2)
        with self.assertRaises(MesSessionError) as ctx:
            client.end(SID)
        self.assertNotIn(TOKEN, str(ctx.exception))


def _record() -> PartRecord:
    frames = [Frame(camera_id="C1", side=Side.TOP, seq=0, timestamp_utc="2026-11-04T08:30:12Z",
                    data=b"img", ext="png")]
    results = [ImageResult(camera_id="C1", side=Side.TOP, score=0.1, defects=())]
    return PartRecord(
        part_id="P-1", session_id="sess-1", station_id="station-1", part_number="PN-100",
        lot="LOT-9", operator="anna.k", supervisor="sup-bob", material_handler="mh-cy",
        started_utc="2026-11-04T08:30:00Z", model_version="m-1",
        thresholds=Thresholds(t_low=0.3, t_img=0.6), frames=frames, results=results,
        machine_verdict=Verdict.GOOD, final_verdict=Verdict.GOOD, override=None,
    )


class PostHeaderTests(_FakeMesCase):
    """MesClient.post sends Station-Session when a session id is given."""

    def _post(self, **kwargs) -> dict:
        cfg = MesConfig(base_url=self.base, token=TOKEN, criteria_ids=["CR-1"],
                        class_to_criterion={}, timeout_s=5)
        self.state.routes[("POST", "/api/machine/inspection-events")] = (201, {"id": 5})
        result = MesClient(cfg).post({"verdict": "GOOD"}, [], idempotency_key="sess-1", **kwargs)
        self.assertTrue(result.ok)
        _, _, headers, _ = self.state.requests[-1]
        return headers

    def test_header_is_sent_with_the_session_id(self):
        headers = self._post(station_session=SID)
        self.assertEqual(headers.get("Station-Session"), SID)
        self.assertEqual(headers.get("Idempotency-Key"), "sess-1")

    def test_header_is_absent_without_a_session(self):
        headers = self._post()
        self.assertNotIn("Station-Session", headers)

    def test_unsafe_session_value_is_refused_before_sending(self):
        cfg = MesConfig(base_url=self.base, token=TOKEN, criteria_ids=["CR-1"],
                        class_to_criterion={}, timeout_s=5)
        with self.assertRaises(ValueError):
            MesClient(cfg).post({"verdict": "GOOD"}, [], idempotency_key="sess-1",
                                station_session="abc\r\nX-Evil: 1")
        self.assertEqual(self.state.requests, [])


class OutboxSessionColumnTests(unittest.TestCase):
    """The outbox keeps each part's MES session and migrates outboxes made before the column."""

    def test_old_outbox_gains_the_column_and_keeps_its_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "outbox").mkdir()
            old = sqlite3.connect(str(root / "outbox.db"))
            old.execute("CREATE TABLE items (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "session_id TEXT NOT NULL UNIQUE, fields_json TEXT NOT NULL, "
                        "files_json TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL "
                        "DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0, last_error TEXT, "
                        "mes_event_id INTEGER, created_at REAL NOT NULL, sent_at REAL)")
            old.execute("INSERT INTO items (session_id, fields_json, files_json, status, "
                        "created_at) VALUES ('sess-old', '{}', '[]', 'pending', 0)")
            old.commit()
            old.close()
            ob = Outbox(root)
            try:
                cols = [r["name"] for r in ob._db.execute("PRAGMA table_info(items)")]
                self.assertIn("station_session", cols)
                self.assertEqual(ob._db.execute("SELECT COUNT(*) FROM items").fetchone()[0], 1)
                ob.enqueue("sess-new", {"verdict": "GOOD"}, [], station_session=SID)
                row = ob._db.execute(
                    "SELECT station_session FROM items WHERE session_id = 'sess-new'").fetchone()
                self.assertEqual(row[0], SID)
            finally:
                ob.close()


if __name__ == "__main__":
    unittest.main()
