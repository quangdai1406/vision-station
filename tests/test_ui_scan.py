"""HTTP tests for the Taking-data endpoints and page (station.ui, plan step 4).

Fakes only: the cameras are fake, the inspection session is FakeSession from test_ui.
The scan session and the scan store are real (a temporary folder), so a saved scan is
checked with station.scanstore.verify_scan. No real camera, no ~/vision-home.
"""
import http.client
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import threading
import types
import unittest
from unittest import mock

from station.app import StationApp, _LockedSession
from station.auth import SessionManager, UserStore, hash_password
from station.contracts import CameraError, Frame, Side
from station.scan import ScanSession
from station.scanstore import ScanStore, list_scans, load_scan, verify_scan
from station.ui import make_server
from tests.test_ui import ADM_PW, ADM_USER, OP_PW, OP_USER, SUP_PW, SUP_USER, FakeSession, Reply

NOW = "2026-10-27T08:30:12Z"
CAMS = ["C1", "C2", "C3", "C4"]
ROUTES = [
    ("POST", "/api/scan/start"),
    ("POST", "/api/scan/press"),
    ("POST", "/api/scan/done-side"),
    ("POST", "/api/scan/mark"),
    ("POST", "/api/scan/finish"),
    ("POST", "/api/scan/cancel"),
    ("POST", "/api/scan/clear-fault"),
    ("POST", "/api/scan/reset"),
    ("GET", "/api/scan/state"),
    ("GET", "/api/scan/step-picture?camera=C1"),
]


class FakeCams:
    """Each capture returns one picture per camera: data is 'TOP-C1-<capture number>'.
    A scripted exception is raised instead of a capture."""

    def __init__(self):
        self.script = []
        self.count = 0

    def capture(self, side):
        self.count += 1
        action = self.script.pop(0) if self.script else None
        if isinstance(action, BaseException):
            raise action
        return [Frame(camera_id=c, side=side, seq=i, timestamp_utc=NOW,
                      data=f"{side.value}-{c}-{self.count}".encode(), ext="png")
                for i, c in enumerate(CAMS)]


def start_body(verdict="GOOD", defects=None, toy="toy-8", **extra):
    label = {"verdict": verdict, "defects": [] if defects is None else defects}
    body = {"toy_id": toy, "label": label}
    body.update(extra)
    return body


class ScanHttpBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        self.cams = FakeCams()
        self.busy = [False]
        self.store = ScanStore(self.root, site="SITE", line="L1")
        ids = iter(f"SC-{i}" for i in range(1, 500))
        self.scan = ScanSession(
            self.cams, self.store, busy=lambda: self.busy[0], debounce_s=0.0, min_flip_s=0.0,
            now_utc=lambda: NOW, scan_id_factory=lambda: next(ids),
        )
        self.inspect = FakeSession()
        self.users = UserStore({
            OP_USER: {"password_hash": hash_password(OP_PW), "role": "operator"},
            SUP_USER: {"password_hash": hash_password(SUP_PW), "role": "supervisor"},
            ADM_USER: {"password_hash": hash_password(ADM_PW), "role": "admin"},
        })
        self.sessions = SessionManager(timeout_s=600)
        self.server = None
        self.start_server(self.inspect, self.scan)
        self.tokens = {
            "op": self.login(OP_USER, OP_PW),
            "sup": self.login(SUP_USER, SUP_PW),
            "adm": self.login(ADM_USER, ADM_PW),
        }

    def start_server(self, station, scan):
        self.stop_server()
        self.server = make_server(station, self.users, self.sessions, host="127.0.0.1",
                                  port=0, scan=scan)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(5)
            self.server = None

    def call(self, method, path, body=None, token=None, raw=None):
        hdrs = {}
        if token:
            hdrs["Cookie"] = "station_session=" + token
        data = raw
        if data is None and body is not None:
            data = json.dumps(body).encode("utf-8")
        if data is not None:
            hdrs["Content-Type"] = "application/json"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request(method, path, body=data, headers=hdrs)
            resp = conn.getresponse()
            return Reply(resp.status, resp.getheaders(), resp.read())
        finally:
            conn.close()

    def login(self, user, password):
        reply = self.call("POST", "/api/login", {"username": user, "password": password})
        self.assertEqual(reply.status, 200, reply.body)
        cookie = dict(reply.headers)["Set-Cookie"]
        return cookie.split(";", 1)[0].split("=", 1)[1]

    def post(self, role, path, body=None):
        return self.call("POST", path, body={} if body is None else body, token=self.tokens[role])

    def get(self, role, path):
        return self.call("GET", path, token=self.tokens[role])

    def ok(self, role, path, body=None):
        reply = self.post(role, path, body)
        self.assertEqual(reply.status, 200, reply.body)
        return json.loads(reply.body.decode("utf-8"))

    def state(self, role="sup"):
        reply = self.get(role, "/api/scan/state")
        self.assertEqual(reply.status, 200, reply.body)
        return json.loads(reply.body.decode("utf-8"))["scan"]

    def saved_scans(self):
        return list_scans(self.root)

    def scan_doc(self, scan_dir):
        return load_scan(scan_dir)


class ScanAuthTests(ScanHttpBase):
    def test_every_scan_route_needs_a_login(self):
        for method, path in ROUTES:
            reply = self.call(method, path, body=None if method == "GET" else {})
            self.assertEqual(reply.status, 401, path)

    def test_operator_is_refused_on_every_scan_route(self):
        for method, path in ROUTES:
            reply = self.call(method, path, body=None if method == "GET" else {},
                              token=self.tokens["op"])
            self.assertEqual(reply.status, 403, path)
            self.assertNotIn(b"Traceback", reply.body)
        # Nothing was started by the refused requests.
        self.assertEqual(self.scan.snapshot()["state"], "IDLE")

    def test_supervisor_and_admin_may_use_the_routes(self):
        for role in ("sup", "adm"):
            state = self.state(role)
            self.assertEqual(state["state"], "IDLE")
            self.assertNotIn("saved_path", state)
            reply = self.get(role, "/api/scan/state")
            self.assertTrue(json.loads(reply.body)["may_scan"])
            self.assertEqual(self.ok(role, "/api/scan/press"), {"accepted": False, "scan": state})

    def test_disabled_scan_is_404_after_the_login_and_role_checks(self):
        self.start_server(self.inspect, None)
        for method, path in ROUTES:
            body = None if method == "GET" else {}
            self.assertEqual(self.call(method, path, body=body).status, 401, path)
            self.assertEqual(self.call(method, path, body=body, token=self.tokens["op"]).status,
                             403, path)
            reply = self.call(method, path, body=body, token=self.tokens["sup"])
            self.assertEqual(reply.status, 404, path)
            self.assertIn("not enabled", json.loads(reply.body)["error"])

    def test_actions_list_offers_scan_only_when_enabled_and_allowed(self):
        def actions(role):
            reply = self.get(role, "/api/state")
            return json.loads(reply.body)["auth"]["actions"]

        self.assertNotIn("scan", actions("op"))
        self.assertIn("scan", actions("sup"))
        self.assertIn("scan", actions("adm"))
        self.start_server(self.inspect, None)
        self.assertNotIn("scan", actions("sup"))
        self.assertNotIn("scan", actions("adm"))

    def test_scan_active_flag_is_visible_to_every_role(self):
        def scan_active(role):
            return json.loads(self.get(role, "/api/state").body)["scan_active"]

        self.assertFalse(scan_active("op"))
        self.ok("sup", "/api/scan/start", start_body())
        self.assertTrue(scan_active("op"))
        self.assertTrue(scan_active("adm"))
        self.ok("sup", "/api/scan/cancel")
        self.assertFalse(scan_active("op"))

    def test_state_snapshot_hides_the_folder_path(self):
        self.ok("sup", "/api/scan/start", start_body())
        self.ok("sup", "/api/scan/press")
        self.ok("sup", "/api/scan/done-side")
        self.ok("sup", "/api/scan/press")
        self.ok("sup", "/api/scan/done-side")
        reply = self.ok("sup", "/api/scan/finish")
        self.assertNotIn("saved_path", reply["scan"])
        self.assertNotIn(str(self.root), json.dumps(reply))
        self.assertNotIn(str(self.root), self.get("sup", "/api/scan/state").body.decode("utf-8"))


class ScanFlowTests(ScanHttpBase):
    def take_good_scan(self, role="sup"):
        self.ok(role, "/api/scan/start", start_body())
        self.assertTrue(self.ok(role, "/api/scan/press")["accepted"])
        self.assertTrue(self.ok(role, "/api/scan/press")["accepted"])
        self.assertEqual(self.ok(role, "/api/scan/done-side")["scan"]["state"], "WAIT_FLIP")
        self.assertTrue(self.ok(role, "/api/scan/press")["accepted"])  # flip, BOTTOM step 1
        self.assertEqual(self.ok(role, "/api/scan/done-side")["scan"]["state"], "SAVING")
        return self.ok(role, "/api/scan/finish")

    def test_good_flow_over_http_saves_a_scan_that_verifies(self):
        started = self.ok("sup", "/api/scan/start", start_body(toy="toy-8"))
        self.assertEqual(started["scan_id"], "SC-1")
        self.assertEqual(started["scan"]["state"], "SCAN_SIDE")
        self.assertEqual(started["scan"]["part_number"], "AISI-100")  # the station's current part
        self.assertEqual(self.state()["prompt"], started["scan"]["prompt"])
        self.assertTrue(self.ok("sup", "/api/scan/press")["accepted"])
        self.assertTrue(self.ok("sup", "/api/scan/press")["accepted"])
        self.assertEqual(self.ok("sup", "/api/scan/done-side")["scan"]["state"], "WAIT_FLIP")
        self.assertTrue(self.ok("sup", "/api/scan/press")["accepted"])
        self.assertEqual(self.ok("sup", "/api/scan/done-side")["scan"]["state"], "SAVING")
        finished = self.ok("sup", "/api/scan/finish")
        self.assertTrue(finished["saved"])
        self.assertEqual(finished["scan"]["state"], "DONE")

        scans = self.saved_scans()
        self.assertEqual(len(scans), 1)
        self.assertEqual(verify_scan(scans[0]), [])
        doc = self.scan_doc(scans[0])
        self.assertEqual(doc["toy_id"], "toy-8")
        self.assertEqual(doc["taken_by"], SUP_USER)
        self.assertEqual(doc["label"]["verdict"], "GOOD")
        self.assertEqual(doc["label"]["status"], "proposed")
        self.assertEqual(doc["label"]["labelled_by"], SUP_USER)
        per_side = {}
        for image in doc["images"]:
            per_side[image["side"]] = per_side.get(image["side"], 0) + 1
        self.assertEqual(per_side, {"TOP": 8, "BOTTOM": 4})  # 2 steps x 4 cameras, 1 x 4
        self.assertEqual(self.ok("sup", "/api/scan/reset")["scan"]["state"], "IDLE")

    def test_admin_label_is_official(self):
        self.take_good_scan("adm")
        doc = self.scan_doc(self.saved_scans()[0])
        self.assertEqual(doc["label"]["status"], "official")
        self.assertEqual(doc["label"]["labelled_by"], ADM_USER)
        self.assertEqual(verify_scan(self.saved_scans()[0]), [])

    def test_client_supplied_label_fields_are_ignored(self):
        body = start_body(toy="toy-9")
        body["label"].update({
            "status": "official", "labelled_by": "someone-else",
            "labelled_at": "2001-01-01T00:00:00Z", "approved_by": "admin-x",
            "approved_at": "2001-01-02T00:00:00Z", "reject_reason": "x",
        })
        body["taken_by"] = "someone-else"
        body["sides"] = ["TOP"]
        self.ok("sup", "/api/scan/start", body)
        self.ok("sup", "/api/scan/press")
        self.ok("sup", "/api/scan/done-side")
        self.ok("sup", "/api/scan/finish")
        doc = self.scan_doc(self.saved_scans()[0])
        self.assertEqual(doc["label"]["status"], "proposed")
        self.assertEqual(doc["label"]["labelled_by"], SUP_USER)
        self.assertNotEqual(doc["label"]["labelled_at"], "2001-01-01T00:00:00Z")
        self.assertIsNone(doc["label"]["approved_by"])
        self.assertEqual(doc["taken_by"], SUP_USER)
        self.assertEqual(verify_scan(self.saved_scans()[0]), [])

    def test_part_number_empty_uses_station_part_and_none_is_refused(self):
        self.ok("sup", "/api/scan/start", start_body(part_number="   "))
        self.assertEqual(self.state()["part_number"], "AISI-100")
        self.ok("sup", "/api/scan/cancel")
        self.inspect.current = {"part_number": "", "model_version": None, "can_inspect": False,
                                "reason": None, "parts_mode": "mes"}
        reply = self.post("sup", "/api/scan/start", start_body())
        self.assertEqual(reply.status, 400)
        self.assertIn("part_number is required", json.loads(reply.body)["error"])
        self.assertEqual(self.state()["state"], "IDLE")

    def test_step_picture_returns_the_last_step_with_no_store(self):
        self.assertEqual(self.get("sup", "/api/scan/step-picture?camera=C1").status, 204)
        self.ok("sup", "/api/scan/start", start_body())
        self.assertEqual(self.get("sup", "/api/scan/step-picture?camera=C1").status, 204)
        self.ok("sup", "/api/scan/press")
        reply = self.get("sup", "/api/scan/step-picture?camera=C1")
        self.assertEqual(reply.status, 200)
        self.assertEqual(reply.body, b"TOP-C1-1")
        headers = dict(reply.headers)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertEqual(self.get("sup", "/api/scan/step-picture?camera=C2").body, b"TOP-C2-1")
        self.ok("sup", "/api/scan/press")
        self.assertEqual(self.get("sup", "/api/scan/step-picture?camera=C1").body, b"TOP-C1-2")
        self.ok("sup", "/api/scan/done-side")
        self.assertEqual(self.get("sup", "/api/scan/step-picture?camera=C1").status, 200)
        self.ok("sup", "/api/scan/press")
        self.ok("sup", "/api/scan/done-side")
        self.ok("sup", "/api/scan/finish")
        self.assertEqual(self.get("sup", "/api/scan/step-picture?camera=C1").status, 204)

    def test_bad_flow_marks_reach_scan_json(self):
        self.ok("sup", "/api/scan/start",
                start_body("BAD", ["doh_small", "doh_edge"], toy="toy-10"))
        self.ok("sup", "/api/scan/press")
        marked = self.ok("sup", "/api/scan/mark",
                         {"camera_id": "C1", "side": "TOP", "step": 1, "visible": True})
        self.assertEqual(marked["scan"]["marks"], {"pictures": 4, "visible": 1, "not_visible": 3})
        self.ok("sup", "/api/scan/press")
        self.ok("sup", "/api/scan/done-side")
        self.ok("sup", "/api/scan/press")
        self.ok("sup", "/api/scan/mark",
                {"camera_id": "C3", "side": "BOTTOM", "step": 1, "visible": True})
        # An earlier step can still be changed until finish.
        self.ok("sup", "/api/scan/mark",
                {"camera_id": "C2", "side": "TOP", "step": 1, "visible": True})
        self.ok("sup", "/api/scan/mark",
                {"camera_id": "C2", "side": "TOP", "step": 1, "visible": False})
        self.ok("sup", "/api/scan/done-side")
        self.ok("sup", "/api/scan/finish")

        scan_dir = self.saved_scans()[0]
        self.assertEqual(verify_scan(scan_dir), [])
        doc = self.scan_doc(scan_dir)
        self.assertEqual(doc["label"]["verdict"], "BAD")
        self.assertEqual(doc["label"]["defects"], ["doh_small", "doh_edge"])
        self.assertEqual(doc["label"]["status"], "proposed")
        visible = {(i["camera_id"], i["side"], i["step"]): i["defect_visible"]
                   for i in doc["images"]}
        self.assertTrue(visible[("C1", "TOP", 1)])
        self.assertTrue(visible[("C3", "BOTTOM", 1)])
        self.assertFalse(visible[("C2", "TOP", 1)])
        self.assertFalse(visible[("C4", "TOP", 1)])
        self.assertEqual(sum(1 for v in visible.values() if v), 2)

    def test_marks_are_refused_on_good_and_on_unknown_pictures(self):
        self.ok("sup", "/api/scan/start", start_body())
        self.ok("sup", "/api/scan/press")
        reply = self.post("sup", "/api/scan/mark",
                          {"camera_id": "C1", "side": "TOP", "step": 1, "visible": True})
        self.assertEqual(reply.status, 400)
        self.assertIn("only a BAD scan has defect marks", json.loads(reply.body)["error"])
        self.ok("sup", "/api/scan/cancel")
        self.ok("sup", "/api/scan/start", start_body("BAD", ["doh_large"]))
        self.ok("sup", "/api/scan/press")
        reply = self.post("sup", "/api/scan/mark",
                          {"camera_id": "C1", "side": "TOP", "step": 7, "visible": True})
        self.assertEqual(reply.status, 400)
        self.assertIn("no picture C1 TOP step 7", json.loads(reply.body)["error"])

    def test_cancel_saves_nothing(self):
        self.ok("sup", "/api/scan/start", start_body("BAD", ["doh_small"]))
        self.ok("sup", "/api/scan/press")
        self.ok("sup", "/api/scan/done-side")
        self.assertEqual(self.ok("sup", "/api/scan/cancel")["scan"]["state"], "IDLE")
        self.assertEqual(self.saved_scans(), [])
        self.assertEqual(list(self.root.rglob("*.png")), [])
        self.assertEqual(list(self.root.rglob("scan.json")), [])

    def test_camera_fault_keeps_the_steps_and_clear_fault_returns_to_the_side(self):
        self.ok("sup", "/api/scan/start", start_body())
        self.ok("sup", "/api/scan/press")
        self.cams.script = [CameraError("C3 did not answer")]
        self.assertTrue(self.ok("sup", "/api/scan/press")["accepted"])
        state = self.state()
        self.assertEqual(state["state"], "FAULT")
        self.assertIn("did not answer", state["fault"])
        self.assertEqual(self.get("sup", "/api/scan/step-picture?camera=C1").status, 204)
        self.assertEqual(self.ok("sup", "/api/scan/clear-fault")["scan"]["state"], "SCAN_SIDE")
        self.assertEqual(self.state()["counts"], {"TOP": 1, "BOTTOM": 0})
        self.assertTrue(self.ok("sup", "/api/scan/press")["accepted"])
        self.assertEqual(self.state()["counts"]["TOP"], 2)

    def test_finish_and_done_need_their_steps_first(self):
        self.ok("sup", "/api/scan/start", start_body())
        reply = self.post("sup", "/api/scan/done-side")
        self.assertEqual(reply.status, 400)
        self.assertIn("has no steps yet", json.loads(reply.body)["error"])
        self.ok("sup", "/api/scan/press")
        self.ok("sup", "/api/scan/done-side")
        reply = self.post("sup", "/api/scan/finish")
        self.assertEqual(reply.status, 400)
        self.assertIn("after Done on the last side", json.loads(reply.body)["error"])

    def test_start_refused_while_inspection_runs_or_a_scan_is_open(self):
        self.busy[0] = True
        reply = self.post("sup", "/api/scan/start", start_body())
        self.assertEqual(reply.status, 400)
        self.assertIn("an inspection is running", json.loads(reply.body)["error"])
        self.busy[0] = False
        self.ok("sup", "/api/scan/start", start_body())
        reply = self.post("sup", "/api/scan/start", start_body())
        self.assertEqual(reply.status, 400)
        self.assertIn("cannot start in state SCAN_SIDE", json.loads(reply.body)["error"])

    def test_reset_then_a_second_scan_gets_a_new_id(self):
        self.take_good_scan()
        self.assertEqual(self.ok("sup", "/api/scan/reset")["scan"]["state"], "IDLE")
        second = self.ok("sup", "/api/scan/start", start_body(toy="toy-11"))
        self.assertEqual(second["scan_id"], "SC-2")
        self.ok("sup", "/api/scan/cancel")
        self.assertEqual(len(self.saved_scans()), 1)

    def test_idle_press_is_ignored_and_not_an_error(self):
        self.assertFalse(self.ok("sup", "/api/scan/press")["accepted"])
        reply = self.post("sup", "/api/scan/done-side")
        self.assertEqual(reply.status, 400)
        self.assertIn("done_side is not allowed in state IDLE", json.loads(reply.body)["error"])


class ScanInputTests(ScanHttpBase):
    def check_400(self, path, body, expected, token="sup"):
        reply = self.post(token, path, body)
        self.assertEqual(reply.status, 400, (path, body, reply.body))
        message = json.loads(reply.body)["error"]
        self.assertIn(expected, message)
        self.assertNotIn(b"Traceback", reply.body)

    def test_start_input_is_validated_with_a_message(self):
        cases = [
            ({"label": {"verdict": "GOOD"}}, "toy_id is required"),
            ({"toy_id": 5, "label": {"verdict": "GOOD"}}, "toy_id is required"),
            (start_body(toy="a/b"), "must not contain / or"),
            (start_body(toy="a\\b"), "must not contain / or"),
            (start_body(toy="..x"), "must not contain / or"),
            (start_body(toy="x" * 65), "toy_id must be 1 to 64 characters"),
            (start_body(toy="a\nb"), "control characters"),
            (start_body(toy="   "), "toy_id must be 1 to 64 characters"),
            ({"toy_id": "t"}, "label must be an object"),
            ({"toy_id": "t", "label": "GOOD"}, "label must be an object"),
            (start_body(verdict="MAYBE"), "verdict must be GOOD or BAD"),
            (start_body("BAD", []), "a BAD label needs at least one defect"),
            (start_body("GOOD", ["doh_small"]), "a GOOD label has no defects"),
            (start_body("BAD", "doh_small"), "defects must be a list"),
            (start_body("BAD", ["d"] * 33), "defects must be a list of up to 32"),
            (start_body("BAD", [1]), "defect names must be text"),
            (start_body("BAD", [" "]), "each defect name must be 1 to 64 characters"),
            (start_body("BAD", ["d" * 65]), "each defect name must be 1 to 64 characters"),
            (start_body("BAD", ["a", "a"]), "must not repeat a name"),
            (start_body(part_number="p" * 101), "part_number must be a string up to 100"),
            (start_body(part_number="p" * 65), "part_number is longer than 64 characters"),
            (start_body(part_number=7), "part_number must be a string"),
            (start_body(sides=["LEFT"]), "sides must be a list of TOP and/or BOTTOM"),
            (start_body(sides=[]), "sides must be a list of TOP and/or BOTTOM"),
            (start_body(sides=["TOP", "TOP"]), "sides must not repeat"),
        ]
        for body, expected in cases:
            self.check_400("/api/scan/start", body, expected)
        self.assertEqual(self.state()["state"], "IDLE")

    def test_mark_input_is_validated_with_a_message(self):
        self.ok("sup", "/api/scan/start", start_body("BAD", ["doh_small"]))
        self.ok("sup", "/api/scan/press")
        good = {"camera_id": "C1", "side": "TOP", "step": 1, "visible": True}
        cases = [
            ({**good, "camera_id": "C9"}, "camera_id must be C1, C2, C3 or C4"),
            ({**good, "camera_id": 1}, "camera_id must be C1, C2, C3 or C4"),
            ({**good, "side": "LEFT"}, "side must be TOP or BOTTOM"),
            ({**good, "step": "1"}, "step must be a whole number of at least 1"),
            ({**good, "step": True}, "step must be a whole number of at least 1"),
            ({**good, "step": 0}, "step must be a whole number of at least 1"),
            ({**good, "visible": "yes"}, "visible must be true or false"),
            ({**good, "visible": 1}, "visible must be true or false"),
        ]
        for body, expected in cases:
            self.check_400("/api/scan/mark", body, expected)

    def test_other_bad_requests(self):
        self.check_400("/api/scan/finish", {}, "only possible after Done")
        self.check_400("/api/scan/mark", good_mark(), "marks cannot change in state IDLE")
        reply = self.call("POST", "/api/scan/press", raw=b"{", token=self.tokens["sup"])
        self.assertEqual(reply.status, 400)
        self.assertIn("invalid JSON", json.loads(reply.body)["error"])
        reply = self.call("POST", "/api/scan/press", raw=b"[]", token=self.tokens["sup"])
        self.assertEqual(reply.status, 400)
        self.assertIn("JSON object required", json.loads(reply.body)["error"])
        reply = self.call("POST", "/api/scan/press", raw=b"x" * (64 * 1024 + 1),
                          token=self.tokens["sup"])
        self.assertEqual(reply.status, 413)
        reply = self.call("GET", "/api/scan/step-picture?camera=C7", token=self.tokens["sup"])
        self.assertEqual(reply.status, 400)
        self.assertIn("camera must be C1", json.loads(reply.body)["error"])
        reply = self.call("POST", "/api/scan/unknown", body={}, token=self.tokens["sup"])
        self.assertEqual(reply.status, 404)

    def test_scan_error_text_never_carries_a_secret_or_traceback(self):
        self.busy[0] = True
        reply = self.post("sup", "/api/scan/start", start_body())
        self.assertNotIn(b"Traceback", reply.body)
        self.assertNotIn(SUP_PW.encode(), reply.body)
        self.assertNotIn(b"password", reply.body.lower())


def good_mark():
    return {"camera_id": "C1", "side": "TOP", "step": 1, "visible": True}


class InspectionGuardTests(ScanHttpBase):
    def test_inspection_press_is_refused_while_a_scan_is_active(self):
        # The real StationApp guard methods, bound to a stub that holds only what they use.
        stub = types.SimpleNamespace(
            _scan_session=self.scan,
            store=types.SimpleNamespace(append_event=lambda *a, **k: None),
            parts_mode="fixed",
            camera_status=lambda: {"open": True, "last_error": None},
            current_part=lambda: dict(self.inspect.current),
        )
        stub.scan_active = types.MethodType(StationApp.scan_active, stub)
        stub.scan_blocks_press = types.MethodType(StationApp.scan_blocks_press, stub)
        locked = _LockedSession(self.inspect, threading.RLock(), stub)
        self.start_server(locked, self.scan)

        self.ok("sup", "/api/scan/start", start_body())
        reply = self.ok("op", "/api/press")
        self.assertEqual(reply, {"accepted": False})
        self.assertNotIn("press", self.inspect.names())
        self.assertTrue(json.loads(self.get("op", "/api/state").body)["scan_active"])

        self.ok("sup", "/api/scan/cancel")
        self.assertEqual(self.ok("op", "/api/press"), {"accepted": True})
        self.assertIn("press", self.inspect.names())


class ScanPageTests(ScanHttpBase):
    def page(self):
        reply = self.call("GET", "/")
        self.assertEqual(reply.status, 200)
        return reply.body.decode("utf-8")

    def test_panel_and_button_are_in_the_page_hidden_by_default(self):
        html = self.page()
        self.assertIn('id="scan-open" class="secondary" hidden', html)
        self.assertIn('<section id="scan-panel" class="card" aria-label="Taking data" hidden>', html)
        for control in ("scan-toy", "scan-part", "scan-start-form", "scan-press", "scan-done",
                        "scan-finish", "scan-cancel", "scan-fault", "scan-reset", "scan-pics",
                        "scan-marks", "scan-busy", "inspect-controls"):
            self.assertIn('id="%s"' % control, html)
        self.assertIn('value="BAD"', html)
        self.assertIn("doh_small, doh_large, doh_edge", html)
        self.assertLess(html.index('id="scan-panel"'), html.index('id="inspect-controls"'))

    def test_live_view_and_step_pictures_stay_on_during_a_scan(self):
        html = self.page()
        # The live view is not hidden while a scan runs; it pauses only while saving or in a fault.
        self.assertNotIn('$("live").hidden', html)
        self.assertIn('scanSnap.state === "SCAN_SIDE" || scanSnap.state === "WAIT_FLIP"', html)
        # Step pictures are listed for GOOD scans too; only BAD scans get the defect checkbox.
        self.assertNotIn('if (s.label_verdict !== "BAD" || count < 1', html)
        self.assertIn("if (isBad) {", html)

    def test_button_visibility_comes_from_the_actions_list(self):
        html = self.page()
        self.assertIn('var scanAllowed = false;', html)
        self.assertIn("scanAllowed = !!allowed.scan;", html)
        self.assertIn("$(\"scan-open\").hidden = !scanAllowed;", html)
        self.assertIn("if (scanAllowed) { refreshScan(); }", html)
        # The server's list is what the browser sees: scan only for the roles that may use it.
        self.assertNotIn("scan", json.loads(self.get("op", "/api/state").body)["auth"]["actions"])
        self.assertIn("scan", json.loads(self.get("sup", "/api/state").body)["auth"]["actions"])

    def test_enter_key_goes_to_the_scan_while_it_runs(self):
        html = self.page()
        key = html.index('addEventListener("keydown"')  # the inspection key handler
        self.assertIn("if (scanIsActive()) {", html[key:])
        self.assertLess(html.index("if (scanIsActive()) {", key),
                        html.index("if (inPressLockout()) { setPressStatus(MSG_WAIT_CAPTURE); return; }", key))
        self.assertIn('"/api/scan/press"', html)
        self.assertIn('"/api/scan/step-picture?camera="', html)

    def test_scan_values_are_set_as_text_only(self):
        html = self.page()
        block = self.scan_block(html)
        self.assertNotIn("innerHTML", block)
        self.assertNotIn("insertAdjacentHTML", block)
        self.assertNotIn("document.write", block)

    def test_scan_review_buttons_do_not_trigger_the_part_label_handler(self):
        # A data-action="label_approve" button also fires the page-wide handler, which would
        # approve the collected part's label for the same click.
        html = self.page()
        start = html.index("function renderScanReview(rows)")
        end = html.index("function refreshScanReview()")
        self.assertNotIn("data-action", html[start:end])

    def test_review_list_is_in_the_panel_and_hidden_by_default(self):
        html = self.page()
        self.assertIn('<section id="scan-review" aria-label="Scans waiting for approval" hidden>', html)
        for control in ("scan-review-none", "scan-review-list"):
            self.assertIn('id="%s"' % control, html)
        self.assertLess(html.index('id="scan-panel"'), html.index('id="scan-review"'))
        self.assertLess(html.index('id="scan-review"'), html.index('id="inspect-controls"'))
        self.assertIn('"/api/scan/pending"', html)
        self.assertIn('"/api/scan/review"', html)
        self.assertIn("reviewAllowed = !!allowed.label_approve;", html)
        self.assertIn('$("scan-review").hidden = !reviewAllowed;', html)
        self.assertNotIn("window.prompt", self.scan_block(html))

    @staticmethod
    def scan_block(html):
        start = html.index("// ----- Taking data (supervisor and admin)")
        end = html.index("// ----- collected-part labels")
        return html[start:end]

    def test_scan_block_passes_node_check_when_node_available(self):
        if shutil.which("node") is None:
            self.skipTest("node not available")
        block = self.scan_block(self.page())
        self.assertIn("function renderScanPanel()", block)
        # Wrapped in a function so the block is checked as the script part it is.
        source = "(function () {\n" + block + "\n})();\n"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scan_block.js")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(source)
            proc = subprocess.run(["node", "--check", path], capture_output=True, text=True,
                                  timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)


PROPOSED_GOOD = {"verdict": "GOOD", "defects": [], "status": "proposed",
                 "labelled_by": SUP_USER, "labelled_at": NOW}


class ScanReviewSessionTests(unittest.TestCase):
    """ScanSession.pending_reviews and ScanSession.review on a real store (fake cameras)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)
        self.stamp = [NOW]
        self.store = ScanStore(self.root, site="SITE", line="L1")
        ids = iter(f"SC-{i}" for i in range(1, 500))
        self.scan = ScanSession(
            FakeCams(), self.store, busy=lambda: False, debounce_s=0.0, min_flip_s=0.0,
            now_utc=lambda: self.stamp[0], scan_id_factory=lambda: next(ids),
        )

    def take(self, verdict="GOOD", defects=None, toy="toy-1", status="proposed",
             taken_by=SUP_USER, when=NOW):
        """One finished scan: TOP step 1, flip, BOTTOM step 1, saved."""
        self.stamp[0] = when
        label = {"verdict": verdict, "defects": [] if defects is None else defects,
                 "status": status, "labelled_by": taken_by, "labelled_at": when}
        scan_id = self.scan.start("AISI-100", toy, label, taken_by)
        self.scan.press()
        self.scan.done_side()
        self.scan.press()
        self.scan.done_side()
        self.scan.finish()
        self.scan.reset()
        return scan_id

    def folder(self, scan_id):
        return next(d for d in list_scans(self.root) if d.name == scan_id)

    def test_pending_lists_only_proposed_scans_newest_first(self):
        older = self.take(when="2026-10-27T08:00:00Z")
        newer = self.take("BAD", ["doh_small"], toy="toy-2", when="2026-10-27T09:00:00Z")
        self.take(status="official", taken_by=ADM_USER, when="2026-10-27T09:30:00Z")
        rejected = self.take(when="2026-10-27T09:45:00Z")
        self.scan.review(rejected, "reject", ADM_USER, "photo blurred")
        rows = self.scan.pending_reviews()
        self.assertEqual([r["scan_id"] for r in rows], [newer, older])
        row = rows[0]
        self.assertEqual(set(row), {"scan_id", "part_number", "toy_id", "verdict", "defects",
                                    "labelled_by", "labelled_at", "steps"})
        self.assertEqual(row["verdict"], "BAD")
        self.assertEqual(row["defects"], ["doh_small"])
        self.assertEqual(row["labelled_by"], SUP_USER)
        self.assertEqual(row["labelled_at"], "2026-10-27T09:00:00Z")
        self.assertEqual(row["steps"], {"TOP": 1, "BOTTOM": 1})
        self.assertEqual(row["part_number"], "AISI-100")

    def test_pending_is_capped(self):
        with mock.patch("station.scan.PENDING_REVIEW_CAP", 2):
            scans = [self.take(when=f"2026-10-27T0{h}:00:00Z") for h in (7, 8, 9)]
            rows = self.scan.pending_reviews()
        self.assertEqual([r["scan_id"] for r in rows], [scans[2], scans[1]])

    def test_a_damaged_scan_is_not_listed_and_not_reviewed(self):
        damaged = self.take(when="2026-10-27T08:00:00Z")
        intact = self.take(toy="toy-2", when="2026-10-27T09:00:00Z")
        picture = next(p for p in self.folder(damaged).iterdir() if p.suffix == ".png")
        picture.write_bytes(b"changed after the save")
        self.assertEqual([r["scan_id"] for r in self.scan.pending_reviews()], [intact])
        with self.assertRaises(ValueError) as ctx:
            self.scan.review(damaged, "approve", ADM_USER)
        self.assertIn("damaged", str(ctx.exception))
        self.assertEqual(load_scan(self.folder(damaged))["label"]["status"], "proposed")

    def test_approve_makes_the_label_official_and_the_scan_verifies(self):
        scan_id = self.take(when="2026-10-27T08:00:00Z")
        self.stamp[0] = "2026-10-27T10:00:00Z"
        label = self.scan.review(scan_id, "approve", ADM_USER)
        self.assertEqual(label["status"], "official")
        self.assertEqual(label["approved_by"], ADM_USER)
        self.assertEqual(label["approved_at"], "2026-10-27T10:00:00Z")
        self.assertEqual(label["labelled_by"], SUP_USER)
        self.assertEqual(verify_scan(self.folder(scan_id)), [])
        self.assertEqual(self.scan.pending_reviews(), [])

    def test_reject_needs_a_reason_and_records_the_reviewer(self):
        scan_id = self.take()
        with self.assertRaises(ValueError):
            self.scan.review(scan_id, "reject", ADM_USER)
        with self.assertRaises(ValueError):
            self.scan.review(scan_id, "reject", ADM_USER, "   ")
        label = self.scan.review(scan_id, "reject", ADM_USER, "photo blurred")
        self.assertEqual(label["status"], "rejected")
        self.assertEqual(label["reject_reason"], "photo blurred")
        self.assertEqual(label["approved_by"], ADM_USER)
        self.assertEqual(verify_scan(self.folder(scan_id)), [])
        self.assertEqual(self.scan.pending_reviews(), [])

    def test_bad_scan_ids_and_reviewers_are_refused(self):
        scan_id = self.take()
        bad_ids = ["../x", "a/b", "..\\x", "", "SC-404", "x\x00y", str(self.folder(scan_id)), None]
        for bad in bad_ids:
            with self.subTest(scan_id=bad):
                with self.assertRaises(ValueError):
                    self.scan.review(bad, "approve", ADM_USER)
        with self.assertRaises(ValueError):
            self.scan.review(scan_id, "approve", "")
        self.assertEqual([r["scan_id"] for r in self.scan.pending_reviews()], [scan_id])

    def test_only_a_proposed_label_can_be_reviewed(self):
        scan_id = self.take()
        self.scan.review(scan_id, "approve", ADM_USER)
        with self.assertRaises(ValueError) as ctx:
            self.scan.review(scan_id, "approve", ADM_USER)
        self.assertIn("only a proposed label", str(ctx.exception))
        official = self.take(status="official", taken_by=ADM_USER, toy="toy-2")
        with self.assertRaises(ValueError):
            self.scan.review(official, "reject", ADM_USER, "no")

    def test_other_scans_can_be_reviewed_while_a_scan_runs(self):
        scan_id = self.take()
        self.scan.start("AISI-100", "toy-3", dict(PROPOSED_GOOD), SUP_USER)
        self.assertEqual(self.scan.snapshot()["state"], "SCAN_SIDE")
        self.scan.review(scan_id, "approve", ADM_USER)
        self.assertEqual(load_scan(self.folder(scan_id))["label"]["status"], "official")


class ScanReviewHttpTests(ScanHttpBase):
    def take_scan(self, role="sup", verdict="GOOD", defects=None, toy="toy-8"):
        started = self.ok(role, "/api/scan/start", start_body(verdict, defects, toy=toy))
        self.ok(role, "/api/scan/press")
        self.ok(role, "/api/scan/done-side")
        self.ok(role, "/api/scan/press")
        self.ok(role, "/api/scan/done-side")
        self.ok(role, "/api/scan/finish")
        self.ok(role, "/api/scan/reset")
        return started["scan_id"]

    def pending_rows(self, role="adm"):
        reply = self.get(role, "/api/scan/pending")
        self.assertEqual(reply.status, 200, reply.body)
        return json.loads(reply.body.decode("utf-8"))["pending"]

    def test_both_routes_need_a_login_then_the_label_approve_action(self):
        self.assertEqual(self.call("GET", "/api/scan/pending").status, 401)
        self.assertEqual(self.call("POST", "/api/scan/review", body={}).status, 401)
        for role in ("op", "sup"):
            self.assertEqual(self.get(role, "/api/scan/pending").status, 403, role)
            reply = self.post(role, "/api/scan/review",
                              {"scan_id": "SC-1", "decision": "approve"})
            self.assertEqual(reply.status, 403, role)
            self.assertNotIn(b"Traceback", reply.body)
        self.assertNotIn("label_approve", json.loads(self.get("sup", "/api/state").body)["auth"]["actions"])
        self.assertIn("label_approve", json.loads(self.get("adm", "/api/state").body)["auth"]["actions"])

    def test_supervisor_scan_is_listed_for_the_admin_without_pictures_or_paths(self):
        scan_id = self.take_scan("sup", toy="toy-8")
        reply = self.get("adm", "/api/scan/pending")
        rows = json.loads(reply.body.decode("utf-8"))["pending"]
        self.assertEqual([r["scan_id"] for r in rows], [scan_id])
        row = rows[0]
        self.assertEqual(row["toy_id"], "toy-8")
        self.assertEqual(row["verdict"], "GOOD")
        self.assertEqual(row["defects"], [])
        self.assertEqual(row["labelled_by"], SUP_USER)
        self.assertEqual(row["steps"], {"TOP": 1, "BOTTOM": 1})
        self.assertNotIn("images", row)
        self.assertNotIn(str(self.root), reply.body.decode("utf-8"))

    def test_admin_approves_and_the_scan_verifies(self):
        scan_id = self.take_scan("sup")
        reply = self.post("adm", "/api/scan/review", {"scan_id": scan_id, "decision": "approve"})
        self.assertEqual(reply.status, 200, reply.body)
        body = json.loads(reply.body.decode("utf-8"))
        self.assertEqual(body["label"]["status"], "official")
        self.assertEqual(body["label"]["approved_by"], ADM_USER)
        self.assertEqual(body["pending"], [])
        folder = self.saved_scans()[0]
        doc = self.scan_doc(folder)
        self.assertEqual(doc["label"]["status"], "official")
        self.assertEqual(doc["label"]["approved_by"], ADM_USER)
        self.assertEqual(doc["label"]["labelled_by"], SUP_USER)
        self.assertEqual(verify_scan(folder), [])

    def test_reviewer_name_comes_from_the_login(self):
        scan_id = self.take_scan("sup")
        reply = self.post("adm", "/api/scan/review", {
            "scan_id": scan_id, "decision": "approve", "by": "someone-else",
            "approved_by": "someone-else", "label": {"status": "rejected"},
        })
        self.assertEqual(reply.status, 200, reply.body)
        self.assertEqual(self.scan_doc(self.saved_scans()[0])["label"]["approved_by"], ADM_USER)

    def test_reject_needs_a_reason_and_is_recorded(self):
        scan_id = self.take_scan("sup")
        for body in ({"scan_id": scan_id, "decision": "reject"},
                     {"scan_id": scan_id, "decision": "reject", "reason": "   "},
                     {"scan_id": scan_id, "decision": "reject", "reason": "x" * 201}):
            with self.subTest(body=body):
                reply = self.post("adm", "/api/scan/review", body)
                self.assertEqual(reply.status, 400, reply.body)
        reply = self.post("adm", "/api/scan/review", {
            "scan_id": scan_id, "decision": "reject", "reason": "  photo blurred  "})
        self.assertEqual(reply.status, 200, reply.body)
        body = json.loads(reply.body.decode("utf-8"))
        self.assertEqual(body["label"]["status"], "rejected")
        self.assertEqual(body["label"]["reject_reason"], "photo blurred")
        self.assertEqual(body["label"]["approved_by"], ADM_USER)
        self.assertEqual(body["pending"], [])
        self.assertEqual(verify_scan(self.saved_scans()[0]), [])

    def test_bad_requests_are_400_with_a_message(self):
        scan_id = self.take_scan("sup")
        cases = [
            ({}, "scan_id is required"),
            ({"scan_id": "", "decision": "approve"}, "scan_id is required"),
            ({"scan_id": 7, "decision": "approve"}, "scan_id is required"),
            ({"scan_id": scan_id}, "decision must be approve or reject"),
            ({"scan_id": scan_id, "decision": "maybe"}, "decision must be approve or reject"),
            ({"scan_id": scan_id, "decision": "approve", "reason": "why"},
             "reason is only for a rejection"),
            ({"scan_id": "SC-404", "decision": "approve"}, "no finished scan"),
            ({"scan_id": "../x", "decision": "approve"}, "scan_id must be a scan folder name"),
        ]
        for body, text in cases:
            with self.subTest(body=body):
                reply = self.post("adm", "/api/scan/review", body)
                self.assertEqual(reply.status, 400, reply.body)
                self.assertIn(text, json.loads(reply.body.decode("utf-8"))["error"])
        self.assertEqual([r["scan_id"] for r in self.pending_rows()], [scan_id])
        self.post("adm", "/api/scan/review", {"scan_id": scan_id, "decision": "approve"})
        reply = self.post("adm", "/api/scan/review", {"scan_id": scan_id, "decision": "approve"})
        self.assertEqual(reply.status, 400)
        self.assertIn("only a proposed label", json.loads(reply.body.decode("utf-8"))["error"])

    def test_disabled_scan_is_404_after_the_login_and_label_check(self):
        self.start_server(self.inspect, None)
        self.assertEqual(self.get("adm", "/api/scan/pending").status, 404)
        reply = self.post("adm", "/api/scan/review", {"scan_id": "SC-1", "decision": "approve"})
        self.assertEqual(reply.status, 404)
        self.assertIn("not enabled", json.loads(reply.body.decode("utf-8"))["error"])
        self.assertEqual(self.get("sup", "/api/scan/pending").status, 403)

    def test_review_works_while_a_scan_is_running(self):
        scan_id = self.take_scan("sup")
        self.ok("sup", "/api/scan/start", start_body(toy="toy-9"))
        self.assertEqual(self.state()["state"], "SCAN_SIDE")
        reply = self.post("adm", "/api/scan/review", {"scan_id": scan_id, "decision": "approve"})
        self.assertEqual(reply.status, 200, reply.body)
        self.assertEqual(self.pending_rows(), [])
        self.assertEqual(self.state()["state"], "SCAN_SIDE")


if __name__ == "__main__":
    unittest.main()
