"""HTTP tests for station.ui using a fake session on 127.0.0.1, port 0."""
import base64
import collections
import http.client
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unittest

from station.auth import SessionManager, UserStore, hash_password
from station.contracts import InvalidTransition, Verdict
from station.ui import make_server

# Throwaway test credentials only.
OP_USER, OP_PW = "op-test", "op-test-pass-1"
SUP_USER, SUP_PW = "sup-test", "sup-test-pass-1"
ADM_USER, ADM_PW = "adm-test", "adm-test-pass-1"

Reply = collections.namedtuple("Reply", "status headers body")

IMAGE_BYTES = b"\x89PNG-fake-image-bytes"
IMAGE_B64 = base64.b64encode(IMAGE_BYTES).decode("ascii")


class FakeSession:
    """Duck-typed session. snapshot() returns the shape the real session returns."""

    def __init__(self):
        self.calls = []
        self.errors = {}
        self.press_result = True
        self.verdict = None
        # None means the key is absent, as in a snapshot from before the change.
        self.partial_station = None
        self.camera_ids = None

    def _call(self, name, **kwargs):
        self.calls.append((name, kwargs))
        exc = self.errors.get(name)
        if exc is not None:
            raise exc

    def names(self):
        return [name for name, _ in self.calls]

    def snapshot(self, include_images=False):
        self._call("snapshot", include_images=include_images)
        defects = [{"defect_class": "scratch", "score": 0.9, "box": [1, 2, 3, 4]}]
        image = {
            "camera_id": "C1",
            "side": "TOP",
            "seq": 7,
            "ext": "png",
            "score": 0.9,
            "defects": defects,
        }
        if include_images:
            image["b64"] = IMAGE_B64
        out = {
            "state": "IDLE",
            "counters": {"total": 3, "good": 2, "bad": 1},
            "part_id": "P-100",
            "machine_verdict": "BAD",
            "final_verdict": None,
            "verdict": self.verdict,
            "fault": None,
            "results": [{"camera_id": "C1", "side": "TOP", "score": 0.9, "defects": defects}],
            "images": [image],
        }
        if self.partial_station is not None:
            out["partial_station"] = self.partial_station
        if self.camera_ids is not None:
            out["camera_ids"] = list(self.camera_ids)
        return out

    def press(self):
        self._call("press")
        return self.press_result

    def resolve_review(self, by, new_verdict, reason):
        self._call("resolve_review", by=by, new_verdict=new_verdict, reason=reason)

    def override(self, by, new_verdict, reason):
        self._call("override", by=by, new_verdict=new_verdict, reason=reason)

    def ack_alarm(self):
        self._call("ack_alarm")

    def clear_fault(self):
        self._call("clear_fault")

    def tick(self):
        self._call("tick")


class UiTestCase(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.users = UserStore({
            OP_USER: {"password_hash": hash_password(OP_PW), "role": "operator"},
            SUP_USER: {"password_hash": hash_password(SUP_PW), "role": "supervisor"},
            ADM_USER: {"password_hash": hash_password(ADM_PW), "role": "admin"},
        })
        self.sessions = SessionManager(timeout_s=600)
        self.server = make_server(self.session, self.users, self.sessions,
                                  host="127.0.0.1", port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def call(self, method, path, body=None, token=None, raw=None, headers=None):
        hdrs = dict(headers or {})
        if token:
            hdrs["Cookie"] = "station_session=" + token
        data = raw
        if data is None and body is not None:
            data = json.dumps(body).encode("utf-8")
        if data is not None and "Content-Type" not in hdrs:
            hdrs["Content-Type"] = "application/json"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(method, path, body=data, headers=hdrs)
            resp = conn.getresponse()
            return Reply(resp.status, resp.getheaders(), resp.read())
        finally:
            conn.close()

    def login(self, user, password):
        reply = self.call("POST", "/api/login", {"username": user, "password": password})
        self.assertEqual(reply.status, 200, reply.body)
        return self.token_from(reply)

    @staticmethod
    def set_cookie(reply):
        for name, value in reply.headers:
            if name.lower() == "set-cookie":
                return value
        return None

    def token_from(self, reply):
        cookie = self.set_cookie(reply)
        if not cookie or not cookie.startswith("station_session="):
            return None
        return cookie.split(";", 1)[0].split("=", 1)[1]

    @staticmethod
    def js(reply):
        return json.loads(reply.body.decode("utf-8"))


class LoginTests(UiTestCase):
    def test_login_ok_sets_hardened_cookie(self):
        reply = self.call("POST", "/api/login", {"username": OP_USER, "password": OP_PW})
        self.assertEqual(reply.status, 200)
        self.assertEqual(self.js(reply), {"role": "operator"})
        cookie = self.set_cookie(reply)
        self.assertIn("station_session=", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)

    def test_wrong_password_is_401_and_generic(self):
        wrong_pw = self.call("POST", "/api/login", {"username": OP_USER, "password": "bad-guess"})
        no_user = self.call("POST", "/api/login", {"username": "ghost", "password": "bad-guess"})
        self.assertEqual(wrong_pw.status, 401)
        self.assertEqual(no_user.status, 401)
        self.assertEqual(wrong_pw.body, no_user.body)
        self.assertIsNone(self.set_cookie(wrong_pw))
        self.assertNotIn(b"bad-guess", wrong_pw.body)

    def test_login_missing_fields_is_400(self):
        reply = self.call("POST", "/api/login", {"username": OP_USER})
        self.assertEqual(reply.status, 400)

    def test_login_bad_json_is_400(self):
        reply = self.call("POST", "/api/login", raw=b"{nope")
        self.assertEqual(reply.status, 400)

    def test_state_without_cookie_is_401(self):
        reply = self.call("GET", "/api/state")
        self.assertEqual(reply.status, 401)

    def test_action_without_cookie_is_401(self):
        self.assertEqual(self.call("POST", "/api/press", {}).status, 401)
        self.assertNotIn("press", self.session.names())

    def test_logout_invalidates_token(self):
        token = self.login(OP_USER, OP_PW)
        self.assertEqual(self.call("GET", "/api/state", token=token).status, 200)
        out = self.call("POST", "/api/logout", {}, token=token)
        self.assertEqual(out.status, 200)
        self.assertIn("Max-Age=0", self.set_cookie(out))
        self.assertEqual(self.call("GET", "/api/state", token=token).status, 401)


class StateTests(UiTestCase):
    def test_state_fields_and_auth_block(self):
        token = self.login(OP_USER, OP_PW)
        reply = self.call("GET", "/api/state", token=token)
        self.assertEqual(reply.status, 200)
        body = self.js(reply)
        self.assertEqual(body["state"], "IDLE")
        self.assertEqual(body["counters"], {"total": 3, "good": 2, "bad": 1})
        self.assertEqual(body["part_id"], "P-100")
        self.assertEqual(body["machine_verdict"], "BAD")
        self.assertEqual(body["auth"]["role"], "operator")
        self.assertIn("press", body["auth"]["actions"])
        self.assertNotIn("resolve", body["auth"]["actions"])
        self.assertIn("tick", self.session.names())
        self.assertEqual(self.session.calls[-1], ("snapshot", {"include_images": False}))

    def test_verdict_key_passes_through(self):
        token = self.login(OP_USER, OP_PW)
        for verdict in ("GOOD", "BAD", "REVIEW", None):
            with self.subTest(verdict=verdict):
                self.session.verdict = verdict
                body = self.js(self.call("GET", "/api/state", token=token))
                self.assertIn("verdict", body)
                self.assertEqual(body["verdict"], verdict)

    def test_without_images_session_called_with_false_and_no_b64(self):
        token = self.login(OP_USER, OP_PW)
        body = self.js(self.call("GET", "/api/state", token=token))
        self.assertEqual(self.session.calls[-1], ("snapshot", {"include_images": False}))
        image = body["images"][0]
        self.assertNotIn("b64", image)
        self.assertEqual(image["score"], 0.9)
        self.assertEqual(image["defects"][0]["box"], [1, 2, 3, 4])

    def test_images_param_passes_include_images_and_b64_comes_through(self):
        token = self.login(SUP_USER, SUP_PW)
        body = self.js(self.call("GET", "/api/state?images=1", token=token))
        self.assertEqual(self.session.calls[-1], ("snapshot", {"include_images": True}))
        image = body["images"][0]
        self.assertEqual(image["b64"], IMAGE_B64)
        self.assertEqual(base64.b64decode(image["b64"]), IMAGE_BYTES)
        self.assertNotIn("data", image)
        self.assertEqual(image["camera_id"], "C1")
        self.assertEqual(image["side"], "TOP")
        self.assertEqual(image["score"], 0.9)
        self.assertEqual(image["defects"][0]["box"], [1, 2, 3, 4])

    def test_unknown_path_is_404(self):
        self.assertEqual(self.call("GET", "/nope").status, 404)


class ActionTests(UiTestCase):
    def test_operator_can_press(self):
        token = self.login(OP_USER, OP_PW)
        reply = self.call("POST", "/api/press", {}, token=token)
        self.assertEqual(reply.status, 200)
        self.assertEqual(self.js(reply), {"accepted": True})
        self.assertIn("press", self.session.names())

    def test_operator_can_ack(self):
        token = self.login(OP_USER, OP_PW)
        self.assertEqual(self.call("POST", "/api/ack", {}, token=token).status, 200)
        self.assertIn("ack_alarm", self.session.names())

    def test_operator_forbidden_from_supervisor_actions(self):
        token = self.login(OP_USER, OP_PW)
        bodies = {
            "/api/resolve": {"verdict": "GOOD", "reason": "ok"},
            "/api/override": {"verdict": "GOOD", "reason": "ok"},
            "/api/clear-fault": {},
        }
        for path, body in bodies.items():
            with self.subTest(path=path):
                self.assertEqual(self.call("POST", path, body, token=token).status, 403)
        self.assertEqual(
            set(self.session.names()) & {"resolve_review", "override", "clear_fault"}, set())

    def test_supervisor_resolve_uses_logged_in_user_for_by(self):
        token = self.login(SUP_USER, SUP_PW)
        reply = self.call("POST", "/api/resolve",
                          {"verdict": "BAD", "reason": "scratch confirmed", "by": "mallory"},
                          token=token)
        self.assertEqual(reply.status, 200)
        name, kwargs = self.session.calls[-1]
        self.assertEqual(name, "resolve_review")
        self.assertEqual(kwargs["by"], SUP_USER)
        self.assertEqual(kwargs["new_verdict"], Verdict.BAD)
        self.assertEqual(kwargs["reason"], "scratch confirmed")

    def test_supervisor_override_and_clear_fault(self):
        token = self.login(SUP_USER, SUP_PW)
        self.assertEqual(self.call("POST", "/api/override",
                                   {"verdict": "GOOD", "reason": "re-inspected"},
                                   token=token).status, 200)
        name, kwargs = self.session.calls[-1]
        self.assertEqual((name, kwargs["by"], kwargs["new_verdict"]),
                         ("override", SUP_USER, Verdict.GOOD))
        self.assertEqual(self.call("POST", "/api/clear-fault", {}, token=token).status, 200)
        self.assertEqual(self.session.calls[-1][0], "clear_fault")

    def test_admin_can_do_everything(self):
        token = self.login(ADM_USER, ADM_PW)
        for path, body in {
            "/api/press": {}, "/api/ack": {}, "/api/clear-fault": {},
            "/api/resolve": {"verdict": "GOOD", "reason": "ok"},
            "/api/override": {"verdict": "BAD", "reason": "ok"},
        }.items():
            with self.subTest(path=path):
                self.assertEqual(self.call("POST", path, body, token=token).status, 200)

    def test_invalid_transition_is_409(self):
        self.session.errors["press"] = InvalidTransition("not idle")
        token = self.login(OP_USER, OP_PW)
        self.assertEqual(self.call("POST", "/api/press", {}, token=token).status, 409)

        self.session.errors["resolve_review"] = InvalidTransition("no review pending")
        token = self.login(SUP_USER, SUP_PW)
        reply = self.call("POST", "/api/resolve", {"verdict": "GOOD", "reason": "x"}, token=token)
        self.assertEqual(reply.status, 409)

    def test_session_value_error_is_400(self):
        self.session.errors["override"] = ValueError("override needs a BAD part")
        token = self.login(SUP_USER, SUP_PW)
        reply = self.call("POST", "/api/override", {"verdict": "GOOD", "reason": "x"}, token=token)
        self.assertEqual(reply.status, 400)

    def test_bad_verdict_is_400(self):
        token = self.login(SUP_USER, SUP_PW)
        for verdict in ("REVIEW", "maybe", "", None, 1):
            with self.subTest(verdict=verdict):
                reply = self.call("POST", "/api/resolve",
                                  {"verdict": verdict, "reason": "x"}, token=token)
                self.assertEqual(reply.status, 400)
                reply = self.call("POST", "/api/override",
                                  {"verdict": verdict, "reason": "x"}, token=token)
                self.assertEqual(reply.status, 400)
        self.assertNotIn("resolve_review", self.session.names())
        self.assertNotIn("override", self.session.names())

    def test_empty_reason_is_400(self):
        token = self.login(SUP_USER, SUP_PW)
        for body in ({"verdict": "GOOD"}, {"verdict": "GOOD", "reason": "   "},
                     {"verdict": "GOOD", "reason": 5}, {"verdict": "GOOD", "reason": "x" * 501}):
            with self.subTest(body=body):
                reply = self.call("POST", "/api/resolve", body, token=token)
                self.assertEqual(reply.status, 400)
        self.assertNotIn("resolve_review", self.session.names())

    def test_bad_json_is_400(self):
        token = self.login(SUP_USER, SUP_PW)
        for raw in (b"{not json", b"[1, 2]", b"\xff\xfe"):
            with self.subTest(raw=raw):
                reply = self.call("POST", "/api/resolve", raw=raw, token=token)
                self.assertEqual(reply.status, 400)

    def test_non_json_content_type_is_415(self):
        token = self.login(SUP_USER, SUP_PW)
        reply = self.call("POST", "/api/resolve", raw=b"verdict=GOOD", token=token,
                          headers={"Content-Type": "text/plain"})
        self.assertEqual(reply.status, 415)

    def test_oversized_body_is_413(self):
        token = self.login(OP_USER, OP_PW)
        reply = self.call("POST", "/api/press", raw=b"{}", token=token,
                          headers={"Content-Length": str(64 * 1024 + 1)})
        self.assertEqual(reply.status, 413)
        self.assertNotIn("press", self.session.names())


class PageTests(UiTestCase):
    def page(self):
        reply = self.call("GET", "/")
        self.assertEqual(reply.status, 200)
        self.assertIn("text/html", dict(reply.headers)["Content-Type"])
        return reply.body.decode("utf-8")

    def test_html_page_served_without_external_urls(self):
        html = self.page()
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertIn('getContext("2d")', html)
        self.assertIn("/api/state", html)

    def test_page_reads_b64_not_data(self):
        html = self.page()
        self.assertIn("b64", html)
        self.assertNotIn("item.data", html)
        self.assertNotIn("im.data", html)

    def test_page_polls_with_and_without_images(self):
        html = self.page()
        self.assertIn('"/api/state"', html)
        self.assertIn('"/api/state?images=1"', html)

    def test_page_shows_both_sides(self):
        html = self.page()
        self.assertIn('"TOP"', html)
        self.assertIn('"BOTTOM"', html)

    def test_state_line_shows_fault_text_via_textcontent(self):
        html = self.page()
        self.assertIn("s.fault", html)
        self.assertIn('"FAULT: "', html)
        self.assertIn("el.state.textContent", html)
        self.assertNotIn("innerHTML", html)

    def test_image_onerror_marks_failed_instead_of_clearing_key(self):
        html = self.page()
        self.assertIn("self.failed = true", html)
        self.assertIn("image failed to load", html)
        self.assertIn("this.failed = false", html)
        # clearing the cache key would make hasImage() false and re-request images every poll
        self.assertNotIn('self.key = ""', html)
        self.assertIn("!(retry && this.failed)", html)

    def test_page_script_passes_node_check_when_node_available(self):
        if shutil.which("node") is None:
            self.skipTest("node not available")
        html = self.page()
        m = re.search(r"<script>(.*?)</script>", html, re.S)
        self.assertIsNotNone(m)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "page.js")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(m.group(1))
            proc = subprocess.run(["node", "--check", path], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_security_headers_present(self):
        headers = dict(self.call("GET", "/").headers)
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")

    def test_enter_key_presses_part_with_focus_guards(self):
        html = self.page()
        self.assertIn('addEventListener("keydown"', html)
        self.assertIn('"Enter"', html)
        self.assertIn("ev.repeat", html)
        self.assertIn("document.activeElement", html)
        for tag in ("INPUT", "TEXTAREA", "SELECT", "BUTTON"):
            self.assertIn('tag === "%s"' % tag, html)
        self.assertIn("pressing", html)
        self.assertIn('PATHS.press', html)
        self.assertNotIn('"Escape"', html)

    def test_press_button_label_and_hint(self):
        html = self.page()
        self.assertIn(">Press (Enter)</button>", html)
        self.assertIn("Enter key or click: start TOP scan, flip the part, "
                      "press again for the BOTTOM scan.", html)

    def test_poc_banner_text_and_partial_gate(self):
        html = self.page()
        self.assertIn('id="poc"', html)
        self.assertIn("partial_station", html)
        self.assertIn("PROOF OF CONCEPT: ", html)
        # the sentence is split across JS string literals; check each fragment
        self.assertIn("This station does not meet the 8-image rule. ", html)
        self.assertIn("Results are not for production decisions.", html)
        self.assertIn("el.poc.textContent", html)
        self.assertIn("el.poc.hidden = !partial", html)
        # the banner sits above the verdict
        self.assertLess(html.index('id="poc"'), html.index('id="verdict"'))

    def test_tiles_built_from_camera_ids_with_c1_to_c4_default(self):
        html = self.page()
        self.assertIn("camera_ids", html)
        self.assertIn('DEFAULT_CAMERAS = ["C1", "C2", "C3", "C4"]', html)
        self.assertIn("ensureTiles(cams)", html)
        self.assertIn("--cols", html)


class PartialStationTests(UiTestCase):
    def test_partial_station_keys_pass_through_state(self):
        self.session.partial_station = True
        self.session.camera_ids = ["C1"]
        token = self.login(OP_USER, OP_PW)
        body = self.js(self.call("GET", "/api/state", token=token))
        self.assertIs(body["partial_station"], True)
        self.assertEqual(body["camera_ids"], ["C1"])
        full = self.js(self.call("GET", "/api/state?images=1", token=token))
        self.assertIs(full["partial_station"], True)
        self.assertEqual(full["camera_ids"], ["C1"])
        self.assertEqual(full["images"][0]["b64"], IMAGE_B64)

    def test_old_snapshot_without_new_keys_still_works(self):
        token = self.login(OP_USER, OP_PW)
        reply = self.call("GET", "/api/state", token=token)
        self.assertEqual(reply.status, 200)
        body = self.js(reply)
        self.assertNotIn("partial_station", body)
        self.assertNotIn("camera_ids", body)
        self.assertEqual(body["state"], "IDLE")
        self.assertEqual(self.call("GET", "/").status, 200)


if __name__ == "__main__":
    unittest.main()
