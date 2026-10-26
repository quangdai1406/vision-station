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
import urllib.parse
from unittest import mock

from station import auth as station_auth
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
JPEG_BYTES = b"\xff\xd8\xff-fake-jpeg-frame"


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
        self.prompt = "Flip the part, then press Enter"
        self.press_ready = True
        self.flip_wait_s = 0.0
        # False leaves prompt, press_ready and flip_wait_s out of snapshot().
        self.new_keys = True
        self.preview_bytes = JPEG_BYTES
        # Part picker: when set, these keys are merged into snapshot() (parts_mode 'mes').
        self.parts_snapshot = None
        self.parts = [{"part_number": "AISI-100", "description": "Liner LH", "part_type": "FG",
                       "lifecycle": "ACTIVE", "unit_cost": 12.5, "bom": ["x"],
                       "planning": {"wo": 3}}]
        self.select_result = {"part_number": "AISI-100", "model_version": "m-7",
                              "can_inspect": True}
        self.current = {"part_number": "AISI-100", "model_version": "m-7", "can_inspect": True,
                        "reason": None, "parts_mode": "mes"}

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
        if self.new_keys:
            out["prompt"] = self.prompt
            out["press_ready"] = self.press_ready
            out["flip_wait_s"] = self.flip_wait_s
        if self.parts_snapshot is not None:
            out.update(self.parts_snapshot)
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

    def preview(self, camera_id):
        self._call("preview", camera_id=camera_id)
        return self.preview_bytes

    def search_parts(self, q, limit=20):
        self._call("search_parts", q=q, limit=limit)
        return list(self.parts)

    def select_part(self, part_number, by):
        self._call("select_part", part_number=part_number, by=by)
        return dict(self.select_result)

    def current_part(self):
        self._call("current_part")
        return dict(self.current)

    def label_part(self, by, verdict, criterion_id=None, part_id=None):
        self._call("label_part", by=by, verdict=verdict, criterion_id=criterion_id,
                   part_id=part_id)
        return {"session_id": "S-9", "verdict": verdict.value, "criterion_id": criterion_id,
                "changed": False}


class PartsError(Exception):
    """Stand-in for the MES client's error. The UI matches it by class name only."""


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

    def test_csp_allows_only_blob_and_data_images(self):
        csp = dict(self.call("GET", "/").headers)["Content-Security-Policy"]
        self.assertIn("img-src data: blob:", csp)
        self.assertIn("default-src 'none'", csp)
        self.assertIn("connect-src 'self'", csp)
        self.assertNotIn("img-src data:;", csp)

    def test_live_view_section_and_polling(self):
        html = self.page()
        self.assertIn("Live view", html)
        self.assertIn("Live view (not saved)", html)
        self.assertIn('id="live-grid"', html)
        self.assertIn("PREVIEW_MS = 250", html)
        self.assertIn('"/api/preview?camera="', html)
        self.assertIn('credentials: "same-origin"', html)
        self.assertIn('document.visibilityState !== "visible"', html)
        self.assertIn('curState === "IDLE" || curState === "WAIT_FLIP"', html)
        self.assertIn("URL.createObjectURL", html)
        self.assertIn("URL.revokeObjectURL", html)
        self.assertIn("liveSchedule(t, g, PREVIEW_MS)", html)
        self.assertIn("PREVIEW_RETRY_MS", html)

    def test_press_lockout_and_capturing_feedback(self):
        html = self.page()
        self.assertIn("PRESS_LOCKOUT_MS = 1500", html)
        self.assertIn("PRESS_MAX_WAIT_MS", html)
        self.assertIn('"Capturing..."', html)
        self.assertIn("Capturing... keep the part still.", html)
        self.assertIn("Wait: the picture is being taken.", html)
        self.assertIn("inPressLockout()", html)

    def test_prompt_banner_wiring(self):
        html = self.page()
        self.assertIn('id="prompt"', html)
        self.assertIn("el.prompt.textContent", html)
        self.assertIn("s.prompt", html)
        self.assertLess(html.index('id="prompt"'), html.index('data-action="press"'))

    def test_wait_label_and_flip_message(self):
        html = self.page()
        self.assertIn("s.press_ready !== false", html)
        self.assertIn("s.flip_wait_s", html)
        self.assertIn('"Wait " + Math.ceil(flipWait) + " s"', html)
        self.assertIn("Press ignored: flip the part first, then press Enter.", html)
        self.assertIn('curState === "WAIT_FLIP"', html)
        self.assertIn("Press was not accepted in the current state.", html)


class PreviewTests(UiTestCase):
    @staticmethod
    def headers_of(reply):
        return {name.lower(): value for name, value in reply.headers}

    def test_preview_without_cookie_is_401(self):
        reply = self.call("GET", "/api/preview?camera=C1")
        self.assertEqual(reply.status, 401)
        self.assertNotIn("preview", self.session.names())

    def test_preview_jpeg_for_every_role(self):
        for user, pw in ((OP_USER, OP_PW), (SUP_USER, SUP_PW), (ADM_USER, ADM_PW)):
            with self.subTest(user=user):
                token = self.login(user, pw)
                reply = self.call("GET", "/api/preview?camera=C1", token=token)
                self.assertEqual(reply.status, 200)
                headers = self.headers_of(reply)
                self.assertEqual(headers["content-type"], "image/jpeg")
                self.assertEqual(headers["cache-control"], "no-store")
                self.assertEqual(reply.body, JPEG_BYTES)

    def test_preview_passes_camera_and_never_ticks_or_acts(self):
        token = self.login(OP_USER, OP_PW)
        self.call("GET", "/api/preview?camera=C3", token=token)
        self.assertEqual(self.session.calls[-1], ("preview", {"camera_id": "C3"}))
        self.assertEqual(set(self.session.names()), {"preview"})

    def test_preview_none_is_204(self):
        self.session.preview_bytes = None
        token = self.login(OP_USER, OP_PW)
        reply = self.call("GET", "/api/preview?camera=C2", token=token)
        self.assertEqual(reply.status, 204)
        self.assertEqual(reply.body, b"")
        self.assertEqual(self.headers_of(reply)["cache-control"], "no-store")

    def test_preview_exception_is_204_not_500(self):
        self.session.errors["preview"] = RuntimeError("camera unplugged")
        token = self.login(SUP_USER, SUP_PW)
        reply = self.call("GET", "/api/preview?camera=C1", token=token)
        self.assertEqual(reply.status, 204)
        self.assertNotIn(b"camera unplugged", reply.body)

    def test_preview_missing_or_invalid_camera_is_400(self):
        token = self.login(OP_USER, OP_PW)
        for raw_query in ("", "camera=", "camera=C9", "camera=" + urllib.parse.quote("../x", safe=""),
                          "camera=c1", "camera=C10"):
            with self.subTest(query=raw_query):
                reply = self.call("GET", "/api/preview?" + raw_query, token=token)
                self.assertEqual(reply.status, 400)
        self.assertNotIn("preview", self.session.names())


class SnapshotKeysTests(UiTestCase):
    def test_new_keys_pass_through_state(self):
        self.session.prompt = "Flip the part"
        self.session.press_ready = False
        self.session.flip_wait_s = 2.5
        token = self.login(OP_USER, OP_PW)
        body = self.js(self.call("GET", "/api/state", token=token))
        self.assertEqual(body["prompt"], "Flip the part")
        self.assertIs(body["press_ready"], False)
        self.assertEqual(body["flip_wait_s"], 2.5)

    def test_old_snapshot_without_new_keys_still_serves(self):
        self.session.new_keys = False
        token = self.login(OP_USER, OP_PW)
        reply = self.call("GET", "/api/state", token=token)
        self.assertEqual(reply.status, 200)
        body = self.js(reply)
        for key in ("prompt", "press_ready", "flip_wait_s"):
            self.assertNotIn(key, body)
        self.assertEqual(self.call("GET", "/").status, 200)


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


class PartsApiTests(UiTestCase):
    def test_search_requires_login(self):
        self.assertEqual(self.call("GET", "/api/parts/search?q=AISI").status, 401)
        self.assertNotIn("search_parts", self.session.names())

    def test_search_open_to_every_role_and_strips_q(self):
        for user, pw in ((OP_USER, OP_PW), (SUP_USER, SUP_PW), (ADM_USER, ADM_PW)):
            with self.subTest(user=user):
                token = self.login(user, pw)
                reply = self.call("GET", "/api/parts/search?q=%20AISI%20", token=token)
                self.assertEqual(reply.status, 200)
        self.assertEqual(self.session.calls[-1], ("search_parts", {"q": "AISI", "limit": 20}))

    def test_search_q_must_be_1_to_100_characters(self):
        token = self.login(OP_USER, OP_PW)
        for query in ("", "q=", "q=%20%20%20", "q=" + "x" * 101):
            with self.subTest(query=query[:12]):
                reply = self.call("GET", "/api/parts/search?" + query, token=token)
                self.assertEqual(reply.status, 400)
        self.assertNotIn("search_parts", self.session.names())
        self.assertEqual(self.call("GET", "/api/parts/search?q=" + "x" * 100, token=token).status,
                         200)

    def test_search_limit_defaults_and_clamps(self):
        token = self.login(OP_USER, OP_PW)
        cases = {"": 20, "&limit=abc": 20, "&limit=0": 1, "&limit=-5": 1,
                 "&limit=7": 7, "&limit=999": 50}
        for suffix, expected in cases.items():
            with self.subTest(suffix=suffix):
                self.session.calls = []
                reply = self.call("GET", "/api/parts/search?q=AISI" + suffix, token=token)
                self.assertEqual(reply.status, 200)
                self.assertEqual(self.session.calls[-1],
                                 ("search_parts", {"q": "AISI", "limit": expected}))

    def test_search_returns_only_the_four_allowed_keys(self):
        token = self.login(OP_USER, OP_PW)
        reply = self.call("GET", "/api/parts/search?q=AISI", token=token)
        self.assertNotIn(b"unit_cost", reply.body)
        self.assertNotIn(b"planning", reply.body)
        self.assertNotIn(b"bom", reply.body)
        self.assertEqual(self.js(reply), {"results": [{
            "part_number": "AISI-100", "description": "Liner LH",
            "part_type": "FG", "lifecycle": "ACTIVE"}]})

    def test_search_partserror_is_502_and_hides_detail(self):
        token = self.login(OP_USER, OP_PW)
        self.session.errors["search_parts"] = PartsError("connect to mes-db.internal failed")
        reply = self.call("GET", "/api/parts/search?q=AISI", token=token)
        self.assertEqual(reply.status, 502)
        self.assertIn("error", self.js(reply))
        self.assertNotIn(b"mes-db", reply.body)

    def test_search_valueerror_is_400(self):
        token = self.login(OP_USER, OP_PW)
        self.session.errors["search_parts"] = ValueError("bad query")
        self.assertEqual(self.call("GET", "/api/parts/search?q=AISI", token=token).status, 400)

    def test_select_requires_login(self):
        reply = self.call("POST", "/api/part/select", {"part_number": "AISI-100"})
        self.assertEqual(reply.status, 401)
        self.assertNotIn("select_part", self.session.names())

    def test_select_open_to_operator_supervisor_admin_and_by_is_logged_in_user(self):
        for user, pw in ((OP_USER, OP_PW), (SUP_USER, SUP_PW), (ADM_USER, ADM_PW)):
            with self.subTest(user=user):
                token = self.login(user, pw)
                reply = self.call("POST", "/api/part/select",
                                  {"part_number": "AISI-100", "by": "mallory"}, token=token)
                self.assertEqual(reply.status, 200)
                self.assertEqual(self.session.calls[-1],
                                 ("select_part", {"part_number": "AISI-100", "by": user}))
        self.assertEqual(self.js(reply)["model_version"], "m-7")

    def test_select_role_without_action_gets_403(self):
        token = self.login(OP_USER, OP_PW)
        with mock.patch.dict(station_auth._POLICY, {"operator": frozenset({"press", "ack"})}):
            reply = self.call("POST", "/api/part/select", {"part_number": "AISI-100"}, token=token)
        self.assertEqual(reply.status, 403)
        self.assertNotIn("select_part", self.session.names())

    def test_select_bad_part_number_is_400(self):
        token = self.login(OP_USER, OP_PW)
        for body in ({"part_number": ""}, {"part_number": "   "}, {"part_number": "x" * 101},
                     {"part_number": 5}, {}):
            with self.subTest(body=body):
                reply = self.call("POST", "/api/part/select", body, token=token)
                self.assertEqual(reply.status, 400)
        self.assertNotIn("select_part", self.session.names())

    def test_select_accepts_exactly_100_characters(self):
        token = self.login(OP_USER, OP_PW)
        reply = self.call("POST", "/api/part/select", {"part_number": "x" * 100}, token=token)
        self.assertEqual(reply.status, 200)

    def test_select_invalid_transition_is_409(self):
        token = self.login(OP_USER, OP_PW)
        self.session.errors["select_part"] = InvalidTransition("part can only change while IDLE")
        reply = self.call("POST", "/api/part/select", {"part_number": "AISI-100"}, token=token)
        self.assertEqual(reply.status, 409)

    def test_select_partserror_is_502_and_hides_detail(self):
        token = self.login(OP_USER, OP_PW)
        self.session.errors["select_part"] = PartsError("MES timeout at 10.0.0.5")
        reply = self.call("POST", "/api/part/select", {"part_number": "AISI-100"}, token=token)
        self.assertEqual(reply.status, 502)
        self.assertNotIn(b"10.0.0.5", reply.body)

    def test_select_valueerror_is_400(self):
        token = self.login(OP_USER, OP_PW)
        self.session.errors["select_part"] = ValueError("unknown part")
        reply = self.call("POST", "/api/part/select", {"part_number": "AISI-100"}, token=token)
        self.assertEqual(reply.status, 400)

    def test_select_response_keeps_only_allowed_keys(self):
        self.session.select_result = {"part_number": "AISI-100", "model_version": None,
                                      "can_inspect": False, "reason": "no approved model",
                                      "warnings": ["model review due"], "unit_cost": 9.9}
        token = self.login(OP_USER, OP_PW)
        body = self.js(self.call("POST", "/api/part/select", {"part_number": "AISI-100"},
                                 token=token))
        self.assertEqual(body, {"part_number": "AISI-100", "model_version": None,
                                "can_inspect": False, "reason": "no approved model",
                                "warnings": ["model review due"]})

    def test_get_part_any_logged_in_role(self):
        self.assertEqual(self.call("GET", "/api/part").status, 401)
        for user, pw in ((OP_USER, OP_PW), (SUP_USER, SUP_PW), (ADM_USER, ADM_PW)):
            with self.subTest(user=user):
                token = self.login(user, pw)
                reply = self.call("GET", "/api/part", token=token)
                self.assertEqual(reply.status, 200)
                body = self.js(reply)
                self.assertEqual(body["part_number"], "AISI-100")
                self.assertEqual(body["parts_mode"], "mes")
                self.assertIs(body["can_inspect"], True)
        self.assertIn("current_part", self.session.names())

    def test_parts_snapshot_keys_pass_through_state(self):
        self.session.parts_snapshot = {
            "parts_mode": "mes", "camera": {"open": False, "last_error": "no frames"},
            "part_number": "AISI-100", "part_selected": True, "model_version": "m-7",
            "can_inspect": False, "block_reason": "no approved model"}
        token = self.login(OP_USER, OP_PW)
        body = self.js(self.call("GET", "/api/state", token=token))
        for key, value in self.session.parts_snapshot.items():
            self.assertEqual(body[key], value)

    def test_old_snapshot_without_parts_keys_still_serves(self):
        token = self.login(OP_USER, OP_PW)
        reply = self.call("GET", "/api/state", token=token)
        self.assertEqual(reply.status, 200)
        body = self.js(reply)
        for key in ("parts_mode", "camera", "part_selected", "block_reason"):
            self.assertNotIn(key, body)


class PartsPageTests(UiTestCase):
    def page(self):
        reply = self.call("GET", "/")
        self.assertEqual(reply.status, 200)
        return reply.body.decode("utf-8")

    def test_part_panel_gated_on_parts_mode(self):
        html = self.page()
        self.assertIn('id="part-panel"', html)
        self.assertIn("<h2>Part number</h2>", html)
        self.assertIn('s.parts_mode === "mes"', html)
        self.assertIn('el.partPanel.hidden = partsMode !== "mes"', html)
        self.assertLess(html.index('id="part-panel"'), html.index('data-action="press"'))

    def test_search_and_select_endpoints_wired(self):
        html = self.page()
        self.assertIn('id="part-q"', html)
        self.assertIn('"/api/parts/search?q="', html)
        self.assertIn('"/api/part/select"', html)
        self.assertIn('el.partForm.addEventListener("submit"', html)

    def test_press_labels_for_part_choice(self):
        html = self.page()
        self.assertIn('"Choose a part"', html)
        self.assertIn('"No approved model"', html)
        self.assertIn('"Press (Enter)"', html)

    def test_camera_not_open_message(self):
        html = self.page()
        self.assertIn('id="cam-status"', html)
        self.assertIn('"Camera not open: "', html)
        self.assertIn("The live view retries every few seconds.", html)
        self.assertLess(html.index('id="cam-status"'), html.index('id="live-grid"'))

    def test_search_debounce_and_stale_response_guard(self):
        html = self.page()
        self.assertIn("PART_SEARCH_MIN_MS = 300", html)
        self.assertIn("partSearchGen", html)
        self.assertIn("if (g !== partSearchGen)", html)

    def test_part_text_is_set_with_textcontent_only(self):
        html = self.page()
        self.assertIn("pn.textContent = row.part_number;", html)
        self.assertIn("desc.textContent", html)
        for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
            self.assertNotIn(bad, html)

    def test_part_script_passes_node_check_when_node_available(self):
        if shutil.which("node") is None:
            self.skipTest("node not available")
        html = self.page()
        m = re.search(r"<script>(.*?)</script>", html, re.S)
        self.assertIsNotNone(m)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "page.js")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(m.group(1))
            proc = subprocess.run(["node", "--check", path], capture_output=True, text=True,
                                  timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class LabelActionTests(UiTestCase):
    def test_every_logged_in_role_can_label_and_by_is_the_login(self):
        for user, pw in ((OP_USER, OP_PW), (SUP_USER, SUP_PW), (ADM_USER, ADM_PW)):
            with self.subTest(user=user):
                token = self.login(user, pw)
                reply = self.call("POST", "/api/label",
                                  {"verdict": "BAD", "criterion_id": "CR-1", "part_id": "P-7",
                                   "by": "mallory"}, token=token)
                self.assertEqual(reply.status, 200, reply.body)
                self.assertEqual(set(self.js(reply)),
                                 {"session_id", "verdict", "criterion_id", "changed"})
        calls = [kw for name, kw in self.session.calls if name == "label_part"]
        self.assertEqual([c["by"] for c in calls], [OP_USER, SUP_USER, ADM_USER])
        self.assertEqual(calls[0]["verdict"], Verdict.BAD)
        self.assertEqual(calls[0]["criterion_id"], "CR-1")
        self.assertEqual(calls[0]["part_id"], "P-7")
        self.assertNotIn("resolve_review", self.session.names())

    def test_label_needs_login(self):
        self.assertEqual(self.call("POST", "/api/label", {"verdict": "GOOD"}).status, 401)
        self.assertNotIn("label_part", self.session.names())

    def test_role_without_label_action_gets_403(self):
        token = self.login(OP_USER, OP_PW)
        with mock.patch.dict(station_auth._POLICY,
                             {"operator": frozenset({"press", "ack", "select_part"})}):
            reply = self.call("POST", "/api/label", {"verdict": "GOOD"}, token=token)
        self.assertEqual(reply.status, 403)
        self.assertNotIn("label_part", self.session.names())

    def test_bad_label_input_is_400(self):
        token = self.login(SUP_USER, SUP_PW)
        cases = [{"verdict": "REVIEW"}, {}, {"verdict": "GOOD", "criterion_id": 5},
                 {"verdict": "BAD", "criterion_id": "x" * 65}, {"verdict": "GOOD", "part_id": ""},
                 {"verdict": "GOOD", "part_id": "p" * 101}]
        for body in cases:
            with self.subTest(body=body):
                self.assertEqual(self.call("POST", "/api/label", body, token=token).status, 400)
        self.assertNotIn("label_part", self.session.names())

    def test_refusal_from_the_station_is_400(self):
        self.session.errors["label_part"] = ValueError(
            "criterion_id must be one of the configured mes.criteria_ids")
        token = self.login(OP_USER, OP_PW)
        reply = self.call("POST", "/api/label", {"verdict": "BAD", "criterion_id": "CR-9"},
                          token=token)
        self.assertEqual(reply.status, 400)

    def test_page_has_label_controls(self):
        page = self.call("GET", "/").body.decode("utf-8")
        self.assertIn('data-action="label"', page)
        self.assertIn('"/api/label"', page)
        self.assertIn('id="criterion"', page)


if __name__ == "__main__":
    unittest.main()
