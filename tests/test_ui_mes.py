"""mes_ticket mode: station.ui routes, session identity and the press rule. Local server, fakes only."""
import http.client
import json
import threading
import unittest

from station.auth import SessionManager, UserStore
from station.contracts import State, Verdict
from station.mes_session import MesSessionError, SessionInfo
from station.ui import SESSION_COOKIE, make_server
from tests.test_session import Harness
from tests.test_ui import FakeSession

ROLE_MAP = {"inspector": "operator", "supervisor": "supervisor"}
SID = "sess-anna-1"
TICKET = "ticket-only-for-tests-1"
INFO = SessionInfo(session_id=SID, username="anna.k", roles=("inspector",),
                   expires_utc="2026-11-04T09:00:00Z")


class FakeMesSessions:
    """Stands in for MesSessionClient. Records calls; the scripted results drive each case."""

    def __init__(self):
        self.redeem_result = INFO
        self.redeem_error = None
        self.validate_result = INFO
        self.validate_error = None
        self.ended = []
        self.calls = []

    def redeem(self, ticket):
        self.calls.append(("redeem", ticket))
        if self.redeem_error is not None:
            raise self.redeem_error
        return self.redeem_result

    def validate(self, session_id):
        self.calls.append(("validate", session_id))
        if self.validate_error is not None:
            raise self.validate_error
        return self.validate_result

    def end(self, session_id):
        self.calls.append(("end", session_id))
        self.ended.append(session_id)


class RecordingSession(FakeSession):
    """FakeSession whose press() records who pressed and the MES session."""

    def __init__(self):
        super().__init__()
        self.press_kwargs = []

    def press(self, **kwargs):
        self.press_kwargs.append(kwargs)
        return super().press()


class MesUiCase(unittest.TestCase):
    def setUp(self):
        self.station = RecordingSession()
        self.mes = FakeMesSessions()
        self.sessions = SessionManager(timeout_s=900)
        self.server = make_server(self.station, UserStore({}), self.sessions, port=0,
                                  auth_mode="mes_ticket", mes_auth=self.mes, role_map=ROLE_MAP)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def req(self, method, path, body=None, cookie=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if cookie:
            headers["Cookie"] = f"{SESSION_COOKIE}={cookie}"
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        out = (resp.status, resp.getheaders(), raw)
        conn.close()
        return out

    def enter(self, ticket=TICKET):
        status, headers, raw = self.req("POST", "/api/enter", {"ticket": ticket})
        cookie = None
        for name, value in headers:
            if name.lower() == "set-cookie" and value.startswith(SESSION_COOKIE + "="):
                cookie = value.split(";", 1)[0].split("=", 1)[1]
        return status, cookie, raw


class EnterPageTests(MesUiCase):
    def test_enter_page_reads_fragment_and_posts_ticket(self):
        status, _, raw = self.req("GET", "/enter")
        self.assertEqual(status, 200)
        text = raw.decode("utf-8")
        self.assertIn("location.hash", text)
        self.assertIn("/api/enter", text)
        self.assertNotIn(TICKET, text)

    def test_ticket_in_query_string_is_rejected(self):
        status, _, raw = self.req("GET", "/enter?ticket=" + TICKET)
        self.assertEqual(status, 400)
        self.assertNotIn(TICKET.encode(), raw)


class EnterTests(MesUiCase):
    def test_redeem_sets_cookie_and_maps_role(self):
        status, cookie, raw = self.enter()
        self.assertEqual(status, 200)
        self.assertTrue(cookie)
        self.assertNotIn(TICKET.encode(), raw)
        self.assertEqual(self.mes.calls[0], ("redeem", TICKET))
        status, _, raw = self.req("GET", "/api/state", cookie=cookie)
        self.assertEqual(status, 200)
        auth = json.loads(raw)["auth"]
        self.assertEqual(auth["username"], "anna.k")
        self.assertEqual(auth["role"], "operator")

    def test_cookie_is_http_only_and_same_site(self):
        _, headers, _ = self.req("POST", "/api/enter", {"ticket": TICKET})
        set_cookie = [v for k, v in headers if k.lower() == "set-cookie"][0]
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("SameSite=", set_cookie)

    def test_replayed_ticket_is_409(self):
        self.mes.redeem_error = MesSessionError("MES refused the ticket redeem", status=409)
        status, cookie, raw = self.enter()
        self.assertEqual(status, 409)
        self.assertIsNone(cookie)
        self.assertNotIn(TICKET.encode(), raw)

    def test_unavailable_mes_is_503(self):
        self.mes.redeem_error = MesSessionError("could not reach MES for ticket redeem: refused")
        status, cookie, _ = self.enter()
        self.assertEqual(status, 503)
        self.assertIsNone(cookie)

    def test_unmapped_role_is_403_and_the_mes_session_is_ended(self):
        self.mes.redeem_result = SessionInfo(session_id=SID, username="visitor.v",
                                             roles=("visitor",), expires_utc="x")
        status, cookie, _ = self.enter()
        self.assertEqual(status, 403)
        self.assertIsNone(cookie)
        self.assertEqual(self.mes.ended, [SID])

    def test_ticket_in_query_string_is_refused_on_post_too(self):
        status, _, _ = self.req("POST", "/api/enter?ticket=" + TICKET, {"ticket": TICKET})
        self.assertEqual(status, 400)
        self.assertEqual(self.mes.calls, [])

    def test_bad_ticket_shape_is_400_before_redeem(self):
        status, _, _ = self.req("POST", "/api/enter", {"ticket": ""})
        self.assertEqual(status, 400)
        status, _, _ = self.req("POST", "/api/enter", {})
        self.assertEqual(status, 400)
        self.assertEqual(self.mes.calls, [])


class LoginRefusedTests(MesUiCase):
    def test_local_password_login_is_refused(self):
        status, headers, _ = self.req("POST", "/api/login",
                                      {"username": "anna.k", "password": "whatever-1"})
        self.assertEqual(status, 403)
        self.assertFalse([v for k, v in headers if k.lower() == "set-cookie"])


class ValidationTests(MesUiCase):
    def test_each_poll_validates_with_the_mes(self):
        _, cookie, _ = self.enter()
        self.req("GET", "/api/state", cookie=cookie)
        self.assertIn(("validate", SID), self.mes.calls)

    def test_ended_session_is_401_with_reason_and_the_station_token_is_dropped(self):
        _, cookie, _ = self.enter()
        self.mes.validate_result = None
        status, _, raw = self.req("GET", "/api/state", cookie=cookie)
        self.assertEqual(status, 401)
        body = json.loads(raw)
        self.assertEqual(body["reason"], "mes_session_ended")
        self.assertIn("Log in again from the MES", body["error"])
        self.assertIsNone(self.sessions.get(cookie))

    def test_mes_unreachable_during_validate_is_503_and_the_session_survives(self):
        _, cookie, _ = self.enter()
        self.mes.validate_error = MesSessionError("could not reach MES for session check")
        status, _, _ = self.req("GET", "/api/state", cookie=cookie)
        self.assertEqual(status, 503)
        self.mes.validate_error = None
        status, _, _ = self.req("GET", "/api/state", cookie=cookie)
        self.assertEqual(status, 200)

    def test_logout_ends_the_mes_session(self):
        _, cookie, _ = self.enter()
        status, _, _ = self.req("POST", "/api/logout", {}, cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(self.mes.ended, [SID])
        self.assertIsNone(self.sessions.get(cookie))

    def test_page_is_marked_for_mes_mode(self):
        status, _, raw = self.req("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b'data-auth="mes"', raw)


class PressIdentityTests(MesUiCase):
    def test_press_passes_the_inspector_and_the_mes_session(self):
        _, cookie, _ = self.enter()
        status, _, _ = self.req("POST", "/api/press", {}, cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(self.station.press_kwargs,
                         [{"by": "anna.k", "station_session": SID}])


class LocalModeTests(unittest.TestCase):
    def test_local_mode_has_no_enter_route_and_no_mes_marker(self):
        station = RecordingSession()
        server = make_server(station, UserStore({}), SessionManager(900), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            for method, path, expect in (("GET", "/enter", 404), ("POST", "/api/enter", 404),
                                         ("GET", "/", 200)):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request(method, path, body=b"{}" if method == "POST" else None,
                             headers={"Content-Type": "application/json"})
                resp = conn.getresponse()
                page = resp.read()
                conn.close()
                self.assertEqual(resp.status, expect, path)
                if path == "/":
                    self.assertNotIn(b'data-auth="mes"', page)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)


class SessionIdentityTests(unittest.TestCase):
    """The press rule in StationSession: the part is recorded under the inspector who started it."""

    def start_part(self, h, by="anna.k", sid="sess-anna-1"):
        self.assertTrue(h.session.press(by=by, station_session=sid))
        self.assertEqual(h.session.state, State.WAIT_FLIP)

    def test_same_inspector_finishes_the_part_under_their_name(self):
        h = Harness(fuse_verdict=Verdict.GOOD)
        self.start_part(h)
        h.clock.t = 1.0
        self.assertTrue(h.session.press(by="anna.k", station_session="sess-anna-2"))
        self.assertEqual(h.session.state, State.IDLE)
        record = h.store.saved[0]
        self.assertEqual(record.operator, "anna.k")
        self.assertEqual(h.session.station_session_for(record.part_id), "sess-anna-1")
        self.assertIsNone(h.session.station_session_for(record.part_id))

    def test_different_inspector_at_the_flip_is_a_fault_and_nothing_is_saved(self):
        h = Harness(fuse_verdict=Verdict.GOOD)
        self.start_part(h)
        h.clock.t = 1.0
        self.assertFalse(h.session.press(by="bob.s", station_session="sess-bob-1"))
        self.assertEqual(h.session.state, State.FAULT)
        self.assertEqual(h.store.saved, [])
        self.assertEqual(h.completed, [])
        self.assertIn("fault", h.store.kinds())
        self.assertIn("another inspector", h.session.snapshot()["fault"])
        h.session.clear_fault()
        self.assertEqual(h.session.state, State.IDLE)
        self.assertEqual(h.store.saved, [])

    def test_local_mode_keeps_the_configured_operator(self):
        h = Harness(fuse_verdict=Verdict.GOOD)
        h.complete_part(0.0)
        self.assertEqual(h.store.saved[0].operator, "op")
        self.assertIsNone(h.session.station_session_for(h.store.saved[0].part_id))


if __name__ == "__main__":
    unittest.main()
