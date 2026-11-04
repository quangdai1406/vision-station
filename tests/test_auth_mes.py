"""Tests for the mes_ticket additions in station/auth.py: role mapping and MES-bound sessions."""
import copy
import os
import sqlite3
import unittest

from station.app import ConfigError, build_station, validate_config
from station.auth import (
    AUTH_MODES,
    SessionManager,
    map_role,
    validate_role_map,
)

from station.cameras import FakeCameraSet
from station.inference import MarkerModel
from station.outputs import RecordingOutputs
from tests.test_app import PAST_MIN_FLIP, TOKEN_ENV, TmpCase, base_config
from tests.test_app import FakeClock as AppClock

ROLE_MAP = {"inspector": "operator", "supervisor": "supervisor"}
MES_AUTH = {"mode": "mes_ticket", "role_map": ROLE_MAP}


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class RoleMapTests(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(AUTH_MODES, ("local", "mes_ticket"))

    def test_mapped_role_is_returned(self):
        self.assertEqual(map_role(ROLE_MAP, ["inspector"]), "operator")
        self.assertEqual(map_role(ROLE_MAP, ["supervisor"]), "supervisor")

    def test_unmapped_role_gets_none(self):
        self.assertIsNone(map_role(ROLE_MAP, ["visitor"]))
        self.assertIsNone(map_role(ROLE_MAP, []))

    def test_highest_mapped_role_wins_when_several_match(self):
        self.assertEqual(map_role(ROLE_MAP, ["inspector", "supervisor"]), "supervisor")

    def test_admin_is_only_given_by_the_map(self):
        self.assertIsNone(map_role(ROLE_MAP, ["admin"]))  # not in the map: no access
        self.assertEqual(map_role({"lead": "admin"}, ["lead"]), "admin")

    def test_validate_accepts_good_map(self):
        self.assertEqual(validate_role_map(ROLE_MAP), ROLE_MAP)

    def test_validate_rejects_bad_maps(self):
        for bad in (None, {}, [], {"inspector": "root"}, {"": "operator"},
                    {"inspector": 3}, {"inspector": ""}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_role_map(bad)


class MesBoundSessionTests(unittest.TestCase):
    def test_mes_session_is_stored_and_read_without_refresh(self):
        clock = FakeClock(0.0)
        sm = SessionManager(60, clock=clock)
        token = sm.create("anna.k", "operator", mes_session="sess-1")
        self.assertEqual(sm.mes_session(token), "sess-1")
        self.assertEqual(sm.get(token), ("anna.k", "operator"))

    def test_plain_session_has_no_mes_session(self):
        sm = SessionManager(60, clock=AppClock())
        token = sm.create("alice", "operator")
        self.assertIsNone(sm.mes_session(token))
        self.assertIsNone(sm.mes_session("unknown"))

    def test_expiry_calls_on_expire_once_with_the_mes_session(self):
        clock = FakeClock(0.0)
        seen = []
        sm = SessionManager(60, clock=clock,
                            on_expire=lambda token, user, sid: seen.append((user, sid)))
        token = sm.create("anna.k", "operator", mes_session="sess-1")
        clock.t = 60.0
        self.assertIsNone(sm.get(token))
        self.assertIsNone(sm.get(token))
        self.assertEqual(seen, [("anna.k", "sess-1")])

    def test_prune_during_create_also_calls_on_expire(self):
        clock = FakeClock(0.0)
        seen = []
        sm = SessionManager(60, clock=clock,
                            on_expire=lambda token, user, sid: seen.append(sid))
        sm.create("anna.k", "operator", mes_session="sess-1")
        clock.t = 100.0
        sm.create("bob", "supervisor", mes_session="sess-2")
        self.assertEqual(seen, ["sess-1"])

    def test_destroy_does_not_call_on_expire(self):
        seen = []
        sm = SessionManager(60, clock=FakeClock(),
                            on_expire=lambda token, user, sid: seen.append(sid))
        token = sm.create("anna.k", "operator", mes_session="sess-1")
        sm.destroy(token)
        self.assertEqual(seen, [])


class AppConfigTests(TmpCase):
    def cfg(self, auth=None, *, with_auth=True):
        cfg = base_config(self.tmp)
        if with_auth:
            cfg["auth"] = auth
        return cfg

    def test_default_auth_is_local(self):
        out = validate_config(self.cfg(with_auth=False))
        self.assertEqual(out["auth"], {"mode": "local", "role_map": None})

    def test_mes_ticket_with_role_map_is_accepted(self):
        out = validate_config(self.cfg(dict(MES_AUTH)))
        self.assertEqual(out["auth"], {"mode": "mes_ticket", "role_map": ROLE_MAP})

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ConfigError):
            validate_config(self.cfg({"mode": "ldap"}))

    def test_mes_ticket_needs_a_role_map(self):
        with self.assertRaises(ConfigError):
            validate_config(self.cfg({"mode": "mes_ticket"}))
        with self.assertRaises(ConfigError):
            validate_config(self.cfg({"mode": "mes_ticket", "role_map": {}}))

    def test_role_map_must_name_station_roles(self):
        with self.assertRaises(ConfigError):
            validate_config(self.cfg({"mode": "mes_ticket", "role_map": {"inspector": "root"}}))

    def test_role_map_in_local_mode_is_refused(self):
        with self.assertRaises(ConfigError):
            validate_config(self.cfg({"mode": "local", "role_map": ROLE_MAP}))

    def test_unknown_auth_key_is_refused(self):
        with self.assertRaises(ConfigError):
            validate_config(self.cfg({"mode": "local", "password_min": 8}))


class AppBuildTests(TmpCase):
    def build(self, token="not-a-real-token-for-tests", auth=None):
        if token is not None:
            os.environ[TOKEN_ENV] = token
        cfg = base_config(self.tmp)
        cfg["auth"] = copy.deepcopy(auth) if auth else copy.deepcopy(MES_AUTH)
        clock = AppClock()
        app = build_station(cfg, cameras=FakeCameraSet(), model=MarkerModel("test-1"),
                            outputs=RecordingOutputs(), clock=clock)
        self.addCleanup(app.stop)
        return app, clock

    def test_mes_ticket_build_wires_the_handler(self):
        app, _ = self.build()
        handler = app.server.RequestHandlerClass
        self.assertEqual(handler.auth_mode, "mes_ticket")
        self.assertEqual(handler.role_map, ROLE_MAP)
        self.assertIsNotNone(handler.mes_auth)

    def test_mes_ticket_without_token_is_a_config_error_that_names_no_value(self):
        os.environ.pop(TOKEN_ENV, None)
        cfg = base_config(self.tmp)
        cfg["auth"] = copy.deepcopy(MES_AUTH)
        with self.assertRaises(ConfigError) as ctx:
            build_station(cfg, cameras=FakeCameraSet(), model=MarkerModel("test-1"),
                          outputs=RecordingOutputs(), clock=AppClock())
        self.assertIn(TOKEN_ENV, str(ctx.exception))

    def test_demo_cannot_use_mes_ticket(self):
        cfg = base_config(self.tmp)
        cfg["auth"] = copy.deepcopy(MES_AUTH)
        with self.assertRaises(ConfigError):
            build_station(cfg, cameras=FakeCameraSet(), model=MarkerModel("test-1"),
                          outputs=RecordingOutputs(), clock=AppClock(), demo=True)

    def test_inspector_name_and_mes_session_travel_with_the_queued_post(self):
        app, clock = self.build()
        self.assertTrue(app.session.press(by="anna.k", station_session="sess-anna-1"))
        clock.now += PAST_MIN_FLIP
        self.assertTrue(app.session.press(by="anna.k", station_session="sess-anna-2"))
        self.assertEqual(app.outbox.pending_count(), 1)
        row = app.outbox._db.execute("SELECT station_session, fields_json FROM items").fetchone()
        self.assertEqual(row[0], "sess-anna-1")
        self.assertIn('"operators": "anna.k"', row[1])

    def test_a_part_with_no_mes_session_is_never_queued(self):
        app, clock = self.build()
        self.assertTrue(app.session.press())
        clock.now += PAST_MIN_FLIP
        self.assertTrue(app.session.press())
        self.assertEqual(app.outbox.pending_count(), 0)
        conn = sqlite3.connect(str(app.store.db_path))
        try:
            kinds = [r[0] for r in conn.execute("SELECT kind FROM events").fetchall()]
        finally:
            conn.close()
        self.assertIn("mes_enqueue_failed", kinds)


if __name__ == "__main__":
    unittest.main()
