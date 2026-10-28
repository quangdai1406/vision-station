"""Tests for the optional `scan` block and the Taking-data wiring in station/app.py.

Fake cameras, a marker model and a temporary data folder only. Time is injected
(FakeClock), so nothing sleeps. Plan: docs/26_1027_PLAN_taking_data_mode.md, step 3.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
import unittest

from station.app import ConfigError, build_station, validate_config
from station.auth import hash_password
from station.cameras import FakeCameraSet
from station.contracts import State
from station.inference import MarkerModel
from station.outputs import RecordingOutputs
from station.scan import ScanError, ScanState
from station.scanstore import load_scan, verify_scan

TOKEN_ENV = "VISION_TEST_SCAN_TOKEN"
GOOD_LABEL = {
    "verdict": "GOOD",
    "defects": [],
    "status": "official",
    "labelled_by": "sup1",
    "labelled_at": "2026-10-27T08:42:00Z",
    "approved_by": "admin1",
    "approved_at": "2026-10-27T09:00:00Z",
}
BAD_LABEL = {
    "verdict": "BAD",
    "defects": ["doh_small"],
    "status": "proposed",
    "labelled_by": "sup1",
    "labelled_at": "2026-10-27T08:42:00Z",
}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def base_config(tmp: str, scan=None) -> dict:
    cfg = {
        "site": "SITE1", "line": "L1", "station_id": "VIS-1", "part_number": "PN-100",
        "lot": "LOT-1", "operator": "op1", "supervisor": "sup1", "material_handler": "mh1",
        "thresholds": {"t_low": 0.5, "t_img": 0.8},
        "data_dir": os.path.join(tmp, "data"),
        "users_file": os.path.join(tmp, "users.json"),
        "ui": {"host": "127.0.0.1", "port": 0, "session_timeout_s": 900},
        "mes": {
            "base_url": "http://127.0.0.1:9",
            "token_env": TOKEN_ENV,
            "criteria_ids": ["CR-1", "CR-2"],
            "class_to_criterion": {"scratch": "CR-1"},
            "default_criterion_id": None,
            "sync_interval_s": 3600,
            "default_bad_disposition": "quarantine",
        },
        "cameras": {"kind": "fake"},
        "model": {"kind": "marker", "version": "test-1"},
        "outputs": {"kind": "console"},
    }
    if scan is not None:
        cfg["scan"] = scan
    return cfg


class ScanConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="station-scan-cfg-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_default_is_disabled_with_the_plan_values(self) -> None:
        out = validate_config(base_config(self.tmp))
        self.assertEqual(out["scan"], {"enabled": False, "min_flip_s": 2.0, "debounce_s": None})

    def test_enabled_block_keeps_its_values_and_validates_again_to_the_same_value(self) -> None:
        out = validate_config(base_config(self.tmp, {"enabled": True, "min_flip_s": 0, "debounce_s": 0.1}))
        self.assertEqual(out["scan"], {"enabled": True, "min_flip_s": 0.0, "debounce_s": 0.1})
        self.assertEqual(validate_config(out)["scan"], out["scan"])

    def test_bad_values_name_the_key(self) -> None:
        cases = [
            ({"enabled": "yes"}, "scan.enabled must be true or false"),
            ({"enabled": 1}, "scan.enabled must be true or false"),
            ({"min_flip_s": -1}, "scan.min_flip_s must be >= 0.0"),
            ({"min_flip_s": True}, "scan.min_flip_s must be a number"),
            ({"min_flip_s": "2"}, "scan.min_flip_s must be a number"),
            ({"debounce_s": "0.3"}, "scan.debounce_s must be a number"),
            ({"debounce_s": -0.1}, "scan.debounce_s must be >= 0.0"),
        ]
        for scan, message in cases:
            with self.subTest(scan=scan):
                with self.assertRaises(ConfigError) as ctx:
                    validate_config(base_config(self.tmp, scan))
                self.assertIn(message, str(ctx.exception))

    def test_unknown_key_is_refused_and_named(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            validate_config(base_config(self.tmp, {"enabled": True, "colour": "red"}))
        self.assertIn("scan: unknown key(s) ['colour']", str(ctx.exception))

    def test_block_must_be_an_object(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            validate_config(base_config(self.tmp, [True]))
        self.assertIn("scan must be a JSON object", str(ctx.exception))


class ScanAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="station-scan-app-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._saved_env = os.environ.pop(TOKEN_ENV, None)
        self.addCleanup(self._restore_env)
        with open(os.path.join(self.tmp, "users.json"), "w", encoding="utf-8") as fh:
            json.dump({"sup1": {"password_hash": hash_password("supervisor-pass-1"),
                                "role": "supervisor"}}, fh)
        self.data_dir = pathlib.Path(self.tmp) / "data"

    def _restore_env(self) -> None:
        os.environ.pop(TOKEN_ENV, None)
        if self._saved_env is not None:
            os.environ[TOKEN_ENV] = self._saved_env

    def _app(self, scan=None):
        clock = FakeClock()
        cameras = FakeCameraSet()
        app = build_station(base_config(self.tmp, scan), cameras=cameras, model=MarkerModel("test-1"),
                            outputs=RecordingOutputs(), clock=clock)
        self.addCleanup(app.stop)
        return app, clock, cameras

    def _good_scan(self, app, clock, label=None) -> pathlib.Path:
        """2 TOP steps, flip, 2 BOTTOM steps, finish. Returns the saved scan folder."""
        app.scan.start("PN-100", "TOY-1", dict(GOOD_LABEL if label is None else label), "sup1")
        app.scan.press()          # TOP step 1
        clock.now += 0.5
        app.scan.press()          # TOP step 2
        clock.now += 0.5
        self.assertEqual(app.scan.done_side(), ScanState.WAIT_FLIP.value)
        clock.now += 2.5          # past min_flip_s (2.0)
        self.assertTrue(app.scan.press())  # flip press: BOTTOM step 1
        clock.now += 0.5
        app.scan.press()          # BOTTOM step 2
        clock.now += 0.5
        self.assertEqual(app.scan.done_side(), ScanState.SAVING.value)
        return app.scan.finish()

    def test_disabled_by_default_changes_nothing(self) -> None:
        app, _, _ = self._app()
        self.assertIsNone(app.scan)
        self.assertIsNone(app._scan_session)
        self.assertFalse((self.data_dir / "scans").exists())
        self.assertFalse(app.scan_active())
        # Inspection still works exactly as before.
        self.assertTrue(app._locked.press())
        self.assertEqual(app.session.state, State.WAIT_FLIP)

    def test_explicitly_disabled_is_the_same_as_the_default(self) -> None:
        app, _, _ = self._app({"enabled": False})
        self.assertIsNone(app.scan)
        self.assertFalse((self.data_dir / "scans").exists())

    def test_enabled_builds_a_scan_session_on_the_same_cameras_and_store_root(self) -> None:
        app, _, cameras = self._app({"enabled": True})
        self.assertIsNotNone(app.scan)
        self.assertIs(app._scan_session._cameras, cameras)
        self.assertEqual(app._scan_session._store.scans_dir, self.data_dir / "scans")
        self.assertEqual(app._scan_session._store.line, "L1")
        self.assertEqual(app._scan_session._camera_ids, ("C1", "C2", "C3", "C4"))

    def test_good_scan_runs_end_to_end_and_verifies_clean(self) -> None:
        app, clock, _ = self._app({"enabled": True})
        scan_dir = self._good_scan(app, clock)
        self.assertEqual(app.scan.snapshot()["state"], ScanState.DONE.value)
        self.assertEqual(verify_scan(scan_dir), [])
        self.assertTrue(str(scan_dir).startswith(str(self.data_dir / "scans")))
        self.assertIsInstance(load_scan(scan_dir), dict)
        pictures = [p for p in scan_dir.rglob("*") if p.is_file() and p.name != "scan.json"
                    and p.name != "manifest.sha256"]
        self.assertEqual(len(pictures), 16)  # 2 TOP steps + 2 BOTTOM steps, 4 cameras each
        app.scan.reset()
        self.assertEqual(app.scan.snapshot()["state"], ScanState.IDLE.value)

    def test_scan_refused_while_an_inspection_part_is_in_progress(self) -> None:
        app, clock, _ = self._app({"enabled": True})
        self.assertTrue(app._locked.press())  # TOP captured: the part is in progress
        self.assertNotEqual(app.session.state, State.IDLE)
        with self.assertRaises(ScanError) as ctx:
            app.scan.start("PN-100", "TOY-1", dict(GOOD_LABEL), "sup1")
        self.assertIn("inspection", str(ctx.exception))
        self.assertEqual(app.scan.snapshot()["state"], ScanState.IDLE.value)
        self.assertEqual(list((self.data_dir / "scans" / "L1").iterdir()), [])  # nothing saved

    def test_busy_is_true_in_every_inspection_state_except_idle(self) -> None:
        app, _, _ = self._app({"enabled": True})
        for state in State:
            with self.subTest(state=state.value):
                app.session.state = state
                self.assertEqual(app._inspection_busy(), state is not State.IDLE)
        app.session.state = State.IDLE

    def test_inspection_press_refused_while_a_scan_is_active(self) -> None:
        app, clock, _ = self._app({"enabled": True})
        app.scan.start("PN-100", "TOY-1", dict(GOOD_LABEL), "sup1")
        self.assertTrue(app.scan_active())
        self.assertFalse(app._locked.press())
        self.assertEqual(app.session.state, State.IDLE)
        self.assertEqual(app.session.counters["total"], 0)
        # Still refused in the middle of a side, and while waiting for the flip.
        app.scan.press()
        self.assertFalse(app._locked.press())
        clock.now += 0.5
        app.scan.done_side()
        self.assertEqual(app.scan.snapshot()["state"], ScanState.WAIT_FLIP.value)
        self.assertFalse(app._locked.press())
        self.assertEqual(app.session.state, State.IDLE)

    def test_inspection_press_allowed_again_after_the_scan_is_cancelled_or_saved(self) -> None:
        app, clock, _ = self._app({"enabled": True})
        app.scan.start("PN-100", "TOY-1", dict(GOOD_LABEL), "sup1")
        app.scan.cancel()
        self.assertFalse(app.scan_active())
        self.assertTrue(app._locked.press())
        self.assertEqual(app.session.state, State.WAIT_FLIP)

        app2, clock2, _ = self._app({"enabled": True})
        self._good_scan(app2, clock2)  # DONE: the cameras are free, inspection may run
        self.assertFalse(app2.scan_active())
        self.assertTrue(app2._locked.press())

    def test_label_rules_are_enforced_by_the_store_through_the_app(self) -> None:
        app, _, _ = self._app({"enabled": True})
        # A BAD label needs at least one defect name (station/scanstore.py label rules).
        with self.assertRaises(ScanError):
            app.scan.start("PN-100", "TOY-1", {"verdict": "BAD", "defects": [], "status": "proposed",
                                               "labelled_by": "sup1",
                                               "labelled_at": "2026-10-27T08:42:00Z"}, "sup1")
        self.assertEqual(app.scan.snapshot()["state"], ScanState.IDLE.value)
        app.scan.start("PN-100", "TOY-1", dict(BAD_LABEL), "sup1")  # a valid BAD label starts
        self.assertEqual(app.scan.snapshot()["state"], ScanState.SCAN_SIDE.value)
        app.scan.cancel()

    def test_min_flip_s_from_the_config_is_used(self) -> None:
        app, clock, _ = self._app({"enabled": True, "min_flip_s": 10})
        app.scan.start("PN-100", "TOY-1", dict(GOOD_LABEL), "sup1")
        app.scan.press()
        clock.now += 0.5
        app.scan.done_side()
        clock.now += 2.5  # enough for the 2.0 default, not for 10
        self.assertFalse(app.scan.press())
        with self.assertRaises(ScanError):
            app.scan.flip_ready()

    def test_invalid_scan_block_stops_the_build_with_a_clear_message(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            build_station(base_config(self.tmp, {"enabled": True, "min_flp_s": 1}),
                          cameras=FakeCameraSet(), model=MarkerModel("test-1"),
                          outputs=RecordingOutputs())
        self.assertIn("unknown key(s) ['min_flp_s']", str(ctx.exception))
        self.assertFalse((self.data_dir / "scans").exists())

    def _http(self, app, method, path, cookie=None, body=None):
        import urllib.error
        import urllib.request
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if cookie:
            headers["Cookie"] = cookie
        req = urllib.request.Request(app.url.rstrip("/") + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as reply:
                return reply.status, reply.headers, reply.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers, exc.read()

    def test_scan_routes_are_served_by_the_real_app_over_http(self) -> None:
        # regression: the UI server must get the scan object from the app, and the scan wrapper
        # must offer step_picture, or the Taking-data screen never works on the real station
        app, _, _ = self._app({"enabled": True})
        app.start()
        status, headers, _ = self._http(app, "POST", "/api/login",
                                        body={"username": "sup1", "password": "supervisor-pass-1"})
        self.assertEqual(status, 200)
        cookie = headers["Set-Cookie"].split(";")[0]
        status, _, raw = self._http(app, "GET", "/api/scan/state", cookie)
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(raw)["may_scan"])
        status, _, _ = self._http(app, "POST", "/api/scan/start", cookie, {
            "toy_id": "TOY-1", "part_number": "PN-100",
            "label": {"verdict": "BAD", "defects": ["doh_small"]}})
        self.assertEqual(status, 200)
        status, _, _ = self._http(app, "POST", "/api/scan/press", cookie, {})
        self.assertEqual(status, 200)
        status, _, picture = self._http(app, "GET", "/api/scan/step-picture?camera=C1", cookie)
        self.assertEqual(status, 200)
        self.assertTrue(picture)
        status, _, _ = self._http(app, "POST", "/api/scan/cancel", cookie, {})
        self.assertEqual(status, 200)

    def test_pending_and_review_go_through_the_locked_wrapper(self) -> None:
        app, clock, _ = self._app({"enabled": True})
        proposed = {"verdict": "GOOD", "defects": [], "status": "proposed",
                    "labelled_by": "sup1", "labelled_at": "2026-10-27T08:42:00Z"}
        scan_dir = self._good_scan(app, clock, proposed)
        self.assertEqual([r["scan_id"] for r in app.scan.pending_reviews()], [scan_dir.name])
        label = app.scan.review(scan_dir.name, "approve", "admin1")
        self.assertEqual(label["status"], "official")
        self.assertEqual(label["approved_by"], "admin1")
        self.assertEqual(app.scan.pending_reviews(), [])
        self.assertEqual(verify_scan(scan_dir), [])

    def test_scan_routes_are_not_served_when_disabled(self) -> None:
        app, _, _ = self._app()
        app.start()
        status, headers, _ = self._http(app, "POST", "/api/login",
                                        body={"username": "sup1", "password": "supervisor-pass-1"})
        self.assertEqual(status, 200)
        cookie = headers["Set-Cookie"].split(";")[0]
        status, _, _ = self._http(app, "GET", "/api/scan/state", cookie)
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
