"""Unit tests for station/app.py, station/users_cli.py and station/__main__.py."""
from __future__ import annotations

import copy
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from unittest import mock

from station import users_cli
from station.app import (
    ConfigError,
    StationApp,
    _LockedSession,
    _PartIdFactory,
    build_station,
    load_config,
    validate_config,
)
from station.auth import UserStore, hash_password, load_users
from station.cameras import FakeCameraSet
from station.contracts import CameraError, Frame, Side, StationError, State, is_full_station
from station.inference import MarkerModel
from station.outputs import RecordingOutputs
from station.storage import Store

try:
    import numpy  # noqa: F401 - only for the golden model tests
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

REPO = pathlib.Path(__file__).resolve().parent.parent
TOKEN_ENV = "VISION_TEST_APP_TOKEN"
# A gap between the TOP and BOTTOM press that is longer than the 2.0 s default min_flip_s.
PAST_MIN_FLIP = 2.5


def base_config(tmp: str) -> dict:
    return {
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
        "flip_timeout_s": 60,
    }


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="station-app-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._saved_env = os.environ.pop(TOKEN_ENV, None)
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        os.environ.pop(TOKEN_ENV, None)
        if self._saved_env is not None:
            os.environ[TOKEN_ENV] = self._saved_env

    def write_config(self, data: dict, name: str = "config.json") -> str:
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return path


class ConfigTests(TmpCase):
    def test_valid_config_loads_and_fills_defaults(self) -> None:
        data = base_config(self.tmp)
        del data["ui"]
        cfg = load_config(self.write_config(data))
        self.assertEqual(cfg["ui"]["host"], "127.0.0.1")
        self.assertEqual(cfg["ui"]["port"], 8000)
        self.assertEqual(cfg["thresholds"], {"t_low": 0.5, "t_img": 0.8})

    def test_relative_paths_are_resolved_against_the_config_folder(self) -> None:
        data = base_config(self.tmp)
        data["data_dir"] = "./data"
        data["users_file"] = "users.json"
        cfg = load_config(self.write_config(data))
        self.assertEqual(cfg["data_dir"], os.path.join(os.path.realpath(self.tmp), "data"))
        self.assertTrue(os.path.isabs(cfg["users_file"]))

    def test_missing_thresholds_is_rejected(self) -> None:
        data = base_config(self.tmp)
        del data["thresholds"]
        with self.assertRaisesRegex(ConfigError, "thresholds"):
            load_config(self.write_config(data))

    def test_thresholds_out_of_order_or_out_of_range_are_rejected(self) -> None:
        for bad in ({"t_low": 0.9, "t_img": 0.8}, {"t_low": -0.1, "t_img": 0.8},
                    {"t_low": 0.1, "t_img": 1.5}, {"t_low": 0.1}):
            data = base_config(self.tmp)
            data["thresholds"] = bad
            with self.assertRaises(ConfigError, msg=str(bad)):
                validate_config(data)

    def test_min_flip_s_defaults_to_2_and_is_validated(self) -> None:
        data = base_config(self.tmp)  # no min_flip_s key: an existing config keeps working
        self.assertEqual(validate_config(data)["min_flip_s"], 2.0)
        data["min_flip_s"] = 0
        self.assertEqual(validate_config(data)["min_flip_s"], 0.0)
        data["min_flip_s"] = 0.75
        self.assertEqual(validate_config(data)["min_flip_s"], 0.75)
        for bad in (-0.5, "2", True, None):
            data["min_flip_s"] = bad
            with self.assertRaisesRegex(ConfigError, "min_flip_s"):
                validate_config(data)

    def test_unknown_top_level_key_is_rejected(self) -> None:
        data = base_config(self.tmp)
        data["thresold"] = {}
        with self.assertRaisesRegex(ConfigError, "unknown key"):
            validate_config(data)

    def test_missing_required_section_is_rejected(self) -> None:
        data = base_config(self.tmp)
        del data["mes"]
        with self.assertRaisesRegex(ConfigError, "mes"):
            validate_config(data)

    def test_placeholder_values_are_rejected(self) -> None:
        data = base_config(self.tmp)
        data["lot"] = "[LOT NUMBER]"
        with self.assertRaisesRegex(ConfigError, "placeholder.*lot"):
            validate_config(data)

    def test_token_value_instead_of_env_name_is_rejected(self) -> None:
        data = base_config(self.tmp)
        data["mes"]["token_env"] = "mtk_live_9f8e7d6c5b4a"
        with self.assertRaisesRegex(ConfigError, "NAME of an environment variable"):
            validate_config(data)

    def test_mes_section_rules_are_enforced(self) -> None:
        data = base_config(self.tmp)
        data["mes"]["base_url"] = "ftp://mes"
        with self.assertRaisesRegex(ConfigError, "http"):
            validate_config(data)
        data = base_config(self.tmp)
        data["mes"]["class_to_criterion"] = {"scratch": "CR-9"}
        with self.assertRaisesRegex(ConfigError, "criteria_ids"):
            validate_config(data)
        data = base_config(self.tmp)
        data["mes"]["default_bad_disposition"] = "scrap"
        with self.assertRaisesRegex(ConfigError, "disposition"):
            validate_config(data)

    def test_camera_and_model_rules(self) -> None:
        data = base_config(self.tmp)
        data["cameras"] = {"kind": "opencv", "devices": [0, 1, 2, 3, 4]}
        with self.assertRaisesRegex(ConfigError, "4 device"):
            validate_config(data)
        data = base_config(self.tmp)
        data["model"] = {"kind": "onnx", "version": "v1"}
        with self.assertRaisesRegex(ConfigError, "model.path"):
            validate_config(data)
        data = base_config(self.tmp)
        data["outputs"] = {"kind": "serial_relay"}
        with self.assertRaisesRegex(ConfigError, "outputs.port"):
            validate_config(data)

    def test_bad_json_and_missing_file_give_clear_errors(self) -> None:
        bad = os.path.join(self.tmp, "bad.json")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        with self.assertRaisesRegex(ConfigError, "not valid JSON"):
            load_config(bad)
        with self.assertRaisesRegex(ConfigError, "cannot read config"):
            load_config(os.path.join(self.tmp, "absent.json"))

    def test_example_config_is_valid_except_for_placeholders(self) -> None:
        example = json.loads((REPO / "config.example.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ConfigError, "placeholders"):
            validate_config(example)
        text = (REPO / "config.example.json").read_text(encoding="utf-8")
        self.assertNotIn("token\":", text.lower())  # no token value anywhere in the example
        self.assertTrue(example["mes"]["token_env"].startswith("["))
        self.assertEqual(example["min_flip_s"], 2.0)  # an engineering default, not a placeholder


class PartIdTests(TmpCase):
    def _store(self) -> Store:
        return Store(pathlib.Path(self.tmp) / "data", site="SITE1", line="L1")

    def test_ids_are_unique_and_counted_per_second(self) -> None:
        factory = _PartIdFactory(self._store(), lambda: "2026-10-23T08:30:12Z")
        ids = [factory() for _ in range(50)]
        self.assertEqual(len(set(ids)), 50)
        self.assertEqual(ids[0], "P202610230830120001")
        self.assertEqual(ids[1], "P202610230830120002")
        for pid in ids:
            self.assertRegex(pid, r"^P\d{14}\d{4}$")

    def test_counter_restarts_in_a_new_second_and_never_repeats(self) -> None:
        times = iter(["2026-10-23T08:30:12Z", "2026-10-23T08:30:12Z", "2026-10-23T08:30:13Z",
                      "2026-10-23T08:30:12Z"])
        factory = _PartIdFactory(self._store(), lambda: next(times))
        ids = [factory() for _ in range(4)]
        self.assertEqual(ids[2], "P202610230830130001")
        self.assertEqual(ids[3], "P202610230830120003")  # clock went back: still no repeat
        self.assertEqual(len(set(ids)), 4)

    def test_id_already_on_disk_is_skipped(self) -> None:
        store = self._store()
        taken = store.store_dir / "L1" / "2026" / "10" / "23" / "P202610230830120001"
        taken.mkdir(parents=True)
        factory = _PartIdFactory(store, lambda: "2026-10-23T08:30:12Z")
        self.assertEqual(factory(), "P202610230830120002")


class BuildTests(TmpCase):
    def setUp(self) -> None:
        super().setUp()
        with open(os.path.join(self.tmp, "users.json"), "w", encoding="utf-8") as fh:
            json.dump({"op1": {"password_hash": hash_password("operator-pass-1"), "role": "operator"}}, fh)

    def test_mes_is_disabled_without_token_but_parts_are_still_inspected(self) -> None:
        cfg = base_config(self.tmp)
        clock = FakeClock()
        cams = FakeCameraSet()
        app = build_station(cfg, cameras=cams, model=MarkerModel("test-1"),
                            outputs=RecordingOutputs(), clock=clock)
        self.addCleanup(app.stop)
        self.assertFalse(app.mes_enabled)
        self.assertEqual(self._event_kinds(app.store), ["mes_disabled"])
        self.assertEqual(app.sync_once(), {"sent": 0, "failed_permanent": 0, "retry_later": 0})

        self.assertTrue(app.session.press())
        clock.now += PAST_MIN_FLIP
        self.assertTrue(app.session.press())
        self.assertEqual(app.session.counters["total"], 1)
        self.assertEqual(app.outbox.pending_count(), 1)  # queued for later, not sent

    def test_token_from_environment_enables_mes(self) -> None:
        os.environ[TOKEN_ENV] = "not-a-real-token-for-tests"
        app = build_station(base_config(self.tmp), cameras=FakeCameraSet(),
                            model=MarkerModel("test-1"), outputs=RecordingOutputs(),
                            clock=FakeClock())
        self.addCleanup(app.stop)
        self.assertTrue(app.mes_enabled)
        self.assertNotIn("mes_disabled", self._event_kinds(app.store))

    def test_onnx_without_injected_model_is_a_clear_error(self) -> None:
        cfg = base_config(self.tmp)
        cfg["model"] = {"kind": "onnx", "path": os.path.join(self.tmp, "m.onnx"), "version": "v1"}
        with self.assertRaisesRegex(Exception, "not wired yet"):
            build_station(cfg, cameras=FakeCameraSet(), outputs=RecordingOutputs())

    def test_missing_users_file_is_a_clear_error(self) -> None:
        os.remove(os.path.join(self.tmp, "users.json"))
        with self.assertRaisesRegex(Exception, "users file not found"):
            build_station(base_config(self.tmp), cameras=FakeCameraSet(),
                          model=MarkerModel("test-1"), outputs=RecordingOutputs())

    def test_enqueue_failure_is_logged_and_does_not_stop_the_session(self) -> None:
        # The defect class has no criterion mapping and there is no default, so
        # build_request raises MappingError when the part is posted.
        cfg = base_config(self.tmp)
        cfg["mes"]["class_to_criterion"] = {}
        marker = b"SCORE:0.92;CLASS:dent;BOX:1,2,3,4"

        def frame_bytes(cam, side, n):
            return marker + b" |" + cam.encode() + b"|" + side.value.encode()

        clock = FakeClock()
        os.environ[TOKEN_ENV] = "not-a-real-token-for-tests"
        app = build_station(cfg, cameras=FakeCameraSet(frame_bytes=frame_bytes),
                            model=MarkerModel("test-1"), outputs=RecordingOutputs(), clock=clock)
        self.addCleanup(app.stop)
        app.session.press()
        clock.now += PAST_MIN_FLIP
        app.session.press()
        self.assertEqual(app.session.snapshot()["state"], State.VERDICT.value)
        app.session.ack_alarm()  # must not raise even though the MES mapping fails

        self.assertEqual(app.session.counters, {"total": 1, "good": 0, "bad": 1})
        self.assertEqual(app.outbox.pending_count(), 0)
        rows = self._events(app.store, "mes_enqueue_failed")
        self.assertEqual(len(rows), 1)
        self.assertIn("dent", rows[0]["payload_json"])
        self.assertNotIn(os.environ[TOKEN_ENV], rows[0]["payload_json"])

    def test_restart_keeps_counters_and_unsent_outbox_items(self) -> None:
        cfg = base_config(self.tmp)
        clock = FakeClock()
        app1 = build_station(cfg, cameras=FakeCameraSet(), model=MarkerModel("test-1"),
                             outputs=RecordingOutputs(), clock=clock)
        app1.session.press()
        clock.now += PAST_MIN_FLIP
        app1.session.press()
        app1.stop()

        os.environ[TOKEN_ENV] = "not-a-real-token-for-tests"
        app2 = build_station(cfg, cameras=FakeCameraSet(), model=MarkerModel("test-1"),
                             outputs=RecordingOutputs(), clock=clock)
        self.addCleanup(app2.stop)
        self.assertEqual(app2.session.counters, {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(app2.outbox.pending_count(), 1)
        self.assertEqual(app2.session.snapshot()["state"], State.IDLE.value)

        # The next part gets a new id and session, and nothing is lost.
        clock.now += PAST_MIN_FLIP
        app2.session.press()
        clock.now += PAST_MIN_FLIP
        app2.session.press()
        self.assertEqual(app2.session.counters["total"], 2)
        self.assertEqual(app2.outbox.pending_count(), 2)

    def test_start_serves_the_ui_and_stop_is_clean(self) -> None:
        app = build_station(base_config(self.tmp), cameras=FakeCameraSet(),
                            model=MarkerModel("test-1"), outputs=RecordingOutputs(),
                            clock=FakeClock())
        app.start()
        try:
            with urllib.request.urlopen(app.url, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn(b"Vision Station", resp.read())
        finally:
            app.stop()
        app.stop()  # second stop is a no-op
        self.assertFalse(app._ui_thread.is_alive())

    def test_sync_thread_drains_on_its_interval(self) -> None:
        cfg = base_config(self.tmp)  # MES at 127.0.0.1:9 refuses connections: a retryable failure
        cfg["mes"]["sync_interval_s"] = 0.05
        os.environ[TOKEN_ENV] = "not-a-real-token-for-tests"
        clock = FakeClock()
        app = build_station(cfg, cameras=FakeCameraSet(), model=MarkerModel("test-1"),
                            outputs=RecordingOutputs(), clock=clock)
        self.addCleanup(app.stop)
        app.session.press()
        clock.now += PAST_MIN_FLIP
        app.session.press()
        self.assertEqual(app.outbox.pending_count(), 1)
        app.start()
        import time as _t

        deadline = _t.monotonic() + 5
        rows: list = []
        while _t.monotonic() < deadline and not rows:
            rows = self._events(app.store, "mes_sync")
            _t.sleep(0.02)
        app.stop()
        self.assertTrue(rows, "the sync thread never ran")
        self.assertIn('"retry_later": 1', rows[0]["payload_json"])
        self.assertFalse(app._sync_thread.is_alive())

    def test_demo_station_never_queues_parts_for_mes(self) -> None:
        from station.__main__ import build_demo_station

        cfg = base_config(self.tmp)
        cfg["min_flip_s"] = 0.0  # real clock here: a 2 s flip gap would make the test wait 2 s
        os.environ[TOKEN_ENV] = "not-a-real-token-for-tests"
        app = build_demo_station(cfg)
        self.addCleanup(app.stop)
        self.assertFalse(app.mes_enabled)
        app.session.press()  # demo uses the real monotonic clock, so wait out the 0.3 s debounce
        import time as _t
        _t.sleep(0.35)
        app.session.press()
        self.assertEqual(app.session.counters["total"], 1)
        self.assertEqual(app.outbox.pending_count(), 0)

    # helpers
    @staticmethod
    def _events(store: Store, kind: str) -> list[dict]:
        import sqlite3

        conn = sqlite3.connect(str(store.db_path))
        try:
            rows = conn.execute("SELECT kind, payload_json FROM events WHERE kind = ?", (kind,)).fetchall()
        finally:
            conn.close()
        return [{"kind": k, "payload_json": p} for k, p in rows]

    @staticmethod
    def _event_kinds(store: Store) -> list[str]:
        import sqlite3

        conn = sqlite3.connect(str(store.db_path))
        try:
            return [r[0] for r in conn.execute("SELECT kind FROM events ORDER BY id").fetchall()]
        finally:
            conn.close()


class CameraConfigTests(TmpCase):
    def test_one_opencv_device_is_a_partial_station(self) -> None:
        data = base_config(self.tmp)
        data["cameras"] = {"kind": "opencv", "devices": [0]}
        cfg = validate_config(data)
        self.assertEqual(cfg["cameras"]["camera_ids"], ["C1"])
        self.assertFalse(is_full_station(cfg["cameras"]["camera_ids"]))

    def test_four_opencv_devices_are_the_full_station(self) -> None:
        data = base_config(self.tmp)
        data["cameras"] = {"kind": "opencv", "devices": [0, 1, 2, 3]}
        cfg = validate_config(data)
        self.assertEqual(cfg["cameras"]["camera_ids"], ["C1", "C2", "C3", "C4"])
        self.assertTrue(is_full_station(cfg["cameras"]["camera_ids"]))

    def test_zero_or_five_devices_are_rejected(self) -> None:
        for devices in ([], [0, 1, 2, 3, 4]):
            data = base_config(self.tmp)
            data["cameras"] = {"kind": "opencv", "devices": devices}
            with self.assertRaisesRegex(ConfigError, "1 to 4 device"):
                validate_config(data)

    def test_duplicate_device_indexes_are_rejected(self) -> None:
        data = base_config(self.tmp)
        data["cameras"] = {"kind": "opencv", "devices": [0, 0]}
        with self.assertRaisesRegex(ConfigError, "duplicate"):
            validate_config(data)

    def test_fake_camera_count_selects_the_camera_ids(self) -> None:
        for count, ids in ((1, ["C1"]), (2, ["C1", "C2"]), (4, ["C1", "C2", "C3", "C4"])):
            data = base_config(self.tmp)
            data["cameras"] = {"kind": "fake", "camera_count": count}
            self.assertEqual(validate_config(data)["cameras"]["camera_ids"], ids)
        data = base_config(self.tmp)  # default is the full 4-camera station
        self.assertEqual(validate_config(data)["cameras"]["camera_ids"], ["C1", "C2", "C3", "C4"])

    def test_fake_camera_count_out_of_range_and_opencv_count_are_rejected(self) -> None:
        for count in (0, 5):
            data = base_config(self.tmp)
            data["cameras"] = {"kind": "fake", "camera_count": count}
            with self.assertRaisesRegex(ConfigError, "camera_count"):
                validate_config(data)
        data = base_config(self.tmp)
        data["cameras"] = {"kind": "opencv", "devices": [0], "camera_count": 1}
        with self.assertRaisesRegex(ConfigError, "only for kind 'fake'"):
            validate_config(data)

    def test_camera_ids_in_config_must_match_the_devices(self) -> None:
        data = base_config(self.tmp)
        data["cameras"] = {"kind": "opencv", "devices": [0], "camera_ids": ["C2"]}
        with self.assertRaisesRegex(ConfigError, "derived"):
            validate_config(data)

    def test_validated_config_validates_again_to_the_same_value(self) -> None:
        data = base_config(self.tmp)  # StationApp validates an already validated config
        data["cameras"] = {"kind": "fake", "camera_count": 1}
        data["mes"]["allow_partial_cameras"] = True
        once = validate_config(data)
        self.assertEqual(validate_config(once), once)

    def test_allow_partial_cameras_must_be_a_boolean(self) -> None:
        data = base_config(self.tmp)
        self.assertFalse(validate_config(data)["mes"]["allow_partial_cameras"])
        data["mes"]["allow_partial_cameras"] = "yes"
        with self.assertRaisesRegex(ConfigError, "allow_partial_cameras"):
            validate_config(data)


class PartialStationTests(TmpCase):
    def setUp(self) -> None:
        super().setUp()
        with open(os.path.join(self.tmp, "users.json"), "w", encoding="utf-8") as fh:
            json.dump({"op1": {"password_hash": hash_password("operator-pass-1"), "role": "operator"}}, fh)

    def _one_camera_config(self, allow: bool = False) -> dict:
        cfg = base_config(self.tmp)
        cfg["cameras"] = {"kind": "fake", "camera_count": 1}
        cfg["mes"]["allow_partial_cameras"] = allow
        return cfg

    def test_one_camera_part_is_stored_with_two_images_and_never_queued(self) -> None:
        os.environ[TOKEN_ENV] = "not-a-real-token-for-tests"  # a token is present, MES still off
        clock = FakeClock()
        app = build_station(self._one_camera_config(), model=MarkerModel("test-1"),
                            outputs=RecordingOutputs(), clock=clock)
        self.addCleanup(app.stop)
        self.assertTrue(app.partial)
        self.assertFalse(app.mes_enabled)
        self.assertEqual(app.session.snapshot()["camera_ids"], ["C1"])

        self.assertTrue(app.session.press())  # TOP
        clock.now += PAST_MIN_FLIP
        self.assertTrue(app.session.press())  # BOTTOM: fuse and save
        self.assertEqual(app.session.counters["total"], 1)

        part = app.store.recent_parts(1)[0]
        with open(os.path.join(part["path"], "part.json"), encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertTrue(doc["partial_station"])
        self.assertEqual(doc["camera_count"], 1)
        self.assertEqual(len(doc["images"]), 2)
        self.assertEqual(app.outbox.pending_count(), 0)

        reasons = [json.loads(r["payload_json"])["reason"]
                   for r in BuildTests._events(app.store, "mes_disabled")]
        self.assertIn("partial station (1 camera(s)): parts are not sent to the MES", reasons)

    def test_partial_station_with_the_flag_queues_the_part(self) -> None:
        os.environ[TOKEN_ENV] = "not-a-real-token-for-tests"
        clock = FakeClock()
        app = build_station(self._one_camera_config(allow=True), model=MarkerModel("test-1"),
                            outputs=RecordingOutputs(), clock=clock)
        self.addCleanup(app.stop)
        self.assertTrue(app.mes_enabled)
        self.assertIn("mes_partial_allowed", BuildTests._event_kinds(app.store))
        self.assertNotIn("mes_disabled", BuildTests._event_kinds(app.store))

        app.session.press()
        clock.now += PAST_MIN_FLIP
        app.session.press()
        self.assertEqual(app.session.counters["total"], 1)
        self.assertEqual(app.outbox.pending_count(), 1)

    def test_partial_station_does_not_post_without_a_token_either_way(self) -> None:
        clock = FakeClock()
        app = build_station(self._one_camera_config(), model=MarkerModel("test-1"),
                            outputs=RecordingOutputs(), clock=clock)
        self.addCleanup(app.stop)
        self.assertFalse(app.mes_enabled)
        kinds = BuildTests._event_kinds(app.store)
        self.assertEqual(kinds.count("mes_disabled"), 2)  # partial reason and no-token reason

    def test_full_station_is_not_partial(self) -> None:
        os.environ[TOKEN_ENV] = "not-a-real-token-for-tests"
        app = build_station(base_config(self.tmp), cameras=FakeCameraSet(),
                            model=MarkerModel("test-1"), outputs=RecordingOutputs(),
                            clock=FakeClock())
        self.addCleanup(app.stop)
        self.assertFalse(app.partial)
        self.assertTrue(app.mes_enabled)
        self.assertNotIn("mes_partial_allowed", BuildTests._event_kinds(app.store))


class GoldenModelConfigTests(TmpCase):
    def test_golden_kind_needs_a_path(self) -> None:
        data = base_config(self.tmp)
        data["model"] = {"kind": "golden", "version": "golden-poc-1"}
        with self.assertRaisesRegex(ConfigError, "model.path is required when model.kind is 'golden'"):
            validate_config(data)

    def test_missing_golden_file_gives_a_clear_error(self) -> None:
        cfg = base_config(self.tmp)
        cfg["model"] = {"kind": "golden", "path": os.path.join(self.tmp, "golden.npz"),
                        "version": "golden-poc-1"}
        with self.assertRaisesRegex(StationError, "golden model file not found.*run python -m station teach first"):
            build_station(cfg, cameras=FakeCameraSet(), outputs=RecordingOutputs())

    @unittest.skipUnless(HAVE_NUMPY, "numpy is needed to build a golden model")
    def test_saved_golden_model_loads_and_predicts(self) -> None:
        import numpy as np

        from station.app import StationApp
        from station.golden import GoldenBuilder

        def decode(frame: Frame):  # the bytes are one grey value: a flat 8x8 image
            return np.full((8, 8), float(frame.data[0]))

        builder = GoldenBuilder(decode)
        for side in (Side.TOP, Side.BOTTOM):
            for value in (100, 102, 98):
                builder.add(Frame(camera_id="C1", side=side, seq=0,
                                  timestamp_utc="2026-10-23T08:30:12Z", data=bytes([value])))
        path = os.path.join(self.tmp, "golden.npz")
        builder.build("file-version").save(path)

        cfg = base_config(self.tmp)
        cfg["cameras"] = {"kind": "fake", "camera_count": 1}
        cfg["model"] = {"kind": "golden", "path": path, "version": "golden-poc-1"}
        with mock.patch("station.golden.cv2_decoder", return_value=decode):
            model = StationApp._make_model(validate_config(cfg))

        self.assertEqual(model.version, "golden-poc-1")  # the config version, not the file's
        ts = "2026-10-23T08:30:12Z"
        good = model.predict(Frame("C1", Side.TOP, 0, ts, bytes([100])))
        bad = model.predict(Frame("C1", Side.TOP, 0, ts, bytes([200])))
        self.assertEqual(good.score, 0.0)
        self.assertGreater(bad.score, 0.5)


class PocConfigTests(TmpCase):
    def test_poc_example_fails_only_on_placeholders_and_loads_once_filled(self) -> None:
        text = (REPO / "config.poc.example.json").read_text(encoding="utf-8")
        example = json.loads(text)
        with self.assertRaisesRegex(ConfigError, "placeholders"):
            validate_config(example)

        filled = copy.deepcopy(example)
        filled.update({
            "site": "SITE1", "line": "L1", "station_id": "VIS-POC-1", "part_number": "PN-100",
            "lot": "LOT-1", "operator": "op1", "supervisor": "sup1", "material_handler": "mh1",
        })
        filled["thresholds"] = {"t_low": 0.5, "t_img": 0.8}
        filled["mes"].update({
            "base_url": "http://127.0.0.1:3000",
            "token_env": "STATION_MES_TOKEN",
            "criteria_ids": ["CR-1"],
            "class_to_criterion": {"scratch": "CR-1"},
        })
        self.assertNotRegex(json.dumps(filled), r'"\[', "a bracketed placeholder is left")

        cfg = load_config(self.write_config(filled, "poc.json"))
        self.assertEqual(cfg["cameras"]["camera_ids"], ["C1"])
        self.assertFalse(is_full_station(cfg["cameras"]["camera_ids"]))  # partial
        self.assertFalse(cfg["mes"]["allow_partial_cameras"])
        self.assertEqual(cfg["model"]["kind"], "golden")
        self.assertTrue(cfg["model"]["path"].endswith("golden.npz"))
        self.assertEqual(cfg["ui"]["port"], 8000)
        self.assertEqual(cfg["min_flip_s"], 2.0)


class MinFlipTests(TmpCase):
    def setUp(self) -> None:
        super().setUp()
        with open(os.path.join(self.tmp, "users.json"), "w", encoding="utf-8") as fh:
            json.dump({"op1": {"password_hash": hash_password("operator-pass-1"), "role": "operator"}}, fh)

    def _app(self, min_flip_s=None) -> tuple[StationApp, FakeClock]:
        cfg = base_config(self.tmp)
        if min_flip_s is not None:
            cfg["min_flip_s"] = min_flip_s
        clock = FakeClock()
        app = build_station(cfg, cameras=FakeCameraSet(), model=MarkerModel("test-1"),
                            outputs=RecordingOutputs(), clock=clock)
        self.addCleanup(app.stop)
        return app, clock

    def test_quick_second_press_is_ignored_with_the_default(self) -> None:
        app, clock = self._app()
        self.assertTrue(app.session.press())  # TOP
        clock.now += 0.5  # past the 0.3 s debounce, but inside the 2.0 s default
        self.assertFalse(app.session.press())
        self.assertEqual(app.session.snapshot()["state"], State.WAIT_FLIP.value)
        self.assertEqual(app.session.counters["total"], 0)

    def test_quick_second_press_is_accepted_when_min_flip_s_is_0(self) -> None:
        app, clock = self._app(min_flip_s=0)
        self.assertTrue(app.session.press())
        clock.now += 0.5
        self.assertTrue(app.session.press())
        self.assertEqual(app.session.counters["total"], 1)


class PreviewWrapperTests(unittest.TestCase):
    class FakeSession:
        def __init__(self, result=b"jpeg-bytes", error=None) -> None:
            self.result = result
            self.error = error
            self.calls: list[str] = []

        def preview(self, camera_id: str):
            self.calls.append(camera_id)
            if self.error is not None:
                raise self.error
            return self.result

    def test_preview_returns_the_session_bytes_and_releases_the_lock(self) -> None:
        lock = threading.RLock()
        session = self.FakeSession()
        wrapped = _LockedSession(session, lock)
        self.assertEqual(wrapped.preview("C1"), b"jpeg-bytes")
        self.assertEqual(session.calls, ["C1"])
        self.assertTrue(lock.acquire(blocking=False))  # not left held
        lock.release()

    def test_preview_returns_none_at_once_while_a_press_holds_the_lock(self) -> None:
        lock = threading.RLock()
        session = self.FakeSession()
        wrapped = _LockedSession(session, lock)
        holding = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with lock:
                holding.set()
                release.wait(10)

        worker = threading.Thread(target=hold, daemon=True)
        worker.start()
        try:
            self.assertTrue(holding.wait(10))
            self.assertIsNone(wrapped.preview("C1"))
            self.assertEqual(session.calls, [])  # the session was never touched
        finally:
            release.set()
            worker.join(10)
        self.assertEqual(wrapped.preview("C1"), b"jpeg-bytes")

    def test_preview_returns_none_when_the_session_raises(self) -> None:
        lock = threading.RLock()
        wrapped = _LockedSession(self.FakeSession(error=StationError("camera gone")), lock)
        self.assertIsNone(wrapped.preview("C2"))
        self.assertTrue(lock.acquire(blocking=False))  # released despite the error
        lock.release()


class ScriptedCameras(FakeCameraSet):
    """A FakeCameraSet whose open() succeeds or raises, like OpenCvCameraSet.open()."""

    def __init__(self, open_error=None) -> None:
        super().__init__()
        self._open_error = open_error
        self.open_calls = 0
        self._open = False
        self._last_error = None

    def open(self) -> None:
        self.open_calls += 1
        if self._open_error is not None:
            self._last_error = str(self._open_error) or type(self._open_error).__name__
            raise self._open_error
        self._open = True
        self._last_error = None

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def last_open_error(self):
        return self._last_error


class CameraOpenTests(TmpCase):
    def setUp(self) -> None:
        super().setUp()
        with open(os.path.join(self.tmp, "users.json"), "w", encoding="utf-8") as fh:
            json.dump({"op1": {"password_hash": hash_password("operator-pass-1"), "role": "operator"}}, fh)

    def _app(self, cameras) -> StationApp:
        app = build_station(base_config(self.tmp), cameras=cameras, model=MarkerModel("test-1"),
                            outputs=RecordingOutputs(), clock=FakeClock())
        self.addCleanup(app.stop)
        return app

    def test_start_opens_the_cameras_once_and_logs_camera_open(self) -> None:
        cams = ScriptedCameras()
        app = self._app(cams)
        app.start()
        app.stop()
        self.assertEqual(cams.open_calls, 1)
        rows = BuildTests._events(app.store, "camera_open")
        self.assertEqual(len(rows), 1)
        self.assertIn('"C1"', rows[0]["payload_json"])
        self.assertNotIn("camera_open_failed", BuildTests._event_kinds(app.store))
        self.assertEqual(app.camera_status(), {"open": True, "last_error": None})

    def test_failed_open_is_logged_printed_and_does_not_stop_start(self) -> None:
        cams = ScriptedCameras(open_error=CameraError("C1: could not open device 0"))
        app = self._app(cams)
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            app.start()  # must not raise
        app.stop()
        self.assertEqual(cams.open_calls, 1)
        rows = BuildTests._events(app.store, "camera_open_failed")
        self.assertEqual(len(rows), 1)
        self.assertIn("could not open device 0", rows[0]["payload_json"])
        self.assertNotIn("camera_open", BuildTests._event_kinds(app.store))
        self.assertEqual(
            err.getvalue(),
            "camera not opened: C1: could not open device 0. "
            "The live view will keep retrying every few seconds.\n",
        )
        self.assertEqual(app.camera_status(), {"open": False, "last_error": "C1: could not open device 0"})

    def test_non_camera_error_from_open_is_also_contained(self) -> None:
        app = self._app(ScriptedCameras(open_error=RuntimeError("driver busy")))
        with mock.patch("sys.stderr", io.StringIO()):
            app.start()
        app.stop()
        rows = BuildTests._events(app.store, "camera_open_failed")
        self.assertIn("driver busy", rows[0]["payload_json"])

    def test_fake_cameras_have_no_open_call_and_report_open(self) -> None:
        app = self._app(FakeCameraSet())
        app.start()
        app.stop()
        kinds = BuildTests._event_kinds(app.store)
        self.assertNotIn("camera_open", kinds)
        self.assertNotIn("camera_open_failed", kinds)
        self.assertEqual(app.camera_status(), {"open": True, "last_error": None})


class UsersCliTests(TmpCase):
    def run_cli(self, argv, passwords=("correct horse 1", "correct horse 1")):
        feed = iter(passwords)
        out, err = io.StringIO(), io.StringIO()
        code = users_cli.main(argv, getpass_fn=lambda prompt: next(feed), out=out, err=err)
        return code, out.getvalue(), err.getvalue()

    def test_add_stores_a_verifiable_hash_and_never_prints_secrets(self) -> None:
        path = os.path.join(self.tmp, "users.json")
        code, out, err = self.run_cli(["add", "alice", "operator", "--users-file", path])
        self.assertEqual(code, 0, err)
        self.assertIn("added user 'alice' with role operator", out)
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        self.assertNotIn("correct horse", raw)
        self.assertNotIn("correct horse", out + err)
        self.assertNotIn("scrypt$", out + err)
        users = load_users(path)
        self.assertEqual(users.authenticate("alice", "correct horse 1"), "operator")
        self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")

    def test_update_changes_role_and_password_and_keeps_other_users(self) -> None:
        path = os.path.join(self.tmp, "users.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"bob": {"password_hash": hash_password("bob-password-1"), "role": "operator"}}, fh)
        code, out, _ = self.run_cli(["add", "bob", "supervisor", "--users-file", path],
                                    passwords=("new-password-22", "new-password-22"))
        self.assertEqual(code, 0)
        self.assertIn("updated user 'bob'", out)
        users = load_users(path)
        self.assertEqual(users.authenticate("bob", "new-password-22"), "supervisor")
        self.assertIsNone(users.authenticate("bob", "bob-password-1"))

    def test_bad_role_and_mismatched_or_short_passwords_write_nothing(self) -> None:
        path = os.path.join(self.tmp, "users.json")
        code, _, _ = self.run_cli(["add", "carol", "boss", "--users-file", path])
        self.assertEqual(code, 2)  # argparse rejects the role before any prompt
        self.assertFalse(os.path.exists(path))

        code, _, err = self.run_cli(["add", "carol", "operator", "--users-file", path],
                                    passwords=("one-password-1", "other-password-1"))
        self.assertEqual(code, 1)
        self.assertIn("do not match", err)
        code, _, err = self.run_cli(["add", "carol", "operator", "--users-file", path],
                                    passwords=("short", "short"))
        self.assertEqual(code, 1)
        self.assertIn("at least 8", err)
        self.assertFalse(os.path.exists(path))

    def test_corrupt_users_file_is_not_overwritten(self) -> None:
        path = os.path.join(self.tmp, "users.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{broken")
        code, _, err = self.run_cli(["add", "dan", "admin", "--users-file", path])
        self.assertEqual(code, 1)
        self.assertIn("not changed", err)
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "{broken")

    def test_users_file_comes_from_config_when_not_given(self) -> None:
        cfg_path = self.write_config(base_config(self.tmp))
        code, out, err = self.run_cli(["add", "erin", "operator", "--config", cfg_path])
        self.assertEqual(code, 0, err)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "users.json")))


class MainTests(unittest.TestCase):
    def test_help_exits_zero(self) -> None:
        proc = subprocess.run([sys.executable, "-m", "station", "--help"], cwd=str(REPO),
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--config", proc.stdout)
        self.assertIn("--demo", proc.stdout)

    def test_bad_config_exits_two_with_a_message(self) -> None:
        proc = subprocess.run([sys.executable, "-m", "station", "--config", "/nonexistent/x.json"],
                              cwd=str(REPO), capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("config error", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main()
