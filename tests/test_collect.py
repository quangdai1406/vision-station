"""Collect mode: the station saves the 8 images and part.json with no model verdict.

Fakes only (FakeCameraSet, RecordingOutputs, an injected clock). Real Store in a temp folder.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
import tempfile
import unittest

from station.app import ConfigError, build_station, validate_config
from station.cameras import FakeCameraSet
from station.contracts import Frame, ImageResult, PartRecord, Side, State, Thresholds, Verdict
from station.inference import CollectModel
from station.outputs import RecordingOutputs
from station.session import COLLECTED_LABEL, StationSession
from station.storage import COLLECTED, Store

TS = "2026-10-26T08:00:00Z"
PAST_MIN_FLIP = 2.5  # longer than the 2.0 s default min_flip_s
TOKEN_ENV = "VISION_TEST_COLLECT_TOKEN"


def collect_config(tmp: str) -> dict:
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
        "model": {"kind": "collect", "version": "collect-1"},
        "outputs": {"kind": "console"},
    }


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class CollectModelTests(unittest.TestCase):
    def test_every_image_is_neutral_with_no_defects(self) -> None:
        model = CollectModel("collect-1")
        self.assertEqual(model.version, "collect-1")
        frame = Frame(camera_id="C2", side=Side.BOTTOM, seq=0, timestamp_utc=TS, data=b"x")
        result = model.predict(frame)
        self.assertEqual((result.camera_id, result.side, result.score), ("C2", Side.BOTTOM, 0.0))
        self.assertEqual(result.defects, ())


class CollectSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="collect-session-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = Store(pathlib_path(self.tmp), site="SITE1", line="L1")
        self.outputs = RecordingOutputs()
        self.clock = _Clock()
        self.completed: list[PartRecord] = []
        self.session = StationSession(
            FakeCameraSet(clock=lambda: TS), CollectModel("collect-1"), self.outputs, self.store,
            Thresholds(t_low=0.5, t_img=0.8),
            station_id="VIS-1", part_number="PN-100", lot="LOT-1", operator="op1",
            supervisor="sup1", material_handler="mh1",
            clock=self.clock, now_utc=lambda: TS,
            part_id_factory=lambda: "part-1", session_id_factory=lambda: "sess-1",
            on_complete=self.completed.append, collect_only=True,
            fuse_fn=_fuse_must_not_run,
        )

    def _inspect_one_part(self) -> None:
        self.assertTrue(self.session.press())
        self.clock.now += PAST_MIN_FLIP
        self.assertTrue(self.session.press())

    def test_part_is_saved_with_eight_images_and_no_judgement(self) -> None:
        self._inspect_one_part()
        self.assertEqual(self.session.state, State.IDLE)
        self.assertEqual(len(self.completed), 1)
        rec = self.completed[0]
        self.assertEqual(len(rec.frames), 8)
        self.assertEqual(len(rec.results), 8)
        self.assertEqual(rec.final_verdict, Verdict.REVIEW)
        self.assertEqual(rec.disposition, COLLECTED)
        self.assertIsNone(rec.override)

        part = self.store.recent_parts(1)[0]
        with open(os.path.join(part["path"], "part.json"), encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual(doc["final_verdict"], "REVIEW")
        self.assertEqual(doc["disposition"], COLLECTED)
        self.assertEqual(len(doc["images"]), 8)
        self.assertEqual(self.store.verify_part(part["path"]), [])

    def test_never_green_or_red_only_clear(self) -> None:
        self._inspect_one_part()
        self.assertNotIn("show_good", self.outputs.history)
        self.assertNotIn("show_bad", self.outputs.history)
        self.assertNotIn("show_review", self.outputs.history)
        self.assertEqual(self.outputs.history[-1], "clear")

    def test_collected_part_is_counted_in_total_only(self) -> None:
        self._inspect_one_part()
        self.assertEqual(self.session.counters, {"total": 1, "good": 0, "bad": 0})
        self.assertEqual(self.store.counters(), {"total": 1, "good": 0, "bad": 0})

    def test_snapshot_says_collected_not_judged(self) -> None:
        self._inspect_one_part()
        snap = self.session.snapshot()
        self.assertEqual(snap["verdict"], COLLECTED_LABEL)
        self.assertEqual(snap["final_verdict"], "REVIEW")
        self.assertNotIn(snap["verdict"], ("GOOD", "BAD"))

    def test_store_failure_faults_and_does_not_light_anything(self) -> None:
        def boom(record):
            raise OSError("disk full")

        self.store.save_part = boom
        self.assertTrue(self.session.press())
        self.clock.now += PAST_MIN_FLIP
        self.session.press()
        self.assertEqual(self.session.state, State.FAULT)
        self.assertEqual(self.session.counters["total"], 0)
        self.assertNotIn("show_good", self.outputs.history)
        del self.store.save_part  # restore the real method
        self.session.clear_fault()
        self.assertEqual(self.session.state, State.IDLE)
        self.assertEqual(self.session.counters["total"], 1)
        self.assertEqual(self.outputs.history[-1], "clear")


class CollectStorageTests(unittest.TestCase):
    def test_review_final_is_accepted_only_for_a_collected_part(self) -> None:
        tmp = tempfile.mkdtemp(prefix="collect-storage-")
        self.addCleanup(shutil.rmtree, tmp, True)
        store = Store(pathlib_path(tmp), site="SITE1", line="L1")
        store.save_part(_record("S-9", "P-9", Verdict.REVIEW, COLLECTED))
        self.assertEqual(store.counters(), {"total": 1, "good": 0, "bad": 0})
        with self.assertRaises(ValueError):
            store.save_part(_record("S-10", "P-10", Verdict.REVIEW, None))


class CollectAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="collect-app-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.clock = _Clock()
        with open(os.path.join(self.tmp, "users.json"), "w", encoding="utf-8") as fh:
            fh.write("{}")  # an empty user store is enough: no login is used here
        self.saved_env = os.environ.pop(TOKEN_ENV, None)
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        if self.saved_env is not None:
            os.environ[TOKEN_ENV] = self.saved_env

    def test_app_saves_collected_part_and_never_queues_it_for_mes(self) -> None:
        # A token is set, so the only reason nothing is queued is collect mode.
        app = build_station(collect_config(self.tmp), cameras=FakeCameraSet(clock=lambda: TS),
                            outputs=RecordingOutputs(), now_utc=lambda: TS, clock=self.clock,
                            token="test-token")
        self.assertFalse(app.mes_enabled)
        self.assertTrue(app.session.press())
        self.clock.now += PAST_MIN_FLIP
        self.assertTrue(app.session.press())
        self.assertEqual(app.session.counters, {"total": 1, "good": 0, "bad": 0})
        self.assertEqual(app.outbox.pending_count(), 0)
        part = app.store.recent_parts(1)[0]
        self.assertEqual(part["final_verdict"], "REVIEW")
        kinds = [e for e in _event_kinds(app)]
        self.assertIn("collected", kinds)
        self.assertIn("mes_disabled", kinds)

    def test_collect_mode_without_injected_model_uses_collect_model(self) -> None:
        app = build_station(collect_config(self.tmp), cameras=FakeCameraSet(clock=lambda: TS),
                            outputs=RecordingOutputs(), now_utc=lambda: TS, clock=self.clock)
        self.assertEqual(app.session.snapshot()["model_version"], "collect-1")


    def _collected_app(self):
        app = build_station(collect_config(self.tmp), cameras=FakeCameraSet(clock=lambda: TS),
                            outputs=RecordingOutputs(), now_utc=lambda: TS, clock=self.clock,
                            token="test-token")
        self.assertTrue(app.session.press())
        self.clock.now += PAST_MIN_FLIP
        self.assertTrue(app.session.press())
        return app

    def test_label_last_collected_part_keeps_verdict_counters_and_outbox(self) -> None:
        app = self._collected_app()
        result = app.label_part(by="op1", verdict=Verdict.BAD, criterion_id="CR-1")
        self.assertEqual(result["verdict"], "BAD")
        self.assertEqual(result["criterion_id"], "CR-1")
        self.assertFalse(result["changed"])
        part = app.store.recent_parts(1)[0]
        self.assertEqual(part["final_verdict"], "REVIEW")
        with open(os.path.join(part["path"], "part.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["operator_label"]["criterion_id"], "CR-1")
        self.assertEqual(app.session.counters, {"total": 1, "good": 0, "bad": 0})
        self.assertEqual(app.outbox.pending_count(), 0)
        again = app.label_part(by="op1", verdict=Verdict.GOOD)
        self.assertTrue(again["changed"])
        self.assertEqual(_event_kinds(app).count("operator_label"), 2)

    def test_label_refusals(self) -> None:
        app = self._collected_app()
        with self.assertRaises(ValueError):
            app.label_part(by="op1", verdict=Verdict.GOOD, criterion_id="CR-1")
        with self.assertRaises(ValueError):  # not in mes.criteria_ids
            app.label_part(by="op1", verdict=Verdict.BAD, criterion_id="CR-9")
        with self.assertRaises(ValueError):
            app.label_part(by="op1", verdict=Verdict.REVIEW)
        with self.assertRaises(ValueError):
            app.label_part(by="op1", verdict=Verdict.GOOD, part_id="P-unknown")
        with self.assertRaises(ValueError):
            app.label_part(by="  ", verdict=Verdict.GOOD)
        self.assertEqual(_event_kinds(app).count("operator_label"), 0)

    def test_label_without_any_collected_part_is_refused(self) -> None:
        app = build_station(collect_config(self.tmp), cameras=FakeCameraSet(clock=lambda: TS),
                            outputs=RecordingOutputs(), now_utc=lambda: TS, clock=self.clock,
                            token="test-token")
        with self.assertRaises(ValueError):
            app.label_part(by="op1", verdict=Verdict.GOOD)

    def test_free_class_name_allowed_when_no_criteria_list(self) -> None:
        app = self._collected_app()
        app._mes_cfg = replace(app._mes_cfg, criteria_ids=[])
        result = app.label_part(by="op1", verdict=Verdict.BAD, criterion_id="Scratch")
        self.assertEqual(result["criterion_id"], "Scratch")


class CollectConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="collect-config-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_collect_kind_is_accepted_in_fixed_mode(self) -> None:
        cfg = validate_config(collect_config(self.tmp))
        self.assertEqual(cfg["model"]["kind"], "collect")
        self.assertIsNone(cfg["model"]["path"])

    def test_collect_kind_is_refused_in_mes_parts_mode(self) -> None:
        data = collect_config(self.tmp)
        data["parts"] = {"mode": "mes", "models_dir": self.tmp}
        data.pop("part_number")
        with self.assertRaisesRegex(ConfigError, "needs parts.mode 'fixed'"):
            validate_config(data)

    def test_collect_kind_refuses_a_model_path(self) -> None:
        data = collect_config(self.tmp)
        data["model"]["path"] = os.path.join(self.tmp, "m.onnx")
        with self.assertRaisesRegex(ConfigError, "model.path is not used"):
            validate_config(data)

    def test_unknown_kind_message_lists_collect(self) -> None:
        data = collect_config(self.tmp)
        data["model"]["kind"] = "guess"
        with self.assertRaisesRegex(ConfigError, "'collect'"):
            validate_config(data)


def _fuse_must_not_run(*args, **kwargs):
    raise AssertionError("fuse ran in collect mode")


def _record(session_id: str, part_id: str, final: Verdict, disposition) -> PartRecord:
    frames, results = [], []
    for side in (Side.TOP, Side.BOTTOM):
        for i, cam in enumerate(("C1", "C2", "C3", "C4")):
            frames.append(Frame(camera_id=cam, side=side, seq=i, timestamp_utc=TS,
                                data=f"{side.value}-{cam}-{session_id}".encode()))
            results.append(ImageResult(camera_id=cam, side=side, score=0.0))
    return PartRecord(
        part_id=part_id, session_id=session_id, station_id="VIS-1", part_number="PN-100",
        lot="LOT-1", operator="op1", supervisor="sup1", material_handler="mh1",
        started_utc=TS, model_version="collect-1", thresholds=Thresholds(t_low=0.5, t_img=0.8),
        frames=frames, results=results, machine_verdict=Verdict.REVIEW, final_verdict=final,
        disposition=disposition,
    )


def pathlib_path(text: str):
    import pathlib

    return pathlib.Path(text)


def _event_kinds(app) -> list[str]:
    import sqlite3

    conn = sqlite3.connect(str(app.store.db_path))
    try:
        return [r[0] for r in conn.execute("SELECT kind FROM events").fetchall()]
    finally:
        conn.close()


if __name__ == "__main__":
    unittest.main()
