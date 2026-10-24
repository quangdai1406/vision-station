import json
import os
import pathlib
import sqlite3
import tempfile
import threading
import unittest

from station.contracts import (
    CAMERA_IDS,
    Defect,
    Frame,
    ImageResult,
    Override,
    PartRecord,
    Side,
    Thresholds,
    Verdict,
)
from station.storage import Store

STARTED = "2026-10-23T08:30:12Z"


def make_record(session_id="S-1", part_id="P-1", final=Verdict.GOOD, **overrides):
    frames, results = [], []
    for side in (Side.TOP, Side.BOTTOM):
        for seq, cam in enumerate(CAMERA_IDS):
            frames.append(
                Frame(
                    camera_id=cam,
                    side=side,
                    seq=seq,
                    timestamp_utc=STARTED,
                    data=f"{session_id}-{cam}-{side.value}".encode() * 40,
                    ext="png",
                )
            )
            if cam == "C2" and side is Side.BOTTOM:
                defects = (Defect("scratch", 0.7, (1, 2, 3, 4)),)
                score = 0.7
            else:
                defects = ()
                score = 0.1
            results.append(ImageResult(camera_id=cam, side=side, score=score, defects=defects))
    fields = dict(
        part_id=part_id,
        session_id=session_id,
        station_id="ST-1",
        part_number="PN-1",
        lot="LOT-1",
        operator="op-1",
        supervisor="sup-1",
        material_handler="mh-1",
        started_utc=STARTED,
        model_version="m-1.0",
        thresholds=Thresholds(t_low=0.3, t_img=0.6),
        frames=frames,
        results=results,
        machine_verdict=final,
        final_verdict=final,
    )
    fields.update(overrides)
    return PartRecord(**fields)


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.store = Store(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def db(self):
        return sqlite3.connect(str(self.root / "station.db"))


class TestSavePart(StoreTestCase):
    def test_save_and_read_back_part_json(self):
        path = self.store.save_part(make_record())
        doc = json.loads((path / "part.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["part_id"], "P-1")
        self.assertEqual(doc["session_id"], "S-1")
        self.assertEqual(doc["final_verdict"], "GOOD")
        self.assertEqual(doc["machine_verdict"], "GOOD")
        self.assertEqual(doc["thresholds"], {"t_low": 0.3, "t_img": 0.6})
        self.assertIsNone(doc["override"])
        self.assertEqual(len(doc["images"]), 8)
        c2 = [i for i in doc["images"] if i["camera_id"] == "C2" and i["side"] == "BOTTOM"][0]
        self.assertEqual(c2["defects"], [{"defect_class": "scratch", "score": 0.7, "box": [1, 2, 3, 4]}])

    def test_override_is_recorded(self):
        rec = make_record(
            final=Verdict.BAD,
            machine_verdict=Verdict.REVIEW,
            override=Override(Verdict.REVIEW, Verdict.BAD, "scratch seen", "sup-1"),
            disposition="quarantine",
        )
        path = self.store.save_part(rec)
        doc = json.loads((path / "part.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["override"]["original_verdict"], "REVIEW")
        self.assertEqual(doc["override"]["new_verdict"], "BAD")
        self.assertEqual(doc["disposition"], "quarantine")

    def test_file_names_and_location(self):
        path = self.store.save_part(make_record())
        expected_dir = self.root / "store" / "L1" / "2026" / "10" / "23" / "P-1"
        self.assertEqual(path, expected_dir)
        expected_names = {
            f"SITE_L1_20261023_083012_P-1_{cam}_{side}.png"
            for cam in CAMERA_IDS
            for side in ("TOP", "BOTTOM")
        }
        self.assertEqual(
            {p.name for p in path.iterdir()},
            expected_names | {"part.json", "manifest.sha256"},
        )

    def test_manifest_verifies(self):
        path = self.store.save_part(make_record())
        lines = (path / "manifest.sha256").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 8)
        for line in lines:
            digest, name = line.split("  ", 1)
            self.assertEqual(len(digest), 64)
            self.assertTrue((path / name).is_file())
        self.assertEqual(self.store.verify_part(path), [])

    def test_tampered_image_reported(self):
        path = self.store.save_part(make_record())
        name = "SITE_L1_20261023_083012_P-1_C1_TOP.png"
        (path / name).write_bytes(b"tampered")
        problems = self.store.verify_part(path)
        self.assertEqual(problems, [f"hash mismatch: {name}"])

    def test_missing_image_reported(self):
        path = self.store.save_part(make_record())
        name = "SITE_L1_20261023_083012_P-1_C3_BOTTOM.png"
        (path / name).unlink()
        self.assertIn(f"missing file: {name}", self.store.verify_part(path))

    def test_extra_file_reported(self):
        path = self.store.save_part(make_record())
        (path / "stray.txt").write_text("x", encoding="utf-8")
        self.assertIn("not in manifest: stray.txt", self.store.verify_part(path))

    def test_missing_part_json_and_manifest_reported(self):
        path = self.store.save_part(make_record())
        (path / "part.json").unlink()
        self.assertIn("missing part.json", self.store.verify_part(path))
        (path / "manifest.sha256").unlink()
        self.assertIn("missing manifest.sha256", self.store.verify_part(path))

    def test_idempotent_second_save(self):
        first = self.store.save_part(make_record())
        before = (first / "part.json").read_bytes()
        second = self.store.save_part(make_record(operator="someone-else"))
        self.assertEqual(first, second)
        self.assertEqual((second / "part.json").read_bytes(), before)
        self.assertEqual(self.store.counters()["total"], 1)
        with self.db() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM images").fetchone()[0], 8)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0], 1)

    def test_seven_frames_raises(self):
        rec = make_record()
        rec.frames = rec.frames[:7]
        with self.assertRaises(ValueError):
            self.store.save_part(rec)

    def test_seven_results_raises(self):
        rec = make_record()
        rec.results = rec.results[:7]
        with self.assertRaises(ValueError):
            self.store.save_part(rec)

    def test_no_tmp_directories_left(self):
        self.store.save_part(make_record())
        for dirpath, dirnames, filenames in os.walk(self.root / "store"):
            for name in dirnames + filenames:
                self.assertNotIn(".tmp", name)

    def test_two_threads_saving_different_parts(self):
        recs = [make_record("S-A", "P-A"), make_record("S-B", "P-B")]
        errors = []

        def worker(rec):
            try:
                self.store.save_part(rec)
            except Exception as exc:  # noqa: BLE001 - collected for the assertion
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(r,)) for r in recs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.store.counters()["total"], 2)
        for rec in recs:
            path = self.root / "store" / "L1" / "2026" / "10" / "23" / rec.part_id
            self.assertEqual(self.store.verify_part(path), [])


class TestEventsAndQueries(StoreTestCase):
    def test_events_cannot_be_updated_or_deleted(self):
        event_id = self.store.append_event("test", {"a": 1}, session_id="S-1")
        self.assertIsInstance(event_id, int)
        conn = self.db()
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("UPDATE events SET kind = 'x' WHERE id = ?", (event_id,))
            with self.assertRaises(sqlite3.DatabaseError):
                conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        finally:
            conn.close()
        conn = self.db()
        try:
            row = conn.execute("SELECT kind, payload_json FROM events WHERE id = ?", (event_id,)).fetchone()
        finally:
            conn.close()
        self.assertEqual(row, ("test", '{"a": 1}'))

    def test_counters(self):
        self.assertEqual(self.store.counters(), {"total": 0, "good": 0, "bad": 0})
        self.store.save_part(make_record("S-1", "P-1", Verdict.GOOD))
        self.store.save_part(make_record("S-2", "P-2", Verdict.BAD))
        self.store.save_part(make_record("S-3", "P-3", Verdict.GOOD))
        self.assertEqual(self.store.counters(), {"total": 3, "good": 2, "bad": 1})

    def test_review_final_verdict_rejected(self):
        with self.assertRaises(ValueError):
            self.store.save_part(make_record(final=Verdict.REVIEW))

    def test_recent_parts_newest_first(self):
        self.store.save_part(make_record("S-1", "P-1"))
        self.store.save_part(make_record("S-2", "P-2"))
        recent = self.store.recent_parts(limit=1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["session_id"], "S-2")
        self.assertEqual(
            set(recent[0]),
            {"session_id", "part_id", "final_verdict", "started_utc", "path"},
        )


if __name__ == "__main__":
    unittest.main()
