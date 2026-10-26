import json
import os
import pathlib
import sqlite3
import tempfile
import threading
import unittest
from unittest import mock

from station.contracts import (
    CAMERA_IDS,
    LABEL_OFFICIAL,
    LABEL_PROPOSED,
    LABEL_REJECTED,
    Defect,
    Frame,
    ImageResult,
    Override,
    PartRecord,
    Side,
    Thresholds,
    OperatorLabel,
    Verdict,
)
from station.storage import (COLLECTED, OPERATOR_LABEL_EVENT, Store, check_criterion_id,
                             check_reject_reason, label_status)

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


def make_pairs_record(pairs, session_id="S-P1", part_id="P-P1", final=Verdict.GOOD):
    """A PartRecord whose frames/results are exactly the given (camera, side) pairs."""
    frames, results = [], []
    for seq, (cam, side) in enumerate(pairs):
        frames.append(
            Frame(
                camera_id=cam,
                side=side,
                seq=seq,
                timestamp_utc=STARTED,
                data=f"{session_id}-{cam}-{side.value}-{seq}".encode() * 40,
                ext="png",
            )
        )
        results.append(ImageResult(camera_id=cam, side=side, score=0.1))
    return make_record(session_id, part_id, final, frames=frames, results=results)


ONE_CAMERA = [("C1", Side.TOP), ("C1", Side.BOTTOM)]


class TestPartialStation(StoreTestCase):
    def setUp(self):
        super().setUp()
        self.partial = Store(self.root, allow_partial=True)

    def test_default_store_rejects_two_frames(self):
        with self.assertRaises(ValueError):
            self.store.save_part(make_pairs_record(ONE_CAMERA))

    def test_partial_saves_one_camera_part(self):
        path = self.partial.save_part(make_pairs_record(ONE_CAMERA))
        self.assertEqual(path, self.root / "store" / "L1" / "2026" / "10" / "23" / "P-P1")
        self.assertEqual(
            {p.name for p in path.iterdir()},
            {
                "SITE_L1_20261023_083012_P-P1_C1_TOP.png",
                "SITE_L1_20261023_083012_P-P1_C1_BOTTOM.png",
                "part.json",
                "manifest.sha256",
            },
        )
        doc = json.loads((path / "part.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["camera_count"], 1)
        self.assertIs(doc["partial_station"], True)
        self.assertEqual(len(doc["images"]), 2)
        self.assertEqual(self.partial.verify_part(path), [])

    def test_four_camera_part_is_not_partial(self):
        path = self.partial.save_part(make_record())
        doc = json.loads((path / "part.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["camera_count"], 4)
        self.assertIs(doc["partial_station"], False)
        self.assertEqual(self.partial.verify_part(path), [])

    def test_default_part_json_has_new_keys(self):
        path = self.store.save_part(make_record())
        doc = json.loads((path / "part.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["camera_count"], 4)
        self.assertIs(doc["partial_station"], False)

    def test_partial_idempotent_second_save(self):
        first = self.partial.save_part(make_pairs_record(ONE_CAMERA))
        before = (first / "part.json").read_bytes()
        second = self.partial.save_part(make_pairs_record(ONE_CAMERA, final=Verdict.BAD))
        self.assertEqual(first, second)
        self.assertEqual((second / "part.json").read_bytes(), before)
        self.assertEqual(self.partial.counters()["total"], 1)

    def test_rejected_shapes(self):
        cases = {
            "duplicate pair": [("C1", Side.TOP), ("C1", Side.TOP),
                               ("C1", Side.BOTTOM), ("C1", Side.BOTTOM)],
            "unequal TOP/BOTTOM sets": [("C1", Side.TOP), ("C2", Side.TOP),
                                        ("C1", Side.BOTTOM), ("C3", Side.BOTTOM)],
            "odd count": [("C1", Side.TOP), ("C1", Side.BOTTOM), ("C2", Side.TOP)],
            "empty": [],
            "ten frames": [(f"C{i}", s) for i in range(1, 6) for s in (Side.TOP, Side.BOTTOM)],
        }
        for name, pairs in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    self.partial.save_part(make_pairs_record(pairs, session_id=f"S-{name}"))
        self.assertEqual(self.partial.counters()["total"], 0)

    def test_partial_results_must_match_frames(self):
        rec = make_pairs_record(ONE_CAMERA)
        rec.results = [
            ImageResult(camera_id="C2", side=Side.TOP, score=0.1),
            ImageResult(camera_id="C1", side=Side.BOTTOM, score=0.1),
        ]
        with self.assertRaises(ValueError):
            self.partial.save_part(rec)


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


TS_LABEL = "2026-10-26T09:00:00Z"


class TestOperatorLabel(StoreTestCase):
    def save_collected(self, session="S-c1", part="P-c1"):
        return self.store.save_part(
            make_record(session, part, final=Verdict.REVIEW, disposition=COLLECTED))

    @staticmethod
    def part_json(part_dir):
        return json.loads((part_dir / "part.json").read_text(encoding="utf-8"))

    @staticmethod
    def image_bytes(part_dir):
        return {p.name: p.read_bytes() for p in sorted(part_dir.glob("*.png"))}

    def label_events(self):
        conn = self.db()
        try:
            return conn.execute(
                "SELECT session_id, payload_json FROM events WHERE kind = ? ORDER BY id",
                (OPERATOR_LABEL_EVENT,),
            ).fetchall()
        finally:
            conn.close()

    def test_record_default_has_no_label_and_new_part_json_has_null(self):
        self.assertIsNone(make_record().operator_label)
        part_dir = self.save_collected()
        self.assertIsNone(self.part_json(part_dir)["operator_label"])
        self.assertEqual(self.store.verify_part(part_dir), [])

    def test_good_label_written_images_and_manifest_untouched(self):
        part_dir = self.save_collected()
        images = self.image_bytes(part_dir)
        manifest = (part_dir / "manifest.sha256").read_bytes()
        previous = self.store.set_operator_label(
            "S-c1", OperatorLabel(Verdict.GOOD, "qa-1", TS_LABEL))
        self.assertIsNone(previous)
        self.assertEqual(self.part_json(part_dir)["operator_label"], {
            "verdict": "GOOD", "criterion_id": None,
            "labelled_by": "qa-1", "labelled_at": TS_LABEL, "status": "official",
            "proposed_by": None, "proposed_at": None, "approved_by": None,
            "approved_at": None, "reject_reason": None})
        self.assertEqual(self.image_bytes(part_dir), images)
        self.assertEqual((part_dir / "manifest.sha256").read_bytes(), manifest)
        self.assertEqual(self.store.verify_part(part_dir), [])
        self.assertEqual(len(self.label_events()), 1)
        self.assertEqual(self.store.recent_parts(1)[0]["final_verdict"], "REVIEW")
        self.assertEqual(self.store.counters(), {"total": 1, "good": 0, "bad": 0})

    def test_bad_label_with_criterion_then_relabel_each_change_logged(self):
        part_dir = self.save_collected()
        self.store.set_operator_label(
            "S-c1", OperatorLabel(Verdict.BAD, "qa-1", TS_LABEL, criterion_id="CR-1"))
        previous = self.store.set_operator_label(
            "S-c1", OperatorLabel(Verdict.GOOD, "qa-2", TS_LABEL))
        self.assertEqual(previous["verdict"], "BAD")
        self.assertEqual(self.part_json(part_dir)["operator_label"]["verdict"], "GOOD")
        events = self.label_events()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1][0], "S-c1")
        second = json.loads(events[1][1])
        self.assertEqual(second["previous"]["criterion_id"], "CR-1")
        self.assertEqual(second["label"]["labelled_by"], "qa-2")
        self.assertEqual(self.store.verify_part(part_dir), [])

    def test_non_collected_part_cannot_be_labelled(self):
        part_dir = self.store.save_part(make_record("S-g1", "P-g1", final=Verdict.GOOD))
        before = (part_dir / "part.json").read_bytes()
        with self.assertRaises(ValueError):
            self.store.set_operator_label("S-g1", OperatorLabel(Verdict.GOOD, "qa-1", TS_LABEL))
        self.assertEqual((part_dir / "part.json").read_bytes(), before)
        self.assertEqual(self.label_events(), [])

    def test_unknown_session_refused(self):
        with self.assertRaises(ValueError):
            self.store.set_operator_label("S-none", OperatorLabel(Verdict.GOOD, "qa-1", TS_LABEL))

    def test_label_on_a_non_collected_record_refused_at_save(self):
        with self.assertRaises(ValueError):
            self.store.save_part(make_record(
                "S-g2", "P-g2", final=Verdict.GOOD,
                operator_label=OperatorLabel(Verdict.GOOD, "qa-1", TS_LABEL)))
        self.assertEqual(self.store.counters()["total"], 0)

    def test_invalid_labels_refused_and_part_json_unchanged(self):
        part_dir = self.save_collected()
        before = (part_dir / "part.json").read_bytes()
        invalid = [
            OperatorLabel(Verdict.REVIEW, "qa-1", TS_LABEL),
            OperatorLabel(Verdict.GOOD, "qa-1", TS_LABEL, criterion_id="CR-1"),
            OperatorLabel(Verdict.BAD, "qa-1", TS_LABEL, criterion_id=""),
            OperatorLabel(Verdict.BAD, "qa-1", TS_LABEL, criterion_id="x" * 65),
            OperatorLabel(Verdict.BAD, "qa-1", TS_LABEL, criterion_id="CR\n1"),
            OperatorLabel(Verdict.BAD, "  ", TS_LABEL),
            OperatorLabel(Verdict.GOOD, "qa-1", "yesterday"),
        ]
        for label in invalid:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self.store.set_operator_label("S-c1", label)
        self.assertEqual((part_dir / "part.json").read_bytes(), before)
        self.assertEqual(self.label_events(), [])

    def test_criterion_id_rules(self):
        self.assertEqual(check_criterion_id("  CR-1 "), "CR-1")
        self.assertEqual(check_criterion_id("Scratch on flange"), "Scratch on flange")
        for bad in ("", "   ", "a\tb", "a\x00b", "x" * 65, 7):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    check_criterion_id(bad)

    def test_failed_event_write_restores_part_json(self):
        part_dir = self.save_collected()
        before = (part_dir / "part.json").read_bytes()
        with mock.patch.object(self.store, "append_event", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.set_operator_label("S-c1", OperatorLabel(Verdict.GOOD, "qa-1", TS_LABEL))
        self.assertEqual((part_dir / "part.json").read_bytes(), before)
        self.assertEqual([p.name for p in part_dir.iterdir() if p.name.startswith(".tmp-")], [])

    def test_part_json_without_the_field_still_loads_and_takes_a_label(self):
        part_dir = self.save_collected()
        doc = self.part_json(part_dir)
        del doc["operator_label"]  # what a part.json saved before this field looks like
        (part_dir / "part.json").write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(self.store.verify_part(part_dir), [])
        previous = self.store.set_operator_label(
            "S-c1", OperatorLabel(Verdict.BAD, "qa-1", TS_LABEL, criterion_id="CR-1"))
        self.assertIsNone(previous)
        self.assertEqual(self.store.verify_part(part_dir), [])

    def test_collected_lookups(self):
        self.assertIsNone(self.store.last_collected_session_id())
        self.save_collected("S-c1", "P-c1")
        self.save_collected("S-c2", "P-c2")
        self.store.save_part(make_record("S-g4", "P-g4", final=Verdict.GOOD))
        self.assertEqual(self.store.last_collected_session_id(), "S-c2")
        self.assertEqual(self.store.collected_session_id_for_part("P-c1"), "S-c1")
        self.assertIsNone(self.store.collected_session_id_for_part("P-g4"))

    # ---- two-step approval: statuses, proposal fields, action in the event log

    def proposal(self, verdict=Verdict.BAD, criterion="CR-1", by="op-1"):
        return OperatorLabel(verdict, by, TS_LABEL, criterion_id=criterion, status=LABEL_PROPOSED,
                             proposed_by=by, proposed_at=TS_LABEL)

    def test_old_label_without_status_key_loads_as_official(self):
        part_dir = self.save_collected()
        doc = self.part_json(part_dir)
        doc["operator_label"] = {"verdict": "BAD", "criterion_id": "CR-1",
                                 "labelled_by": "qa-1", "labelled_at": TS_LABEL}
        (part_dir / "part.json").write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(self.store.verify_part(part_dir), [])
        self.assertEqual(label_status(doc["operator_label"]), LABEL_OFFICIAL)
        self.assertIsNone(label_status(None))
        listed = self.store.collected_labels(status=LABEL_OFFICIAL)
        self.assertEqual([r["session_id"] for r in listed], ["S-c1"])

    def test_proposal_then_approval_logged_with_action(self):
        part_dir = self.save_collected()
        self.store.set_operator_label("S-c1", self.proposal(), action="propose")
        self.assertEqual(self.part_json(part_dir)["operator_label"]["status"], LABEL_PROPOSED)
        approved = OperatorLabel(Verdict.BAD, "admin-1", TS_LABEL, criterion_id="CR-1",
                                 status=LABEL_OFFICIAL, proposed_by="op-1", proposed_at=TS_LABEL,
                                 approved_by="admin-1", approved_at=TS_LABEL)
        previous = self.store.set_operator_label("S-c1", approved, action="approve")
        self.assertEqual(previous["status"], LABEL_PROPOSED)
        events = [json.loads(p) for _, p in self.label_events()]
        self.assertEqual([e["action"] for e in events], ["propose", "approve"])
        self.assertEqual(events[1]["previous"]["proposed_by"], "op-1")
        self.assertEqual(events[1]["label"]["approved_by"], "admin-1")
        self.assertEqual(self.store.verify_part(part_dir), [])
        self.assertEqual(self.store.counters(), {"total": 1, "good": 0, "bad": 0})

    def test_rejection_needs_reason_and_is_stored(self):
        part_dir = self.save_collected()
        self.store.set_operator_label("S-c1", self.proposal(), action="propose")
        rejected = OperatorLabel(Verdict.BAD, "admin-1", TS_LABEL, criterion_id="CR-1",
                                 status=LABEL_REJECTED, proposed_by="op-1", proposed_at=TS_LABEL,
                                 approved_by="admin-1", approved_at=TS_LABEL,
                                 reject_reason="  blurred image  ")
        self.store.set_operator_label("S-c1", rejected, action="reject")
        doc = self.part_json(part_dir)["operator_label"]
        self.assertEqual(doc["status"], LABEL_REJECTED)
        self.assertEqual(doc["reject_reason"], "blurred image")
        no_reason = OperatorLabel(Verdict.BAD, "admin-1", TS_LABEL, status=LABEL_REJECTED,
                                  approved_by="admin-1", approved_at=TS_LABEL)
        with self.assertRaises(ValueError):
            self.store.set_operator_label("S-c1", no_reason, action="reject")

    def test_invalid_status_and_proposal_fields_refused(self):
        self.save_collected()
        bad = [
            OperatorLabel(Verdict.GOOD, "op-1", TS_LABEL, status="maybe"),
            OperatorLabel(Verdict.GOOD, "op-1", TS_LABEL, status=LABEL_PROPOSED),  # no proposed_by
            OperatorLabel(Verdict.GOOD, "op-1", TS_LABEL, status=LABEL_PROPOSED,
                          proposed_by="op-1", proposed_at=TS_LABEL, approved_by="admin"),
            OperatorLabel(Verdict.GOOD, "op-1", TS_LABEL, status=LABEL_OFFICIAL,
                          approved_by="admin"),  # approved_by without approved_at
            OperatorLabel(Verdict.GOOD, "op-1", TS_LABEL, status=LABEL_OFFICIAL,
                          reject_reason="no"),
            OperatorLabel(Verdict.GOOD, "op-1", TS_LABEL, status=LABEL_REJECTED,
                          approved_by="admin", approved_at=TS_LABEL),  # no reason
        ]
        for label in bad:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self.store.set_operator_label("S-c1", label, action="propose")
        self.assertEqual(self.label_events(), [])

    def test_reject_reason_rules(self):
        self.assertEqual(check_reject_reason(" short "), "short")
        for bad in ("", "   ", "a\nb", "x" * 201, 5):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    check_reject_reason(bad)

    def test_unknown_label_action_refused(self):
        self.save_collected()
        with self.assertRaises(ValueError):
            self.store.set_operator_label("S-c1", OperatorLabel(Verdict.GOOD, "a", TS_LABEL),
                                          action="delete")

    def test_collected_part_doc_is_read_only_copy(self):
        self.save_collected()
        doc = self.store.collected_part_doc("S-c1")
        self.assertEqual(doc["part_number"], "PN-1")
        doc["operator_label"] = "tampered"
        self.assertIsNone(self.store.collected_part_doc("S-c1")["operator_label"])
        with self.assertRaises(ValueError):
            self.store.collected_part_doc("S-g9")

    def test_collected_labels_filter_by_status(self):
        self.save_collected("S-c1", "P-c1")
        self.save_collected("S-c2", "P-c2")
        self.save_collected("S-c3", "P-c3")
        self.store.set_operator_label("S-c1", self.proposal(), action="propose")
        self.store.set_operator_label("S-c2", OperatorLabel(
            Verdict.GOOD, "admin", TS_LABEL, status=LABEL_OFFICIAL,
            approved_by="admin", approved_at=TS_LABEL), action="set_official")
        pending = self.store.collected_labels(status=LABEL_PROPOSED)
        self.assertEqual([r["session_id"] for r in pending], ["S-c1"])
        self.assertEqual(pending[0]["part_id"], "P-c1")
        self.assertEqual(len(self.store.collected_labels()), 2)  # S-c3 has no label


if __name__ == "__main__":
    unittest.main()
