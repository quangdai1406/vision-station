"""Tests for tools/export_evidence.py. Run from the project root:

    python3 -m unittest discover -s tests
"""
import contextlib
import io
import json
import os
import pathlib
import tempfile
import unittest
from datetime import date

from station.contracts import (
    CAMERA_IDS,
    IMAGES_PER_PART,
    Defect,
    Frame,
    ImageResult,
    PartRecord,
    Side,
    Thresholds,
    Verdict,
)
from station.storage import Store
from tools import export_evidence as ee


def make_record(session_id, part_id, started):
    frames, results = [], []
    for side in (Side.TOP, Side.BOTTOM):
        for seq, cam in enumerate(CAMERA_IDS):
            frames.append(
                Frame(
                    camera_id=cam,
                    side=side,
                    seq=seq,
                    timestamp_utc=started,
                    data=f"{session_id}-{cam}-{side.value}".encode() * 40,
                    ext="png",
                )
            )
            defects = (Defect("scratch", 0.7, (1, 2, 3, 4)),) if side is Side.BOTTOM and cam == "C2" else ()
            results.append(
                ImageResult(camera_id=cam, side=side, score=0.7 if defects else 0.1, defects=defects)
            )
    return PartRecord(
        part_id=part_id,
        session_id=session_id,
        station_id="ST-1",
        part_number="PN-1",
        lot="LOT-1",
        operator="op-1",
        supervisor="sup-1",
        material_handler="mh-1",
        started_utc=started,
        model_version="m-1.0",
        thresholds=Thresholds(t_low=0.3, t_img=0.6),
        frames=frames,
        results=results,
        machine_verdict=Verdict.GOOD,
        final_verdict=Verdict.GOOD,
    )


def snapshot(root):
    """Map every file under root to its bytes, for before/after comparisons."""
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = pathlib.Path(dirpath) / name
            out[str(path.relative_to(root))] = path.read_bytes()
    return out


class ExportEvidenceTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self._tmp.name)
        self.pi = base / "pi"
        self.dest = base / "laptop_evidence"
        self.store = Store(self.pi, site="SITE", line="L1")
        # Two parts on 2026-10-23 and one on 2026-10-25, all real layout.
        self.parts = {
            "P-1": self._save("S-1", "P-1", "2026-10-23T08:30:12Z"),
            "P-2": self._save("S-2", "P-2", "2026-10-23T09:00:00Z"),
            "P-3": self._save("S-3", "P-3", "2026-10-25T10:00:00Z"),
        }

    def tearDown(self):
        self._tmp.cleanup()

    def _save(self, session_id, part_id, started):
        return self.store.save_part(make_record(session_id, part_id, started))

    def _export(self, **kwargs):
        return ee.export(self.pi, self.dest, **kwargs)

    def _dest_part(self, part_id, day):
        y, m, d = day.split("-")
        return self.dest / "store" / "L1" / y / m / d / part_id


class LayoutAndConstantsTests(ExportEvidenceTestCase):
    def test_image_count_matches_station_contract(self):
        self.assertEqual(ee.IMAGES_PER_PART, IMAGES_PER_PART)

    def test_real_saved_part_is_ok(self):
        check = ee.check_part(self.parts["P-1"])
        self.assertEqual(check.status, "ok")
        self.assertEqual(check.problems, [])
        self.assertEqual(len(check.entries), IMAGES_PER_PART)


class CheckPartTests(ExportEvidenceTestCase):
    def test_missing_image_is_incomplete(self):
        part = self.parts["P-1"]
        image = sorted(p for p in part.iterdir() if p.suffix == ".png")[0]
        image.unlink()
        check = ee.check_part(part)
        self.assertEqual(check.status, "incomplete")
        self.assertTrue(any("missing image" in p for p in check.problems))

    def test_missing_part_json_is_incomplete(self):
        part = self.parts["P-1"]
        (part / ee.PART_JSON).unlink()
        self.assertEqual(ee.check_part(part).status, "incomplete")

    def test_missing_manifest_is_incomplete(self):
        part = self.parts["P-1"]
        (part / ee.MANIFEST).unlink()
        self.assertEqual(ee.check_part(part).status, "incomplete")

    def test_changed_image_bytes_is_corrupt(self):
        part = self.parts["P-1"]
        image = sorted(p for p in part.iterdir() if p.suffix == ".png")[0]
        image.write_bytes(b"tampered")
        check = ee.check_part(part)
        self.assertEqual(check.status, "corrupt")
        self.assertTrue(any("hash mismatch" in p for p in check.problems))

    def test_unlisted_file_is_corrupt(self):
        part = self.parts["P-1"]
        (part / "extra.txt").write_text("x", encoding="utf-8")
        check = ee.check_part(part)
        self.assertEqual(check.status, "corrupt")
        self.assertIn("not in manifest: extra.txt", check.problems)

    def test_part_json_disagreeing_with_manifest_is_corrupt(self):
        part = self.parts["P-1"]
        doc = json.loads((part / ee.PART_JSON).read_text(encoding="utf-8"))
        doc["images"][0]["sha256"] = "0" * 64
        (part / ee.PART_JSON).write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(ee.check_part(part).status, "corrupt")

    def test_malformed_manifest_is_corrupt(self):
        part = self.parts["P-1"]
        (part / ee.MANIFEST).write_text("not a manifest line\n", encoding="utf-8")
        self.assertEqual(ee.check_part(part).status, "corrupt")

    def test_purged_tombstone_is_purged(self):
        part = self.parts["P-1"]
        (part / ee.PURGED_FILE).write_text("{}", encoding="utf-8")
        self.assertEqual(ee.check_part(part).status, "purged")


class ExportTests(ExportEvidenceTestCase):
    def test_copies_all_ok_parts_in_same_layout(self):
        report = self._export()
        c = report["counts"]
        self.assertEqual((c["found"], c["copied"], c["skipped"], c["failed"]), (3, 3, 0, 0))
        for part_id, day in (("P-1", "2026-10-23"), ("P-2", "2026-10-23"), ("P-3", "2026-10-25")):
            dst = self._dest_part(part_id, day)
            self.assertTrue((dst / ee.MANIFEST).is_file())
            self.assertEqual(self.store.verify_part(dst), [])  # real storage verifier agrees

    def test_report_file_written_with_counts_and_no_extra_secrets(self):
        self._export()
        report = json.loads((self.dest / ee.REPORT_FILE).read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], 1)
        self.assertIn("generated_utc", report)
        self.assertEqual(report["counts"]["copied"], 3)
        self.assertNotIn("token", json.dumps(report).lower())

    def test_second_run_is_incremental(self):
        self._export()
        marker = self._dest_part("P-1", "2026-10-23") / ee.MANIFEST
        before = marker.stat().st_mtime_ns
        report = self._export()
        c = report["counts"]
        self.assertEqual((c["copied"], c["already_present"], c["skipped"]), (0, 3, 3))
        self.assertEqual(marker.stat().st_mtime_ns, before)

    def test_corrupt_and_incomplete_parts_are_skipped_and_reported(self):
        sorted(p for p in self.parts["P-1"].iterdir() if p.suffix == ".png")[0].write_bytes(b"bad")
        (self.parts["P-2"] / ee.MANIFEST).unlink()
        report = self._export()
        c = report["counts"]
        self.assertEqual((c["copied"], c["corrupt"], c["incomplete"]), (1, 1, 1))
        self.assertFalse(self._dest_part("P-1", "2026-10-23").exists())
        self.assertFalse(self._dest_part("P-2", "2026-10-23").exists())
        statuses = {item["part"]: item["status"] for item in report["problems"]}
        self.assertEqual(statuses["L1/2026/10/23/P-1"], "corrupt")
        self.assertEqual(statuses["L1/2026/10/23/P-2"], "incomplete")

    def test_purged_part_is_skipped(self):
        (self.parts["P-3"] / ee.PURGED_FILE).write_text("{}", encoding="utf-8")
        c = self._export()["counts"]
        self.assertEqual((c["copied"], c["purged"]), (2, 1))

    def test_interrupted_copy_is_completed_on_next_run(self):
        # A copy that stopped before the manifest was written has no manifest at the destination.
        dst = self._dest_part("P-1", "2026-10-23")
        dst.mkdir(parents=True)
        (dst / ee.PART_JSON).write_text("partial", encoding="utf-8")
        c = self._export()["counts"]
        self.assertEqual(c["copied"], 3)
        self.assertEqual(self.store.verify_part(dst), [])

    def test_destination_manifest_that_differs_is_not_overwritten(self):
        dst = self._dest_part("P-1", "2026-10-23")
        dst.mkdir(parents=True)
        (dst / ee.MANIFEST).write_text("different\n", encoding="utf-8")
        (dst / ee.PART_JSON).write_text("{}", encoding="utf-8")
        c = self._export()["counts"]
        self.assertEqual(c["failed"], 1)
        self.assertEqual((dst / ee.MANIFEST).read_text(encoding="utf-8"), "different\n")

    def test_nothing_is_deleted_at_destination(self):
        self._export()
        stray = self._dest_part("P-1", "2026-10-23") / "keep_me.txt"
        stray.write_text("local note", encoding="utf-8")
        self._export()
        self.assertEqual(stray.read_text(encoding="utf-8"), "local note")

    def test_source_is_never_modified(self):
        before = snapshot(self.pi)
        self._export()
        self._export(since=date(2026, 10, 24))
        self.assertEqual(snapshot(self.pi), before)

    def test_since_filters_by_date_folder(self):
        c = self._export(since=date(2026, 10, 24))["counts"]
        self.assertEqual((c["found"], c["copied"], c["out_of_range"]), (1, 1, 2))
        self.assertTrue(self._dest_part("P-3", "2026-10-25").is_dir())
        self.assertFalse(self._dest_part("P-1", "2026-10-23").exists())

    def test_dry_run_writes_nothing(self):
        report = self._export(dry_run=True)
        self.assertEqual(report["counts"]["copied"], 3)
        self.assertTrue(report["dry_run"])
        self.assertFalse(self.dest.exists())

    def test_accepts_store_folder_as_source(self):
        report = ee.export(self.pi / "store", self.dest)
        self.assertEqual(report["counts"]["copied"], 3)

    def test_source_and_destination_must_not_overlap(self):
        with self.assertRaises(ee.ExportError):
            ee.export(self.pi, self.pi / "store" / "out")
        with self.assertRaises(ee.ExportError):
            ee.export(self.pi, self.pi)

    def test_missing_source_is_an_error(self):
        with self.assertRaises(ee.ExportError):
            ee.export(self.pi / "nope", self.dest)

    def test_missing_destination_parent_is_refused_not_created(self):
        dest = pathlib.Path(self._tmp.name) / "unmounted" / "evidence"
        with self.assertRaises(ee.ExportError):
            ee.export(self.pi, dest)
        self.assertFalse((pathlib.Path(self._tmp.name) / "unmounted").exists())

    def test_temp_folders_are_ignored(self):
        tmp = self.pi / "store" / "L1" / "2026" / "10" / "23" / ".tmp-P-9-abc"
        tmp.mkdir()
        report = self._export()
        self.assertEqual(report["counts"]["found"], 3)
        self.assertEqual(report["counts"]["temp_ignored"], 1)


class CliTests(ExportEvidenceTestCase):
    def test_exit_zero_when_clean(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = ee.main(["--source", str(self.pi), "--dest", str(self.dest)])
        self.assertEqual(code, 0)

    def test_exit_one_when_a_part_is_corrupt(self):
        sorted(p for p in self.parts["P-1"].iterdir() if p.suffix == ".png")[0].write_bytes(b"bad")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = ee.main(["--source", str(self.pi), "--dest", str(self.dest)])
        self.assertEqual(code, 1)
        self.assertIn("[corrupt]", out.getvalue())

    def test_bad_since_is_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            code = ee.main(["--source", str(self.pi), "--dest", str(self.dest), "--since", "23/10/2026"])
        self.assertEqual(code, 2)

    def test_dry_run_flag_prints_summary_without_writing(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = ee.main(["--source", str(self.pi), "--dest", str(self.dest), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", out.getvalue())
        self.assertFalse(self.dest.exists())


if __name__ == "__main__":
    unittest.main()
