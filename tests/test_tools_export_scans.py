"""Tests for the scan part of tools/export_evidence.py (scans/ next to store/).

Fake scans are written with the real station.scanstore.ScanStore. Run from the project root:

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

from station.scanstore import ScanStore, load_scan, verify_scan
from station.storage import Store
from tests.test_tools_export_evidence import make_record
from tools import export_evidence as ee

CAMS = ("C1", "C2", "C3", "C4")
GOOD_LABEL = {
    "verdict": "GOOD",
    "defects": [],
    "status": "proposed",
    "labelled_by": "sup-1",
    "labelled_at": "2026-10-23T08:40:00Z",
}


def snapshot(root):
    """Map every file under root to its bytes, for before/after comparisons."""
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = pathlib.Path(dirpath) / name
            out[str(path.relative_to(root))] = path.read_bytes()
    return out


class ScanExportTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self._tmp.name)
        self.pi = base / "pi"
        self.dest = base / "laptop_evidence"
        self.scans = ScanStore(self.pi, site="SITE", line="L1")
        self.scan_dirs = {
            "S-1": self._save_scan("S-1", "2026-10-23T08:30:12Z"),
            "S-2": self._save_scan("S-2", "2026-10-25T10:00:00Z"),
        }

    def tearDown(self):
        self._tmp.cleanup()

    def _save_scan(self, scan_id, started, steps_top=2, steps_bottom=1):
        writer = self.scans.begin_scan(
            scan_id, "PN-1", "TOY-1", dict(GOOD_LABEL), "op-1", list(CAMS), started
        )
        for side, count in (("TOP", steps_top), ("BOTTOM", steps_bottom)):
            for step in range(1, count + 1):
                for cam in CAMS:
                    data = f"{scan_id}-{cam}-{side}-{step}".encode() * 40
                    writer.add_picture(cam, side, step, data)
        return writer.finish(started)

    def _export(self, **kwargs):
        return ee.export(self.pi, self.dest, **kwargs)

    def _dest_scan(self, scan_id, day):
        y, m, d = day.split("-")
        return self.dest / "scans" / "L1" / y / m / d / scan_id

    def _first_png(self, scan_id):
        return sorted(p for p in self.scan_dirs[scan_id].iterdir() if p.suffix == ".png")[0]


class ScanCopyTests(ScanExportTestCase):
    def test_ok_scans_are_copied_and_verify_at_destination(self):
        report = self._export()
        sc = report["scans"]["counts"]
        self.assertEqual((sc["found"], sc["copied"], sc["skipped"], sc["failed"]), (2, 2, 0, 0))
        for scan_id, day in (("S-1", "2026-10-23"), ("S-2", "2026-10-25")):
            dst = self._dest_scan(scan_id, day)
            self.assertEqual(verify_scan(dst), [])  # station verifier agrees
            self.assertEqual(load_scan(dst)["scan_id"], scan_id)

    def test_scan_json_is_copied_and_listed_in_manifest(self):
        self._export()
        dst = self._dest_scan("S-1", "2026-10-23")
        src = self.scan_dirs["S-1"]
        self.assertEqual((dst / "scan.json").read_bytes(), (src / "scan.json").read_bytes())
        self.assertIn("scan.json", (dst / "manifest.sha256").read_text(encoding="utf-8"))

    def test_second_run_skips_scans_already_present(self):
        self._export()
        marker = self._dest_scan("S-1", "2026-10-23") / "manifest.sha256"
        before = marker.stat().st_mtime_ns
        sc = self._export()["scans"]["counts"]
        self.assertEqual((sc["copied"], sc["already_present"], sc["skipped"]), (0, 2, 2))
        self.assertEqual(marker.stat().st_mtime_ns, before)

    def test_interrupted_copy_is_completed_on_next_run(self):
        self._export()
        dst = self._dest_scan("S-1", "2026-10-23")
        (dst / "manifest.sha256").unlink()
        first_picture = sorted(p.name for p in dst.iterdir() if p.suffix == ".png")[0]
        (dst / first_picture).unlink()
        sc = self._export()["scans"]["counts"]
        self.assertEqual((sc["copied"], sc["already_present"]), (1, 1))
        self.assertEqual(verify_scan(dst), [])

    def test_dry_run_copies_nothing(self):
        report = self._export(dry_run=True)
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["scans"]["counts"]["copied"], 2)
        self.assertFalse(self.dest.exists())

    def test_nothing_is_deleted_at_destination(self):
        self._export()
        stray = self._dest_scan("S-1", "2026-10-23") / "keep_me.txt"
        stray.write_text("local note", encoding="utf-8")
        self._export()
        self.assertEqual(stray.read_text(encoding="utf-8"), "local note")

    def test_source_is_never_modified(self):
        before = snapshot(self.pi)
        self._export()
        self._export(dry_run=True)
        self._export(since=date(2026, 10, 24))
        self.assertEqual(snapshot(self.pi), before)

    def test_since_filters_scan_day_folders(self):
        sc = self._export(since=date(2026, 10, 24))["scans"]["counts"]
        self.assertEqual((sc["found"], sc["copied"], sc["out_of_range"]), (1, 1, 1))
        self.assertTrue(self._dest_scan("S-2", "2026-10-25").is_dir())
        self.assertFalse(self._dest_scan("S-1", "2026-10-23").exists())


class ScanCheckTests(ScanExportTestCase):
    def test_altered_picture_is_corrupt_skipped_and_reported(self):
        self._first_png("S-1").write_bytes(b"tampered")
        report = self._export()
        sc = report["scans"]["counts"]
        self.assertEqual((sc["copied"], sc["corrupt"]), (1, 1))
        self.assertFalse(self._dest_scan("S-1", "2026-10-23").exists())
        item = report["scans"]["problems"][0]
        self.assertEqual((item["scan"], item["status"]), ("L1/2026/10/23/S-1", "corrupt"))
        self.assertTrue(any("hash mismatch" in r for r in item["reasons"]))

    def test_staging_folder_is_incomplete_and_never_copied(self):
        writer = self.scans.begin_scan(
            "S-3", "PN-1", "TOY-3", dict(GOOD_LABEL), "op-1", list(CAMS), "2026-10-23T09:00:00Z"
        )
        writer.add_picture("C1", "TOP", 1, b"half written")  # never finished: stays staged
        report = self._export()
        sc = report["scans"]["counts"]
        self.assertEqual((sc["found"], sc["copied"], sc["incomplete"]), (3, 2, 1))
        day_dest = self.dest / "scans" / "L1" / "2026" / "10" / "23"
        self.assertFalse((day_dest / "S-3").exists())
        self.assertFalse((day_dest / ".S-3.partial").exists())
        staged = [i for i in report["scans"]["problems"] if i["scan"].endswith(".S-3.partial")]
        self.assertEqual(staged[0]["status"], "incomplete")
        self.assertIn("partial", staged[0]["reasons"][0])

    def test_missing_picture_is_incomplete(self):
        self._first_png("S-1").unlink()
        sc = self._export()["scans"]["counts"]
        self.assertEqual((sc["incomplete"], sc["copied"]), (1, 1))

    def test_symlink_in_scan_folder_is_corrupt_and_not_followed(self):
        target = self.pi / "elsewhere.png"
        target.write_bytes(b"outside")
        link = self.scan_dirs["S-1"] / "link.png"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available here")
        report = self._export()
        self.assertEqual(report["scans"]["counts"]["corrupt"], 1)
        self.assertFalse(self._dest_scan("S-1", "2026-10-23").exists())

    def test_symlink_day_entry_is_corrupt_and_not_followed(self):
        day_dir = self.pi / "scans" / "L1" / "2026" / "10" / "24"
        day_dir.mkdir(parents=True)
        try:
            os.symlink(self.scan_dirs["S-1"], day_dir / "S-9")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available here")
        report = self._export()
        self.assertEqual(report["scans"]["counts"]["corrupt"], 1)
        self.assertFalse((self.dest / "scans" / "L1" / "2026" / "10" / "24" / "S-9").exists())


class ScanConflictTests(ScanExportTestCase):
    def test_destination_manifest_conflict_is_not_overwritten(self):
        dst = self._dest_scan("S-1", "2026-10-23")
        dst.mkdir(parents=True)
        (dst / "manifest.sha256").write_text("different\n", encoding="utf-8")
        sc = self._export()["scans"]["counts"]
        self.assertEqual(sc["failed"], 1)
        self.assertEqual((dst / "manifest.sha256").read_text(encoding="utf-8"), "different\n")

    def test_destination_picture_with_other_bytes_is_not_overwritten(self):
        self._export()
        dst = self._dest_scan("S-1", "2026-10-23")
        (dst / "manifest.sha256").unlink()
        picture = sorted(p for p in dst.iterdir() if p.suffix == ".png")[0]
        picture.write_bytes(b"someone else's file")
        sc = self._export()["scans"]["counts"]
        self.assertEqual(sc["failed"], 1)
        self.assertEqual(picture.read_bytes(), b"someone else's file")
        self.assertFalse((dst / "manifest.sha256").exists())


class ScanCliAndCoexistenceTests(ScanExportTestCase):
    def test_parts_and_scans_export_together(self):
        store = Store(self.pi, site="SITE", line="L1")
        store.save_part(make_record("S-P1", "P-1", "2026-10-23T08:30:12Z"))
        report = self._export()
        self.assertEqual((report["counts"]["found"], report["counts"]["copied"]), (1, 1))
        self.assertEqual(report["scans"]["counts"]["copied"], 2)
        dst_part = self.dest / "store" / "L1" / "2026" / "10" / "23" / "P-1"
        self.assertEqual(store.verify_part(dst_part), [])
        self.assertEqual(verify_scan(self._dest_scan("S-1", "2026-10-23")), [])

    def test_report_keeps_parts_keys_and_adds_scans_section(self):
        report = self._export()
        for key in ("counts", "problems", "source", "dest", "since", "dry_run", "schema"):
            self.assertIn(key, report)
        self.assertEqual(set(report["scans"]), {"counts", "problems"})
        self.assertIn("copied", report["scans"]["counts"])
        written = json.loads((self.dest / ee.REPORT_FILE).read_text(encoding="utf-8"))
        self.assertEqual(written["scans"]["counts"]["copied"], 2)

    def test_scans_folder_itself_is_accepted_as_source(self):
        report = ee.export(self.pi / "scans", self.dest)
        self.assertEqual(report["scans"]["counts"]["copied"], 2)
        self.assertEqual(report["counts"]["found"], 0)

    def test_scans_are_not_read_as_parts(self):
        report = ee.export(self.pi / "scans", self.dest)
        self.assertEqual(report["problems"], [])

    def test_exit_one_when_a_scan_is_corrupt(self):
        self._first_png("S-1").write_bytes(b"bad")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = ee.main(["--source", str(self.pi), "--dest", str(self.dest)])
        self.assertEqual(code, 1)
        self.assertIn("[corrupt] scan L1/2026/10/23/S-1", out.getvalue())

    def test_exit_zero_when_scans_are_clean(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = ee.main(["--source", str(self.pi), "--dest", str(self.dest)])
        self.assertEqual(code, 0)
        self.assertIn("scans: found 2", out.getvalue())

    def test_dry_run_flag_prints_scan_summary_without_writing(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = ee.main(["--source", str(self.pi), "--dest", str(self.dest), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", out.getvalue())
        self.assertFalse(self.dest.exists())


if __name__ == "__main__":
    unittest.main()
