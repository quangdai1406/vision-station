import hashlib
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from station import scanstore
from station.scanstore import MANIFEST, SCAN_JSON, ScanStore, list_scans, load_scan, verify_scan

CAMS = ["C1", "C2", "C3", "C4"]
STARTED = "2026-10-27T08:30:12Z"
FINISHED = "2026-10-27T08:41:55Z"
GOOD_LABEL = {
    "verdict": "GOOD",
    "defects": [],
    "status": "official",
    "labelled_by": "sup-1",
    "labelled_at": "2026-10-27T08:42:00Z",
    "approved_by": "admin-1",
    "approved_at": "2026-10-27T09:00:00Z",
}
BAD_LABEL = {
    "verdict": "BAD",
    "defects": ["doh_small", "doh_edge"],
    "status": "proposed",
    "labelled_by": "sup-1",
    "labelled_at": "2026-10-27T08:42:00Z",
}
DAY = ("scans", "L1", "2026", "10", "27")


def pic(tag):
    return f"png-{tag}".encode() * 40


class ScanTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.store = ScanStore(self.root, site="SITE", line="L1")

    def tearDown(self):
        self._tmp.cleanup()

    def begin(self, scan_id="SC-1", label=None, cams=CAMS, started=STARTED, store=None):
        store = store or self.store
        return store.begin_scan(
            scan_id, "PN-1", "toy-3", GOOD_LABEL if label is None else label,
            "op-1", list(cams), started,
        )

    def fill(self, writer, steps, bad=False):
        """steps like {"TOP": 3, "BOTTOM": 5}; BAD scans alternate defect_visible."""
        for side, count in steps.items():
            for step in range(1, count + 1):
                for cam in writer.camera_ids:
                    dv = ((step + CAMS.index(cam)) % 2 == 0) if bad else None
                    writer.add_picture(cam, side, step, pic(f"{cam}-{side}-{step}"), defect_visible=dv)

    def scan_dir(self, scan_id="SC-1"):
        return self.root.joinpath(*DAY, scan_id)

    def finished_scan(self, steps=None, label=None, scan_id="SC-1", bad=False):
        writer = self.begin(scan_id, label=label)
        self.fill(writer, steps or {"TOP": 1, "BOTTOM": 1}, bad=bad)
        return writer.finish(FINISHED)


class TestRoundTrip(ScanTestCase):
    def test_good_scan_round_trip_with_unequal_steps(self):
        writer = self.begin()
        self.fill(writer, {"TOP": 3, "BOTTOM": 5})
        final = writer.finish(FINISHED)

        self.assertEqual(final, self.scan_dir())
        self.assertEqual(sorted(p.name for p in self.root.joinpath(*DAY[:-1], "27").iterdir()), ["SC-1"])
        names = {p.name for p in final.iterdir()}
        self.assertIn("SITE_L1_20261027_083012_SC-1_C1_TOP_S01.png", names)
        self.assertIn("SITE_L1_20261027_083012_SC-1_C4_BOTTOM_S05.png", names)
        self.assertEqual(len(names), 32 + 2)  # 32 pictures, scan.json, manifest.sha256

        doc = load_scan(final)
        self.assertEqual(doc["steps"], {"TOP": 3, "BOTTOM": 5})
        self.assertEqual(doc["label"]["verdict"], "GOOD")
        self.assertEqual(doc["label"]["defects"], [])
        self.assertEqual(doc["camera_ids"], CAMS)
        self.assertEqual(len(doc["images"]), 32)
        self.assertTrue(all(img["defect_visible"] is None for img in doc["images"]))
        self.assertTrue(all(len(img["sha256"]) == 64 for img in doc["images"]))
        self.assertEqual(verify_scan(final), [])
        self.assertEqual(list_scans(self.root), [final])

    def test_bad_scan_defect_visible_per_picture(self):
        writer = self.begin(label=BAD_LABEL)
        self.fill(writer, {"TOP": 2, "BOTTOM": 1}, bad=True)
        final = writer.finish(FINISHED)

        doc = load_scan(final)
        self.assertEqual(doc["label"]["defects"], ["doh_small", "doh_edge"])
        self.assertEqual(len(doc["images"]), 12)
        for img in doc["images"]:
            expected = ((img["step"] + CAMS.index(img["camera_id"])) % 2 == 0)
            self.assertIs(img["defect_visible"], expected)
        self.assertEqual(verify_scan(final), [])

    def test_manifest_lists_pictures_and_scan_json_in_sha256sum_format(self):
        final = self.finished_scan()
        lines = (final / MANIFEST).read_text(encoding="utf-8").splitlines()
        names = [line.split("  ", 1)[1] for line in lines]
        self.assertIn(SCAN_JSON, names)
        self.assertEqual(len(names), 8 + 1)  # 8 pictures (2 sides x 4 cameras) plus scan.json
        for line in lines:
            digest, name = line.split("  ", 1)
            self.assertEqual(len(digest), 64)
            self.assertTrue((final / name).is_file())

    def test_manifest_is_written_after_scan_json(self):
        writer = self.begin()
        self.fill(writer, {"TOP": 1, "BOTTOM": 2})
        written = []
        real = scanstore._write_durable

        def spy(path, data):
            written.append(path.name)
            real(path, data)

        with mock.patch.object(scanstore, "_write_durable", side_effect=spy):
            writer.finish(FINISHED)
        self.assertEqual(written[-2:], [SCAN_JSON, MANIFEST])

    def test_folder_rename_happens_after_manifest_write(self):
        writer = self.begin()
        self.fill(writer, {"TOP": 1, "BOTTOM": 2})
        events = []
        real_write = scanstore._write_durable
        real_rename = os.rename

        def write_spy(path, data):
            events.append(f"write {path.name}")
            real_write(path, data)

        def rename_spy(src, dst):
            events.append(f"rename {pathlib.Path(src).name} -> {pathlib.Path(dst).name}")
            real_rename(src, dst)

        with mock.patch.object(scanstore, "_write_durable", side_effect=write_spy), \
                mock.patch.object(scanstore.os, "rename", side_effect=rename_spy):
            writer.finish(FINISHED)
        renames = [i for i, e in enumerate(events) if e.startswith("rename ")]
        self.assertEqual(events[renames[0]], "rename .SC-1.partial -> SC-1")
        self.assertEqual(len(renames), 1)
        self.assertGreater(renames[0], events.index(f"write {SCAN_JSON}"))
        self.assertGreater(renames[0], events.index(f"write {MANIFEST}"))
        self.assertEqual(renames[0], len(events) - 1)  # nothing is written after the rename
        self.assertEqual(verify_scan(self.scan_dir()), [])


class TestWriterLifecycle(ScanTestCase):
    def test_cancel_leaves_nothing(self):
        writer = self.begin()
        self.fill(writer, {"TOP": 2})
        writer.cancel()
        writer.cancel()  # second cancel is harmless
        self.assertEqual(list((self.root / "scans" / "L1").iterdir()), [])
        self.assertEqual(list_scans(self.root), [])
        with self.assertRaises(RuntimeError):
            writer.add_picture("C1", "TOP", 3, pic("late"))
        with self.assertRaises(RuntimeError):
            writer.finish(FINISHED)

    def test_exception_mid_write_leaves_no_final_folder(self):
        writer = self.begin()
        real = scanstore._write_durable
        calls = {"n": 0}

        def flaky(path, data):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("disk full (simulated)")
            return real(path, data)

        with mock.patch.object(scanstore, "_write_durable", side_effect=flaky):
            writer.add_picture("C1", "TOP", 1, pic("a"))
            writer.add_picture("C2", "TOP", 1, pic("b"))
            with self.assertRaises(RuntimeError):
                writer.add_picture("C3", "TOP", 1, pic("c"))
        self.assertFalse(self.scan_dir().exists())
        self.assertEqual(list_scans(self.root), [])
        self.assertEqual(list((self.root / "scans" / "L1").iterdir()), [])
        # the id is free again after the failed write
        retry = self.begin()
        self.fill(retry, {"TOP": 1, "BOTTOM": 1})
        self.assertEqual(retry.finish(FINISHED), self.scan_dir())

    def test_failed_finish_removes_staging_and_publishes_nothing(self):
        writer = self.begin()
        self.fill(writer, {"TOP": 1, "BOTTOM": 1})
        real = scanstore._write_durable

        def fail_on_scan_json(path, data):
            if path.name == SCAN_JSON:
                raise OSError("write failed (simulated)")
            return real(path, data)

        with mock.patch.object(scanstore, "_write_durable", side_effect=fail_on_scan_json):
            with self.assertRaises(OSError):
                writer.finish(FINISHED)
        self.assertFalse(self.scan_dir().exists())
        self.assertEqual(list_scans(self.root), [])
        self.assertEqual(list((self.root / "scans" / "L1").iterdir()), [])

    def test_interrupted_scan_is_only_staging_and_list_skips_it(self):
        writer = self.begin()
        self.fill(writer, {"TOP": 2})
        # simulate a crash: the writer is dropped without finish() or cancel()
        staging = self.root.joinpath(*DAY[:-1], "27", ".SC-1.partial")
        self.assertTrue(staging.is_dir())
        self.assertFalse(self.scan_dir().exists())
        self.assertEqual(list_scans(self.root), [])
        problems = verify_scan(staging)
        self.assertTrue(any("partial staging folder" in p for p in problems), problems)
        with self.assertRaises(FileExistsError):
            self.begin()


class TestRefusals(ScanTestCase):
    def test_duplicate_scan_id_refused(self):
        self.finished_scan()
        with self.assertRaises(FileExistsError):
            self.begin()
        with self.assertRaises(FileExistsError):
            self.begin(started="2026-10-28T07:00:00Z")  # other day, same id

    def test_duplicate_scan_id_refused_while_first_is_open(self):
        self.begin("SC-2")
        with self.assertRaises(FileExistsError):
            self.begin("SC-2")

    def test_good_with_defect_visible_refused(self):
        writer = self.begin()
        with self.assertRaises(ValueError):
            writer.add_picture("C1", "TOP", 1, pic("x"), defect_visible=True)
        writer.cancel()

    def test_bad_without_defect_visible_refused(self):
        writer = self.begin(label=BAD_LABEL)
        with self.assertRaises(ValueError):
            writer.add_picture("C1", "TOP", 1, pic("x"))
        with self.assertRaises(ValueError):
            writer.add_picture("C1", "TOP", 1, pic("x"), defect_visible="yes")
        writer.cancel()

    def test_label_rules_refused_at_begin(self):
        bad_labels = [
            dict(GOOD_LABEL, defects=["doh_small"]),
            dict(BAD_LABEL, defects=[]),
            dict(GOOD_LABEL, status="maybe"),
            dict(GOOD_LABEL, status="rejected"),  # rejected needs a reason
            dict(GOOD_LABEL, approved_by=None),  # approved_by and approved_at go together
            dict(GOOD_LABEL, extra_key=1),
            dict(GOOD_LABEL, labelled_by="  "),
        ]
        for label in bad_labels:
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.begin(label=label)
        self.assertEqual(list((self.root / "scans" / "L1").iterdir()), [])

    def test_non_contiguous_steps_refused_and_nothing_left(self):
        writer = self.begin()
        for cam in CAMS:
            writer.add_picture(cam, "TOP", 1, pic(cam))
            writer.add_picture(cam, "TOP", 3, pic(cam + "3"))
        with self.assertRaises(ValueError):
            writer.finish(FINISHED)
        self.assertFalse(self.scan_dir().exists())
        self.assertEqual(list((self.root / "scans" / "L1").iterdir()), [])

    def test_step_missing_one_camera_refused(self):
        writer = self.begin()
        for cam in CAMS:
            writer.add_picture(cam, "TOP", 1, pic(cam))
        writer.add_picture("C1", "TOP", 2, pic("c1-2"))
        with self.assertRaises(ValueError):
            writer.finish(FINISHED)
        self.assertFalse(self.scan_dir().exists())

    def test_empty_scan_and_empty_picture_refused(self):
        writer = self.begin()
        with self.assertRaises(ValueError):
            writer.add_picture("C1", "TOP", 1, b"")
        with self.assertRaises(ValueError):
            writer.finish(FINISHED)

    def test_picture_argument_rules(self):
        writer = self.begin()
        writer.add_picture("C1", "TOP", 1, pic("ok"))
        bad_calls = [
            dict(camera_id="C9", side="TOP", step=2),
            dict(camera_id="C1", side="LEFT", step=2),
            dict(camera_id="C1", side="TOP", step=0),
            dict(camera_id="C1", side="TOP", step=1),  # duplicate
            dict(camera_id="C2", side="TOP", step=1, ext="../png"),
            dict(camera_id="C2", side="TOP", step=1, ext="p/g"),
        ]
        for kwargs in bad_calls:
            kwargs.setdefault("ext", "png")
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                writer.add_picture(data=pic("x"), **kwargs)
        writer.cancel()

    def test_unsafe_names_refused(self):
        for scan_id in ("../x", "a/b", ".hidden", "", "a\\b", "..", "x\x00y"):
            with self.subTest(scan_id=scan_id), self.assertRaises(ValueError):
                self.begin(scan_id)
        with self.assertRaises(ValueError):
            self.begin(cams=["C1/..", "C2"])
        with self.assertRaises(ValueError):
            self.begin(cams=["C1", "C1"])
        for site in ("a/b", "..", ""):
            with self.subTest(site=site), self.assertRaises(ValueError):
                ScanStore(self.root, site=site)
        with self.assertRaises(ValueError):
            ScanStore(self.root, line="L/1")

    def test_finish_before_start_refused(self):
        writer = self.begin()
        self.fill(writer, {"TOP": 1})
        with self.assertRaises(ValueError):
            writer.finish("2026-10-27T08:00:00Z")
        self.assertFalse(self.scan_dir().exists())


class TestVerify(ScanTestCase):
    def picture_name(self, cam="C1", side="TOP", step=1):
        return f"SITE_L1_20261027_083012_SC-1_{cam}_{side}_S{step:02d}.png"

    def test_good_scan_verifies(self):
        self.assertEqual(verify_scan(self.finished_scan()), [])

    def test_altered_picture_caught(self):
        final = self.finished_scan()
        name = self.picture_name("C2", "BOTTOM")
        (final / name).write_bytes(b"altered")
        self.assertEqual(verify_scan(final), [f"hash mismatch: {name}"])

    def test_deleted_picture_caught(self):
        final = self.finished_scan()
        name = self.picture_name("C3", "TOP")
        (final / name).unlink()
        self.assertIn(f"missing file: {name}", verify_scan(final))

    def test_extra_file_caught(self):
        final = self.finished_scan()
        (final / "stray.txt").write_text("x", encoding="utf-8")
        self.assertEqual(verify_scan(final), ["not in manifest: stray.txt"])

    def test_changed_manifest_caught(self):
        final = self.finished_scan()
        name = self.picture_name("C1", "TOP")
        lines = (final / MANIFEST).read_text(encoding="utf-8").splitlines()
        lines = [("0" * 64 + "  " + line.split("  ", 1)[1]) if line.endswith(name) else line for line in lines]
        (final / MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertEqual(
            verify_scan(final),
            [f"hash mismatch: {name}", f"scan.json disagrees with manifest: {name}"],
        )

    def test_missing_manifest_caught(self):
        final = self.finished_scan()
        (final / MANIFEST).unlink()
        self.assertEqual(verify_scan(final), ["missing manifest.sha256"])

    def test_edited_label_caught_by_hash_and_rules(self):
        final = self.finished_scan()
        doc = json.loads((final / SCAN_JSON).read_text(encoding="utf-8"))
        doc["label"]["defects"] = ["doh_small"]  # GOOD with a defect
        (final / SCAN_JSON).write_text(json.dumps(doc), encoding="utf-8")
        problems = verify_scan(final)
        self.assertIn(f"hash mismatch: {SCAN_JSON}", problems)
        self.assertIn("scan.json invalid: a GOOD label has no defects", problems)
        with self.assertRaises(ValueError):
            load_scan(final)

    def test_symlink_caught(self):
        final = self.finished_scan()
        target = final / self.picture_name("C4", "BOTTOM")
        link = final / "link.png"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available here")
        self.assertEqual(verify_scan(final), ["symlink: link.png"])

    def test_missing_scan_json_caught(self):
        final = self.finished_scan()
        (final / SCAN_JSON).unlink()
        problems = verify_scan(final)
        self.assertIn("missing scan.json", problems)

    def forge(self, final, edit_doc=None, rename=None):
        """Change scan.json, a picture name, or both, and rebuild the manifest from scan.json.

        This is what someone who edits the whole folder would do; it is not what the writer does.
        """
        doc = json.loads((final / SCAN_JSON).read_text(encoding="utf-8"))
        if edit_doc:
            edit_doc(doc)
        if rename:
            old, new = rename
            os.rename(final / old, final / new)
            for img in doc["images"]:
                if img["file"] == old:
                    img["file"] = new
        data = scanstore._dump_json(doc)
        (final / SCAN_JSON).write_bytes(data)
        lines = [f"{img['sha256']}  {img['file']}" for img in doc["images"]]
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {SCAN_JSON}")
        (final / MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_altered_site_in_scan_json_caught(self):
        final = self.finished_scan()
        doc = json.loads((final / SCAN_JSON).read_text(encoding="utf-8"))
        doc["site"] = "EVIL"
        (final / SCAN_JSON).write_text(json.dumps(doc), encoding="utf-8")  # manifest not updated
        problems = verify_scan(final)
        self.assertIn(f"hash mismatch: {SCAN_JSON}", problems)
        name_problems = [p for p in problems if "file name does not match the scan" in p]
        self.assertEqual(len(name_problems), 8)  # every picture name carries the real site

    def test_altered_site_with_rebuilt_manifest_still_caught_by_names(self):
        final = self.finished_scan()
        self.forge(final, edit_doc=lambda doc: doc.update(site="EVIL"))
        problems = verify_scan(final)
        self.assertNotIn(f"hash mismatch: {SCAN_JSON}", problems)
        self.assertIn("scan.json invalid: image 0 file name does not match the scan", problems)
        self.assertTrue(all("does not match the scan" in p for p in problems), problems)

    def test_altered_line_in_scan_json_caught(self):
        final = self.finished_scan()
        doc = json.loads((final / SCAN_JSON).read_text(encoding="utf-8"))
        doc["line"] = "L2"
        (final / SCAN_JSON).write_text(json.dumps(doc), encoding="utf-8")
        problems = verify_scan(final)
        self.assertIn(f"hash mismatch: {SCAN_JSON}", problems)
        self.assertIn(
            "scan folder path does not match scan.json (line 'L2', started_utc date 2026/10/27)",
            problems,
        )

    def test_scan_moved_to_other_day_folder_caught(self):
        final = self.finished_scan()
        other_day = self.root / "scans" / "L1" / "2026" / "10" / "28"
        other_day.mkdir(parents=True)
        moved = other_day / "SC-1"
        os.rename(final, moved)
        self.assertEqual(
            verify_scan(moved),
            ["scan folder path does not match scan.json (line 'L1', started_utc date 2026/10/27)"],
        )

    def test_picture_renamed_to_wrong_site_caught(self):
        final = self.finished_scan()
        name = self.picture_name("C1", "TOP")
        wrong = name.replace("SITE_", "WRONG_", 1)
        os.rename(final / name, final / wrong)
        problems = verify_scan(final)
        self.assertIn(f"missing file: {name}", problems)
        self.assertIn(f"not in manifest: {wrong}", problems)

    def test_picture_renamed_to_wrong_site_with_rebuilt_manifest_caught_by_name_rule(self):
        final = self.finished_scan()
        name = self.picture_name("C1", "TOP")
        wrong = name.replace("SITE_", "WRONG_", 1)
        self.forge(final, rename=(name, wrong))
        self.assertEqual(verify_scan(final), ["scan.json invalid: image 0 file name does not match the scan"])


class TestReadAndList(ScanTestCase):
    def test_load_scan_refuses_invalid_json_document(self):
        final = self.finished_scan()
        (final / SCAN_JSON).write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_scan(final)

    def test_list_scans_by_line_and_all_lines(self):
        first = self.finished_scan()
        other = ScanStore(self.root, site="SITE", line="L2")
        writer = self.begin("SC-9", store=other)
        self.fill(writer, {"TOP": 1, "BOTTOM": 1})
        second = writer.finish(FINISHED)
        self.assertEqual(list_scans(self.root), sorted([first, second]))
        self.assertEqual(list_scans(self.root, line="L2"), [second])
        self.assertEqual(list_scans(self.root, line="L1"), [first])

    def test_list_scans_empty_root(self):
        self.assertEqual(list_scans(self.root / "nothing-here"), [])


class TestSetDefectVisible(ScanTestCase):
    def test_mark_changed_before_finish_is_written_to_scan_json(self):
        writer = self.begin(label=BAD_LABEL)
        self.fill(writer, {"TOP": 2, "BOTTOM": 1}, bad=True)
        writer.set_defect_visible("C2", "TOP", 2, True)
        writer.set_defect_visible("C2", "TOP", 2, False)
        writer.set_defect_visible("C4", "BOTTOM", 1, True)
        final = writer.finish(FINISHED)
        doc = load_scan(final)
        visible = {(i["camera_id"], i["side"], i["step"]): i["defect_visible"] for i in doc["images"]}
        self.assertIs(visible[("C2", "TOP", 2)], False)
        self.assertIs(visible[("C4", "BOTTOM", 1)], True)
        self.assertIs(visible[("C1", "TOP", 1)], False)
        self.assertEqual(verify_scan(final), [])

    def test_refused_for_good_unknown_closed_and_bad_value(self):
        good = self.begin("SC-G")
        self.fill(good, {"TOP": 1})
        with self.assertRaises(ValueError):
            good.set_defect_visible("C1", "TOP", 1, True)
        good.cancel()

        bad = self.begin("SC-B", label=BAD_LABEL)
        self.fill(bad, {"TOP": 1}, bad=True)
        with self.assertRaises(ValueError):
            bad.set_defect_visible("C1", "TOP", 2, True)  # no such step
        with self.assertRaises(ValueError):
            bad.set_defect_visible("C1", "BOTTOM", 1, True)  # no such side
        with self.assertRaises(ValueError):
            bad.set_defect_visible("C1", "TOP", 1, "yes")  # not a bool
        bad.finish(FINISHED)
        with self.assertRaises(RuntimeError):
            bad.set_defect_visible("C1", "TOP", 1, True)  # after finish
        self.assertEqual(verify_scan(self.scan_dir("SC-B")), [])


if __name__ == "__main__":
    unittest.main()
