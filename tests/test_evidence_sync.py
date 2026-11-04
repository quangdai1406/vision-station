"""Tests for station/evidence_sync.py (plan S4: evidence transport to the laptop).

Run from the project root:

    python3 -m unittest discover -s tests
"""
import contextlib
import hashlib
import io
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from station import evidence_sync as es

CAMERAS = ("C1", "C2", "C3", "C4")
SIDES = ("TOP", "BOTTOM")


def make_part(local_root, part_id="PN-1-0001", day=("2026", "11", "04")):
    """Write one saved part the way station/storage.py does: 8 pictures, part.json, manifest last."""
    part_dir = pathlib.Path(local_root) / "store" / "L1" / day[0] / day[1] / day[2] / part_id
    part_dir.mkdir(parents=True)
    entries = []
    for cam in CAMERAS:
        for side in SIDES:
            name = f"SITE_L1_20261104_120000_{part_id}_{cam}_{side}.png"
            data = f"{part_id}-{cam}-{side}".encode() * 50
            (part_dir / name).write_bytes(data)
            entries.append((name, hashlib.sha256(data).hexdigest()))
    doc = {
        "part_id": part_id,
        "session_id": f"S-{part_id}",
        "images": [{"file": n, "sha256": h} for n, h in entries],
    }
    (part_dir / es.PART_JSON).write_text(json.dumps(doc), encoding="utf-8")
    manifest = "".join(f"{h}  {n}\n" for n, h in entries)
    (part_dir / es.MANIFEST).write_text(manifest, encoding="utf-8")
    return part_dir


def snapshot(root):
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = pathlib.Path(dirpath) / name
            out[str(path.relative_to(root))] = path.read_bytes()
    return out


class EvidenceSyncTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self._tmp.name)
        self.local = base / "station"
        self.dest = base / "laptop_store"  # stands in for the mounted SMB folder
        self.dest.mkdir()
        self.part = make_part(self.local)

    def tearDown(self):
        self._tmp.cleanup()

    def final_path(self, part_id="PN-1-0001", day=("2026", "11", "04")):
        return self.dest / "L1" / day[0] / day[1] / day[2] / part_id


class CopyTests(EvidenceSyncTestCase):
    def test_copy_keeps_layout(self):
        ref = es.sync_part(self.part, self.dest)
        final = self.final_path()
        self.assertEqual(ref, str(final))
        self.assertEqual(snapshot(final), snapshot(self.part))

    def test_source_is_never_deleted(self):
        before = snapshot(self.part)
        es.sync_part(self.part, self.dest)
        self.assertEqual(snapshot(self.part), before)

    def test_every_copied_file_matches_manifest_hash(self):
        es.sync_part(self.part, self.dest)
        final = self.final_path()
        for line in (final / es.MANIFEST).read_text(encoding="utf-8").splitlines():
            digest, _sep, name = line.partition("  ")
            self.assertEqual(hashlib.sha256((final / name).read_bytes()).hexdigest(), digest)

    def test_no_temp_folder_left_behind(self):
        es.sync_part(self.part, self.dest)
        names = [p.name for p in self.final_path().parent.iterdir()]
        self.assertEqual(names, ["PN-1-0001"])

    def test_corrupted_copy_raises_and_leaves_no_final_file(self):
        real_copy = es._copy_file
        calls = {"n": 0}

        def corrupting_copy(src, dst):
            real_copy(src, dst)
            calls["n"] += 1
            if calls["n"] == 3:
                dst.write_bytes(b"damaged in transit")

        with mock.patch.object(es, "_copy_file", side_effect=corrupting_copy):
            with self.assertRaises(es.SyncError):
                es.sync_part(self.part, self.dest)
        self.assertFalse(self.final_path().exists())
        leftovers = list(self.final_path().parent.iterdir()) if self.final_path().parent.exists() else []
        self.assertEqual(leftovers, [])

    def test_tampered_source_is_refused_before_copying(self):
        (self.part / "SITE_L1_20261104_120000_PN-1-0001_C1_TOP.png").write_bytes(b"changed")
        with self.assertRaises(es.SyncError):
            es.sync_part(self.part, self.dest)
        self.assertFalse((self.dest / "L1").exists())

    def test_unlisted_file_in_source_is_refused(self):
        (self.part / "extra.png").write_bytes(b"not in manifest")
        with self.assertRaises(es.SyncError):
            es.sync_part(self.part, self.dest)
        self.assertFalse(self.final_path().exists())

    def test_part_outside_store_layout_is_refused(self):
        loose = pathlib.Path(self._tmp.name) / "elsewhere" / "PN-1-0001"
        loose.mkdir(parents=True)
        with self.assertRaises(es.SyncError):
            es.sync_part(loose, self.dest)


class IdempotencyTests(EvidenceSyncTestCase):
    def test_second_run_copies_nothing_and_returns_same_ref(self):
        first = es.sync_part(self.part, self.dest)
        before = snapshot(self.final_path())
        real_copy = es._copy_file
        with mock.patch.object(es, "_copy_file", side_effect=real_copy) as spy:
            second = es.sync_part(self.part, self.dest)
        self.assertEqual(first, second)
        self.assertEqual(spy.call_count, 0)
        self.assertEqual(snapshot(self.final_path()), before)

    def test_existing_destination_with_different_manifest_is_not_overwritten(self):
        es.sync_part(self.part, self.dest)
        final = self.final_path()
        (final / es.MANIFEST).write_text("0" * 64 + "  other.png\n", encoding="utf-8")
        before = snapshot(final)
        with self.assertRaises(es.SyncError):
            es.sync_part(self.part, self.dest)
        self.assertEqual(snapshot(final), before)

    def test_archived_part_with_tombstone_is_not_recreated(self):
        es.sync_part(self.part, self.dest)
        final = self.final_path()
        for picture in final.glob("*.png"):
            picture.unlink()
        (final / es.PURGED_FILE).write_text("{}", encoding="utf-8")
        ref = es.sync_part(self.part, self.dest)
        self.assertEqual(ref, str(final))
        self.assertEqual(list(final.glob("*.png")), [])


class QueueTests(EvidenceSyncTestCase):
    def test_enqueue_then_pending_lists_the_part(self):
        queue = es.SyncQueue(self.local)
        queue.enqueue(self.part)
        pending = queue.pending()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["part_dir"], str(self.part))

    def test_duplicate_enqueue_is_ignored(self):
        queue = es.SyncQueue(self.local)
        queue.enqueue(self.part)
        queue.enqueue(self.part)
        self.assertEqual(len(queue.pending()), 1)

    def test_queue_write_leaves_no_temp_file(self):
        es.SyncQueue(self.local).enqueue(self.part)
        names = sorted(p.name for p in self.local.iterdir())
        self.assertFalse([n for n in names if n.startswith(".tmp-")])

    def test_queue_survives_a_new_instance(self):
        es.SyncQueue(self.local).enqueue(self.part)
        self.assertEqual(len(es.SyncQueue(self.local).pending()), 1)

    def test_unreachable_destination_keeps_item_queued_and_retries_later(self):
        queue = es.SyncQueue(self.local)
        queue.enqueue(self.part)
        missing = self.dest.parent / "not-mounted-yet"
        report = queue.run_once(missing)
        self.assertEqual((report.synced, report.failed, report.pending), (0, 1, 1))
        self.assertFalse(missing.exists())  # a missing mount must not be created locally
        self.assertEqual(len(queue.pending()), 1)
        self.assertIsNotNone(queue.pending()[0]["last_error"])

        missing.mkdir()
        report = queue.run_once(missing)
        self.assertEqual((report.synced, report.failed, report.pending), (1, 0, 0))
        self.assertEqual(queue.pending(), [])

    def test_item_marked_done_only_after_verified_copy(self):
        queue = es.SyncQueue(self.local)
        queue.enqueue(self.part)
        real_copy = es._copy_file

        def corrupting_copy(src, dst):
            real_copy(src, dst)
            dst.write_bytes(b"bad")

        with mock.patch.object(es, "_copy_file", side_effect=corrupting_copy):
            report = queue.run_once(self.dest)
        self.assertEqual((report.synced, report.failed), (0, 1))
        self.assertEqual(len(queue.pending()), 1)
        self.assertFalse(self.final_path().exists())

        report = queue.run_once(self.dest)
        self.assertEqual((report.synced, report.pending), (1, 0))
        item = json.loads((self.local / es.QUEUE_NAME).read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(item["state"], es.DONE)
        self.assertEqual(item["archive_ref"], str(self.final_path()))

    def test_run_once_retries_each_pending_item_and_keeps_failures(self):
        queue = es.SyncQueue(self.local)
        other = make_part(self.local, part_id="PN-1-0002")
        queue.enqueue(self.part)
        queue.enqueue(other)
        os.remove(other / "SITE_L1_20261104_120000_PN-1-0002_C4_BOTTOM.png")
        report = queue.run_once(self.dest)
        self.assertEqual((report.synced, report.failed, report.pending), (1, 1, 1))
        self.assertEqual(queue.pending()[0]["part_dir"], str(other))


class CliTests(EvidenceSyncTestCase):
    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = es.main(list(argv))
        return code, out.getvalue()

    def test_run_then_status(self):
        es.SyncQueue(self.local).enqueue(self.part)
        code, _ = self.run_cli("run", "--local", str(self.local), "--dest", str(self.dest))
        self.assertEqual(code, 0)
        code, text = self.run_cli("status", "--local", str(self.local))
        self.assertEqual(code, 0)
        self.assertIn("pending 0", text)

    def test_run_with_unreachable_destination_exits_nonzero(self):
        es.SyncQueue(self.local).enqueue(self.part)
        code, text = self.run_cli(
            "run", "--local", str(self.local), "--dest", str(self.dest.parent / "missing")
        )
        self.assertEqual(code, 1)
        self.assertIn("failed 1", text)
        code, text = self.run_cli("status", "--local", str(self.local))
        self.assertIn("pending 1", text)


if __name__ == "__main__":
    unittest.main()
