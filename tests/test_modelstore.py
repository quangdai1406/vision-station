import hashlib
import importlib.util
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from station import modelstore as ms
from station.contracts import Frame, Side, StationError

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np
    from station import golden

NOW1 = "2026-10-23T08:00:00Z"
NOW2 = "2026-10-24T08:00:00Z"
NOW3 = "2026-10-25T08:00:00Z"
PART = "PN-100"


def _manifest(**over):
    base = {
        "type": "golden",
        "camera_ids": ["C1"],
        "trained_on": {"good_images": 10, "bad_images": 0, "since": "2026-10-01",
                       "until": "2026-10-20", "parts": 1},
        "metrics": {"note": "test only"},
        "notes": "",
    }
    base.update(over)
    return base


class _StoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = self.base / "models"
        self.src_dir = self.base / "src"
        self.src_dir.mkdir()
        self.store = ms.ModelStore(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _file(self, name, data: bytes) -> Path:
        path = self.src_dir / name
        path.write_bytes(data)
        return path

    def _candidate(self, version, data=b"model-bytes", part=PART, store=None, now=NOW1, **over):
        src = self._file("model.bin", data)
        return (store or self.store).add_candidate(part, version, src, _manifest(**over), now)

    def _part_dir(self, part=PART):
        return self.root / ms.safe_part(part)

    def _version_dir(self, version, part=PART):
        return self._part_dir(part) / version


class SafePartTests(unittest.TestCase):
    def test_plain_part_keeps_readable_name_plus_hash(self):
        name = ms.safe_part("PN-100")
        self.assertTrue(re.fullmatch(r"PN-100-[0-9a-f]{8}", name), name)
        digest = hashlib.sha256("PN-100".encode("utf-8")).hexdigest()[:8]
        self.assertTrue(name.endswith("-" + digest))

    def test_unsafe_characters_become_underscores(self):
        name = ms.safe_part("A/B\\C:D*E?F")
        self.assertTrue(re.fullmatch(r"A_B_C_D_E_F-[0-9a-f]{8}", name), name)

    def test_colliding_names_get_different_folders(self):
        a, b = ms.safe_part("A/B"), ms.safe_part("A_B")
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("A_B-") and b.startswith("A_B-"))

    def test_reserved_device_names_are_prefixed(self):
        self.assertTrue(ms.safe_part("CON").startswith("part_CON-"))
        self.assertTrue(ms.safe_part("lpt1.txt").startswith("part_lpt1.txt-"))

    def test_trailing_dot_and_space_removed(self):
        name = ms.safe_part("X. ")
        self.assertTrue(name.startswith("X-"), name)

    def test_length_is_capped_at_80(self):
        self.assertEqual(len(ms.safe_part("A" * 200)), 80)

    def test_empty_names_rejected(self):
        for bad in ("", "   "):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ms.safe_part(bad)

    def test_dot_only_name_is_safe(self):
        name = ms.safe_part("..")
        self.assertTrue(re.fullmatch(r"part-[0-9a-f]{8}", name), name)


class ModelStoreTests(_StoreCase):
    def test_add_candidate_copies_and_hashes(self):
        data = b"abc" * 100
        path = self._candidate("v1", data)
        self.assertEqual(path.read_bytes(), data)
        self.assertEqual(path.parent.name, "v1")
        (manifest,) = self.store.list_versions(PART)
        self.assertEqual(manifest["schema"], 1)
        self.assertEqual(manifest["part_number"], PART)
        self.assertEqual(manifest["version"], "v1")
        self.assertEqual(manifest["type"], "golden")
        self.assertEqual(manifest["status"], "candidate")
        self.assertEqual(manifest["created_utc"], NOW1)
        self.assertEqual(manifest["model_file"], "model.bin")
        self.assertEqual(manifest["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(manifest["camera_ids"], ["C1"])
        self.assertEqual(manifest["trained_on"]["good_images"], 10)
        self.assertFalse((self._part_dir() / ms.CURRENT_NAME).exists())
        self.assertEqual(self.store.verify(PART, "v1"), [])

    def test_duplicate_version_refused_and_first_copy_kept(self):
        self._candidate("v1", b"first")
        with self.assertRaises(ms.ModelError):
            self._candidate("v1", b"second")
        self.assertEqual((self._version_dir("v1") / "model.bin").read_bytes(), b"first")

    def test_bad_version_names_refused(self):
        bad = ["", "../x", "a b", "a/b", "x" * 65, ".", "..", ".hidden", "v1.", "CON", "v1\n"]
        for version in bad:
            with self.subTest(version=version):
                with self.assertRaises(ms.ModelError):
                    self._candidate(version)
        self.assertEqual(self.store.list_versions(PART), [])

    def test_bad_manifest_type_refused(self):
        src = self._file("m.bin", b"x")
        with self.assertRaises(ValueError):
            self.store.add_candidate(PART, "v1", src, _manifest(type="Golden!"), NOW1)

    def test_add_candidate_never_changes_current(self):
        self._candidate("v1")
        self.store.approve(PART, "v1", "Dai", "first approval", NOW1)
        self._candidate("v2", b"other", now=NOW2)
        self.assertEqual(self.store.current(PART).version, "v1")

    def test_list_versions_sorted_by_created_then_version(self):
        self._candidate("b", now=NOW2)
        self._candidate("a", now=NOW2)
        self._candidate("c", now=NOW1)
        self.assertEqual([m["version"] for m in self.store.list_versions(PART)], ["c", "a", "b"])
        self.assertEqual(self.store.list_versions("OTHER-PART"), [])

    def test_no_staging_folders_left_behind(self):
        self._candidate("v1")
        with self.assertRaises(ms.ModelError):
            self._candidate("v1")
        leftovers = [p.name for p in self._part_dir().iterdir() if p.name.startswith(".")]
        self.assertEqual(leftovers, [])

    def test_verify_reports_each_problem(self):
        self._candidate("v1")
        vdir = self._version_dir("v1")
        self.assertEqual(self.store.verify(PART, "v1"), [])
        self.assertTrue(self.store.verify(PART, "missing-version"))

        (vdir / "model.bin").write_bytes(b"changed")
        self.assertTrue(any("sha256" in p for p in self.store.verify(PART, "v1")))
        (vdir / "model.bin").unlink()
        self.assertTrue(any("missing" in p for p in self.store.verify(PART, "v1")))

        manifest_path = vdir / ms.MANIFEST_NAME
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["part_number"] = "OTHER"
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        self.assertTrue(any("part_number" in p for p in self.store.verify(PART, "v1")))

        manifest_path.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.store.verify(PART, "v1"), ["manifest.json is not valid JSON"])


class ApproveTests(_StoreCase):
    def test_approve_needs_by_and_reason(self):
        self._candidate("v1")
        for by, reason in (("", "r"), ("   ", "r"), ("Dai", ""), ("Dai", " \t ")):
            with self.subTest(by=by, reason=reason):
                with self.assertRaises(ValueError):
                    self.store.approve(PART, "v1", by, reason, NOW1)
        self.assertIsNone(self.store.current(PART))
        self.assertEqual(self.store.approvals(PART), [])

    def test_approve_unknown_version_refused(self):
        with self.assertRaises(ms.ModelError):
            self.store.approve(PART, "v9", "Dai", "why", NOW1)

    def test_tampered_candidate_cannot_be_approved(self):
        self._candidate("v1")
        (self._version_dir("v1") / "model.bin").write_bytes(b"tampered")
        with self.assertRaises(ms.ModelError):
            self.store.approve(PART, "v1", "Dai", "why", NOW1)
        self.assertIsNone(self.store.current(PART))

    def test_approve_writes_current_manifest_and_log(self):
        self._candidate("v1")
        info = self.store.approve(PART, "v1", "  Dai  ", "  first approval ", NOW1)
        self.assertIsInstance(info, ms.ModelInfo)
        self.assertEqual(info.version, "v1")
        self.assertEqual(info.type, "golden")
        self.assertEqual(info.path, self._version_dir("v1") / "model.bin")
        self.assertEqual(info.approved_by, "Dai")
        self.assertEqual(info.reason, "first approval")
        self.assertEqual(info.manifest["status"], "approved")

        current = json.loads((self._part_dir() / ms.CURRENT_NAME).read_text(encoding="utf-8"))
        self.assertEqual(current["version"], "v1")
        self.assertEqual(current["approved_by"], "Dai")
        self.assertEqual(current["approved_utc"], NOW1)

        (entry,) = self.store.approvals(PART)
        self.assertEqual(entry["action"], "approve")
        self.assertEqual(entry["version"], "v1")
        self.assertEqual(entry["by"], "Dai")
        self.assertIsNone(entry["previous_version"])
        (manifest,) = self.store.list_versions(PART)
        self.assertEqual(manifest["status"], "approved")

    def test_second_approval_retires_first_and_logs_both(self):
        self._candidate("v1")
        self._candidate("v2", b"second", now=NOW2)
        self.store.approve(PART, "v1", "Dai", "first", NOW1)
        self.store.approve(PART, "v2", "Lan", "second", NOW2)

        statuses = {m["version"]: m["status"] for m in self.store.list_versions(PART)}
        self.assertEqual(statuses, {"v1": "retired", "v2": "approved"})
        log = self.store.approvals(PART)
        self.assertEqual([e["action"] for e in log], ["approve", "approve"])
        self.assertEqual([e["version"] for e in log], ["v1", "v2"])
        self.assertEqual(log[1]["previous_version"], "v1")
        self.assertEqual(self.store.current(PART).version, "v2")

    def test_reapproving_same_version_keeps_it_approved(self):
        self._candidate("v1")
        self.store.approve(PART, "v1", "Dai", "first", NOW1)
        self.store.approve(PART, "v1", "Dai", "again", NOW2)
        (manifest,) = self.store.list_versions(PART)
        self.assertEqual(manifest["status"], "approved")
        self.assertEqual(self.store.current(PART).version, "v1")


class CurrentTests(_StoreCase):
    def test_none_when_nothing_approved(self):
        self._candidate("v1")
        self.assertIsNone(self.store.current(PART))
        self.assertIsNone(self.store.current("NEVER-SEEN"))

    def test_returns_approved_model(self):
        self._candidate("v1")
        self.store.approve(PART, "v1", "Dai", "ok", NOW1)
        info = self.store.current(PART)
        self.assertEqual((info.part_number, info.version, info.approved_by, info.approved_utc),
                         (PART, "v1", "Dai", NOW1))
        self.assertEqual(info.reason, "ok")

    def test_tampered_model_file_raises(self):
        self._candidate("v1")
        self.store.approve(PART, "v1", "Dai", "ok", NOW1)
        (self._version_dir("v1") / "model.bin").write_bytes(b"swapped")
        with self.assertRaises(ms.ModelError):
            self.store.current(PART)

    def test_missing_model_file_raises(self):
        self._candidate("v1")
        self.store.approve(PART, "v1", "Dai", "ok", NOW1)
        (self._version_dir("v1") / "model.bin").unlink()
        with self.assertRaises(ms.ModelError):
            self.store.current(PART)

    def test_missing_version_folder_raises(self):
        self._candidate("v1")
        self.store.approve(PART, "v1", "Dai", "ok", NOW1)
        shutil.rmtree(self._version_dir("v1"))
        with self.assertRaises(ms.ModelError):
            self.store.current(PART)

    def test_malformed_current_json_raises(self):
        self._candidate("v1")
        self.store.approve(PART, "v1", "Dai", "ok", NOW1)
        (self._part_dir() / ms.CURRENT_NAME).write_text("{not json", encoding="utf-8")
        with self.assertRaises(ms.ModelError):
            self.store.current(PART)


class RollbackTests(_StoreCase):
    def test_rollback_to_earlier_approved_version(self):
        self._candidate("v1")
        self._candidate("v2", b"second", now=NOW2)
        self.store.approve(PART, "v1", "Dai", "first", NOW1)
        self.store.approve(PART, "v2", "Dai", "second", NOW2)
        info = self.store.rollback(PART, "Lan", "v2 misbehaves", NOW3)
        self.assertEqual(info.version, "v1")
        self.assertEqual(self.store.current(PART).version, "v1")
        statuses = {m["version"]: m["status"] for m in self.store.list_versions(PART)}
        self.assertEqual(statuses, {"v1": "approved", "v2": "retired"})
        self.assertEqual(self.store.approvals(PART)[-1]["action"], "rollback")

    def test_rollback_skips_tampered_version(self):
        self._candidate("v1")
        self._candidate("v2", b"second", now=NOW2)
        self._candidate("v3", b"third", now=NOW3)
        for version, when in (("v1", NOW1), ("v2", NOW2), ("v3", NOW3)):
            self.store.approve(PART, version, "Dai", "step", when)
        (self._version_dir("v2") / "model.bin").write_bytes(b"damaged")
        info = self.store.rollback(PART, "Dai", "v3 bad", NOW3)
        self.assertEqual(info.version, "v1")

    def test_no_previous_version_raises(self):
        self._candidate("v1")
        self.store.approve(PART, "v1", "Dai", "only", NOW1)
        with self.assertRaises(ms.ModelError):
            self.store.rollback(PART, "Dai", "why", NOW2)
        self.assertEqual(self.store.current(PART).version, "v1")

    def test_nothing_approved_raises(self):
        self._candidate("v1")
        with self.assertRaises(ms.ModelError):
            self.store.rollback(PART, "Dai", "why", NOW2)

    def test_rollback_needs_by_and_reason(self):
        self._candidate("v1")
        self.store.approve(PART, "v1", "Dai", "only", NOW1)
        with self.assertRaises(ValueError):
            self.store.rollback(PART, "", "why", NOW2)
        with self.assertRaises(ValueError):
            self.store.rollback(PART, "Dai", " ", NOW2)


class SharedStoreTests(_StoreCase):
    def test_atomic_write_leaves_no_tmp_files(self):
        self._candidate("v1")
        self._candidate("v2", b"second", now=NOW2)
        self.store.approve(PART, "v1", "Dai", "one", NOW1)
        self.store.approve(PART, "v2", "Dai", "two", NOW2)
        leftovers = [
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if p.name.endswith(".tmp") or p.name.startswith(".new-")
        ]
        self.assertEqual(leftovers, [])

    def test_two_instances_see_each_other(self):
        other = ms.ModelStore(self.root)
        self._candidate("v1")
        self._candidate("v2", b"second", now=NOW2, store=other)
        self.store.approve(PART, "v1", "Dai", "one", NOW1)
        self.assertEqual(other.current(PART).version, "v1")
        other.approve(PART, "v2", "Lan", "two", NOW2)
        self.assertEqual(self.store.current(PART).version, "v2")
        self.assertEqual(len(self.store.approvals(PART)), 2)

    def test_approval_log_lines_are_whole_json_records(self):
        self._candidate("v1")
        self._candidate("v2", b"second", now=NOW2)
        self.store.approve(PART, "v1", "Dai", "one", NOW1)
        self.store.approve(PART, "v2", "Dai", "two", NOW2)
        raw = (self._part_dir() / ms.APPROVALS_NAME).read_text(encoding="utf-8")
        self.assertTrue(raw.endswith("\n"))
        lines = raw.splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertEqual(json.loads(line)["action"], "approve")

    def test_damaged_approval_log_raises(self):
        self._candidate("v1")
        self.store.approve(PART, "v1", "Dai", "one", NOW1)
        with (self._part_dir() / ms.APPROVALS_NAME).open("a", encoding="utf-8") as fh:
            fh.write("{broken\n")
        with self.assertRaises(ms.ModelError):
            self.store.approvals(PART)


class ActivationStackTests(_StoreCase):
    def _three_approved(self):
        self._candidate("v1")
        self._candidate("v2", b"second", now=NOW2)
        self._candidate("v3", b"third", now=NOW3)
        for version, when in (("v1", NOW1), ("v2", NOW2), ("v3", NOW3)):
            self.store.approve(PART, version, "Dai", "step", when)

    def test_repeated_rollbacks_walk_down_then_stop(self):
        self._three_approved()
        self.assertEqual(self.store.rollback(PART, "Lan", "v3 bad", NOW3).version, "v2")
        self.assertEqual(self.store.rollback(PART, "Lan", "v2 bad", NOW3).version, "v1")
        with self.assertRaises(ms.ModelError):
            self.store.rollback(PART, "Lan", "v1 bad", NOW3)
        self.assertEqual(self.store.current(PART).version, "v1")

    def test_rolled_back_from_version_never_returns(self):
        self._three_approved()
        self.store.rollback(PART, "Lan", "v3 bad", NOW3)
        self.store.rollback(PART, "Lan", "v2 bad", NOW3)
        self.assertEqual(self.store.history(PART), ["v1"])
        with self.assertRaises(ms.ModelError):
            self.store.rollback(PART, "Lan", "again", NOW3)
        self.assertEqual(self.store.current(PART).version, "v1")

    def test_manual_approve_after_rollback_becomes_current_again(self):
        self._three_approved()
        self.store.rollback(PART, "Lan", "v3 bad", NOW3)
        self.store.approve(PART, "v3", "Dai", "fixed, re-approved", NOW3)
        self.assertEqual(self.store.current(PART).version, "v3")
        self.assertEqual(self.store.history(PART), ["v1", "v2", "v3"])
        self.assertEqual(self.store.rollback(PART, "Lan", "v3 bad again", NOW3).version, "v2")

    def test_rollback_skips_tampered_and_raises_when_none_verifies(self):
        self._three_approved()
        (self._version_dir("v2") / "model.bin").write_bytes(b"damaged")
        (self._version_dir("v1") / "model.bin").write_bytes(b"damaged")
        with self.assertRaises(ms.ModelError):
            self.store.rollback(PART, "Lan", "v3 bad", NOW3)
        self.assertEqual(self.store.current(PART).version, "v3")

    def test_log_line_is_written_before_current_json_changes(self):
        self._three_approved()
        real_replace = os.replace
        seen = []

        def spy_replace(src, dst, *args, **kwargs):
            if Path(dst).name == ms.CURRENT_NAME:
                seen.append([(e["action"], e["version"]) for e in self.store.approvals(PART)])
            return real_replace(src, dst, *args, **kwargs)

        self.store.rollback(PART, "Lan", "v3 bad", NOW3)
        with mock.patch.object(ms.os, "replace", side_effect=spy_replace):
            self.store.approve(PART, "v3", "Dai", "again", NOW3)
        self.assertEqual(seen[-1][-1], ("approve", "v3"))

    def test_failed_switch_logs_failed_line_and_keeps_old_model(self):
        self._candidate("v1")
        self._candidate("v2", b"second", now=NOW2)
        self.store.approve(PART, "v1", "Dai", "first", NOW1)
        real_replace = os.replace

        def failing_replace(src, dst, *args, **kwargs):
            if Path(dst).name == ms.CURRENT_NAME:
                raise OSError("simulated disk failure")
            return real_replace(src, dst, *args, **kwargs)

        with mock.patch.object(ms.os, "replace", side_effect=failing_replace):
            with self.assertRaises(ms.ModelError):
                self.store.approve(PART, "v2", "Lan", "second", NOW2)

        log = [(e["action"], e["version"]) for e in self.store.approvals(PART)]
        self.assertEqual(log, [("approve", "v1"), ("approve", "v2"), ("failed", "v2")])
        self.assertEqual(self.store.current(PART).version, "v1")
        self.assertEqual(self.store.history(PART), ["v1"])
        statuses = {m["version"]: m["status"] for m in self.store.list_versions(PART)}
        self.assertEqual(statuses, {"v1": "approved", "v2": "candidate"})

        # The failed line does not block a later, successful approval.
        self.store.approve(PART, "v2", "Lan", "retry", NOW2)
        self.assertEqual(self.store.current(PART).version, "v2")
        self.assertEqual(self.store.history(PART), ["v1", "v2"])

    def test_history_tracks_approve_and_rollback(self):
        self.assertEqual(self.store.history(PART), [])
        self._three_approved()
        self.assertEqual(self.store.history(PART), ["v1", "v2", "v3"])
        self.store.rollback(PART, "Lan", "v3 bad", NOW3)
        self.assertEqual(self.store.history(PART), ["v1", "v2"])
        self.store.rollback(PART, "Lan", "v2 bad", NOW3)
        self.assertEqual(self.store.history(PART), ["v1"])

    def test_two_instances_see_the_same_stack(self):
        other = ms.ModelStore(self.root)
        self._three_approved()
        self.store.rollback(PART, "Lan", "v3 bad", NOW3)
        self.assertEqual(other.history(PART), ["v1", "v2"])
        other.rollback(PART, "Lan", "v2 bad", NOW3)
        self.assertEqual(self.store.history(PART), ["v1"])
        self.assertEqual(self.store.current(PART).version, "v1")


class LoaderTests(_StoreCase):
    def test_registered_loader_is_used(self):
        self._candidate("v1", type="fake")
        self.store.approve(PART, "v1", "Dai", "ok", NOW1)
        info = self.store.current(PART)

        def fake_loader(model_info, decode):
            return ("loaded", model_info.version, decode)

        with mock.patch.dict(ms.LOADERS, clear=False):
            ms.register_loader("fake", fake_loader)
            out = ms.load_runtime_model(info, "DECODE")
        self.assertEqual(out, ("loaded", "v1", "DECODE"))

    def test_unknown_type_raises_with_hint(self):
        info = ms.ModelInfo(part_number=PART, version="v1", type="nope-type", path=Path("x"),
                            manifest={}, approved_by="Dai", approved_utc=NOW1, reason="r")
        with self.assertRaises(ms.ModelError) as ctx:
            ms.load_runtime_model(info, None)
        self.assertIn("no loader for model type nope-type", str(ctx.exception))
        self.assertIn("optional package", str(ctx.exception))

    def test_register_loader_rejects_bad_type_name(self):
        with self.assertRaises(ValueError):
            ms.register_loader("Bad Name", lambda info, decode: None)

    def test_loader_failure_becomes_model_error(self):
        info = ms.ModelInfo(part_number=PART, version="v1", type="broken", path=Path("x"),
                            manifest={}, approved_by="Dai", approved_utc=NOW1, reason="r")

        def broken(model_info, decode):
            raise RuntimeError("corrupt file")

        with mock.patch.dict(ms.LOADERS, clear=False):
            ms.register_loader("broken", broken)
            with self.assertRaises(ms.ModelError):
                ms.load_runtime_model(info, None)


def _encode(arr):
    h, w = arr.shape
    return f"{h}x{w}:".encode("ascii") + arr.astype(np.uint8).tobytes()


def _decode(frame):
    head, _, body = frame.data.partition(b":")
    h, w = (int(v) for v in head.split(b"x"))
    return np.frombuffer(body, dtype=np.uint8).reshape(h, w).astype(np.float64)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class GoldenLoaderTests(_StoreCase):
    def test_golden_model_loads_through_registry(self):
        builder = golden.GoldenBuilder(_decode)
        for seed in range(3):
            arr = (120 + np.arange(12 * 16).reshape(12, 16) % (seed + 2)).astype(np.uint8)
            builder.add(Frame(camera_id="C1", side=Side.TOP, seq=seed,
                              timestamp_utc="2026-10-23T08:30:00Z", data=_encode(arr)))
        model = builder.build("golden-t")
        saved = self.base / "golden.npz"
        model.save(str(saved))

        self.store.add_candidate(PART, "v1", saved, _manifest(), NOW1)
        self.store.approve(PART, "v1", "Dai", "golden test", NOW1)
        info = self.store.current(PART)
        loaded = ms.load_runtime_model(info, _decode)
        self.assertIsInstance(loaded, golden.GoldenModel)
        self.assertEqual(loaded.version, "golden-t")


if __name__ == "__main__":
    unittest.main()
