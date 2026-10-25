import argparse
import contextlib
import io
import json
import pathlib
import tempfile
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
from station.modelstore import ModelStore
from station.storage import Store
from trainer import purge
from trainer.dataset import PURGED_FILE, iter_parts

FULL = [(cam, side) for side in (Side.TOP, Side.BOTTOM) for cam in CAMERA_IDS]


def make_record(session_id, part_id, part_number, started, final, *, machine=None, override=None):
    frames, results = [], []
    for seq, (cam, side) in enumerate(FULL):
        frames.append(Frame(camera_id=cam, side=side, seq=seq, timestamp_utc=started,
                            data=f"{session_id}-{cam}-{side.value}".encode() * 40, ext="png"))
        results.append(ImageResult(camera_id=cam, side=side, score=0.1, defects=()))
    return PartRecord(
        part_id=part_id, session_id=session_id, station_id="ST-1", part_number=part_number,
        lot="LOT-1", operator="op-1", supervisor="sup-1", material_handler="mh-1",
        started_utc=started, model_version="m-1.0", thresholds=Thresholds(t_low=0.3, t_img=0.6),
        frames=frames, results=results, machine_verdict=machine or final, final_verdict=final,
        override=override,
    )


class PurgeFixture(unittest.TestCase):
    """Parts of PN-1 (day 2026-10-01 unless noted):

    P-G1  S-G1  GOOD            trained                      -> purged
    P-G2  S-G2  GOOD            trained, but holdout         -> kept (holdout)
    P-G3  S-G3  GOOD            not in any training run      -> kept
    P-BAD S-BAD BAD             listed in training           -> kept (BAD)
    P-OVR S-OVR GOOD/BAD+ovr    listed in training           -> kept (overridden)
    P-OTH S-OTH GOOD (PN-2)     trained, other part number   -> not touched
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.store_root = self.root / "evidence"
        self.models_root = self.root / "models"
        store = Store(self.store_root)
        started = "2026-10-01T08:00:00Z"
        store.save_part(make_record("S-G1", "P-G1", "PN-1", started, Verdict.GOOD))
        store.save_part(make_record("S-G2", "P-G2", "PN-1", "2026-10-01T09:00:00Z", Verdict.GOOD))
        store.save_part(make_record("S-G3", "P-G3", "PN-1", "2026-10-01T10:00:00Z", Verdict.GOOD))
        store.save_part(make_record("S-BAD", "P-BAD", "PN-1", "2026-10-01T11:00:00Z", Verdict.BAD))
        store.save_part(make_record(
            "S-OVR", "P-OVR", "PN-1", "2026-10-01T12:00:00Z", Verdict.GOOD, machine=Verdict.BAD,
            override=Override(Verdict.BAD, Verdict.GOOD, "dust", "sup-1"),
        ))
        store.save_part(make_record("S-OTH", "P-OTH", "PN-2", started, Verdict.GOOD))

        model_file = self.root / "model.bin"
        model_file.write_bytes(b"model bytes v1")
        ModelStore(self.models_root).add_candidate(
            "PN-1", "cand-1", model_file, {"type": "golden"}, "2026-10-08T00:00:00Z")
        self.report = self.root / "weekly-2026-10-08.json"
        self.write_report(self.report_doc())

    def tearDown(self):
        self._tmp.cleanup()

    def report_doc(self, **overrides):
        doc = {
            "parts": [{
                "part_number": "PN-1",
                "candidate_version": "cand-1",
                "training_succeeded": True,
                "trained_session_ids": ["S-G1", "S-G2", "S-BAD", "S-OVR"],
                "holdout_session_ids": ["S-G2"],
            }],
        }
        doc.update(overrides)
        return doc

    def write_report(self, doc):
        self.report.write_text(json.dumps(doc), encoding="utf-8")

    def pdir(self, part_id):
        return self.store_root / "store" / "L1" / "2026" / "10" / "01" / part_id

    def pictures(self, part_id):
        return sorted(p for p in self.pdir(part_id).glob("*.png"))

    def plan(self):
        return purge.plan_purge(self.store_root, self.models_root,
                                part_number="PN-1", report_path=self.report)

    def cli(self, *extra):
        parser = argparse.ArgumentParser(prog="t")
        sub = parser.add_subparsers(dest="command", required=True)
        purge.add_arguments(sub)
        args = parser.parse_args([
            "purge", "--store", str(self.store_root), "--models", str(self.models_root),
            "--part", "PN-1", "--weekly-report", str(self.report), *extra,
        ])
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = purge.run(args)
        return code, out.getvalue(), err.getvalue()


class TestPlan(PurgeFixture):
    def test_only_eligible_parts_are_planned(self):
        plan = self.plan()
        self.assertIsNone(plan.refusal)
        self.assertEqual([p.part_id for p in plan.parts], ["P-G1"])
        self.assertEqual(len(plan.parts[0].files), 8)
        reasons = {k.part_dir.name: k.reason for k in plan.kept}
        self.assertEqual(reasons["P-G2"], "holdout evaluation set")
        self.assertIn("not in a successful training run", reasons["P-G3"])
        self.assertEqual(reasons["P-BAD"], "BAD part")
        self.assertEqual(reasons["P-OVR"], "overridden by a person")
        self.assertNotIn("P-OTH", reasons)

    def test_dry_run_deletes_nothing(self):
        before = [p.read_bytes() for p in self.pictures("P-G1")]
        code, out, _ = self.cli()
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN: nothing was deleted", out)
        self.assertEqual([p.read_bytes() for p in self.pictures("P-G1")], before)
        self.assertFalse((self.pdir("P-G1") / PURGED_FILE).exists())

    def test_confirm_deletes_only_eligible_pictures(self):
        code, _, err = self.cli("--confirm")
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(self.pictures("P-G1"), [])
        for part in ("P-G2", "P-G3", "P-BAD", "P-OVR"):
            self.assertEqual(len(self.pictures(part)), 8, part)
        self.assertEqual(len(self.pictures("P-OTH")), 8)
        for part in ("P-G1",):
            self.assertTrue((self.pdir(part) / "part.json").is_file())
            self.assertTrue((self.pdir(part) / "manifest.sha256").is_file())

    def test_tombstone_lists_deleted_files(self):
        self.cli("--confirm")
        tomb = json.loads((self.pdir("P-G1") / PURGED_FILE).read_text(encoding="utf-8"))
        self.assertEqual(tomb["tool"], purge.TOOL_VERSION)
        self.assertEqual(tomb["candidate_version"], "cand-1")
        self.assertEqual(tomb["report"], "weekly-2026-10-08.json")
        self.assertTrue(tomb["utc"].endswith("Z"))
        names = {f["name"] for f in tomb["files"]}
        self.assertEqual(len(names), 8)
        for entry in tomb["files"]:
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertGreater(entry["size"], 0)

    def test_tombstone_written_before_files_removed(self):
        plan = self.plan()
        original = purge.os.unlink
        seen = {}

        def spy(path):
            seen.setdefault("tomb_first", (self.pdir("P-G1") / PURGED_FILE).is_file())
            return original(path)

        purge.os.unlink = spy
        try:
            purge.execute_purge(plan, now_utc="2026-10-08T01:00:00Z")
        finally:
            purge.os.unlink = original
        self.assertTrue(seen["tomb_first"])

    def test_idempotent_rerun(self):
        self.cli("--confirm")
        tomb_before = (self.pdir("P-G1") / PURGED_FILE).read_text(encoding="utf-8")
        plan = self.plan()
        self.assertEqual(plan.parts, [])
        self.assertEqual(plan.already_purged, 1)
        result = purge.execute_purge(plan)
        self.assertEqual((result.files_deleted, result.errors), (0, []))
        self.assertEqual((self.pdir("P-G1") / PURGED_FILE).read_text(encoding="utf-8"), tomb_before)

    def test_purged_part_skipped_by_dataset_reader(self):
        self.cli("--confirm")
        problems = []
        got = [s.part_id for s in iter_parts(self.store_root, part_number="PN-1",
                                              verify=True, problems=problems)]
        self.assertNotIn("P-G1", got)
        self.assertTrue(any(line.startswith(str(self.pdir("P-G1"))) and "purged" in line
                            for line in problems), problems)
        self.assertIn("P-G3", got)


class TestRefusals(PurgeFixture):
    def test_missing_candidate_refuses(self):
        doc = self.report_doc()
        doc["parts"][0]["candidate_version"] = "cand-404"
        self.write_report(doc)
        plan = self.plan()
        self.assertIsNotNone(plan.refusal)
        self.assertEqual(plan.parts, [])
        code, _, err = self.cli("--confirm")
        self.assertEqual(code, 1)
        self.assertIn("refused", err)
        self.assertEqual(len(self.pictures("P-G1")), 8)

    def test_unverifiable_candidate_refuses(self):
        model = ModelStore(self.models_root)._version_dir("PN-1", "cand-1") / "model.bin"
        model.write_bytes(b"tampered")
        plan = self.plan()
        self.assertIsNotNone(plan.refusal)
        self.assertIn("does not verify", plan.refusal)
        self.assertEqual(self.cli("--confirm")[0], 1)
        self.assertEqual(len(self.pictures("P-G1")), 8)

    def test_malformed_entry_refuses(self):
        doc = self.report_doc()
        del doc["parts"][0]["trained_session_ids"]
        self.write_report(doc)
        self.assertIsNotNone(self.plan().refusal)

    def test_missing_report_refuses(self):
        self.report.unlink()
        self.assertIsNotNone(self.plan().refusal)

    def test_no_entry_for_part_refuses(self):
        self.write_report({"parts": []})
        self.assertIn("no training entry", self.plan().refusal)

    def test_failed_training_deletes_nothing(self):
        doc = self.report_doc()
        doc["parts"][0]["training_succeeded"] = False
        self.write_report(doc)
        plan = self.plan()
        self.assertIsNone(plan.refusal)
        self.assertEqual(plan.parts, [])


class TestSafety(PurgeFixture):
    def test_modified_picture_keeps_part_with_warning(self):
        self.pictures("P-G1")[0].write_bytes(b"changed")
        plan = self.plan()
        self.assertEqual(plan.parts, [])
        self.assertTrue(any("changed since it was saved" in w for w in plan.warnings), plan.warnings)
        self.cli("--confirm")
        self.assertEqual(len(self.pictures("P-G1")), 8)
        self.assertFalse((self.pdir("P-G1") / PURGED_FILE).exists())

    def test_path_traversal_refused(self):
        part_json = self.pdir("P-G1") / "part.json"
        doc = json.loads(part_json.read_text(encoding="utf-8"))
        victim = self.store_root / "store" / "victim.png"
        victim.write_bytes(b"not yours")
        doc["images"][0]["file"] = "../../victim.png"
        part_json.write_text(json.dumps(doc), encoding="utf-8")
        plan = self.plan()
        self.assertEqual(plan.parts, [])
        self.cli("--confirm")
        self.assertEqual(victim.read_bytes(), b"not yours")

    def test_absolute_path_refused(self):
        part_json = self.pdir("P-G1") / "part.json"
        doc = json.loads(part_json.read_text(encoding="utf-8"))
        doc["images"][0]["file"] = str(self.root / "model.bin")
        part_json.write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(self.plan().parts, [])
        self.assertTrue((self.root / "model.bin").is_file())

    def test_symlinked_picture_kept(self):
        target = self.root / "outside.png"
        target.write_bytes(b"outside")
        picture = self.pictures("P-G1")[0]
        picture.unlink()
        picture.symlink_to(target)
        plan = self.plan()
        self.assertEqual(plan.parts, [])
        self.assertTrue(any("symlink" in w for w in plan.warnings), plan.warnings)
        self.cli("--confirm")
        self.assertTrue(target.is_file())
        self.assertTrue(picture.is_symlink())

    def test_part_json_and_manifest_never_deleted(self):
        self.cli("--confirm")
        self.assertTrue((self.pdir("P-G1") / "part.json").is_file())
        self.assertTrue((self.pdir("P-G1") / "manifest.sha256").is_file())

    def test_unlisted_file_in_part_folder_kept(self):
        extra = self.pdir("P-G1") / "notes.txt"
        extra.write_text("keep me", encoding="utf-8")
        self.cli("--confirm")
        self.assertEqual(extra.read_text(encoding="utf-8"), "keep me")


class TestArguments(PurgeFixture):
    def test_add_arguments_registers_purge(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        purge.add_arguments(sub)
        args = parser.parse_args(["purge", "--store", "s", "--models", "m", "--part", "p",
                                  "--weekly-report", "r.json"])
        self.assertFalse(args.confirm)
        self.assertEqual(args.part, "p")


if __name__ == "__main__":
    unittest.main()
