"""Tests for trainer/scans.py (Taking-data scans read for the weekly trainer) and the --scans-too path.

Fake scans are written with station.scanstore.ScanStore in temporary folders. The weekly runs use
the same fake extractor and decoder as tests/test_trainer_weekly.py, so no torch is needed.
"""
import contextlib
import io
import json
import pathlib
import tempfile
import unittest

from station.modelstore import ModelStore, safe_part
from station.scanstore import ScanStore
from trainer.__main__ import main
from trainer.dataset import ImageRef
from trainer.scans import (
    ScanImageRef,
    ScanSample,
    bad_defect_counts,
    has_scans,
    image_counts,
    inventory_scans,
    read_scans,
    scans_root_for,
    split_by_toy,
)
from trainer.weekly import run_weekly
from tests.test_trainer_weekly import FakeExtractor, fake_decode, save_part

try:
    import numpy as np
    from station.anomaly import PatchCoreModel
except ImportError:  # numpy missing: the weekly runs are skipped
    np = None

CAMS = ["C1", "C2", "C3", "C4"]
STARTED = "2026-10-27T08:30:12Z"
FINISHED = "2026-10-27T08:41:55Z"
LABELLED = "2026-10-27T08:42:00Z"
APPROVED = "2026-10-27T09:00:00Z"
NOW = "2026-10-27T12:00:00Z"


def pic(tag):
    return f"png-{tag}".encode() * 40


def label_for(verdict, status, defects=()):
    if verdict == "GOOD":
        label = {"verdict": "GOOD", "defects": [], "status": status,
                 "labelled_by": "sup-1", "labelled_at": LABELLED}
    else:
        label = {"verdict": "BAD", "defects": list(defects), "status": status,
                 "labelled_by": "sup-1", "labelled_at": LABELLED}
    if status == "official":
        label.update(approved_by="admin-1", approved_at=APPROVED)
    if status == "rejected":
        label.update(approved_by="admin-1", approved_at=APPROVED, reject_reason="wrong toy")
    return label


def make_scan(store, scan_id, toy, *, verdict="GOOD", status="official", defects=("doh_small",),
              steps=None, part="PN-S", started=STARTED, visible=None):
    """Write one finished scan. BAD pictures alternate defect_visible unless `visible` is given."""
    steps = steps or {"TOP": 2, "BOTTOM": 2}
    writer = store.begin_scan(scan_id, part, toy, label_for(verdict, status, defects),
                              "op-1", list(CAMS), started)
    for side, count in steps.items():
        for step in range(1, count + 1):
            for cam in CAMS:
                dv = None
                if verdict == "BAD":
                    dv = visible if visible is not None else (step + CAMS.index(cam)) % 2 == 0
                writer.add_picture(cam, side, step, pic(f"{scan_id}-{cam}-{side}-{step}"),
                                   defect_visible=dv)
    return writer.finish(FINISHED)


def fake_scan(scan_id, toy, *, part="PN-S", steps=None, verdict="GOOD"):
    """A ScanSample built in memory, for the split tests (no files)."""
    steps = steps or {"TOP": 1}
    images = tuple(
        ScanImageRef(path=pathlib.Path(f"/nowhere/{scan_id}-{cam}-{side}-{step}.png"), camera_id=cam,
                     side=side, score=None, sha256="0" * 64, step=step,
                     defect_visible=None if verdict == "GOOD" else True)
        for side, count in steps.items() for step in range(1, count + 1) for cam in CAMS
    )
    return ScanSample(scan_id=scan_id, toy_id=toy, part_number=part, started_utc=STARTED,
                      verdict=verdict, status="official", defects=(), steps=dict(steps),
                      scan_dir=pathlib.Path(f"/nowhere/{scan_id}"), images=images)


class ScanFolderCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.store = ScanStore(self.root, site="SITE", line="L1")

    def tearDown(self):
        self._tmp.cleanup()


class TestReadScans(ScanFolderCase):
    def test_official_good_scan_is_a_training_scan(self):
        make_scan(self.store, "SC-1", "toy-1")
        scans = read_scans(self.root)
        self.assertEqual(scans.folders, 1)
        self.assertEqual([s.scan_id for s in scans.good], ["SC-1"])
        self.assertEqual(scans.bad, [])
        self.assertEqual(scans.excluded, [])
        self.assertEqual(scans.problems, [])
        sample = scans.good[0]
        self.assertEqual(sample.toy_id, "toy-1")
        self.assertEqual(len(sample.images), 2 * 4 * 2)  # 2 steps, 4 cameras, 2 sides
        self.assertTrue(all(isinstance(img, ImageRef) for img in sample.images))
        self.assertTrue(all(img.score is None for img in sample.images))
        self.assertEqual(image_counts(scans.good), {"C1/BOTTOM": 2, "C1/TOP": 2, "C2/BOTTOM": 2,
                                                    "C2/TOP": 2, "C3/BOTTOM": 2, "C3/TOP": 2,
                                                    "C4/BOTTOM": 2, "C4/TOP": 2})

    def test_proposed_and_rejected_good_scans_are_excluded_with_a_reason(self):
        make_scan(self.store, "SC-P", "toy-1", status="proposed")
        make_scan(self.store, "SC-R", "toy-1", status="rejected")
        scans = read_scans(self.root)
        self.assertEqual(scans.good, [])
        self.assertEqual(scans.bad, [])
        self.assertEqual(scans.problems, [])
        reasons = scans.excluded_reasons()
        self.assertEqual(sum(reasons.values()), 2)
        self.assertEqual(reasons["GOOD label, status proposed: not used (only official labels train)"], 1)
        self.assertEqual(reasons["GOOD label, status rejected: not used (only official labels train)"], 1)

    def test_bad_scans_are_never_training_and_are_counted_per_defect_and_visibility(self):
        make_scan(self.store, "SC-B1", "toy-3", verdict="BAD", defects=("doh_small",))
        make_scan(self.store, "SC-B2", "toy-3", verdict="BAD", defects=("doh_small", "doh_edge"),
                  visible=False)
        make_scan(self.store, "SC-B3", "toy-3", verdict="BAD", defects=("doh_edge",), status="proposed")
        scans = read_scans(self.root)
        self.assertEqual(scans.good, [])
        self.assertEqual([s.scan_id for s in scans.bad], ["SC-B1", "SC-B2"])
        self.assertIn("BAD label, status proposed: not used (only official labels are evaluated)",
                      scans.excluded_reasons())
        counts = bad_defect_counts(scans.bad)
        pictures = 2 * 4 * 2  # per scan
        self.assertEqual(counts["doh_small"]["scans"], 2)
        self.assertEqual(counts["doh_edge"]["scans"], 1)  # the proposed scan is not counted
        self.assertEqual(counts["doh_small"]["visible_true"] + counts["doh_small"]["visible_false"],
                         2 * pictures)
        self.assertEqual(counts["doh_edge"]["visible_false"], pictures)  # SC-B2 was all not visible
        self.assertEqual(counts["doh_edge"]["visible_true"], 0)
        self.assertTrue(all(img.defect_visible in (True, False) for s in scans.bad for img in s.images))

    def test_altered_picture_excludes_the_scan_and_gives_a_problem_line(self):
        folder = make_scan(self.store, "SC-1", "toy-1")
        make_scan(self.store, "SC-2", "toy-2")
        picture = sorted(folder.glob("*.png"))[0]
        picture.write_bytes(b"altered" * 40)
        scans = read_scans(self.root)
        self.assertEqual([s.scan_id for s in scans.good], ["SC-2"])
        self.assertTrue(any(p.startswith(f"{folder}: ") and f"hash mismatch: {picture.name}" in p
                            for p in scans.problems), scans.problems)
        self.assertIn(str(folder), [f for f, reason in scans.excluded])

    def test_unfinished_staging_folder_is_not_read(self):
        self.store.begin_scan("SC-X", "PN-S", "toy-1", label_for("GOOD", "official"),
                              "op-1", list(CAMS), STARTED)
        scans = read_scans(self.root)
        self.assertEqual(scans.folders, 0)
        self.assertEqual(scans.good, [])

    def test_unequal_top_and_bottom_counts_work(self):
        make_scan(self.store, "SC-1", "toy-1", steps={"TOP": 3, "BOTTOM": 1})
        make_scan(self.store, "SC-2", "toy-2", steps={"TOP": 1, "BOTTOM": 4})
        scans = read_scans(self.root)
        self.assertEqual(scans.problems, [])
        counts = image_counts(scans.good)
        self.assertEqual(counts["C1/TOP"], 4)
        self.assertEqual(counts["C1/BOTTOM"], 5)
        self.assertEqual(sorted(s.steps.get("TOP") for s in scans.good), [1, 3])

    def test_scans_root_accepts_the_store_folder_too(self):
        make_scan(self.store, "SC-1", "toy-1")
        self.assertEqual(scans_root_for(self.root), self.root)
        self.assertEqual(scans_root_for(self.root / "store"), self.root)  # store/ without a scans folder
        self.assertTrue(has_scans(self.root))
        self.assertFalse(has_scans(self.root / "nothing-here"))


class TestSplitByToy(unittest.TestCase):
    def sizes_to_scans(self, sizes):
        return [fake_scan(f"{toy}-{i}", toy) for toy, n in sizes.items() for i in range(n)]

    def test_split_never_puts_one_toy_on_both_sides(self):
        scans = self.sizes_to_scans({"t1": 2, "t2": 3, "t3": 1, "t4": 4, "t5": 2, "t6": 2})
        for seed in range(10):
            for fraction in (0.0, 0.2, 0.34, 0.5, 0.9):
                train, hold, held = split_by_toy(scans, fraction, seed)
                train_toys = {s.toy_id for s in train}
                hold_toys = {s.toy_id for s in hold}
                self.assertFalse(train_toys & hold_toys, (seed, fraction))
                self.assertEqual(len(train) + len(hold), len(scans))
                self.assertEqual(set(held), hold_toys)
                self.assertTrue(train, (seed, fraction))  # at least one toy always trains
                if fraction == 0:
                    self.assertEqual(hold, [])

    def test_split_is_repeatable_for_the_same_seed(self):
        scans = self.sizes_to_scans({"t1": 2, "t2": 3, "t3": 1, "t4": 4})
        self.assertEqual(split_by_toy(scans, 0.5, 7)[2], split_by_toy(scans, 0.5, 7)[2])

    def test_explicit_holdout_toys_are_honoured(self):
        scans = self.sizes_to_scans({"t1": 2, "t2": 3, "t3": 1})
        for seed in (0, 1, 2):
            train, hold, held = split_by_toy(scans, 0.0, seed, holdout_toys=["t2"])
            self.assertEqual(held, ["t2"])
            self.assertEqual({s.toy_id for s in hold}, {"t2"})
            self.assertEqual(len(hold), 3)
            self.assertEqual({s.toy_id for s in train}, {"t1", "t3"})

    def test_holdout_fraction_is_checked(self):
        scans = self.sizes_to_scans({"t1": 1, "t2": 1})
        with self.assertRaises(ValueError):
            split_by_toy(scans, 0.95, 0)
        with self.assertRaises(ValueError):
            split_by_toy(scans, True, 0)


class TestInventory(ScanFolderCase):
    def test_counts_per_toy_label_status_and_camera_side(self):
        make_scan(self.store, "SC-1", "toy-1")
        make_scan(self.store, "SC-2", "toy-1")
        make_scan(self.store, "SC-3", "toy-2", status="proposed")
        make_scan(self.store, "SC-4", "toy-3", verdict="BAD", defects=("doh_small",))
        make_scan(self.store, "SC-5", "toy-3", verdict="BAD", defects=("doh_large",), status="rejected")
        inv = inventory_scans(self.root)
        self.assertEqual(inv["folders"], 5)
        self.assertEqual(inv["intact"], 5)
        self.assertEqual(inv["problems"], 0)
        self.assertEqual(inv["by_toy"], {"toy-1": {"GOOD": 2, "BAD": 0},
                                         "toy-2": {"GOOD": 1, "BAD": 0},
                                         "toy-3": {"GOOD": 0, "BAD": 2}})
        self.assertEqual(inv["by_label"], {"GOOD": 3, "BAD": 2})
        self.assertEqual(inv["by_status"], {"official": 3, "proposed": 1, "rejected": 1})
        self.assertEqual(inv["excluded"]["total"], 2)
        self.assertEqual(inv["training_images_by_camera_side"]["C1/TOP"], 4)
        # One official BAD scan: 2 steps x 4 cameras x 2 sides = 16 pictures, half marked visible.
        self.assertEqual(inv["bad_defects_official"], {"doh_small": {"scans": 1, "visible_true": 8,
                                                                    "visible_false": 8}})
        self.assertEqual(inv["scans_by_side"], {"TOP": 5, "BOTTOM": 5})

    def test_inventory_reports_a_bad_scan_as_a_problem(self):
        folder = make_scan(self.store, "SC-1", "toy-1")
        (folder / sorted(p.name for p in folder.glob("*.png"))[0]).unlink()
        inv = inventory_scans(self.root)
        self.assertEqual(inv["intact"], 0)
        self.assertGreater(inv["problems"], 0)
        self.assertEqual(inv["excluded"]["by_reason"], {"failed checks (see problems)": 1})


class TestInventoryCommand(ScanFolderCase):
    def run_inventory(self, store):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["inventory", "--store", str(store)])
        return code, out.getvalue()

    def test_no_scans_folder_leaves_the_output_as_before(self):
        plain = self.root / "plain"  # a store with no scans/ folder (setUp creates one under self.root)
        (plain / "store").mkdir(parents=True)
        code, text = self.run_inventory(plain)
        self.assertEqual(code, 0)
        self.assertIn("no usable parts found", text)
        self.assertNotIn("scans (", text)
        self.assertTrue(text.rstrip().endswith("problems: 0"))

    def test_scans_folder_adds_the_scan_block(self):
        make_scan(self.store, "SC-1", "toy-1")
        code, text = self.run_inventory(self.root)
        self.assertEqual(code, 0)
        self.assertIn("problems: 0", text)  # existing part line is still there
        self.assertIn("scans (", text)
        self.assertIn("toy-1", text)
        self.assertIn("scan problems: 0", text)


@unittest.skipUnless(np is not None, "numpy is not installed")
class TestWeeklyWithScans(ScanFolderCase):
    def setUp(self):
        super().setUp()
        self.models = self.root / "models"
        self.reports = self.root / "reports"

    def args(self, *extra):
        from trainer.__main__ import build_parser
        base = ["weekly", "--store", str(self.root), "--models", str(self.models),
                "--reports", str(self.reports), "--min-good", "2", "--min-holdout", "0",
                "--holdout-fraction", "0", "--now-utc", NOW]
        return build_parser().parse_args(base + list(extra))

    def run_quiet(self, args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = run_weekly(args, extractor_factory=FakeExtractor, decoder_factory=lambda: fake_decode)
        return code, out.getvalue()

    def report(self):
        return json.loads((self.reports / "weekly-2026-10-27.json").read_text(encoding="utf-8"))

    def test_scans_only_candidate_with_unequal_top_bottom_counts(self):
        make_scan(self.store, "SC-1", "toy-1", steps={"TOP": 3, "BOTTOM": 1})
        make_scan(self.store, "SC-2", "toy-2", steps={"TOP": 3, "BOTTOM": 1})
        make_scan(self.store, "SC-3", "toy-3", steps={"TOP": 3, "BOTTOM": 1})
        code, out = self.run_quiet(self.args("--scans-too"))
        self.assertEqual(code, 0, out)
        ms = ModelStore(self.models)
        versions = ms.list_versions("PN-S")
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]["status"], "candidate")
        model = PatchCoreModel.load(ms.root / safe_part("PN-S") / versions[0]["version"] / "patchcore.npz",
                                    fake_decode, FakeExtractor())
        banks = model.describe()["banks"]
        self.assertEqual(banks["C1/TOP"]["train_images"], 9)
        self.assertEqual(banks["C1/BOTTOM"]["train_images"], 3)
        report = self.report()
        self.assertEqual(report["scans"]["training_scans"], 3)
        self.assertEqual(report["scans"]["per_part"]["PN-S"]["train_scans"], 3)
        self.assertEqual(report["parts"][0]["status"], "candidate")
        self.assertIn("Scans (Taking data", (self.reports / "weekly-2026-10-27.txt").read_text(encoding="utf-8"))

    def test_parts_and_scans_with_explicit_holdout_toy(self):
        for day in (1, 2, 3):
            save_part(Store_for(self.root), f"S-{day}", "PN-1", f"2026-10-0{day}T08:00:00Z")
        make_scan(self.store, "SC-T1", "T1", part="PN-1")
        make_scan(self.store, "SC-T2", "T2", part="PN-1")
        make_scan(self.store, "SC-T3", "T3", part="PN-1")
        make_scan(self.store, "SC-T3B", "T3", verdict="BAD", defects=("doh_small",), part="PN-1")
        code, out = self.run_quiet(self.args("--scans-too", "--holdout-toys", "T3", "--min-holdout", "1"))
        self.assertEqual(code, 0, out)
        report = self.report()
        part = next(p for p in report["parts"] if p["part_number"] == "PN-1")
        self.assertEqual(part["status"], "candidate")
        self.assertEqual(part["scans"]["holdout_toys"], ["T3"])
        self.assertEqual(part["scans"]["holdout_rule"], "explicit --holdout-toys")
        self.assertEqual(part["scans"]["held_out_scans"], 1)
        self.assertEqual(part["scans"]["train_scans"], 2)
        self.assertEqual(len(part["holdout_scan_ids"]), 1)
        self.assertEqual(sorted(part["trained_scan_ids"]), ["SC-T1", "SC-T2"])
        self.assertEqual(len(part["trained_session_ids"]), 3)  # purge keys: parts only
        held = part["evaluation"]["scans"]["held_out_good"]
        self.assertEqual(held["scans"], 1)
        self.assertEqual(list(held["per_toy"]), ["T3"])
        bad = part["evaluation"]["scans"]["bad"]
        self.assertEqual(bad["scans"], 1)
        self.assertEqual(list(bad["per_defect"]), ["doh_small"])
        self.assertEqual(bad["per_defect"]["doh_small"]["scans"], 1)
        section = report["scans"]
        self.assertEqual(section["per_toy"]["T3"]["held_out_for_part_numbers"], ["PN-1"])
        self.assertEqual(section["per_toy"]["T1"]["held_out_for_part_numbers"], [])
        self.assertEqual(section["training_scans"], 3)
        self.assertEqual(section["evaluation_scans"], 1)
        self.assertEqual(section["excluded"]["total"], 0)
        text = (self.reports / "weekly-2026-10-27.txt").read_text(encoding="utf-8")
        self.assertIn("held-out toys: T3", text)
        self.assertIn("held-out good scans (1 scans)", text)
        self.assertIn("Scans (Taking data", text)

        ms = ModelStore(self.models)
        version = ms.list_versions("PN-1")[0]["version"]
        model = PatchCoreModel.load(ms.root / safe_part("PN-1") / version / "patchcore.npz",
                                    fake_decode, FakeExtractor())
        # Bank C1/TOP: 3 part pictures (one per part) plus 2 train scans with 2 TOP steps each.
        # The held-out toy T3 adds nothing to the training bank.
        self.assertEqual(model.describe()["banks"]["C1/TOP"]["train_images"], 3 + 2 * 2)

    def test_scans_excluded_are_counted_in_the_report(self):
        make_scan(self.store, "SC-1", "toy-1")
        make_scan(self.store, "SC-2", "toy-2")
        make_scan(self.store, "SC-3", "toy-3", status="proposed")
        code, _ = self.run_quiet(self.args("--scans-too"))
        self.assertEqual(code, 0)
        excluded = self.report()["scans"]["excluded"]
        self.assertEqual(excluded["total"], 1)
        self.assertEqual(sum(excluded["by_reason"].values()), 1)

    def test_dry_run_with_scans_writes_nothing(self):
        make_scan(self.store, "SC-1", "toy-1")
        make_scan(self.store, "SC-2", "toy-2")

        def no_extractor():
            raise AssertionError("dry run must not build the extractor")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = run_weekly(self.args("--scans-too", "--dry-run"), extractor_factory=no_extractor,
                              decoder_factory=None)
        self.assertEqual(code, 0)
        self.assertIn("Scans (Taking data", out.getvalue())
        self.assertFalse(self.reports.exists())
        self.assertFalse(self.models.exists())

    def test_holdout_toys_needs_scans_too(self):
        make_scan(self.store, "SC-1", "toy-1")
        with contextlib.redirect_stderr(io.StringIO()):
            code = run_weekly(self.args("--holdout-toys", "toy-1"), extractor_factory=FakeExtractor,
                              decoder_factory=lambda: fake_decode)
        self.assertEqual(code, 2)
        self.assertFalse(self.reports.exists())

    def test_unknown_holdout_toy_is_a_usage_error(self):
        make_scan(self.store, "SC-1", "toy-1")
        make_scan(self.store, "SC-2", "toy-2")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = run_weekly(self.args("--scans-too", "--holdout-toys", "toy-9"),
                              extractor_factory=FakeExtractor, decoder_factory=lambda: fake_decode)
        self.assertEqual(code, 2)
        self.assertIn("toy-9", err.getvalue())
        self.assertFalse(self.reports.exists())

    def test_altered_scan_is_excluded_and_named_in_store_problems(self):
        make_scan(self.store, "SC-1", "toy-1")
        make_scan(self.store, "SC-2", "toy-2")
        folder = make_scan(self.store, "SC-3", "toy-3")
        sorted(folder.glob("*.png"))[0].write_bytes(b"altered" * 40)
        code, _ = self.run_quiet(self.args("--scans-too"))
        self.assertEqual(code, 0)
        report = self.report()
        self.assertEqual(report["scans"]["training_scans"], 2)
        self.assertGreater(report["store_problems"]["count"], 0)
        self.assertTrue(any(line.startswith(f"{folder}: ") for line in report["store_problems"]["first"]))


def Store_for(root):
    from station.storage import Store
    return Store(root)


if __name__ == "__main__":
    unittest.main()
