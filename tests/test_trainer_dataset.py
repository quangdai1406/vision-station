import contextlib
import dataclasses
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
    OperatorLabel,
    Override,
    PartRecord,
    Side,
    Thresholds,
    Verdict,
)
from station.storage import COLLECTED, Store
from trainer.__main__ import main
from trainer.dataset import (
    Dataset,
    ImageRef,
    PartSample,
    classify,
    iter_parts,
    list_part_numbers,
    split_holdout,
)

FULL = [(cam, side) for side in (Side.TOP, Side.BOTTOM) for cam in CAMERA_IDS]
ONE_CAMERA = [("C1", Side.TOP), ("C1", Side.BOTTOM)]
SIDE_NAMES = ("TOP", "BOTTOM")


def make_record(session_id, part_id, part_number, started, final, *,
                machine=None, override=None, pairs=None, bad=None):
    """A PartRecord shaped like tests/test_storage.make_record; `bad` marks one hot image."""
    pairs = list(pairs or FULL)
    frames, results = [], []
    for seq, (cam, side) in enumerate(pairs):
        frames.append(Frame(
            camera_id=cam,
            side=side,
            seq=seq,
            timestamp_utc=started,
            data=f"{session_id}-{cam}-{side.value}".encode() * 40,
            ext="png",
        ))
        hot = (cam, side) == bad
        results.append(ImageResult(
            camera_id=cam,
            side=side,
            score=0.7 if hot else 0.1,
            defects=(Defect("scratch", 0.7, (1, 2, 3, 4)),) if hot else (),
        ))
    return PartRecord(
        part_id=part_id,
        session_id=session_id,
        station_id="ST-1",
        part_number=part_number,
        lot="LOT-1",
        operator="op-1",
        supervisor="sup-1",
        material_handler="mh-1",
        started_utc=started,
        model_version="m-1.0",
        thresholds=Thresholds(t_low=0.3, t_img=0.6),
        frames=frames,
        results=results,
        machine_verdict=machine or final,
        final_verdict=final,
        override=override,
    )


class EvidenceFixture(unittest.TestCase):
    """Six parts, two part numbers, four dates.

    PN-1  P-A1  10-01 08:00  GOOD machine GOOD              -> good (strict and include)
    PN-1  P-A2  10-01 09:00  BAD  machine BAD               -> bad
    PN-1  P-A3  10-02 08:00  GOOD machine BAD, overridden   -> excluded (strict), good (include)
    PN-1  P-A4  10-03 08:00  GOOD machine GOOD, ONE camera  -> partial station
    PN-1  P-A5  10-05 08:00  GOOD machine GOOD              -> good
    PN-2  P-B1  10-02 08:00  GOOD machine GOOD              -> good (other part number)
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        full = Store(self.root)
        partial = Store(self.root, allow_partial=True)
        full.save_part(make_record("S-A1", "P-A1", "PN-1", "2026-10-01T08:00:00Z", Verdict.GOOD))
        full.save_part(make_record("S-A2", "P-A2", "PN-1", "2026-10-01T09:00:00Z", Verdict.BAD,
                                   bad=("C2", Side.BOTTOM)))
        full.save_part(make_record(
            "S-A3", "P-A3", "PN-1", "2026-10-02T08:00:00Z", Verdict.GOOD,
            machine=Verdict.BAD,
            override=Override(Verdict.BAD, Verdict.GOOD, "dust, not a scratch", "sup-1"),
        ))
        partial.save_part(make_record("S-A4", "P-A4", "PN-1", "2026-10-03T08:00:00Z", Verdict.GOOD,
                                      pairs=ONE_CAMERA))
        full.save_part(make_record("S-A5", "P-A5", "PN-1", "2026-10-05T08:00:00Z", Verdict.GOOD))
        full.save_part(make_record("S-B1", "P-B1", "PN-2", "2026-10-02T08:00:00Z", Verdict.GOOD))

    def tearDown(self):
        self._tmp.cleanup()

    def pdir(self, part_id, day):
        return self.root / "store" / "L1" / "2026" / "10" / f"{day:02d}" / part_id

    def image(self, part_id, day, cam, side):
        return next(self.pdir(part_id, day).glob(f"*_{cam}_{side}.png"))

    @staticmethod
    def ids(samples):
        return [s.part_id for s in samples]


class TestIterParts(EvidenceFixture):
    def test_default_skips_partial_and_orders_by_time_then_part_id(self):
        problems = []
        got = list(iter_parts(self.root, problems=problems))
        # P-A3 and P-B1 share 10-02 08:00, so part_id breaks the tie.
        self.assertEqual(self.ids(got), ["P-A1", "P-A2", "P-A3", "P-B1", "P-A5"])
        self.assertEqual(problems, [])
        for sample in got:
            self.assertEqual(len(sample.images), 8)
            self.assertFalse(sample.partial_station)

    def test_filter_by_part_number(self):
        self.assertEqual(self.ids(iter_parts(self.root, part_number="PN-2")), ["P-B1"])
        self.assertEqual(self.ids(iter_parts(self.root, part_number="PN-404")), [])

    def test_since_inclusive_until_exclusive(self):
        got = list(iter_parts(self.root, part_number="PN-1", full_station_only=False,
                              since="2026-10-02", until="2026-10-03"))
        self.assertEqual(self.ids(got), ["P-A3"])
        got = list(iter_parts(self.root, until="2026-10-02"))
        self.assertEqual(self.ids(got), ["P-A1", "P-A2"])
        got = list(iter_parts(self.root, since="2026-10-02", until="2026-10-02"))
        self.assertEqual(got, [])

    def test_bad_date_raises_value_error(self):
        with self.assertRaises(ValueError):
            iter_parts(self.root, since="2026/10/01")

    def test_partial_station_excluded_by_default_and_included_on_request(self):
        default = list(iter_parts(self.root, part_number="PN-1"))
        self.assertNotIn("P-A4", self.ids(default))
        everything = list(iter_parts(self.root, part_number="PN-1", full_station_only=False))
        self.assertIn("P-A4", self.ids(everything))
        partial = next(s for s in everything if s.part_id == "P-A4")
        self.assertTrue(partial.partial_station)
        self.assertEqual(partial.camera_count, 1)
        self.assertEqual(len(partial.images), 2)

    def test_tampered_image_skipped_with_problem_when_verifying(self):
        self.image("P-A1", 1, "C1", "TOP").write_bytes(b"tampered")
        problems = []
        got = list(iter_parts(self.root, part_number="PN-1", problems=problems))
        self.assertNotIn("P-A1", self.ids(got))
        self.assertTrue(
            any(line.startswith(str(self.pdir("P-A1", 1))) and
                "hash mismatch: SITE_L1_20261001_080000_P-A1_C1_TOP.png" in line
                for line in problems),
            problems,
        )

    def test_tampered_image_kept_without_verify(self):
        self.image("P-A1", 1, "C1", "TOP").write_bytes(b"tampered")
        problems = []
        got = list(iter_parts(self.root, part_number="PN-1", verify=False, problems=problems))
        self.assertIn("P-A1", self.ids(got))
        self.assertEqual(problems, [])

    def test_missing_image_skipped_even_without_verify(self):
        self.image("P-A1", 1, "C2", "BOTTOM").unlink()
        problems = []
        got = list(iter_parts(self.root, part_number="PN-1", verify=False, problems=problems))
        self.assertNotIn("P-A1", self.ids(got))
        self.assertTrue(any("missing image file" in line for line in problems), problems)

    def test_corrupt_part_json_skipped_with_problem(self):
        (self.pdir("P-A2", 1) / "part.json").write_text("{not json", encoding="utf-8")
        problems = []
        got = list(iter_parts(self.root, part_number="PN-1", problems=problems))
        self.assertNotIn("P-A2", self.ids(got))
        self.assertEqual(len(problems), 1, problems)
        self.assertTrue(problems[0].startswith(str(self.pdir("P-A2", 1))))
        self.assertIn("unreadable part.json", problems[0])

    def test_missing_key_skipped_with_problem(self):
        import json

        path = self.pdir("P-A5", 5) / "part.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        del doc["final_verdict"]
        path.write_text(json.dumps(doc), encoding="utf-8")
        problems = []
        got = list(iter_parts(self.root, part_number="PN-1", verify=False, problems=problems))
        self.assertNotIn("P-A5", self.ids(got))
        self.assertTrue(any("missing keys: final_verdict" in line for line in problems), problems)

    def test_both_store_root_forms_work(self):
        from_root = self.ids(iter_parts(self.root))
        from_store = self.ids(iter_parts(self.root / "store"))
        self.assertEqual(from_root, from_store)
        self.assertEqual(len(from_root), 5)

    def test_missing_store_raises(self):
        with self.assertRaises(FileNotFoundError):
            iter_parts(self.root / "nope")


class TestClassify(unittest.TestCase):
    def sample(self, final, machine, override=None):
        return PartSample(
            part_number="PN-1", part_id="P", session_id="S", started_utc="2026-10-01T00:00:00Z",
            machine_verdict=machine, final_verdict=final, override=override, defect_label=None,
            model_version="m", partial_station=False, camera_count=4, images=(), part_dir=pathlib.Path("/x"),
        )

    def test_strict_policy(self):
        note = {"original_verdict": "BAD", "new_verdict": "GOOD", "reason": "r", "by": "s"}
        self.assertEqual(classify(self.sample("GOOD", "GOOD")), "good")
        self.assertEqual(classify(self.sample("GOOD", "BAD", note)), "excluded")
        self.assertEqual(classify(self.sample("GOOD", "GOOD", note)), "excluded")
        self.assertEqual(classify(self.sample("BAD", "BAD")), "bad")
        self.assertEqual(classify(self.sample("BAD", "GOOD", note)), "bad")

    def test_include_overridden_policy(self):
        note = {"original_verdict": "BAD", "new_verdict": "GOOD", "reason": "r", "by": "s"}
        self.assertEqual(classify(self.sample("GOOD", "BAD", note), "include-overridden"), "good")
        self.assertEqual(classify(self.sample("GOOD", "REVIEW", note), "include-overridden"), "good")
        self.assertEqual(classify(self.sample("BAD", "GOOD"), "include-overridden"), "bad")

    def test_review_and_unknown_are_excluded(self):
        for policy in ("strict", "include-overridden"):
            self.assertEqual(classify(self.sample("REVIEW", "REVIEW"), policy), "excluded")
            self.assertEqual(classify(self.sample("WHAT", "GOOD"), policy), "excluded")

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            classify(self.sample("GOOD", "GOOD"), "loose")


class TestCollect(EvidenceFixture):
    def test_strict_policy_labels(self):
        ds = Dataset.collect(self.root, "PN-1")
        self.assertEqual(self.ids(ds.good), ["P-A1", "P-A5"])
        self.assertEqual(self.ids(ds.bad), ["P-A2"])
        self.assertEqual(self.ids(ds.excluded), ["P-A3"])
        self.assertEqual(ds.problems, [])

    def test_include_overridden_moves_override_to_good(self):
        ds = Dataset.collect(self.root, "PN-1", good_policy="include-overridden")
        self.assertEqual(self.ids(ds.good), ["P-A1", "P-A3", "P-A5"])
        self.assertEqual(self.ids(ds.bad), ["P-A2"])
        self.assertEqual(ds.excluded, [])

    def test_counts_per_camera_side_label(self):
        ds = Dataset.collect(self.root, "PN-1")
        counts = ds.counts()
        for cam in CAMERA_IDS:
            for side in SIDE_NAMES:
                self.assertEqual(counts[(cam, side, "good")], 2)
                self.assertEqual(counts[(cam, side, "bad")], 1)
        self.assertEqual(sum(counts.values()), 24)

    def test_exclude_session_ids(self):
        ds = Dataset.collect(self.root, "PN-1", exclude_session_ids=["S-A5"])
        self.assertEqual(self.ids(ds.good), ["P-A1"])
        self.assertEqual(self.ids(ds.excluded), ["P-A3", "P-A5"])

    def test_summary_is_plain_and_complete(self):
        summary = Dataset.collect(self.root, "PN-1").summary()
        self.assertEqual(summary["part_number"], "PN-1")
        self.assertEqual(summary["parts_good"], 2)
        self.assertEqual(summary["parts_bad"], 1)
        self.assertEqual(summary["parts_excluded"], 1)
        self.assertEqual(summary["problems"], 0)
        self.assertEqual(summary["images_by_key"]["C1/TOP/good"], 2)
        self.assertEqual(summary["images_by_key"]["C4/BOTTOM/bad"], 1)
        self.assertEqual(summary["first_utc"], "2026-10-01T08:00:00Z")
        self.assertEqual(summary["last_utc"], "2026-10-05T08:00:00Z")

    def test_problems_reported_on_collect(self):
        self.image("P-A1", 1, "C1", "TOP").write_bytes(b"tampered")
        ds = Dataset.collect(self.root, "PN-1")
        self.assertEqual(self.ids(ds.good), ["P-A5"])
        self.assertEqual(len(ds.problems), 1)


def fake_sample(part_id, started, images=()):
    return PartSample(
        part_number="PN-1", part_id=part_id, session_id="S-" + part_id, started_utc=started,
        machine_verdict="GOOD", final_verdict="GOOD", override=None, defect_label=None,
        model_version="m", partial_station=False, camera_count=4, images=tuple(images),
        part_dir=pathlib.Path("/unused") / part_id,
    )


def fake_images(part_id, n=2):
    return [
        ImageRef(path=pathlib.Path("/unused") / part_id / f"{i}.png", camera_id="C1",
                 side="TOP", score=None, sha256="0" * 64)
        for i in range(n)
    ]


def six_days():
    samples = []
    for day in range(1, 7):
        for k in range(2):
            pid = f"P-{day:02d}{k}"
            samples.append(fake_sample(pid, f"2026-10-{day:02d}T{8 + k:02d}:00:00Z",
                                       fake_images(pid)))
    return samples


class TestSplit(unittest.TestCase):
    def test_deterministic_for_a_seed(self):
        samples = six_days()
        first = split_holdout(samples, 0.5, seed=7)
        second = split_holdout(samples, 0.5, seed=7)
        self.assertEqual([s.part_id for s in first[0]], [s.part_id for s in second[0]])
        self.assertEqual([s.part_id for s in first[1]], [s.part_id for s in second[1]])

    def test_seed_changes_the_holdout(self):
        samples = six_days()
        holdouts = {tuple(s.part_id for s in split_holdout(samples, 0.5, seed=seed)[1])
                    for seed in range(1, 7)}
        self.assertGreater(len(holdouts), 1)

    def test_partition_is_complete_and_never_splits_images(self):
        samples = six_days()
        train, holdout = split_holdout(samples, 0.5, seed=3)
        self.assertEqual(len(train) + len(holdout), len(samples))
        train_paths = {img.path for s in train for img in s.images}
        holdout_paths = {img.path for s in holdout for img in s.images}
        self.assertEqual(train_paths & holdout_paths, set())
        self.assertEqual(train_paths | holdout_paths, {img.path for s in samples for img in s.images})

    def test_whole_days_stay_together(self):
        train, holdout = split_holdout(six_days(), 0.5, seed=3)
        sides = {}
        for side_name, group in (("train", train), ("holdout", holdout)):
            for s in group:
                sides.setdefault(s.started_utc[:10], set()).add(side_name)
        self.assertTrue(all(len(v) == 1 for v in sides.values()), sides)
        self.assertEqual(len(holdout), 6)  # three whole days of two parts

    def test_fraction_zero_keeps_everything_in_train(self):
        samples = six_days()
        train, holdout = split_holdout(samples, 0, seed=1)
        self.assertEqual(holdout, [])
        self.assertEqual(len(train), len(samples))

    def test_bad_fractions_rejected(self):
        for value in (0.95, -0.1, True, "0.5"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    split_holdout(six_days(), value, seed=1)

    def test_single_day_falls_back_to_whole_parts(self):
        samples = [fake_sample(f"P-{k:02d}", f"2026-10-04T{k:02d}:00:00Z", fake_images(f"P-{k:02d}"))
                   for k in range(10)]
        train, holdout = split_holdout(samples, 0.3, seed=4)
        again_train, again_holdout = split_holdout(samples, 0.3, seed=4)
        self.assertEqual(len(holdout), 3)
        self.assertEqual([s.part_id for s in holdout], [s.part_id for s in again_holdout])
        held = {img.path for s in holdout for img in s.images}
        kept = {img.path for s in train for img in s.images}
        self.assertEqual(held & kept, set())


class TestListPartNumbers(EvidenceFixture):
    def test_counts_by_part_number(self):
        self.assertEqual(list_part_numbers(self.root), {"PN-1": 4, "PN-2": 1})

    def test_partial_included_on_request(self):
        self.assertEqual(list_part_numbers(self.root, full_station_only=False),
                         {"PN-1": 5, "PN-2": 1})

    def test_date_window(self):
        self.assertEqual(list_part_numbers(self.root, since="2026-10-02", until="2026-10-03"),
                         {"PN-1": 1, "PN-2": 1})


class TestInventoryCli(EvidenceFixture):
    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_prints_part_numbers_and_counts(self):
        code, out, err = self.run_cli(["inventory", "--store", str(self.root)])
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertIn("part PN-1", out)
        self.assertIn("part PN-2", out)
        self.assertIn("problems: 0", out)
        self.assertIn(["C1", "TOP", "good", "2"], [line.split() for line in out.splitlines()])
        self.assertIn(["C4", "BOTTOM", "bad", "1"], [line.split() for line in out.splitlines()])

    def test_part_filter_and_policy(self):
        code, out, _ = self.run_cli(["inventory", "--store", str(self.root), "--part", "PN-2"])
        self.assertEqual(code, 0)
        self.assertNotIn("part PN-1", out)
        code, out, _ = self.run_cli(["inventory", "--store", str(self.root), "--part", "PN-1",
                                     "--good-policy", "include-overridden"])
        self.assertEqual(code, 0)
        self.assertIn(["C1", "TOP", "good", "3"], [line.split() for line in out.splitlines()])

    def test_problem_lines_shown(self):
        (self.pdir("P-A2", 1) / "part.json").write_text("{bad", encoding="utf-8")
        code, out, _ = self.run_cli(["inventory", "--store", str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn("problems: 1", out)

    def test_missing_store_exit_2(self):
        code, out, err = self.run_cli(["inventory", "--store", str(self.root / "missing")])
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("inventory error: "), err)

    def test_bad_date_exit_2(self):
        code, _out, err = self.run_cli(["inventory", "--store", str(self.root), "--since", "yesterday"])
        self.assertEqual(code, 2)
        self.assertIn("inventory error:", err)

    def test_no_command_exit_2(self):
        code, _out, _err = self.run_cli([])
        self.assertEqual(code, 2)


class CollectedLabelDatasetTests(unittest.TestCase):
    """Collected parts (final REVIEW) are labelled by the operator; other parts keep the strict rule."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        store = Store(self.root)
        when = "2026-10-26T08:00:00Z"

        def collected(session, part, label=None):
            rec = make_record(session, part, "PN-1", when, Verdict.REVIEW, machine=Verdict.REVIEW)
            return dataclasses.replace(rec, disposition=COLLECTED, operator_label=label)

        store.save_part(collected("S-c0", "P-c0"))  # unlabelled
        store.save_part(collected("S-c1", "P-c1", OperatorLabel(Verdict.GOOD, "qa-1", when)))
        store.save_part(collected("S-c2", "P-c2", OperatorLabel(
            Verdict.BAD, "qa-1", when, criterion_id="CR-1")))
        store.save_part(make_record("S-g1", "P-g1", "PN-1", when, Verdict.GOOD))
        store.save_part(make_record("S-b1", "P-b1", "PN-1", when, Verdict.BAD, bad=("C1", Side.TOP)))
        self.samples = {s.part_id: s for s in iter_parts(self.root, problems=None)}

    def tearDown(self):
        self._tmp.cleanup()

    def part_json(self, part_id):
        return next((self.root / "store").rglob(f"{part_id}/part.json"))

    def test_labels_decide_collected_parts(self):
        for policy in ("strict", "include-overridden"):
            with self.subTest(policy=policy):
                self.assertEqual(classify(self.samples["P-c0"], policy), "excluded")
                self.assertEqual(classify(self.samples["P-c1"], policy), "good")
                self.assertEqual(classify(self.samples["P-c2"], policy), "bad")

    def test_non_collected_parts_keep_the_strict_rule(self):
        self.assertEqual(classify(self.samples["P-g1"]), "good")
        self.assertEqual(classify(self.samples["P-b1"]), "bad")
        self.assertEqual(classify(self.samples["P-g1"], "include-overridden"), "good")

    def test_unknown_policy_still_rejected_for_collected(self):
        with self.assertRaises(ValueError):
            classify(self.samples["P-c1"], "loose")

    def test_part_json_without_the_field_loads_as_before(self):
        path = self.part_json("P-g1")
        doc = json.loads(path.read_text(encoding="utf-8"))
        del doc["operator_label"]
        path.write_text(json.dumps(doc), encoding="utf-8")
        problems = []
        parts = {s.part_id: s for s in iter_parts(self.root, problems=problems)}
        self.assertEqual(problems, [])
        self.assertIsNone(parts["P-g1"].operator_label)
        self.assertEqual(classify(parts["P-g1"]), "good")

    def test_only_official_labels_train(self):
        """Proposed and rejected labels stay out; a label with no status key is official."""
        for session, part, status in (("S-p1", "P-p1", "proposed"), ("S-r1", "P-r1", "rejected")):
            store = Store(self.root)
            label = {"verdict": "BAD", "criterion_id": "CR-1", "labelled_by": "op-1",
                     "labelled_at": "2026-10-26T08:00:00Z", "status": status}
            rec = make_record(session, part, "PN-1", "2026-10-26T08:00:00Z", Verdict.REVIEW,
                              machine=Verdict.REVIEW)
            store.save_part(dataclasses.replace(rec, disposition=COLLECTED, operator_label=None))
            path = self.part_json(part)
            doc = json.loads(path.read_text(encoding="utf-8"))
            doc["operator_label"] = label
            path.write_text(json.dumps(doc), encoding="utf-8")
        problems = []
        parts = {s.part_id: s for s in iter_parts(self.root, problems=problems)}
        self.assertEqual(problems, [])
        self.assertEqual(classify(parts["P-p1"]), "excluded")
        self.assertEqual(classify(parts["P-r1"]), "excluded")
        self.assertEqual(classify(self.samples["P-c1"]), "good")  # no status key -> official

    def test_unknown_status_is_a_problem(self):
        path = self.part_json("P-c1")
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["operator_label"]["status"] = "maybe"
        path.write_text(json.dumps(doc), encoding="utf-8")
        problems = []
        ids = [s.part_id for s in iter_parts(self.root, problems=problems)]
        self.assertNotIn("P-c1", ids)
        self.assertTrue(any("bad operator_label" in p for p in problems), problems)

    def test_malformed_operator_label_is_a_problem(self):
        path = self.part_json("P-c1")
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["operator_label"] = {"verdict": "MAYBE"}
        path.write_text(json.dumps(doc), encoding="utf-8")
        problems = []
        ids = [s.part_id for s in iter_parts(self.root, problems=problems)]
        self.assertNotIn("P-c1", ids)
        self.assertTrue(any("bad operator_label" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
