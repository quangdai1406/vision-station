import contextlib
import io
import json
import pathlib
import tempfile
import unittest
import zlib

from station.contracts import (
    CAMERA_IDS,
    IMAGES_PER_PART,
    Frame,
    ImageResult,
    PartRecord,
    Side,
    StationError,
    Thresholds,
    Verdict,
)
from station.modelstore import ModelStore, safe_part
from station.storage import Store
from trainer.__main__ import _lazy_command, build_parser
from trainer.weekly import QUALITY, run_weekly

try:
    import numpy as np
    from station.anomaly import PatchCoreBuilder, PatchCoreModel, _write_npz
except ImportError:  # numpy missing: the numpy-dependent tests are skipped
    np = None

NOW_1 = "2026-10-04T00:00:00Z"
NOW_2 = "2026-10-12T00:00:00Z"
PAIRS = [(cam, side) for side in (Side.TOP, Side.BOTTOM) for cam in CAMERA_IDS]


def save_part(store, session, pn, started, final=Verdict.GOOD, *, poison=False):
    """One saved part with 8 images. The image bytes come from the session id, so each part differs."""
    tag = "POISON" if poison else "OK"
    frames, results = [], []
    for seq, (cam, side) in enumerate(PAIRS):
        frames.append(Frame(camera_id=cam, side=side, seq=seq, timestamp_utc=started,
                            data=f"{session}-{tag}-{cam}-{side.value}".encode() * 40, ext="png"))
        results.append(ImageResult(camera_id=cam, side=side, score=0.1))
    store.save_part(PartRecord(
        part_id=f"P-{session}", session_id=session, station_id="ST-1", part_number=pn,
        lot="LOT-1", operator="op-1", supervisor="sup-1", material_handler="mh-1",
        started_utc=started, model_version="m-1.0", thresholds=Thresholds(t_low=0.3, t_img=0.6),
        frames=frames, results=results, machine_verdict=final, final_verdict=final, override=None,
    ))


def fake_decode(frame):
    if b"POISON" in bytes(frame.data):
        raise StationError("test image cannot be decoded")
    seed = zlib.crc32(bytes(frame.data)) & 0xFFFFFFFF
    return np.random.RandomState(seed).randint(0, 256, size=(16, 16, 3)).astype(np.uint8)


class FakeExtractor:
    name = "fake-test-extractor-v1"
    stride = 2
    input_hw = (16, 16)

    def extract(self, image):
        arr = np.asarray(image, dtype=np.float32)
        return arr[::2, ::2, :3] / 255.0


def as_v1(src, dst):
    """Rewrite a saved v2 model file as the old patchcore-v1 format (same arrays, older schema tag)."""
    with np.load(str(src), allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    meta = json.loads(str(arrays["meta"]))
    meta["schema"] = "patchcore-v1"
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    _write_npz(np, dst, arrays)


class WeeklyBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self._tmp.name)
        self.store_root = root / "evidence"
        self.models = root / "models"
        self.reports = root / "reports"
        self.store = Store(self.store_root)

    def tearDown(self):
        self._tmp.cleanup()

    def args(self, *extra, now=NOW_1):
        base = ["weekly", "--store", str(self.store_root), "--models", str(self.models),
                "--reports", str(self.reports), "--min-good", "2", "--min-holdout", "0",
                "--holdout-fraction", "0", "--now-utc", now]
        return build_parser().parse_args(base + list(extra))

    def run_quiet(self, args, extractor_factory=FakeExtractor, decoder_factory=lambda: fake_decode):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = run_weekly(args, extractor_factory=extractor_factory, decoder_factory=decoder_factory)
        return code, out.getvalue()

    def report_json(self, now_date="2026-10-04"):
        return json.loads((self.reports / f"weekly-{now_date}.json").read_text(encoding="utf-8"))

    def part_result(self, report, pn):
        return next(p for p in report["parts"] if p["part_number"] == pn)


@unittest.skipUnless(np is not None, "numpy is not installed")
class TestWeeklyRuns(WeeklyBase):
    def test_part_below_min_good_is_skipped_never_trained(self):
        save_part(self.store, "S-1", "PN-1", "2026-10-01T08:00:00Z")
        code, _ = self.run_quiet(self.args())
        self.assertEqual(code, 0)
        report = self.report_json()
        self.assertEqual(self.part_result(report, "PN-1")["status"], "skipped")
        self.assertIn("below --min-good", self.part_result(report, "PN-1")["reason"])
        self.assertEqual(ModelStore(self.models).list_versions("PN-1"), [])

    def test_candidate_is_registered_and_never_approved(self):
        for day in (1, 2, 3, 5):
            save_part(self.store, f"S-{day}", "PN-1", f"2026-10-0{day}T08:00:00Z")
        code, _ = self.run_quiet(self.args("--holdout-fraction", "0.25", "--min-holdout", "1"))
        self.assertEqual(code, 0)
        ms = ModelStore(self.models)
        self.assertIsNone(ms.current("PN-1"))
        self.assertEqual(ms.approvals("PN-1"), [])
        versions = ms.list_versions("PN-1")
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0]["status"], "candidate")
        self.assertEqual(ms.verify("PN-1", versions[0]["version"]), [])
        part = self.part_result(self.report_json(), "PN-1")
        self.assertEqual(part["counts"]["good_holdout_parts"], 1)
        self.assertEqual(part["evaluation"]["held_out_good"]["images_scored"], IMAGES_PER_PART)
        self.assertEqual(part["evaluation"]["t_img"], QUALITY)
        self.assertEqual(part["evaluation"]["held_out_good_parts_flagged"], QUALITY)
        # keys that trainer.purge reads
        self.assertEqual(part["candidate_version"], versions[0]["version"])
        self.assertIs(part["training_succeeded"], True)
        self.assertEqual(len(part["holdout_session_ids"]), 1)
        self.assertEqual(len(part["trained_session_ids"]), 3)
        self.assertFalse(set(part["trained_session_ids"]) & set(part["holdout_session_ids"]))

    def test_weekly_report_is_accepted_by_purge_planning(self):
        from trainer.purge import plan_purge
        for day in (1, 2, 3, 5):
            save_part(self.store, f"S-{day}", "PN-1", f"2026-10-0{day}T08:00:00Z")
        code, _ = self.run_quiet(self.args("--holdout-fraction", "0.25", "--min-holdout", "1"))
        self.assertEqual(code, 0)
        report = next(self.reports.glob("weekly-*.json"))
        plan = plan_purge(self.store_root, self.models, part_number="PN-1", report_path=report)
        self.assertIsNone(plan.refusal)
        self.assertEqual(len(plan.parts), 3)  # the 3 trained good parts; the held-out one is kept

    def test_incremental_update_uses_only_new_pictures(self):
        for day in (1, 2, 3):
            save_part(self.store, f"S-{day}", "PN-1", f"2026-10-0{day}T08:00:00Z")
        self.assertEqual(self.run_quiet(self.args())[0], 0)
        ms = ModelStore(self.models)
        first = ms.list_versions("PN-1")[0]["version"]
        ms.approve("PN-1", first, "tester", "first weekly candidate", NOW_1)

        save_part(self.store, "S-10", "PN-1", "2026-10-10T08:00:00Z")
        save_part(self.store, "S-11", "PN-1", "2026-10-11T08:00:00Z")
        self.assertEqual(self.run_quiet(self.args(now=NOW_2))[0], 0)

        second = next(m for m in ms.list_versions("PN-1") if m["version"] != first)
        part = self.part_result(self.report_json("2026-10-12"), "PN-1")
        self.assertEqual(part["mode"], "incremental")
        self.assertEqual(part["parent_version"], first)
        self.assertEqual(part["counts"]["new_train_parts"], 2)
        model = PatchCoreModel.load(
            ms.root / safe_part("PN-1") / second["version"] / "patchcore.npz", fake_decode, FakeExtractor())
        bank = model.describe()["banks"]["C1/TOP"]
        self.assertEqual(bank["train_images"], 2)   # only the two new pictures
        self.assertEqual(bank["images_seen"], 3 + 2)  # parent's 3 plus the 2 new

    def test_v1_parent_trains_fresh_and_says_so(self):
        for day in (1, 2, 3):
            save_part(self.store, f"S-{day}", "PN-1", f"2026-10-0{day}T08:00:00Z")
        self.assertEqual(self.run_quiet(self.args())[0], 0)
        ms = ModelStore(self.models)
        first = ms.list_versions("PN-1")[0]
        legacy_file = self.models / "legacy.npz"
        as_v1(ms.root / safe_part("PN-1") / first["version"] / "patchcore.npz", legacy_file)
        ms.add_candidate("PN-1", "legacy-v1", legacy_file,
                         {"type": "patchcore", "trained_on": first["trained_on"]}, NOW_1)
        ms.approve("PN-1", "legacy-v1", "tester", "old format parent", NOW_1)

        save_part(self.store, "S-10", "PN-1", "2026-10-10T08:00:00Z")
        self.assertEqual(self.run_quiet(self.args(now=NOW_2))[0], 0)
        part = self.part_result(self.report_json("2026-10-12"), "PN-1")
        self.assertEqual(part["mode"], "fresh")
        self.assertIn("patchcore-v1", part["mode_note"])
        self.assertEqual(part["status"], "candidate")
        new = next(m for m in ms.list_versions("PN-1") if m["version"] not in (first["version"], "legacy-v1"))
        model = PatchCoreModel.load(
            ms.root / safe_part("PN-1") / new["version"] / "patchcore.npz", fake_decode, FakeExtractor())
        self.assertEqual(model.describe()["banks"]["C1/TOP"]["train_images"], 4)  # all good parts, not just new

    def test_report_files_written_atomically(self):
        for day in (1, 2):
            save_part(self.store, f"S-{day}", "PN-1", f"2026-10-0{day}T08:00:00Z")
        self.assertEqual(self.run_quiet(self.args())[0], 0)
        text = (self.reports / "weekly-2026-10-04.txt").read_text(encoding="utf-8")
        self.assertIn("Part PN-1: CANDIDATE", text)
        self.assertIn("Nothing was approved", text)
        self.assertEqual(sorted(p.name for p in self.reports.iterdir()),
                         ["weekly-2026-10-04.json", "weekly-2026-10-04.txt"])

    def test_dry_run_builds_and_writes_nothing(self):
        for day in (1, 2):
            save_part(self.store, f"S-{day}", "PN-1", f"2026-10-0{day}T08:00:00Z")

        def no_extractor():
            raise AssertionError("dry run must not build the extractor")

        code, out = self.run_quiet(self.args("--dry-run"), extractor_factory=no_extractor,
                                   decoder_factory=None)
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", out)
        self.assertIn("PLANNED", out)
        self.assertFalse(self.reports.exists())
        self.assertFalse(self.models.exists())

    def test_one_failing_part_does_not_stop_another(self):
        save_part(self.store, "S-A1", "PN-A", "2026-10-01T08:00:00Z")
        save_part(self.store, "S-A2", "PN-A", "2026-10-02T08:00:00Z")
        save_part(self.store, "S-A3", "PN-A", "2026-10-03T08:00:00Z", poison=True)
        save_part(self.store, "S-B1", "PN-B", "2026-10-01T09:00:00Z")
        save_part(self.store, "S-B2", "PN-B", "2026-10-02T09:00:00Z")
        code, _ = self.run_quiet(self.args())
        self.assertEqual(code, 1)
        report = self.report_json()
        self.assertEqual(self.part_result(report, "PN-A")["status"], "error")
        self.assertEqual(self.part_result(report, "PN-B")["status"], "candidate")
        ms = ModelStore(self.models)
        self.assertEqual(ms.list_versions("PN-A"), [])
        self.assertEqual(len(ms.list_versions("PN-B")), 1)


class TestWeeklyArguments(unittest.TestCase):
    def parse_error(self, argv):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            build_parser().parse_args(argv)
        return caught.exception.code

    def test_min_good_and_min_holdout_are_required(self):
        base = ["weekly", "--store", "s", "--models", "m", "--reports", "r"]
        self.assertEqual(self.parse_error(base + ["--min-holdout", "1"]), 2)
        self.assertEqual(self.parse_error(base + ["--min-good", "2"]), 2)

    def test_min_good_must_be_positive(self):
        base = ["weekly", "--store", "s", "--models", "m", "--reports", "r", "--min-holdout", "0"]
        self.assertEqual(self.parse_error(base + ["--min-good", "0"]), 2)

    def test_missing_extractor_without_factory_is_usage_error(self):
        args = build_parser().parse_args(["weekly", "--store", "s", "--models", "m", "--reports", "r",
                                          "--min-good", "2", "--min-holdout", "0", "--holdout-fraction", "0"])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_weekly(args), 2)

    def test_lazy_command_reports_missing_module_clearly(self):
        _add, run = _lazy_command("purge", "trainer.no_such_module_for_test", ("add_arguments",), ("run",))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(run(build_parser().parse_args(["inventory", "--store", "s"])), 2)
        self.assertIn("purge: not available", err.getvalue())


if __name__ == "__main__":
    unittest.main()
