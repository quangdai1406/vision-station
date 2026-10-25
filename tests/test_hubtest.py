"""Tests for tools/hubtest.py. A fake cv2 stands in for OpenCV; numpy is real when installed."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is required only for these tests
    np = None

from station import cameras as station_cameras
from tools import hubtest
from tools.hubtest import MODE_ORDER, MODE_TITLES, main, recommend, run_hubtest, wait_text

DEVICES = [0, 1, 2, 3]
# Real grabs per picture in the station: one picture grab after FLUSH_FRAMES flush grabs.
FLUSH = station_cameras.STALE_FLUSH_FRAMES


class FakeClock:
    """Monotonic clock that advances a fixed step on every call, so timings are deterministic."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.25
        return self.t


class FakeCap:
    def __init__(self, hub: "FakeHubCv2", index: int) -> None:
        self.hub = hub
        self.index = index
        self.is_open = index not in hub.fail_open
        self.w = 64
        self.h = 48
        self.grabbed = False
        self.released = False

    def isOpened(self) -> bool:
        return self.is_open

    def set(self, prop: int, value: float) -> bool:
        if prop == self.hub.CAP_PROP_FRAME_WIDTH:
            self.w = int(value)
        elif prop == self.hub.CAP_PROP_FRAME_HEIGHT:
            self.h = int(value)
        elif prop == self.hub.CAP_PROP_BUFFERSIZE:
            if self.hub.buffersize_fails:
                raise RuntimeError("buffer size not supported")
            self.hub.buffer_sets.append((self.index, int(value)))
        return True

    def grab(self) -> bool:
        self.hub.grab_counts[self.index] = self.hub.grab_counts.get(self.index, 0) + 1
        if self.index in self.hub.raise_grab:
            raise RuntimeError("boom")
        if self.index in self.hub.dead:
            return False
        if self.hub.overlap_fails and (self.hub.pending - {self.index}):
            # Shared hub: a second camera asked for a picture before the first was read.
            return False
        self.hub.pending.add(self.index)
        self.grabbed = True
        return True

    def retrieve(self):
        self.hub.pending.discard(self.index)
        if not self.grabbed:
            return False, None
        self.grabbed = False
        value = 0 if self.index in self.hub.black else 120 + self.index
        return True, np.full((self.h, self.w, 3), value, dtype=np.uint8)

    def release(self) -> None:
        self.hub.pending.discard(self.index)
        self.released = True
        self.hub.released.append(self.index)


class FakeHubCv2:
    CAP_DSHOW = 700
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_BUFFERSIZE = 38
    COLOR_BGR2GRAY = 6

    def __init__(self, *, fail_open=(), dead=(), black=(), overlap_fails=False, raise_grab=(),
                 buffersize_fails=False) -> None:
        self.fail_open = set(fail_open)
        self.dead = set(dead)
        self.black = set(black)
        self.overlap_fails = overlap_fails
        self.raise_grab = set(raise_grab)
        self.buffersize_fails = buffersize_fails
        self.pending: set[int] = set()
        self.open_calls: list[tuple[int, object]] = []
        self.buffer_sets: list[tuple[int, int]] = []
        self.grab_counts: dict[int, int] = {}
        self.released: list[int] = []
        self.imwrite_ok = True
        self.saved: list[str] = []
        # Optional: a clock the fake advances on every open, to simulate slow opening.
        self.clock: FakeClock | None = None
        self.open_cost = 0.0

    def VideoCapture(self, index: int, api: object = None) -> FakeCap:
        self.open_calls.append((index, api))
        if self.clock is not None:
            self.clock.t += self.open_cost
        return FakeCap(self, index)

    def cvtColor(self, img, code):
        assert code == self.COLOR_BGR2GRAY
        return img[..., 0]

    def imwrite(self, path: str, img) -> bool:
        if not self.imwrite_ok:
            return False
        Path(path).write_bytes(b"\x89PNG fake")
        self.saved.append(Path(path).name)
        return True


def _run(hub: FakeHubCv2, out_dir: str, **overrides) -> dict:
    kwargs = dict(devices=DEVICES, width=64, height=48, rounds=1, warmup=1, backend="default",
                  out_path=Path(out_dir) / "hubtest_report.json", cv2_module=hub,
                  clock=FakeClock(), now=lambda: "2026-10-25T09:00:00Z", out=lambda s: None)
    kwargs.update(overrides)
    return run_hubtest(**kwargs)


def _cams(report: dict, mode: str, rnd: int = 1) -> dict:
    rounds = report["modes"][mode]["rounds"]
    return {c["device"]: c for c in rounds[rnd - 1]["cameras"]}


@unittest.skipUnless(np is not None, "numpy is required for hub tests")
class HubTestRuns(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def test_all_ok_recommends_simultaneous_and_samples_saved(self) -> None:
        hub = FakeHubCv2()
        report = _run(hub, self.dir, rounds=2)
        for mode in MODE_ORDER:
            s = report["modes"][mode]["summary"]
            self.assertEqual((s["pictures_ok"], s["pictures_total"]), (8, 8), mode)
            self.assertIsNone(report["modes"][mode]["fatal_error"])
        self.assertEqual(report["recommended_mode"], "simultaneous")
        self.assertEqual(
            report["sample_pictures"],
            {"0": "hubtest_report_cam0_first.png", "1": "hubtest_report_cam1_first.png",
             "2": "hubtest_report_cam2_first.png", "3": "hubtest_report_cam3_first.png"},
        )
        self.assertEqual(sorted(hub.saved), sorted(report["sample_pictures"].values()))
        self.assertTrue((Path(self.dir) / "hubtest_report_cam2_first.png").exists())

    def test_simultaneous_fails_on_shared_hub_sequential_keep_open_worked(self) -> None:
        hub = FakeHubCv2(overlap_fails=True)
        report = _run(hub, self.dir, rounds=2)
        sim = report["modes"]["simultaneous"]
        self.assertEqual(sim["summary"]["pictures_ok"], 2)  # only device 0 per round
        failed = {c["device"] for c in _cams(report, "simultaneous", 2).values() if not c["ok"]}
        self.assertEqual(failed, {1, 2, 3})
        self.assertIn("device 1", _cams(report, "simultaneous")[1]["error"])
        self.assertEqual(report["modes"]["sequential_keep_open"]["summary"]["pictures_ok"], 8)
        self.assertEqual(report["modes"]["sequential_open_per_shot"]["summary"]["pictures_ok"], 8)
        self.assertEqual(report["recommended_mode"], "sequential_keep_open")

    def test_one_camera_fails_in_sequential_is_reported_with_index(self) -> None:
        hub = FakeHubCv2(dead={2})
        report = _run(hub, self.dir, rounds=2)
        for mode in ("sequential_open_per_shot", "sequential_keep_open"):
            cams = _cams(report, mode, 2)
            self.assertFalse(cams[2]["ok"], mode)
            self.assertIn("device 2", cams[2]["error"], mode)
            self.assertEqual(sum(c["ok"] for c in cams.values()), 3, mode)
            self.assertEqual(report["modes"][mode]["summary"]["pictures_ok"], 6, mode)
        self.assertIsNone(report["recommended_mode"])

    def test_camera_that_does_not_open_is_reported_in_both_sequential_modes(self) -> None:
        hub = FakeHubCv2(fail_open={1})
        report = _run(hub, self.dir, rounds=1)
        a1 = _cams(report, "sequential_open_per_shot")[1]
        a2 = _cams(report, "sequential_keep_open")[1]
        self.assertIn("could not open camera", a1["error"])
        self.assertIn("could not open camera", a2["error"])
        self.assertFalse(a1["ok"])
        self.assertTrue(_cams(report, "sequential_open_per_shot")[0]["ok"])

    def test_all_black_frame_is_detected(self) -> None:
        hub = FakeHubCv2(black={3})
        report = _run(hub, self.dir, rounds=1)
        for mode in MODE_ORDER:
            cam = _cams(report, mode)[3]
            self.assertFalse(cam["ok"], mode)
            self.assertIn("all-black", cam["error"], mode)
            self.assertEqual(cam["mean_grey"], 0.0, mode)
        self.assertTrue(_cams(report, "sequential_open_per_shot")[0]["ok"])
        self.assertIsNone(report["sample_pictures"]["3"])  # no sample for a black camera

    def test_exception_in_one_camera_does_not_stop_the_others(self) -> None:
        hub = FakeHubCv2(raise_grab={1})
        report = _run(hub, self.dir, rounds=1)
        cams = _cams(report, "sequential_open_per_shot")
        self.assertIn("grab failed (boom)", cams[1]["error"])
        self.assertTrue(cams[0]["ok"] and cams[2]["ok"] and cams[3]["ok"])
        self.assertIsNone(report["modes"]["sequential_open_per_shot"]["fatal_error"])

    def test_open_per_shot_opens_each_device_every_round_keep_open_only_once(self) -> None:
        hub = FakeHubCv2()
        _run(hub, self.dir, rounds=3, backend="dshow")
        opened = [index for index, _ in hub.open_calls]
        # A1: 4 devices x 3 rounds; A2: 4 once; B: 4 x 3 rounds
        self.assertEqual(opened.count(0), 3 + 1 + 3)
        self.assertEqual(len(opened), 12 + 4 + 12)
        self.assertEqual(hub.open_calls[0][1], FakeHubCv2.CAP_DSHOW)

    def test_default_backend_is_the_station_backend_and_prints_no_warning(self) -> None:
        hub = FakeHubCv2()
        lines: list[str] = []
        report = _run(hub, self.dir, rounds=1, out=lines.append)  # no backend given
        self.assertTrue(all(api is None for _, api in hub.open_calls))
        self.assertEqual(report["requested"]["backend"], "default")
        self.assertEqual(report["station_backend"], "default")
        self.assertFalse([line for line in lines if line.startswith("warning:")])

    def test_dshow_uses_dshow_and_prints_exactly_one_warning_line(self) -> None:
        hub = FakeHubCv2()
        lines: list[str] = []
        report = _run(hub, self.dir, rounds=1, backend="dshow", out=lines.append)
        self.assertTrue(all(api == FakeHubCv2.CAP_DSHOW for _, api in hub.open_calls))
        warnings = [line for line in lines if line.startswith("warning:")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("differs from the station (default)", warnings[0])
        self.assertEqual(report["requested"]["backend"], "dshow")

    def test_json_report_is_valid_and_written_atomically(self) -> None:
        hub = FakeHubCv2(overlap_fails=True)
        out_path = Path(self.dir) / "report.json"
        err, text = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = main(["--devices", "0", "1", "2", "3", "--rounds", "2", "--warmup", "1",
                       "--width", "64", "--height", "48", "--out", str(out_path)],
                      cv2_module=hub, np_module=np, now=lambda: "2026-10-25T09:00:00Z",
                      clock=FakeClock(), out=lambda s: text.write(s + "\n"))
        self.assertEqual(rc, 0, err.getvalue())
        report = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], 1)
        self.assertEqual(report["devices"], DEVICES)
        self.assertEqual(report["requested"]["backend"], "default")  # the station's backend
        self.assertEqual(report["station_backend"], "default")
        self.assertEqual(set(report["modes"]), set(MODE_ORDER))
        self.assertEqual(report["recommended_mode"], "sequential_keep_open")
        leftovers = [p.name for p in Path(self.dir).iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])
        printed = text.getvalue()
        self.assertIn("Recommended mode: sequential keep-open worked", printed)
        self.assertIn("[Quality to define]", printed)
        self.assertIn("Test B (all at once, the current station mode) is expected to fail on a USB hub",
                      printed)
        self.assertIn("Test B failures above are a result, not a crash", printed)

    def test_failed_sample_save_is_skipped_quietly(self) -> None:
        hub = FakeHubCv2()
        hub.imwrite_ok = False
        report = _run(hub, self.dir, rounds=1)
        self.assertEqual(set(report["sample_pictures"].values()), {None})
        self.assertEqual(report["modes"]["sequential_open_per_shot"]["summary"]["pictures_ok"], 4)

    def test_summary_text_when_nothing_worked(self) -> None:
        hub = FakeHubCv2(dead={0})
        text = io.StringIO()
        with contextlib.redirect_stderr(io.StringIO()):
            rc = main(["--devices", "0", "1", "--rounds", "1", "--warmup", "0", "--width", "64",
                       "--height", "48", "--out", str(Path(self.dir) / "r.json")],
                      cv2_module=hub, np_module=np, clock=FakeClock(), out=lambda s: text.write(s + "\n"))
        self.assertEqual(rc, 0)
        printed = text.getvalue()
        self.assertIn("Recommended mode: none worked", printed)
        self.assertIn("Operator wait for 2 pictures: n/a (no mode worked)", printed)
        self.assertIn("Summary: cameras 0 1", printed)

    def test_buffersize_is_set_to_one_on_every_open(self) -> None:
        hub = FakeHubCv2()
        _run(hub, self.dir, rounds=2)
        opened = sorted(index for index, _ in hub.open_calls)
        self.assertEqual(len(opened), 20)  # A1 8 + A2 4 + B 8
        self.assertEqual(sorted(index for index, _ in hub.buffer_sets), opened)
        self.assertTrue(all(value == 1 for _, value in hub.buffer_sets))
        self.assertEqual(hubtest.STATION_BUFFER_FRAMES, 1)

    def test_buffersize_error_is_ignored(self) -> None:
        hub = FakeHubCv2(buffersize_fails=True)
        report = _run(hub, self.dir, rounds=2)
        self.assertEqual(hub.buffer_sets, [])
        self.assertEqual(report["modes"]["sequential_open_per_shot"]["summary"]["pictures_ok"], 8)
        self.assertEqual(report["recommended_mode"], "simultaneous")

    def test_flush_and_warm_up_grabs_match_the_station_in_each_mode(self) -> None:
        # Per device: warm-up 1 grab after each open, FLUSH grabs before each picture, 1 picture grab.
        hub = FakeHubCv2()
        snapshots: dict[str, dict] = {}

        def out(line: str) -> None:
            if line in MODE_TITLES.values():
                snapshots[line] = dict(hub.grab_counts)

        _run(hub, self.dir, rounds=2, warmup=1, out=out)
        final = dict(hub.grab_counts)
        a1, a2, b = (MODE_TITLES[k] for k in ("sequential_open_per_shot", "sequential_keep_open", "simultaneous"))

        def delta(start: dict, end: dict, dev: int) -> int:
            return end.get(dev, 0) - start.get(dev, 0)

        for dev in DEVICES:
            self.assertEqual(delta(snapshots[a1], snapshots[a2], dev), 2 * (1 + FLUSH + 1), f"A1 dev {dev}")
            self.assertEqual(delta(snapshots[a2], snapshots[b], dev), 1 + 2 * (FLUSH + 1), f"A2 dev {dev}")
            self.assertEqual(delta(snapshots[b], final, dev), 2 * (1 + FLUSH + 1), f"B dev {dev}")

    def test_keep_open_round_one_includes_opening_and_later_rounds_do_not(self) -> None:
        clock = FakeClock()
        hub = FakeHubCv2()
        hub.clock = clock
        hub.open_cost = 100.0  # every open takes 100 s on the fake clock
        report = _run(hub, self.dir, rounds=3, clock=clock)
        keep = report["modes"]["sequential_keep_open"]
        summary = keep["summary"]
        self.assertGreater(summary["round1_s"], 400)  # four opens in round 1
        self.assertEqual(summary["round1_s"], keep["rounds"][0]["round_s"])
        later = [r["round_s"] for r in keep["rounds"][1:]]
        self.assertAlmostEqual(summary["later_round_s_mean"], sum(later) / len(later))
        self.assertLess(summary["later_round_s_mean"], 10)

    def test_operator_wait_for_keep_open_uses_rounds_two_and_later(self) -> None:
        clock = FakeClock()
        hub = FakeHubCv2(overlap_fails=True)  # simultaneous fails, keep-open and A1 work
        hub.clock = clock
        hub.open_cost = 100.0
        out_path = Path(self.dir) / "r.json"
        text = io.StringIO()
        rc = main(["--devices", "0", "1", "2", "3", "--rounds", "3", "--warmup", "1", "--width", "64",
                   "--height", "48", "--out", str(out_path)],
                  cv2_module=hub, np_module=np, clock=clock, out=lambda s: text.write(s + "\n"))
        self.assertEqual(rc, 0)
        report = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(report["recommended_mode"], "sequential_keep_open")
        later = report["modes"]["sequential_keep_open"]["summary"]["later_round_s_mean"]
        printed = text.getvalue()
        self.assertIn(f"Operator wait for 4 pictures: {later:.2f} s (keep-open, mean of rounds 2 and later", printed)
        self.assertIn("Round 1 (includes opening and warm-up)", printed)


class WaitTextTests(unittest.TestCase):
    @staticmethod
    def _report(recommended, rounds_n=3, **summary) -> dict:
        base = {"round_s_mean": 9.0, "round1_s": 12.0, "later_round_s_mean": 2.5}
        base.update(summary)
        return {"devices": [0, 1, 2, 3], "requested": {"rounds": rounds_n}, "recommended_mode": recommended,
                "modes": {recommended: {"summary": base}} if recommended else {}}

    def test_keep_open_uses_rounds_two_and_later(self) -> None:
        text = wait_text(self._report("sequential_keep_open"))
        self.assertTrue(text.startswith("Operator wait for 4 pictures: 2.50 s (keep-open, mean of rounds 2 and later"))
        self.assertIn("opening and warm-up not included", text)
        self.assertIn("[Quality to define]", text)

    def test_keep_open_with_one_round_is_not_available(self) -> None:
        text = wait_text(self._report("sequential_keep_open", later_round_s_mean=None))
        self.assertIn("Operator wait for 4 pictures: n/a", text)
        self.assertIn("--rounds 2 or more", text)

    def test_open_per_shot_uses_the_full_per_shot_cycle(self) -> None:
        text = wait_text(self._report("sequential_open_per_shot", rounds_n=3))
        self.assertIn("Operator wait for 4 pictures: 9.00 s", text)
        self.assertIn("full per-shot cycle: open, warm-up, picture, release", text)
        self.assertIn("mean of 3 round(s)", text)

    def test_simultaneous_uses_the_full_round(self) -> None:
        text = wait_text(self._report("simultaneous", rounds_n=2))
        self.assertIn("Operator wait for 4 pictures: 9.00 s", text)
        self.assertIn("full round: open, warm-up, all pictures", text)

    def test_none_worked(self) -> None:
        text = wait_text(self._report(None))
        self.assertEqual(text, "Operator wait for 4 pictures: n/a (no mode worked) "
                               "(decide with Quality if acceptable: [Quality to define])")


class RecommendationRuleTests(unittest.TestCase):
    @staticmethod
    def _mode(ok_flags: list[list[bool]], fatal: str | None = None) -> dict:
        rounds = [{"round": i + 1, "round_s": 1.0,
                   "cameras": [{"device": d, "ok": ok, "error": None if ok else "x", "release_error": None}
                               for d, ok in enumerate(flags)]}
                  for i, flags in enumerate(ok_flags)]
        return {"rounds": rounds, "fatal_error": fatal}

    def test_simultaneous_preferred_when_it_works(self) -> None:
        modes = {k: self._mode([[True, True]]) for k in MODE_ORDER}
        self.assertEqual(recommend(modes), "simultaneous")

    def test_keep_open_next_when_simultaneous_fails(self) -> None:
        modes = {"simultaneous": self._mode([[True, False]]),
                 "sequential_keep_open": self._mode([[True, True]]),
                 "sequential_open_per_shot": self._mode([[True, True]])}
        self.assertEqual(recommend(modes), "sequential_keep_open")

    def test_open_per_shot_when_only_it_works(self) -> None:
        modes = {"simultaneous": self._mode([[False, True]]),
                 "sequential_keep_open": self._mode([[True, False]]),
                 "sequential_open_per_shot": self._mode([[True, True]])}
        self.assertEqual(recommend(modes), "sequential_open_per_shot")

    def test_one_failed_round_disqualifies_a_mode(self) -> None:
        modes = {"simultaneous": self._mode([[True, True], [True, False]]),
                 "sequential_keep_open": self._mode([[True, True], [True, True]])}
        self.assertEqual(recommend(modes), "sequential_keep_open")

    def test_fatal_error_or_no_rounds_disqualify(self) -> None:
        modes = {"simultaneous": self._mode([[True, True]], fatal="boom"),
                 "sequential_keep_open": {"rounds": [], "fatal_error": None},
                 "sequential_open_per_shot": self._mode([[True, True]])}
        self.assertEqual(recommend(modes), "sequential_open_per_shot")

    def test_none_when_nothing_worked(self) -> None:
        modes = {k: self._mode([[True, False]]) for k in MODE_ORDER}
        self.assertIsNone(recommend(modes))
        self.assertIsNone(recommend({}))


class MissingDependencyTests(unittest.TestCase):
    def test_missing_cv2_exits_2_installs_nothing_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "report.json"
            err = io.StringIO()
            with mock.patch.dict(sys.modules, {"cv2": None}), contextlib.redirect_stderr(err):
                rc = main(["--devices", "0", "1", "2", "3", "--out", str(out_path)])
            self.assertEqual(rc, 2)
            self.assertIn("py -m pip install opencv-python numpy", err.getvalue())
            self.assertIn("installs nothing", err.getvalue())
            self.assertFalse(out_path.exists())

    @unittest.skipUnless(np is not None, "numpy is required for hub tests")
    def test_missing_numpy_exits_2(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            err = io.StringIO()
            with mock.patch.dict(sys.modules, {"numpy": None}), contextlib.redirect_stderr(err):
                rc = main(["--devices", "0", "--out", str(Path(d) / "r.json")], cv2_module=FakeHubCv2())
            self.assertEqual(rc, 2)
            self.assertIn("numpy", err.getvalue())

    def test_bad_arguments_exit_2(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(main(["--devices", "0", "0"], cv2_module=FakeHubCv2()), 2)
            self.assertEqual(main(["--devices", "0", "--rounds", "0"], cv2_module=FakeHubCv2()), 2)
        self.assertIn("unique", err.getvalue())


@unittest.skipUnless(np is not None, "numpy is required for hub tests")
class ReportEncodingTests(unittest.TestCase):
    def test_json_value_or_type_error_exits_1_and_writes_nothing(self) -> None:
        for exc in (ValueError("Out of range float values are not JSON compliant"),
                    TypeError("Object of type set is not JSON serializable")):
            with self.subTest(error=type(exc).__name__), tempfile.TemporaryDirectory() as d:
                out_path = Path(d) / "r.json"
                err = io.StringIO()
                with mock.patch.object(hubtest.json, "dumps", side_effect=exc), \
                        contextlib.redirect_stderr(err):
                    rc = main(["--devices", "0", "--rounds", "1", "--warmup", "0", "--width", "64",
                               "--height", "48", "--out", str(out_path)],
                              cv2_module=FakeHubCv2(), np_module=np, clock=FakeClock(), out=lambda s: None)
                self.assertEqual(rc, 1)
                self.assertIn("not valid JSON", err.getvalue())
                self.assertFalse(out_path.exists())
                self.assertEqual([p for p in Path(d).iterdir() if p.name.endswith(".tmp")], [])


class ConstantsTests(unittest.TestCase):
    def test_station_values_are_used(self) -> None:
        self.assertEqual(hubtest.DEFAULT_WARMUP, station_cameras.OPEN_WARMUP_FRAMES)
        self.assertEqual(hubtest.FLUSH_FRAMES, station_cameras.STALE_FLUSH_FRAMES)
        self.assertEqual(hubtest.STATION_BACKEND, "default")
        self.assertEqual(hubtest.build_parser().get_default("backend"), "default")


if __name__ == "__main__":
    unittest.main()
