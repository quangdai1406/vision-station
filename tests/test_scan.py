"""Tests for station.scan.ScanSession. Fakes only; time is injected, no sleeping.

The scan store is real (a temporary folder), so these tests also prove that the
saved scan verifies clean with station.scanstore.verify_scan.
"""
import inspect
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

from station import scanstore
from station.contracts import CAMERA_IDS, CameraError, Frame, Side
from station.scan import ScanError, ScanSession
from station.scanstore import ScanStore, list_scans, load_scan, verify_scan

CAMS = list(CAMERA_IDS)
NOW = "2026-10-27T08:30:12Z"
REPO = pathlib.Path(__file__).resolve().parent.parent
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


def make_frames(side, cams=CAMS):
    return [
        Frame(camera_id=cid, side=side, seq=i, timestamp_utc=NOW,
              data=f"{side.value}-{cid}-{i}".encode() * 20)
        for i, cid in enumerate(cams)
    ]


class FakeCameras:
    """Each capture pops one scripted action: None -> all 4 cameras, list -> those frames,
    exception -> raise it."""

    def __init__(self):
        self.sides = []
        self.script = []

    def capture(self, side):
        self.sides.append(side)
        action = self.script.pop(0) if self.script else None
        if isinstance(action, BaseException):
            raise action
        if isinstance(action, list):
            return action
        return make_frames(side)


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def advance(self, dt):
        self.t += dt

    def __call__(self):
        return self.t


class Harness:
    def __init__(self, busy=False, debounce_s=0.3, min_flip_s=2.0):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.store = ScanStore(self.root, site="SITE", line="L1")
        self.cameras = FakeCameras()
        self.clock = FakeClock(100.0)
        self.busy_flag = [busy]
        self.min_flip_s = min_flip_s
        self.session = ScanSession(
            self.cameras, self.store, busy=lambda: self.busy_flag[0],
            debounce_s=debounce_s, min_flip_s=min_flip_s, clock=self.clock, now_utc=lambda: NOW,
            scan_id_factory=iter(f"SC-{i}" for i in range(1, 100)).__next__,
        )

    def flip(self):
        """Wait out the flip gap (min_flip_s), then flip the part explicitly."""
        self.clock.advance(self.min_flip_s)
        self.session.flip_ready()

    def cleanup(self):
        self._tmp.cleanup()

    def start(self, label=None, sides=None):
        kwargs = {} if sides is None else {"sides": sides}
        return self.session.start(
            "PN-1", "toy-3", GOOD_LABEL if label is None else label, "op-1", **kwargs
        )

    def press(self, dt=1.0):
        """Advance the clock past the debounce, then press. Returns the press result."""
        self.clock.advance(dt)
        return self.session.press()


class ScanSessionTestCase(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.addCleanup(self.h.cleanup)
        self.s = self.h.session

    def take(self, n, dt=1.0):
        """n accepted presses, each after the debounce window."""
        for _ in range(n):
            self.assertTrue(self.h.press(dt), "press was ignored")

    def finish_good(self, top=3, bottom=5):
        self.h.start()
        self.take(top)
        self.assertEqual(self.s.done_side(), "WAIT_FLIP")
        self.h.flip()
        self.take(bottom)
        self.assertEqual(self.s.done_side(), "SAVING")
        return self.s.finish()

    def stored_scan_dirs(self):
        """Everything under the line folder (staging folders and finished scans)."""
        return list(self.h.store.line_dir.rglob("*"))

    def pictures_on_disk(self):
        return [p for p in self.h.store.line_dir.rglob("*") if p.is_file()]


class TestSaveAndVerify(ScanSessionTestCase):
    def test_good_scan_three_top_five_bottom_saved_and_verifies(self):
        final = self.finish_good(top=3, bottom=5)
        self.assertEqual(self.s.snapshot()["state"], "DONE")
        self.assertEqual(verify_scan(final), [])
        doc = load_scan(final)
        self.assertEqual(doc["steps"], {"TOP": 3, "BOTTOM": 5})
        self.assertEqual(len(doc["images"]), (3 + 5) * 4)
        self.assertTrue(all(img["defect_visible"] is None for img in doc["images"]))
        self.assertEqual(self.h.cameras.sides[:3], [Side.TOP] * 3)
        self.assertEqual(self.h.cameras.sides[3:], [Side.BOTTOM] * 5)
        self.assertEqual(list_scans(self.h.root), [final])

    def test_reset_returns_to_idle_and_allows_next_scan(self):
        self.finish_good(top=1, bottom=1)
        self.s.reset()
        self.assertEqual(self.s.snapshot()["state"], "IDLE")
        self.assertEqual(self.h.start(), "SC-2")
        self.assertEqual(self.s.snapshot()["state"], "SCAN_SIDE")

    def test_bad_scan_marks_default_false_and_mark_before_finish_is_saved(self):
        self.h.start(label=BAD_LABEL)
        self.take(2)
        self.s.done_side()
        self.h.flip()
        self.take(1)
        self.s.done_side()
        self.s.mark("C1", "TOP", 2, True)
        self.s.mark("C4", "BOTTOM", 1, True)
        self.s.mark("C4", "BOTTOM", 1, False)  # changed again before finish: last value wins
        self.s.mark("C4", "BOTTOM", 1, True)
        final = self.s.finish()
        doc = load_scan(final)
        visible = {(i["camera_id"], i["side"], i["step"]): i["defect_visible"] for i in doc["images"]}
        self.assertTrue(visible[("C1", "TOP", 2)])
        self.assertTrue(visible[("C4", "BOTTOM", 1)])
        self.assertFalse(visible[("C2", "TOP", 1)])  # default False
        self.assertFalse(visible[("C1", "TOP", 1)])
        self.assertEqual(len(visible), 12)
        self.assertEqual(verify_scan(final), [])


class TestCancelAndRefusals(ScanSessionTestCase):
    def test_cancel_mid_way_leaves_nothing(self):
        self.h.start()
        self.take(2)
        self.s.done_side()
        self.h.flip()
        self.take(1)
        self.s.cancel()
        self.assertEqual(self.s.snapshot()["state"], "IDLE")
        self.assertEqual(list_scans(self.h.root), [])
        self.assertEqual(self.stored_scan_dirs(), [])

    def test_cancel_in_wait_flip_leaves_nothing(self):
        self.h.start()
        self.take(1)
        self.s.done_side()
        self.s.cancel()
        self.assertEqual(self.stored_scan_dirs(), [])

    def test_start_refused_when_busy(self):
        self.h.busy_flag[0] = True
        with self.assertRaises(ScanError):
            self.h.start()
        self.assertEqual(self.s.snapshot()["state"], "IDLE")
        self.assertEqual(self.stored_scan_dirs(), [])

    def test_start_refused_when_not_idle(self):
        self.h.start()
        with self.assertRaises(ScanError):
            self.h.start()
        self.assertEqual(self.s.snapshot()["state"], "SCAN_SIDE")

    def test_start_bad_label_refused_and_stays_idle(self):
        with self.assertRaises(ScanError):
            self.h.start(label={"verdict": "GOOD", "defects": ["x"], "status": "official",
                                "labelled_by": "a", "labelled_at": NOW})
        self.assertEqual(self.s.snapshot()["state"], "IDLE")

    def test_done_side_with_zero_steps_refused(self):
        self.h.start()
        with self.assertRaises(ScanError):
            self.s.done_side()
        self.assertEqual(self.s.snapshot()["state"], "SCAN_SIDE")

    def test_finish_only_after_last_side(self):
        self.h.start()
        self.take(1)
        with self.assertRaises(ScanError):
            self.s.finish()
        self.assertEqual(self.s.snapshot()["state"], "SCAN_SIDE")
        self.assertEqual(list_scans(self.h.root), [])

    def test_mark_refused_for_good_scan(self):
        self.h.start()
        self.take(1)
        with self.assertRaises(ScanError):
            self.s.mark("C1", "TOP", 1, True)

    def test_mark_unknown_picture_refused(self):
        self.h.start(label=BAD_LABEL)
        self.take(1)
        with self.assertRaises(ScanError):
            self.s.mark("C1", "TOP", 9, True)
        with self.assertRaises(ScanError):
            self.s.mark("C1", "BOTTOM", 1, True)

    def test_cancel_and_reset_refused_when_idle_or_done(self):
        with self.assertRaises(ScanError):
            self.s.cancel()
        with self.assertRaises(ScanError):
            self.s.reset()
        self.finish_good(top=1, bottom=1)
        with self.assertRaises(ScanError):
            self.s.cancel()


class TestSides(ScanSessionTestCase):
    def test_default_requires_both_sides(self):
        self.h.start()
        self.take(1)
        self.assertEqual(self.s.done_side(), "WAIT_FLIP")
        self.h.flip()
        self.take(1)
        self.assertEqual(self.s.done_side(), "SAVING")

    def test_explicit_top_only_scan_saves_after_top(self):
        self.h.start(sides=("TOP",))
        self.take(2)
        self.assertEqual(self.s.done_side(), "SAVING")
        final = self.s.finish()
        doc = load_scan(final)
        self.assertEqual(doc["steps"], {"TOP": 2})
        self.assertEqual(verify_scan(final), [])

    def test_explicit_sides_must_be_valid(self):
        for bad in ((), ("TOP", "TOP"), ("LEFT",), "TOP"):
            with self.subTest(sides=bad):
                with self.assertRaises(ScanError):
                    self.h.start(sides=bad)
        self.assertEqual(self.s.snapshot()["state"], "IDLE")


class TestCameraFailure(ScanSessionTestCase):
    def test_camera_failure_names_camera_keeps_steps_and_retry_succeeds(self):
        self.h.start(sides=("TOP",))
        self.take(1)
        self.h.cameras.script = [CameraError("C2: grab failed")]
        self.assertTrue(self.h.press())
        snap = self.s.snapshot()
        self.assertEqual(snap["state"], "FAULT")
        self.assertIn("C2", snap["fault"])
        self.assertIn("TOP step 2", snap["fault"])
        self.assertEqual(snap["counts"]["TOP"], 1)
        self.assertEqual(snap["step"], 1)
        self.assertIn("retry", snap["prompt"])
        self.assertTrue(self.h.press())  # retry of the same step 2
        snap = self.s.snapshot()
        self.assertEqual(snap["state"], "SCAN_SIDE")
        self.assertIsNone(snap["fault"])
        self.assertEqual(snap["counts"]["TOP"], 2)
        self.assertEqual(self.s.done_side(), "SAVING")
        final = self.s.finish()
        self.assertEqual(verify_scan(final), [])
        self.assertEqual(load_scan(final)["steps"], {"TOP": 2})

    def test_missing_camera_is_named_and_writes_nothing(self):
        self.h.start()
        partial = make_frames(Side.TOP, cams=["C1", "C2", "C4"])
        self.h.cameras.script = [partial]
        self.assertTrue(self.h.press())
        snap = self.s.snapshot()
        self.assertEqual(snap["state"], "FAULT")
        self.assertIn("C3", snap["fault"])
        self.assertEqual(snap["counts"]["TOP"], 0)
        self.assertEqual(self.pictures_on_disk(), [])  # nothing from the failed step was written

    def test_clear_fault_returns_to_the_side(self):
        self.h.start()
        self.h.cameras.script = [CameraError("C1: no frame")]
        self.assertTrue(self.h.press())
        self.s.clear_fault()
        self.assertEqual(self.s.snapshot()["state"], "SCAN_SIDE")
        self.assertEqual(self.s.snapshot()["counts"]["TOP"], 0)
        with self.assertRaises(ScanError):
            self.s.clear_fault()

    def test_fault_on_flip_press_keeps_the_flip_and_retry_takes_bottom_step_one(self):
        self.h.start()
        self.take(1)
        self.s.done_side()
        self.h.cameras.script = [CameraError("C3: timeout")]
        # The flip press, 2.0 s after Done (the flip gap): BOTTOM step 1 fails on C3.
        self.assertTrue(self.h.press(dt=2.0))
        snap = self.s.snapshot()
        self.assertEqual(snap["state"], "FAULT")
        self.assertEqual(snap["side"], "BOTTOM")
        self.assertIn("BOTTOM step 1", snap["fault"])
        self.assertTrue(self.h.press())
        self.assertEqual(self.s.snapshot()["state"], "SCAN_SIDE")
        self.assertEqual(self.s.snapshot()["counts"], {"TOP": 1, "BOTTOM": 1})

    def test_retry_that_fails_again_stays_in_fault(self):
        self.h.start()
        self.h.cameras.script = [CameraError("C1: no frame"), CameraError("C1: no frame")]
        self.h.press()
        self.h.press()  # retry fails again
        self.assertEqual(self.s.snapshot()["state"], "FAULT")
        self.assertEqual(self.s.snapshot()["counts"]["TOP"], 0)

    def test_write_failure_discards_scan_and_fault_returns_to_idle(self):
        real_write = scanstore._write_durable
        calls = {"n": 0}

        def flaky(path, data):
            calls["n"] += 1
            if calls["n"] == 2:  # second picture of step 1
                raise OSError("disk full")
            return real_write(path, data)

        self.h.start()
        with mock.patch.object(scanstore, "_write_durable", side_effect=flaky):
            self.assertTrue(self.h.press())
        snap = self.s.snapshot()
        self.assertEqual(snap["state"], "FAULT")
        self.assertIn("discarded", snap["fault"])
        self.assertEqual(self.stored_scan_dirs(), [])
        self.assertEqual(list_scans(self.h.root), [])
        self.s.clear_fault()
        self.assertEqual(self.s.snapshot()["state"], "IDLE")


class TestDebounce(ScanSessionTestCase):
    def test_fast_second_press_is_ignored(self):
        self.h.start()
        self.assertTrue(self.h.press(0.0))
        self.assertFalse(self.h.press(0.1))  # 0.1 s after the accepted press: inside 0.3 s
        self.assertEqual(self.s.snapshot()["counts"]["TOP"], 1)
        self.assertTrue(self.h.press(0.3))  # 0.4 s after the accepted press: allowed
        self.assertEqual(self.s.snapshot()["counts"]["TOP"], 2)
        self.assertEqual(len(self.h.cameras.sides), 2)

    def test_press_refused_in_wrong_state_returns_false(self):
        self.assertFalse(self.h.press())  # IDLE
        self.h.start()
        self.take(1)
        self.s.done_side()
        self.h.flip()
        self.take(1)
        self.s.done_side()
        self.assertFalse(self.h.press())  # SAVING


class TestFlip(ScanSessionTestCase):
    def test_flip_ready_only_in_wait_flip(self):
        self.h.start()
        with self.assertRaises(ScanError):
            self.s.flip_ready()

    def test_done_side_only_in_scan_side(self):
        self.h.start()
        self.take(1)
        self.s.done_side()
        with self.assertRaises(ScanError):
            self.s.done_side()


class TestFlipGap(ScanSessionTestCase):
    """min_flip_s: a flip (press or flip_ready) is refused until it has passed since Done."""

    def take_top_and_done(self):
        self.h.start()
        self.take(1)
        self.assertEqual(self.s.done_side(), "WAIT_FLIP")

    def test_press_right_after_done_side_is_ignored_and_captures_nothing(self):
        self.take_top_and_done()
        calls_before = len(self.h.cameras.sides)
        self.assertFalse(self.h.press(dt=1.0))  # 1.0 s after Done: debounce passed, flip gap not
        self.assertEqual(len(self.h.cameras.sides), calls_before)
        snap = self.s.snapshot()
        self.assertEqual(snap["state"], "WAIT_FLIP")
        self.assertEqual(snap["counts"], {"TOP": 1, "BOTTOM": 0})
        self.assertFalse(snap["press_ready"])

    def test_press_after_min_flip_flips_and_takes_bottom_step_one(self):
        self.take_top_and_done()
        self.assertFalse(self.h.press(dt=1.0))
        self.assertTrue(self.h.press(dt=1.0))  # 2.0 s after Done
        snap = self.s.snapshot()
        self.assertEqual(snap["state"], "SCAN_SIDE")
        self.assertEqual((snap["side"], snap["step"]), ("BOTTOM", 1))
        self.assertEqual(snap["counts"], {"TOP": 1, "BOTTOM": 1})
        self.assertEqual(self.h.cameras.sides[-1], Side.BOTTOM)

    def test_min_flip_zero_keeps_immediate_flip(self):
        h = Harness(min_flip_s=0)
        self.addCleanup(h.cleanup)
        h.start()
        h.press(dt=1.0)
        self.assertEqual(h.session.done_side(), "WAIT_FLIP")
        self.assertTrue(h.press(dt=1.0))  # debounce passed, no flip gap
        self.assertEqual(h.session.snapshot()["counts"], {"TOP": 1, "BOTTOM": 1})

    def test_double_tap_done_then_enter_is_ignored_by_debounce(self):
        h = Harness(min_flip_s=0)  # flip gap off, so only the debounce can stop the tap
        self.addCleanup(h.cleanup)
        h.start()
        h.press(dt=1.0)
        h.session.done_side()
        self.assertFalse(h.press(dt=0.1))  # 0.1 s after Done: inside debounce_s
        self.assertEqual(h.session.snapshot()["state"], "WAIT_FLIP")
        self.assertEqual(len(h.cameras.sides), 1)
        # The default harness: the same double tap is ignored too, and nothing is captured.
        self.take_top_and_done()
        self.assertFalse(self.h.press(dt=0.1))
        self.assertEqual(len(self.h.cameras.sides), 1)

    def test_flip_ready_too_early_raises_with_seconds_left(self):
        self.take_top_and_done()
        self.h.clock.advance(0.5)
        with self.assertRaises(ScanError) as ctx:
            self.s.flip_ready()
        self.assertIn("1.5 s", str(ctx.exception))
        self.assertEqual(self.s.snapshot()["state"], "WAIT_FLIP")
        self.h.clock.advance(1.5)
        self.s.flip_ready()
        self.assertEqual(self.s.snapshot()["state"], "SCAN_SIDE")

    def test_snapshot_shows_flip_wait_and_press_ready(self):
        self.take_top_and_done()
        snap = self.s.snapshot()
        self.assertEqual(snap["flip_wait_s"], 2.0)
        self.assertFalse(snap["press_ready"])
        self.assertIn("Wait 2.0 s", snap["prompt"])
        self.h.clock.advance(1.5)
        snap = self.s.snapshot()
        self.assertEqual(snap["flip_wait_s"], 0.5)
        self.assertFalse(snap["press_ready"])
        self.h.clock.advance(0.5)
        snap = self.s.snapshot()
        self.assertEqual(snap["flip_wait_s"], 0.0)
        self.assertTrue(snap["press_ready"])
        self.assertIn("Flip the part", snap["prompt"])
        self.assertNotIn("Wait", snap["prompt"])

    def test_snapshot_flip_wait_is_zero_on_side_and_idle(self):
        self.assertEqual(self.s.snapshot()["flip_wait_s"], 0.0)
        self.h.start()
        self.assertEqual(self.s.snapshot()["flip_wait_s"], 0.0)

    def test_negative_min_flip_refused(self):
        with self.assertRaises(ValueError):
            ScanSession(self.h.cameras, self.h.store, busy=lambda: False, min_flip_s=-0.1)
        self.assertEqual(self.s.snapshot()["state"], "IDLE")


class TestThreads(ScanSessionTestCase):
    def test_two_threads_pressing_do_not_corrupt_the_scan(self):
        h = Harness(debounce_s=0.0)
        self.addCleanup(h.cleanup)
        h.start(sides=("TOP",))
        results = []
        errors = []

        def presser():
            try:
                for _ in range(10):
                    results.append(h.session.press())
            except Exception as exc:  # any exception is a failure
                errors.append(exc)

        threads = [threading.Thread(target=presser) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(all(results), True)
        self.assertEqual(h.session.snapshot()["counts"]["TOP"], 20)
        h.session.done_side()
        h.session.finish()
        scans = list_scans(h.root)
        self.assertEqual(len(scans), 1)
        self.assertEqual(verify_scan(scans[0]), [])
        self.assertEqual(load_scan(scans[0])["steps"], {"TOP": 20})


class TestSnapshot(ScanSessionTestCase):
    EXPECTED_KEYS = {
        "state", "side", "step", "counts", "sides", "scan_id", "part_number", "toy_id",
        "label_verdict", "defects", "marks", "camera_names", "fault", "saved_path",
        "press_ready", "flip_wait_s", "prompt",
    }

    def test_snapshot_keys_and_idle_values(self):
        snap = self.s.snapshot()
        self.assertEqual(set(snap), self.EXPECTED_KEYS)
        self.assertEqual(snap["state"], "IDLE")
        self.assertEqual(snap["counts"], {"TOP": 0, "BOTTOM": 0})
        self.assertEqual(snap["camera_names"], CAMS)
        self.assertIsNone(snap["scan_id"])
        self.assertTrue(snap["prompt"])

    def test_snapshot_during_scan_and_after_save(self):
        self.h.start(label=BAD_LABEL)
        self.take(2)
        snap = self.s.snapshot()
        self.assertEqual(set(snap), self.EXPECTED_KEYS)
        self.assertEqual(snap["state"], "SCAN_SIDE")
        self.assertEqual((snap["side"], snap["step"]), ("TOP", 2))
        self.assertEqual(snap["scan_id"], "SC-1")
        self.assertEqual(snap["part_number"], "PN-1")
        self.assertEqual(snap["toy_id"], "toy-3")
        self.assertEqual(snap["label_verdict"], "BAD")
        self.assertEqual(snap["defects"], ["doh_small", "doh_edge"])
        self.assertEqual(snap["marks"], {"pictures": 8, "visible": 0, "not_visible": 8})
        self.assertIn("TOP step 3", snap["prompt"])
        self.assertIn("Done with this side", snap["prompt"])
        self.s.done_side()
        self.assertIn("Wait", self.s.snapshot()["prompt"])  # flip gap still running
        self.h.flip()
        self.take(1)
        self.s.done_side()
        self.s.mark("C1", "TOP", 1, True)
        self.s.finish()
        snap = self.s.snapshot()
        self.assertEqual(set(snap), self.EXPECTED_KEYS)
        self.assertEqual(snap["state"], "DONE")
        self.assertTrue(snap["saved_path"].endswith("SC-1"))
        self.assertEqual(snap["marks"]["visible"], 1)


class TestStepPicture(ScanSessionTestCase):
    """step_picture(): the picture of the most recent saved step, for the UI thumbnails."""

    def test_none_before_any_step_and_for_unknown_camera(self):
        self.assertIsNone(self.s.step_picture("C1"))
        self.h.start()
        self.assertIsNone(self.s.step_picture("C1"))
        self.take(1)
        self.assertIsNone(self.s.step_picture("C9"))

    def test_returns_latest_step_bytes_and_ext_per_camera(self):
        self.h.start()
        first = [Frame(camera_id=c, side=Side.TOP, seq=i, timestamp_utc=NOW,
                       data=f"one-{c}".encode(), ext="jpg") for i, c in enumerate(CAMS)]
        second = [Frame(camera_id=c, side=Side.TOP, seq=i, timestamp_utc=NOW,
                        data=f"two-{c}".encode(), ext="png") for i, c in enumerate(CAMS)]
        self.h.cameras.script = [first, second]
        self.take(1)
        self.assertEqual(self.s.step_picture("C2"), (b"one-C2", "jpg"))
        self.take(1)
        self.assertEqual(self.s.step_picture("C2"), (b"two-C2", "png"))
        self.assertEqual(self.s.step_picture("C4"), (b"two-C4", "png"))

    def test_kept_through_done_side_and_flip_wait_then_none_after_finish(self):
        self.h.start()
        self.take(1)
        self.s.done_side()
        self.assertIsNotNone(self.s.step_picture("C1"))
        self.h.flip()
        self.take(1)
        self.s.done_side()
        self.assertIsNotNone(self.s.step_picture("C1"))
        self.s.finish()
        self.assertIsNone(self.s.step_picture("C1"))

    def test_none_after_cancel_and_after_a_camera_fault(self):
        self.h.start()
        self.take(1)
        self.s.cancel()
        self.assertIsNone(self.s.step_picture("C1"))
        self.h.start()
        self.h.cameras.script = [CameraError("C3 lost the image")]
        self.assertTrue(self.h.press(1.0))  # a camera fault is still a processed press
        self.assertEqual(self.s.snapshot()["state"], "FAULT")
        self.assertIsNone(self.s.step_picture("C1"))

    def test_memory_holds_only_the_last_step(self):
        self.h.start()
        self.take(3)
        self.assertEqual(set(self.s._last_pictures), set(CAMS))
        self.assertEqual(len(self.s._last_pictures), len(CAMS))


class TestNoOutputs(unittest.TestCase):
    def test_session_has_no_outputs_parameter(self):
        self.assertNotIn("outputs", inspect.signature(ScanSession).parameters)

    def test_module_does_not_load_outputs_session_fusion_storage_or_mes(self):
        code = (
            "import sys, station.scan\n"
            "bad = [m for m in ('station.outputs', 'station.session', 'station.fusion', "
            "'station.storage', 'station.mes_client') if m in sys.modules]\n"
            "print(bad)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=str(REPO), capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
