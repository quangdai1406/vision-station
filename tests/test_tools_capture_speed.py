"""Tests for tools/capture_speed.py with a stub camera set (no camera, no numpy)."""
import unittest

from station.contracts import CameraError, Frame, Side
from tools.capture_speed import run_rounds


class StubCams:
    camera_ids = ("C1", "C2")

    def __init__(self, fail_round=None):
        self.n = 0
        self.fail_round = fail_round
        self.last_failed_cameras = []
        self.last_worker_times_s = [0.1, 0.2]
        self.last_capture_times_s = [0.05, 0.06]

    def capture(self, side):
        self.n += 1
        if self.n == self.fail_round:
            self.last_failed_cameras = ["C2"]
            raise CameraError("C2: grab failed")
        return [Frame(c, Side.TOP, i, "t", bytes([10 * (i + 1)]), "png") for i, c in enumerate(self.camera_ids)]


class ProbeTests(unittest.TestCase):
    def test_summary(self):
        ticks = iter(x * 0.5 for x in range(100))
        lines = []
        out = run_rounds(StubCams(), 3, decode_mean=lambda b: float(b[0]),
                         clock=lambda: next(ticks), log=lines.append)
        self.assertEqual((out["rounds"], out["ok_rounds"]), (3, 3))
        self.assertAlmostEqual(out["worst_s"], 0.5)
        self.assertEqual(out["brightness"], {"C1": 10.0, "C2": 20.0})
        self.assertAlmostEqual(out["min_grab_wait_s"]["C1"], 0.05)
        self.assertTrue(any("SUMMARY" in l for l in lines))

    def test_failed_round_is_counted_not_fatal(self):
        out = run_rounds(StubCams(fail_round=2), 3, decode_mean=lambda b: 1.0, log=lambda s: None)
        self.assertEqual((out["ok_rounds"], out["failed_rounds"]), (2, 1))
        self.assertEqual(out["failures_by_camera"], {"C1": 0, "C2": 1})


if __name__ == "__main__":
    unittest.main()
