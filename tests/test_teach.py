"""Tests for `python -m station teach` (station/teach.py). Fake cameras, no hardware."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import shutil
import tempfile
import unittest

from station import __main__ as cli
from station import golden, teach
from station.cameras import FakeCameraSet
from station.contracts import Side

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def _decode(frame):
    import numpy as np

    return np.full((16, 16), float(sum(frame.data) % 256))


class _Inputs:
    """Fake input(): records calls; raises exc on call number fail_at (1-based)."""

    def __init__(self, fail_at=None, exc=EOFError):
        self.calls = 0
        self.fail_at = fail_at
        self.exc = exc

    def __call__(self, prompt=""):
        self.calls += 1
        if self.calls == self.fail_at:
            raise self.exc()
        return ""


class _TeachCase(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="teach-test-"))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.out = self.dir / "golden.npz"

    def write_config(self, kind="golden", with_path=True):
        model = {"kind": kind, "version": "golden-test"}
        if with_path:
            model["path"] = "golden.npz"
        data = {
            "site": "SITE", "line": "LINE", "station_id": "ST-1", "part_number": "PN-1",
            "lot": "LOT-1", "operator": "op", "supervisor": "sup", "material_handler": "mh",
            "thresholds": {"t_low": 0.2, "t_img": 0.5},
            "data_dir": str(self.dir / "data"), "users_file": str(self.dir / "users.json"),
            "mes": {"base_url": "https://mes.example.test", "token_env": "STATION_MES_TOKEN",
                    "criteria_ids": ["CRIT-1"],
                    "class_to_criterion": {"difference_from_golden": "CRIT-1"}},
            "cameras": {"kind": "fake"},
            "model": model,
            "outputs": {"kind": "console"},
        }
        path = self.dir / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path


class TeachArgumentTests(_TeachCase):
    """No numpy needed: every case stops before a model is built."""

    def test_model_kind_not_golden_exits_2(self):
        path = self.write_config(kind="marker")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = cli.main(["teach", "--config", str(path)])
        self.assertEqual(rc, 2)
        self.assertIn("model.kind must be 'golden'", err.getvalue())
        self.assertFalse(self.out.exists())

    def test_frames_per_side_one_is_rejected_by_argparse(self):
        path = self.write_config()
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["teach", "--config", str(path), "--frames-per-side", "1"])
        self.assertEqual(ctx.exception.code, 2)

    def test_parts_zero_is_rejected_by_argparse(self):
        path = self.write_config()
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["teach", "--config", str(path), "--parts", "0"])
        self.assertEqual(ctx.exception.code, 2)

    def test_run_teach_rejects_frames_per_side_below_two(self):
        cfg = teach.load_teach_config(self.write_config())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = teach.run_teach(cfg, frames_per_side=1, parts=1, cameras=FakeCameraSet(camera_ids=("C1",)),
                                 input_fn=_Inputs(), sleep=lambda s: None, write=lambda s: None)
        self.assertEqual(rc, 2)
        self.assertIn("at least 2", err.getvalue())
        self.assertFalse(self.out.exists())

    def test_missing_output_path_exits_2(self):
        path = self.write_config(with_path=False)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = cli.main(["teach", "--config", str(path)])
        self.assertEqual(rc, 2)
        # the config itself requires model.path for the golden kind
        self.assertIn("model.path", err.getvalue())

    def test_refuses_to_overwrite_without_force(self):
        self.out.write_bytes(b"old model")
        cfg = teach.load_teach_config(self.write_config())
        cameras = FakeCameraSet(camera_ids=("C1",))
        inputs = _Inputs()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = teach.run_teach(cfg, frames_per_side=2, parts=1, cameras=cameras, input_fn=inputs,
                                 sleep=lambda s: None, write=lambda s: None)
        self.assertEqual(rc, 2)
        self.assertIn("--force", err.getvalue())
        self.assertEqual(self.out.read_bytes(), b"old model")
        self.assertEqual(inputs.calls, 0)
        self.assertEqual(cameras.capture_counts, {Side.TOP: 0, Side.BOTTOM: 0})

    def test_help_still_works(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("teach", out.getvalue())

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["teach", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("--frames-per-side", out.getvalue())


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TeachRunTests(_TeachCase):
    PROMPT_TOP = "Part {k} of {n}: place a GOOD part with the TOP side up, keep it still, then press Enter"
    PROMPT_BOTTOM = "Part {k} of {n}: flip it to the BOTTOM side, keep it still, then press Enter"

    def _run(self, *, parts=2, frames=3, cameras=None, inputs=None, force=False):
        cfg = teach.load_teach_config(self.write_config())
        lines, sleeps = [], []
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = teach.run_teach(
                cfg,
                frames_per_side=frames,
                parts=parts,
                force=force,
                cameras=cameras if cameras is not None else FakeCameraSet(camera_ids=("C1",)),
                builder=golden.GoldenBuilder(_decode),
                input_fn=inputs if inputs is not None else _Inputs(),
                sleep=sleeps.append,
                write=lines.append,
            )
        return rc, lines, sleeps, err.getvalue()

    def test_full_run_writes_loadable_model_with_frame_counts(self):
        cameras = FakeCameraSet(camera_ids=("C1",))
        rc, lines, sleeps, _ = self._run(parts=2, frames=3, cameras=cameras)
        self.assertEqual(rc, 0)
        self.assertTrue(self.out.exists())

        model = golden.GoldenModel.load(self.out, _decode)
        self.assertEqual(model.version, "golden-test")
        keys = model.describe()["keys"]
        self.assertEqual(keys["C1/TOP"]["frames"], 2 * 3)
        self.assertEqual(keys["C1/BOTTOM"]["frames"], 2 * 3)
        self.assertEqual(cameras.capture_counts, {Side.TOP: 6, Side.BOTTOM: 6})

        self.assertIn("  C1/TOP: 6 frames", lines)
        self.assertIn("  C1/BOTTOM: 6 frames", lines)
        # 2 parts x 2 sides x 3 rounds = 12 rounds, a pause between each pair of rounds.
        self.assertEqual(sleeps, [teach.DEFAULT_PAUSE_S] * 11)

    def test_prompts_appear_in_order_top_then_bottom_per_part(self):
        rc, lines, _, _ = self._run(parts=2, frames=2)
        self.assertEqual(rc, 0)
        prompts = [line for line in lines if line.startswith("Part ")]
        self.assertEqual(prompts, [
            self.PROMPT_TOP.format(k=1, n=2),
            self.PROMPT_BOTTOM.format(k=1, n=2),
            self.PROMPT_TOP.format(k=2, n=2),
            self.PROMPT_BOTTOM.format(k=2, n=2),
        ])

    def test_one_input_prompt_per_side_per_part(self):
        inputs = _Inputs()
        rc, _, _, _ = self._run(parts=2, frames=2, inputs=inputs)
        self.assertEqual(rc, 0)
        self.assertEqual(inputs.calls, 4)

    def test_summary_contains_the_reminder(self):
        rc, lines, _, _ = self._run(parts=1, frames=2)
        self.assertEqual(rc, 0)
        self.assertIn(teach.REMINDER, lines)
        self.assertIn("Learned from GOOD parts only. This is a demo heuristic for a proof of concept: "
                      "scores are not validated and thresholds still have to be chosen from labelled data.",
                      lines)

    def test_force_overwrites_existing_file(self):
        self.out.write_bytes(b"old model")
        rc, _, _, _ = self._run(parts=1, frames=2, force=True)
        self.assertEqual(rc, 0)
        self.assertNotEqual(self.out.read_bytes(), b"old model")
        self.assertEqual(golden.GoldenModel.load(self.out, _decode).version, "golden-test")

    def test_camera_failure_exits_1_and_writes_nothing(self):
        cameras = FakeCameraSet(fail_sides=[Side.BOTTOM], camera_ids=("C1",))
        rc, _, _, err = self._run(parts=1, frames=2, cameras=cameras)
        self.assertEqual(rc, 1)
        self.assertIn("teach error:", err)
        self.assertFalse(self.out.exists())

    def test_cameras_are_released_after_a_failure(self):
        cameras = FakeCameraSet(fail_sides=[Side.TOP], camera_ids=("C1",))
        released = []
        cameras.release = lambda: released.append(True)
        rc, _, _, _ = self._run(parts=1, frames=2, cameras=cameras)
        self.assertEqual(rc, 1)
        self.assertEqual(released, [True])

    def test_prompt_abort_exits_1_and_writes_nothing(self):
        for exc in (EOFError, KeyboardInterrupt):
            with self.subTest(exc=exc.__name__):
                inputs = _Inputs(fail_at=2, exc=exc)
                rc, _, _, err = self._run(parts=2, frames=2, inputs=inputs)
                self.assertEqual(rc, 1)
                self.assertIn("teach aborted: nothing was written", err)
                self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
