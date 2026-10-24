"""Tests for tools/capture.py. A fake cv2 module stands in for OpenCV."""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is required only for these tests
    np = None

from tools import ToolDependencyError
from tools import capture as capture_mod
from tools.capture import CaptureError, capture_session, main

PLACEMENT_LINE = "Remove the part, put it back in the fixture, then press Enter"


def _fourcc(*chars: str) -> int:
    value = 0
    for i, c in enumerate(chars):
        value |= ord(c) << (8 * i)
    return value


class FakeCapture:
    """One fake camera.

    Profile keys: ignore (props whose set is dropped), reject ({prop: values} refused
    without change), raise_set (props whose set raises), fail_open, fail_grab_at,
    fail_retrieve_at, image_follows_exposure (brightness follows the exposure prop;
    otherwise the picture ignores exposure while read-back still echoes the value),
    black (every frame is all zeros).
    """

    def __init__(self, cv2: "FakeCv2", index: int, api: object) -> None:
        self._cv2 = cv2
        self.index = index
        self.api = api
        profile = cv2.profiles.get(index, {})
        self.ignore = set(profile.get("ignore", ()))
        self.reject = {prop: set(vals) for prop, vals in profile.get("reject", {}).items()}
        self.raise_set = set(profile.get("raise_set", ()))
        self.fail_open = profile.get("fail_open", False)
        self.fail_grab_at = profile.get("fail_grab_at")
        self.fail_retrieve_at = profile.get("fail_retrieve_at")
        self.image_follows_exposure = profile.get("image_follows_exposure", False)
        self.black = bool(profile.get("black", False))
        c = cv2
        self.props = {
            c.CAP_PROP_AUTOFOCUS: 1.0,
            c.CAP_PROP_AUTO_EXPOSURE: 0.75,
            c.CAP_PROP_AUTO_WB: 1.0,
            c.CAP_PROP_FOCUS: 0.5,
            c.CAP_PROP_EXPOSURE: -5.0,
            c.CAP_PROP_WB_TEMPERATURE: 4500.0,
            c.CAP_PROP_GAIN: 0.0,
            c.CAP_PROP_FPS: 30.0,
            c.CAP_PROP_FRAME_WIDTH: 640.0,
            c.CAP_PROP_FRAME_HEIGHT: 480.0,
            c.CAP_PROP_FOURCC: 0.0,
            c.CAP_PROP_BUFFERSIZE: 4.0,
        }
        self.grab_calls = 0
        self.retrieve_calls = 0
        self.released = False

    def isOpened(self) -> bool:
        return not self.fail_open

    def set(self, prop: int, value: float) -> bool:
        self._cv2.set_log.append((self.index, prop, float(value)))
        if prop in self.raise_set:
            raise RuntimeError("driver refused the property")
        if prop in self.ignore or float(value) in self.reject.get(prop, ()):
            return False
        if prop in (self._cv2.CAP_PROP_FRAME_WIDTH, self._cv2.CAP_PROP_FRAME_HEIGHT):
            mjpg = self.props[self._cv2.CAP_PROP_FOURCC] == float(self._cv2.MJPG)
            limit = 640 if prop == self._cv2.CAP_PROP_FRAME_WIDTH else 480
            if not mjpg and value > limit:
                value = limit  # without MJPG this fake camera only does VGA
        self.props[prop] = float(value)
        return True

    def get(self, prop: int) -> float:
        return float(self.props.get(prop, 0.0))

    def grab(self) -> bool:
        self._cv2.events.append(("grab", self.index))
        call = self.grab_calls
        self.grab_calls += 1
        return call != self.fail_grab_at

    def retrieve(self):
        self._cv2.events.append(("retrieve", self.index))
        n = self.retrieve_calls
        self.retrieve_calls += 1
        if n == self.fail_retrieve_at:
            return False, None
        w = int(self.props[self._cv2.CAP_PROP_FRAME_WIDTH])
        h = int(self.props[self._cv2.CAP_PROP_FRAME_HEIGHT])
        ys, xs = np.mgrid[0:h, 0:w]
        # A stationary scene: the picture does not change from frame to frame, so any
        # change in brightness in these tests comes from the exposure setting only.
        b = (xs * 3 + ys + self.index * 29) % 256
        g = (xs + ys * 2 + self.index * 5) % 256
        r = (xs * ys + self.index * 7) % 256
        img = np.stack([b, g, r], axis=-1)
        if self.image_follows_exposure:
            shift = int(round(self.props[self._cv2.CAP_PROP_EXPOSURE] * 10))
            img = np.clip(img.astype(np.int64) + shift, 0, 255)
        if self.black:
            img = np.zeros_like(img)
        return True, img.astype(np.uint8)

    def release(self) -> None:
        self.released = True


class FakeCv2:
    CAP_DSHOW = 700
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FOURCC = 6
    CAP_PROP_BUFFERSIZE = 38
    CAP_PROP_GAIN = 14
    CAP_PROP_EXPOSURE = 15
    CAP_PROP_FOCUS = 28
    CAP_PROP_AUTO_EXPOSURE = 21
    CAP_PROP_AUTOFOCUS = 39
    CAP_PROP_AUTO_WB = 44
    CAP_PROP_WB_TEMPERATURE = 45
    COLOR_BGR2GRAY = 6
    MJPG = _fourcc("M", "J", "P", "G")

    def __init__(self, profiles: dict | None = None) -> None:
        self.profiles = profiles or {}
        self.set_log: list[tuple[int, int, float]] = []
        self.events: list[tuple] = []  # ("grab", dev) / ("retrieve", dev) / ("input",)
        self.released: list[int] = []
        self.created: list[FakeCapture] = []
        self.imwrites: list[str] = []

    def VideoCapture(self, index: int, api: object = None) -> FakeCapture:
        cap = FakeCapture(self, index, api)
        self.created.append(cap)
        return cap

    def VideoWriter_fourcc(self, *chars: str) -> int:
        return _fourcc(*chars)

    def cvtColor(self, img, code):
        assert code == self.COLOR_BGR2GRAY
        f = img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299
        return f.astype(np.uint8)

    def imwrite(self, path: str, img) -> bool:
        Path(path).write_bytes(b"\x89PNG fake " + repr(img.shape).encode())
        self.imwrites.append(Path(path).name)
        return True


@unittest.skipUnless(np is not None, "numpy is required for capture tests")
class CaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.lines: list[str] = []

    def run_capture(self, cv2: FakeCv2 | None = None, **overrides):
        cv2 = cv2 or FakeCv2()
        kwargs = dict(
            mode="stationary",
            devices=[0],
            out_dir=self.root / "sessions",
            name="s1",
            count=3,
            interval_s=0.0,
            warmup_s=0.0,
            width=64,
            height=48,
            backend="default",
            cv2_module=cv2,
            input_fn=lambda prompt="": "",
            sleep=lambda s: None,
            out=self.lines.append,
        )
        kwargs.update(overrides)
        return capture_session(**kwargs), cv2

    def session_dir(self, name: str = "s1") -> Path:
        return self.root / "sessions" / name

    def load_session(self, name: str = "s1") -> dict:
        return json.loads((self.session_dir(name) / "session.json").read_text(encoding="utf-8"))

    def cli_args(self, name: str, *extra: str) -> list[str]:
        return [
            "--mode", "stationary", "--devices", "0",
            "--out", str(self.root / "sessions"), "--name", name,
            "--warmup-s", "0", "--interval-s", "0", "--width", "64", "--height", "48",
            "--backend", "default", *extra,
        ]

    def input_logger(self, cv2: FakeCv2):
        def fn(prompt: str = "") -> str:
            cv2.events.append(("input",))
            return ""
        return fn

    def assert_schema(self, doc: dict) -> None:
        self.assertEqual(
            set(doc),
            {"schema", "created_utc", "mode", "session_name", "devices", "notes", "requested", "cameras", "frames"},
        )
        self.assertEqual(doc["schema"], 1)
        self.assertTrue(doc["created_utc"].endswith("Z"))
        self.assertIn(doc["mode"], ("stationary", "placement"))
        self.assertEqual(set(doc["notes"]), {"part", "lighting", "operator", "free"})
        self.assertEqual(set(doc["requested"]), {"width", "height", "fps", "fourcc", "manual", "flush_frames"})
        self.assertEqual(set(doc["requested"]["manual"]), {"focus", "exposure", "white_balance", "gain"})
        for cam in doc["cameras"].values():
            self.assertEqual(
                set(cam),
                {"actual_width", "actual_height", "actual_fps", "readback", "applied_ok", "warnings",
                 "buffersize_readback", "auto_exposure_attempts", "exposure_response"},
            )
            self.assertEqual(
                set(cam["readback"]),
                {"autofocus", "focus", "auto_exposure", "exposure", "auto_wb", "white_balance", "gain"},
            )
            self.assertEqual(
                set(cam["applied_ok"]),
                {"autofocus", "auto_exposure", "auto_wb", "focus", "exposure", "white_balance", "gain",
                 "exposure_responds"},
            )
            self.assertIsInstance(cam["warnings"], list)
            self.assertIsInstance(cam["auto_exposure_attempts"], list)
        for frame in doc["frames"]:
            self.assertEqual(set(frame), {"index", "device", "stamp", "mean", "laplacian_var"})

    def test_stationary_layout_and_npz_shapes(self) -> None:
        summary, _ = self.run_capture(count=4, sample_every=2)
        folder = self.session_dir()
        self.assertTrue((folder / "session.json").is_file())
        self.assertEqual(
            sorted(p.name for p in (folder / "samples").iterdir()),
            ["dev0_frame0.png", "dev0_frame2.png", "dev0_frame3.png"],
        )
        with np.load(folder / "frames_dev0.npz") as z:
            self.assertEqual(z["gray"].shape, (4, 48, 64))
            self.assertEqual(z["gray"].dtype, np.uint8)
            self.assertEqual(z["stamps"].shape, (4,))
            self.assertEqual(z["stamps"].dtype, np.float64)
            self.assertTrue(np.all(np.diff(z["stamps"]) >= 0))
            first_mean = float(z["gray"][0].mean())

        doc = self.load_session()
        self.assert_schema(doc)
        self.assertEqual(doc["mode"], "stationary")
        self.assertEqual(doc["devices"], [0])
        self.assertEqual([f["index"] for f in doc["frames"]], [0, 1, 2, 3])
        self.assertAlmostEqual(doc["frames"][0]["mean"], first_mean, places=6)
        self.assertTrue(all(f["laplacian_var"] > 0 for f in doc["frames"]))
        self.assertEqual(doc["cameras"]["0"]["actual_width"], 64)
        self.assertEqual(doc["cameras"]["0"]["actual_height"], 48)
        self.assertEqual(summary["frames_saved"], 4)
        self.assertIsNone(summary["aborted"])

    def test_placement_mode_with_no_prompt(self) -> None:
        def fail_input(prompt: str = "") -> str:
            raise AssertionError("must not wait for input with --no-prompt")

        summary, _ = self.run_capture(mode="placement", count=3, no_prompt=True, input_fn=fail_input)
        doc = self.load_session()
        self.assert_schema(doc)
        self.assertEqual(doc["mode"], "placement")
        self.assertEqual(len(doc["frames"]), 3)
        with np.load(self.session_dir() / "frames_dev0.npz") as z:
            self.assertEqual(z["gray"].shape, (3, 48, 64))
        self.assertTrue(any(PLACEMENT_LINE in line for line in self.lines))
        self.assertEqual(summary["frames_saved"], 3)

    def test_placement_waits_for_enter_once_per_round(self) -> None:
        prompts: list[str] = []
        self.run_capture(mode="placement", count=3, no_prompt=False, input_fn=lambda p="": prompts.append(p) or "")
        self.assertEqual(len(prompts), 3)

    def test_ignored_autofocus_and_white_balance_are_read_back_and_warned(self) -> None:
        cv2 = FakeCv2(profiles={0: {"ignore": {FakeCv2.CAP_PROP_AUTOFOCUS, FakeCv2.CAP_PROP_AUTO_WB}}})
        summary, _ = self.run_capture(cv2=cv2, focus=0.2)
        doc = self.load_session()
        self.assert_schema(doc)
        cam = doc["cameras"]["0"]
        self.assertEqual(cam["readback"]["autofocus"], 1.0)
        self.assertFalse(cam["applied_ok"]["autofocus"])
        self.assertEqual(cam["readback"]["auto_wb"], 1.0)
        self.assertFalse(cam["applied_ok"]["auto_wb"])
        self.assertTrue(cam["applied_ok"]["auto_exposure"])
        self.assertEqual(cam["readback"]["focus"], 0.2)
        self.assertTrue(cam["applied_ok"]["focus"])
        self.assertIsNone(cam["applied_ok"]["exposure"])
        text = " | ".join(cam["warnings"])
        self.assertIn("autofocus", text)
        self.assertIn("auto white balance", text)
        self.assertTrue(any("autofocus" in w for w in summary["warnings"]))

    def test_auto_exposure_falls_back_from_0_25_to_1(self) -> None:
        cv2 = FakeCv2(profiles={0: {"reject": {FakeCv2.CAP_PROP_AUTO_EXPOSURE: {0.25}}}})
        self.run_capture(cv2=cv2)
        cam = self.load_session()["cameras"]["0"]
        self.assertEqual(cam["readback"]["auto_exposure"], 1.0)
        self.assertTrue(cam["applied_ok"]["auto_exposure"])
        self.assertFalse(any("auto exposure" in w for w in cam["warnings"]))

    def test_auto_exposure_both_values_rejected_is_warned(self) -> None:
        cv2 = FakeCv2(profiles={0: {"reject": {FakeCv2.CAP_PROP_AUTO_EXPOSURE: {0.25, 1.0}}}})
        summary, _ = self.run_capture(cv2=cv2)
        cam = self.load_session()["cameras"]["0"]
        self.assertFalse(cam["applied_ok"]["auto_exposure"])
        self.assertTrue(any("auto exposure" in w for w in cam["warnings"]))
        self.assertTrue(summary["warnings"])

    def test_auto_exposure_attempts_are_recorded(self) -> None:
        self.run_capture(count=1)
        cam = self.load_session()["cameras"]["0"]
        self.assertEqual(cam["auto_exposure_attempts"], [{"set": 0.25, "readback": 0.25}])

    def test_every_rejected_auto_exposure_candidate_is_recorded(self) -> None:
        cv2 = FakeCv2(profiles={0: {"reject": {FakeCv2.CAP_PROP_AUTO_EXPOSURE: {0.25}}}})
        self.run_capture(cv2=cv2, count=1)
        cam = self.load_session()["cameras"]["0"]
        self.assertEqual(
            cam["auto_exposure_attempts"],
            [{"set": 0.25, "readback": 0.75}, {"set": 1.0, "readback": 1.0}],
        )

    def test_ignored_requested_value_is_warned(self) -> None:
        cv2 = FakeCv2(profiles={0: {"ignore": {FakeCv2.CAP_PROP_FOCUS}}})
        self.run_capture(cv2=cv2, focus=0.2)
        cam = self.load_session()["cameras"]["0"]
        self.assertFalse(cam["applied_ok"]["focus"])
        self.assertEqual(cam["readback"]["focus"], 0.5)
        self.assertTrue(any(w.startswith("focus:") for w in cam["warnings"]))

    def test_fourcc_is_set_before_width_and_height(self) -> None:
        cv2 = FakeCv2()
        self.run_capture(cv2=cv2, fourcc="MJPG", width=800, height=600, count=1)
        order = [prop for (index, prop, _) in cv2.set_log if index == 0]
        self.assertIn((0, FakeCv2.CAP_PROP_FOURCC, float(FakeCv2.MJPG)), cv2.set_log)
        self.assertLess(order.index(FakeCv2.CAP_PROP_FOURCC), order.index(FakeCv2.CAP_PROP_FRAME_WIDTH))
        self.assertLess(order.index(FakeCv2.CAP_PROP_FOURCC), order.index(FakeCv2.CAP_PROP_FRAME_HEIGHT))
        doc = self.load_session()
        self.assertEqual(doc["requested"]["fourcc"], "MJPG")
        cam = doc["cameras"]["0"]
        self.assertEqual((cam["actual_width"], cam["actual_height"]), (800, 600))
        self.assertFalse(any("resolution" in w for w in cam["warnings"]))

    def test_resolution_ignored_without_mjpg_is_warned(self) -> None:
        self.run_capture(width=800, height=600, count=1)
        cam = self.load_session()["cameras"]["0"]
        self.assertEqual((cam["actual_width"], cam["actual_height"]), (640, 480))
        self.assertTrue(any("resolution" in w for w in cam["warnings"]))

    def test_refuses_to_overwrite_existing_folder(self) -> None:
        self.session_dir().mkdir(parents=True)
        marker = self.session_dir() / "keep.txt"
        marker.write_text("original", encoding="utf-8")
        cv2 = FakeCv2()
        with self.assertRaises(CaptureError) as ctx:
            self.run_capture(cv2=cv2)
        self.assertIn(str(self.session_dir()), str(ctx.exception))
        self.assertEqual(cv2.created, [])
        self.assertEqual(marker.read_text(encoding="utf-8"), "original")

    def test_device_failure_keeps_partial_data_and_notes_aborted(self) -> None:
        cv2 = FakeCv2(profiles={1: {"fail_grab_at": 2}})
        # flush_frames=0 so that grab call 2 is round 2's real grab, as the test intends.
        summary, _ = self.run_capture(cv2=cv2, devices=[0, 1], count=5, flush_frames=0)
        self.assertIsNotNone(summary["aborted"])
        self.assertIn("device 1", summary["aborted"])
        doc = self.load_session()
        self.assert_schema(doc)
        self.assertIn("aborted", doc["notes"]["free"])
        self.assertEqual(len(doc["frames"]), 4)  # two complete rounds on two devices
        for dev in (0, 1):
            with np.load(self.session_dir() / f"frames_dev{dev}.npz") as z:
                self.assertEqual(z["gray"].shape, (2, 48, 64))
                self.assertEqual(z["stamps"].shape, (2,))
        self.assertTrue(all(c.released for c in cv2.created))
        self.assertEqual(summary["frames_saved"], 4)

    def test_two_devices(self) -> None:
        cv2 = FakeCv2()
        summary, _ = self.run_capture(cv2=cv2, devices=[0, 2], count=2)
        folder = self.session_dir()
        for dev in (0, 2):
            with np.load(folder / f"frames_dev{dev}.npz") as z:
                self.assertEqual(z["gray"].shape, (2, 48, 64))
        self.assertEqual(
            sorted(p.name for p in (folder / "samples").iterdir()),
            ["dev0_frame0.png", "dev0_frame1.png", "dev2_frame0.png", "dev2_frame1.png"],
        )
        doc = self.load_session()
        self.assert_schema(doc)
        self.assertEqual(doc["devices"], [0, 2])
        self.assertEqual(set(doc["cameras"]), {"0", "2"})
        self.assertEqual(len(doc["frames"]), 4)
        self.assertEqual(summary["frames_per_device"], {"0": 2, "2": 2})
        self.assertTrue(all(c.released for c in cv2.created))

    def test_dshow_backend_passes_cap_dshow(self) -> None:
        cv2 = FakeCv2()
        self.run_capture(cv2=cv2, backend="dshow", count=1)
        self.assertEqual(cv2.created[0].api, FakeCv2.CAP_DSHOW)

    def test_default_backend_follows_platform(self) -> None:
        cv2 = FakeCv2()
        with mock.patch.object(capture_mod, "_is_windows", return_value=True):
            self.run_capture(cv2=cv2, backend=None, count=1, name="win")
        self.assertEqual(cv2.created[-1].api, FakeCv2.CAP_DSHOW)
        cv2 = FakeCv2()
        with mock.patch.object(capture_mod, "_is_windows", return_value=False):
            self.run_capture(cv2=cv2, backend=None, count=1, name="other")
        self.assertIsNone(cv2.created[-1].api)

    def test_cli_main_writes_session_and_prints_next_command(self) -> None:
        lines: list[str] = []
        code = main(
            [
                "--mode", "stationary", "--devices", "0",
                "--out", str(self.root / "sessions"), "--name", "cli1",
                "--count", "2", "--interval-s", "0", "--warmup-s", "0",
                "--width", "64", "--height", "48", "--backend", "default",
            ],
            cv2_module=FakeCv2(),
            out=lines.append,
        )
        self.assertEqual(code, 0)
        self.assertTrue((self.root / "sessions" / "cli1" / "session.json").is_file())
        self.assertTrue(any("python -m tools.report" in line for line in lines))

    def test_cli_main_returns_error_code_when_folder_exists(self) -> None:
        self.session_dir("cli2").mkdir(parents=True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(
                ["--mode", "stationary", "--devices", "0", "--out", str(self.root / "sessions"),
                 "--name", "cli2", "--count", "1", "--warmup-s", "0", "--width", "64", "--height", "48"],
                cv2_module=FakeCv2(),
                out=self.lines.append,
            )
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err.getvalue())

    def test_missing_cv2_raises_tool_dependency_error(self) -> None:
        with mock.patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaises(ToolDependencyError) as ctx:
                capture_session(
                    mode="stationary", devices=[0], out_dir=self.root / "sessions", name="s1",
                    count=1, interval_s=0.0, warmup_s=0.0, backend="default",
                )
        self.assertIn("pip install opencv-python numpy", str(ctx.exception))
        self.assertFalse(self.session_dir().exists())

    def test_session_json_matches_format_keys(self) -> None:
        self.run_capture(count=2)
        doc = self.load_session()
        self.assert_schema(doc)
        self.assertEqual(
            set(doc["frames"][0]),
            {"index", "device", "stamp", "mean", "laplacian_var"},
        )

    # --- stale frames: flushing ------------------------------------------------

    def test_flush_grabs_happen_before_each_placement_capture(self) -> None:
        cv2 = FakeCv2()
        self.run_capture(cv2=cv2, mode="placement", count=2, no_prompt=False,
                         input_fn=self.input_logger(cv2), flush_frames=3)
        expected = []
        for _ in range(2):
            # Enter, then 3 discarded grabs and the real grab, then retrieve.
            expected += [("input",)] + [("grab", 0)] * 4 + [("retrieve", 0)]
        self.assertEqual(cv2.events, expected)

    def test_flush_grabs_happen_before_each_stationary_round_on_every_device(self) -> None:
        cv2 = FakeCv2()
        self.run_capture(cv2=cv2, devices=[0, 1], count=2, flush_frames=2)
        # Per round: 2 discarded grabs per device, then the real grab per device, then retrieves.
        one_round = [("grab", 0), ("grab", 0), ("grab", 1), ("grab", 1),
                     ("grab", 0), ("grab", 1), ("retrieve", 0), ("retrieve", 1)]
        self.assertEqual(cv2.events, one_round * 2)

    def test_flush_frames_zero_disables_flushing(self) -> None:
        cv2 = FakeCv2()
        self.run_capture(cv2=cv2, devices=[0], count=2, flush_frames=0)
        self.assertEqual(cv2.events, [("grab", 0), ("retrieve", 0)] * 2)
        self.assertEqual(self.load_session()["requested"]["flush_frames"], 0)

    def test_flush_frames_default_is_five_and_recorded(self) -> None:
        cv2 = FakeCv2()
        self.run_capture(cv2=cv2, count=1)
        self.assertEqual(cv2.events, [("grab", 0)] * 6 + [("retrieve", 0)])
        self.assertEqual(self.load_session()["requested"]["flush_frames"], 5)

    def test_negative_flush_frames_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.run_capture(flush_frames=-1)
        self.assertFalse(self.session_dir().exists())

    def test_cli_flush_frames_zero_disables_flushing(self) -> None:
        cv2 = FakeCv2()
        code = main(self.cli_args("cli-flush", "--count", "2", "--flush-frames", "0"),
                    cv2_module=cv2, out=self.lines.append)
        self.assertEqual(code, 0)
        self.assertEqual(cv2.events, [("grab", 0), ("retrieve", 0)] * 2)

    # --- buffer size -----------------------------------------------------------

    def test_buffersize_is_set_to_one_and_read_back_recorded(self) -> None:
        cv2 = FakeCv2()
        self.run_capture(cv2=cv2, count=1)
        self.assertIn((0, FakeCv2.CAP_PROP_BUFFERSIZE, 1.0), cv2.set_log)
        cam = self.load_session()["cameras"]["0"]
        self.assertEqual(cam["buffersize_readback"], 1.0)
        self.assertFalse(any("buffer size" in w for w in cam["warnings"]))

    def test_buffersize_not_taken_is_recorded_and_warned(self) -> None:
        cv2 = FakeCv2(profiles={0: {"ignore": {FakeCv2.CAP_PROP_BUFFERSIZE}}})
        summary, _ = self.run_capture(cv2=cv2, count=1)
        cam = self.load_session()["cameras"]["0"]
        self.assertEqual(cam["buffersize_readback"], 4.0)
        self.assertTrue(any("buffer size" in w for w in cam["warnings"]))
        self.assertIsNone(summary["aborted"])

    def test_buffersize_set_failure_is_ignored(self) -> None:
        cv2 = FakeCv2(profiles={0: {"raise_set": {FakeCv2.CAP_PROP_BUFFERSIZE}}})
        summary, _ = self.run_capture(cv2=cv2, count=1)
        self.assertIsNone(summary["aborted"])
        cam = self.load_session()["cameras"]["0"]
        self.assertEqual(cam["buffersize_readback"], 4.0)
        self.assertTrue(any("buffer size" in w for w in cam["warnings"]))

    # --- exposure response -----------------------------------------------------

    def test_exposure_that_changes_the_picture_is_responding(self) -> None:
        cv2 = FakeCv2(profiles={0: {"image_follows_exposure": True}})
        self.run_capture(cv2=cv2, count=1, exposure=-4.0)
        cam = self.load_session()["cameras"]["0"]
        self.assertIs(cam["applied_ok"]["exposure_responds"], True)
        self.assertGreater(cam["exposure_response"]["delta"], capture_mod.RESPONSE_EPS_GRAY)
        self.assertFalse(any("exposure" in w for w in cam["warnings"]))
        # The requested exposure is restored after the check.
        self.assertEqual(cv2.created[0].props[FakeCv2.CAP_PROP_EXPOSURE], -4.0)

    def test_exposure_that_does_not_change_the_picture_is_flagged(self) -> None:
        # Default fake: read-back echoes the set value, but the picture ignores it.
        summary, _ = self.run_capture(count=1, exposure=-4.0)
        cam = self.load_session()["cameras"]["0"]
        self.assertEqual(cam["readback"]["exposure"], -4.0)
        self.assertIs(cam["applied_ok"]["exposure_responds"], False)
        text = " | ".join(cam["warnings"])
        self.assertIn("exposure changes had no visible effect", text)
        self.assertIn("DirectShow read-back can echo the value you set", text)
        self.assertTrue(any("exposure changes had no visible effect" in w for w in summary["warnings"]))

    def test_response_check_skipped_without_exposure(self) -> None:
        cv2 = FakeCv2(profiles={0: {"image_follows_exposure": True}})
        self.run_capture(cv2=cv2, count=1)
        cam = self.load_session()["cameras"]["0"]
        self.assertIsNone(cam["applied_ok"]["exposure_responds"])
        self.assertIsNone(cam["exposure_response"])
        self.assertFalse(any("exposure" in w for w in cam["warnings"]))
        # No extra grabs: only the flush and the real grab for the one frame.
        self.assertEqual(cv2.events, [("grab", 0)] * 6 + [("retrieve", 0)])

    def test_response_check_skipped_by_flag_is_noted(self) -> None:
        cv2 = FakeCv2(profiles={0: {"image_follows_exposure": True}})
        self.run_capture(cv2=cv2, count=1, exposure=-4.0, response_check=False)
        cam = self.load_session()["cameras"]["0"]
        self.assertIsNone(cam["applied_ok"]["exposure_responds"])
        self.assertIsNone(cam["exposure_response"])
        self.assertTrue(any("skipped" in w and "DirectShow read-back can echo" in w for w in cam["warnings"]))
        self.assertEqual(cv2.events, [("grab", 0)] * 6 + [("retrieve", 0)])

    def test_cli_no_response_check_flag_skips_the_check(self) -> None:
        cv2 = FakeCv2(profiles={0: {"image_follows_exposure": True}})
        code = main(self.cli_args("cli-resp", "--count", "1", "--exposure", "-4", "--no-response-check"),
                    cv2_module=cv2, out=self.lines.append)
        self.assertEqual(code, 0)
        cam = self.load_session("cli-resp")["cameras"]["0"]
        self.assertIsNone(cam["applied_ok"]["exposure_responds"])

    def test_exposure_restore_is_read_back_flushed_and_detail_has_eps(self) -> None:
        cv2 = FakeCv2(profiles={0: {"image_follows_exposure": True}})
        self.run_capture(cv2=cv2, count=1, exposure=-4.0, flush_frames=2)
        cam = self.load_session()["cameras"]["0"]
        detail = cam["exposure_response"]
        self.assertEqual(detail["eps"], capture_mod.RESPONSE_EPS_GRAY)
        self.assertTrue(detail["restore_ok"])
        self.assertEqual(detail["restore_readback"], -4.0)
        self.assertFalse(any("did not take" in w for w in cam["warnings"]))
        # base: flush + 3 frames; alt: flush + 3 frames; restore: flush; real capture: flush + 1.
        mean = [("grab", 0)] * 2 + [("grab", 0), ("retrieve", 0)] * 3
        real = [("grab", 0)] * 2 + [("grab", 0), ("retrieve", 0)]
        self.assertEqual(cv2.events, mean + mean + [("grab", 0)] * 2 + real)
        self.assertEqual(cv2.set_log[-1], (0, FakeCv2.CAP_PROP_EXPOSURE, -4.0))
        self.assertEqual(cv2.created[0].props[FakeCv2.CAP_PROP_EXPOSURE], -4.0)

    def test_exposure_restore_that_does_not_take_is_warned(self) -> None:
        # The driver refuses the set back to -4, so the picture stays at the changed -8.
        cv2 = FakeCv2(profiles={0: {"reject": {FakeCv2.CAP_PROP_EXPOSURE: {-4.0}},
                                    "image_follows_exposure": True}})
        summary, _ = self.run_capture(cv2=cv2, count=1, exposure=-4.0, flush_frames=0)
        cam = self.load_session()["cameras"]["0"]
        self.assertIs(cam["exposure_response"]["restore_ok"], False)
        self.assertEqual(cam["exposure_response"]["restore_readback"], -8.0)
        self.assertTrue(any("did not take" in w for w in cam["warnings"]))
        self.assertTrue(any("did not take" in w for w in summary["warnings"]))

    def test_exposure_is_restored_even_when_a_grab_fails(self) -> None:
        # flush_frames=0: grab calls 0-2 are the base frames, 3 is the alt frame, 4 fails.
        cv2 = FakeCv2(profiles={0: {"image_follows_exposure": True, "fail_grab_at": 4}})
        summary, _ = self.run_capture(cv2=cv2, count=1, exposure=-4.0, flush_frames=0)
        self.assertIsNotNone(summary["aborted"])
        self.assertIn("grab failed", summary["aborted"])
        self.assertEqual(cv2.created[0].props[FakeCv2.CAP_PROP_EXPOSURE], -4.0)
        self.assertTrue(cv2.created[0].released)

    # --- session names and folder creation --------------------------------------

    def test_invalid_session_names_exit_2_with_a_message(self) -> None:
        bad = [
            "a<b", "a>b", "a:b", 'a"b', "a/b", "a\\b", "a|b", "a?b", "a*b",
            "a\tb", "a\x01b", "trailing.", "trailing ", "CON", "nul", "com1",
            "LPT9.txt", "Aux.log", "COM¹", "", "   ", ".", "..",
        ]
        for name in bad:
            with self.subTest(name=name):
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    code = main(self.cli_args(name), cv2_module=FakeCv2(), out=self.lines.append)
                self.assertEqual(code, 2)
                self.assertIn("error:", err.getvalue())
                self.assertIn("session name", err.getvalue())
                self.assertFalse((self.root / "sessions").exists())

    def test_invalid_session_name_is_a_capture_error_too(self) -> None:
        with self.assertRaises(CaptureError) as ctx:
            self.run_capture(name="CON.txt")
        self.assertIn("reserved", str(ctx.exception))

    def test_valid_session_names_are_accepted(self) -> None:
        for name in ["line-1_2026", "Run 2 (left)", "CONSOLE", "com10", "a.b"]:
            with self.subTest(name=name):
                self.run_capture(name=name, count=1)
                self.assertTrue((self.session_dir(name) / "session.json").is_file())

    def test_folder_creation_failure_is_a_capture_error_not_a_traceback(self) -> None:
        blocker = self.root / "blocker"
        blocker.write_text("a file, not a folder", encoding="utf-8")
        with self.assertRaises(CaptureError) as ctx:
            self.run_capture(out_dir=blocker / "sessions")
        self.assertIn(str(blocker / "sessions" / "s1"), str(ctx.exception))

    def test_cli_folder_creation_failure_exits_1_without_traceback(self) -> None:
        blocker = self.root / "blocker"
        blocker.write_text("a file, not a folder", encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(self.cli_args("s9", "--out", str(blocker / "sessions")),
                        cv2_module=FakeCv2(), out=self.lines.append)
        self.assertEqual(code, 1)
        self.assertIn("could not create session folder", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())


@unittest.skipUnless(np is not None, "numpy is required for capture tests")
class ListDevicesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lines: list[str] = []

    def found_lines(self) -> list[tuple[str, str, str]]:
        return re.findall(r"^index (\d+): (\d+)x(\d+)", "\n".join(self.lines), re.M)

    def test_lists_exactly_the_indexes_that_open_with_sizes(self) -> None:
        cv2 = FakeCv2(profiles={1: {"fail_open": True}})
        code = main(["--list-devices", "--max-index", "2", "--backend", "default"],
                    cv2_module=cv2, out=self.lines.append)
        self.assertEqual(code, 0)
        self.assertEqual(self.found_lines(), [("0", "640", "480"), ("2", "640", "480")])
        self.assertIn("Found 2 camera(s)", "\n".join(self.lines))

    def test_black_frame_is_reported_as_all_black(self) -> None:
        cv2 = FakeCv2(profiles={2: {"black": True}})
        main(["--list-devices", "--max-index", "2", "--backend", "default"],
             cv2_module=cv2, out=self.lines.append)
        self.assertIn("index 0: 640x480, not all black", self.lines)
        self.assertIn("index 2: 640x480, all black", self.lines)

    def test_none_open_exits_1_with_hint(self) -> None:
        cv2 = FakeCv2(profiles={0: {"fail_open": True}, 1: {"fail_open": True}})
        code = main(["--list-devices", "--max-index", "1"], cv2_module=cv2, out=self.lines.append)
        self.assertEqual(code, 1)
        text = "\n".join(self.lines)
        self.assertIn("No camera opened at indexes 0..1", text)
        self.assertIn("Close other apps", text)
        self.assertIn("privacy", text.lower())
        self.assertIn("--backend default", text)

    def test_max_index_is_respected_and_defaults_to_five(self) -> None:
        cv2 = FakeCv2()
        main(["--list-devices", "--max-index", "3", "--backend", "default"],
             cv2_module=cv2, out=self.lines.append)
        self.assertEqual([c.index for c in cv2.created], [0, 1, 2, 3])
        cv2 = FakeCv2()
        main(["--list-devices", "--backend", "default"], cv2_module=cv2, out=self.lines.append)
        self.assertEqual([c.index for c in cv2.created], [0, 1, 2, 3, 4, 5])

    def test_negative_max_index_exits_2(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["--list-devices", "--max-index", "-1"], cv2_module=FakeCv2(), out=self.lines.append)
        self.assertEqual(code, 2)
        self.assertIn("max-index", err.getvalue())

    def test_every_opened_camera_is_released(self) -> None:
        cv2 = FakeCv2(profiles={1: {"fail_open": True}, 2: {"black": True}})
        main(["--list-devices", "--max-index", "2", "--backend", "default"],
             cv2_module=cv2, out=self.lines.append)
        self.assertEqual(len(cv2.created), 3)
        self.assertTrue(all(c.released for c in cv2.created))

    def test_capture_args_not_required_for_list_but_required_otherwise(self) -> None:
        cv2 = FakeCv2()
        code = main(["--list-devices", "--max-index", "0", "--backend", "default"],
                    cv2_module=cv2, out=self.lines.append)
        self.assertEqual(code, 0)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as ctx:
            main(["--devices", "0", "--name", "x"], cv2_module=cv2, out=self.lines.append)
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--mode", err.getvalue())
        self.assertEqual(len(cv2.created), 1)  # only the list run opened a camera

    def test_missing_cv2_gives_dependency_message(self) -> None:
        err = io.StringIO()
        with mock.patch.dict(sys.modules, {"cv2": None}), contextlib.redirect_stderr(err):
            code = main(["--list-devices"], out=self.lines.append)
        self.assertEqual(code, 2)
        self.assertIn("pip install opencv-python numpy", err.getvalue())


class DependencyTests(unittest.TestCase):
    def test_missing_numpy_raises_tool_dependency_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sys.modules, {"numpy": None}):
                with self.assertRaises(ToolDependencyError) as ctx:
                    capture_session(
                        mode="stationary", devices=[0], out_dir=tmp, name="s1",
                        count=1, backend="default", cv2_module=object(),
                    )
            self.assertIn("numpy", str(ctx.exception))
            self.assertIn("pip install opencv-python numpy", str(ctx.exception))
            self.assertFalse((Path(tmp) / "s1").exists())


if __name__ == "__main__":
    unittest.main()
