import re
import sys
import threading
import time
import unittest
from unittest import mock

from station.cameras import (
    AUTO_EXPOSURE_MANUAL,
    AUTO_WB_OFF,
    AUTOFOCUS_OFF,
    FakeCameraSet,
    OpenCvCameraSet,
    make_blur_check,
)
from station.contracts import CAMERA_IDS, CameraError, Frame, Side


class FakeCapture:
    def __init__(self, cv2, index, log):
        self.cv2 = cv2
        self.index = index
        self.log = log
        self.props = {}
        self.opened = True
        self.grab_ok = True
        self.retrieve_ok = True
        self.image = f"img{index}"

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value
        self.log.append(("set", self.index, prop, value))
        return True

    def grab(self):
        self.log.append(("grab", self.index))
        return self.grab_ok

    def retrieve(self):
        self.log.append(("retrieve", self.index))
        return (self.retrieve_ok, self.image if self.retrieve_ok else None)

    def release(self):
        self.log.append(("release", self.index))


class FakeBuf:
    def __init__(self, data):
        self._data = data

    def tobytes(self):
        return self._data


class FakeImage:
    """Stands in for a numpy frame: it has a shape and a printable name."""

    def __init__(self, name, width, height):
        self.name = name
        self.shape = (height, width, 3)

    def __str__(self):
        return self.name


class FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_AUTOFOCUS = 39
    CAP_PROP_AUTO_EXPOSURE = 21
    CAP_PROP_AUTO_WB = 44
    CAP_PROP_FOCUS = 28
    CAP_PROP_EXPOSURE = 15
    CAP_PROP_WB_TEMPERATURE = 45
    CAP_PROP_BUFFERSIZE = 38
    IMWRITE_JPEG_QUALITY = 1
    INTER_AREA = 3

    def __init__(self):
        self.log = []
        self.captures = []
        self.encode_ok = True
        self.fail_encode_for = None  # image string that fails encoding
        self.encode_params = []  # (ext, params) per imencode call
        self.resizes = []  # (image name, dsize, interpolation) per resize call

    def VideoCapture(self, index):
        cap = FakeCapture(self, index, self.log)
        self.captures.append(cap)
        return cap

    def imencode(self, ext, image, params=None):
        self.log.append(("imencode", image))
        self.encode_params.append((ext, params))
        if not self.encode_ok or str(image) == self.fail_encode_for:
            return (False, None)
        label = ext.lstrip(".").upper()
        return (True, FakeBuf(f"{label}:{image}".encode("ascii")))

    def resize(self, image, dsize, interpolation=None):
        self.log.append(("resize", str(image)))
        self.resizes.append((str(image), dsize, interpolation))
        width, height = dsize
        return FakeImage(f"{image}>resized", width, height)


def _fixed_clock():
    ticks = iter(f"2026-10-23T08:30:{s:02d}Z" for s in range(100))
    return lambda: next(ticks)


class FakeCameraSetTests(unittest.TestCase):
    def test_capture_returns_four_frames_in_camera_order(self):
        cams = FakeCameraSet(clock=lambda: "2026-10-23T08:30:12Z")
        frames = cams.capture(Side.TOP)
        self.assertEqual([f.camera_id for f in frames], list(CAMERA_IDS))
        self.assertEqual([f.seq for f in frames], [0, 1, 2, 3])
        for f in frames:
            self.assertIsInstance(f, Frame)
            self.assertEqual(f.side, Side.TOP)
            self.assertEqual(f.ext, "png")
            self.assertEqual(f.timestamp_utc, "2026-10-23T08:30:12Z")

    def test_default_data_is_deterministic_and_counts_calls(self):
        cams = FakeCameraSet()
        first = cams.capture(Side.BOTTOM)
        second = cams.capture(Side.BOTTOM)
        self.assertEqual(first[0].data, b"FAKE:C1:BOTTOM:0")
        self.assertEqual(first[3].data, b"FAKE:C4:BOTTOM:0")
        self.assertEqual(second[1].data, b"FAKE:C2:BOTTOM:1")
        self.assertEqual(cams.capture_counts[Side.BOTTOM], 2)
        self.assertEqual(cams.capture_counts[Side.TOP], 0)

    def test_frame_bytes_callable_overrides_data(self):
        cams = FakeCameraSet(frame_bytes=lambda cam, side, n: f"{cam}-{side.value}-{n}".encode())
        frames = cams.capture("TOP")
        self.assertEqual(frames[2].data, b"C3-TOP-0")

    def test_failing_side_raises_camera_error_and_still_counts(self):
        cams = FakeCameraSet(fail_sides=[Side.TOP])
        with self.assertRaises(CameraError):
            cams.capture(Side.TOP)
        self.assertEqual(cams.capture_counts[Side.TOP], 1)
        self.assertEqual(len(cams.capture(Side.BOTTOM)), 4)

    def test_default_clock_is_iso_utc(self):
        frames = FakeCameraSet().capture(Side.TOP)
        self.assertRegex(frames[0].timestamp_utc, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_default_camera_ids_are_all_four(self):
        cams = FakeCameraSet()
        self.assertEqual(cams.camera_ids, CAMERA_IDS)
        self.assertEqual(len(cams.capture(Side.TOP)), 4)

    def test_single_camera_set_returns_one_frame_per_side(self):
        cams = FakeCameraSet(camera_ids=("C1",))
        self.assertEqual(cams.camera_ids, ("C1",))
        for side in (Side.TOP, Side.BOTTOM):
            frames = cams.capture(side)
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].camera_id, "C1")
            self.assertEqual(frames[0].seq, 0)
            self.assertEqual(frames[0].side, side)
        self.assertEqual(cams.capture_counts[Side.TOP], 1)

    def test_two_camera_set_uses_given_ids_in_order(self):
        frames = FakeCameraSet(camera_ids=("C2", "C4")).capture("BOTTOM")
        self.assertEqual([(f.camera_id, f.seq) for f in frames], [("C2", 0), ("C4", 1)])

    def test_empty_camera_ids_rejected(self):
        with self.assertRaises(ValueError):
            FakeCameraSet(camera_ids=())


class OpenCvCameraSetTests(unittest.TestCase):
    def test_open_sets_manual_modes_and_manual_values(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet(
            [0, 1, 2, 3],
            cv2_module=cv2,
            manual_settings={"focus": 120, "exposure": -6, "white_balance": 4500},
        )
        cams.open()
        self.assertEqual(len(cv2.captures), 4)
        for cap in cv2.captures:
            self.assertEqual(cap.props[FakeCv2.CAP_PROP_AUTOFOCUS], AUTOFOCUS_OFF)
            self.assertEqual(cap.props[FakeCv2.CAP_PROP_AUTO_EXPOSURE], AUTO_EXPOSURE_MANUAL)
            self.assertEqual(cap.props[FakeCv2.CAP_PROP_AUTO_WB], AUTO_WB_OFF)
            self.assertEqual(cap.props[FakeCv2.CAP_PROP_FOCUS], 120)
            self.assertEqual(cap.props[FakeCv2.CAP_PROP_EXPOSURE], -6)
            self.assertEqual(cap.props[FakeCv2.CAP_PROP_WB_TEMPERATURE], 4500)
            self.assertEqual(cap.props[FakeCv2.CAP_PROP_FRAME_WIDTH], 2560)
            self.assertEqual(cap.props[FakeCv2.CAP_PROP_FRAME_HEIGHT], 1440)

    def test_unset_manual_values_are_not_touched(self):
        cv2 = FakeCv2()
        OpenCvCameraSet([0, 1, 2, 3], cv2_module=cv2, manual_settings={"focus": 7}).open()
        for cap in cv2.captures:
            self.assertEqual(cap.props[FakeCv2.CAP_PROP_FOCUS], 7)
            self.assertNotIn(FakeCv2.CAP_PROP_EXPOSURE, cap.props)
            self.assertNotIn(FakeCv2.CAP_PROP_WB_TEMPERATURE, cap.props)
            # Auto modes are still forced off even with no manual values.
            self.assertIn(FakeCv2.CAP_PROP_AUTOFOCUS, cap.props)

    def test_unknown_manual_setting_rejected(self):
        with self.assertRaises(ValueError):
            OpenCvCameraSet([0, 1, 2, 3], cv2_module=FakeCv2(), manual_settings={"gain": 1})

    def test_device_count_outside_1_to_4_rejected(self):
        for indexes in ([], [0, 1, 2, 3, 4]):
            with self.subTest(count=len(indexes)):
                with self.assertRaises(CameraError) as ctx:
                    OpenCvCameraSet(indexes, cv2_module=FakeCv2())
                self.assertIn("1 to 4", str(ctx.exception))

    def test_one_two_and_four_devices_name_the_right_ids(self):
        for indexes, expected in (
            ([5], ("C1",)),
            ([5, 6], ("C1", "C2")),
            ([0, 1, 2, 3], ("C1", "C2", "C3", "C4")),
        ):
            with self.subTest(count=len(indexes)):
                cv2 = FakeCv2()
                # Warm-up and stale flush off: this test checks one grab per device, in order.
                cams = OpenCvCameraSet(
                    indexes,
                    cv2_module=cv2,
                    now_utc=_fixed_clock(),
                    open_warmup_frames=0,
                    stale_flush_frames=0,
                )
                self.assertEqual(cams.camera_ids, expected)
                frames = cams.capture(Side.TOP)
                self.assertEqual([f.camera_id for f in frames], list(expected))
                self.assertEqual([f.seq for f in frames], list(range(len(expected))))
                self.assertEqual([c.index for c in cv2.captures], list(indexes))
                self.assertEqual([e[1] for e in cv2.log if e[0] == "grab"], list(indexes))
                cams.release()

    def test_single_device_failure_names_c1(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([7], cv2_module=cv2)
        cv2.VideoCapture = lambda index: _closed(cv2, index)
        with self.assertRaises(CameraError) as ctx:
            cams.capture(Side.BOTTOM)
        self.assertIn("C1", str(ctx.exception))

    def test_two_device_grab_failure_names_c2(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1], cv2_module=cv2)
        cams.open()
        cv2.captures[1].grab_ok = False
        with self.assertRaises(CameraError) as ctx:
            cams.capture(Side.TOP)
        self.assertIn("C2", str(ctx.exception))

    def test_release_closes_only_opened_devices_for_two_cameras(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([4, 9], cv2_module=cv2)
        cams.open()
        cams.release()
        self.assertEqual(sorted(e[1] for e in cv2.log if e[0] == "release"), [4, 9])

    def test_capture_grabs_all_before_any_retrieve(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1, 2, 3], cv2_module=cv2, now_utc=_fixed_clock())
        cams.open()
        cv2.log.clear()  # drop the open-time warm-up reads
        frames = cams.capture(Side.TOP)
        ops = [entry[0] for entry in cv2.log if entry[0] in ("grab", "retrieve")]
        # Updated: 2 stale-flush grabs per device now come first; then the real grabs, then the retrieves.
        self.assertEqual(ops, ["grab"] * 12 + ["retrieve"] * 4)
        last_grab = max(i for i, e in enumerate(cv2.log) if e[0] == "grab")
        first_retrieve = min(i for i, e in enumerate(cv2.log) if e[0] == "retrieve")
        self.assertLess(last_grab, first_retrieve)

        self.assertEqual([f.camera_id for f in frames], list(CAMERA_IDS))
        self.assertEqual([f.seq for f in frames], [0, 1, 2, 3])
        self.assertEqual(frames[0].data, b"PNG:img0")
        self.assertEqual(frames[3].data, b"PNG:img3")
        self.assertEqual(frames[0].ext, "png")
        self.assertEqual(frames[0].timestamp_utc, "2026-10-23T08:30:00Z")
        self.assertEqual(frames[3].timestamp_utc, "2026-10-23T08:30:03Z")

    def test_skew_is_recorded(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1, 2, 3], cv2_module=cv2)
        self.assertIsNone(cams.last_skew_s)
        cams.capture(Side.TOP)
        self.assertIsNotNone(cams.last_skew_s)
        self.assertGreaterEqual(cams.last_skew_s, 0.0)

    def test_grab_failure_names_camera(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1, 2, 3], cv2_module=cv2)
        cams.open()
        cv2.captures[2].grab_ok = False
        with self.assertRaises(CameraError) as ctx:
            cams.capture(Side.TOP)
        self.assertIn("C3", str(ctx.exception))

    def test_retrieve_failure_names_camera(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1, 2, 3], cv2_module=cv2)
        cams.open()
        cv2.captures[1].retrieve_ok = False
        with self.assertRaises(CameraError) as ctx:
            cams.capture(Side.BOTTOM)
        self.assertIn("C2", str(ctx.exception))

    def test_encode_failure_names_camera(self):
        cv2 = FakeCv2()
        cv2.fail_encode_for = "img3"
        cams = OpenCvCameraSet([0, 1, 2, 3], cv2_module=cv2)
        with self.assertRaises(CameraError) as ctx:
            cams.capture(Side.TOP)
        self.assertIn("C4", str(ctx.exception))

    def test_device_that_does_not_open_names_camera(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1, 2, 3], cv2_module=cv2)
        cv2.VideoCapture = lambda index: _closed(cv2, index)
        with self.assertRaises(CameraError) as ctx:
            cams.capture(Side.TOP)
        self.assertIn("C1", str(ctx.exception))

    def test_release_closes_all_devices(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1, 2, 3], cv2_module=cv2)
        cams.open()
        cams.release()
        released = sorted(e[1] for e in cv2.log if e[0] == "release")
        self.assertEqual(released, [0, 1, 2, 3])

    def test_missing_cv2_raises_camera_error_when_not_injected(self):
        with mock.patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaises(CameraError) as ctx:
                OpenCvCameraSet([0, 1, 2, 3])
        self.assertIn("OpenCV", str(ctx.exception))


def _closed(cv2, index):
    cap = FakeCapture(cv2, index, cv2.log)
    cap.opened = False
    return cap


class FakeCameraSetPreviewTests(unittest.TestCase):
    def test_preview_bytes_count_calls_and_unknown_id_is_none(self):
        cams = FakeCameraSet()
        self.assertEqual(cams.preview("C2"), b"FAKEJPEG:C2:0")
        self.assertEqual(cams.preview("C1"), b"FAKEJPEG:C1:1")
        self.assertIsNone(cams.preview("C9"))
        self.assertIsNone(FakeCameraSet(camera_ids=("C1",)).preview("C2"))

    def test_preview_does_not_change_capture_state(self):
        cams = FakeCameraSet()
        cams.preview("C1")
        cams.preview("C1")
        self.assertEqual(cams.capture_counts, {Side.TOP: 0, Side.BOTTOM: 0})
        frames = cams.capture(Side.TOP)
        self.assertEqual(frames[0].data, b"FAKE:C1:TOP:0")


class OpenCvPreviewTests(unittest.TestCase):
    def _opened(self, indexes=(0, 1, 2, 3)):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet(list(indexes), cv2_module=cv2, now_utc=_fixed_clock())
        cams.open()
        return cv2, cams

    def test_preview_returns_jpeg_bytes_for_known_id(self):
        cv2, cams = self._opened()
        self.assertEqual(cams.preview("C1"), b"JPG:img0")
        self.assertEqual(cams.preview("C4"), b"JPG:img3")
        self.assertEqual(cv2.encode_params[0], (".jpg", [FakeCv2.IMWRITE_JPEG_QUALITY, 70]))

    def test_preview_quality_argument_is_passed_to_encoder(self):
        cv2, cams = self._opened()
        cams.preview("C1", quality=40)
        self.assertEqual(cv2.encode_params[-1], (".jpg", [FakeCv2.IMWRITE_JPEG_QUALITY, 40]))

    def test_preview_unknown_id_is_none(self):
        cv2, cams = self._opened()
        self.assertIsNone(cams.preview("C9"))
        self.assertIsNone(cams.preview(""))

    def test_preview_before_open_opens_devices_and_returns_jpeg(self):
        # Changed: the live view opens the devices itself, so the picture shows before the first Enter.
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1], cv2_module=cv2, now_utc=_fixed_clock())
        self.assertFalse(cams.is_open)
        self.assertEqual(cams.preview("C1"), b"JPG:img0")
        self.assertEqual(len(cv2.captures), 2)
        self.assertTrue(cams.is_open)
        self.assertIsNone(cams.last_open_error)

    def test_preview_after_release_reopens_devices(self):
        # Changed: preview() opens closed devices, so after release() the next preview reopens them.
        cv2, cams = self._opened()
        self.assertIsNotNone(cams.preview("C1"))
        cams.release()
        self.assertFalse(cams.is_open)
        self.assertEqual(cams.preview("C1"), b"JPG:img0")
        self.assertTrue(cams.is_open)

    def test_preview_returns_none_immediately_while_capture_holds_lock(self):
        cv2, cams = self._opened()
        entered = threading.Event()
        proceed = threading.Event()
        cap = cv2.captures[0]
        real_grab = cap.grab

        def blocking_grab():
            entered.set()
            proceed.wait(5)
            return real_grab()

        cap.grab = blocking_grab
        result = {}

        def run_capture():
            result["frames"] = cams.capture(Side.TOP)

        worker = threading.Thread(target=run_capture)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            started = time.monotonic()
            self.assertIsNone(cams.preview("C1"))
            self.assertIsNone(cams.preview("C2"))
            self.assertLess(time.monotonic() - started, 1.0)
        finally:
            proceed.set()
            worker.join(5)
        self.assertEqual(len(result["frames"]), 4)
        self.assertEqual(cams.preview("C1"), b"JPG:img0")

    def test_encode_failure_is_none_and_other_cameras_still_work(self):
        cv2, cams = self._opened()
        cv2.fail_encode_for = "img2"
        self.assertIsNone(cams.preview("C3"))
        self.assertEqual(cams.preview("C1"), b"JPG:img0")
        cv2.encode_ok = False
        self.assertIsNone(cams.preview("C1"))

    def test_grab_or_retrieve_problems_are_none_not_raised(self):
        cv2, cams = self._opened()
        cv2.captures[0].grab_ok = False
        self.assertIsNone(cams.preview("C1"))
        cv2.captures[0].grab_ok = True
        cv2.captures[0].retrieve_ok = False
        self.assertIsNone(cams.preview("C1"))
        cv2.captures[0].retrieve_ok = True

        def boom():
            raise RuntimeError("device gone")

        cv2.captures[1].grab = boom
        self.assertIsNone(cams.preview("C2"))
        self.assertEqual(cams.preview("C1"), b"JPG:img0")

    def test_preview_grabs_only_the_requested_device(self):
        cv2, cams = self._opened()
        cv2.log.clear()
        self.assertEqual(cams.preview("C3"), b"JPG:img2")
        grabs = [e[1] for e in cv2.log if e[0] == "grab"]
        self.assertEqual(grabs, [2])

    def test_wide_frame_is_downscaled_to_default_width_keeping_aspect(self):
        cv2, cams = self._opened((0,))
        cv2.captures[0].image = FakeImage("big", 1280, 720)
        self.assertEqual(cams.preview("C1"), b"JPG:big>resized")
        self.assertEqual(cv2.resizes, [("big", (640, 360), FakeCv2.INTER_AREA)])

    def test_width_argument_is_honoured(self):
        cv2, cams = self._opened((0,))
        cv2.captures[0].image = FakeImage("big", 1280, 720)
        cams.preview("C1", width=320)
        self.assertEqual(cv2.resizes[-1][1], (320, 180))

    def test_aspect_ratio_kept_for_odd_sizes(self):
        cv2, cams = self._opened((0,))
        cv2.captures[0].image = FakeImage("tall", 1000, 1000)
        cams.preview("C1")
        self.assertEqual(cv2.resizes[-1][1], (640, 640))
        cv2.captures[0].image = FakeImage("odd", 1000, 333)
        cams.preview("C1")
        self.assertEqual(cv2.resizes[-1][1], (640, 213))

    def test_narrow_frame_is_not_upscaled(self):
        cv2, cams = self._opened((0,))
        cv2.captures[0].image = FakeImage("small", 400, 300)
        self.assertEqual(cams.preview("C1"), b"JPG:small")
        self.assertEqual(cv2.resizes, [])

    def test_capture_still_works_after_previews(self):
        cv2, cams = self._opened()
        cams.preview("C1")
        cams.preview("C2")
        frames = cams.capture(Side.BOTTOM)
        self.assertEqual([f.camera_id for f in frames], list(CAMERA_IDS))
        self.assertEqual(frames[0].data, b"PNG:img0")
        self.assertEqual(frames[3].data, b"PNG:img3")
        self.assertIsNotNone(cams.last_skew_s)
        self.assertIsNotNone(cams.preview("C1"))


class _Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def _working(cv2, index):
    return FakeCapture(cv2, index, cv2.log)


class OpenCvLiveViewOpenTests(unittest.TestCase):
    def test_capture_after_preview_reuses_the_open_devices(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1, 2, 3], cv2_module=cv2, now_utc=_fixed_clock())
        cams.preview("C1")
        self.assertEqual(len(cv2.captures), 4)
        frames = cams.capture(Side.TOP)
        self.assertEqual(len(frames), 4)
        self.assertEqual(len(cv2.captures), 4)  # no second open

    def test_failed_open_returns_none_and_waits_for_the_cooldown(self):
        cv2 = FakeCv2()
        calls = []

        def video_capture(index):
            calls.append(index)
            return _closed(cv2, index)

        cv2.VideoCapture = video_capture
        clock = _Clock(100.0)
        cams = OpenCvCameraSet([0], cv2_module=cv2, monotonic=clock, reopen_cooldown_s=5.0)

        self.assertIsNone(cams.preview("C1"))
        self.assertEqual(calls, [0])
        self.assertIsNotNone(cams.last_open_error)
        self.assertIn("C1", cams.last_open_error)
        self.assertIn("0", cams.last_open_error)
        self.assertFalse(cams.is_open)

        clock.now = 104.9
        self.assertIsNone(cams.preview("C1"))
        self.assertEqual(calls, [0])  # no retry before the cooldown

        clock.now = 105.0
        cv2.VideoCapture = lambda index: _working(cv2, index)
        self.assertEqual(cams.preview("C1"), b"JPG:img0")  # retried after the cooldown
        self.assertIsNone(cams.last_open_error)
        self.assertTrue(cams.is_open)

    def test_preview_while_capture_holds_the_lock_returns_none_and_opens_nothing(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1], cv2_module=cv2)
        self.assertTrue(cams._lock.acquire(blocking=False))
        try:
            started = time.monotonic()
            self.assertIsNone(cams.preview("C1"))
            self.assertLess(time.monotonic() - started, 1.0)
        finally:
            cams._lock.release()
        self.assertEqual(cv2.captures, [])
        self.assertFalse(cams.is_open)

    def test_camera_that_does_not_open_gives_none_and_names_camera_and_index(self):
        cv2 = FakeCv2()
        cv2.VideoCapture = lambda index: _closed(cv2, index)
        cams = OpenCvCameraSet([5, 6], cv2_module=cv2)
        self.assertIsNone(cams.preview("C2"))
        self.assertIn("C1", cams.last_open_error)
        self.assertIn("5", cams.last_open_error)

    def test_second_camera_failing_releases_the_first(self):
        cv2 = FakeCv2()
        cv2.VideoCapture = lambda index: _closed(cv2, index) if index == 6 else _working(cv2, index)
        cams = OpenCvCameraSet([5, 6], cv2_module=cv2)
        self.assertIsNone(cams.preview("C1"))
        self.assertIn("C2", cams.last_open_error)
        self.assertIn("6", cams.last_open_error)
        self.assertEqual(sorted(e[1] for e in cv2.log if e[0] == "release"), [5, 6])
        self.assertFalse(cams.is_open)

    def test_open_failure_in_capture_still_raises_and_records_the_error(self):
        cv2 = FakeCv2()
        cv2.VideoCapture = lambda index: _closed(cv2, index)
        cams = OpenCvCameraSet([7], cv2_module=cv2)
        with self.assertRaises(CameraError):
            cams.capture(Side.TOP)
        self.assertIn("C1", cams.last_open_error)

    def test_open_success_clears_last_open_error(self):
        cv2 = FakeCv2()
        cv2.VideoCapture = lambda index: _closed(cv2, index)
        cams = OpenCvCameraSet([7], cv2_module=cv2, monotonic=_Clock(0.0))
        with self.assertRaises(CameraError):
            cams.open()
        self.assertIsNotNone(cams.last_open_error)
        cv2.VideoCapture = lambda index: _working(cv2, index)
        cams.open()
        self.assertIsNone(cams.last_open_error)
        self.assertTrue(cams.is_open)

    def test_last_open_error_is_read_only(self):
        cams = OpenCvCameraSet([0], cv2_module=FakeCv2())
        with self.assertRaises(AttributeError):
            cams.last_open_error = "x"
        with self.assertRaises(AttributeError):
            cams.is_open = True


class FakeCameraSetStatusTests(unittest.TestCase):
    def test_fake_cameras_report_open_with_no_error(self):
        cams = FakeCameraSet()
        self.assertIs(cams.is_open, True)
        self.assertIsNone(cams.last_open_error)


class BlurCheckTests(unittest.TestCase):
    def _frame(self):
        return Frame(camera_id="C1", side=Side.TOP, seq=0, timestamp_utc="t", data=b"x")

    def test_sharp_image_passes_flat_image_fails(self):
        sharp = [[0 if (x + y) % 2 == 0 else 255 for x in range(8)] for y in range(8)]
        flat = [[128] * 8 for _ in range(8)]
        frame = self._frame()
        sharp_check = make_blur_check(lambda f: sharp, min_variance=100.0)
        flat_check = make_blur_check(lambda f: flat, min_variance=1.0)
        self.assertTrue(sharp_check(frame))
        self.assertFalse(flat_check(frame))

    def test_threshold_is_inclusive(self):
        # Single interior pixel differing by 4 from neighbours -> variance 0 for n=1.
        img = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
        check = make_blur_check(lambda f: img, min_variance=0.0)
        self.assertTrue(check(self._frame()))
        strict = make_blur_check(lambda f: img, min_variance=0.5)
        self.assertFalse(strict(self._frame()))

    def test_frame_smaller_than_3x3_is_not_ok(self):
        for img in ([[1, 2], [3, 4]], [[1, 2, 3, 4, 5]], [[1], [2], [3]], []):
            check = make_blur_check(lambda f, img=img: img, min_variance=0.0)
            self.assertFalse(check(self._frame()), img)

    def test_decode_receives_the_frame(self):
        seen = []
        frame = self._frame()

        def decode(f):
            seen.append(f)
            return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        make_blur_check(decode, min_variance=0.0)(frame)
        self.assertEqual(seen, [frame])


def _recorded(make):
    """VideoCapture factory that also records each capture in cv2.captures."""

    def video_capture(cv2, index):
        cap = make(cv2, index)
        cv2.captures.append(cap)
        return cap

    return video_capture


def _failing_reads(cv2, index):
    cap = FakeCapture(cv2, index, cv2.log)
    cap.grab_ok = False
    cap.retrieve_ok = False
    return cap


class OpenCvStallReopenTests(unittest.TestCase):
    def test_five_consecutive_preview_failures_release_devices_then_cooldown_reopens(self):
        cv2 = FakeCv2()
        clock = _Clock(100.0)
        cams = OpenCvCameraSet([0], cv2_module=cv2, monotonic=clock, reopen_cooldown_s=5.0)
        cams.open()
        cv2.captures[0].grab_ok = False
        for _ in range(4):
            self.assertIsNone(cams.preview("C1"))
        self.assertTrue(cams.is_open)
        self.assertEqual(cams.status()["consecutive_failures"], 4)

        self.assertIsNone(cams.preview("C1"))  # the fifth failure stalls the set
        self.assertFalse(cams.is_open)
        self.assertEqual([e[1] for e in cv2.log if e[0] == "release"], [0])
        self.assertEqual(cams.last_open_error, "camera stopped delivering pictures; reopening")
        self.assertEqual(cams.status()["consecutive_failures"], 0)

        clock.now = 104.9
        self.assertIsNone(cams.preview("C1"))
        self.assertEqual(len(cv2.captures), 1)  # still cooling down, no reopen

        clock.now = 105.0
        cv2.VideoCapture = lambda index: _recorded(_working)(cv2, index)
        self.assertEqual(cams.preview("C1"), b"JPG:img0")  # reopened, picture back
        self.assertEqual(len(cv2.captures), 2)
        self.assertTrue(cams.is_open)
        self.assertIsNone(cams.last_open_error)

    def test_one_success_resets_the_failure_count(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0], cv2_module=cv2)
        cams.open()
        cap = cv2.captures[0]
        cap.grab_ok = False
        for _ in range(4):
            self.assertIsNone(cams.preview("C1"))
        cap.grab_ok = True
        self.assertEqual(cams.preview("C1"), b"JPG:img0")
        self.assertEqual(cams.status()["consecutive_failures"], 0)
        cap.grab_ok = False
        for _ in range(4):
            self.assertIsNone(cams.preview("C1"))
        self.assertTrue(cams.is_open)
        self.assertIsNone(cams.last_open_error)
        self.assertEqual(cams.status()["consecutive_failures"], 4)

    def test_fails_before_reopen_kwarg_is_honoured(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0], cv2_module=cv2, fails_before_reopen=2)
        cams.open()
        cv2.captures[0].grab_ok = False
        cams.preview("C1")
        self.assertTrue(cams.is_open)
        cams.preview("C1")
        self.assertFalse(cams.is_open)

    def test_capture_failure_raises_and_the_next_capture_reopens(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0], cv2_module=cv2, now_utc=_fixed_clock())
        cams.open()
        cv2.captures[0].grab_ok = False
        with self.assertRaises(CameraError):
            cams.capture(Side.TOP)
        self.assertFalse(cams.is_open)
        self.assertEqual([e[1] for e in cv2.log if e[0] == "release"], [0])
        frames = cams.capture(Side.TOP)  # no cooldown for capture
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(cv2.captures), 2)
        self.assertTrue(cams.is_open)


class OpenCvReleaseTests(unittest.TestCase):
    def test_release_twice_is_safe(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1], cv2_module=cv2)
        cams.open()
        cams.release()
        cams.release()
        self.assertFalse(cams.is_open)
        self.assertEqual(sorted(e[1] for e in cv2.log if e[0] == "release"), [0, 1])

    def test_release_takes_the_lock_and_leaves_it_free(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0], cv2_module=cv2)
        cams.open()
        cams.release()
        self.assertTrue(cams._lock.acquire(blocking=False))
        cams._lock.release()

    def test_release_does_not_hang_while_a_preview_holds_the_lock(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0], cv2_module=cv2)
        cams.open()
        held = threading.Event()
        done = threading.Event()

        def hold():
            with cams._lock:
                held.set()
                done.wait(5)

        holder = threading.Thread(target=hold)
        holder.start()
        try:
            self.assertTrue(held.wait(5))
            with mock.patch("station.cameras.RELEASE_LOCK_TIMEOUT_S", 0.05):
                started = time.monotonic()
                cams.release()
                self.assertLess(time.monotonic() - started, 1.0)
        finally:
            done.set()
            holder.join(5)
        self.assertFalse(cams.is_open)
        self.assertEqual([e[1] for e in cv2.log if e[0] == "release"], [0])


class OpenCvWarmupAndFlushTests(unittest.TestCase):
    def test_open_discards_warmup_frames_on_each_device(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1], cv2_module=cv2)
        cams.open()
        self.assertEqual([e[1] for e in cv2.log if e[0] == "grab"], [0] * 5 + [1] * 5)
        self.assertEqual([e[1] for e in cv2.log if e[0] == "retrieve"], [0] * 5 + [1] * 5)

    def test_zero_warmup_frames_disables_warmup(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1], cv2_module=cv2, open_warmup_frames=0)
        cams.open()
        self.assertEqual([e for e in cv2.log if e[0] in ("grab", "retrieve")], [])

    def test_warmup_failures_are_ignored(self):
        cv2 = FakeCv2()
        cv2.VideoCapture = lambda index: _failing_reads(cv2, index)
        cams = OpenCvCameraSet([0], cv2_module=cv2)
        cams.open()
        self.assertTrue(cams.is_open)
        self.assertIsNone(cams.last_open_error)

    def test_buffer_size_is_set_to_one_best_effort(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0], cv2_module=cv2)
        cams.open()
        self.assertEqual(cv2.captures[0].props[FakeCv2.CAP_PROP_BUFFERSIZE], 1)

    def test_buffer_size_error_is_ignored(self):
        class NoBufferSize(FakeCapture):
            def set(self, prop, value):
                if prop == FakeCv2.CAP_PROP_BUFFERSIZE:
                    raise RuntimeError("driver refuses buffer size")
                return super().set(prop, value)

        cv2 = FakeCv2()
        cv2.VideoCapture = lambda index: NoBufferSize(cv2, index, cv2.log)
        cams = OpenCvCameraSet([0], cv2_module=cv2)
        cams.open()
        self.assertTrue(cams.is_open)
        self.assertIsNone(cams.last_open_error)

    def test_stale_flush_runs_before_the_real_grabs(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet(
            [0, 1, 2, 3], cv2_module=cv2, now_utc=_fixed_clock(), open_warmup_frames=0
        )
        cams.open()
        cv2.log.clear()
        frames = cams.capture(Side.TOP)
        self.assertEqual(len(frames), 4)
        grabs = [e[1] for e in cv2.log if e[0] == "grab"]
        self.assertEqual(grabs, [0, 1, 2, 3] * 3)  # 2 flush rounds, then the real round

    def test_zero_stale_flush_grabs_each_device_once(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1], cv2_module=cv2, stale_flush_frames=0)
        cams.open()
        cv2.log.clear()
        cams.capture(Side.BOTTOM)
        self.assertEqual([e[1] for e in cv2.log if e[0] == "grab"], [0, 1])

    def test_skew_excludes_the_flush_time(self):
        cv2 = FakeCv2()
        clock = _Clock(0.0)
        cams = OpenCvCameraSet(
            [0, 1, 2, 3],
            cv2_module=cv2,
            now_utc=_fixed_clock(),
            monotonic=clock,
            open_warmup_frames=0,
        )
        cams.open()
        calls = []

        def make_ticking(cap):
            real_grab = cap.grab

            def ticking():
                calls.append(None)
                # The first 8 grabs are the flush: a long time passes there.
                clock.now += 5.0 if len(calls) <= 8 else 0.01
                return real_grab()

            return ticking

        for cap in cv2.captures:
            cap.grab = make_ticking(cap)
        cams.capture(Side.TOP)
        self.assertEqual(len(calls), 12)
        self.assertAlmostEqual(cams.last_skew_s, 0.03, places=6)


class StatusTests(unittest.TestCase):
    def test_fake_camera_set_status(self):
        self.assertEqual(
            FakeCameraSet().status(),
            {
                "open": True,
                "last_error": None,
                "consecutive_failures": 0,
                "capture_mode": "simultaneous",
                "sequential_keep_open": False,
            },
        )

    def test_opencv_status_tracks_open_error_and_failures(self):
        cv2 = FakeCv2()
        cv2.VideoCapture = lambda index: _recorded(_closed)(cv2, index)
        cams = OpenCvCameraSet([7], cv2_module=cv2, monotonic=_Clock(0.0))
        with self.assertRaises(CameraError):
            cams.open()
        status = cams.status()
        self.assertFalse(status["open"])
        self.assertIn("C1", status["last_error"])
        self.assertEqual(status["consecutive_failures"], 0)

        cv2.VideoCapture = lambda index: _recorded(_working)(cv2, index)
        cams.open()
        self.assertEqual(
            cams.status(),
            {
                "open": True,
                "last_error": None,
                "consecutive_failures": 0,
                "capture_mode": "simultaneous",
                "sequential_keep_open": False,
            },
        )
        cv2.captures[-1].grab_ok = False
        self.assertIsNone(cams.preview("C1"))
        self.assertEqual(cams.status()["consecutive_failures"], 1)


def _logged_opens(cv2, make=_working):
    """VideoCapture factory that logs ("open", index), so open/release order can be checked."""

    def video_capture(index):
        cv2.log.append(("open", index))
        return make(cv2, index)

    return video_capture


def _peak_open(log):
    """Most devices open at once, from the ("open", i) and ("release", i) entries of a log."""
    now = peak = 0
    for entry in log:
        if entry[0] == "open":
            now += 1
            peak = max(peak, now)
        elif entry[0] == "release":
            now -= 1
    return peak


def _device_ops(log):
    return [e for e in log if e[0] in ("open", "grab", "retrieve", "release")]


def _opens_of(log, index):
    return sum(1 for e in log if e == ("open", index))


class _Ticker:
    """Monotonic clock that moves 1.0 s on every read: 0, 1, 2, ..."""

    def __init__(self):
        self.n = -1

    def __call__(self):
        self.n += 1
        return float(self.n)


class SequentialCaptureTests(unittest.TestCase):
    def test_capture_mode_must_be_simultaneous_or_sequential(self):
        with self.assertRaises(ValueError):
            OpenCvCameraSet([0, 1, 2, 3], cv2_module=FakeCv2(), capture_mode="parallel")
        with self.assertRaises(ValueError):
            FakeCameraSet(capture_mode="parallel")

    def test_status_reports_capture_mode_and_keep_open(self):
        cams = OpenCvCameraSet(
            [0, 1], cv2_module=FakeCv2(), capture_mode="sequential", sequential_keep_open=True
        )
        status = cams.status()
        self.assertEqual(status["capture_mode"], "sequential")
        self.assertIs(status["sequential_keep_open"], True)

    def test_sequential_shoots_c1_to_c4_one_device_at_a_time(self):
        cv2 = FakeCv2()
        cv2.VideoCapture = _logged_opens(cv2)
        cams = OpenCvCameraSet(
            [0, 1, 2, 3],
            cv2_module=cv2,
            now_utc=_fixed_clock(),
            capture_mode="sequential",
            open_warmup_frames=0,
            stale_flush_frames=0,
        )
        frames = cams.capture(Side.TOP)
        self.assertEqual([f.camera_id for f in frames], list(CAMERA_IDS))
        self.assertEqual([f.seq for f in frames], [0, 1, 2, 3])
        self.assertEqual(
            [f.data for f in frames], [b"PNG:img0", b"PNG:img1", b"PNG:img2", b"PNG:img3"]
        )
        self.assertEqual(
            [f.timestamp_utc for f in frames],
            [f"2026-10-23T08:30:0{i}Z" for i in range(4)],
        )
        expected = []
        for i in range(4):
            expected += [("open", i), ("grab", i), ("retrieve", i), ("release", i)]
        self.assertEqual(_device_ops(cv2.log), expected)
        self.assertEqual(_peak_open(cv2.log), 1)
        self.assertFalse(cams.is_open)

    def test_open_per_shot_with_default_warmup_and_flush_never_holds_two_devices(self):
        cv2 = FakeCv2()
        cv2.VideoCapture = _logged_opens(cv2)
        cams = OpenCvCameraSet(
            [0, 1, 2, 3], cv2_module=cv2, now_utc=_fixed_clock(), capture_mode="sequential"
        )
        frames = cams.capture(Side.BOTTOM)
        self.assertEqual([f.camera_id for f in frames], list(CAMERA_IDS))
        self.assertEqual([e[1] for e in cv2.log if e[0] == "open"], [0, 1, 2, 3])
        self.assertEqual(_peak_open(cv2.log), 1)
        self.assertEqual(sorted(e[1] for e in cv2.log if e[0] == "release"), [0, 1, 2, 3])

    def test_keep_open_opens_all_but_never_grabs_two_before_retrieving_the_first(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet(
            [0, 1, 2, 3],
            cv2_module=cv2,
            now_utc=_fixed_clock(),
            capture_mode="sequential",
            sequential_keep_open=True,
            open_warmup_frames=0,
        )
        cams.open()
        self.assertEqual(len(cv2.captures), 4)
        cv2.log.clear()
        frames = cams.capture(Side.TOP)
        self.assertEqual([f.camera_id for f in frames], list(CAMERA_IDS))
        self.assertEqual(len(cv2.captures), 4)  # nothing reopened
        # Per camera: two stale-flush grabs, the real grab, then its retrieve.
        expected = []
        for i in range(4):
            expected += [("grab", i), ("grab", i), ("grab", i), ("retrieve", i)]
        self.assertEqual([e for e in cv2.log if e[0] in ("grab", "retrieve")], expected)
        self.assertEqual([e for e in cv2.log if e[0] == "release"], [])
        self.assertTrue(cams.is_open)

    def test_failure_on_camera_3_releases_everything_and_next_capture_reopens(self):
        for keep_open in (False, True):
            with self.subTest(keep_open=keep_open):
                cv2 = FakeCv2()
                state = {"fail": True}

                def make(cv2, index, state=state):
                    cap = FakeCapture(cv2, index, cv2.log)
                    if index == 2 and state["fail"]:
                        cap.grab_ok = False
                    return cap

                cv2.VideoCapture = _logged_opens(cv2, make)
                cams = OpenCvCameraSet(
                    [0, 1, 2, 3],
                    cv2_module=cv2,
                    now_utc=_fixed_clock(),
                    capture_mode="sequential",
                    sequential_keep_open=keep_open,
                    open_warmup_frames=0,
                )
                with self.assertRaises(CameraError) as ctx:
                    cams.capture(Side.TOP)
                self.assertIn("C3", str(ctx.exception))
                self.assertFalse(cams.is_open)
                released = sorted(e[1] for e in cv2.log if e[0] == "release")
                # Open-per-shot never opened C4; keep-open had opened all four up front.
                self.assertEqual(released, [0, 1, 2, 3] if keep_open else [0, 1, 2])
                self.assertEqual(
                    [e[1] for e in cv2.log if e[0] == "open"],
                    [0, 1, 2, 3] if keep_open else [0, 1, 2],
                )

                state["fail"] = False
                cv2.log.clear()
                frames = cams.capture(Side.TOP)  # no cooldown for capture
                self.assertEqual([f.camera_id for f in frames], list(CAMERA_IDS))
                self.assertEqual([e[1] for e in cv2.log if e[0] == "open"], [0, 1, 2, 3])
                self.assertEqual(_peak_open(cv2.log), 1 if not keep_open else 4)

    def test_timing_attributes_are_set_in_both_modes(self):
        for mode in ("simultaneous", "sequential"):
            with self.subTest(mode=mode):
                cams = OpenCvCameraSet(
                    [0, 1, 2, 3],
                    cv2_module=FakeCv2(),
                    now_utc=_fixed_clock(),
                    monotonic=_Ticker(),
                    capture_mode=mode,
                )
                cams.capture(Side.TOP)
                # Each camera's grab takes one tick; grab ends are 1, 3, 5, 7.
                self.assertEqual(cams.last_capture_times_s, [1.0, 1.0, 1.0, 1.0])
                self.assertEqual(cams.last_skew_s, 6.0)
                self.assertEqual(cams.last_capture_total_s, 7.0)


class SequentialPreviewTests(unittest.TestCase):
    def test_round_robin_opens_one_device_at_a_time_and_serves_cached_frames_off_turn(self):
        cv2 = FakeCv2()
        cv2.VideoCapture = _logged_opens(cv2)
        cams = OpenCvCameraSet(
            [0, 1, 2, 3], cv2_module=cv2, now_utc=_fixed_clock(), capture_mode="sequential"
        )
        # Every call grabs the turn camera and caches it, whatever was requested.
        results = [
            cams.preview("C1"),  # turn C1: fresh
            cams.preview("C2"),  # turn C2: fresh
            cams.preview("C1"),  # turn C3: C1 from the cache
            cams.preview("C3"),  # turn C4: C3 cached on its turn two calls ago
            cams.preview("C3"),  # turn C1: C3 from the cache
            cams.preview("C4"),  # turn C2: C4 cached on turn C4
            cams.preview("C3"),  # turn C3: fresh
        ]
        self.assertEqual(
            results,
            [b"JPG:img0", b"JPG:img1", b"JPG:img0", b"JPG:img2", b"JPG:img2", b"JPG:img3", b"JPG:img2"],
        )
        self.assertEqual(_peak_open(cv2.log), 1)
        self.assertFalse(cams.is_open)

    def test_failed_open_waits_for_the_cooldown_for_that_camera_only(self):
        cv2 = FakeCv2()

        def make(cv2, index):
            return _closed(cv2, index) if index == 1 else FakeCapture(cv2, index, cv2.log)

        cv2.VideoCapture = _logged_opens(cv2, make)
        clock = _Clock(100.0)
        cams = OpenCvCameraSet(
            [0, 1],
            cv2_module=cv2,
            monotonic=clock,
            reopen_cooldown_s=5.0,
            capture_mode="sequential",
        )
        self.assertEqual(cams.preview("C1"), b"JPG:img0")  # turn C1
        self.assertIsNone(cams.preview("C2"))  # turn C2: the open fails
        self.assertEqual(_opens_of(cv2.log, 1), 1)
        self.assertIn("C2", cams.last_open_error)
        self.assertEqual(cams.preview("C1"), b"JPG:img0")  # turn C1 still works

        self.assertIsNone(cams.preview("C2"))  # turn C2 inside the cooldown: no open attempt
        self.assertEqual(_opens_of(cv2.log, 1), 1)

        clock.now = 105.0
        cams.preview("C1")
        self.assertIsNone(cams.preview("C2"))  # turn C2 after the cooldown: tried again
        self.assertEqual(_opens_of(cv2.log, 1), 2)

    def test_keep_open_preview_opens_one_device_per_turn_and_keeps_it(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet(
            [0, 1, 2, 3],
            cv2_module=cv2,
            now_utc=_fixed_clock(),
            capture_mode="sequential",
            sequential_keep_open=True,
            open_warmup_frames=0,
        )
        cams.preview("C1")
        self.assertEqual(len(cv2.captures), 1)
        for cam in ("C2", "C3", "C4"):
            cams.preview(cam)
        self.assertEqual(len(cv2.captures), 4)
        self.assertEqual([e for e in cv2.log if e[0] == "release"], [])
        self.assertTrue(cams.is_open)
        frames = cams.capture(Side.TOP)
        self.assertEqual(len(frames), 4)
        self.assertEqual(len(cv2.captures), 4)  # the capture reuses the open devices


class FakeCaptureModeTests(unittest.TestCase):
    def test_fake_records_the_order_of_camera_shots(self):
        cams = FakeCameraSet(capture_mode="sequential", sequential_keep_open=True)
        cams.capture(Side.TOP)
        cams.capture(Side.BOTTOM)
        self.assertEqual(
            cams.capture_order,
            [("TOP", c) for c in CAMERA_IDS] + [("BOTTOM", c) for c in CAMERA_IDS],
        )
        self.assertIsNone(cams.last_capture_total_s)
        self.assertEqual(
            cams.status(),
            {
                "open": True,
                "last_error": None,
                "consecutive_failures": 0,
                "capture_mode": "sequential",
                "sequential_keep_open": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
