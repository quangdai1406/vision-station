import re
import sys
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


class FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_AUTOFOCUS = 39
    CAP_PROP_AUTO_EXPOSURE = 21
    CAP_PROP_AUTO_WB = 44
    CAP_PROP_FOCUS = 28
    CAP_PROP_EXPOSURE = 15
    CAP_PROP_WB_TEMPERATURE = 45

    def __init__(self):
        self.log = []
        self.captures = []
        self.encode_ok = True
        self.fail_encode_for = None  # image string that fails encoding

    def VideoCapture(self, index):
        cap = FakeCapture(self, index, self.log)
        self.captures.append(cap)
        return cap

    def imencode(self, ext, image):
        self.log.append(("imencode", image))
        if not self.encode_ok or image == self.fail_encode_for:
            return (False, None)
        return (True, FakeBuf(b"PNG:" + image.encode("ascii")))


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

    def test_wrong_device_count_rejected(self):
        with self.assertRaises(CameraError):
            OpenCvCameraSet([0, 1, 2], cv2_module=FakeCv2())

    def test_capture_grabs_all_before_any_retrieve(self):
        cv2 = FakeCv2()
        cams = OpenCvCameraSet([0, 1, 2, 3], cv2_module=cv2, now_utc=_fixed_clock())
        frames = cams.capture(Side.TOP)
        ops = [entry[0] for entry in cv2.log if entry[0] in ("grab", "retrieve")]
        self.assertEqual(ops, ["grab"] * 4 + ["retrieve"] * 4)
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


if __name__ == "__main__":
    unittest.main()
