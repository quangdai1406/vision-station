import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from station import golden
from station.contracts import Frame, Side, StationError

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np


def _encode(arr):
    h, w = arr.shape
    return f"{h}x{w}:".encode("ascii") + arr.astype(np.uint8).tobytes()


def _decode(frame):
    # Test decode: data is b"<h>x<w>:" followed by uint8 pixels. No cv2 involved.
    head, _, body = frame.data.partition(b":")
    h, w = (int(v) for v in head.split(b"x"))
    return np.frombuffer(body, dtype=np.uint8).reshape(h, w).astype(np.float64)


def _frame(arr, camera="C1", side=Side.TOP):
    return Frame(camera_id=camera, side=side, seq=0, timestamp_utc="2026-10-23T08:30:00Z",
                 data=_encode(arr))


def _still_set(shape, n, base=120, seed=0):
    """n frames of a still scene with 0/1 gray noise."""
    rng = np.random.RandomState(seed)
    return [(base + rng.randint(0, 2, shape)).astype(np.uint8) for _ in range(n)]


def _build(frames_by_key, version="g-1", **kwargs):
    builder = golden.GoldenBuilder(_decode, **kwargs)
    for (camera, side), arrays in frames_by_key.items():
        for arr in arrays:
            builder.add(_frame(arr, camera, side))
    return builder.build(version)


def _blob(shape, top, left, h, w, base=120, lift=150):
    arr = np.full(shape, base, dtype=np.int64)
    arr[top:top + h, left:left + w] += lift
    return np.clip(arr, 0, 255).astype(np.uint8)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class GoldenBuilderTests(unittest.TestCase):
    def test_needs_at_least_two_frames_per_key(self):
        builder = golden.GoldenBuilder(_decode)
        builder.add(_frame(_still_set((20, 30), 1)[0], "C1", Side.TOP))
        with self.assertRaises(ValueError) as ctx:
            builder.build("v")
        self.assertIn("C1/TOP", str(ctx.exception))

    def test_no_frames_rejected(self):
        with self.assertRaises(ValueError):
            golden.GoldenBuilder(_decode).build("v")

    def test_frames_for_one_key_must_share_shape(self):
        builder = golden.GoldenBuilder(_decode)
        builder.add(_frame(np.zeros((20, 30), np.uint8)))
        with self.assertRaises(ValueError):
            builder.add(_frame(np.zeros((20, 31), np.uint8)))

    def test_bad_settings_rejected(self):
        for kwargs in ({"std_floor": 0.0}, {"z_ref": -1.0}, {"max_width_px": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    golden.GoldenBuilder(_decode, **kwargs)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class GoldenPredictTests(unittest.TestCase):
    def test_still_training_with_noise_scores_low_on_similar_frame(self):
        shape = (24, 32)
        model = _build({("C1", Side.TOP): _still_set(shape, 6)})
        rng = np.random.RandomState(99)
        new = (120 + rng.randint(0, 2, shape)).astype(np.uint8)
        result = model.predict(_frame(new))
        self.assertLess(result.score, 0.1)
        self.assertEqual(result.defects, ())
        self.assertEqual(result.camera_id, "C1")
        self.assertEqual(result.side, Side.TOP)

    def test_clear_bright_blob_scores_high_with_box_on_blob(self):
        shape = (120, 160)  # block factor 1
        model = _build({("C1", Side.TOP): _still_set(shape, 6)})
        blob = _blob(shape, top=40, left=50, h=40, w=40)
        result = model.predict(_frame(blob))
        self.assertGreater(result.score, 0.5)
        self.assertEqual(len(result.defects), 1)
        defect = result.defects[0]
        self.assertEqual(defect.defect_class, "difference_from_golden")
        self.assertEqual(defect.box, (50, 40, 40, 40))  # x, y, w, h in original pixels
        self.assertGreater(defect.score, 0.5)

    def test_flat_images_do_not_divide_by_zero(self):
        flat = np.full((10, 12), 100, dtype=np.uint8)
        model = _build({("C1", Side.TOP): [flat, flat, flat]})
        same = model.predict(_frame(flat))
        self.assertEqual(same.score, 0.0)
        self.assertEqual(same.defects, ())
        off = np.full((10, 12), 140, dtype=np.uint8)
        result = model.predict(_frame(off))
        self.assertTrue(0.0 < result.score <= 1.0)

    def test_top_and_bottom_keys_are_independent(self):
        shape = (16, 20)
        model = _build({
            ("C1", Side.TOP): [np.full(shape, 100, np.uint8)] * 3,
            ("C1", Side.BOTTOM): [np.full(shape, 200, np.uint8)] * 3,
        })
        bottom_ok = model.predict(_frame(np.full(shape, 200, np.uint8), side=Side.BOTTOM))
        top_wrong = model.predict(_frame(np.full(shape, 200, np.uint8), side=Side.TOP))
        self.assertEqual(bottom_ok.score, 0.0)
        self.assertEqual(top_wrong.score, 1.0)

    def test_unknown_key_raises_naming_it(self):
        model = _build({("C1", Side.TOP): _still_set((8, 8), 2)})
        with self.assertRaises(StationError) as ctx:
            model.predict(_frame(np.zeros((8, 8), np.uint8), camera="C2", side=Side.TOP))
        self.assertIn("C2/TOP", str(ctx.exception))

    def test_bad_shape_raises_station_error(self):
        model = _build({("C1", Side.TOP): _still_set((8, 8), 2)})
        with self.assertRaises(StationError):
            model.predict(_frame(np.zeros((8, 9), np.uint8)))

    def test_box_scaling_for_block_factor_two(self):
        shape = (120, 320)  # width 320 with max 160 -> block 2
        model = _build({("C1", Side.TOP): _still_set(shape, 4)})
        self.assertEqual(model.describe()["keys"]["C1/TOP"]["block_factor"], 2)
        blob = _blob(shape, top=20, left=100, h=40, w=60)  # aligned to the 2 px grid
        defect = model.predict(_frame(blob)).defects[0]
        self.assertEqual(defect.box, (100, 20, 60, 40))

    def test_box_scaling_for_block_factor_four(self):
        shape = (120, 640)  # width 640 with max 160 -> block 4
        model = _build({("C1", Side.TOP): _still_set(shape, 4)})
        self.assertEqual(model.describe()["keys"]["C1/TOP"]["block_factor"], 4)
        blob = _blob(shape, top=40, left=200, h=40, w=80)  # aligned to the 4 px grid
        defect = model.predict(_frame(blob)).defects[0]
        self.assertEqual(defect.box, (200, 40, 80, 40))

    def test_tiny_region_gives_no_box(self):
        shape = (40, 40)
        model = _build({("C1", Side.TOP): _still_set(shape, 4)})
        speck = _blob(shape, top=10, left=10, h=1, w=3)  # 3 px < 4 px minimum
        result = model.predict(_frame(speck))
        self.assertEqual(result.defects, ())


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class GoldenPersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _model(self):
        return _build({
            ("C1", Side.TOP): _still_set((120, 160), 4, seed=1),
            ("C1", Side.BOTTOM): _still_set((120, 160), 4, base=60, seed=2),
        }, version="golden-test")

    def test_save_load_round_trip_gives_identical_predictions(self):
        model = self._model()
        path = os.path.join(self.dir, "g.npz")
        model.save(path)
        loaded = golden.GoldenModel.load(path, _decode)
        self.assertEqual(loaded.version, "golden-test")
        self.assertEqual(loaded.describe(), model.describe())
        blob = _blob((120, 160), 30, 30, 30, 30)
        for side in (Side.TOP, Side.BOTTOM):
            a = model.predict(_frame(blob, side=side))
            b = loaded.predict(_frame(blob, side=side))
            self.assertEqual(a, b)

    def test_saves_are_deterministic(self):
        model = self._model()
        first = os.path.join(self.dir, "a.npz")
        second = os.path.join(self.dir, "b.npz")
        model.save(first)
        model.save(second)
        with open(first, "rb") as fa, open(second, "rb") as fb:
            self.assertEqual(fa.read(), fb.read())
        meta_a = str(np.load(first)["meta"])
        meta_b = str(np.load(second)["meta"])
        self.assertEqual(meta_a, meta_b)
        self.assertNotIn("created", meta_a)
        self.assertEqual(json.loads(meta_a)["schema"], golden.SCHEMA)

    def test_unknown_schema_raises_station_error(self):
        path = os.path.join(self.dir, "other.npz")
        np.savez(path, meta=np.array(json.dumps({"schema": "golden-v0"})))
        with self.assertRaises(StationError) as ctx:
            golden.GoldenModel.load(path, _decode)
        self.assertIn("golden-v0", str(ctx.exception))

    def test_unreadable_file_raises_station_error(self):
        path = os.path.join(self.dir, "junk.npz")
        with open(path, "wb") as fh:
            fh.write(b"not a zip file")
        with self.assertRaises(StationError):
            golden.GoldenModel.load(path, _decode)

    def test_describe_lists_constants_and_custom_values(self):
        default = self._model().describe()
        self.assertEqual(default["std_floor"], 2.0)
        self.assertEqual(default["z_ref"], 8.0)
        self.assertEqual(default["max_width_px"], 160)
        self.assertIn("demo", default["heuristic"])
        custom = _build({("C1", Side.TOP): _still_set((8, 8), 2)},
                        std_floor=3.0, z_ref=10.0, max_width_px=4).describe()
        self.assertEqual((custom["std_floor"], custom["z_ref"], custom["max_width_px"]),
                         (3.0, 10.0, 4))
        self.assertEqual(custom["keys"]["C1/TOP"]["block_factor"], 2)


class GoldenCv2AndNumpyTests(unittest.TestCase):
    def test_missing_cv2_gives_clear_error(self):
        with mock.patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaises(StationError) as ctx:
                golden.cv2_decoder()
        self.assertIn("OpenCV", str(ctx.exception))

    def test_injected_cv2_decoder_converts_to_gray_float(self):
        class FakeCv2:
            IMREAD_COLOR = 1
            COLOR_BGR2GRAY = 6

            def __init__(self):
                self.calls = []

            def imdecode(self, buf, flags):
                self.calls.append(("imdecode", len(buf), flags))
                return "BGR"

            def cvtColor(self, image, code):
                self.calls.append(("cvtColor", image, code))
                return np.full((3, 4), 7, dtype=np.uint8) if HAS_NUMPY else None

        if not HAS_NUMPY:
            self.skipTest("numpy not installed")
        cv2 = FakeCv2()
        decode = golden.cv2_decoder(cv2)
        out = decode(Frame(camera_id="C1", side=Side.TOP, seq=0, timestamp_utc="t", data=b"\x89PNGxx"))
        self.assertEqual(out.shape, (3, 4))
        self.assertEqual(out.dtype, np.float64)
        self.assertEqual(cv2.calls[0], ("imdecode", 6, 1))
        self.assertEqual(cv2.calls[1][2], 6)

    def test_undecodable_frame_raises_station_error(self):
        if not HAS_NUMPY:
            self.skipTest("numpy not installed")
        cv2 = mock.Mock()
        cv2.imdecode.return_value = None
        frame = Frame(camera_id="C2", side=Side.BOTTOM, seq=1, timestamp_utc="t", data=b"junk")
        with self.assertRaises(StationError) as ctx:
            golden.cv2_decoder(cv2)(frame)
        self.assertIn("C2", str(ctx.exception))

    def test_missing_numpy_gives_clear_error(self):
        with mock.patch.dict(sys.modules, {"numpy": None}):
            with self.assertRaises(StationError) as ctx:
                golden._np()
        self.assertIn("numpy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
