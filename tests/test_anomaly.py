import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from station import anomaly
from station.contracts import Frame, Side, StationError

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np

SHAPE = (80, 96)  # grey test images; feature grid is (10, 12)


class BlockExtractor:
    """Fake extractor: 8x8 block statistics -> (H/8, W/8, 4) = mean, max, min, std."""

    name = "fake-block-8-v1"
    stride = 8
    input_hw = SHAPE

    def extract(self, image):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        h, w = arr.shape
        hb, wb = h // 8, w // 8
        blocks = arr[: hb * 8, : wb * 8].reshape(hb, 8, wb, 8).transpose(0, 2, 1, 3).reshape(hb, wb, 64)
        return np.stack(
            [blocks.mean(-1), blocks.max(-1), blocks.min(-1), blocks.std(-1)], axis=-1
        ).astype(np.float32)


def _encode(arr):
    h, w = arr.shape
    return f"{h}x{w}:".encode("ascii") + arr.astype(np.uint8).tobytes()


def _decode(frame):
    head, _, body = frame.data.partition(b":")
    h, w = (int(v) for v in head.split(b"x"))
    return np.frombuffer(body, dtype=np.uint8).reshape(h, w).astype(np.float64)


def _frame(arr, camera="C1", side=Side.TOP):
    return Frame(camera_id=camera, side=side, seq=0, timestamp_utc="2026-10-24T08:30:00Z",
                 data=_encode(arr))


def _good(level=120, seed=0, shape=SHAPE):
    """Still good scene: level plus 0/1 grey noise."""
    rng = np.random.RandomState(seed)
    return (level + rng.randint(0, 2, shape)).astype(np.uint8)


def _blob_image(seed=0, top=20, left=30, h=24, w=24, shape=SHAPE):
    arr = _good(seed=seed, shape=shape).astype(np.int64)
    arr[top:top + h, left:left + w] = 255
    return arr.astype(np.uint8), (left, top, w, h)


def _builder(extractor=None, **kwargs):
    return anomaly.PatchCoreBuilder(extractor or BlockExtractor(), _decode, **kwargs)


def _overlap_fraction(box, blob):
    bx, by, bw, bh = box
    lx, ly, lw, lh = blob
    ix = max(0, min(bx + bw, lx + lw) - max(bx, lx))
    iy = max(0, min(by + bh, ly + lh) - max(by, ly))
    return (ix * iy) / float(lw * lh)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class CoresetTests(unittest.TestCase):
    def test_same_seed_gives_same_indices(self):
        rng = np.random.RandomState(1)
        feats = rng.randn(500, 200).astype(np.float32)  # projected: C > 128
        a = anomaly.greedy_coreset(feats, 40, seed=3)
        b = anomaly.greedy_coreset(feats, 40, seed=3)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(np.unique(a)), 40)
        self.assertTrue(np.all(np.diff(a) > 0))

    def test_n_select_at_least_n_returns_all(self):
        feats = np.arange(30, dtype=np.float32).reshape(10, 3)
        np.testing.assert_array_equal(anomaly.greedy_coreset(feats, 10, 0), np.arange(10))
        np.testing.assert_array_equal(anomaly.greedy_coreset(feats, 50, 0), np.arange(10))

    def test_far_outlier_is_always_selected(self):
        rng = np.random.RandomState(2)
        feats = rng.rand(300, 4).astype(np.float32) * 0.1
        feats = np.vstack([feats, np.full((1, 4), 1000.0, dtype=np.float32)])
        for seed in range(5):
            chosen = anomaly.greedy_coreset(feats, 5, seed=seed)
            self.assertIn(300, chosen)

    def test_progress_callback_called(self):
        calls = []
        anomaly.greedy_coreset(np.random.RandomState(0).rand(50, 4), 10, 0,
                               progress=lambda done, total: calls.append((done, total)))
        self.assertEqual(calls[-1], (10, 10))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class NumpyBackendTests(unittest.TestCase):
    def test_chunking_matches_brute_force(self):
        rng = np.random.RandomState(4)
        q = rng.randn(37, 6).astype(np.float32) * 3 + 50
        b = rng.randn(23, 6).astype(np.float32) * 3 + 50
        brute = np.sqrt(((q[:, None, :].astype(np.float64) - b[None].astype(np.float64)) ** 2).sum(-1)).min(1)
        for max_bytes in (4 * 23 * 2, 4 * 5, 10 ** 9):  # forces many blocks, then one block
            backend = anomaly.NumpyBackend(max_chunk_bytes=max_bytes)
            got = backend.nn_distance(q, b)
            self.assertEqual(got.dtype, np.float32)
            np.testing.assert_allclose(got, brute, rtol=1e-4, atol=1e-3)

    def test_block_never_exceeds_budget(self):
        backend = anomaly.NumpyBackend(max_chunk_bytes=256 * 1024 * 1024)
        rows, cols = backend.block_shape(10 ** 6, 10 ** 6)
        self.assertLessEqual(rows * cols * 4, 256 * 1024 * 1024)
        rows, cols = backend.block_shape(5, 10 ** 9)
        self.assertLessEqual(rows * cols * 4, 256 * 1024 * 1024)

    def test_large_bank_and_queries_run_in_bounded_time(self):
        rng = np.random.RandomState(5)
        bank = rng.randn(20000, 32).astype(np.float32)
        queries = rng.randn(5000, 32).astype(np.float32)
        start = time.perf_counter()
        got = anomaly.NumpyBackend().nn_distance(queries, bank)
        elapsed = time.perf_counter() - start
        self.assertEqual(got.shape, (5000,))
        self.assertLess(elapsed, 60.0)
        sample = np.sqrt(((queries[:20, None, :].astype(np.float64) - bank[None].astype(np.float64)) ** 2).sum(-1)).min(1)
        np.testing.assert_allclose(got[:20], sample, rtol=1e-4, atol=1e-3)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class PatchCoreTests(unittest.TestCase):
    def _train(self, n=6, level=120, **kwargs):
        builder = _builder(**kwargs)
        for i in range(n):
            builder.add(_frame(_good(level, seed=i)))
        return builder

    def test_needs_two_images_per_key(self):
        builder = _builder()
        builder.add(_frame(_good(seed=0)))
        with self.assertRaises(ValueError) as ctx:
            builder.build("v")
        self.assertIn("C1/TOP", str(ctx.exception))

    def test_array_without_key_is_rejected(self):
        with self.assertRaises(ValueError):
            _builder().add(_good())

    def test_good_scores_low_and_blob_scores_high_with_box_on_blob(self):
        builder = self._train()
        for i in range(2):
            builder.add_holdout(_frame(_good(seed=100 + i)))
        model = builder.build("pc-1")
        good = model.predict(_frame(_good(seed=50)))
        self.assertLess(good.score, 0.5)
        self.assertEqual(good.camera_id, "C1")
        self.assertEqual(good.side, Side.TOP)

        arr, blob = _blob_image(seed=51)
        bad = model.predict(_frame(arr))
        self.assertGreater(bad.score, 0.5)
        self.assertEqual(len(bad.defects), 1)
        defect = bad.defects[0]
        self.assertEqual(defect.defect_class, "anomaly")
        x, y, w, h = defect.box
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, SHAPE[1])
        self.assertLessEqual(y + h, SHAPE[0])
        self.assertGreater(_overlap_fraction(defect.box, blob), 0.5)

    def test_holdout_sets_norm_ref_and_source(self):
        builder = self._train()
        # two held-out images: this fixture's block statistics are so discrete that one
        # float16 held-out image can match the float16 bank exactly (raw score 0)
        builder.add_holdout(_frame(_good(seed=100)))
        builder.add_holdout(_frame(_good(seed=101)))
        model = builder.build("pc-h")
        info = model.describe()["banks"]["C1/TOP"]
        self.assertEqual(info["norm_ref_source"], "holdout")
        self.assertEqual(info["holdout_images"], 2)
        self.assertGreater(info["norm_ref"], 0.0)
        raw_holdout = model._keys[("C1", "TOP")].calibration_raw
        self.assertEqual(len(raw_holdout), 2)
        self.assertAlmostEqual(info["norm_ref"], 2.0 * max(raw_holdout), places=9)

    def test_training_residual_path(self):
        model = self._train(n=10).build("pc-r")
        info = model.describe()["banks"]["C1/TOP"]
        self.assertEqual(info["norm_ref_source"], "training_residual")
        self.assertEqual(info["holdout_images"], 0)
        raw = model._keys[("C1", "TOP")].calibration_raw
        self.assertGreaterEqual(len(raw), 1)
        self.assertAlmostEqual(info["norm_ref"], 2.0 * float(np.percentile(raw, 99)), places=9)

    def test_top_and_bottom_are_independent(self):
        builder = _builder()
        for i in range(4):
            builder.add(_frame(_good(120, seed=i), side=Side.TOP))
            builder.add(_frame(_good(60, seed=i), side=Side.BOTTOM))
        builder.add_holdout(_frame(_good(120, seed=40), side=Side.TOP))
        builder.add_holdout(_frame(_good(60, seed=40), side=Side.BOTTOM))
        model = builder.build("pc-tb")
        top = model.predict(_frame(_good(120, seed=9), side=Side.TOP))
        bottom_same_as_top = model.predict(_frame(_good(120, seed=9), side=Side.BOTTOM))
        self.assertLess(top.score, 0.5)
        self.assertGreater(bottom_same_as_top.score, 0.5)
        self.assertEqual(sorted(model.describe()["banks"]), ["C1/BOTTOM", "C1/TOP"])

    def test_unknown_key_is_error_naming_it(self):
        model = self._train().build("pc-u")
        with self.assertRaises(StationError) as ctx:
            model.predict(_frame(_good(), camera="C3"))
        self.assertIn("C3/TOP", str(ctx.exception))

    def test_anomaly_map_is_normalised_and_grid_shaped(self):
        model = self._train().build("pc-m")
        amap = model.anomaly_map(_frame(_blob_image(seed=3)[0]))
        self.assertEqual(amap.shape, (10, 12))
        self.assertEqual(amap.dtype, np.float32)
        self.assertTrue(0.0 <= float(amap.min()) and float(amap.max()) <= 1.0)

    def test_good_image_has_no_box(self):
        model = self._train().build("pc-nb")
        result = model.predict(_frame(_good(120, seed=77)))
        if result.score == 0.0:
            self.assertEqual(result.defects, ())

    def test_frame_input_is_decoded_on_add(self):
        builder = _builder()
        for i in range(3):
            builder.add(_frame(_good(seed=i)))
        self.assertEqual(builder.build("pc-f").describe()["banks"]["C1/TOP"]["train_images"], 3)

    def test_save_load_round_trip_identical_predictions_and_bytes(self):
        builder = self._train()
        builder.add_holdout(_frame(_good(seed=300)))
        model = builder.build("pc-s")
        arr, _ = _blob_image(seed=301)
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "a.npz")
            second = os.path.join(tmp, "b.npz")
            third = os.path.join(tmp, "c.npz")
            model.save(first)
            model.save(second)
            with open(first, "rb") as fa, open(second, "rb") as fb:
                self.assertEqual(fa.read(), fb.read())
            loaded = anomaly.PatchCoreModel.load(first, _decode, BlockExtractor())
            loaded.save(third)
            with open(first, "rb") as fa, open(third, "rb") as fc:
                self.assertEqual(fa.read(), fc.read())
        for frame in (_frame(_good(seed=302)), _frame(arr)):
            a, b = model.predict(frame), loaded.predict(frame)
            self.assertEqual(a.score, b.score)
            self.assertEqual(a.defects, b.defects)
        self.assertEqual(loaded.version, "pc-s")

    def test_extractor_name_mismatch_is_rejected(self):
        model = self._train().build("pc-n")
        other = BlockExtractor()
        other.name = "something-else"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.npz")
            model.save(path)
            with self.assertRaises(StationError) as ctx:
                anomaly.PatchCoreModel.load(path, _decode, other)
        self.assertIn("fake-block-8-v1", str(ctx.exception))

    def test_describe_is_json_friendly(self):
        import json
        model = self._train().build("pc-d")
        text = json.dumps(model.describe(), sort_keys=True)
        self.assertIn("fake-block-8-v1", text)

    def test_build_caps_bank_at_max_bank(self):
        builder = _builder(max_bank=7, coreset_ratio=1.0)
        for i in range(3):
            builder.add(_frame(_good(seed=i)))
        built = builder.build("pc-cap")
        self.assertEqual(built.describe()["banks"]["C1/TOP"]["rows"], 7)

    def test_register_loader_works_without_modelstore(self):
        builder = self._train()
        model = builder.build("pc-reg")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.npz")
            model.save(path)
            with mock.patch.dict("sys.modules", {"station.modelstore": None}):
                loader = anomaly.register_patchcore_loader(BlockExtractor)
                loaded = loader(path, _decode)
        self.assertEqual(loaded.predict(_frame(_good(seed=5))).score,
                         model.predict(_frame(_good(seed=5))).score)

    def test_registers_when_modelstore_offers_register_loader(self):
        calls = []
        fake_store = mock.Mock()
        fake_store.register_loader = lambda kind, loader: calls.append(kind)
        with mock.patch.dict("sys.modules", {"station.modelstore": fake_store}):
            anomaly.register_patchcore_loader(BlockExtractor)
        self.assertEqual(calls, ["patchcore"])


def _fit(level=120, seeds=range(4), holdout_seeds=(100, 101), version="v1", extractor=None, **kwargs):
    builder = _builder(extractor, **kwargs)
    for s in seeds:
        builder.add(_frame(_good(level, seed=s)))
    for s in holdout_seeds:
        builder.add_holdout(_frame(_good(level, seed=s)))
    return builder.build(version)


def _update(model, seeds=(), level=120, holdout_seeds=(), version="v2", extractor=None, **kwargs):
    builder = anomaly.PatchCoreBuilder.from_model(model, extractor or BlockExtractor(), _decode, **kwargs)
    for s in seeds:
        builder.add(_frame(_good(level, seed=s)))
    for s in holdout_seeds:
        builder.add_holdout(_frame(_good(level, seed=s)))
    return builder.build(version)


def _raw_against(model, key, feats):
    bank32 = np.asarray(model.bank(key), dtype=np.float32)
    return float(anomaly.NumpyBackend().nn_distance(np.asarray(feats, dtype=np.float32), bank32).max())


def _block_feats(seed, level=120):
    return BlockExtractor().extract(_good(level, seed=seed)).reshape(-1, 4).astype(np.float16)


class _OtherDimExtractor(BlockExtractor):
    """Same name and input size, but 3 features per cell instead of 4."""

    def extract(self, image):
        return super().extract(image)[..., :3]


def _write_legacy_v1(path, model):
    """Write a file in the patchcore-v1 layout (no holdout features, no parent, no input_hw)."""
    arrays = {}
    entries = []
    for key in model.keys():
        tag = f"{key[0]}__{key[1]}"
        bank = np.asarray(model.bank(key), dtype=np.float16)
        arrays[f"bank__{tag}"] = bank
        entries.append({
            "camera_id": key[0], "side": key[1], "bank_rows": int(bank.shape[0]),
            "feature_dim": int(bank.shape[1]), "norm_ref": model.norm_ref(key),
            "norm_ref_source": "holdout", "train_images": 4, "holdout_images": 2,
            "calibration_raw": [],
        })
    meta = {"schema": "patchcore-v1", "version": model.version,
            "extractor": {"name": BlockExtractor.name, "stride": 8}, "settings": {}, "keys": entries}
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    anomaly._write_npz(np, path, arrays)


class _FakeCv2:
    IMREAD_COLOR = 1

    def __init__(self, image):
        self.image = image
        self.buffers = []

    def imdecode(self, buf, flag):
        self.buffers.append((len(buf), flag))
        return self.image


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class IncrementalTests(unittest.TestCase):
    def test_accessors(self):
        model = _fit()
        self.assertEqual(model.keys(), [("C1", "TOP")])
        self.assertEqual(model.bank(("C1", "TOP")).dtype, np.float16)
        self.assertEqual(model.norm_ref(("C1", "TOP")), model.describe()["banks"]["C1/TOP"]["norm_ref"])
        self.assertEqual(model.extractor_name, BlockExtractor.name)
        self.assertEqual(model.input_hw, SHAPE)
        self.assertIsNone(model.parent_version)
        with self.assertRaises(StationError):
            model.bank(("C9", "TOP"))

    def test_from_model_rejects_extractor_name_mismatch(self):
        model = _fit()
        other = BlockExtractor()
        other.name = "other-extractor"
        with self.assertRaises(StationError) as ctx:
            anomaly.PatchCoreBuilder.from_model(model, other, _decode)
        self.assertIn(BlockExtractor.name, str(ctx.exception))
        self.assertIn("other-extractor", str(ctx.exception))

    def test_from_model_rejects_input_hw_mismatch(self):
        model = _fit()
        other = BlockExtractor()
        other.input_hw = (64, 64)
        with self.assertRaises(StationError) as ctx:
            anomaly.PatchCoreBuilder.from_model(model, other, _decode)
        self.assertIn("input size", str(ctx.exception))

    def test_from_model_rejects_feature_dimension_mismatch_on_add(self):
        model = _fit()
        builder = anomaly.PatchCoreBuilder.from_model(model, _OtherDimExtractor(), _decode)
        with self.assertRaises(StationError) as ctx:
            builder.add(_frame(_good(seed=9)))
        self.assertIn("feature dimension 3", str(ctx.exception))
        self.assertIn("4", str(ctx.exception))

    def test_update_extends_bank_and_new_style_scores_drop(self):
        model = _fit()
        before = model.predict(_frame(_good(122, seed=60))).score
        self.assertGreater(before, 0.5)
        updated = _update(model, seeds=range(10, 13), level=122)
        self.assertLess(updated.predict(_frame(_good(122, seed=60))).score, before)
        self.assertLess(updated.predict(_frame(_good(122, seed=61))).score, 0.5)
        self.assertLess(updated.predict(_frame(_good(120, seed=50))).score, 0.5)
        self.assertGreater(updated.describe()["banks"]["C1/TOP"]["rows"], 0)

    def test_defect_after_update_still_scores_high_with_original_box(self):
        model = _fit()
        updated = _update(model, seeds=range(10, 13), level=122)
        arr, blob = _blob_image(seed=51)
        bad = updated.predict(_frame(arr))
        self.assertGreater(bad.score, 0.5)
        self.assertEqual(len(bad.defects), 1)
        box = bad.defects[0].box
        self.assertLessEqual(box[0] + box[2], SHAPE[1])
        self.assertLessEqual(box[1] + box[3], SHAPE[0])
        self.assertGreater(_overlap_fraction(box, blob), 0.5)

    def test_keys_only_in_old_model_are_kept_unchanged(self):
        builder = _builder()
        for i in range(3):
            builder.add(_frame(_good(120, seed=i), side=Side.TOP))
            builder.add(_frame(_good(60, seed=i), side=Side.BOTTOM))
        builder.add_holdout(_frame(_good(120, seed=40), side=Side.TOP))
        builder.add_holdout(_frame(_good(60, seed=40), side=Side.BOTTOM))
        model = builder.build("both")
        bottom_bank = np.array(model.bank(("C1", "BOTTOM")), copy=True)
        bottom_norm = model.norm_ref(("C1", "BOTTOM"))
        updated = _update(model, seeds=range(10, 12), level=120)
        self.assertEqual(updated.keys(), [("C1", "BOTTOM"), ("C1", "TOP")])
        np.testing.assert_array_equal(updated.bank(("C1", "BOTTOM")), bottom_bank)
        self.assertEqual(updated.norm_ref(("C1", "BOTTOM")), bottom_norm)
        self.assertEqual(updated.describe()["banks"]["C1/BOTTOM"]["images_seen"], 3)
        self.assertEqual(updated.describe()["banks"]["C1/TOP"]["images_seen"], 5)

    def test_key_new_in_update_needs_two_images(self):
        model = _fit()
        builder = anomaly.PatchCoreBuilder.from_model(model, BlockExtractor(), _decode)
        builder.add(_frame(_good(seed=5), camera="C2"))
        with self.assertRaises(ValueError) as ctx:
            builder.build("v2")
        self.assertIn("C2/TOP", str(ctx.exception))
        builder = anomaly.PatchCoreBuilder.from_model(model, BlockExtractor(), _decode)
        builder.add(_frame(_good(seed=5), camera="C2"))
        builder.add(_frame(_good(seed=6), camera="C2"))
        updated = builder.build("v2")
        self.assertEqual(updated.keys(), [("C1", "TOP"), ("C2", "TOP")])
        self.assertEqual(updated.describe()["banks"]["C2/TOP"]["images_seen"], 2)

    def test_parent_version_and_images_seen_accumulate_over_two_updates(self):
        v1 = _fit(seeds=range(3), version="v1")
        self.assertEqual(v1.describe()["banks"]["C1/TOP"]["images_seen"], 3)
        v2 = _update(v1, seeds=range(10, 12), version="v2")
        self.assertEqual(v2.parent_version, "v1")
        self.assertEqual(v2.describe()["parent_version"], "v1")
        self.assertEqual(v2.describe()["banks"]["C1/TOP"]["images_seen"], 5)
        self.assertEqual(v2.describe()["banks"]["C1/TOP"]["train_images"], 2)
        v3 = _update(v2, seeds=[20], version="v3")
        self.assertEqual(v3.parent_version, "v2")
        self.assertEqual(v3.describe()["banks"]["C1/TOP"]["images_seen"], 6)
        self.assertEqual(v3.describe()["banks"]["C1/TOP"]["train_images"], 1)
        self.assertEqual(v3.describe()["banks"]["C1/TOP"]["holdout_images"], 2)

    def test_holdout_cap_keeps_newest(self):
        builder = _builder(max_holdout_images=2)
        for i in range(3):
            builder.add(_frame(_good(seed=i)))
        for s in (200, 201, 202, 203):
            builder.add_holdout(_frame(_good(seed=s)))
        model = builder.build("cap")
        retained = model._keys[("C1", "TOP")].holdout_features
        self.assertEqual(len(retained), 2)
        np.testing.assert_array_equal(retained[0][0], _block_feats(202))
        np.testing.assert_array_equal(retained[1][0], _block_feats(203))
        self.assertEqual(len(model._keys[("C1", "TOP")].calibration_raw), 2)

        updated = _update(model, holdout_seeds=[210], seeds=[30], max_holdout_images=2)
        retained = updated._keys[("C1", "TOP")].holdout_features
        self.assertEqual(len(retained), 2)
        np.testing.assert_array_equal(retained[0][0], _block_feats(203))
        np.testing.assert_array_equal(retained[1][0], _block_feats(210))

    def test_recalibration_uses_old_holdout_features_without_old_pictures(self):
        model = _fit(holdout_seeds=(100, 101))
        # the old training and held-out pictures are gone: only new pictures are passed in
        updated = _update(model, seeds=range(10, 13), level=120)
        info = updated.describe()["banks"]["C1/TOP"]
        self.assertEqual(info["norm_ref_source"], "holdout")
        self.assertEqual(info["holdout_images"], 2)
        raws = [_raw_against(updated, ("C1", "TOP"), _block_feats(s)) for s in (100, 101)]
        self.assertAlmostEqual(info["norm_ref"], 2.0 * max(raws), places=5)
        self.assertEqual(len(updated._keys[("C1", "TOP")].calibration_raw), 2)

    def test_update_without_any_holdout_falls_back_to_training_residual(self):
        model = _fit(holdout_seeds=())
        updated = _update(model, seeds=range(10, 13), level=120)
        self.assertEqual(updated.describe()["banks"]["C1/TOP"]["norm_ref_source"], "training_residual")

    def test_update_is_deterministic_and_saves_byte_identical(self):
        model = _fit()
        first = _update(model, seeds=range(10, 13), level=122)
        second = _update(model, seeds=range(10, 13), level=122)
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = (os.path.join(tmp, n) for n in ("a.npz", "b.npz", "c.npz"))
            first.save(a)
            second.save(b)
            with open(a, "rb") as fa, open(b, "rb") as fb:
                self.assertEqual(fa.read(), fb.read())
            loaded = anomaly.PatchCoreModel.load(a, _decode, BlockExtractor())
            loaded.save(c)
            with open(a, "rb") as fa, open(c, "rb") as fc:
                self.assertEqual(fa.read(), fc.read())
            self.assertEqual(loaded.parent_version, "v1")
            self.assertEqual(loaded.describe()["banks"]["C1/TOP"]["holdout_images"], 2)
            # the loaded model can be updated again with only new pictures
            again = _update(loaded, seeds=[40], level=122, version="v3")
            self.assertEqual(again.describe()["banks"]["C1/TOP"]["images_seen"], 8)  # 4 + 3 + 1

    def test_legacy_v1_file_scores_but_refuses_update(self):
        model = _fit()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.npz")
            _write_legacy_v1(path, model)
            loaded = anomaly.PatchCoreModel.load(path, _decode, BlockExtractor())
        self.assertTrue(loaded.legacy)
        self.assertIsNone(loaded.parent_version)
        self.assertIsNone(loaded.input_hw)
        self.assertEqual(loaded.describe()["banks"]["C1/TOP"]["images_seen"], 4)
        self.assertEqual(loaded.predict(_frame(_good(seed=3))).score, model.predict(_frame(_good(seed=3))).score)
        with self.assertRaises(StationError) as ctx:
            anomaly.PatchCoreBuilder.from_model(loaded, BlockExtractor(), _decode)
        self.assertIn("Retrain", str(ctx.exception))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class Cv2RgbDecoderTests(unittest.TestCase):
    def test_bgr_is_converted_to_rgb(self):
        bgr = np.zeros((4, 5, 3), dtype=np.uint8)
        bgr[..., 0] = 10  # blue
        bgr[..., 1] = 20  # green
        bgr[..., 2] = 30  # red
        cv2 = _FakeCv2(bgr)
        frame = Frame(camera_id="C1", side=Side.TOP, seq=0, timestamp_utc="2026-10-24T08:30:00Z",
                      data=b"encoded-bytes")
        out = anomaly.cv2_rgb_decoder(cv2)(frame)
        self.assertEqual(out.shape, (4, 5, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(tuple(out[0, 0]), (30, 20, 10))
        self.assertEqual(cv2.buffers, [(len(b"encoded-bytes"), 1)])

    def test_undecodable_payload_is_station_error(self):
        frame = Frame(camera_id="C1", side=Side.TOP, seq=0, timestamp_utc="2026-10-24T08:30:00Z", data=b"bad")
        with self.assertRaises(StationError):
            anomaly.cv2_rgb_decoder(_FakeCv2(None))(frame)

    def test_missing_cv2_is_station_error(self):
        with mock.patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaises(StationError) as ctx:
                anomaly.cv2_rgb_decoder()
        self.assertIn("opencv", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
