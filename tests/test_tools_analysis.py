"""Tests for tools/analysis.py (numpy-only repeatability metrics) with synthetic data."""

import math
import os
import tempfile
import time
import unittest

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None
    HAS_NUMPY = False

if HAS_NUMPY:
    from tools import analysis


def _texture(H, W, seed, cut=40.0):
    """Broadband random texture in float, roughly 25..225 (Gaussian spectral envelope)."""
    rng = np.random.default_rng(seed)
    spec = np.fft.fft2(rng.standard_normal((H, W)))
    fy = np.fft.fftfreq(H)[:, None] * H
    fx = np.fft.fftfreq(W)[None, :] * W
    img = np.fft.ifft2(spec * np.exp(-(np.hypot(fy, fx) / cut) ** 2)).real
    img = (img - img.min()) / (img.max() - img.min())
    return img * 200.0 + 25.0


def _fourier_shift(img, dx, dy):
    """Circular sub-pixel shift: result(x) = img(x - dx, y - dy)."""
    H, W = img.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    phase = np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.fft.ifft2(np.fft.fft2(img) * phase).real


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestBrightness(unittest.TestCase):
    def test_constant_frames_have_zero_range_and_slope(self):
        g = np.full((5, 16, 16), 90, dtype=np.uint8)
        r = analysis.brightness_stats(g)
        self.assertEqual(r["mean_per_frame"], [90.0] * 5)
        self.assertAlmostEqual(r["range_pct"], 0.0)
        self.assertAlmostEqual(r["slope_per_frame"], 0.0)
        self.assertAlmostEqual(r["std"], 0.0)

    def test_known_ramp_slope_is_recovered(self):
        T, H, W = 10, 8, 8
        base = np.full((H, W), 100.0)
        g = np.stack([base + 0.75 * t for t in range(T)])  # float input
        r = analysis.brightness_stats(g)
        self.assertAlmostEqual(r["slope_per_frame"], 0.75, places=9)
        self.assertAlmostEqual(r["min"], 100.0)
        self.assertAlmostEqual(r["max"], 100.0 + 0.75 * 9)
        expected_range = (r["max"] - r["min"]) / r["mean"] * 100.0
        self.assertAlmostEqual(r["range_pct"], expected_range)

    def test_uint8_ramp_slope_is_recovered(self):
        T = 20
        g = np.stack([np.full((10, 10), 50 + 2 * t, dtype=np.uint8) for t in range(T)])
        r = analysis.brightness_stats(g)
        self.assertAlmostEqual(r["slope_per_frame"], 2.0, places=9)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestSharpness(unittest.TestCase):
    def test_constant_frames_have_zero_laplacian_variance(self):
        g = np.full((3, 12, 12), 7, dtype=np.uint8)
        r = analysis.sharpness_stats(g)
        self.assertEqual(r["lap_var_per_frame"], [0.0, 0.0, 0.0])
        self.assertEqual(r["cv"], 0.0)

    def test_sharp_image_scores_higher_than_blurred(self):
        rng = np.random.default_rng(1)
        sharp = rng.integers(0, 256, size=(64, 64)).astype(np.float64)
        blurred = sharp.copy()
        for _ in range(3):  # repeated 3x3 box blur
            p = np.pad(blurred, 1, mode="edge")
            blurred = sum(p[i:i + 64, j:j + 64] for i in range(3) for j in range(3)) / 9.0
        g = np.stack([sharp, blurred]).astype(np.float64)
        r = analysis.sharpness_stats(g)
        self.assertGreater(r["lap_var_per_frame"][0], r["lap_var_per_frame"][1] * 5)
        self.assertEqual(len(r["lap_var_per_frame"]), 2)

    def test_laplacian_variance_matches_direct_formula(self):
        rng = np.random.default_rng(2)
        f = rng.integers(0, 256, size=(1, 9, 11)).astype(np.float64)
        lap = np.zeros((7, 9))
        for y in range(1, 8):
            for x in range(1, 10):
                c = f[0, y, x]
                lap[y - 1, x - 1] = 4 * c - f[0, y - 1, x] - f[0, y + 1, x] - f[0, y, x - 1] - f[0, y, x + 1]
        r = analysis.sharpness_stats(f)
        self.assertAlmostEqual(r["lap_var_per_frame"][0], lap.var(), places=6)

    def test_frame_smaller_than_3x3_raises(self):
        with self.assertRaises(ValueError):
            analysis.sharpness_stats(np.zeros((2, 2, 5), dtype=np.uint8))
        with self.assertRaises(ValueError):
            analysis.sharpness_stats(np.zeros((2, 5, 2), dtype=np.uint8))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestTemporalNoise(unittest.TestCase):
    def test_constant_frames_have_zero_noise(self):
        g = np.full((6, 20, 20), 128, dtype=np.uint8)
        r = analysis.temporal_noise(g)
        for key in ("pixel_std_mean", "pixel_std_p50", "pixel_std_p95",
                    "pixel_std_p99", "pixel_std_max"):
            self.assertEqual(r[key], 0.0, key)

    def test_gaussian_noise_sigma_two_is_recovered(self):
        rng = np.random.default_rng(3)
        T, H, W = 50, 120, 160
        base = np.full((H, W), 128.0)
        frames = base + rng.normal(0.0, 2.0, size=(T, H, W))
        g = np.clip(np.rint(frames), 0, 255).astype(np.uint8)
        r = analysis.temporal_noise(g)
        self.assertAlmostEqual(r["pixel_std_mean"], 2.0, delta=0.15)
        self.assertLessEqual(r["pixel_std_p50"], r["pixel_std_p95"])
        self.assertLessEqual(r["pixel_std_p95"], r["pixel_std_p99"])
        self.assertLessEqual(r["pixel_std_p99"], r["pixel_std_max"])

    def test_roi_restricts_to_window(self):
        rng = np.random.default_rng(4)
        T, H, W = 40, 60, 80
        g = np.full((T, H, W), 100.0)
        # Noise only in the window x in [10, 30), y in [5, 25).
        g[:, 5:25, 10:30] += rng.normal(0.0, 3.0, size=(T, 20, 20))
        noisy = analysis.temporal_noise(g, roi=(10, 5, 20, 20))
        quiet = analysis.temporal_noise(g, roi=(50, 40, 20, 15))
        self.assertAlmostEqual(noisy["pixel_std_mean"], 3.0, delta=0.2)
        self.assertEqual(quiet["pixel_std_max"], 0.0)
        full = analysis.temporal_noise(g)
        self.assertLess(full["pixel_std_p50"], noisy["pixel_std_mean"])

    def test_needs_two_frames(self):
        with self.assertRaises(ValueError):
            analysis.temporal_noise(np.zeros((1, 4, 4), dtype=np.uint8))

    def test_bad_roi_raises(self):
        g = np.zeros((3, 10, 10), dtype=np.uint8)
        for roi in [(0, 0, 11, 5), (-1, 0, 3, 3), (0, 0, 0, 3), (8, 8, 4, 4), (1, 2, 3)]:
            with self.assertRaises(ValueError, msg=str(roi)):
                analysis.temporal_noise(g, roi=roi)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestEstimateShifts(unittest.TestCase):
    def test_integer_shift_recovered_with_sign_convention(self):
        ref = _texture(96, 128, seed=5)
        # np.roll moves content right for +shift on axis 1 and down for +shift on axis 0.
        frame_moved = np.roll(ref, shift=(-3, 7), axis=(0, 1))  # dy=-3 (up), dx=+7 (right)
        g = np.stack([ref, frame_moved])
        r = analysis.estimate_shifts(g, ref_index=0)
        self.assertEqual(r[0]["dx"], 0.0)
        self.assertEqual(r[0]["dy"], 0.0)
        self.assertAlmostEqual(r[1]["dx"], 7.0, delta=0.05)
        self.assertAlmostEqual(r[1]["dy"], -3.0, delta=0.05)
        self.assertGreater(r[1]["response"], 0.3)

    def test_reference_index_other_than_zero(self):
        ref = _texture(64, 64, seed=6)
        g = np.stack([np.roll(ref, 2, axis=1), ref])
        r = analysis.estimate_shifts(g, ref_index=1)
        self.assertEqual(r[1]["dx"], 0.0)
        self.assertAlmostEqual(r[0]["dx"], 2.0, delta=0.05)

    def test_subpixel_shift_recovered(self):
        ref = _texture(96, 96, seed=7)
        true_dx, true_dy = 2.4, -1.3
        moved = _fourier_shift(ref, true_dx, true_dy)
        r = analysis.estimate_shifts(np.stack([ref, moved]), ref_index=0)
        self.assertAlmostEqual(r[1]["dx"], true_dx, delta=0.15)
        self.assertAlmostEqual(r[1]["dy"], true_dy, delta=0.15)

    def test_constant_frames_give_zero_shift(self):
        g = np.full((3, 32, 32), 60, dtype=np.uint8)
        for entry in analysis.estimate_shifts(g):
            self.assertEqual(entry["dx"], 0.0)
            self.assertEqual(entry["dy"], 0.0)

    def test_bad_index_raises(self):
        with self.assertRaises(ValueError):
            analysis.estimate_shifts(np.zeros((2, 8, 8), dtype=np.uint8), ref_index=2)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestResidual(unittest.TestCase):
    def test_constant_frames_have_zero_residual(self):
        g = np.full((5, 24, 24), 200, dtype=np.uint8)
        for align in (True, False):
            r = analysis.residual_scores(g, align=align)
            self.assertEqual(r["per_frame"], [0.0] * 5)
            self.assertEqual(r["max"], 0.0)
            self.assertEqual(r["p99_abs_diff_per_frame"], [0.0] * 5)

    def test_gaussian_noise_residual_matches_expected_mean_abs(self):
        rng = np.random.default_rng(8)
        T, H, W = 51, 120, 160
        frames = 128.0 + rng.normal(0.0, 2.0, size=(T, H, W))
        g = np.clip(np.rint(frames), 0, 255).astype(np.uint8)
        r = analysis.residual_scores(g, align=False)
        # Each frame minus a per-pixel median of T noisy frames: the difference has
        # variance sigma^2 + var(median), with var(median) ~ (pi/2) sigma^2 / T.
        var_q = 2.0 ** 2 + 1.0 / 12.0  # including uint8 rounding
        var_diff = var_q + (math.pi / 2.0) * var_q / T
        expected = math.sqrt(2.0 / math.pi) * math.sqrt(var_diff) / 255.0
        self.assertAlmostEqual(r["mean"] / expected, 1.0, delta=0.05)
        self.assertEqual(len(r["per_frame"]), T)
        self.assertEqual(len(r["p99_abs_diff_per_frame"]), T)
        self.assertLessEqual(r["p95"], r["max"])

    def test_shifted_frames_small_residual_with_align_large_without(self):
        base = np.random.default_rng(9).integers(0, 256, size=(80, 100)).astype(np.uint8)
        shifts = [(0, 0), (2, -3), (-1, 4), (3, 5)]
        g = np.stack([np.roll(base, shift=s, axis=(0, 1)) for s in shifts])
        aligned = analysis.residual_scores(g, align=True)
        raw = analysis.residual_scores(g, align=False)
        self.assertLess(aligned["max"], 0.01)
        self.assertGreater(raw["mean"], 0.1)

    def test_explicit_reference_with_align(self):
        base = np.random.default_rng(10).integers(0, 256, size=(60, 60)).astype(np.float64)
        g = np.stack([np.roll(base, 4, axis=1), np.roll(base, -2, axis=0)]).astype(np.uint8)
        r = analysis.residual_scores(g, reference=base, align=True)
        self.assertLess(r["max"], 0.01)
        self.assertEqual(len(r["per_frame"]), 2)

    def test_bad_reference_shape_raises(self):
        g = np.zeros((2, 10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            analysis.residual_scores(g, reference=np.zeros((9, 10)))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestInputValidation(unittest.TestCase):
    def test_bad_shapes_raise_value_error(self):
        funcs = [analysis.brightness_stats, analysis.sharpness_stats,
                 analysis.temporal_noise, analysis.estimate_shifts,
                 analysis.residual_scores]
        bad_inputs = [np.zeros((10, 10), dtype=np.uint8),
                      np.zeros((2, 3, 4, 5), dtype=np.uint8),
                      np.zeros((0, 4, 4), dtype=np.uint8),
                      [[[1, 2]]]]
        for fn in funcs:
            for bad in bad_inputs:
                with self.subTest(fn=fn.__name__, shape=getattr(bad, "shape", None)):
                    with self.assertRaises(ValueError):
                        fn(bad)

    def test_bad_dtype_raises(self):
        with self.assertRaises(ValueError):
            analysis.brightness_stats(np.zeros((2, 4, 4), dtype=bool))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestLargeStack(unittest.TestCase):
    def test_memory_conscious_path_runs_quickly(self):
        rng = np.random.default_rng(11)
        g = rng.integers(0, 256, size=(20, 600, 800), dtype=np.uint8)
        start = time.monotonic()
        b = analysis.brightness_stats(g)
        s = analysis.sharpness_stats(g)
        n = analysis.temporal_noise(g)
        sh = analysis.estimate_shifts(g)
        res = analysis.residual_scores(g)
        elapsed = time.monotonic() - start
        self.assertEqual(len(b["mean_per_frame"]), 20)
        self.assertEqual(len(s["lap_var_per_frame"]), 20)
        self.assertEqual(len(sh), 20)
        self.assertEqual(len(res["per_frame"]), 20)
        self.assertGreater(n["pixel_std_mean"], 0.0)
        self.assertLess(elapsed, 120.0)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestLoadSessionGray(unittest.TestCase):
    def test_round_trip(self):
        gray = np.random.default_rng(12).integers(0, 256, size=(4, 6, 8), dtype=np.uint8)
        stamps = np.array([1.0, 2.5, 3.0, 4.25])
        with tempfile.TemporaryDirectory() as d:
            np.savez(os.path.join(d, "frames_dev2.npz"), gray=gray, stamps=stamps)
            g2, s2 = analysis.load_session_gray(d, 2)
        self.assertEqual(g2.dtype, np.uint8)
        np.testing.assert_array_equal(g2, gray)
        np.testing.assert_allclose(s2, stamps)

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                analysis.load_session_gray(d, 0)


if __name__ == "__main__":
    unittest.main()
