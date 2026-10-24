"""Tests for tools/report.py. Synthetic session folders in a temp dir; numpy tests are skipped without numpy."""
import contextlib
import csv
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover - depends on the environment
    np = None
    HAS_NUMPY = False

from tools import ToolDependencyError
from tools import analysis
from tools import report

NEED_NUMPY = unittest.skipUnless(HAS_NUMPY, "numpy is not installed")


def _build_session(root, name, mode, *, shifts=None, frames=6, h=48, w=64,
                   autofocus_applied=True, seed=1, base=None, sigma=2.0,
                   offsets=None, autofocus_readback=None):
    """Write a session folder that follows tools/FORMAT.md. Placement uses integer (dx, dy) shifts.

    base: optional fixed image for a stationary session (pure noise, no drift); the
    frame size then comes from base. offsets: optional per-frame brightness offsets
    (stationary only, one per frame). autofocus_readback: optional read-back value for autofocus.
    """
    rng = np.random.default_rng(seed)
    if base is None:
        base = rng.integers(30, 226, size=(h, w)).astype(np.float64)
        stationary_seq = [base + 0.5 * t for t in range(frames)]
    else:
        base = np.asarray(base, dtype=np.float64)
        h, w = base.shape
        stationary_seq = [base] * frames
    if offsets is not None:
        assert len(offsets) == frames
        stationary_seq = [f + float(offsets[t]) for t, f in enumerate(stationary_seq)]
    if mode == "placement":
        seq = [np.roll(base, (dy, dx), axis=(0, 1)) for dx, dy in shifts]
    else:
        seq = stationary_seq
    gray = np.stack([np.clip(f + rng.normal(0.0, sigma, size=(h, w)), 0, 255)
                     for f in seq]).astype(np.uint8)
    count = gray.shape[0]
    stamps = 1000.0 + np.arange(count) * 0.1
    sdir = Path(root) / name
    (sdir / "samples").mkdir(parents=True)
    np.savez(sdir / "frames_dev0.npz", gray=gray, stamps=stamps)
    warnings = [] if autofocus_applied else [
        "autofocus: requested off but the camera reports auto (readback 1)"]
    meta = {
        "schema": 1,
        "created_utc": "2026-10-24T08:00:00Z",
        "mode": mode,
        "session_name": name,
        "devices": [0],
        "notes": {"part": "P-1 good unchanged", "lighting": "bench LED",
                  "operator": "test", "free": ""},
        "requested": {"width": w, "height": h, "fps": 10, "fourcc": "MJPG",
                      "manual": {"focus": 30, "exposure": 100,
                                 "white_balance": 4500, "gain": 10}},
        "cameras": {"0": {
            "actual_width": w, "actual_height": h, "actual_fps": 10.0,
            "readback": {"autofocus": (autofocus_readback if autofocus_readback is not None
                                       else (0 if autofocus_applied else 1)),
                         "focus": 30,
                         "auto_exposure": 0, "exposure": 100, "auto_wb": 0,
                         "white_balance": 4500, "gain": 10},
            "applied_ok": {"autofocus": autofocus_applied, "auto_exposure": True,
                           "auto_wb": True, "focus": True, "exposure": True,
                           "white_balance": True, "gain": True},
            "warnings": warnings,
        }},
        "frames": [{"index": i, "device": 0, "stamp": float(stamps[i]),
                    "mean": float(gray[i].mean()), "laplacian_var": 1.0}
                   for i in range(count)],
    }
    (sdir / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sdir


def _quiet_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = report.main(argv)
    return rc, out.getvalue(), err.getvalue()


def _section(md, start, end):
    return md.split(start, 1)[1].split(end, 1)[0]


@NEED_NUMPY
class ReportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _read(self, path):
        return Path(path).read_text(encoding="utf-8")

    def test_writes_three_files_to_default_folder(self):
        sdir = _build_session(self.root, "stat_ok", "stationary")
        rc, _, _ = _quiet_main([str(sdir)])
        self.assertEqual(rc, 0)
        out = sdir / "report"
        for name in ("report.json", "report.md", "summary.csv"):
            self.assertTrue((out / name).is_file(), name)
        data = json.loads(self._read(out / "report.json"))
        self.assertEqual(len(data["rows"]), 1)

    def test_out_option_is_used(self):
        sdir = _build_session(self.root, "stat_out", "stationary")
        custom = self.root / "custom_out"
        rc, _, _ = _quiet_main([str(sdir), "--out", str(custom)])
        self.assertEqual(rc, 0)
        self.assertTrue((custom / "report.md").is_file())
        self.assertTrue((custom / "summary.csv").is_file())

    def test_warning_block_for_ignored_control(self):
        sdir = _build_session(self.root, "stat_ign", "stationary", autofocus_applied=False)
        self.assertEqual(_quiet_main([str(sdir)])[0], 0)
        md = self._read(sdir / "report" / "report.md")
        self.assertIn("WARNING", md)
        cam_section = _section(md, "## 2. Camera settings", "## 3. Stationary")
        self.assertIn("WARNING", cam_section)
        auto_row = [ln for ln in cam_section.splitlines() if ln.startswith("| Autofocus |")]
        self.assertTrue(auto_row)
        self.assertIn("not confirmed", auto_row[0])
        self.assertIn("reports auto", cam_section)  # session warning copied verbatim

    def test_no_warning_block_when_all_applied(self):
        sdir = _build_session(self.root, "stat_fine", "stationary")
        self.assertEqual(_quiet_main([str(sdir)])[0], 0)
        md = self._read(sdir / "report" / "report.md")
        self.assertNotIn("WARNING", md)
        self.assertIn("No camera control was reported as ignored.", md)

    def test_noise_floor_section_has_observed_numbers(self):
        sdir = _build_session(self.root, "stat_nf", "stationary")
        self.assertEqual(_quiet_main([str(sdir)])[0], 0)
        out = sdir / "report"
        md = self._read(out / "report.md")
        data = json.loads(self._read(out / "report.json"))
        row = data["noise_floor"]["rows"][0]
        resid = data["sessions"][0]["per_device"]["0"]["residual_unaligned"]
        self.assertAlmostEqual(row["mean"], resid["mean"])
        section = _section(md, "## 5. Noise floor", "## 6.")
        self.assertIn("observed noise floor, not a threshold", section.lower())
        self.assertIn(f"{row['mean']:.4f}", section)
        self.assertIn(f"{row['p95']:.4f}", section)
        self.assertIn(f"{row['max']:.4f}", section)

    def _assert_noise_floor_is_unaligned(self, sdir):
        self.assertEqual(_quiet_main([str(sdir)])[0], 0)
        out = sdir / "report"
        data = json.loads(self._read(out / "report.json"))
        row = data["noise_floor"]["rows"][0]
        gray, _ = analysis.load_session_gray(sdir, 0)
        expected = analysis.residual_scores(gray, align=False)["mean"]
        self.assertAlmostEqual(row["mean"], expected, places=9)
        self.assertEqual(data["noise_floor"]["basis"], report.NOISE_FLOOR_BASIS)
        try:
            aligned = analysis.residual_scores(gray, align=True)["mean"]
        except ValueError:
            aligned = None  # the aligned figure may be unavailable; it is not used
        if aligned is not None:
            self.assertNotAlmostEqual(row["mean"], aligned, places=6)
        md = self._read(out / "report.md")
        section = _section(md, "## 5. Noise floor", "## 6.")
        self.assertIn("no alignment", md)
        self.assertNotIn("whole-pixel alignment", md)
        self.assertIn("Alignment is NOT applied", section)
        self.assertIn("observed noise floor, not a threshold", section.lower())

    def test_noise_floor_unaligned_on_flat_pure_noise(self):
        flat = np.full((48, 64), 128.0)
        sdir = _build_session(self.root, "nf_flat", "stationary", base=flat,
                              frames=12, sigma=3.0, seed=11)
        self._assert_noise_floor_is_unaligned(sdir)

    def test_noise_floor_unaligned_on_weak_texture(self):
        yy, xx = np.mgrid[0:48, 0:64].astype(np.float64)
        weak = 120.0 + 0.05 * xx + 0.05 * yy
        sdir = _build_session(self.root, "nf_weak", "stationary", base=weak,
                              frames=12, sigma=3.0, seed=12)
        self._assert_noise_floor_is_unaligned(sdir)

    def test_stationary_flat_session_never_aligns(self):
        # Previously the aligned residual could raise ValueError here; it must not be used.
        real = analysis.residual_scores

        def guarded(gray, *args, **kwargs):
            if kwargs.get("align", True):
                raise ValueError("estimated shifts leave no overlapping area to compare")
            return real(gray, *args, **kwargs)

        sdir = _build_session(self.root, "stat_spur", "stationary",
                              base=np.full((48, 64), 128.0), frames=8, sigma=3.0, seed=2)
        with mock.patch.object(analysis, "residual_scores", side_effect=guarded):
            rc, _, err = _quiet_main([str(sdir)])
        self.assertEqual(rc, 0, err)
        data = json.loads(self._read(sdir / "report" / "report.json"))
        self.assertIsNone(data["sessions"][0]["per_device"]["0"].get("residual_aligned"))

    def test_drift_line_present_and_omitted_on_estimate_failure(self):
        sdir = _build_session(self.root, "stat_drift", "stationary")
        self.assertEqual(_quiet_main([str(sdir)])[0], 0)
        md = self._read(sdir / "report" / "report.md")
        self.assertIn("apparent drift/vibration", md)
        self.assertIn("itself noisy", md)
        with mock.patch.object(analysis, "estimate_shifts", side_effect=ValueError("no shifts")):
            rc, _, _ = _quiet_main([str(sdir), "--out", str(self.root / "drift_fail")])
        self.assertEqual(rc, 0)
        md = self._read(self.root / "drift_fail" / "report.md")
        self.assertNotIn("apparent drift/vibration", md)
        self.assertIn("## 5. Noise floor", md)

    def test_analysis_value_error_gives_clean_nonzero_exit(self):
        sdir = _build_session(self.root, "stat_bad", "stationary")
        with mock.patch.object(analysis, "sharpness_stats",
                               side_effect=ValueError("boom from analysis")):
            rc, out, err = _quiet_main([str(sdir)])
        self.assertEqual(rc, 2)
        self.assertIn("boom from analysis", err)
        self.assertNotIn("Traceback", err)
        self.assertFalse((sdir / "report" / "report.md").exists())

    def test_cannot_tell_section_always_present(self):
        stat = _build_session(self.root, "stat_c", "stationary")
        place = _build_session(self.root, "place_c", "placement",
                               shifts=[(0, 0), (2, 0), (0, 3), (-2, 1)])
        self.assertEqual(_quiet_main([str(stat)])[0], 0)
        self.assertEqual(_quiet_main([str(place), "--out", str(self.root / "p")])[0], 0)
        for md_path in (stat / "report" / "report.md", self.root / "p" / "report.md"):
            md = self._read(md_path)
            self.assertIn("## 6. What this test CANNOT tell you", md)
            self.assertIn("tools/thresholds.py", md)
            self.assertIn("[Quality to define]", md)
            self.assertIn("4-camera USB bandwidth", md)

    def test_no_pass_or_fail_words(self):
        a = _build_session(self.root, "stat_a", "stationary", seed=3)
        b = _build_session(self.root, "stat_b", "stationary", seed=4, autofocus_applied=False)
        out = self.root / "both"
        self.assertEqual(_quiet_main([str(a), str(b), "--out", str(out)])[0], 0)
        md = self._read(out / "report.md")
        self.assertIsNone(re.search(r"\b(pass|fail)", md, re.IGNORECASE))

    def test_placement_shift_spread_matches_known_shifts(self):
        shifts = [(0, 0), (3, 1), (-2, 0), (4, -2), (-3, 2), (1, 1)]
        sdir = _build_session(self.root, "place_s", "placement", shifts=shifts)
        self.assertEqual(_quiet_main([str(sdir)])[0], 0)
        data = json.loads(self._read(sdir / "report" / "report.json"))
        sh = data["sessions"][0]["per_device"]["0"]["shifts"]
        dx = np.array([s[0] for s in shifts], dtype=float)
        dy = np.array([s[1] for s in shifts], dtype=float)
        self.assertAlmostEqual(sh["max_abs_dx"], np.max(np.abs(dx)), delta=1.0)
        self.assertAlmostEqual(sh["max_abs_dy"], np.max(np.abs(dy)), delta=1.0)
        self.assertAlmostEqual(sh["std_dx"], np.std(dx), delta=1.0)
        self.assertAlmostEqual(sh["std_dy"], np.std(dy), delta=1.0)
        md = self._read(sdir / "report" / "report.md")
        self.assertIn("## 4. Placement results", md)
        self.assertIn("Residual with alignment", md)
        self.assertIn("Residual without alignment", md)

    def test_multi_session_comparison(self):
        a = _build_session(self.root, "cmp_a", "stationary", seed=5)
        b = _build_session(self.root, "cmp_b", "stationary", seed=6, frames=9)
        out = self.root / "cmp"
        self.assertEqual(_quiet_main([str(a), str(b), "--out", str(out)])[0], 0)
        data = json.loads(self._read(out / "report.json"))
        self.assertEqual(len(data["rows"]), 2)
        self.assertIsNotNone(data["comparison"])
        self.assertTrue(data["comparison"]["metrics"])
        md = self._read(out / "report.md")
        self.assertIn("## Comparison across sessions and devices", md)
        self.assertIn("cmp_a", md)
        self.assertIn("cmp_b", md)
        with open(out / "summary.csv", newline="", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        self.assertEqual(len(lines), 3)  # header + two rows

    def test_missing_npz_gives_clear_error(self):
        sdir = _build_session(self.root, "stat_miss", "stationary")
        (sdir / "frames_dev0.npz").unlink()
        with self.assertRaises(report.ReportInputError) as cm:
            report.summarize_session(sdir)
        self.assertIn("frames_dev0.npz", str(cm.exception))
        rc, _, err = _quiet_main([str(sdir)])
        self.assertEqual(rc, 2)
        self.assertIn("frames_dev0.npz", err)
        self.assertFalse((sdir / "report").exists())

    def test_missing_session_json_gives_clear_error(self):
        sdir = _build_session(self.root, "stat_nojson", "stationary")
        (sdir / "session.json").unlink()
        with self.assertRaises(report.ReportInputError) as cm:
            report.summarize_session(sdir)
        self.assertIn("session.json", str(cm.exception))

    @staticmethod
    def _table_cells(section, label):
        rows = [ln for ln in section.splitlines() if ln.startswith(f"| {label} |")]
        if not rows:
            raise AssertionError(f"no table row for {label!r}")
        return [c.strip() for c in rows[0].strip().strip("|").split("|")]

    def test_dark_first_frame_is_flagged_with_sensitivity_view(self):
        sdir = _build_session(self.root, "out_dark", "stationary",
                              offsets=[-40.0, 0, 0, 0, 0, 0])
        self.assertEqual(_quiet_main([str(sdir)])[0], 0)
        out = sdir / "report"
        data = json.loads(self._read(out / "report.json"))
        e = data["sessions"][0]["per_device"]["0"]
        self.assertEqual([f["index"] for f in e["outliers"]["flagged"]], [0])
        self.assertIsNotNone(e["excluding_flagged"])
        md = self._read(out / "report.md")
        self.assertIsNone(re.search(r"\b(pass|fail)", md, re.IGNORECASE))
        sec3 = _section(md, "## 3. Stationary results", "## 4. Placement results")
        self.assertIn("frame 0 (mean brightness", sec3)
        self.assertIn("Sensitivity view", sec3)
        self.assertIn("1 flagged frame changes these numbers", sec3)
        all_range = float(self._table_cells(sec3, "Brightness range (% of mean)")[1])
        excl_range = float(self._table_cells(sec3, "Brightness range (% of mean)")[2])
        self.assertLess(excl_range, all_range)
        for label in ("Temporal noise mean (gray levels)", "Temporal noise p95 (gray levels)",
                      "Residual, no alignment: mean", "Residual, no alignment: p95",
                      "Residual, no alignment: max"):
            cells = self._table_cells(sec3, label)
            self.assertLess(float(cells[2]), float(cells[1]), label)
        sec5 = _section(md, "## 5. Noise floor", "## 6.")
        self.assertIn("all frames (headline)", sec5)
        self.assertIn("excluding flagged frames (sensitivity)", sec5)
        with open(out / "summary.csv", newline="", encoding="utf-8") as fh:
            row = list(csv.DictReader(fh))[0]
        self.assertEqual(row["flagged_frames"], "1")
        self.assertEqual(row["flagged_frame_indices"], "0")
        self.assertTrue(row["excl_flagged_brightness_range_pct"])
        self.assertLess(float(row["excl_flagged_brightness_range_pct"]), all_range)

    def test_no_flagged_frames_means_no_sensitivity_view_and_empty_csv_columns(self):
        sdir = _build_session(self.root, "out_none", "stationary")
        self.assertEqual(_quiet_main([str(sdir)])[0], 0)
        out = sdir / "report"
        md = self._read(out / "report.md")
        self.assertIn("Flagged: none.", md)
        self.assertNotIn("Sensitivity view", md)
        self.assertNotIn("excluding flagged frames (sensitivity)", md)
        with open(out / "summary.csv", newline="", encoding="utf-8") as fh:
            row = list(csv.DictReader(fh))[0]
        self.assertEqual(row["flagged_frame_indices"], "")
        self.assertEqual(row["excl_flagged_brightness_range_pct"], "")
        self.assertEqual(row["excl_flagged_residual_unaligned_max"], "")

    def test_skip_first_removes_frame_zero_and_says_so(self):
        sdir = _build_session(self.root, "skip1", "stationary",
                              offsets=[-40.0, 0, 0, 0, 0, 0])
        self.assertEqual(_quiet_main([str(sdir), "--skip-first", "1"])[0], 0)
        out = sdir / "report"
        data = json.loads(self._read(out / "report.json"))
        self.assertEqual(data["settings"]["skip_first"], 1)
        e = data["sessions"][0]["per_device"]["0"]
        self.assertEqual(e["frames_used"], 5)
        self.assertEqual(len(e["brightness"]["mean_per_frame"]), 5)
        gray, _ = analysis.load_session_gray(sdir, 0)
        expected = analysis.brightness_stats(gray[1:])["range_pct"]
        self.assertAlmostEqual(e["brightness"]["range_pct"], expected, places=9)
        self.assertEqual(e["outliers"]["flagged"], [])
        md = self._read(out / "report.md")
        self.assertIn("Frames skipped at the start (--skip-first): 1", md)
        self.assertIn("Frames used: 5 of 6", md)
        self.assertNotIn("Sensitivity view", md)

    def test_skip_first_leaving_too_few_frames_is_an_input_error(self):
        sdir = _build_session(self.root, "skip_all", "stationary", frames=6)
        rc, _, err = _quiet_main([str(sdir), "--skip-first", "5"])
        self.assertEqual(rc, 2)
        self.assertIn("--skip-first 5", err)
        self.assertNotIn("Traceback", err)

    def test_noisy_flat_image_suppresses_drift_line_with_real_response(self):
        sdir = _build_session(self.root, "flat_noise", "stationary",
                              base=np.full((48, 64), 128.0), frames=12, sigma=3.0, seed=11)
        self.assertEqual(_quiet_main([str(sdir)])[0], 0)
        out = sdir / "report"
        data = json.loads(self._read(out / "report.json"))
        est = data["sessions"][0]["per_device"]["0"]["shift_estimate"]
        self.assertFalse(est["reported"])
        self.assertLess(est["median_response"], report.DEFAULT_MIN_SHIFT_RESPONSE)
        md = self._read(out / "report.md")
        sec3 = _section(md, "## 3. Stationary results", "## 4. Placement results")
        match = re.search(r"Apparent drift/vibration not reported: image too flat or too noisy "
                          r"to measure shift \(median peak response ([0-9.]+), needs at least "
                          r"0\.10\)\.", sec3)
        self.assertIsNotNone(match)
        self.assertAlmostEqual(float(match.group(1)), est["median_response"], delta=0.0006)
        self.assertNotIn("shift spread std dx", sec3)

    def test_textured_stable_image_still_shows_drift_line(self):
        sdir = _build_session(self.root, "tex_stable", "stationary", seed=21)
        self.assertEqual(_quiet_main([str(sdir)])[0], 0)
        out = sdir / "report"
        data = json.loads(self._read(out / "report.json"))
        est = data["sessions"][0]["per_device"]["0"]["shift_estimate"]
        self.assertTrue(est["reported"])
        self.assertGreaterEqual(est["median_response"], report.DEFAULT_MIN_SHIFT_RESPONSE)
        md = self._read(out / "report.md")
        sec3 = _section(md, "## 3. Stationary results", "## 4. Placement results")
        self.assertIn("- Apparent drift/vibration (estimate;", sec3)
        self.assertNotIn("not reported", sec3)

    def test_min_shift_response_option_suppresses_drift_line(self):
        sdir = _build_session(self.root, "tex_thresh", "stationary", seed=21)
        self.assertEqual(_quiet_main([str(sdir), "--min-shift-response", "1.01",
                                      "--out", str(self.root / "thr")])[0], 0)
        md = self._read(self.root / "thr" / "report.md")
        self.assertIn("Apparent drift/vibration not reported", md)
        self.assertIn("needs at least 1.01", md)

    def test_garbled_wording_is_gone_and_minus_one_is_explained(self):
        sdir = _build_session(self.root, "af_minus1", "stationary",
                              autofocus_applied=False, autofocus_readback=-1.0)
        self.assertEqual(_quiet_main([str(sdir)])[0], 0)
        md = self._read(sdir / "report" / "report.md")
        self.assertNotIn("requested not requested", md)
        self.assertNotIn("was requested not", md)
        self.assertIn("Autofocus was requested off but the camera did not confirm it "
                      "(read back: -1.0).", md)
        self.assertIn("does not expose this control", md)
        self.assertIn("so this does not prove auto is on", md)
        cam = _section(md, "## 2. Camera settings", "## 3. Stationary")
        self.assertIn("| Autofocus | off (always requested) |", cam)
        self.assertIn("| Auto exposure | off (always requested) |", cam)
        self.assertIn("| Focus | 30 |", cam)
        self.assertIn("| Gain | 10 |", cam)
        self.assertIsNone(re.search(r"\b(pass|fail)", md, re.IGNORECASE))

    def test_main_returns_zero_on_success(self):
        sdir = _build_session(self.root, "stat_rc", "stationary")
        self.assertEqual(report.main([str(sdir)]), 0)
        shutil.rmtree(sdir / "report", ignore_errors=True)


class ReportOptionTests(unittest.TestCase):
    def test_defaults_and_options_parse(self):
        parser = report.build_parser()
        args = parser.parse_args(["sess"])
        self.assertEqual(args.outlier_sigma, 5.0)
        self.assertEqual(args.outlier_floor_gray, 2.0)
        self.assertEqual(args.skip_first, 0)
        self.assertEqual(args.min_shift_response, 0.1)
        args = parser.parse_args(["sess", "--outlier-sigma", "3", "--outlier-floor-gray", "1.5",
                                  "--skip-first", "2", "--min-shift-response", "0.2"])
        self.assertEqual(args.outlier_sigma, 3.0)
        self.assertEqual(args.outlier_floor_gray, 1.5)
        self.assertEqual(args.skip_first, 2)
        self.assertEqual(args.min_shift_response, 0.2)

    def test_negative_or_non_numeric_values_are_rejected(self):
        parser = report.build_parser()
        for argv in (["sess", "--skip-first", "-1"], ["sess", "--outlier-sigma", "abc"],
                     ["sess", "--min-shift-response", "-0.5"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(argv)


class ToolDependencyTests(unittest.TestCase):
    def test_missing_numpy_raises_tool_dependency_error(self):
        with mock.patch.dict(sys.modules, {"numpy": None}):
            with self.assertRaises(ToolDependencyError):
                report.summarize_session(Path(tempfile.gettempdir()) / "no_such_session")


if __name__ == "__main__":
    unittest.main()
