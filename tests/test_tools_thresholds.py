"""Tests for tools/thresholds.py. Run from the project root:

    python3 -m unittest
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

from tools import thresholds as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(scores, label, classes=None):
    out = []
    for i, s in enumerate(scores):
        out.append(
            {
                "label": label,
                "score": float(s),
                "defect_class": classes[i] if classes else None,
                "part_id": None,
            }
        )
    return out


class LoadScoresTests(unittest.TestCase):
    def _write(self, tmp, text):
        path = os.path.join(tmp, "scores.csv")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_valid_file_normalises_label_and_optional_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                "label,score,defect_class,part_id\n"
                "GOOD,0.1,,p1\n"
                "Bad,0.8,scratch,p2\n",
            )
            got = th.load_scores(path)
        self.assertEqual([r["label"] for r in got], ["good", "bad"])
        self.assertEqual([r["score"] for r in got], [0.1, 0.8])
        self.assertIsNone(got[0]["defect_class"])
        self.assertEqual(got[1]["defect_class"], "scratch")
        self.assertEqual(got[1]["part_id"], "p2")

    def test_optional_columns_may_be_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "label,score\ngood,0.2\n")
            got = th.load_scores(path)
        self.assertIsNone(got[0]["defect_class"])
        self.assertIsNone(got[0]["part_id"])

    def test_bad_label_reports_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "label,score\ngood,0.1\nmaybe,0.2\n")
            with self.assertRaisesRegex(th.ThresholdInputError, "line 3.*maybe"):
                th.load_scores(path)

    def test_missing_column_is_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "label,value\ngood,0.1\n")
            with self.assertRaisesRegex(th.ThresholdInputError, "missing required column.*score"):
                th.load_scores(path)

    def test_non_numeric_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "label,score\ngood,abc\n")
            with self.assertRaisesRegex(th.ThresholdInputError, "not a number"):
                th.load_scores(path)

    def test_empty_score_is_non_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "label,score\ngood,\n")
            with self.assertRaisesRegex(th.ThresholdInputError, "not a number"):
                th.load_scores(path)

    def test_nan_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "label,score\nbad,nan\n")
            with self.assertRaisesRegex(th.ThresholdInputError, "NaN"):
                th.load_scores(path)

    def test_scores_outside_unit_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in ("1.5", "-0.1", "inf"):
                path = self._write(tmp, f"label,score\ngood,{value}\n")
                with self.subTest(value=value):
                    with self.assertRaisesRegex(th.ThresholdInputError, "outside 0..1"):
                        th.load_scores(path)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "")
            with self.assertRaisesRegex(th.ThresholdInputError, "empty file"):
                th.load_scores(path)

    def test_unreadable_file(self):
        with self.assertRaisesRegex(th.ThresholdInputError, "cannot read"):
            th.load_scores(os.path.join(ROOT, "no_such_scores_file.csv"))


class WilsonAndRuleOfThreeTests(unittest.TestCase):
    def test_wilson_zero_of_ten(self):
        lo, hi = th.wilson_interval(0, 10)
        self.assertEqual(round(lo, 3), 0.0)
        self.assertEqual(round(hi, 3), 0.278)

    def test_wilson_five_of_ten(self):
        lo, hi = th.wilson_interval(5, 10)
        self.assertEqual(round(lo, 3), 0.237)
        self.assertEqual(round(hi, 3), 0.763)

    def test_wilson_n_zero(self):
        self.assertEqual(th.wilson_interval(0, 0), (0.0, 1.0))

    def test_wilson_rejects_k_above_n(self):
        with self.assertRaises(ValueError):
            th.wilson_interval(11, 10)

    def test_rule_of_three(self):
        self.assertAlmostEqual(th.rule_of_three_upper(30), 0.1)
        self.assertAlmostEqual(th.rule_of_three_upper(4), 0.75)

    def test_rule_of_three_n_zero_and_capped(self):
        self.assertEqual(th.rule_of_three_upper(0), 1.0)
        self.assertEqual(th.rule_of_three_upper(2), 1.0)


class CandidateTableTests(unittest.TestCase):
    GOOD = [0.1, 0.2, 0.6]
    BAD = [0.5, 0.7, 0.9]

    def _row(self, table, t):
        return next(r for r in table if r["threshold"] == t)

    def test_candidates_sorted_and_include_zero_and_top(self):
        table = th.candidate_table(self.GOOD, self.BAD)
        thresholds = [r["threshold"] for r in table]
        self.assertEqual(thresholds, sorted(thresholds))
        self.assertEqual(thresholds[0], 0.0)
        self.assertGreater(thresholds[-1], 0.9)
        self.assertEqual(len(thresholds), 8)  # 0.0 + 6 distinct observed + top

    def test_escape_and_false_reject_counts(self):
        table = th.candidate_table(self.GOOD, self.BAD)
        r = self._row(table, 0.5)
        self.assertEqual((r["escapes"], r["false_rejects"]), (0, 1))
        r = self._row(table, 0.7)
        self.assertEqual((r["escapes"], r["false_rejects"]), (1, 0))
        self.assertAlmostEqual(r["escape_rate"], 1 / 3)
        r = self._row(table, 0.0)
        self.assertEqual((r["escapes"], r["false_rejects"]), (0, 3))
        top = table[-1]
        self.assertEqual((top["escapes"], top["false_rejects"]), (3, 0))

    def test_rates_have_wilson_intervals(self):
        r = self._row(th.candidate_table(self.GOOD, self.BAD), 0.7)
        lo, hi = th.wilson_interval(1, 3)
        self.assertAlmostEqual(r["escape_ci_low"], lo)
        self.assertAlmostEqual(r["escape_ci_high"], hi)


class SuggestTests(unittest.TestCase):
    NOTE = (
        "Candidate values from labelled samples only. They are not approved thresholds; "
        "Quality must approve them and confirm the sample counts are adequate."
    )

    def test_refuses_below_minimums_and_gives_no_numbers(self):
        good = rows([0.1, 0.2, 0.3], "good")
        bad = rows([0.8, 0.9, 0.95, 0.99], "bad")
        result = th.suggest(good, bad, min_good=5, min_bad=3, review_from_good_percentile=90)
        self.assertFalse(result["ok"])
        self.assertIn("good 3 (minimum 5)", result["reason"])
        for key in ("t_img", "t_low", "table", "per_class", "at_t_img", "separable"):
            self.assertNotIn(key, result)
        self.assertEqual(result["note"], self.NOTE)

    def test_refuses_when_bad_below_minimum(self):
        good = rows([0.1, 0.2, 0.3, 0.4], "good")
        bad = rows([0.8], "bad")
        result = th.suggest(good, bad, min_good=3, min_bad=2, review_from_good_percentile=90)
        self.assertFalse(result["ok"])
        self.assertIn("bad 1 (minimum 2)", result["reason"])
        self.assertNotIn("t_img", result)

    def test_min_arguments_are_required(self):
        with self.assertRaises(TypeError):
            th.suggest([], [], review_from_good_percentile=90)  # no min_good / min_bad

    def test_separable_gives_t_img_min_bad_and_zero_escapes(self):
        good = rows([0.1, 0.2, 0.3], "good")
        bad = rows([0.6, 0.8, 0.9, 0.95], "bad")
        result = th.suggest(good, bad, min_good=3, min_bad=4, review_from_good_percentile=100)
        self.assertTrue(result["ok"])
        self.assertTrue(result["separable"])
        self.assertEqual(result["t_img"], 0.6)
        self.assertEqual(result["at_t_img"]["escapes"], 0)
        self.assertEqual(result["at_t_img"]["false_rejects"], 0)
        self.assertAlmostEqual(result["rule_of_three_escape_upper"], 0.75)
        self.assertEqual(result["note"], self.NOTE)

    def test_overlap_is_not_separable_and_gives_no_single_t_img(self):
        good = rows([0.1, 0.5, 0.7], "good")
        bad = rows([0.4, 0.9], "bad")
        result = th.suggest(good, bad, min_good=3, min_bad=2, review_from_good_percentile=90)
        self.assertTrue(result["ok"])
        self.assertFalse(result["separable"])
        self.assertIsNone(result["t_img"])
        self.assertIsNone(result["t_low"])
        self.assertEqual(result["overlap"]["low"], 0.4)
        self.assertEqual(result["overlap"]["high"], 0.7)
        self.assertEqual(result["overlap"]["good"], 2)
        self.assertEqual(result["overlap"]["bad"], 1)
        thresholds = [r["threshold"] for r in result["table"]]
        self.assertIn(0.4, thresholds)
        self.assertEqual(result["note"], self.NOTE)

    def test_t_low_percentile_nearest_rank(self):
        good_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        good = rows(good_scores, "good")
        bad = rows([0.97, 0.98, 0.99], "bad")
        expected = {0: 0.1, 50: 0.5, 90: 0.9, 95: 0.95, 100: 0.95}
        for p, want in expected.items():
            with self.subTest(percentile=p):
                result = th.suggest(good, bad, min_good=10, min_bad=3, review_from_good_percentile=p)
                self.assertEqual(result["t_low"], want)
                self.assertFalse(result["t_low_capped"])
                self.assertEqual(result["t_img"], 0.97)

    def test_pair_counts_for_candidate_pair(self):
        good = rows([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], "good")
        bad = rows([0.97, 0.98, 0.99], "bad")
        result = th.suggest(good, bad, min_good=10, min_bad=3, review_from_good_percentile=90)
        self.assertEqual(result["verdict_counts"]["good_parts"], {"GOOD": 8, "REVIEW": 2, "BAD": 0})
        self.assertEqual(result["verdict_counts"]["bad_parts"], {"GOOD": 0, "REVIEW": 0, "BAD": 3})

    def test_t_low_cap_behaviour(self):
        # Percentile above t_img is capped to t_img and flagged.
        self.assertEqual(th.t_low_candidate([0.1, 0.2, 0.5], 100, 0.3), (0.3, True))
        self.assertEqual(th.t_low_candidate([0.1, 0.2, 0.5], 50, 0.3), (0.2, False))

    def test_per_class_recall_and_bounds(self):
        bad_rows = [
            {"label": "bad", "score": 0.7, "defect_class": "scratch"},
            {"label": "bad", "score": 0.9, "defect_class": "scratch"},
            {"label": "bad", "score": 0.85, "defect_class": "scratch"},
            {"label": "bad", "score": 0.95, "defect_class": "dent"},
        ]
        got = th.per_class_recall(bad_rows, 0.8)
        by_name = {c["defect_class"]: c for c in got}
        self.assertEqual(by_name["scratch"]["n_bad"], 3)
        self.assertEqual(by_name["scratch"]["caught"], 2)
        self.assertAlmostEqual(by_name["scratch"]["recall"], 2 / 3)
        self.assertEqual(by_name["dent"]["n_bad"], 1)
        self.assertEqual(by_name["dent"]["recall"], 1.0)
        self.assertEqual(by_name["dent"]["miss_rule_of_three_upper"], 1.0)

    def test_per_class_unspecified_and_suggest_per_class(self):
        self.assertEqual(
            th.per_class_recall([{"label": "bad", "score": 0.5, "defect_class": None}], 0.5)[0]["defect_class"],
            th.UNSPECIFIED_CLASS,
        )
        good = rows([0.1, 0.2, 0.3], "good")
        bad = rows([0.6, 0.7, 0.8, 0.9], "bad", classes=["scratch", "scratch", "scratch", "scratch"])
        result = th.suggest(good, bad, min_good=3, min_bad=4, review_from_good_percentile=90)
        self.assertEqual(result["per_class"][0]["n_bad"], 4)
        self.assertAlmostEqual(result["per_class"][0]["miss_rule_of_three_upper"], 0.75)
        no_class = th.suggest(good, rows([0.6, 0.7, 0.8, 0.9], "bad"), min_good=3, min_bad=4,
                              review_from_good_percentile=90)
        self.assertIsNone(no_class["per_class"])


PER_CLASS_LABEL_TEXT = (
    "Per-class sample counts and rule-of-three bounds (recall is 100% at t_img = min(bad) by "
    "construction, so it is not evidence of detection quality on new parts)"
)


class ThresholdEvaluationTests(unittest.TestCase):
    # Hand-made data. Good max 0.9 > bad min 0.2, so the sample is not separable.
    GOOD = [0.1, 0.2, 0.3, 0.4, 0.9]
    BAD = [
        {"label": "bad", "score": 0.5, "defect_class": "scratch"},
        {"label": "bad", "score": 0.7, "defect_class": "scratch"},
        {"label": "bad", "score": 0.8, "defect_class": "scratch"},
        {"label": "bad", "score": 0.6, "defect_class": "dent"},
        {"label": "bad", "score": 0.2, "defect_class": "dent"},
    ]

    def _suggest(self, good_scores, bad, **kw):
        return th.suggest(rows(good_scores, "good"), bad, min_good=5, min_bad=5,
                          review_from_good_percentile=90, **kw)

    def test_evaluation_numbers_on_hand_made_data(self):
        # t_img = 0.6: BAD if score >= 0.6.
        # scratch: 0.5 escapes, 0.7 and 0.8 caught -> recall 2/3.
        # dent: 0.2 escapes, 0.6 caught (boundary counts as caught) -> recall 1/2.
        # overall: 3 of 5 caught, 2 escapes. good: 0.9 is a false reject (1 of 5).
        result = self._suggest(self.GOOD, self.BAD, t_img=0.6)
        self.assertTrue(result["ok"])
        self.assertFalse(result["separable"])
        ev = result["evaluation"]
        self.assertEqual(ev["t_img"], 0.6)
        self.assertEqual((ev["bad"]["n"], ev["bad"]["caught"], ev["bad"]["escapes"]), (5, 3, 2))
        self.assertAlmostEqual(ev["bad"]["recall"], 0.6)
        self.assertEqual((ev["good"]["n"], ev["good"]["false_rejects"]), (5, 1))
        self.assertAlmostEqual(ev["good"]["false_reject_rate"], 0.2)
        self.assertEqual(ev["bad"]["recall_ci_low"], th.wilson_interval(3, 5)[0])
        self.assertEqual(ev["bad"]["recall_ci_high"], th.wilson_interval(3, 5)[1])
        self.assertEqual(ev["good"]["false_reject_ci_high"], th.wilson_interval(1, 5)[1])

        pc = {c["defect_class"]: c for c in ev["per_class"]}
        self.assertEqual(sorted(pc), ["dent", "scratch"])
        self.assertEqual((pc["scratch"]["n_bad"], pc["scratch"]["caught"], pc["scratch"]["escapes"]), (3, 2, 1))
        self.assertAlmostEqual(pc["scratch"]["recall"], 2 / 3)
        self.assertEqual(pc["scratch"]["recall_ci_low"], th.wilson_interval(2, 3)[0])
        self.assertEqual(pc["scratch"]["recall_ci_high"], th.wilson_interval(2, 3)[1])
        self.assertEqual((pc["dent"]["n_bad"], pc["dent"]["caught"], pc["dent"]["escapes"]), (2, 1, 1))
        self.assertAlmostEqual(pc["dent"]["recall"], 0.5)
        self.assertEqual(pc["dent"]["recall_ci_high"], th.wilson_interval(1, 2)[1])

    def test_evaluation_on_separable_sample_without_classes(self):
        good = rows([0.1, 0.2, 0.3], "good")
        bad = rows([0.6, 0.8, 0.9, 0.95], "bad")
        at_06 = th.suggest(good, bad, min_good=3, min_bad=4, review_from_good_percentile=100, t_img=0.6)
        self.assertTrue(at_06["separable"])
        self.assertEqual(at_06["evaluation"]["bad"]["escapes"], 0)
        self.assertEqual(at_06["evaluation"]["good"]["false_rejects"], 0)
        self.assertIsNone(at_06["evaluation"]["per_class"])
        at_07 = th.suggest(good, bad, min_good=3, min_bad=4, review_from_good_percentile=100, t_img=0.7)
        self.assertEqual(at_07["evaluation"]["bad"]["escapes"], 1)  # 0.6 escapes
        self.assertAlmostEqual(at_07["evaluation"]["bad"]["recall"], 0.75)

    def test_no_t_img_means_no_evaluation_and_same_result(self):
        good = rows([0.1, 0.2, 0.3], "good")
        bad = rows([0.6, 0.7, 0.8, 0.9], "bad", classes=["scratch"] * 4)
        plain = th.suggest(good, bad, min_good=3, min_bad=4, review_from_good_percentile=90)
        explicit_none = th.suggest(good, bad, min_good=3, min_bad=4, review_from_good_percentile=90, t_img=None)
        self.assertEqual(plain, explicit_none)
        self.assertNotIn("evaluation", plain)
        self.assertEqual(plain["per_class"], [{"defect_class": "scratch", "n_bad": 4,
                                               "miss_rule_of_three_upper": 0.75}])

    def test_t_img_validation(self):
        for bad_value in (1.5, -0.1, float("nan"), True, "0.5"):
            with self.subTest(t_img=bad_value), self.assertRaises(ValueError):
                self._suggest(self.GOOD, self.BAD, t_img=bad_value)

    def test_refused_result_has_no_evaluation(self):
        result = th.suggest(rows(self.GOOD, "good"), self.BAD, min_good=50, min_bad=5,
                            review_from_good_percentile=90, t_img=0.6)
        self.assertFalse(result["ok"])
        self.assertNotIn("evaluation", result)

    def test_min_arguments_still_required_with_t_img(self):
        with self.assertRaises(TypeError):
            th.suggest(self.GOOD, self.BAD, min_bad=5, review_from_good_percentile=90, t_img=0.6)

    def test_report_label_and_evaluation_section(self):
        good = rows([0.1, 0.2, 0.3], "good")
        bad = rows([0.6, 0.7, 0.8, 0.9], "bad", classes=["scratch"] * 4)
        plain = th.suggest(good, bad, min_good=3, min_bad=4, review_from_good_percentile=90)
        report = th.render_report(plain, source="s.csv", model_version=None, n_rows=7)
        self.assertIn("### " + PER_CLASS_LABEL_TEXT, report)
        self.assertNotIn("recall at t_img", report)
        self.assertIn(SuggestTests.NOTE, report)
        self.assertNotIn("Evaluation of a chosen threshold", report)

        with_eval = th.suggest(good, bad, min_good=3, min_bad=4, review_from_good_percentile=90, t_img=0.7)
        report = th.render_report(with_eval, source="s.csv", model_version=None, n_rows=7)
        self.assertIn("## Evaluation of a chosen threshold", report)
        self.assertIn("Evaluation of t_img = `0.7`", report)
        self.assertIn("not a recommended value", report)
        self.assertIn(SuggestTests.NOTE, report)

    def test_cli_evaluate_flag_writes_evaluation_and_absent_flag_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scores.csv")
            lines = ["label,score,defect_class"]
            lines += [f"good,{s}," for s in (0.05, 0.1, 0.12, 0.2, 0.25, 0.3, 0.33, 0.4, 0.45, 0.5)]
            lines += ["bad,0.6,scratch", "bad,0.7,dent", "bad,0.75,scratch", "bad,0.8,dent", "bad,0.9,scratch"]
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            base = [path, "--min-good", "5", "--min-bad", "5", "--review-from-good-percentile", "90"]

            out_md, out_json = os.path.join(tmp, "a.md"), os.path.join(tmp, "a.json")
            with contextlib.redirect_stdout(io.StringIO()):
                code = th.main(base + ["--evaluate-t-img", "0.6", "--out", out_md, "--json", out_json])
            self.assertEqual(code, 0)
            with open(out_md, encoding="utf-8") as fh:
                report = fh.read()
            with open(out_json, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertIn("## Evaluation of a chosen threshold", report)
            self.assertEqual(data["evaluation"]["bad"]["escapes"], 0)
            self.assertEqual(data["evaluation"]["good"]["false_rejects"], 0)
            self.assertEqual(data["evaluation"]["bad"]["recall"], 1.0)

            out_md2, out_json2 = os.path.join(tmp, "b.md"), os.path.join(tmp, "b.json")
            with contextlib.redirect_stdout(io.StringIO()):
                code = th.main(base + ["--out", out_md2, "--json", out_json2])
            self.assertEqual(code, 0)
            with open(out_md2, encoding="utf-8") as fh:
                report2 = fh.read()
            with open(out_json2, encoding="utf-8") as fh:
                data2 = json.load(fh)
            self.assertNotIn("Evaluation of a chosen threshold", report2)
            self.assertNotIn("evaluation", data2)
            self.assertIn(PER_CLASS_LABEL_TEXT, report2)

    def test_cli_evaluate_out_of_range_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("label,score\ngood,0.1\ngood,0.2\nbad,0.8\nbad,0.9\n")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = th.main([path, "--min-good", "1", "--min-bad", "1",
                                "--review-from-good-percentile", "90", "--evaluate-t-img", "1.5"])
        self.assertEqual(code, 1)
        self.assertIn("t_img", err.getvalue())


class CliTests(unittest.TestCase):
    def _csv(self, tmp):
        path = os.path.join(tmp, "scores.csv")
        lines = ["label,score,defect_class,part_id"]
        for i, s in enumerate([0.05, 0.1, 0.12, 0.2, 0.25, 0.3, 0.33, 0.4, 0.45, 0.5]):
            lines.append(f"good,{s},,g{i}")
        for i, (s, c) in enumerate([(0.6, "scratch"), (0.7, "dent"), (0.75, "scratch"),
                                    (0.8, "dent"), (0.9, "scratch")]):
            lines.append(f"bad,{s},{c},b{i}")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return path

    def test_missing_required_arguments_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._csv(tmp)
            base = [sys.executable, "-m", "tools.thresholds", path]
            full = ["--min-good", "5", "--min-bad", "5", "--review-from-good-percentile", "90"]
            for drop in range(0, len(full), 2):
                args = full[:drop] + full[drop + 2:]
                with self.subTest(missing=full[drop]):
                    proc = subprocess.run(base + args, cwd=ROOT, capture_output=True, text=True)
                    self.assertEqual(proc.returncode, 2)
                    self.assertIn("required", proc.stderr)

    def test_cli_end_to_end_writes_report_with_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._csv(tmp)
            out_md = os.path.join(tmp, "report.md")
            out_json = os.path.join(tmp, "result.json")
            proc = subprocess.run(
                [sys.executable, "-m", "tools.thresholds", path,
                 "--min-good", "5", "--min-bad", "5", "--review-from-good-percentile", "90",
                 "--model-version", "test-model-1", "--out", out_md, "--json", out_json],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out_md, encoding="utf-8") as fh:
                report = fh.read()
            with open(out_json, encoding="utf-8") as fh:
                data = json.load(fh)
        self.assertIn(SuggestTests.NOTE, report)
        self.assertIn(SuggestTests.NOTE, proc.stdout)
        self.assertIn("test-model-1", report)
        self.assertIn(th.PER_CLASS_COUNTS_LABEL, report)
        self.assertEqual(data["note"], SuggestTests.NOTE)
        self.assertEqual(data["t_img"], 0.6)

    def test_cli_refusal_still_writes_report_without_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._csv(tmp)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = th.main([path, "--min-good", "50", "--min-bad", "5",
                                "--review-from-good-percentile", "90"])
        self.assertEqual(code, 0)
        self.assertIn("Refused", out.getvalue())
        self.assertIn("NOT RECORDED", out.getvalue())
        self.assertIn(SuggestTests.NOTE, out.getvalue())

    def test_cli_bad_input_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("label,score\ngood,2.0\n")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = th.main([path, "--min-good", "1", "--min-bad", "1",
                                "--review-from-good-percentile", "90"])
        self.assertEqual(code, 1)
        self.assertIn("outside 0..1", err.getvalue())


if __name__ == "__main__":
    unittest.main()
