"""Tests for station/fusion.py. Run from the project root:

    python3 -m unittest
"""
import math
import unittest

from station.contracts import (
    IMAGES_PER_PART,
    Defect,
    ImageResult,
    Side,
    Thresholds,
    Verdict,
)
from station.fusion import classify_image, fuse, worst_defects

TH = Thresholds(t_low=0.40, t_img=0.70)


def img(cam: str, score: float, side: Side = Side.TOP, defects=()) -> ImageResult:
    return ImageResult(camera_id=cam, side=side, score=score, defects=tuple(defects))


def part(scores: dict) -> list:
    """Build 8 results (C1..C4 x TOP/BOTTOM). `scores` maps (cam, side) -> score."""
    out = []
    for side in (Side.TOP, Side.BOTTOM):
        for cam in ("C1", "C2", "C3", "C4"):
            out.append(img(cam, scores.get((cam, side), 0.0), side))
    return out


def all_good_part() -> list:
    return part({})


class ClassifyImageTests(unittest.TestCase):
    def test_exact_t_low_is_review(self):
        self.assertIs(classify_image(img("C1", TH.t_low), TH), Verdict.REVIEW)

    def test_exact_t_img_is_bad(self):
        self.assertIs(classify_image(img("C1", TH.t_img), TH), Verdict.BAD)

    def test_just_below_t_low_is_good(self):
        score = math.nextafter(TH.t_low, 0.0)
        self.assertIs(classify_image(img("C1", score), TH), Verdict.GOOD)

    def test_just_below_t_img_is_review(self):
        score = math.nextafter(TH.t_img, 0.0)
        self.assertIs(classify_image(img("C1", score), TH), Verdict.REVIEW)

    def test_zero_is_good(self):
        self.assertIs(classify_image(img("C1", 0.0), TH), Verdict.GOOD)

    def test_review_band_is_never_good(self):
        for score in (TH.t_low, 0.5, math.nextafter(TH.t_img, 0.0)):
            self.assertIsNot(classify_image(img("C1", score), TH), Verdict.GOOD)

    def test_thresholds_are_not_hardcoded(self):
        other = Thresholds(t_low=0.10, t_img=0.20)
        self.assertIs(classify_image(img("C1", 0.15), other), Verdict.REVIEW)
        self.assertIs(classify_image(img("C1", 0.15), TH), Verdict.GOOD)


class FuseTests(unittest.TestCase):
    def test_all_good_is_good(self):
        pv = fuse(all_good_part(), TH)
        self.assertIs(pv.verdict, Verdict.GOOD)
        self.assertEqual(pv.reason, "no image above T_low")

    def test_bad_beats_review(self):
        results = part({("C1", Side.TOP): 0.45, ("C2", Side.BOTTOM): 0.80})
        pv = fuse(results, TH)
        self.assertIs(pv.verdict, Verdict.BAD)

    def test_one_review_among_seven_good_is_review(self):
        results = part({("C3", Side.BOTTOM): 0.40})
        pv = fuse(results, TH)
        self.assertIs(pv.verdict, Verdict.REVIEW)

    def test_review_is_never_returned_as_good(self):
        for score in (TH.t_low, 0.55, math.nextafter(TH.t_img, 0.0)):
            results = part({("C4", Side.TOP): score})
            self.assertIsNot(fuse(results, TH).verdict, Verdict.GOOD)

    def test_exact_t_img_single_image_is_bad(self):
        results = part({("C2", Side.TOP): TH.t_img})
        self.assertIs(fuse(results, TH).verdict, Verdict.BAD)

    def test_just_below_t_low_everywhere_is_good(self):
        score = math.nextafter(TH.t_low, 0.0)
        results = part({(c, s): score for c in ("C1", "C2", "C3", "C4")
                        for s in (Side.TOP, Side.BOTTOM)})
        self.assertIs(fuse(results, TH).verdict, Verdict.GOOD)

    def test_wrong_count_raises(self):
        base = all_good_part()
        for n in (7, 9, 0, 1):
            with self.subTest(n=n):
                seq = (base * 2)[:n]
                with self.assertRaises(ValueError):
                    fuse(seq, TH)

    def test_correct_count_accepted(self):
        self.assertEqual(len(all_good_part()), IMAGES_PER_PART)
        fuse(all_good_part(), TH)  # does not raise

    def test_order_preserved(self):
        results = part({("C1", Side.TOP): 0.5, ("C4", Side.BOTTOM): 0.1})
        pv = fuse(results, TH)
        self.assertEqual(pv.results, tuple(results))
        self.assertEqual(
            [(r.camera_id, r.side) for r in pv.results],
            [(r.camera_id, r.side) for r in results],
        )
        # Reversed input comes back reversed, not sorted.
        rev = list(reversed(results))
        self.assertEqual(fuse(rev, TH).results, tuple(rev))

    def test_reason_names_worst_camera_bad(self):
        results = part({("C1", Side.TOP): 0.75, ("C3", Side.BOTTOM): 0.95})
        pv = fuse(results, TH)
        self.assertIn("C3", pv.reason)
        self.assertIn("BOTTOM", pv.reason)
        self.assertIn("0.950", pv.reason)
        self.assertNotIn("C1", pv.reason)

    def test_reason_names_worst_camera_review(self):
        results = part({("C2", Side.TOP): 0.41, ("C4", Side.TOP): 0.52})
        pv = fuse(results, TH)
        self.assertIs(pv.verdict, Verdict.REVIEW)
        self.assertIn("C4", pv.reason)
        self.assertNotIn("C2", pv.reason)

    def test_reason_ignores_review_images_when_bad_present(self):
        results = part({("C1", Side.TOP): 0.99, ("C2", Side.TOP): 0.50})
        pv = fuse(results, TH)
        self.assertIs(pv.verdict, Verdict.BAD)
        self.assertIn("C1", pv.reason)
        self.assertNotIn("C2", pv.reason)

    def test_reason_lists_defect_classes(self):
        results = part({
            ("C1", Side.TOP): 0.80,
            ("C2", Side.TOP): 0.30,
        })
        results[0] = img("C1", 0.80, Side.TOP,
                         defects=[Defect("scratch", 0.80, (0, 0, 5, 5))])
        results[1] = img("C2", 0.30, Side.TOP,
                         defects=[Defect("dent", 0.30, (0, 0, 5, 5))])
        pv = fuse(results, TH)
        self.assertIs(pv.verdict, Verdict.BAD)
        self.assertIn("scratch", pv.reason)
        self.assertNotIn("dent", pv.reason)

    def test_good_reason_text(self):
        self.assertEqual(fuse(all_good_part(), TH).reason, "no image above T_low")


class ThresholdsTests(unittest.TestCase):
    def test_t_low_above_t_img_raises(self):
        with self.assertRaises(ValueError):
            Thresholds(t_low=0.8, t_img=0.5)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            Thresholds(t_low=-0.1, t_img=0.5)
        with self.assertRaises(ValueError):
            Thresholds(t_low=0.1, t_img=1.5)

    def test_equal_thresholds_allowed(self):
        Thresholds(t_low=0.5, t_img=0.5)


class WorstDefectsTests(unittest.TestCase):
    def test_sorted_descending_across_images(self):
        results = [
            img("C1", 0.3, defects=[Defect("a", 0.3, (0, 0, 1, 1))]),
            img("C2", 0.9, defects=[Defect("b", 0.9, (0, 0, 1, 1)),
                                    Defect("c", 0.5, (0, 0, 1, 1))]),
            img("C3", 0.0),
        ]
        scores = [d.score for d in worst_defects(results)]
        self.assertEqual(scores, [0.9, 0.5, 0.3])

    def test_empty_when_no_defects(self):
        self.assertEqual(worst_defects(all_good_part()), [])


if __name__ == "__main__":
    unittest.main()
