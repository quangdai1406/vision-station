"""Tests for station/remote_model.py: the station's client for the laptop inference service (S2)."""
from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from station.contracts import Defect, Frame, ImageResult, Side, StationError, Verdict
from station.fusion import classify_image
from station.contracts import Thresholds
from station.remote_model import RemoteModel
from tests.fake_laptop import FAKE_LAPTOP_TOKEN, FakeLaptop

TOKEN_ENV = "VISION_TEST_REMOTE_TOKEN"
PART = "PN-A"


def frame(camera: str = "C1", side: Side = Side.TOP) -> Frame:
    return Frame(camera_id=camera, side=side, seq=0, timestamp_utc="2026-11-04T08:00:00Z",
                 data=b"\x89PNG-fake-picture-bytes", ext="png")


class RemoteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.laptop = FakeLaptop(models={PART: "2026-11-01-a"})
        self.addCleanup(self.laptop.close)
        patcher = mock.patch.dict(os.environ, {TOKEN_ENV: FAKE_LAPTOP_TOKEN})
        patcher.start()
        self.addCleanup(patcher.stop)

    def model(self, timeout_s: float = 5.0, part: str = PART) -> RemoteModel:
        return RemoteModel(self.laptop.base, TOKEN_ENV, timeout_s, part)


class PredictSuccessTests(RemoteTestCase):
    def test_success_maps_the_reply_to_an_image_result(self) -> None:
        self.laptop.predict_score = 0.25
        self.laptop.predict_fields = {"defects": [
            {"defect_class": "scratch", "score": 0.25, "box": [10, 20, 30, 40]},
        ]}
        result = self.model().predict(frame("C2", Side.BOTTOM))
        self.assertIsInstance(result, ImageResult)
        self.assertEqual(result.camera_id, "C2")
        self.assertEqual(result.side, Side.BOTTOM)
        self.assertEqual(result.score, 0.25)
        self.assertEqual(result.defects, (Defect("scratch", 0.25, (10, 20, 30, 40)),))

    def test_request_carries_the_part_camera_side_and_raw_picture(self) -> None:
        model = self.model()
        model.predict(frame("C3", Side.TOP))
        method, path, headers = self.laptop.requests[-1]
        self.assertEqual((method, path), ("POST", "/v1/predict"))
        self.assertEqual(headers["Authorization"], f"Bearer {FAKE_LAPTOP_TOKEN}")
        self.assertEqual(headers["Part-Number"], PART)
        self.assertEqual(headers["Camera-Id"], "C3")
        self.assertEqual(headers["Side"], "TOP")
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertEqual(self.laptop.last_predict_body, b"\x89PNG-fake-picture-bytes")

    def test_version_comes_from_the_first_reply_when_not_checked_first(self) -> None:
        model = self.model()
        self.assertEqual(model.version, "")
        model.predict(frame())
        self.assertEqual(model.version, "2026-11-01-a")

    def test_approval_check_sets_the_version_from_health(self) -> None:
        model = self.model()
        self.assertEqual(model.check_approved(), "2026-11-01-a")
        self.assertEqual(model.version, "2026-11-01-a")
        self.assertEqual(self.laptop.requests[-1][0], "GET")


class ApprovalTests(RemoteTestCase):
    def test_part_not_listed_by_the_laptop_raises(self) -> None:
        with self.assertRaisesRegex(StationError, "does not list an approved model for part PN-X"):
            self.model(part="PN-X").check_approved()

    def test_health_error_status_raises(self) -> None:
        self.laptop.health_status = 503
        self.laptop.health_raw = b'{"error": "no"}'
        with self.assertRaises(StationError):
            self.model().check_approved()

    def test_unreachable_laptop_raises_station_error(self) -> None:
        self.laptop.close()
        with self.assertRaises(StationError):
            self.model().check_approved()


class FailureTests(RemoteTestCase):
    def test_timeout_raises_station_error(self) -> None:
        self.laptop.delay = 2.0
        with self.assertRaisesRegex(StationError, "timed out"):
            self.model(timeout_s=0.2).predict(frame())

    def test_server_error_raises_station_error(self) -> None:
        self.laptop.predict_status = 500
        with self.assertRaisesRegex(StationError, "500"):
            self.model().predict(frame())

    def test_503_raises_station_error_with_the_laptop_message(self) -> None:
        self.laptop.predict_status = 503
        with self.assertRaisesRegex(StationError, "GPU out of memory"):
            self.model().predict(frame())

    def test_bad_json_raises_station_error(self) -> None:
        self.laptop.predict_raw = b"<html>not json</html>"
        with self.assertRaisesRegex(StationError, "not JSON"):
            self.model().predict(frame())

    def test_missing_score_raises_station_error(self) -> None:
        self.laptop.predict_drop = ["score"]
        with self.assertRaisesRegex(StationError, "score"):
            self.model().predict(frame())

    def test_score_out_of_range_or_not_a_number_raises(self) -> None:
        for bad in (1.5, -0.1, "0.3", True, float("nan")):
            with self.subTest(score=bad):
                self.laptop.predict_fields = {"score": bad}
                with self.assertRaises(StationError):
                    self.model().predict(frame())

    def test_reply_for_another_side_or_camera_raises(self) -> None:
        self.laptop.predict_fields = {"side": "BOTTOM"}
        with self.assertRaises(StationError):
            self.model().predict(frame(side=Side.TOP))
        self.laptop.predict_fields = {"camera_id": "C4"}
        with self.assertRaises(StationError):
            self.model().predict(frame(camera="C1"))

    def test_malformed_defect_raises(self) -> None:
        self.laptop.predict_fields = {"defects": [{"defect_class": "x", "score": 0.5, "box": [1, 2, 3]}]}
        with self.assertRaises(StationError):
            self.model().predict(frame())

    def test_model_version_changed_mid_session_raises(self) -> None:
        model = self.model()
        self.assertEqual(model.check_approved(), "2026-11-01-a")
        self.laptop.models[PART] = "2026-11-02-b"  # a promotion on the laptop
        with self.assertRaisesRegex(StationError, "model version"):
            model.predict(frame())

    def test_unauthorised_token_raises(self) -> None:
        self.laptop.token = "another-token-the-station-does-not-have"
        with self.assertRaisesRegex(StationError, "401"):
            self.model().predict(frame())


class TokenSafetyTests(RemoteTestCase):
    def test_token_is_never_in_an_error_text(self) -> None:
        # The laptop echoes the token back in its error body; the station must not repeat it.
        self.laptop.predict_status = 500
        self.laptop.predict_raw = json.dumps({"error": f"bad header {FAKE_LAPTOP_TOKEN}"}).encode("utf-8")
        with self.assertRaises(StationError) as ctx:
            self.model().predict(frame())
        self.assertNotIn(FAKE_LAPTOP_TOKEN, str(ctx.exception))

    def test_unreachable_error_text_has_no_token(self) -> None:
        self.laptop.close()
        with self.assertRaises(StationError) as ctx:
            self.model().check_approved()
        self.assertNotIn(FAKE_LAPTOP_TOKEN, str(ctx.exception))

    def test_missing_token_variable_names_the_variable_not_a_value(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(TOKEN_ENV, None)
            with self.assertRaisesRegex(StationError, TOKEN_ENV):
                RemoteModel(self.laptop.base, TOKEN_ENV, 5.0, PART)

    def test_token_is_not_kept_in_the_repr(self) -> None:
        self.assertNotIn(FAKE_LAPTOP_TOKEN, repr(self.model()))


class ConstructionTests(unittest.TestCase):
    def test_timeout_must_be_positive(self) -> None:
        with mock.patch.dict(os.environ, {TOKEN_ENV: "x"}):
            with self.assertRaises(ValueError):
                RemoteModel("http://127.0.0.1:9", TOKEN_ENV, 0, PART)

    def test_base_url_must_be_http(self) -> None:
        with mock.patch.dict(os.environ, {TOKEN_ENV: "x"}):
            with self.assertRaises(ValueError):
                RemoteModel("laptop:8765", TOKEN_ENV, 5.0, PART)


class SafetyTests(unittest.TestCase):
    def test_review_range_score_from_the_laptop_is_never_green(self) -> None:
        # t_low <= score < t_img is REVIEW: the remote model must not change that rule.
        th = Thresholds(t_low=0.5, t_img=0.8)
        result = ImageResult(camera_id="C1", side=Side.TOP, score=0.6)
        self.assertIs(classify_image(result, th), Verdict.REVIEW)


if __name__ == "__main__":
    unittest.main()
