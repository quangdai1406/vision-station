import sys
import unittest
from unittest import mock

from station.contracts import Defect, Frame, ImageResult, Side, StationError
from station.inference import MarkerModel, OnnxModel, ScriptedModel


def _frame(data=b"", camera="C1", side=Side.TOP):
    return Frame(camera_id=camera, side=side, seq=0, timestamp_utc="2026-10-23T08:30:00Z", data=data)


class ScriptedModelTests(unittest.TestCase):
    def test_version_and_scripted_result(self):
        scripted = ImageResult(camera_id="C2", side=Side.BOTTOM, score=0.9,
                               defects=(Defect("dent", 0.9, (1, 2, 3, 4)),))
        model = ScriptedModel("v1", {("C2", "BOTTOM"): scripted})
        self.assertEqual(model.version, "v1")
        self.assertIs(model.predict(_frame(camera="C2", side=Side.BOTTOM)), scripted)

    def test_unscripted_key_gets_default_score_and_no_defects(self):
        model = ScriptedModel("v1", default_score=0.25)
        result = model.predict(_frame(camera="C4", side=Side.TOP))
        self.assertEqual(result, ImageResult(camera_id="C4", side=Side.TOP, score=0.25))
        self.assertEqual(result.defects, ())

    def test_default_score_is_zero(self):
        self.assertEqual(ScriptedModel("v").predict(_frame()).score, 0.0)

    def test_scripted_key_is_side_specific(self):
        scripted = ImageResult(camera_id="C1", side=Side.TOP, score=0.5)
        model = ScriptedModel("v", {("C1", "TOP"): scripted})
        self.assertEqual(model.predict(_frame(side=Side.BOTTOM)).score, 0.0)

    def test_default_score_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            ScriptedModel("v", default_score=1.5)


class MarkerModelTests(unittest.TestCase):
    def setUp(self):
        self.model = MarkerModel("demo-1")

    def test_full_marker(self):
        result = self.model.predict(_frame(b"SCORE:0.72;CLASS:scratch;BOX:10,20,30,40"))
        self.assertEqual(result.score, 0.72)
        self.assertEqual(result.camera_id, "C1")
        self.assertEqual(result.side, Side.TOP)
        self.assertEqual(result.defects, (Defect("scratch", 0.72, (10, 20, 30, 40)),))
        self.assertEqual(self.model.version, "demo-1")

    def test_no_marker_is_score_zero_no_defects(self):
        result = self.model.predict(_frame(b"\x89PNG plain bytes"))
        self.assertEqual(result, ImageResult(camera_id="C1", side=Side.TOP, score=0.0))

    def test_marker_may_be_embedded_in_other_bytes(self):
        result = self.model.predict(_frame(b"\x00\x01FAKE:C1:TOP:0 SCORE:0.3"))
        self.assertEqual(result.score, 0.3)
        self.assertEqual(result.defects, ())

    def test_score_only_has_no_defects(self):
        result = self.model.predict(_frame(b"SCORE:0.4"))
        self.assertEqual(result.score, 0.4)
        self.assertEqual(result.defects, ())

    def test_class_without_box_uses_zero_box(self):
        result = self.model.predict(_frame(b"CLASS:stain"))
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.defects, (Defect("stain", 0.0, (0, 0, 0, 0)),))

    def test_malformed_markers_raise(self):
        bad = [
            b"SCORE:abc",
            b"SCORE:",
            b"SCORE:1.5",
            b"SCORE:-0.1",
            b"SCORE:nan",
            b"SCORE:0.1;SCORE:0.2",
            b"CLASS:scratch;CLASS:dent",
            b"CLASS:scratch;BOX:1,2,3",
            b"CLASS:scratch;BOX:a,b,c,d",
            b"CLASS:scratch;BOX:1,2,3,-4",
            b"CLASS:scratch;BOX:1.5,2,3,4",
            b"BOX:1,2,3,4",
            b"SCORE;CLASS:scratch",
            b"SCORE:0.2;",
        ]
        for data in bad:
            with self.subTest(data=data):
                with self.assertRaises(StationError):
                    self.model.predict(_frame(data))


class _FakeInput:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def __init__(self, output=None, fail=False):
        self.output = output if output is not None else ["RAW"]
        self.fail = fail
        self.calls = []

    def get_inputs(self):
        return [_FakeInput("images")]

    def run(self, output_names, feed):
        self.calls.append((output_names, feed))
        if self.fail:
            raise RuntimeError("bad tensor")
        return self.output


def _pre(frame):
    return f"TENSOR({frame.camera_id})"


def _post(frame, raw):
    return ImageResult(camera_id=frame.camera_id, side=frame.side,
                       score=0.5 if raw == ["RAW"] else 0.0)


class OnnxModelTests(unittest.TestCase):
    def test_predict_runs_session_with_preprocessed_tensor(self):
        session = FakeSession()
        model = OnnxModel("/models/x.onnx", "onnx-7", _pre, _post, session=session)
        self.assertEqual(model.version, "onnx-7")
        frame = _frame(camera="C3", side=Side.BOTTOM)
        result = model.predict(frame)
        self.assertEqual(session.calls, [(None, {"images": "TENSOR(C3)"})])
        self.assertEqual(result, ImageResult(camera_id="C3", side=Side.BOTTOM, score=0.5))

    def test_session_failure_raises_station_error(self):
        model = OnnxModel("m.onnx", "v", _pre, _post, session=FakeSession(fail=True))
        with self.assertRaises(StationError):
            model.predict(_frame())

    def test_injected_ort_module_builds_session_from_path(self):
        session = FakeSession()
        ort = mock.Mock()
        ort.InferenceSession.return_value = session
        model = OnnxModel("/models/x.onnx", "v", _pre, _post, ort_module=ort)
        ort.InferenceSession.assert_called_once_with("/models/x.onnx")
        model.predict(_frame())
        self.assertEqual(len(session.calls), 1)

    def test_missing_onnxruntime_without_session_raises_station_error(self):
        with mock.patch.dict(sys.modules, {"onnxruntime": None}):
            with self.assertRaises(StationError) as ctx:
                OnnxModel("m.onnx", "v", _pre, _post)
        self.assertIn("onnxruntime", str(ctx.exception))

    def test_session_load_failure_raises_station_error(self):
        ort = mock.Mock()
        ort.InferenceSession.side_effect = RuntimeError("corrupt file")
        with self.assertRaises(StationError):
            OnnxModel("m.onnx", "v", _pre, _post, ort_module=ort)


if __name__ == "__main__":
    unittest.main()
