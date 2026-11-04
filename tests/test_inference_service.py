"""Tests for inference_service (L1): the laptop model server.

Fake model and fake extractor only: no torch, numpy or cv2 is needed. Each test runs the
real HTTP server on 127.0.0.1 with port 0 and a real ModelStore in a temporary folder.
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import quote

from inference_service import registry as reg
from inference_service.server import build_server, main, parse_predict_headers
from station import modelstore as ms
from station.contracts import Defect, ImageResult, Side, StationError

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "test-token-7f3a9c1e5b2d4a6f8e0c"  # a fake token, used only in these tests
TOKEN_ENV = "INFERENCE_TEST_TOKEN_NAME"
PART = "PN-100-A"
NOW = "2026-11-04T08:00:00Z"


class OutOfMemoryError(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError: the registry matches it by class name."""


class FakeModel:
    def __init__(self, version, score, faults, gate):
        self.version = version
        self._score = score
        self._faults = faults  # list of exceptions, one raised per predict call
        self._gate = gate  # optional threading.Event that predict waits on

    def predict(self, frame):
        if self._gate is not None:
            self._gate.wait(10)
        if self._faults:
            raise self._faults.pop(0)
        return ImageResult(
            camera_id=frame.camera_id,
            side=Side(frame.side),
            score=self._score,
            defects=(Defect(defect_class="anomaly", score=self._score, box=(1, 2, 3, 4)),),
        )


def auth(token):
    """Headers for a bearer token; no header at all when token is None."""
    return {} if token is None else {"Authorization": "Bearer " + token}


class ServiceCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="inference_service_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = ms.ModelStore(self.tmp / "models")
        self.loads: list[str] = []
        self.scores: dict[str, float] = {}
        self.faults: list[BaseException] = []  # raised by the next predict calls
        self.load_faults: list[BaseException] = []  # raised by the next loads
        self.gate = None
        self.registry = reg.ModelRegistry(self.store, self._loader)
        self.server = build_server(self.registry, token=TOKEN, host="127.0.0.1", port=0)
        self.port = self.server.server_address[1]
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._stop, thread)

    def _stop(self, thread):
        self.server.shutdown()
        self.server.server_close()
        thread.join(5)

    def _loader(self, info):
        self.loads.append(info.version)
        if self.load_faults:
            raise self.load_faults.pop(0)
        return FakeModel(info.version, self.scores.get(info.version, 0.25), self.faults, self.gate)

    def add_approved(self, version, content=b"fake model bytes", approve=True, part=PART):
        src = self.tmp / f"{version}.npz"
        src.write_bytes(content)
        self.store.add_candidate(part, version, src, {"type": "patchcore"}, NOW)
        if approve:
            self.store.approve(part, version, by="tester", reason="test", now_utc=NOW)

    def model_file(self, version, part=PART):
        return self.tmp / "models" / ms.safe_part(part) / version / f"{version}.npz"

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            data = resp.read()
            return resp.status, data, {k.lower(): v for k, v in resp.getheaders()}
        finally:
            conn.close()

    def predict_headers(self, changes=None):
        headers = {
            "Authorization": "Bearer " + TOKEN,
            "Part-Number": PART,
            "Camera-Id": "C1",
            "Side": "TOP",
            "Content-Type": "image/png",
        }
        for key, value in (changes or {}).items():
            if value is None:
                headers.pop(key, None)
            else:
                headers[key] = value
        return headers

    def predict(self, changes=None, body=b"\x89PNG fake picture"):
        return self.request("POST", "/v1/predict", body=body, headers=self.predict_headers(changes))

    def health(self, token=None):
        return self.request("GET", "/v1/health", headers=auth(token))

    def status(self, token=TOKEN):
        return self.request("GET", "/v1/status", headers=auth(token))

    def model_lookup(self, part, token=TOKEN):
        path = "/v1/models/" + quote(part, safe="")
        return self.request("GET", path, headers=auth(token))


class PredictStatusTests(ServiceCase):
    def test_unapproved_part_is_503(self):
        status, data, _ = self.predict()
        self.assertEqual(status, 503)
        self.assertIn("error", json.loads(data))
        self.assertEqual(self.loads, [])

    def test_tampered_model_is_503_and_never_loaded(self):
        self.add_approved("v1")
        self.model_file("v1").write_bytes(b"tampered bytes")
        status, data, _ = self.predict()
        self.assertEqual(status, 503)
        self.assertIn("error", json.loads(data))
        self.assertEqual(self.loads, [])

    def test_reload_after_promotion_and_no_reload_without_change(self):
        self.add_approved("v1")
        status, data, _ = self.predict()
        self.assertEqual(json.loads(data)["model_version"], "v1")
        self.add_approved("v2")  # approve writes current.json with a new file
        status, data, _ = self.predict()
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["model_version"], "v2")
        self.predict()
        self.assertEqual(self.loads, ["v1", "v2"])

    def test_never_falls_back_to_older_version(self):
        self.add_approved("v1")
        self.assertEqual(self.predict()[0], 200)
        self.add_approved("v2")  # approved and verified, then changed on disk before any request loads it
        self.model_file("v2").write_bytes(b"tampered")
        status, data, _ = self.predict()
        self.assertEqual(status, 503)
        self.assertNotIn(b'"v1"', data)
        self.assertEqual(self.loads, ["v1"])

    def test_success_returns_imageresult_compatible_json(self):
        self.add_approved("v1")
        self.scores["v1"] = 0.42
        status, data, headers = self.predict()
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json")
        body = json.loads(data)
        self.assertEqual(body["model_version"], "v1")
        self.assertEqual(body["score"], 0.42)
        self.assertEqual(body["defects"], [{"defect_class": "anomaly", "score": 0.42, "box": [1, 2, 3, 4]}])
        self.assertNotIn("t_low", body)  # raw scores only; the station applies thresholds
        self.assertNotIn("t_img", body)
        rebuilt = ImageResult(
            camera_id=body["camera_id"],
            side=Side(body["side"]),
            score=body["score"],
            defects=tuple(Defect(d["defect_class"], d["score"], tuple(d["box"])) for d in body["defects"]),
        )
        self.assertEqual(
            rebuilt,
            ImageResult(camera_id="C1", side=Side.TOP, score=0.42,
                        defects=(Defect("anomaly", 0.42, (1, 2, 3, 4)),)),
        )

    def test_out_of_memory_is_503_not_a_crash_and_cache_is_freed(self):
        self.add_approved("v1")
        self.faults.append(OutOfMemoryError("CUDA out of memory"))
        status, data, _ = self.predict()
        self.assertEqual(status, 503)
        self.assertIn("memory", json.loads(data)["error"])
        self.assertEqual(self.health()[0], 200)  # the server still answers
        self.assertEqual(self.predict()[0], 200)  # next request reloads the model
        self.assertEqual(self.loads, ["v1", "v1"])

    def test_out_of_memory_while_loading_is_503(self):
        self.add_approved("v1")
        self.load_faults.append(OutOfMemoryError("CUDA out of memory"))
        status, _, _ = self.predict()
        self.assertEqual(status, 503)
        self.assertEqual(self.health()[0], 200)

    def test_missing_torch_is_503_with_the_station_message(self):
        self.add_approved("v1")
        self.load_faults.append(StationError("PyTorch is not installed; the torch feature extractor needs torch"))
        status, data, _ = self.predict()
        self.assertEqual(status, 503)
        self.assertIn("PyTorch is not installed", json.loads(data)["error"])

    def test_bad_requests_are_400_or_415(self):
        self.add_approved("v1")
        self.assertEqual(self.predict({"Camera-Id": None})[0], 400)
        self.assertEqual(self.predict({"Side": "MIDDLE"})[0], 400)
        self.assertEqual(self.predict({"Part-Number": ""})[0], 400)
        self.assertEqual(self.predict({"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.predict(body=b"")[0], 400)
        self.assertEqual(self.request("POST", "/v1/nope", headers=self.predict_headers())[0], 404)

    def test_parse_predict_headers_accepts_lowercase_side(self):
        part, camera, side = parse_predict_headers({"Part-Number": PART, "Camera-Id": "C2", "Side": "bottom"})
        self.assertEqual((part, camera, side), (PART, "C2", Side.BOTTOM))


class ModelLookupTests(ServiceCase):
    def test_approved_model_is_reported_without_loading_it(self):
        self.add_approved("v1")
        status, data, _ = self.model_lookup(PART)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data), {"part_number": PART, "approved": True, "model_version": "v1"})
        self.assertEqual(self.loads, [])

    def test_lookup_reports_the_current_version_after_a_promotion(self):
        self.add_approved("v1")
        self.add_approved("v2")
        self.assertEqual(json.loads(self.model_lookup(PART)[1])["model_version"], "v2")
        self.assertEqual(self.loads, [])

    def test_no_approved_model_is_404_with_a_reason(self):
        status, data, _ = self.model_lookup(PART)
        self.assertEqual(status, 404)
        body = json.loads(data)
        self.assertIs(body["approved"], False)
        self.assertIn("No approved model", body["reason"])
        self.assertEqual(self.loads, [])

    def test_tampered_model_is_503_and_never_loaded(self):
        self.add_approved("v1")
        self.model_file("v1").write_bytes(b"tampered bytes")
        status, data, _ = self.model_lookup(PART)
        self.assertEqual(status, 503)
        body = json.loads(data)
        self.assertIs(body["approved"], False)
        self.assertNotIn("model_version", body)
        self.assertEqual(self.loads, [])

    def test_lookup_never_falls_back_to_an_older_version(self):
        self.add_approved("v1")
        self.add_approved("v2")
        self.model_file("v2").write_bytes(b"tampered")
        status, data, _ = self.model_lookup(PART)
        self.assertEqual(status, 503)
        self.assertNotIn(b'"v1"', data)
        self.assertNotIn("model_version", json.loads(data))

    def test_part_numbers_with_spaces_slashes_and_unicode_round_trip(self):
        parts = ["PN 100 A", "PN/100/A", "PN-ÄÖ-100", "A+B", "PN%2F", "PNé #7?"]
        for part in parts:
            with self.subTest(part=part):
                self.add_approved("v1", part=part)
                status, data, _ = self.model_lookup(part)
                self.assertEqual(status, 200, data)
                self.assertEqual(json.loads(data), {"part_number": part, "approved": True, "model_version": "v1"})

    def test_plus_sign_is_not_read_as_a_space(self):
        self.add_approved("v1", part="A+B")
        self.assertEqual(self.model_lookup("A+B")[0], 200)  # the raw '+' is the part's own character
        self.assertEqual(self.request("GET", "/v1/models/A%20B", headers=auth(TOKEN))[0], 404)  # a space is not '+'

    def test_unencoded_slash_in_the_part_is_400(self):
        self.add_approved("v1", part="PN/100")
        status, data, _ = self.request("GET", "/v1/models/PN/100", headers=auth(TOKEN))
        self.assertEqual(status, 400)
        self.assertIn("%2F", json.loads(data)["error"])

    def test_empty_part_is_400(self):
        status, _, _ = self.request("GET", "/v1/models/", headers=auth(TOKEN))
        self.assertEqual(status, 400)

    def test_lookup_needs_the_token(self):
        self.add_approved("v1")
        self.assertEqual(self.model_lookup(PART, token=None)[0], 401)
        self.assertEqual(self.model_lookup(PART, token="not-the-token")[0], 401)
        self.assertEqual(self.loads, [])


class AuthTests(ServiceCase):
    def test_wrong_token_is_401(self):
        self.add_approved("v1")
        self.assertEqual(self.predict({"Authorization": "Bearer not-the-token"})[0], 401)
        self.assertEqual(self.status(token="not-the-token")[0], 401)
        self.assertEqual(self.loads, [])

    def test_missing_or_other_scheme_is_401(self):
        self.add_approved("v1")
        self.assertEqual(self.predict({"Authorization": None})[0], 401)
        self.assertEqual(self.predict({"Authorization": "Basic " + TOKEN})[0], 401)
        self.assertEqual(self.request("GET", "/v1/status")[0], 401)

    def test_status_is_token_protected(self):
        self.assertEqual(self.status(token=None)[0], 401)
        self.assertEqual(self.status(token="wrong")[0], 401)
        self.assertEqual(self.status(token=TOKEN)[0], 200)

    def test_token_never_appears_in_responses_or_stderr(self):
        self.add_approved("v1")
        captured = io.StringIO()
        responses = []
        with contextlib.redirect_stderr(captured):
            responses.append(self.predict())
            responses.append(self.predict({"Authorization": "Bearer wrong"}))
            responses.append(self.predict({"Part-Number": "UNKNOWN-PART"}))
            responses.append(self.predict({"Side": "MIDDLE"}))
            responses.append(self.health())
            responses.append(self.health(token="wrong"))
            responses.append(self.status())
            responses.append(self.status(token="wrong"))
            responses.append(self.model_lookup(PART))
            responses.append(self.model_lookup(PART, token="wrong"))
        for status, data, headers in responses:
            self.assertNotIn(TOKEN, data.decode("utf-8", "replace"))
            for value in headers.values():
                self.assertNotIn(TOKEN, value)
        self.assertNotIn(TOKEN, captured.getvalue())
        self.assertNotIn("Bearer", captured.getvalue())


class HealthTests(ServiceCase):
    def test_health_needs_no_token_and_answers_only_busy_and_status(self):
        status, data, _ = self.health()
        self.assertEqual(status, 200)
        body = json.loads(data)
        self.assertEqual(set(body), {"status", "busy", "in_flight"})
        self.assertEqual(body["status"], "ok")
        self.assertIs(body["busy"], False)
        self.assertEqual(body["in_flight"], 0)

    def test_health_busy_is_a_json_boolean_for_the_weekly_script_call(self):
        # The same call run_weekly_training.ps1 makes: GET with no Authorization header.
        status, data, headers = self.request("GET", "/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertIn(b'"busy": false', data)
        self.assertIs(json.loads(data)["busy"], False)

    def test_health_does_not_list_models_or_gpu(self):
        self.add_approved("v1")
        self.assertEqual(self.predict()[0], 200)
        body = json.loads(self.health()[1])
        self.assertNotIn("models", body)
        self.assertNotIn("gpu", body)
        self.assertNotIn("version", body)

    def test_busy_is_true_while_a_request_is_in_progress(self):
        self.add_approved("v1")
        self.gate = threading.Event()
        worker = threading.Thread(target=self.predict, daemon=True)
        worker.start()
        try:
            busy = False
            for _ in range(500):
                body = json.loads(self.health()[1])
                if body["busy"]:
                    busy = True
                    break
                time.sleep(0.01)
            self.assertTrue(busy)
            self.assertGreaterEqual(body["in_flight"], 1)
        finally:
            self.gate.set()
            worker.join(10)
        self.assertFalse(json.loads(self.health()[1])["busy"])


class StatusTests(ServiceCase):
    def test_status_lists_loaded_models_gpu_and_version_with_the_token(self):
        self.add_approved("v1")
        self.assertEqual(self.predict()[0], 200)
        status, data, _ = self.status()
        self.assertEqual(status, 200)
        body = json.loads(data)
        self.assertEqual(body["models"], [{"part_number": PART, "version": "v1"}])
        self.assertEqual(body["version"], "1")
        self.assertFalse(body["busy"])
        self.assertIn("gpu", body)
        self.assertIn("torch_loaded", body["gpu"])

    def test_unknown_path_with_the_token_is_404(self):
        self.assertEqual(self.request("GET", "/v1/nope", headers=auth(TOKEN))[0], 404)


class StartupTests(unittest.TestCase):
    def test_importing_the_service_does_not_load_torch_numpy_or_cv2(self):
        code = (
            "import sys, inference_service.server, inference_service.registry; "
            "print(any(m in sys.modules for m in ('torch', 'cv2', 'numpy')))"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
                             text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False")

    def test_missing_token_variable_refuses_to_start_and_names_only_the_variable(self):
        tmp = tempfile.mkdtemp(prefix="inference_service_cli_")
        self.addCleanup(shutil.rmtree, tmp, True)
        err = io.StringIO()
        env = {k: v for k, v in os.environ.items() if k != TOKEN_ENV}
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stderr(err):
            code = main(["--models", tmp, "--token-env", TOKEN_ENV])
        self.assertEqual(code, 2)
        self.assertIn(TOKEN_ENV, err.getvalue())


if __name__ == "__main__":
    unittest.main()
