"""Tests for tools/pi_check.py. A fake cv2 stands in for OpenCV; a localhost fake inference
server stands in for the laptop. Nothing here touches a camera, the LAN or the real SMB share."""
from __future__ import annotations

import contextlib
import io
import json
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from tools import pi_check
from tools.pi_check import main

TOKEN = "pi-check-test-token-7f3a9c1e"
PART = "PN-TEST-001"


class FakeFrame:
    """Stands in for a numpy image: shape, and any() for the all-black test."""

    def __init__(self, value: int, h: int = 48, w: int = 64) -> None:
        self.shape = (h, w, 3)
        self._any = value != 0

    def any(self) -> bool:
        return self._any


class FakeCap:
    def __init__(self, cv2: "FakeCv2", index: int) -> None:
        self.cv2 = cv2
        self.index = index
        self.grabbed = False

    def isOpened(self) -> bool:
        return self.index not in self.cv2.dead

    def grab(self) -> bool:
        if self.index in self.cv2.no_frame:
            return False
        self.grabbed = True
        return True

    def retrieve(self):
        if not self.grabbed:
            return False, None
        self.grabbed = False
        return True, FakeFrame(0 if self.index in self.cv2.black else 100 + self.index)

    def release(self) -> None:
        self.cv2.released.append(self.index)


class FakeCv2:
    CAP_V4L2 = 200
    __version__ = "4.0.0-fake"

    def __init__(self, *, dead=(), no_frame=(), black=()) -> None:
        self.dead = set(dead)
        self.no_frame = set(no_frame)
        self.black = set(black)
        self.calls: list[tuple[int, object]] = []
        self.released: list[int] = []

    def VideoCapture(self, index: int, api: object = None) -> FakeCap:
        self.calls.append((index, api))
        return FakeCap(self, index)


class _FakeLaptop(BaseHTTPRequestHandler):
    """Localhost stand-in for the laptop inference service (GET /v1/health, POST /v1/predict)."""

    server_version = "FakeLaptop"
    sys_version = ""

    def log_message(self, format, *args) -> None:  # noqa: A002
        return

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorised(self) -> bool:
        if self.headers.get("Authorization") == f"Bearer {self.server.token}":
            return True
        self._send(401, {"error": "unauthorized"})
        return False

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorised():
            return
        if self.path != "/v1/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {"status": "ok", "version": "1", "models": [{"part_number": PART, "version": "v1"}],
                         "busy": False, "in_flight": 0, "gpu": {}})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorised():
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.server.predict_headers.append({k: self.headers.get(k) for k in
                                            ("Part-Number", "Camera-Id", "Side", "Content-Type")})
        if self.server.predict_status != 200:
            self._send(self.server.predict_status, {"error": self.server.predict_error})
            return
        self._send(200, {"camera_id": self.headers.get("Camera-Id"), "side": self.headers.get("Side"),
                         "score": 0.12, "defects": [], "model_version": "v1"})


class FakeLaptopServer:
    def __init__(self, token: str = TOKEN) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLaptop)
        self.httpd.token = token
        self.httpd.predict_headers = []
        self.httpd.predict_status = 200
        self.httpd.predict_error = "internal error"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "FakeLaptopServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def predict_headers(self) -> list:
        return self.httpd.predict_headers


def _closed_port_url() -> str:
    """A localhost URL where nothing listens: the port was free a moment ago."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.lines: list[str] = []
        self.report_path = self.dir / "pi_check_report.json"

    def run_main(self, argv, *, cv2=None, env=None, platform_name="linux", no_cameras=True):
        args = list(argv) + ["--out", str(self.report_path)]
        if no_cameras:
            args.append("--no-cameras")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = main(args, cv2_module=cv2, environ=env if env is not None else {},
                      platform_name=platform_name, out=lambda s: self.lines.append(str(s)))
        self.stderr = err.getvalue()
        self.text = "\n".join(self.lines)
        return rc

    def report(self) -> dict:
        return json.loads(self.report_path.read_text(encoding="utf-8"))


class CameraListTests(_Base):
    def test_each_working_camera_is_listed_by_index(self) -> None:
        rc = self.run_main([], cv2=FakeCv2(), no_cameras=False)
        self.assertEqual(rc, 0, self.stderr)
        for index in range(pi_check.DEFAULT_MAX_INDEX + 1):
            self.assertIn(f"index {index}: OK", self.text)

    def test_dead_camera_is_reported_by_index(self) -> None:
        rc = self.run_main(["--max-index", "3"], cv2=FakeCv2(dead={2}), no_cameras=False)
        self.assertEqual(rc, 0)  # camera results are data, not an error
        self.assertIn("index 2: NOT WORKING: does not open", self.text)
        cams = self.report()["cameras"]
        self.assertEqual(cams["working"], [0, 1, 3])
        self.assertEqual(cams["not_working"], [{"index": 2, "reason": "does not open"}])

    def test_camera_that_opens_but_gives_no_frame_is_reported(self) -> None:
        self.run_main(["--max-index", "1"], cv2=FakeCv2(no_frame={1}), no_cameras=False)
        self.assertIn("index 1: NOT WORKING: opens but gives no frame", self.text)

    def test_all_black_picture_is_reported_by_index(self) -> None:
        rc = self.run_main(["--max-index", "0"], cv2=FakeCv2(black={0}), no_cameras=False)
        self.assertEqual(rc, 0)
        self.assertIn("index 0: NOT WORKING: all-black picture", self.text)
        cams = self.report()["cameras"]
        self.assertEqual(cams["working"], [])
        self.assertEqual(cams["not_working"], [{"index": 0, "reason": "all-black picture"}])
        self.assertFalse(cams["details"][0]["not_black"])

    def test_linux_uses_v4l2_and_says_why(self) -> None:
        cv2 = FakeCv2()
        self.run_main(["--max-index", "1"], cv2=cv2, platform_name="linux", no_cameras=False)
        self.assertTrue(all(api == FakeCv2.CAP_V4L2 for _, api in cv2.calls))
        self.assertIn("V4L2", self.text)
        self.assertIn("DirectShow", self.text)
        self.assertEqual(self.report()["cameras"]["backend"], "v4l2")

    def test_other_platform_uses_default_backend(self) -> None:
        cv2 = FakeCv2()
        self.run_main(["--max-index", "0"], cv2=cv2, platform_name="win32", no_cameras=False)
        self.assertTrue(all(api is None for _, api in cv2.calls))
        self.assertEqual(self.report()["cameras"]["backend"], "default")

    def test_every_camera_handle_is_released_even_a_dead_one(self) -> None:
        cv2 = FakeCv2(dead={1})
        self.run_main(["--max-index", "2"], cv2=cv2, no_cameras=False)
        self.assertEqual(sorted(cv2.released), [0, 1, 2])

    def test_missing_opencv_is_reported_and_exits_1(self) -> None:
        with mock.patch.dict("sys.modules", {"cv2": None}):
            rc = self.run_main(["--max-index", "0"], cv2=None, no_cameras=False)
        self.assertEqual(rc, 1)
        self.assertIn("OpenCV", self.text)
        self.assertIn("docs/pi_setup.md", self.text)
        self.assertTrue(self.report()["cameras"]["error"])


class LaptopTests(_Base):
    def test_unreachable_laptop_is_reported_and_exits_1(self) -> None:
        url = _closed_port_url()
        rc = self.run_main(["--inference-url", url, "--token-env", "PI_TOKEN", "--part-number", PART],
                           env={"PI_TOKEN": TOKEN})
        self.assertEqual(rc, 1)
        self.assertIn("laptop unreachable", self.text)
        self.assertIn("No picture was sent", self.text)
        inf = self.report()["inference"]
        self.assertFalse(inf["health"]["ok"])
        self.assertEqual(inf["images"], [])

    def test_round_trip_timings_for_eight_pictures(self) -> None:
        with FakeLaptopServer() as laptop:
            rc = self.run_main(["--inference-url", laptop.url, "--token-env", "PI_TOKEN",
                                "--part-number", PART], env={"PI_TOKEN": TOKEN})
            headers = laptop.predict_headers
        self.assertEqual(rc, 0, self.stderr)
        self.assertEqual(len(headers), pi_check.DEFAULT_IMAGES)
        self.assertEqual(pi_check.DEFAULT_IMAGES, 8)
        self.assertEqual({h["Part-Number"] for h in headers}, {PART})
        self.assertEqual({h["Content-Type"] for h in headers}, {"image/png"})
        self.assertEqual({(h["Camera-Id"], h["Side"]) for h in headers},
                         {(c, s) for c in ("C1", "C2", "C3", "C4") for s in ("TOP", "BOTTOM")})
        for word in ("min", "median", "max", "Total"):
            self.assertIn(word, self.text)
        self.assertIn("Operator wait limit: [Quality to define]", self.lines)
        self.assertEqual(self.lines[-2], "Operator wait limit: [Quality to define]")  # then "Report written"
        inf = self.report()["inference"]
        self.assertEqual(inf["stats"]["count_ok"], 8)
        self.assertLessEqual(inf["stats"]["min_s"], inf["stats"]["median_s"])
        self.assertLessEqual(inf["stats"]["median_s"], inf["stats"]["max_s"])
        self.assertEqual(inf["health"]["version"], "1")
        self.assertEqual(inf["health"]["models"], 1)

    def test_timings_are_printed_without_pass_or_fail_judgement(self) -> None:
        with FakeLaptopServer() as laptop:
            self.run_main(["--inference-url", laptop.url, "--token-env", "PI_TOKEN",
                           "--part-number", PART, "--images", "2"], env={"PI_TOKEN": TOKEN})
        lowered = self.text.lower()
        self.assertNotIn("pass", lowered)
        self.assertNotIn("threshold", lowered)
        self.assertIn("2 OK", self.text)

    def test_refused_token_is_reported_and_no_picture_is_sent(self) -> None:
        with FakeLaptopServer(token="someone-else") as laptop:
            rc = self.run_main(["--inference-url", laptop.url, "--token-env", "PI_TOKEN",
                                "--part-number", PART], env={"PI_TOKEN": TOKEN})
            headers = laptop.predict_headers
        self.assertEqual(rc, 1)
        self.assertIn("HTTP 401", self.text)
        self.assertEqual(headers, [])

    def test_failing_predict_is_listed_and_exits_1(self) -> None:
        with FakeLaptopServer() as laptop:
            laptop.httpd.predict_status = 503
            laptop.httpd.predict_error = "no approved model"
            rc = self.run_main(["--inference-url", laptop.url, "--token-env", "PI_TOKEN",
                                "--part-number", PART, "--images", "2"], env={"PI_TOKEN": TOKEN})
        self.assertEqual(rc, 1)
        self.assertIn("HTTP 503", self.text)
        self.assertIn("no approved model", self.text)
        self.assertEqual(self.report()["inference"]["stats"]["count_ok"], 0)

    def test_token_never_appears_in_output_or_json_even_if_a_server_echoes_it(self) -> None:
        with FakeLaptopServer() as laptop:
            laptop.httpd.predict_status = 500
            laptop.httpd.predict_error = f"bad token {TOKEN} seen"  # a broken server echoing it
            self.run_main(["--inference-url", laptop.url, "--token-env", "PI_TOKEN",
                           "--part-number", PART, "--images", "1"], env={"PI_TOKEN": TOKEN})
        self.assertNotIn(TOKEN, self.text)
        self.assertNotIn(TOKEN, self.stderr)
        self.assertNotIn(TOKEN, self.report_path.read_text(encoding="utf-8"))

    def test_token_is_not_printed_on_success(self) -> None:
        with FakeLaptopServer() as laptop:
            self.run_main(["--inference-url", laptop.url, "--token-env", "PI_TOKEN",
                           "--part-number", PART], env={"PI_TOKEN": TOKEN})
        self.assertNotIn(TOKEN, self.text)
        self.assertNotIn(TOKEN, self.report_path.read_text(encoding="utf-8"))
        self.assertIn("PI_TOKEN", self.text)  # the variable's NAME may be shown

    def test_missing_token_variable_is_an_argument_error(self) -> None:
        rc = self.run_main(["--inference-url", "http://127.0.0.1:9", "--token-env", "PI_TOKEN",
                            "--part-number", PART], env={})
        self.assertEqual(rc, 2)
        self.assertIn("PI_TOKEN", self.stderr)
        self.assertFalse(self.report_path.exists())

    def test_url_with_credentials_is_refused(self) -> None:
        rc = self.run_main(["--inference-url", "http://user:pw@127.0.0.1:9", "--token-env", "PI_TOKEN",
                            "--part-number", PART], env={"PI_TOKEN": TOKEN})
        self.assertEqual(rc, 2)
        self.assertNotIn("pw", self.stderr)

    def test_token_env_without_url_is_an_argument_error(self) -> None:
        self.assertEqual(self.run_main(["--token-env", "PI_TOKEN"], env={"PI_TOKEN": TOKEN}), 2)

    def test_image_file_must_be_png_or_jpeg(self) -> None:
        bad = self.dir / "picture.bmp"
        bad.write_bytes(b"BM")
        rc = self.run_main(["--inference-url", "http://127.0.0.1:9", "--token-env", "PI_TOKEN",
                            "--part-number", PART, "--image", str(bad)], env={"PI_TOKEN": TOKEN})
        self.assertEqual(rc, 2)

    def test_user_supplied_png_is_sent(self) -> None:
        png = self.dir / "picture.png"
        png.write_bytes(pi_check.synthetic_png(16, 16))
        with FakeLaptopServer() as laptop:
            rc = self.run_main(["--inference-url", laptop.url, "--token-env", "PI_TOKEN",
                                "--part-number", PART, "--image", str(png), "--images", "1"],
                               env={"PI_TOKEN": TOKEN})
        self.assertEqual(rc, 0, self.stderr)
        self.assertIn(str(png), self.text)

    def test_synthetic_png_has_a_png_signature(self) -> None:
        data = pi_check.synthetic_png(8, 8)
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))


class EvidenceTests(_Base):
    def test_writable_folder_is_reported_with_free_space_and_no_leftovers(self) -> None:
        share = self.dir / "share"
        share.mkdir()
        rc = self.run_main(["--evidence-dest", str(share)])
        self.assertEqual(rc, 0, self.stderr)
        ev = self.report()["evidence"]
        self.assertTrue(ev["writable"])
        self.assertGreater(ev["free_bytes"], 0)
        self.assertIn("writable", self.text)
        self.assertEqual(list(share.iterdir()), [])

    def test_missing_mount_point_is_reported_and_exits_1(self) -> None:
        rc = self.run_main(["--evidence-dest", str(self.dir / "not-mounted")])
        self.assertEqual(rc, 1)
        self.assertIn("not a directory", self.text)
        self.assertFalse(self.report()["evidence"]["writable"])


class ReportTests(_Base):
    def test_version_lines_are_in_the_report(self) -> None:
        self.run_main([], cv2=FakeCv2(), no_cameras=False)
        v = self.report()["versions"]
        self.assertTrue(v["python"])
        self.assertTrue(v["platform"])
        self.assertEqual(v["opencv"], "4.0.0-fake")

    def test_opencv_none_when_not_installed(self) -> None:
        with mock.patch.dict("sys.modules", {"cv2": None}):
            self.run_main([], cv2=None)
        self.assertIsNone(self.report()["versions"]["opencv"])

    def test_report_is_written_atomically_and_no_temp_file_is_left(self) -> None:
        self.run_main([])
        self.assertEqual(self.report()["schema"], 1)
        self.assertEqual([p.name for p in self.dir.iterdir() if p.name.endswith(".tmp")], [])

    def test_json_encoding_failure_exits_1_and_keeps_the_old_report(self) -> None:
        self.report_path.write_text("old report\n", encoding="utf-8")
        with mock.patch.object(pi_check.json, "dumps", side_effect=TypeError("not serialisable")):
            rc = self.run_main([])
        self.assertEqual(rc, 1)
        self.assertIn("not valid JSON", self.stderr)
        self.assertEqual(self.report_path.read_text(encoding="utf-8"), "old report\n")
        self.assertEqual([p.name for p in self.dir.iterdir() if p.name.endswith(".tmp")], [])

    def test_operator_wait_limit_placeholder_is_in_the_report(self) -> None:
        self.run_main([])
        self.assertEqual(self.report()["operator_wait_limit"], "[Quality to define]")


class ArgumentTests(_Base):
    def test_bad_max_index_exits_2(self) -> None:
        self.assertEqual(self.run_main(["--max-index", "-1"]), 2)


if __name__ == "__main__":
    unittest.main()
