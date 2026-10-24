"""End-to-end: the ONE-CAMERA proof-of-concept station through the real stack.

Real StationApp, session, storage, outbox, MES client and UI HTTP server. Injected:
the cameras (FakeCameraSet, whose frame bytes are small grey images) and the model
(a GoldenModel built with GoldenBuilder and a fake decode function). The MES is a
fake http.server on 127.0.0.1. Nothing touches the network beyond 127.0.0.1 and
there are no real sleeps.

The thresholds (t_low 0.5, t_img 0.8), the golden settings and the image sizes
in this file exist only to exercise the flow. They have no quality meaning.
Quality owns the real thresholds.
"""
from __future__ import annotations

import contextlib
import http.client
import json
import os
import pathlib
import shutil
import sqlite3
import tempfile
import threading
import unittest

try:
    import numpy as np
except ImportError:  # the proof-of-concept flow needs numpy (golden model); skip without it
    np = None

from station.app import build_station
from station.auth import hash_password
from station.cameras import FakeCameraSet
from station.contracts import Frame, Side, StationError
from station.golden import GoldenBuilder
from station.inference import MarkerModel
from station.outputs import RecordingOutputs
from tests.test_e2e import FakeClock, FakeMes, Ui, parse_multipart, press_part

TOKEN_ENV = "VISION_TEST_POC_E2E_TOKEN"
TOKEN = "poc-e2e-test-token-not-a-real-secret"
OPERATOR = ("operator1", "operator-pass-1")
SUPERVISOR = ("supervisor1", "supervisor-pass-1")

DEFECT_CLASS = "difference_from_golden"  # the class GoldenModel reports
MODEL_VERSION = "poc-golden-1"
H, W = 40, 60
BLOB_X, BLOB_Y, BLOB_W, BLOB_H = 20, 10, 8, 8
BLOB_BOX = (BLOB_X, BLOB_Y, BLOB_W, BLOB_H)
GOLDEN_FRAMES = 5  # good frames per (C1, side) used to learn the golden image


# ------------------------------------------------------------ image helpers

def image_bytes(seed: int, blob: bool = False) -> bytes:
    """A flat grey 40x60 image (128) with +-1 noise, optionally with a bright 8x8 blob."""
    rng = np.random.default_rng(seed)
    img = (128 + rng.integers(-1, 2, size=(H, W))).astype(np.uint8)
    if blob:
        img[BLOB_Y:BLOB_Y + BLOB_H, BLOB_X:BLOB_X + BLOB_W] = 250
    return img.tobytes()


def decode_frame(frame: Frame) -> np.ndarray:
    """Fake decode: the frame bytes are the raw 40x60 grey image."""
    if len(frame.data) != H * W:
        raise StationError(f"test frame has {len(frame.data)} bytes, expected {H * W}")
    return np.frombuffer(frame.data, dtype=np.uint8).reshape(H, W).astype(np.float64)


def build_golden_model():
    builder = GoldenBuilder(decode_frame)
    for side in (Side.TOP, Side.BOTTOM):
        for seed in range(1, GOLDEN_FRAMES + 1):
            builder.add(Frame(camera_id="C1", side=side, seq=0,
                              timestamp_utc="2026-10-24T00:00:00Z", data=image_bytes(seed)))
    return builder.build(MODEL_VERSION)


def _overlap_area(a, b) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = min(ax + aw, bx + bw) - max(ax, bx)
    h = min(ay + ah, by + bh) - max(ay, by)
    return max(0, w) * max(0, h)


# ------------------------------------------------------------ harness

@unittest.skipIf(np is None, "numpy not installed")
class _Harness(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="poc-e2e-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

        self.mes = FakeMes()
        threading.Thread(target=self.mes.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
        self.addCleanup(self._stop_mes)

        os.environ[TOKEN_ENV] = TOKEN
        self.addCleanup(os.environ.pop, TOKEN_ENV, None)

        self.users_file = os.path.join(self.tmp, "users.json")
        with open(self.users_file, "w", encoding="utf-8") as fh:
            json.dump({
                OPERATOR[0]: {"password_hash": hash_password(OPERATOR[1]), "role": "operator"},
                SUPERVISOR[0]: {"password_hash": hash_password(SUPERVISOR[1]), "role": "supervisor"},
            }, fh)

        self.blobs: set[tuple[str, str]] = set()  # (camera_id, side value) that get a blob
        self.clock = FakeClock()
        self.outputs = RecordingOutputs()

    def _stop_mes(self) -> None:
        self.mes.shutdown()
        self.mes.server_close()

    def _frame_bytes(self, camera_id: str, side: Side, n: int) -> bytes:
        blob = (camera_id, side.value) in self.blobs
        return image_bytes(n + (0 if side is Side.TOP else 100), blob=blob)

    def config(self, *, allow_partial: bool, cameras: dict, model: dict) -> dict:
        return {
            "site": "SITE1", "line": "L1", "station_id": "VIS-POC-1", "part_number": "PN-100",
            "lot": "LOT-7", "operator": "operator1", "supervisor": "supervisor1",
            "material_handler": "mh1",
            # Test values: they only exercise the flow. They have no quality meaning.
            "thresholds": {"t_low": 0.5, "t_img": 0.8},
            "data_dir": os.path.join(self.tmp, "data"),
            "users_file": self.users_file,
            "ui": {"host": "127.0.0.1", "port": 0, "session_timeout_s": 900},
            "mes": {
                "base_url": self.mes.base_url,
                "token_env": TOKEN_ENV,
                "criteria_ids": ["CR-1", "CR-2"],
                "class_to_criterion": {DEFECT_CLASS: "CR-1"},
                "default_criterion_id": None,
                "sync_interval_s": 3600,  # the tests call sync_once() themselves
                "default_bad_disposition": "quarantine",
                "allow_partial_cameras": allow_partial,
            },
            "cameras": cameras,
            "model": model,
            "outputs": {"kind": "console"},
            "flip_timeout_s": 60,
        }

    def start(self, cfg: dict, *, cameras, model):
        app = build_station(cfg, cameras=cameras, model=model, outputs=self.outputs, clock=self.clock)
        app.start()
        self.addCleanup(app.stop)
        self.app = app
        self.ui = Ui(app)
        self.op = self.ui.login(*OPERATOR)
        self.sup = self.ui.login(*SUPERVISOR)
        return app

    def one_camera(self, *, allow_partial: bool):
        """A 1-camera station. The model is injected; the config model block is only validated."""
        cfg = self.config(
            allow_partial=allow_partial,
            cameras={"kind": "fake", "camera_count": 1},
            model={"kind": "golden", "path": os.path.join(self.tmp, "golden.npz"), "version": MODEL_VERSION},
        )
        cameras = FakeCameraSet(frame_bytes=self._frame_bytes, camera_ids=("C1",))
        return self.start(cfg, cameras=cameras, model=build_golden_model())

    # helpers
    def press_part(self) -> None:
        """TOP press, flip gap, BOTTOM press (module press_part)."""
        press_part(self.ui, self.op, self.clock)

    def state(self) -> dict:
        status, body, _ = self.ui.req("GET", "/api/state", token=self.op)
        self.assertEqual(status, 200, body)
        return body

    def saved_part(self):
        row = self.app.store.recent_parts(1)[0]
        return row["session_id"], pathlib.Path(row["path"])

    @staticmethod
    def part_json(part_dir: pathlib.Path) -> dict:
        return json.loads((part_dir / "part.json").read_text(encoding="utf-8"))

    @staticmethod
    def pngs(part_dir: pathlib.Path) -> list[str]:
        return sorted(p.name for p in part_dir.iterdir() if p.suffix == ".png")

    def events(self, kind: str) -> list[dict]:
        with contextlib.closing(sqlite3.connect(str(self.app.store.db_path))) as conn:
            rows = conn.execute(
                "SELECT payload_json FROM events WHERE kind = ? ORDER BY id", (kind,)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def posted(self) -> list[dict]:
        out = []
        for rec in self.mes.requests:
            fields, files = parse_multipart(rec["ctype"], rec["body"])
            out.append({"record": rec, "fields": fields, "files": files})
        return out

    def raw_get(self, path: str) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection(self.ui.host, self.ui.port, timeout=10)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()


# ------------------------------------------------------------ one camera

@unittest.skipIf(np is None, "numpy not installed")
class OneCameraPocE2E(_Harness):

    # 1. GOOD
    def test_good_part_is_saved_and_shows_green(self) -> None:
        self.one_camera(allow_partial=False)
        self.press_part()

        st = self.state()
        self.assertEqual(st["state"], "IDLE")
        self.assertEqual(st["verdict"], "GOOD")
        self.assertEqual(st["counters"], {"total": 1, "good": 1, "bad": 0})
        self.assertTrue(self.outputs.green)
        self.assertFalse(self.outputs.red)
        self.assertIs(st["partial_station"], True)
        self.assertEqual(st["camera_ids"], ["C1"])

        session_id, part_dir = self.saved_part()
        self.assertEqual(len(self.pngs(part_dir)), 2)
        self.assertTrue(any(n.endswith("_C1_TOP.png") for n in self.pngs(part_dir)))
        self.assertTrue(any(n.endswith("_C1_BOTTOM.png") for n in self.pngs(part_dir)))
        doc = self.part_json(part_dir)
        self.assertEqual(doc["camera_count"], 1)
        self.assertIs(doc["partial_station"], True)
        self.assertEqual(doc["final_verdict"], "GOOD")
        self.assertEqual(self.app.store.verify_part(part_dir), [])

        # The supervisor sees the same station state.
        status, body, _ = self.ui.req("GET", "/api/state", token=self.sup)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["verdict"], "GOOD")

    # 2. BAD
    def test_bad_part_stays_red_until_ack_and_keeps_the_defect_box(self) -> None:
        self.one_camera(allow_partial=False)
        self.blobs.add(("C1", "TOP"))
        self.press_part()

        st = self.state()
        self.assertEqual(st["state"], "VERDICT")
        self.assertEqual(st["verdict"], "BAD")
        self.assertEqual(st["counters"]["total"], 0)  # not counted until acknowledged
        self.assertTrue(self.outputs.red)
        self.assertTrue(self.outputs.alarm)
        self.assertFalse(self.outputs.green)
        top = next(i for i in st["images"] if i["camera_id"] == "C1" and i["side"] == "TOP")
        bottom = next(i for i in st["images"] if i["camera_id"] == "C1" and i["side"] == "BOTTOM")
        self.assertTrue(top["defects"])
        self.assertGreater(top["score"], bottom["score"])

        status, body, _ = self.ui.req("POST", "/api/ack", token=self.op)
        self.assertEqual((status, body), (200, {"ok": True}))
        st = self.state()
        self.assertEqual(st["state"], "IDLE")
        self.assertEqual(st["counters"], {"total": 1, "good": 0, "bad": 1})
        self.assertTrue(self.outputs.red)
        self.assertFalse(self.outputs.alarm)

        session_id, part_dir = self.saved_part()
        self.assertEqual(len(self.pngs(part_dir)), 2)
        self.assertEqual(self.app.store.verify_part(part_dir), [])
        doc = self.part_json(part_dir)
        self.assertEqual(doc["final_verdict"], "BAD")
        self.assertEqual(doc["disposition"], "quarantine")
        top_doc = next(i for i in doc["images"] if i["camera_id"] == "C1" and i["side"] == "TOP")
        bottom_doc = next(i for i in doc["images"] if i["camera_id"] == "C1" and i["side"] == "BOTTOM")
        self.assertTrue(top_doc["defects"])
        box = tuple(top_doc["defects"][0]["box"])
        self.assertGreater(_overlap_area(box, BLOB_BOX), 0)  # in original pixel coordinates
        self.assertEqual(box, BLOB_BOX)  # block factor 1 for a 60-pixel-wide image
        self.assertEqual(bottom_doc["defects"], [])

    # 3a. MES safety: a partial station sends nothing, even with a token
    def test_partial_station_sends_nothing_to_mes_even_with_a_token(self) -> None:
        self.one_camera(allow_partial=False)
        self.blobs.add(("C1", "TOP"))
        self.press_part()
        self.ack()

        self.assertFalse(self.app.mes_enabled)
        self.assertEqual(self.app.sync_once(), {"sent": 0, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(self.mes.requests, [])
        self.assertEqual(self.app.outbox.pending_count(), 0)
        self.assertEqual(self.app.outbox.list_items(), [])

        reasons = [e["reason"] for e in self.events("mes_disabled")]
        self.assertTrue(any("partial station" in r for r in reasons), reasons)
        self.assertEqual(self.state()["counters"], {"total": 1, "good": 0, "bad": 1})

    # 3b. MES: the user's explicit choice path
    def test_partial_station_with_allow_flag_posts_the_bad_part_once(self) -> None:
        self.one_camera(allow_partial=True)
        self.assertTrue(self.app.mes_enabled)
        self.blobs.add(("C1", "TOP"))
        self.press_part()
        self.ack()

        self.assertEqual(self.app.sync_once(), {"sent": 1, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(self.app.sync_once(), {"sent": 0, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(len(self.mes.requests), 1)

        session_id, _ = self.saved_part()
        post = self.posted()[0]
        self.assertEqual(post["record"]["key"], session_id)
        self.assertEqual(post["record"]["auth"], f"Bearer {TOKEN}")
        self.assertEqual(post["record"]["status"], 201)
        self.assertEqual(post["fields"]["verdict"], "BAD")
        self.assertEqual(post["fields"]["visionSessionId"], session_id)
        self.assertEqual(json.loads(post["fields"]["checklistResults"]), {"CR-1": "fail", "CR-2": "pass"})
        self.assertEqual(list(post["files"]), ["defectPhoto_CR-1"])  # exactly one defect photo
        filename, data = post["files"]["defectPhoto_CR-1"]
        self.assertEqual(filename, "C1_TOP.png")
        self.assertEqual(data, image_bytes(0, blob=True))  # the TOP frame of the BAD part
        self.assertTrue(self.events("mes_partial_allowed"))

    def ack(self) -> None:
        status, body, _ = self.ui.req("POST", "/api/ack", token=self.op)
        self.assertEqual((status, body), (200, {"ok": True}))

    # 5. Banner data and the served page
    def test_snapshot_and_page_carry_the_proof_of_concept_banner(self) -> None:
        self.one_camera(allow_partial=False)
        st = self.state()
        self.assertIn("partial_station", st)
        self.assertIn("camera_ids", st)
        self.assertIs(st["partial_station"], True)
        self.assertEqual(st["camera_ids"], ["C1"])

        status, body = self.raw_get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"PROOF OF CONCEPT", body)
        self.assertIn(b'"keydown"', body)  # the Enter key handler
        self.assertIn(b'"Enter"', body)


# ------------------------------------------------------------ four cameras

@unittest.skipIf(np is None, "numpy not installed")
class FourCameraDefaultE2E(_Harness):

    # 4. The 4-camera default is untouched
    def test_four_camera_default_is_not_partial_and_posts_with_a_token(self) -> None:
        def frame_bytes(cam: str, side: Side, n: int) -> bytes:
            return f"SCORE:0.05 |{cam}|{side.value}|{n}".encode("ascii")

        cfg = self.config(
            allow_partial=False,
            cameras={"kind": "fake"},
            model={"kind": "marker", "version": "poc-4cam-marker"},
        )
        self.start(cfg, cameras=FakeCameraSet(frame_bytes=frame_bytes),
                   model=MarkerModel(version="poc-4cam-marker"))

        st = self.state()
        self.assertIs(st["partial_station"], False)
        self.assertEqual(st["camera_ids"], ["C1", "C2", "C3", "C4"])
        self.assertTrue(self.app.mes_enabled)
        self.assertEqual(self.events("mes_disabled"), [])

        self.press_part()
        self.assertEqual(self.state()["verdict"], "GOOD")
        session_id, part_dir = self.saved_part()
        self.assertEqual(len(self.pngs(part_dir)), 8)
        self.assertIs(self.part_json(part_dir)["partial_station"], False)

        self.assertEqual(self.app.sync_once(), {"sent": 1, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(len(self.mes.requests), 1)
        self.assertEqual(self.mes.requests[0]["key"], session_id)


# ------------------------------------------------------------ live view

@unittest.skipIf(np is None, "numpy not installed")
class OneCameraLiveViewE2E(_Harness):
    """GET /api/preview: a read-only live view that never changes the station."""

    def preview_get(self, camera: str, token: str | None):
        """Return (status, headers with lower-case names, body) for one live-view request."""
        conn = http.client.HTTPConnection(self.ui.host, self.ui.port, timeout=10)
        try:
            headers = {"Cookie": f"station_session={token}"} if token else {}
            conn.request("GET", "/api/preview?camera=" + camera, headers=headers)
            resp = conn.getresponse()
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
        finally:
            conn.close()

    def event_count(self) -> int:
        with contextlib.closing(sqlite3.connect(str(self.app.store.db_path))) as conn:
            return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def test_preview_returns_a_jpeg_frame_to_a_logged_in_operator(self) -> None:
        self.one_camera(allow_partial=False)
        status, headers, body = self.preview_get("C1", self.op)
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "image/jpeg")
        self.assertIn("no-store", headers["cache-control"])
        self.assertTrue(body.startswith(b"FAKEJPEG:C1"), body[:40])

    def test_preview_needs_a_login(self) -> None:
        self.one_camera(allow_partial=False)
        status, _, _ = self.preview_get("C1", None)
        self.assertEqual(status, 401)

    def test_preview_rejects_a_name_that_is_not_a_camera(self) -> None:
        self.one_camera(allow_partial=False)
        status, _, _ = self.preview_get("C9", self.op)
        self.assertEqual(status, 400)

    def test_preview_of_a_camera_this_station_does_not_have_is_204(self) -> None:
        self.one_camera(allow_partial=False)  # C1 only
        status, _, body = self.preview_get("C2", self.op)
        self.assertEqual((status, body), (204, b""))

    def test_previews_change_no_state_counters_or_events(self) -> None:
        self.one_camera(allow_partial=False)
        self.clock.now += 1.0
        status, body, _ = self.ui.req("POST", "/api/press", token=self.op)  # TOP taken: WAIT_FLIP
        self.assertEqual((status, body), (200, {"accepted": True}))
        state_before = self.state()
        events_before = self.event_count()

        for cam, expected in (("C1", 200), ("C1", 200), ("C2", 204), ("C1", 200)):
            status, _, _ = self.preview_get(cam, self.op)
            self.assertEqual(status, expected, cam)

        self.assertEqual(self.state(), state_before)
        self.assertEqual(self.event_count(), events_before)
        self.assertEqual(self.state()["state"], "WAIT_FLIP")


if __name__ == "__main__":
    unittest.main()
