"""Tests for station/mes_client.py: request building and posting to a local fake MES."""
from __future__ import annotations

import email.parser
import email.policy
import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from station.contracts import (
    Defect,
    Frame,
    ImageResult,
    Override,
    PartRecord,
    Side,
    Thresholds,
    Verdict,
)
from station.mes_client import MappingError, MesClient, MesConfig, build_request

TOKEN = "test-token-123"
CAMERAS = ("C1", "C2", "C3", "C4")
SIDES = (Side.TOP, Side.BOTTOM)


def make_frames() -> list[Frame]:
    return [
        Frame(
            camera_id=cam, side=side, seq=0, timestamp_utc="2026-10-23T08:30:12Z",
            data=f"{cam}-{side.value}".encode(), ext="png",
        )
        for side in SIDES
        for cam in CAMERAS
    ]


def make_results(defects: dict[tuple[str, Side], list[Defect]] | None = None,
                 scores: dict[tuple[str, Side], float] | None = None) -> list[ImageResult]:
    defects = defects or {}
    scores = scores or {}
    out = []
    for side in SIDES:
        for cam in CAMERAS:
            ds = tuple(defects.get((cam, side), ()))
            score = max([d.score for d in ds] + [scores.get((cam, side), 0.0)])
            out.append(ImageResult(camera_id=cam, side=side, score=score, defects=ds))
    return out


def make_record(results: list[ImageResult], final: Verdict = Verdict.BAD,
                machine: Verdict = Verdict.BAD, override: Override | None = None) -> PartRecord:
    return PartRecord(
        part_id="P-1", session_id="sess-1", station_id="station-1", part_number="PN-100",
        lot="LOT-9", operator="op-anna", supervisor="sup-bob", material_handler="mh-cy",
        started_utc="2026-10-23T08:30:00Z", model_version="m-2.4",
        thresholds=Thresholds(t_low=0.3, t_img=0.6), frames=make_frames(), results=results,
        machine_verdict=machine, final_verdict=final, override=override,
    )


CRITERIA = ["crit_scratch", "crit_dent", "crit_label"]


def make_cfg(**overrides) -> MesConfig:
    base = dict(
        base_url="http://127.0.0.1:1", token=TOKEN, criteria_ids=CRITERIA,
        class_to_criterion={"scratch": "crit_scratch", "dent": "crit_dent"},
    )
    base.update(overrides)
    return MesConfig(**base)


class BuildRequestTests(unittest.TestCase):
    def test_good_part_passes_everything_and_sends_no_photos(self):
        record = make_record(make_results(), final=Verdict.GOOD, machine=Verdict.GOOD)
        fields, files = build_request(record, make_cfg(), archive_ref="arch/sess-1")

        self.assertEqual(files, [])
        self.assertEqual(json.loads(fields["checklistResults"]),
                         {c: "pass" for c in CRITERIA})
        self.assertEqual(fields["verdict"], "GOOD")
        self.assertEqual(fields["quantityInspected"], "1")
        self.assertEqual(fields["stationId"], "station-1")
        self.assertEqual(fields["partNumber"], "PN-100")
        self.assertEqual(fields["supervisor"], "sup-bob")
        self.assertEqual(fields["operators"], "op-anna")
        self.assertEqual(fields["materialHandler"], "mh-cy")
        self.assertEqual(fields["visionSessionId"], "sess-1")
        self.assertEqual(fields["modelVersion"], "m-2.4")
        self.assertEqual(fields["archiveRef"], "arch/sess-1")
        self.assertIn("sess-1", fields["notes"])
        self.assertIn("m-2.4", fields["notes"])
        self.assertIn("GOOD", fields["notes"])
        for key in ("rejectedQuantity", "rejectionReason", "disposition", "overrideBy"):
            self.assertNotIn(key, fields)

    def test_bad_part_with_two_classes_gets_two_correct_photos(self):
        defects = {
            ("C1", Side.TOP): [Defect("scratch", 0.7, (0, 0, 5, 5))],
            ("C2", Side.TOP): [Defect("scratch", 0.9, (0, 0, 5, 5))],
            ("C3", Side.BOTTOM): [Defect("dent", 0.8, (1, 1, 5, 5))],
            ("C4", Side.BOTTOM): [Defect("dent", 0.95, (1, 1, 5, 5))],
        }
        record = make_record(make_results(defects))
        fields, files = build_request(record, make_cfg(), archive_ref="arch/x")

        self.assertEqual(json.loads(fields["checklistResults"]),
                         {"crit_scratch": "fail", "crit_dent": "fail", "crit_label": "pass"})
        self.assertEqual(fields["verdict"], "BAD")
        self.assertEqual(fields["rejectedQuantity"], "1")
        self.assertEqual(fields["rejectionReason"], "dent, scratch")
        self.assertEqual(fields["disposition"], "quarantine")

        by_name = {name: (filename, data) for name, filename, data in files}
        self.assertEqual(sorted(by_name), ["defectPhoto_crit_dent", "defectPhoto_crit_scratch"])
        self.assertEqual(by_name["defectPhoto_crit_scratch"], ("C2_TOP.png", b"C2-TOP"))
        self.assertEqual(by_name["defectPhoto_crit_dent"], ("C4_BOTTOM.png", b"C4-BOTTOM"))
        self.assertEqual(len(files), 2)  # never the other images

    def test_unknown_class_falls_back_to_default(self):
        defects = {("C1", Side.TOP): [Defect("crack", 0.9, (0, 0, 5, 5))]}
        record = make_record(make_results(defects))
        cfg = make_cfg(default_criterion_id="crit_label", default_bad_disposition="released")
        fields, files = build_request(record, cfg, archive_ref="a")

        self.assertEqual(json.loads(fields["checklistResults"]),
                         {"crit_scratch": "pass", "crit_dent": "pass", "crit_label": "fail"})
        self.assertEqual(fields["rejectionReason"], "crack")
        self.assertEqual(fields["disposition"], "released")
        self.assertEqual([f[0] for f in files], ["defectPhoto_crit_label"])
        self.assertEqual(files[0][1], "C1_TOP.png")

    def test_unknown_class_without_default_raises(self):
        defects = {("C1", Side.TOP): [Defect("crack", 0.9, (0, 0, 5, 5))]}
        record = make_record(make_results(defects))
        with self.assertRaises(MappingError):
            build_request(record, make_cfg(), archive_ref="a")

    def test_bad_part_without_defects_uses_default_and_highest_frame(self):
        results = make_results(scores={("C3", Side.BOTTOM): 0.4})  # score with no defect listed
        record = make_record(results)  # supervisor resolved REVIEW to BAD
        cfg = make_cfg(default_criterion_id="crit_label")
        fields, files = build_request(record, cfg, archive_ref="a")

        self.assertEqual(fields["rejectionReason"], "supervisor decision")
        self.assertEqual(json.loads(fields["checklistResults"])["crit_label"], "fail")
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0][0], "defectPhoto_crit_label")
        self.assertEqual(files[0][1], "C3_BOTTOM.png")  # highest-score frame overall
        self.assertEqual(files[0][2], b"C3-BOTTOM")

    def test_bad_part_without_defects_and_without_default_raises(self):
        record = make_record(make_results())
        with self.assertRaises(MappingError):
            build_request(record, make_cfg(), archive_ref="a")

    def test_review_is_rejected(self):
        record = make_record(make_results(), final=Verdict.REVIEW, machine=Verdict.REVIEW)
        with self.assertRaises(ValueError):
            build_request(record, make_cfg(), archive_ref="a")

    def test_override_to_good_is_released_with_override_fields(self):
        override = Override(original_verdict=Verdict.BAD, new_verdict=Verdict.GOOD,
                            reason="dust, not a defect", by="sup-bob")
        record = make_record(make_results(), final=Verdict.GOOD, machine=Verdict.BAD,
                             override=override)
        fields, files = build_request(record, make_cfg(), archive_ref="a")

        self.assertEqual(fields["disposition"], "released")
        self.assertEqual(fields["overrideOriginalVerdict"], "BAD")
        self.assertEqual(fields["overrideReason"], "dust, not a defect")
        self.assertEqual(fields["overrideBy"], "sup-bob")
        self.assertIn("machine verdict BAD", fields["notes"])
        self.assertEqual(files, [])

    def test_token_is_not_in_repr(self):
        self.assertNotIn(TOKEN, repr(make_cfg()))


def parse_multipart(content_type: str, body: bytes) -> dict[str, list[tuple[str | None, bytes]]]:
    raw = b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(raw)
    parts: dict[str, list[tuple[str | None, bytes]]] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        parts.setdefault(name, []).append((part.get_filename(), part.get_payload(decode=True)))
    return parts


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (http.server naming)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.seen.append({
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "body": body,
        })
        status, payload = self.server.reply
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # noqa: A002
        pass


class MesClientPostTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.seen = []
        self.server.reply = (201, {"id": 42})
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.cfg = make_cfg(base_url=f"http://127.0.0.1:{self.server.server_address[1]}")
        self.client = MesClient(self.cfg)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post_bad_part(self, key="sess-1"):
        defects = {("C2", Side.TOP): [Defect("scratch", 0.9, (0, 0, 5, 5))]}
        record = make_record(make_results(defects))
        fields, files = build_request(record, self.cfg, archive_ref="arch/sess-1")
        return self.client.post(fields, files, idempotency_key=key)

    def test_201_is_ok_and_body_matches_the_request(self):
        result = self.post_bad_part()

        self.assertEqual((result.status, result.ok, result.replay), (201, True, False))
        self.assertEqual(result.event_id, 42)
        self.assertIsNone(result.error)
        self.assertFalse(result.retryable)

        req = self.server.seen[0]
        self.assertEqual(req["path"], "/api/machine/inspection-events")
        self.assertEqual(req["headers"]["authorization"], f"Bearer {TOKEN}")
        self.assertEqual(req["headers"]["idempotency-key"], "sess-1")
        self.assertTrue(req["headers"]["content-type"].startswith("multipart/form-data; boundary="))

        parts = parse_multipart(req["headers"]["content-type"], req["body"])
        self.assertEqual(parts["stationId"][0][1], b"station-1")
        self.assertEqual(parts["verdict"][0][1], b"BAD")
        self.assertEqual(parts["rejectionReason"][0][1], b"scratch")
        self.assertEqual(json.loads(parts["checklistResults"][0][1]),
                         {"crit_scratch": "fail", "crit_dent": "pass", "crit_label": "pass"})
        self.assertEqual(parts["defectPhoto_crit_scratch"], [("C2_TOP.png", b"C2-TOP")])
        self.assertNotIn("defectPhoto_crit_dent", parts)
        self.assertEqual(sorted(parts), sorted([
            "stationId", "partNumber", "quantityInspected", "checklistResults", "supervisor",
            "operators", "materialHandler", "verdict", "visionSessionId", "modelVersion",
            "archiveRef", "notes", "rejectedQuantity", "rejectionReason", "disposition",
            "defectPhoto_crit_scratch",
        ]))

    def test_200_is_replay(self):
        self.server.reply = (200, {"id": 42, "replay": True})
        result = self.post_bad_part()
        self.assertEqual((result.status, result.ok, result.replay), (200, True, True))
        self.assertEqual(result.event_id, 42)

    def test_422_is_permanent_data_error(self):
        self.server.reply = (422, {"error": "verdict disagrees with checklistResults"})
        result = self.post_bad_part()
        self.assertFalse(result.ok)
        self.assertFalse(result.retryable)
        self.assertEqual(result.status, 422)
        self.assertIn("verdict disagrees", result.error)

    def test_401_is_retryable_and_never_echoes_the_token(self):
        self.server.reply = (401, {"error": f"invalid token {TOKEN}"})
        result = self.post_bad_part()
        self.assertFalse(result.ok)
        self.assertTrue(result.retryable)
        self.assertEqual(result.status, 401)
        self.assertIn("token", result.error)
        self.assertNotIn(TOKEN, result.error)

    def test_500_is_retryable(self):
        self.server.reply = (500, {"error": "database down"})
        result = self.post_bad_part()
        self.assertFalse(result.ok)
        self.assertTrue(result.retryable)
        self.assertEqual(result.status, 500)

    def test_connection_refused_is_retryable(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()  # nothing listens on this port now
        client = MesClient(make_cfg(base_url=f"http://127.0.0.1:{port}", timeout_s=2.0))
        record = make_record(make_results(), final=Verdict.GOOD, machine=Verdict.GOOD)
        fields, files = build_request(record, client.cfg, archive_ref="a")
        result = client.post(fields, files, idempotency_key="sess-x")
        self.assertFalse(result.ok)
        self.assertTrue(result.retryable)
        self.assertEqual(result.status, 0)
        self.assertNotIn(TOKEN, result.error)

    def test_good_post_has_no_file_parts(self):
        record = make_record(make_results(), final=Verdict.GOOD, machine=Verdict.GOOD)
        fields, files = build_request(record, self.cfg, archive_ref="a")
        self.client.post(fields, files, idempotency_key="sess-good")
        parts = parse_multipart(self.server.seen[0]["headers"]["content-type"],
                                self.server.seen[0]["body"])
        self.assertFalse([n for n in parts if n.startswith("defectPhoto_")])
        self.assertNotIn("rejectedQuantity", parts)


if __name__ == "__main__":
    unittest.main()
