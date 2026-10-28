"""Tests for the HOME TRIAL fake MES (tools/fake_mes.py). Loopback only, standard library."""
from __future__ import annotations

import http.client
import json
import os
import pathlib
import secrets
import subprocess
import sys
import tempfile
import unittest

from station.mes_client import MesClient, MesConfig
from station.parts import MesPartsClient, PartsError
from tools.fake_mes import FakeMesServer, load_criteria, load_parts

REPO = pathlib.Path(__file__).resolve().parent.parent
TOKEN = "fake-mes-test-" + secrets.token_hex(12)  # made up per run, never written anywhere
UNSET_ENV = "VISION_FAKE_MES_TEST_UNSET_TOKEN"
BOUNDARY = "fakemesboundary7d3f"


def multipart(fields: dict[str, str], photos: dict[str, tuple[str, bytes]]) -> bytes:
    out = bytearray()
    for name, value in fields.items():
        out += (f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n').encode()
        out += value.encode("utf-8") + b"\r\n"
    for name, (filename, data) in photos.items():
        out += (f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\nContent-Type: image/png\r\n\r\n').encode()
        out += data + b"\r\n"
    out += f"--{BOUNDARY}--\r\n".encode()
    return bytes(out)


EVENT_FIELDS = {"stationId": "ST-1", "partNumber": "HT-LINER-1", "verdict": "BAD",
                "checklistResults": '{"CR-1": "fail", "CR-2": "pass"}'}
EVENT_PHOTO = {"defectPhoto_CR-1": ("C1_TOP.png", b"\x89PNG-fake-bytes" * 10)}


class FakeMesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="fake-mes-test-")
        self.addCleanup(tmp.cleanup)
        self.out_dir = pathlib.Path(tmp.name) / "out"
        self.server = FakeMesServer(TOKEN, out_dir=self.out_dir).start()
        self.addCleanup(self.server.stop)
        self.port = self.server.server_address[1]

    def request(self, method: str, path: str, *, token: str | None = TOKEN,
                body: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, dict]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            hdrs = dict(headers or {})
            if token is not None:
                hdrs["Authorization"] = f"Bearer {token}"
            conn.request(method, path, body=body, headers=hdrs)
            resp = conn.getresponse()
            raw = resp.read()
            return resp.status, json.loads(raw.decode("utf-8")) if raw else {}
        finally:
            conn.close()

    def post_event(self, key: str | None, fields: dict[str, str] = EVENT_FIELDS,
                   photos: dict[str, tuple[str, bytes]] = EVENT_PHOTO) -> tuple[int, dict]:
        headers = {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"}
        if key is not None:
            headers["Idempotency-Key"] = key
        return self.request("POST", "/api/machine/inspection-events",
                            body=multipart(fields, photos), headers=headers)

    def events_lines(self) -> list[str]:
        path = self.out_dir / "events.jsonl"
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8").splitlines()


class TestAuth(FakeMesTestCase):
    def test_get_without_token_is_401(self) -> None:
        status, _ = self.request("GET", "/api/machine/parts/search?q=HT", token=None)
        self.assertEqual(status, 401)

    def test_get_with_wrong_token_is_401(self) -> None:
        status, _ = self.request("GET", "/api/machine/parts/HT-LINER-1", token="wrong")
        self.assertEqual(status, 401)

    def test_post_without_token_is_401_and_stores_nothing(self) -> None:
        headers = {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}",
                   "Idempotency-Key": "k-noauth"}
        status, _ = self.request("POST", "/api/machine/inspection-events", token=None,
                                 body=multipart(EVENT_FIELDS, EVENT_PHOTO), headers=headers)
        self.assertEqual(status, 401)
        self.assertEqual(self.events_lines(), [])


class TestParts(FakeMesTestCase):
    def test_search_returns_sample_part(self) -> None:
        status, rows = self.request("GET", "/api/machine/parts/search?q=sample&limit=5")
        self.assertEqual(status, 200)
        self.assertEqual([r["part_number"] for r in rows], ["HT-LINER-1"])

    def test_search_no_match_is_empty_list(self) -> None:
        status, rows = self.request("GET", "/api/machine/parts/search?q=zzz-nothing")
        self.assertEqual((status, rows), (200, []))

    def test_part_lookup_and_unknown_part(self) -> None:
        status, part = self.request("GET", "/api/machine/parts/HT-LINER-1")
        self.assertEqual(status, 200)
        self.assertEqual(part["part_number"], "HT-LINER-1")
        status, _ = self.request("GET", "/api/machine/parts/NOPE")
        self.assertEqual(status, 404)

    def test_criteria_two_visual_items(self) -> None:
        status, items = self.request("GET", "/api/machine/criteria/HT-LINER-1")
        self.assertEqual(status, 200)
        self.assertEqual([i["id"] for i in items], ["CR-1", "CR-2"])
        self.assertTrue(all(i["check_type"] == "visual" and i["active"] for i in items))

    def test_criteria_unknown_part_is_404(self) -> None:
        status, _ = self.request("GET", "/api/machine/criteria/NOPE")
        self.assertEqual(status, 404)

    def test_unknown_route_is_404(self) -> None:
        status, _ = self.request("GET", "/api/machine/other")
        self.assertEqual(status, 404)


class TestInspectionEvents(FakeMesTestCase):
    def test_first_post_is_201_and_replay_is_200_same_id(self) -> None:
        status, first = self.post_event("key-1")
        self.assertEqual((status, first), (201, {"id": 1}))
        status, again = self.post_event("key-1")
        self.assertEqual((status, again), (200, {"id": 1}))
        self.assertEqual(self.server.replays, 1)

    def test_new_key_gets_next_id(self) -> None:
        self.post_event("key-a")
        status, body = self.post_event("key-b")
        self.assertEqual((status, body), (201, {"id": 2}))

    def test_missing_idempotency_key_is_400(self) -> None:
        status, body = self.post_event(None)
        self.assertEqual(status, 400)
        self.assertIn("Idempotency-Key", body["error"])
        self.assertEqual(self.events_lines(), [])

    def test_wrong_content_type_is_400(self) -> None:
        status, _ = self.request("POST", "/api/machine/inspection-events", body=b"{}",
                                 headers={"Content-Type": "application/json",
                                          "Idempotency-Key": "k-json"})
        self.assertEqual(status, 400)

    def test_one_jsonl_line_per_accepted_event_without_photo_bytes(self) -> None:
        self.post_event("key-1")
        self.post_event("key-1")  # replay: not written again
        self.post_event("key-2")
        lines = self.events_lines()
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["id"], 1)
        self.assertEqual(first["key"], "key-1")
        self.assertEqual(first["fields"]["partNumber"], "HT-LINER-1")
        self.assertEqual(first["fields"]["verdict"], "BAD")
        self.assertEqual(first["photos"], [{"field": "defectPhoto_CR-1", "filename": "C1_TOP.png",
                                            "bytes": len(EVENT_PHOTO["defectPhoto_CR-1"][1])}])
        self.assertNotIn("fake-bytes", lines[0])

    def test_token_never_appears_in_events_file(self) -> None:
        self.post_event("key-tok", fields={**EVENT_FIELDS, "notes": "no secret here"})
        self.post_event("key-tok")
        text = (self.out_dir / "events.jsonl").read_text(encoding="utf-8")
        self.assertNotIn(TOKEN, text)
        self.assertNotIn("Bearer", text)

    def test_restart_keeps_ids_for_known_keys(self) -> None:
        self.post_event("key-1")
        self.server.stop()
        restarted = FakeMesServer(TOKEN, out_dir=self.out_dir).start()
        self.addCleanup(restarted.stop)
        self.server = restarted
        self.port = restarted.server_address[1]
        self.assertEqual(self.post_event("key-1"), (200, {"id": 1}))
        self.assertEqual(self.post_event("key-2"), (201, {"id": 2}))


class TestLoadFiles(unittest.TestCase):
    def test_parts_and_criteria_files_are_served(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fake-mes-files-") as tmp:
            parts_path = os.path.join(tmp, "parts.json")
            crit_path = os.path.join(tmp, "criteria.json")
            with open(parts_path, "w", encoding="utf-8") as fh:
                json.dump([{"part_number": "X-9", "description": "Custom door panel"}], fh)
            with open(crit_path, "w", encoding="utf-8") as fh:
                json.dump({"X-9": [{"id": 5, "criteria_name": "Crack", "check_type": "visual",
                                    "active": 1}]}, fh)
            server = FakeMesServer(TOKEN, parts=load_parts(parts_path),
                                   criteria=load_criteria(crit_path),
                                   out_dir=os.path.join(tmp, "out")).start()
            self.addCleanup(server.stop)
            client = MesPartsClient(server.base_url, TOKEN)
            self.assertEqual([p["part_number"] for p in client.search("door")], ["X-9"])
            self.assertEqual(client.criteria("X-9")[0]["criteria_name"], "Crack")


class TestStationClients(FakeMesTestCase):
    def test_station_clients_work_against_fake(self) -> None:
        base = self.server.base_url

        parts = MesPartsClient(base, TOKEN)
        found = parts.search("sample", limit=5)
        self.assertEqual([p["part_number"] for p in found], ["HT-LINER-1"])
        self.assertEqual(parts.get_part("HT-LINER-1")["part_number"], "HT-LINER-1")
        self.assertIsNone(parts.get_part("NOPE"))
        self.assertEqual([c["id"] for c in parts.criteria("HT-LINER-1")], ["CR-1", "CR-2"])
        with self.assertRaises(PartsError) as ctx:
            MesPartsClient(base, "not-the-token").search("sample")
        self.assertNotIn("not-the-token", str(ctx.exception))

        cfg = MesConfig(base_url=base, token=TOKEN, criteria_ids=["CR-1", "CR-2"],
                        class_to_criterion={}, default_criterion_id="CR-1")
        client = MesClient(cfg)
        fields = {"stationId": "ST-1", "partNumber": "HT-LINER-1", "verdict": "BAD",
                  "visionSessionId": "sess-1", "checklistResults": "{}"}
        photos = [("defectPhoto_CR-1", "C1_TOP.png", b"\x89PNG-fake")]
        first = client.post(fields, photos, "sess-1")
        self.assertEqual((first.status, first.ok, first.replay, first.event_id),
                         (201, True, False, 1))
        again = client.post(fields, photos, "sess-1")
        self.assertEqual((again.status, again.ok, again.replay, again.event_id),
                         (200, True, True, 1))


class TestCommandLine(unittest.TestCase):
    def test_missing_token_env_exits_nonzero(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != UNSET_ENV}
        proc = subprocess.run(
            [sys.executable, "-m", "tools.fake_mes", "--token-env", UNSET_ENV, "--port", "0"],
            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(UNSET_ENV, proc.stderr)


if __name__ == "__main__":
    unittest.main()
