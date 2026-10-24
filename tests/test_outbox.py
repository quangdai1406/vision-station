"""Tests for station/outbox.py: durable queue, retry timing and ordering."""
from __future__ import annotations

import pathlib
import tempfile
import unittest

from station.mes_client import PostResult
from station.outbox import Outbox


class ScriptedClient:
    """Stands in for MesClient. Returns queued results in order; records every call."""

    def __init__(self, *results: PostResult):
        self.queue = list(results)
        self.calls: list[dict] = []

    def post(self, fields, files, idempotency_key):
        self.calls.append({"fields": fields, "files": files, "key": idempotency_key})
        if self.queue:
            return self.queue.pop(0)
        return PostResult(201, True, False, 1000 + len(self.calls), None, False)


def ok(event_id: int = 7) -> PostResult:
    return PostResult(201, True, False, event_id, None, False)


def permanent(msg: str = "MES rejected this inspection (HTTP 422): bad") -> PostResult:
    return PostResult(422, False, False, None, msg, False)


def retryable(msg: str = "MES temporarily unavailable (HTTP 500)") -> PostResult:
    return PostResult(500, False, False, None, msg, True)


FILES = [("defectPhoto_crit_scratch", "C2_TOP.png", b"\x89PNG-bytes")]


class OutboxTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.box = Outbox(self.root)

    def tearDown(self):
        self.box.close()
        self._tmp.cleanup()

    def enqueue(self, sid: str, fields: dict | None = None):
        return self.box.enqueue(sid, fields or {"stationId": "station-1", "verdict": "BAD"}, FILES)

    def test_success_marks_sent_with_event_id_and_sends_bytes(self):
        self.enqueue("sess-1")
        client = ScriptedClient(ok(event_id=7))

        counts = self.box.drain(client, now=100.0)

        self.assertEqual(counts, {"sent": 1, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(self.box.pending_count(), 0)
        call = client.calls[0]
        self.assertEqual(call["key"], "sess-1")
        self.assertEqual(call["fields"], {"stationId": "station-1", "verdict": "BAD"})
        self.assertEqual(call["files"], [("defectPhoto_crit_scratch", "C2_TOP.png", b"\x89PNG-bytes")])
        item = self.box.list_items("sent")[0]
        self.assertEqual((item["session_id"], item["mes_event_id"]), ("sess-1", 7))

    def test_retry_backoff_timing_with_injected_now(self):
        self.enqueue("sess-1")
        client = ScriptedClient(retryable(), retryable(), ok())

        # attempt 1 fails at t=1000: attempts=1, next = 1000 + 5 * 2**1 = 1010
        counts = self.box.drain(client, now=1000.0, base_delay_s=5.0, max_delay_s=600.0)
        self.assertEqual(counts, {"sent": 0, "failed_permanent": 0, "retry_later": 1})
        item = self.box.list_items("pending")[0]
        self.assertEqual((item["attempts"], item["next_attempt"]), (1, 1010.0))
        self.assertIn("500", item["last_error"])

        # not due yet: nothing is posted
        self.assertEqual(self.box.drain(client, now=1009.9), {"sent": 0, "failed_permanent": 0, "retry_later": 0})
        self.assertEqual(len(client.calls), 1)

        # attempt 2 fails at t=1010: next = 1010 + 5 * 2**2 = 1030
        self.box.drain(client, now=1010.0, base_delay_s=5.0, max_delay_s=600.0)
        item = self.box.list_items("pending")[0]
        self.assertEqual((item["attempts"], item["next_attempt"]), (2, 1030.0))

        # attempt 3 succeeds at t=1030
        self.assertEqual(self.box.drain(client, now=1030.0)["sent"], 1)
        self.assertEqual(self.box.pending_count(), 0)

    def test_backoff_is_capped_at_max_delay(self):
        self.enqueue("sess-1")
        client = ScriptedClient(retryable(), retryable())
        self.box.drain(client, now=0.0, base_delay_s=5.0, max_delay_s=12.0)
        self.assertEqual(self.box.list_items("pending")[0]["next_attempt"], 10.0)  # 5 * 2**1
        self.box.drain(client, now=10.0, base_delay_s=5.0, max_delay_s=12.0)
        self.assertEqual(self.box.list_items("pending")[0]["next_attempt"], 22.0)  # capped: 10 + 12

    def test_ordering_is_kept_when_first_item_fails_retryably(self):
        self.enqueue("sess-A")
        self.enqueue("sess-B")
        client = ScriptedClient(retryable(), ok(8), ok(9))

        counts = self.box.drain(client, now=0.0)
        self.assertEqual(counts, {"sent": 0, "failed_permanent": 0, "retry_later": 1})
        self.assertEqual([c["key"] for c in client.calls], ["sess-A"])  # B was not attempted
        self.assertEqual(self.box.pending_count(), 2)

        counts = self.box.drain(client, now=10.0)
        self.assertEqual(counts["sent"], 2)
        self.assertEqual([c["key"] for c in client.calls], ["sess-A", "sess-A", "sess-B"])

    def test_permanent_failure_is_not_retried_and_does_not_block_others(self):
        self.enqueue("sess-A")
        self.enqueue("sess-B")
        client = ScriptedClient(permanent("MES rejected this inspection (HTTP 422): verdict mismatch"), ok())

        counts = self.box.drain(client, now=0.0)
        self.assertEqual(counts, {"sent": 1, "failed_permanent": 1, "retry_later": 0})

        # later drains never post sess-A again
        self.box.drain(client, now=1e9)
        self.assertEqual([c["key"] for c in client.calls], ["sess-A", "sess-B"])
        failed = self.box.failed_permanent_items()
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["session_id"], "sess-A")
        self.assertEqual(failed[0]["attempts"], 1)
        self.assertIn("verdict mismatch", failed[0]["last_error"])
        self.assertEqual(self.box.pending_count(), 0)

    def test_data_survives_restart(self):
        self.enqueue("sess-1")
        self.box.close()

        box2 = Outbox(self.root)  # new process, same root
        try:
            self.assertEqual(box2.pending_count(), 1)
            client = ScriptedClient(ok(event_id=5))
            self.assertEqual(box2.drain(client, now=0.0)["sent"], 1)
        finally:
            box2.close()

        self.box = Outbox(self.root)  # restart again: sent state is kept
        self.assertEqual(self.box.pending_count(), 0)
        self.assertEqual(self.box.list_items("sent")[0]["mes_event_id"], 5)

    def test_enqueue_is_idempotent_per_session(self):
        self.assertTrue(self.enqueue("sess-1", {"verdict": "BAD"}))
        self.assertFalse(self.box.enqueue("sess-1", {"verdict": "GOOD"}, []))

        self.assertEqual(len(self.box.list_items()), 1)
        client = ScriptedClient(ok())
        self.box.drain(client, now=0.0)
        self.assertEqual(client.calls[0]["fields"], {"verdict": "BAD"})  # first enqueue kept
        self.assertEqual(len(client.calls), 1)

    def test_unsafe_session_id_is_rejected(self):
        for bad in ("../escape", "a/b", "", ".hidden"):
            with self.assertRaises(ValueError):
                self.enqueue(bad)

    def test_oldest_pending_age(self):
        self.assertIsNone(self.box.oldest_pending_age_s())

        self.enqueue("sess-1")
        created = self.box.list_items()[0]["created_at"]
        self.assertAlmostEqual(self.box.oldest_pending_age_s(now=created + 30.0), 30.0, places=6)

        self.enqueue("sess-2")  # newer item does not change the oldest age
        self.assertAlmostEqual(self.box.oldest_pending_age_s(now=created + 45.0), 45.0, places=6)

        self.box.drain(ScriptedClient(ok(), ok()), now=created + 50.0)
        self.assertIsNone(self.box.oldest_pending_age_s(now=created + 60.0))

    def test_list_items_filters_by_status(self):
        self.enqueue("sess-A")
        self.enqueue("sess-B")
        self.box.drain(ScriptedClient(ok(), permanent()), now=0.0)
        self.assertEqual([i["session_id"] for i in self.box.list_items("sent")], ["sess-A"])
        self.assertEqual([i["session_id"] for i in self.box.list_items("failed_permanent")], ["sess-B"])
        self.assertEqual(len(self.box.list_items()), 2)


if __name__ == "__main__":
    unittest.main()
