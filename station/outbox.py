"""Durable outbox for inspection events waiting to be posted to the MES.

Layout under root:
  root/outbox.db                      SQLite table of items (one row per session)
  root/outbox/<session_id>/<n>.bin    raw bytes of each photo file

Standard library only.
"""
from __future__ import annotations

import json
import pathlib
import re
import sqlite3
import time
from typing import Any, Optional, Sequence

from station.mes_client import MesClient, PostResult

STATUS_PENDING = "pending"
STATUS_SENT = "sent"
STATUS_FAILED_PERMANENT = "failed_permanent"

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL UNIQUE,
    fields_json   TEXT NOT NULL,
    files_json    TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('pending', 'sent', 'failed_permanent')),
    attempts      INTEGER NOT NULL DEFAULT 0,
    next_attempt  REAL NOT NULL DEFAULT 0,
    last_error    TEXT,
    mes_event_id  INTEGER,
    created_at    REAL NOT NULL,
    sent_at       REAL
)
"""


class Outbox:
    def __init__(self, root: pathlib.Path):
        self.root = pathlib.Path(root)
        (self.root / "outbox").mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.root / "outbox.db"))
        self._db.row_factory = sqlite3.Row
        with self._db:
            self._db.execute(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def enqueue(
        self,
        session_id: str,
        fields: dict[str, str],
        files: Sequence[tuple[str, str, bytes]],
    ) -> bool:
        """Store one post. Returns False (and changes nothing) if session_id is already queued."""
        if not _SESSION_ID_RE.match(session_id):
            raise ValueError(f"unsafe session_id: {session_id!r}")
        if self._db.execute(
            "SELECT 1 FROM items WHERE session_id = ?", (session_id,)
        ).fetchone():
            return False

        file_rows = []
        directory = self.root / "outbox" / session_id
        directory.mkdir(parents=True, exist_ok=True)
        for index, (field_name, filename, data) in enumerate(files):
            rel = f"outbox/{session_id}/{index}.bin"
            (self.root / rel).write_bytes(bytes(data))
            file_rows.append({"field": field_name, "filename": filename, "path": rel})

        with self._db:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO items "
                "(session_id, fields_json, files_json, status, attempts, next_attempt, created_at) "
                "VALUES (?, ?, ?, ?, 0, 0, ?)",
                (
                    session_id,
                    json.dumps(dict(fields), sort_keys=True),
                    json.dumps(file_rows),
                    STATUS_PENDING,
                    time.time(),
                ),
            )
        return cur.rowcount == 1

    def drain(
        self,
        client: MesClient,
        now: Optional[float] = None,
        base_delay_s: float = 5.0,
        max_delay_s: float = 600.0,
        max_items: int = 50,
    ) -> dict[str, int]:
        """Post due items oldest first. Stops at the first retryable failure."""
        now = time.time() if now is None else float(now)
        counts = {"sent": 0, "failed_permanent": 0, "retry_later": 0}
        rows = self._db.execute(
            "SELECT * FROM items WHERE status = ? AND next_attempt <= ? "
            "ORDER BY seq LIMIT ?",
            (STATUS_PENDING, now, max_items),
        ).fetchall()

        for row in rows:
            try:
                fields = json.loads(row["fields_json"])
                files = self._load_files(row["files_json"])
            except (OSError, ValueError, KeyError) as e:
                self._finish(row, STATUS_FAILED_PERMANENT, now, error=f"stored item unreadable: {e}")
                counts["failed_permanent"] += 1
                continue

            try:
                result: PostResult = client.post(fields, files, idempotency_key=row["session_id"])
            except Exception as e:  # never lose the queue to a client bug; retry later
                result = PostResult(
                    status=0, ok=False, replay=False, event_id=None,
                    error=f"unexpected client error: {type(e).__name__}", retryable=True,
                )

            if result.ok:
                self._finish(row, STATUS_SENT, now, event_id=result.event_id)
                counts["sent"] += 1
            elif not result.retryable:
                self._finish(row, STATUS_FAILED_PERMANENT, now, error=result.error)
                counts["failed_permanent"] += 1
            else:
                attempts = row["attempts"] + 1
                exponent = min(attempts, 60)  # keep the power bounded
                delay = min(base_delay_s * (2 ** exponent), max_delay_s)
                with self._db:
                    self._db.execute(
                        "UPDATE items SET attempts = ?, last_error = ?, next_attempt = ? "
                        "WHERE seq = ?",
                        (attempts, result.error, now + delay, row["seq"]),
                    )
                counts["retry_later"] += 1
                break  # keep order: MES is probably down, do not post later items

        return counts

    def pending_count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM items WHERE status = ?", (STATUS_PENDING,)
        ).fetchone()
        return int(row["n"])

    def failed_permanent_items(self) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT session_id, last_error, attempts FROM items "
            "WHERE status = ? ORDER BY seq",
            (STATUS_FAILED_PERMANENT,),
        ).fetchall()
        return [
            {"session_id": r["session_id"], "last_error": r["last_error"], "attempts": r["attempts"]}
            for r in rows
        ]

    def oldest_pending_age_s(self, now: Optional[float] = None) -> Optional[float]:
        now = time.time() if now is None else float(now)
        row = self._db.execute(
            "SELECT MIN(created_at) AS oldest FROM items WHERE status = ?",
            (STATUS_PENDING,),
        ).fetchone()
        if row["oldest"] is None:
            return None
        return now - float(row["oldest"])

    def list_items(self, status: Optional[str] = None) -> list[dict[str, Any]]:
        if status is None:
            rows = self._db.execute("SELECT * FROM items ORDER BY seq").fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM items WHERE status = ? ORDER BY seq", (status,)
            ).fetchall()
        keys = ("session_id", "status", "attempts", "next_attempt", "last_error",
                "mes_event_id", "created_at", "sent_at")
        return [{k: r[k] for k in keys} for r in rows]

    # internals

    def _load_files(self, files_json: str) -> list[tuple[str, str, bytes]]:
        out = []
        for entry in json.loads(files_json):
            data = (self.root / entry["path"]).read_bytes()
            out.append((entry["field"], entry["filename"], data))
        return out

    def _finish(
        self,
        row: sqlite3.Row,
        status: str,
        now: float,
        error: Optional[str] = None,
        event_id: Optional[int] = None,
    ) -> None:
        with self._db:
            self._db.execute(
                "UPDATE items SET status = ?, attempts = attempts + 1, last_error = ?, "
                "mes_event_id = ?, sent_at = ? WHERE seq = ?",
                (
                    status,
                    None if status == STATUS_SENT else error,
                    event_id,
                    now if status == STATUS_SENT else None,
                    row["seq"],
                ),
            )
