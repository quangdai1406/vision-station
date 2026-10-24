"""Storage for finished inspection parts and the append-only event log.

Standard library only. Layout under ``root``:

    station.db
    store/{line}/{YYYY}/{MM}/{DD}/{part_id}/
        {site}_{line}_{YYYYMMDD}_{HHMMSS}_{part_id}_{camera_id}_{side}.{ext}
        part.json
        manifest.sha256   (sha256sum text format, images only)

Plan reference: 26_1023_inspection_system_plan.md (storage section).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from typing import Iterator

from station.contracts import (
    IMAGES_PER_PART,
    PartRecord,
    Side,
    Verdict,
)

PART_JSON = "part.json"
MANIFEST = "manifest.sha256"
_META_FILES = frozenset({PART_JSON, MANIFEST})
_TMP_PREFIX = ".tmp-"
_BAD_PATH_CHARS = frozenset("/\\\x00")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    part_id TEXT NOT NULL,
    final_verdict TEXT NOT NULL,
    machine_verdict TEXT NOT NULL,
    started_utc TEXT NOT NULL,
    path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    file TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    side TEXT NOT NULL,
    seq INTEGER NOT NULL,
    sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    kind TEXT NOT NULL,
    session_id TEXT,
    payload_json TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only: UPDATE not allowed');
END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only: DELETE not allowed');
END;
"""


def _component(value: str, name: str) -> str:
    """Reject values that cannot safely be used as one path component."""
    if (
        not isinstance(value, str)
        or value in ("", ".", "..")
        or value.startswith(_TMP_PREFIX.rstrip("-"))
        or any(c in _BAD_PATH_CHARS for c in value)
    ):
        raise ValueError(f"{name} is not usable in a file path: {value!r}")
    return value


def _parse_utc(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _write_durable(path: pathlib.Path, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump_json(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _insert_rows(conn: sqlite3.Connection, doc: dict, path: str) -> None:
    cur = conn.execute(
        "INSERT OR IGNORE INTO parts "
        "(session_id, part_id, final_verdict, machine_verdict, started_utc, path) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            doc["session_id"],
            doc["part_id"],
            doc["final_verdict"],
            doc["machine_verdict"],
            doc["started_utc"],
            path,
        ),
    )
    if cur.rowcount == 0:  # session already recorded; keep the original rows
        return
    conn.executemany(
        "INSERT INTO images (session_id, file, camera_id, side, seq, sha256) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                doc["session_id"],
                img["file"],
                img["camera_id"],
                img["side"],
                img["seq"],
                img["sha256"],
            )
            for img in doc["images"]
        ],
    )


class Store:
    def __init__(self, root: pathlib.Path, site: str = "SITE", line: str = "L1") -> None:
        self.root = pathlib.Path(root)
        self.site = _component(site, "site")
        self.line = _component(line, "line")
        self.db_path = self.root / "station.db"
        self.store_dir = self.root / "store"
        self._lock = threading.RLock()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        with self._db() as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ db
    @contextlib.contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        # One connection per call: nothing is shared between threads.
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    # --------------------------------------------------------------- parts
    def save_part(self, record: PartRecord) -> pathlib.Path:
        frames = list(record.frames)
        results = list(record.results)
        if len(frames) != IMAGES_PER_PART or len(results) != IMAGES_PER_PART:
            raise ValueError(
                f"a part needs exactly {IMAGES_PER_PART} frames and "
                f"{IMAGES_PER_PART} results, got {len(frames)} frames "
                f"and {len(results)} results"
            )
        keys = [(_component(f.camera_id, "camera_id"), Side(f.side).value) for f in frames]
        if len(set(keys)) != IMAGES_PER_PART:
            raise ValueError("each camera must appear exactly once per side")
        by_key = {(r.camera_id, Side(r.side).value): r for r in results}
        if set(by_key) != set(keys) or len(by_key) != IMAGES_PER_PART:
            raise ValueError("results must match frames one-to-one")
        if Verdict(record.final_verdict) is Verdict.REVIEW:
            raise ValueError("final_verdict must be GOOD or BAD (REVIEW is resolved first)")
        if not record.session_id:
            raise ValueError("session_id is required")
        part_id = _component(record.part_id, "part_id")
        started = _parse_utc(record.started_utc)

        with self._lock:
            known = self._known_path(record.session_id)
            if known is not None:
                return known  # idempotent: never rewrite recorded evidence
            part_dir = (
                self.store_dir
                / self.line
                / started.strftime("%Y")
                / started.strftime("%m")
                / started.strftime("%d")
                / part_id
            )
            if part_dir.exists():
                return self._adopt(part_dir, record)
            return self._write_new(part_dir, record, frames, by_key, started)

    def _known_path(self, session_id: str) -> pathlib.Path | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT path FROM parts WHERE session_id = ?", (session_id,)
            ).fetchone()
        return pathlib.Path(row[0]) if row else None

    def _adopt(self, part_dir: pathlib.Path, record: PartRecord) -> pathlib.Path:
        """A finished directory exists but has no DB row (crash after rename)."""
        try:
            doc = json.loads((part_dir / PART_JSON).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FileExistsError(f"{part_dir} exists but has no readable part.json") from exc
        if doc.get("session_id") != record.session_id:
            raise FileExistsError(
                f"{part_dir} already belongs to session {doc.get('session_id')!r}"
            )
        with self._db() as conn:
            _insert_rows(conn, doc, str(part_dir))
        return part_dir

    def _write_new(
        self,
        part_dir: pathlib.Path,
        record: PartRecord,
        frames: list,
        by_key: dict,
        started: datetime,
    ) -> pathlib.Path:
        date_stamp = started.strftime("%Y%m%d")
        time_stamp = started.strftime("%H%M%S")
        part_id = record.part_id

        staged: list[tuple[str, bytes]] = []  # (file name, bytes) in capture order
        images: list[dict] = []
        for frame in frames:
            side = Side(frame.side).value
            result = by_key[(frame.camera_id, side)]
            ext = frame.ext.lstrip(".")
            if not ext.isalnum():
                raise ValueError(f"bad image extension: {frame.ext!r}")
            name = (
                f"{self.site}_{self.line}_{date_stamp}_{time_stamp}_{part_id}_"
                f"{_component(frame.camera_id, 'camera_id')}_{side}.{ext}"
            )
            digest = hashlib.sha256(frame.data).hexdigest()
            staged.append((name, frame.data))
            images.append(
                {
                    "file": name,
                    "camera_id": frame.camera_id,
                    "side": side,
                    "seq": frame.seq,
                    "timestamp_utc": frame.timestamp_utc,
                    "sha256": digest,
                    "score": result.score,
                    "defects": [
                        {
                            "defect_class": d.defect_class,
                            "score": d.score,
                            "box": list(d.box),
                        }
                        for d in result.defects
                    ],
                }
            )

        doc = {
            "part_id": part_id,
            "session_id": record.session_id,
            "station_id": record.station_id,
            "part_number": record.part_number,
            "lot": record.lot,
            "operator": record.operator,
            "supervisor": record.supervisor,
            "material_handler": record.material_handler,
            "started_utc": record.started_utc,
            "model_version": record.model_version,
            "thresholds": {
                "t_low": record.thresholds.t_low,
                "t_img": record.thresholds.t_img,
            },
            "machine_verdict": Verdict(record.machine_verdict).value,
            "final_verdict": Verdict(record.final_verdict).value,
            "override": None
            if record.override is None
            else {
                "original_verdict": Verdict(record.override.original_verdict).value,
                "new_verdict": Verdict(record.override.new_verdict).value,
                "reason": record.override.reason,
                "by": record.override.by,
            },
            "disposition": record.disposition,
            "images": images,
        }
        manifest = "".join(
            f"{img['sha256']}  {img['file']}\n" for img in images
        ).encode("utf-8")

        part_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp = pathlib.Path(
            tempfile.mkdtemp(prefix=f"{_TMP_PREFIX}{part_id}-", dir=str(part_dir.parent))
        )
        try:
            for name, data in staged:
                _write_durable(tmp / name, data)
            _write_durable(tmp / PART_JSON, _dump_json(doc))
            _write_durable(tmp / MANIFEST, manifest)
            os.chmod(tmp, 0o755)
            os.rename(tmp, part_dir)  # atomic publish of the whole part
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        with self._db() as conn:
            _insert_rows(conn, doc, str(part_dir))
        return part_dir

    def verify_part(self, part_dir: pathlib.Path) -> list[str]:
        """Return a list of problems; an empty list means the part is intact."""
        part_dir = pathlib.Path(part_dir)
        if not part_dir.is_dir():
            return [f"missing part directory: {part_dir}"]
        problems: list[str] = []
        part_json = part_dir / PART_JSON
        manifest_path = part_dir / MANIFEST
        if not part_json.is_file():
            problems.append("missing part.json")
        if not manifest_path.is_file():
            problems.append("missing manifest.sha256")
            return problems

        expected: dict[str, str] = {}
        for lineno, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            digest, sep, name = line.partition("  ")
            if (
                not sep
                or len(digest) != 64
                or not name
                or name in (".", "..")
                or "/" in name
                or "\\" in name
            ):
                problems.append(f"malformed manifest line {lineno}")
                continue
            expected[name] = digest.lower()

        for name, digest in expected.items():
            path = part_dir / name
            if not path.is_file():
                problems.append(f"missing file: {name}")
            elif _sha256_file(path) != digest:
                problems.append(f"hash mismatch: {name}")

        for entry in sorted(part_dir.iterdir()):
            if entry.name in _META_FILES or entry.name in expected:
                continue
            problems.append(f"not in manifest: {entry.name}")

        if part_json.is_file():
            try:
                doc = json.loads(part_json.read_text(encoding="utf-8"))
                for img in doc.get("images", []):
                    if expected.get(img.get("file")) != img.get("sha256"):
                        problems.append(f"part.json disagrees with manifest: {img.get('file')}")
            except (ValueError, AttributeError, TypeError):
                problems.append("unreadable part.json")
        return problems

    def archive_ref(self, part_dir: pathlib.Path) -> str:
        return str(part_dir)

    # -------------------------------------------------------------- events
    def append_event(self, kind: str, payload: dict, session_id: str | None = None) -> int:
        if not kind:
            raise ValueError("kind is required")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        with self._db() as conn:
            cur = conn.execute(
                "INSERT INTO events (ts_utc, kind, session_id, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (ts, kind, session_id, body),
            )
            new_id = cur.lastrowid
        return int(new_id)

    # ------------------------------------------------------------- queries
    def counters(self) -> dict:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT final_verdict, COUNT(*) FROM parts GROUP BY final_verdict"
            ).fetchall()
        by_verdict = {verdict: count for verdict, count in rows}
        good = by_verdict.get(Verdict.GOOD.value, 0)
        bad = by_verdict.get(Verdict.BAD.value, 0)
        return {"total": sum(by_verdict.values()), "good": good, "bad": bad}

    def recent_parts(self, limit: int = 20) -> list[dict]:
        with self._db() as conn:
            rows = conn.execute(
                "SELECT session_id, part_id, final_verdict, started_utc, path "
                "FROM parts ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {
                "session_id": r[0],
                "part_id": r[1],
                "final_verdict": r[2],
                "started_utc": r[3],
                "path": r[4],
            }
            for r in rows
        ]
