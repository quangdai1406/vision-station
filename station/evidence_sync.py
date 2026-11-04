"""Copy saved part folders from the Pi to the laptop evidence store.

Standard library only. ``sync_part`` copies one part folder (8 pictures, part.json,
manifest.sha256) written by station/storage.py, keeping its layout under the
destination ``store`` folder: ``<dest>/<line>/YYYY/MM/DD/<part_id>/``. The local copy is
never deleted. Every file is hashed against manifest.sha256 before copying and again
inside a temporary folder after copying; only then is the folder renamed into place.

``SyncQueue`` keeps the parts still to copy in ``<local>/evidence_sync_queue.jsonl``
(one JSON object per line, rewritten atomically). An item is marked done only after
``sync_part`` returns; failed items stay pending for the next run.

CLI (run from the project root)::

    python3 -m station.evidence_sync run --local <station root> --dest <laptop store folder>
    python3 -m station.evidence_sync status --local <station root>

``--local`` is the station data folder that holds ``store/`` and the queue file.
``--dest`` is the laptop folder that holds the ``L1/`` tree (the trainer's ``store/``).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

PART_JSON = "part.json"
MANIFEST = "manifest.sha256"
PURGED_FILE = "purged.json"  # tombstone written by trainer/purge.py
STORE_DIR = "store"
QUEUE_NAME = "evidence_sync_queue.jsonl"
PENDING = "pending"
DONE = "done"
_TMP_PREFIX = ".tmp-"
_META_FILES = frozenset({PART_JSON, MANIFEST})
_HEX64 = re.compile(r"[0-9a-f]{64}")
_YEAR = re.compile(r"\d{4}")
_MONTH_DAY = re.compile(r"\d{2}")


class SyncError(RuntimeError):
    """A part could not be copied and verified. The message never contains a secret."""


@dataclass
class _Source:
    manifest: bytes
    entries: list  # [(file name, sha256)] for the pictures
    part_json_sha: str


@dataclass
class SyncReport:
    synced: int = 0
    failed: int = 0
    pending: int = 0
    errors: list = field(default_factory=list)  # "<part folder>: <reason>"


# ------------------------------------------------------------------ helpers

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain_name(name: str) -> bool:
    return (
        name not in ("", ".", "..")
        and "/" not in name
        and "\\" not in name
        and "\x00" not in name
        and not name.startswith(_TMP_PREFIX)
    )


def _layout(part_dir: pathlib.Path) -> pathlib.Path:
    """Relative path ``<line>/YYYY/MM/DD/<part_id>`` below the store folder. Raises SyncError."""
    parts = part_dir.parts
    if len(parts) < 6 or parts[-6] != STORE_DIR:
        raise SyncError(f"not a part folder in the store layout: {part_dir.name}")
    line, year, month, day, part_id = parts[-5:]
    if not (_plain_name(line) and _YEAR.fullmatch(year) and _MONTH_DAY.fullmatch(month)
            and _MONTH_DAY.fullmatch(day) and _plain_name(part_id)):
        raise SyncError(f"not a part folder in the store layout: {part_id}")
    return pathlib.Path(line, year, month, day, part_id)


def _require_regular(path: pathlib.Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise SyncError(f"missing or not a regular file: {path.name}")


def _parse_manifest(data: bytes) -> list:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SyncError("manifest is not UTF-8") from exc
    entries: list = []
    seen: set = set()
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        digest, sep, name = line.partition("  ")
        if (not sep or not _HEX64.fullmatch(digest) or not _plain_name(name)
                or name in _META_FILES or name in seen):
            raise SyncError(f"malformed manifest line {lineno}")
        seen.add(name)
        entries.append((name, digest))
    if not entries:
        raise SyncError("manifest lists no files")
    return entries


def _read_source(part_dir: pathlib.Path) -> _Source:
    """Check the local part against its own manifest. Nothing is copied or changed."""
    if not part_dir.is_dir() or part_dir.is_symlink():
        raise SyncError(f"part folder not found: {part_dir.name}")
    manifest_path = part_dir / MANIFEST
    _require_regular(manifest_path)
    manifest = manifest_path.read_bytes()
    entries = _parse_manifest(manifest)
    listed = {name for name, _ in entries}
    for name, digest in entries:
        path = part_dir / name
        _require_regular(path)
        if _sha256_file(path) != digest:
            raise SyncError(f"local hash mismatch: {name}")
    _require_regular(part_dir / PART_JSON)
    for entry in part_dir.iterdir():
        if entry.name not in listed and entry.name not in _META_FILES:
            raise SyncError(f"local folder has a file not in manifest: {entry.name}")
    return _Source(manifest, entries, _sha256_file(part_dir / PART_JSON))


def _copy_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy bytes and flush them to disk. The hash is checked by the caller."""
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            fout.write(chunk)
        fout.flush()
        os.fsync(fout.fileno())


def _write_bytes(path: pathlib.Path, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _verify_folder(folder: pathlib.Path, src: _Source) -> None:
    """Every listed picture and part.json must hash as recorded; the manifest must match."""
    for name, digest in src.entries:
        path = folder / name
        _require_regular(path)
        if _sha256_file(path) != digest:
            raise SyncError(f"hash mismatch after copy: {name}")
    _require_regular(folder / PART_JSON)
    if _sha256_file(folder / PART_JSON) != src.part_json_sha:
        raise SyncError("hash mismatch after copy: part.json")
    _require_regular(folder / MANIFEST)
    if (folder / MANIFEST).read_bytes() != src.manifest:
        raise SyncError("manifest differs after copy")


def _adopt_existing(final: pathlib.Path, src: _Source) -> str:
    """The destination folder already exists: accept it only if it is the same part."""
    if final.is_symlink() or not final.is_dir():
        raise SyncError(f"destination exists and is not a folder: {final.name}")
    if (final / PURGED_FILE).is_file():
        # Pictures were deleted on purpose after a verified training run: never recreate them.
        return str(final)
    manifest_path = final / MANIFEST
    if not manifest_path.is_file() or manifest_path.read_bytes() != src.manifest:
        raise SyncError(f"destination holds a different part (not overwritten): {final.name}")
    _verify_folder(final, src)
    return str(final)


# ----------------------------------------------------------------- copying

def sync_part(part_dir, dest_root) -> str:
    """Copy one saved part to ``dest_root`` (the laptop store folder); return the final path.

    Idempotent: a part already there with the same manifest is left alone. Raises
    SyncError when the destination is unreachable, the local part or the copy fails
    verification, or the destination holds a different part. No final folder is left
    behind after a failed copy.
    """
    part_dir = pathlib.Path(part_dir)
    dest_root = pathlib.Path(dest_root)
    rel = _layout(part_dir)
    try:
        src = _read_source(part_dir)
    except OSError as exc:
        raise SyncError(f"local read failed: {part_dir.name}: {exc.strerror or type(exc).__name__}") from exc
    if not dest_root.is_dir():
        # Do not create the folder: a missing SMB mount must not turn into a local folder.
        raise SyncError("destination is not reachable")
    final = dest_root / rel
    if os.path.lexists(final):
        return _adopt_existing(final, src)

    tmp: Optional[pathlib.Path] = None
    try:
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"{_TMP_PREFIX}{rel.name}-", dir=str(final.parent)))
        for name, _digest in src.entries:
            _copy_file(part_dir / name, tmp / name)
        _copy_file(part_dir / PART_JSON, tmp / PART_JSON)
        _write_bytes(tmp / MANIFEST, src.manifest)
        _verify_folder(tmp, src)
        os.rename(tmp, final)
    except SyncError:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    except OSError as exc:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        raise SyncError(f"copy failed: {part_dir.name}: {exc.strerror or type(exc).__name__}") from exc
    return str(final)


# ------------------------------------------------------------------- queue

def _write_atomic(path: pathlib.Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f"{_TMP_PREFIX}queue-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


class SyncQueue:
    """Parts waiting to be copied to the laptop, kept in a JSON-lines file."""

    def __init__(self, folder) -> None:
        self.folder = pathlib.Path(folder)
        self.path = self.folder / QUEUE_NAME

    def _load(self) -> list:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise SyncError(f"queue unreadable: {exc.strerror or type(exc).__name__}") from exc
        items = []
        for lineno, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except ValueError as exc:
                raise SyncError(f"queue line {lineno} is malformed") from exc
            if not isinstance(item, dict) or "part_dir" not in item or "state" not in item:
                raise SyncError(f"queue line {lineno} is malformed")
            items.append(item)
        return items

    def _save(self, items: list) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        data = "".join(json.dumps(item, sort_keys=True) + "\n" for item in items)
        _write_atomic(self.path, data.encode("utf-8"))

    def enqueue(self, part_dir) -> None:
        key = str(pathlib.Path(part_dir))
        _layout(pathlib.Path(part_dir))  # refuse anything that is not a store part
        items = self._load()
        if any(item["part_dir"] == key for item in items):
            return
        items.append({
            "part_dir": key,
            "state": PENDING,
            "enqueued_utc": _now(),
            "attempts": 0,
            "last_error": None,
            "archive_ref": None,
        })
        self._save(items)

    def pending(self) -> list:
        return [item for item in self._load() if item["state"] == PENDING]

    def done(self) -> list:
        return [item for item in self._load() if item["state"] == DONE]

    def run_once(self, dest_root) -> SyncReport:
        items = self._load()
        report = SyncReport()
        for item in items:
            if item["state"] != PENDING:
                continue
            try:
                ref = sync_part(item["part_dir"], dest_root)
            except SyncError as exc:
                item["attempts"] = int(item.get("attempts", 0)) + 1
                item["last_error"] = str(exc)
                report.failed += 1
                report.errors.append(f"{pathlib.Path(item['part_dir']).name}: {exc}")
            else:
                item.update(state=DONE, archive_ref=ref, last_error=None, done_utc=_now())
                report.synced += 1
            self._save(items)  # each item is recorded as soon as it is settled
        report.pending = sum(1 for item in items if item["state"] == PENDING)
        return report


# --------------------------------------------------------------------- CLI

def _status(queue: SyncQueue) -> int:
    items = queue._load()
    pending = [i for i in items if i["state"] == PENDING]
    done = [i for i in items if i["state"] == DONE]
    print(f"pending {len(pending)}, done {len(done)}")
    for item in pending:
        print(f"pending: {item['part_dir']} attempts={item.get('attempts', 0)} "
              f"last_error={item.get('last_error')}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m station.evidence_sync")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="copy every pending part to the laptop once")
    run_p.add_argument("--local", required=True, help="station data folder (holds store/ and the queue)")
    run_p.add_argument("--dest", required=True, help="laptop store folder (the mounted share)")
    status_p = sub.add_parser("status", help="show pending and done counts")
    status_p.add_argument("--local", required=True)
    args = parser.parse_args(argv)

    queue = SyncQueue(args.local)
    try:
        if args.command == "status":
            return _status(queue)
        report = queue.run_once(args.dest)
    except SyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"synced {report.synced}, failed {report.failed}, pending {report.pending}")
    for line in report.errors:
        print(f"failed: {line}")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
