"""Export saved evidence from the Pi station to a destination folder.

Standard library only. Read-only at the source. Nothing is ever deleted at the
source or at the destination. Plain file copy: the destination can be an SMB
mount, a USB drive or a local staging folder. This tool has no network code.

Source: the station evidence root (the folder that holds store/ and/or scans/, or store/ or
scans/ itself).
Layout read: store/{line}/{YYYY}/{MM}/{DD}/{part_id}/ (station/storage.py).
Scans (Taking data mode, station/scanstore.py) are read from scans/{line}/{YYYY}/{MM}/{DD}/{scan_id}/
and exported to DEST/scans/... with the same rules (see "Scans" below).

Each part folder is checked against its manifest.sha256 (sha256sum text format)
before anything is copied. A part is:
  ok          manifest lists 8 images, all present, hashes match, part.json present and
              agrees with the manifest, no unlisted files, no symlinks
  incomplete  part.json, manifest or one of the 8 images is missing
  corrupt     hash mismatch, malformed manifest, unlisted file, symlink, bad part.json
  purged      purged.json tombstone present (pictures deleted by trainer/purge.py)
Only ok parts are copied. Incomplete, corrupt and purged parts are skipped and reported.

Destination: parts are written to DEST/store/{line}/{YYYY}/{MM}/{DD}/{part_id}/, the same
layout, so trainer/dataset.py can read the destination directly. A part already at the
destination whose manifest.sha256 is byte-identical to the source is skipped. A destination
manifest that differs is never overwritten (reported as failed). The manifest is written
last, so an interrupted copy is completed on the next run.

Scans: each scan folder is checked with station.scanstore.verify_scan, the same function the
station's tests use, so the export cannot drift from the rule the station writes by (the
module is standard library only). A scan is:
  ok          verify_scan reports no problem (scan.json and every listed picture present,
              hashes match, no unlisted files, no symlinks)
  incomplete  manifest, scan.json or a listed picture is missing (verify_scan "missing ..."),
              or a hidden .{scan_id}.partial staging folder (never copied, reported)
  corrupt     any other verify_scan problem, or a symlink
  purged      purged.json present (same tombstone name as parts; no scan purge exists today)
Only ok scans are copied to DEST/scans/{line}/{YYYY}/{MM}/{DD}/{scan_id}/. The copy writes
every file the manifest lists (scan.json included) to a temporary name, checks its hash, renames
it into place, and writes manifest.sha256 last. A destination file whose hash differs is not
overwritten (failed). Scans are counted in report["scans"]; the parts counts are unchanged.
The SharePoint transfer is postponed (plan D-C); the destination can be any folder.

Report: DEST/export_report.json (counts and problem list; no secrets, no picture data).
Parts counts stay in "counts" and "problems"; scans are in "scans" ({"counts", "problems"}).

CLI (run from the repo root):
    python3 -m tools.export_evidence --source EVIDENCE_ROOT --dest DEST_FOLDER \
        [--since YYYY-MM-DD] [--dry-run]

Exit codes: 0 no corrupt or failed part or scan, 1 some part or scan is corrupt or failed,
2 bad arguments or unusable source/destination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Sequence

from station.scanstore import verify_scan

PART_JSON = "part.json"
MANIFEST = "manifest.sha256"
PURGED_FILE = "purged.json"  # tombstone written by trainer/purge.py
STORE_DIR = "store"
SCANS_DIR = "scans"
SCAN_JSON = "scan.json"
REPORT_FILE = "export_report.json"
IMAGES_PER_PART = 8  # same value as station/contracts.py IMAGES_PER_PART
_TMP_PREFIX = ".tmp-"
_META_FILES = frozenset({PART_JSON, MANIFEST})
_HEX64 = re.compile(r"[0-9a-f]{64}")
_YEAR = re.compile(r"\d{4}")
_MONTH_DAY = re.compile(r"\d{2}")
_COUNT_KEYS = (
    "found",
    "copied",
    "already_present",
    "incomplete",
    "corrupt",
    "purged",
    "failed",
    "skipped",
    "out_of_range",
    "temp_ignored",
)


class ExportError(RuntimeError):
    """The source or destination cannot be used at all."""


@dataclass
class PartCheck:
    status: str  # ok | incomplete | corrupt | purged
    problems: list = field(default_factory=list)
    manifest_bytes: Optional[bytes] = None
    entries: list = field(default_factory=list)  # [(file name, sha256)]


# ------------------------------------------------------------ verification

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_manifest(data: bytes) -> tuple[list, list]:
    """Parse sha256sum text. Returns (entries, corrupt problems)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [], ["manifest is not UTF-8"]
    entries: list = []
    problems: list = []
    seen: set = set()
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        digest, sep, name = line.partition("  ")
        if (
            not sep
            or not _HEX64.fullmatch(digest)
            or not name
            or name in (".", "..")
            or "/" in name
            or "\\" in name
        ):
            problems.append(f"malformed manifest line {lineno}")
            continue
        if name in _META_FILES:
            problems.append(f"manifest lists {name}")
            continue
        if name in seen:
            problems.append(f"manifest lists {name} twice")
            continue
        seen.add(name)
        entries.append((name, digest))
    return entries, problems


def check_part(part_dir: Path) -> PartCheck:
    """Verify one part folder without changing it."""
    if (part_dir / PURGED_FILE).exists():
        return PartCheck("purged", [f"{PURGED_FILE} present (pictures deleted)"])

    corrupt: list = []
    incomplete: list = []
    part_json = part_dir / PART_JSON
    manifest = part_dir / MANIFEST

    for path in (part_json, manifest):
        if path.is_symlink():
            corrupt.append(f"symlink: {path.name}")
        elif not path.is_file():
            incomplete.append(f"missing {path.name}")

    manifest_bytes: Optional[bytes] = None
    entries: list = []
    if manifest.is_file() and not manifest.is_symlink():
        manifest_bytes = manifest.read_bytes()
        entries, probs = _parse_manifest(manifest_bytes)
        corrupt.extend(probs)
        if len(entries) < IMAGES_PER_PART:
            incomplete.append(f"manifest lists {len(entries)} images, need {IMAGES_PER_PART}")
        elif len(entries) > IMAGES_PER_PART:
            corrupt.append(f"manifest lists {len(entries)} images, expected {IMAGES_PER_PART}")

    listed = {name for name, _ in entries}
    for name, digest in entries:
        path = part_dir / name
        if path.is_symlink():
            corrupt.append(f"symlink: {name}")
        elif not path.is_file():
            incomplete.append(f"missing image: {name}")
        elif _sha256_file(path) != digest:
            corrupt.append(f"hash mismatch: {name}")

    # Without a readable manifest the part is only incomplete; the unlisted-file and
    # part.json checks need the manifest to compare against.
    if manifest_bytes is not None:
        for entry in sorted(part_dir.iterdir()):
            if entry.name in _META_FILES or entry.name in listed:
                continue
            corrupt.append(f"not in manifest: {entry.name}")

    if manifest_bytes is not None and part_json.is_file() and not part_json.is_symlink():
        try:
            doc = json.loads(part_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = None
        images = doc.get("images") if isinstance(doc, dict) else None
        if not isinstance(images, list):
            corrupt.append("part.json has no images list")
        else:
            try:
                recorded = sorted((img["file"], img["sha256"]) for img in images)
            except (KeyError, TypeError):
                recorded = None
            if recorded is None or recorded != sorted(entries):
                corrupt.append("part.json disagrees with manifest")

    if corrupt:
        return PartCheck("corrupt", corrupt + incomplete, manifest_bytes, entries)
    if incomplete:
        return PartCheck("incomplete", incomplete, manifest_bytes, entries)
    return PartCheck("ok", [], manifest_bytes, entries)


# ------------------------------------------------------------------ copying

def _copy_verified(src: Path, dst: Path, expected: str) -> None:
    """Copy one file and re-hash the copy. Raises OSError on any mismatch."""
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            fout.write(chunk)
        fout.flush()
        os.fsync(fout.fileno())
    if _sha256_file(dst) != expected:
        raise OSError(f"copied file does not match its hash: {dst.name}")


def _write_manifest(dst_dir: Path, data: bytes) -> None:
    tmp = dst_dir / ".manifest.sha256.part"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dst_dir / MANIFEST)


def _export_part(check: PartCheck, src: Path, dst: Path, dry_run: bool) -> tuple[str, str]:
    """Return (outcome, reason). Outcome: copied | already_present | failed."""
    dst_manifest = dst / MANIFEST
    if dst_manifest.is_file():
        if dst_manifest.read_bytes() != check.manifest_bytes:
            return "failed", "destination manifest differs from source (not overwritten)"
        if (dst / PART_JSON).is_file():
            return "already_present", ""
    if dry_run:
        return "copied", ""
    try:
        dst.mkdir(parents=True, exist_ok=True)
        for name, digest in check.entries:
            _copy_verified(src / name, dst / name, digest)
        _copy_verified(src / PART_JSON, dst / PART_JSON, _sha256_file(src / PART_JSON))
        _write_manifest(dst, check.manifest_bytes)
    except OSError as exc:
        return "failed", f"copy error: {exc}"
    return "copied", ""


# ------------------------------------------------------------- scan records

def check_scan(scan_dir: Path) -> PartCheck:
    """Verify one scan folder with station.scanstore.verify_scan. Nothing is changed."""
    if (scan_dir / PURGED_FILE).exists():
        return PartCheck("purged", [f"{PURGED_FILE} present (pictures deleted)"])
    problems = verify_scan(scan_dir)
    if problems:
        # verify_scan words a missing manifest, scan.json or picture as "missing ...".
        if any(not p.startswith("missing") for p in problems):
            return PartCheck("corrupt", problems)
        return PartCheck("incomplete", problems)
    manifest_bytes = (scan_dir / MANIFEST).read_bytes()
    entries, _ = _parse_manifest(manifest_bytes)  # verify_scan already checked the format
    return PartCheck("ok", [], manifest_bytes, entries)


def _copy_atomic(src: Path, dst: Path, expected: str) -> None:
    """Copy one file under a temporary name, re-hash it, then rename it into place."""
    tmp = dst.parent / f".{dst.name}.part"
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            fout.write(chunk)
        fout.flush()
        os.fsync(fout.fileno())
    if _sha256_file(tmp) != expected:
        raise OSError(f"copied file does not match its hash: {dst.name}")
    os.replace(tmp, dst)


def _export_scan(check: PartCheck, src: Path, dst: Path, dry_run: bool) -> tuple[str, str]:
    """Return (outcome, reason). Outcome: copied | already_present | failed.

    Copies every file the manifest lists (scan.json included), then writes the manifest last.
    A destination file that already exists with a different hash is never overwritten.
    """
    dst_manifest = dst / MANIFEST
    manifest_present = os.path.lexists(dst_manifest)
    if manifest_present and (
        dst_manifest.is_symlink()
        or not dst_manifest.is_file()
        or dst_manifest.read_bytes() != check.manifest_bytes
    ):
        return "failed", "destination manifest differs from source (not overwritten)"
    to_copy: list = []
    for name, digest in check.entries:
        target = dst / name
        if target.is_symlink():
            return "failed", f"destination is a symlink (not overwritten): {name}"
        if target.is_file():
            if _sha256_file(target) != digest:
                return "failed", f"destination file differs from source (not overwritten): {name}"
        else:
            to_copy.append((name, digest))
    if not to_copy and manifest_present:
        return "already_present", ""
    if dry_run:
        return "copied", ""
    try:
        dst.mkdir(parents=True, exist_ok=True)
        for name, digest in to_copy:
            _copy_atomic(src / name, dst / name, digest)
        if not manifest_present:
            _write_manifest(dst, check.manifest_bytes)
    except OSError as exc:
        return "failed", f"copy error: {exc}"
    return "copied", ""


# ------------------------------------------------------------------- walking

def _resolve_sources(source) -> tuple[Optional[Path], Optional[Path]]:
    """Return (store folder, scans folder) to read. Either is None when absent."""
    root = Path(source)
    if not root.is_dir():
        raise ExportError(f"source folder not found: {root}")
    if (root / STORE_DIR).is_dir():
        scans = root / SCANS_DIR
        return root / STORE_DIR, (scans if scans.is_dir() else None)
    if root.name == SCANS_DIR:
        return None, root
    if (root / SCANS_DIR).is_dir():
        # An evidence root with scans but no parts: never read scans/ as if it were store/.
        return None, root / SCANS_DIR
    return root, None


def _subdirs(path: Path, counts: dict) -> list:
    out = []
    for entry in sorted(path.iterdir()):
        if not entry.is_dir() or entry.is_symlink():
            continue
        if entry.name.startswith(_TMP_PREFIX):
            counts["temp_ignored"] += 1
            continue
        out.append(entry)
    return out


def iter_part_dirs(store: Path, counts: dict) -> Iterator[tuple[str, date, Path]]:
    """Yield (line, day, part_dir) for store/{line}/{YYYY}/{MM}/{DD}/{part_id}/."""
    for line_dir in _subdirs(store, counts):
        for year_dir in _subdirs(line_dir, counts):
            if not _YEAR.fullmatch(year_dir.name):
                continue
            for month_dir in _subdirs(year_dir, counts):
                if not _MONTH_DAY.fullmatch(month_dir.name):
                    continue
                for day_dir in _subdirs(month_dir, counts):
                    if not _MONTH_DAY.fullmatch(day_dir.name):
                        continue
                    try:
                        day = date(int(year_dir.name), int(month_dir.name), int(day_dir.name))
                    except ValueError:
                        continue
                    for part_dir in _subdirs(day_dir, counts):
                        yield line_dir.name, day, part_dir


def _plain_dirs(path: Path) -> list:
    """Sorted sub-folders, symlinks left out. Hidden folders are kept (the caller decides)."""
    return [e for e in sorted(path.iterdir()) if e.is_dir() and not e.is_symlink()]


def iter_scan_dirs(scans: Path) -> Iterator[tuple[str, date, Path, str]]:
    """Yield (line, day, entry, kind) for scans/{line}/{YYYY}/{MM}/{DD}/{entry}.

    kind: "scan" (a finished scan folder), "staging" (a hidden .{scan_id}.partial folder),
    "symlink" (a symlink in a day folder: never followed).
    """
    for line_dir in _plain_dirs(scans):
        for year_dir in _plain_dirs(line_dir):
            if not _YEAR.fullmatch(year_dir.name):
                continue
            for month_dir in _plain_dirs(year_dir):
                if not _MONTH_DAY.fullmatch(month_dir.name):
                    continue
                for day_dir in _plain_dirs(month_dir):
                    if not _MONTH_DAY.fullmatch(day_dir.name):
                        continue
                    try:
                        day = date(int(year_dir.name), int(month_dir.name), int(day_dir.name))
                    except ValueError:
                        continue
                    for entry in sorted(day_dir.iterdir()):
                        if entry.is_symlink():
                            kind = "symlink"
                        elif not entry.is_dir():
                            continue
                        elif entry.name.startswith("."):
                            kind = "staging"
                        else:
                            kind = "scan"
                        yield line_dir.name, day, entry, kind


def _check_separate(store: Path, dest: Path) -> None:
    s = store.resolve()
    d = dest.resolve()
    if d == s or d.is_relative_to(s) or s.is_relative_to(d):
        raise ExportError("source and destination must not be inside each other")


def _write_report(dest: Path, report: dict) -> Path:
    path = dest / REPORT_FILE
    tmp = dest / ".export_report.json.part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------- export

def export(source, dest, since: Optional[date] = None, dry_run: bool = False) -> dict:
    """Copy every verified part and scan from source to dest. Returns the report dict."""
    store, scans = _resolve_sources(source)
    dest = Path(dest)
    for root in (store, scans):
        if root is not None:
            _check_separate(root, dest)
    if dest.exists() and not dest.is_dir():
        raise ExportError(f"destination is not a folder: {dest}")
    if not dest.exists() and not dry_run:
        if not dest.parent.is_dir():
            raise ExportError(
                f"destination parent does not exist (is the share or drive mounted?): {dest.parent}"
            )
        dest.mkdir()

    counts = {key: 0 for key in _COUNT_KEYS}
    problems: list = []
    parts_iter = iter_part_dirs(store, counts) if store is not None else iter(())
    for line, day, part_dir in parts_iter:
        if since is not None and day < since:
            counts["out_of_range"] += 1
            continue
        counts["found"] += 1
        rel = f"{line}/{day:%Y}/{day:%m}/{day:%d}/{part_dir.name}"
        check = check_part(part_dir)
        if check.status == "ok":
            dst = dest / STORE_DIR / line / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}" / part_dir.name
            outcome, reason = _export_part(check, part_dir, dst, dry_run)
            counts[outcome] += 1
            if outcome == "failed":
                problems.append({"part": rel, "status": "failed", "reasons": [reason]})
            continue
        counts[check.status] += 1
        problems.append({"part": rel, "status": check.status, "reasons": check.problems})

    counts["skipped"] = (
        counts["already_present"] + counts["incomplete"] + counts["corrupt"] + counts["purged"]
    )

    scounts = {key: 0 for key in _COUNT_KEYS}
    sproblems: list = []
    scan_iter = iter_scan_dirs(scans) if scans is not None else iter(())
    for line, day, entry, kind in scan_iter:
        if since is not None and day < since:
            scounts["out_of_range"] += 1
            continue
        scounts["found"] += 1
        rel = f"{line}/{day:%Y}/{day:%m}/{day:%d}/{entry.name}"
        if kind == "staging":
            check = PartCheck("incomplete", ["partial staging folder (interrupted scan), never copied"])
        elif kind == "symlink":
            check = PartCheck("corrupt", ["symlink, not followed"])
        else:
            check = check_scan(entry)
        if check.status == "ok":
            dst = dest / SCANS_DIR / line / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}" / entry.name
            outcome, reason = _export_scan(check, entry, dst, dry_run)
            scounts[outcome] += 1
            if outcome == "failed":
                sproblems.append({"scan": rel, "status": "failed", "reasons": [reason]})
            continue
        scounts[check.status] += 1
        sproblems.append({"scan": rel, "status": check.status, "reasons": check.problems})
    scounts["skipped"] = (
        scounts["already_present"] + scounts["incomplete"] + scounts["corrupt"] + scounts["purged"]
    )

    report = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": bool(dry_run),
        "source": str(store) if store is not None else None,
        "scans_source": str(scans) if scans is not None else None,
        "dest": str(dest),
        "since": since.isoformat() if since else None,
        "counts": counts,
        "problems": problems,
        "scans": {"counts": scounts, "problems": sproblems},
    }
    if not dry_run:
        _write_report(dest, report)
    return report


# ---------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", required=True,
                   help="Pi evidence root (the folder that holds store/ and/or scans/)")
    p.add_argument("--dest", required=True,
                   help="destination folder: SMB mount, USB drive or local staging folder")
    p.add_argument("--since", metavar="YYYY-MM-DD", default=None,
                   help="only export parts and scans whose date folder is on or after this day")
    p.add_argument("--dry-run", action="store_true",
                   help="verify and count only; write nothing at the destination")
    return p


def _format_summary(report: dict) -> str:
    c = report["counts"]
    prefix = "DRY RUN (nothing written): " if report["dry_run"] else ""
    lines = [
        f"{prefix}found {c['found']} parts; "
        f"copied {c['copied']}, skipped {c['skipped']} "
        f"(already present {c['already_present']}, incomplete {c['incomplete']}, "
        f"corrupt {c['corrupt']}, purged {c['purged']}), failed {c['failed']}",
    ]
    if c["out_of_range"]:
        lines.append(f"before --since: {c['out_of_range']} parts not considered")
    for item in report["problems"]:
        lines.append(f"  [{item['status']}] {item['part']}: {'; '.join(item['reasons'])}")
    if report.get("scans_source") is not None:
        sc = report["scans"]["counts"]
        lines.append(
            f"{prefix}scans: found {sc['found']}; "
            f"copied {sc['copied']}, skipped {sc['skipped']} "
            f"(already present {sc['already_present']}, incomplete {sc['incomplete']}, "
            f"corrupt {sc['corrupt']}, purged {sc['purged']}), failed {sc['failed']}"
        )
        if sc["out_of_range"]:
            lines.append(f"before --since: {sc['out_of_range']} scans not considered")
        for item in report["scans"]["problems"]:
            lines.append(f"  [{item['status']}] scan {item['scan']}: {'; '.join(item['reasons'])}")
    if not report["dry_run"]:
        lines.append(f"report: {Path(report['dest']) / REPORT_FILE}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    since = None
    if args.since is not None:
        try:
            since = date.fromisoformat(args.since)
        except ValueError:
            print(f"error: --since must be YYYY-MM-DD, got {args.since!r}", file=sys.stderr)
            return 2
    try:
        report = export(args.source, args.dest, since=since, dry_run=args.dry_run)
    except (ExportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(_format_summary(report))
    c = report["counts"]
    sc = report["scans"]["counts"]
    return 1 if (c["corrupt"] or c["failed"] or sc["corrupt"] or sc["failed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
