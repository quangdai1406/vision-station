"""Scan records for "Taking data" mode: 360-degree pictures for training.

Standard library only. A scan is a separate record from an inspected part: it
never has a part verdict, never lights anything and never goes to the MES.
Layout under ``root``:

    scans/{line}/{YYYY}/{MM}/{DD}/{scan_id}/
        {site}_{line}_{YYYYMMDD}_{HHMMSS}_{scan_id}_{CAM}_{SIDE}_S{step:02d}.{ext}
        scan.json
        manifest.sha256   (sha256sum text format; lists every picture and scan.json, written last)

A scan is built in a hidden staging folder ``.{scan_id}.partial`` inside its day
folder and renamed to ``{scan_id}`` only when it is complete, so an interrupted
write never leaves a folder that looks finished.

Plan reference: docs/26_1027_PLAN_taking_data_mode.md (sections 3, 4, 5, 8).
Format reference: tools/FORMAT.md ("Scan record").
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import re
import shutil
import threading
from datetime import datetime, timezone
from typing import Optional

SCAN_JSON = "scan.json"
MANIFEST = "manifest.sha256"
SIDES = ("TOP", "BOTTOM")
VERDICTS = ("GOOD", "BAD")
LABEL_STATUSES = ("proposed", "official", "rejected")
MAX_NAME_CHARS = 64
MAX_REASON_CHARS = 200
_STAGING_SUFFIX = ".partial"
_META_FILES = frozenset({SCAN_JSON, MANIFEST})
_BAD_PATH_CHARS = frozenset("/\\\x00")
_UTC_FMT = "%Y-%m-%dT%H:%M:%SZ"
_LABEL_KEYS = (
    "verdict",
    "defects",
    "status",
    "labelled_by",
    "labelled_at",
    "approved_by",
    "approved_at",
    "reject_reason",
)
_SHA_RE = re.compile(r"[0-9a-f]{64}")


def _component(value: str, name: str) -> str:
    """Reject values that cannot safely be used as one path component."""
    if (
        not isinstance(value, str)
        or value in ("", ".", "..")
        or value.startswith(".")
        or any(c in _BAD_PATH_CHARS for c in value)
    ):
        raise ValueError(f"{name} is not usable in a file path: {value!r}")
    return value


def _parse_utc(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO 8601 UTC time string")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 UTC time string") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _text(value: object, name: str, max_chars: int = MAX_NAME_CHARS) -> str:
    """A short printable string. Returns it stripped."""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    if len(text) > max_chars:
        raise ValueError(f"{name} is longer than {max_chars} characters")
    if not text.isprintable():
        raise ValueError(f"{name} must not contain control characters")
    return text


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _try(fn, *args) -> list[str]:
    """Run a check; return [] when it passes, [message] when it raises."""
    try:
        fn(*args)
    except (ValueError, TypeError) as exc:
        return [str(exc)]
    return []


def _write_durable(path: pathlib.Path, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _fsync_dir(path: pathlib.Path) -> None:
    """Make a rename durable where the platform allows it (not on Windows)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump_json(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _label_doc(label: object) -> dict:
    """The scan.json form of a label dict: every known key present, unknown keys refused."""
    if not isinstance(label, dict):
        raise ValueError("label must be an object")
    extra = set(label) - set(_LABEL_KEYS)
    if extra:
        raise ValueError(f"unknown label keys: {sorted(extra)}")
    doc = {key: label.get(key) for key in _LABEL_KEYS}
    if doc["defects"] is None:
        doc["defects"] = []
    return doc


def _label_problems(label: object) -> list[str]:
    """Label rules, shared by begin_scan (refuses) and verify_scan (reports)."""
    if not isinstance(label, dict):
        return ["label must be an object"]
    problems: list[str] = []
    verdict = label.get("verdict")
    status = label.get("status")
    defects = label.get("defects")
    if verdict not in VERDICTS:
        problems.append("label verdict must be GOOD or BAD")
    if not isinstance(defects, list):
        problems.append("label defects must be a list")
    else:
        for name in defects:
            problems += _try(_text, name, "defect name")
        if len({str(d) for d in defects}) != len(defects):
            problems.append("label defects must not repeat a name")
        if verdict == "GOOD" and defects:
            problems.append("a GOOD label has no defects")
        if verdict == "BAD" and not defects:
            problems.append("a BAD label needs at least one defect")
    if status not in LABEL_STATUSES:
        problems.append("label status must be proposed, official or rejected")
    problems += _try(_text, label.get("labelled_by"), "labelled_by")
    problems += _try(_parse_utc, label.get("labelled_at"), "labelled_at")
    approved_by = label.get("approved_by")
    approved_at = label.get("approved_at")
    if (approved_by is None) != (approved_at is None):
        problems.append("approved_by and approved_at go together")
    if approved_by is not None:
        problems += _try(_text, approved_by, "approved_by")
    if approved_at is not None:
        problems += _try(_parse_utc, approved_at, "approved_at")
    reason = label.get("reject_reason")
    if status == "rejected":
        if approved_by is None:
            problems.append("a rejected label needs approved_by and approved_at")
        problems += _try(_text, reason, "reject_reason", MAX_REASON_CHARS)
    else:
        if reason is not None:
            problems.append("reject_reason is only for a rejected label")
        if status == "proposed" and approved_by is not None:
            problems.append("a proposed label has no approval")
    return problems


def _scan_problems(doc: object) -> list[str]:
    """Every rule a scan.json must follow. Empty list means valid."""
    if not isinstance(doc, dict):
        return ["scan.json is not an object"]
    problems: list[str] = []
    scan_id = doc.get("scan_id")
    problems += _try(_component, scan_id, "scan_id")
    site_problems = _try(_component, doc.get("site"), "site")
    line_problems = _try(_component, doc.get("line"), "line")
    problems += site_problems + line_problems
    naming_ok = not site_problems and not line_problems
    problems += _try(_text, doc.get("part_number"), "part_number")
    problems += _try(_text, doc.get("toy_id"), "toy_id")
    problems += _try(_text, doc.get("taken_by"), "taken_by")

    started_dt = finished_dt = None
    try:
        started_dt = _parse_utc(doc.get("started_utc"), "started_utc")
    except ValueError as exc:
        problems.append(str(exc))
    try:
        finished_dt = _parse_utc(doc.get("finished_utc"), "finished_utc")
    except ValueError as exc:
        problems.append(str(exc))
    if started_dt and finished_dt and finished_dt < started_dt:
        problems.append("finished_utc is before started_utc")

    label = doc.get("label")
    problems += _label_problems(label)
    verdict = label.get("verdict") if isinstance(label, dict) else None

    cams = doc.get("camera_ids")
    if not isinstance(cams, list) or not cams:
        problems.append("camera_ids must be a non-empty list")
        cams = []
    else:
        for cam in cams:
            problems += _try(_component, cam, "camera_id")
        if len({str(c) for c in cams}) != len(cams):
            problems.append("camera_ids must not repeat")

    steps = doc.get("steps")
    if not isinstance(steps, dict) or not steps:
        problems.append("steps must be an object with TOP and/or BOTTOM")
        steps = {}
    steps_ok: dict[str, int] = {}
    for side, count in steps.items():
        if side not in SIDES:
            problems.append(f"unknown side in steps: {side!r}")
        elif not _is_int(count) or count < 1:
            problems.append(f"steps {side} must be a whole number of at least 1")
        else:
            steps_ok[side] = count

    images = doc.get("images")
    if not isinstance(images, list):
        problems.append("images must be a list")
        images = []
    if not images:
        problems.append("a scan needs at least one picture")
    date = started_dt.strftime("%Y%m%d") if started_dt else None
    hms = started_dt.strftime("%H%M%S") if started_dt else None
    seen_files: set[str] = set()
    seen_keys: set[tuple] = set()
    by_side: dict[str, dict[int, list[str]]] = {}
    for i, img in enumerate(images):
        where = f"image {i}"
        if not isinstance(img, dict):
            problems.append(f"{where} is not an object")
            continue
        fname = img.get("file")
        problems += _try(_component, fname, f"{where} file")
        cam = img.get("camera_id")
        side = img.get("side")
        step = img.get("step")
        if not isinstance(cam, str) or cam not in cams:
            problems.append(f"{where} camera_id {cam!r} is not in camera_ids")
        if side not in SIDES:
            problems.append(f"{where} side must be TOP or BOTTOM")
        step_ok = _is_int(step) and step >= 1
        if not step_ok:
            problems.append(f"{where} step must be a whole number of at least 1")
        sha = img.get("sha256")
        if not isinstance(sha, str) or not _SHA_RE.fullmatch(sha):
            problems.append(f"{where} sha256 is not a sha256 hex digest")
        defect_visible = img.get("defect_visible")
        if verdict == "BAD" and not isinstance(defect_visible, bool):
            problems.append(f"{where} needs defect_visible true or false (BAD scan)")
        if verdict == "GOOD" and defect_visible is not None:
            problems.append(f"{where} defect_visible must be null (GOOD scan)")
        if naming_ok and step_ok and date and isinstance(fname, str) and isinstance(cam, str):
            stem, dot, ext = fname.rpartition(".")
            expected_stem = (
                f"{doc['site']}_{doc['line']}_{date}_{hms}_{scan_id}_{cam}_{side}_S{step:02d}"
            )
            if not (dot and ext.isalnum() and stem == expected_stem):
                problems.append(f"{where} file name does not match the scan")
        if isinstance(fname, str):
            if fname in seen_files:
                problems.append(f"{where} repeats file {fname}")
            seen_files.add(fname)
        if isinstance(cam, str) and step_ok and side in SIDES:
            key = (cam, side, step)
            if key in seen_keys:
                problems.append(f"{where} repeats camera {cam} {side} step {step}")
            seen_keys.add(key)
            by_side.setdefault(side, {}).setdefault(step, []).append(cam)

    for side in sorted(set(steps_ok) | set(by_side)):
        if side not in steps_ok:
            problems.append(f"{side} has pictures but no step count")
            continue
        if side not in by_side:
            problems.append(f"{side} has a step count but no pictures")
            continue
        count = steps_ok[side]
        found = by_side[side]
        if sorted(found) != list(range(1, count + 1)):
            problems.append(f"{side} steps must be 1 to {count} with no gaps")
        for step, found_cams in sorted(found.items()):
            if sorted(map(str, found_cams)) != sorted(map(str, cams)):
                problems.append(
                    f"{side} step {step} must have exactly one picture per camera"
                )
    return problems


class ScanWriter:
    """Builds one scan step by step. Create it with ScanStore.begin_scan().

    Each picture is written to the staging folder as soon as it is added, so
    only small metadata stays in memory. finish() publishes the scan;
    cancel() removes everything. Any error while writing closes the writer and
    removes its staging folder.
    """

    def __init__(
        self,
        store: "ScanStore",
        scan_id: str,
        doc_fields: dict,
        started_dt: datetime,
        day_dir: pathlib.Path,
        staging: pathlib.Path,
        final_dir: pathlib.Path,
    ) -> None:
        self._store = store
        self.scan_id = scan_id
        self.camera_ids = tuple(doc_fields["camera_ids"])
        self.verdict = doc_fields["label"]["verdict"]
        self._fields = doc_fields
        self._started_dt = started_dt
        self._date = started_dt.strftime("%Y%m%d")
        self._time = started_dt.strftime("%H%M%S")
        self.day_dir = day_dir
        self.staging = staging
        self.final_dir = final_dir
        self._images: list[dict] = []
        self._keys: set[tuple] = set()
        self._closed = False

    def add_picture(
        self,
        camera_id: str,
        side: str,
        step: int,
        data: bytes,
        ext: str = "png",
        defect_visible: Optional[bool] = None,
    ) -> str:
        """Write one picture to staging. Returns its file name."""
        if self._closed:
            raise RuntimeError("scan writer is closed")
        if camera_id not in self.camera_ids:
            raise ValueError(f"camera_id {camera_id!r} is not in this scan's camera_ids")
        if side not in SIDES:
            raise ValueError(f"side must be TOP or BOTTOM, got {side!r}")
        if not _is_int(step) or step < 1:
            raise ValueError(f"step must be a whole number of at least 1, got {step!r}")
        key = (camera_id, side, step)
        if key in self._keys:
            raise ValueError(f"{camera_id} {side} step {step} was already added")
        if not isinstance(ext, str) or not ext.lstrip(".").isalnum():
            raise ValueError(f"bad image extension: {ext!r}")
        ext = ext.lstrip(".")
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)
        if not isinstance(data, bytes) or not data:
            raise ValueError("picture data must be non-empty bytes")
        if self.verdict == "BAD":
            if not isinstance(defect_visible, bool):
                raise ValueError("a BAD scan needs defect_visible (True or False) for every picture")
        elif defect_visible is not None:
            raise ValueError("a GOOD scan takes no defect_visible")

        name = (
            f"{self._store.site}_{self._store.line}_{self._date}_{self._time}_"
            f"{self.scan_id}_{camera_id}_{side}_S{step:02d}.{ext}"
        )
        digest = hashlib.sha256(data).hexdigest()
        try:
            _write_durable(self.staging / name, data)
        except BaseException:
            self._discard()
            raise
        self._images.append(
            {
                "file": name,
                "camera_id": camera_id,
                "side": side,
                "step": step,
                "sha256": digest,
                "defect_visible": defect_visible,
            }
        )
        self._keys.add(key)
        return name

    def set_defect_visible(self, camera_id: str, side: str, step: int, value: bool) -> None:
        """Change one picture's defect_visible. Only on a BAD scan, and only before finish()."""
        with self._store._lock:
            if self._closed:
                raise RuntimeError("scan writer is closed")
            if self.verdict != "BAD":
                raise ValueError("defect_visible can only be set on a BAD scan")
            if not isinstance(value, bool):
                raise ValueError(f"defect_visible must be True or False, got {value!r}")
            for img in self._images:
                if img["camera_id"] == camera_id and img["side"] == side and img["step"] == step:
                    img["defect_visible"] = value
                    return
            raise ValueError(f"no picture {camera_id} {side} step {step} in this scan")

    def finish(self, finished_utc: str) -> pathlib.Path:
        """Validate, write scan.json, then the manifest (last), then publish the folder."""
        if self._closed:
            raise RuntimeError("scan writer is closed")
        try:
            finished_dt = _parse_utc(finished_utc, "finished_utc")
            if finished_dt < self._started_dt:
                raise ValueError("finished_utc is before started_utc")
            order = {side: i for i, side in enumerate(SIDES)}
            cam_order = {cam: i for i, cam in enumerate(self.camera_ids)}
            images = sorted(
                self._images,
                key=lambda i: (order[i["side"]], i["step"], cam_order[i["camera_id"]]),
            )
            steps: dict[str, int] = {}
            for img in images:
                steps[img["side"]] = max(steps.get(img["side"], 0), img["step"])
            doc = dict(self._fields)
            doc["steps"] = {side: steps[side] for side in SIDES if side in steps}
            doc["finished_utc"] = finished_dt.strftime(_UTC_FMT)
            doc["images"] = images
            problems = _scan_problems(doc)
            if problems:
                raise ValueError("; ".join(problems))
            scan_bytes = _dump_json(doc)
            _write_durable(self.staging / SCAN_JSON, scan_bytes)
            lines = [f"{img['sha256']}  {img['file']}\n" for img in images]
            lines.append(f"{hashlib.sha256(scan_bytes).hexdigest()}  {SCAN_JSON}\n")
            _write_durable(self.staging / MANIFEST, "".join(lines).encode("utf-8"))
            _fsync_dir(self.staging)
            with self._store._lock:
                if self.final_dir.exists():
                    raise FileExistsError(f"scan folder already exists: {self.final_dir}")
                os.rename(self.staging, self.final_dir)  # atomic publish of the whole scan
            _fsync_dir(self.day_dir)
        except BaseException:
            self._discard()
            raise
        self._closed = True
        return self.final_dir

    def cancel(self) -> None:
        """Abandon the scan and remove its staging folder. Safe to call twice."""
        if not self._closed:
            self._discard()

    def _discard(self) -> None:
        self._closed = True
        with self._store._lock:
            shutil.rmtree(self.staging, ignore_errors=True)
            self._prune_empty_days()

    def _prune_empty_days(self) -> None:
        """Remove day/month/year folders this scan left empty (never the line folder)."""
        stop = self._store.line_dir
        path = self.day_dir
        while path != stop and stop in path.parents:
            try:
                path.rmdir()
            except OSError:
                break
            path = path.parent


class ScanStore:
    """Writes scan records under ``root/scans/<line>/`` (see module docstring)."""

    def __init__(self, root: pathlib.Path, site: str = "SITE", line: str = "L1") -> None:
        self.root = pathlib.Path(root)
        self.site = _component(site, "site")
        self.line = _component(line, "line")
        self.scans_dir = self.root / "scans"
        self.line_dir = self.scans_dir / self.line
        self._lock = threading.RLock()
        self.line_dir.mkdir(parents=True, exist_ok=True)

    def begin_scan(
        self,
        scan_id: str,
        part_number: str,
        toy_id: str,
        label: dict,
        taken_by: str,
        camera_ids: list,
        started_utc: str,
    ) -> ScanWriter:
        """Open a new scan. Refuses a scan_id that already exists anywhere under this root."""
        scan_id = _component(scan_id, "scan_id")
        part_number = _text(part_number, "part_number")
        toy_id = _text(toy_id, "toy_id")
        taken_by = _text(taken_by, "taken_by")
        if not isinstance(camera_ids, (list, tuple)) or not camera_ids:
            raise ValueError("camera_ids must be a non-empty list")
        cams = [_component(c, "camera_id") for c in camera_ids]
        if len(set(cams)) != len(cams):
            raise ValueError("camera_ids must not repeat")
        started_dt = _parse_utc(started_utc, "started_utc")
        label_doc = _label_doc(label)
        problems = _label_problems(label_doc)
        if problems:
            raise ValueError("; ".join(problems))

        day_dir = (
            self.line_dir
            / started_dt.strftime("%Y")
            / started_dt.strftime("%m")
            / started_dt.strftime("%d")
        )
        fields = {
            "scan_id": scan_id,
            "site": self.site,
            "line": self.line,
            "part_number": part_number,
            "toy_id": toy_id,
            "label": label_doc,
            "taken_by": taken_by,
            "started_utc": started_dt.strftime(_UTC_FMT),
            "camera_ids": cams,
        }
        with self._lock:
            if self._scan_id_used(scan_id):
                raise FileExistsError(f"scan {scan_id!r} already exists")
            day_dir.mkdir(parents=True, exist_ok=True)
            staging = day_dir / f".{scan_id}{_STAGING_SUFFIX}"
            staging.mkdir()
            return ScanWriter(
                self, scan_id, fields, started_dt, day_dir, staging, day_dir / scan_id
            )

    def _scan_id_used(self, scan_id: str) -> bool:
        """True if a finished or staging scan with this id exists under any line."""
        staging_name = f".{scan_id}{_STAGING_SUFFIX}"
        for entry in self.scans_dir.glob("*/*/*/*/*"):
            if entry.name in (scan_id, staging_name):
                return True
        return False


def verify_scan(scan_dir: pathlib.Path) -> list[str]:
    """Return a list of problems; an empty list means the scan is intact and valid."""
    scan_dir = pathlib.Path(scan_dir)
    if scan_dir.is_symlink():
        return ["scan folder is a symlink"]
    if not scan_dir.is_dir():
        return [f"missing scan directory: {scan_dir}"]
    if scan_dir.name.startswith("."):
        return [f"partial staging folder, not a finished scan: {scan_dir.name}"]

    problems: list[str] = []
    for entry in sorted(scan_dir.iterdir()):
        if entry.is_symlink():
            problems.append(f"symlink: {entry.name}")
        elif entry.name.startswith("."):
            problems.append(f"partial staging entry: {entry.name}")

    manifest_path = scan_dir / MANIFEST
    scan_path = scan_dir / SCAN_JSON
    if not manifest_path.is_file() or manifest_path.is_symlink():
        if not manifest_path.is_symlink():
            problems.append("missing manifest.sha256")
        return problems
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return problems + ["unreadable manifest.sha256"]

    expected: dict[str, str] = {}
    for lineno, line in enumerate(manifest_text.splitlines(), 1):
        if not line.strip():
            continue
        digest, sep, name = line.partition("  ")
        if (
            not sep
            or not _SHA_RE.fullmatch(digest)
            or not name
            or name in (".", "..")
            or "/" in name
            or "\\" in name
        ):
            problems.append(f"malformed manifest line {lineno}")
            continue
        expected[name] = digest

    for name, digest in expected.items():
        path = scan_dir / name
        if path.is_symlink():
            continue  # already reported above
        if not path.is_file():
            problems.append(f"missing file: {name}")
        elif _sha256_file(path) != digest:
            problems.append(f"hash mismatch: {name}")

    for entry in sorted(scan_dir.iterdir()):
        if entry.is_symlink() or entry.name.startswith("."):
            continue  # already reported above
        if entry.name not in expected and entry.name not in _META_FILES:
            problems.append(f"not in manifest: {entry.name}")

    if SCAN_JSON not in expected:
        problems.append("scan.json not in manifest")
    if not scan_path.is_file() or scan_path.is_symlink():
        if not scan_path.is_symlink():
            problems.append("missing scan.json")
        return problems
    try:
        doc = json.loads(scan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return problems + ["scan.json is unreadable or not JSON"]

    problems += [f"scan.json invalid: {p}" for p in _scan_problems(doc)]
    if not isinstance(doc, dict):
        return problems
    if doc.get("scan_id") != scan_dir.name:
        problems.append("scan_id does not match the folder name")
    try:
        started = _parse_utc(doc.get("started_utc"), "started_utc")
        expected_path = (
            doc.get("line"),
            started.strftime("%Y"),
            started.strftime("%m"),
            started.strftime("%d"),
        )
        if tuple(scan_dir.resolve().parts[-5:-1]) != expected_path:
            problems.append(
                "scan folder path does not match scan.json "
                f"(line {doc.get('line')!r}, started_utc date {'/'.join(expected_path[1:])})"
            )
    except ValueError:
        pass

    images = doc.get("images") if isinstance(doc.get("images"), list) else []
    listed: set[str] = set()
    for img in images:
        if not isinstance(img, dict) or not isinstance(img.get("file"), str):
            continue
        name = img["file"]
        listed.add(name)
        if name not in expected:
            problems.append(f"image not in manifest: {name}")
        elif expected[name] != img.get("sha256"):
            problems.append(f"scan.json disagrees with manifest: {name}")
    for name in sorted(expected):
        if name != SCAN_JSON and name not in listed:
            problems.append(f"manifest lists a file scan.json does not: {name}")
    return problems


def load_scan(scan_dir: pathlib.Path) -> dict:
    """Read and validate scan.json. Raises ValueError if it is invalid.

    This checks scan.json only; use verify_scan() to check the pictures and hashes.
    """
    path = pathlib.Path(scan_dir) / SCAN_JSON
    if path.is_symlink():
        raise ValueError("scan.json is a symlink")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("scan.json is not UTF-8 text") from exc
    problems = _scan_problems(doc)
    if problems:
        raise ValueError("invalid scan.json: " + "; ".join(problems))
    return doc


def list_scans(root: pathlib.Path, line: Optional[str] = None) -> list[pathlib.Path]:
    """Finished scan folders under ``root/scans`` (one line, or all). Staging folders are skipped."""
    base = pathlib.Path(root) / "scans"
    if not base.is_dir():
        return []
    if line is not None:
        lines = [_component(line, "line")]
    else:
        lines = sorted(p.name for p in base.iterdir() if p.is_dir() and not p.is_symlink())
    found: list[pathlib.Path] = []
    for name in lines:
        line_dir = base / name
        if not line_dir.is_dir() or line_dir.is_symlink():
            continue
        for scan_dir in line_dir.glob("*/*/*/*"):
            if scan_dir.name.startswith(".") or scan_dir.is_symlink() or not scan_dir.is_dir():
                continue
            found.append(scan_dir)
    return sorted(found)
