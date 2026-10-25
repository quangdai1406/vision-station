"""Delete the pictures of GOOD parts that a verified training run already used.

``python3 -m trainer purge --store PATH --models PATH --part PN --weekly-report FILE [--confirm]``

Dry run is the default: it prints what would be deleted and deletes nothing.
Only ``--confirm`` deletes. Standard library only.

Weekly report (JSON, written by trainer/weekly.py). Read defensively; this is the
key set this module relies on::

    {
      "parts": [
        {
          "part_number": "PN-1",                 # required, str
          "candidate_version": "2026-10-19-a",   # required, str: ModelStore version trained
          "training_succeeded": true,            # required, bool
          "trained_session_ids": ["S-A1", ...],  # required, list of str: session_id of each
                                                 #   part used as a GOOD training input
          "holdout_session_ids": ["S-A9", ...]   # optional, list of str
        }
      ],
      "holdout_session_ids": ["S-A9", ...]       # optional, top level, list of str
    }

A session is protected if it appears in ANY holdout list (top level or per entry).
Entries for other part numbers are ignored, except for their holdout lists. A report
with no entry for --part, a malformed entry for --part, or a succeeded entry whose
candidate is missing from the ModelStore or fails verification REFUSES the whole run
(exit 1, nothing deleted).

Eligible pictures (all must hold, else the part is kept):
  - the part is classified GOOD under the strict policy (trainer.dataset.classify):
    final GOOD, machine GOOD, override null;
  - it is not a partial station, and its verdicts are not REVIEW (supervisor review);
  - its session_id is in trained_session_ids of a succeeded entry for the same part
    number whose candidate verifies, and it is not a holdout session;
  - every picture listed in part.json is a plain file name listed in
    manifest.sha256 with the same sha256, is a regular file (not a symlink), and
    still hashes to that sha256.

A part with one changed, symlinked or unlisted picture is kept as a whole (a
half-purged part would be left behind). BAD and overridden parts are never touched.
part.json, manifest.sha256 and any file not listed as a picture are never deleted.

Per purged part, purged.json is written atomically BEFORE any picture is deleted::

    {"schema": 1, "tool": "trainer.purge/1", "utc": "...Z", "part_number", "part_id",
     "session_id", "candidate_version", "report": "<report file name>",
     "files": [{"name", "sha256", "size"}, ...]}

part.json is left untouched. Re-running is idempotent: a part whose pictures are all
gone is counted as already purged; a part interrupted half-way has its remaining
listed pictures deleted and the tombstone's file list merged.
trainer/dataset.py skips tombstoned parts with a note.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from station.modelstore import ModelStore
from trainer.dataset import (
    MANIFEST,
    PART_JSON,
    PURGED_FILE,
    PartSample,
    _image_ref,
    _part_json_paths,
    _plain_name,
    _read_part_doc,
    _resolve_store,
    _sha256_file,
    classify,
)

TOOL_VERSION = "trainer.purge/1"
SCHEMA = 1
DRY_RUN_REMINDER = (
    "DRY RUN: nothing was deleted. Re-run with --confirm to delete the pictures listed above."
)
WARNINGS_SHOWN = 20
_META_NAMES = frozenset({PART_JSON, MANIFEST, PURGED_FILE})


class PurgeRefusal(Exception):
    """The whole run must stop; nothing may be deleted."""


class _PartSkip(Exception):
    """This one part must be kept; the reason is reported."""


# ----------------------------------------------------------------- data
@dataclass(frozen=True)
class PurgeFile:
    name: str
    path: pathlib.Path
    sha256: str
    size: int


@dataclass(frozen=True)
class PurgePart:
    part_number: str
    part_id: str
    session_id: str
    part_dir: pathlib.Path
    candidate_version: str
    files: tuple  # tuple of PurgeFile, pictures still present and verified
    previous_tombstone: Optional[dict]  # purged.json already there (interrupted run)

    @property
    def bytes(self) -> int:
        return sum(f.size for f in self.files)


@dataclass(frozen=True)
class Kept:
    part_dir: pathlib.Path
    reason: str


@dataclass
class PurgePlan:
    part_number: str
    report_name: str
    refusal: Optional[str] = None
    parts: list = field(default_factory=list)
    kept: list = field(default_factory=list)
    already_purged: int = 0
    warnings: list = field(default_factory=list)

    @property
    def pictures(self) -> int:
        return sum(len(p.files) for p in self.parts)

    @property
    def bytes(self) -> int:
        return sum(p.bytes for p in self.parts)


@dataclass
class PurgeResult:
    parts_purged: int = 0
    files_deleted: int = 0
    bytes_freed: int = 0
    errors: list = field(default_factory=list)


# ----------------------------------------------------------------- helpers
def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _inside(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return False
    return True


def _str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) and x for x in value)


def _load_report(path) -> Any:
    report_path = pathlib.Path(path)
    if not report_path.is_file():
        raise PurgeRefusal(f"weekly report not found: {report_path.name}")
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise PurgeRefusal(f"weekly report unreadable: {type(exc).__name__}") from exc


def _parse_report(report: Any, part_number: str) -> tuple[list, set]:
    """Return (entries for part_number, every protected holdout session id)."""
    if not isinstance(report, dict):
        raise PurgeRefusal("weekly report is not a JSON object")
    top_holdout = report.get("holdout_session_ids", [])
    if not _str_list(top_holdout):
        raise PurgeRefusal("weekly report: holdout_session_ids must be a list of strings")
    holdout = set(top_holdout)
    parts = report.get("parts")
    if not isinstance(parts, list):
        raise PurgeRefusal("weekly report has no 'parts' list")
    mine: list = []
    for entry in parts:
        if not isinstance(entry, dict) or not isinstance(entry.get("part_number"), str):
            raise PurgeRefusal("weekly report: a 'parts' entry has no part_number")
        entry_holdout = entry.get("holdout_session_ids", [])
        if not _str_list(entry_holdout):
            raise PurgeRefusal(f"weekly report: holdout_session_ids of {entry['part_number']} is malformed")
        holdout.update(entry_holdout)
        if entry["part_number"] == part_number:
            mine.append(entry)
    if not mine:
        raise PurgeRefusal(f"weekly report has no training entry for part {part_number}")
    return mine, holdout


def _verified_sessions(model_store: ModelStore, part_number: str, entries: list,
                       plan: PurgePlan) -> dict:
    """session_id -> candidate_version, for succeeded runs whose candidate verifies."""
    sessions: dict = {}
    for entry in entries:
        version = entry.get("candidate_version")
        trained = entry.get("trained_session_ids")
        succeeded = entry.get("training_succeeded")
        if not isinstance(version, str) or not version:
            raise PurgeRefusal(f"weekly report: candidate_version missing for part {part_number}")
        if not isinstance(succeeded, bool):
            raise PurgeRefusal(f"weekly report: training_succeeded missing for part {part_number}")
        if not _str_list(trained):
            raise PurgeRefusal(f"weekly report: trained_session_ids malformed for part {part_number}")
        if not succeeded:
            plan.warnings.append(
                f"training did not succeed for candidate {version}: nothing it used is purged"
            )
            continue
        problems = model_store.verify(part_number, version)
        if problems:
            raise PurgeRefusal(
                f"candidate {version} for part {part_number} does not verify: " + "; ".join(problems)
            )
        for session_id in trained:
            sessions.setdefault(session_id, version)
    return sessions


def _manifest_map(part_dir: pathlib.Path) -> dict:
    path = part_dir / MANIFEST
    if not path.is_file():
        raise _PartSkip("kept: missing manifest.sha256")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        raise _PartSkip("kept: unreadable manifest.sha256")
    out: dict = {}
    for line in lines:
        if not line.strip():
            continue
        digest, sep, name = line.partition("  ")
        if not sep or len(digest) != 64 or not _plain_name(name):
            raise _PartSkip("kept: malformed manifest.sha256")
        out[name] = digest.lower()
    return out


def _label_sample(doc: dict, part_dir: pathlib.Path) -> PartSample:
    """A PartSample without images: enough for trainer.dataset.classify."""
    return PartSample(
        part_number=doc["part_number"], part_id=doc["part_id"], session_id=doc["session_id"],
        started_utc=doc["started_utc"], machine_verdict=doc["machine_verdict"],
        final_verdict=doc["final_verdict"], override=doc["override"],
        defect_label=doc.get("defect_label"), model_version=doc["model_version"],
        partial_station=doc["partial_station"], camera_count=doc["camera_count"],
        images=(), part_dir=part_dir,
    )


def _read_tombstone(part_dir: pathlib.Path) -> dict:
    path = part_dir / PURGED_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        raise _PartSkip("kept: purged.json is unreadable")
    if not isinstance(data, dict) or not isinstance(data.get("files"), list):
        raise _PartSkip("kept: purged.json is malformed")
    return data


def _pictures(part_dir: pathlib.Path, store_dir: pathlib.Path, doc: dict,
              tombstone: Optional[dict]) -> list:
    """Verified pictures still present. Raises _PartSkip for anything that keeps the part."""
    refs = []
    seen: set = set()
    for entry in doc["images"]:
        ref, reason = _image_ref(part_dir, entry)
        if ref is None:
            raise _PartSkip(f"kept: {reason}")
        name = ref.path.name
        if name in _META_NAMES:
            raise _PartSkip(f"kept: refuses to treat {name} as a picture")
        if name in seen:
            raise _PartSkip(f"kept: picture listed twice: {name}")
        seen.add(name)
        if ref.path.is_symlink():
            raise _PartSkip(f"kept: picture is a symlink: {name}")
        if not _inside(ref.path, store_dir):
            raise _PartSkip(f"kept: picture path leaves the store: {name}")
        refs.append(ref)

    manifest = _manifest_map(part_dir)
    for ref in refs:
        if manifest.get(ref.path.name) != ref.sha256.lower():
            raise _PartSkip(f"kept: part.json disagrees with manifest: {ref.path.name}")

    files = []
    for ref in refs:
        path, name = ref.path, ref.path.name
        if path.is_symlink():
            raise _PartSkip(f"kept: picture is a symlink: {name}")
        if not path.exists():
            if tombstone is not None:
                continue  # deleted by an earlier, interrupted purge
            raise _PartSkip(f"kept: picture missing: {name}")
        if not path.is_file():
            raise _PartSkip(f"kept: not a regular file: {name}")
        digest = _sha256_file(path)
        if digest != manifest[name]:
            raise _PartSkip(f"kept: picture changed since it was saved: {name}")
        files.append(PurgeFile(name=name, path=path, sha256=digest, size=path.stat().st_size))
    return files


def _consider(plan: PurgePlan, part_dir: pathlib.Path, store_dir: pathlib.Path, doc: dict,
              sessions: dict, holdout: set) -> None:
    session_id = doc["session_id"]

    def keep(reason: str, warn: bool = False) -> None:
        plan.kept.append(Kept(part_dir=part_dir, reason=reason))
        if warn:
            plan.warnings.append(f"{part_dir}: {reason}")

    if session_id in holdout:
        return keep("holdout evaluation set")
    if doc["partial_station"]:
        return keep("partial station (never a training input)")
    if doc["override"] is not None:
        return keep("overridden by a person")
    if doc["machine_verdict"] == "REVIEW" or doc["final_verdict"] == "REVIEW":
        return keep("supervisor review")
    if doc["final_verdict"] == "BAD":
        return keep("BAD part")
    if classify(_label_sample(doc, part_dir), "strict") != "good":
        return keep(f"not GOOD under strict policy (machine {doc['machine_verdict']}, "
                    f"final {doc['final_verdict']})")
    if session_id not in sessions:
        return keep("not in a successful training run's trained_session_ids")

    tomb_path = part_dir / PURGED_FILE
    if tomb_path.is_symlink():
        return keep("kept: purged.json is a symlink", warn=True)
    try:
        tombstone = _read_tombstone(part_dir) if tomb_path.exists() else None
        files = _pictures(part_dir, store_dir, doc, tombstone)
    except _PartSkip as exc:
        return keep(str(exc), warn=True)
    if not files:
        plan.already_purged += 1
        return
    plan.parts.append(PurgePart(
        part_number=doc["part_number"], part_id=doc["part_id"], session_id=session_id,
        part_dir=part_dir, candidate_version=sessions[session_id], files=tuple(files),
        previous_tombstone=tombstone,
    ))


def plan_purge(store, models, *, part_number: str, report_path) -> PurgePlan:
    """Work out what could be deleted. Reads the store and models; deletes nothing.

    store: the folder that holds store/ (or store/ itself). models: the ModelStore root
    (or a ModelStore). A refusal is returned as plan.refusal, never raised.
    """
    plan = PurgePlan(part_number=str(part_number), report_name=pathlib.Path(report_path).name)
    try:
        if not isinstance(part_number, str) or not part_number.strip():
            raise PurgeRefusal("a part number is required")
        report = _load_report(report_path)
        entries, holdout = _parse_report(report, part_number)
        store_dir = _resolve_store(store)
        model_store = models if isinstance(models, ModelStore) else ModelStore(models)
        sessions = _verified_sessions(model_store, part_number, entries, plan)
        for part_json in _part_json_paths(store_dir):
            part_dir = part_json.parent
            if part_json.is_symlink() or part_dir.is_symlink() or not _inside(part_dir, store_dir):
                plan.warnings.append(f"{part_dir}: symlink or outside the store, not touched")
                continue
            doc, reason = _read_part_doc(part_json)
            if doc is None:
                plan.warnings.append(f"{part_dir}: {reason}")
                continue
            if doc["part_number"] != part_number:
                continue
            _consider(plan, part_dir, store_dir, doc, sessions, holdout)
    except PurgeRefusal as exc:
        plan.refusal = str(exc)
        plan.parts = []
    except (OSError, ValueError) as exc:
        plan.refusal = f"could not read the store or models: {exc}"
        plan.parts = []
    return plan


def _write_json_atomic(path: pathlib.Path, data: dict) -> None:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".purged.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _recheck(part: PurgePart) -> None:
    for f in part.files:
        if f.path.is_symlink() or not f.path.is_file():
            raise _PartSkip(f"picture changed since the plan: {f.name}")
        if _sha256_file(f.path) != f.sha256:
            raise _PartSkip(f"picture changed since the plan: {f.name}")


def _tombstone(part: PurgePart, report_name: str, now: str) -> dict:
    merged: dict = {}
    previous = part.previous_tombstone or {}
    for entry in previous.get("files", []):
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            merged[entry["name"]] = entry
    for f in part.files:
        merged[f.name] = {"name": f.name, "sha256": f.sha256, "size": f.size}
    utc = previous.get("utc") if isinstance(previous.get("utc"), str) else now
    return {
        "schema": SCHEMA,
        "tool": TOOL_VERSION,
        "utc": utc,
        "part_number": part.part_number,
        "part_id": part.part_id,
        "session_id": part.session_id,
        "candidate_version": part.candidate_version,
        "report": report_name,
        "files": [merged[name] for name in sorted(merged)],
    }


def execute_purge(plan: PurgePlan, *, now_utc: Optional[str] = None) -> PurgeResult:
    """Write each part's tombstone, then delete its pictures. Refuses a refused plan."""
    if plan.refusal is not None:
        raise PurgeRefusal(plan.refusal)
    now = now_utc or _utc_now()
    result = PurgeResult()
    for part in plan.parts:
        try:
            _recheck(part)
            _write_json_atomic(part.part_dir / PURGED_FILE, _tombstone(part, plan.report_name, now))
        except (_PartSkip, OSError) as exc:
            result.errors.append(f"{part.part_dir}: {exc}; nothing deleted from this part")
            continue
        for f in part.files:
            try:
                os.unlink(f.path)
            except OSError as exc:
                result.errors.append(f"{f.path}: could not delete: {exc.strerror or type(exc).__name__}")
                continue
            result.files_deleted += 1
            result.bytes_freed += f.size
        result.parts_purged += 1
    return result


# ----------------------------------------------------------------- CLI
def add_arguments(sub: Any) -> None:
    p = sub.add_parser(
        "purge",
        help="delete GOOD pictures already used by a verified training run (dry run by default)",
    )
    p.add_argument("--store", required=True, metavar="PATH",
                   help="folder that holds store/ (or the store/ folder itself)")
    p.add_argument("--models", required=True, metavar="PATH",
                   help="ModelStore root with the per-part model versions")
    p.add_argument("--part", required=True, metavar="PN", help="the part number to purge")
    p.add_argument("--weekly-report", required=True, metavar="FILE",
                   help="weekly report JSON written by trainer/weekly.py")
    p.add_argument("--confirm", action="store_true",
                   help="really delete; without it nothing is deleted")


def _print_plan(plan: PurgePlan) -> None:
    print(f"purge plan for part {plan.part_number} (report {plan.report_name})")
    print(f"  to purge: {len(plan.parts)} parts, {plan.pictures} pictures, {plan.bytes} bytes")
    print(f"  already purged: {plan.already_purged}")
    for reason, count in sorted(Counter(k.reason for k in plan.kept).items()):
        print(f"  kept {count}: {reason}")
    for part in plan.parts:
        print(f"  {part.part_id} session {part.session_id} candidate {part.candidate_version}: "
              f"{len(part.files)} pictures, {part.bytes} bytes")
    print(f"warnings: {len(plan.warnings)}")
    for line in plan.warnings[:WARNINGS_SHOWN]:
        print(f"  {line}")
    if len(plan.warnings) > WARNINGS_SHOWN:
        print(f"  ... {len(plan.warnings) - WARNINGS_SHOWN} more")


def run(args: argparse.Namespace) -> int:
    try:
        plan = plan_purge(args.store, args.models, part_number=args.part,
                          report_path=args.weekly_report)
    except OSError as exc:
        print(f"purge error: {exc}", file=sys.stderr)
        return 1
    if plan.refusal is not None:
        print(f"purge refused: {plan.refusal}", file=sys.stderr)
        print("nothing was deleted", file=sys.stderr)
        return 1
    _print_plan(plan)
    if not args.confirm:
        print()
        print(DRY_RUN_REMINDER)
        return 0
    try:
        result = execute_purge(plan)
    except PurgeRefusal as exc:
        print(f"purge refused: {exc}", file=sys.stderr)
        return 1
    print()
    print(f"deleted {result.files_deleted} pictures ({result.bytes_freed} bytes) "
          f"from {result.parts_purged} parts; tombstones written as {PURGED_FILE}")
    for line in result.errors:
        print(f"error: {line}", file=sys.stderr)
    return 1 if result.errors else 0
