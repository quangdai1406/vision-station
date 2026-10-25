"""Dataset reader for the weekly trainer.

Reads the evidence the stations saved under
``<store>/store/<line>/YYYY/MM/DD/<part_id>/`` (layout: ``station/storage.py``)
and turns it into per-part GOOD and BAD sets. Standard library only. Nothing
here writes to the store.

Label policy, from part.json:
  BAD                        final_verdict == BAD
  GOOD, strict (default)     final GOOD and machine GOOD and override is null
  GOOD, include-overridden   final GOOD, even when the machine said BAD or REVIEW
                             (a human decided)
  anything else              excluded (REVIEW never reaches final_verdict)

Parts with a purged.json tombstone (pictures deleted by trainer/purge.py) are
skipped with a note, not read as problems of the store.

Problem lines are strings ``"<part folder>: <reason>"``. An unreadable part.json
cannot be tied to a part number, so it is reported by every scan that reads it.

Split: split_holdout() assigns whole calendar days (UTC) to the holdout, never
single images. If the parts fall on fewer than two days it assigns whole parts.
The holdout size is approximate: days (or parts) are added greedily in a seeded
random order, only while that brings the holdout closer to the target size.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Iterable, Iterator, Optional, Sequence

PART_JSON = "part.json"
MANIFEST = "manifest.sha256"
PURGED_FILE = "purged.json"  # tombstone written by trainer/purge.py after pictures were deleted
STORE_DIR = "store"
GOOD_POLICIES = ("strict", "include-overridden")
SIDES = ("TOP", "BOTTOM")
_TMP_PREFIX = ".tmp-"
_META_FILES = frozenset({PART_JSON, MANIFEST})
_REQUIRED_KEYS = (
    "part_id",
    "session_id",
    "part_number",
    "started_utc",
    "model_version",
    "machine_verdict",
    "final_verdict",
    "override",
    "camera_count",
    "partial_station",
    "images",
)
_TEXT_KEYS = (
    "part_id",
    "session_id",
    "part_number",
    "started_utc",
    "model_version",
    "machine_verdict",
    "final_verdict",
)


@dataclass(frozen=True)
class ImageRef:
    path: pathlib.Path
    camera_id: str
    side: str  # 'TOP' | 'BOTTOM'
    score: Optional[float]
    sha256: str  # as recorded in part.json


@dataclass(frozen=True)
class PartSample:
    part_number: str
    part_id: str
    session_id: str
    started_utc: str
    machine_verdict: str
    final_verdict: str
    override: Optional[dict]
    defect_label: Optional[str]  # part.json 'defect_label' when present
    model_version: str
    partial_station: bool
    camera_count: int
    images: tuple  # tuple of ImageRef
    part_dir: pathlib.Path


# ----------------------------------------------------------------- helpers
def _check_policy(good_policy: str) -> str:
    policy = str(good_policy).strip().lower().replace("_", "-")
    if policy not in GOOD_POLICIES:
        raise ValueError(
            f"good_policy must be one of {', '.join(GOOD_POLICIES)}, got {good_policy!r}"
        )
    return policy


def _parse_day(text: Optional[str], name: str) -> Optional[date]:
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a YYYY-MM-DD date, got {text!r}") from exc


def _parse_utc(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_store(store_root) -> pathlib.Path:
    """Accept the folder that contains 'store/' or the 'store/' folder itself."""
    root = pathlib.Path(store_root)
    if not root.is_dir():
        raise FileNotFoundError(f"store folder not found: {root}")
    inner = root / STORE_DIR
    return inner if inner.is_dir() else root


def _part_json_paths(store_dir: pathlib.Path) -> list[pathlib.Path]:
    found = []
    for path in store_dir.rglob(PART_JSON):
        if not path.is_file():
            continue
        parents = path.relative_to(store_dir).parts[:-1]
        if any(p.startswith(_TMP_PREFIX) for p in parents):
            continue  # an interrupted save, never published
        found.append(path)
    return sorted(found)


def _plain_name(name) -> bool:
    return (
        isinstance(name, str)
        and name not in ("", ".", "..")
        and "/" not in name
        and "\\" not in name
        and "\x00" not in name
    )


def _check_doc(doc: dict) -> Optional[str]:
    missing = [k for k in _REQUIRED_KEYS if k not in doc]
    if missing:
        return f"missing keys: {', '.join(missing)}"
    for key in _TEXT_KEYS:
        if not isinstance(doc[key], str) or not doc[key]:
            return f"bad value for {key}"
    try:
        _parse_utc(doc["started_utc"])
    except ValueError:
        return "bad started_utc"
    if doc["override"] is not None and not isinstance(doc["override"], dict):
        return "bad override"
    if not isinstance(doc["partial_station"], bool):
        return "bad partial_station"
    count = doc["camera_count"]
    if isinstance(count, bool) or not isinstance(count, int):
        return "bad camera_count"
    if not isinstance(doc["images"], list) or not doc["images"]:
        return "images is not a non-empty list"
    label = doc.get("defect_label")
    if label is not None and not isinstance(label, str):
        return "bad defect_label"
    return None


def _read_part_doc(part_json: pathlib.Path) -> tuple[Optional[dict], Optional[str]]:
    try:
        doc = json.loads(part_json.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, f"unreadable part.json: {type(exc).__name__}"
    if not isinstance(doc, dict):
        return None, "part.json is not a JSON object"
    reason = _check_doc(doc)
    if reason:
        return None, reason
    return doc, None


def _passes(doc: dict, part_number, since_day, until_day, full_station_only) -> bool:
    if part_number is not None and doc["part_number"] != part_number:
        return False
    day = _parse_utc(doc["started_utc"]).date()
    if since_day is not None and day < since_day:
        return False
    if until_day is not None and day >= until_day:
        return False
    if full_station_only and doc["partial_station"]:
        return False
    return True


def _image_ref(part_dir: pathlib.Path, entry) -> tuple[Optional[ImageRef], Optional[str]]:
    if not isinstance(entry, dict):
        return None, "bad image entry"
    name = entry.get("file")
    camera = entry.get("camera_id")
    side = entry.get("side")
    digest = entry.get("sha256")
    score = entry.get("score")
    if not _plain_name(name):
        return None, f"bad image file name: {name!r}"
    if not isinstance(camera, str) or not camera:
        return None, f"bad camera_id in {name}"
    if side not in SIDES:
        return None, f"bad side in {name}"
    if not isinstance(digest, str) or not digest:
        return None, f"bad sha256 in {name}"
    if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
        return None, f"bad score in {name}"
    ref = ImageRef(
        path=part_dir / name,
        camera_id=camera,
        side=side,
        score=None if score is None else float(score),
        sha256=digest,
    )
    return ref, None


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_problems(part_dir: pathlib.Path, refs: Sequence[ImageRef]) -> list[str]:
    """Same rules as station.storage.Store.verify_part, without needing a Store."""
    manifest_path = part_dir / MANIFEST
    if not manifest_path.is_file():
        return ["missing manifest.sha256"]
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ["unreadable manifest.sha256"]
    problems: list[str] = []
    expected: dict[str, str] = {}
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        digest, sep, name = line.partition("  ")
        if not sep or len(digest) != 64 or not _plain_name(name):
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
    for ref in refs:
        if expected.get(ref.path.name) != ref.sha256:
            problems.append(f"part.json disagrees with manifest: {ref.path.name}")
    return problems


def _make_sample(part_dir: pathlib.Path, doc: dict, verify: bool) -> tuple[Optional[PartSample], list[str]]:
    refs: list[ImageRef] = []
    for entry in doc["images"]:
        ref, reason = _image_ref(part_dir, entry)
        if ref is None:
            return None, [reason]
        refs.append(ref)
    missing = [r.path.name for r in refs if not r.path.is_file()]
    if missing:
        return None, [f"missing image file: {name}" for name in missing]
    if verify:
        problems = _verify_problems(part_dir, refs)
        if problems:
            return None, problems
    sample = PartSample(
        part_number=doc["part_number"],
        part_id=doc["part_id"],
        session_id=doc["session_id"],
        started_utc=doc["started_utc"],
        machine_verdict=doc["machine_verdict"],
        final_verdict=doc["final_verdict"],
        override=doc["override"],
        defect_label=doc.get("defect_label"),
        model_version=doc["model_version"],
        partial_station=doc["partial_station"],
        camera_count=doc["camera_count"],
        images=tuple(refs),
        part_dir=part_dir,
    )
    return sample, []


def _note(problems: Optional[list], part_dir: pathlib.Path, reason: str) -> None:
    if problems is not None:
        problems.append(f"{part_dir}: {reason}")


def _sort_key(sample: PartSample):
    return (_parse_utc(sample.started_utc), sample.part_id, sample.session_id, str(sample.part_dir))


def _fmt_utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ reading
def iter_parts(
    store_root,
    *,
    part_number: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    full_station_only: bool = True,
    verify: bool = True,
    problems: Optional[list] = None,
) -> Iterator[PartSample]:
    """Yield intact parts, ordered by started_utc then part_id.

    since is inclusive and until exclusive, compared with the UTC date of
    started_utc. Parts that fail a check are skipped and described in
    ``problems`` (a list you pass in). Raises FileNotFoundError when the store
    folder is missing and ValueError for a malformed date.
    """
    since_day = _parse_day(since, "since")
    until_day = _parse_day(until, "until")
    store_dir = _resolve_store(store_root)
    found: list[PartSample] = []
    for part_json in _part_json_paths(store_dir):
        part_dir = part_json.parent
        doc, reason = _read_part_doc(part_json)
        if doc is None:
            _note(problems, part_dir, reason)
            continue
        if not _passes(doc, part_number, since_day, until_day, full_station_only):
            continue
        if (part_dir / PURGED_FILE).is_file():
            # Pictures were deleted on purpose after a verified training run (trainer purge).
            _note(problems, part_dir, "skipped: pictures purged after training (see purged.json)")
            continue
        sample, reasons = _make_sample(part_dir, doc, verify)
        if sample is None:
            for text in reasons:
                _note(problems, part_dir, text)
            continue
        found.append(sample)
    found.sort(key=_sort_key)
    return iter(found)


def list_part_numbers(
    store_root,
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    full_station_only: bool = True,
) -> dict:
    """Map part_number -> number of parts. Reads part.json only; no image checks."""
    since_day = _parse_day(since, "since")
    until_day = _parse_day(until, "until")
    store_dir = _resolve_store(store_root)
    counts: Counter = Counter()
    for part_json in _part_json_paths(store_dir):
        doc, _reason = _read_part_doc(part_json)
        if doc is None:
            continue
        if _passes(doc, None, since_day, until_day, full_station_only):
            counts[doc["part_number"]] += 1
    return dict(sorted(counts.items()))


# ----------------------------------------------------------------- labels
def classify(sample: PartSample, good_policy: str = "strict") -> str:
    """Return 'good', 'bad' or 'excluded' for one part under the label policy."""
    policy = _check_policy(good_policy)
    if sample.final_verdict == "BAD":
        return "bad"
    if sample.final_verdict == "GOOD":
        if policy == "include-overridden":
            return "good"
        if sample.machine_verdict == "GOOD" and sample.override is None:
            return "good"
    return "excluded"


@dataclass
class Dataset:
    part_number: str
    good_policy: str = "strict"
    good: list = field(default_factory=list)
    bad: list = field(default_factory=list)
    excluded: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    @classmethod
    def from_samples(
        cls,
        part_number: str,
        samples: Iterable[PartSample],
        *,
        good_policy: str = "strict",
        problems: Optional[list] = None,
        exclude_session_ids: Iterable[str] = (),
    ) -> "Dataset":
        """Label already-read samples. Sessions listed in exclude_session_ids go to excluded."""
        policy = _check_policy(good_policy)
        dataset = cls(part_number=part_number, good_policy=policy, problems=list(problems or []))
        skip = set(exclude_session_ids)
        buckets = {"good": dataset.good, "bad": dataset.bad, "excluded": dataset.excluded}
        for sample in samples:
            if sample.session_id in skip:
                dataset.excluded.append(sample)
                continue
            buckets[classify(sample, policy)].append(sample)
        return dataset

    @classmethod
    def collect(
        cls,
        store_root,
        part_number: str,
        *,
        since: Optional[str] = None,
        until: Optional[str] = None,
        good_policy: str = "strict",
        full_station_only: bool = True,
        verify: bool = True,
        exclude_session_ids: Iterable[str] = (),
    ) -> "Dataset":
        """Read one part number from the store and label its parts."""
        if not isinstance(part_number, str) or not part_number:
            raise ValueError("part_number is required")
        policy = _check_policy(good_policy)
        problems: list[str] = []
        samples = iter_parts(
            store_root,
            part_number=part_number,
            since=since,
            until=until,
            full_station_only=full_station_only,
            verify=verify,
            problems=problems,
        )
        return cls.from_samples(
            part_number,
            samples,
            good_policy=policy,
            problems=problems,
            exclude_session_ids=exclude_session_ids,
        )

    def counts(self) -> dict:
        """(camera_id, side, 'good'|'bad') -> number of images."""
        tally: Counter = Counter()
        for label, group in (("good", self.good), ("bad", self.bad)):
            for sample in group:
                for img in sample.images:
                    tally[(img.camera_id, img.side, label)] += 1
        return dict(sorted(tally.items()))

    def summary(self) -> dict:
        """Plain, JSON-ready overview. first_utc/last_utc cover good, bad and excluded parts."""
        everything = self.good + self.bad + self.excluded
        if everything:
            times = [_parse_utc(s.started_utc) for s in everything]
            first, last = _fmt_utc(min(times)), _fmt_utc(max(times))
        else:
            first = last = None
        return {
            "part_number": self.part_number,
            "parts_good": len(self.good),
            "parts_bad": len(self.bad),
            "parts_excluded": len(self.excluded),
            "problems": len(self.problems),
            "images_by_key": {
                f"{cam}/{side}/{label}": n for (cam, side, label), n in self.counts().items()
            },
            "first_utc": first,
            "last_utc": last,
        }


# ------------------------------------------------------------------ split
def _check_fraction(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 0.9:
        raise ValueError(f"holdout_fraction must be a number in [0, 0.9], got {value!r}")
    return float(value)


def split_holdout(samples: Iterable[PartSample], holdout_fraction: float, seed) -> tuple[list, list]:
    """Split parts into (train, holdout) by whole calendar day (UTC).

    Days are shuffled with random.Random(seed) and added to the holdout while
    that brings its part count closer to holdout_fraction * n. At least one
    day stays in train. With fewer than two days, whole parts are used
    instead, so no part's images are ever split. Both lists are in
    started_utc, part_id order.
    """
    fraction = _check_fraction(holdout_fraction)
    ordered = sorted(samples, key=_sort_key)
    if fraction == 0 or not ordered:
        return ordered, []
    by_day: dict = {}
    for sample in ordered:
        by_day.setdefault(_parse_utc(sample.started_utc).date(), []).append(sample)
    if len(by_day) >= 2:
        units = [by_day[day] for day in sorted(by_day)]
    else:
        units = [[sample] for sample in ordered]
    target = fraction * len(ordered)
    order = list(range(len(units)))
    random.Random(seed).shuffle(order)
    held: set = set()
    held_count = 0
    for idx in order:
        if len(held) >= len(units) - 1:
            break
        size = len(units[idx])
        if abs(held_count + size - target) < abs(held_count - target):
            held.add(idx)
            held_count += size
    train = [s for i, unit in enumerate(units) if i not in held for s in unit]
    holdout = [s for i, unit in enumerate(units) if i in held for s in unit]
    return train, holdout
