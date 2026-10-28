"""Scan reader for the weekly trainer (Taking data mode, plan step 6).

Reads the scan records the station saved under ``<root>/scans/<line>/YYYY/MM/DD/<scan_id>/``
(layout and checks: ``station/scanstore.py``; format: ``tools/FORMAT.md`` "Scan record").
Standard library only. Nothing here writes to the store.

Rules (plan docs/26_1027_PLAN_taking_data_mode.md, sections 4 and 6):
  training   a scan that passes verify_scan, with label verdict GOOD and status official.
             Its pictures are GOOD training images.
  evaluation a scan that passes verify_scan, with label verdict BAD and status official.
             It is returned separately and never becomes a training image.
  excluded   every other scan (proposed or rejected label, or a failed check). Each one is
             counted with a reason; a failed check also gives problem lines.

Scan pictures are returned as ImageRef objects (the type trainer/dataset.py uses for part
pictures), so the weekly trainer feeds them the same way. ScanImageRef adds the step and the
"defect visible" mark. A scan is kept as one ScanSample, which has the attributes the weekly
trainer reads from a part (images, started_utc).

Split: split_by_toy() holds out whole toys, never single scans or pictures.

Problem lines are strings ``"<scan folder>: <reason>"``, the same shape as trainer/dataset.py.
"""
from __future__ import annotations

import pathlib
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from station.scanstore import list_scans, load_scan, verify_scan
from trainer.dataset import ImageRef

STORE_DIR = "store"  # same name as trainer/dataset.py: the folder that holds the parts
SCANS_DIR = "scans"
GOOD = "GOOD"
BAD = "BAD"
OFFICIAL = "official"
STATUSES = ("official", "proposed", "rejected")
SIDES = ("TOP", "BOTTOM")
REASON_INVALID = "failed checks (see problems)"
RULE_EXPLICIT = "explicit --holdout-toys"
RULE_SEEDED = "seeded whole toys (holdout_fraction)"


@dataclass(frozen=True)
class ScanImageRef(ImageRef):
    """One scan picture: an ImageRef plus its step and its BAD-scan 'defect visible' mark."""
    step: int = 0
    defect_visible: Optional[bool] = None  # True/False on a BAD scan, None on a GOOD scan


@dataclass(frozen=True)
class ScanSample:
    """One intact scan. Carries the attributes the weekly trainer reads from a part."""
    scan_id: str
    toy_id: str
    part_number: str
    started_utc: str
    verdict: str  # GOOD | BAD
    status: str  # proposed | official | rejected
    defects: tuple  # defect names (BAD only; empty for GOOD)
    steps: dict  # {'TOP': n, 'BOTTOM': m}, sides present only
    scan_dir: pathlib.Path
    images: tuple  # tuple of ScanImageRef


@dataclass
class ScanSet:
    """Everything read from one store root. Nothing is written."""
    root: pathlib.Path
    folders: int = 0  # finished scan folders found
    valid: list = field(default_factory=list)  # intact scans, any label and status
    good: list = field(default_factory=list)  # official GOOD: training images
    bad: list = field(default_factory=list)  # official BAD: evaluation only
    excluded: list = field(default_factory=list)  # (scan folder, reason), one per unused scan
    problems: list = field(default_factory=list)  # "<scan folder>: <reason>"

    def excluded_reasons(self) -> dict:
        return dict(sorted(Counter(reason for _folder, reason in self.excluded).items()))


# ------------------------------------------------------------------ helpers
def scans_root_for(store_root) -> pathlib.Path:
    """The folder that contains ``scans/``. Accepts the folder holding ``store/`` or ``store/`` itself."""
    root = pathlib.Path(store_root)
    if (root / SCANS_DIR).is_dir():
        return root
    if root.name == STORE_DIR and (root.parent / SCANS_DIR).is_dir():
        return root.parent
    return root


def has_scans(store_root) -> bool:
    """True when a scans/ folder exists for this store root."""
    return (scans_root_for(store_root) / SCANS_DIR).is_dir()


def _order(sample: ScanSample):
    return (sample.started_utc, sample.scan_id, str(sample.scan_dir))


def _check_fraction(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 0.9:
        raise ValueError(f"holdout_fraction must be a number in [0, 0.9], got {value!r}")
    return float(value)


def _status_reason(sample: ScanSample) -> str:
    if sample.verdict == GOOD:
        return f"GOOD label, status {sample.status}: not used (only official labels train)"
    return f"BAD label, status {sample.status}: not used (only official labels are evaluated)"


def _sample(scan_dir: pathlib.Path) -> tuple[Optional[ScanSample], list[str]]:
    """(sample, []) for an intact scan, (None, problem lines) otherwise."""
    problems = verify_scan(scan_dir)
    if problems:
        return None, [f"{scan_dir}: {text}" for text in problems]
    try:
        doc = load_scan(scan_dir)
    except (OSError, ValueError) as exc:
        return None, [f"{scan_dir}: {exc}"]
    images = tuple(
        ScanImageRef(
            path=scan_dir / img["file"],
            camera_id=img["camera_id"],
            side=img["side"],
            score=None,
            sha256=img["sha256"],
            step=img["step"],
            defect_visible=img["defect_visible"],
        )
        for img in doc["images"]
    )
    label = doc["label"]
    sample = ScanSample(
        scan_id=doc["scan_id"],
        toy_id=doc["toy_id"],
        part_number=doc["part_number"],
        started_utc=doc["started_utc"],
        verdict=label["verdict"],
        status=label["status"],
        defects=tuple(label["defects"]),
        steps=dict(doc["steps"]),
        scan_dir=scan_dir,
        images=images,
    )
    return sample, []


# ------------------------------------------------------------------ reading
def read_scans(store_root) -> ScanSet:
    """Read every scan under the store root and sort it into good, bad, excluded and problems."""
    root = scans_root_for(store_root)
    folders = list_scans(root)  # finished folders only; staging folders are skipped
    scan_set = ScanSet(root=root, folders=len(folders))
    for folder in folders:
        sample, problems = _sample(folder)
        if sample is None:
            scan_set.problems.extend(problems)
            scan_set.excluded.append((str(folder), REASON_INVALID))
            continue
        scan_set.valid.append(sample)
        if sample.status == OFFICIAL and sample.verdict == GOOD:
            scan_set.good.append(sample)
        elif sample.status == OFFICIAL and sample.verdict == BAD:
            scan_set.bad.append(sample)
        else:
            scan_set.excluded.append((str(folder), _status_reason(sample)))
    for group in (scan_set.valid, scan_set.good, scan_set.bad):
        group.sort(key=_order)
    return scan_set


# -------------------------------------------------------------------- split
def split_by_toy(scans, holdout_fraction, seed, holdout_toys=None) -> tuple[list, list, list]:
    """Split GOOD scans into (train, held_out, held_toys) by whole toy_id. Never by picture or scan.

    holdout_toys given (a list of toy ids): exactly those toys are held out.
    holdout_toys None: toys are shuffled with random.Random(seed) and added to the holdout while
    that brings its scan count closer to holdout_fraction * n, as trainer/dataset.py does for days.
    At least one toy stays in train. Both lists are in started_utc, scan_id order.
    """
    fraction = _check_fraction(holdout_fraction)
    by_toy: dict = {}
    for scan in sorted(scans, key=_order):
        by_toy.setdefault(scan.toy_id, []).append(scan)
    if holdout_toys is not None:
        held = {toy for toy in holdout_toys if toy in by_toy}
    else:
        held = set()
        toys = sorted(by_toy)
        if fraction and toys:
            target = fraction * len(scans)
            order = list(range(len(toys)))
            random.Random(seed).shuffle(order)
            count = 0
            for idx in order:
                if len(held) >= len(toys) - 1:
                    break
                size = len(by_toy[toys[idx]])
                if abs(count + size - target) < abs(count - target):
                    held.add(toys[idx])
                    count += size
    train = [s for toy in sorted(by_toy) if toy not in held for s in by_toy[toy]]
    holdout = [s for toy in sorted(by_toy) if toy in held for s in by_toy[toy]]
    return train, holdout, sorted(held)


# ----------------------------------------------------------------- counting
def image_counts(samples) -> dict:
    """'camera/side' -> number of pictures."""
    tally: Counter = Counter()
    for sample in samples:
        for img in sample.images:
            tally[f"{img.camera_id}/{img.side}"] += 1
    return dict(sorted(tally.items()))


def bad_defect_counts(samples) -> dict:
    """Per defect name: scans, and pictures marked defect visible True / False.

    A scan with several defects counts its pictures under each defect name. The mark is per picture.
    """
    tally: dict = {}
    for sample in samples:
        for defect in sample.defects:
            row = tally.setdefault(defect, {"scans": 0, "visible_true": 0, "visible_false": 0})
            row["scans"] += 1
            for img in sample.images:
                row["visible_true" if img.defect_visible else "visible_false"] += 1
    return dict(sorted(tally.items()))


def inventory_scans(store_root) -> dict:
    """Plain, JSON-ready counts for every scan under the store root (all scans; no part filter)."""
    return summarize(read_scans(store_root))


def summarize(scan_set: ScanSet) -> dict:
    """Plain, JSON-ready counts for one ScanSet (see inventory_scans)."""
    by_toy: dict = {}
    by_label: Counter = Counter()
    by_status: Counter = Counter()
    sides: Counter = Counter()
    for sample in scan_set.valid:
        by_toy.setdefault(sample.toy_id, {GOOD: 0, BAD: 0})[sample.verdict] += 1
        by_label[sample.verdict] += 1
        by_status[sample.status] += 1
        for side in sample.steps:
            sides[side] += 1
    return {
        "scans_root": str(scan_set.root),
        "folders": scan_set.folders,
        "intact": len(scan_set.valid),
        "problems": len(scan_set.problems),
        "by_toy": dict(sorted(by_toy.items())),
        "by_label": {GOOD: by_label[GOOD], BAD: by_label[BAD]},
        "by_status": {name: by_status[name] for name in STATUSES},
        "scans_by_side": {side: sides[side] for side in SIDES},
        "images_by_camera_side": image_counts(scan_set.valid),
        "training_images_by_camera_side": image_counts(scan_set.good),
        "excluded": {"total": len(scan_set.excluded), "by_reason": scan_set.excluded_reasons()},
        "bad_defects_official": bad_defect_counts(scan_set.bad),
    }
