from __future__ import annotations

import pathlib
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from station.scanstore import list_scans, load_scan, verify_scan
from trainer.dataset import ImageRef

STORE_DIR = "store"
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
    step: int = 0
    defect_visible: Optional[bool] = None


@dataclass(frozen=True)
class ScanSample:
    scan_id: str
    toy_id: str
    part_number: str
    started_utc: str
    verdict: str
    status: str
    defects: tuple
    steps: dict
    scan_dir: pathlib.Path
    images: tuple


@dataclass
class ScanSet:
    root: pathlib.Path
    folders: int = 0
    valid: list = field(default_factory=list)
    good: list = field(default_factory=list)
    bad: list = field(default_factory=list)
    excluded: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    def excluded_reasons(self) -> dict:
        return dict(sorted(Counter(reason for _folder, reason in self.excluded).items()))


def scans_root_for(store_root) -> pathlib.Path:
    root = pathlib.Path(store_root)
    if (root / SCANS_DIR).is_dir():
        return root
    if root.name == STORE_DIR and (root.parent / SCANS_DIR).is_dir():
        return root.parent
    return root


def has_scans(store_root) -> bool:
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


def read_scans(store_root) -> ScanSet:
    root = scans_root_for(store_root)
    folders = list_scans(root)
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


def split_by_toy(scans, holdout_fraction, seed, holdout_toys=None) -> tuple[list, list, list]:
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


def image_counts(samples) -> dict:
    tally: Counter = Counter()
    for sample in samples:
        for img in sample.images:
            tally[f"{img.camera_id}/{img.side}"] += 1
    return dict(sorted(tally.items()))


def bad_defect_counts(samples) -> dict:
    tally: dict = {}
    for sample in samples:
        for defect in sample.defects:
            row = tally.setdefault(defect, {"scans": 0, "visible_true": 0, "visible_false": 0})
            row["scans"] += 1
            for img in sample.images:
                row["visible_true" if img.defect_visible else "visible_false"] += 1
    return dict(sorted(tally.items()))


def inventory_scans(store_root) -> dict:
    return summarize(read_scans(store_root))


def summarize(scan_set: ScanSet) -> dict:
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
