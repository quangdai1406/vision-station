"""Weekly PatchCore candidates for the trainer: ``python3 -m trainer weekly ...``.

For each part number: read its good parts (strict policy), split held-out good
parts by whole UTC day, build one PatchCore candidate for all (camera, side)
keys, evaluate it, and register it with ModelStore.add_candidate. A candidate
is never approved or promoted here; a person does that.

Incremental update: when the part's approved model is patchcore-v2, the builder
starts from it (PatchCoreBuilder.from_model) and only good parts that started
after the parent's manifest trained_until_utc are added. A patchcore-v1 parent,
an incompatible parent or a non-patchcore parent means a fresh build, and the
report says why.

Thresholds (t_img, t_low) and the count of held-out good parts that would be
flagged are NOT chosen here; the report leaves them as "[Quality to define]".
--min-good and --min-holdout have no defaults on purpose; they are Quality's.

--scans-too (off by default) adds the official GOOD Taking-data scans of the same part
number to the banks as training images. Scans are split by whole toy_id (never by
picture): --holdout-toys names the toys held out, otherwise whole toys are chosen by the
seeded split with holdout_fraction. Parts keep the day split. With scans included, min-good
and min-holdout count parts and scans together, and the candidate is built fresh (an
approved parent is not updated yet). Held-out scan images are scored and reported
separately, by part and by toy. Official BAD scans are only evaluated, never trained.

Reports: weekly-<UTC date>.json and weekly-<UTC date>.txt in --reports, written
atomically. --dry-run prints the plan and writes nothing.

Exit codes: 0 ok (skipped parts are fine), 1 at least one part failed,
2 usage error or missing dependency (numpy, cv2, torch).

numpy, cv2 and torch are imported only when a real run needs them. The feature
extractor can be injected (extractor_factory) and so can the image decoder
(decoder_factory), which is how the tests run without torch. Nothing here reads
secrets or touches the network.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from station.anomaly import (
    DEFAULT_CORESET_RATIO,
    DEFAULT_MAX_BANK,
    DEFAULT_MAX_HOLDOUT,
    DEFAULT_NORM_FACTOR,
    DEFAULT_SEED,
    PatchCoreBuilder,
    PatchCoreModel,
    cv2_rgb_decoder,
)
from station.contracts import Frame, Side, StationError
from station.modelstore import ModelError, ModelStore
from trainer.dataset import Dataset, iter_parts, split_holdout
from trainer.scans import (
    BAD,
    GOOD,
    RULE_EXPLICIT,
    RULE_SEEDED,
    ScanSet,
    read_scans,
    split_by_toy,
    summarize,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
GOOD_POLICY = "strict"
MODEL_TYPE = "patchcore"
MODEL_FILE = "patchcore.npz"
QUALITY = "[Quality to define]"
REPORT_PREFIX = "weekly-"
REPORT_SCHEMA = "trainer-weekly-report/1"
PROBLEMS_SHOWN = 5
SCANS_FRESH_NOTE = "scans included: built fresh (incremental updates do not include scans yet)"
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)  # cutoff when no good parts were trained

Extractor = Any
Decode = Callable[[Frame], Any]


class UsageError(Exception):
    """Bad command line or inputs. Exit code 2."""


class DependencyError(Exception):
    """A needed package is missing. Exit code 2."""


class _Skip(Exception):
    """A part is left out on purpose; the reason goes in the report."""


@dataclass
class _Runtime:
    extractor: Extractor
    decode: Decode


# ------------------------------------------------------------------ arguments
def _at_least(minimum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"must be an integer, got {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value

    parse.__name__ = f"integer>={minimum}"
    return parse


def _fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number, got {text!r}") from None
    if not 0.0 <= value <= 0.9:
        raise argparse.ArgumentTypeError(f"must be in [0, 0.9], got {value}")
    return value


def add_arguments(sub: Any) -> None:
    p = sub.add_parser(
        "weekly",
        help="build PatchCore candidates from the week's good parts (never approves)",
    )
    p.add_argument("--store", required=True, metavar="PATH",
                   help="folder that holds store/ (or the store/ folder itself)")
    p.add_argument("--models", required=True, metavar="PATH", help="ModelStore root folder")
    p.add_argument("--reports", required=True, metavar="PATH", help="folder for weekly-<date>.json/.txt")
    p.add_argument("--part", action="extend", nargs="+", metavar="PN",
                   help="part numbers to train (default: every part number in the store)")
    p.add_argument("--min-good", required=True, type=_at_least(1), metavar="N",
                   help="skip a part with fewer good parts than N (required; Quality decides N)")
    p.add_argument("--min-holdout", required=True, type=_at_least(0), metavar="N",
                   help="skip a part with fewer held-out good parts than N (required; Quality decides N)")
    p.add_argument("--holdout-fraction", required=True, type=_fraction, metavar="F",
                   help="share of good parts held out by whole day (required; Quality decides F)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, metavar="S",
                   help=f"seed for the split and the coreset (default {DEFAULT_SEED})")
    p.add_argument("--extractor-config", metavar="FILE",
                   help='JSON: {"arch", "input_hw": [h, w], "weights_path", "device"}; needed unless --dry-run')
    p.add_argument("--dry-run", action="store_true", help="print the plan; build and write nothing")
    p.add_argument("--now-utc", metavar="ISO",
                   help="run time, ISO 8601 UTC (default: now); names the report and the candidate version")
    p.add_argument("--scans-too", action="store_true",
                   help="also train on official GOOD Taking-data scans, split by whole toy (default: off)")
    p.add_argument("--holdout-toys", metavar="A,B,C",
                   help="with --scans-too: toy ids held out whole (default: chosen by the seeded split)")


def _toy_list(text: str) -> list:
    """'A, B,C' -> ['A', 'B', 'C']. Empty entries are dropped; duplicates keep their first place."""
    toys = list(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
    if not toys:
        raise UsageError("--holdout-toys needs at least one toy id")
    return toys


def run(args: argparse.Namespace) -> int:
    return run_weekly(args)


# ----------------------------------------------------------------- settings
def _parse_utc(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_now(text: Optional[str]) -> datetime:
    if text is None:
        return datetime.now(timezone.utc).replace(microsecond=0)
    try:
        return _parse_utc(text)
    except ValueError:
        raise UsageError(f"--now-utc must be an ISO 8601 time, got {text!r}") from None


def _load_config(path: str) -> dict:
    """Validate the extractor config. Its contents are never printed."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise UsageError(f"cannot read --extractor-config ({type(exc).__name__})") from None
    if not isinstance(data, dict):
        raise UsageError("--extractor-config must be a JSON object")
    arch = data.get("arch")
    weights = data.get("weights_path")
    hw = data.get("input_hw")
    device = data.get("device", "auto")
    if not isinstance(arch, str) or not arch:
        raise UsageError("--extractor-config needs a text 'arch'")
    if not isinstance(weights, str) or not weights:
        raise UsageError("--extractor-config needs a text 'weights_path'")
    if (not isinstance(hw, list) or len(hw) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in hw)):
        raise UsageError("--extractor-config 'input_hw' must be [height, width] integers")
    if not isinstance(device, str):
        raise UsageError("--extractor-config 'device' must be text")
    return {"arch": arch, "weights_path": weights, "input_hw": (hw[0], hw[1]), "device": device}


def _make_runtime(args: argparse.Namespace, extractor_factory, decoder_factory) -> _Runtime:
    try:
        importlib.import_module("numpy")
    except ImportError:
        raise DependencyError("numpy is required for the weekly trainer; install numpy") from None
    if extractor_factory is None:
        if not args.extractor_config:
            raise UsageError("--extractor-config is required for a real run (not needed with --dry-run)")
        cfg = _load_config(args.extractor_config)
        try:
            from station.features_torch import TorchExtractor  # noqa: PLC0415 - needs torch, lazily
            extractor = TorchExtractor(cfg["arch"], device=cfg["device"], input_hw=cfg["input_hw"],
                                       weights_path=cfg["weights_path"])
        except ImportError as exc:
            raise DependencyError(f"torch and torchvision are required for the feature extractor ({exc})") from None
        except StationError as exc:
            raise DependencyError(str(exc)) from None
    else:
        extractor = extractor_factory()
    if decoder_factory is None:
        try:
            decode = cv2_rgb_decoder()
        except StationError as exc:
            raise DependencyError(str(exc)) from None
    else:
        decode = decoder_factory()
    return _Runtime(extractor=extractor, decode=decode)


def _new_builder(runtime: _Runtime, args: argparse.Namespace) -> PatchCoreBuilder:
    return PatchCoreBuilder(
        runtime.extractor, runtime.decode,
        coreset_ratio=DEFAULT_CORESET_RATIO, max_bank=DEFAULT_MAX_BANK, seed=args.seed,
        norm_factor=DEFAULT_NORM_FACTOR, max_holdout_images=DEFAULT_MAX_HOLDOUT,
    )


# ------------------------------------------------------------------- parts
def _frame(ref: Any, sample: Any, seq: int) -> Frame:
    return Frame(
        camera_id=ref.camera_id,
        side=Side(ref.side),
        seq=seq,
        timestamp_utc=sample.started_utc,
        data=ref.path.read_bytes(),
        ext=ref.path.suffix.lstrip(".") or "png",
    )


def _when(sample: Any) -> datetime:
    return _parse_utc(sample.started_utc)


def _parent_plan(parent: Any) -> tuple[Optional[datetime], str, Optional[str]]:
    """(cutoff or None for fresh, note, parent version). The v1/compatibility check needs the file; see _load_base."""
    if parent is None:
        return None, "no approved model: trained fresh", None
    version = parent.version
    if parent.type != MODEL_TYPE:
        return None, f"approved model {version} is type {parent.type!r}, not {MODEL_TYPE}: trained fresh", version
    trained_on = parent.manifest.get("trained_on")
    until = trained_on.get("trained_until_utc") if isinstance(trained_on, dict) else None
    try:
        cutoff = _parse_utc(until) if isinstance(until, str) else None
    except ValueError:
        cutoff = None
    if cutoff is None:
        return None, f"approved model {version} has no trained_until_utc: trained fresh", version
    return cutoff, "", version


def _load_base(parent: Any, runtime: _Runtime, args: argparse.Namespace) -> tuple[Optional[PatchCoreBuilder], str]:
    """Builder seeded from the approved parent, or (None, why a fresh build is needed)."""
    version = parent.version
    try:
        base = PatchCoreModel.load(parent.path, runtime.decode, runtime.extractor)
    except StationError as exc:
        return None, f"approved model {version} could not be loaded ({exc}): trained fresh"
    if base.legacy:
        return None, f"approved model {version} is patchcore-v1, which cannot be updated: trained fresh"
    try:
        return PatchCoreBuilder.from_model(
            base, runtime.extractor, runtime.decode,
            coreset_ratio=DEFAULT_CORESET_RATIO, max_bank=DEFAULT_MAX_BANK, seed=args.seed,
            norm_factor=DEFAULT_NORM_FACTOR, max_holdout_images=DEFAULT_MAX_HOLDOUT,
        ), ""
    except StationError as exc:
        return None, f"approved model {version} is incompatible ({exc}): trained fresh"


def _feed(builder: PatchCoreBuilder, sample: Any, holdout: bool, runtime: _Runtime,
          tally: dict) -> None:
    label = "good_holdout" if holdout else "good_train"
    for seq, ref in enumerate(sample.images):
        frame = _frame(ref, sample, seq)
        if holdout:
            builder.add_holdout(frame)
        else:
            builder.add(frame)
        bucket = tally.setdefault(f"{ref.camera_id}/{ref.side}", {"good_train": 0, "good_holdout": 0})
        bucket[label] += 1


def _summary(values: list) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
    }


def _score_samples(model: Any, samples: list, runtime: _Runtime) -> tuple[list, list]:
    scores: list = []
    part_max: list = []
    for sample in samples:
        part = [model.predict(_frame(ref, sample, seq)).score for seq, ref in enumerate(sample.images)]
        scores.extend(part)
        if part:
            part_max.append(max(part))
    return scores, part_max


def _evaluate(model: Any, held_out: list, bad: list, runtime: _Runtime) -> dict:
    good_scores, good_max = _score_samples(model, held_out, runtime)
    bad_scores, bad_max = _score_samples(model, bad, runtime)
    return {
        "held_out_good": {
            "parts": len(held_out),
            "images_scored": len(good_scores),
            "scores": _summary(good_scores),
            "part_max_scores": sorted(round(x, 4) for x in good_max),
        },
        "bad": {
            "parts": len(bad),
            "images_scored": len(bad_scores),
            "scores": _summary(bad_scores),
            "part_max_scores": sorted(round(x, 4) for x in bad_max),
        },
        "held_out_good_parts_flagged": QUALITY,
        "t_img": QUALITY,
        "t_low": QUALITY,
    }


def _scored_scans(model: Any, samples: list) -> list:
    """(scan, [(picture, score), ...]) for each scan."""
    return [
        (scan, [(ref, model.predict(_frame(ref, scan, seq)).score) for seq, ref in enumerate(scan.images)])
        for scan in samples
    ]


def _scan_group(rows: list) -> dict:
    scores = [score for _scan, pairs in rows for _ref, score in pairs]
    return {"scans": len(rows), "images_scored": len(scores), "scores": _summary(scores)}


def _evaluate_scans(model: Any, held: list, bad: list) -> dict:
    """Held-out GOOD scans and official BAD scans, scored like held-out parts, kept apart from parts.

    BAD pictures are grouped per defect name and by their 'defect visible' mark.
    """
    held_rows = _scored_scans(model, held)
    bad_rows = _scored_scans(model, bad)
    held_toys = sorted({scan.toy_id for scan in held})
    bad_toys = sorted({scan.toy_id for scan in bad})
    per_defect: dict = {}
    for scan, pairs in bad_rows:
        for defect in scan.defects:
            bucket = per_defect.setdefault(defect, {"scans": 0, "visible_true": [], "visible_false": []})
            bucket["scans"] += 1
            for ref, score in pairs:
                bucket["visible_true" if ref.defect_visible else "visible_false"].append(score)
    return {
        "held_out_good": dict(
            _scan_group(held_rows),
            per_toy={toy: _scan_group([r for r in held_rows if r[0].toy_id == toy]) for toy in held_toys},
        ),
        "bad": dict(
            _scan_group(bad_rows),
            per_toy={toy: _scan_group([r for r in bad_rows if r[0].toy_id == toy]) for toy in bad_toys},
            per_defect={
                defect: {
                    "scans": b["scans"],
                    "visible_true": _summary(b["visible_true"]),
                    "visible_false": _summary(b["visible_false"]),
                }
                for defect, b in sorted(per_defect.items())
            },
        ),
        "held_out_good_scans_flagged": QUALITY,
    }


def _register(ms: ModelStore, part_number: str, version: str, model: Any, manifest: dict, now_iso: str) -> Path:
    ms.root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".weekly-", dir=str(ms.root)))
    try:
        model_file = staging / MODEL_FILE
        model.save(model_file)
        return ms.add_candidate(part_number, version, model_file, manifest, now_iso)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _manifest(*, model: Any, mode: str, note: str, parent_version: Optional[str], cutoff_iso: str,
              args: argparse.Namespace, now_iso: str, counts: dict, evaluation: dict,
              runtime: _Runtime) -> dict:
    np = importlib.import_module("numpy")
    extractor = runtime.extractor
    return {
        "type": MODEL_TYPE,
        "camera_ids": sorted({cam for cam, _side in model.keys()}),
        "trained_on": {
            "mode": mode,
            "mode_note": note,
            "parent_version": parent_version,
            "trained_until_utc": cutoff_iso,
            "good_policy": GOOD_POLICY,
            "seed": args.seed,
            "holdout_fraction": args.holdout_fraction,
            "holdout_split": "whole UTC days, or whole parts when the parts fall on fewer than two days",
            "counts": counts,
            "extractor": {
                "name": str(getattr(extractor, "name", "")),
                "input_hw": list(getattr(extractor, "input_hw", []) or []),
                "stride": getattr(extractor, "stride", None),
            },
            "settings": model.describe()["settings"],
            "provenance": {
                "generator": "trainer.weekly",
                "run_utc": now_iso,
                "python": platform.python_version(),
                "numpy": str(np.__version__),
                "source": "no git information recorded",
            },
        },
        "metrics": evaluation,
        "notes": f"Weekly PatchCore candidate ({mode}). Not approved. t_img and t_low: {QUALITY}.",
    }


class _Context:
    """Per-run values shared by every part."""

    def __init__(self, args: argparse.Namespace, ms: ModelStore, runtime: Optional[_Runtime],
                 now: datetime, dry_run: bool, scans: Optional[ScanSet] = None,
                 holdout_toys: Optional[list] = None) -> None:
        self.args = args
        self.ms = ms
        self.runtime = runtime
        self.now = now
        self.now_iso = _iso(now)
        self.version = "weekly-" + now.strftime("%Y%m%dT%H%M%SZ")
        self.dry_run = dry_run
        # Scans are used only when --scans-too was given (scans is then a ScanSet).
        self.scans_too = scans is not None
        self.holdout_toys = holdout_toys  # None: the seeded split chooses the toys
        self.scan_good: dict = {}  # part number -> official GOOD scans
        self.scan_bad: dict = {}  # part number -> official BAD scans (evaluation only)
        if scans is not None:
            for scan in scans.good:
                self.scan_good.setdefault(scan.part_number, []).append(scan)
            for scan in scans.bad:
                self.scan_bad.setdefault(scan.part_number, []).append(scan)
        self.scan_set = scans


def _run_part(part_number: str, samples: list, ctx: _Context, result: dict) -> None:
    args = ctx.args
    ds = Dataset.from_samples(part_number, samples, good_policy=GOOD_POLICY)
    good = ds.good
    scan_good = ctx.scan_good.get(part_number, []) if ctx.scans_too else []
    scan_bad = ctx.scan_bad.get(part_number, []) if ctx.scans_too else []
    result["counts"].update(good_parts=len(good), bad_parts=len(ds.bad), excluded_parts=len(ds.excluded))
    if ctx.scans_too:
        result["counts"].update(good_scans=len(scan_good), bad_scans=len(scan_bad))
    n_good = len(good) + len(scan_good)
    if n_good < args.min_good:
        kind = "good parts and scans" if ctx.scans_too else "good parts"
        raise _Skip(f"{n_good} {kind}, below --min-good {args.min_good}")
    train_all, hold_all = split_holdout(good, args.holdout_fraction, args.seed)
    result["counts"].update(good_train_parts=len(train_all), good_holdout_parts=len(hold_all))
    scan_train: list = []
    scan_hold: list = []
    held_toys: list = []
    rule = RULE_EXPLICIT if ctx.holdout_toys is not None else RULE_SEEDED
    if ctx.scans_too:
        scan_train, scan_hold, held_toys = split_by_toy(
            scan_good, args.holdout_fraction, args.seed, ctx.holdout_toys)
        result["counts"].update(good_train_scans=len(scan_train), good_holdout_scans=len(scan_hold))
        result["scans"] = {"holdout_rule": rule, "holdout_toys": held_toys,
                           "train_scans": len(scan_train), "held_out_scans": len(scan_hold),
                           "images_by_key": {}}
    n_hold = len(hold_all) + len(scan_hold)
    if n_hold < args.min_holdout:
        kind = "held-out good parts and scans" if ctx.scans_too else "held-out good parts"
        raise _Skip(f"{n_hold} {kind}, below --min-holdout {args.min_holdout}")

    builder: Optional[PatchCoreBuilder] = None
    if ctx.scans_too:
        # Scans are not part of the incremental update yet, so the candidate is always built fresh.
        cutoff, note, parent_version = None, SCANS_FRESH_NOTE, None
    else:
        parent = ctx.ms.current(part_number)  # ModelError when the approved model is unusable: a part error
        cutoff, note, parent_version = _parent_plan(parent)
        if cutoff is not None:
            if ctx.dry_run:
                note = "incremental if the approved model is patchcore-v2 and compatible; otherwise fresh (checked at build)"
            else:
                builder, note = _load_base(parent, ctx.runtime, args)
                if builder is None:
                    cutoff = None  # v1, incompatible or unreadable parent: fresh build
    mode = "incremental" if cutoff is not None else "fresh"

    if cutoff is not None:
        new_train = [s for s in train_all if _when(s) > cutoff]
        new_hold = [s for s in hold_all if _when(s) > cutoff]
        if not new_train:
            raise _Skip(f"no good training parts since trained_until_utc {_iso(cutoff)}")
    else:
        new_train, new_hold = train_all, hold_all
        n_train = len(train_all) + len(scan_train)
        if n_train < 2:
            kind = "good training parts and scans" if ctx.scans_too else "good training parts"
            raise _Skip(f"fewer than 2 {kind} after the holdout split; each camera/side needs 2")
    result["counts"].update(new_train_parts=len(new_train), new_holdout_parts=len(new_hold))
    result.update(mode=mode, mode_note=note, parent_version=parent_version)

    if ctx.dry_run:
        result["status"] = "planned"
        return

    if builder is None:
        builder = _new_builder(ctx.runtime, args)
    tally: dict = {}
    for sample in new_train:
        _feed(builder, sample, False, ctx.runtime, tally)
    for sample in new_hold:
        _feed(builder, sample, True, ctx.runtime, tally)
    scan_tally: dict = {}
    for scan in scan_train:
        _feed(builder, scan, False, ctx.runtime, scan_tally)
    for scan in scan_hold:
        _feed(builder, scan, True, ctx.runtime, scan_tally)
    model = builder.build(ctx.version)

    # Parts only: the cutoff is read by the next incremental run, which picks up parts after it.
    # Scans are not in the incremental path, so they must not move the cutoff.
    trained_until = max((_when(s) for s in good), default=_EPOCH)
    cutoff_iso = _iso(trained_until)
    evaluation = _evaluate(model, hold_all, ds.bad, ctx.runtime)
    counts = dict(result["counts"], images_by_key=dict(sorted(tally.items())))
    if ctx.scans_too:
        evaluation["scans"] = _evaluate_scans(model, scan_hold, scan_bad)
        result["scans"]["images_by_key"] = dict(sorted(scan_tally.items()))
        counts["scans"] = result["scans"]
    manifest = _manifest(
        model=model, mode=mode, note=note, parent_version=parent_version, cutoff_iso=cutoff_iso,
        args=args, now_iso=ctx.now_iso, counts=counts, evaluation=evaluation, runtime=ctx.runtime,
    )
    _register(ctx.ms, part_number, ctx.version, model, manifest, ctx.now_iso)
    # Keys read by trainer.purge: only pictures fed to the builder this run are listed as trained.
    # Scan ids have their own keys: they are not parts and purge never deletes them.
    result.update(status="candidate", version=ctx.version, trained_until_utc=cutoff_iso,
                  counts=counts, evaluation=evaluation,
                  candidate_version=ctx.version, training_succeeded=True,
                  trained_session_ids=sorted(s.session_id for s in new_train),
                  holdout_session_ids=sorted(s.session_id for s in hold_all))
    if ctx.scans_too:
        result["trained_scan_ids"] = sorted(s.scan_id for s in scan_train)
        result["holdout_scan_ids"] = sorted(s.scan_id for s in scan_hold)


def _process_part(part_number: str, samples: list, ctx: _Context) -> dict:
    result: dict = {
        "part_number": part_number,
        "status": "",
        "reason": "",
        "version": None,
        "mode": None,
        "mode_note": "",
        "parent_version": None,
        "trained_until_utc": None,
        "counts": {},
        "evaluation": None,
    }
    try:
        _run_part(part_number, samples, ctx, result)
    except _Skip as exc:
        result["status"], result["reason"] = "skipped", str(exc)
    except Exception as exc:  # one bad part must not stop the others; the report names it
        result["status"], result["reason"] = "error", f"{type(exc).__name__}: {exc}"
    return result


# ------------------------------------------------------------------ report
def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
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


def _fmt_scores(block: dict) -> str:
    s = block["scores"]
    if not s["count"]:
        return "no images scored"
    return f"{s['count']} images, score min/median/max {s['min']} / {s['median']} / {s['max']}"


def _render_text(report: dict) -> str:
    inp = report["inputs"]
    lines = [f"Weekly trainer report, run {report['run_utc']}"]
    if report["dry_run"]:
        lines.append("DRY RUN: nothing was built or written.")
    lines += [
        f"min good {inp['min_good']}, min held-out {inp['min_holdout']}, "
        f"holdout fraction {inp['holdout_fraction']}, seed {inp['seed']}, good policy {GOOD_POLICY}",
        f"t_img: {QUALITY}   t_low: {QUALITY}   held-out good parts flagged: {QUALITY}",
        "",
    ]
    for part in report["parts"]:
        head = f"Part {part['part_number']}: {part['status'].upper()}"
        if part["version"]:
            head += f" -> candidate {part['version']}"
        lines.append(head)
        if part["reason"]:
            lines.append(f"  reason: {part['reason']}")
        if part["mode"]:
            extra = f" ({part['mode_note']})" if part["mode_note"] else ""
            parent = f", parent {part['parent_version']}" if part["parent_version"] else ""
            lines.append(f"  mode: {part['mode']}{parent}{extra}")
        c = part["counts"]
        if c:
            lines.append(f"  good parts {c.get('good_parts', 0)} (train {c.get('good_train_parts', 0)}, "
                         f"held-out {c.get('good_holdout_parts', 0)}), bad {c.get('bad_parts', 0)}, "
                         f"excluded {c.get('excluded_parts', 0)}")
        if "new_train_parts" in c and part["mode"] == "incremental":
            lines.append(f"  new since parent: train {c['new_train_parts']}, held-out {c['new_holdout_parts']}")
        ev = part["evaluation"]
        if ev:
            lines.append(f"  held-out good ({ev['held_out_good']['parts']} parts): {_fmt_scores(ev['held_out_good'])}")
            lines.append(f"  labelled BAD ({ev['bad']['parts']} parts): {_fmt_scores(ev['bad'])}")
            lines.append(f"  held-out good parts flagged: {QUALITY}")
            if "scans" in ev:
                lines += _scan_eval_lines(ev["scans"])
        if "scans" in part and part["scans"] is not None:
            sc = part["scans"]
            toys = ", ".join(sc["holdout_toys"]) or "none"
            lines.append(f"  scans: train {sc['train_scans']}, held-out {sc['held_out_scans']} "
                         f"({sc['holdout_rule']}); held-out toys: {toys}")
        lines.append("")
    if "scans" in report:
        lines += _scans_section_lines(report["scans"])
    problems = report["store_problems"]
    lines.append(f"store problems: {problems['count']}")
    for item in problems["first"][:PROBLEMS_SHOWN]:
        lines.append(f"  {item}")
    s = report["summary"]
    lines += [
        "",
        f"candidates {s['candidate']}, planned {s['planned']}, skipped {s['skipped']}, failed {s['error']}",
        "Nothing was approved. Approval is a separate, named step.",
        f"exit code {report['exit_code']}",
    ]
    return "\n".join(lines) + "\n"


def _fmt_summary(s: dict) -> str:
    if not s["count"]:
        return "none"
    return f"{s['count']} images, score min/median/max {s['min']} / {s['median']} / {s['max']}"


def _scan_eval_lines(block: dict) -> list:
    """Held-out scan scores under one part (kept apart from the part pictures)."""
    held = block["held_out_good"]
    bad = block["bad"]
    lines = [f"  held-out good scans ({held['scans']} scans): {_fmt_scores(held)}"]
    for toy, row in held["per_toy"].items():
        lines.append(f"    toy {toy} ({row['scans']} scans): {_fmt_scores(row)}")
    lines.append(f"  labelled BAD scans ({bad['scans']} scans, evaluation only): {_fmt_scores(bad)}")
    for defect, row in bad["per_defect"].items():
        lines.append(f"    defect {defect} ({row['scans']} scans): defect visible "
                     f"{_fmt_summary(row['visible_true'])}; not visible {_fmt_summary(row['visible_false'])}")
    lines.append(f"  held-out good scans flagged: {QUALITY}")
    return lines


def _build_scans_section(ctx: _Context, parts: list) -> dict:
    """The report's 'scans' section: what was read, excluded, held out and scored."""
    scan_set = ctx.scan_set
    inv = summarize(scan_set)
    per_part: dict = {}
    for part in parts:
        sc = part.get("scans")
        if sc is None:
            continue
        per_part[part["part_number"]] = {
            "status": part["status"],
            "train_scans": sc["train_scans"],
            "held_out_scans": sc["held_out_scans"],
            "holdout_rule": sc["holdout_rule"],
            "holdout_toys": sc["holdout_toys"],
            "images_by_key": sc["images_by_key"],
        }
    held_for: dict = {}
    for pn, row in per_part.items():
        for toy in row["holdout_toys"]:
            held_for.setdefault(toy, []).append(pn)
    per_toy = {
        toy: {"GOOD": row[GOOD], "BAD": row[BAD], "held_out_for_part_numbers": held_for.get(toy, [])}
        for toy, row in inv["by_toy"].items()
    }
    return {
        "folders": inv["folders"],
        "intact": inv["intact"],
        "problems": inv["problems"],
        "by_label": inv["by_label"],
        "by_status": inv["by_status"],
        "scans_by_side": inv["scans_by_side"],
        "training_scans": len(scan_set.good),
        "evaluation_scans": len(scan_set.bad),
        "excluded": inv["excluded"],
        "holdout_rule": RULE_EXPLICIT if ctx.holdout_toys is not None else RULE_SEEDED,
        "holdout_toys_requested": list(ctx.holdout_toys or []),
        "per_toy": per_toy,
        "images_by_camera_side": inv["images_by_camera_side"],
        "training_images_by_camera_side": inv["training_images_by_camera_side"],
        "bad_defects_official": inv["bad_defects_official"],
        "per_part": per_part,
    }


def _scans_section_lines(sec: dict) -> list:
    lines = [
        "Scans (Taking data, --scans-too)",
        f"  folders {sec['folders']}, intact {sec['intact']}, problems {sec['problems']}",
        f"  labels: GOOD {sec['by_label'][GOOD]}, BAD {sec['by_label'][BAD]}; status official "
        f"{sec['by_status']['official']}, proposed {sec['by_status']['proposed']}, "
        f"rejected {sec['by_status']['rejected']}",
        f"  training scans (official GOOD): {sec['training_scans']}; "
        f"evaluation only (official BAD): {sec['evaluation_scans']}",
    ]
    excl = sec["excluded"]
    reasons = "".join(f"\n    {n} x {reason}" for reason, n in excl["by_reason"].items())
    lines.append(f"  excluded: {excl['total']}{reasons}")
    toys = ", ".join(sec["holdout_toys_requested"]) if sec["holdout_toys_requested"] else "none given"
    lines.append(f"  holdout rule: {sec['holdout_rule']}; toys named: {toys}")
    lines.append("  per toy (intact scans):")
    for toy, row in sec["per_toy"].items():
        held = ", ".join(row["held_out_for_part_numbers"]) or "no"
        lines.append(f"    {toy}: GOOD {row[GOOD]}, BAD {row[BAD]}; held out for part {held}")
    lines.append("  pictures per camera/side, all intact scans: " + (", ".join(
        f"{k} {v}" for k, v in sec["images_by_camera_side"].items()) or "none"))
    for pn, row in sec["per_part"].items():
        imgs = ", ".join(
            f"{k} train {v.get('good_train', 0)} held-out {v.get('good_holdout', 0)}"
            for k, v in row["images_by_key"].items()) or "none"
        held = ", ".join(row["holdout_toys"]) or "none"
        lines.append(f"  part {pn} ({row['status']}): train scans {row['train_scans']}, "
                     f"held-out scans {row['held_out_scans']}, held-out toys {held}")
        lines.append(f"    pictures: {imgs}")
    for defect, row in sec["bad_defects_official"].items():
        lines.append(f"  BAD defect {defect}: {row['scans']} scans, pictures defect visible "
                     f"{row['visible_true']}, not visible {row['visible_false']}")
    lines.append("")
    return lines


def _build_report(ctx: _Context, store: Path, models: Path, reports: Path, parts: list,
                  store_problems: list) -> dict:
    args = ctx.args
    summary = {"candidate": 0, "planned": 0, "skipped": 0, "error": 0}
    for part in parts:
        summary[part["status"]] = summary.get(part["status"], 0) + 1
    exit_code = EXIT_ERROR if summary["error"] else EXIT_OK
    report = {
        "schema": REPORT_SCHEMA,
        "run_utc": ctx.now_iso,
        "dry_run": ctx.dry_run,
        "inputs": {
            "store": str(store),
            "models": str(models),
            "reports": str(reports),
            "min_good": args.min_good,
            "min_holdout": args.min_holdout,
            "holdout_fraction": args.holdout_fraction,
            "seed": args.seed,
            "good_policy": GOOD_POLICY,
            "parts_requested": list(args.part) if args.part else "all",
            "extractor_config": args.extractor_config,
            "extractor": None if ctx.runtime is None else str(getattr(ctx.runtime.extractor, "name", "")),
        },
        "quality_to_define": {"t_img": QUALITY, "t_low": QUALITY, "held_out_good_parts_flagged": QUALITY},
        "store_problems": {"count": len(store_problems), "first": store_problems[:50]},
        "summary": summary,
        "parts": parts,
        "exit_code": exit_code,
    }
    if ctx.scans_too:
        report["scans"] = _build_scans_section(ctx, parts)
    return report


# -------------------------------------------------------------------- run
def run_weekly(args: argparse.Namespace, *, extractor_factory: Optional[Callable[[], Any]] = None,
               decoder_factory: Optional[Callable[[], Decode]] = None) -> int:
    """Run the weekly job. extractor_factory/decoder_factory are for tests and alternative backends."""
    try:
        now = _parse_now(args.now_utc)
        store = Path(args.store)
        models = Path(args.models)
        reports = Path(args.reports)
        if not store.is_dir():
            raise UsageError(f"store folder not found: {store}")
        if models.exists() and not models.is_dir():
            raise UsageError(f"--models is not a folder: {models}")
        holdout_toys = None
        if args.holdout_toys is not None:
            if not args.scans_too:
                raise UsageError("--holdout-toys needs --scans-too")
            holdout_toys = _toy_list(args.holdout_toys)
        if args.extractor_config and not args.dry_run:
            _load_config(args.extractor_config)  # validate early; contents are not printed
        runtime = None
        if not args.dry_run:
            runtime = _make_runtime(args, extractor_factory, decoder_factory)
    except UsageError as exc:
        print(f"weekly: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except DependencyError as exc:
        print(f"weekly: missing dependency: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        problems: list[str] = []
        groups: dict = {}
        for sample in iter_parts(store, full_station_only=True, verify=True, problems=problems):
            groups.setdefault(sample.part_number, []).append(sample)
        scan_set = None
        if args.scans_too:
            scan_set = read_scans(store)
            problems.extend(scan_set.problems)
    except (OSError, ValueError) as exc:
        print(f"weekly: could not read the store: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if holdout_toys is not None:
        known = {scan.toy_id for scan in scan_set.valid}
        unknown = [toy for toy in holdout_toys if toy not in known]
        if unknown:
            print("weekly: --holdout-toys names toys with no intact scan in the store: "
                  + ", ".join(unknown), file=sys.stderr)
            return EXIT_USAGE

    names = set(groups)
    if scan_set is not None:
        names |= {scan.part_number for scan in scan_set.good + scan_set.bad}
    wanted = list(dict.fromkeys(args.part)) if args.part else sorted(names)
    ctx = _Context(args, ModelStore(models), runtime, now, bool(args.dry_run),
                   scans=scan_set, holdout_toys=holdout_toys)
    parts = [_process_part(pn, groups.get(pn, []), ctx) for pn in wanted]
    report = _build_report(ctx, store, models, reports, parts, problems)
    text = _render_text(report)
    if ctx.dry_run:
        print(text, end="")
        return report["exit_code"]

    try:
        reports.mkdir(parents=True, exist_ok=True)
        base = f"{REPORT_PREFIX}{now.strftime('%Y-%m-%d')}"
        _write_atomic(reports / f"{base}.json",
                      json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        _write_atomic(reports / f"{base}.txt", text)
    except OSError as exc:
        print(f"weekly: could not write the report: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(text, end="")
    return report["exit_code"]
