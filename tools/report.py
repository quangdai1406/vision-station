"""Plain-language repeatability report for capture session folders.

CLI:  python -m tools.report SESSION_DIR [SESSION_DIR ...] [--out report_dir]
      [--outlier-sigma K] [--outlier-floor-gray F] [--skip-first N] [--min-shift-response R]

Reads session.json and frames_dev<N>.npz (see tools/FORMAT.md) and writes
report.json, report.md and summary.csv. The report is descriptive only: it gives
measurements and their plain meaning, no pass/fail verdict and no acceptance limit.
Numpy is needed; it is imported lazily so that the module itself loads without it.
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from tools import ToolDependencyError

SCHEMA = 1
MODES = ("stationary", "placement")
# Stationary noise floor is measured WITHOUT alignment: phase-correlation alignment against a
# single noisy frame can fit sensor noise and would hide real mount vibration.
NOISE_FLOOR_BASIS = ("per-frame residual against the median frame of the series, "
                     "no alignment, stationary part")

# Frame-outlier rule for stationary sessions. A frame is flagged when
#   |mean_i - median(means)| > max(K * 1.4826 * MAD, F).
# K and F are engineering settings for SPOTTING bad frames; they are not quality thresholds.
MAD_SCALE = 1.4826
DEFAULT_OUTLIER_SIGMA = 5.0
DEFAULT_OUTLIER_FLOOR_GRAY = 2.0
DEFAULT_SKIP_FIRST = 0
DEFAULT_MIN_SHIFT_RESPONSE = 0.1
DEFAULT_SETTINGS = {
    "skip_first": DEFAULT_SKIP_FIRST,
    "outlier_sigma": DEFAULT_OUTLIER_SIGMA,
    "outlier_floor_gray": DEFAULT_OUTLIER_FLOOR_GRAY,
    "min_shift_response": DEFAULT_MIN_SHIFT_RESPONSE,
}
OUTLIER_FORMULA = "flagged when |mean - median| > max(K x 1.4826 x MAD, F)"
OUTLIER_RULE_TEXT = ("a rule to help you spot frames taken before the camera settled or "
                     "disturbed by light changes, not a quality threshold")

# Camera controls shown in section 2: (label, key in readback / applied_ok).
CONTROLS = [
    ("Autofocus", "autofocus"),
    ("Focus", "focus"),
    ("Auto exposure", "auto_exposure"),
    ("Exposure", "exposure"),
    ("Auto white balance", "auto_wb"),
    ("White balance", "white_balance"),
    ("Gain", "gain"),
]
# The capture tool always requests these auto controls off.
AUTO_CONTROL_KEYS = ("autofocus", "auto_exposure", "auto_wb")
AUTO_REQUESTED_TEXT = "off (always requested)"

# Metrics compared across sessions/devices: (csv column, label, digits).
COMPARE_METRICS = [
    ("brightness_range_pct", "Brightness range (% of mean)", 2),
    ("sharpness_cv", "Sharpness variation (cv)", 4),
    ("temporal_noise_mean", "Temporal noise, mean (gray levels)", 3),
    ("residual_aligned_mean", "Frame-to-frame residual, aligned (mean)", 4),
    ("residual_unaligned_mean", "Frame-to-frame residual, unaligned (mean)", 4),
    ("shift_std_dx", "Shift spread dx (px)", 2),
    ("shift_std_dy", "Shift spread dy (px)", 2),
]

CSV_FIELDS = [
    "session_name", "session_dir", "mode", "device", "frames", "frames_used", "skip_first",
    "width", "height",
    "measured_fps",
    "brightness_range_pct", "brightness_slope_per_frame",
    "sharpness_cv",
    "temporal_noise_mean", "temporal_noise_p95", "temporal_noise_p99",
    "residual_aligned_mean", "residual_aligned_p95", "residual_aligned_max",
    "shift_max_abs_dx", "shift_max_abs_dy", "shift_std_dx", "shift_std_dy",
    "residual_unaligned_mean", "residual_unaligned_p95", "residual_unaligned_max",
    "flagged_frames", "flagged_frame_indices",
    "excl_flagged_brightness_range_pct",
    "excl_flagged_temporal_noise_mean", "excl_flagged_temporal_noise_p95",
    "excl_flagged_residual_unaligned_mean", "excl_flagged_residual_unaligned_p95",
    "excl_flagged_residual_unaligned_max",
    "autofocus_applied_ok", "warnings_count",
]

LIMITATIONS = [
    "It does not detect defects, and it does not choose t_low or t_img. Those need "
    "labelled good AND bad parts and the model; use tools/thresholds.py for that.",
    "One camera cannot test 4-camera USB bandwidth or inter-camera timing.",
    "Results hold only for the lighting, part and mount used in these sessions.",
    "Judging whether the repeatability is good enough needs a requirement from Quality "
    "(smallest defect size and contrast): [Quality to define].",
]


class ReportInputError(ValueError):
    """A session folder is missing files or its metadata cannot be used."""


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ToolDependencyError(
            "tools/report.py needs numpy, which is not installed") from exc
    return np


def _analysis():
    _numpy()
    from tools import analysis
    return analysis


# ---------------------------------------------------------------- reading sessions

def _read_session_json(session_dir):
    path = session_dir / "session.json"
    if not path.is_file():
        raise ReportInputError(f"missing file: {path}")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportInputError(f"cannot read {path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise ReportInputError(f"{path} must contain a JSON object")
    if meta.get("mode") not in MODES:
        raise ReportInputError(f"{path}: mode must be one of {MODES}; got {meta.get('mode')!r}")
    return meta


def _device_list(session_dir, meta):
    devices = meta.get("devices") or []
    if not devices:
        raise ReportInputError(f"{session_dir / 'session.json'} lists no devices")
    devices = [int(d) for d in devices]
    missing = [session_dir / f"frames_dev{d}.npz" for d in devices
               if not (session_dir / f"frames_dev{d}.npz").is_file()]
    if missing:
        raise ReportInputError(
            "missing file(s): " + ", ".join(str(p) for p in missing))
    return devices


def _requested_text(requested, key):
    if key in AUTO_CONTROL_KEYS:
        return AUTO_REQUESTED_TEXT
    value = (requested.get("manual") or {}).get(key)
    return "not requested" if value is None else str(value)


def _camera_info(meta, dev):
    cam = (meta.get("cameras") or {}).get(str(dev)) or {}
    return {
        "actual_width": cam.get("actual_width"),
        "actual_height": cam.get("actual_height"),
        "actual_fps": cam.get("actual_fps"),
        "readback": cam.get("readback") or {},
        "applied_ok": cam.get("applied_ok") or {},
        "warnings": list(cam.get("warnings") or []),
        "requested": {key: _requested_text(meta.get("requested") or {}, key)
                      for _, key in CONTROLS},
    }


def _noise_metrics(analysis, stack):
    """Metrics that stationary variants share: brightness, temporal noise, unaligned residual."""
    return {
        "brightness": analysis.brightness_stats(stack),
        "temporal_noise": analysis.temporal_noise(stack),
        # No alignment for a stationary part (see NOISE_FLOOR_BASIS).
        "residual_unaligned": analysis.residual_scores(stack, align=False),
    }


def flag_outliers(means, sigma, floor):
    """Flag frames whose mean brightness is far from the median of the series.

    Returns (positions, median, mad, limit); positions index into means.
    A frame is flagged when |mean - median| > max(sigma * 1.4826 * MAD, floor).
    """
    np = _numpy()
    m = np.asarray(means, dtype=np.float64)
    med = float(np.median(m))
    mad = float(np.median(np.abs(m - med)))
    limit = max(float(sigma) * MAD_SCALE * mad, float(floor))
    flagged = [i for i, v in enumerate(m.tolist()) if abs(v - med) > limit]
    return flagged, med, mad, limit


def _shift_estimate(analysis, np, stack, min_response):
    """Apparent drift from phase correlation; reported only if the median peak response is high."""
    n = int(stack.shape[0])
    base = {"frames": n, "min_response": float(min_response), "median_response": None,
            "estimate_ok": False, "reported": False}
    if n < 2:
        return base
    try:
        shifts = analysis.estimate_shifts(stack, ref_index=0)
    except (ValueError, ArithmeticError):
        return base
    dx = np.array([s["dx"] for s in shifts], dtype=np.float64)
    dy = np.array([s["dy"] for s in shifts], dtype=np.float64)
    resp = np.array([s["response"] for s in shifts], dtype=np.float64)
    median_response = float(np.median(resp))
    return {
        "frames": n,
        "min_response": float(min_response),
        "median_response": median_response,
        "estimate_ok": True,
        "reported": median_response >= float(min_response),
        "max_abs_dx": float(np.max(np.abs(dx))),
        "max_abs_dy": float(np.max(np.abs(dy))),
        "std_dx": float(np.std(dx)),
        "std_dy": float(np.std(dy)),
    }


def _stationary_entry(analysis, np, gray, opts):
    total = int(gray.shape[0])
    skip = int(opts["skip_first"])
    if total - skip < 2:
        raise ReportInputError(
            f"--skip-first {skip} leaves {max(total - skip, 0)} of {total} frame(s); "
            "at least 2 are needed for stationary metrics")
    used = gray[skip:]
    noise = _noise_metrics(analysis, used)
    headline = {"sharpness": analysis.sharpness_stats(used), **noise}
    means = headline["brightness"]["mean_per_frame"]
    positions, median, mad, limit = flag_outliers(
        means, opts["outlier_sigma"], opts["outlier_floor_gray"])
    flagged = [{"index": skip + p, "mean": float(means[p])} for p in positions]
    clean = None
    excluding = None
    if flagged:
        flagged_set = set(positions)
        keep = [p for p in range(len(means)) if p not in flagged_set]
        clean = used[np.asarray(keep, dtype=np.intp)]
        if clean.shape[0] >= 2:
            excluding = {"frames": int(clean.shape[0]), **_noise_metrics(analysis, clean)}
    # Apparent drift is estimated on the frames left after removing flagged outliers.
    shift = _shift_estimate(analysis, np, used if clean is None else clean,
                            opts["min_shift_response"])
    return {
        "frames_used": int(used.shape[0]),
        "skip_first": skip,
        **headline,
        "outliers": {
            "sigma": float(opts["outlier_sigma"]),
            "floor_gray": float(opts["outlier_floor_gray"]),
            "median_brightness": median,
            "mad": mad,
            "limit": limit,
            "flagged": flagged,
        },
        "excluding_flagged": excluding,
        "shift_estimate": shift,
    }


def _summarize_device(analysis, np, session_dir, meta, mode, dev, opts):
    try:
        gray, stamps = analysis.load_session_gray(session_dir, dev)
    except (OSError, ValueError) as exc:
        raise ReportInputError(f"cannot read frames_dev{dev}.npz in {session_dir}: {exc}") from exc
    frames = int(gray.shape[0])
    duration = float(stamps[-1] - stamps[0]) if frames > 1 else 0.0
    entry = {
        "device": dev,
        "frames": frames,
        "width": int(gray.shape[2]),
        "height": int(gray.shape[1]),
        "duration_s": duration,
        "measured_fps": (frames - 1) / duration if duration > 0 else None,
        "camera": _camera_info(meta, dev),
    }
    if mode == "stationary":
        entry.update(_stationary_entry(analysis, np, gray, opts))
    else:
        entry["frames_used"] = frames
        entry["skip_first"] = 0
        entry["brightness"] = analysis.brightness_stats(gray)
        entry["sharpness"] = analysis.sharpness_stats(gray)
        shifts = analysis.estimate_shifts(gray, ref_index=0)
        dx = np.array([s["dx"] for s in shifts], dtype=np.float64)
        dy = np.array([s["dy"] for s in shifts], dtype=np.float64)
        entry["shifts"] = {
            "max_abs_dx": float(np.max(np.abs(dx))),
            "max_abs_dy": float(np.max(np.abs(dy))),
            "std_dx": float(np.std(dx)),
            "std_dy": float(np.std(dy)),
            "per_frame": shifts,
        }
        entry["residual_aligned"] = analysis.residual_scores(gray, align=True)
        entry["residual_unaligned"] = analysis.residual_scores(gray, align=False)
    return entry


def summarize_session(session_dir, settings=None):
    """Return a JSON-ready dict with all measurements for one session folder."""
    analysis = _analysis()
    np = _numpy()
    opts = dict(DEFAULT_SETTINGS)
    opts.update(settings or {})
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise ReportInputError(f"session folder does not exist: {session_dir}")
    meta = _read_session_json(session_dir)
    devices = _device_list(session_dir, meta)
    mode = meta["mode"]
    try:
        per_device = {str(dev): _summarize_device(analysis, np, session_dir, meta, mode, dev, opts)
                      for dev in devices}
    except ReportInputError:
        raise
    except ValueError as exc:
        raise ReportInputError(f"cannot analyse session {session_dir}: {exc}") from exc
    return {
        "session_dir": str(session_dir),
        "session_name": str(meta.get("session_name") or session_dir.name),
        "created_utc": meta.get("created_utc"),
        "mode": mode,
        "skip_first": int(opts["skip_first"]) if mode == "stationary" else 0,
        "devices": devices,
        "notes": dict(meta.get("notes") or {}),
        "requested": meta.get("requested") or {},
        "per_device": per_device,
    }


# ---------------------------------------------------------------- building the report

def _plain(obj):
    """Convert numpy and non-finite values into JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "tolist"):
        return _plain(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _csv_row(summary, dev, e):
    b = e["brightness"]
    sh = e["sharpness"]
    tn = e.get("temporal_noise") or {}
    ra = e.get("residual_aligned") or {}
    ru = e.get("residual_unaligned") or {}
    shifts = e.get("shifts") or {}
    cam = e["camera"]
    outliers = e.get("outliers")
    flagged = outliers["flagged"] if outliers else None
    ex = e.get("excluding_flagged") or {}
    ex_b = ex.get("brightness") or {}
    ex_tn = ex.get("temporal_noise") or {}
    ex_ru = ex.get("residual_unaligned") or {}
    return {
        "session_name": summary["session_name"],
        "session_dir": summary["session_dir"],
        "mode": summary["mode"],
        "device": dev,
        "frames": e["frames"],
        "frames_used": e["frames_used"],
        "skip_first": e["skip_first"],
        "width": e["width"],
        "height": e["height"],
        "measured_fps": e["measured_fps"],
        "brightness_range_pct": b["range_pct"],
        "brightness_slope_per_frame": b["slope_per_frame"],
        "sharpness_cv": sh["cv"],
        "temporal_noise_mean": tn.get("pixel_std_mean"),
        "temporal_noise_p95": tn.get("pixel_std_p95"),
        "temporal_noise_p99": tn.get("pixel_std_p99"),
        "residual_aligned_mean": ra.get("mean"),
        "residual_aligned_p95": ra.get("p95"),
        "residual_aligned_max": ra.get("max"),
        "shift_max_abs_dx": shifts.get("max_abs_dx"),
        "shift_max_abs_dy": shifts.get("max_abs_dy"),
        "shift_std_dx": shifts.get("std_dx"),
        "shift_std_dy": shifts.get("std_dy"),
        "residual_unaligned_mean": ru.get("mean"),
        "residual_unaligned_p95": ru.get("p95"),
        "residual_unaligned_max": ru.get("max"),
        "flagged_frames": None if flagged is None else len(flagged),
        "flagged_frame_indices": None if flagged is None else
        ";".join(str(f["index"]) for f in flagged),
        "excl_flagged_brightness_range_pct": ex_b.get("range_pct"),
        "excl_flagged_temporal_noise_mean": ex_tn.get("pixel_std_mean"),
        "excl_flagged_temporal_noise_p95": ex_tn.get("pixel_std_p95"),
        "excl_flagged_residual_unaligned_mean": ex_ru.get("mean"),
        "excl_flagged_residual_unaligned_p95": ex_ru.get("p95"),
        "excl_flagged_residual_unaligned_max": ex_ru.get("max"),
        "autofocus_applied_ok": cam["applied_ok"].get("autofocus"),
        "warnings_count": len(cam["warnings"]),
    }


def _rows(summaries):
    rows = []
    for s in summaries:
        for dev in s["devices"]:
            rows.append(_csv_row(s, dev, s["per_device"][str(dev)]))
    return rows


def _comparison(rows):
    if len(rows) < 2:
        return None
    label = lambda r: f"{r['session_name']} (device {r['device']})"
    metrics = {}
    for col, text, digits in COMPARE_METRICS:
        present = [r for r in rows if r[col] is not None]
        if len(present) < 2:
            continue
        hi = max(present, key=lambda r: r[col])
        lo = min(present, key=lambda r: r[col])
        if hi[col] == lo[col]:
            sentence = f"{text}: {label(hi)} and {label(lo)} are equal."
        else:
            sentence = (f"{text}: {label(hi)} is higher than {label(lo)} "
                        f"({hi[col]:.{digits}f} vs {lo[col]:.{digits}f}).")
        metrics[col] = {"label": text, "highest": label(hi), "lowest": label(lo),
                        "sentence": sentence}
    return {"metrics": metrics}


def _noise_floor_row(session, dev, variant, frames, residual, drift):
    return {
        "session_name": session["session_name"],
        "device": dev,
        "part": (session["notes"] or {}).get("part") or "not recorded",
        "variant": variant,
        "frames": frames,
        "mean": residual["mean"],
        "p95": residual["p95"],
        "max": residual["max"],
        "drift_std_dx": drift.get("std_dx"),
        "drift_std_dy": drift.get("std_dy"),
        "drift_max_abs_dx": drift.get("max_abs_dx"),
        "drift_max_abs_dy": drift.get("max_abs_dy"),
    }


def _noise_floor(summaries):
    rows = []
    for s in summaries:
        if s["mode"] != "stationary":
            continue
        for dev in s["devices"]:
            e = s["per_device"][str(dev)]
            sh = e.get("shift_estimate") or {}
            # Drift is shown only when the estimate is reported (see _shift_estimate).
            drift = sh if sh.get("reported") else {}
            rows.append(_noise_floor_row(s, dev, "all_frames", e["frames_used"],
                                         e["residual_unaligned"], drift))
            ex = e.get("excluding_flagged")
            if ex:
                rows.append(_noise_floor_row(s, dev, "excluding_flagged", ex["frames"],
                                             ex["residual_unaligned"], {}))
    return {
        "basis": NOISE_FLOOR_BASIS,
        "observed_not_threshold": True,
        "rows": rows,
    }


def build_report(summaries, settings=None):
    rows = _rows(summaries)
    opts = dict(DEFAULT_SETTINGS)
    opts.update(settings or {})
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settings": opts,
        "outlier_rule": {"formula": OUTLIER_FORMULA, "meaning": OUTLIER_RULE_TEXT},
        "sessions": summaries,
        "rows": rows,
        "comparison": _comparison(rows),
        "noise_floor": _noise_floor(summaries),
        "limitations": list(LIMITATIONS),
    }


# ---------------------------------------------------------------- formatting

def _f(value, digits=2):
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v) or math.isinf(v):
        return "n/a"
    return f"{v:.{digits}f}"


def _text(value):
    return "n/a" if value is None else str(value)


def _applied_text(value):
    if value is True:
        return "yes"
    if value is False:
        return "not confirmed"
    return "not reported"


def _not_confirmed(cam, key):
    """True when a control was requested but the camera did not confirm it."""
    if cam["applied_ok"].get(key) is False:
        return True
    if key in AUTO_CONTROL_KEYS:
        readback = cam["readback"].get(key)
        return readback is not None and readback not in (0, False)
    return False


def _not_confirmed_text(label, key, requested_text, readback):
    rb = _text(readback)
    if key in AUTO_CONTROL_KEYS:
        example = " (for example a fixed-focus lens)" if key == "autofocus" else ""
        return (f"{label} was requested off but the camera did not confirm it (read back: {rb}). "
                f"On Windows -1 often means the camera does not expose this control{example}, "
                "so this does not prove auto is on.")
    if requested_text == "not requested":
        return f"{label}: the camera did not confirm the setting (read back: {rb})."
    return (f"{label} was requested as {requested_text} but the camera did not confirm it "
            f"(read back: {rb}).")


MODE_TEXT = {
    "stationary": "stationary (part and camera held still; measures drift, sharpness and noise)",
    "placement": "placement (part placed again between frames; measures position change and image noise)",
}


def _md_measured(summaries):
    lines = ["## 1. What was measured", ""]
    for s in summaries:
        lines += [f"### Session `{s['session_name']}`", ""]
        lines.append(f"- Folder: `{s['session_dir']}`")
        lines.append(f"- Created (UTC): {_text(s['created_utc'])}")
        lines.append(f"- Mode: {MODE_TEXT[s['mode']]}")
        lines.append(f"- Devices: {', '.join(str(d) for d in s['devices'])}")
        skip = s["skip_first"]
        if s["mode"] == "stationary":
            lines.append(
                f"- Frames skipped at the start (--skip-first): {skip}"
                + (" (these frames are in no stationary metric)." if skip
                   else " (none; all frames are used)."))
        for dev in s["devices"]:
            e = s["per_device"][str(dev)]
            used = (f" ({e['frames_used']} used after skipping the first {skip})"
                    if s["mode"] == "stationary" and skip else "")
            lines.append(
                f"- Device {dev}: {e['frames']} frames{used}, {e['width']} x {e['height']} px, "
                f"measured {_f(e['measured_fps'], 1)} fps over {_f(e['duration_s'], 1)} s")
        notes = s["notes"]
        for key, label in (("part", "Part"), ("lighting", "Lighting"),
                           ("operator", "Operator"), ("free", "Notes")):
            lines.append(f"- {label}: {notes.get(key) or 'not recorded'}")
        lines.append("")
    return lines


def _md_camera(summaries):
    lines = ["## 2. Camera settings", ""]
    lines.append("Requested values: the capture tool always requests the auto controls off "
                 "(autofocus, auto exposure, auto white balance). Focus, exposure, white balance "
                 "and gain show the manual value from session.json when one was set.")
    lines.append("")
    unconfirmed = []
    notes = []
    for s in summaries:
        for dev in s["devices"]:
            cam = s["per_device"][str(dev)]["camera"]
            for w in cam["warnings"]:
                notes.append(f"- Session `{s['session_name']}`, device {dev}, reported by the "
                             f"session: \"{w}\"")
            for label, key in CONTROLS:
                if _not_confirmed(cam, key):
                    unconfirmed.append(
                        f"- Session `{s['session_name']}`, device {dev}: "
                        + _not_confirmed_text(label, key, cam["requested"][key],
                                              cam["readback"].get(key)))
    if unconfirmed or notes:
        lines += ["> **WARNING: the camera did not confirm some settings.** Auto settings that "
                  "stay on change brightness, focus and noise during a run, so the numbers "
                  "below may not describe the settings that were requested.", ""]
        lines += unconfirmed + notes
        lines.append("")
    else:
        lines += ["No camera control was reported as ignored.", ""]
    for s in summaries:
        for dev in s["devices"]:
            cam = s["per_device"][str(dev)]["camera"]
            lines += [f"### Session `{s['session_name']}`, device {dev}", "",
                      "| Control | Requested | Read back | Applied OK |",
                      "|---|---|---|---|"]
            for label, key in CONTROLS:
                lines.append(f"| {label} | {cam['requested'][key]} | "
                             f"{_text(cam['readback'].get(key))} | "
                             f"{_applied_text(cam['applied_ok'].get(key))} |")
            lines.append("")
    return lines


def _outlier_lines(e):
    o = e["outliers"]
    lines = [
        f"- Outlier frames (mean brightness): {OUTLIER_FORMULA}, K = {_f(o['sigma'], 1)}, "
        f"F = {_f(o['floor_gray'], 1)} gray levels; median {_f(o['median_brightness'], 2)}, "
        f"MAD {_f(o['mad'], 2)}, limit {_f(o['limit'], 2)} gray levels. Meaning: "
        f"{OUTLIER_RULE_TEXT}."]
    if o["flagged"]:
        items = "; ".join(f"frame {f['index']} (mean brightness {_f(f['mean'], 2)})"
                          for f in o["flagged"])
        lines.append(f"- Flagged: {items}.")
    else:
        lines.append("- Flagged: none.")
    return lines


def _drift_line(e):
    sh = e.get("shift_estimate") or {}
    resp = _f(sh.get("median_response"), 3)
    if sh.get("reported"):
        return [
            "- Apparent drift/vibration (estimate; itself noisy on a flat or weak-texture part, "
            "so it can reflect sensor noise): shift spread std dx "
            f"{_f(sh['std_dx'], 2)} px, dy {_f(sh['std_dy'], 2)} px; largest single-frame shift "
            f"|dx| {_f(sh['max_abs_dx'], 2)} px, |dy| {_f(sh['max_abs_dy'], 2)} px. Estimated on "
            f"{sh['frames']} frame(s) with flagged frames removed; median peak response {resp}."]
    if not sh.get("estimate_ok"):
        return ["- Apparent drift/vibration not reported: the shift estimate could not be "
                "computed for this session."]
    return [f"- Apparent drift/vibration not reported: image too flat or too noisy to measure "
            f"shift (median peak response {resp}, needs at least "
            f"{_f(sh.get('min_response'), 2)})."]


def _sensitivity_lines(e):
    n = len(e["outliers"]["flagged"])
    if not n:
        return []
    ex = e.get("excluding_flagged")
    if ex is None:
        return ["", "Excluding flagged frames leaves fewer than 2 frames, so no sensitivity view "
                "is shown.", ""]
    b, tn, ru = e["brightness"], e["temporal_noise"], e["residual_unaligned"]
    xb, xtn, xru = ex["brightness"], ex["temporal_noise"], ex["residual_unaligned"]
    rows = [
        ("Frames", str(e["frames_used"]), str(ex["frames"])),
        ("Brightness range (% of mean)", _f(b["range_pct"]), _f(xb["range_pct"])),
        ("Temporal noise mean (gray levels)", _f(tn["pixel_std_mean"], 3),
         _f(xtn["pixel_std_mean"], 3)),
        ("Temporal noise p95 (gray levels)", _f(tn["pixel_std_p95"], 3),
         _f(xtn["pixel_std_p95"], 3)),
        ("Residual, no alignment: mean", _f(ru["mean"], 4), _f(xru["mean"], 4)),
        ("Residual, no alignment: p95", _f(ru["p95"], 4), _f(xru["p95"], 4)),
        ("Residual, no alignment: max", _f(ru["max"], 4), _f(xru["max"], 4)),
    ]
    lines = ["", "**Sensitivity view, not the headline:** all frames versus excluding the "
             "flagged frames.", "",
             "| Metric | All frames (headline) | Excluding flagged frames (sensitivity) |",
             "|---|---|---|"]
    lines += [f"| {m} | {a} | {c} |" for m, a, c in rows]
    verb = "changes" if n == 1 else "change"
    lines += ["", f"{n} flagged frame{'' if n == 1 else 's'} {verb} these numbers: check whether "
              "the camera had settled before the first frames.", ""]
    return lines


def _md_stationary(summaries):
    lines = ["## 3. Stationary results", ""]
    for s in summaries:
        if s["mode"] != "stationary":
            continue
        for dev in s["devices"]:
            e = s["per_device"][str(dev)]
            b, sh = e["brightness"], e["sharpness"]
            tn, ru = e["temporal_noise"], e["residual_unaligned"]
            n_used = e["frames_used"]
            total = b["slope_per_frame"] * (n_used - 1)
            lines += [f"### Session `{s['session_name']}`, device {dev}", ""]
            skip_note = (f" (the first {s['skip_first']} skipped by --skip-first, in no metric "
                         "below)" if s["skip_first"] else "")
            lines.append(f"- Frames used: {n_used} of {e['frames']}{skip_note}.")
            lines.append(
                f"- Brightness drift: range {_f(b['range_pct'])} % of mean; slope "
                f"{_f(b['slope_per_frame'], 4)} gray levels per frame ({_f(total)} over the run). "
                "Meaning: how far average brightness moved during the run. A small range and "
                "slope mean steadier light.")
            lines.append(
                f"- Sharpness variation: cv {_f(sh['cv'], 4)} (mean Laplacian variance "
                f"{_f(sh['mean'], 1)}). Meaning: how much sharpness changed from frame to frame "
                "(std divided by mean). Lower means steadier focus.")
            lines.append(
                f"- Temporal noise: per-pixel std across frames, mean {_f(tn['pixel_std_mean'], 3)}, "
                f"p95 {_f(tn['pixel_std_p95'], 3)}, p99 {_f(tn['pixel_std_p99'], 3)} gray levels. "
                "Meaning: a pixel of a still part jitters about this much between frames; "
                "p95 and p99 show the noisier pixels.")
            lines.append(
                f"- Frame-to-frame residual, no alignment: mean {_f(ru['mean'], 4)}, "
                f"p95 {_f(ru['p95'], 4)}, max {_f(ru['max'], 4)} (0 to 1 scale). Meaning: how far "
                "a typical frame differs from the median frame of the series. Nothing is aligned, "
                "so any real movement of the part or mount is included.")
            lines += _outlier_lines(e)
            lines += _drift_line(e)
            lines += _sensitivity_lines(e)
            lines.append("")
    if len(lines) == 2:
        lines += ["No stationary session was given.", ""]
    return lines


def _md_placement(summaries):
    lines = ["## 4. Placement results", ""]
    for s in summaries:
        if s["mode"] != "placement":
            continue
        for dev in s["devices"]:
            e = s["per_device"][str(dev)]
            sh = e["shifts"]
            ra, ru = e["residual_aligned"], e["residual_unaligned"]
            lines += [f"### Session `{s['session_name']}`, device {dev}", ""]
            lines.append(
                f"- Position change: shift spread (std) dx {_f(sh['std_dx'], 2)} px, "
                f"dy {_f(sh['std_dy'], 2)} px; largest single-frame shift |dx| "
                f"{_f(sh['max_abs_dx'], 2)} px, |dy| {_f(sh['max_abs_dy'], 2)} px. Meaning: how far "
                "the part position varied between frames. The spread is the typical variation; "
                "the largest shift is the worst frame.")
            lines.append(
                f"- Residual with alignment: mean {_f(ra['mean'], 4)}, p95 {_f(ra['p95'], 4)}, "
                f"max {_f(ra['max'], 4)}. Meaning: image difference left after moving each frame "
                "back onto the reference by whole pixels. This is mostly image noise and surface "
                "change, not position.")
            lines.append(
                f"- Residual without alignment: mean {_f(ru['mean'], 4)}, p95 {_f(ru['p95'], 4)}, "
                f"max {_f(ru['max'], 4)}. Meaning: image difference including the position change. "
                "The gap to the aligned residual is mostly position change.")
            lines.append("")
    if len(lines) == 2:
        lines += ["No placement session was given.", ""]
    return lines


VARIANT_TEXT = {
    "all_frames": "all frames (headline)",
    "excluding_flagged": "excluding flagged frames (sensitivity)",
}


def _md_noise_floor(report):
    nf = report["noise_floor"]
    lines = ["## 5. Noise floor for threshold work", ""]
    if not nf["rows"]:
        lines += ["No stationary session was given, so no noise floor was measured. Run a "
                  "stationary session on a good, unchanged part to obtain one.", ""]
        return lines
    lines += [
        f"Basis: {nf['basis']}, assumed to hold a good, unchanged part.",
        "",
        "Alignment is NOT applied. Aligning each frame to a single noisy frame can fit sensor "
        "noise, and it would hide real mount vibration, which this repeatability noise floor "
        "should include. A defect score must clearly exceed this distribution to be "
        "distinguishable from noise.",
        "",
        "**Observed noise floor, not a threshold.** The all-frames row is the headline. Where "
        "frames were flagged, the excluding-flagged row beside it is a sensitivity view only.",
        "",
        "| Session | Device | Part | Frames used | Variant | Residual mean | Residual p95 | Residual max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in nf["rows"]:
        lines.append(f"| `{r['session_name']}` | {r['device']} | {r['part']} | {r['frames']} | "
                     f"{VARIANT_TEXT[r['variant']]} | {_f(r['mean'], 4)} | {_f(r['p95'], 4)} | "
                     f"{_f(r['max'], 4)} |")
    lines.append("")
    for r in nf["rows"]:
        if r.get("drift_std_dx") is None:
            continue
        lines.append(
            f"- `{r['session_name']}`, device {r['device']}: apparent drift/vibration (estimate; "
            "itself noisy on a flat or weak-texture part, so it can reflect sensor noise): shift "
            f"spread std dx {_f(r['drift_std_dx'], 2)} px, dy {_f(r['drift_std_dy'], 2)} px; "
            f"largest single-frame shift |dx| {_f(r['drift_max_abs_dx'], 2)} px, "
            f"|dy| {_f(r['drift_max_abs_dy'], 2)} px.")
    if any(r.get("drift_std_dx") is not None for r in nf["rows"]):
        lines.append("")
    return lines


def _md_comparison(report):
    comp = report["comparison"]
    if comp is None:
        return []
    lines = ["## Comparison across sessions and devices", ""]
    header = ["Session", "Device", "Mode", "Brightness range %", "Sharpness cv",
              "Temporal noise mean", "Residual aligned mean", "Residual unaligned mean",
              "Shift std dx px", "Shift std dy px"]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in report["rows"]:
        lines.append("| " + " | ".join([
            f"`{r['session_name']}`", str(r["device"]), r["mode"],
            _f(r["brightness_range_pct"], 2), _f(r["sharpness_cv"], 4),
            _f(r["temporal_noise_mean"], 3), _f(r["residual_aligned_mean"], 4),
            _f(r["residual_unaligned_mean"], 4),
            _f(r["shift_std_dx"], 2), _f(r["shift_std_dy"], 2)]) + " |")
    lines.append("")
    for item in comp["metrics"].values():
        lines.append(f"- {item['sentence']}")
    lines += ["", "Differences describe these sessions only.", ""]
    return lines


def _md_limitations():
    lines = ["## 6. What this test CANNOT tell you", ""]
    lines += [f"- {text}" for text in LIMITATIONS]
    lines.append("")
    return lines


def render_markdown(report):
    summaries = report["sessions"]
    s_opts = report["settings"]
    lines = [
        "# Repeatability report", "",
        f"Generated (UTC): {report['generated_utc']}  ",
        f"Sessions: {len(summaries)}. Session-device rows: {len(report['rows'])}.",
        "",
        "This report gives measurements and what they mean. It gives no verdict and no "
        "acceptance limit.",
        "",
        f"Settings used: frame-outlier spotting rule with K = {_f(s_opts['outlier_sigma'], 1)} "
        f"and F = {_f(s_opts['outlier_floor_gray'], 1)} gray levels ({OUTLIER_RULE_TEXT}); "
        f"frames skipped at the start = {s_opts['skip_first']} (0 means nothing is excluded); "
        "apparent drift is reported only when the median peak response is at least "
        f"{_f(s_opts['min_shift_response'], 2)}.",
        "",
    ]
    lines += _md_measured(summaries)
    lines += _md_camera(summaries)
    lines += _md_stationary(summaries)
    lines += _md_placement(summaries)
    lines += _md_noise_floor(report)
    lines += _md_comparison(report)
    lines += _md_limitations()
    return "\n".join(lines) + "\n"


def write_report(session_dirs, out_dir, settings=None):
    """Summarize the sessions and write report.json, report.md and summary.csv to out_dir."""
    opts = dict(DEFAULT_SETTINGS)
    opts.update(settings or {})
    summaries = [summarize_session(d, opts) for d in session_dirs]
    report = _plain(build_report(summaries, opts))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow(row)
    return report


def _nonneg_float(text):
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number; got {text!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more; got {text!r}")
    return value


def _nonneg_int(text):
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a whole number; got {text!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more; got {text!r}")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m tools.report",
        description="Write a plain-language repeatability report for capture session folders.")
    parser.add_argument("session_dirs", nargs="+", metavar="SESSION_DIR")
    parser.add_argument("--out", metavar="REPORT_DIR", default=None,
                        help="output folder (default: a 'report' folder inside the first session)")
    parser.add_argument("--outlier-sigma", type=_nonneg_float, default=DEFAULT_OUTLIER_SIGMA,
                        metavar="K",
                        help="spotting rule for frame outliers: K times the robust sigma "
                             f"(default {DEFAULT_OUTLIER_SIGMA:g}). Not a quality threshold.")
    parser.add_argument("--outlier-floor-gray", type=_nonneg_float,
                        default=DEFAULT_OUTLIER_FLOOR_GRAY, metavar="F",
                        help="minimum brightness gap in gray levels for the outlier rule "
                             f"(default {DEFAULT_OUTLIER_FLOOR_GRAY:g}). Not a quality threshold.")
    parser.add_argument("--skip-first", type=_nonneg_int, default=DEFAULT_SKIP_FIRST, metavar="N",
                        help="exclude the first N frames from all stationary metrics "
                             "(default 0: nothing excluded)")
    parser.add_argument("--min-shift-response", type=_nonneg_float,
                        default=DEFAULT_MIN_SHIFT_RESPONSE, metavar="R",
                        help="below this median peak response the image is too flat, dark or "
                             f"noisy for shift estimates (default {DEFAULT_MIN_SHIFT_RESPONSE:g})")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    settings = {
        "skip_first": args.skip_first,
        "outlier_sigma": args.outlier_sigma,
        "outlier_floor_gray": args.outlier_floor_gray,
        "min_shift_response": args.min_shift_response,
    }
    session_dirs = [Path(p) for p in args.session_dirs]
    out_dir = Path(args.out) if args.out else session_dirs[0] / "report"
    try:
        write_report(session_dirs, out_dir, settings)
    except ToolDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except ReportInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {out_dir / 'report.md'}, {out_dir / 'report.json'} and {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
