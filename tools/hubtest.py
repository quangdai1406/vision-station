"""USB hub feasibility test for several webcams. Run on the user's Windows PC.

    py -m tools.hubtest --devices 0 1 2 3 [--width 1920 --height 1080] [--rounds 5]
        [--warmup 5] [--backend default|dshow] [--out hubtest_report.json]

Question: do the cameras behind one USB hub deliver pictures when they are used one
after another, and does simultaneous capture work on this hub? Three modes, each run
for --rounds rounds. The camera handling imitates station/cameras.py (same backend,
same CAP_PROP_BUFFERSIZE, same warm-up after open, same stale-frame flush before each
picture):

  sequential_open_per_shot  (Test A1) per device, in order: open, warm up, flush, one
                            picture, release.
  sequential_keep_open      (Test A2) open all devices once, warm up each, then per device
                            in order, every round: flush, one picture.
  simultaneous              (Test B, all at once) open all, warm up, flush all, grab all,
                            then retrieve all: the way station/cameras.py captures. Expected
                            to fail on a hub; a pass is also fine and is then recommended
                            first. A failure is reported and the run keeps going.

Backend: the station calls cv2.VideoCapture(index) with no backend, so the default is
"default". "dshow" (cv2.CAP_DSHOW) is selectable; when it is chosen, one warning line says
it differs from the station. Run both backends if the first one fails.

Each camera is handled on its own: one failure never stops the others, and every failure
is recorded with its camera index. Warm-up and flush grabs are discarded; their failures
are not counted, the picture step reports them. A picture counts as OK when it is non-empty
and not all black. The 4 pictures of one side are taken one after another, so the part must
stay still on the fixture for the whole side.

Nothing is judged against a threshold here. The summary reports counts and times. The
recommendation follows only recommend(): the first mode in MODE_ORDER in which every
round and every camera succeeded.

Timings (seconds, from the monotonic clock):
  open_s         opening and setting the camera (not the warm-up).
  first_frame_s  open-per-shot: from opening the camera to its picture (includes warm-up).
                 keep-open: from the flush to the picture. simultaneous: from the start of
                 the grab-all step to this camera's picture.
  round_s        wall time of one round, i.e. one picture from each device. Round 1 of
                 keep-open includes opening and warm-up; the summary reports it separately
                 (round1_s) from the mean of rounds 2 and later (later_round_s_mean).
  Operator wait: keep-open uses rounds 2 and later; open-per-shot and simultaneous use the
                 full cycle (opening and warm-up included), mean of all rounds.

Exit codes: 0 when the run finished and the report was written (camera failures are data,
not an error); 1 when the report could not be written, including when it could not be turned
into JSON (ValueError or TypeError from json); 2 when cv2 or numpy is missing or an argument
is wrong. This tool installs nothing.

Sample pictures: the first picture of each camera from Test A1 round 1 is saved as PNG
next to --out (<stem>_cam<N>_first.png) so the operator can see which corner each index
shows. A failed save is skipped quietly.

JSON report, schema 1, written atomically to --out:
{
  "schema": 1,
  "tool": "tools.hubtest",
  "started_utc": "2026-10-25T09:00:00Z",
  "finished_utc": "2026-10-25T09:05:00Z",
  "devices": [0, 1, 2, 3],
  "requested": {"width": 1920, "height": 1080, "rounds": 5, "warmup": 5, "backend": "default"},
  "station_backend": "default",
  "modes": {
    "sequential_open_per_shot": {
      "title": "Test A1 - ...",
      "fatal_error": null,
      "rounds": [
        {"round": 1, "round_s": 4.2,
         "cameras": [{"device": 0, "open_s": 0.3, "first_frame_s": 1.1, "ok": true,
                      "mean_grey": 118.2, "width": 1920, "height": 1080,
                      "error": null, "release_error": null}]}
      ],
      "summary": {"pictures_ok": 20, "pictures_total": 20, "slowest_s": 1.5,
                  "fastest_s": 0.9, "round_s_mean": 4.1, "round1_s": 4.2,
                  "later_round_s_mean": 4.0}
    },
    "sequential_keep_open": {...same shape...},
    "simultaneous": {...same shape...}
  },
  "recommended_mode": "sequential_keep_open",
  "sample_pictures": {"0": "hubtest_report_cam0_first.png", "1": null, "2": null, "3": null}
}
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from station.cameras import OPEN_WARMUP_FRAMES, STALE_FLUSH_FRAMES
from tools import ToolDependencyError

SCHEMA_VERSION = 1
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_ROUNDS = 5
# Same value as the station (station/cameras.py OPEN_WARMUP_FRAMES): grabs read and
# discarded right after each open. Engineering setting, not a threshold.
DEFAULT_WARMUP = OPEN_WARMUP_FRAMES
# Same as the station (STALE_FLUSH_FRAMES): grabs dropped before each picture.
FLUSH_FRAMES = STALE_FLUSH_FRAMES
# The station opens cameras with cv2.VideoCapture(index) and no backend name
# (station/cameras.py OpenCvCameraSet.open and _open_for_shot).
STATION_BACKEND = "default"
# The station sets CAP_PROP_BUFFERSIZE to 1 after configuring a device, errors ignored
# (station/cameras.py _open_one).
STATION_BUFFER_FRAMES = 1
DEFAULT_OUT = "hubtest_report.json"
INSTALL_HINT = "py -m pip install opencv-python numpy"
NOT_INSTALLED_NOTE = "this tool installs nothing; install it yourself with: " + INSTALL_HINT
# The order in which the recommendation rule looks at the modes.
MODE_ORDER = ("simultaneous", "sequential_keep_open", "sequential_open_per_shot")
RUN_ORDER = ("sequential_open_per_shot", "sequential_keep_open", "simultaneous")
MODE_TITLES = {
    "sequential_open_per_shot": "Test A1 - sequential, open per picture",
    "sequential_keep_open": "Test A2 - sequential, all cameras kept open",
    "simultaneous": "Test B - simultaneous (the current station mode)",
}
MODE_NAMES = {
    "simultaneous": "simultaneous",
    "sequential_keep_open": "sequential keep-open",
    "sequential_open_per_shot": "sequential open-per-shot",
}
# What the operator waits for in one round, per mode (used in the wait line).
MODE_CYCLE = {
    "sequential_open_per_shot": "full per-shot cycle: open, warm-up, picture, release",
    "simultaneous": "full round: open, warm-up, all pictures",
}
QUALITY_NOTE = "decide with Quality if acceptable: [Quality to define]"
SIMULTANEOUS_NOTE = (
    "Test B (all at once, the current station mode) is expected to fail on a USB hub. "
    "A pass is also a result: then it is recommended first. A failure is reported; the run keeps going."
)
BACKEND_WARNING = (
    "warning: backend {backend} differs from the station ({station}); the station opens "
    "cameras with cv2.VideoCapture(index) and no backend. Run with --backend {station} too."
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ReportWriteError(Exception):
    """The report could not be turned into JSON, so nothing was written (exit 1)."""


def _import_numpy() -> Any:
    try:
        import numpy  # noqa: PLC0415 - lazy on purpose
    except ImportError as exc:
        raise ToolDependencyError(f"numpy is needed for the hub test; {NOT_INSTALLED_NOTE}") from exc
    return numpy


def _import_cv2() -> Any:
    try:
        import cv2  # noqa: PLC0415 - lazy on purpose
    except ImportError as exc:
        raise ToolDependencyError(f"OpenCV (cv2) is needed for the hub test; {NOT_INSTALLED_NOTE}") from exc
    return cv2


def _err_text(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def _new_cam(dev: int) -> dict:
    return {
        "device": dev,
        "open_s": None,
        "first_frame_s": None,
        "ok": False,
        "mean_grey": None,
        "width": None,
        "height": None,
        "error": None,
        "release_error": None,
    }


@dataclass
class _Plan:
    cv2: Any
    np: Any
    devices: tuple
    api: Any
    width: int
    height: int
    rounds: int
    warmup: int
    clock: Callable[[], float]
    out: Callable[[str], Any]


def _open_camera(cv2: Any, dev: int, api: Any, width: int, height: int,
                 clock: Callable[[], float], cam: dict) -> Optional[Any]:
    start = clock()
    cap = None
    try:
        cap = cv2.VideoCapture(dev) if api is None else cv2.VideoCapture(dev, api)
        if not cap.isOpened():
            raise RuntimeError("the camera did not open")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    except Exception as exc:
        cam["error"] = f"device {dev}: could not open camera ({_err_text(exc)})"
        if cap is not None:
            with contextlib.suppress(Exception):
                cap.release()
        cam["open_s"] = clock() - start
        return None
    # Best effort, as the station does: a short driver queue. Errors ignored.
    with contextlib.suppress(Exception):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, STATION_BUFFER_FRAMES)
    cam["open_s"] = clock() - start
    return cap


def _warm_up(cap: Any, n: int) -> None:
    """Read and discard n frames after open (grab and retrieve, as the station does).

    Best effort: failures here are not counted, the picture step reports them.
    """
    for _ in range(n):
        try:
            cap.grab()
            cap.retrieve()
        except Exception:
            pass


def _flush_stale_all(caps: Sequence[Any], n: int) -> None:
    """Drop n stale grabs per device before a picture, in the station's order (untimed).

    Failures are ignored here; the real grab reports them.
    """
    for _ in range(n):
        for cap in caps:
            try:
                cap.grab()
            except Exception:
                pass


def _grab_one(dev: int, cap: Any, cam: dict) -> bool:
    try:
        ok = cap.grab()
    except Exception as exc:
        cam["error"] = f"device {dev}: grab failed ({_err_text(exc)})"
        return False
    if not ok:
        cam["error"] = f"device {dev}: grab gave no frame"
        return False
    return True


def _retrieve_one(cv2: Any, np: Any, dev: int, cap: Any, cam: dict,
                  clock: Callable[[], float], start: float) -> Any:
    """Retrieve one picture and check it. Returns the image when it is OK, else None."""
    try:
        ok, image = cap.retrieve()
    except Exception as exc:
        cam["error"] = f"device {dev}: retrieve failed ({_err_text(exc)})"
        return None
    if not ok or image is None:
        cam["error"] = f"device {dev}: retrieve gave no frame"
        return None
    cam["first_frame_s"] = clock() - start
    try:
        arr = np.asarray(image)
        if arr.size == 0:
            cam["error"] = f"device {dev}: empty frame"
            return None
        cam["width"], cam["height"] = int(arr.shape[1]), int(arr.shape[0])
        gray = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        cam["mean_grey"] = float(np.mean(gray))
        black = not bool(np.any(arr))
    except Exception as exc:
        cam["error"] = f"device {dev}: could not read the picture ({_err_text(exc)})"
        return None
    if black:
        cam["error"] = f"device {dev}: all-black frame (mean grey {cam['mean_grey']:.1f})"
        return None
    cam["ok"] = True
    return arr


def _release(dev: int, cap: Any, cam: dict) -> None:
    try:
        cap.release()
    except Exception as exc:
        cam["release_error"] = f"device {dev}: release failed ({_err_text(exc)})"


def _report_camera(plan: _Plan, label: str, rnd: int, cam: dict) -> None:
    if cam["ok"]:
        plan.out(f"{label} round {rnd} camera {cam['device']}: OK, first picture "
                 f"{cam['first_frame_s']:.2f} s, mean grey {cam['mean_grey']:.1f}")
    else:
        plan.out(f"{label} round {rnd} camera {cam['device']}: FAILED: "
                 f"{cam['error'] or 'no picture'}")


def _run_open_per_shot(plan: _Plan) -> tuple[list, dict]:
    rounds: list[dict] = []
    samples: dict[int, Any] = {}
    for rnd in range(1, plan.rounds + 1):
        round_start = plan.clock()
        cams: list[dict] = []
        for dev in plan.devices:
            cam = _new_cam(dev)
            start = plan.clock()
            cap = _open_camera(plan.cv2, dev, plan.api, plan.width, plan.height, plan.clock, cam)
            if cap is not None:
                try:
                    _warm_up(cap, plan.warmup)
                    _flush_stale_all([cap], FLUSH_FRAMES)
                    if _grab_one(dev, cap, cam):
                        image = _retrieve_one(plan.cv2, plan.np, dev, cap, cam, plan.clock, start)
                        if image is not None and rnd == 1:
                            samples[dev] = image
                finally:
                    _release(dev, cap, cam)
            cams.append(cam)
            _report_camera(plan, "A1", rnd, cam)
        rounds.append({"round": rnd, "round_s": plan.clock() - round_start, "cameras": cams})
    return rounds, samples


def _run_keep_open(plan: _Plan) -> tuple[list, dict]:
    session_start = plan.clock()
    caps: dict[int, Any] = {}
    setup: dict[int, dict] = {}
    for dev in plan.devices:
        cam = _new_cam(dev)
        cap = _open_camera(plan.cv2, dev, plan.api, plan.width, plan.height, plan.clock, cam)
        if cap is not None:
            _warm_up(cap, plan.warmup)
            caps[dev] = cap
        setup[dev] = cam
    rounds: list[dict] = []
    try:
        for rnd in range(1, plan.rounds + 1):
            # Round 1 includes opening and warm-up: the operator waits for those too.
            round_start = session_start if rnd == 1 else plan.clock()
            cams: list[dict] = []
            for dev in plan.devices:
                cam = _new_cam(dev)
                if rnd == 1:
                    cam["open_s"] = setup[dev]["open_s"]
                    cam["error"] = setup[dev]["error"]
                cap = caps.get(dev)
                if cap is None:
                    cam["error"] = cam["error"] or f"device {dev}: not open (failed in round 1)"
                else:
                    _flush_stale_all([cap], FLUSH_FRAMES)
                    start = plan.clock()
                    if _grab_one(dev, cap, cam):
                        _retrieve_one(plan.cv2, plan.np, dev, cap, cam, plan.clock, start)
                cams.append(cam)
                _report_camera(plan, "A2", rnd, cam)
            rounds.append({"round": rnd, "round_s": plan.clock() - round_start, "cameras": cams})
    finally:
        last = {c["device"]: c for c in rounds[-1]["cameras"]} if rounds else {}
        for dev, cap in caps.items():
            _release(dev, cap, last.get(dev, _new_cam(dev)))
    return rounds, {}


def _run_simultaneous(plan: _Plan) -> tuple[list, dict]:
    rounds: list[dict] = []
    for rnd in range(1, plan.rounds + 1):
        round_start = plan.clock()
        cams = {dev: _new_cam(dev) for dev in plan.devices}
        caps: dict[int, Any] = {}
        try:
            for dev in plan.devices:
                cap = _open_camera(plan.cv2, dev, plan.api, plan.width, plan.height, plan.clock, cams[dev])
                if cap is not None:
                    _warm_up(cap, plan.warmup)
                    caps[dev] = cap
            _flush_stale_all(list(caps.values()), FLUSH_FRAMES)
            grab_start = plan.clock()
            grabbed: list[int] = []
            for dev in plan.devices:
                if dev in caps and _grab_one(dev, caps[dev], cams[dev]):
                    grabbed.append(dev)
            for dev in grabbed:
                _retrieve_one(plan.cv2, plan.np, dev, caps[dev], cams[dev], plan.clock, grab_start)
        finally:
            for dev, cap in caps.items():
                _release(dev, cap, cams[dev])
        round_cams = [cams[dev] for dev in plan.devices]
        for cam in round_cams:
            _report_camera(plan, "B", rnd, cam)
        rounds.append({"round": rnd, "round_s": plan.clock() - round_start, "cameras": round_cams})
    return rounds, {}


_RUNNERS = {
    "sequential_open_per_shot": _run_open_per_shot,
    "sequential_keep_open": _run_keep_open,
    "simultaneous": _run_simultaneous,
}


def _summarise(rounds: list) -> dict:
    cams = [c for r in rounds for c in r["cameras"]]
    ok = [c for c in cams if c["ok"]]
    times = [c["first_frame_s"] for c in ok if c["first_frame_s"] is not None]
    round_times = [r["round_s"] for r in rounds]
    later = round_times[1:]
    return {
        "pictures_ok": len(ok),
        "pictures_total": len(cams),
        "slowest_s": max(times) if times else None,
        "fastest_s": min(times) if times else None,
        "round_s_mean": sum(round_times) / len(round_times) if round_times else None,
        "round1_s": round_times[0] if round_times else None,
        "later_round_s_mean": sum(later) / len(later) if later else None,
    }


def recommend(modes: dict) -> Optional[str]:
    """First mode in MODE_ORDER with no fatal error in which every round and every camera succeeded.

    Returns the mode key, or None when no mode worked. No time limit or threshold is applied.
    """
    for key in MODE_ORDER:
        mode = modes.get(key) or {}
        rounds = mode.get("rounds") or []
        if mode.get("fatal_error") or not rounds:
            continue
        if all(cam["ok"] for rnd in rounds for cam in rnd["cameras"]):
            return key
    return None


def _save_picture(cv2: Any, path: Path, image: Any) -> bool:
    try:
        return bool(cv2.imwrite(str(path), image))
    except Exception:
        return False


def write_report_atomic(path: Path, report: dict) -> None:
    """Write the JSON report to a temporary file in the same folder, then replace the target.

    Raises ReportWriteError when json cannot encode the report (ValueError or TypeError).
    """
    try:
        data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    except (ValueError, TypeError) as exc:
        raise ReportWriteError(f"the report is not valid JSON, nothing was written ({_err_text(exc)})") from exc
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def run_hubtest(
    *,
    devices: Sequence[int],
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    rounds: int = DEFAULT_ROUNDS,
    warmup: int = DEFAULT_WARMUP,
    backend: str = STATION_BACKEND,
    out_path: Any = DEFAULT_OUT,
    cv2_module: Any = None,
    np_module: Any = None,
    now: Callable[[], str] = utc_now_iso,
    clock: Callable[[], float] = time.monotonic,
    out: Callable[[str], Any] = print,
) -> dict:
    """Run Tests A1, A2 and B, write the sample pictures and the JSON report, return the report.

    Raises ValueError for bad arguments, ToolDependencyError when cv2 or numpy is missing
    (before any camera is opened) and ReportWriteError when the report cannot be encoded.
    Camera failures never raise; they are in the report.
    """
    devices = [int(d) for d in devices]
    if not devices:
        raise ValueError("at least one device is required")
    if len(set(devices)) != len(devices):
        raise ValueError("device indexes must be unique")
    if any(d < 0 for d in devices):
        raise ValueError("device indexes must not be negative")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be 0 or more")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if backend not in ("dshow", "default"):
        raise ValueError("backend must be dshow or default")

    np = np_module if np_module is not None else _import_numpy()
    cv2 = cv2_module if cv2_module is not None else _import_cv2()
    api = cv2.CAP_DSHOW if backend == "dshow" else None
    out_path = Path(out_path)
    started = now()
    if backend != STATION_BACKEND:
        out(BACKEND_WARNING.format(backend=backend, station=STATION_BACKEND))

    plan = _Plan(cv2=cv2, np=np, devices=tuple(devices), api=api, width=int(width),
                 height=int(height), rounds=int(rounds), warmup=int(warmup), clock=clock, out=out)
    modes: dict[str, dict] = {}
    samples: dict[int, Any] = {}
    for key in RUN_ORDER:
        title = MODE_TITLES[key]
        out("")
        out(title)
        if key == "simultaneous":
            out(SIMULTANEOUS_NOTE)
        fatal: Optional[str] = None
        try:
            rounds_done, extra = _RUNNERS[key](plan)
        except Exception as exc:  # safety net: a bug must not lose the other tests
            rounds_done, extra, fatal = [], {}, _err_text(exc)
            out(f"{title} stopped unexpectedly: {fatal}")
        if key == "sequential_open_per_shot":
            samples = extra
        modes[key] = {
            "title": title,
            "fatal_error": fatal,
            "rounds": rounds_done,
            "summary": _summarise(rounds_done),
        }

    sample_pictures: dict[str, Optional[str]] = {str(d): None for d in devices}
    for dev, image in samples.items():
        name = f"{out_path.stem}_cam{dev}_first.png"
        if _save_picture(cv2, out_path.with_name(name), image):
            sample_pictures[str(dev)] = name

    key = recommend(modes)
    report = {
        "schema": SCHEMA_VERSION,
        "tool": "tools.hubtest",
        "started_utc": started,
        "finished_utc": now(),
        "devices": devices,
        "requested": {"width": int(width), "height": int(height), "rounds": int(rounds),
                      "warmup": int(warmup), "backend": backend},
        "station_backend": STATION_BACKEND,
        "modes": {k: modes[k] for k in MODE_ORDER if k in modes},
        "recommended_mode": key,
        "sample_pictures": sample_pictures,
    }
    write_report_atomic(out_path, report)
    return report


def _problems(mode: dict) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    for rnd in mode["rounds"]:
        for cam in rnd["cameras"]:
            if not cam["ok"]:
                found.append((cam["device"], rnd["round"], cam["error"] or "no picture"))
            if cam["release_error"]:
                found.append((cam["device"], rnd["round"], cam["release_error"]))
    return found


def wait_text(report: dict) -> str:
    """The 'Operator wait' line for the recommended mode, with the basis of the figure."""
    n = len(report["devices"])
    key = report["recommended_mode"]
    if key is None:
        return f"Operator wait for {n} pictures: n/a (no mode worked) ({QUALITY_NOTE})"
    summary = report["modes"][key]["summary"]
    if key == "sequential_keep_open":
        seconds = summary["later_round_s_mean"]
        if seconds is None:
            return (f"Operator wait for {n} pictures: n/a (keep-open needs --rounds 2 or more, "
                    f"because rounds 2 and later are used) ({QUALITY_NOTE})")
        basis = "keep-open, mean of rounds 2 and later; opening and warm-up not included"
    else:
        seconds = summary["round_s_mean"]
        rounds_n = report["requested"]["rounds"]
        basis = f"{MODE_NAMES[key]}, {MODE_CYCLE[key]}, mean of {rounds_n} round(s)"
    return f"Operator wait for {n} pictures: {seconds:.2f} s ({basis}) ({QUALITY_NOTE})"


def print_summary(report: dict, out: Callable[[str], Any] = print) -> None:
    devices = report["devices"]
    req = report["requested"]
    n = len(devices)
    out("")
    out(f"Summary: cameras {' '.join(str(d) for d in devices)}, {req['width']}x{req['height']}, "
        f"{req['rounds']} round(s), warm-up {req['warmup']}, backend {req['backend']}.")
    out("The script judges nothing against a threshold; it reports what happened.")
    for key in RUN_ORDER:
        mode = report["modes"][key]
        s = mode["summary"]
        out("")
        out(MODE_TITLES[key])
        out(f"  {s['pictures_ok']}/{s['pictures_total']} pictures OK")
        if s["slowest_s"] is None:
            out("  Slowest and fastest camera: n/a (no picture was taken)")
        else:
            out(f"  Slowest camera: {s['slowest_s']:.2f} s, fastest camera: {s['fastest_s']:.2f} s")
        if s["round_s_mean"] is None:
            out(f"  Total time for {n} pictures: n/a")
        else:
            out(f"  Total time for {n} pictures: {s['round_s_mean']:.2f} s "
                f"(mean of {req['rounds']} round(s))")
        if key == "sequential_keep_open" and s["round1_s"] is not None:
            out(f"  Round 1 (includes opening and warm-up): {s['round1_s']:.2f} s")
            if s["later_round_s_mean"] is None:
                out("  Rounds 2 and later: n/a (only one round was run)")
            else:
                out(f"  Rounds 2 and later, mean: {s['later_round_s_mean']:.2f} s (opening not included)")
        if mode["fatal_error"]:
            out(f"  Stopped unexpectedly: {mode['fatal_error']}")
        problems = _problems(mode)
        if problems:
            out("  Failures:")
            for dev, rnd, text in problems:
                out(f"    camera {dev}, round {rnd}: {text}")
        else:
            out("  Failures: none")

    key = report["recommended_mode"]
    out("")
    if key is None:
        out("Recommended mode: none worked")
    else:
        out(f"Recommended mode: {MODE_NAMES[key]} worked")
    out(wait_text(report))
    if _problems(report["modes"]["simultaneous"]):
        out("Test B failures above are a result, not a crash: a hub often cannot send all cameras at once.")
    saved = [name for name in report["sample_pictures"].values() if name]
    if saved:
        out("Sample pictures (first picture per camera, Test A1): " + ", ".join(saved))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="py -m tools.hubtest",
        description="Check whether webcams behind one USB hub deliver pictures, one at a time "
                    "and all at once. Writes a JSON report and sample pictures.",
    )
    p.add_argument("--devices", nargs="+", type=int, required=True,
                   help="camera device indexes, e.g. 0 1 2 3")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                   help=f"rounds per mode, one picture per camera per round (default {DEFAULT_ROUNDS})")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                   help="frames read and discarded right after each camera opens, as the station "
                        f"does (engineering setting, not a threshold; default {DEFAULT_WARMUP})")
    p.add_argument("--backend", choices=("default", "dshow"), default=STATION_BACKEND,
                   help="OpenCV backend. default (the station's, cv2.VideoCapture(index)) is the "
                        "default; dshow is the Windows DirectShow backend, which prints a warning")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"JSON report path (default {DEFAULT_OUT})")
    return p


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    cv2_module: Any = None,
    np_module: Any = None,
    now: Callable[[], str] = utc_now_iso,
    clock: Callable[[], float] = time.monotonic,
    out: Callable[[str], Any] = print,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_hubtest(
            devices=args.devices,
            width=args.width,
            height=args.height,
            rounds=args.rounds,
            warmup=args.warmup,
            backend=args.backend,
            out_path=args.out,
            cv2_module=cv2_module,
            np_module=np_module,
            now=now,
            clock=clock,
            out=out,
        )
    except (ToolDependencyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ReportWriteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: could not write the report to {args.out}: {exc.strerror or exc}", file=sys.stderr)
        return 1
    print_summary(report, out)
    out(f"Report written to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
