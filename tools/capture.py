"""Real-webcam capture for the repeatability study. Run on the user's Windows PC.

    python -m tools.capture --mode stationary --devices 0 --out sessions --name NAME
    python -m tools.capture --list-devices [--max-index N] [--backend dshow|default]

Writes the session folder described in tools/FORMAT.md. cv2 and numpy are
imported lazily; a cv2 module can be injected (the tests do this).

The exposure response check is a 'did the picture respond at all' check, NOT a
quality threshold. It compares mean gray levels against an engineering epsilon of
1.0 gray level (RESPONSE_EPS_GRAY). It can read as responsive under mains flicker
or auto-drift, and as unresponsive in a saturated or black scene. Before trusting
it, check that the scene is mid-grey and that the light is steady.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from tools import ToolDependencyError

SCHEMA_VERSION = 1
MODES = ("stationary", "placement")
DEFAULT_COUNT = {"stationary": 50, "placement": 30}
PLACEMENT_PROMPT = "Remove the part, put it back in the fixture, then press Enter"
INSTALL_HINT = "pip install opencv-python numpy"
# V4L2: 0.25 is manual, 0.75 is auto. DirectShow may use 1 for manual instead.
MANUAL_EXPOSURE_CANDIDATES = (0.25, 1.0)
AUTOFOCUS_OFF = 0.0
AUTO_WB_OFF = 0.0
# Engineering setting, not a threshold: frames grabbed and discarded before each
# real capture so that stale frames queued by the driver are not used. 0 disables.
DEFAULT_FLUSH_FRAMES = 5
BUFFERSIZE_WANTED = 1.0
# Exposure response check: did the picture respond at all? NOT a quality threshold.
RESPONSE_FRAMES = 3
RESPONSE_EPS_GRAY = 1.0
ECHO_NOTE = "DirectShow read-back can echo the value you set"
NO_EFFECT_NOTE = (
    "exposure changes had no visible effect: the camera may be ignoring manual exposure "
    f"({ECHO_NOTE}). This is a 'did the picture respond at all' check with a 1.0 gray-level "
    "engineering epsilon, NOT a quality threshold; it can read as responsive under mains "
    "flicker or auto-drift and as unresponsive in a saturated or black scene, so check that "
    "the scene is mid-grey and the light is steady"
)
RESTORE_NOTE_FMT = (
    "exposure: restoring the requested value {requested:g} did not take (read back {readback}); "
    "later frames may keep the changed exposure"
)
# Engineering setting, not a threshold: how many camera indexes --list-devices probes (0..N).
DEFAULT_MAX_INDEX = 5
# Brightness settle check, after the timed warm-up and before the first recorded frame.
# Engineering settings, NOT quality thresholds. The tolerance is a 'did it stop changing'
# epsilon in gray levels, the same kind of epsilon as RESPONSE_EPS_GRAY.
DEFAULT_SETTLE_MAX_S = 30.0
DEFAULT_SETTLE_FRAMES = 5
DEFAULT_SETTLE_TOL_GRAY = 1.0
SETTLE_INTERVAL_S = 0.2
# Engineering hint only: a first recorded frame this far (fraction of the median) off the median.
SETTLE_HINT_FRACTION = 0.10
SETTLE_WARNING_FMT = (
    "brightness had not settled after {max_s:g} s (last frame means: {means}); the first frames "
    "of this run may be unrepresentative: increase --settle-max-s or fix the lighting"
)
SETTLE_HINT_FMT = (
    "device {dev}: first frame brightness {first:.1f} differs from the median {median:.1f}: the "
    "camera may not have settled; check the report's outlier section."
)
SKIPPED_NOTE = (
    "exposure response check skipped (--no-response-check): the camera may be ignoring "
    f"manual exposure ({ECHO_NOTE})"
)
_WINDOWS_BAD_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
    + ["COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³"]
)
_ABS_TOL = 1e-6
_REL_TOL = 0.02


class CaptureError(RuntimeError):
    """A camera could not be opened, configured, grabbed or retrieved."""


class SessionNameError(CaptureError, ValueError):
    """A session name Windows cannot use as a folder name. The CLI exits with 2."""


@dataclass
class _Settings:
    width: int
    height: int
    fps: Optional[float]
    fourcc: Optional[str]
    focus: Optional[float]
    exposure: Optional[float]
    white_balance: Optional[float]
    gain: Optional[float]
    flush_frames: int
    response_check: bool


@dataclass
class _DeviceLog:
    grays: list = field(default_factory=list)
    stamps: list = field(default_factory=list)
    stats: list = field(default_factory=list)  # (mean, laplacian_var) per frame


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def _import_numpy() -> Any:
    try:
        import numpy  # noqa: PLC0415 - lazy on purpose
    except ImportError as exc:
        raise ToolDependencyError(
            f"numpy is needed for capture. Install it with: {INSTALL_HINT}"
        ) from exc
    return numpy


def _import_cv2() -> Any:
    try:
        import cv2  # noqa: PLC0415 - lazy on purpose
    except ImportError as exc:
        raise ToolDependencyError(
            f"OpenCV is needed for capture. Install it with: {INSTALL_HINT}"
        ) from exc
    return cv2


def _check_session_name(name: str) -> None:
    """Reject names Windows cannot use as a folder name, saying what to change."""
    hint = "use letters, digits, '-' or '_' instead"
    if not name.strip():
        raise SessionNameError("session name is empty; give the session a short name such as line1-run2")
    bad = sorted({c for c in name if c in _WINDOWS_BAD_CHARS})
    if bad:
        raise SessionNameError(
            f"session name {name!r} has {', '.join(repr(c) for c in bad)}, which Windows does not "
            f"allow in folder names; {hint}"
        )
    control = [c for c in name if ord(c) < 32 or ord(c) == 127]
    if control:
        raise SessionNameError(
            f"session name {name!r} has a control character (code {ord(control[0])}); {hint}"
        )
    if name.endswith((".", " ")):
        raise SessionNameError(
            f"session name {name!r} ends with a dot or space, which Windows does not allow; "
            "remove the trailing character"
        )
    stem = name.split(".", 1)[0].strip().upper()
    if stem in _WINDOWS_RESERVED:
        raise SessionNameError(
            f"session name {name!r} is a reserved Windows device name ({stem}), also with an "
            "extension; choose a different name"
        )


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= _ABS_TOL


def _value_ok(readback: float, requested: float) -> bool:
    return abs(readback - requested) <= max(_ABS_TOL, _REL_TOL * abs(requested))


def _safe(dev: int, fn: Callable[..., Any], *args: Any) -> Any:
    try:
        return fn(*args)
    except Exception as exc:
        raise CaptureError(f"device {dev}: camera call failed ({exc})") from exc


def _open_device(cv2: Any, dev: int, api: Any) -> Any:
    try:
        return cv2.VideoCapture(dev) if api is None else cv2.VideoCapture(dev, api)
    except Exception as exc:
        raise CaptureError(f"device {dev}: could not open camera ({exc})") from exc


def _configure(cv2: Any, np: Any, dev: int, cap: Any, s: _Settings) -> dict:
    """Set buffer size, format, then manual controls, then read everything back."""
    call = lambda fn, *args: _safe(dev, fn, *args)  # noqa: E731

    # Small driver queue so grab() returns fresher frames. Best effort: a refused
    # set is visible in the read-back below, so it is not raised here.
    buffersize_readback: Optional[float] = None
    with contextlib.suppress(Exception):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFERSIZE_WANTED)
    with contextlib.suppress(Exception):
        buffersize_readback = float(cap.get(cv2.CAP_PROP_BUFFERSIZE))

    # Format first: MJPG must be set before width/height/fps for 2K on USB webcams.
    if s.fourcc:
        call(cap.set, cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*s.fourcc))
    call(cap.set, cv2.CAP_PROP_FRAME_WIDTH, s.width)
    call(cap.set, cv2.CAP_PROP_FRAME_HEIGHT, s.height)
    if s.fps is not None:
        call(cap.set, cv2.CAP_PROP_FPS, s.fps)

    # Then manual mode. Every auto-exposure candidate is recorded with its read-back.
    call(cap.set, cv2.CAP_PROP_AUTOFOCUS, AUTOFOCUS_OFF)
    chosen_exposure: Optional[float] = None
    attempts: list[dict] = []
    for candidate in MANUAL_EXPOSURE_CANDIDATES:
        call(cap.set, cv2.CAP_PROP_AUTO_EXPOSURE, candidate)
        echoed = float(call(cap.get, cv2.CAP_PROP_AUTO_EXPOSURE))
        attempts.append({"set": candidate, "readback": echoed})
        if _same(echoed, candidate):
            chosen_exposure = candidate
            break
    call(cap.set, cv2.CAP_PROP_AUTO_WB, AUTO_WB_OFF)

    requested = {
        "focus": (cv2.CAP_PROP_FOCUS, s.focus),
        "exposure": (cv2.CAP_PROP_EXPOSURE, s.exposure),
        "white_balance": (cv2.CAP_PROP_WB_TEMPERATURE, s.white_balance),
        "gain": (cv2.CAP_PROP_GAIN, s.gain),
    }
    for prop, value in requested.values():
        if value is not None:
            call(cap.set, prop, float(value))

    readback = {
        "autofocus": call(cap.get, cv2.CAP_PROP_AUTOFOCUS),
        "focus": call(cap.get, cv2.CAP_PROP_FOCUS),
        "auto_exposure": call(cap.get, cv2.CAP_PROP_AUTO_EXPOSURE),
        "exposure": call(cap.get, cv2.CAP_PROP_EXPOSURE),
        "auto_wb": call(cap.get, cv2.CAP_PROP_AUTO_WB),
        "white_balance": call(cap.get, cv2.CAP_PROP_WB_TEMPERATURE),
        "gain": call(cap.get, cv2.CAP_PROP_GAIN),
    }
    readback = {k: float(v) for k, v in readback.items()}

    warnings: list[str] = []
    applied: dict[str, Optional[bool]] = {}

    applied["autofocus"] = _same(readback["autofocus"], AUTOFOCUS_OFF)
    if not applied["autofocus"]:
        warnings.append(
            f"autofocus: camera/driver ignored autofocus-off (read back {readback['autofocus']:g}); "
            "focus may hunt between frames"
        )

    applied["auto_exposure"] = chosen_exposure is not None
    if not applied["auto_exposure"]:
        warnings.append(
            "auto exposure: camera/driver ignored manual exposure (tried 0.25 and 1, read back "
            f"{readback['auto_exposure']:g}); exposure may vary between frames"
        )

    applied["auto_wb"] = _same(readback["auto_wb"], AUTO_WB_OFF)
    if not applied["auto_wb"]:
        warnings.append(
            f"auto white balance: camera/driver ignored auto-WB-off (read back {readback['auto_wb']:g}); "
            "colour may drift, and 'auto lighting correction' may be uncontrollable on some webcams"
        )

    for name, (_, value) in requested.items():
        if value is None:
            applied[name] = None
            continue
        ok = _value_ok(readback[name], float(value))
        applied[name] = ok
        if not ok:
            warnings.append(
                f"{name}: camera/driver ignored the requested value {float(value):g} "
                f"(read back {readback[name]:g})"
            )

    if buffersize_readback is not None and not _same(buffersize_readback, BUFFERSIZE_WANTED):
        warnings.append(
            f"buffer size: camera/driver did not take 1 (read back {buffersize_readback:g}); "
            "grab() may return queued old frames"
        )

    actual_w = int(round(float(call(cap.get, cv2.CAP_PROP_FRAME_WIDTH))))
    actual_h = int(round(float(call(cap.get, cv2.CAP_PROP_FRAME_HEIGHT))))
    actual_fps = float(call(cap.get, cv2.CAP_PROP_FPS))
    if (actual_w, actual_h) != (s.width, s.height):
        warnings.append(
            f"resolution: camera delivers {actual_w}x{actual_h}, requested {s.width}x{s.height} "
            "(camera/driver ignored the size; for 2K on USB try --fourcc MJPG)"
        )
    if s.fps is not None and actual_fps > 0 and not _value_ok(actual_fps, s.fps):
        warnings.append(f"fps: camera reports {actual_fps:g}, requested {s.fps:g}")

    exposure_response: Optional[dict] = None
    if s.exposure is None:
        applied["exposure_responds"] = None
    elif not s.response_check:
        applied["exposure_responds"] = None
        warnings.append(SKIPPED_NOTE)
    else:
        responds, exposure_response = _exposure_response(np, cv2, dev, cap, s)
        applied["exposure_responds"] = responds
        if not responds:
            warnings.append(NO_EFFECT_NOTE)
        if not exposure_response["restore_ok"]:
            readback = exposure_response["restore_readback"]
            shown = "no value" if readback is None else f"{readback:g}"
            warnings.append(RESTORE_NOTE_FMT.format(requested=float(s.exposure), readback=shown))

    return {
        "actual_width": actual_w,
        "actual_height": actual_h,
        "actual_fps": actual_fps,
        "buffersize_readback": buffersize_readback,
        "readback": readback,
        "applied_ok": applied,
        "auto_exposure_attempts": attempts,
        "exposure_response": exposure_response,
        "warnings": warnings,
    }


def _grab(dev: int, cap: Any) -> None:
    try:
        ok = cap.grab()
    except Exception as exc:
        raise CaptureError(f"device {dev}: grab failed ({exc})") from exc
    if not ok:
        raise CaptureError(f"device {dev}: grab failed, camera gave no frame")


def _retrieve(dev: int, cap: Any) -> Any:
    try:
        ok, image = cap.retrieve()
    except Exception as exc:
        raise CaptureError(f"device {dev}: retrieve failed ({exc})") from exc
    if not ok or image is None:
        raise CaptureError(f"device {dev}: retrieve failed, camera gave no frame")
    return image


def _flush(caps: list, n: int) -> None:
    """Grab and discard n frames from every device, so the next grab is fresh."""
    for dev, cap in caps:
        for _ in range(n):
            _grab(dev, cap)


def _gray_of(np: Any, cv2: Any, image: Any) -> Any:
    arr = np.asarray(image)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    elif arr.ndim == 2:
        gray = arr
    else:
        raise CaptureError(f"unexpected image shape {arr.shape}")
    return np.ascontiguousarray(gray, dtype=np.uint8)


def _frame_stats(np: Any, gray: Any) -> tuple[float, float]:
    g = gray.astype(np.float64)
    mean = float(g.mean())
    if g.shape[0] >= 3 and g.shape[1] >= 3:
        inner = g[1:-1, 1:-1]
        lap = 4 * inner - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        lap_var = float(lap.var())
    else:
        lap_var = 0.0
    return mean, lap_var


def _mean_after_flush(np: Any, cv2: Any, dev: int, cap: Any, flush_frames: int) -> float:
    """Mean gray level of a few fresh frames, after discarding stale ones."""
    _flush([(dev, cap)], flush_frames)
    means = []
    for _ in range(RESPONSE_FRAMES):
        _grab(dev, cap)
        gray = _gray_of(np, cv2, _retrieve(dev, cap))
        means.append(float(gray.astype(np.float64).mean()))
    return sum(means) / len(means)


def _restore_exposure(cv2: Any, dev: int, cap: Any, requested: float, flush_frames: int) -> dict:
    """Put the requested exposure back, read it back, then discard stale frames."""
    readback: Optional[float] = None
    restore_ok = False
    try:
        cap.set(cv2.CAP_PROP_EXPOSURE, requested)
        readback = float(cap.get(cv2.CAP_PROP_EXPOSURE))
        restore_ok = _value_ok(readback, requested)
    except Exception:
        readback = None  # the driver refused the call: reported as not taken
    _flush([(dev, cap)], flush_frames)
    return {"restored_to": requested, "restore_readback": readback, "restore_ok": restore_ok}


def _exposure_response(np: Any, cv2: Any, dev: int, cap: Any, s: _Settings) -> tuple[bool, dict]:
    """Change exposure and see whether the picture moves at all. Not a quality check.

    Uses requested*2 if the driver reads it back, else requested/2. The requested
    value is restored in a finally block, so it is restored even if a grab fails.
    """
    call = lambda fn, *args: _safe(dev, fn, *args)  # noqa: E731
    requested = float(s.exposure)
    primary, fallback = (requested * 2, requested / 2) if requested != 0 else (1.0, -1.0)
    alt = primary
    try:
        base = _mean_after_flush(np, cv2, dev, cap, s.flush_frames)
        call(cap.set, cv2.CAP_PROP_EXPOSURE, primary)
        if not _value_ok(float(call(cap.get, cv2.CAP_PROP_EXPOSURE)), primary):
            call(cap.set, cv2.CAP_PROP_EXPOSURE, fallback)
            alt = fallback
        alt_mean = _mean_after_flush(np, cv2, dev, cap, s.flush_frames)
    finally:
        restore = _restore_exposure(cv2, dev, cap, requested, s.flush_frames)
    delta = abs(alt_mean - base)
    detail = {
        "alt_value": alt,
        "base_mean": base,
        "alt_mean": alt_mean,
        "delta": delta,
        "eps": RESPONSE_EPS_GRAY,
        **restore,
    }
    return delta > RESPONSE_EPS_GRAY, detail


def _warm_up(caps: list, warmup_s: float, clock: Callable[[], float]) -> None:
    if warmup_s <= 0:
        return
    end = clock() + warmup_s
    while clock() < end:
        for dev, cap in caps:
            _grab(dev, cap)


def _window_settled(means: list, frames: int, tol: float) -> bool:
    """True when the last `frames` means differ from each other by at most `tol` gray levels."""
    if len(means) < frames:
        return False
    window = means[-frames:]
    return max(window) - min(window) <= tol


def _settle_record(*, checked: bool, settled: bool, seconds: float, frames_used: int,
                   last_means: list, tol: float, frames: int, max_s: float) -> dict:
    return {
        "checked": bool(checked),
        "settled": bool(settled),
        "seconds": float(seconds),
        "frames_used": int(frames_used),
        "last_means": [float(m) for m in last_means],
        "tol_gray": float(tol),
        "frames": int(frames),
        "max_s": float(max_s),
    }


def _settle(np: Any, cv2: Any, caps: list, cameras: dict, *, max_s: float, frames: int,
            tol: float, clock: Callable[[], float], sleep: Callable[[float], Any]) -> None:
    """Wait until each camera's brightness stops changing, then record cameras[dev]["settle"].

    Grabs and retrieves every camera about every SETTLE_INTERVAL_S seconds. A camera has
    settled when the means of its last `frames` frames differ by at most `tol` gray levels.
    Stops when every camera has settled or `max_s` seconds have passed. These frames are
    discarded, not recorded. The record is written even when a grab fails.
    """
    means: dict[int, list] = {dev: [] for dev, _ in caps}
    settled_at: dict[int, float] = {}
    start = clock()
    elapsed = 0.0
    # Hard cap on passes so a wall-clock that jumps backwards (or a stuck test clock)
    # can never keep this loop running: max_s / interval, plus a few spare passes.
    max_passes = int(max_s / SETTLE_INTERVAL_S) + max(frames, 1) + 5
    passes = 0
    try:
        while True:
            passes += 1
            for dev, cap in caps:
                _grab(dev, cap)
            for dev, cap in caps:
                gray = _gray_of(np, cv2, _retrieve(dev, cap))
                means[dev].append(float(gray.astype(np.float64).mean()))
            elapsed = clock() - start
            for dev, _ in caps:
                if dev not in settled_at and _window_settled(means[dev], frames, tol):
                    settled_at[dev] = elapsed
            if len(settled_at) == len(caps) or elapsed >= max_s or passes >= max_passes:
                break
            sleep(SETTLE_INTERVAL_S)
    finally:
        for dev, _ in caps:
            done = dev in settled_at
            cameras[str(dev)]["settle"] = _settle_record(
                checked=True,
                settled=done,
                seconds=settled_at.get(dev, elapsed),
                frames_used=len(means[dev]),
                last_means=means[dev][-frames:],
                tol=tol,
                frames=frames,
                max_s=max_s,
            )
    for dev, _ in caps:
        if dev not in settled_at:
            last = ", ".join(f"{m:.1f}" for m in means[dev][-frames:])
            cameras[str(dev)]["warnings"].append(SETTLE_WARNING_FMT.format(max_s=max_s, means=last))


def _first_frame_hints(np: Any, devices: list, logs: dict) -> list[str]:
    """Engineering hint only: a first recorded frame more than 10 % off the median of the frames."""
    hints: list[str] = []
    for dev in devices:
        means = [mean for mean, _ in logs[dev].stats]
        if not means:
            continue
        median = float(np.median(means))
        first = float(means[0])
        if abs(first - median) > SETTLE_HINT_FRACTION * median:
            hints.append(SETTLE_HINT_FMT.format(dev=dev, first=first, median=median))
    return hints


def _wait_for_placement(index: int, count: int, no_prompt: bool, input_fn: Callable[[str], str], out: Callable[[str], Any]) -> None:
    out(f"Round {index + 1}/{count}: {PLACEMENT_PROMPT}")
    if not no_prompt:
        try:
            input_fn("")
        except EOFError as exc:
            raise CaptureError("input closed while waiting for the placement round") from exc


def _capture_round(
    np: Any,
    cv2: Any,
    caps: list,
    logs: dict,
    index: int,
    count: int,
    sample_every: int,
    folder: Path,
    clock: Callable[[], float],
) -> None:
    """Grab every device first (close in time), then retrieve, then record."""
    # Stamps are taken at grab time (when grab() returns), not at exposure time.
    stamps: dict[int, float] = {}
    for dev, cap in caps:
        _grab(dev, cap)
        stamps[dev] = clock()
    images = {dev: _retrieve(dev, cap) for dev, cap in caps}

    for dev, _ in caps:
        gray = _gray_of(np, cv2, images[dev])
        mean, lap_var = _frame_stats(np, gray)
        log = logs[dev]
        log.grays.append(gray)
        log.stamps.append(stamps[dev])
        log.stats.append((mean, lap_var))

    is_sample = index == 0 or index == count - 1 or index % sample_every == 0
    if is_sample:
        for dev, _ in caps:
            path = folder / "samples" / f"dev{dev}_frame{index}.png"
            if not cv2.imwrite(str(path), images[dev]):
                raise CaptureError(f"device {dev}: could not write sample {path.name}")


def _release_all(caps: list) -> dict[int, str]:
    errors: dict[int, str] = {}
    for dev, cap in caps:
        try:
            cap.release()
        except Exception as exc:
            errors[dev] = f"release failed ({exc})"
    return errors


def _write_session(
    np: Any,
    folder: Path,
    *,
    created: str,
    mode: str,
    name: str,
    devices: list,
    settings: _Settings,
    notes: dict,
    cameras: dict,
    logs: dict,
    aborted: Optional[str],
) -> list:
    frames: list[dict] = []
    for dev in devices:
        log = logs[dev]
        if log.grays:
            np.savez_compressed(
                str(folder / f"frames_dev{dev}.npz"),
                gray=np.stack(log.grays).astype(np.uint8),
                stamps=np.asarray(log.stamps, dtype=np.float64),
            )
        for idx, ((mean, lap_var), stamp) in enumerate(zip(log.stats, log.stamps)):
            frames.append(
                {
                    "index": idx,
                    "device": dev,
                    "stamp": float(stamp),
                    "mean": float(mean),
                    "laplacian_var": float(lap_var),
                }
            )

    free_parts = [notes.get("free", "")]
    if aborted:
        free_parts.append(f"aborted: {aborted}")
    doc = {
        "schema": SCHEMA_VERSION,
        "created_utc": created,
        "mode": mode,
        "session_name": name,
        "devices": devices,
        "notes": {
            "part": notes.get("part", ""),
            "lighting": notes.get("lighting", ""),
            "operator": notes.get("operator", ""),
            "free": "; ".join(p for p in free_parts if p),
        },
        "requested": {
            "width": settings.width,
            "height": settings.height,
            "fps": settings.fps,
            "fourcc": settings.fourcc,
            "manual": {
                "focus": settings.focus,
                "exposure": settings.exposure,
                "white_balance": settings.white_balance,
                "gain": settings.gain,
            },
            "flush_frames": settings.flush_frames,
        },
        "cameras": cameras,
        "frames": frames,
    }
    (folder / "session.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return frames


def capture_session(
    *,
    mode: str,
    devices: Sequence[int],
    out_dir: Any,
    name: str,
    count: Optional[int] = None,
    interval_s: float = 0.5,
    warmup_s: float = 5.0,
    width: int = 2560,
    height: int = 1440,
    fps: Optional[float] = None,
    fourcc: Optional[str] = None,
    focus: Optional[float] = None,
    exposure: Optional[float] = None,
    white_balance: Optional[float] = None,
    gain: Optional[float] = None,
    part: str = "",
    lighting: str = "",
    operator: str = "",
    note: str = "",
    no_prompt: bool = False,
    sample_every: int = 10,
    flush_frames: int = DEFAULT_FLUSH_FRAMES,
    response_check: bool = True,
    settle_max_s: float = DEFAULT_SETTLE_MAX_S,
    settle_frames: int = DEFAULT_SETTLE_FRAMES,
    settle_tol_gray: float = DEFAULT_SETTLE_TOL_GRAY,
    backend: Optional[str] = None,
    cv2_module: Any = None,
    input_fn: Callable[[str], str] = input,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Any] = time.sleep,
    out: Callable[[str], Any] = print,
) -> dict:
    """Capture one session and write its folder. Returns a summary dict.

    Raises ValueError for bad arguments (SessionNameError, a ValueError, for a
    bad session name), ToolDependencyError when numpy or cv2 is missing, and
    CaptureError when the folder exists, cannot be created, or the cameras
    cannot be opened. A failure during capture does not raise: the partial
    data is kept and summary["aborted"] holds the reason.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    devices = [int(d) for d in devices]
    if not devices:
        raise ValueError("at least one device is required")
    if len(set(devices)) != len(devices):
        raise ValueError("device indexes must be unique")
    if any(d < 0 for d in devices):
        raise ValueError("device indexes must not be negative")
    if count is None:
        count = DEFAULT_COUNT[mode]
    if count < 1:
        raise ValueError("count must be at least 1")
    if interval_s < 0 or warmup_s < 0:
        raise ValueError("interval and warm-up must not be negative")
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1")
    if flush_frames < 0:
        raise ValueError("flush-frames must be 0 or more (0 disables flushing)")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if settle_max_s < 0:
        raise ValueError("settle-max-s must be 0 or more (0 disables the brightness settle check)")
    if settle_frames < 2:
        raise ValueError("settle-frames must be at least 2")
    if settle_tol_gray < 0:
        raise ValueError("settle-tol-gray must be 0 or more")
    if fourcc is not None and len(fourcc) != 4:
        raise ValueError("fourcc must be four characters, e.g. MJPG")
    if backend not in (None, "dshow", "default"):
        raise ValueError("backend must be dshow or default")
    _check_session_name(name)

    settings = _Settings(
        width=int(width),
        height=int(height),
        fps=None if fps is None else float(fps),
        fourcc=fourcc,
        focus=focus,
        exposure=exposure,
        white_balance=white_balance,
        gain=gain,
        flush_frames=int(flush_frames),
        response_check=bool(response_check),
    )
    api_mode = backend or ("dshow" if _is_windows() else "default")

    np = _import_numpy()
    cv2 = cv2_module if cv2_module is not None else _import_cv2()

    folder = Path(out_dir) / name
    if folder.exists():
        raise CaptureError(f"refusing to overwrite existing session folder: {folder}")
    try:
        folder.mkdir(parents=True)
        (folder / "samples").mkdir()
    except OSError as exc:
        with contextlib.suppress(OSError):
            folder.rmdir()  # leave no half-made folder behind
        raise CaptureError(
            f"could not create session folder {folder}: {exc.strerror or exc}. "
            "Check that the parent folder exists and is writable, or choose another --out"
        ) from exc

    created = utc_now_iso()
    api = cv2.CAP_DSHOW if api_mode == "dshow" else None
    caps: list[tuple[int, Any]] = []
    logs = {dev: _DeviceLog() for dev in devices}
    cameras: dict[str, dict] = {}
    aborted: Optional[str] = None
    release_problems: list[str] = []
    notes = {"part": part, "lighting": lighting, "operator": operator, "free": note}

    try:
        for dev in devices:
            cap = _open_device(cv2, dev, api)
            caps.append((dev, cap))
            if not cap.isOpened():
                raise CaptureError(f"device {dev}: could not open camera at index {dev}")
            cameras[str(dev)] = _configure(cv2, np, dev, cap, settings)
        _warm_up(caps, warmup_s, clock)
        if settle_max_s > 0:
            _settle(np, cv2, caps, cameras, max_s=settle_max_s, frames=settle_frames,
                    tol=settle_tol_gray, clock=clock, sleep=sleep)
        else:
            for dev in devices:
                cameras[str(dev)]["settle"] = _settle_record(
                    checked=False, settled=False, seconds=0.0, frames_used=0, last_means=[],
                    tol=settle_tol_gray, frames=settle_frames, max_s=0.0,
                )
        for index in range(count):
            if mode == "placement":
                _wait_for_placement(index, count, no_prompt, input_fn, out)
            _flush(caps, settings.flush_frames)
            _capture_round(np, cv2, caps, logs, index, count, sample_every, folder, clock)
            if mode == "stationary" and index < count - 1 and interval_s > 0:
                sleep(interval_s)
    except CaptureError as exc:
        aborted = str(exc)
    except KeyboardInterrupt:
        aborted = "interrupted by the user"
    except Exception as exc:
        aborted = f"unexpected error: {exc!r}"
        raise
    finally:
        for dev, message in _release_all(caps).items():
            if str(dev) in cameras:
                cameras[str(dev)]["warnings"].append(f"{message}")
            else:
                release_problems.append(f"device {dev}: {message}")
        frames = _write_session(
            np,
            folder,
            created=created,
            mode=mode,
            name=name,
            devices=devices,
            settings=settings,
            notes=notes,
            cameras=cameras,
            logs=logs,
            aborted=aborted,
        )

    hints = _first_frame_hints(np, devices, logs)
    warnings = [f"device {dev}: {w}" for dev in devices for w in cameras.get(str(dev), {}).get("warnings", [])]
    warnings.extend(release_problems)
    return {
        "folder": str(folder),
        "frames_saved": len(frames),
        "frames_per_device": {str(dev): len(logs[dev].stats) for dev in devices},
        "warnings": warnings,
        "aborted": aborted,
        "settle": {str(dev): cameras[str(dev)]["settle"]
                   for dev in devices if "settle" in cameras.get(str(dev), {})},
        "hints": hints,
    }


def _api_mode(backend: Optional[str]) -> str:
    return backend or ("dshow" if _is_windows() else "default")


def list_devices(
    *,
    max_index: int = DEFAULT_MAX_INDEX,
    backend: Optional[str] = None,
    cv2_module: Any = None,
    out: Callable[[str], Any] = print,
) -> list[dict]:
    """Probe camera indexes 0..max_index and report those that open and deliver a frame.

    Every camera that was opened is released again. Returns one dict per camera
    found: index, width, height, not_black. Raises ValueError for bad arguments and
    ToolDependencyError when numpy or cv2 is missing.
    """
    if max_index < 0:
        raise ValueError("max-index must be 0 or more")
    if backend not in (None, "dshow", "default"):
        raise ValueError("backend must be dshow or default")
    np = _import_numpy()
    cv2 = cv2_module if cv2_module is not None else _import_cv2()
    mode = _api_mode(backend)
    api = cv2.CAP_DSHOW if mode == "dshow" else None
    out(f"Probing camera indexes 0..{max_index} with the {mode} backend")

    found: list[dict] = []
    for index in range(max_index + 1):
        try:
            cap = _open_device(cv2, index, api)
        except CaptureError:
            continue
        try:
            if not cap.isOpened():
                continue
            try:
                _flush([(index, cap)], DEFAULT_FLUSH_FRAMES)  # let the camera settle
                _grab(index, cap)
                image = _retrieve(index, cap)
            except CaptureError:
                continue
            arr = np.asarray(image)
            width, height = int(arr.shape[1]), int(arr.shape[0])
            not_black = bool(np.any(arr))
            found.append({"index": index, "width": width, "height": height, "not_black": not_black})
            out(f"index {index}: {width}x{height}, {'not all black' if not_black else 'all black'}")
        finally:
            for _, message in _release_all([(index, cap)]).items():
                out(f"index {index}: {message}")

    if found:
        out(f"Found {len(found)} camera(s). Pass their indexes to --devices.")
    else:
        out(
            f"No camera opened at indexes 0..{max_index}. Close other apps that use the camera "
            "(video calls, the Windows Camera app), check the Windows privacy settings "
            "(Settings > Privacy & security > Camera), and try --backend default."
        )
    return found


def _require_capture_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in (("--mode", args.mode), ("--devices", args.devices), ("--name", args.name))
        if value is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.capture",
        description="Capture a repeatability session from USB webcams (see tools/FORMAT.md).",
    )
    p.add_argument("--mode", choices=MODES, default=None, help="required unless --list-devices")
    p.add_argument("--devices", nargs="+", type=int, default=None,
                   help="camera device indexes, e.g. 0 1 2 3 (required unless --list-devices)")
    p.add_argument("--out", default="sessions", help="parent folder for the session (default: sessions)")
    p.add_argument("--name", default=None,
                   help="session folder name; must not exist yet. No < > : \" / \\ | ? *, control "
                        "characters, trailing dot or space, or names like CON, NUL, COM1, LPT1")
    p.add_argument("--count", type=int, default=None, help="frames per device (default 50 stationary, 30 placement)")
    p.add_argument("--interval-s", type=float, default=0.5, help="seconds between stationary frames")
    p.add_argument("--warmup-s", type=float, default=5.0, help="seconds of discarded frames before capture")
    p.add_argument("--width", type=int, default=2560)
    p.add_argument("--height", type=int, default=1440)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--fourcc", default=None, help="e.g. MJPG (needed for 2K on many USB webcams)")
    p.add_argument("--focus", type=float, default=None)
    p.add_argument("--exposure", type=float, default=None)
    p.add_argument("--white-balance", dest="white_balance", type=float, default=None)
    p.add_argument("--gain", type=float, default=None)
    p.add_argument("--part", default="")
    p.add_argument("--lighting", default="")
    p.add_argument("--operator", default="")
    p.add_argument("--note", default="")
    p.add_argument("--no-prompt", action="store_true", help="placement mode: do not wait for Enter")
    p.add_argument("--sample-every", type=int, default=10, help="save a colour PNG every K frames")
    p.add_argument("--flush-frames", type=int, default=DEFAULT_FLUSH_FRAMES,
                   help="frames discarded from every camera before each capture, so stale driver "
                        "frames are not used (engineering setting, not a threshold; 0 disables)")
    p.add_argument("--no-response-check", action="store_true",
                   help="skip the check that the picture responds to --exposure changes")
    p.add_argument("--settle-max-s", type=float, default=DEFAULT_SETTLE_MAX_S,
                   help="seconds to wait, after warm-up and before the first recorded frame, for the "
                        "brightness to stop changing (engineering setting, not a threshold; 0 disables)")
    p.add_argument("--settle-frames", type=int, default=DEFAULT_SETTLE_FRAMES,
                   help="consecutive frames whose means must agree for the brightness to count as "
                        "settled (default 5, at least 2)")
    p.add_argument("--settle-tol-gray", type=float, default=DEFAULT_SETTLE_TOL_GRAY,
                   help="largest allowed max-min of those frame means, in gray levels (engineering "
                        "epsilon, not a quality threshold)")
    p.add_argument("--backend", choices=("dshow", "default"), default=None,
                   help="OpenCV backend (default: dshow on Windows, otherwise default)")
    p.add_argument("--list-devices", action="store_true",
                   help="probe camera indexes 0..--max-index and print those that deliver a frame; "
                        "captures nothing and needs no --mode/--devices/--name")
    p.add_argument("--max-index", type=int, default=DEFAULT_MAX_INDEX,
                   help=f"highest camera index --list-devices probes (default {DEFAULT_MAX_INDEX})")
    return p


def _settle_line(dev: str, rec: dict) -> str:
    if not rec["checked"]:
        return f"Brightness settle check: disabled (device {dev})"
    if rec["settled"]:
        return (f"Brightness settled after {rec['seconds']:.1f} s (device {dev}, "
                f"{rec['frames_used']} frames used for the check)")
    return f"Brightness NOT settled after {rec['max_s']:g} s (device {dev}); see the warnings"


def _print_summary(summary: dict, out: Callable[[str], Any]) -> None:
    out(f"Session folder: {summary['folder']}")
    per_dev = ", ".join(f"device {d}: {n}" for d, n in summary["frames_per_device"].items())
    out(f"Frames saved: {summary['frames_saved']} ({per_dev})")
    for dev, rec in summary.get("settle", {}).items():
        out(_settle_line(dev, rec))
    for hint in summary.get("hints", []):
        out(f"Hint: {hint}")
    if summary["warnings"]:
        out("Warnings:")
        for w in summary["warnings"]:
            out(f"  - {w}")
    else:
        out("Warnings: none")
    if summary["aborted"]:
        out(f"ABORTED: {summary['aborted']}. Partial data kept; session.json notes say aborted.")
    out(f"Next: python -m tools.report {summary['folder']}")


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    cv2_module: Any = None,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], Any] = print,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Any] = time.sleep,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_devices:
        try:
            found = list_devices(max_index=args.max_index, backend=args.backend,
                                 cv2_module=cv2_module, out=out)
        except (ToolDependencyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0 if found else 1
    _require_capture_args(parser, args)
    try:
        summary = capture_session(
            mode=args.mode,
            devices=args.devices,
            out_dir=args.out,
            name=args.name,
            count=args.count,
            interval_s=args.interval_s,
            warmup_s=args.warmup_s,
            width=args.width,
            height=args.height,
            fps=args.fps,
            fourcc=args.fourcc,
            focus=args.focus,
            exposure=args.exposure,
            white_balance=args.white_balance,
            gain=args.gain,
            part=args.part,
            lighting=args.lighting,
            operator=args.operator,
            note=args.note,
            no_prompt=args.no_prompt,
            sample_every=args.sample_every,
            flush_frames=args.flush_frames,
            response_check=not args.no_response_check,
            settle_max_s=args.settle_max_s,
            settle_frames=args.settle_frames,
            settle_tol_gray=args.settle_tol_gray,
            backend=args.backend,
            cv2_module=cv2_module,
            input_fn=input_fn,
            out=out,
            clock=clock,
            sleep=sleep,
        )
    except (ToolDependencyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_summary(summary, out)
    return 1 if summary["aborted"] else 0


if __name__ == "__main__":
    sys.exit(main())
