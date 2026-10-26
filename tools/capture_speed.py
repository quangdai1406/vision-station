"""Timing probe for the keep-open camera session: N rounds of one picture per camera.

    python -m tools.capture_speed --devices 0 2 4 6 --width 1280 --height 720 \\
        --mjpg --threaded --flush 1 --rounds 20

Prints, per round, the wall time of the four pictures, each camera's time and
mean brightness, then a summary. numpy and cv2 are imported lazily.

Frame-age check (a proxy, not a measurement): the 'grab wait' of a camera is how
long the timed grab blocked after the flush. A wait near zero means a frame was
already queued, so it may be older than the shot; a wait of about one frame
period means the grab waited for a fresh frame. No threshold is applied here.
The wall clock cannot tell the true age of a frame.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any, Callable, Optional, Sequence

from tools import ToolDependencyError


def _decode_mean(data: bytes) -> float:
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        raise ToolDependencyError("numpy and OpenCV (cv2) are needed for the capture-speed probe") from exc
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return float(img.mean()) if img is not None else float("nan")


def run_rounds(
    cams: Any,
    rounds: int,
    *,
    decode_mean: Callable[[bytes], float] = _decode_mean,
    clock: Callable[[], float] = time.monotonic,
    pause_s: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> dict:
    """Run `rounds` captures; return a summary dict. A failing round is counted, not fatal."""
    ids = list(cams.camera_ids)
    rows: list[dict] = []
    failures: dict[str, int] = {c: 0 for c in ids}
    for n in range(rounds):
        if n and pause_s:
            sleep(pause_s)
        t0 = clock()
        error: Optional[str] = None
        frames: Sequence = ()
        try:
            frames = cams.capture("TOP")
        except Exception as exc:
            error = str(exc)
        wall = clock() - t0
        row: dict = {"round": n + 1, "wall_s": wall, "error": error, "mean": {}, "cam_s": {}, "wait_s": {}}
        if error is None:
            for f in frames:
                row["mean"][f.camera_id] = decode_mean(f.data)
            for key, attr in (("cam_s", "last_worker_times_s"), ("wait_s", "last_capture_times_s")):
                vals = getattr(cams, attr, None)
                if vals:
                    row[key] = dict(zip(ids, vals))
        else:
            for c in getattr(cams, "last_failed_cameras", None) or []:
                failures[c] = failures.get(c, 0) + 1
        rows.append(row)
        log(_format_row(row))
    return _summary(rows, ids, failures, log)


def _fmt(d: dict, ids: Sequence[str], spec: str, scale: float = 1.0) -> str:
    return " ".join(f"{c}={format(d[c] * scale, spec)}" if c in d else f"{c}=--" for c in ids)


def _format_row(row: dict) -> str:
    ids = sorted(set(row["mean"]) | set(row["cam_s"]) | set(row["wait_s"]))
    if row["error"]:
        return f"round {row['round']:3d}  {row['wall_s']:.3f}s  FAILED: {row['error']}"
    return (f"round {row['round']:3d}  {row['wall_s']:.3f}s  cam_s[{_fmt(row['cam_s'], ids, '.2f')}] "
            f"grab_wait_ms[{_fmt(row['wait_s'], ids, '.0f', 1000)}] mean[{_fmt(row['mean'], ids, '.1f')}]")


def _summary(rows: list[dict], ids: Sequence[str], failures: dict, log: Callable[[str], None]) -> dict:
    ok = [r for r in rows if r["error"] is None]
    walls = [r["wall_s"] for r in ok]
    out: dict = {
        "rounds": len(rows), "ok_rounds": len(ok), "failed_rounds": len(rows) - len(ok),
        "failures_by_camera": failures,
        "mean_s": statistics.fmean(walls) if walls else None,
        "worst_s": max(walls) if walls else None,
        "best_s": min(walls) if walls else None,
        "brightness": {c: statistics.fmean([r["mean"][c] for r in ok if c in r["mean"]])
                       for c in ids if any(c in r["mean"] for r in ok)},
        "min_grab_wait_s": {c: min(r["wait_s"][c] for r in ok if c in r["wait_s"])
                            for c in ids if any(c in r["wait_s"] for r in ok)},
    }
    log(f"SUMMARY rounds={out['rounds']} ok={out['ok_rounds']} failed={out['failed_rounds']}")
    if walls:
        log(f"  four pictures: mean {out['mean_s']:.3f}s  worst {out['worst_s']:.3f}s  best {out['best_s']:.3f}s")
    log(f"  failures by camera: {failures}")
    log("  mean brightness: " + _fmt(out["brightness"], ids, ".1f"))
    log("  min grab wait ms: " + _fmt(out["min_grab_wait_s"], ids, ".0f", 1000))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--devices", type=int, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--pause", type=float, default=1.0, help="idle seconds between rounds")
    ap.add_argument("--mjpg", action="store_true")
    ap.add_argument("--threaded", action="store_true")
    ap.add_argument("--flush", type=int, default=1)
    args = ap.parse_args(argv)
    from station.cameras import OpenCvCameraSet  # noqa: PLC0415

    cams = OpenCvCameraSet(args.devices, width=args.width, height=args.height,
                           session_mode="keep_open", mjpg=args.mjpg, threaded=args.threaded,
                           stale_flush_frames=args.flush)
    t0 = time.monotonic()
    try:
        cams.open()
    except Exception as exc:
        print(f"open failed: {exc}")
        return 1
    print(f"open: {time.monotonic() - t0:.2f}s failed={cams.last_failed_cameras} "
          f"{args.width}x{args.height} mjpg={args.mjpg} threaded={args.threaded} flush={args.flush}")
    try:
        run_rounds(cams, args.rounds, pause_s=args.pause)
    finally:
        cams.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
