"""`python -m station teach`: learn the golden model from GOOD parts on the real cameras.

    python -m station teach --config config.json [--frames-per-side N] [--parts K]
                            [--out PATH] [--force]

DEMO HEURISTIC for the proof of concept: the golden model is learned from good
parts only and its scores are not validated. Standard library at module level;
numpy and cv2 are imported lazily by station.golden. Nothing is written until
every part has been captured.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from station import golden
from station.app import ConfigError, StationApp, load_config
from station.contracts import Side, StationError

DEFAULT_FRAMES_PER_SIDE = 5  # engineering setting, not a quality limit
DEFAULT_PARTS = 3  # engineering setting, not a quality limit
DEFAULT_PAUSE_S = 0.3
REMINDER = (
    "Learned from GOOD parts only. This is a demo heuristic for a proof of concept: "
    "scores are not validated and thresholds still have to be chosen from labelled data."
)
_TOP_PROMPT = "Part {k} of {parts}: place a GOOD part with the TOP side up, keep it still, then press Enter"
_BOTTOM_PROMPT = "Part {k} of {parts}: flip it to the BOTTOM side, keep it still, then press Enter"
_SIDE_PROMPTS = ((Side.TOP, _TOP_PROMPT), (Side.BOTTOM, _BOTTOM_PROMPT))


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _at_least(minimum: int, label: str) -> Callable[[str], int]:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{label} must be a whole number, got {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"{label} must be at least {minimum}, got {value}")
        return value

    return parse


def add_teach_arguments(parser: argparse.ArgumentParser) -> None:
    """The teach options, shared by `python -m station teach` and its tests."""
    parser.add_argument("--config", required=True, help="path to the station config (JSON); cameras come from it")
    parser.add_argument("--frames-per-side", type=_at_least(2, "--frames-per-side"),
                        default=DEFAULT_FRAMES_PER_SIDE, metavar="N",
                        help=f"frames per camera and side for each part (default {DEFAULT_FRAMES_PER_SIDE}, at least 2)")
    parser.add_argument("--parts", type=_at_least(1, "--parts"), default=DEFAULT_PARTS, metavar="K",
                        help=f"number of GOOD parts to photograph (default {DEFAULT_PARTS}, at least 1)")
    parser.add_argument("--out", metavar="PATH", default=None,
                        help="output file (default: model.path from the config)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")


def load_teach_config(path: str | os.PathLike) -> dict:
    """Load the station config (model.kind 'golden' is accepted by app.load_config)."""
    return load_config(path)


def teach_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m station teach",
        description="Learn the golden model from GOOD parts on the real cameras (demo heuristic).",
    )
    add_teach_arguments(parser)
    args = parser.parse_args(list(argv))
    try:
        config = load_teach_config(args.config)
    except ConfigError as exc:
        _err(f"config error: {exc}")
        return 2
    return run_teach(config, frames_per_side=args.frames_per_side, parts=args.parts,
                     out=args.out, force=args.force)


def _release(cameras: Any) -> None:
    release = getattr(cameras, "release", None)
    if callable(release):
        try:
            release()
        except Exception as exc:  # the model is already saved or the run already failed
            _err(f"teach warning: could not release the cameras: {exc}")


def _summary(out: Any, counts: Mapping[str, int], described: Mapping[str, Any],
             write: Callable[[str], Any]) -> None:
    write(f"Golden model saved to {out}")
    write("Frames per camera and side:")
    for key in sorted(counts):
        write(f"  {key}: {counts[key]} frames")
    write("Model constants:")
    for name, value in described.items():
        if name != "keys":
            write(f"  {name}: {value}")
    write(REMINDER)


def run_teach(
    config: Mapping[str, Any],
    *,
    frames_per_side: int = DEFAULT_FRAMES_PER_SIDE,
    parts: int = DEFAULT_PARTS,
    out: Optional[str] = None,
    force: bool = False,
    cameras=None,
    builder=None,
    input_fn: Callable[[str], str] = input,
    sleep: Callable[[float], Any] = time.sleep,
    write: Callable[[str], Any] = print,
) -> int:
    """Photograph K good parts (top and bottom), learn the golden model and save it.

    config: a validated config dict with model.kind 'golden'. out defaults to model.path.
    cameras and builder are injectable for tests; by default they are built from config.
    Returns 0 on success, 2 for a usage or config problem (nothing built), 1 for a
    camera or build error, or an aborted prompt. Nothing is written unless every part succeeded.
    """
    kind = config["model"]["kind"]
    if kind != "golden":
        _err(f"teach: model.kind must be 'golden' to learn a golden model, not {kind!r}")
        return 2
    if frames_per_side < 2:
        _err("teach: --frames-per-side must be at least 2")
        return 2
    if parts < 1:
        _err("teach: --parts must be at least 1")
        return 2
    out_path = out if out is not None else config["model"].get("path")
    if not out_path:
        _err("teach: no output file: pass --out or set model.path in the config")
        return 2
    if os.path.exists(out_path) and not force:
        _err(f"teach: {out_path} already exists; pass --force to overwrite it")
        return 2

    counts: dict[str, int] = {}
    captures = 0
    try:
        if cameras is None:
            cameras = StationApp._make_cameras(config)
        if builder is None:
            builder = golden.GoldenBuilder(golden.cv2_decoder())
        for k in range(1, parts + 1):
            for side, template in _SIDE_PROMPTS:
                write(template.format(k=k, parts=parts))
                input_fn("")
                for _ in range(frames_per_side):
                    if captures:
                        sleep(DEFAULT_PAUSE_S)
                    captures += 1
                    for frame in cameras.capture(side):
                        builder.add(frame)
                        key = f"{frame.camera_id}/{frame.side.value}"
                        counts[key] = counts.get(key, 0) + 1
        model = builder.build(version=config["model"]["version"])
        model.save(out_path)
        _summary(out_path, counts, model.describe(), write)
    except (EOFError, KeyboardInterrupt):
        _err("teach aborted: nothing was written")
        return 1
    except (StationError, ValueError) as exc:
        _err(f"teach error: {exc}")
        return 1
    finally:
        _release(cameras)
    return 0
