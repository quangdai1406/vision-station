"""Entry point.

    python -m station --config config.json [--demo]
    python -m station users add <username> <role> [--config config.json | --users-file FILE]
    python -m station teach --config config.json [--frames-per-side N] [--parts K] [--out PATH] [--force]
"""
from __future__ import annotations

import argparse
import sys
import threading
from typing import Optional, Sequence

from station.app import ConfigError, StationApp, build_station, load_config

EPILOG = (
    "other commands:\n"
    "  python -m station users add <username> <role> [--config FILE | --users-file FILE]\n"
    "      create a login (roles: operator, supervisor, admin); the password is prompted for\n"
    "  python -m station teach --config FILE [--frames-per-side N] [--parts K] [--out PATH] [--force]\n"
    "      learn the golden model from GOOD parts on the real cameras (demo heuristic)\n"
)


def build_demo_station(config: dict) -> StationApp:
    """Fake cameras, the marker model and console outputs. MES posting is off."""
    from station.cameras import FakeCameraSet
    from station.inference import MarkerModel
    from station.outputs import ConsoleOutputs

    return build_station(
        config,
        cameras=FakeCameraSet(),
        model=MarkerModel(version=config["model"]["version"]),
        outputs=ConsoleOutputs(),
        demo=True,
    )


def _run(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m station",
        description="Camera inspection station: inspect parts, store evidence, post results to the MES.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="path to the station config (JSON)")
    parser.add_argument("--demo", action="store_true",
                        help="use fake cameras, the marker model and console outputs; "
                             "parts are stored but never sent to the MES")
    args = parser.parse_args(list(argv))

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        app = build_demo_station(config) if args.demo else build_station(config)
    except Exception as exc:  # StationError, camera and file errors: one line, no traceback
        print(f"startup error: {exc}", file=sys.stderr)
        return 2

    app.start()
    mode = " (demo: fake cameras, MES posting off)" if args.demo else ""
    print(f"station running{mode}. UI: {app.url}   Ctrl-C to stop", flush=True)
    stop = threading.Event()
    try:
        while not stop.wait(1.0):  # a short timeout keeps Ctrl-C responsive on every platform
            pass
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
    print("station stopped", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "users":
        from station.users_cli import main as users_main

        return users_main(args[1:])
    if args and args[0] == "teach":
        from station.teach import teach_main

        return teach_main(args[1:])
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
