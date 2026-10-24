"""Command line for the weekly trainer: ``python3 -m trainer <command> ...``.

Commands live in COMMANDS: name -> (add_arguments(subparsers), run(args) -> int).
Add a command by writing both functions and one entry in that dict.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Optional, Sequence

from trainer.dataset import GOOD_POLICIES, Dataset, iter_parts

PROG = "python3 -m trainer"
PROBLEM_LINES_SHOWN = 5


def _add_inventory(sub: Any) -> None:
    p = sub.add_parser(
        "inventory",
        help="count parts and images per part number, camera, side and label",
    )
    p.add_argument("--store", required=True, metavar="PATH",
                   help="folder that holds store/ (or the store/ folder itself)")
    p.add_argument("--part", metavar="PN", help="only this part number")
    p.add_argument("--since", metavar="YYYY-MM-DD", help="inclusive, UTC date of started_utc")
    p.add_argument("--until", metavar="YYYY-MM-DD", help="exclusive, UTC date of started_utc")
    p.add_argument("--good-policy", choices=GOOD_POLICIES, default="strict",
                   help="strict (default) or include-overridden")


def _print_dataset(ds: Dataset) -> None:
    summary = ds.summary()
    print(f"part {ds.part_number}")
    print(f"  parts: good {summary['parts_good']}, bad {summary['parts_bad']}, "
          f"excluded {summary['parts_excluded']}  (good policy: {ds.good_policy})")
    print(f"  {'camera':<8}{'side':<8}{'label':<8}{'images':>6}")
    for (camera, side, label), count in ds.counts().items():
        print(f"  {camera:<8}{side:<8}{label:<8}{count:>6}")
    print()


def _run_inventory(args: argparse.Namespace) -> int:
    problems: list[str] = []
    try:
        samples = list(iter_parts(
            args.store,
            part_number=args.part,
            since=args.since,
            until=args.until,
            full_station_only=True,
            verify=True,
            problems=problems,
        ))
        groups: dict = {}
        for sample in samples:
            groups.setdefault(sample.part_number, []).append(sample)
        datasets = [
            Dataset.from_samples(pn, groups[pn], good_policy=args.good_policy)
            for pn in sorted(groups)
        ]
    except (OSError, ValueError) as exc:
        print(f"inventory error: {exc}", file=sys.stderr)
        return 2
    if not datasets:
        print("no usable parts found")
    for ds in datasets:
        _print_dataset(ds)
    # Problems are store-wide: an unreadable part.json cannot be tied to one part.
    print(f"problems: {len(problems)}")
    for line in problems[:PROBLEM_LINES_SHOWN]:
        print(f"  {line}")
    return 0


# name -> (add_arguments, run)
COMMANDS: dict = {
    "inventory": (_add_inventory, _run_inventory),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Weekly trainer tools: read the station evidence store.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    for add_arguments, _run in COMMANDS.values():
        add_arguments(sub)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits on --help and on usage errors
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 2
    _add, run = COMMANDS[args.command]
    result: Callable[[argparse.Namespace], int] = run
    return result(args)


if __name__ == "__main__":
    raise SystemExit(main())
