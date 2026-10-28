"""Command line for the weekly trainer: ``python3 -m trainer <command> ...``.

Commands live in COMMANDS: name -> (add_arguments(subparsers), run(args) -> int).
Add a command by writing both functions and one entry in that dict.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any, Callable, Optional, Sequence

from trainer import weekly
from trainer.dataset import GOOD_POLICIES, Dataset, iter_parts
from trainer.scans import BAD, GOOD, ScanSet, has_scans, read_scans, summarize

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
    if has_scans(args.store):
        _print_scans(read_scans(args.store))
    return 0


def _print_scans(scan_set: ScanSet) -> None:
    """Taking-data scans, after the part output. Part filters (--part, --since, --until) do not apply."""
    inv = summarize(scan_set)
    print(f"scans ({inv['scans_root']}; all scans, --part/--since/--until do not apply)")
    print(f"  folders {inv['folders']}, intact {inv['intact']}, problems {inv['problems']}")
    labels, statuses = inv["by_label"], inv["by_status"]
    print(f"  label GOOD {labels[GOOD]}, BAD {labels[BAD]}; status official {statuses['official']}, "
          f"proposed {statuses['proposed']}, rejected {statuses['rejected']}")
    print(f"  {'toy':<16}{'GOOD':>6}{'BAD':>6}")
    for toy, row in inv["by_toy"].items():
        print(f"  {toy:<16}{row[GOOD]:>6}{row[BAD]:>6}")
    sides = inv["scans_by_side"]
    print(f"  scans with TOP {sides['TOP']}, BOTTOM {sides['BOTTOM']} (counts may differ per side)")
    intact, training = inv["images_by_camera_side"], inv["training_images_by_camera_side"]
    print(f"  {'camera/side':<12}{'intact':>8}{'training':>10}")
    for key in sorted(set(intact) | set(training)):
        print(f"  {key:<12}{intact.get(key, 0):>8}{training.get(key, 0):>10}")
    excluded = inv["excluded"]
    print(f"  excluded {excluded['total']}")
    for reason, n in excluded["by_reason"].items():
        print(f"    {n} x {reason}")
    for name, row in inv["bad_defects_official"].items():
        print(f"  BAD defect {name}: {row['scans']} scans, pictures defect visible "
              f"{row['visible_true']}, not visible {row['visible_false']}")
    print(f"scan problems: {len(scan_set.problems)}")
    for line in scan_set.problems[:PROBLEM_LINES_SHOWN]:
        print(f"  {line}")


def _lazy_module(module_name: str) -> tuple[Any, Optional[str]]:
    """Import a command module by name. (module, None) or (None, reason) when it is missing."""
    try:
        return importlib.import_module(module_name), None
    except ImportError as exc:
        return None, str(exc)


def _first_callable(module: Any, names: Sequence[str]) -> Optional[Callable[..., Any]]:
    if module is None:
        return None
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def _lazy_command(command: str, module_name: str, add_names: Sequence[str],
                  run_names: Sequence[str]) -> tuple[Callable[[Any], None], Callable[[argparse.Namespace], int]]:
    """Wire a command whose module may not exist yet. The module is imported when the command is
    registered and again when it runs; a missing module gives a clear error instead of breaking the CLI."""

    def add_arguments(sub: Any) -> None:
        module, _reason = _lazy_module(module_name)
        add = _first_callable(module, add_names)
        if add is not None:
            add(sub)
            return
        placeholder = sub.add_parser(command, help=f"not available: {module_name} is missing", add_help=False)
        placeholder.add_argument("rest", nargs=argparse.REMAINDER)

    def run(args: argparse.Namespace) -> int:
        module, reason = _lazy_module(module_name)
        fn = _first_callable(module, run_names)
        if fn is None:
            detail = reason or f"{module_name} has none of: {', '.join(run_names)}"
            print(f"{command}: not available ({detail})", file=sys.stderr)
            return 2
        return int(fn(args))

    return add_arguments, run


# name -> (add_arguments(subparsers), run(args) -> int)
COMMANDS: dict = {
    "inventory": (_add_inventory, _run_inventory),
    "weekly": (weekly.add_arguments, weekly.run),
    "purge": _lazy_command("purge", "trainer.purge", ("add_arguments",), ("run",)),
    "promote": _lazy_command("promote", "trainer.promote",
                             ("add_promote_arguments", "add_arguments"), ("run_promote", "run")),
    "rollback": _lazy_command("rollback", "trainer.promote",
                              ("add_rollback_arguments", "add_arguments"), ("run_rollback", "run")),
    "selftest": _lazy_command("selftest", "trainer.selftest", ("add_arguments",), ("run",)),
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
