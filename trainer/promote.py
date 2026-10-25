"""Human-approved promotion and rollback of part models: ``python3 -m trainer promote|rollback|versions``.

Promotion is a human decision. The CLI needs a named person (--by) and a reason
(--reason), refuses names that look like automation, verifies the candidate's
sha256 before anything changes, prints what is being approved and what is
approved now, and asks for the word ``yes`` unless --yes is given. The actual
switch is ModelStore.approve / ModelStore.rollback, which write the approval log.

Exit codes: 0 done, 1 refused or error (nothing changed).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from station.modelstore import ModelError, ModelInfo, ModelStore

AUTOMATION_WORDS = frozenset({"trainer", "weekly", "system", "auto", "automation", "scheduled"})
CONFIRM_WORD = "yes"


class Refusal(Exception):
    """The command must stop without changing anything. The message is shown to the operator."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _person(value: Any, label: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise Refusal(f"--{label} is required: give the full name of the person making this decision")
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    if tokens & AUTOMATION_WORDS:
        raise Refusal(
            f"--{label} {text!r} looks like an automated name; a person must approve, not a job or system"
        )
    return text


def _reason(value: Any) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise Refusal("--reason is required: say why this model is approved or rolled back")
    return text


def _store(args: argparse.Namespace) -> ModelStore:
    return ModelStore(args.models)


def _find_record(store: ModelStore, part: str, version: str) -> dict:
    for record in store.list_versions(part):
        if record.get("version") == version:
            return record
    raise Refusal(f"version {version} not found for part {part}")


def _show(label: str, value: Any) -> None:
    print(f"  {label:<16}{value}")


def _print_candidate(record: dict, current: Optional[ModelInfo]) -> None:
    trained = record.get("trained_on") or {}
    print("Candidate to approve:")
    _show("part", record.get("part_number"))
    _show("version", record.get("version"))
    _show("type", record.get("type"))
    _show("status", record.get("status"))
    _show("created (UTC)", record.get("created_utc"))
    _show("sha256", str(record.get("sha256", ""))[:16] + "...")
    _show("parent_version", record.get("parent_version", "(not recorded)"))
    _show("report", record.get("report", trained.get("report", "(not recorded)")))
    _show("cameras", ", ".join(record.get("camera_ids") or []) or "(none)")
    _show("trained_on", json.dumps(trained, sort_keys=True) if trained else "(not recorded)")
    _show("metrics", json.dumps(record.get("metrics") or {}, sort_keys=True))
    if record.get("notes"):
        _show("notes", record["notes"])
    print("Currently approved:")
    if current is None:
        print("  none (nothing approved yet for this part)")
    else:
        print(f"  version {current.version}, approved by {current.approved_by} at {current.approved_utc}")
        print(f"  reason: {current.reason}")


def _confirm(input_fn: Callable[[str], str], prompt: str) -> bool:
    try:
        return input_fn(prompt).strip() == CONFIRM_WORD
    except EOFError:
        return False


def _print_approval_line(store: ModelStore, part: str) -> None:
    entries = store.approvals(part)
    if entries:
        print("Approval log line:")
        print("  " + json.dumps(entries[-1], sort_keys=True, ensure_ascii=False))


def add_promote_arguments(sub: Any) -> None:
    p = sub.add_parser("promote", help="approve a verified candidate model for one part (human decision)")
    p.add_argument("--models", required=True, metavar="PATH", help="model store root")
    p.add_argument("--part", required=True, metavar="PN", help="part number")
    p.add_argument("--version", required=True, metavar="V", help="candidate version to approve")
    p.add_argument("--by", required=True, metavar="NAME", help="full name of the approving person")
    p.add_argument("--reason", required=True, metavar="TEXT", help="why this version is approved")
    p.add_argument("--yes", action="store_true", help="do not ask for the typed 'yes' confirmation")


def run_promote(
    args: argparse.Namespace,
    *,
    now: Optional[str] = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    try:
        by = _person(args.by, "by")
        reason = _reason(args.reason)
        part = args.part
        version = args.version
        store = _store(args)
        problems = store.verify(part, version)
        if problems:
            raise Refusal(f"version {version} failed verification, not approved: " + "; ".join(problems))
        record = _find_record(store, part, version)
        try:
            current = store.current(part)
        except ModelError as exc:
            raise Refusal(f"the currently approved model is unusable, fix that first: {exc}") from exc
        if current is not None and current.version == version:
            raise Refusal(f"version {version} is already the approved model for part {part}")
        _print_candidate(record, current)
        if not args.yes and not _confirm(input_fn, f"Type {CONFIRM_WORD} to approve {version} for {part}: "):
            raise Refusal("not confirmed; nothing was changed")
        info = store.approve(part, version, by, reason, now or utc_now_iso())
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except (ModelError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Approved: part {part} version {info.version} by {info.approved_by} at {info.approved_utc}")
    _print_approval_line(store, part)
    return 0


def add_rollback_arguments(sub: Any) -> None:
    p = sub.add_parser("rollback", help="re-approve the nearest earlier verified model for one part")
    p.add_argument("--models", required=True, metavar="PATH", help="model store root")
    p.add_argument("--part", required=True, metavar="PN", help="part number")
    p.add_argument("--by", required=True, metavar="NAME", help="full name of the approving person")
    p.add_argument("--reason", required=True, metavar="TEXT", help="why the model is rolled back")
    p.add_argument("--yes", action="store_true", help="do not ask for the typed 'yes' confirmation")


def _rollback_target(store: ModelStore, part: str, current_version: str) -> Optional[str]:
    stack = store.history(part)
    if current_version not in stack:
        return None
    for version in reversed(stack[: stack.index(current_version)]):
        if not store.verify(part, version):
            return version
    return None


def run_rollback(
    args: argparse.Namespace,
    *,
    now: Optional[str] = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    try:
        by = _person(args.by, "by")
        reason = _reason(args.reason)
        part = args.part
        store = _store(args)
        try:
            current = store.current(part)
        except ModelError as exc:
            raise Refusal(f"the currently approved model is unusable: {exc}") from exc
        if current is None:
            raise Refusal(f"nothing is approved for part {part}: nothing to roll back from")
        target = _rollback_target(store, part, current.version)
        if target is None:
            raise Refusal(
                f"no earlier approved model for part {part} still verifies: nothing to roll back to"
            )
        print(f"Currently approved: version {current.version}, approved by {current.approved_by} "
              f"at {current.approved_utc}")
        print(f"Rolling back to: version {target}")
        if not args.yes and not _confirm(input_fn, f"Type {CONFIRM_WORD} to roll back {part}: "):
            raise Refusal("not confirmed; nothing was changed")
        info = store.rollback(part, by, reason, now or utc_now_iso())
    except Refusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except (ModelError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Rolled back: part {part} now approved version {info.version} by {info.approved_by} "
          f"at {info.approved_utc}")
    _print_approval_line(store, part)
    return 0


def add_versions_arguments(sub: Any) -> None:
    p = sub.add_parser("versions", help="list a part's model versions, statuses and approval history")
    p.add_argument("--models", required=True, metavar="PATH", help="model store root")
    p.add_argument("--part", required=True, metavar="PN", help="part number")


def run_versions(args: argparse.Namespace) -> int:
    try:
        part = args.part
        store = _store(args)
        records = store.list_versions(part)
        try:
            current = store.current(part)
            print(f"approved: {current.version} (by {current.approved_by} at {current.approved_utc})"
                  if current else "approved: none")
        except ModelError as exc:
            print(f"approved: UNUSABLE ({exc})")
        if not records:
            print(f"no versions for part {part}")
            return 0
        print(f"{'version':<20}{'status':<12}{'created (UTC)':<22}{'check':<10}")
        for record in records:
            version = str(record.get("version"))
            check = "ok" if not store.verify(part, version) else "FAILED"
            print(f"{version:<20}{str(record.get('status')):<12}{str(record.get('created_utc')):<22}{check:<10}")
        print("history (oldest first): " + (" -> ".join(store.history(part)) or "(empty)"))
    except (ModelError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
