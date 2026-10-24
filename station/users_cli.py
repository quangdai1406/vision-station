"""`python -m station users add <username> <role>`: create or update a station login.

The password is read with getpass (asked twice), hashed with auth.hash_password and
written to the users JSON file. Neither the password nor the hash is printed.
The file is written atomically with mode 0600.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import pathlib
import sys
import tempfile
from typing import Callable, Optional, Sequence, TextIO

from station.auth import ROLES, hash_password

MIN_PASSWORD_LENGTH = 8
MAX_USERNAME_LENGTH = 64
DEFAULT_CONFIG = "config.json"


def _valid_username(name: str) -> bool:
    return (
        0 < len(name) <= MAX_USERNAME_LENGTH
        and not any(ch.isspace() for ch in name)
        and name.isprintable()
    )


def _users_path(args: argparse.Namespace, err: TextIO) -> Optional[pathlib.Path]:
    if args.users_file:
        return pathlib.Path(args.users_file)
    # Imported here so that `users` works without reading the whole config schema first.
    from station.app import ConfigError, load_config

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=err)
        print("hint: pass --users-file FILE to write the users file without a config", file=err)
        return None
    return pathlib.Path(cfg["users_file"])


def _load_existing(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))  # raises on corrupt JSON: never overwrite it
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return data


def _write_atomic(path: pathlib.Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".users-", suffix=".tmp")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(
    argv: Sequence[str],
    *,
    getpass_fn: Callable[[str], str] = getpass.getpass,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m station users",
                                     description="Manage station logins.")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="add a user, or change an existing user's role and password")
    add.add_argument("username")
    add.add_argument("role", choices=ROLES, help="operator, supervisor or admin")
    where = add.add_mutually_exclusive_group()
    where.add_argument("--config", default=DEFAULT_CONFIG,
                       help=f"config file whose users_file is used (default: {DEFAULT_CONFIG})")
    where.add_argument("--users-file", help="users JSON file to write, instead of the config's")
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return int(exc.code or 0)

    if not _valid_username(args.username):
        print("username must be 1-64 printable characters with no spaces", file=err)
        return 1

    path = _users_path(args, err)
    if path is None:
        return 1

    password = getpass_fn("Password: ")
    if password != getpass_fn("Repeat password: "):
        print("passwords do not match; nothing was written", file=err)
        return 1
    if len(password) < MIN_PASSWORD_LENGTH:
        print(f"password must be at least {MIN_PASSWORD_LENGTH} characters; nothing was written", file=err)
        return 1

    try:
        users = _load_existing(path)
    except (OSError, ValueError) as exc:
        print(f"cannot read users file {path}: {exc}; it was not changed", file=err)
        return 1

    existed = args.username in users
    users[args.username] = {"password_hash": hash_password(password), "role": args.role}
    try:
        _write_atomic(path, users)
    except OSError as exc:
        print(f"cannot write users file {path}: {exc.strerror or exc}", file=err)
        return 1

    verb = "updated" if existed else "added"
    print(f"{verb} user {args.username!r} with role {args.role} in {path}", file=out)
    return 0
