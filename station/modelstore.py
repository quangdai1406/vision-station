"""Per-part model versions on disk, with human approval.

Layout under the store root::

    <root>/<safe_part>/<version>/<model file>
    <root>/<safe_part>/<version>/manifest.json
    <root>/<safe_part>/current.json       the approved version (written only by approve/rollback)
    <root>/<safe_part>/approvals.jsonl    append-only history of approve and rollback

A station may inspect a part only with that part's approved version. A model is never
approved automatically: approve() needs a named person and a reason. current() raises
ModelError rather than falling back when current.json points at a missing or tampered
version.

Standard library only. Model files are loaded through LOADERS, so importing this module
needs neither numpy nor the golden model.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from station.contracts import StationError

SCHEMA = 1
MANIFEST_NAME = "manifest.json"
CURRENT_NAME = "current.json"
APPROVALS_NAME = "approvals.jsonl"
_RESERVED_NAMES = {MANIFEST_NAME, CURRENT_NAME, APPROVALS_NAME}
_VERSION_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
_TYPE_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
_FOLDER_MAX = 80
_HASH_LEN = 8
_BASE_MAX = _FOLDER_MAX - _HASH_LEN - 1
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {
    f"lpt{i}" for i in range(1, 10)
}

Loader = Callable[["ModelInfo", Any], Any]


class ModelError(StationError):
    """A model version is missing, unverified, not approved, or tampered with."""


@dataclass(frozen=True)
class ModelInfo:
    part_number: str
    version: str
    type: str
    path: Path  # the model file inside the version folder
    manifest: dict
    approved_by: str
    approved_utc: str
    reason: str


# ---------------------------------------------------------------------------
# Naming and small file helpers
# ---------------------------------------------------------------------------

def safe_part(part_number: str) -> str:
    """Folder name for a part number. Always ends in '-' + 8 hex chars of sha256(part_number).

    The suffix is on every name, so two different part numbers that clean to the same text
    still get different folders, and the mapping is unambiguous.
    """
    if not isinstance(part_number, str) or not part_number.strip():
        raise ValueError("part number must be a non-empty string")
    base = re.sub(r"[^A-Za-z0-9._-]", "_", part_number.strip())
    base = re.sub(r"_+", "_", base).strip(" .")[:_BASE_MAX].rstrip(" .")
    if not base:
        base = "part"
    if base.split(".")[0].lower() in _WINDOWS_RESERVED:
        base = "part_" + base
    digest = hashlib.sha256(part_number.encode("utf-8")).hexdigest()[:_HASH_LEN]
    return f"{base}-{digest}"


def _check_version(version: Any) -> str:
    if (
        not isinstance(version, str)
        or not _VERSION_RE.fullmatch(version)
        or version.startswith(".")
        or version.endswith(".")
        or version.split(".")[0].lower() in _WINDOWS_RESERVED
    ):
        raise ModelError(
            f"bad version name {version!r}: use 1 to 64 letters, digits, '.', '_' or '-'"
        )
    return version


def _check_type(type_name: Any) -> str:
    if not isinstance(type_name, str) or not _TYPE_RE.fullmatch(type_name):
        raise ValueError(f"model type must be a lowercase word, got {type_name!r}")
    return type_name


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _bad_file_name(name: str) -> bool:
    return (
        name in ("", ".", "..")
        or name in _RESERVED_NAMES
        or "/" in name
        or "\\" in name
        or Path(name).name != name
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    """Write to a temp file in the same folder, fsync, then os.replace. No .tmp is left behind."""
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    except OSError as exc:
        raise ModelError(f"could not write {path.name}: {exc}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        _unlink_quietly(tmp)
        raise ModelError(f"could not write {path.name}: {exc}") from exc
    except BaseException:
        _unlink_quietly(tmp)
        raise


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _json_text(data: Any) -> str:
    try:
        return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data is not JSON-serialisable: {exc}") from exc


def _append_line(path: Path, line: str) -> None:
    """One write call, O_APPEND, trailing newline included in the data."""
    data = line.encode("utf-8")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(str(path), flags, 0o644)
        try:
            written = os.write(fd, data)
        finally:
            os.close(fd)
    except OSError as exc:
        raise ModelError(f"could not append to {path.name}: {exc}") from exc
    if written != len(data):
        raise ModelError(f"short write to {path.name}; the approval log may be damaged")


def _load_manifest(vdir: Path) -> Optional[dict]:
    try:
        data = json.loads((vdir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ModelStore:
    """Model versions for many parts under one root. Nothing is cached: every call reads the disk,
    so two instances on the same root see each other's changes."""

    def __init__(self, root: Any) -> None:
        self.root = Path(root)

    def _part_dir(self, part_number: str) -> Path:
        return self.root / safe_part(part_number)

    def _version_dir(self, part_number: str, version: str) -> Path:
        return self._part_dir(part_number) / _check_version(version)

    def add_candidate(
        self,
        part_number: str,
        version: str,
        model_file: Path,
        manifest: dict,
        now_utc: str,
    ) -> Path:
        """Copy a model file into a new version folder as a 'candidate'. Never changes current.json.

        Returns the path of the copied model file. Refuses to overwrite an existing version.
        """
        pdir = self._part_dir(part_number)
        _check_version(version)
        src = Path(model_file)
        if not src.is_file():
            raise ModelError(f"model file not found: {src}")
        name = src.name
        if _bad_file_name(name):
            raise ValueError(f"model file name {name!r} is not allowed")
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a dict")
        mtype = _check_type(manifest.get("type"))
        created = _required_text(now_utc, "now_utc")
        camera_ids = manifest.get("camera_ids", [])
        if not isinstance(camera_ids, list) or not all(isinstance(c, str) for c in camera_ids):
            raise ValueError("camera_ids must be a list of strings")
        trained_on = manifest.get("trained_on", {})
        metrics = manifest.get("metrics", {})
        notes = manifest.get("notes", "")
        if not isinstance(trained_on, dict) or not isinstance(metrics, dict):
            raise ValueError("trained_on and metrics must be dicts")
        if not isinstance(notes, str):
            raise ValueError("notes must be a string")

        vdir = pdir / version
        if vdir.exists():
            raise ModelError(f"version {version} already exists for part {part_number}")
        # Everything is built in a staging folder, then renamed into place.
        record = {
            "schema": SCHEMA,
            "part_number": part_number,
            "version": version,
            "type": mtype,
            "created_utc": created,
            "model_file": name,
            "sha256": None,  # filled in after the copy
            "camera_ids": list(camera_ids),
            "trained_on": dict(trained_on),
            "metrics": dict(metrics),
            "status": "candidate",
            "notes": notes,
        }
        try:
            pdir.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".new-", dir=str(pdir)))
        except OSError as exc:
            raise ModelError(f"could not create version folder for {part_number}: {exc}") from exc
        try:
            dest = staging / name
            shutil.copyfile(src, dest)
            record["sha256"] = _sha256_file(dest)
            text = _json_text(record)
            _write_text_atomic(staging / MANIFEST_NAME, text)
            if vdir.exists():
                raise ModelError(f"version {version} already exists for part {part_number}")
            os.rename(staging, vdir)
        except OSError as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise ModelError(f"could not store model version {version}: {exc}") from exc
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return vdir / name

    def list_versions(self, part_number: str) -> list[dict]:
        """Manifests of this part's versions, sorted by created_utc then version."""
        pdir = self._part_dir(part_number)
        if not pdir.is_dir():
            return []
        out = []
        for child in pdir.iterdir():
            if child.name.startswith(".") or not child.is_dir():
                continue
            manifest = _load_manifest(child)
            if manifest is not None:
                out.append(manifest)
        out.sort(key=lambda m: (str(m.get("created_utc", "")), str(m.get("version", ""))))
        return out

    def verify(self, part_number: str, version: str) -> list[str]:
        """Problems with one version. An empty list means it is intact."""
        try:
            vdir = self._version_dir(part_number, version)
        except (ModelError, ValueError) as exc:
            return [str(exc)]
        if not vdir.is_dir():
            return [f"version {version} not found"]
        if not (vdir / MANIFEST_NAME).is_file():
            return ["manifest.json is missing"]
        manifest = _load_manifest(vdir)
        if manifest is None:
            return ["manifest.json is not valid JSON"]
        problems = []
        if manifest.get("part_number") != part_number:
            problems.append("manifest part_number does not match")
        if manifest.get("version") != version:
            problems.append("manifest version does not match the folder")
        name = manifest.get("model_file")
        if not isinstance(name, str) or _bad_file_name(name):
            return problems + ["manifest model_file is not a plain file name"]
        model = vdir / name
        if not model.is_file():
            return problems + [f"model file {name} is missing"]
        try:
            actual = _sha256_file(model)
        except OSError as exc:
            return problems + [f"model file {name} cannot be read: {exc}"]
        if actual != manifest.get("sha256"):
            problems.append("model file sha256 does not match the manifest")
        return problems

    def approve(self, part_number: str, version: str, by: str, reason: str, now_utc: str) -> ModelInfo:
        """A named person makes this version the approved one for the part."""
        by_text = _required_text(by, "by")
        reason_text = _required_text(reason, "reason")
        return self._activate(part_number, version, by_text, reason_text, now_utc, "approve")

    def rollback(self, part_number: str, by: str, reason: str, now_utc: str) -> ModelInfo:
        """Re-approve the nearest version below the current one on the activation stack that
        still verifies. Versions rolled back from are off the stack, so they never come back.

        Raises ModelError when nothing is approved now or no earlier version can be used.
        """
        by_text = _required_text(by, "by")
        reason_text = _required_text(reason, "reason")
        _required_text(now_utc, "now_utc")
        current_data = self._read_current_raw(part_number)
        if current_data is None:
            raise ModelError(f"no approved model for part {part_number}: nothing to roll back from")
        current_version = current_data["version"]
        stack = self._stack(part_number)
        # Nearest first.
        candidates: list[str] = (
            list(reversed(stack[: stack.index(current_version)])) if current_version in stack else []
        )
        for version in candidates:
            if not self.verify(part_number, version):
                return self._activate(part_number, version, by_text, reason_text, now_utc, "rollback")
        raise ModelError(
            f"no earlier approved model for part {part_number} still verifies: nothing to roll back to"
        )

    def current(self, part_number: str) -> Optional[ModelInfo]:
        """The approved model, None if nothing was ever approved, ModelError if it is unusable."""
        data = self._read_current_raw(part_number)
        if data is None:
            return None
        version = data["version"]
        problems = self.verify(part_number, version)
        if problems:
            raise ModelError(
                f"approved model {version} for part {part_number} is not usable: " + "; ".join(problems)
            )
        vdir = self._version_dir(part_number, version)
        manifest = _load_manifest(vdir)
        if manifest is None or manifest.get("status") != "approved":
            status = None if manifest is None else manifest.get("status")
            raise ModelError(
                f"approved model {version} for part {part_number} is not marked approved (status {status!r})"
            )
        return _info(part_number, version, vdir, manifest, data["approved_by"],
                     data["approved_utc"], data["reason"])

    def approvals(self, part_number: str) -> list[dict]:
        """The approval log for a part, oldest first."""
        path = self._part_dir(part_number) / APPROVALS_NAME
        if not path.exists():
            return []
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ModelError(f"cannot read the approval log for part {part_number}: {exc}") from exc
        out = []
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                raise ModelError(f"approval log for part {part_number}: line {number} is not valid JSON") from None
            if not isinstance(entry, dict):
                raise ModelError(f"approval log for part {part_number}: line {number} is not an object")
            out.append(entry)
        return out

    def history(self, part_number: str) -> list[str]:
        """The activation stack for a part, oldest first and the current version last.

        Built by replaying the approval log; 'failed' lines cancel the attempt they name.
        """
        return self._stack(part_number)

    # -- internals ----------------------------------------------------------

    def _stack(self, part_number: str) -> list[str]:
        stack = _replay(self.approvals(part_number))
        current = self._read_current_raw(part_number)
        # The log is written before current.json, so after a crash it can lead by one
        # unfinished line. current.json is the truth for what is active.
        if current is not None and current["version"] in stack:
            stack = stack[: stack.index(current["version"]) + 1]
        return stack

    def _log_failed_switch(self, part_number: str, version: str, action: str, now: str,
                           exc: BaseException) -> None:
        entry = {
            "action": "failed",
            "part_number": part_number,
            "version": version,
            "intended": action,
            "error": str(exc)[:300],
            "utc": now,
        }
        try:
            _append_line(self._part_dir(part_number) / APPROVALS_NAME,
                         json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        except ModelError:
            pass  # best effort: the stack is still anchored to current.json

    def _read_current_raw(self, part_number: str) -> Optional[dict]:
        path = self._part_dir(part_number) / CURRENT_NAME
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            raise ModelError(f"current.json for part {part_number} is unreadable") from None
        required = ("part_number", "version", "approved_by", "approved_utc", "reason")
        if not isinstance(data, dict) or any(not isinstance(data.get(k), str) for k in required):
            raise ModelError(f"current.json for part {part_number} is malformed")
        if data["part_number"] != part_number:
            raise ModelError(f"current.json for part {part_number} names another part")
        return data

    def _set_status(self, part_number: str, version: str, status: str) -> None:
        vdir = self._version_dir(part_number, version)
        manifest = _load_manifest(vdir)
        if manifest is None or manifest.get("status") == status:
            return
        manifest["status"] = status
        _write_text_atomic(vdir / MANIFEST_NAME, _json_text(manifest))

    def _activate(self, part_number: str, version: str, by: str, reason: str,
                  now_utc: str, action: str) -> ModelInfo:
        now = _required_text(now_utc, "now_utc")
        vdir = self._version_dir(part_number, version)
        problems = self.verify(part_number, version)
        if problems:
            raise ModelError(
                f"version {version} for part {part_number} cannot be approved: " + "; ".join(problems)
            )
        previous = self._read_current_raw(part_number)
        previous_version = previous["version"] if previous else None
        manifest = _load_manifest(vdir)
        if manifest is None:  # verify() passed, so this is a race with an outside edit
            raise ModelError(f"manifest for version {version} changed while approving")
        old_status = manifest.get("status")
        entry = {
            "action": action,
            "part_number": part_number,
            "version": version,
            "by": by,
            "reason": reason,
            "utc": now,
            "previous_version": previous_version,
        }
        # 1. The intended action goes to the log first, so the log never lags current.json.
        _append_line(self._part_dir(part_number) / APPROVALS_NAME,
                     json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        # 2. Then switch current.json. If the switch fails, a 'failed' line cancels the intent.
        manifest_changed = False
        try:
            manifest["status"] = "approved"
            _write_text_atomic(vdir / MANIFEST_NAME, _json_text(manifest))
            manifest_changed = True
            current = {
                "part_number": part_number,
                "version": version,
                "approved_by": by,
                "approved_utc": now,
                "reason": reason,
            }
            _write_text_atomic(self._part_dir(part_number) / CURRENT_NAME, _json_text(current))
        except Exception as exc:
            if manifest_changed:
                manifest["status"] = old_status
                try:
                    _write_text_atomic(vdir / MANIFEST_NAME, _json_text(manifest))
                except Exception:
                    pass  # best effort; the status is not what decides what is active
            self._log_failed_switch(part_number, version, action, now, exc)
            raise ModelError(
                f"could not switch the approved model for part {part_number} to version {version}: {exc}"
            ) from exc
        # 3. Retire the old version. Cosmetic: current.json already names the new one.
        if previous_version is not None and previous_version != version:
            try:
                self._set_status(part_number, previous_version, "retired")
            except (ModelError, OSError):
                pass
        return _info(part_number, version, vdir, manifest, by, now, reason)


_ACTIVATIONS = ("approve", "rollback")


def _replay(entries: list[dict]) -> list[str]:
    """The activation stack (oldest first, current last) from the raw approval log.

    approve: the version moves to the top of the stack.
    rollback: the version rolled back from is dropped and the stack is cut down to the version
    rolled back to, so everything above it goes.
    failed: cancels the nearest earlier unresolved approve/rollback for the same version.
    """
    cancelled: set[int] = set()
    open_intents: list[int] = []
    for idx, entry in enumerate(entries):
        action = entry.get("action")
        if action in _ACTIVATIONS:
            open_intents.append(idx)
        elif action == "failed":
            for pos in range(len(open_intents) - 1, -1, -1):
                if entries[open_intents[pos]].get("version") == entry.get("version"):
                    cancelled.add(open_intents.pop(pos))
                    break
    stack: list[str] = []
    for idx, entry in enumerate(entries):
        version = entry.get("version")
        if idx in cancelled or entry.get("action") not in _ACTIVATIONS or not isinstance(version, str):
            continue
        if version in stack:
            if entry["action"] == "approve":
                stack.remove(version)
                stack.append(version)
            else:
                stack = stack[: stack.index(version) + 1]
        else:
            stack.append(version)
    return stack


def _info(part_number: str, version: str, vdir: Path, manifest: dict,
          approved_by: str, approved_utc: str, reason: str) -> ModelInfo:
    return ModelInfo(
        part_number=part_number,
        version=version,
        type=str(manifest.get("type", "")),
        path=vdir / str(manifest["model_file"]),
        manifest=dict(manifest),
        approved_by=approved_by,
        approved_utc=approved_utc,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Runtime loaders
# ---------------------------------------------------------------------------

LOADERS: dict[str, Loader] = {}


def register_loader(type_name: str, fn: Loader) -> None:
    """Register how to turn a model file of this type into the station's InferenceModel."""
    _check_type(type_name)
    LOADERS[type_name] = fn


def _load_golden(info: ModelInfo, decode: Any) -> Any:
    from station.golden import GoldenModel  # noqa: PLC0415 - lazy import, keeps this module light

    return GoldenModel.load(str(info.path), decode)


register_loader("golden", _load_golden)


def load_runtime_model(info: ModelInfo, decode: Any) -> Any:
    """Build the station's model object for an approved version. ModelError if no loader fits."""
    fn = LOADERS.get(info.type)
    if fn is None:
        raise ModelError(f"no loader for model type {info.type}: is the optional package installed?")
    try:
        return fn(info, decode)
    except StationError:
        raise
    except Exception as exc:
        raise ModelError(
            f"could not load {info.type} model {info.version} for part {info.part_number}: {exc}"
        ) from exc
