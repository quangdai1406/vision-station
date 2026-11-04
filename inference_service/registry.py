"""Which approved model each part uses on the laptop: loaded once, reloaded when it changes.

registry.get(part) returns only the version ModelStore.current() returns, which is sha256
verified. Anything unusable (no approval, tampered file, missing torch, GPU out of memory,
load error) is ModelUnavailable, which the server answers with 503. An older version is
never used in its place.

The cache is keyed by the stat of current.json (inode, mtime, size). A promotion or rollback
replaces that file, so the next request reloads. The model file itself is hashed only when
it is loaded, so a file tampered with after loading is caught at the next reload.

Standard library only. torch is never imported here; a torch OOM is recognised by class name.
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable

from station import modelstore as ms
from station.contracts import StationError


class ModelUnavailable(StationError):
    """No approved, verified or loadable model for this part right now. The server answers 503."""


def is_out_of_memory(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory error, recognised by name so torch is not needed here."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def free_gpu_cache() -> None:
    """Return cached GPU memory to the driver, if torch is already loaded. Best effort."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - freeing is best effort; the 503 is what matters
        pass


@dataclass(frozen=True)
class LoadedModel:
    part_number: str
    version: str
    model: Any  # anything with predict(frame) -> ImageResult
    signature: tuple


def _current_signature(store: ms.ModelStore, part_number: str) -> tuple | None:
    path = store.root / ms.safe_part(part_number) / ms.CURRENT_NAME
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


class ModelRegistry:
    """Loaded models, one per part. load_model(info) builds the station-style model object."""

    def __init__(self, store: ms.ModelStore, load_model: Callable[[ms.ModelInfo], Any]) -> None:
        self._store = store
        self._load_model = load_model
        self._lock = threading.Lock()
        self._loaded: dict[str, LoadedModel] = {}

    def get(self, part_number: str) -> LoadedModel:
        with self._lock:
            sig = _current_signature(self._store, part_number)
            cached = self._loaded.get(part_number)
            if cached is not None and sig is not None and cached.signature == sig:
                return cached
            self._loaded.pop(part_number, None)
            try:
                info = self._store.current(part_number)
            except ms.ModelError as exc:
                raise ModelUnavailable(str(exc)) from None
            if info is None:
                raise ModelUnavailable(f"No approved model for part {part_number}. Ask Quality to approve one.")
            try:
                model = self._load_model(info)
            except Exception as exc:  # noqa: BLE001 - every load failure becomes a 503
                if is_out_of_memory(exc):
                    free_gpu_cache()
                    raise ModelUnavailable(
                        f"GPU out of memory while loading version {info.version}; the model was "
                        "unloaded and will be retried on the next request"
                    ) from None
                if isinstance(exc, StationError):
                    raise ModelUnavailable(str(exc)) from None
                raise ModelUnavailable(
                    f"could not load version {info.version} for part {part_number}: {type(exc).__name__}"
                ) from None
            loaded = LoadedModel(part_number, info.version, model, sig)
            if sig is not None:
                self._loaded[part_number] = loaded
            return loaded

    def release(self, part_number: str) -> None:
        """Drop a part's loaded model and free GPU cache (after an out-of-memory error)."""
        with self._lock:
            self._loaded.pop(part_number, None)
            free_gpu_cache()

    def loaded(self) -> list[dict]:
        with self._lock:
            return [{"part_number": m.part_number, "version": m.version}
                    for m in sorted(self._loaded.values(), key=lambda m: m.part_number)]
