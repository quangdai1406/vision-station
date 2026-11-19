from __future__ import annotations

import math
import pathlib
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Sequence

from station.contracts import CAMERA_IDS, Side, StationError
from station.scanstore import (
    SIDES, ScanStore, list_scans, load_scan, review_scan_label, verify_scan,
)


class ScanState(str, Enum):
    IDLE = "IDLE"
    SCAN_SIDE = "SCAN_SIDE"
    WAIT_FLIP = "WAIT_FLIP"
    SAVING = "SAVING"
    DONE = "DONE"
    FAULT = "FAULT"


class ScanError(StationError):
    pass


PENDING_REVIEW_CAP = 200


def _default_scan_id() -> str:
    return "SCAN-" + uuid.uuid4().hex[:16]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seconds_left(wait: float) -> float:
    return math.ceil(round(wait, 6) * 10) / 10


def _side_name(side: object) -> str:
    value = side.value if isinstance(side, Side) else side
    if value not in SIDES:
        raise ScanError(f"side must be TOP or BOTTOM, got {side!r}")
    return value


def _validate_sides(sides: object) -> tuple[str, ...]:
    if isinstance(sides, (str, bytes)):
        raise ScanError("sides must be a list of TOP and/or BOTTOM")
    try:
        names = tuple(_side_name(s) for s in sides)
    except TypeError as exc:
        raise ScanError("sides must be a list of TOP and/or BOTTOM") from exc
    if not names:
        raise ScanError("sides must not be empty")
    if len(set(names)) != len(names):
        raise ScanError("sides must not repeat")
    return names


class ScanSession:

    def __init__(
        self,
        cameras,
        store: ScanStore,
        busy: Callable[[], bool],
        *,
        camera_ids: Sequence[str] = CAMERA_IDS,
        debounce_s: float = 0.3,
        min_flip_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        now_utc: Callable[[], str] = _utc_now,
        scan_id_factory: Callable[[], str] = _default_scan_id,
    ) -> None:
        cams = tuple(camera_ids)
        if not cams:
            raise ValueError("camera_ids must not be empty")
        if len(set(cams)) != len(cams):
            raise ValueError("camera_ids must not repeat")
        if debounce_s < 0:
            raise ValueError(f"debounce_s must be >= 0, got {debounce_s}")
        if min_flip_s < 0:
            raise ValueError(f"min_flip_s must be >= 0, got {min_flip_s}")
        self._cameras = cameras
        self._store = store
        self._busy = busy
        self._camera_ids = cams
        self._debounce_s = float(debounce_s)
        self._min_flip_s = float(min_flip_s)
        self._clock = clock
        self._now_utc = now_utc
        self._scan_id_factory = scan_id_factory
        self._lock = threading.RLock()

        self._state = ScanState.IDLE
        self._fault: Optional[str] = None
        self._fault_return: Optional[ScanState] = None
        self._last_done: Optional[float] = None
        self._flip_started: Optional[float] = None
        self._saved_path: Optional[pathlib.Path] = None
        self._clear_scan()


    def start(
        self,
        part_number: str,
        toy_id: str,
        label: dict,
        taken_by: str,
        sides: Sequence[str] = ("TOP", "BOTTOM"),
    ) -> str:
        with self._lock:
            if self._busy():
                raise ScanError("an inspection is running; a scan cannot start now")
            if self._state is not ScanState.IDLE:
                raise ScanError(f"a scan cannot start in state {self._state.value}")
            side_names = _validate_sides(sides)
            scan_id = self._scan_id_factory()
            try:
                writer = self._store.begin_scan(
                    scan_id, part_number, toy_id, label, taken_by,
                    list(self._camera_ids), self._now_utc(),
                )
            except (ValueError, FileExistsError) as exc:
                raise ScanError(str(exc)) from exc
            self._clear_scan()
            self._writer = writer
            self._scan_id = scan_id
            self._part_number = part_number
            self._toy_id = toy_id
            self._label_verdict = writer.verdict
            self._defects = list(label.get("defects") or [])
            self._sides = side_names
            self._idx = 0
            self._counts = {side: 0 for side in SIDES}
            self._visible: dict[tuple[str, str, int], bool] = {}
            self._state = ScanState.SCAN_SIDE
            return scan_id

    def press(self) -> bool:
        with self._lock:
            now = self._clock()
            if not self._debounce_ok(now):
                return False
            state = self._state
            if state is ScanState.WAIT_FLIP:
                if self._flip_wait(now) > 0:
                    return False
                self._flip(now)
            elif state is ScanState.FAULT:
                if self._fault_return is not ScanState.SCAN_SIDE:
                    return False
                self._fault = None
                self._fault_return = None
                self._state = ScanState.SCAN_SIDE
            elif state is not ScanState.SCAN_SIDE:
                return False
            self._take_step()
            self._last_done = self._clock()
            return True

    def flip_ready(self) -> None:
        with self._lock:
            if self._state is not ScanState.WAIT_FLIP:
                raise ScanError(f"flip_ready is not allowed in state {self._state.value}")
            self._flip(self._clock())

    def done_side(self) -> str:
        with self._lock:
            if self._state is not ScanState.SCAN_SIDE:
                raise ScanError(f"done_side is not allowed in state {self._state.value}")
            side = self._current_side()
            if self._counts[side] < 1:
                raise ScanError(f"{side} has no steps yet: take at least one step before Done")
            now = self._clock()
            if self._idx < len(self._sides) - 1:
                self._state = ScanState.WAIT_FLIP
                self._flip_started = now
            else:
                self._state = ScanState.SAVING
            self._last_done = now
            return self._state.value

    def mark(self, camera_id: str, side: str, step: int, visible: bool) -> None:
        with self._lock:
            if self._state not in (ScanState.SCAN_SIDE, ScanState.WAIT_FLIP, ScanState.SAVING):
                raise ScanError(f"marks cannot change in state {self._state.value}")
            if self._label_verdict != "BAD":
                raise ScanError("only a BAD scan has defect marks")
            if not isinstance(visible, bool):
                raise ScanError("visible must be True or False")
            side_name = _side_name(side)
            key = (camera_id, side_name, step)
            if key not in self._visible:
                raise ScanError(f"no picture {camera_id} {side_name} step {step} in this scan")
            try:
                self._writer.set_defect_visible(camera_id, side_name, step, visible)
            except (ValueError, RuntimeError) as exc:
                raise ScanError(str(exc)) from exc
            self._visible[key] = visible

    def finish(self) -> pathlib.Path:
        with self._lock:
            if self._state is not ScanState.SAVING:
                raise ScanError("finish is only possible after Done on the last side")
            empty = [s for s in self._sides if self._counts[s] < 1]
            if empty:
                raise ScanError(f"{', '.join(empty)} has no steps")
            try:
                path = self._writer.finish(self._now_utc())
            except Exception as exc:
                self._drop_scan(f"scan not saved: {exc}")
                raise ScanError(self._fault) from exc
            self._writer = None
            self._saved_path = path
            self._state = ScanState.DONE
            return path

    def cancel(self) -> None:
        with self._lock:
            if self._state in (ScanState.IDLE, ScanState.DONE):
                raise ScanError(f"nothing to cancel in state {self._state.value}")
            if self._writer is not None:
                self._writer.cancel()
            self._clear_scan()
            self._state = ScanState.IDLE

    def reset(self) -> None:
        with self._lock:
            if self._state is not ScanState.DONE:
                raise ScanError(f"reset is only possible after a saved scan, not in {self._state.value}")
            self._clear_scan()
            self._state = ScanState.IDLE

    def clear_fault(self) -> None:
        with self._lock:
            if self._state is not ScanState.FAULT:
                raise ScanError(f"clear_fault is not allowed in state {self._state.value}")
            if self._fault_return is None:
                self._clear_scan()
                self._state = ScanState.IDLE
                return
            self._fault = None
            self._fault_return = None
            self._state = ScanState.SCAN_SIDE

    def step_picture(self, camera_id: str) -> Optional[tuple[bytes, str]]:
        with self._lock:
            if self._state not in (ScanState.SCAN_SIDE, ScanState.WAIT_FLIP, ScanState.SAVING):
                return None
            return self._last_pictures.get(camera_id)

    def pending_reviews(self) -> list[dict]:
        with self._lock:
            found = []
            for scan_dir in list_scans(self._store.root, self._store.line):
                try:
                    doc = load_scan(scan_dir)
                except (OSError, ValueError):
                    continue
                if doc["label"]["status"] == "proposed":
                    found.append((doc, scan_dir))
            found.sort(key=lambda item: item[0]["started_utc"], reverse=True)
            rows: list[dict] = []
            for doc, scan_dir in found:
                if len(rows) >= PENDING_REVIEW_CAP:
                    break
                if verify_scan(scan_dir):
                    continue
                label = doc["label"]
                rows.append({
                    "scan_id": doc["scan_id"],
                    "part_number": doc["part_number"],
                    "toy_id": doc["toy_id"],
                    "verdict": label["verdict"],
                    "defects": list(label["defects"]),
                    "labelled_by": label["labelled_by"],
                    "labelled_at": label["labelled_at"],
                    "steps": dict(doc["steps"]),
                })
            return rows

    def review(self, scan_id: str, decision: str, by: str, reason: Optional[str] = None) -> dict:
        with self._lock:
            if not isinstance(scan_id, str) or not scan_id:
                raise ValueError("scan_id is required")
            if any(ch in scan_id for ch in "/\\\x00"):
                raise ValueError("scan_id must be a scan folder name, not a path")
            matches = [d for d in list_scans(self._store.root, self._store.line) if d.name == scan_id]
            if len(matches) != 1:
                raise ValueError(f"no finished scan {scan_id!r} on this line")
            try:
                return review_scan_label(matches[0], decision, by, self._now_utc(), reason)
            except OSError as exc:
                raise ValueError(
                    "the scan label could not be written; nothing was approved "
                    f"({exc.__class__.__name__})"
                ) from exc

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            wait = self._flip_wait(now)
            side = self._current_side() if self._sides and self._state in (
                ScanState.SCAN_SIDE, ScanState.WAIT_FLIP, ScanState.SAVING, ScanState.FAULT
            ) else None
            marks = None
            if self._label_verdict == "BAD":
                values = list(self._visible.values())
                marks = {
                    "pictures": len(values),
                    "visible": sum(1 for v in values if v),
                    "not_visible": sum(1 for v in values if not v),
                }
            return {
                "state": self._state.value,
                "side": side,
                "step": self._counts[side] if side else 0,
                "counts": dict(self._counts),
                "sides": list(self._sides),
                "scan_id": self._scan_id,
                "part_number": self._part_number,
                "toy_id": self._toy_id,
                "label_verdict": self._label_verdict,
                "defects": list(self._defects),
                "marks": marks,
                "camera_names": list(self._camera_ids),
                "fault": self._fault,
                "saved_path": str(self._saved_path) if self._saved_path else None,
                "press_ready": self._press_ready(now),
                "flip_wait_s": _seconds_left(wait) if wait > 0 else 0.0,
                "prompt": self._prompt(side, wait),
            }


    def _take_step(self) -> bool:
        side = self._current_side()
        step = self._counts[side] + 1
        try:
            frames = list(self._cameras.capture(Side(side)))
            pictures = self._check_frames(side, step, frames)
        except Exception as exc:
            text = str(exc) or type(exc).__name__
            self._fault_at(f"{side} step {step}: {text}", ScanState.SCAN_SIDE)
            return False
        defect_visible = False if self._label_verdict == "BAD" else None
        try:
            for cam in self._camera_ids:
                data, ext = pictures[cam]
                self._writer.add_picture(
                    cam, side, step, data, ext, defect_visible=defect_visible,
                )
        except Exception as exc:
            self._drop_scan(
                f"{side} step {step}: a picture could not be written ({exc}); "
                "the whole scan was discarded"
            )
            return False
        for cam in self._camera_ids:
            self._visible[(cam, side, step)] = bool(defect_visible)
        self._last_pictures = pictures
        self._counts[side] = step
        return True

    def _check_frames(self, side: str, step: int, frames: list) -> dict[str, tuple[bytes, str]]:
        by_cam: dict[str, tuple[bytes, str]] = {}
        for frame in frames:
            cam = frame.camera_id
            if cam not in self._camera_ids:
                raise ValueError(f"unexpected camera {cam!r} sent a picture")
            if cam in by_cam:
                raise ValueError(f"{cam} sent two pictures")
            data = frame.data
            if not isinstance(data, (bytes, bytearray, memoryview)) or not len(data):
                raise ValueError(f"{cam} sent an empty picture")
            ext = frame.ext.lstrip(".") if isinstance(frame.ext, str) else ""
            if not ext.isalnum():
                raise ValueError(f"{cam} sent a bad picture type {frame.ext!r}")
            by_cam[cam] = (bytes(data), ext)
        missing = [c for c in self._camera_ids if c not in by_cam]
        if missing:
            raise ValueError(f"no picture from {', '.join(missing)}")
        return by_cam

    def _fault_at(self, text: str, return_state: Optional[ScanState]) -> None:
        self._fault = text
        self._fault_return = return_state
        self._state = ScanState.FAULT

    def _drop_scan(self, text: str) -> None:
        if self._writer is not None:
            self._writer.cancel()
        self._writer = None
        self._fault_at(text, None)

    def _current_side(self) -> str:
        return self._sides[self._idx]

    def _debounce_ok(self, now: float) -> bool:
        return self._last_done is None or now - self._last_done >= self._debounce_s

    def _flip_wait(self, now: float) -> float:
        if self._state is not ScanState.WAIT_FLIP or self._flip_started is None:
            return 0.0
        return max(0.0, self._min_flip_s - (now - self._flip_started))

    def _flip(self, now: float) -> None:
        wait = self._flip_wait(now)
        if wait > 0:
            raise ScanError(
                f"the part is not turned yet: wait {_seconds_left(wait)} s before the flip"
            )
        self._idx += 1
        self._state = ScanState.SCAN_SIDE

    def _press_ready(self, now: float) -> bool:
        if not self._debounce_ok(now):
            return False
        if self._state is ScanState.SCAN_SIDE:
            return True
        if self._state is ScanState.WAIT_FLIP:
            return self._flip_wait(now) <= 0
        return self._state is ScanState.FAULT and self._fault_return is ScanState.SCAN_SIDE

    def _prompt(self, side: Optional[str], wait: float = 0.0) -> str:
        st = self._state
        if st is ScanState.IDLE:
            return "Choose the part, the toy and the label, then start the scan."
        if st is ScanState.SCAN_SIDE:
            n = self._counts[side]
            if n == 0:
                return f"{side} step 1: place the part, then press Enter."
            return f"{side} step {n + 1}: turn the part, then press Enter, or press Done with this side"
        if st is ScanState.WAIT_FLIP:
            nxt = self._sides[self._idx + 1]
            done = f"{side} done with {self._counts[side]} step(s)."
            if wait > 0:
                return (f"{done} Wait {_seconds_left(wait)} s, then flip the part to the "
                        f"{nxt} side and press Enter.")
            return f"{done} Flip the part to the {nxt} side, then press Enter."
        if st is ScanState.SAVING:
            return "Press Finish to save the scan, or Cancel to drop it."
        if st is ScanState.DONE:
            return "Scan saved. Reset to start another scan."
        if st is ScanState.FAULT:
            if self._fault_return is ScanState.SCAN_SIDE:
                return f"Fault: {self._fault}. Press Enter to retry this step, or clear the fault."
            return f"Fault: {self._fault}. Clear the fault to return to the start."
        return ""

    def _clear_scan(self) -> None:
        self._writer = None
        self._scan_id: Optional[str] = None
        self._part_number = ""
        self._toy_id = ""
        self._label_verdict: Optional[str] = None
        self._defects: list[str] = []
        self._sides: tuple[str, ...] = ()
        self._idx = 0
        self._counts = {side: 0 for side in SIDES}
        self._visible: dict[tuple[str, str, int], bool] = {}
        self._fault = None
        self._fault_return = None
        self._flip_started = None
        self._saved_path = None
        self._last_pictures: dict[str, tuple[bytes, str]] = {}
