"""Shared types and interfaces for the camera inspection station.

Every module codes against this file; do not change it without the
orchestrator. Standard library only.

Plan reference: 26_1023_inspection_system_plan.md (sections 3, 5, 7).

Decisions made here (flag to the user):
- Thresholds use two numbers, not three. t_low is the plan's [T_low] and t_img
  is [T_img]; the REVIEW band is the contiguous range t_low <= score < t_img
  (so [T_high] == [T_img]). Quality owns the real values.
- Image scores are 0.0..1.0 where higher means more defective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol, Sequence


class Side(str, Enum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"


class Verdict(str, Enum):
    GOOD = "GOOD"
    BAD = "BAD"
    REVIEW = "REVIEW"  # never green; a supervisor resolves it to GOOD or BAD


class State(str, Enum):
    IDLE = "IDLE"
    CAPTURE_TOP = "CAPTURE_TOP"
    INFER_TOP = "INFER_TOP"
    WAIT_FLIP = "WAIT_FLIP"
    CAPTURE_BOTTOM = "CAPTURE_BOTTOM"
    INFER_BOTTOM = "INFER_BOTTOM"
    VERDICT = "VERDICT"
    SUPERVISOR_REVIEW = "SUPERVISOR_REVIEW"
    FAULT = "FAULT"


CAMERA_IDS = ("C1", "C2", "C3", "C4")
IMAGES_PER_PART = 8  # 4 cameras x 2 sides


@dataclass(frozen=True)
class Frame:
    camera_id: str
    side: Side
    seq: int  # 0..3 within one side
    timestamp_utc: str  # ISO 8601, e.g. 2026-10-23T08:30:12Z
    data: bytes
    ext: str = "png"


@dataclass(frozen=True)
class Defect:
    defect_class: str
    score: float
    box: tuple[int, int, int, int]  # x, y, width, height in pixels


@dataclass(frozen=True)
class ImageResult:
    camera_id: str
    side: Side
    score: float  # worst defect score in this image, 0.0 if none
    defects: tuple[Defect, ...] = ()


@dataclass(frozen=True)
class Thresholds:
    t_low: float  # score >= t_low (and < t_img) -> REVIEW
    t_img: float  # score >= t_img -> BAD

    def __post_init__(self) -> None:
        if not (0.0 <= self.t_low <= self.t_img <= 1.0):
            raise ValueError("need 0 <= t_low <= t_img <= 1")


@dataclass(frozen=True)
class PartVerdict:
    verdict: Verdict
    results: tuple[ImageResult, ...]  # all 8, in capture order
    reason: str = ""


@dataclass(frozen=True)
class Override:
    original_verdict: Verdict
    new_verdict: Verdict  # GOOD or BAD only
    reason: str
    by: str


@dataclass
class PartRecord:
    """One finished part: what storage writes and the MES client posts."""

    part_id: str
    session_id: str  # also the MES Idempotency-Key
    station_id: str
    part_number: str
    lot: str
    operator: str
    supervisor: str
    material_handler: str
    started_utc: str
    model_version: str
    thresholds: Thresholds
    frames: list[Frame]  # 8
    results: list[ImageResult]  # 8
    machine_verdict: Verdict  # what fusion said before any human resolved it
    final_verdict: Verdict  # GOOD or BAD (REVIEW is resolved before posting)
    override: Optional[Override] = None
    disposition: Optional[str] = None  # "quarantine" | "released" | None


class StationError(Exception):
    pass


class CameraError(StationError):
    """A camera dropped out, timed out or returned an unusable frame."""


class InvalidTransition(StationError):
    """An action that the current state does not allow."""


class CameraSet(Protocol):
    def capture(self, side: Side) -> Sequence[Frame]:
        """Capture one frame from each of the 4 cameras. Raise CameraError on failure."""


class InferenceModel(Protocol):
    version: str

    def predict(self, frame: Frame) -> ImageResult: ...


class OutputDriver(Protocol):
    """Lights and alarm. Red is on only for BAD; green only for GOOD."""

    def show_good(self) -> None: ...
    def show_bad(self) -> None: ...  # red on and alarm on
    def show_review(self) -> None: ...  # neither green nor alarm; amber/blink is the driver's choice
    def show_fault(self) -> None: ...
    def alarm_off(self) -> None: ...
    def clear(self) -> None: ...  # all off, ready for the next part
