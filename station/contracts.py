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


def camera_ids_for(count: int) -> tuple[str, ...]:
    """Camera ids for a station with `count` cameras (1..4). Fewer than 4 is proof-of-concept only."""
    if not 1 <= count <= len(CAMERA_IDS):
        raise ValueError(f"camera count must be 1..{len(CAMERA_IDS)}, got {count}")
    return CAMERA_IDS[:count]


def images_for(camera_ids: "tuple[str, ...] | list[str]") -> int:
    """Images per part for these cameras: every camera, top and bottom."""
    return 2 * len(camera_ids)


def is_full_station(camera_ids: "tuple[str, ...] | list[str]") -> bool:
    """True only for the real 4-camera, 8-image station (the only mode that may post to the MES by default)."""
    return tuple(camera_ids) == CAMERA_IDS


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


LABEL_PROPOSED = "proposed"  # an operator proposed it; never used for training
LABEL_OFFICIAL = "official"  # approved by an admin (or set by one); the only label training uses
LABEL_REJECTED = "rejected"  # an admin rejected the proposal, with a reason; never used for training
LABEL_STATUSES = (LABEL_PROPOSED, LABEL_OFFICIAL, LABEL_REJECTED)


@dataclass(frozen=True)
class OperatorLabel:
    """A human label on a collected part (collect mode), so it can be used for training.

    Never changes the part's final_verdict or counters; it is metadata on the saved part.
    criterion_id is only meaningful for BAD and is None for GOOD.

    Two-step approval: an operator proposes (status 'proposed'); only an admin makes it
    'official' or 'rejected'. labelled_by/labelled_at are who wrote this record last.
    proposed_* is the proposer; approved_* is the admin who approved, rejected or set it.
    A label saved before the status field existed has no status and counts as 'official'.
    """

    verdict: Verdict  # GOOD or BAD only
    labelled_by: str
    labelled_at: str  # ISO 8601 UTC
    criterion_id: Optional[str] = None
    status: str = LABEL_OFFICIAL  # 'proposed' | 'official' | 'rejected'
    proposed_by: Optional[str] = None
    proposed_at: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    reject_reason: Optional[str] = None


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
    operator_label: Optional[OperatorLabel] = None  # collected parts only; None until labelled


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
