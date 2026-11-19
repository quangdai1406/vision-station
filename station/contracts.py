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
    REVIEW = "REVIEW"


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
IMAGES_PER_PART = 8


def camera_ids_for(count: int) -> tuple[str, ...]:
    if not 1 <= count <= len(CAMERA_IDS):
        raise ValueError(f"camera count must be 1..{len(CAMERA_IDS)}, got {count}")
    return CAMERA_IDS[:count]


def images_for(camera_ids: "tuple[str, ...] | list[str]") -> int:
    return 2 * len(camera_ids)


def is_full_station(camera_ids: "tuple[str, ...] | list[str]") -> bool:
    return tuple(camera_ids) == CAMERA_IDS


@dataclass(frozen=True)
class Frame:
    camera_id: str
    side: Side
    seq: int
    timestamp_utc: str
    data: bytes
    ext: str = "png"


@dataclass(frozen=True)
class Defect:
    defect_class: str
    score: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageResult:
    camera_id: str
    side: Side
    score: float
    defects: tuple[Defect, ...] = ()


@dataclass(frozen=True)
class Thresholds:
    t_low: float
    t_img: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.t_low <= self.t_img <= 1.0):
            raise ValueError("need 0 <= t_low <= t_img <= 1")


@dataclass(frozen=True)
class PartVerdict:
    verdict: Verdict
    results: tuple[ImageResult, ...]
    reason: str = ""


@dataclass(frozen=True)
class Override:
    original_verdict: Verdict
    new_verdict: Verdict
    reason: str
    by: str


LABEL_PROPOSED = "proposed"
LABEL_OFFICIAL = "official"
LABEL_REJECTED = "rejected"
LABEL_STATUSES = (LABEL_PROPOSED, LABEL_OFFICIAL, LABEL_REJECTED)


@dataclass(frozen=True)
class OperatorLabel:

    verdict: Verdict
    labelled_by: str
    labelled_at: str
    criterion_id: Optional[str] = None
    status: str = LABEL_OFFICIAL
    proposed_by: Optional[str] = None
    proposed_at: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    reject_reason: Optional[str] = None


@dataclass
class PartRecord:

    part_id: str
    session_id: str
    station_id: str
    part_number: str
    lot: str
    operator: str
    supervisor: str
    material_handler: str
    started_utc: str
    model_version: str
    thresholds: Thresholds
    frames: list[Frame]
    results: list[ImageResult]
    machine_verdict: Verdict
    final_verdict: Verdict
    override: Optional[Override] = None
    disposition: Optional[str] = None
    operator_label: Optional[OperatorLabel] = None


class StationError(Exception):
    pass


class CameraError(StationError):
    pass


class InvalidTransition(StationError):
    pass


class CameraSet(Protocol):
    def capture(self, side: Side) -> Sequence[Frame]:
        pass


class InferenceModel(Protocol):
    version: str

    def predict(self, frame: Frame) -> ImageResult: ...


class OutputDriver(Protocol):

    def show_good(self) -> None: ...
    def show_bad(self) -> None: ...
    def show_review(self) -> None: ...
    def show_fault(self) -> None: ...
    def alarm_off(self) -> None: ...
    def clear(self) -> None: ...
