"""One inspection session: from the first press of a part to a completed record.

Plan reference: 26_1023_inspection_system_plan.md (sections 3, 5, 7).
Standard library only. Cameras, model, outputs and storage are injected.

Flow: IDLE --press--> (TOP capture, infer) WAIT_FLIP --press--> (BOTTOM capture,
infer, fuse) --> IDLE (GOOD), VERDICT (BAD), SUPERVISOR_REVIEW (REVIEW).
Any failure while capturing or inferring goes to FAULT.

Completion order (every path that ends a part: machine GOOD, ack of BAD,
override, review resolution):
  1. store.save_part(record) runs first.
  2. Only after it succeeds are the part counted and the lights changed.
     Green is never shown for a part that was not saved.

Save failures (decided here):
  - If save_part raises, the session goes to FAULT (show_fault, state FAULT,
    'fault' event, snapshot fault text). Nothing is counted and no light is
    changed to green. Frames, results, machine verdict and the pending decision
    (final verdict, override, disposition) are kept in `_pending`.
  - clear_fault() retries that same save once (same part_id and session_id, so a
    store that keys on session_id can deduplicate). If the retry succeeds the
    part is saved and counted once and the station returns to IDLE. If it fails
    again the station stays in FAULT with the new error text. The station does
    not return to the pre-completion state (VERDICT / SUPERVISOR_REVIEW /
    INFER_BOTTOM): the operator's decision is kept and only the save is retried.

on_complete failures (decided here):
  - on_complete runs after a successful save. If it raises, the part stays saved
    and counted, an 'on_complete_failed' event is logged, the lights are updated
    as for a normal completion and the station returns to IDLE. No fault. The
    outbox is expected to retry the upload.

Output failures (decided here):
  - The lights run after a successful save. If an OutputDriver call raises, the
    part stays saved and counted, the station returns to IDLE, an 'outputs_failed'
    event with the error text is logged, the remaining lights are still tried and
    on_complete still runs. No fault.
  - The BAD and REVIEW alarms (show_bad, show_review) run after the state is set
    to VERDICT / SUPERVISOR_REVIEW, and show_fault runs after the state is set to
    FAULT. If one raises, the state is kept, an 'outputs_failed' event is logged,
    and nothing is raised out of press(). The operator can still ack, override or
    resolve, and the part is saved on completion.

Snapshot: see StationSession.snapshot for the exact shape the UI reads.
"""
from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from station.contracts import (
    CAMERA_IDS,
    IMAGES_PER_PART,
    CameraError,
    Frame,
    ImageResult,
    InvalidTransition,
    Override,
    PartRecord,
    PartVerdict,
    Side,
    State,
    StationError,
    Thresholds,
    Verdict,
)


@dataclass(frozen=True)
class _PendingCompletion:
    final: Verdict
    override: Optional[Override]
    disposition: Optional[str]
    lights: tuple[str, ...]  # OutputDriver methods to call only after a successful save


def _default_session_id() -> str:
    return uuid.uuid4().hex


def _accept_all(frame: Frame) -> bool:
    return True


def _defects_list(r: ImageResult) -> list[dict[str, Any]]:
    return [
        {"defect_class": d.defect_class, "score": d.score, "box": list(d.box)}
        for d in r.defects
    ]


def _result_dict(r: ImageResult) -> dict[str, Any]:
    return {
        "camera_id": r.camera_id,
        "side": r.side.value,
        "score": r.score,
        "defects": _defects_list(r),
    }


def _enum_value(x: Any) -> Optional[str]:
    return None if x is None else x.value


class StationSession:
    def __init__(
        self,
        cameras,
        model,
        outputs,
        store,
        thresholds: Thresholds,
        *,
        station_id: str,
        part_number: str,
        lot: str,
        operator: str,
        supervisor: str,
        material_handler: str,
        flip_timeout_s: float = 60.0,
        debounce_s: float = 0.3,
        frame_ok: Optional[Callable[[Frame], bool]] = None,
        max_retakes: int = 2,
        clock: Callable[[], float] = time.monotonic,
        now_utc: Callable[[], str],
        part_id_factory: Callable[[], str],
        session_id_factory: Callable[[], str] = _default_session_id,
        fuse_fn: Optional[Callable[[Sequence[ImageResult], Thresholds], PartVerdict]] = None,
        on_complete: Optional[Callable[[PartRecord], None]] = None,
        initial_counters: Optional[dict[str, int]] = None,
    ) -> None:
        self._cameras = cameras
        self._model = model
        self._outputs = outputs
        self._store = store
        self._thresholds = thresholds
        self._station_id = station_id
        self._part_number = part_number
        self._lot = lot
        self._operator = operator
        self._supervisor = supervisor
        self._material_handler = material_handler
        self._flip_timeout_s = flip_timeout_s
        self._debounce_s = debounce_s
        self._frame_ok = frame_ok or _accept_all
        self._max_retakes = max_retakes
        self._clock = clock
        self._now_utc = now_utc
        self._part_id_factory = part_id_factory
        self._session_id_factory = session_id_factory
        self._fuse_fn = fuse_fn
        self._on_complete = on_complete

        self.state: State = State.IDLE
        self.counters: dict[str, int] = {"total": 0, "good": 0, "bad": 0}
        if initial_counters is not None:
            for key in self.counters:
                self.counters[key] = int(initial_counters.get(key, 0))

        self._last_press: Optional[float] = None
        self._fault: Optional[str] = None
        self._fault_return: Optional[State] = None
        self._clear_part()

    # ------------------------------------------------------------------ public

    def press(self) -> bool:
        now = self._clock()
        if self._last_press is not None and now - self._last_press < self._debounce_s:
            self._log("press_ignored", {"reason": "debounce", "state": self.state.value},
                      session_id=self._active_session_id())
            return False
        if self.state not in (State.IDLE, State.WAIT_FLIP):
            self._log("press_ignored", {"reason": "state", "state": self.state.value},
                      session_id=self._active_session_id())
            return False

        self._last_press = now
        if self.state is State.IDLE:
            self._outputs.clear()
            self._begin_part()
            self._log("press", {"side": Side.TOP.value, "part_id": self._part_id},
                      session_id=self._session_id)
            if self._run_side(Side.TOP, fail_state=State.IDLE):
                self.state = State.WAIT_FLIP
        else:
            self._log("press", {"side": Side.BOTTOM.value, "part_id": self._part_id},
                      session_id=self._session_id)
            self._run_bottom()
        return True

    def ack_alarm(self) -> None:
        if self.state is not State.VERDICT or self._machine != Verdict.BAD:
            raise InvalidTransition(f"ack_alarm is not allowed in {self.state.value}")
        self._log("ack", {"action": "ack_alarm", "part_id": self._part_id, "verdict": Verdict.BAD.value},
                  session_id=self._session_id)
        self._finish(Verdict.BAD, override=None, disposition="quarantine",
                     lights=("show_bad", "alarm_off"))

    def override(self, by: str, new_verdict: Verdict, reason: str) -> None:
        new = self._check_decision(by, new_verdict, reason)
        if self.state is not State.VERDICT:
            raise InvalidTransition(f"override is not allowed in {self.state.value}")
        ov = Override(original_verdict=Verdict.BAD, new_verdict=new, reason=reason, by=by)
        self._log("override", {"original_verdict": ov.original_verdict.value, "new_verdict": new.value,
                               "reason": reason, "by": by, "part_id": self._part_id},
                  session_id=self._session_id)
        if new is Verdict.GOOD:
            lights = ("alarm_off", "show_good")
        else:
            lights = ("show_bad", "alarm_off")
        self._finish(new, override=ov, disposition="released" if new is Verdict.GOOD else "quarantine",
                     lights=lights)

    def resolve_review(self, by: str, new_verdict: Verdict, reason: str) -> None:
        new = self._check_decision(by, new_verdict, reason)
        if self.state is not State.SUPERVISOR_REVIEW:
            raise InvalidTransition(f"resolve_review is not allowed in {self.state.value}")
        ov = Override(original_verdict=Verdict.REVIEW, new_verdict=new, reason=reason, by=by)
        self._log("review_resolved", {"new_verdict": new.value, "reason": reason, "by": by,
                                      "part_id": self._part_id},
                  session_id=self._session_id)
        if new is Verdict.GOOD:
            lights = ("show_good",)
        else:
            lights = ("show_bad", "alarm_off")
        self._finish(new, override=ov, disposition="released" if new is Verdict.GOOD else "quarantine",
                     lights=lights)

    def tick(self) -> bool:
        """Void a part whose flip press never came. Returns True if a session was voided."""
        if self.state is not State.WAIT_FLIP or self._last_press is None:
            return False
        if self._clock() - self._last_press <= self._flip_timeout_s:
            return False
        self._log("session_voided", {"reason": "flip_timeout", "part_id": self._part_id},
                  session_id=self._session_id)
        self._outputs.clear()
        self._clear_part()
        self.state = State.IDLE
        return True

    def clear_fault(self) -> None:
        """Leave FAULT. A failed save is retried once; other faults return to their pre-fault state."""
        if self.state is not State.FAULT:
            raise InvalidTransition(f"clear_fault is not allowed in {self.state.value}")
        self._outputs.clear()
        pending = self._pending
        back = State.IDLE if pending is not None else (self._fault_return or State.IDLE)
        self._log("fault_cleared", {"return_state": back.value, "part_id": self._part_id,
                                    "retry_save": pending is not None},
                  session_id=self._session_id)
        if pending is not None:
            self._try_complete()  # on failure it re-enters FAULT with the new error
            return
        self._fault = None
        self._fault_return = None
        if back is State.IDLE:
            self._clear_part()  # the part was never completed; drop it
        elif back is State.WAIT_FLIP:
            self._last_press = self._clock()  # restart the flip timeout
        self.state = back

    def snapshot(self, include_images: bool = False) -> dict[str, Any]:
        """State for the UI.

        Keys: state, counters {total, good, bad}, part_id, machine_verdict,
        final_verdict (set only once the part is saved), verdict (what the big
        indicator shows: REVIEW in SUPERVISOR_REVIEW, else final_verdict, else
        machine_verdict; GOOD is never shown before it is saved; None when no
        part), fault, results (one per captured image, so far), and images
        (one metadata entry per captured frame, up to 8, with score and defects
        merged in from the matching result; 'b64' only when include_images).
        """
        results = self._all_results()
        by_key = {(r.camera_id, r.side): r for r in results}
        images: list[dict[str, Any]] = []
        for f in self._all_frames()[:IMAGES_PER_PART]:
            r = by_key.get((f.camera_id, f.side))
            entry: dict[str, Any] = {
                "camera_id": f.camera_id,
                "side": f.side.value,
                "seq": f.seq,
                "ext": f.ext,
                "score": r.score if r is not None else 0.0,
                "defects": _defects_list(r) if r is not None else [],
            }
            if include_images:
                entry["b64"] = base64.b64encode(f.data).decode("ascii")
            images.append(entry)
        return {
            "state": self.state.value,
            "counters": dict(self.counters),
            "part_id": self._part_id,
            "machine_verdict": _enum_value(self._machine),
            "final_verdict": _enum_value(self._final),
            "verdict": self._display_verdict(),
            "fault": self._fault,
            "results": [_result_dict(r) for r in results],
            "images": images,
        }

    # ----------------------------------------------------------------- helpers

    def _display_verdict(self) -> Optional[str]:
        if self.state is State.SUPERVISOR_REVIEW:
            return Verdict.REVIEW.value
        if self._final is not None:
            return self._final.value
        if self._machine is None or self._machine is Verdict.GOOD:
            return None  # a machine GOOD is not shown until it is saved
        return self._machine.value

    def _run_bottom(self) -> None:
        self._bottom_frames = []
        self._bottom_results = []
        if not self._run_side(Side.BOTTOM, fail_state=State.WAIT_FLIP):
            return
        self._fuse_and_route()

    def _run_side(self, side: Side, fail_state: State) -> bool:
        """Capture and infer one side. On failure, enter FAULT and return False."""
        try:
            frames = self._capture_side(side)
            self.state = State.INFER_TOP if side is Side.TOP else State.INFER_BOTTOM
            results = [self._model.predict(f) for f in frames]
        except Exception as exc:  # CameraError, StationError or anything from the model
            self._fail(exc, stage=side.value, return_state=fail_state)
            return False
        if side is Side.TOP:
            self._top_frames, self._top_results = frames, results
        else:
            self._bottom_frames, self._bottom_results = frames, results
        return True

    def _capture_side(self, side: Side) -> list[Frame]:
        retakes = 0
        while True:
            self.state = State.CAPTURE_TOP if side is Side.TOP else State.CAPTURE_BOTTOM
            frames = list(self._cameras.capture(side))
            self._check_frame_set(side, frames)
            rejected = [f.camera_id for f in frames if not self._frame_ok(f)]
            self._log("capture", {"side": side.value, "attempt": retakes + 1,
                                  "frames": len(frames), "rejected": rejected},
                      session_id=self._session_id)
            if not rejected:
                return frames
            if retakes >= self._max_retakes:
                raise StationError(
                    f"{side.value}: frame check failed after {retakes} retake(s); cameras {rejected}"
                )
            retakes += 1

    @staticmethod
    def _check_frame_set(side: Side, frames: list[Frame]) -> None:
        if len(frames) != len(CAMERA_IDS):
            raise CameraError(f"{side.value}: expected {len(CAMERA_IDS)} frames, got {len(frames)}")
        if sorted(f.camera_id for f in frames) != sorted(CAMERA_IDS):
            raise CameraError(
                f"{side.value}: camera ids {sorted(f.camera_id for f in frames)} do not match {list(CAMERA_IDS)}"
            )

    def _fuse_and_route(self) -> None:
        results = self._all_results()
        try:
            if len(results) != IMAGES_PER_PART:
                raise StationError(f"need {IMAGES_PER_PART} results to fuse, have {len(results)}")
            fuse = self._fuse_fn
            if fuse is None:
                from station.fusion import fuse as default_fuse
                fuse = default_fuse
            pv = fuse(results, self._thresholds)
            if pv.verdict not in (Verdict.GOOD, Verdict.BAD, Verdict.REVIEW):
                raise StationError(f"fusion returned unknown verdict {pv.verdict!r}")
        except Exception as exc:
            self._fail(exc, stage="fuse", return_state=State.WAIT_FLIP)
            return

        self._machine = pv.verdict
        self._log("verdict", {"verdict": pv.verdict.value, "reason": pv.reason, "part_id": self._part_id},
                  session_id=self._session_id)
        if pv.verdict is Verdict.GOOD:
            self._finish(Verdict.GOOD, override=None, disposition=None, lights=("show_good",))
        elif pv.verdict is Verdict.BAD:
            self.state = State.VERDICT  # set before the light: the operator must be able to ack
            self._light("show_bad")
        else:
            self.state = State.SUPERVISOR_REVIEW  # set before the light: resolve_review must work
            self._light("show_review")

    def _light(self, name: str) -> None:
        """Call one OutputDriver method for a state already set. A failure is logged, never raised."""
        try:
            getattr(self._outputs, name)()
        except Exception as exc:
            self._log("outputs_failed",
                      {"part_id": self._part_id, "error": f"{name}: {str(exc) or type(exc).__name__}"},
                      session_id=self._session_id)

    def _finish(self, final: Verdict, override: Optional[Override], disposition: Optional[str],
                lights: tuple[str, ...]) -> None:
        """Save and count a completed part exactly once, then light the outputs and return to IDLE.

        On a save failure the session enters FAULT and keeps the decision in `_pending`.
        """
        self._pending = _PendingCompletion(final=final, override=override,
                                           disposition=disposition, lights=lights)
        self._try_complete()

    def _try_complete(self) -> bool:
        """Attempt the save of `_pending`. Returns True if the part was saved."""
        p = self._pending
        if p is None:
            return False
        record = self._build_record(p)
        try:
            self._store.save_part(record)
        except Exception as exc:
            self._fail(exc, stage="save", return_state=None)
            return False

        self._pending = None
        self._fault = None
        self._fault_return = None
        self._final = p.final
        self._override = p.override
        self._disposition = p.disposition
        self.counters["total"] += 1
        self.counters["good" if p.final is Verdict.GOOD else "bad"] += 1
        self.state = State.IDLE
        # Every light is attempted even if an earlier one raises (e.g. alarm_off
        # still runs when show_bad raises). A light failure never undoes the save.
        light_errors: list[str] = []
        for name in p.lights:
            try:
                getattr(self._outputs, name)()
            except Exception as exc:
                light_errors.append(f"{name}: {str(exc) or type(exc).__name__}")
        if light_errors:
            self._log("outputs_failed",
                      {"part_id": self._part_id, "error": "; ".join(light_errors)},
                      session_id=self._session_id)
        if self._on_complete is not None:
            try:
                self._on_complete(record)
            except Exception as exc:
                self._log("on_complete_failed",
                          {"part_id": self._part_id, "error": str(exc) or type(exc).__name__},
                          session_id=self._session_id)
        return True

    def _build_record(self, p: _PendingCompletion) -> PartRecord:
        return PartRecord(
            part_id=self._part_id,
            session_id=self._session_id,
            station_id=self._station_id,
            part_number=self._part_number,
            lot=self._lot,
            operator=self._operator,
            supervisor=self._supervisor,
            material_handler=self._material_handler,
            started_utc=self._started_utc,
            model_version=self._model.version,
            thresholds=self._thresholds,
            frames=self._all_frames(),
            results=self._all_results(),
            machine_verdict=self._machine,
            final_verdict=p.final,
            override=p.override,
            disposition=p.disposition,
        )

    def _fail(self, exc: BaseException, stage: str, return_state: Optional[State]) -> None:
        text = str(exc) or type(exc).__name__
        self._fault = text
        self._fault_return = return_state
        self.state = State.FAULT
        self._light("show_fault")
        self._log("fault", {"stage": stage, "error": text,
                            "return_state": return_state.value if return_state is not None else "retry_save"},
                  session_id=self._session_id)

    @staticmethod
    def _check_decision(by: str, new_verdict: Verdict, reason: str) -> Verdict:
        if new_verdict not in (Verdict.GOOD, Verdict.BAD):
            raise ValueError("new_verdict must be GOOD or BAD")
        if not isinstance(by, str) or not by.strip():
            raise ValueError("by is required")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason is required")
        return Verdict(new_verdict)

    def _all_frames(self) -> list[Frame]:
        return list(self._top_frames) + list(self._bottom_frames)

    def _all_results(self) -> list[ImageResult]:
        return list(self._top_results) + list(self._bottom_results)

    def _active_session_id(self) -> Optional[str]:
        return None if self.state is State.IDLE else self._session_id

    def _log(self, kind: str, payload: dict[str, Any], *, session_id: Optional[str]) -> None:
        self._store.append_event(kind, payload, session_id=session_id)

    def _clear_part(self) -> None:
        self._part_id: Optional[str] = None
        self._session_id: Optional[str] = None
        self._started_utc: Optional[str] = None
        self._top_frames: list[Frame] = []
        self._top_results: list[ImageResult] = []
        self._bottom_frames: list[Frame] = []
        self._bottom_results: list[ImageResult] = []
        self._machine: Optional[Verdict] = None
        self._final: Optional[Verdict] = None
        self._override: Optional[Override] = None
        self._disposition: Optional[str] = None
        self._pending: Optional[_PendingCompletion] = None
        self._fault = None
        self._fault_return = None

    def _begin_part(self) -> None:
        self._clear_part()
        self._part_id = str(self._part_id_factory())
        self._session_id = str(self._session_id_factory())
        self._started_utc = self._now_utc()
