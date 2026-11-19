from __future__ import annotations

import base64
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from station.storage import COLLECTED
from station.defect_proposal import (DefectProposal, ProposalRecord, add_proposal, check_proposal,
                                     remove_proposal)
from station.contracts import (
    CAMERA_IDS,
    IMAGES_PER_PART,
    CameraError,
    Frame,
    ImageResult,
    InvalidTransition,
    Override,
    images_for,
    is_full_station,
    PartRecord,
    PartVerdict,
    Side,
    State,
    StationError,
    Thresholds,
    Verdict,
)


COLLECTED_LABEL = "COLLECTED"


@dataclass(frozen=True)
class _PendingCompletion:
    final: Verdict
    override: Optional[Override]
    disposition: Optional[str]
    lights: tuple[str, ...]


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


def _version_of(model) -> str:
    return "" if model is None else str(getattr(model, "version", "") or "")


def _validate_camera_ids(camera_ids: Sequence[str]) -> tuple[str, ...]:
    ids = tuple(camera_ids)
    unknown = [c for c in ids if c not in CAMERA_IDS]
    if unknown:
        raise ValueError(f"unknown camera ids {unknown}; must be a subset of {list(CAMERA_IDS)}")
    if not ids:
        raise ValueError("camera_ids must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError(f"camera_ids must be unique, got {list(ids)}")
    return ids


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
        min_flip_s: float = 2.0,
        frame_ok: Optional[Callable[[Frame], bool]] = None,
        max_retakes: int = 2,
        clock: Callable[[], float] = time.monotonic,
        now_utc: Callable[[], str],
        part_id_factory: Callable[[], str],
        session_id_factory: Callable[[], str] = _default_session_id,
        fuse_fn: Optional[Callable[[Sequence[ImageResult], Thresholds], PartVerdict]] = None,
        on_complete: Optional[Callable[[PartRecord], None]] = None,
        initial_counters: Optional[dict[str, int]] = None,
        camera_ids: Sequence[str] = CAMERA_IDS,
        collect_only: bool = False,
    ) -> None:
        self._camera_ids = _validate_camera_ids(camera_ids)
        self._collect_only = bool(collect_only)
        self._expected_images = images_for(self._camera_ids)
        self._cameras = cameras
        self._model = model
        self._model_version = _version_of(model)
        self._outputs = outputs
        self._store = store
        self._thresholds = thresholds
        self._station_id = station_id
        self._part_number = part_number or ""
        self._lot = lot
        self._operator = operator
        self._supervisor = supervisor
        self._part_operator: Optional[str] = None
        self._station_session_by_part: dict[str, str] = {}
        self._material_handler = material_handler
        self._flip_timeout_s = flip_timeout_s
        if min_flip_s < 0:
            raise ValueError(f"min_flip_s must be >= 0, got {min_flip_s}")
        self._debounce_s = debounce_s
        self._min_flip_s = float(min_flip_s)
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
        self._last_done: Optional[float] = None
        self._top_done_at: Optional[float] = None
        self._fault: Optional[str] = None
        self._fault_return: Optional[State] = None
        self._block_reason: str = ""
        self._completed_proposals: dict[str, list[ProposalRecord]] = {}
        self._clear_part()

    def press(self, by: Optional[str] = None, station_session: Optional[str] = None) -> bool:
        if self.state is State.IDLE and not self.can_inspect:
            self._log("press_ignored", {"reason": "no_part_or_model", "state": self.state.value,
                                        "block_reason": self._block_reason},
                      session_id=self._active_session_id())
            return False
        now = self._clock()
        if not self._debounce_ok(now):
            self._log("press_ignored", {"reason": "debounce", "state": self.state.value},
                      session_id=self._active_session_id())
            return False
        if self.state not in (State.IDLE, State.WAIT_FLIP):
            self._log("press_ignored", {"reason": "state", "state": self.state.value},
                      session_id=self._active_session_id())
            return False
        wait = self._flip_wait(now)
        if wait > 0:
            self._log("press_ignored", {"reason": "flip_too_soon", "state": self.state.value,
                                        "seconds_remaining": round(wait, 1)},
                      session_id=self._active_session_id())
            return False

        if (self.state is State.WAIT_FLIP and by is not None and self._part_operator is not None
                and by != self._part_operator):
            self._station_session_by_part.pop(self._part_id, None)
            self._fail(StationError(
                "another inspector pressed the flip for this part, so it is not recorded under "
                "their name. Clear the fault to discard the part."),
                stage="inspector_changed", return_state=None)
            return False

        self._last_press = now
        try:
            if self.state is State.IDLE:
                self._outputs.clear()
                self._begin_part()
                self._part_operator = by
                if by is not None and station_session is not None:
                    self._station_session_by_part[str(self._part_id)] = station_session
                self._log("press", {"side": Side.TOP.value, "part_id": self._part_id, **self._station_fields()},
                          session_id=self._session_id)
                if self._run_side(Side.TOP, fail_state=State.IDLE):
                    self.state = State.WAIT_FLIP
                    self._top_done_at = self._clock()
            else:
                self._log("press", {"side": Side.BOTTOM.value, "part_id": self._part_id, **self._station_fields()},
                          session_id=self._session_id)
                self._run_bottom()
        finally:
            self._last_done = self._clock()
        return True

    def propose_defect(self, proposal: DefectProposal, *, by: str, criteria_ids) -> bool:
        if self._machine is Verdict.GOOD or self._final is Verdict.GOOD:
            raise ValueError("proposing a defect on a part the model passed is not supported yet")
        if self.state not in (State.VERDICT, State.SUPERVISOR_REVIEW):
            raise InvalidTransition(f"propose_defect is not allowed in {self.state.value}")
        if not isinstance(by, str) or not by.strip():
            raise ValueError("by is required")
        check_proposal(proposal, criteria_ids=criteria_ids, image_keys=self._image_keys())
        row = ProposalRecord(criterion_id=proposal.criterion_id, camera_id=proposal.camera_id,
                             side=proposal.side, note=proposal.note, proposed_by=by.strip(),
                             proposed_at=self._now_utc())
        self._proposals, replaced = add_proposal(self._proposals, row)
        self._log("defect_proposed", {"part_id": self._part_id, "criterion_id": row.criterion_id,
                                      "camera_id": row.camera_id, "side": row.side.value,
                                      "replaced": replaced, "by": row.proposed_by},
                  session_id=self._session_id)
        return replaced

    def remove_defect(self, criterion_id: str) -> bool:
        if self.state not in (State.VERDICT, State.SUPERVISOR_REVIEW):
            raise InvalidTransition(f"remove_defect is not allowed in {self.state.value}")
        rows, removed = remove_proposal(self._proposals, criterion_id)
        if removed:
            self._proposals = rows
            self._log("defect_removed", {"part_id": self._part_id, "criterion_id": criterion_id},
                      session_id=self._session_id)
        return removed

    def proposals_for(self, part_id: str) -> list[ProposalRecord]:
        return self._completed_proposals.pop(str(part_id), [])

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
        if self.state is not State.FAULT:
            raise InvalidTransition(f"clear_fault is not allowed in {self.state.value}")
        self._outputs.clear()
        pending = self._pending
        back = State.IDLE if pending is not None else (self._fault_return or State.IDLE)
        self._log("fault_cleared", {"return_state": back.value, "part_id": self._part_id,
                                    "retry_save": pending is not None},
                  session_id=self._session_id)
        if pending is not None:
            self._try_complete()
            return
        self._fault = None
        self._fault_return = None
        if back is State.IDLE:
            self._clear_part()
        elif back is State.WAIT_FLIP:
            self._last_press = self._clock()
            self._top_done_at = -math.inf
        self.state = back

    def set_part(self, part_number: str, model, model_version: str) -> None:
        if self.state is not State.IDLE:
            raise InvalidTransition(f"set_part is not allowed in {self.state.value}")
        number = str(part_number or "").strip()
        if not number:
            raise ValueError("part_number is required")
        if model is None:
            raise ValueError("model is required")
        version = str(model_version or "")
        self._part_number = number
        self._model = model
        self._model_version = version
        self._block_reason = ""
        self._log("part_selected", {"part_number": number, "model_version": version}, session_id=None)

    def clear_part(self) -> None:
        if self.state is not State.IDLE:
            raise InvalidTransition(f"clear_part is not allowed in {self.state.value}")
        self._part_number = ""
        self._model = None
        self._model_version = ""
        self._block_reason = ""
        self._log("part_cleared", {}, session_id=None)

    def set_blocked(self, reason: Optional[str]) -> None:
        if self.state is not State.IDLE:
            raise InvalidTransition(f"set_blocked is not allowed in {self.state.value}")
        self._block_reason = "" if reason is None else str(reason).strip()

    @property
    def part_number(self) -> str:
        return self._part_number

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def can_inspect(self) -> bool:
        return bool(self._part_number) and self._model is not None and not self._block_reason

    @property
    def block_reason(self) -> str:
        return self._block_reason

    def snapshot(self, include_images: bool = False) -> dict[str, Any]:
        now = self._clock()
        wait = self._flip_wait(now)
        results = self._all_results()
        by_key = {(r.camera_id, r.side): r for r in results}
        images: list[dict[str, Any]] = []
        for f in self._all_frames()[:self._expected_images]:
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
            "partial_station": self.is_partial,
            "camera_ids": list(self._camera_ids),
            "prompt": self._prompt(wait),
            "press_ready": self._press_ready(now),
            "flip_wait_s": round(wait, 1),
            "part_number": self._part_number,
            "part_selected": bool(self._part_number),
            "model_version": self._model_version,
            "can_inspect": self.can_inspect,
            "block_reason": self._block_reason,
            "defect_proposals": ([] if self.state is State.IDLE
                                 else [r.as_dict() for r in self._proposals]),
        }

    @property
    def is_partial(self) -> bool:
        return not is_full_station(self._camera_ids)

    def preview(self, camera_id: str) -> Optional[bytes]:
        if camera_id not in self._camera_ids:
            return None
        try:
            fn = getattr(self._cameras, "preview", None)
            if not callable(fn):
                return None
            return fn(camera_id)
        except Exception:
            return None

    def _debounce_ok(self, now: float) -> bool:
        return self._last_done is None or now - self._last_done >= self._debounce_s

    def _flip_wait(self, now: float) -> float:
        if self.state is not State.WAIT_FLIP or self._top_done_at is None:
            return 0.0
        return max(0.0, self._min_flip_s - (now - self._top_done_at))

    def _press_ready(self, now: float) -> bool:
        if self.state is State.IDLE and not self.can_inspect:
            return False
        return (self.state in (State.IDLE, State.WAIT_FLIP)
                and self._debounce_ok(now)
                and self._flip_wait(now) <= 0)

    def _prompt(self, wait: float) -> str:
        st = self.state
        if st is State.IDLE:
            if not self.can_inspect:
                if not self._part_number:
                    return "Choose a part number to start."
                if self._block_reason:
                    return self._block_reason
                return "No approved model for this part."
            return "Step 1: place the part TOP side up, then press Enter."
        if st is State.WAIT_FLIP:
            if wait > 0:
                return f"Step 2: flip the part to the BOTTOM side. You can press Enter in {math.ceil(wait)} s."
            return "Step 2: flip the part to the BOTTOM side, then press Enter."
        if st is State.VERDICT:
            return "BAD: check the part, then acknowledge the alarm."
        if st is State.SUPERVISOR_REVIEW:
            return "REVIEW: a supervisor must decide GOOD or BAD."
        if st is State.FAULT:
            return f"Fault: {self._fault}" if self._fault else "Fault: clear it, then start again."
        return ""

    def _station_fields(self) -> dict[str, Any]:
        return {"partial_station": self.is_partial, "camera_ids": list(self._camera_ids)}

    def _display_verdict(self) -> Optional[str]:
        if self._disposition == COLLECTED:
            return COLLECTED_LABEL
        if self.state is State.SUPERVISOR_REVIEW:
            return Verdict.REVIEW.value
        if self._final is not None:
            return self._final.value
        if self._machine is None or self._machine is Verdict.GOOD:
            return None
        return self._machine.value

    def _run_bottom(self) -> None:
        self._bottom_frames = []
        self._bottom_results = []
        if not self._run_side(Side.BOTTOM, fail_state=State.WAIT_FLIP):
            return
        self._fuse_and_route()

    def _run_side(self, side: Side, fail_state: State) -> bool:
        try:
            frames = self._capture_side(side)
            self.state = State.INFER_TOP if side is Side.TOP else State.INFER_BOTTOM
            results = [self._model.predict(f) for f in frames]
        except Exception as exc:
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

    def _check_frame_set(self, side: Side, frames: list[Frame]) -> None:
        if len(frames) != len(self._camera_ids):
            raise CameraError(f"{side.value}: expected {len(self._camera_ids)} frames, got {len(frames)}")
        if sorted(f.camera_id for f in frames) != sorted(self._camera_ids):
            raise CameraError(
                f"{side.value}: camera ids {sorted(f.camera_id for f in frames)} do not match {list(self._camera_ids)}"
            )

    def _fuse_and_route(self) -> None:
        results = self._all_results()
        if self._collect_only:
            self._route_collected(results)
            return
        try:
            if len(results) != self._expected_images:
                raise StationError(f"need {self._expected_images} results to fuse, have {len(results)}")
            fuse = self._fuse_fn
            if fuse is None:
                from station.fusion import fuse as default_fuse
                fuse = default_fuse
            if self._expected_images != IMAGES_PER_PART:
                pv = fuse(results, self._thresholds, expected=self._expected_images)
            else:
                pv = fuse(results, self._thresholds)
            if pv.verdict not in (Verdict.GOOD, Verdict.BAD, Verdict.REVIEW):
                raise StationError(f"fusion returned unknown verdict {pv.verdict!r}")
        except Exception as exc:
            self._fail(exc, stage="fuse", return_state=State.WAIT_FLIP)
            return

        self._machine = pv.verdict
        self._log("verdict", {"verdict": pv.verdict.value, "reason": pv.reason, "part_id": self._part_id,
                              **self._station_fields()},
                  session_id=self._session_id)
        if pv.verdict is Verdict.GOOD:
            self._finish(Verdict.GOOD, override=None, disposition=None, lights=("show_good",))
        elif pv.verdict is Verdict.BAD:
            self.state = State.VERDICT
            self._light("show_bad")
        else:
            self.state = State.SUPERVISOR_REVIEW
            self._light("show_review")

    def _route_collected(self, results: list[ImageResult]) -> None:
        if len(results) != self._expected_images:
            self._fail(StationError(f"need {self._expected_images} results, have {len(results)}"),
                       stage="collect", return_state=State.WAIT_FLIP)
            return
        self._machine = Verdict.REVIEW
        self._log("collected", {"part_id": self._part_id, "images": len(results),
                                **self._station_fields()},
                  session_id=self._session_id)
        self._finish(Verdict.REVIEW, override=None, disposition=COLLECTED, lights=("clear",))

    def _light(self, name: str) -> None:
        try:
            getattr(self._outputs, name)()
        except Exception as exc:
            self._log("outputs_failed",
                      {"part_id": self._part_id, "error": f"{name}: {str(exc) or type(exc).__name__}"},
                      session_id=self._session_id)

    def _finish(self, final: Verdict, override: Optional[Override], disposition: Optional[str],
                lights: tuple[str, ...]) -> None:
        if self._proposals and final is Verdict.GOOD:
            self._log("defect_proposal_forces_bad", {"part_id": self._part_id,
                                                     "proposals": len(self._proposals)},
                      session_id=self._session_id)
            final, override, disposition = Verdict.BAD, None, "quarantine"
            lights = ("show_bad", "alarm_off")
        self._pending = _PendingCompletion(final=final, override=override,
                                           disposition=disposition, lights=lights)
        self._try_complete()

    def _try_complete(self) -> bool:
        p = self._pending
        if p is None:
            return False
        record = self._build_record(p)
        rows = list(self._proposals)
        try:
            if rows:
                self._store.save_part(record, defect_proposals=[r.as_dict() for r in rows])
            else:
                self._store.save_part(record)
        except Exception as exc:
            self._fail(exc, stage="save", return_state=None)
            return False

        if rows:
            self._completed_proposals[str(record.part_id)] = rows
        self._pending = None
        self._fault = None
        self._fault_return = None
        self._final = p.final
        self._override = p.override
        self._disposition = p.disposition
        self.counters["total"] += 1
        if p.disposition != COLLECTED:
            self.counters["good" if p.final is Verdict.GOOD else "bad"] += 1
        self.state = State.IDLE
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
            operator=self._part_operator or self._operator,
            supervisor=self._supervisor,
            material_handler=self._material_handler,
            started_utc=self._started_utc,
            model_version=self._model_version,
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

    def station_session_for(self, part_id: str) -> Optional[str]:
        return self._station_session_by_part.pop(part_id, None)

    @staticmethod
    def _check_decision(by: str, new_verdict: Verdict, reason: str) -> Verdict:
        if new_verdict not in (Verdict.GOOD, Verdict.BAD):
            raise ValueError("new_verdict must be GOOD or BAD")
        if not isinstance(by, str) or not by.strip():
            raise ValueError("by is required")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason is required")
        return Verdict(new_verdict)

    def _image_keys(self) -> list[tuple[str, Side]]:
        return [(cid, side) for cid in self._camera_ids for side in (Side.TOP, Side.BOTTOM)]

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
        self._proposals: list[ProposalRecord] = []
        self._fault = None
        self._fault_return = None

    def _begin_part(self) -> None:
        self._clear_part()
        self._part_id = str(self._part_id_factory())
        self._session_id = str(self._session_id_factory())
        self._started_utc = self._now_utc()
