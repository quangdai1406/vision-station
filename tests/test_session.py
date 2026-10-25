"""Tests for station.session.StationSession. Fakes only; time is injected, no sleeping."""
import base64
import unittest

from station.contracts import (
    CAMERA_IDS,
    CameraError,
    Defect,
    Frame,
    ImageResult,
    InvalidTransition,
    PartRecord,
    PartVerdict,
    Side,
    State,
    StationError,
    Thresholds,
    Verdict,
)
from station.session import StationSession

TH = Thresholds(t_low=0.3, t_img=0.6)
TS = "2026-10-23T08:00:00Z"


def make_frames(side, n=4):
    return [
        Frame(camera_id=cid, side=side, seq=i, timestamp_utc=TS,
              data=f"{side.value}-{cid}".encode())
        for i, cid in enumerate(CAMERA_IDS[:n])
    ]


class FakeCameras:
    def __init__(self, default_n=4):
        self.sides = []
        self.default_n = default_n
        self.script = []  # per call: None -> default_n frames, int -> that many frames, list -> those frames, exception -> raise
        self.on_capture = None  # optional hook, called at the start of each capture (e.g. to advance a fake clock)

    def capture(self, side):
        self.sides.append(side)
        if self.on_capture is not None:
            self.on_capture(side)
        action = self.script.pop(0) if self.script else None
        if isinstance(action, BaseException):
            raise action
        if isinstance(action, list):
            return action
        return make_frames(side, action or self.default_n)


class FakeModel:
    version = "model-test-1"

    def __init__(self):
        self.fail_with = None
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return ImageResult(
            camera_id=frame.camera_id, side=frame.side, score=0.1,
            defects=(Defect("scratch", 0.1, (1, 2, 3, 4)),),
        )


class FakeOutputs:
    def __init__(self):
        self.calls = []

    def show_good(self):
        self.calls.append("show_good")

    def show_bad(self):
        self.calls.append("show_bad")

    def show_review(self):
        self.calls.append("show_review")

    def show_fault(self):
        self.calls.append("show_fault")

    def alarm_off(self):
        self.calls.append("alarm_off")

    def clear(self):
        self.calls.append("clear")


class FakeStore:
    def __init__(self):
        self.saved = []
        self.events = []
        self.save_script = []  # per save: None -> ok, exception -> raise
        self.on_save = None    # optional hook, called before each save attempt

    def save_part(self, record):
        if self.on_save is not None:
            self.on_save()
        action = self.save_script.pop(0) if self.save_script else None
        if isinstance(action, BaseException):
            raise action
        self.saved.append(record)

    def append_event(self, kind, payload, session_id=None):
        self.events.append((kind, payload, session_id))

    def kinds(self):
        return [e[0] for e in self.events]


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def advance(self, dt):
        self.t += dt

    def __call__(self):
        return self.t


class FakeFuse:
    def __init__(self, verdict=Verdict.GOOD):
        self.verdict = verdict
        self.calls = []

    def __call__(self, results, th):
        self.calls.append(list(results))
        assert len(results) == 8
        return PartVerdict(verdict=self.verdict, results=tuple(results), reason="fake")


class _Ids:
    def __init__(self, prefix):
        self.prefix = prefix
        self.n = 0

    def __call__(self):
        self.n += 1
        return f"{self.prefix}-{self.n}"


class RecordingFuse:
    """Fuse fake that accepts any result count and records the keyword arguments it was called with."""

    def __init__(self, verdict=Verdict.GOOD):
        self.verdict = verdict
        self.calls = []
        self.kwargs = []

    def __call__(self, results, th, **kwargs):
        self.calls.append(list(results))
        self.kwargs.append(kwargs)
        return PartVerdict(verdict=self.verdict, results=tuple(results), reason="fake")


KEEP_DEFAULT = object()  # Harness(min_flip_s=KEEP_DEFAULT) builds the session without the argument (default 2.0)


class PreviewCameras(FakeCameras):
    """FakeCameras that also has a preview() passthrough."""

    def __init__(self, error=None, **kw):
        super().__init__(**kw)
        self.error = error  # exception to raise from preview(), if set
        self.previews = []

    def preview(self, camera_id):
        self.previews.append(camera_id)
        if self.error is not None:
            raise self.error
        return f"preview-{camera_id}".encode()


class Harness:
    def __init__(self, fuse_verdict=Verdict.GOOD, frame_ok=None, max_retakes=2,
                 on_complete=None, initial_counters=None, camera_ids=CAMERA_IDS, fuse=None,
                 min_flip_s=0.0, cameras=None, part_number="PN-1", with_model=True):
        self.cameras = cameras if cameras is not None else FakeCameras(default_n=len(camera_ids))
        self.model = FakeModel()
        self.outputs = FakeOutputs()
        self.store = FakeStore()
        self.clock = FakeClock()
        self.fuse = fuse if fuse is not None else FakeFuse(fuse_verdict)
        self.completed = []
        self.session = StationSession(
            self.cameras, self.model if with_model else None, self.outputs, self.store, TH,
            station_id="ST-1", part_number=part_number, lot="LOT-1", operator="op",
            supervisor="sup", material_handler="mh",
            frame_ok=frame_ok, max_retakes=max_retakes, clock=self.clock,
            now_utc=lambda: TS, part_id_factory=_Ids("part"),
            session_id_factory=_Ids("sess"), fuse_fn=self.fuse,
            on_complete=on_complete or self.completed.append,
            initial_counters=initial_counters,
            camera_ids=camera_ids,
            **({} if min_flip_s is KEEP_DEFAULT else {"min_flip_s": min_flip_s}),
        )

    def press_at(self, t):
        self.clock.t = t
        return self.session.press()

    def complete_part(self, t0=0.0):
        """Two accepted presses: TOP then BOTTOM."""
        assert self.press_at(t0) is True
        assert self.press_at(t0 + 1.0) is True


class _Base(unittest.TestCase):
    def harness(self, **kw):
        return Harness(**kw)


class GoodPathTests(_Base):
    def test_full_good_path(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        s = h.session
        self.assertTrue(h.press_at(0.0))
        self.assertEqual(s.state, State.WAIT_FLIP)
        self.assertTrue(h.press_at(1.0))

        self.assertEqual(s.state, State.IDLE)
        self.assertEqual(h.outputs.calls[-1], "show_good")
        self.assertNotIn("show_bad", h.outputs.calls)
        self.assertEqual(len(h.store.saved), 1)
        self.assertEqual(s.counters, {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(len(h.completed), 1)
        self.assertIs(h.completed[0], h.store.saved[0])

        rec = h.store.saved[0]
        self.assertIsInstance(rec, PartRecord)
        self.assertEqual(len(rec.frames), 8)
        self.assertEqual(len(rec.results), 8)
        self.assertEqual(rec.machine_verdict, Verdict.GOOD)
        self.assertEqual(rec.final_verdict, Verdict.GOOD)
        self.assertIsNone(rec.disposition)
        self.assertIsNone(rec.override)
        self.assertEqual(rec.model_version, "model-test-1")
        self.assertEqual(rec.part_id, "part-1")
        self.assertEqual(rec.session_id, "sess-1")
        self.assertEqual([f.side for f in rec.frames], [Side.TOP] * 4 + [Side.BOTTOM] * 4)
        self.assertEqual(len(h.fuse.calls[0]), 8)
        self.assertEqual(h.cameras.sides, [Side.TOP, Side.BOTTOM])

        kinds = h.store.kinds()
        for k in ("press", "capture", "verdict"):
            self.assertIn(k, kinds)

    def test_green_stays_until_next_press_clears_it(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.complete_part(0.0)
        self.assertEqual(h.outputs.calls[-1], "show_good")
        self.assertTrue(h.press_at(5.0))
        self.assertEqual(h.outputs.calls[-1], "clear")
        self.assertEqual(h.session.state, State.WAIT_FLIP)
        self.assertEqual(h.session.counters["total"], 1)


class BadPathTests(_Base):
    def test_bad_waits_for_ack_and_is_not_saved_yet(self):
        h = self.harness(fuse_verdict=Verdict.BAD)
        h.complete_part(0.0)
        self.assertEqual(h.session.state, State.VERDICT)
        self.assertEqual(h.outputs.calls[-1], "show_bad")
        self.assertEqual(h.store.saved, [])
        self.assertEqual(h.session.counters, {"total": 0, "good": 0, "bad": 0})
        self.assertEqual(h.completed, [])

    def test_ack_alarm_completes_as_bad_and_keeps_red(self):
        h = self.harness(fuse_verdict=Verdict.BAD)
        h.complete_part(0.0)
        h.session.ack_alarm()
        self.assertEqual(h.outputs.calls[-2:], ["show_bad", "alarm_off"])
        self.assertNotIn("clear", h.outputs.calls[h.outputs.calls.index("show_bad"):])
        self.assertNotIn("show_good", h.outputs.calls)
        self.assertEqual(h.session.state, State.IDLE)
        self.assertEqual(len(h.store.saved), 1)
        rec = h.store.saved[0]
        self.assertEqual(rec.machine_verdict, Verdict.BAD)
        self.assertEqual(rec.final_verdict, Verdict.BAD)
        self.assertEqual(rec.disposition, "quarantine")
        self.assertIsNone(rec.override)
        self.assertEqual(h.session.counters, {"total": 1, "good": 0, "bad": 1})
        self.assertEqual(len(h.completed), 1)
        self.assertIn("ack", h.store.kinds())

    def test_press_ignored_in_verdict(self):
        h = self.harness(fuse_verdict=Verdict.BAD)
        h.complete_part(0.0)
        self.assertFalse(h.press_at(10.0))
        self.assertEqual(h.session.state, State.VERDICT)
        self.assertIn("press_ignored", h.store.kinds())

    def test_override_bad_to_good(self):
        h = self.harness(fuse_verdict=Verdict.BAD)
        h.complete_part(0.0)
        h.session.override("supervisor-1", Verdict.GOOD, "re-measured, within spec")
        self.assertEqual(h.outputs.calls[-3:], ["show_bad", "alarm_off", "show_good"])
        self.assertEqual(h.session.state, State.IDLE)
        rec = h.store.saved[0]
        self.assertEqual(rec.machine_verdict, Verdict.BAD)
        self.assertEqual(rec.final_verdict, Verdict.GOOD)
        self.assertEqual(rec.disposition, "released")
        self.assertEqual(rec.override.original_verdict, Verdict.BAD)
        self.assertEqual(rec.override.new_verdict, Verdict.GOOD)
        self.assertEqual(rec.override.by, "supervisor-1")
        self.assertEqual(rec.override.reason, "re-measured, within spec")
        self.assertEqual(h.session.counters, {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(len(h.completed), 1)
        self.assertIn("override", h.store.kinds())

    def test_override_bad_to_bad_is_quarantined_and_counted_bad(self):
        h = self.harness(fuse_verdict=Verdict.BAD)
        h.complete_part(0.0)
        h.session.override("supervisor-1", "BAD", "confirmed defect")
        rec = h.store.saved[0]
        self.assertEqual(rec.final_verdict, Verdict.BAD)
        self.assertEqual(rec.disposition, "quarantine")
        self.assertNotIn("show_good", h.outputs.calls)
        self.assertEqual(h.session.counters, {"total": 1, "good": 0, "bad": 1})


class ReviewPathTests(_Base):
    def test_review_never_shows_good_before_resolution(self):
        h = self.harness(fuse_verdict=Verdict.REVIEW)
        h.complete_part(0.0)
        self.assertEqual(h.session.state, State.SUPERVISOR_REVIEW)
        self.assertEqual(h.outputs.calls[-1], "show_review")
        self.assertNotIn("show_good", h.outputs.calls)
        self.assertEqual(h.store.saved, [])
        snap = h.session.snapshot()
        self.assertEqual(snap["machine_verdict"], "REVIEW")
        self.assertIsNone(snap["final_verdict"])
        self.assertEqual(h.session.counters["total"], 0)

    def test_press_ignored_in_supervisor_review(self):
        h = self.harness(fuse_verdict=Verdict.REVIEW)
        h.complete_part(0.0)
        self.assertFalse(h.press_at(10.0))
        self.assertEqual(h.session.state, State.SUPERVISOR_REVIEW)

    def test_resolve_review_to_bad(self):
        h = self.harness(fuse_verdict=Verdict.REVIEW)
        h.complete_part(0.0)
        h.session.resolve_review("sup-2", Verdict.BAD, "scratch confirmed")
        self.assertEqual(h.outputs.calls[-2:], ["show_bad", "alarm_off"])
        self.assertNotIn("show_good", h.outputs.calls)
        self.assertEqual(h.session.state, State.IDLE)
        rec = h.store.saved[0]
        self.assertEqual(rec.machine_verdict, Verdict.REVIEW)
        self.assertEqual(rec.final_verdict, Verdict.BAD)
        self.assertEqual(rec.disposition, "quarantine")
        self.assertEqual(rec.override.original_verdict, Verdict.REVIEW)
        self.assertEqual(h.session.counters, {"total": 1, "good": 0, "bad": 1})
        self.assertEqual(len(h.completed), 1)
        self.assertIn("review_resolved", h.store.kinds())

    def test_resolve_review_to_good(self):
        h = self.harness(fuse_verdict=Verdict.REVIEW)
        h.complete_part(0.0)
        h.session.resolve_review("sup-2", Verdict.GOOD, "clean on re-look")
        self.assertEqual(h.outputs.calls[-1], "show_good")
        rec = h.store.saved[0]
        self.assertEqual(rec.final_verdict, Verdict.GOOD)
        self.assertEqual(rec.disposition, "released")
        self.assertEqual(rec.override.original_verdict, Verdict.REVIEW)
        self.assertEqual(h.session.counters, {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(h.session.state, State.IDLE)


class DebounceTests(_Base):
    def test_double_press_within_debounce_is_ignored(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        self.assertTrue(h.press_at(0.0))
        self.assertFalse(h.press_at(0.1))
        self.assertEqual(h.session.state, State.WAIT_FLIP)
        self.assertEqual(h.cameras.sides, [Side.TOP])
        ignored = [e for e in h.store.events if e[0] == "press_ignored"]
        self.assertEqual(len(ignored), 1)

    def test_ignored_press_does_not_extend_the_window(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        self.assertTrue(h.press_at(0.0))
        self.assertFalse(h.press_at(0.2))   # ignored, measured from the accepted press at 0.0
        self.assertTrue(h.press_at(0.35))   # 0.35 s after the accepted press
        self.assertEqual(h.session.state, State.IDLE)
        self.assertEqual(h.session.counters["total"], 1)


class FlipTimeoutTests(_Base):
    def test_timeout_voids_without_saving_or_counting(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        self.assertTrue(h.press_at(0.0))
        h.clock.t = 60.0
        self.assertFalse(h.session.tick())
        self.assertEqual(h.session.state, State.WAIT_FLIP)
        h.clock.t = 60.5
        self.assertTrue(h.session.tick())
        self.assertEqual(h.session.state, State.IDLE)
        self.assertEqual(h.outputs.calls[-1], "clear")
        self.assertEqual(h.store.saved, [])
        self.assertEqual(h.session.counters, {"total": 0, "good": 0, "bad": 0})
        self.assertIn("session_voided", h.store.kinds())
        self.assertFalse(h.session.tick())

    def test_new_press_after_void_starts_a_fresh_part(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.press_at(0.0)
        h.clock.t = 100.0
        h.session.tick()
        h.complete_part(200.0)
        self.assertEqual(h.session.counters["total"], 1)
        self.assertEqual(h.store.saved[0].part_id, "part-2")


class CameraFaultTests(_Base):
    def test_camera_failure_on_top_then_clear_fault_resumes_idle(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.cameras.script = [CameraError("C2 timeout")]
        self.assertTrue(h.press_at(0.0))
        self.assertEqual(h.session.state, State.FAULT)
        self.assertEqual(h.outputs.calls[-1], "show_fault")
        self.assertEqual(h.store.saved, [])
        self.assertEqual(h.session.counters["total"], 0)
        faults = [e for e in h.store.events if e[0] == "fault"]
        self.assertEqual(len(faults), 1)
        self.assertIn("C2 timeout", faults[0][1]["error"])
        self.assertIn("C2 timeout", h.session.snapshot()["fault"])

        self.assertFalse(h.press_at(5.0))  # presses ignored in FAULT
        h.session.clear_fault()
        self.assertEqual(h.session.state, State.IDLE)
        self.assertEqual(h.outputs.calls[-1], "clear")
        self.assertIsNone(h.session.snapshot()["fault"])

        h.complete_part(10.0)
        self.assertEqual(h.session.counters, {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(h.cameras.sides, [Side.TOP, Side.TOP, Side.BOTTOM])

    def test_camera_failure_on_bottom_keeps_top_and_resumes_waiting_for_flip(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.cameras.script = [None, CameraError("C4 dropped")]
        self.assertTrue(h.press_at(0.0))
        self.assertTrue(h.press_at(1.0))
        self.assertEqual(h.session.state, State.FAULT)
        snap = h.session.snapshot()
        self.assertEqual(len(snap["results"]), 4)
        self.assertEqual(h.store.saved, [])
        self.assertEqual(h.session.counters["total"], 0)
        self.assertEqual(h.fuse.calls, [])

        h.session.clear_fault()
        self.assertEqual(h.session.state, State.WAIT_FLIP)
        self.assertEqual(h.outputs.calls[-1], "clear")
        self.assertTrue(h.press_at(5.0))
        self.assertEqual(h.session.state, State.IDLE)

        self.assertEqual(h.cameras.sides, [Side.TOP, Side.BOTTOM, Side.BOTTOM])
        self.assertEqual(h.session.counters, {"total": 1, "good": 1, "bad": 0})
        rec = h.store.saved[0]
        self.assertEqual(len(rec.frames), 8)
        self.assertEqual([f.side for f in rec.frames], [Side.TOP] * 4 + [Side.BOTTOM] * 4)
        self.assertEqual(len(h.fuse.calls[0]), 8)

    def test_inference_error_is_a_fault(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.model.fail_with = RuntimeError("GPU lost")
        self.assertTrue(h.press_at(0.0))
        self.assertEqual(h.session.state, State.FAULT)
        self.assertIn("GPU lost", h.session.snapshot()["fault"])
        self.assertEqual(h.store.saved, [])
        h.model.fail_with = None
        h.session.clear_fault()
        self.assertEqual(h.session.state, State.IDLE)

    def test_clear_fault_outside_fault_is_invalid(self):
        h = self.harness()
        with self.assertRaises(InvalidTransition):
            h.session.clear_fault()


class RetakeTests(_Base):
    def test_blur_then_pass_retakes_once(self):
        calls = []

        def frame_ok(frame):
            calls.append(frame)
            return len(calls) != 1  # only the very first frame is blurred

        h = self.harness(fuse_verdict=Verdict.GOOD, frame_ok=frame_ok)
        h.complete_part(0.0)
        self.assertEqual(h.cameras.sides, [Side.TOP, Side.TOP, Side.BOTTOM])
        self.assertEqual(h.session.counters["total"], 1)

    def test_blur_past_max_retakes_is_a_fault(self):
        h = self.harness(fuse_verdict=Verdict.GOOD, frame_ok=lambda f: False, max_retakes=2)
        self.assertTrue(h.press_at(0.0))
        self.assertEqual(h.session.state, State.FAULT)
        self.assertEqual(h.cameras.sides, [Side.TOP] * 3)  # first try + 2 retakes
        self.assertIn("after 2 retake", h.session.snapshot()["fault"])
        self.assertEqual(h.store.saved, [])
        self.assertEqual(h.session.counters["total"], 0)


class ShortCaptureTests(_Base):
    def test_three_frames_from_camera_is_a_fault(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.cameras.script = [3]
        self.assertTrue(h.press_at(0.0))
        self.assertEqual(h.session.state, State.FAULT)
        self.assertIn("expected 4 frames", h.session.snapshot()["fault"])
        self.assertEqual(h.fuse.calls, [])
        self.assertEqual(h.store.saved, [])
        self.assertEqual(h.session.counters["total"], 0)

    def test_short_bottom_capture_is_a_fault_and_never_fuses(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.cameras.script = [None, 3]
        h.press_at(0.0)
        h.press_at(1.0)
        self.assertEqual(h.session.state, State.FAULT)
        self.assertEqual(h.fuse.calls, [])
        self.assertEqual(h.store.saved, [])


class SnapshotTests(_Base):
    def test_snapshot_keys_idle(self):
        h = self.harness()
        snap = h.session.snapshot()
        self.assertEqual(
            set(snap),
            {"state", "counters", "part_id", "machine_verdict", "final_verdict",
             "verdict", "results", "fault", "images", "partial_station", "camera_ids",
             "prompt", "press_ready", "flip_wait_s", "part_number", "part_selected",
             "model_version", "can_inspect", "block_reason"},
        )
        self.assertEqual(snap["state"], "IDLE")
        self.assertEqual(snap["counters"], {"total": 0, "good": 0, "bad": 0})
        self.assertIsNone(snap["part_id"])
        self.assertIsNone(snap["verdict"])
        self.assertEqual(snap["results"], [])
        self.assertEqual(snap["images"], [])

    def test_snapshot_results_shape_mid_part(self):
        h = self.harness()
        h.press_at(0.0)
        snap = h.session.snapshot()
        self.assertEqual(snap["state"], "WAIT_FLIP")
        self.assertEqual(len(snap["results"]), 4)
        r = snap["results"][0]
        self.assertEqual(set(r), {"camera_id", "side", "score", "defects"})
        self.assertEqual(r["side"], "TOP")
        self.assertEqual(r["defects"][0]["box"], [1, 2, 3, 4])
        self.assertEqual(set(r["defects"][0]), {"defect_class", "score", "box"})
        self.assertIsNone(snap["verdict"])  # nothing judged yet

    def test_snapshot_images_metadata_is_present_without_b64(self):
        h = self.harness()
        h.press_at(0.0)  # TOP only: 4 frames captured, inferred
        snap = h.session.snapshot()
        self.assertEqual(len(snap["images"]), 4)
        img = snap["images"][0]
        self.assertEqual(set(img), {"camera_id", "side", "seq", "ext", "score", "defects"})
        self.assertEqual(img["side"], "TOP")
        self.assertAlmostEqual(img["score"], 0.1)
        self.assertEqual(img["defects"], [{"defect_class": "scratch", "score": 0.1, "box": [1, 2, 3, 4]}])

    def test_snapshot_images_when_requested(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.complete_part(0.0)
        snap = h.session.snapshot(include_images=True)
        self.assertEqual(snap["final_verdict"], "GOOD")
        self.assertEqual(len(snap["images"]), 8)
        img = snap["images"][0]
        self.assertEqual(set(img), {"camera_id", "side", "seq", "ext", "score", "defects", "b64"})
        self.assertEqual(base64.b64decode(img["b64"]), b"TOP-C1")
        self.assertEqual(snap["images"][4]["side"], "BOTTOM")
        self.assertNotIn("b64", h.session.snapshot()["images"][0])

    def test_snapshot_verdict_values(self):
        h = self.harness(fuse_verdict=Verdict.BAD)
        h.complete_part(0.0)
        self.assertEqual(h.session.snapshot()["verdict"], "BAD")  # shown before ack, not saved
        self.assertIsNone(h.session.snapshot()["final_verdict"])
        h.session.ack_alarm()
        self.assertEqual(h.session.snapshot()["verdict"], "BAD")
        self.assertEqual(h.session.snapshot()["final_verdict"], "BAD")

        r = self.harness(fuse_verdict=Verdict.REVIEW)
        r.complete_part(0.0)
        self.assertEqual(r.session.snapshot()["verdict"], "REVIEW")
        r.session.resolve_review("sup", Verdict.GOOD, "clean")
        self.assertEqual(r.session.snapshot()["verdict"], "GOOD")

    def test_snapshot_verdict_good_only_once_saved(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.complete_part(0.0)
        self.assertEqual(h.session.snapshot()["verdict"], "GOOD")
        self.assertEqual(h.session.snapshot()["final_verdict"], "GOOD")


class InvalidTransitionTests(_Base):
    def test_actions_in_wrong_state_raise(self):
        h = self.harness()
        with self.assertRaises(InvalidTransition):
            h.session.ack_alarm()  # IDLE
        with self.assertRaises(InvalidTransition):
            h.session.override("sup", Verdict.GOOD, "reason")  # IDLE
        with self.assertRaises(InvalidTransition):
            h.session.resolve_review("sup", Verdict.GOOD, "reason")  # IDLE
        with self.assertRaises(InvalidTransition):
            h.session.clear_fault()  # IDLE

    def test_ack_and_override_not_allowed_in_supervisor_review(self):
        h = self.harness(fuse_verdict=Verdict.REVIEW)
        h.complete_part(0.0)
        with self.assertRaises(InvalidTransition):
            h.session.ack_alarm()
        with self.assertRaises(InvalidTransition):
            h.session.override("sup", Verdict.GOOD, "reason")

    def test_resolve_review_not_allowed_in_verdict(self):
        h = self.harness(fuse_verdict=Verdict.BAD)
        h.complete_part(0.0)
        with self.assertRaises(InvalidTransition):
            h.session.resolve_review("sup", Verdict.GOOD, "reason")
        self.assertEqual(h.session.state, State.VERDICT)

    def test_ack_not_allowed_in_wait_flip(self):
        h = self.harness()
        h.press_at(0.0)
        with self.assertRaises(InvalidTransition):
            h.session.ack_alarm()


class ValueErrorTests(_Base):
    def test_override_rejects_review_and_empty_fields(self):
        h = self.harness(fuse_verdict=Verdict.BAD)
        h.complete_part(0.0)
        with self.assertRaises(ValueError):
            h.session.override("sup", Verdict.REVIEW, "reason")
        with self.assertRaises(ValueError):
            h.session.override("sup", "REVIEW", "reason")
        with self.assertRaises(ValueError):
            h.session.override("", Verdict.GOOD, "reason")
        with self.assertRaises(ValueError):
            h.session.override("sup", Verdict.GOOD, "   ")
        self.assertEqual(h.session.state, State.VERDICT)
        self.assertEqual(h.store.saved, [])

    def test_resolve_review_rejects_review_and_empty_fields(self):
        h = self.harness(fuse_verdict=Verdict.REVIEW)
        h.complete_part(0.0)
        with self.assertRaises(ValueError):
            h.session.resolve_review("sup", Verdict.REVIEW, "reason")
        with self.assertRaises(ValueError):
            h.session.resolve_review("", Verdict.GOOD, "reason")
        with self.assertRaises(ValueError):
            h.session.resolve_review("sup", Verdict.BAD, "")
        self.assertEqual(h.session.state, State.SUPERVISOR_REVIEW)
        self.assertEqual(h.store.saved, [])


class SaveBeforeLightTests(_Base):
    def test_green_is_not_lit_when_the_save_happens(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        seen = []
        h.store.on_save = lambda: seen.append(list(h.outputs.calls))
        h.complete_part(0.0)
        self.assertEqual(len(seen), 1)
        self.assertNotIn("show_good", seen[0])

    def test_override_to_good_is_not_lit_when_the_save_happens(self):
        h = self.harness(fuse_verdict=Verdict.BAD)
        h.complete_part(0.0)
        seen = []
        h.store.on_save = lambda: seen.append(list(h.outputs.calls))
        h.session.override("sup", Verdict.GOOD, "re-measured")
        self.assertEqual(len(seen), 1)
        self.assertNotIn("show_good", seen[0])
        self.assertNotIn("alarm_off", seen[0])
        self.assertEqual(h.outputs.calls[-1], "show_good")

    def test_resolve_review_to_good_is_not_lit_when_the_save_happens(self):
        h = self.harness(fuse_verdict=Verdict.REVIEW)
        h.complete_part(0.0)
        seen = []
        h.store.on_save = lambda: seen.append(list(h.outputs.calls))
        h.session.resolve_review("sup", Verdict.GOOD, "clean")
        self.assertNotIn("show_good", seen[0])


class SaveFailureTests(_Base):
    def test_save_failure_on_good_goes_to_fault_counts_nothing_and_stays_dark(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.store.save_script = [OSError("disk full")]
        h.complete_part(0.0)  # must not raise
        s = h.session
        self.assertEqual(s.state, State.FAULT)
        self.assertEqual(h.outputs.calls[-1], "show_fault")
        self.assertNotIn("show_good", h.outputs.calls)
        self.assertEqual(h.store.saved, [])
        self.assertEqual(s.counters, {"total": 0, "good": 0, "bad": 0})
        self.assertEqual(h.completed, [])
        faults = [e for e in h.store.events if e[0] == "fault"]
        self.assertEqual(len(faults), 1)
        self.assertIn("disk full", faults[0][1]["error"])
        snap = s.snapshot()
        self.assertIn("disk full", snap["fault"])
        self.assertIsNone(snap["final_verdict"])
        self.assertEqual(snap["machine_verdict"], "GOOD")
        self.assertIsNone(snap["verdict"])  # GOOD is not shown until it is saved
        self.assertEqual(len(snap["results"]), 8)  # results kept for the retry

    def test_clear_fault_retries_save_once_and_counts_once(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.store.save_script = [OSError("disk full")]
        h.complete_part(0.0)
        h.session.clear_fault()
        s = h.session
        self.assertEqual(s.state, State.IDLE)
        self.assertEqual(h.outputs.calls[-1], "show_good")
        self.assertEqual(len(h.store.saved), 1)
        self.assertEqual(s.counters, {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(len(h.completed), 1)
        self.assertIsNone(s.snapshot()["fault"])
        self.assertEqual(s.snapshot()["final_verdict"], "GOOD")
        rec = h.store.saved[0]
        self.assertEqual(rec.part_id, "part-1")
        self.assertEqual(rec.session_id, "sess-1")
        self.assertEqual(len(rec.frames), 8)
        self.assertIn("fault_cleared", h.store.kinds())

    def test_clear_fault_retry_failing_again_stays_in_fault(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.store.save_script = [OSError("first"), OSError("second")]
        h.complete_part(0.0)
        h.session.clear_fault()
        s = h.session
        self.assertEqual(s.state, State.FAULT)
        self.assertIn("second", s.snapshot()["fault"])
        self.assertEqual(h.outputs.calls[-1], "show_fault")
        self.assertNotIn("show_good", h.outputs.calls)
        self.assertEqual(s.counters, {"total": 0, "good": 0, "bad": 0})
        self.assertEqual(h.store.saved, [])
        self.assertFalse(h.press_at(10.0))  # still in FAULT

        h.session.clear_fault()  # store has recovered
        self.assertEqual(s.state, State.IDLE)
        self.assertEqual(s.counters, {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(len(h.store.saved), 1)

    def test_bad_ack_save_failure_keeps_part_unsaved_and_retry_keeps_red(self):
        h = self.harness(fuse_verdict=Verdict.BAD)
        h.complete_part(0.0)
        h.store.save_script = [OSError("db locked")]
        h.session.ack_alarm()
        self.assertEqual(h.session.state, State.FAULT)
        self.assertNotIn("show_good", h.outputs.calls)
        self.assertEqual(h.session.counters, {"total": 0, "good": 0, "bad": 0})
        self.assertEqual(h.store.saved, [])

        h.session.clear_fault()
        self.assertEqual(h.session.state, State.IDLE)
        self.assertEqual(h.outputs.calls[-2:], ["show_bad", "alarm_off"])
        self.assertEqual(h.store.saved[0].final_verdict, Verdict.BAD)
        self.assertEqual(h.store.saved[0].disposition, "quarantine")
        self.assertEqual(h.session.counters, {"total": 1, "good": 0, "bad": 1})
        self.assertEqual(len(h.completed), 1)

    def test_override_to_good_save_failure_keeps_override_and_retry_releases(self):
        h = self.harness(fuse_verdict=Verdict.BAD)
        h.complete_part(0.0)
        h.store.save_script = [OSError("timeout")]
        h.session.override("sup-1", Verdict.GOOD, "re-measured")
        self.assertEqual(h.session.state, State.FAULT)
        self.assertNotIn("show_good", h.outputs.calls)
        h.session.clear_fault()
        self.assertEqual(h.session.state, State.IDLE)
        self.assertEqual(h.outputs.calls[-1], "show_good")
        rec = h.store.saved[0]
        self.assertEqual(rec.final_verdict, Verdict.GOOD)
        self.assertEqual(rec.override.by, "sup-1")
        self.assertEqual(rec.override.reason, "re-measured")
        self.assertEqual(h.session.counters, {"total": 1, "good": 1, "bad": 0})

    def test_resolve_review_save_failure_keeps_review_and_retry_completes(self):
        h = self.harness(fuse_verdict=Verdict.REVIEW)
        h.complete_part(0.0)
        h.store.save_script = [OSError("timeout")]
        h.session.resolve_review("sup-2", Verdict.BAD, "scratch confirmed")
        self.assertEqual(h.session.state, State.FAULT)
        self.assertEqual(h.session.counters["total"], 0)
        h.session.clear_fault()
        self.assertEqual(h.session.state, State.IDLE)
        rec = h.store.saved[0]
        self.assertEqual(rec.override.original_verdict, Verdict.REVIEW)
        self.assertEqual(rec.final_verdict, Verdict.BAD)
        self.assertEqual(h.session.counters, {"total": 1, "good": 0, "bad": 1})

    def test_clear_fault_after_successful_completion_is_invalid_and_nothing_resaves(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.complete_part(0.0)
        self.assertEqual(h.session.counters["total"], 1)
        with self.assertRaises(InvalidTransition):
            h.session.clear_fault()
        self.assertEqual(len(h.store.saved), 1)
        self.assertEqual(h.session.counters["total"], 1)

    def test_on_complete_failure_keeps_part_saved_and_counted_without_fault(self):
        def boom(record):
            raise ValueError("mes down")

        h = self.harness(fuse_verdict=Verdict.GOOD, on_complete=boom)
        h.complete_part(0.0)
        s = h.session
        self.assertEqual(s.state, State.IDLE)
        self.assertEqual(len(h.store.saved), 1)
        self.assertEqual(s.counters, {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(h.outputs.calls[-1], "show_good")
        self.assertNotIn("show_fault", h.outputs.calls)
        self.assertNotIn("fault", h.store.kinds())
        failed = [e for e in h.store.events if e[0] == "on_complete_failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn("mes down", failed[0][1]["error"])
        self.assertIsNone(s.snapshot()["fault"])


class RaisingOutputs(FakeOutputs):
    """OutputDriver whose named methods raise after recording the call."""

    def __init__(self, raising):
        super().__init__()
        self.raising = set(raising)
        self.armed = False  # off until the test reaches the call it wants to fail

    def _maybe_raise(self, name):
        self.calls.append(name)
        if self.armed and name in self.raising:
            raise RuntimeError(f"{name} driver offline")

    def show_good(self):
        self._maybe_raise("show_good")

    def show_bad(self):
        self._maybe_raise("show_bad")

    def alarm_off(self):
        self._maybe_raise("alarm_off")

    def show_review(self):
        self._maybe_raise("show_review")

    def show_fault(self):
        self._maybe_raise("show_fault")


class OutputFailureTests(_Base):
    def _harness(self, fuse_verdict, raising, on_complete=None):
        h = self.harness(fuse_verdict=fuse_verdict, on_complete=on_complete)
        h.outputs = RaisingOutputs(raising)
        h.session._outputs = h.outputs
        return h

    def _assert_saved_despite_output_failure(self, h, light, final):
        s = h.session
        self.assertEqual(s.state, State.IDLE)
        self.assertEqual(len(h.store.saved), 1)
        self.assertEqual(h.store.saved[0].final_verdict, final)
        self.assertEqual(s.counters["total"], 1)
        self.assertNotIn("show_fault", h.outputs.calls)
        self.assertNotIn("fault", h.store.kinds())
        self.assertIsNone(s.snapshot()["fault"])
        failed = [e for e in h.store.events if e[0] == "outputs_failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn(light, failed[0][1]["error"])
        self.assertIn("driver offline", failed[0][1]["error"])
        self.assertEqual(failed[0][1]["part_id"], h.store.saved[0].part_id)

    def test_machine_good_light_failure_keeps_part_saved_and_idle(self):
        h = self._harness(Verdict.GOOD, ("show_good",))
        self.assertTrue(h.press_at(0.0))
        h.outputs.armed = True
        self.assertTrue(h.press_at(1.0))  # must not raise
        self._assert_saved_despite_output_failure(h, "show_good", Verdict.GOOD)
        self.assertEqual(h.session.counters, {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(len(h.completed), 1)
        self.assertIs(h.completed[0], h.store.saved[0])

    def test_machine_good_light_failure_still_runs_on_complete(self):
        seen = []
        h = self._harness(Verdict.GOOD, ("show_good",), on_complete=seen.append)
        h.outputs.armed = True
        h.complete_part(0.0)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], h.store.saved[0])
        self.assertNotIn("on_complete_failed", h.store.kinds())

    def test_bad_ack_light_failure_keeps_part_saved_and_idle(self):
        h = self._harness(Verdict.BAD, ("show_bad",))
        h.complete_part(0.0)
        h.outputs.armed = True
        h.session.ack_alarm()  # must not raise
        self._assert_saved_despite_output_failure(h, "show_bad", Verdict.BAD)
        self.assertEqual(h.store.saved[0].disposition, "quarantine")
        self.assertEqual(h.session.counters, {"total": 1, "good": 0, "bad": 1})
        # the remaining light is still attempted after show_bad raised
        self.assertIn("alarm_off", h.outputs.calls[-2:])
        self.assertEqual(len(h.completed), 1)

    def test_override_to_good_light_failure_keeps_part_saved_and_idle(self):
        h = self._harness(Verdict.BAD, ("alarm_off",))
        h.complete_part(0.0)
        h.outputs.armed = True
        h.session.override("sup-1", Verdict.GOOD, "re-measured")  # must not raise
        self._assert_saved_despite_output_failure(h, "alarm_off", Verdict.GOOD)
        self.assertEqual(h.store.saved[0].disposition, "released")
        self.assertEqual(h.session.counters, {"total": 1, "good": 1, "bad": 0})
        # show_good is still attempted after alarm_off raised
        self.assertIn("show_good", h.outputs.calls)
        self.assertEqual(len(h.completed), 1)

    def test_override_to_bad_light_failure_keeps_part_saved_and_idle(self):
        h = self._harness(Verdict.BAD, ("show_bad",))
        h.complete_part(0.0)
        h.outputs.armed = True
        h.session.override("sup-1", "BAD", "confirmed defect")  # must not raise
        self._assert_saved_despite_output_failure(h, "show_bad", Verdict.BAD)
        self.assertEqual(h.session.counters, {"total": 1, "good": 0, "bad": 1})
        self.assertIn("alarm_off", h.outputs.calls)
        self.assertEqual(len(h.completed), 1)

    def test_no_outputs_failed_event_when_lights_succeed(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.complete_part(0.0)
        self.assertNotIn("outputs_failed", h.store.kinds())


class AlarmAndFaultLightFailureTests(_Base):
    """show_bad / show_review / show_fault run after their state is set; a raise must not wedge the station."""

    def _harness(self, fuse_verdict, raising):
        h = self.harness(fuse_verdict=fuse_verdict)
        h.outputs = RaisingOutputs(raising)
        h.session._outputs = h.outputs
        return h

    def _outputs_failed(self, h, light):
        failed = [e for e in h.store.events if e[0] == "outputs_failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn(light, failed[0][1]["error"])
        self.assertIn("driver offline", failed[0][1]["error"])
        return failed[0]

    def test_a_show_bad_raises_on_fuse_step_then_ack_saves_once(self):
        h = self._harness(Verdict.BAD, ("show_bad",))
        h.outputs.armed = True  # fails only during the BAD fuse step
        self.assertTrue(h.press_at(0.0))
        self.assertTrue(h.press_at(1.0))  # must not raise out of press()
        s = h.session
        self.assertEqual(s.state, State.VERDICT)
        self.assertEqual(s.snapshot()["verdict"], "BAD")
        self.assertNotIn("fault", h.store.kinds())
        self.assertNotIn("show_fault", h.outputs.calls)
        event = self._outputs_failed(h, "show_bad")
        self.assertEqual(event[1]["part_id"], "part-1")
        self.assertEqual(h.store.saved, [])  # not saved until the ack
        self.assertEqual(s.counters, {"total": 0, "good": 0, "bad": 0})

        h.outputs.armed = False  # the driver is back for the operator's ack
        s.ack_alarm()
        self.assertEqual(s.state, State.IDLE)
        self.assertEqual(len(h.store.saved), 1)
        self.assertEqual(h.store.saved[0].final_verdict, Verdict.BAD)
        self.assertEqual(h.store.saved[0].disposition, "quarantine")
        self.assertEqual(s.counters, {"total": 1, "good": 0, "bad": 1})
        self.assertEqual(len(h.completed), 1)
        self.assertEqual(len([e for e in h.store.events if e[0] == "outputs_failed"]), 1)

    def test_b_show_review_raises_then_resolve_review_saves_once(self):
        h = self._harness(Verdict.REVIEW, ("show_review",))
        h.outputs.armed = True
        self.assertTrue(h.press_at(0.0))
        self.assertTrue(h.press_at(1.0))
        s = h.session
        self.assertEqual(s.state, State.SUPERVISOR_REVIEW)
        self.assertEqual(s.snapshot()["verdict"], "REVIEW")
        self.assertNotIn("fault", h.store.kinds())
        self._outputs_failed(h, "show_review")
        self.assertEqual(h.store.saved, [])

        h.outputs.armed = False
        s.resolve_review("sup-2", Verdict.GOOD, "clean on re-look")
        self.assertEqual(s.state, State.IDLE)
        self.assertEqual(h.outputs.calls[-1], "show_good")
        self.assertEqual(len(h.store.saved), 1)
        self.assertEqual(h.store.saved[0].final_verdict, Verdict.GOOD)
        self.assertEqual(s.counters, {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(len(h.completed), 1)

    def test_c_show_fault_raises_during_camera_fault_and_clear_fault_works(self):
        h = self._harness(Verdict.GOOD, ("show_fault",))
        h.cameras.script = [CameraError("C2 timeout")]
        h.outputs.armed = True
        self.assertTrue(h.press_at(0.0))  # must not raise out of press()
        s = h.session
        self.assertEqual(s.state, State.FAULT)
        self.assertIn("C2 timeout", s.snapshot()["fault"])
        self.assertIn("fault", h.store.kinds())
        self._outputs_failed(h, "show_fault")
        self.assertEqual(h.store.saved, [])

        h.outputs.armed = False
        s.clear_fault()
        self.assertEqual(s.state, State.IDLE)
        self.assertIsNone(s.snapshot()["fault"])
        self.assertEqual(h.outputs.calls[-1], "clear")

    def test_d_station_is_not_wedged_after_a_bad_part_with_failed_alarm(self):
        h = self._harness(Verdict.BAD, ("show_bad",))
        h.outputs.armed = True
        h.complete_part(0.0)
        h.outputs.armed = False
        h.session.ack_alarm()
        self.assertEqual(h.session.state, State.IDLE)

        self.assertTrue(h.press_at(10.0))  # next part starts normally
        self.assertEqual(h.session.state, State.WAIT_FLIP)
        self.assertEqual(h.session.snapshot()["part_id"], "part-2")
        self.assertTrue(h.press_at(11.0))
        self.assertEqual(h.session.state, State.VERDICT)
        h.session.ack_alarm()
        self.assertEqual([r.part_id for r in h.store.saved], ["part-1", "part-2"])
        self.assertEqual(h.session.counters, {"total": 2, "good": 0, "bad": 2})


class FaultClearTests(_Base):
    def test_fault_cleared_event_is_logged(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.cameras.script = [CameraError("C1 timeout")]
        h.press_at(0.0)
        h.session.clear_fault()
        self.assertIn("fault_cleared", h.store.kinds())

    def test_clear_fault_to_wait_flip_restarts_flip_timeout(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.cameras.script = [None, CameraError("C4 dropped")]
        h.press_at(0.0)
        h.press_at(1.0)  # bottom capture fails
        self.assertEqual(h.session.state, State.FAULT)
        h.clock.t = 100.0
        h.session.clear_fault()
        self.assertEqual(h.session.state, State.WAIT_FLIP)
        h.clock.t = 150.0  # 50 s after the clear: within the 60 s flip timeout
        self.assertFalse(h.session.tick())
        self.assertEqual(h.session.state, State.WAIT_FLIP)
        h.clock.t = 160.5  # more than 60 s after the clear
        self.assertTrue(h.session.tick())
        self.assertEqual(h.session.state, State.IDLE)


class InitialCounterTests(_Base):
    def test_initial_counters_seed_the_totals(self):
        h = self.harness(fuse_verdict=Verdict.GOOD,
                         initial_counters={"total": 5, "good": 3, "bad": 2})
        self.assertEqual(h.session.counters, {"total": 5, "good": 3, "bad": 2})
        h.complete_part(0.0)
        self.assertEqual(h.session.counters, {"total": 6, "good": 4, "bad": 2})

    def test_default_counters_are_zero(self):
        h = self.harness()
        self.assertEqual(h.session.counters, {"total": 0, "good": 0, "bad": 0})


def _events(h, kind):
    return [e[1] for e in h.store.events if e[0] == kind]


class PartialStationTests(_Base):
    """Proof-of-concept station with fewer than 4 cameras (1 camera here)."""

    def partial(self, fuse_verdict=Verdict.GOOD, **kw):
        return self.harness(fuse_verdict=fuse_verdict, camera_ids=("C1",),
                            fuse=RecordingFuse(fuse_verdict), **kw)

    def test_one_camera_good_part_completes_with_two_images(self):
        h = self.partial(Verdict.GOOD)
        s = h.session
        self.assertTrue(h.press_at(0.0))
        self.assertEqual(s.state, State.WAIT_FLIP)
        self.assertEqual(len(s.snapshot()["results"]), 1)
        self.assertTrue(h.press_at(1.0))

        self.assertEqual(s.state, State.IDLE)
        self.assertEqual(h.outputs.calls[-1], "show_good")
        self.assertEqual(len(h.store.saved), 1)
        self.assertEqual(s.counters, {"total": 1, "good": 1, "bad": 0})
        self.assertEqual(len(h.completed), 1)
        rec = h.store.saved[0]
        self.assertEqual(len(rec.frames), 2)
        self.assertEqual(len(rec.results), 2)
        self.assertEqual([f.side for f in rec.frames], [Side.TOP, Side.BOTTOM])
        self.assertEqual([r.camera_id for r in rec.results], ["C1", "C1"])
        self.assertEqual(rec.final_verdict, Verdict.GOOD)
        self.assertEqual(h.cameras.sides, [Side.TOP, Side.BOTTOM])
        self.assertEqual(h.model.calls, 2)

        snap = s.snapshot()
        self.assertIs(snap["partial_station"], True)
        self.assertEqual(snap["camera_ids"], ["C1"])

    def test_fuse_receives_expected_two_for_one_camera(self):
        h = self.partial(Verdict.GOOD)
        h.complete_part(0.0)
        self.assertEqual(h.fuse.kwargs, [{"expected": 2}])
        self.assertEqual(len(h.fuse.calls[0]), 2)

    def test_one_camera_bad_part_needs_ack_alarm(self):
        h = self.partial(Verdict.BAD)
        h.complete_part(0.0)
        self.assertEqual(h.session.state, State.VERDICT)
        self.assertEqual(h.outputs.calls[-1], "show_bad")
        self.assertEqual(h.store.saved, [])
        self.assertEqual(h.session.counters, {"total": 0, "good": 0, "bad": 0})
        h.session.ack_alarm()
        self.assertEqual(h.session.state, State.IDLE)
        self.assertEqual(len(h.store.saved), 1)
        self.assertEqual(h.store.saved[0].final_verdict, Verdict.BAD)
        self.assertEqual(h.session.counters, {"total": 1, "good": 0, "bad": 1})

    def test_one_camera_review_never_lights_green_before_resolution(self):
        h = self.partial(Verdict.REVIEW)
        h.complete_part(0.0)
        self.assertEqual(h.session.state, State.SUPERVISOR_REVIEW)
        self.assertEqual(h.outputs.calls[-1], "show_review")
        self.assertNotIn("show_good", h.outputs.calls)
        self.assertEqual(h.store.saved, [])
        self.assertIsNone(h.session.snapshot()["final_verdict"])
        self.assertEqual(h.session.counters["total"], 0)

    def test_camera_returning_unconfigured_id_is_a_fault(self):
        h = self.partial(Verdict.GOOD)
        h.cameras.script = [[Frame(camera_id="C2", side=Side.TOP, seq=0,
                                   timestamp_utc=TS, data=b"TOP-C2")]]
        self.assertTrue(h.press_at(0.0))
        self.assertEqual(h.session.state, State.FAULT)
        fault = h.session.snapshot()["fault"]
        self.assertIn("C2", fault)
        self.assertIn("C1", fault)
        self.assertEqual(h.model.calls, 0)
        self.assertEqual(h.fuse.calls, [])
        self.assertEqual(h.store.saved, [])
        self.assertEqual(h.session.counters["total"], 0)

    def test_one_camera_short_capture_is_a_fault(self):
        h = self.partial(Verdict.GOOD)
        h.cameras.script = [[]]
        self.assertTrue(h.press_at(0.0))
        self.assertEqual(h.session.state, State.FAULT)
        self.assertIn("expected 1 frames", h.session.snapshot()["fault"])
        self.assertEqual(h.fuse.calls, [])

    def test_invalid_camera_ids_raise_value_error(self):
        for bad in ((), ("C1", "C1"), ("C1", "C9"), ("C9",)):
            with self.subTest(camera_ids=bad):
                with self.assertRaises(ValueError):
                    self.harness(camera_ids=bad)

    def test_partial_flag_is_in_snapshot_and_in_press_and_verdict_events(self):
        h = self.partial(Verdict.GOOD)
        h.complete_part(0.0)
        presses = _events(h, "press")
        verdicts = _events(h, "verdict")
        self.assertEqual(len(presses), 2)
        self.assertEqual(len(verdicts), 1)
        for payload in presses + verdicts:
            self.assertIs(payload["partial_station"], True)
            self.assertEqual(payload["camera_ids"], ["C1"])

    def test_is_partial_is_read_only_and_true_for_partial_station(self):
        h = self.partial()
        self.assertIs(h.session.is_partial, True)
        with self.assertRaises(AttributeError):
            h.session.is_partial = False
        self.assertIs(self.harness().session.is_partial, False)
        self.assertIs(self.harness(camera_ids=("C1", "C2", "C3")).session.is_partial, True)


class FullStationDefaultTests(_Base):
    def test_default_session_is_full_station(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        self.assertIs(h.session.is_partial, False)
        snap = h.session.snapshot()
        self.assertIs(snap["partial_station"], False)
        self.assertEqual(snap["camera_ids"], list(CAMERA_IDS))

    def test_default_session_calls_fuse_without_expected_keyword(self):
        fuse = RecordingFuse(Verdict.GOOD)
        h = self.harness(fuse=fuse)
        h.complete_part(0.0)
        self.assertEqual(fuse.kwargs, [{}])
        self.assertEqual(len(fuse.calls[0]), 8)
        self.assertEqual(h.store.saved[0].final_verdict, Verdict.GOOD)

    def test_default_press_and_verdict_events_carry_full_station_fields(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.complete_part(0.0)
        presses = _events(h, "press")
        verdicts = _events(h, "verdict")
        self.assertEqual(len(presses), 2)
        self.assertEqual(len(verdicts), 1)
        for payload in presses + verdicts:
            self.assertIs(payload["partial_station"], False)
            self.assertEqual(payload["camera_ids"], list(CAMERA_IDS))


class FlipGapTests(_Base):
    """A BOTTOM press waits min_flip_s after the TOP capture finished; debounce runs from the end of a press."""

    def test_default_min_flip_rejects_0_5s_and_accepts_2_1s_after_top(self):
        h = self.harness(fuse_verdict=Verdict.GOOD, min_flip_s=KEEP_DEFAULT)
        s = h.session
        self.assertTrue(h.press_at(0.0))
        self.assertFalse(h.press_at(0.5))
        self.assertEqual(s.state, State.WAIT_FLIP)
        self.assertEqual(h.cameras.sides, [Side.TOP])
        ignored = _events(h, "press_ignored")
        self.assertEqual(ignored[-1]["reason"], "flip_too_soon")
        self.assertAlmostEqual(ignored[-1]["seconds_remaining"], 1.5)
        self.assertTrue(h.press_at(2.1))
        self.assertEqual(s.state, State.IDLE)
        self.assertEqual(s.counters["total"], 1)

    def test_second_press_right_after_slow_top_capture_is_not_taken_as_bottom(self):
        # Regression for the real bug: the TOP capture takes 5 s, so a press that
        # arrives just after it finished used to pass a debounce measured from its start.
        h = self.harness(fuse_verdict=Verdict.GOOD, min_flip_s=KEEP_DEFAULT)
        h.cameras.on_capture = lambda side: h.clock.advance(5.0)
        s = h.session
        self.assertTrue(h.press_at(0.0))
        self.assertEqual(h.clock.t, 5.0)
        self.assertEqual(s.state, State.WAIT_FLIP)

        self.assertFalse(h.press_at(5.0))  # 0 s after the TOP finished: inside the debounce
        self.assertEqual(_events(h, "press_ignored")[-1]["reason"], "debounce")

        self.assertFalse(h.press_at(5.5))  # 0.5 s after: past the debounce, inside the 2.0 s gap
        ignored = _events(h, "press_ignored")[-1]
        self.assertEqual(ignored["reason"], "flip_too_soon")
        self.assertAlmostEqual(ignored["seconds_remaining"], 1.5)
        self.assertEqual(s.state, State.WAIT_FLIP)
        self.assertEqual(h.cameras.sides, [Side.TOP])
        self.assertEqual(s.counters["total"], 0)

        self.assertTrue(h.press_at(7.1))  # 2.1 s after the TOP finished
        self.assertEqual(s.state, State.IDLE)
        self.assertEqual(h.cameras.sides, [Side.TOP, Side.BOTTOM])
        self.assertEqual(s.counters["total"], 1)

    def test_min_flip_zero_accepts_a_quick_bottom(self):
        h = self.harness(fuse_verdict=Verdict.GOOD, min_flip_s=0.0)
        self.assertTrue(h.press_at(0.0))
        self.assertTrue(h.press_at(0.5))
        self.assertEqual(h.session.state, State.IDLE)
        self.assertEqual(h.session.counters["total"], 1)

    def test_debounce_runs_from_the_end_of_the_previous_press(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.cameras.on_capture = lambda side: h.clock.advance(1.0)  # each capture takes 1 s
        self.assertTrue(h.press_at(0.0))   # finishes at 1.0
        self.assertFalse(h.press_at(1.1))  # 0.1 s after the finish; from the start it would pass
        self.assertEqual(_events(h, "press_ignored")[-1]["reason"], "debounce")
        self.assertEqual(h.cameras.sides, [Side.TOP])
        self.assertTrue(h.press_at(1.5))   # 0.5 s after the finish
        self.assertEqual(h.session.state, State.IDLE)

    def test_negative_min_flip_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.harness(min_flip_s=-0.1)

    def test_clear_fault_back_to_wait_flip_needs_no_second_flip_gap(self):
        h = self.harness(fuse_verdict=Verdict.GOOD, min_flip_s=KEEP_DEFAULT)
        h.cameras.script = [None, CameraError("C4 dropped")]
        h.press_at(0.0)
        self.assertTrue(h.press_at(3.0))  # BOTTOM capture fails
        self.assertEqual(h.session.state, State.FAULT)
        h.session.clear_fault()
        self.assertEqual(h.session.state, State.WAIT_FLIP)
        h.clock.t = 3.5  # past the debounce measured from the failed press
        snap = h.session.snapshot()
        self.assertIs(snap["press_ready"], True)
        self.assertEqual(snap["flip_wait_s"], 0.0)
        self.assertEqual(snap["prompt"], "Step 2: flip the part to the BOTTOM side, then press Enter.")
        self.assertTrue(h.press_at(3.5))
        self.assertEqual(h.session.state, State.IDLE)


class PromptAndReadinessTests(_Base):
    def test_prompt_idle(self):
        snap = self.harness().session.snapshot()
        self.assertEqual(snap["prompt"], "Step 1: place the part TOP side up, then press Enter.")
        self.assertIs(snap["press_ready"], True)
        self.assertEqual(snap["flip_wait_s"], 0.0)

    def test_prompt_wait_flip_counts_down_then_is_ready(self):
        h = self.harness(min_flip_s=KEEP_DEFAULT)
        h.press_at(0.0)
        h.clock.t = 0.5
        snap = h.session.snapshot()
        self.assertEqual(snap["state"], "WAIT_FLIP")
        self.assertEqual(snap["prompt"], "Step 2: flip the part to the BOTTOM side. You can press Enter in 2 s.")
        self.assertIs(snap["press_ready"], False)
        self.assertEqual(snap["flip_wait_s"], 1.5)

        h.clock.t = 2.0
        snap = h.session.snapshot()
        self.assertEqual(snap["prompt"], "Step 2: flip the part to the BOTTOM side, then press Enter.")
        self.assertIs(snap["press_ready"], True)
        self.assertEqual(snap["flip_wait_s"], 0.0)

    def test_prompt_verdict(self):
        h = self.harness(fuse_verdict=Verdict.BAD)
        h.complete_part(0.0)
        snap = h.session.snapshot()
        self.assertEqual(snap["prompt"], "BAD: check the part, then acknowledge the alarm.")
        self.assertIs(snap["press_ready"], False)
        self.assertEqual(snap["flip_wait_s"], 0.0)

    def test_prompt_supervisor_review(self):
        h = self.harness(fuse_verdict=Verdict.REVIEW)
        h.complete_part(0.0)
        snap = h.session.snapshot()
        self.assertEqual(snap["prompt"], "REVIEW: a supervisor must decide GOOD or BAD.")
        self.assertIs(snap["press_ready"], False)

    def test_prompt_fault_shows_the_fault_text(self):
        h = self.harness(fuse_verdict=Verdict.GOOD)
        h.cameras.script = [CameraError("C2 timeout")]
        h.press_at(0.0)
        snap = h.session.snapshot()
        self.assertEqual(snap["state"], "FAULT")
        self.assertEqual(snap["prompt"], "Fault: C2 timeout")
        self.assertIs(snap["press_ready"], False)

    def test_prompt_is_empty_in_transitional_states(self):
        h = self.harness()
        seen = []
        h.cameras.on_capture = lambda side: seen.append(h.session.snapshot())
        h.press_at(0.0)
        self.assertEqual(seen[0]["state"], "CAPTURE_TOP")
        self.assertEqual(seen[0]["prompt"], "")
        self.assertIs(seen[0]["press_ready"], False)


class PreviewTests(_Base):
    def test_preview_passthrough_returns_bytes_from_cameras(self):
        cams = PreviewCameras()
        h = self.harness(cameras=cams)
        self.assertEqual(h.session.preview("C2"), b"preview-C2")
        self.assertEqual(cams.previews, ["C2"])

    def test_preview_is_none_without_a_preview_method(self):
        h = self.harness()  # FakeCameras has no preview()
        self.assertIsNone(h.session.preview("C1"))

    def test_preview_is_none_for_an_unknown_camera_id(self):
        cams = PreviewCameras()
        h = self.harness(cameras=cams)
        self.assertIsNone(h.session.preview("C9"))
        partial = self.harness(cameras=PreviewCameras(), camera_ids=("C1",))
        self.assertIsNone(partial.session.preview("C2"))
        self.assertEqual(partial.session.preview("C1"), b"preview-C1")
        self.assertEqual(cams.previews, [])

    def test_preview_is_none_when_the_camera_raises(self):
        for err in (StationError("C1 offline"), RuntimeError("boom")):
            with self.subTest(error=type(err).__name__):
                h = self.harness(cameras=PreviewCameras(error=err))
                self.assertIsNone(h.session.preview("C1"))

    def test_preview_changes_no_state_counters_outputs_or_events(self):
        cams = PreviewCameras()
        h = self.harness(cameras=cams)
        h.press_at(0.0)  # TOP captured, waiting for the flip
        s = h.session

        def fingerprint():
            return (s.state, s.snapshot(), dict(s.counters), list(h.outputs.calls),
                    list(h.store.events), list(cams.sides))

        before = fingerprint()
        s.preview("C1")
        s.preview("C9")
        cams.error = StationError("boom")
        self.assertIsNone(s.preview("C2"))
        cams.error = None
        self.assertEqual(s.preview("C3"), b"preview-C3")
        self.assertEqual(fingerprint(), before)


class PartSelectionTests(_Base):
    """Part number and model chosen at run time (multi-part station)."""

    def _part_model(self, version):
        m = FakeModel()
        m.version = version
        return m

    def test_constructor_without_part_or_model_cannot_inspect(self):
        h = self.harness(part_number="", with_model=False)
        s = h.session
        self.assertEqual(s.state, State.IDLE)
        self.assertEqual(s.part_number, "")
        self.assertEqual(s.model_version, "")
        self.assertIs(s.can_inspect, False)
        self.assertEqual(s.block_reason, "")
        self.assertIs(s.snapshot()["part_selected"], False)
        self.assertIs(s.snapshot()["press_ready"], False)
        self.assertEqual(s.snapshot()["prompt"], "Choose a part number to start.")

        self.assertFalse(h.press_at(0.0))
        self.assertEqual(h.cameras.sides, [])
        self.assertEqual(s.state, State.IDLE)
        ignored = _events(h, "press_ignored")
        self.assertEqual(ignored[-1]["reason"], "no_part_or_model")
        self.assertEqual(ignored[-1]["block_reason"], "")

    def test_constructor_with_part_and_model_can_inspect(self):
        s = self.harness().session
        self.assertIs(s.can_inspect, True)
        self.assertEqual(s.part_number, "PN-1")
        self.assertEqual(s.model_version, "model-test-1")

    def test_set_part_then_good_part_saves_record_with_new_part_and_model(self):
        h = self.harness(part_number="", with_model=False, fuse_verdict=Verdict.GOOD)
        m = self._part_model("model-PN-9-2")
        h.session.set_part("  PN-9 ", m, "model-PN-9-2")
        self.assertIs(h.session.can_inspect, True)
        self.assertEqual(h.session.part_number, "PN-9")
        self.assertEqual(h.session.model_version, "model-PN-9-2")
        self.assertEqual(_events(h, "part_selected"),
                         [{"part_number": "PN-9", "model_version": "model-PN-9-2"}])

        h.complete_part(0.0)
        self.assertEqual(len(h.store.saved), 1)
        rec = h.store.saved[0]
        self.assertEqual(rec.part_number, "PN-9")
        self.assertEqual(rec.model_version, "model-PN-9-2")
        self.assertEqual(rec.final_verdict, Verdict.GOOD)
        self.assertEqual(m.calls, 8)
        self.assertEqual(h.model.calls, 0)

    def test_set_part_refused_outside_idle(self):
        cases = {}
        wf = self.harness()
        wf.press_at(0.0)
        cases["WAIT_FLIP"] = wf
        bad = self.harness(fuse_verdict=Verdict.BAD)
        bad.complete_part(0.0)
        cases["VERDICT"] = bad
        rev = self.harness(fuse_verdict=Verdict.REVIEW)
        rev.complete_part(0.0)
        cases["SUPERVISOR_REVIEW"] = rev
        fault = self.harness()
        fault.cameras.script = [CameraError("C1 timeout")]
        fault.press_at(0.0)
        cases["FAULT"] = fault
        for name, h in cases.items():
            with self.subTest(state=name):
                self.assertEqual(h.session.state, State(name))
                with self.assertRaises(InvalidTransition):
                    h.session.set_part("PN-2", self._part_model("v2"), "v2")
                with self.assertRaises(InvalidTransition):
                    h.session.clear_part()
                with self.assertRaises(InvalidTransition):
                    h.session.set_blocked("reason")
                self.assertEqual(h.session.part_number, "PN-1")
                self.assertEqual(h.session.model_version, "model-test-1")

    def test_set_part_validation_raises_value_error(self):
        h = self.harness(part_number="", with_model=False)
        for bad_pn in ("", "   "):
            with self.subTest(part_number=bad_pn):
                with self.assertRaises(ValueError):
                    h.session.set_part(bad_pn, self._part_model("v"), "v")
        with self.assertRaises(ValueError):
            h.session.set_part("PN-2", None, "v")
        self.assertIs(h.session.can_inspect, False)
        self.assertEqual(h.session.part_number, "")
        self.assertNotIn("part_selected", h.store.kinds())

    def test_clear_part_resets_selection_and_blocks_press(self):
        h = self.harness()
        h.session.clear_part()
        s = h.session
        self.assertEqual(s.part_number, "")
        self.assertEqual(s.model_version, "")
        self.assertIs(s.snapshot()["part_selected"], False)
        self.assertIs(s.can_inspect, False)
        self.assertIn("part_cleared", h.store.kinds())
        self.assertEqual(s.snapshot()["prompt"], "Choose a part number to start.")
        self.assertFalse(h.press_at(0.0))
        self.assertEqual(h.cameras.sides, [])

    def test_set_blocked_shows_reason_and_blocks_press_until_cleared(self):
        h = self.harness()
        s = h.session
        s.set_blocked("No approved model for PN-1")
        snap = s.snapshot()
        self.assertEqual(snap["block_reason"], "No approved model for PN-1")
        self.assertEqual(snap["prompt"], "No approved model for PN-1")
        self.assertIs(snap["can_inspect"], False)
        self.assertIs(snap["press_ready"], False)
        self.assertEqual(s.block_reason, "No approved model for PN-1")

        self.assertFalse(h.press_at(0.0))
        self.assertEqual(h.cameras.sides, [])
        ignored = _events(h, "press_ignored")[-1]
        self.assertEqual(ignored["reason"], "no_part_or_model")
        self.assertEqual(ignored["block_reason"], "No approved model for PN-1")

        s.set_blocked("   ")  # blank clears, like None
        self.assertIs(s.can_inspect, True)
        self.assertEqual(s.block_reason, "")
        self.assertEqual(s.snapshot()["block_reason"], "")
        self.assertTrue(h.press_at(5.0))
        self.assertEqual(s.state, State.WAIT_FLIP)

    def test_set_part_clears_block_reason(self):
        h = self.harness()
        h.session.set_blocked("no model")
        h.session.set_part("PN-2", self._part_model("v2"), "v2")
        self.assertIs(h.session.can_inspect, True)
        self.assertEqual(h.session.block_reason, "")

    def test_two_parts_back_to_back_get_their_own_part_numbers(self):
        h = self.harness(part_number="", with_model=False, fuse_verdict=Verdict.GOOD)
        m1 = self._part_model("v-PN-A-1")
        h.session.set_part("PN-A", m1, "v-PN-A-1")
        h.complete_part(0.0)
        m2 = self._part_model("v-PN-B-1")
        h.session.set_part("PN-B", m2, "v-PN-B-1")
        h.complete_part(10.0)

        self.assertEqual([r.part_number for r in h.store.saved], ["PN-A", "PN-B"])
        self.assertEqual([r.model_version for r in h.store.saved], ["v-PN-A-1", "v-PN-B-1"])
        self.assertEqual([r.part_id for r in h.store.saved], ["part-1", "part-2"])
        self.assertEqual(h.session.counters, {"total": 2, "good": 2, "bad": 0})
        self.assertEqual((m1.calls, m2.calls), (8, 8))

    def test_snapshot_part_keys_reflect_selection(self):
        h = self.harness()
        snap = h.session.snapshot()
        self.assertEqual(snap["part_number"], "PN-1")
        self.assertIs(snap["part_selected"], True)
        self.assertEqual(snap["model_version"], "model-test-1")
        self.assertIs(snap["can_inspect"], True)
        self.assertEqual(snap["block_reason"], "")

        h.session.clear_part()
        snap = h.session.snapshot()
        self.assertEqual(snap["part_number"], "")
        self.assertIs(snap["part_selected"], False)
        self.assertEqual(snap["model_version"], "")
        self.assertIs(snap["can_inspect"], False)


if __name__ == "__main__":
    unittest.main()
