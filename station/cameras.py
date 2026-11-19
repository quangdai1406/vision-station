from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from station.contracts import CAMERA_IDS, CameraError, Frame, Side, camera_ids_for

NowUtc = Callable[[], str]
FrameBytes = Callable[[str, Side, int], bytes]

CAPTURE_MODES = ("simultaneous", "sequential")

AUTO_EXPOSURE_MANUAL = 0.25
AUTOFOCUS_OFF = 0
AUTO_WB_OFF = 0

PREVIEW_WIDTH = 640
PREVIEW_JPEG_QUALITY = 70
PREVIEW_REOPEN_COOLDOWN_S = 5.0
PREVIEW_FAILS_BEFORE_REOPEN = 5
PREVIEW_STALL_MESSAGE = "camera stopped delivering pictures; reopening"
OPEN_WARMUP_FRAMES = 5
STALE_FLUSH_FRAMES = 2
RELEASE_LOCK_TIMEOUT_S = 2.0
PREVIEW_SEQ_MIN_INTERVAL_S = 1.0
PREVIEW_CACHE_MAX_AGE_S = 10.0

SESSION_MODES = ("default", "keep_open")

_MANUAL_KEYS = ("focus", "exposure", "white_balance")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_capture_mode(mode: str) -> str:
    if mode not in CAPTURE_MODES:
        raise ValueError(f"capture_mode must be one of {CAPTURE_MODES}, got {mode!r}")
    return mode


def _default_frame_bytes(camera_id: str, side: Side, n: int) -> bytes:
    return f"FAKE:{camera_id}:{side.value}:{n}".encode("ascii")


class FakeCameraSet:

    def __init__(
        self,
        fail_sides: Sequence[Side | str] = (),
        frame_bytes: Optional[FrameBytes] = None,
        clock: NowUtc = utc_now_iso,
        camera_ids: Sequence[str] = CAMERA_IDS,
        capture_mode: str = "simultaneous",
        sequential_keep_open: bool = False,
    ) -> None:
        ids = tuple(camera_ids)
        if not ids:
            raise ValueError("camera_ids must not be empty")
        self._fail_sides = frozenset(Side(s) for s in fail_sides)
        self._frame_bytes = frame_bytes or _default_frame_bytes
        self._clock = clock
        self._camera_ids: tuple[str, ...] = ids
        self._capture_mode = _check_capture_mode(capture_mode)
        self._sequential_keep_open = bool(sequential_keep_open)
        self.capture_counts: dict[Side, int] = {Side.TOP: 0, Side.BOTTOM: 0}
        self.capture_order: list[tuple[str, str]] = []
        self.last_capture_total_s: Optional[float] = None
        self._preview_calls = 0

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return self._camera_ids

    @property
    def is_open(self) -> bool:
        return True

    @property
    def last_open_error(self) -> Optional[str]:
        return None

    def status(self) -> dict[str, Any]:
        return {
            "open": True,
            "last_error": None,
            "consecutive_failures": 0,
            "capture_mode": self._capture_mode,
            "sequential_keep_open": self._sequential_keep_open,
        }

    def capture(self, side: Side) -> Sequence[Frame]:
        side = Side(side)
        n = self.capture_counts[side]
        self.capture_counts[side] = n + 1
        if side in self._fail_sides:
            raise CameraError(f"fake camera set failed on the {side.value} side")
        frames = [
            Frame(
                camera_id=cam,
                side=side,
                seq=seq,
                timestamp_utc=self._clock(),
                data=self._frame_bytes(cam, side, n),
                ext="png",
            )
            for seq, cam in enumerate(self._camera_ids)
        ]
        self.capture_order.extend((side.value, f.camera_id) for f in frames)
        return frames

    def preview(self, camera_id: str) -> Optional[bytes]:
        if camera_id not in self._camera_ids:
            return None
        n = self._preview_calls
        self._preview_calls = n + 1
        return f"FAKEJPEG:{camera_id}:{n}".encode("ascii")


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise CameraError(
            "OpenCV (cv2) is not installed; install opencv-python or inject cv2_module"
        ) from exc
    return cv2


class OpenCvCameraSet:

    def __init__(
        self,
        device_indexes: Sequence[int],
        *,
        width: int = 2560,
        height: int = 1440,
        cv2_module: Any = None,
        now_utc: NowUtc = utc_now_iso,
        manual_settings: Optional[Mapping[str, Optional[float]]] = None,
        reopen_cooldown_s: float = PREVIEW_REOPEN_COOLDOWN_S,
        monotonic: Callable[[], float] = time.monotonic,
        fails_before_reopen: int = PREVIEW_FAILS_BEFORE_REOPEN,
        open_warmup_frames: int = OPEN_WARMUP_FRAMES,
        stale_flush_frames: int = STALE_FLUSH_FRAMES,
        capture_mode: str = "simultaneous",
        sequential_keep_open: bool = False,
        preview_seq_min_interval_s: float = PREVIEW_SEQ_MIN_INTERVAL_S,
        preview_cache_max_age_s: float = PREVIEW_CACHE_MAX_AGE_S,
        session_mode: str = "default",
        mjpg: bool = False,
        threaded: bool = False,
    ) -> None:
        indexes = list(device_indexes)
        if not 1 <= len(indexes) <= len(CAMERA_IDS):
            raise CameraError(
                f"expected 1 to {len(CAMERA_IDS)} camera devices, got {len(indexes)}"
            )
        manual = dict(manual_settings or {})
        unknown = set(manual) - set(_MANUAL_KEYS)
        if unknown:
            raise ValueError(f"unknown manual setting(s): {sorted(unknown)}")
        self._capture_mode = _check_capture_mode(capture_mode)
        if session_mode not in SESSION_MODES:
            raise ValueError(f"session_mode must be one of {SESSION_MODES}, got {session_mode!r}")
        if session_mode == "keep_open" and self._capture_mode != "simultaneous":
            raise ValueError("session_mode 'keep_open' needs capture_mode 'simultaneous'")
        if (mjpg or threaded) and session_mode != "keep_open":
            raise ValueError("mjpg and threaded need session_mode 'keep_open'")
        self._session_keep_open = session_mode == "keep_open"
        self._mjpg = bool(mjpg)
        self._threaded = bool(threaded)
        self.last_failed_cameras: list[str] = []
        self.last_worker_times_s: Optional[list[float]] = None
        self._sequential_keep_open = bool(sequential_keep_open)
        self._indexes = indexes
        self._camera_ids: tuple[str, ...] = camera_ids_for(len(indexes))
        self._width = width
        self._height = height
        self._cv2 = cv2_module if cv2_module is not None else _import_cv2()
        self._now_utc = now_utc
        self._manual = manual
        self._caps: Optional[list[Any]] = None
        self._lock = threading.Lock()
        self.last_skew_s: Optional[float] = None
        self.last_capture_total_s: Optional[float] = None
        self.last_capture_times_s: Optional[list[float]] = None
        self._reopen_cooldown_s = float(reopen_cooldown_s)
        self._monotonic = monotonic
        self._last_open_error: Optional[str] = None
        self._failed_open_at: Optional[float] = None
        if fails_before_reopen < 1:
            raise ValueError("fails_before_reopen must be at least 1")
        if open_warmup_frames < 0 or stale_flush_frames < 0:
            raise ValueError("open_warmup_frames and stale_flush_frames must not be negative")
        self._fails_before_reopen = int(fails_before_reopen)
        self._open_warmup_frames = int(open_warmup_frames)
        self._stale_flush_frames = int(stale_flush_frames)
        self._consecutive_failures = 0
        self._seq_keep = self._capture_mode == "sequential" and self._sequential_keep_open
        self._seq_devices: dict[str, Any] = {}
        self._seq_failed_at: dict[str, float] = {}
        self._seq_fail_counts: dict[str, int] = {}
        self._preview_turn = 0
        self._preview_cache: dict[str, tuple[float, bytes]] = {}
        self._cache_lock = threading.Lock()
        if preview_seq_min_interval_s < 0 or preview_cache_max_age_s < 0:
            raise ValueError("preview_seq_min_interval_s and preview_cache_max_age_s must not be negative")
        self._preview_seq_min_interval_s = float(preview_seq_min_interval_s)
        self._preview_cache_max_age_s = float(preview_cache_max_age_s)
        self._seq_last_refresh_at: Optional[float] = None

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return self._camera_ids

    @property
    def is_open(self) -> bool:
        if self._capture_mode == "sequential" and not self._seq_keep:
            return True
        return self._caps is not None or bool(self._seq_devices)

    @property
    def last_open_error(self) -> Optional[str]:
        return self._last_open_error

    def status(self) -> dict[str, Any]:
        return {
            "open": self.is_open,
            "last_error": self._last_open_error,
            "consecutive_failures": self._consecutive_failures,
            "capture_mode": self._capture_mode,
            "sequential_keep_open": self._sequential_keep_open,
        }

    def open(self) -> None:
        if self._capture_mode == "sequential":
            if self._seq_keep:
                self._open_seq_devices()
            return
        if self._session_keep_open:
            self._open_keep_open_devices()
            return
        if self._caps is not None:
            return
        cv2 = self._cv2
        caps: list[Any] = []
        try:
            for cam, index in zip(self._camera_ids, self._indexes):
                try:
                    cap = cv2.VideoCapture(index)
                except Exception as exc:
                    raise CameraError(f"{cam}: could not open device {index}: {exc}") from exc
                caps.append(cap)
                self._open_one(cam, index, cap)
        except BaseException as exc:
            _release_quietly(caps)
            self._last_open_error = str(exc) or type(exc).__name__
            self._failed_open_at = self._monotonic()
            raise
        self._caps = caps
        self._last_open_error = None
        self._failed_open_at = None
        self._consecutive_failures = 0

    def _open_keep_open_devices(self) -> None:
        caps: list[Any] = list(self._caps) if self._caps is not None else [None] * len(self._indexes)
        errors: dict[str, str] = {}

        def work(i: int) -> None:
            cam = self._camera_ids[i]
            try:
                caps[i] = self._open_for_shot(cam, self._indexes[i])
            except BaseException as exc:
                errors[cam] = str(exc) or type(exc).__name__

        todo = [i for i, cap in enumerate(caps) if cap is None]
        threads = [threading.Thread(target=work, args=(i,)) for i in todo]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self._caps = caps
        self.last_failed_cameras = [c for c in self._camera_ids if c in errors]
        if errors:
            self._last_open_error = "; ".join(errors[c] for c in self.last_failed_cameras)
            self._failed_open_at = self._monotonic()
            if all(cap is None for cap in caps):
                self._caps = None
                raise CameraError(self._last_open_error)
        else:
            self._last_open_error = None
            self._failed_open_at = None
            self._consecutive_failures = 0

    def _open_seq_devices(self) -> None:
        for cam, index in zip(self._camera_ids, self._indexes):
            if cam in self._seq_devices:
                continue
            try:
                self._seq_devices[cam] = self._open_for_shot(cam, index)
            except BaseException:
                self._drop_devices_locked()
                raise
        self._last_open_error = None
        self._consecutive_failures = 0

    def _open_for_shot(self, cam: str, index: int, *, warm_up: bool = True) -> Any:
        try:
            try:
                cap = self._cv2.VideoCapture(index)
            except Exception as exc:
                raise CameraError(f"{cam}: could not open device {index}: {exc}") from exc
            try:
                self._open_one(cam, index, cap, warm_up=warm_up)
            except BaseException:
                _release_quietly([cap])
                raise
        except BaseException as exc:
            self._last_open_error = str(exc) or type(exc).__name__
            raise
        return cap

    def _open_one(self, cam: str, index: int, cap: Any, *, warm_up: bool = True) -> None:
        cv2 = self._cv2
        try:
            if not cap.isOpened():
                raise CameraError(f"{cam}: could not open device {index}")
            if self._mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, AUTOFOCUS_OFF)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_MANUAL)
            cap.set(cv2.CAP_PROP_AUTO_WB, AUTO_WB_OFF)
            if self._manual.get("focus") is not None:
                cap.set(cv2.CAP_PROP_FOCUS, self._manual["focus"])
            if self._manual.get("exposure") is not None:
                cap.set(cv2.CAP_PROP_EXPOSURE, self._manual["exposure"])
            if self._manual.get("white_balance") is not None:
                cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self._manual["white_balance"])
        except CameraError:
            raise
        except Exception as exc:
            raise CameraError(f"{cam}: failed to open or configure camera: {exc}") from exc
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if warm_up:
            self._warm_up(cap)

    def _warm_up(self, cap: Any) -> None:
        for _ in range(self._open_warmup_frames):
            try:
                cap.grab()
                cap.retrieve()
            except Exception:
                pass

    def capture(self, side: Side) -> Sequence[Frame]:
        with self._lock:
            try:
                return self._capture_locked(side)
            except CameraError:
                if not self._session_keep_open:
                    self._drop_devices_locked()
                raise

    def preview(
        self,
        camera_id: str,
        *,
        width: int = PREVIEW_WIDTH,
        quality: int = PREVIEW_JPEG_QUALITY,
    ) -> Optional[bytes]:
        if camera_id not in self._camera_ids:
            return None
        if self._capture_mode == "sequential":
            return self._preview_sequential(camera_id, width, quality)
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if self._caps is None:
                if not self._reopen_due():
                    return None
                try:
                    self.open()
                except Exception:
                    return None
            caps = self._caps
            if caps is None:
                return None
            cap = caps[self._camera_ids.index(camera_id)]
            if cap is None:
                return None
            image = self._read_preview_locked(cap)
            if image is None:
                return None
            return self._encode_preview(image, width, quality)
        except Exception:
            return None
        finally:
            self._lock.release()

    def _preview_sequential(self, camera_id: str, width: int, quality: int) -> Optional[bytes]:
        if not self._lock.acquire(blocking=False):
            return self._cached_preview(camera_id)
        try:
            return self._preview_sequential_locked(camera_id, width, quality)
        except Exception:
            return None
        finally:
            self._lock.release()

    def _cached_preview(self, camera_id: str) -> Optional[bytes]:
        with self._cache_lock:
            entry = self._preview_cache.get(camera_id)
            if entry is None:
                return None
            stamp, data = entry
            if self._monotonic() - stamp > self._preview_cache_max_age_s:
                del self._preview_cache[camera_id]
                return None
            return data

    def _cache_put(self, camera_id: str, data: bytes) -> None:
        with self._cache_lock:
            self._preview_cache[camera_id] = (self._monotonic(), data)

    def _encode_preview(self, image: Any, width: int, quality: int) -> Optional[bytes]:
        cv2 = self._cv2
        shape = getattr(image, "shape", None)
        if shape is not None and width > 0 and int(shape[1]) > width:
            new_h = max(1, round(int(shape[0]) * width / int(shape[1])))
            image = cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok or buf is None:
            return None
        return bytes(buf.tobytes())

    def _read_once(self, cap: Any) -> Any:
        image = None
        try:
            if cap.grab():
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    image = frame
        except Exception:
            image = None
        return image

    def _read_preview_locked(self, cap: Any) -> Any:
        image = self._read_once(cap)
        if image is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._fails_before_reopen:
                self._stall_locked()
            return None
        self._consecutive_failures = 0
        return image

    def _stall_locked(self) -> None:
        self._drop_devices_locked()
        self._last_open_error = PREVIEW_STALL_MESSAGE
        self._failed_open_at = self._monotonic()

    def _drop_devices_locked(self) -> None:
        caps = self._caps or []
        seq = list(self._seq_devices.values())
        self._caps = None
        self._seq_devices = {}
        self._consecutive_failures = 0
        _release_quietly(caps + seq)

    def _reopen_due(self) -> bool:
        if self._failed_open_at is None:
            return True
        return self._monotonic() - self._failed_open_at >= self._reopen_cooldown_s

    def _capture_locked(self, side: Side) -> Sequence[Frame]:
        if self._capture_mode == "sequential":
            return self._capture_sequential_locked(side)
        side = Side(side)
        if self._session_keep_open:
            return self._capture_keep_open_locked(side)
        if self._caps is None:
            self.open()
        caps = self._caps
        cv2 = self._cv2
        assert caps is not None

        for _ in range(self._stale_flush_frames):
            for cap in caps:
                try:
                    cap.grab()
                except Exception:
                    pass

        grab_starts: list[float] = []
        grab_times: list[float] = []
        stamps: list[str] = []
        for cam, cap in zip(self._camera_ids, caps):
            grab_starts.append(self._monotonic())
            try:
                ok = cap.grab()
            except Exception as exc:
                raise CameraError(f"{cam}: grab failed: {exc}") from exc
            if not ok:
                raise CameraError(f"{cam}: grab failed")
            grab_times.append(self._monotonic())
            stamps.append(self._now_utc())
        self.last_skew_s = max(grab_times) - min(grab_times)
        self.last_capture_total_s = grab_times[-1] - grab_starts[0]
        self.last_capture_times_s = [end - start for start, end in zip(grab_starts, grab_times)]

        frames: list[Frame] = []
        for seq, (cam, cap, stamp) in enumerate(zip(self._camera_ids, caps, stamps)):
            image = self._retrieve_image(cam, cap)
            data = self._encode_png(cam, image)
            frames.append(
                Frame(
                    camera_id=cam,
                    side=side,
                    seq=seq,
                    timestamp_utc=stamp,
                    data=data,
                    ext="png",
                )
            )

        if len(frames) != len(self._camera_ids):
            raise CameraError(
                f"expected {len(self._camera_ids)} frames, got {len(frames)}"
            )
        return frames

    def _capture_keep_open_locked(self, side: Side) -> Sequence[Frame]:
        if self._caps is None or any(cap is None for cap in self._caps):
            self._open_keep_open_devices()
        caps = self._caps
        assert caps is not None
        n = len(self._camera_ids)
        results: list[Any] = [None] * n
        errors: dict[str, str] = {}

        def work(i: int) -> None:
            cam, cap = self._camera_ids[i], caps[i]
            t0 = self._monotonic()
            try:
                if cap is None:
                    raise CameraError(f"{cam}: not open: {self._last_open_error}")
                for _ in range(self._stale_flush_frames):
                    try:
                        cap.grab()
                    except Exception:
                        pass
                start = self._monotonic()
                try:
                    ok = cap.grab()
                except Exception as exc:
                    raise CameraError(f"{cam}: grab failed: {exc}") from exc
                if not ok:
                    raise CameraError(f"{cam}: grab failed")
                end = self._monotonic()
                stamp = self._now_utc()
                data = self._encode_png(cam, self._retrieve_image(cam, cap))
                results[i] = (start, end, stamp, data, self._monotonic() - t0)
            except Exception as exc:
                errors[cam] = str(exc) or type(exc).__name__

        if self._threaded:
            threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:
            for i in range(n):
                work(i)

        failed = [c for c in self._camera_ids if c in errors]
        self.last_failed_cameras = failed
        if failed:
            for i, cam in enumerate(self._camera_ids):
                if cam in errors and caps[i] is not None:
                    _release_quietly([caps[i]])
                    caps[i] = None
            self._last_open_error = "; ".join(errors[c] for c in failed)
            raise CameraError("; ".join(errors[c] for c in failed))
        starts = [r[0] for r in results]
        ends = [r[1] for r in results]
        self.last_skew_s = max(ends) - min(ends)
        self.last_capture_total_s = max(ends) - min(starts)
        self.last_capture_times_s = [e - s for s, e in zip(starts, ends)]
        self.last_worker_times_s = [r[4] for r in results]
        self._last_open_error = None
        return [
            Frame(camera_id=cam, side=side, seq=seq, timestamp_utc=r[2], data=r[3], ext="png")
            for seq, (cam, r) in enumerate(zip(self._camera_ids, results))
        ]

    def _capture_sequential_locked(self, side: Side) -> Sequence[Frame]:
        side = Side(side)
        if self._seq_keep:
            self.open()
        frames: list[Frame] = []
        grab_starts: list[float] = []
        grab_ends: list[float] = []
        for seq, (cam, index) in enumerate(zip(self._camera_ids, self._indexes)):
            if self._seq_keep:
                frame, start, end = self._sequential_shot(cam, self._seq_devices[cam], side, seq)
            else:
                cap = self._open_for_shot(cam, index)
                try:
                    frame, start, end = self._sequential_shot(cam, cap, side, seq)
                finally:
                    _release_quietly([cap])
            frames.append(frame)
            grab_starts.append(start)
            grab_ends.append(end)

        self.last_skew_s = max(grab_ends) - min(grab_ends)
        self.last_capture_total_s = grab_ends[-1] - grab_starts[0]
        self.last_capture_times_s = [end - start for start, end in zip(grab_starts, grab_ends)]
        self._last_open_error = None
        return frames

    def _sequential_shot(
        self, cam: str, cap: Any, side: Side, seq: int
    ) -> tuple[Frame, float, float]:
        for _ in range(self._stale_flush_frames):
            try:
                cap.grab()
            except Exception:
                pass
        start = self._monotonic()
        try:
            ok = cap.grab()
        except Exception as exc:
            raise CameraError(f"{cam}: grab failed: {exc}") from exc
        if not ok:
            raise CameraError(f"{cam}: grab failed")
        end = self._monotonic()
        stamp = self._now_utc()
        image = self._retrieve_image(cam, cap)
        data = self._encode_png(cam, image)
        frame = Frame(camera_id=cam, side=side, seq=seq, timestamp_utc=stamp, data=data, ext="png")
        return frame, start, end

    def _retrieve_image(self, cam: str, cap: Any) -> Any:
        try:
            ok, image = cap.retrieve()
        except Exception as exc:
            raise CameraError(f"{cam}: retrieve failed: {exc}") from exc
        if not ok or image is None:
            raise CameraError(f"{cam}: retrieve returned no frame")
        return image

    def _encode_png(self, cam: str, image: Any) -> bytes:
        try:
            ok, buf = self._cv2.imencode(".png", image)
            data = buf.tobytes() if ok else None
        except Exception as exc:
            raise CameraError(f"{cam}: PNG encode failed: {exc}") from exc
        if not ok or data is None:
            raise CameraError(f"{cam}: PNG encode failed")
        return bytes(data)

    def _preview_sequential_locked(
        self, camera_id: str, width: int, quality: int
    ) -> Optional[bytes]:
        if not self._seq_keep and self._seq_last_refresh_at is not None:
            if self._monotonic() - self._seq_last_refresh_at < self._preview_seq_min_interval_s:
                return self._cached_preview(camera_id)
        count = len(self._camera_ids)
        turn = self._preview_turn % count
        self._preview_turn = (turn + 1) % count
        turn_cam = self._camera_ids[turn]
        data = self._preview_turn_locked(turn, turn_cam, width, quality)
        if data is not None:
            self._cache_put(turn_cam, data)
        return self._cached_preview(camera_id)

    def _preview_turn_locked(self, idx: int, cam: str, width: int, quality: int) -> Optional[bytes]:
        if self._seq_keep:
            cap = self._seq_devices.get(cam)
            if cap is None:
                if not self._seq_reopen_due(cam):
                    return None
                try:
                    cap = self._open_for_shot(cam, self._indexes[idx])
                except Exception:
                    self._seq_failed_at[cam] = self._monotonic()
                    return None
                self._seq_devices[cam] = cap
                self._seq_failed_at.pop(cam, None)
            image = self._read_once(cap)
            if image is None:
                fails = self._seq_fail_counts.get(cam, 0) + 1
                if fails >= self._fails_before_reopen:
                    self._drop_seq_device_locked(cam)
                else:
                    self._seq_fail_counts[cam] = fails
                return None
            self._seq_fail_counts.pop(cam, None)
            return self._encode_preview(image, width, quality)

        if not self._seq_reopen_due(cam):
            return None
        self._seq_last_refresh_at = self._monotonic()
        try:
            cap = self._open_for_shot(cam, self._indexes[idx], warm_up=False)
        except Exception:
            self._seq_failed_at[cam] = self._monotonic()
            return None
        self._seq_failed_at.pop(cam, None)
        try:
            image = self._read_once(cap)
        finally:
            _release_quietly([cap])
        if image is None:
            return None
        return self._encode_preview(image, width, quality)

    def _seq_reopen_due(self, cam: str) -> bool:
        failed = self._seq_failed_at.get(cam)
        return failed is None or self._monotonic() - failed >= self._reopen_cooldown_s

    def _drop_seq_device_locked(self, cam: str) -> None:
        cap = self._seq_devices.pop(cam, None)
        self._seq_fail_counts.pop(cam, None)
        if cap is not None:
            _release_quietly([cap])
        self._last_open_error = f"{cam}: {PREVIEW_STALL_MESSAGE}"
        self._seq_failed_at[cam] = self._monotonic()

    def release(self) -> None:
        acquired = self._lock.acquire(timeout=RELEASE_LOCK_TIMEOUT_S)
        errors: list[str] = []
        try:
            caps, self._caps = self._caps or [], None
            seq = list(self._seq_devices.items())
            self._seq_devices = {}
            self._consecutive_failures = 0
            for cam, cap in list(zip(self._camera_ids, caps)) + seq:
                if cap is None:
                    continue
                try:
                    cap.release()
                except Exception as exc:
                    errors.append(f"{cam}: {exc}")
        finally:
            if acquired:
                self._lock.release()
        if errors:
            raise CameraError("camera release failed: " + "; ".join(errors))


def _release_quietly(caps: list[Any]) -> None:
    for cap in caps:
        try:
            cap.release()
        except Exception:
            pass


def _laplacian_variance(grey: Sequence[Sequence[float]]) -> Optional[float]:
    height = len(grey)
    if height == 0:
        return None
    width = len(grey[0])
    if any(len(row) != width for row in grey):
        raise ValueError("decoded image rows have different lengths")
    if height < 3 or width < 3:
        return None
    values = []
    for y in range(1, height - 1):
        up, mid, down = grey[y - 1], grey[y], grey[y + 1]
        for x in range(1, width - 1):
            values.append(
                4 * mid[x] - up[x] - down[x] - mid[x - 1] - mid[x + 1]
            )
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def make_blur_check(
    decode: Callable[[Frame], Sequence[Sequence[float]]],
    min_variance: float,
) -> Callable[[Frame], bool]:

    def blur_ok(frame: Frame) -> bool:
        variance = _laplacian_variance(decode(frame))
        if variance is None:
            return False
        return variance >= min_variance

    return blur_ok
