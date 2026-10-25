"""Camera sources for the inspection station.

FakeCameraSet is used by tests and the demo. OpenCvCameraSet drives the real
USB cameras through OpenCV. cv2 is imported lazily (or injected), so this
module imports cleanly on machines without OpenCV.

make_blur_check() is a pure-Python sharpness gate over a decoded grey image.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from station.contracts import CAMERA_IDS, CameraError, Frame, Side, camera_ids_for

NowUtc = Callable[[], str]
FrameBytes = Callable[[str, Side, int], bytes]

# Value OpenCV/V4L2 uses for "manual" exposure. Auto is 0.75 on the same scale.
AUTO_EXPOSURE_MANUAL = 0.25
AUTOFOCUS_OFF = 0
AUTO_WB_OFF = 0

# Live-preview defaults (engineering choices for a small, fast view, not
# inspection limits): preview frames are at most this many pixels wide and
# JPEG-encoded at this quality.
PREVIEW_WIDTH = 640
PREVIEW_JPEG_QUALITY = 70
# After a failed open, preview() waits this long before it tries to open the
# devices again (engineering choice: keeps the live view from re-opening a
# missing camera on every poll).
PREVIEW_REOPEN_COOLDOWN_S = 5.0
# After this many grab/retrieve failures in a row, preview() releases the devices
# so the cooldown reopen can recover an unplugged or sleeping camera (engineering
# choice, no restart needed).
PREVIEW_FAILS_BEFORE_REOPEN = 5
PREVIEW_STALL_MESSAGE = "camera stopped delivering pictures; reopening"
# Frames read and discarded right after a device opens (engineering choice: the
# first DirectShow frames are often dark or stale). 0 disables.
OPEN_WARMUP_FRAMES = 5
# Frames dropped per device before each capture's real grab, so a long gap in
# polling cannot return a stale buffered frame (engineering choice). 0 disables.
STALE_FLUSH_FRAMES = 2
# Seconds release() waits for a running capture or preview before it releases anyway.
RELEASE_LOCK_TIMEOUT_S = 2.0

_MANUAL_KEYS = ("focus", "exposure", "white_balance")


def utc_now_iso() -> str:
    """Current UTC time as ISO 8601 with a Z suffix, e.g. 2026-10-23T08:30:12Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_frame_bytes(camera_id: str, side: Side, n: int) -> bytes:
    return f"FAKE:{camera_id}:{side.value}:{n}".encode("ascii")


class FakeCameraSet:
    """Deterministic CameraSet for tests and the demo.

    capture(side) returns one Frame per camera in camera_ids order (default
    CAMERA_IDS). The data is b'FAKE:<camera>:<side>:<n>' where n is the
    zero-based number of the capture call for that side, unless
    frame_bytes(camera_id, side, n) supplies the bytes. Every call is counted
    in capture_counts, including calls that raise.
    """

    def __init__(
        self,
        fail_sides: Sequence[Side | str] = (),
        frame_bytes: Optional[FrameBytes] = None,
        clock: NowUtc = utc_now_iso,
        camera_ids: Sequence[str] = CAMERA_IDS,
    ) -> None:
        ids = tuple(camera_ids)
        if not ids:
            raise ValueError("camera_ids must not be empty")
        self._fail_sides = frozenset(Side(s) for s in fail_sides)
        self._frame_bytes = frame_bytes or _default_frame_bytes
        self._clock = clock
        self._camera_ids: tuple[str, ...] = ids
        self.capture_counts: dict[Side, int] = {Side.TOP: 0, Side.BOTTOM: 0}
        self._preview_calls = 0

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return self._camera_ids

    @property
    def is_open(self) -> bool:
        """Always True: the fake cameras need no device."""
        return True

    @property
    def last_open_error(self) -> Optional[str]:
        return None

    def status(self) -> dict[str, Any]:
        return {"open": True, "last_error": None, "consecutive_failures": 0}

    def capture(self, side: Side) -> Sequence[Frame]:
        side = Side(side)
        n = self.capture_counts[side]
        self.capture_counts[side] = n + 1
        if side in self._fail_sides:
            raise CameraError(f"fake camera set failed on the {side.value} side")
        return [
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

    def preview(self, camera_id: str) -> Optional[bytes]:
        """Fake live-preview bytes b'FAKEJPEG:<camera_id>:<n>' or None.

        n counts preview calls (zero-based) for known ids. Unknown ids return
        None and do not advance n. Capture state is never touched.
        """
        if camera_id not in self._camera_ids:
            return None
        n = self._preview_calls
        self._preview_calls = n + 1
        return f"FAKEJPEG:{camera_id}:{n}".encode("ascii")


def _import_cv2() -> Any:
    try:
        import cv2  # noqa: PLC0415 - imported lazily on purpose
    except ImportError as exc:
        raise CameraError(
            "OpenCV (cv2) is not installed; install opencv-python or inject cv2_module"
        ) from exc
    return cv2


class OpenCvCameraSet:
    """CameraSet backed by 1 to 4 OpenCV VideoCapture devices.

    device_indexes map in order to camera_ids_for(len(device_indexes)): one
    device is C1, two are C1 and C2, and so on up to C4. Devices are opened on
    the first capture(), by an explicit open(), or by the live view (preview()),
    whichever comes first. Each capture grabs all cameras first, then retrieves
    and PNG-encodes them, so the frames are close in time. last_skew_s holds
    max-min of the grab times for the last capture.

    last_open_error is the text of the last failed open (None when the last open
    worked). After a failed open, preview() waits reopen_cooldown_s before it
    tries again; capture() always tries at once.

    A run of fails_before_reopen grab/retrieve failures in preview() releases
    the devices (last_open_error says so), and the cooldown then reopens them.
    A failed capture() releases the devices too, so the next call reopens them.
    status() reports open, last_error and consecutive_failures.
    """

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
        self._indexes = indexes
        self._camera_ids: tuple[str, ...] = camera_ids_for(len(indexes))
        self._width = width
        self._height = height
        self._cv2 = cv2_module if cv2_module is not None else _import_cv2()
        self._now_utc = now_utc
        self._manual = manual
        self._caps: Optional[list[Any]] = None
        # Held by capture() for its whole duration. preview() only tries it
        # (non-blocking), so a preview never interleaves with a capture.
        self._lock = threading.Lock()
        self.last_skew_s: Optional[float] = None
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

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return self._camera_ids

    @property
    def is_open(self) -> bool:
        return self._caps is not None

    @property
    def last_open_error(self) -> Optional[str]:
        return self._last_open_error

    def status(self) -> dict[str, Any]:
        return {
            "open": self.is_open,
            "last_error": self._last_open_error,
            "consecutive_failures": self._consecutive_failures,
        }

    def open(self) -> None:
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

    def _open_one(self, cam: str, index: int, cap: Any) -> None:
        cv2 = self._cv2
        try:
            if not cap.isOpened():
                raise CameraError(f"{cam}: could not open device {index}")
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
        # Best effort: a short driver queue keeps old frames from piling up. Errors ignored.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._warm_up(cap)

    def _warm_up(self, cap: Any) -> None:
        """Read and discard the first frames after open. Failures are ignored."""
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
                # Do not retry a dead handle: the next capture() or preview() reopens.
                self._drop_devices_locked()
                raise

    def preview(
        self,
        camera_id: str,
        *,
        width: int = PREVIEW_WIDTH,
        quality: int = PREVIEW_JPEG_QUALITY,
    ) -> Optional[bytes]:
        """JPEG bytes of the latest frame of one camera, for the live view.

        Engineering defaults: width=PREVIEW_WIDTH (640) and quality=70; a frame
        wider than `width` is downscaled with INTER_AREA, aspect ratio kept,
        and never upscaled. If the devices are not open, preview opens them
        (inside the lock), unless the last open failed less than
        reopen_cooldown_s ago. Returns None, and never raises, when camera_id
        is unknown, the devices cannot be opened, a capture holds the lock
        (this call does not wait), or any grab, retrieve or encode problem
        occurs. Only this camera is grabbed.
        """
        if camera_id not in self._camera_ids:
            return None
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if self._caps is None:
                if not self._reopen_due():
                    return None
                try:
                    self.open()  # records last_open_error, or clears it on success
                except Exception:
                    return None
            caps = self._caps
            if caps is None:
                return None
            cap = caps[self._camera_ids.index(camera_id)]
            cv2 = self._cv2
            image = self._read_preview_locked(cap)
            if image is None:
                return None
            shape = getattr(image, "shape", None)
            if shape is not None and width > 0 and int(shape[1]) > width:
                new_h = max(1, round(int(shape[0]) * width / int(shape[1])))
                image = cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok or buf is None:
                return None
            return bytes(buf.tobytes())
        except Exception:
            return None
        finally:
            self._lock.release()

    def _read_preview_locked(self, cap: Any) -> Any:
        """One grab and retrieve for the live view, or None. Counts failures.

        After fails_before_reopen failures in a row the devices are released
        (see _stall_locked). A success resets the count.
        """
        image = None
        try:
            if cap.grab():
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    image = frame
        except Exception:
            image = None
        if image is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._fails_before_reopen:
                self._stall_locked()
            return None
        self._consecutive_failures = 0
        return image

    def _stall_locked(self) -> None:
        """Release every device; the cooldown then gates the reopen (lock held)."""
        self._drop_devices_locked()
        self._last_open_error = PREVIEW_STALL_MESSAGE
        self._failed_open_at = self._monotonic()

    def _drop_devices_locked(self) -> None:
        """Release every open device quietly and forget the handles (lock held)."""
        caps = self._caps or []
        self._caps = None
        self._consecutive_failures = 0
        _release_quietly(caps)

    def _reopen_due(self) -> bool:
        if self._failed_open_at is None:
            return True
        return self._monotonic() - self._failed_open_at >= self._reopen_cooldown_s

    def _capture_locked(self, side: Side) -> Sequence[Frame]:
        side = Side(side)
        if self._caps is None:
            self.open()
        caps = self._caps
        cv2 = self._cv2
        assert caps is not None

        # Stage 0: drop frames the driver may have buffered while nobody was polling.
        # Untimed, so it does not count towards last_skew_s. Failures are ignored here;
        # the real grab below reports them.
        for _ in range(self._stale_flush_frames):
            for cap in caps:
                try:
                    cap.grab()
                except Exception:
                    pass

        # Stage 1: grab all cameras before retrieving any, to keep them close in time.
        grab_times: list[float] = []
        stamps: list[str] = []
        for cam, cap in zip(self._camera_ids, caps):
            try:
                ok = cap.grab()
            except Exception as exc:
                raise CameraError(f"{cam}: grab failed: {exc}") from exc
            if not ok:
                raise CameraError(f"{cam}: grab failed")
            grab_times.append(self._monotonic())
            stamps.append(self._now_utc())
        self.last_skew_s = max(grab_times) - min(grab_times)

        # Stage 2: retrieve and encode.
        frames: list[Frame] = []
        for seq, (cam, cap, stamp) in enumerate(zip(self._camera_ids, caps, stamps)):
            try:
                ok, image = cap.retrieve()
            except Exception as exc:
                raise CameraError(f"{cam}: retrieve failed: {exc}") from exc
            if not ok or image is None:
                raise CameraError(f"{cam}: retrieve returned no frame")
            try:
                ok, buf = cv2.imencode(".png", image)
                data = buf.tobytes() if ok else None
            except Exception as exc:
                raise CameraError(f"{cam}: PNG encode failed: {exc}") from exc
            if not ok or data is None:
                raise CameraError(f"{cam}: PNG encode failed")
            frames.append(
                Frame(
                    camera_id=cam,
                    side=side,
                    seq=seq,
                    timestamp_utc=stamp,
                    data=bytes(data),
                    ext="png",
                )
            )

        if len(frames) != len(self._camera_ids):
            raise CameraError(
                f"expected {len(self._camera_ids)} frames, got {len(frames)}"
            )
        return frames

    def release(self) -> None:
        # Wait briefly for a running capture or preview. If it does not finish,
        # release anyway: the handles are dropped either way.
        acquired = self._lock.acquire(timeout=RELEASE_LOCK_TIMEOUT_S)
        errors: list[str] = []
        try:
            caps, self._caps = self._caps or [], None
            self._consecutive_failures = 0
            for cam, cap in zip(self._camera_ids, caps):
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
    """Population variance of the 4-neighbour Laplacian over interior pixels.

    Returns None when the image is smaller than 3x3 (no interior pixels).
    """
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
    """Build a sharpness gate. True when Laplacian variance >= min_variance.

    decode(frame) must return a 2D list of grey values. A frame smaller than
    3x3 is never ok.
    """

    def blur_ok(frame: Frame) -> bool:
        variance = _laplacian_variance(decode(frame))
        if variance is None:
            return False
        return variance >= min_variance

    return blur_ok
