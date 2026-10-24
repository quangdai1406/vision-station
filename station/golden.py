"""Compare-with-a-good-part model for the one-camera proof of concept.

DEMO HEURISTIC, NOT A VALIDATED QUALITY MODEL. GoldenBuilder learns a per-pixel
mean and standard deviation from good parts photographed on a fixed rig;
GoldenModel scores a new image by how far its pixels stray from that golden
image. The constants (std_floor, z_ref, max_width_px, the z_ref/2 region
cut-off and the 4-pixel minimum region) are engineering settings that make the
demo behave. They are NOT quality thresholds or acceptance limits. The real
t_low and t_img are chosen later from labelled scores and approved by Quality.

numpy is imported lazily inside functions, and cv2 only by cv2_decoder(), so
this module imports cleanly without either. Standard library otherwise.
"""
from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from station.contracts import Defect, Frame, ImageResult, Side, StationError

SCHEMA = "golden-v1"
DEFAULT_STD_FLOOR = 2.0  # gray levels; stops a perfectly still training set giving infinite scores
DEFAULT_Z_REF = 8.0  # z value that maps to score 1.0
DEFAULT_MAX_WIDTH_PX = 160  # working image is at most this wide (integer block averaging)
DEFECT_CLASS = "difference_from_golden"
_MIN_REGION_PX = 4  # downscaled pixels
_NPZ_DATE = (1980, 1, 1, 0, 0, 0)  # fixed zip timestamp so saves are byte-deterministic

Decode = Callable[[Frame], Any]
Key = tuple[str, str]


def _np() -> Any:
    try:
        import numpy  # noqa: PLC0415 - imported lazily on purpose
    except ImportError as exc:
        raise StationError("numpy is required for the golden-image model; install numpy") from exc
    return numpy


def _import_cv2() -> Any:
    try:
        import cv2  # noqa: PLC0415 - imported lazily on purpose
    except ImportError as exc:
        raise StationError(
            "OpenCV (cv2) is not installed; install opencv-python or inject cv2_module"
        ) from exc
    return cv2


def _key(camera_id: str, side: Any) -> Key:
    return (str(camera_id), Side(side).value)


def _key_text(key: Key) -> str:
    return f"{key[0]}/{key[1]}"


def _check_settings(max_width_px: int, std_floor: float, z_ref: float) -> None:
    if isinstance(max_width_px, bool) or not isinstance(max_width_px, int) or max_width_px < 1:
        raise ValueError(f"max_width_px must be a positive integer, got {max_width_px!r}")
    for name, value in (("std_floor", std_floor), ("z_ref", z_ref)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be a positive number, got {value!r}")


def _block_factor(width: int, max_width_px: int) -> int:
    return max(1, -(-width // max_width_px))  # ceiling division


def _block_mean(arr: Any, block: int) -> Any:
    np = _np()
    h, w = arr.shape
    bh, bw = h // block, w // block
    if bh < 1 or bw < 1:
        raise ValueError(f"image {h}x{w} is too small for block factor {block}")
    cropped = arr[: bh * block, : bw * block]
    return cropped.reshape(bh, block, bw, block).mean(axis=(1, 3)).astype(np.float64)


def _to_gray_array(value: Any, where: str) -> Any:
    np = _np()
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise StationError(
            f"{where}: decoded image must be a non-empty 2D grayscale array, got shape {arr.shape}"
        )
    return arr


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _largest_region(mask: Any) -> list[tuple[int, int]]:
    """Pixels (y, x) of the largest 4-connected True region, by pixel count."""
    np = _np()
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    best: list[tuple[int, int]] = []
    for y0, x0 in zip(*np.nonzero(mask)):
        y0, x0 = int(y0), int(x0)
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack = [(y0, x0)]
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) > len(best):
            best = pixels
    return best


@dataclass(frozen=True)
class LearnedKey:
    mean: Any  # downscaled per-pixel mean, float64 2D
    std: Any  # downscaled per-pixel sample standard deviation, float64 2D
    block_factor: int
    frames: int
    source_shape: tuple[int, int]  # decoded (full-size) image shape


class _Running:
    def __init__(self, block: int, source_shape: tuple[int, int]) -> None:
        self.block = block
        self.source_shape = source_shape
        self.n = 0
        self.mean: Any = None
        self.m2: Any = None


class GoldenBuilder:
    """Accumulates good frames per (camera_id, side) and builds a GoldenModel.

    Uses Welford's running mean and sum of squared deviations, so no frames are
    kept. Each frame is decoded, block-averaged to the key's working size and
    then folded in. All frames for one key must decode to the same shape.
    """

    def __init__(
        self,
        decode: Decode,
        *,
        max_width_px: int = DEFAULT_MAX_WIDTH_PX,
        std_floor: float = DEFAULT_STD_FLOOR,
        z_ref: float = DEFAULT_Z_REF,
    ) -> None:
        _check_settings(max_width_px, std_floor, z_ref)
        self._decode = decode
        self._max_width_px = max_width_px
        self._std_floor = float(std_floor)
        self._z_ref = float(z_ref)
        self._running: dict[Key, _Running] = {}

    def add(self, frame: Frame) -> None:
        key = _key(frame.camera_id, frame.side)
        where = _key_text(key)
        arr = _to_gray_array(self._decode(frame), where)
        run = self._running.get(key)
        if run is not None and arr.shape != run.source_shape:
            raise ValueError(f"{where}: frame shape {arr.shape} differs from {run.source_shape}")
        block = run.block if run is not None else _block_factor(arr.shape[1], self._max_width_px)
        small = _block_mean(arr, block)  # may raise before the key is registered
        if run is None:
            run = _Running(block, (int(arr.shape[0]), int(arr.shape[1])))
            self._running[key] = run
        np = _np()
        run.n += 1
        if run.mean is None:
            run.mean = small.copy()
            run.m2 = np.zeros_like(small)
        else:
            delta = small - run.mean
            run.mean = run.mean + delta / run.n
            run.m2 = run.m2 + delta * (small - run.mean)

    def build(self, version: str) -> "GoldenModel":
        if not self._running:
            raise ValueError("no golden frames added")
        np = _np()
        keys: dict[Key, LearnedKey] = {}
        for key in sorted(self._running):
            run = self._running[key]
            if run.n < 2:
                raise ValueError(
                    f"{_key_text(key)}: need at least 2 golden frames, got {run.n}"
                )
            std = np.sqrt(np.maximum(run.m2, 0.0) / (run.n - 1))
            keys[key] = LearnedKey(
                mean=run.mean.copy(),
                std=std,
                block_factor=run.block,
                frames=run.n,
                source_shape=run.source_shape,
            )
        return GoldenModel(
            version,
            keys,
            self._decode,
            std_floor=self._std_floor,
            z_ref=self._z_ref,
            max_width_px=self._max_width_px,
        )


class GoldenModel:
    """InferenceModel: score = clip(p99(z) / z_ref, 0, 1), z = |img - mean| / max(std, std_floor).

    Defect: the bounding box, in original pixels, of the largest region of
    pixels with z >= z_ref / 2 (at least 4 downscaled pixels). Its score is the
    region's peak z over z_ref, clipped to 0..1. No defect when the image
    score is 0 or no region is large enough.

    Note: the image score is p99-based, so it can differ from the defect peak, and an
    image can reach a high score with no box (no region of at least 4 downscaled
    pixels). Defects smaller than about 1 % of the working pixels barely move p99.
    All constants are demo heuristics, not quality thresholds.
    """

    def __init__(
        self,
        version: str,
        keys: Mapping[Any, LearnedKey],
        decode: Decode,
        *,
        std_floor: float = DEFAULT_STD_FLOOR,
        z_ref: float = DEFAULT_Z_REF,
        max_width_px: int = DEFAULT_MAX_WIDTH_PX,
    ) -> None:
        _check_settings(max_width_px, std_floor, z_ref)
        if not keys:
            raise ValueError("a golden model needs at least one learned camera/side")
        self.version = str(version)
        self.std_floor = float(std_floor)
        self.z_ref = float(z_ref)
        self.max_width_px = int(max_width_px)
        self._decode = decode
        self._keys: dict[Key, LearnedKey] = {_key(c, s): v for (c, s), v in keys.items()}

    def predict(self, frame: Frame) -> ImageResult:
        np = _np()
        key = _key(frame.camera_id, frame.side)
        where = _key_text(key)
        learned = self._keys.get(key)
        if learned is None:
            raise StationError(f"no golden image learned for {where}")
        arr = _to_gray_array(self._decode(frame), where)
        if arr.shape != learned.source_shape:
            raise StationError(
                f"{where}: image shape {arr.shape} does not match golden shape {learned.source_shape}"
            )
        small = _block_mean(arr, learned.block_factor)
        z = np.abs(small - learned.mean) / np.maximum(learned.std, self.std_floor)
        if not np.isfinite(z).all():
            raise StationError(f"{where}: non-finite pixel values in image")
        score = _clip01(float(np.percentile(z, 99)) / self.z_ref)
        defects: tuple[Defect, ...] = ()
        if score > 0.0:
            region = _largest_region(z >= self.z_ref / 2)
            if len(region) >= _MIN_REGION_PX:
                defects = (self._defect(z, region, learned.block_factor),)
        return ImageResult(
            camera_id=frame.camera_id,
            side=Side(frame.side),
            score=score,
            defects=defects,
        )

    def _defect(self, z: Any, region: list[tuple[int, int]], block: int) -> Defect:
        np = _np()
        ys = np.array([p[0] for p in region], dtype=np.int64)
        xs = np.array([p[1] for p in region], dtype=np.int64)
        peak = _clip01(float(z[ys, xs].max()) / self.z_ref)
        x0, y0 = int(xs.min()) * block, int(ys.min()) * block
        w = (int(xs.max()) - int(xs.min()) + 1) * block
        h = (int(ys.max()) - int(ys.min()) + 1) * block
        return Defect(defect_class=DEFECT_CLASS, score=peak, box=(x0, y0, w, h))

    def describe(self) -> dict[str, Any]:
        return {
            "model": DEFECT_CLASS,
            "heuristic": "demo only; not a validated quality model",
            "version": self.version,
            "std_floor": self.std_floor,
            "z_ref": self.z_ref,
            "max_width_px": self.max_width_px,
            "defect_region_cutoff_z": self.z_ref / 2,
            "min_region_px": _MIN_REGION_PX,
            "keys": {
                _key_text(key): {
                    "frames": learned.frames,
                    "block_factor": learned.block_factor,
                    "source_shape": list(learned.source_shape),
                    "working_shape": [int(v) for v in learned.mean.shape],
                }
                for key, learned in sorted(self._keys.items())
            },
        }

    def save(self, path: Any) -> None:
        np = _np()
        arrays: dict[str, Any] = {}
        entries = []
        for key in sorted(self._keys):
            learned = self._keys[key]
            tag = f"{key[0]}__{key[1]}"
            arrays[f"mean__{tag}"] = np.asarray(learned.mean, dtype=np.float64)
            arrays[f"std__{tag}"] = np.asarray(learned.std, dtype=np.float64)
            entries.append({
                "camera_id": key[0],
                "side": key[1],
                "block_factor": learned.block_factor,
                "frames": learned.frames,
                "source_shape": list(learned.source_shape),
            })
        meta = {
            "schema": SCHEMA,
            "version": self.version,
            "std_floor": self.std_floor,
            "z_ref": self.z_ref,
            "max_width_px": self.max_width_px,
            "keys": entries,
        }
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
        _write_npz(np, path, arrays)

    @classmethod
    def load(cls, path: Any, decode: Decode) -> "GoldenModel":
        np = _np()
        try:
            with np.load(str(path), allow_pickle=False) as data:
                meta = _read_meta(data)
                keys: dict[Key, LearnedKey] = {}
                for entry in meta["keys"]:
                    key = _key(entry["camera_id"], entry["side"])
                    block = int(entry["block_factor"])
                    frames = int(entry["frames"])
                    shape = tuple(int(v) for v in entry["source_shape"])
                    if block < 1 or frames < 2 or len(shape) != 2:
                        raise StationError(f"golden model entry for {_key_text(key)} is invalid")
                    tag = f"{key[0]}__{key[1]}"
                    mean = np.asarray(data[f"mean__{tag}"], dtype=np.float64)
                    std = np.asarray(data[f"std__{tag}"], dtype=np.float64)
                    expected = (shape[0] // block, shape[1] // block)
                    if mean.shape != expected or std.shape != expected:
                        raise StationError(f"golden arrays for {_key_text(key)} have the wrong shape")
                    keys[key] = LearnedKey(mean, std, block, frames, shape)  # type: ignore[arg-type]
                return cls(
                    str(meta["version"]),
                    keys,
                    decode,
                    std_floor=float(meta["std_floor"]),
                    z_ref=float(meta["z_ref"]),
                    max_width_px=int(meta["max_width_px"]),
                )
        except StationError:
            raise
        except Exception as exc:
            raise StationError(f"could not read golden model {path}: {exc}") from exc


def _read_meta(data: Any) -> dict[str, Any]:
    if "meta" not in data.files:
        raise StationError("golden model file has no metadata")
    try:
        meta = json.loads(str(data["meta"]))
    except ValueError as exc:
        raise StationError("golden model metadata is not valid JSON") from exc
    schema = meta.get("schema") if isinstance(meta, dict) else None
    if schema != SCHEMA:
        raise StationError(f"unknown golden model schema {schema!r}; expected {SCHEMA!r}")
    return meta


def _write_npz(np: Any, path: Any, arrays: Mapping[str, Any]) -> None:
    """Write an .npz with a fixed zip timestamp, so identical models give identical bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name in sorted(arrays):
            raw = io.BytesIO()
            np.save(raw, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_NPZ_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw.getvalue())
    try:
        with open(path, "wb") as fh:
            fh.write(buf.getvalue())
    except OSError as exc:
        raise StationError(f"could not write golden model {path}: {exc}") from exc


def cv2_decoder(cv2_module: Optional[Any] = None) -> Decode:
    """Decode function for real frames: cv2.imdecode, then grayscale, as float64 2D."""
    cv2 = cv2_module if cv2_module is not None else _import_cv2()

    def decode(frame: Frame) -> Any:
        np = _np()
        buf = np.frombuffer(frame.data, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise StationError(f"could not decode {frame.camera_id} {frame.side.value} frame")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.asarray(gray, dtype=np.float64)

    return decode
