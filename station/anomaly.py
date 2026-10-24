"""PatchCore-style anomaly detection trained on GOOD parts only.

PatchCoreBuilder collects patch feature vectors from good images for each
(camera_id, side), keeps a greedy k-center coreset as the memory bank, and
calibrates a norm_ref from held-out good images (or from training residuals
when no held-out images are given). A new image scores by its worst patch's
distance to the bank, divided by norm_ref and clipped to 0..1.

Incremental updates: training pictures are deleted after each weekly run, so
PatchCoreBuilder.from_model(previous_model, ...) starts from the saved memory
bank and the retained held-out features of the previous model. Only the NEW
pictures are added. The coreset is then re-run over (old bank rows + new rows).
Re-running the coreset on an existing coreset is a design choice that bounds
memory; the old rows are kept first and the new allowance is
coreset_ratio x new patches, capped by max_bank. Held-out features are kept
per key (newest max_holdout_images) so norm_ref is recalibrated from ALL
retained held-out images against the new bank.

Engineering settings, NOT quality thresholds: coreset_ratio (default 0.1),
max_bank (100000), seed (0), norm_factor (2.0; a held-out good image scores
about 1/norm_factor), max_holdout_images (20), the 0.5 region cut-off, the 10 %
residual sample, and the 128-dimension random projection used by the coreset.
The real t_low and t_img come from labelled scores via tools/thresholds.py and
are approved by Quality.

numpy is imported lazily inside functions. No torch here: the torch feature
extractor and torch backend live in station/features_torch.py. cv2 is imported
lazily by cv2_rgb_decoder. Standard library otherwise.
"""
from __future__ import annotations

import importlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol

from station.contracts import Defect, Frame, ImageResult, Side, StationError

SCHEMA = "patchcore-v2"
SCHEMA_V1 = "patchcore-v1"  # written before incremental updates: loads for scoring, cannot be updated
DEFECT_CLASS = "anomaly"
DEFAULT_CORESET_RATIO = 0.1
DEFAULT_MAX_BANK = 100_000
DEFAULT_SEED = 0
DEFAULT_NORM_FACTOR = 2.0
DEFAULT_MAX_HOLDOUT = 20  # held-out GOOD images kept per key for recalibration
PROJECTION_DIM = 128
REGION_CUTOFF_FRACTION = 0.5  # cells >= 0.5 * raw form the defect region
RESIDUAL_SAMPLE_FRACTION = 0.1  # share of training images scored for norm_ref without holdout
CHUNK_BYTES = 256 * 1024 * 1024  # largest distance block held at once
_MIN_NORM_REF = 1e-9  # divide-by-zero guard only, not a quality limit
_NPZ_DATE = (1980, 1, 1, 0, 0, 0)  # fixed zip timestamp so saves are byte-deterministic

Key = tuple[str, str]
Decode = Callable[[Frame], Any]


class FeatureExtractor(Protocol):
    """name: identifies the extractor in saved models. stride: input pixels per
    feature cell (informational; box mapping uses original/feature shape ratios).
    input_hw: the (h, w) the extractor resizes to. extract: one decoded image
    (2D grey or 3D RGB, RGB channel order) -> float32 (H', W', C)."""

    name: str
    stride: int
    input_hw: tuple[int, int]

    def extract(self, image: Any) -> Any: ...


class Backend(Protocol):
    def nn_distance(self, queries: Any, bank: Any) -> Any:
        """(N, C) queries, (M, C) bank -> (N,) float32 Euclidean distance to the nearest bank row."""


def _np() -> Any:
    try:
        import numpy  # noqa: PLC0415 - imported lazily on purpose
    except ImportError as exc:
        raise StationError("numpy is required for the PatchCore anomaly model; install numpy") from exc
    return numpy


def _key(camera_id: str, side: Any) -> Key:
    return (str(camera_id), Side(side).value)


def _key_text(key: Key) -> str:
    return f"{key[0]}/{key[1]}"


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return float(value)


def _input_hw_of(extractor: Any) -> Optional[tuple[int, int]]:
    hw = getattr(extractor, "input_hw", None)
    if hw is None:
        return None
    h, w = (int(v) for v in hw)
    return (h, w)


class NumpyBackend:
    """Nearest-neighbour distances in numpy. Distance blocks never exceed max_chunk_bytes.

    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab on data centred by the bank mean
    (float32 matmul), so the square root is taken once per query at the end.
    """

    def __init__(self, max_chunk_bytes: int = CHUNK_BYTES) -> None:
        self.max_chunk_bytes = _positive_int(max_chunk_bytes, "max_chunk_bytes")
        if self.max_chunk_bytes < 4:
            raise ValueError("max_chunk_bytes must hold at least one float32")

    def block_shape(self, n_queries: int, bank_rows: int) -> tuple[int, int]:
        """(query rows, bank rows) per block so that rows * cols * 4 <= max_chunk_bytes."""
        max_elems = self.max_chunk_bytes // 4
        rows = min(max(1, n_queries), max(1, max_elems // max(1, bank_rows)))
        cols = min(max(1, bank_rows), max(1, max_elems // rows))
        return rows, cols

    def nn_distance(self, queries: Any, bank: Any) -> Any:
        np = _np()
        q = np.asarray(queries, dtype=np.float32)
        b = np.asarray(bank, dtype=np.float32)
        if q.ndim != 2 or b.ndim != 2 or q.shape[1] != b.shape[1]:
            raise ValueError(f"queries {q.shape} and bank {b.shape} must be (N, C) and (M, C) with equal C")
        if b.shape[0] == 0:
            raise ValueError("bank is empty")
        n, m = q.shape[0], b.shape[0]
        out = np.empty(n, dtype=np.float32)
        if n == 0:
            return out
        mu = b.mean(axis=0, dtype=np.float64).astype(np.float32)
        bc = b - mu
        b_sq = np.einsum("ij,ij->i", bc, bc)
        rows, cols = self.block_shape(n, m)
        for rs in range(0, n, rows):
            qc = q[rs:rs + rows] - mu
            q_sq = np.einsum("ij,ij->i", qc, qc)
            best = np.full(qc.shape[0], np.inf, dtype=np.float32)
            for cs in range(0, m, cols):
                d2 = qc @ bc[cs:cs + cols].T
                d2 *= -2.0
                d2 += q_sq[:, None]
                d2 += b_sq[cs:cs + cols][None, :]
                np.minimum(best, d2.min(axis=1), out=best)
            out[rs:rs + qc.shape[0]] = np.sqrt(np.maximum(best, 0.0))
        return out


def _random_projection(x: Any, rng: Any) -> Any:
    np = _np()
    dim = x.shape[1]
    if dim <= PROJECTION_DIM:
        return x.copy()
    proj = rng.standard_normal((dim, PROJECTION_DIM)).astype(np.float32) / math.sqrt(PROJECTION_DIM)
    return (x @ proj).astype(np.float32)


def greedy_coreset(features: Any, n_select: int, seed: int = DEFAULT_SEED, *, progress: Optional[Callable[[int, int], None]] = None) -> Any:
    """PatchCore greedy k-center selection on a seeded random projection to 128 dims.

    Returns sorted unique row indices (int64). n_select >= N returns every index.
    The first pick is seeded; each next pick is the point farthest from the
    chosen set, so an outlier far from everything is always picked. Cost is
    O(n_select * N * 128), which is slow for large sets. progress(done, total)
    is optional.
    """
    np = _np()
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"features must be a non-empty (N, C) array, got shape {x.shape}")
    _positive_int(n_select, "n_select")
    n = x.shape[0]
    if n_select >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    proj = _random_projection(x, rng)
    start = int(rng.randint(n))
    chosen = np.empty(n_select, dtype=np.int64)
    chosen[0] = start
    min_d = _sq_dist(proj, proj[start])
    min_d[start] = -1.0  # -1 marks chosen rows; they never win the argmax again
    report_every = max(1, n_select // 100)
    for i in range(1, n_select):
        idx = int(np.argmax(min_d))
        chosen[i] = idx
        np.minimum(min_d, _sq_dist(proj, proj[idx]), out=min_d)
        min_d[idx] = -1.0
        if progress is not None and (i % report_every == 0 or i == n_select - 1):
            progress(i + 1, n_select)
    chosen.sort()
    return chosen


def _sq_dist(points: Any, point: Any) -> Any:
    np = _np()
    diff = points - point
    return np.einsum("ij,ij->i", diff, diff).astype(np.float32)


def _largest_region(mask: Any) -> tuple[Any, Any]:
    """(ys, xs) index arrays of the largest 4-connected True region (first one wins ties)."""
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
    ys = np.array([p[0] for p in best], dtype=np.int64)
    xs = np.array([p[1] for p in best], dtype=np.int64)
    return ys, xs


def _cell_box(ys: Any, xs: Any, grid: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Bounding box of feature cells, scaled to original pixels (x, y, w, h)."""
    h_feat, w_feat = grid
    h_img, w_img = shape
    cx0, cx1 = int(xs.min()), int(xs.max()) + 1
    cy0, cy1 = int(ys.min()), int(ys.max()) + 1
    x0 = cx0 * w_img // w_feat
    x1 = min(w_img, -(-(cx1 * w_img) // w_feat))
    y0 = cy0 * h_img // h_feat
    y1 = min(h_img, -(-(cy1 * h_img) // h_feat))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _image_shape(image: Any, where: str, error: type[Exception]) -> tuple[int, int]:
    np = _np()
    arr = np.asarray(image)
    if arr.ndim not in (2, 3) or arr.size == 0:
        raise error(f"{where}: decoded image must be 2D grey or 3D RGB, got shape {arr.shape}")
    return int(arr.shape[0]), int(arr.shape[1])


def _to_grid(value: Any, where: str, error: type[Exception]) -> Any:
    np = _np()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3 or 0 in arr.shape:
        raise error(f"{where}: extractor must return a non-empty (H, W, C) array, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise error(f"{where}: extractor returned non-finite features")
    return arr


@dataclass(frozen=True)
class PatchCoreKey:
    bank: Any  # float16 (M, C) memory bank
    norm_ref: float
    norm_ref_source: str  # "holdout" | "training_residual"
    train_images: int  # images in the build that produced this bank (new images for an update)
    holdout_images: int  # retained held-out images used for calibration
    calibration_raw: tuple[float, ...] = ()  # raw image scores used to set norm_ref
    images_seen: int = 0  # cumulative training images behind the bank (old + new)
    # Retained held-out GOOD features, oldest first: ((h*w, C) float16, (h, w)) per image.
    holdout_features: tuple[tuple[Any, tuple[int, int]], ...] = ()


class PatchCoreBuilder:
    """Accumulates good images per (camera_id, side) and builds a PatchCoreModel.

    add() takes a station Frame (decoded with `decode`) or a decoded ndarray with
    key=(camera_id, side). add_holdout() takes held-out GOOD images used only to
    calibrate norm_ref; their features are kept (float16, newest max_holdout_images
    per key) so a later update can recalibrate without the pictures.
    Every key needs at least 2 training images, unless it is an update of a key
    that already has a memory bank (then 1 new image is enough).
    """

    def __init__(
        self,
        extractor: Any,
        decode: Decode,
        *,
        coreset_ratio: float = DEFAULT_CORESET_RATIO,
        max_bank: int = DEFAULT_MAX_BANK,
        seed: int = DEFAULT_SEED,
        backend: Any = None,
        norm_factor: float = DEFAULT_NORM_FACTOR,
        max_holdout_images: int = DEFAULT_MAX_HOLDOUT,
    ) -> None:
        if extractor is None or decode is None:
            raise ValueError("extractor and decode are required")
        if not (isinstance(coreset_ratio, (int, float)) and 0.0 < coreset_ratio <= 1.0):
            raise ValueError(f"coreset_ratio must be in (0, 1], got {coreset_ratio!r}")
        self._extractor = extractor
        self._decode = decode
        self._ratio = float(coreset_ratio)
        self._max_bank = _positive_int(max_bank, "max_bank")
        self._seed = int(seed)
        self._backend = backend if backend is not None else NumpyBackend()
        self._norm_factor = _finite_positive(norm_factor, "norm_factor")
        self._max_holdout = _positive_int(max_holdout_images, "max_holdout_images")
        self._train: dict[Key, list[tuple[Any, tuple[int, int]]]] = {}
        self._holdout: dict[Key, list[tuple[Any, tuple[int, int]]]] = {}
        self._given_holdout: set[Key] = set()  # keys with held-out images added in this call
        self._dims: dict[Key, int] = {}
        self._pinned: set[Key] = set()  # keys whose dimension comes from a saved model
        self._base: Optional["PatchCoreModel"] = None

    @classmethod
    def from_model(
        cls,
        model: "PatchCoreModel",
        extractor: Any,
        decode: Decode,
        *,
        coreset_ratio: float = DEFAULT_CORESET_RATIO,
        max_bank: int = DEFAULT_MAX_BANK,
        seed: int = DEFAULT_SEED,
        backend: Any = None,
        norm_factor: float = DEFAULT_NORM_FACTOR,
        max_holdout_images: int = DEFAULT_MAX_HOLDOUT,
    ) -> "PatchCoreBuilder":
        """Builder pre-seeded with an earlier model's banks and held-out features.

        Checks extractor name, input_hw and feature dimension against the model.
        add() then appends NEW images; build(version) extends the banks.
        """
        if not isinstance(model, PatchCoreModel):
            raise StationError("from_model needs a PatchCoreModel")
        if model.legacy:
            raise StationError(
                f"model {model.version!r} was saved in the old {SCHEMA_V1} format, which kept no "
                "held-out features; it can score but cannot be updated. Retrain once from the full "
                "set of good images."
            )
        name = str(getattr(extractor, "name", ""))
        if name != model.extractor_name:
            raise StationError(
                f"extractor mismatch: model {model.version!r} was built with {model.extractor_name!r}, "
                f"got {name!r}"
            )
        hw = _input_hw_of(extractor)
        if hw != model.input_hw:
            raise StationError(
                f"input size mismatch: model {model.version!r} was built at {model.input_hw}, "
                f"the extractor gives {hw}"
            )
        builder = cls(
            extractor, decode, coreset_ratio=coreset_ratio, max_bank=max_bank, seed=seed,
            backend=backend, norm_factor=norm_factor, max_holdout_images=max_holdout_images,
        )
        builder._base = model
        for key, learned in model._keys.items():
            builder._dims[key] = int(learned.bank.shape[1])
            builder._pinned.add(key)
            if learned.holdout_features:
                builder._holdout[key] = list(learned.holdout_features)[-builder._max_holdout:]
        return builder

    def add(self, frame_or_array: Any, key: Optional[tuple[str, str]] = None) -> None:
        self._store(False, frame_or_array, key)

    def add_holdout(self, frame_or_array: Any, key: Optional[tuple[str, str]] = None) -> None:
        self._store(True, frame_or_array, key)

    def _store(self, holdout: bool, value: Any, key: Optional[tuple[str, str]]) -> None:
        np = _np()
        if isinstance(value, Frame):
            derived = _key(value.camera_id, value.side)
            if key is not None and _key(key[0], key[1]) != derived:
                raise ValueError(f"key {key!r} does not match frame {_key_text(derived)}")
            k = derived
            image = self._decode(value)
        else:
            if key is None:
                raise ValueError("an ndarray image needs key=(camera_id, side)")
            k = _key(key[0], key[1])
            image = value
        where = _key_text(k)
        _image_shape(image, where, ValueError)
        feats = _to_grid(self._extractor.extract(image), where, ValueError)
        dim = self._dims.setdefault(k, int(feats.shape[2]))
        if feats.shape[2] != dim:
            if k in self._pinned:
                raise StationError(
                    f"{where}: feature dimension {feats.shape[2]} differs from the model's {dim}; "
                    "the extractor does not match the saved model"
                )
            raise ValueError(f"{where}: feature dimension {feats.shape[2]} differs from {dim}")
        h, w, _ = feats.shape
        flat = feats.reshape(h * w, dim)
        grid = (int(h), int(w))
        if holdout:
            self._given_holdout.add(k)
            items = self._holdout.setdefault(k, [])
            items.append((flat.astype(np.float16), grid))
            if len(items) > self._max_holdout:
                del items[: len(items) - self._max_holdout]  # keep the newest
        else:
            self._train.setdefault(k, []).append((flat, grid))

    def build(self, version: str, *, progress: Optional[Callable[[int, int], None]] = None) -> "PatchCoreModel":
        base = self._base
        if not self._train and base is None:
            raise ValueError("no training images added")
        stray = sorted(self._given_holdout - set(self._train))
        if stray:
            raise ValueError(f"{_key_text(stray[0])}: held-out images given but no training images")
        base_keys = dict(base._keys) if base is not None else {}
        keys: dict[Key, PatchCoreKey] = {}
        for k in sorted(set(self._train) | set(base_keys)):
            if k not in self._train:
                keys[k] = base_keys[k]  # only in the old model: kept unchanged
                continue
            keys[k] = self._build_key(k, base_keys.get(k), progress)
        settings = {
            "norm_factor": self._norm_factor,
            "coreset_ratio": self._ratio,
            "max_bank": self._max_bank,
            "max_holdout_images": self._max_holdout,
            "seed": self._seed,
        }
        return PatchCoreModel(
            version,
            keys,
            self._extractor,
            self._decode,
            backend=self._backend,
            settings=settings,
            parent_version=base.version if base is not None else None,
            input_hw=_input_hw_of(self._extractor),
        )

    def _build_key(self, k: Key, old: Optional[PatchCoreKey], progress: Optional[Callable[[int, int], None]]) -> PatchCoreKey:
        np = _np()
        images = self._train[k]
        where = _key_text(k)
        if old is None and len(images) < 2:
            raise ValueError(f"{where}: need at least 2 training images, got {len(images)}")
        new_feats = [f for f, _ in images]
        if old is None:
            feats = np.concatenate(new_feats, axis=0)
            image_ids = np.concatenate([np.full(f.shape[0], i, dtype=np.int64) for i, f in enumerate(new_feats)])
            n_select = min(self._max_bank, max(1, int(round(self._ratio * feats.shape[0]))))
            images_seen = len(images)
        else:
            old_rows = np.asarray(old.bank, dtype=np.float32)  # the old bank rows are raw rows here
            n_new_rows = sum(f.shape[0] for f in new_feats)
            feats = np.concatenate([old_rows] + new_feats, axis=0)
            image_ids = np.concatenate(
                [np.full(old_rows.shape[0], -1, dtype=np.int64)]
                + [np.full(f.shape[0], i, dtype=np.int64) for i, f in enumerate(new_feats)]
            )
            n_select = min(self._max_bank, old_rows.shape[0] + max(1, int(round(self._ratio * n_new_rows))))
            images_seen = old.images_seen + len(images)
        chosen = greedy_coreset(feats, n_select, self._seed, progress=progress)
        bank = feats[chosen].astype(np.float16)
        bank32 = bank.astype(np.float32)  # what predict() will search
        holdout = self._holdout.get(k, [])
        if holdout:
            raw = [self._image_score(f.astype(np.float32), bank32) for f, _ in holdout]
            value, source = max(raw), "holdout"
        else:
            ids = image_ids[chosen]
            counts = np.bincount(ids[ids >= 0], minlength=len(images))
            order = np.lexsort((np.arange(len(images)), counts))  # fewest bank patches first
            n_res = max(1, int(math.ceil(RESIDUAL_SAMPLE_FRACTION * len(images))))
            raw = [self._image_score(images[int(i)][0], bank32) for i in order[:n_res]]
            value, source = float(np.percentile(raw, 99)), "training_residual"
        norm_ref = self._norm_factor * float(value)
        if not math.isfinite(norm_ref):
            raise ValueError(f"{where}: norm_ref is not finite; check the training images")
        norm_ref = max(norm_ref, _MIN_NORM_REF)
        return PatchCoreKey(
            bank=bank,
            norm_ref=float(norm_ref),
            norm_ref_source=source,
            train_images=len(images),
            holdout_images=len(holdout),
            calibration_raw=tuple(float(r) for r in raw),
            images_seen=images_seen,
            holdout_features=tuple(holdout),
        )

    def _image_score(self, feats: Any, bank32: Any) -> float:
        return float(self._backend.nn_distance(feats, bank32).max())


class PatchCoreModel:
    """InferenceModel: score = clip(max patch distance / norm_ref, 0, 1).

    Defect: the largest 4-connected region of anomaly-map cells >= 0.5 * raw,
    as a box in original pixels, with Defect('anomaly', region peak / norm_ref).
    No defect when the score is 0. Unknown camera/side -> StationError.
    """

    def __init__(
        self,
        version: str,
        keys: Mapping[Any, PatchCoreKey],
        extractor: Any,
        decode: Decode,
        *,
        backend: Any = None,
        settings: Optional[Mapping[str, Any]] = None,
        parent_version: Optional[str] = None,
        input_hw: Optional[tuple[int, int]] = None,
        legacy: bool = False,
    ) -> None:
        if not keys:
            raise ValueError("a PatchCore model needs at least one camera/side")
        self.version = str(version)
        self._keys: dict[Key, PatchCoreKey] = {_key(c, s): v for (c, s), v in keys.items()}
        self._extractor = extractor
        self._decode = decode
        self._backend = backend if backend is not None else NumpyBackend()
        self._settings = dict(settings or {})
        self._bank32: dict[Key, Any] = {}
        self._parent_version = None if parent_version is None else str(parent_version)
        self._extractor_name = str(getattr(extractor, "name", ""))
        self._input_hw = None if input_hw is None else (int(input_hw[0]), int(input_hw[1]))
        self._legacy = bool(legacy)

    @property
    def parent_version(self) -> Optional[str]:
        return self._parent_version

    @property
    def extractor_name(self) -> str:
        return self._extractor_name

    @property
    def input_hw(self) -> Optional[tuple[int, int]]:
        return self._input_hw

    @property
    def legacy(self) -> bool:
        """True for a file saved before incremental updates (scores only, cannot be updated)."""
        return self._legacy

    def keys(self) -> list[Key]:
        return sorted(self._keys)

    def bank(self, key: Any) -> Any:
        """float16 (M, C) memory bank for (camera_id, side). Not copied."""
        return self._learned(key[0], key[1])[1].bank

    def norm_ref(self, key: Any) -> float:
        return self._learned(key[0], key[1])[1].norm_ref

    def _learned(self, camera_id: str, side: Any) -> tuple[Key, PatchCoreKey]:
        key = _key(camera_id, side)
        learned = self._keys.get(key)
        if learned is None:
            raise StationError(f"no anomaly memory bank for {_key_text(key)}")
        return key, learned

    def _bank_f32(self, key: Key, learned: PatchCoreKey) -> Any:
        np = _np()
        if key not in self._bank32:
            self._bank32[key] = np.asarray(learned.bank, dtype=np.float32)
        return self._bank32[key]

    def _raw_map(self, image: Any, key: Key, learned: PatchCoreKey) -> tuple[Any, tuple[int, int], tuple[int, int]]:
        np = _np()
        where = _key_text(key)
        shape = _image_shape(image, where, StationError)
        feats = _to_grid(self._extractor.extract(image), where, StationError)
        h, w, c = feats.shape
        bank = self._bank_f32(key, learned)
        if c != bank.shape[1]:
            raise StationError(f"{where}: feature dimension {c} differs from bank dimension {bank.shape[1]}")
        dist = np.asarray(self._backend.nn_distance(feats.reshape(h * w, c), bank), dtype=np.float32)
        return dist.reshape(h, w), (int(h), int(w)), shape

    def predict(self, frame: Frame) -> ImageResult:
        key, learned = self._learned(frame.camera_id, frame.side)
        raw_map, grid, shape = self._raw_map(self._decode(frame), key, learned)
        raw = float(raw_map.max())
        score = _clip01(raw / learned.norm_ref)
        side = Side(frame.side)
        if score <= 0.0:
            return ImageResult(camera_id=frame.camera_id, side=side, score=0.0)
        mask = raw_map >= REGION_CUTOFF_FRACTION * raw
        ys, xs = _largest_region(mask)
        peak = _clip01(float(raw_map[ys, xs].max()) / learned.norm_ref)
        box = _cell_box(ys, xs, grid, shape)
        return ImageResult(
            camera_id=frame.camera_id,
            side=side,
            score=score,
            defects=(Defect(defect_class=DEFECT_CLASS, score=peak, box=box),),
        )

    def anomaly_map(self, frame: Frame) -> Any:
        """Normalised anomaly map (H', W') float32, 0..1, for heatmap overlays."""
        np = _np()
        key, learned = self._learned(frame.camera_id, frame.side)
        raw_map, _, _ = self._raw_map(self._decode(frame), key, learned)
        return np.clip(raw_map / learned.norm_ref, 0.0, 1.0).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        banks = {}
        for key, learned in sorted(self._keys.items()):
            banks[_key_text(key)] = {
                "rows": int(learned.bank.shape[0]),
                "dim": int(learned.bank.shape[1]),
                "norm_ref": learned.norm_ref,
                "norm_ref_source": learned.norm_ref_source,
                "train_images": learned.train_images,
                "images_seen": learned.images_seen,
                "holdout_images": learned.holdout_images,
            }
        return {
            "model": DEFECT_CLASS,
            "heuristic": "good-parts-only memory bank; not a validated quality model",
            "version": self.version,
            "parent_version": self._parent_version,
            "extractor": getattr(self._extractor, "name", None),
            "input_hw": None if self._input_hw is None else list(self._input_hw),
            "stride": getattr(self._extractor, "stride", None),
            "settings": dict(self._settings),
            "constants": {
                "projection_dim": PROJECTION_DIM,
                "region_cutoff_fraction": REGION_CUTOFF_FRACTION,
                "residual_sample_fraction": RESIDUAL_SAMPLE_FRACTION,
            },
            "banks": banks,
        }

    def save(self, path: Any) -> None:
        np = _np()
        arrays: dict[str, Any] = {}
        entries = []
        for key in sorted(self._keys):
            learned = self._keys[key]
            tag = f"{key[0]}__{key[1]}"
            arrays[f"bank__{tag}"] = np.asarray(learned.bank, dtype=np.float16)
            grids = []
            for i, (feats, grid) in enumerate(learned.holdout_features):
                arrays[f"hold__{tag}__{i:04d}"] = np.asarray(feats, dtype=np.float16)
                grids.append([int(grid[0]), int(grid[1])])
            entries.append({
                "camera_id": key[0],
                "side": key[1],
                "bank_rows": int(learned.bank.shape[0]),
                "feature_dim": int(learned.bank.shape[1]),
                "norm_ref": float(learned.norm_ref),
                "norm_ref_source": learned.norm_ref_source,
                "train_images": int(learned.train_images),
                "images_seen": int(learned.images_seen),
                "holdout_images": int(learned.holdout_images),
                "holdout_grids": grids,
                "calibration_raw": [float(r) for r in learned.calibration_raw],
            })
        meta = {
            "schema": SCHEMA,
            "version": self.version,
            "parent_version": self._parent_version,
            "input_hw": None if self._input_hw is None else list(self._input_hw),
            "extractor": {
                "name": str(getattr(self._extractor, "name", "")),
                "stride": int(getattr(self._extractor, "stride", 0) or 0),
            },
            "settings": self._settings,
            "keys": entries,
        }
        arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
        _write_npz(np, path, arrays)

    @classmethod
    def load(cls, path: Any, decode: Decode, extractor: Any, backend: Any = None) -> "PatchCoreModel":
        np = _np()
        if extractor is None:
            raise StationError("loading a PatchCore model needs the matching feature extractor")
        try:
            with np.load(str(path), allow_pickle=False) as data:
                meta = _read_meta(data)
                legacy = meta["schema"] == SCHEMA_V1
                saved_name = meta["extractor"]["name"]
                if getattr(extractor, "name", None) != saved_name:
                    raise StationError(
                        f"extractor mismatch: model was built with {saved_name!r}, "
                        f"got {getattr(extractor, 'name', None)!r}"
                    )
                keys: dict[Key, PatchCoreKey] = {}
                for entry in meta["keys"]:
                    key = _key(entry["camera_id"], entry["side"])
                    tag = f"{key[0]}__{key[1]}"
                    bank = np.asarray(data[f"bank__{tag}"], dtype=np.float16)
                    if bank.shape != (int(entry["bank_rows"]), int(entry["feature_dim"])):
                        raise StationError(f"bank for {_key_text(key)} has the wrong shape")
                    holdout: list[tuple[Any, tuple[int, int]]] = []
                    if legacy:
                        images_seen = int(entry["train_images"])
                    else:
                        images_seen = int(entry["images_seen"])
                        for i, grid in enumerate(entry["holdout_grids"]):
                            h, w = int(grid[0]), int(grid[1])
                            feats = np.asarray(data[f"hold__{tag}__{i:04d}"], dtype=np.float16)
                            if feats.shape != (h * w, int(entry["feature_dim"])):
                                raise StationError(f"held-out features for {_key_text(key)} have the wrong shape")
                            holdout.append((feats, (h, w)))
                    keys[key] = PatchCoreKey(
                        bank=bank,
                        norm_ref=float(entry["norm_ref"]),
                        norm_ref_source=str(entry["norm_ref_source"]),
                        train_images=int(entry["train_images"]),
                        holdout_images=int(entry["holdout_images"]) if legacy else len(holdout),
                        calibration_raw=tuple(float(r) for r in entry["calibration_raw"]),
                        images_seen=images_seen,
                        holdout_features=tuple(holdout),
                    )
                input_hw = None
                parent = None
                if not legacy:
                    raw_hw = meta.get("input_hw")
                    input_hw = None if raw_hw is None else (int(raw_hw[0]), int(raw_hw[1]))
                    parent = meta.get("parent_version")
                return cls(
                    str(meta["version"]),
                    keys,
                    extractor,
                    decode,
                    backend=backend,
                    settings=meta.get("settings", {}),
                    parent_version=parent,
                    input_hw=input_hw,
                    legacy=legacy,
                )
        except StationError:
            raise
        except Exception as exc:
            raise StationError(f"could not read PatchCore model {path}: {exc}") from exc


def _read_meta(data: Any) -> dict[str, Any]:
    if "meta" not in data.files:
        raise StationError("PatchCore model file has no metadata")
    try:
        meta = json.loads(str(data["meta"]))
    except ValueError as exc:
        raise StationError("PatchCore model metadata is not valid JSON") from exc
    schema = meta.get("schema") if isinstance(meta, dict) else None
    if schema not in (SCHEMA, SCHEMA_V1):
        raise StationError(f"unknown PatchCore model schema {schema!r}; expected {SCHEMA!r}")
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
        raise StationError(f"could not write PatchCore model {path}: {exc}") from exc


def cv2_rgb_decoder(cv2_module: Any = None) -> Decode:
    """Decoder for encoded image bytes in Frame.data (JPEG/PNG/BMP), returning RGB uint8 (H, W, 3).

    cv2.imdecode returns BGR; this converts to RGB, which is the channel order
    the feature extractors expect. cv2 is imported lazily; a missing install is
    a StationError. A payload cv2 cannot decode is a StationError.
    """
    if cv2_module is None:
        try:
            import cv2 as imported  # noqa: PLC0415 - imported lazily on purpose
        except ImportError as exc:
            raise StationError("opencv (cv2) is required to decode camera images; install opencv-python") from exc
        cv2_module = imported

    def decode(frame: Frame) -> Any:
        np = _np()
        buf = np.frombuffer(bytes(frame.data), dtype=np.uint8)
        bgr = cv2_module.imdecode(buf, cv2_module.IMREAD_COLOR)
        if bgr is None or getattr(bgr, "ndim", 0) != 3 or bgr.shape[2] != 3:
            raise StationError(f"could not decode the image from camera {frame.camera_id}")
        return np.ascontiguousarray(np.asarray(bgr)[:, :, ::-1])

    return decode


def register_patchcore_loader(extractor_factory: Callable[[], Any]) -> Callable[..., PatchCoreModel]:
    """Return a loader(info_or_path, decode, backend=None) for saved PatchCore models.

    Call this from the application. It imports station.modelstore only at call
    time, and registers the loader with modelstore.register_loader('patchcore', ...)
    when that module and function exist. Otherwise the loader is returned
    unregistered. The extractor is built by extractor_factory() at load time.
    """

    def loader(source: Any, decode: Decode, backend: Any = None) -> PatchCoreModel:
        # modelstore passes a ModelInfo; direct callers may pass the model path itself.
        path = getattr(source, "path", source)
        return PatchCoreModel.load(path, decode, extractor_factory(), backend=backend)

    try:
        modelstore: Any = importlib.import_module("station.modelstore")
    except ImportError:
        modelstore = None
    register = getattr(modelstore, "register_loader", None) if modelstore is not None else None
    if callable(register):
        register("patchcore", loader)
    return loader
