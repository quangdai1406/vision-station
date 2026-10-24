"""Inference models for the inspection station.

ScriptedModel returns preset results (tests). MarkerModel reads a score
marker embedded in frame bytes (end-to-end demo). OnnxModel wraps an ONNX
Runtime session; onnxruntime is imported lazily and never needed when a
session is injected.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Optional

from station.contracts import Defect, Frame, ImageResult, StationError

_MARKER_RE = re.compile(rb"(?:SCORE|CLASS|BOX)[A-Za-z0-9._,:;-]*")
_MARKER_KEYS = ("SCORE", "CLASS", "BOX")


class ScriptedModel:
    """Returns the scripted ImageResult for (camera_id, side), else a default."""

    def __init__(
        self,
        version: str,
        results_by_key: Optional[Mapping[tuple[str, str], ImageResult]] = None,
        default_score: float = 0.0,
    ) -> None:
        if not 0.0 <= default_score <= 1.0:
            raise ValueError("default_score must be between 0.0 and 1.0")
        self.version = version
        self._results = dict(results_by_key or {})
        self._default_score = default_score

    def predict(self, frame: Frame) -> ImageResult:
        found = self._results.get((frame.camera_id, frame.side.value))
        if found is not None:
            return found
        return ImageResult(
            camera_id=frame.camera_id,
            side=frame.side,
            score=self._default_score,
        )


def _parse_marker(data: bytes) -> dict[str, str]:
    """Return the marker fields in frame data, or {} when there is no marker."""
    match = _MARKER_RE.search(data)
    if match is None:
        return {}
    fields: dict[str, str] = {}
    for part in match.group(0).decode("ascii").split(";"):
        key, sep, value = part.partition(":")
        if not sep or key not in _MARKER_KEYS or value == "":
            raise StationError(f"malformed marker field {part!r}")
        if key in fields:
            raise StationError(f"duplicate marker field {key}")
        fields[key] = value
    return fields


def _parse_score(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise StationError(f"malformed marker SCORE {text!r}") from exc
    if not 0.0 <= value <= 1.0:  # also rejects NaN
        raise StationError(f"marker SCORE out of range 0..1: {text!r}")
    return value


def _parse_box(text: str) -> tuple[int, int, int, int]:
    parts = text.split(",")
    if len(parts) != 4:
        raise StationError(f"malformed marker BOX {text!r}: want x,y,w,h")
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise StationError(f"malformed marker BOX {text!r}") from exc
    if any(v < 0 for v in values):
        raise StationError(f"marker BOX has negative values: {text!r}")
    return values  # type: ignore[return-value]


class MarkerModel:
    """Demo model that reads a marker such as b'SCORE:0.72;CLASS:scratch;BOX:10,20,30,40'.

    Every field is optional. No marker gives score 0.0 and no defects. A
    CLASS field produces one Defect (box defaults to 0,0,0,0). A BOX field
    needs a CLASS field. Malformed markers raise StationError.
    """

    def __init__(self, version: str) -> None:
        self.version = version

    def predict(self, frame: Frame) -> ImageResult:
        fields = _parse_marker(frame.data)
        score = _parse_score(fields["SCORE"]) if "SCORE" in fields else 0.0
        defects: tuple[Defect, ...] = ()
        if "BOX" in fields and "CLASS" not in fields:
            raise StationError("marker BOX given without CLASS")
        if "CLASS" in fields:
            box = _parse_box(fields["BOX"]) if "BOX" in fields else (0, 0, 0, 0)
            defects = (
                Defect(defect_class=fields["CLASS"], score=score, box=box),
            )
        return ImageResult(
            camera_id=frame.camera_id,
            side=frame.side,
            score=score,
            defects=defects,
        )


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime  # noqa: PLC0415 - imported lazily on purpose
    except ImportError as exc:
        raise StationError(
            "onnxruntime is not installed; install it or inject session/ort_module"
        ) from exc
    return onnxruntime


class OnnxModel:
    """Runs an ONNX model: postprocess(frame, preprocess(frame) -> session.run)."""

    def __init__(
        self,
        path: Any,
        version: str,
        preprocess: Callable[[Frame], Any],
        postprocess: Callable[[Frame, Any], ImageResult],
        *,
        session: Any = None,
        ort_module: Any = None,
    ) -> None:
        self.version = version
        self._preprocess = preprocess
        self._postprocess = postprocess
        if session is None:
            ort = ort_module if ort_module is not None else _import_onnxruntime()
            try:
                session = ort.InferenceSession(str(path))
            except Exception as exc:
                raise StationError(f"could not load ONNX model {path}: {exc}") from exc
        self._session = session
        try:
            self._input_name = session.get_inputs()[0].name
        except Exception as exc:
            raise StationError(f"ONNX session has no usable input: {exc}") from exc

    def predict(self, frame: Frame) -> ImageResult:
        tensor = self._preprocess(frame)
        try:
            raw = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:
            raise StationError(
                f"ONNX inference failed for {frame.camera_id} {frame.side.value}: {exc}"
            ) from exc
        return self._postprocess(frame, raw)
