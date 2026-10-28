"""Wire the station together: config, build, run, and MES sync.

Plan reference: 26_1023_inspection_system_plan.md (sections 3, 5, 7).
Standard library only. The config schema is documented in README.md and
config.example.json.

Thread model:
  - UI server thread: HTTP requests. Every call into the session goes through
    _LockedSession, so one request at a time touches the session state.
  - MES sync thread: calls sync_once() every sync_interval_s.
  - Outbox: sqlite3 connections are opened per call (see _with_outbox), because
    a connection created on one thread cannot be used from another.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import re
import sys
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional

from station.auth import SessionManager, UserStore, load_users
from station.cameras import (STALE_FLUSH_FRAMES, FakeCameraSet, OpenCvCameraSet, make_blur_check,
                             utc_now_iso)
from station.contracts import (
    CAMERA_IDS,
    LABEL_OFFICIAL,
    LABEL_PROPOSED,
    LABEL_REJECTED,
    Frame,
    InvalidTransition,
    OperatorLabel,
    PartRecord,
    State,
    StationError,
    Thresholds,
    Verdict,
    camera_ids_for,
    images_for,
    is_full_station,
)
from station.inference import CollectModel, MarkerModel
from station.mes_client import MesClient, MesConfig, build_request
from station.modelstore import ModelError, ModelStore, load_runtime_model, register_loader
from station.outbox import Outbox
from station.outputs import ConsoleOutputs, SerialRelayOutputs
from station.parts import MesPartsClient, PartsError, build_part_config
from station.session import StationSession
from station.storage import (COLLECTED, Store, check_criterion_id, check_reject_reason,
                             label_status)
from station.ui import make_server

DEFAULT_UI_PORT = 8000
DEFAULT_SESSION_TIMEOUT_S = 900.0
DEFAULT_SYNC_INTERVAL_S = 5.0
DEFAULT_FLIP_TIMEOUT_S = 60.0
DEFAULT_MIN_FLIP_S = 2.0
DEFAULT_CAMERA_WIDTH = 2560
DEFAULT_CAMERA_HEIGHT = 1440
DEFAULT_BAD_DISPOSITION = "quarantine"
DEFAULT_CAPTURE_MODE = "simultaneous"
_CAPTURE_MODES = ("simultaneous", "sequential")

_PLACEHOLDER_RE = re.compile(r"^\[.*\]$")
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_LAMPS = ("green", "red", "alarm", "amber", "fault")

_REQUIRED_KEYS = (
    "site", "line", "station_id", "lot", "operator", "supervisor",
    "material_handler", "thresholds", "data_dir", "users_file", "mes", "cameras",
    "model", "outputs",
)
# part_number is required in 'fixed' mode only (checked in validate_config).
_OPTIONAL_KEYS = ("part_number", "parts", "ui", "flip_timeout_s", "min_flip_s", "frame_ok")
_ALL_KEYS = _REQUIRED_KEYS + _OPTIONAL_KEYS
_STRING_KEYS = (
    "site", "line", "station_id", "lot", "operator", "supervisor",
    "material_handler",
)
_EXTRACTOR_ARCHS = ("resnet18", "wide_resnet50_2")
_EXTRACTOR_DEVICES = ("auto", "cpu", "cuda")
_NEEDS_TORCH = "This part needs the anomaly model, which needs torch on this PC."


class ConfigError(ValueError):
    """The config file cannot be read, is not JSON, or breaks the schema. The message says where."""


# --------------------------------------------------------------------- config

def _where(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _check_keys(obj: dict, allowed: tuple[str, ...], path: str, required: tuple[str, ...] = ()) -> None:
    unknown = sorted(set(obj) - set(allowed))
    if unknown:
        raise ConfigError(f"{path or 'config'}: unknown key(s) {unknown}")
    missing = [k for k in required if k not in obj]
    if missing:
        raise ConfigError(f"{path or 'config'}: missing required key(s) {missing}")


def _obj(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a JSON object")
    return value


def _str(obj: dict, key: str, path: str) -> str:
    value = obj[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{_where(path, key)} must be a non-empty string")
    return value


def _number(value: Any, path: str, *, lo: Optional[float] = None, hi: Optional[float] = None,
            positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    if positive and value <= 0:
        raise ConfigError(f"{path} must be greater than 0")
    if lo is not None and value < lo:
        raise ConfigError(f"{path} must be >= {lo}")
    if hi is not None and value > hi:
        raise ConfigError(f"{path} must be <= {hi}")
    return float(value)


def _int(value: Any, path: str, *, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if not lo <= value <= hi:
        raise ConfigError(f"{path} must be between {lo} and {hi}")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be true or false")
    return value


def _find_placeholders(value: Any, path: str = "", found: Optional[list] = None) -> list[str]:
    if found is None:
        found = []
    if isinstance(value, dict):
        for key, item in value.items():
            _find_placeholders(item, _where(path, str(key)), found)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _find_placeholders(item, f"{path}[{index}]", found)
    elif isinstance(value, str) and _PLACEHOLDER_RE.match(value):
        found.append(path)
    return found


def _validate_mes(raw: Any) -> dict:
    path = "mes"
    mes = _obj(raw, path)
    _check_keys(mes, ("base_url", "token_env", "criteria_ids", "class_to_criterion",
                      "default_criterion_id", "sync_interval_s", "default_bad_disposition",
                      "allow_partial_cameras"),
                path, required=("base_url", "token_env", "criteria_ids", "class_to_criterion"))
    base_url = _str(mes, "base_url", path)
    if not base_url.startswith(("http://", "https://")):
        raise ConfigError("mes.base_url must start with http:// or https://")
    token_env = _str(mes, "token_env", path)
    if not _ENV_NAME_RE.match(token_env):
        raise ConfigError(
            "mes.token_env must be the NAME of an environment variable (upper case letters, "
            "digits and underscores, e.g. STATION_MES_TOKEN), not the token itself"
        )
    criteria = mes["criteria_ids"]
    if not isinstance(criteria, list) or not all(isinstance(c, str) and c for c in criteria):
        raise ConfigError("mes.criteria_ids must be a list of non-empty strings")
    if len(set(criteria)) != len(criteria):
        raise ConfigError("mes.criteria_ids must not contain duplicates")
    mapping = _obj(mes["class_to_criterion"], "mes.class_to_criterion")
    for defect_class, criterion in mapping.items():
        if not isinstance(criterion, str) or criterion not in criteria:
            raise ConfigError(
                f"mes.class_to_criterion[{defect_class!r}] must be one of mes.criteria_ids"
            )
    default_criterion = mes.get("default_criterion_id")
    if default_criterion is not None and (
        not isinstance(default_criterion, str) or default_criterion not in criteria
    ):
        raise ConfigError("mes.default_criterion_id must be null or one of mes.criteria_ids")
    disposition = mes.get("default_bad_disposition", DEFAULT_BAD_DISPOSITION)
    if disposition not in ("quarantine", "released"):
        raise ConfigError("mes.default_bad_disposition must be 'quarantine' or 'released'")
    return {
        "base_url": base_url.rstrip("/"),
        "token_env": token_env,
        "criteria_ids": list(criteria),
        "class_to_criterion": dict(mapping),
        "default_criterion_id": default_criterion,
        "sync_interval_s": _number(mes.get("sync_interval_s", DEFAULT_SYNC_INTERVAL_S),
                                   "mes.sync_interval_s", positive=True),
        "default_bad_disposition": disposition,
        "allow_partial_cameras": _bool(mes.get("allow_partial_cameras", False),
                                       "mes.allow_partial_cameras"),
    }


def _validate_cameras(raw: Any) -> dict:
    path = "cameras"
    cams = _obj(raw, path)
    _check_keys(cams, ("kind", "devices", "camera_count", "camera_ids", "width", "height", "manual",
                       "capture_mode", "sequential_keep_open", "session_mode", "capture"),
                path, required=("kind",))
    capture_mode = cams.get("capture_mode", DEFAULT_CAPTURE_MODE)
    if capture_mode not in _CAPTURE_MODES:
        raise ConfigError("cameras.capture_mode must be 'simultaneous' or 'sequential'")
    if capture_mode == "simultaneous" and "sequential_keep_open" in cams:
        raise ConfigError("cameras.sequential_keep_open only applies when cameras.capture_mode "
                          "is 'sequential'; remove it or set capture_mode to 'sequential'")
    session_mode = cams.get("session_mode", "default")
    if session_mode not in ("default", "keep_open"):
        raise ConfigError("cameras.session_mode must be 'default' or 'keep_open'")
    if session_mode == "default" and "capture" in cams:
        raise ConfigError("cameras.capture only applies when cameras.session_mode is "
                          "'keep_open'; remove it or set session_mode to 'keep_open'")
    if session_mode == "keep_open" and capture_mode != "simultaneous":
        raise ConfigError("cameras.session_mode 'keep_open' needs cameras.capture_mode 'simultaneous'")
    kind = cams["kind"]
    if kind not in ("fake", "opencv"):
        raise ConfigError("cameras.kind must be 'fake' or 'opencv'")
    devices = cams.get("devices")
    if kind == "opencv":
        if not isinstance(devices, list) or not 1 <= len(devices) <= len(CAMERA_IDS):
            raise ConfigError("cameras.devices must be a list of 1 to 4 device indexes (C1..C4)")
        devices = [_int(d, "cameras.devices[]", lo=0, hi=255) for d in devices]
        if len(set(devices)) != len(devices):
            raise ConfigError("cameras.devices must not contain duplicate device indexes")
        if "camera_count" in cams:
            raise ConfigError("cameras.camera_count is only for kind 'fake'; "
                              "with 'opencv' the number of devices sets the count")
        count = len(devices)
    else:
        if devices is not None:
            devices = [_int(d, "cameras.devices[]", lo=0, hi=255) for d in devices]
        count = _int(cams.get("camera_count", len(CAMERA_IDS)), "cameras.camera_count",
                     lo=1, hi=len(CAMERA_IDS))
    camera_ids = list(camera_ids_for(count))
    if "camera_ids" in cams and cams["camera_ids"] != camera_ids:
        raise ConfigError("cameras.camera_ids is derived from the devices (or camera_count); "
                          "remove it from the config")
    manual_raw = _obj(cams.get("manual", {}), "cameras.manual")
    _check_keys(manual_raw, ("focus", "exposure", "white_balance"), "cameras.manual")
    manual = {
        key: None if value is None else _number(value, f"cameras.manual.{key}")
        for key, value in manual_raw.items()
    }
    out = {
        "kind": kind,
        "devices": devices,
        "camera_ids": camera_ids,
        "width": _int(cams.get("width", DEFAULT_CAMERA_WIDTH), "cameras.width", lo=1, hi=100000),
        "height": _int(cams.get("height", DEFAULT_CAMERA_HEIGHT), "cameras.height", lo=1, hi=100000),
        "manual": manual,
        "capture_mode": capture_mode,
    }
    if capture_mode == "sequential":
        # Only written for sequential mode, so a validated config validates again to the same value.
        out["sequential_keep_open"] = _bool(cams.get("sequential_keep_open", False),
                                            "cameras.sequential_keep_open")
    if session_mode == "keep_open":
        # Only written for keep_open, so the default config validates to the same value as before.
        cap_raw = _obj(cams.get("capture", {}), "cameras.capture")
        _check_keys(cap_raw, ("mjpg", "threaded", "flush_frames"), "cameras.capture")
        out["session_mode"] = "keep_open"
        out["capture"] = {
            "mjpg": _bool(cap_raw.get("mjpg", False), "cameras.capture.mjpg"),
            "threaded": _bool(cap_raw.get("threaded", False), "cameras.capture.threaded"),
            "flush_frames": _int(cap_raw.get("flush_frames", STALE_FLUSH_FRAMES),
                                 "cameras.capture.flush_frames", lo=0, hi=100),
        }
    if kind == "fake":
        out["camera_count"] = count  # kept so a validated config validates again to the same value
    return out


def _validate_extractor(raw: Any) -> Optional[dict]:
    """model.extractor: the feature extractor for a PatchCore part model. Optional."""
    if raw is None:
        return None
    path = "model.extractor"
    ex = _obj(raw, path)
    _check_keys(ex, ("arch", "input_hw", "weights_path", "device"), path, required=("arch", "input_hw"))
    arch = ex["arch"]
    if arch not in _EXTRACTOR_ARCHS:
        raise ConfigError("model.extractor.arch must be 'resnet18' or 'wide_resnet50_2'")
    hw = ex["input_hw"]
    if (not isinstance(hw, list) or len(hw) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in hw)):
        raise ConfigError("model.extractor.input_hw must be [height, width] as two whole numbers")
    if min(hw) < 32:
        raise ConfigError("model.extractor.input_hw must be at least 32 pixels each side")
    weights = ex.get("weights_path")
    if weights is not None and (not isinstance(weights, str) or not weights):
        raise ConfigError("model.extractor.weights_path must be a file path, or left out")
    device = ex.get("device", "auto")
    if device not in _EXTRACTOR_DEVICES:
        raise ConfigError("model.extractor.device must be 'auto', 'cpu' or 'cuda'")
    return {"arch": arch, "input_hw": [hw[0], hw[1]], "weights_path": weights, "device": device}


def _validate_model(raw: Any) -> dict:
    path = "model"
    model = _obj(raw, path)
    _check_keys(model, ("kind", "path", "version", "extractor"), path, required=("kind", "version"))
    kind = model["kind"]
    if kind not in ("marker", "onnx", "golden", "collect"):
        raise ConfigError("model.kind must be 'marker', 'onnx', 'golden' or 'collect'")
    if kind == "collect" and model.get("path") is not None:
        raise ConfigError("model.path is not used when model.kind is 'collect': remove it")
    model_path = model.get("path")
    if kind in ("onnx", "golden") and (not isinstance(model_path, str) or not model_path):
        raise ConfigError(f"model.path is required when model.kind is '{kind}'")
    if model_path is not None and not isinstance(model_path, str):
        raise ConfigError("model.path must be a string")
    return {
        "kind": kind,
        "path": model_path,
        "version": _str(model, "version", path),
        "extractor": _validate_extractor(model.get("extractor")),
    }


def _validate_parts(raw: Any) -> dict:
    """parts: 'fixed' (one configured part, the default) or 'mes' (the operator picks a part)."""
    parts = _obj({} if raw is None else raw, "parts")
    _check_keys(parts, ("mode", "models_dir"), "parts")
    mode = parts.get("mode", "fixed")
    if mode not in ("fixed", "mes"):
        raise ConfigError("parts.mode must be 'fixed' or 'mes'")
    models_dir = parts.get("models_dir", "./models")
    if not isinstance(models_dir, str) or not models_dir.strip():
        raise ConfigError("parts.models_dir must be a non-empty string")
    if mode == "mes" and not os.path.isdir(models_dir):
        raise ConfigError(
            f"parts.models_dir {models_dir} does not exist: create the folder that holds the "
            "approved models, or fix the path"
        )
    return {"mode": mode, "models_dir": models_dir}


def _validate_outputs(raw: Any) -> dict:
    path = "outputs"
    outputs = _obj(raw, path)
    _check_keys(outputs, ("kind", "port", "channel_map"), path, required=("kind",))
    kind = outputs["kind"]
    if kind not in ("console", "serial_relay"):
        raise ConfigError("outputs.kind must be 'console' or 'serial_relay'")
    port = outputs.get("port")
    if kind == "serial_relay" and (not isinstance(port, str) or not port):
        raise ConfigError("outputs.port is required when outputs.kind is 'serial_relay'")
    channel_map = outputs.get("channel_map")
    if channel_map is not None:
        channel_map = _obj(channel_map, "outputs.channel_map")
        _check_keys(channel_map, _LAMPS, "outputs.channel_map")
        channel_map = {k: _int(v, f"outputs.channel_map.{k}", lo=1, hi=255)
                       for k, v in channel_map.items()}
    return {"kind": kind, "port": port, "channel_map": channel_map}


def validate_config(data: Any) -> dict:
    """Check a config dict against the schema and return a normalised copy with defaults filled.

    Paths are used as given. load_config() resolves them against the config file's folder first.
    """
    cfg = _obj(data, "config")
    _check_keys(cfg, _ALL_KEYS, "")
    placeholders = _find_placeholders(cfg)
    if placeholders:
        raise ConfigError(
            "these values are still placeholders, fill them in: " + ", ".join(placeholders)
        )

    parts = _validate_parts(cfg.get("parts"))
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if parts["mode"] == "fixed" and "part_number" not in cfg:
        missing.append("part_number")  # a fixed station names its part; a 'mes' station picks one
    if missing:
        raise ConfigError(f"config: missing required key(s) {missing}")

    out: dict[str, Any] = {"parts": parts}
    for key in _STRING_KEYS:
        out[key] = _str(cfg, key, "")
    if parts["mode"] == "fixed":
        out["part_number"] = _str(cfg, "part_number", "")
    else:
        part_number = cfg.get("part_number", "")
        if not isinstance(part_number, str):
            raise ConfigError("part_number must be a string (it may be left empty in 'mes' mode)")
        out["part_number"] = part_number.strip()
    for key in ("data_dir", "users_file"):
        out[key] = _str(cfg, key, "")

    thresholds = _obj(cfg["thresholds"], "thresholds")
    _check_keys(thresholds, ("t_low", "t_img"), "thresholds", required=("t_low", "t_img"))
    t_low = _number(thresholds["t_low"], "thresholds.t_low", lo=0.0, hi=1.0)
    t_img = _number(thresholds["t_img"], "thresholds.t_img", lo=0.0, hi=1.0)
    if not t_low <= t_img:
        raise ConfigError("thresholds.t_low must be <= thresholds.t_img")
    out["thresholds"] = {"t_low": t_low, "t_img": t_img}

    ui = _obj(cfg.get("ui", {}), "ui")
    _check_keys(ui, ("host", "port", "session_timeout_s"), "ui")
    host = ui.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host:
        raise ConfigError("ui.host must be a non-empty string")
    out["ui"] = {
        "host": host,
        "port": _int(ui.get("port", DEFAULT_UI_PORT), "ui.port", lo=0, hi=65535),
        "session_timeout_s": _number(ui.get("session_timeout_s", DEFAULT_SESSION_TIMEOUT_S),
                                     "ui.session_timeout_s", positive=True),
    }

    out["mes"] = _validate_mes(cfg["mes"])
    out["cameras"] = _validate_cameras(cfg["cameras"])
    out["model"] = _validate_model(cfg["model"])
    if out["model"]["kind"] == "collect" and parts["mode"] != "fixed":
        raise ConfigError("model.kind 'collect' needs parts.mode 'fixed': collect mode has no approved model per part")
    out["outputs"] = _validate_outputs(cfg["outputs"])
    out["flip_timeout_s"] = _number(cfg.get("flip_timeout_s", DEFAULT_FLIP_TIMEOUT_S),
                                    "flip_timeout_s", positive=True)
    out["min_flip_s"] = _number(cfg.get("min_flip_s", DEFAULT_MIN_FLIP_S), "min_flip_s", lo=0.0)

    frame_ok = cfg.get("frame_ok")
    if frame_ok is not None:
        frame_ok = _obj(frame_ok, "frame_ok")
        _check_keys(frame_ok, ("min_variance",), "frame_ok", required=("min_variance",))
        frame_ok = {"min_variance": _number(frame_ok["min_variance"], "frame_ok.min_variance", lo=0.0)}
    out["frame_ok"] = frame_ok
    return out


def load_config(path: str | os.PathLike) -> dict:
    """Read, resolve and validate a JSON config file. Relative paths are relative to the file."""
    cfg_path = pathlib.Path(path)
    try:
        text = cfg_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {cfg_path}: {exc.strerror or exc}") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ConfigError(f"config {cfg_path} is not valid JSON: {exc}") from exc
    base = cfg_path.resolve().parent
    if isinstance(data, dict):
        for key in ("data_dir", "users_file"):
            data[key] = _resolve_path(data.get(key), base)
        model = data.get("model")
        if isinstance(model, dict):
            model["path"] = _resolve_path(model.get("path"), base)
            extractor = model.get("extractor")
            if isinstance(extractor, dict):
                extractor["weights_path"] = _resolve_path(extractor.get("weights_path"), base)
        if data.get("parts") is None:
            data["parts"] = {}
        parts = data["parts"]
        if isinstance(parts, dict):
            parts["models_dir"] = _resolve_path(parts.get("models_dir", "./models"), base)
    return validate_config(data)


def _resolve_path(value: Any, base: pathlib.Path) -> Any:
    if not isinstance(value, str) or not value or _PLACEHOLDER_RE.match(value):
        return value
    p = pathlib.Path(value).expanduser()
    return str(p if p.is_absolute() else base / p)


# ---------------------------------------------------------------- part ids

def _utc_datetime(text: str) -> _dt.datetime:
    dt = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt.astimezone(_dt.timezone.utc)


class _PartIdFactory:
    """P<YYYYMMDDHHMMSS><nnnn>. nnnn counts from 0001 within each second.

    An id is never issued twice in this run, and an id whose part folder already
    exists (from an earlier run in the same second) is skipped.
    """

    def __init__(self, store: Store, now_utc: Callable[[], str]) -> None:
        self._store = store
        self._now_utc = now_utc
        self._lock = threading.Lock()
        self._issued: set[str] = set()
        self._second: Optional[str] = None
        self._counter = 0

    def __call__(self) -> str:
        with self._lock:
            ts = _utc_datetime(self._now_utc())
            second = ts.strftime("%Y%m%d%H%M%S")
            if second != self._second:
                self._second = second
                self._counter = 0
            while True:
                self._counter += 1
                if self._counter > 9999:
                    raise StationError("more than 9999 parts in one second")
                part_id = f"P{second}{self._counter:04d}"
                if part_id not in self._issued and not self._on_disk(ts, part_id):
                    break
            self._issued.add(part_id)
            return part_id

    def _on_disk(self, ts: _dt.datetime, part_id: str) -> bool:
        folder = (self._store.store_dir / self._store.line / ts.strftime("%Y")
                  / ts.strftime("%m") / ts.strftime("%d") / part_id)
        return folder.exists()


# ----------------------------------------------------------- session wrapper

def _need_name(by: Any) -> None:
    if not isinstance(by, str) or not by.strip():
        raise ValueError("by is required")


def _label_result(session_id: str, label: OperatorLabel, previous: Optional[dict]) -> dict:
    return {"session_id": session_id, "verdict": Verdict(label.verdict).value,
            "criterion_id": label.criterion_id, "status": label.status,
            "changed": previous is not None}


class _LockedSession:
    """Serialises every UI call into the session.

    app (optional) supplies the parts mode, the camera state and the part calls.
    """

    def __init__(self, session: StationSession, lock: threading.RLock, app: Any = None) -> None:
        self._session = session
        self._lock = lock
        self._app = app

    def snapshot(self, include_images: bool = False) -> dict:
        with self._lock:
            data = self._session.snapshot(include_images=include_images)
            app = self._app
            data["parts_mode"] = app.parts_mode if app is not None else "fixed"
            data["camera"] = app.camera_status() if app is not None else {"open": True, "last_error": None}
            return data

    def search_parts(self, q: str, limit: int = 20) -> list:
        # A network call: not under the session lock, so a slow MES does not freeze the station.
        return self._need_app().search_parts(q, limit)

    def select_part(self, part_number: str, by: str) -> dict:
        # No lock here: StationApp.select_part does its MES calls unlocked, then takes the lock.
        return self._need_app().select_part(part_number, by)

    def current_part(self) -> dict:
        with self._lock:
            return self._need_app().current_part()

    def _need_app(self) -> Any:
        if self._app is None:
            raise StationError("part selection is not wired to a station")
        return self._app

    def press(self) -> bool:
        with self._lock:
            return self._session.press()

    def ack_alarm(self) -> None:
        with self._lock:
            self._session.ack_alarm()

    def clear_fault(self) -> None:
        with self._lock:
            self._session.clear_fault()

    def resolve_review(self, by: str, new_verdict, reason: str) -> None:
        with self._lock:
            self._session.resolve_review(by=by, new_verdict=new_verdict, reason=reason)

    def override(self, by: str, new_verdict, reason: str) -> None:
        with self._lock:
            self._session.override(by=by, new_verdict=new_verdict, reason=reason)

    def label_part(self, by: str, verdict, criterion_id: Optional[str] = None,
                   part_id: Optional[str] = None) -> dict:
        # Stored parts only: the store locks itself, so the session lock is not needed.
        return self._need_app().label_part(by=by, verdict=verdict, criterion_id=criterion_id,
                                           part_id=part_id)

    def approve_label(self, by: str, part_id: Optional[str] = None) -> dict:
        return self._need_app().approve_label(by=by, part_id=part_id)

    def reject_label(self, by: str, reason: str, part_id: Optional[str] = None) -> dict:
        return self._need_app().reject_label(by=by, reason=reason, part_id=part_id)

    def set_official_label(self, by: str, verdict, criterion_id: Optional[str] = None,
                           part_id: Optional[str] = None) -> dict:
        return self._need_app().set_official_label(by=by, verdict=verdict,
                                                   criterion_id=criterion_id, part_id=part_id)

    def last_label_status(self) -> Optional[dict]:
        return self._need_app().last_label_status()

    def pending_labels(self) -> list:
        return self._need_app().pending_labels()

    def tick(self) -> bool:
        with self._lock:
            return self._session.tick()

    def preview(self, camera_id: str) -> Optional[bytes]:
        """Latest live frame for one camera, or None. Never blocks and never raises.

        A press holds the lock while it captures, so a preview request that cannot
        get the lock at once gets None instead of waiting for the press to finish.
        """
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._session.preview(camera_id)
        except Exception:
            return None
        finally:
            self._lock.release()


# --------------------------------------------------------------- the app

def _grey_decoder() -> Callable[[Frame], list]:
    try:
        import cv2  # noqa: PLC0415 - optional dependency
        import numpy  # noqa: PLC0415 - optional dependency
    except ImportError as exc:
        raise StationError("frame_ok.min_variance needs opencv-python and numpy installed") from exc

    def decode(frame: Frame) -> list:
        img = cv2.imdecode(numpy.frombuffer(frame.data, dtype=numpy.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise StationError(f"{frame.camera_id}: frame is not a decodable image")
        return img.tolist()

    return decode


def _load_golden_model(model: dict) -> Any:
    """Load a golden .npz made by `python -m station teach`. station.golden is imported here, lazily."""
    from station.golden import GoldenModel, cv2_decoder  # noqa: PLC0415 - needs numpy and cv2 at run time

    path = model["path"]
    if not os.path.isfile(path):
        raise StationError(f"golden model file not found: {path}; run python -m station teach first")
    golden = GoldenModel.load(path, cv2_decoder())
    golden.version = model["version"]  # stored records show the config version, not the file's
    return golden


def _torch_importable() -> bool:
    """True when torch can be imported. Checked only; the extractor itself imports it when built."""
    try:
        import torch  # noqa: F401,PLC0415 - optional dependency
    except ImportError:
        return False
    return True


class StationApp:
    """A fully wired station. Build with build_station(); call start() to run it."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        cameras=None,
        model=None,
        outputs=None,
        now_utc: Optional[Callable[[], str]] = None,
        clock: Optional[Callable[[], float]] = None,
        token: Optional[str] = None,
        demo: bool = False,
    ) -> None:
        cfg = validate_config(dict(config))
        self.config = cfg
        self._parts_mode = cfg["parts"]["mode"]
        self._models_dir = cfg["parts"]["models_dir"]
        self._part_configs: dict[str, dict] = {}  # part number -> criteria config from the MES
        self._parts_client: Optional[MesPartsClient] = None
        self._now_utc = now_utc or utc_now_iso
        self._clock = clock or time.monotonic
        self._stop_event = threading.Event()
        self._sync_lock = threading.Lock()
        self._session_lock = threading.RLock()
        self._label_lock = threading.Lock()  # one label write at a time: check, then write
        self._started = False
        self._closed = False
        self._ui_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self.sync_interval_s = cfg["mes"]["sync_interval_s"]
        self._collect_only = cfg["model"]["kind"] == "collect"

        if demo and self._parts_mode == "mes":
            raise ConfigError("the demo station uses the demo model: set parts.mode to 'fixed' for --demo")

        # Devices and model first: a failure here leaves nothing open.
        cameras = cameras if cameras is not None else self._make_cameras(cfg)
        self._cameras = cameras
        self._camera_ids = tuple(cfg["cameras"]["camera_ids"])
        if self._parts_mode == "mes":
            model = None  # the session starts with no part and no model; an injected model is ignored
        else:
            model = model if model is not None else self._make_model(cfg)
        outputs = outputs if outputs is not None else self._make_outputs(cfg)

        # A partial station (fewer than the 4 cameras) is proof-of-concept only: it is the
        # only kind of station allowed to store fewer than 8 images.
        self.partial = not is_full_station(cfg["cameras"]["camera_ids"])
        self.store = Store(pathlib.Path(cfg["data_dir"]), site=cfg["site"], line=cfg["line"],
                           allow_partial=self.partial)
        self._outbox_root = pathlib.Path(cfg["data_dir"]) / "outbox"
        self.outbox = Outbox(self._outbox_root)
        try:
            self._wire(cfg, cameras, model, outputs, token=token, demo=demo)
        except BaseException:
            self.outbox.close()
            raise

    # ----------------------------------------------------------- wiring

    @staticmethod
    def _make_cameras(cfg: dict):
        cams = cfg["cameras"]
        if cams["kind"] == "fake":
            return FakeCameraSet(camera_ids=cams["camera_ids"])
        # The capture kwargs are passed only in sequential mode, so the default
        # (simultaneous) construction is unchanged.
        mode_kwargs: dict[str, Any] = {}
        if cams.get("capture_mode") == "sequential":
            mode_kwargs = {"capture_mode": "sequential",
                           "sequential_keep_open": cams["sequential_keep_open"]}
        if cams.get("session_mode") == "keep_open":
            cap = cams["capture"]
            mode_kwargs.update(session_mode="keep_open", mjpg=cap["mjpg"], threaded=cap["threaded"],
                               stale_flush_frames=cap["flush_frames"])
        return OpenCvCameraSet(
            cams["devices"], width=cams["width"], height=cams["height"],
            manual_settings={k: v for k, v in cams["manual"].items() if v is not None},
            **mode_kwargs,
        )

    @staticmethod
    def _make_model(cfg: dict):
        model = cfg["model"]
        if model["kind"] == "marker":
            return MarkerModel(version=model["version"])
        if model["kind"] == "golden":
            return _load_golden_model(model)
        if model["kind"] == "collect":
            return CollectModel(version=model["version"])
        raise StationError(
            "model.kind 'onnx' is not wired yet: the ONNX pre- and postprocessing are not written "
            "(see README, 'Not done'). Use model.kind 'marker' with --demo, or inject a model."
        )

    @staticmethod
    def _make_outputs(cfg: dict):
        out = cfg["outputs"]
        if out["kind"] == "console":
            return ConsoleOutputs()
        return SerialRelayOutputs(_serial_writer(out["port"]), channel_map=out["channel_map"])

    def _wire(self, cfg: dict, cameras, model, outputs, *, token: Optional[str], demo: bool) -> None:
        self.session_thresholds = Thresholds(t_low=cfg["thresholds"]["t_low"],
                                             t_img=cfg["thresholds"]["t_img"])

        # MES: a missing token disables posting, not inspection. A partial station never
        # posts unless mes.allow_partial_cameras is true.
        mes = cfg["mes"]
        self._demo = demo
        camera_ids = tuple(cfg["cameras"]["camera_ids"])
        allow_partial = mes["allow_partial_cameras"]
        self._partial_blocked = self.partial and not allow_partial
        if token is None:
            token = os.environ.get(mes["token_env"], "")
        self.mes_enabled = bool(token) and not demo and not self._partial_blocked and not self._collect_only
        self._mes_cfg = MesConfig(
            base_url=mes["base_url"],
            token=token or "",
            criteria_ids=mes["criteria_ids"],
            class_to_criterion=mes["class_to_criterion"],
            default_criterion_id=mes["default_criterion_id"],
            default_bad_disposition=mes["default_bad_disposition"],
        )
        self._client = MesClient(self._mes_cfg) if self.mes_enabled else None
        # Part search needs the token too, whatever demo or partial rules say about posting.
        self._parts_client = (MesPartsClient(mes["base_url"], token)
                              if self._parts_mode == "mes" and token else None)
        if self._collect_only:
            self.store.append_event("mes_disabled", {
                "reason": "collect mode: parts are saved with no verdict and are not sent to MES",
            })
        if demo:
            self.store.append_event("mes_disabled", {"reason": "demo mode: parts are not sent to MES"})
        else:
            if self._partial_blocked:
                self.store.append_event("mes_disabled", {
                    "reason": f"partial station ({len(camera_ids)} camera(s)): "
                              "parts are not sent to the MES",
                })
            if not token:
                self.store.append_event("mes_disabled", {
                    "reason": f"no token: the environment variable {mes['token_env']} is not set",
                })
        if self.partial and allow_partial:
            self.store.append_event("mes_partial_allowed", {
                "camera_ids": list(camera_ids),
                "images_per_part": images_for(camera_ids),
                "reason": "mes.allow_partial_cameras is true: parts with fewer than 8 images are sent to the MES",
            })

        frame_ok = None
        if cfg["frame_ok"] is not None:
            frame_ok = make_blur_check(_grey_decoder(), cfg["frame_ok"]["min_variance"])

        self.session = StationSession(
            cameras, model, outputs, self.store, self.session_thresholds,
            station_id=cfg["station_id"],
            part_number="" if self._parts_mode == "mes" else cfg["part_number"],
            lot=cfg["lot"],
            operator=cfg["operator"],
            supervisor=cfg["supervisor"],
            material_handler=cfg["material_handler"],
            flip_timeout_s=cfg["flip_timeout_s"],
            min_flip_s=cfg["min_flip_s"],
            frame_ok=frame_ok,
            clock=self._clock,
            now_utc=self._now_utc,
            part_id_factory=_PartIdFactory(self.store, self._now_utc),
            on_complete=self._enqueue,
            initial_counters=self.store.counters(),
            camera_ids=camera_ids,
            collect_only=self._collect_only,
        )

        try:
            users = load_users(cfg["users_file"])
        except FileNotFoundError as exc:
            raise StationError(
                f"users file not found: {cfg['users_file']} "
                "(create users with: python -m station users add <username> <role>)"
            ) from exc
        self.users: UserStore = users
        self.sessions = SessionManager(timeout_s=cfg["ui"]["session_timeout_s"])
        self._locked = _LockedSession(self.session, self._session_lock, self)
        self.server = make_server(self._locked, users, self.sessions,
                                  host=cfg["ui"]["host"], port=cfg["ui"]["port"])

    # ------------------------------------------------------------ MES

    def _with_outbox(self, fn: Callable[[Outbox], Any]) -> Any:
        """Run fn on a fresh Outbox connection, closed afterwards (safe from any thread)."""
        ob = Outbox(self._outbox_root)
        try:
            return fn(ob)
        finally:
            ob.close()

    def _enqueue(self, record: PartRecord) -> None:
        """on_complete hook: queue the saved part for MES. Never raises."""
        if self._demo:
            return  # demo parts must never reach the outbox, which a later real run would send
        if record.disposition == COLLECTED:
            return  # collected parts have no verdict and are never posted to the MES
        if self._partial_blocked:
            return  # a partial station posts only when mes.allow_partial_cameras is true
        try:
            part_dir = self.store._known_path(record.session_id)
            if part_dir is None:
                raise StationError("saved part is not in the store index")
            fields, files = build_request(record, self._mes_cfg_for(record.part_number),
                                          self.store.archive_ref(part_dir))
            self._with_outbox(lambda ob: ob.enqueue(record.session_id, fields, files))
        except Exception as exc:
            self.store.append_event(
                "mes_enqueue_failed",
                {"part_id": record.part_id, "error": str(exc) or type(exc).__name__},
                session_id=record.session_id,
            )

    def sync_once(self, now: Optional[float] = None) -> dict[str, int]:
        """Drain the outbox once. Logs a 'mes_sync' event when anything was attempted."""
        empty = {"sent": 0, "failed_permanent": 0, "retry_later": 0}
        if self._client is None:
            return empty
        with self._sync_lock:
            counts = self._with_outbox(lambda ob: ob.drain(self._client, now=now))
        if any(counts.values()):
            self.store.append_event("mes_sync", dict(counts))
        return counts

    def _sync_loop(self) -> None:
        while not self._stop_event.wait(self.sync_interval_s):
            try:
                self.sync_once()
            except Exception as exc:  # keep the thread alive; the next round retries
                self.store.append_event("mes_sync_failed", {"error": str(exc) or type(exc).__name__})

    # ------------------------------------------------------- lifecycle

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._open_cameras()
        self._ui_thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.2},
            name="station-ui", daemon=True,
        )
        self._ui_thread.start()
        if self._client is not None:
            self._sync_thread = threading.Thread(target=self._sync_loop, name="mes-sync", daemon=True)
            self._sync_thread.start()

    def _open_cameras(self) -> None:
        """Open the cameras before the UI starts. Never raises.

        A failure is logged ('camera_open_failed' and one stderr line) and the
        station still starts: the live view retries, and capture() retries on Enter.
        """
        opener = getattr(self._cameras, "open", None)
        if not callable(opener):
            return
        try:
            opener()
        except Exception as exc:
            text = str(exc) or type(exc).__name__
            self._log_quietly("camera_open_failed", {"error": text})
            try:
                print(f"camera not opened: {text}. The live view will keep retrying every few seconds.",
                      file=sys.stderr)
            except Exception:
                pass
            return
        self._log_quietly("camera_open", {"camera_ids": list(self._camera_ids)})

    def _log_quietly(self, kind: str, payload: dict) -> None:
        try:
            self.store.append_event(kind, payload)
        except Exception:
            pass

    def camera_status(self) -> dict:
        """{'open': bool, 'last_error': str or None} for the UI. Cameras without is_open count as open."""
        cams = self._cameras
        if not hasattr(cams, "is_open"):
            return {"open": True, "last_error": None}
        return {"open": bool(cams.is_open), "last_error": getattr(cams, "last_open_error", None)}

    # ------------------------------------------------------------ parts

    @property
    def parts_mode(self) -> str:
        return self._parts_mode

    def search_parts(self, q: str, limit: int = 20) -> list[dict]:
        """MES part search for the UI. A network call: callers must not hold the session lock."""
        self._require_mes_parts()
        return self._require_parts_client().search(q, limit)

    def current_part(self) -> dict:
        with self._session_lock:
            s = self.session
            return {
                "part_number": s.part_number,
                "model_version": s.model_version,
                "can_inspect": s.can_inspect,
                "reason": s.block_reason,
                "parts_mode": self._parts_mode,
            }

    def select_part(self, part_number: str, by: str) -> dict:
        """Make a part the one being inspected, with that part's approved model. Takes the session lock.

        Raises ValueError (fixed station or bad input), InvalidTransition (not IDLE) and PartsError
        (not in the MES, MES unreachable, no token). A part with no usable approved model is not an
        error: the session is blocked with the reason and the result has can_inspect False.
        """
        self._require_mes_parts()
        if not isinstance(part_number, str) or not part_number.strip():
            raise ValueError("part number must be a non-empty string")
        if not isinstance(by, str) or not by.strip():
            raise ValueError("by is required")
        pn = part_number.strip()
        # MES calls can take seconds: do them before the lock so presses and snapshots keep working.
        client = self._require_parts_client()
        if client.get_part(pn) is None:
            raise PartsError(f"part {pn} not found in the MES")
        part_cfg = build_part_config(client.criteria(pn))
        warnings = list(part_cfg["warnings"])
        with self._session_lock:
            if self.session.state is not State.IDLE:
                raise InvalidTransition(f"a part cannot be selected while {self.session.state.value}")
            self._part_configs[pn] = part_cfg
            try:
                info = ModelStore(self._models_dir).current(pn)
            except ModelError as exc:
                return self._block_part(pn, str(exc), warnings)
            if info is None:
                return self._block_part(pn, f"No approved model for part {pn}. Ask Quality to approve one.",
                                        warnings)
            try:
                model = self._load_part_model(info)
            except StationError as exc:
                return self._block_part(pn, str(exc), warnings)
            self.session.set_blocked(None)
            self.session.set_part(pn, model, info.version)
            self.store.append_event("part_selected",
                                    {"part_number": pn, "model_version": info.version, "by": by.strip()})
            return {"part_number": pn, "model_version": info.version, "can_inspect": True,
                    "warnings": warnings}

    # ------------------------------------------------------------ labels
    # Two-step approval. Operators PROPOSE (status 'proposed'); only an admin APPROVES, REJECTS or
    # SETS the official label. Only an 'official' label is used for training. Nothing here changes
    # the part's verdict, counters, images, manifest or the MES outbox.

    def _label_target(self, part_id: Optional[str]) -> tuple[str, dict]:
        """(session_id, part.json) of the named collected part, else of the last collected part."""
        if part_id is not None:
            session_id = self.store.collected_session_id_for_part(str(part_id))
            if session_id is None:
                raise ValueError("no collected part with that part_id")
        else:
            session_id = self.store.last_collected_session_id()
            if session_id is None:
                raise ValueError("there is no collected part to label")
        return session_id, self.store.collected_part_doc(session_id)

    def _allowed_criteria(self, part_number: Any) -> set[str]:
        """Configured mes.criteria_ids plus the part's own MES criteria (parts mode). May be empty."""
        allowed: set[str] = set()
        mes_cfg = getattr(self, "_mes_cfg", None)
        if mes_cfg is not None:
            allowed.update(mes_cfg.criteria_ids)
        if self._parts_mode == "mes" and part_number:
            part = self._part_configs.get(str(part_number))
            if part is not None:
                allowed.update(part["criteria_ids"])
        return allowed

    def _check_label_criterion(self, verdict: Verdict, criterion_id: Optional[str],
                               part_number: Any) -> Optional[str]:
        if criterion_id is None:
            return None
        if verdict is Verdict.GOOD:
            raise ValueError("a GOOD label takes no criterion_id")
        criterion = check_criterion_id(criterion_id)
        allowed = self._allowed_criteria(part_number)
        # Neither list available (fixed mode, no criteria_ids): any valid free name, as before.
        if allowed and criterion not in allowed:
            raise ValueError("criterion_id must be one of the configured mes.criteria_ids "
                             "or this part's MES criteria")
        return criterion

    def _write_label(self, session_id: str, label: OperatorLabel, action: str) -> Optional[dict]:
        return self.store.set_operator_label(session_id, label, action=action)

    def label_part(self, by: str, verdict: Any, criterion_id: Optional[str] = None,
                   part_id: Optional[str] = None) -> dict:
        """PROPOSE a label for a collected part (operator, supervisor or admin). Status 'proposed'.

        A new proposal replaces an earlier proposal or rejection. An official label cannot be
        proposed over: only an admin changes it (set_official_label).
        """
        new_verdict = Verdict(verdict)
        if new_verdict not in (Verdict.GOOD, Verdict.BAD):
            raise ValueError("verdict must be GOOD or BAD")
        if not isinstance(by, str) or not by.strip():
            raise ValueError("by is required")
        with self._label_lock:
            session_id, doc = self._label_target(part_id)
            criterion = self._check_label_criterion(new_verdict, criterion_id, doc.get("part_number"))
            if label_status(doc.get("operator_label")) == LABEL_OFFICIAL:
                raise ValueError("this part has an official label: only an admin can change it")
            now = self._now_utc()
            label = OperatorLabel(verdict=new_verdict, labelled_by=by.strip(), labelled_at=now,
                                  criterion_id=criterion, status=LABEL_PROPOSED,
                                  proposed_by=by.strip(), proposed_at=now)
            previous = self._write_label(session_id, label, "propose")
        return {"session_id": session_id, "verdict": new_verdict.value, "criterion_id": criterion,
                "status": LABEL_PROPOSED, "changed": previous is not None}

    def approve_label(self, by: str, part_id: Optional[str] = None) -> dict:
        """ADMIN: make the pending proposal of a collected part OFFICIAL."""
        _need_name(by)
        with self._label_lock:
            session_id, doc = self._label_target(part_id)
            current = doc.get("operator_label")
            if label_status(current) != LABEL_PROPOSED:
                raise ValueError("there is no pending proposal on this part to approve")
            now = self._now_utc()
            label = OperatorLabel(verdict=Verdict(current["verdict"]), labelled_by=by.strip(),
                                  labelled_at=now, criterion_id=current.get("criterion_id"),
                                  status=LABEL_OFFICIAL, proposed_by=current["proposed_by"],
                                  proposed_at=current["proposed_at"], approved_by=by.strip(),
                                  approved_at=now)
            previous = self._write_label(session_id, label, "approve")
        return _label_result(session_id, label, previous)

    def reject_label(self, by: str, reason: str, part_id: Optional[str] = None) -> dict:
        """ADMIN: reject the pending proposal of a collected part, with a short reason."""
        _need_name(by)
        reject_reason = check_reject_reason(reason)
        with self._label_lock:
            session_id, doc = self._label_target(part_id)
            current = doc.get("operator_label")
            if label_status(current) != LABEL_PROPOSED:
                raise ValueError("there is no pending proposal on this part to reject")
            now = self._now_utc()
            label = OperatorLabel(verdict=Verdict(current["verdict"]), labelled_by=by.strip(),
                                  labelled_at=now, criterion_id=current.get("criterion_id"),
                                  status=LABEL_REJECTED, proposed_by=current["proposed_by"],
                                  proposed_at=current["proposed_at"], approved_by=by.strip(),
                                  approved_at=now, reject_reason=reject_reason)
            previous = self._write_label(session_id, label, "reject")
        return _label_result(session_id, label, previous)

    def set_official_label(self, by: str, verdict: Any, criterion_id: Optional[str] = None,
                           part_id: Optional[str] = None) -> dict:
        """ADMIN: set the OFFICIAL label of a collected part directly (replaces any earlier label)."""
        _need_name(by)
        new_verdict = Verdict(verdict)
        if new_verdict not in (Verdict.GOOD, Verdict.BAD):
            raise ValueError("verdict must be GOOD or BAD")
        with self._label_lock:
            session_id, doc = self._label_target(part_id)
            criterion = self._check_label_criterion(new_verdict, criterion_id, doc.get("part_number"))
            now = self._now_utc()
            label = OperatorLabel(verdict=new_verdict, labelled_by=by.strip(), labelled_at=now,
                                  criterion_id=criterion, status=LABEL_OFFICIAL,
                                  approved_by=by.strip(), approved_at=now)
            previous = self._write_label(session_id, label, "set_official")
        return _label_result(session_id, label, previous)

    def last_label_status(self) -> Optional[dict]:
        """The last collected part and its label (any status), or None if there is none."""
        session_id = self.store.last_collected_session_id()
        if session_id is None:
            return None
        doc = self.store.collected_part_doc(session_id)
        label = doc.get("operator_label")
        return {"session_id": session_id, "part_id": doc.get("part_id"),
                "part_number": doc.get("part_number"),
                "label_status": label_status(label) if label is not None else None,
                "verdict": label.get("verdict") if label is not None else None,
                "criterion_id": label.get("criterion_id") if label is not None else None,
                "reject_reason": label.get("reject_reason") if label is not None else None}

    def pending_labels(self) -> list[dict]:
        """Collected parts with a proposal waiting for an admin, oldest first."""
        rows = self.store.collected_labels(status=LABEL_PROPOSED)
        return [{"session_id": r["session_id"], "part_id": r["part_id"],
                 "part_number": r["part_number"], "verdict": r["operator_label"]["verdict"],
                 "criterion_id": r["operator_label"].get("criterion_id"),
                 "proposed_by": r["operator_label"].get("proposed_by"),
                 "proposed_at": r["operator_label"].get("proposed_at")} for r in rows]

    def _block_part(self, part_number: str, reason: str, warnings: list[str]) -> dict:
        self.session.clear_part()
        self.session.set_blocked(reason)
        return {"part_number": part_number, "can_inspect": False, "reason": reason, "warnings": warnings}

    def _load_part_model(self, info: Any) -> Any:
        """Build the runtime model for an approved version. StationError (incl. ModelError) if unusable."""
        from station.golden import cv2_decoder  # noqa: PLC0415 - needs cv2 and numpy at run time

        if info.type == "patchcore" and not _torch_importable():
            raise StationError(_NEEDS_TORCH)
        decode = cv2_decoder()
        if info.type == "patchcore":
            # PatchCore is trained on RGB pictures (trainer/weekly.py); the golden decoder gives grayscale
            from station.anomaly import cv2_rgb_decoder, register_patchcore_loader  # noqa: PLC0415 - lazy
            decode = cv2_rgb_decoder()
            register_patchcore_loader(self._make_extractor)
        model = load_runtime_model(info, decode)
        model.version = info.version  # the approved version name is what the records show
        return model

    def _make_extractor(self) -> Any:
        spec = self.config["model"].get("extractor")
        if spec is None:
            raise StationError("this part needs the anomaly model, but model.extractor is not set in the config")
        from station.features_torch import TorchExtractor  # noqa: PLC0415 - needs torch at run time
        return TorchExtractor(spec["arch"], device=spec["device"], input_hw=tuple(spec["input_hw"]),
                              weights_path=spec["weights_path"])

    def _require_mes_parts(self) -> None:
        if self._parts_mode != "mes":
            raise ValueError("this station is configured for a fixed part")

    def _require_parts_client(self) -> MesPartsClient:
        if self._parts_client is None:
            raise PartsError("MES token not set: part search is unavailable")
        return self._parts_client

    def _mes_cfg_for(self, part_number: str) -> MesConfig:
        """MES config for one part: its own criteria in 'mes' mode, the configured ones otherwise."""
        if self._parts_mode != "mes":
            return self._mes_cfg
        part = self._part_configs.get(part_number)
        if part is None:
            raise StationError(f"no MES criteria are loaded for part {part_number}")
        return replace(self._mes_cfg, criteria_ids=list(part["criteria_ids"]),
                       class_to_criterion=dict(part["class_to_criterion"]))

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._started:
            self.server.shutdown()
        self.server.server_close()
        for thread in (self._ui_thread, self._sync_thread):
            if thread is not None:
                thread.join(timeout=10)
        self.outbox.close()


def _serial_writer(port: str) -> Callable[[bytes], None]:
    """Open the relay board's serial device for writing. Baud rate and line settings are not configured."""
    handle = open(port, "wb", buffering=0)  # noqa: SIM115 - kept open for the station's lifetime

    def write(data: bytes) -> None:
        handle.write(data)

    return write


def build_station(
    config: Mapping[str, Any],
    *,
    cameras=None,
    model=None,
    outputs=None,
    now_utc: Optional[Callable[[], str]] = None,
    clock: Optional[Callable[[], float]] = None,
    token: Optional[str] = None,
    demo: bool = False,
) -> StationApp:
    """Build the station from a validated or raw config dict. Nothing is started yet.

    token: the MES bearer token. Default None reads the environment variable named
    by mes.token_env. An empty token disables MES posting (parts are still stored).
    cameras, model, outputs: inject to replace the ones the config would build.
    demo: parts are never queued for MES (the outbox would send them later).
    """
    return StationApp(config, cameras=cameras, model=model, outputs=outputs, now_utc=now_utc,
                      clock=clock, token=token, demo=demo)
