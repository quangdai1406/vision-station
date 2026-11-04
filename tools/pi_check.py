"""Raspberry Pi check: cameras, laptop round trip, evidence share, versions. Run on the Pi.

    python3 -m tools.pi_check [--max-index 5] [--no-cameras]
        [--inference-url http://LAPTOP:8765 --token-env NAME --part-number PN
         [--images 8] [--image PICTURE.png] [--timeout-s 30]]
        [--evidence-dest PATH] [--out pi_check_report.json]

It decides nothing and installs nothing. It reports what it measured: which camera indexes
work, how long the laptop takes to score pictures, whether the evidence share takes a file,
and the software versions. There are no pass/fail limits here; Quality owns the operator
wait limit, shown as [Quality to define].

Cameras: on Linux the OpenCV V4L2 backend (cv2.CAP_V4L2) is used, because Linux cameras are
V4L2 devices (DirectShow is Windows only). Each index 0..--max-index is opened, warmed up
and asked for one picture. Dead indexes are reported by number. cv2 is imported lazily and
can be injected (the tests do this).

Laptop: GET /v1/health, then --images predict calls (one per picture of a part: cameras C1..C4,
sides TOP and BOTTOM in turn). Pictures are synthetic PNGs unless --image gives a real one.
The token is read only from the environment variable named by --token-env and is never
printed, written or returned in an error. Requests ignore proxy environment variables, so the
LAN address is reached directly.

Exit codes: 0 when the report was written and every requested check ran; 1 when a requested
check failed (laptop unreachable or refused, a predict call failed, OpenCV missing for the
camera check, evidence share not writable) or the report could not be written; 2 when an
argument is wrong. Dead cameras are data, not an error.

JSON report, schema 1, written atomically to --out (no token, no secret).
"""
from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import platform
import shutil
import statistics
import struct
import sys
import tempfile
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from station.cameras import OPEN_WARMUP_FRAMES
from station.contracts import CAMERA_IDS

SCHEMA_VERSION = 1
TOOL_NAME = "tools.pi_check"
DEFAULT_MAX_INDEX = 5  # engineering setting: how many camera indexes are probed (0..N)
DEFAULT_IMAGES = 8  # one part: 4 cameras x 2 sides
DEFAULT_TIMEOUT_S = 30.0  # socket timeout for this tool's requests; NOT the RemoteModel timeout
DEFAULT_OUT = "pi_check_report.json"
OPERATOR_WAIT_LIMIT = "[Quality to define]"
OPERATOR_WAIT_LINE = f"Operator wait limit: {OPERATOR_WAIT_LIMIT}"
HEALTH_PATH = "/v1/health"
PREDICT_PATH = "/v1/predict"
PART_MAX = 128  # same limit as the laptop service (Part-Number header)
TOKEN_REDACTED = "[token redacted]"
SERVER_DETAIL_MAX = 120
SIDES = ("TOP", "BOTTOM")
SYNTHETIC_SIZE = 64
INSTALL_NOTE = "see docs/pi_setup.md for the OpenCV install step; this tool installs nothing"
CONTENT_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
# Requests must not pick up proxy settings from the environment (the laptop is on the LAN).
_OPENER = urlrequest.build_opener(urlrequest.ProxyHandler({}))
_TRANSPORT_ERRORS = (OSError, http.client.HTTPException)


class ReportWriteError(Exception):
    """The report could not be turned into JSON, so nothing was written (exit 1)."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _err_text(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def _try_import_cv2() -> Any:
    try:
        import cv2  # noqa: PLC0415 - lazy on purpose
    except ImportError:
        return None
    return cv2


# ---------------------------------------------------------------- secrets

def _scrub_text(text: str, token: str) -> str:
    return text.replace(token, TOKEN_REDACTED) if token else text


def _scrub_obj(obj: Any, token: str) -> Any:
    """Replace the token in every string of a report (defence in depth: no path should carry it)."""
    if not token:
        return obj
    if isinstance(obj, str):
        return _scrub_text(obj, token)
    if isinstance(obj, dict):
        return {k: _scrub_obj(v, token) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_obj(v, token) for v in obj]
    return obj


# ---------------------------------------------------------------- report file

def write_report_atomic(path: Path, report: dict) -> None:
    """Write the JSON report to a temporary file in the same folder, then replace the target.

    Raises ReportWriteError when json cannot encode the report (ValueError or TypeError).
    """
    try:
        data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    except (ValueError, TypeError) as exc:
        raise ReportWriteError(f"the report is not valid JSON, nothing was written ({_err_text(exc)})") from exc
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        with contextlib.suppress(OSError):
            os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


# ---------------------------------------------------------------- versions

def versions(cv2: Any) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": getattr(cv2, "__version__", None) if cv2 is not None else None,
    }


# ---------------------------------------------------------------- cameras

def _backend_for(cv2: Any, platform_name: str) -> tuple[Any, str, str]:
    """(api, backend name, plain-language reason). api None means OpenCV's default backend."""
    if platform_name.startswith("linux"):
        api = getattr(cv2, "CAP_V4L2", None)
        if api is not None:
            return api, "v4l2", (
                "Linux uses the OpenCV V4L2 backend (cv2.CAP_V4L2): Linux cameras are V4L2 "
                "devices. DirectShow is Windows only and is not used here."
            )
    return None, "default", (
        f"{platform_name}: OpenCV's default backend is used (this check is written for the Pi, "
        "which runs Linux)."
    )


def _warm_up(cap: Any, n: int) -> None:
    for _ in range(n):
        with contextlib.suppress(Exception):
            cap.grab()


def _probe_one(cv2: Any, index: int, api: Any) -> dict:
    result: dict = {"index": index, "ok": False, "width": None, "height": None,
                    "not_black": None, "reason": None, "release_error": None}
    try:
        cap = cv2.VideoCapture(index) if api is None else cv2.VideoCapture(index, api)
    except Exception as exc:
        result["reason"] = f"could not open ({_err_text(exc)})"
        return result
    try:
        if not cap.isOpened():
            result["reason"] = "does not open"
            return result
        try:
            _warm_up(cap, OPEN_WARMUP_FRAMES)
            if not cap.grab():
                result["reason"] = "opens but gives no frame"
                return result
            ok, image = cap.retrieve()
            if not ok or image is None:
                result["reason"] = "opens but gives no frame"
                return result
            result["height"], result["width"] = int(image.shape[0]), int(image.shape[1])
            result["not_black"] = bool(image.any())
        except Exception as exc:
            result["reason"] = f"read failed ({_err_text(exc)})"
            return result
        if not result["not_black"]:
            result["reason"] = "all-black picture"
            return result
        result["ok"] = True
        return result
    finally:
        try:
            cap.release()
        except Exception as exc:
            result["release_error"] = f"release failed ({_err_text(exc)})"


def probe_cameras(cv2: Any, *, max_index: int, platform_name: str, out: Callable[[str], Any]) -> dict:
    api, backend, reason = _backend_for(cv2, platform_name)
    out(f"Cameras: {reason}")
    out(f"Probing camera indexes 0..{max_index}")
    details = []
    for index in range(max_index + 1):
        result = _probe_one(cv2, index, api)
        details.append(result)
        if result["ok"]:
            out(f"index {index}: OK, {result['width']}x{result['height']}, picture not all black")
        else:
            out(f"index {index}: NOT WORKING: {result['reason']}")
        if result["release_error"]:
            out(f"index {index}: {result['release_error']}")
    working = [r["index"] for r in details if r["ok"]]
    not_working = [{"index": r["index"], "reason": r["reason"]} for r in details if not r["ok"]]
    if working:
        out("Working cameras: " + ", ".join(str(i) for i in working)
            + (". Not working: " + ", ".join(str(r["index"]) for r in not_working) + "."
               if not_working else "."))
    else:
        out("No working camera found at these indexes.")
    return {"backend": backend, "backend_reason": reason, "max_index": max_index,
            "working": working, "not_working": not_working, "details": details, "error": None}


def _cameras_without_opencv(max_index: int) -> dict:
    msg = f"OpenCV (cv2) is not installed, so the camera check could not run; {INSTALL_NOTE}"
    return {"backend": None, "backend_reason": None, "max_index": max_index, "working": [],
            "not_working": [], "details": [], "error": msg}


# ---------------------------------------------------------------- laptop

def _base_url(text: str) -> str:
    """http(s)://HOST:PORT with no user name, password, path, query or fragment."""
    parsed = urlparse.urlsplit(text)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("--inference-url must start with http:// or https:// and name a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("--inference-url must not contain a user name or password; the token "
                         "goes in the environment variable named by --token-env")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("--inference-url takes only scheme, host and port, for example http://192.168.0.10:8765")
    return f"{parsed.scheme}://{parsed.netloc}"


def _request(url: str, *, method: str, headers: dict, body: Optional[bytes], timeout: float) -> tuple[int, bytes]:
    """(HTTP status, body). HTTP error answers are returned, transport errors are raised."""
    req = urlrequest.Request(url, data=body, headers=headers, method=method)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urlerror.HTTPError as exc:
        try:
            data = exc.read()
        except Exception:  # noqa: BLE001 - the status is what matters
            data = b""
        return exc.code, data


def _server_detail(data: bytes) -> str:
    """The server's own short error text (the laptop never echoes the token)."""
    try:
        payload = json.loads(data.decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("error"), str):
            text = payload["error"]
        else:
            text = data.decode("utf-8", errors="replace")
    except ValueError:
        text = data.decode("utf-8", errors="replace")
    return text[:SERVER_DETAIL_MAX]


def check_health(base: str, token: str, timeout: float) -> dict:
    result: dict = {"ok": False, "http_status": None, "error": None, "version": None,
                    "models": None, "busy": None}
    try:
        status, data = _request(base + HEALTH_PATH, method="GET",
                                headers={"Authorization": f"Bearer {token}"}, body=None, timeout=timeout)
    except _TRANSPORT_ERRORS as exc:
        result["error"] = f"laptop unreachable at {base} ({_err_text(exc)})"
        return result
    result["http_status"] = status
    if status != 200:
        result["error"] = f"HTTP {status}: {_server_detail(data)}"
        return result
    try:
        payload = json.loads(data.decode("utf-8"))
    except ValueError:
        result["error"] = "the health answer is not JSON"
        return result
    if not isinstance(payload, dict):
        result["error"] = "the health answer is not a JSON object"
        return result
    result.update(ok=True, version=payload.get("version"),
                  models=len(payload.get("models") or []), busy=payload.get("busy"))
    return result


def synthetic_png(width: int = SYNTHETIC_SIZE, height: int = SYNTHETIC_SIZE) -> bytes:
    """A plain grey-gradient PNG, written with the standard library only."""
    if width < 1 or height < 1:
        raise ValueError("the synthetic picture needs a positive size")
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type: none
        for x in range(width):
            v = (x * 4 + y * 3) % 256
            rows += bytes((v, v, v))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b""))


def _predict_one(base: str, token: str, part: str, camera: str, side: str,
                 body: bytes, content_type: str, timeout: float, clock: Callable[[], float]) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Part-Number": part,
        "Camera-Id": camera,
        "Side": side,
        "Content-Type": content_type,
    }
    result: dict = {"camera_id": camera, "side": side, "seconds": None, "http_status": None,
                    "ok": False, "error": None, "score": None, "model_version": None}
    start = clock()
    try:
        status, data = _request(base + PREDICT_PATH, method="POST", headers=headers, body=body, timeout=timeout)
    except _TRANSPORT_ERRORS as exc:
        result["seconds"] = clock() - start
        result["error"] = f"laptop unreachable ({_err_text(exc)})"
        return result
    result["seconds"] = clock() - start
    result["http_status"] = status
    if status != 200:
        result["error"] = f"HTTP {status}: {_server_detail(data)}"
        return result
    try:
        payload = json.loads(data.decode("utf-8"))
        result["score"] = float(payload["score"])
        result["model_version"] = payload.get("model_version")
    except (ValueError, KeyError, TypeError):
        result["error"] = "the answer is not a score"
        return result
    result["ok"] = True
    return result


def timing_stats(seconds: list[float]) -> dict:
    if not seconds:
        return {"count_ok": 0, "min_s": None, "median_s": None, "max_s": None}
    return {"count_ok": len(seconds), "min_s": min(seconds),
            "median_s": statistics.median(seconds), "max_s": max(seconds)}


def _load_picture(path: Optional[str]) -> tuple[bytes, str, str]:
    """(bytes, content type, description). The synthetic PNG is used when no file is given."""
    if path is None:
        return synthetic_png(), "image/png", f"synthetic {SYNTHETIC_SIZE}x{SYNTHETIC_SIZE} PNG (no --image given)"
    p = Path(path)
    content_type = CONTENT_TYPES.get(p.suffix.lower())
    if content_type is None:
        raise ValueError("--image must be a .png, .jpg or .jpeg file")
    if not p.is_file():
        raise ValueError(f"--image file not found: {p}")
    return p.read_bytes(), content_type, str(p)


def check_laptop(base: str, token: str, token_env: str, part: str, n: int,
                 picture: tuple[bytes, str, str], timeout: float, clock: Callable[[], float],
                 out: Callable[[str], Any]) -> dict:
    body, content_type, description = picture
    out(f"Laptop inference service at {base}: token from environment variable {token_env} "
        "(value not shown)")
    health = check_health(base, token, timeout)
    inf: dict = {"url": base, "part_number": part, "pictures_requested": n, "picture": description,
                 "health": health, "images": [], "stats": timing_stats([]), "total_s": None}
    if not health["ok"]:
        out(f"Health: FAILED, {health['error']}. No picture was sent.")
        out(OPERATOR_WAIT_LINE)
        return inf
    out(f"Health: reachable, HTTP 200, service version {health['version']}, "
        f"models loaded {health['models']}, busy {str(health['busy']).lower()}")
    out(f"Predict: {n} picture(s) for part {part}, {description}")
    images = []
    for i in range(n):
        camera = CAMERA_IDS[i % len(CAMERA_IDS)]
        side = SIDES[(i // len(CAMERA_IDS)) % len(SIDES)]
        res = _predict_one(base, token, part, camera, side, body, content_type, timeout, clock)
        res["index"] = i + 1
        images.append(res)
        if res["ok"]:
            out(f"  {i + 1}/{n} {camera} {side}: {res['seconds']:.3f} s, HTTP 200, "
                f"model {res['model_version']}")
        else:
            out(f"  {i + 1}/{n} {camera} {side}: {res['seconds']:.3f} s, FAILED: {res['error']}")
    inf["images"] = images
    ok_times = [r["seconds"] for r in images if r["ok"]]
    inf["stats"] = timing_stats(ok_times)
    inf["total_s"] = sum(r["seconds"] for r in images)
    if ok_times:
        s = inf["stats"]
        out(f"Per picture ({s['count_ok']} OK of {n}): min {s['min_s']:.3f} s, "
            f"median {s['median_s']:.3f} s, max {s['max_s']:.3f} s")
    else:
        out(f"Per picture: n/a (no call of {n} succeeded)")
    out(f"Total for {n} pictures: {inf['total_s']:.3f} s (all calls, one after another)")
    out(OPERATOR_WAIT_LINE)
    return inf


# ---------------------------------------------------------------- evidence share

def check_evidence(dest: str) -> dict:
    result: dict = {"path": dest, "writable": False, "free_bytes": None, "error": None}
    folder = Path(dest)
    if not folder.is_dir():
        result["error"] = "not a directory: the share is not mounted, or the path is wrong"
        return result
    with contextlib.suppress(OSError):
        result["free_bytes"] = shutil.disk_usage(str(folder)).free
    try:
        fd, name = tempfile.mkstemp(dir=str(folder), prefix=".pi_check_", suffix=".tmp")
    except OSError as exc:
        result["error"] = f"not writable ({exc.strerror or exc})"
        return result
    try:
        try:
            fh = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with fh:
            fh.write(b"pi_check write test\n")
    except OSError as exc:
        result["error"] = f"write test failed ({exc.strerror or exc})"
    finally:
        try:
            os.unlink(name)
        except OSError as exc:
            if result["error"] is None:
                result["error"] = f"test file could not be removed ({exc.strerror or exc}); delete {name} by hand"
    if result["error"] is None:
        result["writable"] = True
    return result


# ---------------------------------------------------------------- run

def _validate(*, max_index: int, inference_url: Optional[str], token_env: Optional[str],
              part_number: Optional[str], images: int, image_path: Optional[str],
              timeout_s: float) -> Optional[str]:
    """Returns the normalised base URL, or None when no laptop check was asked for."""
    if max_index < 0:
        raise ValueError("--max-index must be 0 or more")
    if images < 1:
        raise ValueError("--images must be at least 1")
    if timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    if inference_url is None:
        if token_env or part_number or image_path:
            raise ValueError("--token-env, --part-number and --image need --inference-url")
        return None
    if not token_env:
        raise ValueError("--inference-url needs --token-env NAME (the name of the variable that holds the token)")
    if not part_number or not part_number.strip():
        raise ValueError("--inference-url needs --part-number")
    if len(part_number.strip()) > PART_MAX or any(c in part_number for c in "\r\n"):
        raise ValueError(f"--part-number must be 1 to {PART_MAX} characters on one line")
    return _base_url(inference_url)


def run_pi_check(
    *,
    max_index: int = DEFAULT_MAX_INDEX,
    cameras: bool = True,
    inference_url: Optional[str] = None,
    token_env: Optional[str] = None,
    part_number: Optional[str] = None,
    images: int = DEFAULT_IMAGES,
    image_path: Optional[str] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    evidence_dest: Optional[str] = None,
    out_path: Any = DEFAULT_OUT,
    cv2_module: Any = None,
    environ: Optional[dict] = None,
    platform_name: Optional[str] = None,
    now: Callable[[], str] = utc_now_iso,
    clock: Callable[[], float] = time.monotonic,
    out: Callable[[str], Any] = print,
) -> dict:
    """Run the requested checks, write the JSON report and return it.

    Raises ValueError for bad arguments (before anything runs), ReportWriteError when the
    report cannot be encoded and OSError when it cannot be written. Failed checks never raise;
    they are in the report.
    """
    base = _validate(max_index=max_index, inference_url=inference_url, token_env=token_env,
                     part_number=part_number, images=images, image_path=image_path, timeout_s=timeout_s)
    env = os.environ if environ is None else environ
    token = ""
    if base is not None:
        token = env.get(token_env, "")
        if not token:
            raise ValueError(f"the environment variable {token_env} is not set or empty "
                             "(its value is never printed)")
        picture = _load_picture(image_path)
    platform_name = sys.platform if platform_name is None else platform_name
    started = now()
    out_path = Path(out_path)

    def printable(text: Any) -> None:
        out(_scrub_text(str(text), token))

    cv2 = cv2_module if cv2_module is not None else _try_import_cv2()
    report: dict = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "started_utc": started,
        "finished_utc": None,
        "versions": versions(cv2),
        "cameras": None,
        "inference": None,
        "evidence": None,
        "operator_wait_limit": OPERATOR_WAIT_LIMIT,
    }
    v = report["versions"]
    printable(f"Versions: Python {v['python']}, {v['platform']}, "
              f"OpenCV {v['opencv'] or 'not installed'}.")

    if cameras:
        if cv2 is None:
            report["cameras"] = _cameras_without_opencv(max_index)
            printable(f"Cameras: {report['cameras']['error']}")
        else:
            report["cameras"] = probe_cameras(cv2, max_index=max_index, platform_name=platform_name,
                                              out=printable)

    if base is not None:
        report["inference"] = check_laptop(base, token, token_env, part_number.strip(), images,
                                           picture, timeout_s, clock, printable)

    if evidence_dest is not None:
        ev = check_evidence(evidence_dest)
        report["evidence"] = ev
        if ev["writable"]:
            gb = "" if ev["free_bytes"] is None else f", free space {ev['free_bytes'] / 1e9:.1f} GB"
            printable(f"Evidence share {evidence_dest}: writable{gb}")
        else:
            printable(f"Evidence share {evidence_dest}: NOT WRITABLE: {ev['error']}")

    report["finished_utc"] = now()
    report = _scrub_obj(report, token)
    write_report_atomic(out_path, report)
    return report


def _exit_code(report: dict) -> int:
    cams = report.get("cameras")
    if cams is not None and cams.get("error"):
        return 1
    inf = report.get("inference")
    if inf is not None:
        if not inf["health"]["ok"]:
            return 1
        if inf["stats"]["count_ok"] != len(inf["images"]):
            return 1
    ev = report.get("evidence")
    if ev is not None and not ev["writable"]:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m tools.pi_check",
        description="Check the Pi: camera indexes, the round trip to the laptop inference service, "
                    "the evidence share and versions. Decides nothing; writes a JSON report.",
    )
    p.add_argument("--max-index", type=int, default=DEFAULT_MAX_INDEX,
                   help=f"highest camera index probed, 0..N (default {DEFAULT_MAX_INDEX})")
    p.add_argument("--no-cameras", action="store_true", help="skip the camera check")
    p.add_argument("--inference-url", default=None,
                   help="laptop inference service, e.g. http://192.168.0.10:8765 (no user name or password)")
    p.add_argument("--token-env", default=None,
                   help="NAME of the environment variable that holds the token (the value is never printed)")
    p.add_argument("--part-number", default=None, help="part number sent with each picture")
    p.add_argument("--images", type=int, default=DEFAULT_IMAGES,
                   help=f"predict calls, one per picture of a part (default {DEFAULT_IMAGES})")
    p.add_argument("--image", default=None, help="a real .png or .jpg picture to send (default: synthetic PNG)")
    p.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S,
                   help="socket timeout for this tool's requests (default 30; not the station's RemoteModel timeout)")
    p.add_argument("--evidence-dest", default=None,
                   help="folder on the evidence share (SMB mount); a small test file is written and removed")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"JSON report path (default {DEFAULT_OUT})")
    return p


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    cv2_module: Any = None,
    environ: Optional[dict] = None,
    platform_name: Optional[str] = None,
    now: Callable[[], str] = utc_now_iso,
    clock: Callable[[], float] = time.monotonic,
    out: Callable[[str], Any] = print,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_pi_check(
            max_index=args.max_index,
            cameras=not args.no_cameras,
            inference_url=args.inference_url,
            token_env=args.token_env,
            part_number=args.part_number,
            images=args.images,
            image_path=args.image,
            timeout_s=args.timeout_s,
            evidence_dest=args.evidence_dest,
            out_path=args.out,
            cv2_module=cv2_module,
            environ=environ,
            platform_name=platform_name,
            now=now,
            clock=clock,
            out=out,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ReportWriteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: could not write the report to {args.out}: {exc.strerror or exc}", file=sys.stderr)
        return 1
    out(f"Report written to {Path(args.out).resolve()}")
    return _exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
