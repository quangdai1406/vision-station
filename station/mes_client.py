"""MES client for the camera inspection station.

Builds the inspection-event post for one finished part and sends it to
POST /api/machine/inspection-events on the MES. Standard library only.

API reference: /home/user/aisi_mes/docs/vision-station-integration.md
"""
from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from station.contracts import Frame, ImageResult, PartRecord, StationError, Verdict

EVENTS_PATH = "/api/machine/inspection-events"


class MappingError(StationError):
    """A defect or a failed criterion cannot be mapped to an MES criterion id."""


@dataclass
class MesConfig:
    base_url: str  # e.g. http://mes:3000
    token: str = field(repr=False)  # never printed
    criteria_ids: list[str]  # all MES inspection_criteria ids for the part
    class_to_criterion: dict[str, str]  # defect class -> criterion id
    default_criterion_id: Optional[str] = None
    default_bad_disposition: str = "quarantine"
    timeout_s: float = 10.0


@dataclass(frozen=True)
class PostResult:
    status: int  # HTTP status; 0 when no response was received
    ok: bool
    replay: bool  # True when MES answered 200 for a key it already stored
    event_id: Optional[int]
    error: Optional[str]
    retryable: bool


def _side_value(side: Any) -> str:
    return str(getattr(side, "value", side))


def _check_criterion(criterion_id: str, cfg: MesConfig) -> str:
    if criterion_id not in cfg.criteria_ids:
        raise MappingError(f"criterion id {criterion_id!r} is not one of the part's criteria")
    return criterion_id


def criterion_for(defect_class: str, cfg: MesConfig) -> str:
    """Map a defect class to its criterion id, falling back to the default."""
    criterion_id = cfg.class_to_criterion.get(defect_class, cfg.default_criterion_id)
    if criterion_id is None:
        raise MappingError(
            f"defect class {defect_class!r} has no criterion mapping and no default_criterion_id"
        )
    return _check_criterion(criterion_id, cfg)


def build_request(
    record: PartRecord, cfg: MesConfig, archive_ref: str
) -> tuple[dict[str, str], list[tuple[str, str, bytes]]]:
    """Return (fields, files) for one inspection event.

    files is a list of (field_name, filename, data): one defect photo per failed
    criterion, named defectPhoto_<criterion_id>. No other images are included.
    """
    if record.final_verdict not in (Verdict.GOOD, Verdict.BAD):
        raise ValueError(f"only GOOD or BAD can be posted, got {record.final_verdict!r}")
    if record.override is not None and record.override.new_verdict is not record.final_verdict:
        raise ValueError("override.new_verdict must equal final_verdict")
    is_bad = record.final_verdict is Verdict.BAD

    frame_by_key: dict[tuple[str, Any], Frame] = {(f.camera_id, f.side): f for f in record.frames}
    defect_classes: set[str] = set()
    failed: set[str] = set()
    # criterion id -> (highest defect score mapped to it, the image result it came from)
    best: dict[str, tuple[float, ImageResult]] = {}

    if is_bad:
        for result in record.results:
            for defect in result.defects:
                criterion_id = criterion_for(defect.defect_class, cfg)
                defect_classes.add(defect.defect_class)
                failed.add(criterion_id)
                if criterion_id not in best or defect.score > best[criterion_id][0]:
                    best[criterion_id] = (defect.score, result)
        if not failed:
            # e.g. a supervisor resolved a REVIEW to BAD with no defects recorded
            if cfg.default_criterion_id is None:
                raise MappingError("BAD part has no defects and no default_criterion_id is set")
            failed.add(_check_criterion(cfg.default_criterion_id, cfg))

    checklist = {cid: ("fail" if cid in failed else "pass") for cid in cfg.criteria_ids}

    photos: list[tuple[str, str, bytes]] = []
    for criterion_id in sorted(failed):
        if criterion_id in best:
            result = best[criterion_id][1]
        else:
            if not record.results:
                raise MappingError("no image results to choose a defect photo from")
            result = max(record.results, key=lambda r: r.score)
        frame = frame_by_key.get((result.camera_id, result.side))
        if frame is None:
            raise MappingError(
                f"no frame for {result.camera_id} {_side_value(result.side)} "
                f"to attach as photo for {criterion_id}"
            )
        photos.append((
            f"defectPhoto_{criterion_id}",
            f"{frame.camera_id}_{_side_value(frame.side)}.{frame.ext}",
            bytes(frame.data),
        ))

    fields: dict[str, str] = {
        "stationId": record.station_id,
        "partNumber": record.part_number,
        "quantityInspected": "1",
        "checklistResults": json.dumps(checklist, sort_keys=True),
        "supervisor": record.supervisor,
        "operators": record.operator,
        "materialHandler": record.material_handler,
        "verdict": record.final_verdict.value,
        "visionSessionId": record.session_id,
        "modelVersion": record.model_version,
        "archiveRef": archive_ref,
        "notes": (
            f"session {record.session_id}; model {record.model_version}; "
            f"machine verdict {record.machine_verdict.value}"
        ),
    }
    if is_bad:
        fields["rejectedQuantity"] = "1"
        fields["rejectionReason"] = ", ".join(sorted(defect_classes)) or "supervisor decision"
        fields["disposition"] = cfg.default_bad_disposition
    elif record.override is not None and record.override.new_verdict is Verdict.GOOD:
        fields["disposition"] = "released"
    if record.override is not None:
        fields["overrideOriginalVerdict"] = record.override.original_verdict.value
        fields["overrideReason"] = record.override.reason
        fields["overrideBy"] = record.override.by

    return fields, photos


def _q(text: str) -> str:
    """Make a value safe inside a quoted multipart header parameter."""
    return text.replace('"', "_").replace("\r", "_").replace("\n", "_")


def _encode_multipart(
    fields: Mapping[str, str], files: Sequence[tuple[str, str, bytes]], boundary: str
) -> bytes:
    out = bytearray()
    for name, value in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{_q(name)}"\r\n\r\n'.encode()
        out += str(value).encode("utf-8") + b"\r\n"
    for field_name, filename, data in files:
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        out += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{_q(field_name)}"; '
            f'filename="{_q(filename)}"\r\nContent-Type: {ctype}\r\n\r\n'
        ).encode("utf-8")
        out += bytes(data) + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not follow redirects: the bearer token must not follow a redirect elsewhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _read_quietly(resp: Any) -> bytes:
    try:
        return resp.read()
    except Exception:
        return b""


def _json_body(raw: bytes) -> dict:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


class MesClient:
    def __init__(self, cfg: MesConfig):
        self.cfg = cfg
        self._opener = urllib.request.build_opener(_NoRedirect())

    def post(
        self,
        fields: Mapping[str, str],
        files: Sequence[tuple[str, str, bytes]],
        idempotency_key: str,
    ) -> PostResult:
        boundary = uuid.uuid4().hex
        body = _encode_multipart(fields, files, boundary)
        req = urllib.request.Request(
            self.cfg.base_url.rstrip("/") + EVENTS_PATH,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.cfg.token}",
                "Idempotency-Key": idempotency_key,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
        )
        try:
            with self._opener.open(req, timeout=self.cfg.timeout_s) as resp:
                status = resp.status
                raw = resp.read()
        except urllib.error.HTTPError as e:
            status = e.code
            raw = _read_quietly(e)
        except (urllib.error.URLError, OSError) as e:  # connection refused, DNS, timeouts
            reason = getattr(e, "reason", e)
            return PostResult(
                status=0, ok=False, replay=False, event_id=None,
                error=self._redact(f"connection to MES failed: {type(reason).__name__}"),
                retryable=True,
            )
        return self._result(status, raw)

    def _redact(self, text: str) -> str:
        return text.replace(self.cfg.token, "[redacted]") if self.cfg.token else text

    def _result(self, status: int, raw: bytes) -> PostResult:
        body = _json_body(raw)
        raw_id = body.get("id")
        event_id = raw_id if isinstance(raw_id, int) and not isinstance(raw_id, bool) else None
        server_msg = body.get("error") if isinstance(body.get("error"), str) else None
        detail = self._redact(server_msg or "no detail")[:300]

        if status == 201:
            return PostResult(status, True, False, event_id, None, False)
        if status == 200:
            return PostResult(status, True, True, event_id, None, False)
        if status in (401, 403):
            return PostResult(
                status, False, False, None,
                self._redact(
                    f"MES refused the station token (HTTP {status}); check the machine token "
                    f"and its role: {detail}"
                ),
                True,
            )
        if status in (400, 404, 422):
            return PostResult(
                status, False, False, None,
                f"MES rejected this inspection (HTTP {status}): {detail}",
                False,
            )
        if status in (408, 429) or status >= 500:
            return PostResult(
                status, False, False, None,
                f"MES temporarily unavailable (HTTP {status}): {detail}",
                True,
            )
        return PostResult(
            status, False, False, None,
            f"unexpected MES response (HTTP {status}): {detail}",
            False,
        )
