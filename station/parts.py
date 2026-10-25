"""Client for the MES part search and per-part inspection criteria.

Standard library only. The MES bearer token goes only in the Authorization header and is
never put in an error message or a file.

Proxies: as in tools/criteria.py, environment proxies are bypassed by default so the token
goes straight to the MES. Pass use_env_proxy=True only when the MES is reachable only
through a corporate proxy (a token over plain http is then visible to that proxy).

Redirects are not followed, so the token is never sent to another host.

build_part_config() mirrors tools/criteria.py (same slug and collision rules). The functions
are copied here on purpose: station code must not import tools/.
"""
from __future__ import annotations

import http.client
import json
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any

from station.contracts import StationError

CHECK_TYPES = ("visual", "measurement")
SEARCH_LIMIT_MIN = 1
SEARCH_LIMIT_MAX = 50
QUERY_MAX_LENGTH = 100
_MISSING = object()  # sentinel: the MES answered 404 for a lookup that may be absent


class PartsError(StationError):
    """The MES could not supply part data. The message never contains the token."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not follow redirects: the token must not be sent to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _scrub(text: str, token: str) -> str:
    return text.replace(token, "<token>") if token else text


def _http_message(code: int, what: str) -> str:
    if code == 401:
        return ("MES rejected the token (401 Unauthorized). Check the token in the environment "
                "variable and that the machine token has not been revoked.")
    if code == 403:
        return (f"MES refused access to {what} (403 Forbidden). "
                "The station user may lack permission.")
    if code == 404:
        return f"MES has no {what} (404 Not Found)."
    if 300 <= code < 400:
        return (f"MES answered with a redirect ({code}); redirects are not followed. "
                "Check the MES base URL.")
    if code >= 500:
        return f"MES server error ({code}) while fetching {what}. Try again later."
    return f"MES answered {code} while fetching {what}."


def _part_text(part_number: Any) -> str:
    if not isinstance(part_number, str) or not part_number.strip():
        raise ValueError("part number must be a non-empty string")
    return part_number.strip()


def _expect_list_of_dicts(data: Any, what: str) -> list[dict]:
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise PartsError(f"MES returned an unexpected format for {what}")
    return data


class MesPartsClient:
    """Read-only client for the MES part search, part lookup and criteria endpoints."""

    def __init__(self, base_url: str, token: str, *, timeout_s: float = 10.0,
                 use_env_proxy: bool = False) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise PartsError("MES base URL must start with http:// or https://")
        if not token or not isinstance(token, str):
            raise PartsError("no MES token supplied")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = float(timeout_s)
        if use_env_proxy:
            self._opener = urllib.request.build_opener(_NoRedirect)
        else:
            self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)

    def _get_json(self, path: str, *, what: str, missing_ok: bool = False) -> Any:
        req = urllib.request.Request(
            self._base + path,
            headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
            method="GET",
        )
        try:
            with self._opener.open(req, timeout=self._timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()  # release the connection; the body is not used
            if code == 404 and missing_ok:
                return _MISSING
            raise PartsError(_scrub(_http_message(code, what), self._token)) from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", None) or type(exc).__name__
            raise PartsError(
                _scrub(f"could not reach MES at {self._base}: {reason}", self._token)
            ) from None
        except ValueError:
            raise PartsError("invalid MES request (check the base URL and token)") from None
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise PartsError(f"MES returned something other than JSON for {what}") from None

    def search(self, q: str, limit: int = 20) -> list[dict]:
        """Parts matching q. limit is clamped to 1..50. ValueError for an empty or over-long q."""
        if not isinstance(q, str):
            raise ValueError("search text must be a string")
        text = q.strip()
        if not text:
            raise ValueError("search text is empty")
        if len(text) > QUERY_MAX_LENGTH:
            raise ValueError(f"search text is longer than {QUERY_MAX_LENGTH} characters")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer")
        n = min(max(limit, SEARCH_LIMIT_MIN), SEARCH_LIMIT_MAX)
        path = f"/api/machine/parts/search?q={urllib.parse.quote(text, safe='')}&limit={n}"
        data = self._get_json(path, what="part search")
        return _expect_list_of_dicts(data, "part search")

    def get_part(self, part_number: str) -> dict | None:
        """One part, or None when the MES answers 404."""
        part = _part_text(part_number)
        path = "/api/machine/parts/" + urllib.parse.quote(part, safe="")
        data = self._get_json(path, what=f"part {part}", missing_ok=True)
        if data is _MISSING:
            return None
        if not isinstance(data, dict):
            raise PartsError(f"MES returned an unexpected format for part {part}")
        return data

    def criteria(self, part_number: str) -> list[dict]:
        """Inspection criteria for a part. A 404 raises PartsError (no criteria for that part)."""
        part = _part_text(part_number)
        path = "/api/machine/criteria/" + urllib.parse.quote(part, safe="")
        data = self._get_json(path, what=f"criteria for part {part}")
        return _expect_list_of_dicts(data, f"criteria for part {part}")


# ---------------------------------------------------------------------------
# Criteria -> station config (mirrors tools/criteria.py)
# ---------------------------------------------------------------------------

def _validate(data: Any, source: str) -> list[dict]:
    if not isinstance(data, list):
        raise PartsError(f"{source}: expected a list of criteria, got {type(data).__name__}")
    out = []
    seen = set()
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise PartsError(f"{source}: entry {i} is not an object")
        cid = item.get("id")
        if isinstance(cid, bool) or not isinstance(cid, (int, str)) or not str(cid).strip():
            raise PartsError(f"{source}: entry {i} has no usable 'id'")
        key = str(cid).strip()
        where = f"{source}: criterion {key}"
        if key in seen:
            raise PartsError(f"{where} appears more than once")
        seen.add(key)
        name = item.get("criteria_name")
        if not isinstance(name, str) or not name.strip():
            raise PartsError(f"{where} has no 'criteria_name'")
        ctype = item.get("check_type")
        if not isinstance(ctype, str) or ctype not in CHECK_TYPES:
            raise PartsError(f"{where} has check_type {ctype!r}; expected 'visual' or 'measurement'")
        active = item.get("active")
        if isinstance(active, int) and not isinstance(active, bool) and active in (0, 1):
            active = bool(active)
        if not isinstance(active, bool):
            raise PartsError(f"{where} has no boolean 'active'")
        entry = dict(item)
        entry["criteria_description"] = item.get("criteria_description") or ""
        entry["id"] = cid
        entry["criteria_name"] = name
        entry["check_type"] = ctype
        entry["active"] = active
        out.append(entry)
    return out


def _key(c: dict) -> str:
    return str(c["id"]).strip()


def _safe_id(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_") or "x"


def slugify(name: str) -> str:
    """Lowercase ASCII snake_case from a criteria name. Raises ValueError if nothing is left."""
    if not isinstance(name, str):
        raise ValueError("criteria name must be a string")
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", folded.lower()).strip("_")
    if not slug:
        raise ValueError(f"criteria name {name!r} has no letters or digits")
    return slug


def derive_class_names(criteria: list[dict]) -> dict[str, str]:
    """Defect-class name for every criterion id. Colliding names get '_<id>' on every member."""
    criteria = _validate(criteria, "criteria")
    base: dict[str, str | None] = {}
    for c in criteria:
        try:
            base[_key(c)] = slugify(c["criteria_name"])
        except ValueError:
            base[_key(c)] = None
    counts = Counter(v for v in base.values() if v)
    names: dict[str, str] = {}
    for c in criteria:
        key = _key(c)
        slug = base[key]
        if slug is None:
            names[key] = f"criterion_{_safe_id(key)}"
        elif counts[slug] > 1:
            names[key] = f"{slug}_{_safe_id(key)}"
        else:
            names[key] = slug
    if len(set(names.values())) != len(names):
        raise PartsError("could not derive unique defect class names for these criteria")
    return names


def split_by_type(criteria: list[dict]) -> tuple[list[dict], list[dict]]:
    """(active visual, active measurement). Inactive criteria are in neither list."""
    criteria = _validate(criteria, "criteria")
    visual = [c for c in criteria if c["active"] and c["check_type"] == "visual"]
    measurement = [c for c in criteria if c["active"] and c["check_type"] == "measurement"]
    return visual, measurement


def build_part_config(criteria: list[dict]) -> dict[str, Any]:
    """Station config pieces for one part, derived from the MES criteria.

    criteria_ids: every active criterion id (the MES needs a result for each).
    class_to_criterion: active visual criteria only, {defect class: criterion id}.
    measurement_criteria: active measurement criteria, which the camera does not check.
    warnings: things Quality or the station owner must decide.
    """
    entries = _validate(criteria, "criteria")
    names = derive_class_names(entries)
    visual, measurement = split_by_type(entries)
    warnings = []
    for c in measurement:
        warnings.append(
            f"measurement criterion {_key(c)} '{c['criteria_name']}' has no camera defect class: "
            "the camera classifier does not check it, and the station would post it as 'pass' "
            "unless it is handled (manual check or a measurement method)"
        )
    if not visual:
        warnings.append("no active visual criteria: the model has no defect classes to output")
    return {
        "criteria_ids": [_key(c) for c in entries if c["active"]],
        "class_to_criterion": {names[_key(c)]: _key(c) for c in visual},
        "measurement_criteria": [{"id": _key(c), "name": c["criteria_name"]} for c in measurement],
        "warnings": warnings,
        "inactive_count": sum(1 for c in entries if not c["active"]),
    }
