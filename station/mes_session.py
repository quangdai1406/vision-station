"""MES session client for the station: redeem a ticket, validate and end a session.

Standard library only. The MES bearer token goes only in the Authorization header. It is
never put in an error message, a repr or a log. Redirects are not followed, so the token
never reaches another host. Proxies are bypassed, as in station/parts.py.

API (machine routes on the MES):
  POST /api/machine/sessions/redeem      {ticket}  -> 200 {sessionId, username, roles, expiresAt}
                                                      409 when the ticket was already used
  GET  /api/machine/sessions/:id                   -> 200 {active, username, roles, expiresAt}
                                                      401 when the session is not valid
  POST /api/machine/sessions/:id/end               -> 200 (404 when already gone)
"""
from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from station.contracts import StationError

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_TICKET_CHARS = 256


class MesSessionError(StationError):
    """The MES could not redeem, validate or end a session. The text never holds the token or a ticket."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status  # HTTP status from the MES; 0 when no response was received


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    username: str
    roles: tuple[str, ...]
    expires_utc: str


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do not follow redirects: the token must not be sent to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


def _scrub(text: str, token: str) -> str:
    return text.replace(token, "<token>") if token else text


def _read_json(resp: Any) -> Any:
    try:
        return json.loads(resp.read().decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def _info_from(data: Any, session_id: Optional[str] = None) -> SessionInfo:
    if not isinstance(data, dict):
        raise MesSessionError("MES returned an unexpected session format")
    sid = session_id if session_id is not None else data.get("sessionId")
    username = data.get("username")
    roles = data.get("roles")
    expires = data.get("expiresAt")
    if (not isinstance(sid, str) or not sid or not isinstance(username, str) or not username
            or not isinstance(roles, list) or not all(isinstance(r, str) for r in roles)
            or not isinstance(expires, str)):
        raise MesSessionError("MES returned an unexpected session format")
    return SessionInfo(session_id=sid, username=username, roles=tuple(roles), expires_utc=expires)


class MesSessionClient:
    """Redeems inspector tickets and manages MES sessions for this station."""

    def __init__(self, base_url: str, token: str, *, timeout_s: float = 10.0) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise MesSessionError("MES base URL must start with http:// or https://")
        if not token or not isinstance(token, str):
            raise MesSessionError("no MES token supplied")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = float(timeout_s)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)

    def __repr__(self) -> str:
        return f"MesSessionClient(base_url={self._base!r})"

    def redeem(self, ticket: str) -> SessionInfo:
        """Exchange a one-time ticket for a session. A second use raises with status 409."""
        if not isinstance(ticket, str) or not ticket or len(ticket) > _MAX_TICKET_CHARS:
            raise ValueError("ticket must be a non-empty string")
        body = json.dumps({"ticket": ticket}).encode("utf-8")
        status, data = self._call("POST", "/api/machine/sessions/redeem", body=body,
                                  what="ticket redeem")
        if status != 200:
            raise MesSessionError(self._refusal("ticket redeem", status), status=status)
        return _info_from(data)

    def validate(self, session_id: str) -> Optional[SessionInfo]:
        """The session if the MES still holds it active, otherwise None (ended or unknown)."""
        sid = self._checked_id(session_id)
        status, data = self._call("GET", f"/api/machine/sessions/{sid}", what="session check")
        if status in (401, 404):
            return None
        if status != 200:
            raise MesSessionError(self._refusal("session check", status), status=status)
        if not isinstance(data, dict):
            raise MesSessionError("MES returned an unexpected session format")
        if data.get("active") is not True:
            return None
        return _info_from(data, session_id=sid)

    def end(self, session_id: str) -> None:
        """End the MES session. An already-gone session (404) is not an error."""
        sid = self._checked_id(session_id)
        status, _ = self._call("POST", f"/api/machine/sessions/{sid}/end", what="session end")
        if status in (200, 204, 404):
            return
        raise MesSessionError(self._refusal("session end", status), status=status)

    # ----- internals -----------------------------------------------------

    @staticmethod
    def _checked_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not _SESSION_ID_RE.match(session_id):
            raise ValueError("session id is not a valid MES session id")
        return session_id

    def _call(self, method: str, path: str, *, what: str,
              body: Optional[bytes] = None) -> tuple[int, Any]:
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self._base + path, data=body, method=method, headers=headers)
        try:
            with self._opener.open(req, timeout=self._timeout) as resp:
                status = resp.status
                data = _read_json(resp)
        except urllib.error.HTTPError as exc:
            status = exc.code
            data = None
            exc.close()
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", None) or type(exc).__name__
            raise MesSessionError(
                _scrub(f"could not reach MES for {what}: {type(reason).__name__}", self._token)
            ) from None
        return status, data

    def _refusal(self, what: str, status: int) -> str:
        if status == 409:
            return f"MES refused the {what}: the ticket was already used or has expired"
        if status in (401, 403):
            return (f"MES refused the {what} (HTTP {status}); check the machine token "
                    "and its role")
        if status in (408, 429) or status >= 500:
            return f"MES temporarily unavailable for {what} (HTTP {status})"
        return f"MES answered {status} for the {what}"
