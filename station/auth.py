from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from typing import Any, Callable, Mapping, Optional

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
_SALT_BYTES = 16
_KEY_BYTES = 32
_SCHEME = "scrypt"

_MAX_N = 2**17
_MAX_R = 8
_MAX_P = 4
_MIN_DIGEST_BYTES = 16
_MAX_DIGEST_BYTES = 64

ACTIONS = ("press", "ack", "resolve", "override", "clear_fault", "select_part",
           "label_propose", "label_approve", "scan", "propose_defect")
ROLES = ("operator", "supervisor", "admin")

_POLICY: dict[str, frozenset[str]] = {
    "operator": frozenset({"press", "ack", "select_part", "label_propose", "propose_defect"}),
    "supervisor": frozenset({"press", "ack", "resolve", "override", "clear_fault", "select_part",
                             "label_propose", "scan", "propose_defect"}),
    "admin": frozenset(ACTIONS),
}


def _derive(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    maxmem = 128 * r * (n + p + 2) + 1024 * 1024
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        maxmem=maxmem,
        dklen=dklen,
    )


def hash_password(password: str) -> str:
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = _derive(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P, _KEY_BYTES)
    return f"{_SCHEME}${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    if not isinstance(password, str) or not isinstance(stored, str):
        return False
    try:
        parts = stored.split("$")
        if len(parts) != 6 or parts[0] != _SCHEME:
            return False
        if not all(part.isdigit() for part in parts[1:4]):
            return False
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        if n < 2 or n > _MAX_N or n & (n - 1):
            return False
        if not (1 <= r <= _MAX_R) or not (1 <= p <= _MAX_P):
            return False
        salt = bytes.fromhex(parts[4])
        expected = bytes.fromhex(parts[5])
        if not salt or not (_MIN_DIGEST_BYTES <= len(expected) <= _MAX_DIGEST_BYTES):
            return False
        actual = _derive(password, salt, n, r, p, len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


AUTH_MODES = ("local", "mes_ticket")


def validate_role_map(role_map: Any) -> dict[str, str]:
    if not isinstance(role_map, Mapping) or not role_map:
        raise ValueError("role_map must be a non-empty object")
    out: dict[str, str] = {}
    for mes_role, station_role in role_map.items():
        if not isinstance(mes_role, str) or not mes_role.strip():
            raise ValueError("role_map keys must be non-empty MES role names")
        if station_role not in ROLES:
            raise ValueError(f"role_map[{mes_role!r}] must be one of {', '.join(ROLES)}")
        out[mes_role] = station_role
    return out


def map_role(role_map: Mapping[str, str], mes_roles: Any) -> Optional[str]:
    if not isinstance(mes_roles, (list, tuple)):
        return None
    mapped = [role_map[r] for r in mes_roles if isinstance(r, str) and r in role_map]
    if not mapped:
        return None
    return max(mapped, key=ROLES.index)


def role_allows(role: str, action: str) -> bool:
    allowed = _POLICY.get(role)
    return allowed is not None and action in allowed


class UserStore:
    def __init__(self, users: dict[str, dict]):
        self._users: dict[str, tuple[str, str]] = {}
        for username, entry in users.items():
            if not isinstance(entry, dict):
                raise ValueError(f"user entry for {username!r} must be an object")
            role = entry.get("role")
            if role not in ROLES:
                raise ValueError(f"bad role for {username!r}")
            password_hash = entry.get("password_hash")
            if not isinstance(password_hash, str):
                raise ValueError(f"missing password_hash for {username!r}")
            self._users[username] = (password_hash, role)
        self._dummy_hash = hash_password(secrets.token_urlsafe(16))

    def authenticate(self, username: str, password: str) -> Optional[str]:
        entry = self._users.get(username) if isinstance(username, str) else None
        if entry is None:
            verify_password(password, self._dummy_hash)
            return None
        stored, role = entry
        return role if verify_password(password, stored) else None

    @staticmethod
    def role_allows(role: str, action: str) -> bool:
        return role_allows(role, action)


def load_users(path: str) -> UserStore:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("users file must hold a JSON object")
    return UserStore(data)


class SessionManager:
    def __init__(self, timeout_s: float, clock: Callable[[], float] = time.monotonic,
                 on_expire: Optional[Callable[[str, str, Optional[str]], None]] = None):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._timeout = float(timeout_s)
        self._clock = clock
        self._on_expire = on_expire
        self._lock = threading.Lock()
        self._sessions: dict[str, list] = {}

    def create(self, username: str, role: str, mes_session: Optional[str] = None) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = self._clock()
            expired = self._prune_locked(now)
            self._sessions[token] = [username, role, now, mes_session]
        self._notify(expired)
        return token

    def get(self, token: str) -> Optional[tuple[str, str]]:
        if not isinstance(token, str) or not token:
            return None
        expired = []
        with self._lock:
            entry = self._sessions.get(token)
            if entry is None:
                return None
            now = self._clock()
            if now - entry[2] >= self._timeout:
                del self._sessions[token]
                expired.append((token, entry[0], entry[3]))
                result = None
            else:
                entry[2] = now
                result = (entry[0], entry[1])
        self._notify(expired)
        return result

    def mes_session(self, token: str) -> Optional[str]:
        if not isinstance(token, str) or not token:
            return None
        with self._lock:
            entry = self._sessions.get(token)
            return entry[3] if entry is not None else None

    def destroy(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)

    def _prune_locked(self, now: float) -> list:
        expired = []
        for token, entry in list(self._sessions.items()):
            if now - entry[2] >= self._timeout:
                del self._sessions[token]
                expired.append((token, entry[0], entry[3]))
        return expired

    def _notify(self, expired: list) -> None:
        if self._on_expire is None:
            return
        for token, username, mes_session in expired:
            if mes_session is not None:
                self._on_expire(token, username, mes_session)
