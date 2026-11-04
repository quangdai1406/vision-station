"""Password hashing, user store, role policy and in-memory session tokens.

Standard library only. Passwords are hashed with scrypt; the stored format is
    scrypt$<n>$<r>$<p>$<salt_hex>$<hash_hex>
"""
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

# Bounds applied when parsing a stored hash, so a corrupt or hostile record
# cannot ask for an enormous amount of work or memory.
_MAX_N = 2**17
_MAX_R = 8
_MAX_P = 4
_MIN_DIGEST_BYTES = 16
_MAX_DIGEST_BYTES = 64

# label_propose: anyone who inspects may propose a label (it is not official until an admin approves).
# label_approve: admin only; approves, rejects or sets an official label.
# scan: "Taking data" mode (start, press, finish, cancel a scan); supervisor and admin only.
# propose_defect: an inspector adds a human defect proposal (with a picture) to the part being shown.
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
    """Return a scrypt hash string for `password` using a fresh random salt."""
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = _derive(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P, _KEY_BYTES)
    return f"{_SCHEME}${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """True if `password` matches the `stored` hash. False for any malformed input."""
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


# Login modes. "local" is the username/password login; "mes_ticket" takes a one-time ticket
# from the MES and never accepts a local password.
AUTH_MODES = ("local", "mes_ticket")


def validate_role_map(role_map: Any) -> dict[str, str]:
    """Check an auth.role_map: MES role name -> station role. Raises ValueError."""
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
    """The station role for a user's MES roles, or None when no role is in the map.

    When several MES roles map to station roles, the most privileged one wins.
    """
    if not isinstance(mes_roles, (list, tuple)):
        return None
    mapped = [role_map[r] for r in mes_roles if isinstance(r, str) and r in role_map]
    if not mapped:
        return None
    return max(mapped, key=ROLES.index)


def role_allows(role: str, action: str) -> bool:
    """Policy check: may a user with `role` perform `action`?"""
    allowed = _POLICY.get(role)
    return allowed is not None and action in allowed


class UserStore:
    """Users keyed by username. Each entry: {'password_hash': str, 'role': str}."""

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
        # Verified against when the username is unknown, so both paths cost the same.
        self._dummy_hash = hash_password(secrets.token_urlsafe(16))

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Return the user's role if the credentials are valid, otherwise None."""
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
    """Load a UserStore from a JSON file that maps username -> entry object."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("users file must hold a JSON object")
    return UserStore(data)


class SessionManager:
    """Opaque tokens that expire after `timeout_s` without a `get`.

    A session may carry the MES session id it was made from (mes_ticket mode). When a
    session expires, `on_expire(token, username, mes_session)` is called once, outside the
    lock, so the caller can end the MES session. It is not called for destroy().
    """

    def __init__(self, timeout_s: float, clock: Callable[[], float] = time.monotonic,
                 on_expire: Optional[Callable[[str, str, Optional[str]], None]] = None):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._timeout = float(timeout_s)
        self._clock = clock
        self._on_expire = on_expire
        self._lock = threading.Lock()
        # token -> [username, role, last_seen, mes_session]
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
        """Return (username, role) and refresh the timer, or None if unknown or expired."""
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
        """The MES session id bound to a live token. Does not refresh the timer."""
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
