"""Unit tests for station.auth: hashing, role policy, user store, sessions."""
import json
import os
import tempfile
import unittest

from station.auth import (
    ACTIONS,
    SessionManager,
    UserStore,
    hash_password,
    load_users,
    role_allows,
    verify_password,
)

# Throwaway test credentials only.
PW = "test-only-pass-1"


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _flip(hex_text: str) -> str:
    first = "1" if hex_text[0] == "0" else "0"
    return first + hex_text[1:]


class HashTests(unittest.TestCase):
    def test_round_trip(self):
        stored = hash_password(PW)
        self.assertTrue(verify_password(PW, stored))
        self.assertFalse(verify_password(PW + "x", stored))
        self.assertFalse(verify_password("", stored))

    def test_format(self):
        parts = hash_password(PW).split("$")
        self.assertEqual(len(parts), 6)
        self.assertEqual(parts[0], "scrypt")
        self.assertEqual(parts[1:4], ["16384", "8", "1"])
        self.assertEqual(len(bytes.fromhex(parts[4])), 16)
        self.assertEqual(len(bytes.fromhex(parts[5])), 32)

    def test_salt_is_random(self):
        self.assertNotEqual(hash_password(PW), hash_password(PW))

    def test_tampered_hash_fails(self):
        parts = hash_password(PW).split("$")
        tampered = "$".join(parts[:5] + [_flip(parts[5])])
        self.assertFalse(verify_password(PW, tampered))

    def test_tampered_salt_fails(self):
        parts = hash_password(PW).split("$")
        tampered = "$".join(parts[:4] + [_flip(parts[4]), parts[5]])
        self.assertFalse(verify_password(PW, tampered))

    def test_tampered_params_fail(self):
        parts = hash_password(PW).split("$")
        tampered = "$".join([parts[0], "8192"] + parts[2:])
        self.assertFalse(verify_password(PW, tampered))

    def test_malformed_input_is_false(self):
        good_digest = "00" * 32
        salt = "00" * 16
        cases = [
            "",
            "garbage",
            "bcrypt$16384$8$1$00$00",
            "scrypt$abc$8$1$00$00",
            "scrypt$16384$8$1$$" + good_digest,
            "scrypt$3$8$1$" + salt + "$" + good_digest,  # n not a power of two
            "scrypt$16384$8$1$zz$" + good_digest,  # not hex
            "scrypt$16384$8$1$" + salt,  # too few fields
            "scrypt$16384$8$1$" + salt + "$" + "00" * 4,  # digest too short
        ]
        for stored in cases:
            with self.subTest(stored=stored):
                self.assertFalse(verify_password(PW, stored))
        self.assertFalse(verify_password(None, hash_password(PW)))
        self.assertFalse(verify_password(PW, None))

    def test_empty_password_cannot_be_hashed(self):
        with self.assertRaises(ValueError):
            hash_password("")


class RolePolicyTests(unittest.TestCase):
    def test_policy_table(self):
        table = {
            "operator": {"press": True, "ack": True, "resolve": False,
                         "override": False, "clear_fault": False, "select_part": True,
                         "label_propose": True, "label_approve": False, "scan": False},
            "supervisor": {"press": True, "ack": True, "resolve": True,
                           "override": True, "clear_fault": True, "select_part": True,
                           "label_propose": True, "label_approve": False, "scan": True},
            "admin": {a: True for a in ("press", "ack", "resolve", "override", "clear_fault",
                                        "select_part", "label_propose", "label_approve", "scan")},
        }
        for role, row in table.items():
            for action, expected in row.items():
                with self.subTest(role=role, action=action):
                    self.assertEqual(role_allows(role, action), expected)

    def test_unknown_role_or_action_denied(self):
        self.assertFalse(role_allows("visitor", "press"))
        self.assertFalse(role_allows("admin", "delete_everything"))
        self.assertEqual(set(ACTIONS),
                         {"press", "ack", "resolve", "override", "clear_fault", "select_part",
                          "label_propose", "label_approve", "scan"})
        self.assertFalse(role_allows("operator", "scan"))  # Taking data is not for operators
        self.assertFalse(role_allows("operator", "label"))  # the old single 'label' action is gone


class UserStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = UserStore({
            "alice": {"password_hash": hash_password(PW), "role": "operator"},
            "bob": {"password_hash": hash_password(PW), "role": "supervisor"},
        })

    def test_authenticate_ok_returns_role(self):
        self.assertEqual(self.store.authenticate("alice", PW), "operator")
        self.assertEqual(self.store.authenticate("bob", PW), "supervisor")

    def test_wrong_password_and_unknown_user_return_none(self):
        self.assertIsNone(self.store.authenticate("alice", "nope"))
        self.assertIsNone(self.store.authenticate("nobody", PW))
        self.assertIsNone(self.store.authenticate(None, PW))

    def test_bad_role_rejected(self):
        with self.assertRaises(ValueError):
            UserStore({"eve": {"password_hash": hash_password(PW), "role": "root"}})

    def test_load_users_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "users.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"carol": {"password_hash": hash_password(PW), "role": "admin"}}, fh)
            store = load_users(path)
        self.assertEqual(store.authenticate("carol", PW), "admin")
        self.assertIsNone(store.authenticate("carol", "wrong"))


class SessionManagerTests(unittest.TestCase):
    def test_create_get_destroy(self):
        clock = FakeClock(0.0)
        sm = SessionManager(60, clock=clock)
        token = sm.create("alice", "operator")
        self.assertIsInstance(token, str)
        self.assertGreaterEqual(len(token), 32)
        self.assertEqual(sm.get(token), ("alice", "operator"))
        sm.destroy(token)
        self.assertIsNone(sm.get(token))

    def test_unknown_and_empty_tokens(self):
        sm = SessionManager(60, clock=FakeClock())
        self.assertIsNone(sm.get("no-such-token"))
        self.assertIsNone(sm.get(""))
        self.assertIsNone(sm.get(None))
        sm.destroy("no-such-token")  # no error

    def test_expires_after_timeout(self):
        clock = FakeClock(0.0)
        sm = SessionManager(60, clock=clock)
        token = sm.create("alice", "operator")
        clock.t = 59.9
        self.assertIsNotNone(sm.get(token))  # still valid, refreshed at 59.9
        clock.t = 59.9 + 60.0  # 60s of inactivity since the refresh
        self.assertIsNone(sm.get(token))

    def test_get_refreshes_timer(self):
        clock = FakeClock(0.0)
        sm = SessionManager(60, clock=clock)
        token = sm.create("alice", "operator")
        clock.t = 50.0
        self.assertEqual(sm.get(token), ("alice", "operator"))  # refresh at 50
        clock.t = 100.0  # 50s since the refresh, but 100s since creation
        self.assertEqual(sm.get(token), ("alice", "operator"))  # refresh at 100
        clock.t = 159.0
        self.assertEqual(sm.get(token), ("alice", "operator"))
        clock.t = 220.0  # 61s of inactivity
        self.assertIsNone(sm.get(token))
        self.assertIsNone(sm.get(token))  # stays gone

    def test_invalid_timeout(self):
        with self.assertRaises(ValueError):
            SessionManager(0, clock=FakeClock())


if __name__ == "__main__":
    unittest.main()
