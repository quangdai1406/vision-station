import http.server
import json
import os
import socket
import threading
import unittest
import urllib.parse
from unittest import mock

from station import parts
from station.contracts import StationError

try:  # the mirror test needs tools.criteria; skip it cleanly if that import ever fails
    from tools import criteria as tools_criteria
except Exception:  # noqa: BLE001
    tools_criteria = None

FAKE_TOKEN = "fake-token-for-tests-only-0001"
SEARCH = "/api/machine/parts/search"


class _State:
    def __init__(self):
        self.routes = {}  # path (no query) -> (status, headers, body bytes)
        self.requests = []  # (raw path, Authorization header)

    def route(self, path):
        return self.routes.get(path, (404, {}, b'{"error":"not found"}'))


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        state = self.server.state
        state.requests.append((self.path, self.headers.get("Authorization")))
        status, headers, body = state.route(urllib.parse.urlsplit(self.path).path)
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep test output quiet
        pass


def _json(data, status=200):
    return (status, {}, json.dumps(data).encode("utf-8"))


class _FakeMesCase(unittest.TestCase):
    def setUp(self):
        self.state = _State()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.state = self.state
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.client = parts.MesPartsClient(self.base, FAKE_TOKEN, timeout_s=5)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


class SearchTests(_FakeMesCase):
    def test_search_returns_list_with_bearer_and_quoted_query(self):
        rows = [{"part_number": "A B/1", "name": "Bracket"}]
        self.state.routes[SEARCH] = _json(rows)
        result = self.client.search("  A B/1 ", limit=500)
        self.assertEqual(result, rows)
        raw_path, auth = self.state.requests[0]
        self.assertEqual(auth, "Bearer " + FAKE_TOKEN)
        self.assertEqual(raw_path, SEARCH + "?q=A%20B%2F1&limit=50")

    def test_limit_is_clamped_to_one(self):
        self.state.routes[SEARCH] = _json([])
        self.client.search("x", limit=0)
        self.client.search("x", limit=-5)
        self.assertTrue(all(p.endswith("&limit=1") for p, _ in self.state.requests))

    def test_limit_must_be_an_integer(self):
        with self.assertRaises(ValueError):
            self.client.search("x", limit="5")

    def test_empty_and_long_queries_rejected_without_a_request(self):
        for bad in ("   ", "", "a" * 101):
            with self.subTest(bad=bad[:5]):
                with self.assertRaises(ValueError):
                    self.client.search(bad)
        self.assertEqual(self.state.requests, [])

    def test_100_character_query_is_allowed(self):
        self.state.routes[SEARCH] = _json([])
        self.assertEqual(self.client.search("a" * 100), [])
        self.assertEqual(len(self.state.requests), 1)

    def test_unexpected_search_shape_is_an_error(self):
        self.state.routes[SEARCH] = _json({"items": []})
        with self.assertRaises(parts.PartsError):
            self.client.search("x")


class ErrorTests(_FakeMesCase):
    def test_auth_and_server_errors_raise_without_the_token(self):
        for code in (401, 403, 500):
            self.state.routes[SEARCH] = (code, {}, b"{}")
            with self.subTest(code=code):
                with self.assertRaises(parts.PartsError) as ctx:
                    self.client.search("x")
                message = str(ctx.exception)
                self.assertIn(str(code), message)
                self.assertNotIn(FAKE_TOKEN, message)
                self.assertNotIn("Bearer", message)

    def test_same_errors_from_get_part_and_criteria(self):
        self.state.routes["/api/machine/parts/PN-1"] = (401, {}, b"{}")
        self.state.routes["/api/machine/criteria/PN-1"] = (500, {}, b"{}")
        with self.assertRaises(parts.PartsError) as ctx:
            self.client.get_part("PN-1")
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))
        with self.assertRaises(parts.PartsError) as ctx:
            self.client.criteria("PN-1")
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))

    def test_connection_refused_raises_without_the_token(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()  # nothing listens on this port now
        client = parts.MesPartsClient(f"http://127.0.0.1:{port}", FAKE_TOKEN, timeout_s=2)
        with self.assertRaises(parts.PartsError) as ctx:
            client.search("x")
        self.assertIn("could not reach MES", str(ctx.exception))
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))

    def test_redirect_is_not_followed(self):
        self.state.routes[SEARCH] = (302, {"Location": self.base + "/api/machine/parts/other"}, b"")
        self.state.routes["/api/machine/parts/other"] = _json([{"part_number": "X"}])
        with self.assertRaises(parts.PartsError) as ctx:
            self.client.search("x")
        self.assertIn("redirect", str(ctx.exception))
        self.assertEqual([p for p, _ in self.state.requests], [SEARCH + "?q=x&limit=20"])

    def test_non_json_body_is_an_error(self):
        self.state.routes[SEARCH] = (200, {}, b"<html>not json</html>")
        with self.assertRaises(parts.PartsError):
            self.client.search("x")

    def test_bad_base_url_and_empty_token_rejected(self):
        with self.assertRaises(parts.PartsError):
            parts.MesPartsClient("ftp://example.invalid", FAKE_TOKEN)
        with self.assertRaises(parts.PartsError) as ctx:
            parts.MesPartsClient(self.base, "")
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))


class LookupTests(_FakeMesCase):
    def test_get_part_404_returns_none(self):
        self.assertIsNone(self.client.get_part("PN-404"))

    def test_get_part_returns_dict_and_quotes_the_path(self):
        self.state.routes["/api/machine/parts/A%2FB%201"] = _json({"part_number": "A/B 1"})
        self.assertEqual(self.client.get_part("A/B 1"), {"part_number": "A/B 1"})
        self.assertEqual(self.state.requests[0][0], "/api/machine/parts/A%2FB%201")

    def test_criteria_404_raises(self):
        with self.assertRaises(parts.PartsError) as ctx:
            self.client.criteria("PN-404")
        self.assertIn("404", str(ctx.exception))

    def test_criteria_returns_list(self):
        rows = [{"id": 1, "criteria_name": "Scratch", "check_type": "visual", "active": True}]
        self.state.routes["/api/machine/criteria/PN-1"] = _json(rows)
        self.assertEqual(self.client.criteria("PN-1"), rows)

    def test_empty_part_number_rejected(self):
        with self.assertRaises(ValueError):
            self.client.get_part("  ")
        with self.assertRaises(ValueError):
            self.client.criteria("")


class ProxyTests(_FakeMesCase):
    def test_environment_proxy_is_ignored_by_default(self):
        self.state.routes[SEARCH] = _json([])
        env = {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9"}
        with mock.patch.dict(os.environ, env):
            client = parts.MesPartsClient(self.base, FAKE_TOKEN, timeout_s=5)
            self.assertEqual(client.search("x"), [])
        self.assertEqual(len(self.state.requests), 1)


SAMPLE = [
    {"id": 10, "criteria_name": "Scratch", "check_type": "visual", "active": True},
    {"id": 11, "criteria_name": "Dent", "check_type": "visual", "active": False},
    {"id": "12", "criteria_name": "Bore diameter", "check_type": "measurement",
     "active": True, "measurement_uom": "mm"},
    {"id": 13, "criteria_name": "Scratch!", "check_type": "visual", "active": True},
]


class BuildPartConfigTests(unittest.TestCase):
    def test_sample_gives_expected_pieces(self):
        cfg = parts.build_part_config(SAMPLE)
        self.assertEqual(cfg["criteria_ids"], ["10", "12", "13"])
        self.assertEqual(cfg["class_to_criterion"], {"scratch_10": "10", "scratch_13": "13"})
        self.assertEqual(cfg["measurement_criteria"], [{"id": "12", "name": "Bore diameter"}])
        self.assertEqual(cfg["inactive_count"], 1)
        self.assertEqual(len(cfg["warnings"]), 1)
        self.assertIn("Bore diameter", cfg["warnings"][0])

    def test_inactive_visual_is_not_a_class(self):
        cfg = parts.build_part_config(SAMPLE)
        self.assertNotIn("dent", " ".join(cfg["class_to_criterion"]))
        self.assertNotIn("11", cfg["criteria_ids"])

    def test_no_active_visual_gives_warning(self):
        only_measure = [{"id": 1, "criteria_name": "Bore", "check_type": "measurement", "active": True}]
        cfg = parts.build_part_config(only_measure)
        self.assertEqual(cfg["class_to_criterion"], {})
        self.assertTrue(any("no active visual" in w for w in cfg["warnings"]))

    def test_bad_entries_rejected(self):
        with self.assertRaises(StationError):
            parts.build_part_config([{"id": 1, "criteria_name": "X", "check_type": "odd", "active": True}])
        with self.assertRaises(StationError):
            parts.build_part_config("not a list")

    def test_slugify_matches_expected_forms(self):
        self.assertEqual(parts.slugify("Ünïcode  Dent!"), "unicode_dent")
        with self.assertRaises(ValueError):
            parts.slugify("!!!")

    @unittest.skipIf(tools_criteria is None, "tools.criteria is not importable")
    def test_matches_tools_criteria_output(self):
        samples = [
            SAMPLE,
            [
                {"id": "A-1", "criteria_name": "Pit mark", "check_type": "visual", "active": True},
                {"id": 2, "criteria_name": "Ünïcode Dent", "check_type": "visual", "active": True},
                {"id": 3, "criteria_name": "Length", "check_type": "measurement", "active": False},
                {"id": 4, "criteria_name": "Pit mark!", "check_type": "visual", "active": True},
            ],
        ]
        for sample in samples:
            with self.subTest(sample=[c["id"] for c in sample]):
                ours = parts.build_part_config(sample)
                theirs = tools_criteria.build_config_pieces(sample)
                self.assertEqual(ours["criteria_ids"], theirs["criteria_ids"])
                self.assertEqual(ours["class_to_criterion"], theirs["class_to_criterion"])


if __name__ == "__main__":
    unittest.main()
