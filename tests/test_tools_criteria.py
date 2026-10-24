"""Tests for tools/criteria.py. The fake MES runs on 127.0.0.1 inside the test; tokens are fake."""

import contextlib
import csv
import io
import json
import os
import socket
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from tools import criteria as C

FAKE_TOKEN = "fake-token-not-real-0123456789"
TOKEN_ENV = "VISION_TEST_MES_TOKEN"


def _crit(cid, name, check_type="visual", active=True, **extra):
    d = {
        "id": cid,
        "scope_type": "part",
        "scope_value": "P1",
        "criteria_name": name,
        "criteria_description": f"description of {name}",
        "check_type": check_type,
        "measurement_uom": "mm" if check_type == "measurement" else None,
        "measurement_target": 10.0 if check_type == "measurement" else None,
        "measurement_tolerance": 0.2 if check_type == "measurement" else None,
        "standard_picture_path": f"/pics/{cid}.png",
        "active": active,
    }
    d.update(extra)
    return d


class _FakeMESHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.requests.append((self.path, self.headers.get("Authorization")))
        status, body, headers = self.server.routes.get(
            self.path, (404, b'{"error":"not found"}', {})
        )
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _MESTestCase(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _FakeMESHandler)
        self.server.routes = {}
        self.server.requests = []
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def route(self, path, status, body=b"[]", headers=None):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode("utf-8")
        self.server.routes[path] = (status, body, headers or {})


class SlugifyTests(unittest.TestCase):
    def test_basic_names(self):
        self.assertEqual(C.slugify("Scratch"), "scratch")
        self.assertEqual(C.slugify("Dent / Gouge (small)"), "dent_gouge_small")
        self.assertEqual(C.slugify("  Paint--run!!  "), "paint_run")
        self.assertEqual(C.slugify("3mm hole"), "3mm_hole")

    def test_accents_fold_to_ascii(self):
        self.assertEqual(C.slugify("Café Stain"), "cafe_stain")

    def test_empty_or_symbol_only_names_raise(self):
        for bad in ("", "   ", "___", "!!!"):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                C.slugify(bad)

    def test_collisions_get_id_suffix(self):
        crit = [_crit(7, "Scratch"), _crit(9, "scratch!"), _crit(3, "Dent")]
        names = C.derive_class_names(crit)
        self.assertEqual(names, {"7": "scratch_7", "9": "scratch_9", "3": "dent"})

    def test_collision_with_inactive_criterion_still_suffixes(self):
        crit = [_crit(1, "Scratch"), _crit(2, "Scratch", active=False)]
        names = C.derive_class_names(crit)
        self.assertEqual(names, {"1": "scratch_1", "2": "scratch_2"})

    def test_unslugable_name_falls_back_to_id(self):
        names = C.derive_class_names([_crit(5, "!!!")])
        self.assertEqual(names, {"5": "criterion_5"})


class SplitAndSummaryTests(unittest.TestCase):
    def setUp(self):
        self.crit = [
            _crit(1, "Scratch"),
            _crit(2, "Dent", active=False),
            _crit(3, "Bore diameter", check_type="measurement"),
            _crit(4, "Flange height", check_type="measurement", active=False),
        ]

    def test_split_by_type_keeps_only_active(self):
        visual, measurement = C.split_by_type(self.crit)
        self.assertEqual([c["id"] for c in visual], [1])
        self.assertEqual([c["id"] for c in measurement], [3])

    def test_summary_counts_and_warns_about_measurement(self):
        summary = C.summarize(self.crit)
        self.assertEqual(summary["visual_count"], 1)
        self.assertEqual(summary["measurement_count"], 1)
        self.assertEqual(summary["inactive_count"], 2)
        self.assertEqual(summary["classes"], ["scratch"])
        joined = " ".join(summary["warnings"])
        self.assertIn("Bore diameter", joined)
        self.assertIn("'pass'", joined)
        self.assertIn("Quality", joined)

    def test_no_visual_criteria_warns(self):
        summary = C.summarize([_crit(3, "Bore", check_type="measurement")])
        self.assertTrue(any("no active visual" in w for w in summary["warnings"]))


class ConfigPiecesTests(unittest.TestCase):
    def setUp(self):
        self.crit = [
            _crit(1, "Scratch"),
            _crit(2, "Dent", active=False),
            _crit(3, "Bore diameter", check_type="measurement"),
            _crit(4, "Flange height", check_type="measurement", active=False),
            _crit(11, "Paint run"),
        ]

    def test_criteria_ids_lists_all_active_visual_and_measurement(self):
        pieces = C.build_config_pieces(self.crit)
        self.assertEqual(pieces["criteria_ids"], ["1", "3", "11"])
        self.assertTrue(all(isinstance(i, str) for i in pieces["criteria_ids"]))

    def test_class_to_criterion_only_active_visual(self):
        pieces = C.build_config_pieces(self.crit)
        self.assertEqual(pieces["class_to_criterion"], {"scratch": "1", "paint_run": "11"})

    def test_default_is_null_and_marked_for_quality(self):
        pieces = C.build_config_pieces(self.crit)
        self.assertIsNone(pieces["default_criterion_id"])
        notes = pieces["_notes"]
        self.assertIn("default_criterion_id", notes["decision_for_quality"])
        self.assertIn("criteria_ids", notes["derived_from_mes"])
        self.assertTrue(any("Bore diameter" in w for w in notes["warnings"]))


class ChecklistTests(unittest.TestCase):
    def test_rows_for_active_only_with_placeholders(self):
        crit = [
            _crit(1, "Scratch"),
            _crit(2, "Dent", active=False),
            _crit(3, "Bore diameter", check_type="measurement"),
        ]
        rows = C.build_checklist_rows(crit)
        self.assertEqual([r["criterion_id"] for r in rows], ["1", "3"])
        self.assertEqual(rows[0]["defect_class"], "scratch")
        self.assertEqual(rows[1]["defect_class"], "")
        self.assertEqual(rows[1]["check_type"], "measurement")
        for r in rows:
            self.assertEqual(r["good_samples_needed"], "[Quality to define]")
            self.assertEqual(r["bad_samples_needed"], "[Quality to define]")
            self.assertEqual(r["collected_good"], "")
            self.assertEqual(r["collected_bad"], "")
            self.assertEqual(r["standard_picture_path"], f"/pics/{r['criterion_id']}.png")

    def test_write_outputs_csv_header_and_placeholder(self):
        with tempfile.TemporaryDirectory() as d:
            paths = C.write_outputs([_crit(1, "Scratch")], "P-1", d)
            with paths["checklist"].open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
        self.assertEqual(list(rows[0].keys()), C.CHECKLIST_FIELDS)
        self.assertEqual(rows[0]["good_samples_needed"], "[Quality to define]")


class PartNameTests(unittest.TestCase):
    def test_path_characters_are_replaced(self):
        name = C.safe_part_name("../../evil/part\\x:y")
        self.assertNotIn("/", name)
        self.assertNotIn("\\", name)
        self.assertNotIn(":", name)
        self.assertNotIn("..", name)

    def test_windows_reserved_and_empty_names(self):
        self.assertEqual(C.safe_part_name("CON"), "part_CON")
        self.assertEqual(C.safe_part_name("nul.txt"), "part_nul.txt")
        self.assertEqual(C.safe_part_name(""), "part")
        self.assertEqual(C.safe_part_name("..."), "part")

    def test_long_names_are_capped(self):
        self.assertLessEqual(len(C.safe_part_name("A" * 500)), 100)

    def test_normal_part_number_unchanged(self):
        self.assertEqual(C.safe_part_name("817D-LINER-L"), "817D-LINER-L")


class FetchTests(_MESTestCase):
    def test_200_returns_list_and_sends_bearer_header(self):
        self.route("/api/machine/criteria/817D-LINER-L", 200, [_crit(1, "Scratch")])
        result = C.fetch_criteria(self.base_url, FAKE_TOKEN, "817D-LINER-L")
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[0]["check_type"], "visual")
        self.assertEqual(self.server.requests[0][1], f"Bearer {FAKE_TOKEN}")

    def test_part_number_is_url_quoted(self):
        self.route("/api/machine/criteria/A%2FB%201", 200, [])
        C.fetch_criteria(self.base_url, FAKE_TOKEN, "A/B 1")
        self.assertEqual(self.server.requests[0][0], "/api/machine/criteria/A%2FB%201")

    def test_401_message_has_no_token(self):
        self.route("/api/machine/criteria/P1", 401, {"error": "bad token"})
        with self.assertRaises(C.CriteriaFetchError) as cm:
            C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1")
        self.assertIn("401", str(cm.exception))
        self.assertNotIn(FAKE_TOKEN, str(cm.exception))

    def test_403_404_500_raise_plain_errors(self):
        for status in (403, 404, 500, 503):
            with self.subTest(status=status):
                self.route("/api/machine/criteria/P1", status, {"error": "x"})
                with self.assertRaises(C.CriteriaFetchError) as cm:
                    C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1")
                self.assertIn(str(status), str(cm.exception))
                self.assertNotIn(FAKE_TOKEN, str(cm.exception))

    def test_redirect_is_not_followed(self):
        self.route("/api/machine/criteria/P1", 302, b"", {"Location": "/elsewhere"})
        self.route("/elsewhere", 200, [])
        with self.assertRaises(C.CriteriaFetchError) as cm:
            C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1")
        self.assertIn("redirect", str(cm.exception))
        paths = [p for p, _ in self.server.requests]
        self.assertNotIn("/elsewhere", paths)

    def test_connection_refused(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        with self.assertRaises(C.CriteriaFetchError) as cm:
            C.fetch_criteria(f"http://127.0.0.1:{port}", FAKE_TOKEN, "P1", timeout_s=3)
        self.assertIn("could not reach MES", str(cm.exception))
        self.assertNotIn(FAKE_TOKEN, str(cm.exception))

    def test_non_json_and_wrong_shape_raise(self):
        self.route("/api/machine/criteria/P1", 200, b"<html>oops</html>")
        with self.assertRaises(C.CriteriaFetchError):
            C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1")
        self.route("/api/machine/criteria/P1", 200, {"not": "a list"})
        with self.assertRaises(C.CriteriaFetchError):
            C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1")

    def test_bad_base_url_scheme_refused(self):
        with self.assertRaises(C.CriteriaFetchError):
            C.fetch_criteria("file:///etc", FAKE_TOKEN, "P1")

    def test_environment_proxy_is_not_used(self):
        self.route("/api/machine/criteria/P1", 200, [])
        env = {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9"}
        with mock.patch.dict(os.environ, env):
            result = C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1")
        self.assertEqual(result, [])


def _dead_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ProxyModeTests(_MESTestCase):
    """HTTP_PROXY points at a dead port. Default mode must ignore it; --use-env-proxy must use it."""

    def setUp(self):
        super().setUp()
        self.dead_proxy = f"http://127.0.0.1:{_dead_port()}"
        env = {"HTTP_PROXY": self.dead_proxy, "http_proxy": self.dead_proxy,
               "NO_PROXY": "", "no_proxy": ""}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = C.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_default_ignores_env_proxy_and_reaches_mes(self):
        self.route("/api/machine/criteria/P1", 200, [_crit(1, "Scratch")])
        result = C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1")
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(len(self.server.requests), 1)

    def test_use_env_proxy_sends_request_to_the_proxy(self):
        self.route("/api/machine/criteria/P1", 200, [])
        with self.assertRaises(C.CriteriaFetchError) as cm:
            C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1", timeout_s=3, use_env_proxy=True)
        self.assertIn("could not reach MES", str(cm.exception))
        self.assertNotIn(FAKE_TOKEN, str(cm.exception))
        self.assertEqual(self.server.requests, [])  # the MES was never reached directly

    def test_opener_handlers_differ_by_mode(self):
        self.route("/api/machine/criteria/P1", 200, [])
        real_build = urllib.request.build_opener
        built = []

        def spy(*handlers):
            opener = real_build(*handlers)
            built.append(opener)
            return opener

        with mock.patch.object(urllib.request, "build_opener", side_effect=spy):
            C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1")
            # Env mode goes through the dead proxy, so this call is expected to fail.
            with self.assertRaises(C.CriteriaFetchError):
                C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1", timeout_s=3, use_env_proxy=True)
        default_proxies = [h for h in built[0].handlers if isinstance(h, urllib.request.ProxyHandler)]
        env_proxies = [h for h in built[1].handlers if isinstance(h, urllib.request.ProxyHandler)]
        # An empty ProxyHandler({}) registers no protocol handlers, so the default opener has no proxy handler.
        self.assertEqual(default_proxies, [])
        self.assertEqual(len(env_proxies), 1)
        self.assertEqual(env_proxies[0].proxies.get("http"), self.dead_proxy)

    def test_token_never_in_errors_in_either_mode(self):
        self.route("/api/machine/criteria/P1", 401, {"error": "no"})
        with self.assertRaises(C.CriteriaFetchError) as cm:
            C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1")
        self.assertIn("401", str(cm.exception))
        self.assertNotIn(FAKE_TOKEN, str(cm.exception))
        # Env mode, with 127.0.0.1 bypassed so the 401 comes back from the MES itself.
        with mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}):
            with self.assertRaises(C.CriteriaFetchError) as cm:
                C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1", use_env_proxy=True)
        self.assertIn("401", str(cm.exception))
        self.assertNotIn(FAKE_TOKEN, str(cm.exception))
        # Env mode through the dead proxy: connection error, still no token.
        with self.assertRaises(C.CriteriaFetchError) as cm:
            C.fetch_criteria(self.base_url, FAKE_TOKEN, "P1", timeout_s=3, use_env_proxy=True)
        self.assertNotIn(FAKE_TOKEN, str(cm.exception))

    def test_cli_default_succeeds_and_flag_fails_through_dead_proxy(self):
        self.route("/api/machine/criteria/P1", 200, [_crit(1, "Scratch")])
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {TOKEN_ENV: FAKE_TOKEN}):
            argv = ["--base-url", self.base_url, "--token-env", TOKEN_ENV, "--part", "P1", "--out-dir", d]
            code, out, err = self._run(argv)
            self.assertEqual(code, 0, err)
            self.assertNotIn(FAKE_TOKEN, out + err)
            code, out, err = self._run(argv + ["--use-env-proxy"])
            self.assertEqual(code, 1)
            self.assertIn("could not reach MES", err)
            self.assertNotIn(FAKE_TOKEN, out + err)

    def test_cli_use_env_proxy_refused_with_from_json(self):
        code, _, err = self._run(["--from-json", "x.json", "--part", "P1", "--use-env-proxy"])
        self.assertEqual(code, 2)
        self.assertIn("--use-env-proxy", err)

    def test_help_documents_default_and_flag(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            C._parse_args(["--help"])
        text = " ".join(out.getvalue().split())
        self.assertIn("--use-env-proxy", text)
        self.assertIn("Default: connect directly", text)
        self.assertIn("https", text)


class LoadJsonTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_loads_list(self):
        p = self.dir / "c.json"
        p.write_text(json.dumps([_crit(1, "Scratch")]), encoding="utf-8")
        self.assertEqual(C.load_criteria_json(p)[0]["criteria_name"], "Scratch")

    def test_missing_file_is_clear(self):
        with self.assertRaises(C.CriteriaFileError) as cm:
            C.load_criteria_json(self.dir / "nope.json")
        self.assertIn("not found", str(cm.exception))

    def test_bad_json_and_bad_entries(self):
        p = self.dir / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        with self.assertRaises(C.CriteriaFileError):
            C.load_criteria_json(p)
        p.write_text(json.dumps([{"id": 1, "criteria_name": "X", "check_type": "colour", "active": True}]),
                     encoding="utf-8")
        with self.assertRaisesRegex(C.CriteriaFileError, "check_type"):
            C.load_criteria_json(p)
        p.write_text(json.dumps([_crit(1, "A"), _crit(1, "B")]), encoding="utf-8")
        with self.assertRaisesRegex(C.CriteriaFileError, "more than once"):
            C.load_criteria_json(p)


class CliTests(_MESTestCase):
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        env = mock.patch.dict(os.environ, {TOKEN_ENV: FAKE_TOKEN})
        env.start()
        self.addCleanup(env.stop)

    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = C.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_from_json_writes_three_files(self):
        src = self.dir / "in.json"
        src.write_text(json.dumps([_crit(1, "Scratch"), _crit(3, "Bore", check_type="measurement")]),
                       encoding="utf-8")
        out_dir = self.dir / "out"
        code, stdout, _ = self._run(
            ["--from-json", str(src), "--part", "817D-LINER-L", "--out-dir", str(out_dir)]
        )
        self.assertEqual(code, 0)
        self.assertTrue((out_dir / "criteria_817D-LINER-L.json").is_file())
        self.assertTrue((out_dir / "class_map.json").is_file())
        self.assertTrue((out_dir / "collection_checklist.csv").is_file())
        self.assertIn("Active visual (camera defect classes): 1", stdout)
        self.assertIn("WARNING", stdout)
        pieces = json.loads((out_dir / "class_map.json").read_text(encoding="utf-8"))
        self.assertEqual(pieces["class_to_criterion"], {"scratch": "1"})

    def test_refuses_when_token_env_unset(self):
        os.environ.pop("VISION_TEST_UNSET_VAR", None)
        code, _, err = self._run([
            "--base-url", self.base_url, "--token-env", "VISION_TEST_UNSET_VAR",
            "--part", "P1", "--out-dir", str(self.dir / "o"),
        ])
        self.assertNotEqual(code, 0)
        self.assertIn("VISION_TEST_UNSET_VAR is not set", err)
        self.assertEqual(self.server.requests, [])

    def test_token_passed_as_name_is_refused_without_echo(self):
        code, out, err = self._run([
            "--base-url", self.base_url, "--token-env", FAKE_TOKEN,
            "--part", "P1", "--out-dir", str(self.dir / "o"),
        ])
        self.assertNotEqual(code, 0)
        self.assertNotIn(FAKE_TOKEN, out + err)
        self.assertEqual(self.server.requests, [])

    def test_fetch_path_never_prints_token(self):
        self.route("/api/machine/criteria/P1", 200, [_crit(1, "Scratch")])
        code, out, err = self._run([
            "--base-url", self.base_url, "--token-env", TOKEN_ENV,
            "--part", "P1", "--out-dir", str(self.dir / "o"),
        ])
        self.assertEqual(code, 0)
        self.assertNotIn(FAKE_TOKEN, out + err)
        self.assertEqual(self.server.requests[0][1], f"Bearer {FAKE_TOKEN}")

    def test_fetch_error_exit_code_and_no_token(self):
        self.route("/api/machine/criteria/P1", 401, {"error": "no"})
        code, out, err = self._run([
            "--base-url", self.base_url, "--token-env", TOKEN_ENV,
            "--part", "P1", "--out-dir", str(self.dir / "o"),
        ])
        self.assertEqual(code, 1)
        self.assertIn("401", err)
        self.assertNotIn(FAKE_TOKEN, out + err)

    def test_mixed_sources_refused(self):
        code, _, err = self._run([
            "--from-json", "x.json", "--base-url", self.base_url, "--token-env", TOKEN_ENV, "--part", "P1",
        ])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
