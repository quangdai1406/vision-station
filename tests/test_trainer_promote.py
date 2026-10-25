import argparse
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from station import modelstore as ms
from trainer import promote

PART = "PN-200"
NOW1 = "2026-10-23T08:00:00Z"
NOW2 = "2026-10-24T08:00:00Z"
NOW3 = "2026-10-25T08:00:00Z"


def _parse(argv):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    promote.add_promote_arguments(sub)
    promote.add_rollback_arguments(sub)
    promote.add_versions_arguments(sub)
    return parser.parse_args(argv)


def _no_input(prompt):
    raise AssertionError("confirmation must not be asked in this test")


class PromoteCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.root = self.base / "models"
        self.store = ms.ModelStore(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _candidate(self, version, data, now):
        src = self.base / f"{version}.bin"
        src.write_bytes(data)
        manifest = {
            "type": "golden",
            "camera_ids": ["C1"],
            "trained_on": {"good_images": 8, "bad_images": 0},
            "metrics": {"note": "test only"},
            "notes": "",
        }
        self.store.add_candidate(PART, version, src, manifest, now)
        return self.root / ms.safe_part(PART) / version / src.name

    def _run(self, fn, argv, **kw):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fn(_parse(argv) if isinstance(argv, list) else argv, **kw)
        return rc, out.getvalue(), err.getvalue()

    def _promote_argv(self, version, by="Dai Tran", reason="first model", extra=()):
        return ["promote", "--models", str(self.root), "--part", PART, "--version", version,
                "--by", by, "--reason", reason, *extra]

    def _current_version(self):
        cur = self.store.current(PART)
        return cur.version if cur else None


class PromoteTests(PromoteCase):
    def test_approves_with_yes_and_records_approval(self):
        self._candidate("v1", b"model-one", NOW1)
        rc, out, _ = self._run(promote.run_promote,
                               self._promote_argv("v1", extra=["--yes"]), now=NOW2, input_fn=_no_input)
        self.assertEqual(rc, 0)
        self.assertEqual(self._current_version(), "v1")
        cur = self.store.current(PART)
        self.assertEqual(cur.approved_by, "Dai Tran")
        self.assertEqual(cur.reason, "first model")
        self.assertEqual(cur.approved_utc, NOW2)
        log = self.store.approvals(PART)
        self.assertEqual(log[-1]["action"], "approve")
        self.assertIn('"action": "approve"', out)
        self.assertIn("Currently approved:", out)
        self.assertIn("none (nothing approved yet", out)

    def test_typed_yes_on_stdin_approves(self):
        self._candidate("v1", b"model-one", NOW1)
        rc, _, _ = self._run(promote.run_promote, self._promote_argv("v1"), now=NOW2,
                             input_fn=lambda prompt: "yes\n")
        self.assertEqual(rc, 0)
        self.assertEqual(self._current_version(), "v1")

    def test_anything_but_yes_refuses_and_changes_nothing(self):
        self._candidate("v1", b"model-one", NOW1)
        for answer in ("y", "no", "", "YES please"):
            rc, _, err = self._run(promote.run_promote, self._promote_argv("v1"), now=NOW2,
                                   input_fn=lambda prompt, a=answer: a)
            self.assertEqual(rc, 1, answer)
            self.assertIn("not confirmed", err)
        self.assertIsNone(self._current_version())

    def test_eof_on_stdin_refuses(self):
        self._candidate("v1", b"model-one", NOW1)

        def eof(prompt):
            raise EOFError

        rc, _, _ = self._run(promote.run_promote, self._promote_argv("v1"), now=NOW2, input_fn=eof)
        self.assertEqual(rc, 1)
        self.assertIsNone(self._current_version())

    def test_by_and_reason_are_required_and_non_blank(self):
        self._candidate("v1", b"model-one", NOW1)
        for by, reason in (("", "why"), ("   ", "why"), ("Dai Tran", ""), ("Dai Tran", "  \t ")):
            rc, _, err = self._run(promote.run_promote,
                                   self._promote_argv("v1", by=by, reason=reason, extra=["--yes"]),
                                   now=NOW2, input_fn=_no_input)
            self.assertEqual(rc, 1, (by, reason))
            self.assertIn("required", err)
        self.assertIsNone(self._current_version())

    def test_refuses_automation_names(self):
        self._candidate("v1", b"model-one", NOW1)
        for by in ("trainer", "Weekly Trainer", "SYSTEM", "auto", "Auto-Bot", "scheduled job",
                   "nightly automation"):
            rc, _, err = self._run(promote.run_promote,
                                   self._promote_argv("v1", by=by, extra=["--yes"]),
                                   now=NOW2, input_fn=_no_input)
            self.assertEqual(rc, 1, by)
            self.assertIn("automated name", err)
        self.assertIsNone(self._current_version())

    def test_person_name_containing_such_substring_is_allowed(self):
        self._candidate("v1", b"model-one", NOW1)
        rc, _, _ = self._run(promote.run_promote,
                             self._promote_argv("v1", by="Systematic Lee", extra=["--yes"]),
                             now=NOW2, input_fn=_no_input)
        self.assertEqual(rc, 0)

    def test_refuses_tampered_version(self):
        model = self._candidate("v1", b"model-one", NOW1)
        model.write_bytes(b"tampered bytes")
        rc, _, err = self._run(promote.run_promote, self._promote_argv("v1", extra=["--yes"]),
                               now=NOW2, input_fn=_no_input)
        self.assertEqual(rc, 1)
        self.assertIn("failed verification", err)
        self.assertIsNone(self._current_version())

    def test_refuses_unknown_version(self):
        self._candidate("v1", b"model-one", NOW1)
        rc, _, err = self._run(promote.run_promote, self._promote_argv("v9", extra=["--yes"]),
                               now=NOW2, input_fn=_no_input)
        self.assertEqual(rc, 1)
        self.assertIn("not found", err)

    def test_refuses_version_already_approved(self):
        self._candidate("v1", b"model-one", NOW1)
        self._run(promote.run_promote, self._promote_argv("v1", extra=["--yes"]), now=NOW2,
                  input_fn=_no_input)
        rc, _, err = self._run(promote.run_promote, self._promote_argv("v1", extra=["--yes"]),
                               now=NOW3, input_fn=_no_input)
        self.assertEqual(rc, 1)
        self.assertIn("already the approved", err)

    def test_second_promotion_shows_current_and_moves_approval(self):
        self._candidate("v1", b"model-one", NOW1)
        self._candidate("v2", b"model-two", NOW2)
        self._run(promote.run_promote, self._promote_argv("v1", extra=["--yes"]), now=NOW2,
                  input_fn=_no_input)
        rc, out, _ = self._run(promote.run_promote,
                               self._promote_argv("v2", by="Anna Le", reason="better recall",
                                                  extra=["--yes"]),
                               now=NOW3, input_fn=_no_input)
        self.assertEqual(rc, 0)
        self.assertIn("version v1, approved by Dai Tran", out)
        self.assertEqual(self._current_version(), "v2")
        self.assertEqual(self.store.history(PART), ["v1", "v2"])

    def test_refuses_when_current_approved_model_is_unusable(self):
        self._candidate("v1", b"model-one", NOW1)
        self._run(promote.run_promote, self._promote_argv("v1", extra=["--yes"]), now=NOW2,
                  input_fn=_no_input)
        self._candidate("v2", b"model-two", NOW2)
        (self.root / ms.safe_part(PART) / "v1" / f"v1.bin").write_bytes(b"broken")
        rc, _, err = self._run(promote.run_promote, self._promote_argv("v2", extra=["--yes"]),
                               now=NOW3, input_fn=_no_input)
        self.assertEqual(rc, 1)
        self.assertIn("unusable", err)


class RollbackTests(PromoteCase):
    def _approve(self, version, now):
        self._run(promote.run_promote, self._promote_argv(version, extra=["--yes"]), now=now,
                  input_fn=_no_input)

    def _rollback_argv(self, by="Dai Tran", reason="v2 misses scratches", extra=()):
        return ["rollback", "--models", str(self.root), "--part", PART,
                "--by", by, "--reason", reason, *extra]

    def test_rollback_to_earlier_verified_version(self):
        self._candidate("v1", b"model-one", NOW1)
        self._candidate("v2", b"model-two", NOW1)
        self._approve("v1", NOW2)
        self._approve("v2", NOW3)
        rc, out, _ = self._run(promote.run_rollback, self._rollback_argv(extra=["--yes"]),
                               now="2026-10-26T08:00:00Z", input_fn=_no_input)
        self.assertEqual(rc, 0)
        self.assertEqual(self._current_version(), "v1")
        self.assertIn("Rolling back to: version v1", out)
        self.assertEqual(self.store.approvals(PART)[-1]["action"], "rollback")
        self.assertIn('"action": "rollback"', out)

    def test_rollback_needs_confirmation_and_prints_target(self):
        self._candidate("v1", b"model-one", NOW1)
        self._candidate("v2", b"model-two", NOW1)
        self._approve("v1", NOW2)
        self._approve("v2", NOW3)
        rc, out, err = self._run(promote.run_rollback, self._rollback_argv(),
                                 now="2026-10-26T08:00:00Z", input_fn=lambda p: "no")
        self.assertEqual(rc, 1)
        self.assertIn("Rolling back to: version v1", out)
        self.assertEqual(self._current_version(), "v2")

    def test_rollback_refused_when_nothing_earlier(self):
        self._candidate("v1", b"model-one", NOW1)
        self._approve("v1", NOW2)
        rc, _, err = self._run(promote.run_rollback, self._rollback_argv(extra=["--yes"]),
                               now=NOW3, input_fn=_no_input)
        self.assertEqual(rc, 1)
        self.assertIn("nothing to roll back to", err)
        self.assertEqual(self._current_version(), "v1")

    def test_rollback_refused_when_nothing_approved(self):
        self._candidate("v1", b"model-one", NOW1)
        rc, _, err = self._run(promote.run_rollback, self._rollback_argv(extra=["--yes"]),
                               now=NOW3, input_fn=_no_input)
        self.assertEqual(rc, 1)
        self.assertIn("nothing is approved", err)

    def test_rollback_skips_tampered_earlier_version_and_refuses_if_none_left(self):
        model1 = self._candidate("v1", b"model-one", NOW1)
        self._candidate("v2", b"model-two", NOW1)
        self._approve("v1", NOW2)
        self._approve("v2", NOW3)
        model1.write_bytes(b"tampered")
        rc, _, err = self._run(promote.run_rollback, self._rollback_argv(extra=["--yes"]),
                               now="2026-10-26T08:00:00Z", input_fn=_no_input)
        self.assertEqual(rc, 1)
        self.assertIn("nothing to roll back to", err)
        self.assertEqual(self._current_version(), "v2")

    def test_rollback_requires_by_reason_and_refuses_automation(self):
        self._candidate("v1", b"model-one", NOW1)
        self._candidate("v2", b"model-two", NOW1)
        self._approve("v1", NOW2)
        self._approve("v2", NOW3)
        for argv in (self._rollback_argv(by="", extra=["--yes"]),
                     self._rollback_argv(reason=" ", extra=["--yes"]),
                     self._rollback_argv(by="weekly-trainer", extra=["--yes"])):
            rc, _, _ = self._run(promote.run_rollback, argv, now="2026-10-26T08:00:00Z",
                                 input_fn=_no_input)
            self.assertEqual(rc, 1)
        self.assertEqual(self._current_version(), "v2")


class VersionsTests(PromoteCase):
    def test_lists_statuses_and_history(self):
        self._candidate("v1", b"model-one", NOW1)
        self._candidate("v2", b"model-two", NOW1)
        self._run(promote.run_promote, self._promote_argv("v1", extra=["--yes"]), now=NOW2,
                  input_fn=_no_input)
        self._run(promote.run_promote, self._promote_argv("v2", extra=["--yes"]), now=NOW3,
                  input_fn=_no_input)
        rc, out, _ = self._run(promote.run_versions,
                               _parse(["versions", "--models", str(self.root), "--part", PART]))
        self.assertEqual(rc, 0)
        self.assertIn("approved: v2", out)
        self.assertIn("retired", out)
        self.assertIn("history (oldest first): v1 -> v2", out)

    def test_no_versions(self):
        rc, out, _ = self._run(promote.run_versions,
                               _parse(["versions", "--models", str(self.root), "--part", PART]))
        self.assertEqual(rc, 0)
        self.assertIn("no versions", out)


if __name__ == "__main__":
    unittest.main()
