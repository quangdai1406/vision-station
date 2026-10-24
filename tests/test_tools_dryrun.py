"""Chained dry run of the repeatability tools, plus checks on the operator protocol.

The capture, report and threshold runs need numpy (skipped without it). They use the
FakeCv2 from tests/test_tools_capture.py, so no camera or opencv is needed. Every output
goes into a temporary folder. The protocol checks need no numpy.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is needed only for the chained runs
    np = None

from tests.test_tools_capture import FakeCv2
from tools import capture as capture_mod
from tools import criteria as criteria_mod
from tools import report as report_mod
from tools import thresholds as thresholds_mod

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "repeatability_protocol.md"
README = REPO / "README.md"
NEED_NUMPY = unittest.skipUnless(np is not None, "numpy is not installed")
NOTE_SENTENCE = "Candidate values from labelled samples only"


def _quiet(fn, argv):
    """Run a CLI main() and return (exit code, stdout text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        code = fn(argv)
    return code, buf.getvalue()


def _capture(out_dir, name, mode, **overrides):
    """A tiny capture with the fake camera: small frames, no waiting, no flush-delay."""
    kwargs = dict(
        mode=mode,
        devices=[0],
        out_dir=out_dir,
        name=name,
        count=4 if mode == "stationary" else 3,
        interval_s=0.0,
        warmup_s=0.0,
        width=64,
        height=48,
        cv2_module=FakeCv2(),
        out=lambda line: None,
        sleep=lambda seconds: None,
    )
    kwargs.update(overrides)
    return capture_mod.capture_session(**kwargs)


@NEED_NUMPY
class CaptureReportChainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.sessions = self.root / "sessions"

    def test_stationary_capture_then_report(self):
        summary = _capture(self.sessions, "stationary-1", "stationary",
                           part="DRYRUN-PART", lighting="test light", operator="tester")
        folder = self.sessions / "stationary-1"
        self.assertIsNone(summary["aborted"])
        self.assertEqual(summary["frames_per_device"], {"0": 4})
        self.assertTrue((folder / "session.json").is_file())
        self.assertTrue((folder / "frames_dev0.npz").is_file())
        self.assertTrue((folder / "samples" / "dev0_frame0.png").is_file())

        out = self.root / "report"
        code, _ = _quiet(report_mod.main, [str(folder), "--out", str(out)])
        self.assertEqual(code, 0)
        md = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("Observed noise floor, not a threshold", md)
        self.assertIn("## 6. What this test CANNOT tell you", md)
        self.assertIn("DRYRUN-PART", md)
        self.assertTrue((out / "report.json").is_file())
        self.assertTrue((out / "summary.csv").is_file())

    def test_placement_without_prompt_then_report_both_sessions(self):
        lines = []

        def no_input(prompt):
            raise AssertionError("placement with no_prompt=True must not wait for input")

        _capture(self.sessions, "stationary-1", "stationary", part="DRYRUN-PART")
        summary = _capture(self.sessions, "placement-1", "placement", no_prompt=True,
                           input_fn=no_input, out=lines.append)
        self.assertIsNone(summary["aborted"])
        self.assertEqual(summary["frames_per_device"], {"0": 3})
        self.assertTrue(any("Round 1/3" in line for line in lines))

        out = self.root / "both"
        code, _ = _quiet(report_mod.main, [
            str(self.sessions / "stationary-1"),
            str(self.sessions / "placement-1"),
            "--out", str(out),
        ])
        self.assertEqual(code, 0)
        md = (out / "report.md").read_text(encoding="utf-8")
        self.assertIn("## 3. Stationary results", md)
        self.assertIn("## 4. Placement results", md)
        self.assertIn("Residual with alignment", md)
        self.assertIn("Observed noise floor, not a threshold", md)
        self.assertIn("## 6. What this test CANNOT tell you", md)
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["rows"]), 2)
        with (out / "summary.csv").open(newline="", encoding="utf-8") as fh:
            self.assertEqual(len(list(csv.DictReader(fh))), 2)


class CriteriaDryRunTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_from_json_writes_class_map_and_checklist(self):
        criteria = [
            {"id": 101, "criteria_name": "Scratch on face", "criteria_description": "Visible scratch",
             "check_type": "visual", "active": True, "standard_picture_path": "std/scratch.png"},
            {"id": 102, "criteria_name": "Hole diameter", "criteria_description": "Diameter in tolerance",
             "check_type": "measurement", "active": True, "measurement_uom": "mm",
             "measurement_target": 8.0, "measurement_tolerance": 0.1},
            {"id": 103, "criteria_name": "Old dent", "criteria_description": "Retired",
             "check_type": "visual", "active": False},
        ]
        source = self.root / "criteria.json"
        source.write_text(json.dumps(criteria), encoding="utf-8")
        out = self.root / "criteria_out"

        code, text = _quiet(criteria_mod.main, [
            "--from-json", str(source), "--part", "DRYRUN-PART", "--out-dir", str(out),
        ])
        self.assertEqual(code, 0)
        self.assertIn("WARNING", text)

        class_map = json.loads((out / "class_map.json").read_text(encoding="utf-8"))
        self.assertEqual(class_map["class_to_criterion"], {"scratch_on_face": "101"})
        self.assertEqual(class_map["criteria_ids"], ["101", "102"])
        self.assertIsNone(class_map["default_criterion_id"])

        checklist_text = (out / "collection_checklist.csv").read_text(encoding="utf-8")
        self.assertIn("[Quality to define]", checklist_text)
        with (out / "collection_checklist.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["criterion_id"] for r in rows], ["101", "102"])
        by_id = {r["criterion_id"]: r for r in rows}
        self.assertEqual(by_id["101"]["defect_class"], "scratch_on_face")
        self.assertEqual(by_id["102"]["defect_class"], "")
        for r in rows:
            self.assertEqual(r["good_samples_needed"], "[Quality to define]")
            self.assertEqual(r["bad_samples_needed"], "[Quality to define]")
        self.assertNotIn("Old dent", checklist_text)


class ThresholdsDryRunTests(unittest.TestCase):
    GOOD = [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
    BAD = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def _scores_csv(self, good, bad):
        path = self.root / "scores.csv"
        lines = ["label,score,defect_class,part_id"]
        for i, s in enumerate(good):
            lines.append(f"good,{s},,G{i:03d}")
        for i, s in enumerate(bad):
            cls = "scratch" if i % 2 == 0 else "dent"
            lines.append(f"bad,{s},{cls},B{i:03d}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_separable_scores_give_candidate_values(self):
        path = self._scores_csv(self.GOOD, self.BAD)
        md = self.root / "threshold_report.md"
        js = self.root / "result.json"
        code, text = _quiet(thresholds_mod.main, [
            str(path), "--min-good", "10", "--min-bad", "10",
            "--review-from-good-percentile", "90", "--model-version", "dryrun-1",
            "--out", str(md), "--json", str(js),
        ])
        self.assertEqual(code, 0)
        report = md.read_text(encoding="utf-8")
        self.assertIn(NOTE_SENTENCE, report)
        self.assertIn(NOTE_SENTENCE, text)
        self.assertIn("## Candidate values", report)
        self.assertIn("Model version: dryrun-1", report)
        # t_img is the lowest bad score; the 90th percentile of 10 good scores is the 9th value.
        self.assertIn("`0.6`", report)
        self.assertIn("`0.28`", report)
        result = json.loads(js.read_text(encoding="utf-8"))
        self.assertTrue(result["separable"])
        self.assertAlmostEqual(result["t_img"], 0.6)
        self.assertAlmostEqual(result["t_low"], 0.28)

    def test_too_few_samples_gives_refusal_without_numbers(self):
        path = self._scores_csv(self.GOOD, self.BAD)
        md = self.root / "refusal.md"
        code, _ = _quiet(thresholds_mod.main, [
            str(path), "--min-good", "50", "--min-bad", "10",
            "--review-from-good-percentile", "90", "--out", str(md),
        ])
        self.assertEqual(code, 0)
        report = md.read_text(encoding="utf-8")
        self.assertIn("Not enough labelled samples", report)
        self.assertIn("No candidate values are given", report)
        self.assertNotIn("## Candidate values", report)
        self.assertNotIn("`0.6`", report)
        self.assertIn(NOTE_SENTENCE, report)


class ProtocolDocTests(unittest.TestCase):
    TOOLS = {
        "capture": capture_mod.main,
        "report": report_mod.main,
        "criteria": criteria_mod.main,
        "thresholds": thresholds_mod.main,
    }

    def _doc_text(self):
        return DOC.read_text(encoding="utf-8")

    def _doc_commands(self):
        """(tool name, command text) for every `py -m tools.<name>` in the protocol.

        The command text runs to the end of the line, or to a closing backtick for an
        inline command in prose.
        """
        commands = []
        for line in self._doc_text().splitlines():
            for match in re.finditer(r"py -m tools\.([a-z_]+)", line):
                rest = line[match.start():]
                rest = rest.split("`", 1)[0]
                commands.append((match.group(1), rest.strip()))
        return commands

    def _help_text(self, name):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
            self.TOOLS[name](["--help"])
        return buf.getvalue()

    def test_every_protocol_command_uses_a_known_tool(self):
        commands = self._doc_commands()
        self.assertTrue(commands, "no 'py -m tools.<name>' lines found in the protocol")
        for name, _ in commands:
            self.assertIn(name, self.TOOLS, f"protocol names unknown tool tools.{name}")
        # Every use of -m tools. in the doc must be written as 'py -m tools.<name>'.
        for line in self._doc_text().splitlines():
            self.assertNotRegex(line, r"(?<!py )-m tools\.", line)
        used = {name for name, _ in commands}
        self.assertTrue({"capture", "report", "criteria", "thresholds"} <= used)

    def test_every_flag_in_a_protocol_command_exists_in_that_tools_help(self):
        help_text = {name: self._help_text(name) for name in self.TOOLS}
        checked = 0
        for name, line in self._doc_commands():
            for flag in re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", line):
                checked += 1
                self.assertIn(flag, help_text[name],
                              f"{flag} in the protocol is not an option of tools.{name}: {line}")
        self.assertGreater(checked, 0)

    def test_protocol_states_its_limits_and_the_go_no_go_role(self):
        text = self._doc_text()
        self.assertIn("does NOT", text)
        self.assertIn("candidates, not approved values", text)
        self.assertIn("go/no-go input for Quality", text)
        self.assertIn("[Quality to define]", text)

    def test_readme_links_the_protocol_and_lists_the_tools(self):
        text = README.read_text(encoding="utf-8")
        self.assertIn("## Repeatability and threshold tools", text)
        self.assertIn("docs/repeatability_protocol.md", text)
        for module in ("capture", "report", "criteria", "thresholds", "analysis"):
            self.assertIn(f"tools/{module}.py", text)


if __name__ == "__main__":
    unittest.main()
