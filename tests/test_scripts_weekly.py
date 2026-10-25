"""Static checks for the Windows weekly-training scripts (standard library only).

These do not run PowerShell. They check the files exist, the example config parses with
placeholders, and the safety rules hold: no promote or purge invocation outside comments,
no password literal, and the Sunday 06:00 schedule is stated.
"""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WIN = ROOT / "scripts" / "windows"
RUN_PS1 = WIN / "run_weekly_training.ps1"
INSTALL_PS1 = WIN / "install_weekly_task.ps1"
EXAMPLE_JSON = WIN / "trainer.example.json"
RUNBOOK = ROOT / "docs" / "weekly_training.md"
PLACEHOLDER = "[Quality to define]"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _code_lines(text: str) -> list[str]:
    """Lines that are not comments (a line whose first non-space character is '#')."""
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


class WeeklyScriptFilesTest(unittest.TestCase):
    def test_files_exist(self) -> None:
        for path in (RUN_PS1, INSTALL_PS1, EXAMPLE_JSON, RUNBOOK):
            with self.subTest(path=str(path)):
                self.assertTrue(path.is_file(), f"missing: {path}")

    def test_powershell_scripts_mention_weekly_sunday_0600(self) -> None:
        for path in (RUN_PS1, INSTALL_PS1):
            text = _read(path).lower()
            with self.subTest(path=path.name):
                self.assertIn("weekly", text)
                self.assertIn("sunday", text)
                self.assertIn("06:00", text)

    def test_powershell_scripts_have_strict_mode(self) -> None:
        for path in (RUN_PS1, INSTALL_PS1):
            with self.subTest(path=path.name):
                self.assertIn("Set-StrictMode -Version Latest", _read(path))


class ExampleConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = json.loads(_read(EXAMPLE_JSON))

    def test_parses_and_has_required_keys(self) -> None:
        for key in ("store", "models", "reports", "logs", "python", "extractor",
                    "min_good", "min_holdout", "holdout_fraction"):
            with self.subTest(key=key):
                self.assertIn(key, self.cfg)
        for key in ("arch", "input_hw", "weights_path", "device"):
            with self.subTest(extractor_key=key):
                self.assertIn(key, self.cfg["extractor"])

    def test_min_counts_are_quality_placeholders(self) -> None:
        self.assertEqual(self.cfg["min_good"], PLACEHOLDER)
        self.assertEqual(self.cfg["min_holdout"], PLACEHOLDER)

    def test_holdout_fraction_is_quality_placeholder(self) -> None:
        self.assertEqual(self.cfg["holdout_fraction"], PLACEHOLDER)

    def test_every_setting_is_a_bracketed_placeholder(self) -> None:
        values = [self.cfg[k] for k in ("store", "models", "reports", "logs", "python")]
        for value in self.cfg["extractor"].values():
            values += value if isinstance(value, list) else [value]
        for value in values:
            with self.subTest(value=value):
                self.assertRegex(value, r"^\[.*\]$")


class ExtractorContractTest(unittest.TestCase):
    """The JSON the .ps1 writes for the trainer must pass trainer.weekly._load_config."""

    def test_filled_example_is_accepted_by_the_trainer(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from trainer.weekly import _load_config

        cfg = json.loads(_read(EXAMPLE_JSON))["extractor"]
        # what run_weekly_training.ps1 writes: arch, input_hw as [h, w] ints, weights_path, device
        filled = {"arch": "resnet18", "input_hw": [480, 640], "weights_path": "weights.pth", "device": "cuda"}
        self.assertEqual(set(filled), set(cfg))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extractor-config.json"
            path.write_text(json.dumps(filled), encoding="utf-8")
            loaded = _load_config(str(path))
        self.assertEqual(loaded["input_hw"], (480, 640))

    def test_script_keeps_input_hw_a_list_of_ints(self) -> None:
        text = _read(RUN_PS1)
        self.assertIn("[int[]]", text)
        self.assertNotIn("Get-RequiredValue $extractor 'input_hw'", text)


class SafetyRulesTest(unittest.TestCase):
    def test_no_promote_or_purge_invocation_outside_comments(self) -> None:
        forbidden = ("trainer promote", "trainer purge", "--confirm")
        for path in (RUN_PS1, INSTALL_PS1):
            for line in _code_lines(_read(path)):
                for needle in forbidden:
                    with self.subTest(path=path.name, line=line):
                        self.assertNotIn(needle, line)

    def test_no_password_literal_values(self) -> None:
        for path in (RUN_PS1, INSTALL_PS1, EXAMPLE_JSON):
            text = _read(path)
            with self.subTest(path=path.name, check="key: value"):
                self.assertIsNone(re.search(r"password\s*[:=]\s*\S", text, re.IGNORECASE))
            with self.subTest(path=path.name, check="-Password literal"):
                self.assertIsNone(re.search(r"-Password\s+['\"]", text, re.IGNORECASE))

    def test_run_script_does_not_read_secret_variables(self) -> None:
        text = _read(RUN_PS1)
        self.assertNotIn("STATION_MES_TOKEN", text)
        self.assertIsNone(re.search(r"\$env:\w*(TOKEN|PASSWORD|SECRET)", text, re.IGNORECASE))

    def test_run_script_checks_onedrive_online_only_attributes(self) -> None:
        text = _read(RUN_PS1)
        self.assertIn("0x00400000", text)  # FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
        self.assertIn("0x00040000", text)  # FILE_ATTRIBUTE_RECALL_ON_OPEN

    def test_run_script_requires_min_counts_without_defaults(self) -> None:
        text = _read(RUN_PS1)
        self.assertIn("Get-RequiredValue $cfg 'min_good'", text)
        self.assertIn("Get-RequiredValue $cfg 'min_holdout'", text)
        self.assertIn("Get-RequiredValue $cfg 'holdout_fraction'", text)
        self.assertIn("--holdout-fraction", text)
        self.assertIsNone(re.search(r"min_good\s*=\s*['\"]?\d", text))

    def test_install_script_never_names_a_password_parameter(self) -> None:
        code = "\n".join(_code_lines(_read(INSTALL_PS1)))
        self.assertNotIn("-Password", code)
        self.assertNotIn(".Password", code)
        self.assertNotIn("GetNetworkCredential", code)

    def test_install_script_is_unattended_limited_with_required_settings(self) -> None:
        text = _read(INSTALL_PS1)
        for needle in ("-RunLevel Limited", "-StartWhenAvailable", "-WakeToRun",
                       "-ExecutionTimeLimit", "IgnoreNew", "-AllowStartIfOnBatteries:$false",
                       "-Uninstall", "-WhatIf", "VisionStation Weekly Training"):
            with self.subTest(needle=needle):
                self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
