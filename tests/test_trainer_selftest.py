import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from trainer import selftest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np


def _args(**over):
    base = {"extractor_config": None, "images": 2}
    base.update(over)
    return argparse.Namespace(**base)


def _fake_torch(cuda_ok=True):
    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_ok,
        get_device_name=lambda i: "Fake GPU 9000",
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=8 * 1024 ** 3),
        max_memory_allocated=lambda: 300 * 1024 ** 2,
        reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None,
    )
    return types.SimpleNamespace(__version__="2.9.0-fake", cuda=cuda)


class _FakeExtractor:
    def __init__(self, arch="resnet18", device="auto", input_hw=(320, 320), weights_path=None,
                 keep_aspect=True, torch_module=None, torchvision_module=None, channels=384, fail=False):
        self.name = f"{arch}-l2l3-aa-{input_hw[0]}x{input_hw[1]}-letterbox-imagenet-default"
        self.device = "cuda" if device == "auto" else device
        self.input_hw = tuple(input_hw)
        self.channels = channels
        self.fail = fail

    def extract(self, image):
        if self.fail:
            raise RuntimeError("boom on device")
        return np.zeros((4, 4, self.channels), dtype=np.float32)


def _fake_features(**extractor_kw):
    def make(*a, **kw):
        return _FakeExtractor(*a, **kw, **extractor_kw)

    return types.SimpleNamespace(
        TorchExtractor=make,
        selftest=lambda extractor, torch_module=None, torchvision_module=None: {
            "name": extractor.name,
            "device": extractor.device,
            "feature_shape": [4, 4, extractor.channels],
            "seconds": 0.01,
            "cuda_available": True,
            "torch_version": "2.9.0-fake",
        },
    )


def _run(args=None, **kw):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = selftest.run(args or _args(), **kw)
    return rc, out.getvalue() + err.getvalue()


class SelftestMissingDepsTests(unittest.TestCase):
    def test_missing_torch_exits_2_with_install_hint_and_installs_nothing(self):
        fake_np = types.ModuleType("numpy")
        with mock.patch.dict(sys.modules, {"torch": None, "numpy": fake_np}):
            rc, out = _run()
        self.assertEqual(rc, 2)
        self.assertIn("PyTorch is not installed", out)
        self.assertIn("pytorch.org", out)
        self.assertIn("nothing was installed", out)
        self.assertNotIn("SELFTEST SUMMARY", out)

    def test_missing_torchvision_exits_2(self):
        fake_np = types.ModuleType("numpy")
        with mock.patch.dict(sys.modules, {"torchvision": None, "numpy": fake_np}):
            rc, out = _run(torch_module=_fake_torch())
        self.assertEqual(rc, 2)
        self.assertIn("torchvision is not installed", out)

    def test_missing_numpy_exits_2(self):
        with mock.patch.dict(sys.modules, {"numpy": None}):
            rc, out = _run(torch_module=_fake_torch(), torchvision_module=object())
        self.assertEqual(rc, 2)
        self.assertIn("numpy is not installed", out)


@unittest.skipUnless(HAS_NUMPY, "numpy needed for the fake extractor output")
class SelftestFakeSuccessTests(unittest.TestCase):
    def test_success_prints_gpu_info_checks_and_summary(self):
        rc, out = _run(torch_module=_fake_torch(True), torchvision_module=object(),
                       features=_fake_features())
        self.assertEqual(rc, 0)
        self.assertIn("first time", out)
        self.assertIn("Fake GPU 9000", out)
        self.assertIn("8192.0 MiB", out)
        self.assertIn("[PASS] build extractor", out)
        self.assertIn("[PASS] feature shape", out)
        self.assertIn("[PASS] features finite", out)
        self.assertIn("[PASS] features_torch.selftest", out)
        self.assertIn("[INFO] peak GPU memory: 300.0 MiB", out)
        self.assertIn("=== SELFTEST SUMMARY (paste this back) ===", out)
        self.assertIn("result: PASS", out)
        self.assertIn("seconds_per_image:", out)
        self.assertIn("not measured: defect-detection accuracy", out)

    def test_cpu_only_warns_and_timings_are_labelled_cpu(self):
        rc, out = _run(torch_module=_fake_torch(False), torchvision_module=object(),
                       features=_fake_features())
        self.assertEqual(rc, 0)
        self.assertIn("[WARN] CUDA: not available: timings below are CPU timings", out)
        self.assertNotIn("peak GPU memory", out)

    def test_wrong_channel_count_fails(self):
        rc, out = _run(torch_module=_fake_torch(), torchvision_module=object(),
                       features=_fake_features(channels=100))
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] feature shape", out)
        self.assertIn("result: FAIL", out)

    def test_extract_error_is_reported_not_raised(self):
        rc, out = _run(torch_module=_fake_torch(), torchvision_module=object(),
                       features=_fake_features(fail=True))
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] timed batch", out)
        self.assertIn("boom on device", out)

    def test_extractor_construction_error_is_reported(self):
        class Boom(Exception):
            pass

        def make(*a, **kw):
            raise Boom("CUDA is not available")

        features = types.SimpleNamespace(TorchExtractor=make, selftest=None)
        rc, out = _run(torch_module=_fake_torch(), torchvision_module=object(), features=features)
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL] build extractor: Boom: CUDA is not available", out)

    def test_config_file_is_applied_and_unknown_keys_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "cfg.json"
            good.write_text(json.dumps({"arch": "wide_resnet50_2", "input_hw": [256, 256]}), encoding="utf-8")
            rc, out = _run(_args(extractor_config=str(good)), torch_module=_fake_torch(),
                           torchvision_module=object(), features=_fake_features(channels=1536))
            self.assertEqual(rc, 0)
            self.assertIn("arch wide_resnet50_2", out)
            self.assertIn("input 256x256", out)
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"arch": "resnet18", "token": "x"}), encoding="utf-8")
            rc, _ = _run(_args(extractor_config=str(bad)), torch_module=_fake_torch(),
                         torchvision_module=object(), features=_fake_features())
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
