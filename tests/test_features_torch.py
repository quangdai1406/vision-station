"""Tests for station/features_torch.py using FAKE torch and torchvision modules.

These prove the call sequence, shapes and device choice only. The real torch
path is UNPROVEN UNTIL RUN ON THE GPU LAPTOP (python -m trainer selftest).
"""
import contextlib
import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from station import anomaly, features_torch
from station.contracts import StationError

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np


class FakeTensor:
    """A numpy array with just enough torch-like methods for the extractor and backend."""

    def __init__(self, data):
        self.data = np.asarray(data)

    @property
    def shape(self):
        return tuple(self.data.shape)

    def to(self, device, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data

    def permute(self, *dims):
        return FakeTensor(np.transpose(self.data, dims))

    def contiguous(self):
        return self

    def __getitem__(self, index):
        return FakeTensor(self.data[index])

    def min(self, dim):
        return SimpleNamespace(values=FakeTensor(self.data.min(axis=dim)))


def _a(value):
    return value.data if isinstance(value, FakeTensor) else np.asarray(value)


class Recorder:
    def __init__(self):
        self.calls = []

    def log(self, *args):
        self.calls.append(args)

    def names(self):
        return [c[0] for c in self.calls]


class FakeLayer:
    """Subsamples by `stride` and expands to `channels` channels (values are stand-ins)."""

    def __init__(self, rec, name, stride, channels):
        self.rec, self.name, self.stride, self.channels = rec, name, stride, channels

    def __call__(self, x):
        self.rec.log(self.name, _a(x).shape)
        a = _a(x)[:, :, ::self.stride, ::self.stride]
        base = a.mean(axis=1, keepdims=True)
        out = base + np.arange(self.channels, dtype=np.float32).reshape(1, self.channels, 1, 1)
        return FakeTensor(out.astype(np.float32))


class FakeIdentity:
    def __init__(self, rec, name):
        self.rec, self.name = rec, name

    def __call__(self, x):
        self.rec.log(self.name)
        return x


class FakeBackbone:
    def __init__(self, rec):
        self.rec = rec
        self.conv1 = FakeLayer(rec, "conv1", 2, 4)
        self.bn1 = FakeIdentity(rec, "bn1")
        self.relu = FakeIdentity(rec, "relu")
        self.maxpool = FakeLayer(rec, "maxpool", 2, 4)
        self.layer1 = FakeLayer(rec, "layer1", 1, 4)
        self.layer2 = FakeLayer(rec, "layer2", 2, 6)  # stride 8 overall, 6 channels
        self.layer3 = FakeLayer(rec, "layer3", 2, 10)  # stride 16 overall, 10 channels

    def eval(self):
        self.rec.log("eval")
        return self

    def to(self, device):
        self.rec.log("to", device)
        return self

    def load_state_dict(self, state):
        self.rec.log("load_state_dict", state)


def make_torch(cuda=False):
    rec = Recorder()

    def avg_pool2d(x, kernel_size, stride=1, padding=0):
        rec.log("avg_pool2d", kernel_size, stride, padding)
        a = _a(x)
        k, p = int(kernel_size), int(padding)
        padded = np.pad(a, ((0, 0), (0, 0), (p, p), (p, p)))
        h_out = padded.shape[2] - k + 1
        w_out = padded.shape[3] - k + 1
        total = sum(padded[:, :, i:i + h_out, j:j + w_out] for i in range(k) for j in range(k))
        return FakeTensor(total / float(k * k))

    def interpolate(x, size=None, mode="nearest", align_corners=None):
        rec.log("interpolate", tuple(size), mode)
        a = _a(x)
        h_out, w_out = (int(v) for v in size)
        rows = np.arange(h_out) * a.shape[2] // h_out
        cols = np.arange(w_out) * a.shape[3] // w_out
        return FakeTensor(a[:, :, rows][:, :, :, cols])

    def cdist(a, b):
        rec.log("cdist", _a(a).shape, _a(b).shape)
        x, y = _a(a), _a(b)
        return FakeTensor(np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)))

    def from_numpy(array):
        rec.log("from_numpy", np.asarray(array).shape)
        return FakeTensor(np.asarray(array))

    def cat(tensors, dim=0):
        rec.log("cat", dim, len(tensors))
        return FakeTensor(np.concatenate([_a(t) for t in tensors], axis=dim))

    def load(path, map_location=None, weights_only=False):
        rec.log("load", str(path), weights_only)
        return {"fake_state": 1}

    torch = SimpleNamespace(
        __version__="2.0.0-fake",
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            synchronize=lambda: rec.log("synchronize"),
        ),
        device=lambda name: str(name),
        from_numpy=from_numpy,
        no_grad=lambda: _recorded_no_grad(rec),
        cat=cat,
        cdist=cdist,
        load=load,
        nn=SimpleNamespace(functional=SimpleNamespace(avg_pool2d=avg_pool2d, interpolate=interpolate)),
    )
    return torch, rec


@contextlib.contextmanager
def _recorded_no_grad(rec):
    rec.log("no_grad")
    yield


def make_torchvision(rec):
    def build(arch):
        def factory(weights=None):
            rec.log("build", arch, weights)
            return FakeBackbone(rec)
        return factory

    models = SimpleNamespace(resnet18=build("resnet18"), wide_resnet50_2=build("wide_resnet50_2"))
    return SimpleNamespace(models=models)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TorchExtractorTests(unittest.TestCase):
    def _extractor(self, cuda=False, **kwargs):
        torch, rec = make_torch(cuda=cuda)
        tv = make_torchvision(rec)
        kwargs.setdefault("input_hw", (64, 64))
        extractor = features_torch.TorchExtractor(
            torch_module=torch, torchvision_module=tv, **kwargs
        )
        return extractor, rec

    def test_missing_torch_gives_clear_station_error(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(StationError) as ctx:
                features_torch.TorchExtractor(torchvision_module=make_torchvision(Recorder()))
        self.assertIn("PyTorch is not installed", str(ctx.exception))

    def test_missing_torchvision_gives_clear_station_error(self):
        torch, _ = make_torch()
        with mock.patch.dict(sys.modules, {"torchvision": None}):
            with self.assertRaises(StationError) as ctx:
                features_torch.TorchExtractor(torch_module=torch)
        self.assertIn("torchvision", str(ctx.exception))

    def test_extract_returns_concatenated_float32_grid(self):
        extractor, rec = self._extractor()
        image = np.full((64, 64), 128.0)
        feats = extractor.extract(image)
        self.assertEqual(feats.shape, (8, 8, 16))  # 6 (layer2) + 10 (layer3)
        self.assertEqual(feats.dtype, np.float32)
        self.assertEqual(extractor.name, "resnet18-l2l3-64x64")
        self.assertEqual(extractor.stride, 8)
        names = rec.names()
        self.assertLess(names.index("conv1"), names.index("layer2"))
        self.assertLess(names.index("layer2"), names.index("layer3"))
        self.assertEqual(names.count("avg_pool2d"), 2)
        self.assertIn("interpolate", names)
        self.assertIn("cat", names)
        self.assertIn("no_grad", names)
        self.assertIn(("eval",), rec.calls)

    def test_grey_is_repeated_to_three_channels(self):
        extractor, rec = self._extractor()
        extractor.extract(np.zeros((64, 64)))
        self.assertIn(("from_numpy", (1, 3, 64, 64)), rec.calls)

    def test_rgb_image_keeps_three_channels(self):
        extractor, rec = self._extractor()
        extractor.extract(np.zeros((64, 64, 3)))
        self.assertIn(("from_numpy", (1, 3, 64, 64)), rec.calls)

    def test_bad_image_shape_is_station_error(self):
        extractor, _ = self._extractor()
        with self.assertRaises(StationError):
            extractor.extract(np.zeros((64, 64, 4)))

    def test_device_auto_picks_cpu_when_cuda_unavailable(self):
        extractor, rec = self._extractor(cuda=False, device="auto")
        self.assertEqual(extractor.device, "cpu")
        self.assertIn(("to", "cpu"), rec.calls)

    def test_device_auto_picks_cuda_when_available(self):
        extractor, _ = self._extractor(cuda=True, device="auto")
        self.assertEqual(extractor.device, "cuda")

    def test_explicit_cuda_without_cuda_is_station_error(self):
        with self.assertRaises(StationError):
            self._extractor(cuda=False, device="cuda")

    def test_default_weights_request_downloads_default(self):
        extractor, rec = self._extractor()
        self.assertIn(("build", "resnet18", "DEFAULT"), rec.calls)
        self.assertFalse(any(c[0] == "load" for c in rec.calls))

    def test_local_weights_path_is_loaded_without_download(self):
        extractor, rec = self._extractor(weights_path="/models/r18.pth")
        self.assertIn(("build", "resnet18", None), rec.calls)
        self.assertIn(("load", "/models/r18.pth", True), rec.calls)
        self.assertIn(("load_state_dict", {"fake_state": 1}), rec.calls)

    def test_unsupported_arch_and_small_input_rejected(self):
        torch, rec = make_torch()
        tv = make_torchvision(rec)
        with self.assertRaises(StationError):
            features_torch.TorchExtractor("vgg16", torch_module=torch, torchvision_module=tv)
        with self.assertRaises(StationError):
            features_torch.TorchExtractor("resnet18", input_hw=(16, 16), torch_module=torch, torchvision_module=tv)

    def test_wide_resnet_builds_its_own_backbone(self):
        torch, rec = make_torch()
        tv = make_torchvision(rec)
        extractor = features_torch.TorchExtractor("wide_resnet50_2", input_hw=(64, 64),
                                                  torch_module=torch, torchvision_module=tv)
        self.assertEqual(extractor.name, "wide_resnet50_2-l2l3-64x64")
        self.assertIn(("build", "wide_resnet50_2", "DEFAULT"), rec.calls)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TorchBackendTests(unittest.TestCase):
    def test_nn_distance_matches_numpy_backend(self):
        torch, _ = make_torch()
        rng = np.random.RandomState(7)
        q = rng.randn(90, 5).astype(np.float32) * 2 + 100
        b = rng.randn(40, 5).astype(np.float32) * 2 + 100
        got = features_torch.TorchBackend(torch_module=torch).nn_distance(q, b)
        want = anomaly.NumpyBackend().nn_distance(q, b)
        self.assertEqual(got.shape, (90,))
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)

    def test_nn_distance_is_chunked_when_budget_is_small(self):
        torch, rec = make_torch()
        q = np.random.RandomState(8).randn(50, 3).astype(np.float32)
        b = np.random.RandomState(9).randn(10, 3).astype(np.float32)
        backend = features_torch.TorchBackend(torch_module=torch, max_chunk_bytes=4 * 10 * 7)
        got = backend.nn_distance(q, b)
        cdist_calls = [c for c in rec.calls if c[0] == "cdist"]
        self.assertGreater(len(cdist_calls), 1)
        want = anomaly.NumpyBackend().nn_distance(q, b)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)

    def test_auto_device_cpu_without_cuda(self):
        torch, _ = make_torch(cuda=False)
        self.assertEqual(features_torch.TorchBackend(torch_module=torch).device, "cpu")


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class SelftestTests(unittest.TestCase):
    def test_selftest_returns_expected_keys(self):
        torch, rec = make_torch(cuda=False)
        tv = make_torchvision(rec)
        extractor = features_torch.TorchExtractor(input_hw=(64, 64), torch_module=torch, torchvision_module=tv)
        report = features_torch.selftest(extractor)
        self.assertEqual(
            set(report),
            {"name", "device", "feature_shape", "seconds", "cuda_available", "torch_version"},
        )
        self.assertEqual(report["name"], "resnet18-l2l3-64x64")
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["feature_shape"], [8, 8, 16])
        self.assertFalse(report["cuda_available"])
        self.assertEqual(report["torch_version"], "2.0.0-fake")
        self.assertGreaterEqual(report["seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
