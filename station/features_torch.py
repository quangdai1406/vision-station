"""PyTorch feature extractor and cdist backend for the PatchCore anomaly model.

UNPROVEN UNTIL RUN ON THE GPU LAPTOP (python -m trainer selftest). Nothing in
this module has been run against real torch or torchvision yet; the tests use
injected fakes only.

torch and torchvision are imported lazily, in the constructors, and a missing
install raises StationError. Pass torch_module / torchvision_module to inject
them (tests). Standard library and numpy only at module level.

Pretrained weights: with weights_path=None, torchvision downloads its default
ImageNet weights once (needs internet). On a corporate PC without internet,
pass weights_path to a local state_dict file instead.

Decoded images are assumed to be RGB (or grey) with values 0..255.
"""
from __future__ import annotations

import time
from typing import Any

from station.contracts import StationError

SUPPORTED_ARCHS = ("resnet18", "wide_resnet50_2")
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CHUNK_BYTES = 256 * 1024 * 1024  # largest distance block held at once


def _np() -> Any:
    try:
        import numpy  # noqa: PLC0415 - imported lazily on purpose
    except ImportError as exc:
        raise StationError("numpy is required for the torch feature extractor; install numpy") from exc
    return numpy


def _import_torch(torch_module: Any) -> Any:
    """UNPROVEN UNTIL RUN ON THE GPU LAPTOP (python -m trainer selftest)."""
    if torch_module is not None:
        return torch_module
    try:
        import torch  # noqa: PLC0415 - imported lazily on purpose
    except ImportError as exc:
        raise StationError(
            "PyTorch is not installed; the torch feature extractor needs torch "
            "(install it on the training laptop, or use a station without the torch model)"
        ) from exc
    return torch


def _import_torchvision(torchvision_module: Any) -> Any:
    """UNPROVEN UNTIL RUN ON THE GPU LAPTOP (python -m trainer selftest)."""
    if torchvision_module is not None:
        return torchvision_module
    try:
        import torchvision  # noqa: PLC0415 - imported lazily on purpose
    except ImportError as exc:
        raise StationError(
            "torchvision is not installed; the torch feature extractor needs torchvision"
        ) from exc
    return torchvision


def _resolve_device(torch: Any, device: str) -> str:
    """UNPROVEN UNTIL RUN ON THE GPU LAPTOP (python -m trainer selftest).

    'auto' picks cuda when torch reports it available, else cpu. An explicit
    cuda device without CUDA is a StationError.
    """
    if not isinstance(device, str) or not device:
        raise StationError(f"device must be 'auto', 'cpu' or 'cuda', got {device!r}")
    cuda_ok = bool(torch.cuda.is_available())
    if device == "auto":
        return "cuda" if cuda_ok else "cpu"
    if device.startswith("cuda") and not cuda_ok:
        raise StationError(f"device {device!r} requested but CUDA is not available; use 'cpu' or 'auto'")
    return device


class TorchExtractor:
    """Layer2 + layer3 features of a torchvision ResNet, as (H', W', C) float32 numpy.

    Steps: ImageNet normalisation; grey repeated to 3 channels; resize to
    input_hw; layer2 and layer3 maps; 3x3 average pooling on each; layer3
    bilinearly upsampled to layer2 size; concatenated on channels.
    stride = input_h / H' (8 for ResNet layer2 at a size divisible by 8).

    UNPROVEN UNTIL RUN ON THE GPU LAPTOP (python -m trainer selftest).
    """

    def __init__(
        self,
        arch: str = "resnet18",
        *,
        device: str = "auto",
        input_hw: tuple[int, int] = (320, 320),
        weights_path: Any = None,
        torch_module: Any = None,
        torchvision_module: Any = None,
    ) -> None:
        """UNPROVEN UNTIL RUN ON THE GPU LAPTOP (python -m trainer selftest)."""
        if arch not in SUPPORTED_ARCHS:
            raise StationError(f"unsupported arch {arch!r}; use one of {', '.join(SUPPORTED_ARCHS)}")
        try:
            h, w = (int(v) for v in input_hw)
        except (TypeError, ValueError) as exc:
            raise StationError(f"input_hw must be (height, width), got {input_hw!r}") from exc
        if min(h, w) < 32:
            raise StationError(f"input_hw must be at least 32 pixels each side, got {(h, w)}")
        torch = _import_torch(torch_module)
        tv = _import_torchvision(torchvision_module)
        self.arch = arch
        self.input_hw = (h, w)
        self.name = f"{arch}-l2l3-{h}x{w}"
        self.stride = 8  # layer2 of a torchvision ResNet; refreshed from the real map in extract()
        self.device = _resolve_device(torch, device)
        self._torch = torch
        self._torch_device = torch.device(self.device)
        builder = getattr(tv.models, arch)
        if weights_path is None:
            model = builder(weights="DEFAULT")  # downloads the default ImageNet weights once
        else:
            model = builder(weights=None)
            try:
                state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
                model.load_state_dict(state)
            except Exception as exc:
                raise StationError(f"could not load weights from {weights_path}: {exc}") from exc
        model = model.eval()
        self._model = model.to(self._torch_device)

    def extract(self, image: Any) -> Any:
        """UNPROVEN UNTIL RUN ON THE GPU LAPTOP (python -m trainer selftest)."""
        np = _np()
        torch = self._torch
        F = torch.nn.functional
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif not (arr.ndim == 3 and arr.shape[2] == 3):
            raise StationError(f"image must be 2D grey or 3D RGB, got shape {arr.shape}")
        if arr.size == 0:
            raise StationError("image is empty")
        x = arr * np.float32(1.0 / 255.0)
        x = (x - np.asarray(_IMAGENET_MEAN, dtype=np.float32)) / np.asarray(_IMAGENET_STD, dtype=np.float32)
        chw = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)
        with torch.no_grad():
            t = torch.from_numpy(chw).to(self._torch_device)
            t = F.interpolate(t, size=self.input_hw, mode="bilinear", align_corners=False)
            m = self._model
            x0 = m.maxpool(m.relu(m.bn1(m.conv1(t))))
            x1 = m.layer1(x0)
            f2 = m.layer2(x1)
            f3 = m.layer3(f2)
            f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
            f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
            f3 = F.interpolate(f3, size=tuple(int(v) for v in f2.shape[-2:]), mode="bilinear", align_corners=False)
            feat = torch.cat([f2, f3], dim=1)
            out = feat[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
        self.stride = max(1, int(round(self.input_hw[0] / out.shape[0])))
        return out


class TorchBackend:
    """nn_distance with torch.cdist, in row chunks of at most ~256 MB of distances.

    Data is centred on the bank mean before the distances are taken, as in NumpyBackend.

    UNPROVEN UNTIL RUN ON THE GPU LAPTOP (python -m trainer selftest).
    """

    def __init__(self, device: str = "auto", *, torch_module: Any = None, max_chunk_bytes: int = _CHUNK_BYTES) -> None:
        """UNPROVEN UNTIL RUN ON THE GPU LAPTOP (python -m trainer selftest)."""
        torch = _import_torch(torch_module)
        self.device = _resolve_device(torch, device)
        self._torch = torch
        self._torch_device = torch.device(self.device)
        self.max_chunk_bytes = max(4, int(max_chunk_bytes))

    def nn_distance(self, queries: Any, bank: Any) -> Any:
        """UNPROVEN UNTIL RUN ON THE GPU LAPTOP (python -m trainer selftest)."""
        np = _np()
        torch = self._torch
        q = np.asarray(queries, dtype=np.float32)
        b = np.asarray(bank, dtype=np.float32)
        if q.ndim != 2 or b.ndim != 2 or q.shape[1] != b.shape[1]:
            raise ValueError(f"queries {q.shape} and bank {b.shape} must be (N, C) and (M, C) with equal C")
        if b.shape[0] == 0:
            raise ValueError("bank is empty")
        n, m = q.shape[0], b.shape[0]
        out = np.empty(n, dtype=np.float32)
        if n == 0:
            return out
        mu = b.mean(axis=0, dtype=np.float64).astype(np.float32)
        bt = torch.from_numpy(np.ascontiguousarray(b - mu)).to(self._torch_device)
        rows = min(n, max(1, (self.max_chunk_bytes // 4) // m))
        with torch.no_grad():
            for rs in range(0, n, rows):
                qc = np.ascontiguousarray(q[rs:rs + rows] - mu)
                d = torch.cdist(torch.from_numpy(qc).to(self._torch_device), bt)
                out[rs:rs + qc.shape[0]] = d.min(dim=1).values.cpu().numpy()
        return out


def selftest(
    extractor: Any = None,
    *,
    torch_module: Any = None,
    torchvision_module: Any = None,
) -> dict:
    """Run one synthetic 3-channel image through the extractor (laptop self-test).

    Returns name, device, feature_shape, seconds (one warm-up call, then one timed
    call), cuda_available and torch_version.

    UNPROVEN UNTIL RUN ON THE GPU LAPTOP (python -m trainer selftest).
    """
    np = _np()
    if extractor is None:
        extractor = TorchExtractor(torch_module=torch_module, torchvision_module=torchvision_module)
    torch = torch_module if torch_module is not None else getattr(extractor, "_torch", None)
    if torch is None:
        torch = _import_torch(None)
    h, w = (int(v) for v in extractor.input_hw)
    rng = np.random.RandomState(0)
    image = rng.randint(0, 256, size=(h, w, 3)).astype(np.float32)
    image[h // 4: h // 2, w // 4: w // 2, :] = 255.0  # a bright square so the input is not flat
    extractor.extract(image)  # warm-up: CUDA init and allocator are not timed
    start = time.perf_counter()
    feats = extractor.extract(image)
    if bool(torch.cuda.is_available()):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return {
        "name": extractor.name,
        "device": getattr(extractor, "device", "unknown"),
        "feature_shape": [int(v) for v in feats.shape],
        "seconds": round(float(seconds), 4),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(getattr(torch, "__version__", "unknown")),
    }
