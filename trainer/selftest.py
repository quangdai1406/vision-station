"""``python3 -m trainer selftest [--extractor-config FILE]``: first run of the torch feature extractor.

features_torch.py has never run against real torch or torchvision. This command is
the first run on the GPU laptop. It reports what it measures and what it does not:
it checks that the extractor runs, returns finite features of the expected width,
and times it. It does not measure defect-detection accuracy; that needs labelled
images and Quality's decision.

torch and numpy are imported only inside run(). A missing package prints what to
install and returns 2. Nothing is installed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Optional

SELFTEST_IMAGES = 8  # images timed in one batch; a measurement setting, not an acceptance limit
DEFAULT_CONFIG = {
    "arch": "resnet18",
    "device": "auto",
    "input_hw": [320, 320],
    "weights_path": None,
    "keep_aspect": True,
}
# Channels of concatenated layer2 + layer3 features for each supported torchvision arch.
EXPECTED_CHANNELS = {"resnet18": 128 + 256, "wide_resnet50_2": 512 + 1024}
INSTALL_HINT = (
    "Install the packages on this laptop yourself (nothing was installed):\n"
    "  PyTorch with CUDA: use the install command for your CUDA version from https://pytorch.org\n"
    "  (this installs torch and torchvision together)\n"
    "  numpy: pip install numpy"
)
GPU_NOTICE = (
    "NOTICE: GPU speed and accuracy are being measured for the first time on this code.\n"
    "        Timings and feature checks below are the first real numbers; treat them as unproven\n"
    "        until reviewed. Accuracy is NOT measured here: that needs labelled defect images\n"
    "        and Quality's decision."
)


def add_arguments(sub: Any) -> None:
    p = sub.add_parser(
        "selftest",
        help="run the torch feature extractor on this machine: GPU, speed, memory, finite features",
    )
    p.add_argument("--extractor-config", metavar="FILE",
                   help="JSON with any of: arch, device, input_hw [h, w], weights_path, keep_aspect")
    p.add_argument("--images", type=int, default=SELFTEST_IMAGES, metavar="N",
                   help=f"images in the timed batch (default {SELFTEST_IMAGES})")


def _load_config(path: Optional[str]) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path is None:
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read extractor config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("extractor config must be a JSON object")
    unknown = set(data) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown extractor config keys: {', '.join(sorted(unknown))}")
    cfg.update(data)
    return cfg


def _import_or_none(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError:
        return None


def _mib(n_bytes: float) -> float:
    return round(float(n_bytes) / (1024 * 1024), 1)


class _Report:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def line(self, status: str, name: str, detail: str = "") -> None:
        self.results.append((status, name, detail))
        suffix = f": {detail}" if detail else ""
        print(f"[{status}] {name}{suffix}")

    @property
    def failed(self) -> bool:
        return any(status == "FAIL" for status, _, _ in self.results)


def run(
    args: argparse.Namespace,
    *,
    torch_module: Any = None,
    torchvision_module: Any = None,
    features: Any = None,
) -> int:
    """torch_module, torchvision_module and features are injection points for tests.

    features is a module-like object with TorchExtractor and selftest (default: station.features_torch).
    """
    print(GPU_NOTICE)
    print()
    np = _import_or_none("numpy")
    if np is None:
        print("[FAIL] numpy is not installed")
        print(INSTALL_HINT)
        return 2
    torch = torch_module
    if torch is None:
        torch = _import_or_none("torch")
    if torch is None:
        print("[FAIL] PyTorch is not installed")
        print(INSTALL_HINT)
        return 2
    tv = torchvision_module
    if tv is None:
        tv = _import_or_none("torchvision")
    if tv is None:
        print("[FAIL] torchvision is not installed")
        print(INSTALL_HINT)
        return 2
    try:
        cfg = _load_config(getattr(args, "extractor_config", None))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if features is None:
        from station import features_torch as features  # noqa: PLC0415 - lazy on purpose

    rep = _Report()
    cuda = torch.cuda
    cuda_ok = bool(cuda.is_available())
    gpu_name = cuda.get_device_name(0) if cuda_ok else "none"
    vram_mib = _mib(cuda.get_device_properties(0).total_memory) if cuda_ok else None
    torch_version = str(getattr(torch, "__version__", "unknown"))
    print(f"torch {torch_version}, torchvision {getattr(tv, '__version__', 'unknown')}, "
          f"numpy {getattr(np, '__version__', 'unknown')}")
    print(f"CUDA available: {cuda_ok}; GPU: {gpu_name}; VRAM: "
          f"{vram_mib if vram_mib is not None else 'n/a'} MiB")
    if not cuda_ok:
        rep.line("WARN", "CUDA", "not available: timings below are CPU timings, not GPU timings")
    else:
        rep.line("PASS", "CUDA", gpu_name)

    h, w = (int(v) for v in cfg["input_hw"])
    print(f"config: arch {cfg['arch']}, device {cfg['device']}, input {h}x{w}, "
          f"keep_aspect {cfg['keep_aspect']}, weights {'file' if cfg['weights_path'] else 'torchvision default'}")

    try:
        extractor = features.TorchExtractor(
            cfg["arch"],
            device=cfg["device"],
            input_hw=(h, w),
            weights_path=cfg["weights_path"],
            keep_aspect=cfg["keep_aspect"],
            torch_module=torch,
            torchvision_module=tv,
        )
    except Exception as exc:  # StationError or any construction failure is reported, not raised
        rep.line("FAIL", "build extractor", f"{type(exc).__name__}: {exc}")
        return _finish(rep, cfg, gpu_name, vram_mib, torch_version, {})
    rep.line("PASS", "build extractor", f"{extractor.name} on {extractor.device}")

    timings: dict = {}
    try:
        rng = np.random.RandomState(0)
        images = []
        for _ in range(max(1, int(args.images))):
            img = rng.randint(0, 256, size=(h, w, 3)).astype(np.float32)
            img[h // 4: h // 2, w // 4: w // 2, :] = 255.0
            images.append(img)
        extractor.extract(images[0])  # warm-up: CUDA init and allocator are not timed
        if cuda_ok:
            cuda.synchronize()
            cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        feats = [extractor.extract(img) for img in images]
        if cuda_ok:
            cuda.synchronize()
        total = time.perf_counter() - start
        timings["batch_images"] = len(images)
        timings["seconds_per_image"] = round(total / len(images), 4)
        timings["peak_gpu_mib"] = _mib(cuda.max_memory_allocated()) if cuda_ok else None
        shape = [int(v) for v in feats[0].shape]
        expected = EXPECTED_CHANNELS.get(cfg["arch"])
        if len(shape) != 3:
            rep.line("FAIL", "feature shape", f"expected (H, W, C), got {shape}")
        elif expected is not None and shape[2] != expected:
            rep.line("FAIL", "feature shape", f"expected {expected} channels for {cfg['arch']}, got {shape[2]}")
        else:
            rep.line("PASS", "feature shape", f"{shape}")
        bad = sum(1 for f in feats if not np.all(np.isfinite(f)))
        if bad:
            rep.line("FAIL", "features finite", f"{bad} of {len(feats)} images gave NaN or inf")
        else:
            rep.line("PASS", "features finite", f"all {len(feats)} images")
        rep.line("PASS", "timed batch", f"{timings['seconds_per_image']} s per image over {len(images)} images")
        if cuda_ok:
            rep.line("INFO", "peak GPU memory", f"{timings['peak_gpu_mib']} MiB during the batch")
    except Exception as exc:
        rep.line("FAIL", "timed batch", f"{type(exc).__name__}: {exc}")

    try:
        st = features.selftest(extractor, torch_module=torch, torchvision_module=tv)
        rep.line("PASS", "features_torch.selftest", f"{st.get('seconds')} s for one image, "
                 f"shape {st.get('feature_shape')}, device {st.get('device')}")
    except Exception as exc:
        rep.line("FAIL", "features_torch.selftest", f"{type(exc).__name__}: {exc}")

    return _finish(rep, cfg, gpu_name, vram_mib, torch_version, timings)


def _finish(rep: _Report, cfg: dict, gpu_name: str, vram_mib: Any, torch_version: str, timings: dict) -> int:
    overall = "FAIL" if rep.failed else "PASS"
    print()
    print("=== SELFTEST SUMMARY (paste this back) ===")
    print(f"result: {overall}")
    print(f"torch: {torch_version}")
    print(f"gpu: {gpu_name}; vram_mib: {vram_mib}")
    print(f"arch: {cfg['arch']}; device: {cfg['device']}; input_hw: {cfg['input_hw']}; "
          f"keep_aspect: {cfg['keep_aspect']}")
    print(f"seconds_per_image: {timings.get('seconds_per_image', 'n/a')} "
          f"(batch of {timings.get('batch_images', 'n/a')}, warm-up excluded)")
    print(f"peak_gpu_mib: {timings.get('peak_gpu_mib', 'n/a')}")
    print("checks:")
    for status, name, detail in rep.results:
        print(f"  {status} {name}" + (f": {detail}" if detail else ""))
    print("not measured: defect-detection accuracy (needs labelled images and Quality's decision)")
    print("=== END SELFTEST SUMMARY ===")
    return 1 if rep.failed else 0
