"""Numpy-only image repeatability metrics. Shapes and return keys: see tools/FORMAT.md.

This module only measures. It makes no pass/fail judgements and has no thresholds.
gray is an ndarray of shape (T, H, W) with dtype uint8 (or float). Frames are
converted to float64 one at a time so that large stacks do not need a float copy
of the whole array.
"""

import os

from tools import ToolDependencyError

# Work-size bound for chunked row processing (not a judgement threshold).
_CHUNK_ELEMS = 4_000_000
# Relative floor for the phase-correlation normalisation (numerical, not a judgement).
_PHASE_FLOOR = 1e-6


def _np():
    try:
        import numpy as np
    except ImportError as exc:
        raise ToolDependencyError(
            "tools/analysis.py needs numpy, which is not installed") from exc
    return np


def _check_stack(gray):
    np = _np()
    if not isinstance(gray, np.ndarray):
        raise ValueError("gray must be a numpy ndarray of shape (T, H, W)")
    if gray.ndim != 3:
        raise ValueError(f"gray must have shape (T, H, W); got {gray.shape}")
    if 0 in gray.shape:
        raise ValueError(f"gray must not be empty; got shape {gray.shape}")
    if not (np.issubdtype(gray.dtype, np.integer) or np.issubdtype(gray.dtype, np.floating)):
        raise ValueError(f"gray must be uint8 or float; got dtype {gray.dtype}")
    return gray


def _check_index(index, count):
    try:
        i = int(index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frame index must be an integer; got {index!r}") from exc
    if not -count <= i < count:
        raise ValueError(f"frame index {i} out of range for {count} frames")
    return i % count


def brightness_stats(gray):
    np = _np()
    g = _check_stack(gray)
    means = np.array([g[t].mean(dtype=np.float64) for t in range(g.shape[0])],
                     dtype=np.float64)
    mean = float(means.mean())
    mn = float(means.min())
    mx = float(means.max())
    range_pct = (mx - mn) / mean * 100.0 if mean != 0 else 0.0
    T = means.size
    if T >= 2:
        x = np.arange(T, dtype=np.float64)
        xc = x - x.mean()
        slope = float(np.dot(xc, means - means.mean()) / np.dot(xc, xc))
    else:
        slope = 0.0
    return {
        "mean_per_frame": means.tolist(),
        "mean": mean,
        "std": float(means.std()),
        "min": mn,
        "max": mx,
        "range_pct": float(range_pct),
        "slope_per_frame": slope,
    }


def sharpness_stats(gray):
    np = _np()
    g = _check_stack(gray)
    T, H, W = g.shape
    if H < 3 or W < 3:
        raise ValueError(f"frames must be at least 3x3 for the Laplacian; got {H}x{W}")
    lap_var = []
    for t in range(T):
        f = g[t].astype(np.float64)
        lap = (4.0 * f[1:-1, 1:-1] - f[:-2, 1:-1] - f[2:, 1:-1]
               - f[1:-1, :-2] - f[1:-1, 2:])
        lap_var.append(float(lap.var()))
        del f, lap
    arr = np.asarray(lap_var, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    cv = std / mean if mean != 0 else 0.0
    return {"lap_var_per_frame": lap_var, "mean": mean, "std": std, "cv": cv}


def _validate_roi(roi, H, W):
    try:
        x, y, w, h = (int(v) for v in roi)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"roi must be (x, y, w, h); got {roi!r}") from exc
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
        raise ValueError(f"roi {(x, y, w, h)} does not fit inside the {W}x{H} frame")
    return x, y, w, h


def temporal_noise(gray, roi=None):
    np = _np()
    g = _check_stack(gray)
    T, H, W = g.shape
    if T < 2:
        raise ValueError(f"temporal noise needs at least 2 frames; got {T}")
    if roi is None:
        x, y, w, h = 0, 0, W, H
    else:
        x, y, w, h = _validate_roi(roi, H, W)
    # Welford's streaming mean and sum of squared deviations, one frame at a time.
    mean_acc = np.zeros((h, w), dtype=np.float64)
    m2 = np.zeros((h, w), dtype=np.float64)
    for k in range(T):
        xk = g[k, y:y + h, x:x + w].astype(np.float64)
        delta = xk - mean_acc
        mean_acc += delta / (k + 1)
        m2 += delta * (xk - mean_acc)
        del xk, delta
    std = np.sqrt(m2 / T).ravel()
    return {
        "pixel_std_mean": float(std.mean()),
        "pixel_std_p50": float(np.percentile(std, 50)),
        "pixel_std_p95": float(np.percentile(std, 95)),
        "pixel_std_p99": float(np.percentile(std, 99)),
        "pixel_std_max": float(std.max()),
    }


def _parabolic(left, centre, right):
    """Sub-pixel offset of a peak from three samples at -1, 0, +1."""
    den = left - 2.0 * centre + right
    if den == 0:
        return 0.0
    d = 0.5 * (left - right) / den
    return float(min(max(d, -0.5), 0.5))


def _phase_shifts(g, ref2d):
    """Phase correlation of every frame in g against ref2d.

    Positive dx means the frame content moved right relative to ref2d
    (frame(x) = ref(x - dx)); positive dy means it moved down.
    """
    np = _np()
    T, H, W = g.shape
    ref2d = np.asarray(ref2d, dtype=np.float64)
    if ref2d.shape != (H, W):
        raise ValueError(f"reference shape {ref2d.shape} does not match frames {(H, W)}")
    win = np.outer(np.hanning(H), np.hanning(W))
    ref_flat = bool(ref2d.max() == ref2d.min())
    R = np.conj(np.fft.rfft2((ref2d - ref2d.mean()) * win)) if not ref_flat else None
    out = []
    for t in range(T):
        f = g[t].astype(np.float64)
        if ref_flat or f.max() == f.min():
            # A flat image has no phase information: report no shift and no response.
            out.append({"dx": 0.0, "dy": 0.0, "response": 0.0})
            continue
        F = np.fft.rfft2((f - f.mean()) * win)
        cross = F * R
        mag = np.abs(cross)
        # Phase-only normalisation; the small floor stops round-off in empty bands from
        # being amplified to unit magnitude.
        cross /= mag + _PHASE_FLOOR * mag.max() + 1e-300
        r = np.fft.irfft2(cross, s=(H, W))
        py, px = (int(v) for v in np.unravel_index(np.argmax(r), r.shape))
        peak = float(r[py, px])
        dy_sub = _parabolic(r[(py - 1) % H, px], peak, r[(py + 1) % H, px])
        dx_sub = _parabolic(r[py, (px - 1) % W], peak, r[py, (px + 1) % W])
        iy = py if py <= H // 2 else py - H
        ix = px if px <= W // 2 else px - W
        out.append({"dx": float(ix + dx_sub), "dy": float(iy + dy_sub), "response": peak})
        del f, F, cross, mag, r
    return out


def estimate_shifts(gray, ref_index=0):
    g = _check_stack(gray)
    ref_index = _check_index(ref_index, g.shape[0])
    out = _phase_shifts(g, g[ref_index])
    out[ref_index]["dx"] = 0.0
    out[ref_index]["dy"] = 0.0
    return out


def residual_scores(gray, reference=None, align=True):
    np = _np()
    g = _check_stack(gray)
    T, H, W = g.shape
    ref_full = None
    if reference is not None:
        ref_full = np.asarray(reference, dtype=np.float64)
        if ref_full.shape != (H, W):
            raise ValueError(f"reference shape {ref_full.shape} does not match frames {(H, W)}")
    if align:
        shifts = _phase_shifts(g, ref_full) if ref_full is not None else estimate_shifts(g, 0)
        sy = [int(np.rint(s["dy"])) for s in shifts]
        sx = [int(np.rint(s["dx"])) for s in shifts]
    else:
        sy = [0] * T
        sx = [0] * T
    # Aligned frame k at (y, x) is frame_k at (y + sy_k, x + sx_k). Only the region where
    # every frame stays inside the frame is compared, so the wrapped border is excluded.
    y0 = max(0, -min(sy))
    y1 = min(H, H - max(sy))
    x0 = max(0, -min(sx))
    x1 = min(W, W - max(sx))
    if y1 <= y0 or x1 <= x0:
        raise ValueError("estimated shifts leave no overlapping area to compare")
    Hr, Wr = y1 - y0, x1 - x0
    if ref_full is not None:
        ref = ref_full[y0:y1, x0:x1]
    else:
        ref = np.empty((Hr, Wr), dtype=np.float64)
        rows = max(1, _CHUNK_ELEMS // (T * Wr))
        for a in range(y0, y1, rows):
            b = min(y1, a + rows)
            stack = np.stack([g[k, a + sy[k]:b + sy[k], x0 + sx[k]:x1 + sx[k]]
                              for k in range(T)])
            ref[a - y0:b - y0] = np.median(stack, axis=0)
            del stack
    per_frame = []
    p99 = []
    for k in range(T):
        fk = g[k, y0 + sy[k]:y1 + sy[k], x0 + sx[k]:x1 + sx[k]].astype(np.float64)
        ad = np.abs(fk - ref)
        per_frame.append(float(ad.mean()) / 255.0)
        p99.append(float(np.percentile(ad, 99)) / 255.0)
        del fk, ad
    pf = np.asarray(per_frame, dtype=np.float64)
    return {
        "per_frame": per_frame,
        "mean": float(pf.mean()),
        "std": float(pf.std()),
        "p95": float(np.percentile(pf, 95)),
        "max": float(pf.max()),
        "p99_abs_diff_per_frame": p99,
    }


def load_session_gray(session_dir, device):
    np = _np()
    path = os.path.join(os.fspath(session_dir), f"frames_dev{int(device)}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no frames file for device {device}: {path}")
    with np.load(path, allow_pickle=False) as z:
        if "gray" not in z or "stamps" not in z:
            raise ValueError(f"{path} must contain 'gray' and 'stamps' arrays")
        gray = np.array(z["gray"])
        stamps = np.array(z["stamps"], dtype=np.float64)
    _check_stack(gray)
    if stamps.shape != (gray.shape[0],):
        raise ValueError(f"stamps shape {stamps.shape} does not match {gray.shape[0]} frames")
    return gray, stamps
