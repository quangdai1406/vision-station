# Repeatability tools: shared formats (do not change without the orchestrator)

Tools need numpy (and opencv-python only for capture on the user's PC). They are NOT part of the station core, which stays standard-library only. Import numpy/cv2 lazily and raise a clear error (`ToolDependencyError`, defined in tools/__init__.py) when missing. Tests that need numpy use `unittest.skipUnless`.

## Session folder (written by tools/capture.py, read by tools/report.py)
<out>/<session_name>/
  session.json
  frames_dev<N>.npz      arrays: gray (T,H,W) uint8, stamps (T,) float64 epoch seconds
  samples/dev<N>_frame<idx>.png   colour PNGs: first, last and every Kth frame

session.json:
{ "schema": 1, "created_utc": "...Z", "mode": "stationary" | "placement", "session_name": str,
  "devices": [0], "notes": {"part": str, "lighting": str, "operator": str, "free": str},
  "requested": {"width": int, "height": int, "fps": number|null, "fourcc": "MJPG"|null,
                "manual": {"focus": number|null, "exposure": number|null, "white_balance": number|null, "gain": number|null}},
  "cameras": {"<dev>": {"actual_width": int, "actual_height": int, "actual_fps": number,
              "readback": {"autofocus": x, "focus": x, "auto_exposure": x, "exposure": x, "auto_wb": x, "white_balance": x, "gain": x},
              "applied_ok": {"autofocus": bool, "auto_exposure": bool, "auto_wb": bool, "focus": bool|null, "exposure": bool|null, "white_balance": bool|null, "gain": bool|null},
              "warnings": [str]}},
  "frames": [{"index": int, "device": int, "stamp": float, "mean": float, "laplacian_var": float}] }

## tools/analysis.py API (numpy only; gray = ndarray (T,H,W) uint8)
brightness_stats(gray) -> {"mean_per_frame": [..], "mean", "std", "min", "max", "range_pct", "slope_per_frame"}
sharpness_stats(gray) -> {"lap_var_per_frame": [..], "mean", "std", "cv"}   # 4-neighbour Laplacian variance over interior pixels
temporal_noise(gray, roi=None) -> {"pixel_std_mean", "pixel_std_p50", "pixel_std_p95", "pixel_std_p99", "pixel_std_max"}  # per-pixel std across frames; roi=(x,y,w,h)|None
estimate_shifts(gray, ref_index=0) -> [{"dx": float, "dy": float, "response": float}, ...]   # phase correlation, per frame vs reference
residual_scores(gray, reference=None, align=True) -> {"per_frame": [..], "mean", "std", "p95", "max", "p99_abs_diff_per_frame": [..]}  # per_frame = mean absolute difference vs reference (default: per-pixel median of all frames), after integer alignment, scaled to 0..1 by /255
## tools/thresholds.py: input CSV columns label(good|bad), score (float, higher = more defective), optional defect_class, part_id.
## tools/criteria.py: MES GET /api/machine/criteria/<part> returns a JSON list of {id, scope_type, scope_value, criteria_name, criteria_description, check_type('visual'|'measurement'), measurement_uom, measurement_target, measurement_tolerance, standard_picture_path, active}.

## Additive session.json keys (added in round 2; consumers must tolerate their absence)
requested.flush_frames (int): frames grabbed and discarded before each real capture
cameras.<dev>.buffersize_readback: read-back of CAP_PROP_BUFFERSIZE (best effort, set to 1)
cameras.<dev>.auto_exposure_attempts: [{"set": x, "readback": y}, ...] every auto-exposure value tried
cameras.<dev>.exposure_response: detail of the exposure response check, or null
cameras.<dev>.applied_ok.exposure_responds: true | false | null (null = not requested/skipped); false means the picture did not respond to an exposure change (the camera may ignore manual exposure; DirectShow read-back can echo the value you set)

## Stationary noise floor (report.py)
Per-frame residual against the median frame, NO alignment (alignment can fit sensor noise and hides real vibration). Placement sessions report both aligned and unaligned residuals.

## Settle check (capture.py, added after the first real run)
Runs after the timed warm-up and before the first recorded frame (placement: before the first prompt). Options: --settle-max-s (default 30, 0 disables), --settle-frames (default 5), --settle-tol-gray (default 1.0). These are engineering settings meaning "brightness stopped changing", not quality thresholds. Check frames are discarded, never recorded.
cameras.<dev>.settle: {"checked": bool, "settled": bool, "seconds": float, "frames_used": int, "last_means": [float], "tol_gray": float, "frames": int, "max_s": float}  (absent if the run aborted before the check)
An unsettled camera adds a warning to cameras.<dev>.warnings. The end-of-run summary also hints when the first recorded frame is >10 % away from the median brightness.

## report.py additions
Options: --outlier-sigma (default 5.0), --outlier-floor-gray (default 2.0): a stationary frame is FLAGGED when |mean - median| > max(sigma * 1.4826 * MAD, floor). Flagging only helps you spot frames; nothing is dropped unless you pass --skip-first N. When frames are flagged, the report shows all-frames vs excluding-flagged side by side. --min-shift-response (default 0.1): the apparent drift line is hidden below this phase-correlation peak response and the real value is printed.
summary.csv extra columns: frames_used, skip_first, flagged_frames, flagged_frame_indices and six excl_flagged_* columns (empty when nothing is flagged). report.json gains "settings" and the median shift response.

