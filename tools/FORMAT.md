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


## Scan record (Taking data mode; written by station/scanstore.py, read by its verify_scan/load_scan)
Standard library only. A scan is kept apart from inspected parts: it has no part verdict and never goes to the MES. Plan: docs/26_1027_PLAN_taking_data_mode.md.

```
<root>/scans/<line>/YYYY/MM/DD/<scan_id>/
  <site>_<line>_<YYYYMMDD>_<HHMMSS>_<scan_id>_<CAM>_<SIDE>_S<step:02d>.<ext>
  scan.json          the record (below)
  manifest.sha256    sha256sum text format: one "<sha256>  <file>" line per picture and one for scan.json; written LAST
```
- The date and time in the picture names come from `started_utc`, in UTC. `CAM` is a camera id (C1..C4), `SIDE` is TOP or BOTTOM, `step` is 1-based and contiguous per side, `ext` defaults to png.
- While a scan is being written it lives in the hidden folder `.<scan_id>.partial` inside its day folder. It is renamed to `<scan_id>` only when complete. A folder whose name starts with `.` is never a finished scan: readers (`list_scans`) skip it, `verify_scan` reports it.
- `scan.json` keys: `scan_id`, `site`, `line`, `part_number`, `toy_id`, `label`, `steps`, `taken_by`, `started_utc`, `finished_utc` (both `YYYY-MM-DDTHH:MM:SSZ`), `camera_ids` (list), `images` (list). `site` and `line` are copied from the `ScanStore` that wrote the scan; each picture name must start with them.
  - `label`: `verdict` GOOD or BAD; `defects` (list of defect names: empty for GOOD, at least one for BAD); `status` proposed, official or rejected; `labelled_by`, `labelled_at`; `approved_by`, `approved_at` (both or neither; required for rejected); `reject_reason` (rejected only). Labels are never changed by this writer.
  - `steps`: `{"TOP": n, "BOTTOM": m}`. A side that was not scanned is absent. Each side that is present has n >= 1 and steps 1..n with no gaps.
  - `images`: one object per picture: `file`, `camera_id`, `side`, `step`, `sha256`, `defect_visible`. For a BAD scan `defect_visible` is true or false on every picture. For a GOOD scan it is null on every picture.
  - Each step has exactly one picture per listed camera.
- Writers: `ScanStore(root, site, line).begin_scan(...)` returns a `ScanWriter`; `add_picture(...)` writes each picture as it arrives; `finish(finished_utc)` writes scan.json, then the manifest, then renames the folder; `cancel()` removes the staging folder.
- Scan session (`station/scan.py`, `ScanSession`) drives one scan from the screen: states IDLE, SCAN_SIDE, WAIT_FLIP, SAVING, DONE, FAULT. A step is all cameras or nothing; a camera failure goes to FAULT with the camera named, keeps the steps taken, and a press retries the same step. A failed picture write drops the whole scan (the writer's rule), so clear_fault() returns to IDLE. Only DONE is a saved scan. `ScanWriter.set_defect_visible(camera_id, side, step, value)` changes one BAD picture until finish().
  - Flip gap: after "Done with this side" the part cannot be flipped for `min_flip_s` seconds (default 2.0, an engineering setting, 0 turns it off). A press in that time is ignored and captures nothing, so a double tap cannot photograph the unflipped part as BOTTOM.
  - Known limitation 1: a disk write error during a scan discards the whole scan. The store drops the scan's staging folder, so none of its pictures are kept.
  - Known limitation 2: `press()` holds the session lock while the cameras capture, so a `snapshot()` call waits until that capture ends.
- Readers: `verify_scan(scan_dir)` returns a list of problems (empty = intact: missing, extra or altered file, manifest or scan.json mismatch, label or step rules broken, symlink, staging folder). `verify_scan` checks each picture's full name (`site_line_date_time_scanid_CAM_SIDE_Sxx.ext`) against scan.json, and that the folder sits at `<line>/YYYY/MM/DD` as scan.json says. `load_scan(scan_dir)` returns the validated scan.json dict or raises ValueError. `list_scans(root, line=None)` returns finished scan folders only.
- Staging folder blocks the id: while a leftover `.<scan_id>.partial` folder exists, that `scan_id` cannot be used again (`begin_scan` refuses it, and `verify_scan` reports the staging folder). After a crash, the operator removes that hidden folder by hand. There is no code for this yet.
- HTTP (station/ui.py, `scan` role = supervisor and admin; login required; 404 when scan is disabled, 403 for an operator): POST `/api/scan/start` {toy_id, part_number?, label {verdict, defects}, sides?}, `/api/scan/press`, `/api/scan/done-side`, `/api/scan/mark` {camera_id, side, step, visible}, `/api/scan/finish`, `/api/scan/cancel`, `/api/scan/clear-fault`, `/api/scan/reset`; GET `/api/scan/state` and `/api/scan/step-picture?camera=C1` (a PNG/JPEG, 204 when none). The label status and labeller come from the login (supervisor: proposed, admin: official); client values for them are ignored. Scan responses carry no folder path.
- Copying or exporting a scan: `manifest.sha256` lists `scan.json` as well as the pictures. A copy or export tool must copy the whole scan folder, or copy the files the manifest lists and check them against it. It must not write a new manifest from the pictures alone, because that would drop the scan.json hash and the record would no longer match what was written.
