# Plan for the office day (written 2026-10-26, evening)

Continue from here. Master tracker: `docs/26_1026_PLAN_pi_station_progress.md`. Detailed commands:
`docs/office_checklist.md`. Numbers and limits that belong to Quality are `[Quality to define]`. Nothing
here invents a threshold or a sample count.

## Where we stand

- Pi 5 station with 4 USB cameras (device indexes 0, 2, 4, 6; MJPG 1280x720; keep-open, threaded, flush 1).
- Plan step 1.3b is DONE. Collect mode ran 20 parts on the real cameras: 8 PNGs + `part.json` +
  `manifest.sha256` each, all 160 files verify, verdict REVIEW / `collected`, nothing queued for the MES,
  40 of 40 presses accepted (10 with a 2.5 s flip wait, 10 back-to-back at about 2.0 s), no faults.
  Press time: top 0.13 s mean (max 0.14), bottom 0.15 s mean (max 0.17). Checked by a Sonnet checker (PASS).
- Pictures after the light and aim fixes: mean brightness 124-129 of 255 on every camera, no overexposure,
  1-9 percent nearly black pixels (shadowed box wall). The car (test object) is fully in frame on all four
  cameras. The evenness limit is `[Quality to define]`.
- The test used ONE toy car in ONE position for all 20 parts. It proves capture, saving and speed. It says
  nothing about GOOD versus BAD.
- Thresholds in the throwaway test config (0.3 and 0.6) are placeholders with no quality meaning.
- Live view helper (not in the repo): `/home/quangdai1406/vision-liveview/liveview.py`, serves a 2x2 live
  view with a centre cross and a brightness readout at http://127.0.0.1:8090/. Stop the station first;
  only one program can hold the cameras.
- Evidence copy on the Pi: `~/Pictures/vision-station-collect-test-26_1026/` and pictures in
  `~/Pictures/vision-liveview/`.

## Decisions still open (owner: user)

| ID | Decision | Recommended | Answer |
|---|---|---|---|
| D-A | Defect names or GOOD/BAD only first | GOOD/BAD first (anomaly model) | open |
| D-B | Model runs on the Pi | Yes; lighter model if too slow | open |
| D-C | Pi to laptop transfer: SMB share, scp/rsync or USB drive | SMB share or scp | open |

## Before leaving the house / on arrival (5 min)

1. Do not move the cameras or the light. If they move, redo the 20-part collect test.
2. Optional: fix exposure and focus with `cameras.manual` so the office light does not change the pictures.
3. Plug the 4 cameras as at home (two on each USB 2.0 root). Run
   `python -m tools.capture_speed --devices 0 2 4 6 --width 1280 --height 720 --rounds 10 --pause 1.0 --mjpg --threaded --flush 1`
   and expect 0 failures.

## Order of work at the office

| # | Step | Plan step | Owner | Verify |
|---|---|---|---|---|
| 1 | Network: find the laptop address, `ping`, `curl` the MES port; check the Windows firewall | B | user | the Pi reaches the MES |
| 2 | Get a machine token from the MES administrator. Put it only in an environment variable | 1.1 | user | token never in a file or a log |
| 3 | `config.json`: `mes.base_url`, `mes.token_env`, `parts.mode: "mes"`; search one real part number on the station page | 1.1 | user | one live search works |
| 4 | Pull criteria with `python -m tools.criteria`; fill `mes.criteria_ids` and `mes.class_to_criterion` | 1.2 | Claude | mapping complete for the first part |
| 5 | Ask Quality for `min_good`, `min_holdout`, `holdout_fraction` | 1.5 | Quality | written numbers `[Quality to define]` |
| 6 | Collect REAL GOOD parts in `model.kind: "collect"`: many different parts, not one part repeated | 1.3 | user | counts per Quality, all 8 images each |
| 7 | Collect REAL BAD parts per defect type. Operator proposes the label, admin makes it official | 1.4b | user | each BAD part has an official label matching an MES criterion |
| 8 | Unplug and replug one camera during a run | 1.3c | user, Claude watches | the station names the camera and recovers |
| 9 | Dry-run then real `python3 -m tools.export_evidence` to the laptop; read `export_report.json` | 2.1 | user | counts and manifests match, exit code 0 |
| 10 | Laptop setup per `docs/weekly_training.md`: NVIDIA driver, Python, CUDA torch, `trainer.json`; `py -m trainer selftest` | 2.2 | user | shows the GPU and True |
| 11 | First candidate: `python -m trainer weekly` | 2.3 | laptop | candidate in `models\`, report in `reports\` |

A candidate model is NOT used by any station until Quality approves it and a named person runs
`python -m trainer promote --part PN --version V --by "Name" --reason "..."`.

## Not for tomorrow

- Thresholds `t_low` / `t_img` (3.4, 3.5): need labelled held-out scores from a trained model, then Quality
  approves in writing `[Quality to define]`.
- Alarm and light wiring (0.4): `serial_relay` channel map and the alarm acknowledge method.
- Quality accepting MJPG pictures as evidence (0.3).
- Shadow run and live run (Phase 4).
- Defect-name model and ONNX wiring (2.4, 3.3), only if D-A = names.

## Risks and checks to remember

- Pi model speed: resnet18 at 360x640 takes about 2.37 s for 8 pictures (fits a 3 s budget). Whether small
  defects are still found at that size is unmeasured. Test it with the real labelled pictures from steps 6 and 7.
- Few BAD examples limit defect naming; the anomaly model only needs GOOD parts.
- A saved part whose MES enqueue failed (for example an unmapped class) is not re-queued automatically.
  Look for `mes_enqueue_failed` events.
- After a station restart the part's own MES criteria are not reloaded until the part is selected again.
- The UI stays on 127.0.0.1 (no TLS, no login rate limit). Do not expose it to the network.

## Bring back

Any dropped frame, long wait, error line or surprise from steps 1-11, the `export_report.json`, and the
selftest output.
