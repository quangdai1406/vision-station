# Office checklist (first day at the office)

Status: written 2026-10-26 from the plan in `docs/26_1026_PLAN_pi_station_progress.md`. Nothing here has been run
against the real MES or the laptop. Numbers and limits that belong to Quality are `[Quality to define]`.

The Windows training laptop is also the AISI MES server. The Pi must reach it over the office network.

## A. Before you start (2 minutes, on the Pi)

1. `cd ~/vision-station && git status` shows a clean tree on `main` (merged work is local; nothing pushed).
2. Install node if not done: `sudo apt install -y nodejs`, then run the UI tests.
3. Plug the 4 cameras as in the home test (two on each USB 2.0 root). Run
   `python -m tools.capture_speed --devices 0 2 4 6 --width 1280 --height 720 --rounds 10 --pause 1.0 --mjpg --threaded --flush 1`
   and check 0 failures.

## B. Network: can the Pi reach the laptop/MES?

1. Get the laptop's address (PowerShell on the laptop: `ipconfig`). Prefer a fixed address or a hostname.
2. On the Pi: `ping <laptop>`; then check the MES port, for example `curl -s -o /dev/null -w "%{http_code}\n" http://<laptop>:<port>/`.
3. If it fails, check the Windows firewall on the laptop (allow the MES port, and SMB 445 only if you use a share).
4. The station UI stays on 127.0.0.1 (no TLS, no login rate limit): do not expose it to the network.

## C. MES token and part search (step 1.1)

1. Ask the MES administrator for a machine token. Put it ONLY in an environment variable on the Pi
   (for example `export STATION_MES_TOKEN=...` in the shell that starts the station). Never write it in a file.
2. In `config.json`: `mes.base_url`, `mes.token_env` (the variable name), `parts.mode: "mes"`.
3. Do one part search from the station page and confirm a real part number appears.
4. Pull the criteria: `python -m tools.criteria` (see its `--help`) for the part, and fill
   `mes.criteria_ids` and `mes.class_to_criterion` from the output.

## D. Collect (steps 1.3, 1.4b)

1. Start the station with `model.kind: "collect"`. Collect GOOD parts first, then BAD parts per defect class.
2. Operators propose labels; an admin reviews and makes them official. Only official labels reach training.
3. Counts to collect: `min_good`, `min_holdout`, `holdout_fraction` = `[Quality to define]`.
4. Unplug one camera during a run and plug it back (step 1.3c). Expect the station to name the camera and recover.

## E. Move evidence to the laptop (step 2.1)

1. Choose one: SMB share mounted on the Pi, `scp`/`rsync` over SSH, or a USB drive. See `docs/export_evidence.md`.
2. Dry run first: `python3 -m tools.export_evidence --source <pi data dir> --dest <mounted folder> --dry-run`.
3. Real run, then read `export_report.json` (counts copied, skipped, failed). Exit code 1 means corrupt or failed parts.

## F. Laptop setup and first training (steps 2.2, 2.3)

Follow `docs/weekly_training.md` sections 2 and 3 (PowerShell): NVIDIA driver, Python, CUDA torch,
`trainer.json` with no placeholders, then `py -m trainer selftest` (paste the full output into the setup record).
Then one manual `python -m trainer weekly`. A candidate is NOT used by any station until Quality approves and a
named person runs `python -m trainer promote`.

## G. Pi model speed (step 3.1)

Measured on this Pi 5 (resnet18, 4 threads, 8 real 1280x720 pictures, seconds for 8):
320x320 1.10, 360x640 2.37, 432x768 3.35, 480x640 3.09, 576x1024 6.71. Only 320x320 and 360x640 fit 3 s.
Whether small defects are still found at 360x640 is unmeasured: test it with the labelled pictures once collected.
The torch environment on the Pi is `/home/quangdai1406/.venvs/torchpi` (CPU torch, 965 MB).

## H. Bring back

Any dropped frame, long wait, error line or surprise from sections B to F, and the selftest output.
