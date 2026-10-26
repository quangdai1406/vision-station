# Vision station

The camera inspection station for the liner line. It takes 8 images per part
(4 cameras x top and bottom), scores them, fuses the scores into GOOD, BAD or
REVIEW, stores the evidence on disk, and posts each finished part to the MES.
It is standard-library Python. The web UI is served by the station itself on
127.0.0.1.

Plan: `26_1023_inspection_system_plan.md`. This code implements sections
**3** (inspection flow and states), **5** (decision rules and REVIEW) and
**7** (storage, outbox and MES integration). The MES side of the contract is
`/home/user/aisi_mes/docs/vision-station-integration.md`.

## Install

Nothing to install for the core: standard library only (tested on Python 3.13).

Optional, only on the real station:

- `opencv-python` and `numpy`: real USB cameras (`cameras.kind: opencv`) and the
  sharpness gate (`frame_ok`).
- `onnxruntime`: real model inference (not wired yet, see below).

## Configuration

Copy `config.example.json` to `config.json` and replace every `[bracketed]`
value. The station refuses to start while a placeholder is left, and names it.

| Key | Meaning |
|---|---|
| `site`, `line`, `station_id`, `part_number`, `lot` | Identity written to storage and to the MES. `station_id` must be registered in the MES. |
| `operator`, `supervisor`, `material_handler` | Names for the shift, sent with every part. |
| `thresholds` | `t_low` and `t_img`, 0..1, `t_low <= t_img`. **Owned by Quality; there is no default.** Score `>= t_img` is BAD, `t_low <= score < t_img` is REVIEW. |
| `data_dir` | Where parts, `station.db`, `outbox/` and `outbox.db` live. Relative paths are relative to the config file. |
| `users_file` | JSON file of logins, written by `python -m station users add`. |
| `ui.host`, `ui.port`, `ui.session_timeout_s` | Default `127.0.0.1`, `8000`, `900`. |
| `mes.base_url` | MES root, for example `http://127.0.0.1:3000`. |
| `mes.token_env` | **The name** of the environment variable that holds the MES token. The token is never written in the config. |
| `mes.criteria_ids` | The MES criteria ids for the part (`GET /api/machine/criteria/:partNumber`). |
| `mes.class_to_criterion` | Defect class from the model to criterion id. Quality/engineering decide this. |
| `mes.default_criterion_id` | Used for a defect class that has no mapping. `null` means such a part is not posted (the error is logged). |
| `mes.sync_interval_s`, `mes.default_bad_disposition` | Outbox drain period (default 5 s); `quarantine` or `released` (default `quarantine`). |
| `cameras` | `kind` `fake` or `opencv`, `devices` (4 indexes for C1..C4), `width`, `height`, `manual` (focus, exposure, white_balance; `null` = auto). |
| `model` | `kind` `marker`, `onnx`, `golden` or `collect`, `path`, `version` (sent to the MES as `modelVersion`). |
| `outputs` | `kind` `console` or `serial_relay` (with `port`, optional `channel_map`). |
| `flip_timeout_s` | Seconds the station waits for the flip press before voiding a part (default 60). |
| `frame_ok.min_variance` | Optional sharpness gate; a blurred frame is retaken (up to 2 times). |

## Run it

Demo mode uses fake cameras, the marker model and console outputs. Parts are
stored but never queued for the MES. Use a copy of the config with its own
`data_dir`, so demo parts do not mix with real counters:

    python3 -m station --config config.json --demo

It prints the UI URL (for example `http://127.0.0.1:8000/`). Log in, press
**Press** once for the top side, turn the part, and press again for the bottom.
Ctrl-C stops the station cleanly.

Normal mode is the same without `--demo`, with the real cameras, model and
outputs from the config.

### Collect mode (no verdict)

Set `"model": {"kind": "collect", "version": "..."}` with `"parts": {"mode": "fixed"}`
to run the real cameras and save evidence without any model verdict. Each part
is saved as usual (8 images, `part.json`, `manifest.sha256`) with final verdict
`REVIEW` and `disposition` `collected`. The screen shows `COLLECTED (not judged)`,
no light is lit green or red, the counters count the part in total only, and
nothing is queued for the MES. The per-image `score` in `part.json` is a
placeholder `0.0` and must not be read as a result.

## Create users

    python3 -m station users add alice operator
    python3 -m station users add shiftlead supervisor
    python3 -m station users add itadmin admin

The password is asked for twice and must be at least 8 characters. It is
hashed with scrypt and never printed. Running `users add` for an existing
name updates its role and password. Roles:

- `operator`: press, acknowledge a BAD alarm.
- `supervisor`: all of the above, plus resolve REVIEW, override BAD, clear faults.
- `admin`: all actions.

Use `--users-file FILE` to write a users file without a config.

## MES token

On the MES machine, from the MES repo:

    npm run migrate                      # once: adds machine_tokens and the vision columns
    npm run machine-token -- station-1   # prints the bearer token ONCE; stored hashed

Then, on the station machine, export it in the variable named by `mes.token_env`:

    export STATION_MES_TOKEN='...the printed token...'
    python3 -m station --config config.json

Revoke with `npm run machine-token -- --revoke station-1`. Without the variable
the station still inspects and stores parts, but logs `mes_disabled` and does
not post.

## How it works end to end

1. **Press (top).** The session captures one frame from each camera, runs the
   model, and waits for the flip (state `WAIT_FLIP`).
2. **Press (bottom).** The same for the bottom side. The 8 results are fused:
   any BAD image makes the part BAD; otherwise any REVIEW image makes it REVIEW;
   otherwise GOOD.
3. **Save first.** GOOD is saved to `data_dir/store/...` (8 PNGs, `part.json`,
   `manifest.sha256`, a row in `station.db`) before the green light turns on or
   the counter moves. A save failure goes to FAULT and is retried by "Clear
   fault", never shown as green.
4. **BAD.** Red and alarm come on; the part is saved when an operator acknowledges
   the alarm (`/api/ack`).
5. **REVIEW.** Amber only. A supervisor resolves it to GOOD or BAD with a reason
   (`/api/resolve`). Only then is the part saved.
6. **Queue.** When a part completes, its MES record (fields plus, for BAD, one
   `defectPhoto_<criterion>` per failed criterion) is written to the outbox
   (`data_dir/outbox.db` and `data_dir/outbox/<session>/`). Raw images stay on
   the station; the MES gets `archiveRef`, the path of the saved part.
7. **Sync.** Every `sync_interval_s` the outbox sends due items, oldest first. A
   network error, a 408, 429 or 5xx, or a 401/403 keeps the item pending with
   exponential backoff (a 401/403 means the token must be fixed). A 400, 404 or
   422 (for example, the verdict disagrees with the checklist) marks it
   `failed_permanent`, and it is not retried. Every POST carries `Idempotency-Key` = the session id, so a
   retry can never create a second MES event.
8. **Restart.** Counters come back from `station.db`, and unsent outbox items
   are still there.

Events (press, capture, verdict, ack, override, review_resolved, fault, mes_sync,
mes_enqueue_failed, ...) are appended to the `events` table, which cannot be
updated or deleted.

## Tests

    python3 -m unittest discover -s tests -v

`tests/test_e2e.py` runs the whole chain against a local fake MES: GOOD, BAD
with ack, REVIEW with supervisor resolution, MES down then back, 422, and restart.
`tests/test_app.py` covers config validation, part ids, MES disabled without a
token, enqueue failures, restart, the users command and the CLI.

## Safety rules this code enforces

- **REVIEW is never green.** REVIEW shows amber and no alarm, and goes to
  SUPERVISOR_REVIEW. Only a supervisor resolution can end it, and the MES is
  told about it only after that.
- **Any-of-8.** One BAD image makes the part BAD, and one REVIEW image makes it
  REVIEW. A part is GOOD only when all 8 images are below `t_low`.
- **Save before green.** The green light and the counters change only after the save succeeds. Red (with the alarm) and amber show immediately, but the part is saved only on acknowledge or resolve. A light-driver failure is logged as an outputs_failed event and never blocks saving.
- **Idempotency.** `Idempotency-Key` is the session id on every MES POST, and the
  outbox is keyed by session id. A part is never queued twice.
- **REVIEW is resolved before MES.** The MES never receives REVIEW; the final
  verdict and the override (`overrideOriginalVerdict`, `overrideReason`,
  `overrideBy`) are posted together.
- **Roles on every action.** The server checks the role, not only the UI, and a
  disallowed action returns 403.
- **Secrets stay out of files.** The token comes from the environment, is never
  logged, and is redacted from error text. Passwords are stored as scrypt hashes.
- **Local only.** The UI binds to 127.0.0.1 and sends a `SameSite=Strict`
  HttpOnly cookie. The MES client does not follow redirects, so the token cannot
  be sent elsewhere.

## Not done / needs real hardware or decisions

- **Validating the webcams.** `cameras.kind: opencv` has not been run on the
  real USB cameras. Device indexes, exposure and focus settings, and the 4-camera
  grab skew need checking on the line.
- **Real model.** No model is trained. `model.kind: onnx` is refused at build
  time because the ONNX pre- and postprocessing are not written. Only the
  `marker` model (demo and tests) works.
- **GPIO and relay wiring.** `serial_relay` writes `ON<n>`/`OFF<n>` bytes to the
  port, with no baud rate or line settings. The channel map and the alarm
  acknowledge method (button, or only the screen) are not decided.
- **Part-ID scanner.** Part ids are generated by the station (`P` + time + counter).
  The real part identity and lot are typed into the config. A scanner is not wired.
- **Thresholds from Quality.** `t_low` and `t_img` are placeholders until Quality
  signs them off. There is no default on purpose.
- **Class-to-criterion mapping.** `mes.class_to_criterion` needs the model's defect
  classes and the MES criteria for the part. Until then, a BAD part with an
  unmapped class is not posted (`mes_enqueue_failed` is logged).
- **TLS and Secure cookie.** Served over plain HTTP on 127.0.0.1 only. If the UI is
  ever reached from another machine, put it behind TLS and add the `Secure` flag.
- **Login rate limiting.** There is none. Keep the UI on localhost until it is added.
- **Sharpness gate speed.** `frame_ok` converts full frames to Python lists, which
  is slow at 2560x1440. Use it only after measuring on the line.
- A saved part whose MES enqueue failed (for example an unmapped defect class) is not re-queued automatically; look for mes_enqueue_failed events and fix the mapping, then re-enqueue by hand.
- The Outbox's long-lived connection must be closed from the thread that created it (StationApp.stop() on the main thread); the app opens a fresh Outbox per call elsewhere.

## Repeatability and threshold tools

The operator procedure for the real-webcam test is
[docs/repeatability_protocol.md](docs/repeatability_protocol.md). These tools need numpy
(and opencv-python for capture). They are not part of the station core.

- `tools/capture.py`: captures a stationary or placement session from USB webcams.
- `tools/report.py`: writes a plain-language report (`report.md`, `report.json`, `summary.csv`) from session folders.
- `tools/criteria.py`: turns the MES criteria for a part into `class_map.json` and a sample-collection checklist.
- `tools/thresholds.py`: computes candidate `t_low` and `t_img` from a labelled scores CSV. The values are candidates, not approved thresholds.
- `tools/analysis.py`: the numpy-only measurements that `report.py` uses.

## One-camera proof of concept

To see the flow work with one USB webcam on a Windows PC (top picture, flip,
bottom picture, GOOD or BAD on screen), follow
[docs/poc_quickstart.md](docs/poc_quickstart.md). It is a demo: two images per
part, a demo golden-image model, demo thresholds with no quality meaning, and
no MES posting.
