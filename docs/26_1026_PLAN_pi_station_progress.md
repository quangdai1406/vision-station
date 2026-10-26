# Plan: Pi 5 station -> collect -> train on Windows laptop -> inspect (progress checker)

Created 2026-10-26. This file is the progress checker. Update the Status column and the Log at the bottom
after each step. Status values: `TODO`, `DOING`, `DONE`, `BLOCKED`.

Rules (from CLAUDE.md): `station/` is standard library only; `station/contracts.py` is not changed
without the user's say-so; REVIEW is never green and never posted; any BAD image of the 8 makes the part
BAD; never invent thresholds or counts (write `[Quality to define]`); no secrets in files or logs;
`python3 -m unittest discover -s tests` must pass; regenerate `docs/CODEGRAPH.md` after code changes.

## Facts

- Hardware: Raspberry Pi 5 (4 cores, 15 GB RAM), touch screen, 4 USB 2.0 cameras split across two USB 2.0
  roots, 1280x720 MJPG. Four pictures take about 0.2 s with keep-open + MJPG + threaded capture
  (branch `feature/keep-open-threaded-capture`, uncommitted).
- Repo design: PatchCore anomaly model trained on GOOD parts only (`station/anomaly.py`), weekly trainer
  on the laptop (`trainer/`), per-part model store with approval and rollback (`station/modelstore.py`).
  `model.kind: onnx` is refused at build time (`station/app.py`). The `marker` model is a demo only.
- Part numbers and defect classes come from the AISI MES (`station/parts.py`, `tools/criteria.py`).
- Training laptop: Windows, NVIDIA GPU. Torch and onnxruntime are NOT installed on the Pi yet.
- Targets: 4 pictures per side in <= 1 s; model check of 8 pictures in about 2-3 s.

## Decisions (owner: user)

| ID | Decision | Recommended | Answer |
|---|---|---|---|
| D-A | Defect names (scratch, burn, ...) or GOOD/BAD only first | GOOD/BAD first (anomaly model) | open (label field will carry an optional MES criterion id) |
| D-B | Model runs on the Pi | Yes; fall back to a lighter ONNX model if too slow | open |
| D-C | Pi -> laptop transfer (SMB share, scp/rsync, USB drive) | SMB share or scp | open |

## Phase 0 - Station ready

| # | Step | Verify | Owner | Status |
|---|---|---|---|---|
| 0.1 | Review and merge `feature/keep-open-threaded-capture` | unit tests pass | user reviews, Claude merges | TODO |
| 0.2 | Fix mount, lighting, focus, exposure; set `cameras.manual`; name the four corners; run the repeatability study | same part looks the same in 10 presses | user | TODO |
| 0.3 | Quality accepts MJPG pictures as evidence | written sign-off | Quality | TODO |
| 0.4 | Decide lighting/alarm wiring (`serial_relay`, channel map) | red/green/alarm work on a test press | user | TODO |

## Phase 1 - Collect on the Pi

| # | Step | Verify | Owner | Status |
|---|---|---|---|---|
| 1.1 | `parts.mode: "mes"`, MES token in env var, search real part numbers | one live search works | user | TODO |
| 1.2 | Pull criteria with `tools/criteria.py`; build `class_to_criterion` for the first part | mapping complete | Claude | TODO |
| 1.3 | Build a "collect only" mode (no fake verdict) that saves 8 images + part.json per part | 20 parts saved with all 8 images, tests pass | Claude builds, user runs | DONE (code, 1118 tests). Real-camera 20-part run still pending |
| 1.4 | Build BAD tagging: operator picks the defect type (MES criterion) on a BAD part | each BAD part has a label matching an MES criterion id | Claude builds, user runs | DONE (code, 1144 tests). Real-camera labelling not run; page JS not syntax-checked (no node) |
| 1.5 | Counts: `min_good`, `min_holdout`, `holdout_fraction` | written numbers | Quality `[Quality to define]` | TODO |

## Phase 2 - Move data and train (Windows laptop)

| # | Step | Verify | Owner | Status |
|---|---|---|---|---|
| 2.1 | Pi -> laptop transfer script (per D-C) | file counts and `manifest.sha256` match | Claude writes, user runs | DONE (tools/export_evidence.py, docs/export_evidence.md). Real transfer not run |
| 2.2 | Laptop setup per `docs/weekly_training.md`: driver, Python, CUDA torch, `trainer.json` | `py -m trainer selftest` shows GPU and True | user | TODO |
| 2.3 | First candidate: `python -m trainer weekly` | candidate in `models\`, report in `reports\` | laptop | TODO |
| 2.4 | Optional (D-A = names): small classifier on labelled BAD pictures, export ONNX, check with onnxruntime on the Pi | same scores laptop vs Pi within a tolerance Quality accepts | Claude | TODO |

## Phase 3 - Run the model on the Pi

| # | Step | Verify | Owner | Status |
|---|---|---|---|---|
| 3.1 | Install CPU torch on the Pi; run `python -m trainer selftest` there (measure seconds per image) | seconds/image x 8 fits the 2-3 s budget, else lighter extractor or ONNX | Claude | DONE: target NOT met at 1280x720 (see Log) |
| 3.2 | Copy approved model to the Pi; `python -m trainer promote --part PN --version V --by "Name" --reason "..."` | station shows model version on screen and in part.json | named person | TODO |
| 3.3 | If a defect-name model exists: wire `model.kind: onnx` (pre/post-processing) | unit tests + Pi timing run | Claude | TODO |
| 3.4 | Candidate `t_low`, `t_img` from held-out scores with `tools/thresholds.py` | score table with candidate pair | Claude | TODO |
| 3.5 | Quality approves thresholds | written sign-off | Quality `[Quality to define]` | TODO |

## Phase 4 - Shadow run, live, weekly cycle

| # | Step | Verify | Owner | Status |
|---|---|---|---|---|
| 4.1 | Shadow run: operator judges, model only logs; compare | disagreement list reviewed by Quality | operators, Quality | TODO |
| 4.2 | Live on one part number; REVIEW never green, never posted | 20 live parts logged, MES posts correct | user | TODO |
| 4.3 | Failure tests: unplug/replug camera, long run, back-to-back presses with no pause | no stuck state; failing camera named | Claude | TODO |
| 4.4 | Weekly cycle: Sunday train, Monday Quality review, manual promote, rollback ready | Monday report reviewed | Quality | TODO |
| 4.5 | Add part numbers one at a time | each one repeats 4.1-4.2 | user | TODO |

## Risks

- Pi inference speed is unproven until 3.1; it can change the design.
- Few BAD examples limit defect naming; the anomaly model needs only GOOD parts.
- Strict-GOOD training may bias toward easy parts (repo item D5). Retention of GOOD pictures: `[Quality to define]`.
- Unproven in the repo: torch extractor, GPU laptop scripts, live MES search.

## Log

| Date | Step | Note |
|---|---|---|
| 2026-10-26 | - | Plan created. |
| 2026-10-26 | 1.3, 2.1, 3.1 | Round 1 dispatched (3 Haiku workers, parallel). Decisions D-A..D-C still open; defaults used: GOOD/BAD first, model on Pi, scp/rsync transfer. |
| 2026-10-26 | 1.3 | Round 1 PASS (Sonnet checker). New model.kind "collect": saves 8 images + part.json, verdict REVIEW / disposition "collected", no green, no MES post. Fake cameras only. contracts.py untouched. |
| 2026-10-26 | 2.1 | Round 1 PASS. tools/export_evidence.py: verifies manifest, copies ok parts, incremental, never deletes. 30 tests. Not run against a real share. |
| 2026-10-26 | 3.1 | Round 1 PASS. Venv /home/quangdai1406/.venvs/torchpi (CPU torch 2.14.1, 965 MB). resnet18 at 320x320: 0.149 s/image (about 1.2 s for 8). At 1280x720: resnet18 10.5 s for 8 images, wide_resnet50_2 72 s. 2-3 s target NOT met at full resolution; accuracy at 320x320 unmeasured. |
| 2026-10-26 | 1.4 | BLOCKED: collected parts have no label field and trainer/dataset.py excludes final_verdict REVIEW from training. Needs a station/contracts.py change (user's say-so). |
| 2026-10-26 | 1.4, 3.1b | User allowed the contracts.py change (optional label field) and chose to test smaller input sizes. Round 2 dispatched: worker D (1.4), worker E (speed at smaller sizes). |
| 2026-10-26 | 1.4 | Round 2 PASS. contracts.py: OperatorLabel + optional PartRecord.operator_label (approved). storage.set_operator_label (atomic, logged, rollback), POST /api/label (operator/supervisor/admin), trainer/dataset.py trains on labelled collected parts, unlabelled stay excluded. Full suite 1144 OK. |
| 2026-10-26 | 3.1b | Round 2 PASS. resnet18, 8 real 1280x720 images, seconds for 8: 320x320 1.10, 360x640 2.37 (fits 3 s), 480x640 3.09, 432x768 3.35, 576x1024 6.71. Recommended input 360x640 (16:9). Defect accuracy at that size unmeasured; needs labelled pictures. |
| 2026-10-26 | - | STOP: remaining steps need the user, Quality or physical parts (see Waiting on). Rounds 2 of 6, workers 6 of 40, checker calls 2 of 6. |

## Waiting on (all remaining steps)

- 0.1 user reviews and merges the branch (nothing is committed yet).
- 0.2 mount, lighting, focus/exposure; 0.4 lighting/alarm wiring.
- 0.3 Quality accepts MJPG pictures.
- 1.1 MES token and a live part search, then 1.2 (criteria and class mapping).
- 1.3 real run: collect 20 parts on the real cameras; 1.4 label them.
- 1.5 Quality sets min_good, min_holdout, holdout_fraction.
- 2.2/2.3 laptop setup and first training run; 2.1 first real transfer.
- 3.2 promote, 3.4 thresholds (needs labelled scores), 3.5 Quality approval; 3.3/2.4 only if defect names are wanted (D-A).
- 4.x shadow run, live run, failure tests (unplug/replug), weekly cycle.
- Small open items: confirm operators may label BAD; per-part MES criteria are not used by the label check; run the page JS where node exists.

## Round 3 steps (added 2026-10-26)

| # | Step | Verify | Owner | Status |
|---|---|---|---|---|
| 0.1b | Commit and merge into local main (not pushed) | full suite OK on merged main; CODEGRAPH current | Claude | DONE (merge d5d5fa0, 1144 tests) |
| 1.4b | Two-step labels: operator PROPOSES, admin makes OFFICIAL; trainer uses official only; check criterion against configured list AND the part's own MES criteria (branch feature/two-step-labels) | tests + checker | Claude (Haiku worker) | DONE (1164 tests, checker PASS) |
| 1.3b | Real-camera collect test with the test object: 20 parts, picture quality, back-to-back presses | report + saved evidence in scratch | Claude (Haiku worker) | DONE (rerun 2026-10-26 evening after light and aim fixes: 20/20 parts, 160 files verify, 0.13-0.15 s/press, mean brightness 124-129; checker PASS). One toy car, one position: proves capture only |
| 1.3c | Unplug/replug a camera during a run | station names the camera and recovers | user unplugs, Claude watches | TODO |
| 0.5 | Install node (`! sudo apt install -y nodejs`), run the 2 skipped UI tests | tests not skipped | user installs | DONE: node v24.19 via pip (no sudo) in ~/.venvs/node; both page-script node checks pass |
| 0.6 | Office checklist for tomorrow (MES token, part search, laptop setup, transfer) | file docs/office_checklist.md | Claude | DONE: docs/office_checklist.md |

| Date | Step | Note |
|---|---|---|
| 2026-10-26 | 0.1b | Committed in 4 commits and merged into local main (merge d5d5fa0). Not pushed; main is 5 commits ahead of origin/main. Git identity is not set on the Pi; commits used -c user.name/user.email. |
| 2026-10-26 | 1.4b | User approved extending OperatorLabel in contracts.py and asked for Haiku workers. |
| 2026-10-26 | 1.4b | Round 3 PASS. Two-step labels: operator/supervisor/admin PROPOSE, admin only approves, rejects (reason required, 200-char engineering limit) or sets official directly. Only official (or legacy no-status) labels reach training. Criterion is checked against mes.criteria_ids AND the part's own criteria. Known gap: after a station restart the part's own criteria are not reloaded until the part is selected again. |
| 2026-10-26 | 1.3b | Round 3 PASS. Real cameras 0,2,4,6, collect mode, 25 parts: 8 PNGs + part.json + manifest each, no failed press, outbox 0 rows. Seconds mean/max: top 0.192/0.271, bottom 0.199/0.237. Picture mean brightness 35-40 of 255, uneven. |
| 2026-10-26 | 0.5 | node installed without sudo (pip nodejs-wheel-binaries). To use it: PATH=/home/quangdai1406/.venvs/node/lib/python3.13/site-packages/nodejs_wheel/bin:$PATH. |
| 2026-10-26 | 1.3b | Rerun PASS (Sonnet checker). Light fixed (mean 124-129, no overexposure); object in frame on all 4 cameras after three aim rounds. 20 parts, 160 files verify, all presses accepted incl. back-to-back (about 2.0 s), 0 faults, outbox 0. Evidence copied to ~/Pictures/vision-station-collect-test-26_1026/. Live view helper at /home/quangdai1406/vision-liveview/liveview.py (outside the repo). Next: docs/26_1026_PLAN_office_day.md. |
