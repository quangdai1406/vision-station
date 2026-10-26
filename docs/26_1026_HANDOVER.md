# Handover: end of 2026-10-26 (for the office day, 2026-10-27)

Start here. Then read `docs/26_1026_PLAN_office_day.md` (order of work) and
`docs/26_1026_PLAN_pi_station_progress.md` (master tracker with the log).
Numbers that belong to Quality are `[Quality to define]`. None were invented.

## State in one paragraph

The Pi 5 station captures 4 USB cameras (device indexes 0, 2, 4, 6, MJPG 1280x720) in collect mode. A 20-part
real-camera test passed (checker PASS): 160 files verify, verdict REVIEW / `collected`, nothing queued for the
MES, 40 of 40 presses accepted, about 0.13-0.15 s per press. The camera light and aim are good enough
(mean brightness 119-133, car fully in frame on all four). That test used one toy car in one position, so it
proves capture, saving and speed only. No model is trained. `main` is pushed: GitHub `origin/main` has
everything up to the commit that adds this file.

## Do first tomorrow (Pi, 5 minutes)

1. `cd ~/vision-station && git status` should be clean on `main`.
2. Do not move the cameras or the light. If they move, redo the 20-part collect test.
3. Run `python -m tools.liveview` (needs the vision venv), open http://127.0.0.1:8090/ and check that all 4
   cameras show the part, mid brightness, no red text. Stop it before starting the station (Ctrl-C).
4. Run `python -m tools.capture_speed --devices 0 2 4 6 --width 1280 --height 720 --rounds 10 --pause 1.0 --mjpg --threaded --flush 1`
   and expect 0 failures.

Then follow the table in `docs/26_1026_PLAN_office_day.md`, steps 1 to 11: network, MES token, part search,
criteria mapping, Quality's counts, real GOOD and BAD parts, camera unplug test, export to the laptop, laptop
setup, first training run.

## Open items (all need you, Quality or physical parts)

- Decisions D-A (defect names or GOOD/BAD first), D-B (model on the Pi), D-C (transfer method) are still open.
  Defaults used so far: GOOD/BAD first, model on the Pi, scp or SMB.
- Quality: `min_good`, `min_holdout`, `holdout_fraction`, the picture-evenness limit, MJPG accepted as evidence,
  thresholds `t_low`/`t_img` later.
- Step 0.2 is still TODO: `cameras.manual` (fixed exposure and focus) is not set, and the 10-press repeatability
  study has not been run.
- Step 0.4: alarm and light wiring (`serial_relay` channel map, acknowledge method) is not decided.
- Step 1.3c: unplug and replug a camera during a run has not been tested.

## Facts and gotchas

- Evidence of the 20-part test: `~/Pictures/vision-station-collect-test-26_1026/` (143 MB). Camera pictures:
  `~/Pictures/vision-liveview/`. The throwaway test config there has placeholder thresholds 0.3 and 0.6 with no
  quality meaning. Do not reuse it for real data; make a new config with the real part number.
- The test login password was in a scratch folder that may be cleared. It only protected a throwaway test login.
  Create real logins with `python3 -m station users add ...`.
- Only one program can hold the cameras at a time: stop the live view and any station before starting another.
- GitHub login on the Pi: `gh` 2.103.3 is in `~/bin/gh` (not on PATH unless `~/bin` is added), login stored in the
  keyring, git uses it. Git identity is not set on the Pi: commits used
  `git -c user.name=quangdai1406 -c user.email=quangdai1406@gmail.com`. Remove the login with `~/bin/gh auth logout`.
- Node for the UI tests: `PATH=/home/quangdai1406/.venvs/node/lib/python3.13/site-packages/nodejs_wheel/bin:$PATH`.
- Full test suite: `python3 -m unittest discover -s tests` (1169 tests, about 2 minutes, all OK on 2026-10-26).
- CLAUDE.md rules: read `docs/CODEGRAPH.md` first; `station/contracts.py` only with the user's say-so;
  regenerate the code graph overview after code changes; never commit `.codegraph/`.
- Pi model speed: resnet18 at 360x640 takes about 2.37 s for 8 pictures. Whether small defects are found at that
  size is unmeasured. Test it on the real labelled pictures.
- Multi-agent runner this evening: 3 rounds, 5 Haiku workers, 3 Sonnet checks. A new run starts with fresh limits.

## Known gaps

- After a station restart the part's own MES criteria are not reloaded until the part is selected again.
- A saved part whose MES enqueue failed (for example an unmapped class) is not re-queued automatically.
- `tools/liveview.py` has no login. Keep it on 127.0.0.1 unless the network is trusted.

## Bring back tonight's questions to the office

Any dropped frame, long wait, error line or surprise, the `export_report.json`, and the laptop `selftest` output.
