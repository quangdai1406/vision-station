# Plan: "Taking data" mode (360-degree scans) for training (2026-10-27) - FOR APPROVAL

Status: PLAN ONLY. No code is changed. Numbers marked `[HOME TRIAL]` are home-only, not Quality's.
`[Quality to define]` stays where a number is theirs. `station/contracts.py` is NOT changed by this plan.

## 1. Goal (from the owner)

- Operators only inspect. Their pictures are saved and later copied to SharePoint for the weekend training.
- A supervisor or admin can enter **Taking data** mode: press Enter = 4 pictures (one per camera), turn the
  part, press Enter again, until the 360 degrees are done. Then repeat for the other side.
- All scan pictures are also saved for SharePoint and weekly training.
- First step: the 360-degree GOOD scans are the initial training data. Top and bottom counts need not match.
- Then 360-degree BAD scans, many defect types, also used for training (see section 6: what "training" means
  for BAD pictures).
- Revised 2026-10-27: BAD scans are now in scope, with per-picture "defect visible" marks (section 3 and 4).

## 2. Key decision: a SEPARATE scan record, not a bigger "part"

An inspected part keeps exactly 8 images (4 cameras x TOP/BOTTOM), and every safety rule stays as it is
(REVIEW never green or posted, any BAD image makes the part BAD, a verdict needs all 8, save before green,
Idempotency-Key = session_id). Taking-data pictures go into a new **scan record** that never has a verdict,
never lights green or red, and never goes to the MES.

Why: making the part variable would change `IMAGES_PER_PART`, `fusion`, `storage`, `session`, `mes_client`
and the UI together (graph: `station/contracts.py` 32 dependents / 35 tests, `station/storage.py` 31 / 32).
A new module touches none of them.

## 3. Operator view (supervisor/admin only)

1. Log in as supervisor or admin, choose **Taking data**. Type the toy/part ID and pick the label:
   GOOD, or BAD plus one or more defect types (MES criterion ids; home trial: names from a list in the config).
   A BAD part may have more than one defect.
   The part number is chosen first: at home it is the fixed part (HomeToy); later the AISI MES part search
   (`parts.mode: "mes"`, `/api/part/select`, already built) is used to choose the part before scanning.
   BAD scans only: after each step the screen shows the 4 pictures and the supervisor ticks "defect visible"
   PER CAMERA PICTURE (default off). A defect is rarely visible to all 4 cameras, or from all angles of a
   turn, so labelling every picture BAD would put wrong labels into the evaluation and the classifier.
   The ticks are saved per picture and can be corrected before Finish.
2. Side TOP. Place the part. Press Enter: 4 pictures (C1-C4), step 1. Turn the part. Press Enter: step 2 ...
   until the last step. The screen shows "TOP step n of N".
3. Screen says flip. Same for BOTTOM.
4. Finish: the scan is saved (pictures first, then the record). Cancel before finish saves nothing.
5. Taking data is allowed only when the station is IDLE with no part in progress. The cameras are held by one
   program at a time, so it cannot run during an inspection.

The number of steps is NOT fixed (decided 2026-10-27): the supervisor presses Enter for each step and
presses "Done with this side" when the turn is complete. The count is saved per side, so TOP and BOTTOM can
differ. No step count is a quality limit; a minimum or maximum per side, if wanted, is `[Quality to define]`.
Each step gives 4 pictures (one per camera).
The turn is by hand, so only the step index is recorded, not a measured angle.

## 4. Data format (new, documented in tools/FORMAT.md)

```
scans/<line>/YYYY/MM/DD/<scan_id>/
  <site>_<line>_<date>_<time>_<scan_id>_C1_TOP_S03.png   (camera, side, step)
  scan.json        scan_id, part_number, toy_id, label {verdict GOOD|BAD, defects [criterion ids] (BAD only),
                   status, labelled_by/at, approved_by/at}, steps_per_side per side, taken_by, UTC times,
                   camera ids, image list with sha256 and, for BAD scans, defect_visible per picture
                   (per camera and step, decided 2026-10-27)
  manifest.sha256  same text format as parts
```

Label rules mirror the existing two-step labels: a supervisor's label is `proposed`, an admin's is `official`
(or an admin approves it later). The trainer uses `official` only.
No `contracts.py` change is planned. If the existing `Frame` type is reused, its comment "seq 0..3" becomes stale;
the scan code can save bytes directly and avoid that.

## 5. Files (code graph: `impact`, `symbol`)

| File | Change | Graph: dependents / tests | Risk |
|---|---|---|---|
| `station/scan.py` NEW | scan state machine (IDLE, SCAN_SIDE, WAIT_FLIP, SAVING, DONE), uses the `CameraSet` protocol | new, 0 | low |
| `station/scanstore.py` NEW | atomic write, hashes, manifest, the same helpers as `storage.py` (copy or promote the private ones) | new | low |
| `station/auth.py` | new action `scan` for supervisor and admin only | 9 importers, 8 tests | medium: role policy |
| `station/app.py` | build the scan session, config block `scan`; refuse when inspection is active | 8 importers | medium |
| `station/ui.py` | Taking-data screen, `/api/scan/start`, `/press`, `/finish`, `/cancel`, `/state`; role gate | 13 dependents, 18 tests | medium: page script must be node-checked |
| `tools/export_evidence.py` | also verify and copy `scans/` (same manifest check, never deletes) | 1 test file | low |
| `trainer/scans.py` NEW | reader for official GOOD scan images by (camera, side) | new | low |
| `trainer/weekly.py`, `trainer/dataset.py` | add scan GOOD images to the per-(camera, side) memory banks; new split | dataset: 24 dependents / 23 tests | **high: training and split** |
| `station/session.py`, `fusion.py`, `mes_client.py`, `contracts.py` | NOT changed | - | - |

What the graph cannot see: the config keys (plain strings in `app.py`), the page script versus the JSON the
server returns (change both together; `tests/test_ui.py` guards it), `tools/FORMAT.md` as the writer/reader
contract, and the SharePoint side.

## 6. Training design

- PatchCore learns from GOOD only. Scan GOOD images go into the same per-(camera, side) banks, so the model
  format and the station's inspection path do not change. Uneven top/bottom counts are fine because each bank
  is independent (each needs at least 2 good training images, as `weekly.py` already checks).
- A scan is made of near-duplicate pictures. The existing coreset step keeps the bank size manageable.
  Bank size also affects Pi inference time: **unmeasured, test before relying on it**. Pi today: about 2.3 s per
  4-image press with resnet18 at 360x640.
- **Split by toy/scan, never by picture.** Hold out whole scans (and whole days where possible), using the
  `toy_id`, so no toy's pictures are on both sides. This fixes the leak seen in the smoke test.
- **BAD pictures cannot train PatchCore.** It models GOOD only; showing it BAD pictures would teach it that the
  defect is normal. So BAD scans are used in two other ways, both in scope now:
  1. **Evaluation and thresholds (stage 1, first).** Held-out GOOD pictures and the pictures marked
     "defect visible" give scores per defect type. `tools/thresholds.py` turns them into candidate
     `t_low`/`t_img`; Quality approves. This also shows which defect types the model misses.
  2. **Defect-name classifier (stage 2, after stage 1 works).** A small classifier trained on the
     "defect visible" pictures of each defect type plus GOOD pictures, so a BAD part gets a defect name for the
     MES criterion. This is Decision D-A, and `model.kind: onnx` is refused at build time today
     (plan step 3.3), so it needs new code in the station (pre- and post-processing) and the trainer.
  Pictures of a BAD part where the defect is NOT visible are never added to the GOOD training data
  (GOOD strict policy).
- **Split by physical part, per defect type.** Hold out whole BAD scans (a given defect instance on a given
  part), never single pictures: neighbouring steps of one scan look almost the same.
- **Home BAD parts (decided 2026-10-27): play-doh on the toys.** Defect types for the home run are named
  variations, for example by size, place and colour of the play-doh. Limits to remember: a coloured blob is
  an easy, synthetic defect; scores and thresholds learned from it are HOME TRIAL only, not evidence for real
  liner defects. Keep the paired design: toys held out of GOOD training (for example 3 of the 10 toys, chosen
  before training and never changed after looking at scores) get one GOOD scan and several play-doh scans.
  The model then meets the same unseen toy clean and defected, which is the fair test.
- **Counts per defect type** (how many parts and pictures are enough): `[Quality to define]`;
  `[HOME TRIAL]` values only for the home run. Rare defect types may never have enough examples; the plan
  reports that per type instead of hiding it.
- Operators' inspection pictures keep today's rules (GOOD strict) and join the data as before.

## 7. SharePoint

`tools/export_evidence.py` is a plain file copy to a folder (SMB mount, USB or staging). It has no network
code. Owner's note (2026-10-27): the AISI MES already backs up to SharePoint, so the station can do the same.
How the MES does it (script, sync client, API) is not in this repo (the MES repo is separate), so the method
is still to be confirmed: Decision D-C. Whatever it is, no credential goes into a file, a log or an error text
(same rule as the MES token: environment variable only).
The laptop flow in `docs/weekly_training.md` reads `evidence/` from the synced folder.

## 8. Build order (small steps; each ends with tests and the checker)

1. `scanstore.py` + format doc + tests (fake files). Verify: write, verify, interrupted write is not half-saved.
2. `scan.py` state machine + tests with fake cameras. Verify: N steps per side, cancel saves nothing, refuses
   while inspecting, no verdict and no outputs.
3. `auth.py` action + `app.py` wiring + config validation. Verify: operator refused, supervisor allowed.
4. `ui.py` screen and endpoints. Verify: UI tests, node page-script check, real cameras at home.
5. `export_evidence.py` for scans. Verify: dry run, hash check, counts.
6. `trainer/scans.py` + split by toy + weekly on a home scan. Verify: no toy on both sides; selftest and a home
   candidate. HOME TRIAL numbers only.
7. Home 360-degree GOOD scans of the toys; train; inspect with the promoted model; check scores on held-out toys.
8. BAD scans at home: defect list in the config, per-step "defect visible" tick in the UI (extends steps 2 and 4),
   `trainer/scans.py` reads BAD scans, `tools/thresholds.py` input from held-out GOOD and defect-visible BAD
   scores per defect type. Verify: scores table per defect type, no scan on both sides of the split.
9. Quality approval of thresholds and per-type counts `[Quality to define]`.
10. Stage 2 (own plan, after 8 works): defect-name classifier, ONNX wiring (3.3), timing on the Pi.

Full suite `python3 -m unittest discover -s tests` and `gen_codegraph_doc.py` after each code step.

## 9. Decisions (2026-10-27) and what is still open

Decided by the owner:
1. Thresholds first, defect-name classifier second.
2. Home BAD parts: play-doh on the toys.
3. "Defect visible" is marked per camera picture.
4. The part is turned by hand, so only the step index is recorded.
5. SharePoint: the MES already backs up there; the station should do the same.
6. The supervisor types the toy ID at scan start. Later the AISI MES part search chooses the part first.

Decided by the engineer (2026-10-27, before any scores exist; do not change after seeing results):
- Held-out toys: toys 8, 9 and 10 of the logbook (HOME TRIAL). Toys 1-7 train the GOOD model.
- Home defect names: `doh_small`, `doh_large`, `doh_edge` (play-doh; HOME TRIAL, synthetic).

Still open:
- D-C: SharePoint. Postponed until this is merged into the AISI MES system.
- Number of steps per side is variable; any minimum or maximum is `[Quality to define]`.
- Picture retention and per-defect-type counts: `[Quality to define]`.

## Progress log

| Date | Step | Note |
|---|---|---|
| 2026-10-27 | 1 | DONE: `station/scanstore.py` (ScanStore, ScanWriter, verify_scan, load_scan, list_scans), 38 tests, `tools/FORMAT.md` "Scan record" section. Two checker rounds PASS; full suite 1227 OK (2 node skips). Crash-safe write (staging, pictures, scan.json, manifest last, rename), variable steps per side, defect_visible per picture, site/line checked on verify. Open: no cleanup helper for a crashed `.partial` folder (manual removal, documented). contracts.py untouched. |
| 2026-10-27 | 2 | DONE: `station/scan.py` (ScanSession: IDLE, SCAN_SIDE, WAIT_FLIP, SAVING, DONE, FAULT), 79 scan tests, full suite 1268 OK (2 node skips). Variable steps per side, Done with this side, explicit side skip, per-picture marks (new `ScanWriter.set_defect_visible`), cancel saves nothing, refused while an inspection is busy, camera named on failure with step retry, `min_flip_s` 2.0 (engineering setting) so an unflipped part is never labelled BOTTOM (found by the checker). Known limitations: a disk write error drops the whole scan; press() holds the lock during capture so snapshot() waits. Not wired into the station yet. |
| 2026-10-27 | 3 | DONE (overnight run, checker PASS): `scan` action for supervisor/admin; optional `scan` config block (enabled, min_flip_s, debounce_s) builds ScanStore + ScanSession on the same cameras in `station/app.py`; scan start refused while an inspection is mid-part, inspection press refused while a scan is active (guard in `_LockedSession.press`, the only UI path). Full suite 1329 OK, 0 skips (node on PATH). |
| 2026-10-27 | 5 | DONE (checker PASS): `tools/export_evidence.py` also verifies (station.scanstore.verify_scan) and copies `scans/`, manifest last, never deletes, dry-run, report key `scans`. Not run on a real share; SharePoint postponed. |
| 2026-10-27 | 6 | DONE (checker PASS): `trainer/scans.py`, `weekly --scans-too [--holdout-toys A,B,C]`: official GOOD scans join the per-(camera, side) banks, split whole toys only; BAD scans read for evaluation only. Fake extractor tests only: no real torch run on scans yet. |
| 2026-10-28 | 4 | DONE (checker PASS): `station/ui.py` Taking-data endpoints (/api/scan/start, press, done-side, mark, finish, cancel, clear-fault, reset, state, step-picture) gated by the `scan` action (operator 403), label status from the login (supervisor proposed, admin official; forged client fields ignored), page panel with per-camera "defect visible" checkboxes on BAD scans, `ScanSession.step_picture`, `_LockedScan.step_picture` and `scan=self.scan` in `app.py`. Real-app HTTP regression tests. Full suite 1366 OK, 0 skips. NOT run in a real browser yet. |
| 2026-10-28 | smoke | DONE (checker PASS): SMOKE TEST, not Quality. 9 synthetic scans from ONE toy car (8 GOOD official, TOP 2 / BOTTOM 3 steps, 1 BAD with a painted black rectangle) in scratch; `trainer weekly --scans-too --holdout-toys toy7,toy8` with the REAL resnet18 extractor on the Pi: 10 min 08 s, candidate built, nothing promoted; train toys 1-6, held-out 7 and 8, no toy on both sides; held-out GOOD 40 pictures min/median/max 0.3558/0.4245/0.5; BAD defect-visible pictures 1.0, not-visible 0.361-0.480. Export of the 9 scans: copied 9, identical, verify clean. |

## Open list (parked for the owner, from the overnight checker)

- Step 6: `--min-holdout` counts held-out scans, not toys; one held-out toy with 2 scans passes `--min-holdout 2`. Counting held-out toys is stricter. Decide the rule.
- Step 6: per-defect picture counts overlap for scans with several defects; do not sum across defects (add a note to the docs).
- Step 3: every inspection press path must go through `app._locked`; a future hardware-button thread calling `app.session.press()` would bypass the scan guard. A cleaner guard inside `StationSession.press` needs a `session.py` change (owner's say-so).
- Step 3: a scan capture holds the session lock, so inspection UI polling waits during a capture. Releasing the lock during capture is possible; not done.
- Step 5: every file of an already-exported scan is re-hashed at the destination each run; a failed copy leaves a `.name.part` temp file (never deleted by design).
- Step 4: after a page reload the checkboxes show unticked although the server holds the marks; only the last step's pictures can be re-marked on screen (the server accepts a mark on any picture until Finish). Tick from the snapshot marks or keep all steps' pictures: decide.
- Step 4: a scan press while IDLE returns 200 {accepted:false}, unlike other refusals (cosmetic).
- Step 4: the page has only been covered by `node --check` and server-side tests. Run it once in a real browser on the Pi at home (supervisor login, GOOD and BAD scan).
- Evaluation: scores are clipped to 1.0 (`anomaly.py` `_clip01`), so every BAD picture at or above the reference reads exactly 1.0 and the size of the gap is hidden. Recommended (checker): add unclipped raw ratio and a count of saturated pictures to the weekly evaluation JSON (additive, evaluation only, no `contracts.py` change; needs a raw-score accessor in `station/anomaly.py`), plus a docs note meanwhile. Needs the owner's decision.
- Evaluation: held-out GOOD max is exactly 0.5 for both toys (likely `norm_ref` normalisation); one-line check once raw values are reported.

### 2026-10-28 (day) Scan label approval (added after step 7 started)
Why: scans taken by a supervisor are saved `proposed` and the trainer only uses `official`; there was no way to approve a scan. Built: `scanstore.review_scan_label`, `ScanSession.pending_reviews/review`, `GET /api/scan/pending`, `POST /api/scan/review` (admin only, action `label_approve`), and a "Scans waiting for approval" list in the Taking-data panel. Checker PASS (real scans, real HTTP roles, path traversal refused, full suite 1395 OK). Open: the list loads only when the admin clicks Taking data (not on auto-open); damaged scans drop out of the list silently ("needs attention" view not built); crash between the two renames needs manual repair (documented in tools/FORMAT.md); a rejection stores the reviewer in approved_by/approved_at because the label check requires both.

## Handover note (end of the overnight run, 2026-10-28)

State: plan steps 1 to 6 are DONE and committed locally (not pushed): scan store, ScanSession, auth + app wiring, UI screen and endpoints, scan export, trainer with `--scans-too`. Full suite 1366 OK, 0 skipped (node on PATH). `contracts.py` was never touched. The running collect station and `~/vision-home` were not touched. A real-extractor smoke test on synthetic scans passed.

Not done (needs the user or the real station):
1. Step 7: run Taking-data mode on the Pi with real toys. Enable it in `~/vision-home/config.json` with `"scan": {"enabled": true}` (the home config for collect mode has NO scan block yet; stop the station first, edit, restart). Log in as a supervisor or admin (`op1` is an operator and cannot scan; create a supervisor with `python -m station users add --config ~/vision-home/config.json sup1 supervisor`). Run the page once in a real browser (only `node --check` and server tests cover it).
2. Home split: toys 8, 9, 10 held out (decided 2026-10-27), defect names doh_small, doh_large, doh_edge.
3. Train: `python -m trainer weekly ... --scans-too --holdout-toys <toy ids> --extractor-config docs/home_trial/extractor.home.json` with HOME TRIAL numbers only, then a named person promotes.
4. Everything in the Open list above; SharePoint transfer (D-C) is postponed until the merge into the AISI MES system.
5. Push: nothing is on GitHub since commit 115c2a4. Say the word and the commits can be pushed.

