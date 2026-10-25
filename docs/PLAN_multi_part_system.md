# Plan: multi-part inspection with a weekly training cycle

Status: built and tested with fake cameras and fake feature extractors. Not yet run on the real
station PC, the real MES data, or the GPU laptop. Section 8 lists what is unproven.
Numbers that belong to Quality are shown as `[Quality to define]`.

## 1. What the system does

1. The operator logs in at the check station and searches for a part number (the search runs in
   the AISI MES). The operator picks one.
2. The station loads that part's approved model and its MES criteria, then waits for the start
   button. With no part, or no approved model for the part, the start button stays blocked.
3. Press 1 takes 4 TOP pictures. The operator flips the part. Press 2 takes 4 BOTTOM pictures.
   The model scores all 8. GOOD saves first, then lights green. BAD lights red, sounds the alarm
   and marks the defect boxes. REVIEW is never green and never posted to the MES.
4. Every part is saved as evidence (8 pictures, part.json, hashes), with the part number and the
   model version used.
5. Once a week the laptop trains a new candidate model per part from the new GOOD pictures. A
   named person approves it before any station uses it.

## 2. Architecture

```
 MES (parts, criteria, inspection-events)
   ^  GET /api/machine/parts/search, /parts/:pn, /criteria/:pn   (identity fields only)
   |  POST /api/machine/inspection-events  (Idempotency-Key = session_id)
 Station PC  --4 USB cameras-->  station/  (stdlib only)
   |  evidence -> synced folder   models <- synced folder (approved versions only)
 SharePoint  .../Quality/Automotive/VisionStation/{evidence,models,reports}
   |  OneDrive client sync
 Laptop (i7 10th gen, 32 GB RAM, 8 GB NVIDIA GPU)  trainer/  (weekly, Sunday 06:00)
```

| Piece | Where | Notes |
|---|---|---|
| Part search and criteria | `station/parts.py`, MES `machineRoutes.ts` | token auth; MES returns part_number, description, part_type, lifecycle only |
| Part selection | `station/app.py` `select_part`, `station/session.py`, `station/ui.py` | IDLE only; operator action `select_part`; MES calls run outside the session lock |
| Model store | `station/modelstore.py` | candidate, approve, rollback stack, sha256 check, approval log written before the switch, no silent fallback |
| Model types | `station/golden.py`, `station/anomaly.py` | golden reference and PatchCore-style anomaly model (schema v2, incremental) |
| Feature extractor | `station/features_torch.py` | resnet18 or wide_resnet50_2; UNPROVEN until `python -m trainer selftest` runs on the GPU laptop |
| Weekly trainer | `trainer/weekly.py` | candidates only, incremental from the approved model |
| Promote, rollback | `trainer/promote.py` | named person and reason required; automation names refused |
| Purge | `trainer/purge.py` | dry-run by default; `--confirm` to delete; GOOD pictures of trained parts only |
| Scheduling | `scripts/windows/`, `docs/weekly_training.md` | Task Scheduler, Sunday 06:00, runs a missed task when the laptop wakes |

## 3. Training design

- **Learning from good parts only.** The model learns what GOOD looks like per camera and side.
  A picture is BAD when its anomaly score is at or above `t_img`. A part is BAD when any of its 8
  images is. `t_img` and `t_low` come from labelled data through `tools/thresholds.py` and are
  approved by Quality: `[Quality to define]`.
- **Weekly increment.** GOOD pictures are deleted after training (if purge is used), so the model
  grows by adding new pictures to its saved memory bank. The coreset keeps the bank bounded.
  A v1 model, or one built with a different extractor, is retrained fresh once.
- **Held-out set.** Whole days of GOOD parts are held out and scored by the new candidate. The
  report lists score ranges and the BAD parts' scores. It does not choose a threshold.
- **Defect types** (burned, torn, short, slugs). The anomaly model flags that something is
  wrong and where. It does not name the defect. Naming needs labelled BAD examples. Today only
  the BAD/GOOD label exists. See decision D3.
- **Strict GOOD.** A training picture comes from a part with machine GOOD, final GOOD and no
  supervisor override. This keeps mistakes out of the training set. It also means the set can
  lean toward easy parts (D5).

## 4. Weekly cycle

| When | What | Who |
|---|---|---|
| All week | Stations save evidence to the synced folder | Operators |
| Sunday 06:00 | `python -m trainer weekly` makes a candidate and a report per part | Laptop |
| Monday | Quality reads the report in `reports/` and decides | Quality Engineer |
| After review | `python -m trainer promote --part PN --version V --by "Name" --reason "..."` | Named person |
| Next press at stations | Stations load the approved version when a part is selected | Station |
| Optional, manual | `python -m trainer purge` (dry run), then `--confirm` | Named person |
| If a problem appears | `python -m trainer rollback ...` | Named person |

The weekly job never approves and never deletes. Promotion and purge are manual steps.

## 5. Hardware fit

- **Laptop (32 GB RAM, 8 GB GPU).** Feature extraction at 64x64 patch grids with resnet18 or
  wide_resnet50_2 should fit in 8 GB for single-image batches. This is a design estimate. Run
  `python -m trainer selftest` first. It prints GPU name, VRAM, seconds per image and peak GPU
  memory. If it runs out of memory, lower the input size or use resnet18.
- **Station PC.** Inference must score 8 images per part. If the station has no GPU, CPU speed
  decides how long the operator waits. Measure it with the selftest on the station PC (D1).
- **Cameras.** Four USB webcams: give each its own USB port on the PC (no hub). The system
  expects all 4 cameras. Partial stations never post to the MES.

## 6. Live view (the original bug)

The live view was empty because the cameras opened only on the first press and the preview never
opened them. Now the station opens cameras at start, the preview reopens a closed camera every
few seconds, a stalled camera is released and reopened, and the page shows "Camera not open" with
the reason. All of this is tested with fake cameras only. See section 8.

## 7. Safety rules locked by tests

REVIEW is never green and never auto-posted. Any of 8 images at or above `t_img` makes the part
BAD. A verdict needs all 8 images. Save before green. The MES post uses `session_id` as the
Idempotency-Key. No token appears in files, logs or error text. Station and trainer refuse an
unapproved or tampered model, with no fallback to an older one.

## 8. What is unproven

| Item | Proof needed |
|---|---|
| Live view and 4-camera capture on Windows (DirectShow behaviour, USB bandwidth, skew) | Run the station on the real PC with the 4 cameras. Check live view, then 20 parts. |
| Torch feature extractor, GPU speed and memory | `python -m trainer selftest` on the laptop |
| Detection quality (does it catch burned, torn, short, slug?) | Labelled BAD examples and an evaluation run reviewed by Quality |
| Bank growth and update time near the size cap | Time a weekly run on real data after several weeks |
| OneDrive sync of the evidence folder (file counts, online-only placeholders) | A trial week; the script refuses to train on online-only files |
| Part search against the live MES | One search on the real server with a machine token |
| Camera `release()` timeout path on Windows | If it ever times out while a grab is running, native code could crash. Watch for it. |

## 9. Phases

1. **Now.** Merge the work. Run `python -m trainer selftest` on the laptop. Run the station with
   4 cameras and confirm live view.
2. **Collect.** Run in fixed-part mode or parts mode, collecting GOOD and BAD evidence for two or
   three chosen part numbers. Quality sets `min_good`, `min_holdout`, `holdout_fraction`.
3. **Train and review.** First manual weekly run. Quality reviews the report, sets `t_img`, and a
   named person promotes. Run the station against the approved model in shadow (operator still
   decides) before trusting it.
4. **Schedule.** Install the Sunday 06:00 task. Review every Monday.
5. **Widen.** Add parts one at a time.

## 10. Decisions needed from you

- **D1. Station PC hardware.** Does the station PC have an NVIDIA GPU? Without one, check the
  CPU time per part from the selftest.
- **D2. Evidence volume.** The evidence of one part is 8 pictures. Confirm the SharePoint
  folder and OneDrive sync can carry the weekly volume, or keep evidence on a local server and
  sync only reports and models.
- **D3. Defect names.** Do you want the operator to pick a defect type (burned, torn, short,
  slugs) when a part is BAD? That gives labelled data for later defect-type models. Not built.
- **D4. Retention.** Quality records need a retention period. How long must GOOD pictures stay
  before purge is allowed? `[Quality to define]`.
- **D5. Training bias.** Strict GOOD leaves out parts that an operator overrode. Decide whether
  borderline parts may join training.
- **D6. Folder permissions.** Who can read `Quality/Automotive/VisionStation`? It holds
  customer part pictures.
- **D7. Roles.** The station has operator, supervisor and admin. Do you want a read-only role?
- **D8. Part types.** Part number to model mapping is one model per part number. If several part
  numbers share one look, they could share a model. Not built.
