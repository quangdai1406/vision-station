# Home trial smoke test (2026-10-27): train -> promote -> inspect -> post

SMOKE TEST ONLY. NOT QUALITY. One toy car, one position, 20 parts from one UTC day, pictures replayed from disk
(no live cameras). Nothing here is evidence of detection quality.

## What ran (scratch copy, original evidence untouched)

1. Copy of `~/Pictures/vision-station-collect-test-26_1026/`; all 20 parts given an OFFICIAL GOOD label in the COPY.
2. `python -m trainer weekly` on CPU (resnet18, 360x640, `docs/home_trial/extractor.home.json`). HOME TRIAL numbers:
   min_good 10, min_holdout 3, holdout_fraction 0.3. One UTC day, so the split was by whole parts (train 14,
   held-out 6), not by day. Held-out good scores (48 images): min 0.3991, median 0.4757, max 0.5. Took about 4 minutes.
3. `python -m trainer promote` in the scratch model store. The confirmation was piped (`yes`), so NO human approved
   it. Never treat it as an approval.
4. Station with `parts.mode: "mes"`, fake cameras replaying saved PNGs, fake MES (`tools/fake_mes.py`).
   Smoke thresholds t_low 0.60 / t_img 0.80 are NOT Quality's: `[Quality to define]`.

| Case | Result |
|---|---|
| A good part | GOOD, max image 0.398, saved, posted (201), Idempotency-Key == session_id |
| B synthetic BAD (black rectangle on one image) | that image 1.0, other 7 about 0.44-0.50, part BAD, acknowledged, posted with CR-1 fail |
| C synthetic REVIEW (2x2 black patch) | score 0.705, amber, never green, NOT posted until a supervisor resolved it to BAD, then posted |
| D replay of case A's post | HTTP 200, same event id |

## Found and fixed

- `station/app.py` decoded PatchCore pictures as grayscale (`golden.cv2_decoder`) but the trainer trains on RGB
  (`anomaly.cv2_rgb_decoder`). Every image scored 1.0, so a good part became BAD. Fixed; test
  `test_patchcore_part_is_decoded_as_rgb_and_golden_as_grayscale`.

## Open items for Quality / the owner

- Scores saturate: a 5x5 px black patch gives 1.0, 3x3 gives 0.807, 2x2 gives 0.705 (synthetic, on 1280x720).
  How sensitive `t_img` is, and whether 360x640 keeps small real defects, is UNRESOLVED. Needs real labelled BAD parts.
- Speed on the Pi (CPU, station HTTP path): about 2.3 s per 4-image press, about 4.6 s for 8 images, plus about 3.9 s
  model load at start. The earlier selftest figure was 2.37 s for 8 images; the 2-3 s budget is not met end to end.
- With a single UTC day, `trainer weekly` splits by whole parts, so one day can sit in both train and holdout.
- `python -m trainer versions` is described in `trainer/promote.py` and `docs/weekly_training.md` but is not
  registered in `trainer/__main__.py` (not changed; needs the owner's say-so).
- `station.db` of a copied evidence folder stores absolute `parts.path` values; labelling a copy needs them repointed.
