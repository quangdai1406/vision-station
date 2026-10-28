# Home trial: training inputs for the Pi

**HOME TRIAL, NOT QUALITY.** Every number on this page that is not in a config file is a
home-trial setting picked by the engineer to exercise the pipeline. None of them is an acceptance
limit, a sample count for production, or a threshold. Quality has not approved any of them.

## Home-only numbers (HOME TRIAL, NOT QUALITY)

| Setting | Value | Where it is used |
|---|---|---|
| `--min-good` | 10 | `python3 -m trainer weekly`: skip a part with fewer good parts |
| `--min-holdout` | 3 | `python3 -m trainer weekly`: skip a part with fewer held-out good parts |
| `--holdout-fraction` | 0.3 | `python3 -m trainer weekly`: target share of good parts held out, by whole UTC day |
| `--seed` | 0 (trainer default) | day split and coreset |

Not set here, still `[Quality to define]`: `thresholds.t_low` and `thresholds.t_img`. They come from
labelled data through `tools/thresholds.py` and must be approved by Quality. Quality also owns the
`mes.criteria_ids` and `mes.class_to_criterion` values in the config example.

## Files

- `extractor.home.json`: feature extractor for the trainer and the selftest. `resnet18`, input
  `[360, 640]`, `device` `cpu`, `keep_aspect` true. `weights_path` points at the ImageNet
  state dict already on the Pi (see below). The trainer's `selftest` rejects unknown keys, so
  no comments are inside this file.
- `config.home.example.json`: station config for 4 cameras (indexes 0, 2, 4, 6; 1280x720; MJPG;
  keep_open; flush 1, as in the 20-part collect test). `mes.base_url` is `http://127.0.0.1:3000`,
  `mes.token_env` is `STATION_MES_TOKEN`. **No token is in this file**; the token is read from that
  environment variable only. `parts.mode` is `fixed`, and `model.kind` is `collect`. Copy it to
  `config.json` on the Pi, fill the `[...]` placeholders, and do not commit the copy.

## Pretrained weights (offline)

- File: `/home/quangdai1406/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` (46.8 MB).
  It is already on the Pi, so nothing was downloaded. The name's hash prefix `f37072fd` matches
  torchvision's published ResNet-18 weights, and the file's sha256 starts `f37072fd47e8`.
- `weekly` needs `weights_path` to be a file path (it refuses a null value), so this file is set
  explicitly. The `selftest` also accepts it.
- The extractor name then carries `sha256-f37072fd47e8`, so a model built with other weights is
  refused on load.

## Selftest result (CPU, this Pi, 2026-10-27)

Command: `~/.venvs/torchpi/bin/python -m trainer selftest --extractor-config docs/home_trial/extractor.home.json`

- torch 2.14.1+cpu, torchvision 0.29.1+cpu, numpy 2.5.4. CUDA not available, so these are CPU timings.
- Result PASS: built on cpu; feature shape `[45, 80, 384]`; features finite for all 8 images.
- **0.2939 s per image** (batch of 8, warm-up excluded). Single-image selftest: 0.2828 s.
- Speed only. Defect detection accuracy is not measured. Whether small defects survive the
  360x640 input is still unmeasured; check on real labelled pictures before trusting any threshold.

## Weekly dry run (empty store)

Command: `~/.venvs/torchpi/bin/python -m trainer weekly --store <empty> --models <empty> --reports <empty> --min-good 10 --min-holdout 3 --holdout-fraction 0.3 --dry-run`

Output: "DRY RUN: nothing was built or written", min good 10, min held-out 3, holdout fraction 0.3,
seed 0, good policy strict, t_img/t_low `[Quality to define]`, store problems 0, candidates 0,
exit code 0, "Nothing was approved." `--dry-run` does not read `--extractor-config`, so the
dry run does not test the weights.

## What the store must look like for `weekly` to make a candidate

Read from `trainer/weekly.py` and `trainer/dataset.py`. These are the rules, not Quality's.

1. Parts are read only if their `part.json` is intact, every image hash verifies, and the part is a
   full station (all 4 cameras, both sides, 8 images; `partial_station` false).
2. Collect-mode parts (`disposition` `collected`) train only when an **official** operator label has
   verdict GOOD. Labels with status `proposed` or `rejected` are excluded. Parts purged after a
   training run are skipped. Non-collect parts count as good only if `final_verdict` is GOOD,
   `machine_verdict` is GOOD and there is no override.
3. Per part number, good parts must number at least 10 (`--min-good`), or the part is skipped.
4. The holdout split is by whole UTC day (`started_utc`). It needs at least 2 distinct days. With
   fewer than 2 days, the trainer splits whole parts instead, so one day's parts can land in both
   train and holdout. Days are added to the holdout while that brings the held-out count closer to
   0.3 of the good parts, and at least one day always stays in train. Result: the held-out count
   can be far from 30%. Example: 10 good parts split 9 and 1 across two days gives 1 held-out part,
   and the part is skipped (needs 3).
5. Held-out good parts must number at least 3 (`--min-holdout`), or the part is skipped.
6. A fresh build needs at least 2 good training parts after the split. Each camera and side needs 2.
7. The result is a CANDIDATE in the models folder plus a report. Nothing is approved or promoted;
   approval is a separate named step.

Practical minimum for one candidate: about 10 or more good, full-station, officially labelled parts
for one part number, spread over at least 2 UTC days, with enough day sizes that the held-out
count reaches 3. Count with `python3 -m trainer inventory --store <folder>` before running weekly.

## Real run (not done here)

`weekly` without `--dry-run` needs numpy and cv2 in the same venv. `~/.venvs/torchpi` has cv2 5.0.0.
The system `python3` does not have cv2. Use the venv:

    ~/.venvs/torchpi/bin/python -m trainer weekly --store <store> --models <models> --reports <reports> --min-good 10 --min-holdout 3 --holdout-fraction 0.3 --extractor-config docs/home_trial/extractor.home.json
