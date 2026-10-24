# Repeatability protocol: one USB webcam on a Windows PC

This is the operator procedure for the real-webcam repeatability test. Commands are for
PowerShell. Run them from the vision-station folder.

Defaults in this protocol (50 frames, 30 placements, 5 flushed frames, 5 s warm-up) are
engineering defaults. You can change them with the flags shown. They are not thresholds.

## 0. What this does and does NOT do

This test measures how repeatable one camera's picture is on your rig: brightness drift,
sharpness change, image noise of a still part, and how much the part position changes
when you place it again.

It does NOT:

- test 4-camera USB bandwidth or timing (it uses one camera);
- detect defects;
- choose `t_low` or `t_img`. Those need a model and labelled good AND bad parts from the
  same rig.

The result is an input for Quality. It is not a pass or a fail. The tools give
measurements and candidate numbers. They do not decide.

## 1. Setup on Windows

1. Install Python 3 if it is not installed. The `py` launcher comes with it.
2. Install the two libraries. Run this once:

   ```
   py -m pip install opencv-python numpy
   ```

3. Copy or unzip the vision-station folder to the PC, for example to `C:\vision-station`.
   Keep the folder structure as it is.
4. Open PowerShell in that folder:

   ```
   cd C:\vision-station
   ```

5. Find the camera index. Plug in the camera and run:

   ```
   py -m tools.capture --list-devices
   ```

   Write down the index that belongs to your camera (for example `0`). Use that number
   for `--devices`. If nothing is listed, see section 9.

Session folders are written to `sessions\` inside this folder.

## 2. Rig rules

- Fix the camera mount. Do not move the camera between runs.
- Use a steady, diffuse light. Avoid direct sun, flickering lamps and light that changes
  when people walk past.
- Use a mid-grey scene. The picture should not be saturated (white) or black. Check the
  sample picture after Run A. If it is mostly white or mostly black, change the light or
  the exposure, then repeat Run A.
- Use the real part in the real fixture. Use the same fixture for every run.
- Close other programs that use the camera: video calls, the Windows Camera app and
  browser tabs with the camera on.
- Write down the lighting: which lights, where, and whether the room light is on.
- Do not touch the camera, the light, the mount or the fixture during a run. Keep your
  hands out of the view.

## 3. Run A: stationary (about 50 frames)

The part and the camera stay still. The tool takes 50 frames, one every 0.5 s. Before
each frame it discards 5 stale frames. It waits 5 s before the first frame.

At the defaults, the waiting is about 25 s (50 frames, 0.5 s apart). Add the warm-up and
the reads, and a run takes about a minute on a fast PC. The exact time depends on the
camera.

Run A, without manual settings:

```
py -m tools.capture --mode stationary --devices 0 --out sessions --name stationary-1 --part "<PART NUMBER>" --lighting "<describe the light>" --operator "<YOUR NAME>"
```

Run A, with manual settings. Use this form once you have chosen the values:

```
py -m tools.capture --mode stationary --devices 0 --out sessions --name stationary-1 --part "<PART NUMBER>" --lighting "<describe the light>" --operator "<YOUR NAME>" --focus <FOCUS> --exposure <EXPOSURE> --white-balance <WHITE BALANCE>
```

Why set manual values: with auto settings on, the camera can change focus, exposure and
white balance during the run. Those changes look like drift and noise, so the numbers
no longer describe the part.

How to choose the values:

1. Run a short test to see the picture. Use a new name, because a name can be used only
   once:

   ```
   py -m tools.capture --mode stationary --devices 0 --count 5 --out sessions --name try-1 --part "<PART NUMBER>"
   ```

2. Open `sessions\try-1\samples\dev0_frame0.png`. Check that the picture is mid-grey and
   sharp. Change the light or the focus until it is.
3. The right numbers differ between cameras. The tool does not pick them for you. Use
   the same values for every run in this study.
4. Write the values in your notes.

After the run, the tool prints a **Warnings** list. If it says the camera ignored a
control, for example:

- `autofocus: camera/driver ignored autofocus-off`
- `auto exposure: camera/driver ignored manual exposure`
- `exposure changes had no visible effect`

then do this:

1. Do not ignore the warning. Write it in your notes.
2. Try once more with the other backend:

   ```
   py -m tools.capture --mode stationary --devices 0 --out sessions --name stationary-1b --part "<PART NUMBER>" --lighting "<describe the light>" --operator "<YOUR NAME>" --backend default
   ```

3. If the warning stays, keep the run. It is still a record, and the report flags it.
   Tell Quality that the camera ignored that control.

The warning about "exposure changes had no visible effect" means the picture did not change
when the exposure was changed. The camera may be ignoring manual exposure, even if the
read-back shows the value you set. `--no-response-check` skips that test. Do not use it
unless you have a reason.

The exposure test, and its warnings, run only when you pass `--exposure`. If you leave the
camera on auto exposure you will not see them.

## 4. Run B: placement (30 re-placements)

This run measures how much the part position and the image change when the part is
placed again.

```
py -m tools.capture --mode placement --devices 0 --out sessions --name placement-1 --part "<PART NUMBER>" --lighting "<describe the light>" --operator "<YOUR NAME>"
```

Each round, the screen shows:

`Round 1/30: Remove the part, put it back in the fixture, then press Enter`

"Remove the part, put it back" means: lift the part fully out of the fixture, then put
it back into the same fixture the way you normally load it. Do not seat it more carefully
than you normally would. Keep your hands out of the camera view, then press Enter.

Then the tool discards 5 frames and takes one capture. Keep the same fixture, the same
camera and the same light. Do not touch the camera.

There is no undo. If a round went wrong (for example, the part fell out), finish the run
or stop it with Ctrl+C, then start a new run with a new name. Ctrl+C keeps the data that
was captured, and `session.json` marks the run as aborted.

The default is 30 rounds. Use `--count` to change it.

## 5. Optional runs C and D

Name each session with the run and a number, for example `stationary-2-lights-off`.
Use only letters, digits, `-` and `_`. Never reuse a name. The tool refuses to write into
a folder that already exists.

**Run C: room light changed.** Repeat Run A with the room light changed on purpose, for
example room light off. Write the change in `--lighting`:

```
py -m tools.capture --mode stationary --devices 0 --out sessions --name stationary-2-lights-off --part "<PART NUMBER>" --lighting "room lights off, rig light only" --operator "<YOUR NAME>"
```

**Run D: after 30 minutes of warm-up.** Switch on the camera and the light. Leave them on
and do not touch them for 30 minutes. Then run Run A with a new name, for example
`stationary-3-warm-30min`. The tool opens the camera only during a run, so the 30-minute
wait is done by hand.

## 6. Make the report

Run the report on the sessions you made. Add the optional runs to the list if you did them:

```
py -m tools.report sessions\stationary-1 sessions\placement-1 --out report
```

The report folder is `report\` inside this folder. It contains `report.md`,
`report.json` and `summary.csv`.

What each section means, in plain words:

1. **What was measured.** The mode, frame count, size, measured frame rate, and the part,
   lighting and notes you wrote.
2. **Camera settings.** What you asked for, what the camera reported back, and whether
   it applied the setting. A warning block lists any control the camera ignored. Read this
   section first.
3. **Stationary results.**
   - Brightness drift: how far the average brightness moved during the run.
   - Sharpness variation (cv): how much sharpness changed from frame to frame. Lower
     means steadier focus.
   - Temporal noise: how much each pixel of a still part jitters between frames.
   - Residual without alignment: how far a typical frame differs from the median frame.
     Nothing is aligned, so real movement is included.
4. **Placement results.**
   - Position change: how much the part position varied between frames (shift spread, in
     pixels), and the largest single shift.
   - Residual with alignment: image difference after moving each frame by whole pixels.
     This is mostly noise and surface change, not position.
   - Residual without alignment: image difference including the position change.
5. **Noise floor for threshold work.** The residual of the stationary runs, shown as
   observed numbers. This is **observed noise floor, not a threshold**. It describes the
   noise on a good, unchanged part. It is not an accept limit.
6. **What this test CANNOT tell you.** The limits of this test.

The comparison section, when present, gives the differences between sessions and cameras
in sentences. The report has no pass or fail and no acceptance limit.

**What to send back:**

- The report folder: `report\report.md`, `report\report.json`, `report\summary.csv`.
- For each session: the `samples` folder (colour pictures, for example
  `sessions\stationary-1\samples`) and `sessions\stationary-1\session.json`.
- Do not send the `frames_dev*.npz` files unless Quality asks. They are large.

## 7. Defect classes from the MES

The MES lists the inspection criteria for a part. This step turns them into the station
settings and a sample-collection checklist.

**7.1 Create a machine token.** Do this on the MES machine, from the MES repo. It needs
to run once per station:

```
npm run machine-token -- station-1
```

The command prints the token once. The MES keeps only a hash. Give the token to the person
who runs the PC through a secure channel. Do not paste it into chat, email or a file.

**7.2 Set the token in PowerShell.** This lasts until you close PowerShell:

```
$env:STATION_MES_TOKEN = '<token>'
```

Never write the token into a file, a script or `config.json`.

**7.3 Fetch the criteria.** Replace the `<...>` parts. Do not keep the angle brackets. In
PowerShell `<` is a redirection character:

```
py -m tools.criteria --base-url http://<MES-host>:3000 --token-env STATION_MES_TOKEN --part <PART-NUMBER> --out-dir criteria_<PART>
```

`--token-env` takes the **name** of the variable (`STATION_MES_TOKEN`), not the token.

The tool writes three things into the output folder:

- `criteria_<part>.json`: the criteria as the MES sent them.
- `class_map.json`: the station settings (see below).
- `collection_checklist.csv`: the sample-collection checklist (see below).

If you have a saved copy of the criteria, you can run it offline:

```
py -m tools.criteria --from-json <FILE> --part <PART-NUMBER> --out-dir criteria_<PART>
```

**Proxy.** By default the tool connects to the MES directly and ignores proxy settings.
That keeps the token away from any proxy. Use `--use-env-proxy` only if the MES can be
reached only through a corporate proxy, and set the proxy variables as your IT team tells
you. A token sent over plain `http` through a proxy is visible to that proxy. If you use a
proxy, prefer an `https` MES address.

**class_map.json** has four parts:

- `criteria_ids`: all active criteria. Goes into `mes.criteria_ids` in `config.json`.
- `class_to_criterion`: defect class name to criterion id, for active visual criteria
  only. Goes into `mes.class_to_criterion`. The class names come from the criterion names,
  for example "Scratch on face" becomes `scratch_on_face`. They must match the class names
  the model outputs. Agree that with the model owner and Quality.
- `default_criterion_id`: `null`. This is a decision for Quality. It goes into
  `mes.default_criterion_id`.
- `_notes`: for people only. Delete it before you paste the values into `config.json`.

**collection_checklist.csv** has one row for each active criterion. The columns
`good_samples_needed` and `bad_samples_needed` say `[Quality to define]`. Quality fills
those in. You fill in `collected_good` and `collected_bad` as you collect parts.

**Visual and measurement criteria.**

- A **visual** criterion is something the camera can look for. It gets a defect class and
  a line in `class_map.json`.
- A **measurement** criterion (for example a dimension with a unit and a tolerance) gets no
  defect class. The camera classifier does not check it. The tool warns about it. Decide
  with Quality how it will be checked before you rely on the station for it.
- An **inactive** criterion is ignored.

The tool also warns that `default_criterion_id` is a decision for Quality.

## 8. Order of work to reach thresholds

1. **Camera repeatability now.** This protocol, Runs A to B (and C and D if you do them).
2. **Criteria and defect classes now.** Section 7.
3. **Collect labelled good AND bad parts on the same rig.** Use the same camera, light and
   fixture. The counts are decided by Quality: [Quality to define].
4. **Train a first model.** Start with an anomaly model trained on good parts. Note that
   the station cannot run an ONNX model yet (see the README). This step is outside this
   protocol.
5. **Export per-image scores to a CSV** with the columns `label,score,defect_class,part_id`.
   `label` is `good` or `bad`. `score` is between 0 and 1, and higher means more defective.
   Then run:

   ```
   py -m tools.thresholds scores.csv --min-good N --min-bad N --review-from-good-percentile P --model-version V --out threshold_report.md
   ```

   Quality chooses `N` (the minimum counts) and `P` (the percentile of good scores that
   starts the REVIEW band). `V` is the model version string. If the counts are below the
   minimums, the tool refuses and gives no numbers. If the scores separate good from bad,
   it gives candidate values.

   Optional: `--json FILE` writes the result as JSON. `--evaluate-t-img VALUE` evaluates a
   value that Quality has chosen. It does not recommend a value.
6. **Quality approves the values.** They go into `config.json` under `thresholds`:
   `t_low` and `t_img`, each from 0 to 1, with `t_low <= t_img`. A score `>= t_img` is BAD.
   A score `>= t_low` and `< t_img` is REVIEW.
7. **Re-validate after any change** to the camera, light, mount, fixture, model or scoring
   code. Repeat Run A and the threshold step.

Plain statements:

- The thresholds from this process are **candidates, not approved values**.
- The real-webcam test result is a **go/no-go input for Quality**. The tools do not give
  a verdict.

## 9. Troubleshooting

**Notes before you start.**

- Keep the `sessions` folder on a local disk with free space. Each 2K frame is about 3.7 MB
  uncompressed in memory; the saved files are compressed but can still reach hundreds of
  MB per session.
- The `opencv-python` and `numpy` versions are not pinned. If the install breaks, run
  `py -m pip list`, copy the two version lines, and send them back with the error.
- The sample pictures are in each session folder (`sessions\<name>\samples`), not in the
  report folder. Send both.

**Camera not found.** `py -m tools.capture --list-devices` lists nothing or the wrong camera.

- Plug the camera directly into the PC, not into a hub.
- Close other programs that use the camera.
- Check Device Manager under "Cameras" or "Imaging devices".
- Try another index, for example `--devices 1`.

**"could not open camera at index N".** The index is wrong, or another program has the
camera. Run `--list-devices` again and close other programs.

**Black frames.** The sample picture is black or nearly black. Check the light, remove any
lens cover, and check the privacy settings below.

**Permission or privacy settings.** Open Settings, then Privacy & security, then Camera.
Turn on camera access, and turn on "Let desktop apps access your camera".

**DirectShow ignores manual exposure.** The warning is in section 3. Try `--backend default`.
If the warning stays, keep the run and tell Quality.

**Very large files.** A 2560 x 1440 grey frame is about 3.7 MB in memory. Fifty frames are
about 185 MB per camera, and thirty frames are about 110 MB per camera. The `.npz` file on
disk is compressed, so it is smaller. To use less memory, lower `--width` and `--height`,
or close other programs. For 2K on USB webcams, `--fourcc MJPG` is often needed.

**numpy or OpenCV errors.** The tool says "numpy is needed" or "OpenCV is needed". Run
`py -m pip install opencv-python numpy` again. Close and reopen PowerShell, then run the
command again. Use the 64-bit Python.

**"refusing to overwrite existing session folder".** Use a new `--name`.

**"session name ... does not allow".** Remove the characters the message names. Use
letters, digits, `-` and `_`.

**Run stopped.** The data that was captured is kept, and `session.json` marks the run as
aborted. Read the reason in the printed summary.

## 10. When 4 cameras are available

Use the same commands. Give all four indexes from `--list-devices`:

```
py -m tools.capture --mode stationary --devices 0 1 2 3 --out sessions --name stationary-4cam-1 --part "<PART NUMBER>" --lighting "<describe the light>" --operator "<YOUR NAME>"
```

```
py -m tools.capture --mode placement --devices 0 1 2 3 --out sessions --name placement-4cam-1 --part "<PART NUMBER>" --lighting "<describe the light>" --operator "<YOUR NAME>"
```

The four indexes also go into `config.json` under `cameras.devices` (for C1 to C4).

Check these extra points:

- **USB controllers.** Cameras that share one USB controller share its bandwidth. Plug
  them into different controllers where you can. Device Manager can show which controller
  each device uses.
- **Frame rate.** Section 1 of the report shows the measured frame rate for each camera.
  A camera that is lower than the others needs a look.
- **Warnings.** Read the warnings in each `session.json` and in section 2 of the report.
- **Skew between cameras.** The tool grabs the four cameras one after another in each
  round. For the same round, compare the `stamps` arrays in `frames_dev0.npz` to
  `frames_dev3.npz`. The gap between them is the skew. The report does not compute this yet.
