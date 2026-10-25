# USB hub feasibility test (4 webcams behind one hub)

Purpose: find out whether the cameras behind your USB hub can take pictures one after
another, and whether the current simultaneous capture works on that hub. Run this
before relying on the hub. The script judges nothing against a threshold; it reports what
happened. Quality decides what is acceptable.

Run it on the Windows PC, in PowerShell, from the vision-station folder.

## 1. Setup (once)

```powershell
py -m pip install opencv-python numpy
```

The test script does not install anything itself. If OpenCV or numpy is missing it stops
with exit code 2 and prints this command.

## 2. Plug in

- Use a powered hub (one with its own power adapter) if you have one.
- Close every other app that uses a camera (video calls, the Windows Camera app).
- Plug the four cameras in one at a time. Plug them in the same order every time you
  repeat the test.
- Find out which index each camera has:

```powershell
py -m tools.capture --list-devices
```

## 3. Run

```powershell
py -m tools.hubtest --devices 0 1 2 3 --width 1920 --height 1080 --rounds 5 --out hubtest_report.json
```

Options:

- `--devices`: the camera indexes from step 2, e.g. `0 1 2 3`.
- `--width`, `--height`: picture size requested from each camera.
- `--rounds`: how many times each test takes one picture from every camera. Use 2 or more:
  the operator wait figure for keep-open is taken from rounds 2 and later.
- `--warmup`: frames read and thrown away right after each camera opens, as the station
  does (engineering setting, default 5).
- `--backend default` (the default) or `dshow`. Use `default` first: the station opens the
  cameras with the default backend, so this is the test that matches the station. `dshow`
  is the DirectShow backend; if you choose it, the script prints one warning line that it
  differs from the station.
- `--out`: the JSON report. Sample pictures are saved next to it.

If the first run gives no pictures, or a camera fails in every test, run it again with the
other backend (`--backend dshow`), use another `--out` name (for example
`hubtest_dshow.json`), and send both reports (step 6).

Before you run, keep the part still on the fixture from the first picture to the last
picture of a side. The four pictures of a side are taken one after another, so a part that
moves in between gives pictures of different positions.

The run takes a while: three tests, each with the number of rounds you give.

## 4. What the test does

The camera handling imitates the station: the same backend, a short driver queue (buffer
size 1), 5 frames read and thrown away after each camera opens, and 2 stale frames dropped
before each picture.

- **Test A1, sequential, open per picture:** for each camera in order: open it, warm it up,
  drop stale frames, take one picture, close it. Repeated for every round.
- **Test A2, sequential, all cameras kept open:** open all cameras once and warm each up,
  then for each camera in order, every round: drop stale frames, take one picture.
- **Test B, simultaneous (all at once, the current station mode):** open all cameras, warm
  them up, drop stale frames from all of them, grab from all of them, then read all of them.
  This is expected to fail on a hub. A pass is also fine: then the test recommends it first.
  A failure is a result; the test keeps going.

Each camera is tested on its own. A failure on one camera does not stop the others, and
each failure is printed with its camera index.

## 5. Check which camera is which

The first picture from each camera in Test A1 is saved as a PNG next to the JSON report:

```
hubtest_report_cam0_first.png
hubtest_report_cam1_first.png
hubtest_report_cam2_first.png
hubtest_report_cam3_first.png
```

Open them and note which corner or view each index shows. A camera with no PNG had no
good picture in round 1 of Test A1.

## 6. What to send back

1. The file `hubtest_report.json` (in the folder you ran the command from, or the path
   you gave to `--out`). If you ran both backends, send both files.
2. A photo of the hub: the model name or label on it, and how the four cameras are
   connected to it.
3. The Windows version (run `winver` from the Start menu search) and which backend was used
   (`default` or `dshow`).

The PNG files are optional but help with step 5.

## 7. How to read the summary

The last part of the printed output looks like this (numbers are examples):

```
Test A1 - sequential, open per picture
  20/20 pictures OK
  Slowest camera: 1.90 s, fastest camera: 0.80 s
  Total time for 4 pictures: 5.10 s (mean of 5 round(s))
  Failures: none
...
Test A2 - sequential, all cameras kept open
  ...
  Round 1 (includes opening and warm-up): 6.20 s
  Rounds 2 and later, mean: 4.00 s (opening not included)
...
Recommended mode: sequential keep-open worked
Operator wait for 4 pictures: 4.00 s (keep-open, mean of rounds 2 and later; opening and warm-up not included) (decide with Quality if acceptable: [Quality to define])
```

- **`N/N pictures OK`**: how many pictures passed (not empty, not all black). Each round
  asks for one picture per camera, so the total is rounds times cameras.
- **Slowest and fastest camera**: the time for one picture, from the start of that
  camera's step to the picture. In A1 this includes opening the camera and warming it up.
- **Total time for 4 pictures**: the average time for one round, i.e. one picture from
  each of the four cameras. In A2 it includes round 1, so the next two lines show round 1
  and rounds 2 and later separately.
- **Round 1 / Rounds 2 and later** (A2 only): round 1 includes opening and warm-up; the
  later rounds do not.
- **Failures**: each camera and round that did not give a good picture, with the reason.
- **Test B failures**: a result, not a crash. On a hub they usually show why sequential
  capture may be needed. A pass in Test B is also a result.
- **Recommended mode**: the first of these, in this order, that gave a good picture from
  every camera in every round:
  1. simultaneous
  2. sequential keep-open
  3. sequential open-per-shot

  If none did, it says `none worked`. The script applies no other rule.
- **Operator wait for 4 pictures**: the time an operator waits to get one picture from
  each camera in the recommended mode. The printed line says which figure it is:
  - keep-open: the mean of rounds 2 and later. Opening and warm-up are not included,
    because they happen once, before the first picture.
  - open-per-shot: the full per-shot cycle (opening, warm-up, picture and release), mean
    of all rounds. Opening and warm-up are included, because they are repeated for every
    picture.
  - simultaneous: the full round (opening, warm-up and all pictures), mean of all rounds.

  Whether that is acceptable is a Quality decision: `[Quality to define]`.

If the recommendation is `none worked`, or a camera fails in every test, send the report
(step 6) and do not change the station setup yet. If the first backend failed and the
second one worked, say which one in your message.

## 8. Exit codes

- `0`: the test ran and the report was written. Camera failures are in the report and do
  not make the exit code non-zero.
- `1`: the report could not be written: check the `--out` path, or the report could not be
  turned into JSON (then nothing is written).
- `2`: OpenCV or numpy is missing, or an argument is wrong.
