# One-camera proof of concept on Windows

This guide is for one USB webcam on one Windows PC, with Python and PowerShell. The goal is to
see the whole flow work: take a TOP picture, flip the part, take a BOTTOM picture, then see
GOOD, REVIEW or BAD on the screen.

Commands are for PowerShell. Run them from `C:\vision-station` unless a step says otherwise.
Check each command against the help text with `py -m station --help` and
`py -m station teach --help`.

## 0. What this proves and what it does NOT

It proves that:

- one camera takes two pictures per part (TOP and BOTTOM);
- the station compares them with the good-part picture you taught it, and shows GOOD, REVIEW or BAD;
- the evidence for each finished part is stored on this PC.

It does NOT prove:

- **Quality.** One camera gives 2 images per part. The real station takes 8 (4 cameras, top and bottom).
- **A validated model.** The "golden image" model is a demo heuristic. It compares each new picture
  with the good parts you taught it. It is not a validated quality model.
- **Any threshold.** The `t_low` and `t_img` numbers in this guide have no quality meaning. They only
  decide when a picture turns REVIEW or BAD.
- **MES posting.** Parts are stored but NOT sent to the MES. The screen shows a PROOF OF CONCEPT banner.

Do not use these results for production decisions.

## 1. Setup

1. Install Python 3 from python.org if it is not installed. The `py` launcher comes with it.
   Use the 64-bit version.

2. Install the two libraries. Run this once:

   ```powershell
   py -m pip install opencv-python numpy
   ```

3. Get the code. With git:

   ```powershell
   git clone https://github.com/quangdai1406/vision-station.git C:\vision-station
   ```

   Or unzip a copy of the vision-station folder to `C:\vision-station` and keep the folder
   structure as it is.

4. Open PowerShell in that folder:

   ```powershell
   cd C:\vision-station
   ```

5. Plug the camera directly into the PC (not into a hub), then find its index:

   ```powershell
   py -m tools.capture --list-devices
   ```

   Write down the index of your camera (for example `0`). You need it in step 2.
   If nothing is listed, see section 9.

6. Set up the rig. The rules are in `docs\repeatability_protocol.md`, section 2. In short:
   - fix the camera mount, and do not move it;
   - use a steady, diffuse light (no sun, no flickering lamp);
   - use the same fixture and the same light for teaching and for every test;
   - close other programs that use the camera (video calls, the Windows Camera app, browser tabs).

## 2. Configure: config.json

Copy the example and open the copy:

```powershell
Copy-Item config.poc.example.json config.json
notepad config.json
```

The station refuses to start while any value still looks like `[ ... ]`, and it names the
value. It does not check whether your text is right. Replace every value below with your own
text. Keep it valid JSON: double quotes, commas between items, no comments, `true` and `false`
in lower case.

Full file for this proof of concept. The text values are examples. Replace them with yours:

```json
{
  "site": "AISI1",
  "line": "POC",
  "station_id": "poc-1",
  "part_number": "817D-LINER-L",
  "lot": "POC-LOT-1",
  "operator": "Your name",
  "supervisor": "Supervisor name",
  "material_handler": "Handler name",

  "thresholds": {
    "t_low": 0.3,
    "t_img": 0.6
  },

  "min_flip_s": 2.0,

  "data_dir": "./data",
  "users_file": "./users.json",

  "ui": {
    "host": "127.0.0.1",
    "port": 8000,
    "session_timeout_s": 900
  },

  "mes": {
    "base_url": "http://127.0.0.1:3000",
    "token_env": "STATION_MES_TOKEN",
    "criteria_ids": [],
    "class_to_criterion": {},
    "default_criterion_id": null,
    "sync_interval_s": 5,
    "default_bad_disposition": "quarantine",
    "allow_partial_cameras": false
  },

  "cameras": {
    "kind": "opencv",
    "devices": [0],
    "width": 1920,
    "height": 1080
  },

  "model": {
    "kind": "golden",
    "path": "./golden.npz",
    "version": "golden-poc-1"
  },

  "outputs": {
    "kind": "console"
  }
}
```

What each part means:

| Key | What to type | Why |
|---|---|---|
| `site`, `line` | Your site code and line code, for example `AISI1` and `POC`. | They become part of the file names and the folder names. |
| `station_id` | Any text, for example `poc-1`. | It is written to each part. It is not checked here because nothing is sent. |
| `part_number` | The part you test, for example `817D-LINER-L`. | Written to each part record. |
| `lot` | The lot for this run, for example `POC-LOT-1`. | Written to each part record. |
| `operator`, `supervisor`, `material_handler` | The names for this session. | Written to each part record. |
| `thresholds.t_low`, `thresholds.t_img` | **Demo starting values: 0.3 and 0.6.** | See below. |
| `data_dir` | `./data` | Evidence goes to `C:\vision-station\data`. |
| `users_file` | `./users.json` | The logins file, written in step 3. |
| `ui.host`, `ui.port`, `ui.session_timeout_s` | Keep `127.0.0.1`, `8000` and `900`. | `127.0.0.1` means only this PC can open the page. |
| `mes` | Keep the values above. No MES is used here. | See below. |
| `cameras.devices` | One index from step 1.5, in a list, for example `[0]`. | One camera means one number. |
| `cameras.width`, `cameras.height` | Keep `1920` and `1080`. | If the picture is wrong, fix the light first. See section 9. |
| `model.path` | Keep `./golden.npz`. | Teach (step 4) writes this file. |
| `model.version` | Any text, for example `golden-poc-1`. | Written to each part record. |
| `outputs.kind` | Keep `console`. | Lights and the alarm are printed in the PowerShell window. There are no lamps. |
| `min_flip_s` | Keep `2.0`. | Seconds after the TOP picture before the BOTTOM press is accepted. It stops one tap from taking both sides. It is an engineering setting, not a quality limit. `0` turns it off. |

**Thresholds (demo values only).** Choose two numbers between 0 and 1, with `t_low` smaller than
or equal to `t_img`. The starter values are `t_low` 0.3 and `t_img` 0.6. These are demo values.
They have no quality meaning, and Quality has not approved them.

- score `>=` `t_img` gives BAD (red);
- score `>=` `t_low` and `<` `t_img` gives REVIEW (amber);
- score `<` `t_low` gives GOOD (green).

You can change them by looking at the scores (section 8). After you edit `config.json`, stop the
station and start it again. The values are read at start.

**MES (not used here).** The station does not send anything to the MES in this test. The values
above are enough:

- `base_url`: any `http://` address. The example is fine.
- `token_env`: the NAME of an environment variable. Type `STATION_MES_TOKEN`. You do not set it.
  Never put a token in this file.
- `criteria_ids` and `class_to_criterion`: keep them empty, `[]` and `{}`. Nothing is posted,
  so no criteria are needed.

## 3. Create the logins

Do this before the first start. The station will not start without the users file.

```powershell
py -m station users add operator1 operator
py -m station users add supervisor1 supervisor
```

Each command asks for a password, then asks to repeat it. Use at least 8 characters. Nothing is
printed. The file keeps only a hash of each password.

- `operator1` can press and acknowledge a BAD alarm.
- `supervisor1` can also resolve a REVIEW, override a BAD, and clear a fault. You need it for step 6.

The command reads `config.json` in the current folder. Finish step 2 first.

## 4. Teach the good-part picture

Teach photographs GOOD parts only. It saves what the good parts look like in `golden.npz`.
The station compares each new picture with that.

Before you start: stop the station if it is running. Teach and the station both use the camera.

```powershell
py -m station teach --config config.json
```

The PowerShell window asks you for six steps with 3 parts (the default). Each prompt looks like this:

1. `Part 1 of 3: place a GOOD part with the TOP side up, keep it still, then press Enter`
   Put a good part in the fixture, TOP side up, keep it still, then press Enter.
2. `Part 1 of 3: flip it to the BOTTOM side, keep it still, then press Enter`
   Flip the same part, keep it still, then press Enter.

Repeat for parts 2 and 3. For each side, the tool takes 5 frames by itself with a short pause.
So with the defaults you get 3 parts x 2 sides x 5 frames.

Rules:

- Use GOOD parts only. A damaged part teaches the wrong picture.
- Keep the light steady. Use the same fixture and mount as in the test.
- If something goes wrong, press Ctrl+C. Nothing is written when teach stops early.

The defaults are engineering settings, not quality limits. Change them only if you have a reason:

```powershell
py -m station teach --config config.json --parts 3 --frames-per-side 5
```

When teach finishes, it prints `Golden model saved to ...golden.npz` with the frame counts.

Teach refuses to overwrite an existing `golden.npz`. To learn again on purpose, add `--force`:

```powershell
py -m station teach --config config.json --force
```

## 5. Run the station

```powershell
py -m station --config config.json
```

The window prints `station running. UI: http://127.0.0.1:8000/   Ctrl-C to stop`. Leave this
window open. Do not add `--demo`: that uses fake cameras.

1. Open the address from the window in a browser on the same PC (for example Edge or Chrome).
2. Log in as `operator1`, or as `supervisor1` for the REVIEW test.

The login expires after a while (`ui.session_timeout_s`, 900 seconds by default). Log in
again if the page asks you to.

What the screen shows:

- **Banner (orange):** `PROOF OF CONCEPT: 1 camera(s)...`. It stays on while there is one camera.
- **State line:** the station state, or `FAULT:` with the reason.
- **Verdict box:** `GOOD` (green), `BAD` (red), `REVIEW` (amber). It shows `-` before the first part.
- **Counters:** Total, Good, Bad.
- **Two rows, TOP and BOTTOM:** one tile per camera, so one tile per row here. Each tile shows the
  picture and a caption such as `C1  score 0.12`. Red boxes on the picture show the areas that differ
  from the taught good part, with the label `difference_from_golden` and a score.
- **Prompt line (above the buttons):** the next step, for example
  `Step 1: place the part TOP side up, then press Enter.`
- **Live view (not saved):** one picture per camera, each with its label. It shows while the station
  waits: before the TOP press and while you flip the part. It stops while a picture is being taken
  and while a verdict is shown. Use it to place the part and to check light and focus before you
  press Enter. Nothing from the live view is stored.
- **Buttons:** `Press (Enter)`, `Acknowledge alarm`, `Clear fault`, the reason box, and the
  `Resolve REVIEW` and `Override BAD to GOOD` buttons. A button is greyed out when your role may
  not use it. The Press button shows `Capturing...` while a picture is taken, and `Wait N s` while
  the station waits for the flip.

## 6. Use it

Test sequence for one part. One Enter takes one side.

1. Put the part in the fixture, TOP side up. Use the live view to check the placement, the light
   and the focus.
2. Click the grey background of the page once, so the focus is not in a box. Then press **Enter**,
   or click **Press (Enter)**. This takes the TOP picture. The button shows `Capturing...` for a
   moment. The verdict stays `-`. The screen then says `Step 2: flip the part to the BOTTOM side,
   then press Enter.`
3. Flip the part and keep it still. The button shows `Wait N s` first. The station does not accept
   a second press for a short time (`min_flip_s`, 2 seconds by default). This stops one tap from
   taking both sides. When the button shows `Press (Enter)` again, the flip press is accepted. The
   live view comes back while you flip. The default wait for the flip is 60 seconds
   (`flip_timeout_s`). After that the part is voided, and you start again.
4. Press **Enter** again. This takes the BOTTOM picture. The station compares the two pictures and
   shows the verdict. The tiles under the live view show the TOP and BOTTOM pictures.

After any press, the page ignores presses for 1.5 seconds.

What each verdict does:

- **GOOD (green).** The part is saved at once. The Good counter goes up.
- **BAD (red).** The alarm is on. Click **Acknowledge alarm** (operator or supervisor). The part is
  saved after the acknowledge.
- **REVIEW (amber).** No alarm, and never green. A supervisor types a reason in the reason box, then
  clicks **Resolve REVIEW: GOOD** or **Resolve REVIEW: BAD**. The part is saved only after that.
- **FAULT.** A save failure stops the part and shows `FAULT:`. It is never shown as GOOD.
  A supervisor clicks **Clear fault** to retry. Read the message in the State line first.

Try these parts, one at a time. Write down the score and the verdict for each:

1. A taught good part, again.
2. A damaged part.
3. A part with a mark drawn on it (marker or pen on the surface).
4. No part in the fixture when you press.

The purpose is to see how the scores move. It is not a pass or fail test.

## 7. Look at the evidence

Each saved part gets one folder:

```
C:\vision-station\data\store\<line>\<year>\<month>\<day>\<part id>\
```

For the example config this is, for example, `C:\vision-station\data\store\POC\2026\10\24\P...`.
The part id starts with `P` followed by the date and time and a counter.

The folder holds:

- **Two PNG pictures**, one TOP and one BOTTOM. The file name holds the site, line, date, time,
  part id, camera id and side, for example `AISI1_POC_<date>_<time>_P<part id>_C1_TOP.png`.
- **`part.json`**: the record for the part. Check these keys:
  - `camera_count`: `1`;
  - `partial_station`: `true`;
  - `machine_verdict` and `final_verdict`;
  - `thresholds`, `model_version`, `part_number`, `lot`, the names;
  - `images`: one entry per picture, with its score and its defect boxes.
- **`manifest.sha256`**: one line per picture, with its SHA-256 hash and file name. To check a
  picture, run this and compare the hash with its line in the manifest:

  ```powershell
  Get-FileHash -Algorithm SHA256 "C:\vision-station\data\store\POC\2026\10\24\P...\AISI1_POC_...._C1_TOP.png"
  ```

  Replace the `P...` parts with the real folder and file names.

The same part is also recorded in `C:\vision-station\data\station.db`. That database also has an
event that says parts are not sent to the MES, because this is a partial station. Nothing is
queued for the MES.

## 8. Tuning and honesty

- **The golden image needs a rigid rig.** The model compares each picture with what you taught
  it. If the camera, the light or the fixture moves between teach and test, you will see
  false BAD results. Small marks (under about 1 % of the picture) may not change the score
  at all, so test with a clear mark first.

- **Everything GOOD.** The pictures look like the taught part, or the thresholds are too high.
  Check that the light and the mount are the same as during teach. If they changed, re-teach
  (`--force`). Otherwise lower both demo thresholds in `config.json`, keeping `t_low` at or below
  `t_img`. Then stop and start the station.
- **Everything BAD.** The pictures differ from the taught part, or the thresholds are too low.
  Check the light and the mount first. Then re-teach, or raise both demo thresholds.
- **Choose the numbers by looking at the scores.** Use the scores on screen and in `part.json`.
  Do not pick numbers to get the result you want. Write down what you changed and why.
- **These are demo values.** They do not decide quality. Do not present a result as a quality
  decision.

Next steps, in this order:

1. **Camera repeatability.** Run the protocol in `docs\repeatability_protocol.md` (Runs A and B).
2. **Four cameras.** Put four indexes in `cameras.devices`. The station then takes 8 images per part.
3. **Labelled parts.** Collect GOOD and BAD parts on the same rig. Quality decides how many.
4. **Thresholds.** Use `tools\thresholds.py` with Quality's approval (protocol section 8).
5. **MES posting.** Only on a full 4-camera station with a real model. A real model is not wired
   yet (see `README.md`, "Not done"). Keep `mes.allow_partial_cameras` set to `false`.

## 9. Troubleshooting

**Camera not found.** `py -m tools.capture --list-devices` lists nothing, or the wrong camera.
Plug the camera directly into the PC. Close other programs that use the camera. Check Device Manager
under "Cameras" or "Imaging devices". Try another index.

**`C1: could not open device 0`** (or another index). The index is wrong, or another program has the
camera. Run the list command again and close other programs. Then start the station again.

**Black picture.** Check the light, remove any lens cover, and check the Windows camera privacy
settings. Open Settings, then Privacy & security, then Camera. Turn on camera access, and turn on
"Let desktop apps access your camera". `docs\repeatability_protocol.md`, section 9, has the same steps.

**Port already in use.** The station stops with a `startup error:` line. Change `ui.port` in
`config.json`, for example to `8001`. Start the station again and open the new address.

**`golden model file not found: ...; run python -m station teach first`.** Run step 4 first.

**`users file not found`.** Run step 3 first. Create the logins before the first start.

**Login fails.** The page shows `invalid username or password`. Check the spelling of the name
(`operator1`, `supervisor1`). To set a new password, run the `users add` command again with the same
name. It updates the password.

**`config error:` or `these values are still placeholders, fill them in: ...`.** The message names
the key. Fix that key in `config.json` and start again. Check the JSON commas and quotes.

**`thresholds.t_low must be <= thresholds.t_img`.** Swap or change the two numbers.

**The page does not react to Enter.** Click the grey background of the page once, so the focus is
not in a box. Enter is ignored while the focus is in the reason box or on a button. Click
**Press (Enter)** instead if you prefer. The page also ignores presses for 1.5 seconds after each
press.

**The button says `Wait N s`.** This is normal. The station waits for the flip before it accepts
the BOTTOM press (`min_flip_s`, 2 seconds by default). Flip the part. When the button shows
`Press (Enter)` again, press Enter.

**The live view is empty.** The camera is busy while a picture is taken, so an empty frame during a
capture is normal. If the live view stays empty, check the camera index in `cameras.devices` (step
1.5), close other programs that use the camera, and reload the page.

**I still got both pictures (TOP and BOTTOM from one tap).** The station ignores a second press
that comes too soon, and logs it as `press_ignored`. Check the events in
`C:\vision-station\data\station.db`. Run this in PowerShell from `C:\vision-station`, then send us
the lines it prints:

```powershell
py -c 'import sqlite3; c = sqlite3.connect("data/station.db"); [print(r) for r in c.execute("select ts_utc, kind, payload_json from events order by id") if r[1] in ("press", "press_ignored")]'
```

Each `press` line is an accepted press. Each `press_ignored` line is a press the station refused,
with its reason.

**`Press was not accepted in the current state.`** Look at the State line. The station may be
waiting for the flip, or for an acknowledge. Do the step it asks for.

**`FAULT:` on the screen.** A supervisor clicks **Clear fault**. Read the message first.

**Teach stopped with `teach aborted: nothing was written`.** Nothing was saved. Run teach again.

**Stop the station.** Press Ctrl+C in its window. It prints `station stopped`.
