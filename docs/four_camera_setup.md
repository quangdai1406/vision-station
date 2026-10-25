# Four-camera setup on Windows

Use this when you have all four webcams. The one-camera test is in `poc_quickstart.md`.

1. Plug each camera into its own USB port on the PC. Preferably use separate USB ports; if you
   use a hub, see the hub section below.
2. Find the camera indexes (PowerShell, from the vision-station folder):
   `py -m tools.capture --list-devices`. Then note which corner each index shows.
3. Copy `config.example.json` to `config.json`. Set `cameras.devices` to the four indexes in
   corner order, for example `[0, 1, 2, 3]`. Keep `mes.allow_partial_cameras` as `false`.
4. For the part-number flow set `"parts": {"mode": "mes", "models_dir": "<models folder>"}`. In
   `mes` mode the operator searches for a part number on the page. The MES token comes only
   from the environment variable named in the config.
5. Close every other program that uses a camera (Teams, Camera app, browser tabs).
6. Start the station and open the operator page. Check, before running parts:
   - all four live tiles show a picture within a few seconds;
   - unplug one camera: the page shows "Camera not open" and the tile returns after you plug it
     back in (it retries every few seconds);
   - press once with a part: four TOP pictures, then flip, then four BOTTOM pictures.
7. Run 20 parts. Write down any dropped frame, long wait or error and send it back. These checks
   are the first test on real hardware.

## If you must use a USB hub

A USB hub usually cannot carry the streams of four cameras at the same time. Do not assume a
hub works because the tiles show pictures: test it first.

1. With the hub and all four cameras connected, run the hub test first, from the vision-station
   folder (PowerShell): `py -m tools.hubtest --devices 0 1 2 3`. What the test checks and how to
   read its result is in `docs/hub_feasibility.md`.
2. Follow the test's recommendation. In `config.json`, inside `cameras`, set:
   `"capture_mode": "sequential"`. If the test recommends it, also set
   `"sequential_keep_open": true`. Leave both out to keep the default, `"simultaneous"`.
3. With sequential capture, the station does this:
   - the live view shows one camera at a time, in turn, so each tile refreshes more slowly;
   - the four pictures of a side are taken one camera after another, so the operator waits
     longer for each side;
   - the part must stay still from the first picture to the last picture of a side. This is a
     fixture requirement: check the part is clamped before you press.
4. The longest wait the operator may accept for one side is [Quality to define].
