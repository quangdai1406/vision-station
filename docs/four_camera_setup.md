# Four-camera setup on Windows

Use this when you have all four webcams. The one-camera test is in `poc_quickstart.md`.

1. Plug each camera into its own USB port on the PC. Do not use a hub. Four 1080p cameras on
   one USB controller can fail or drop frames.
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
