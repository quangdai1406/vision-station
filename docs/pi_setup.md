# Raspberry Pi setup for the inspection station

Audience: the engineer who sets up the Pi. Commands are for the Pi's shell (bash) unless a
line says Windows. Everything in `<ANGLE_BRACKETS>` is a placeholder the engineer fills in
for this site. Nothing in this guide is a secret, and none may be added: passwords and tokens
stay in the files and variables named below, never in this document, a command line or a log.

**Status: not yet run on a Pi.** The station has never run on a Pi. The laptop inference
service (plan task L1) is built, but its round trip from a Pi is unmeasured. Section 9 lists
what is unproven.

## 1. Raspberry Pi OS and Python

1. Install Raspberry Pi OS (64-bit, with desktop or without; the station needs no desktop).
   Set the hostname, the service user `<PI_USER>` and a login method the engineer records in
   the setup record.
2. Update the system: `sudo apt update && sudo apt full-upgrade`.
3. Check the Python version: `python3 --version`. The station and tools use only the Python 3
   standard library at import time; the Pi's version is recorded in the setup record.
4. Create a virtual environment for the station, owned by `<PI_USER>`:

   ```bash
   sudo apt install python3-venv python3-pip
   python3 -m venv <VENV_DIR>
   <VENV_DIR>/bin/python -m pip install <PACKAGES_FOR_REAL_CAMERAS>
   ```

   `<PACKAGES_FOR_REAL_CAMERAS>`: the OpenCV and numpy packages the station needs for real
   cameras. The exact install method for this Pi (pip wheel, or the OS package) is
   `[Engineer to define]`; it has not been tested on a Pi. The station core is standard
   library only. The neural-network packages are not needed on the Pi: the scoring runs on
   the laptop (see `docs/inference_service.md`).
5. Copy the station folder to `<VISION_STATION_DIR>` (for example from the repository) and
   keep the real `config.json` there, outside version control.

## 2. Powered USB hub for four cameras

- Use a hub with its own power adapter. A hub that draws power from the Pi cannot be relied
  on for four cameras.
- Plug the four cameras into the hub one at a time, in the same order each time.
- A USB hub may not carry four camera streams at once. Do not assume it does: run the hub
  test first (section 10, step 3) and follow its recommendation. The sequential-capture
  settings are in `docs/four_camera_setup.md`, section "If you must use a USB hub".

## 3. Camera device paths and indexes on Linux (V4L2)

- Linux exposes cameras as `/dev/videoN`. OpenCV's index `N` is the `N` in that name, and
  the station's `cameras.devices` list uses those indexes (`[0, 1, 2, 3]` for example).
- Linux cameras are V4L2 devices. DirectShow is Windows only. `tools/pi_check.py` opens the
  cameras with OpenCV's V4L2 backend and says so in its output. The station opens cameras
  with OpenCV's default backend, which on Linux is normally V4L2 as well; confirm this with
  `pi_check` before relying on it.
- A USB camera can show up as more than one `/dev/videoN` node (one for pictures, one for
  metadata). Only the nodes that deliver pictures matter; `pi_check` lists which indexes
  open and deliver a picture, and the index you need is the one that does.
- Indexes can change after a reboot or when a camera is replugged in a different port. Find
  them with `python3 -m tools.pi_check --max-index 7`, which lists the indexes that deliver a
  picture, then check against the corner each camera shows.
- **Stable names (udev), a note only.** To keep a camera on the same name, a udev rule can
  match the camera's own attributes and create a fixed link. Find the attributes with
  `udevadm info -a -n /dev/videoN`, then write the rule by hand. This guide gives no rule
  text, because the attributes depend on the camera. After any rule change, reload with
  `sudo udevadm control --reload-rules && sudo udevadm trigger`, then check the link exists.
  The station still takes indexes from `config.json`, so point those at the right nodes.

## 4. The station as a systemd service

Start the station at boot, as `<PI_USER>`, from `<VISION_STATION_DIR>`. The unit file
contains paths, the user and the NAME of an environment variable, never a value. Create it
as `/etc/systemd/system/visionstation.service`:

```ini
[Unit]
Description=VisionStation inspection station
After=network-online.target
Wants=network-online.target

[Service]
User=<PI_USER>
WorkingDirectory=<VISION_STATION_DIR>
EnvironmentFile=<ENV_FILE_PATH>
ExecStart=<VENV_DIR>/bin/python -m station --config <CONFIG_PATH>
Restart=on-failure
RestartSec=<RESTART_DELAY_S>

[Install]
WantedBy=multi-user.target
```

- `EnvironmentFile` names a file that holds the two secrets the station reads, one line each
  in the form `NAME=value`: the MES token (the variable named by `mes.token_env` in
  `config.json`) and the inference token (the variable named by `model.remote.token_env`).
  Those variable names are chosen in `config.json`; this guide gives no names.
- Create the file as root, then give it to `<PI_USER>` with read access for that user only.
  Its mode must be **600** (owner read and write, nobody else). Nothing else may read it,
  and it is never copied into the unit file, a script, a log or this document.
- After creating or changing the unit: `sudo systemctl daemon-reload`, then
  `sudo systemctl enable --now visionstation`. Check with `systemctl status visionstation`
  and `journalctl -u visionstation -n 100`. The journal must not show a token value.
- The station's operator page address is the `host` and `port` in `config.json` (see
  `config.example.json`). Keep the host to the Pi's LAN address only after the LAN decision
  in section 6.

## 5. LAN addresses

- Give the Pi a fixed address: either a static address in its network settings or a fixed
  lease (reservation) on the router. Record it in the setup record: `<PI_ADDRESS>`.
- The laptop has its own fixed address `<LAPTOP_ADDRESS>`. The inference service listens on
  `<LAPTOP_ADDRESS>:<PORT>` only when the engineer sets its `--host` (its default is
  `127.0.0.1`, which the Pi cannot reach). See `docs/inference_service.md`.
- The Pi should reach the laptop by address, not by a name that depends on DNS or on a
  Wi-Fi network. Test with `ping -c 3 <LAPTOP_ADDRESS>` before anything else.
- Check that the Pi and laptop are on the same plant LAN segment that the owner has confirmed
  (section 6).

## 6. Clock sync (chrony or systemd-timesyncd)

Both machines need correct time. The MES ticket and session checks allow a skew of about
5 minutes (engineering default, plan decision 7), and evidence timestamps are only useful if
the clocks agree.

- Use one time source on the LAN, or the default internet pool if the Pi has access; the
  choice is `[Engineer to define]`.
- With systemd-timesyncd (default on many Raspberry Pi OS images), check with
  `timedatectl status`: the line `System clock synchronized` must read `yes`.
- With chrony (`sudo apt install chrony`), check with `chronyc tracking` (`Leap status: Normal`)
  and `chronyc sources`. Use one of these two, not both.
- Record the check result and the time source in the setup record.

## 7. SMB mount for the laptop evidence share

The Pi saves each part's pictures locally first, then copies them to the laptop over an SMB
share (plan decision 5). The mount is the Pi's view of that share.

1. On the laptop, create a share `<SHARE_NAME>` that only the Pi's account can write to, and
   only to the evidence folder. Record the share account's name (not its password).
2. On the Pi, create the mount point and the credentials file:

   ```bash
   sudo mkdir -p <MOUNT_POINT>
   sudo nano <CREDENTIALS_FILE>        # two lines: username=<SHARE_USER>, password=<SET_BY_ENGINEER>
   sudo chmod 600 <CREDENTIALS_FILE>   # owner read and write only, root-owned
   ```

   The password is typed by the engineer into that file. It is never written in this guide,
   in a script, or in `/etc/fstab` comments.
3. Add one line to `/etc/fstab`:

   ```
   //<LAPTOP_ADDRESS>/<SHARE_NAME>  <MOUNT_POINT>  cifs  credentials=<CREDENTIALS_FILE>,uid=<PI_UID>,gid=<PI_GID>,vers=<SMB_VERSION>,_netdev,nofail  0  0
   ```

   `<SMB_VERSION>` is the SMB protocol version the laptop accepts; the engineer picks it
   after testing. `_netdev` and `nofail` keep the Pi booting when the laptop is off.
4. Mount and check: `sudo mount -a`, then `python3 -m tools.pi_check --no-cameras
   --evidence-dest <MOUNT_POINT>/<EVIDENCE_FOLDER>`. The check writes and removes a small
   test file and prints the free space.

Do not use a guest share, and do not put the credentials on the command line.

Evidence sync: use EITHER `scripts/pi/sync_evidence.sh` OR the station's built-in sync thread (`evidence.sync` in the config), not both: both write the same queue file. `archiveRef` in the MES is the planned laptop path of the part, set when the part is queued; it is valid once the sync has copied and verified the part.

## 8. Pilot without TLS (decision 6, open)

- The laptop inference service and the MES are plain HTTP. The token travels unencrypted on
  the LAN, and the MES session cookie is not marked secure today.
- The pilot is decided to run on the plant LAN without TLS, **only** if the owner confirms the
  LAN is isolated from guests and Wi-Fi. This is plan decision 6 with confidence 0.65 and is
  **escalated to the owner**. It is not settled by this guide.
- TLS is required before the station is used outside the pilot. Until the owner has answered,
  do not connect the Pi to a network that guests or Wi-Fi can reach.

## 9. What is unproven

- The Pi has never run the station, the systemd unit, or the SMB mount.
- The hub test has not been run on a Pi; the hub test's own guide is written for Windows.
- The camera indexes on the Pi, and whether the station's default backend picks V4L2, are
  not confirmed.
- The Pi-to-laptop round trip (8 pictures) has never been measured. Its numbers are not
  judged here: the operator wait limit is `[Quality to define]`.
- Power of the hub with four cameras, the time sync choice, the boot order of the services,
  and the stable camera names are not tested.
- No threshold or acceptance limit is set by this guide.

## 10. First-day checklist

Work through in order. Keep the printed output and the report files in the setup record.

1. [ ] Section 1 done: OS updated, Python checked, `<VENV_DIR>` works, station folder in place.
2. [ ] Section 5 and 6 done: fixed Pi address, laptop reachable (`ping`), clock synchronized.
3. [ ] Hub test. Four cameras on the powered hub, plugged in order, the part held still on the
       fixture. From `<VISION_STATION_DIR>`:

       ```bash
       python3 -m tools.hubtest --devices 0 1 2 3 --rounds 5 --out hubtest_report.json
       ```

       On the Windows PC the same command starts with `py`. Read `docs/hub_feasibility.md`,
       section 7, and follow the recommendation it prints.
4. [ ] `pi_check`, with the laptop running the inference service and the token variable set
       for this shell only (for example with `read -rs` and then `export`, so it is not in a
       file or in the shell history):

       ```bash
       python3 -m tools.pi_check --max-index 5 \
           --inference-url http://<LAPTOP_ADDRESS>:<PORT> \
           --token-env <TOKEN_ENV_NAME> --part-number <PART_NUMBER> \
           --evidence-dest <MOUNT_POINT>/<EVIDENCE_FOLDER> --out pi_check_report.json
       ```

       It prints the working camera indexes, the health reply, the time for each of the 8
       pictures and the total, and whether the evidence folder takes a file. It decides
       nothing. Record the numbers and send them with the report. The operator wait limit
       line says `[Quality to define]`.
5. [ ] Station started (`systemctl status visionstation` shows active; the operator page
       shows four live tiles within a few seconds).
6. [ ] Twenty parts run through the station. For each part, write down any dropped frame,
       long wait, error or FAULT, and send the list back (`docs/four_camera_setup.md`,
       step 7). These are the first real-hardware results.

Keep the setup record current: addresses, mount point, service and task names, the
time source, the hub model's label (not named here), and the date of each check.
