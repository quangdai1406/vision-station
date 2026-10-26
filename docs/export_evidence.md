# Exporting station evidence to the training laptop

`tools/export_evidence.py` copies saved evidence parts from the Pi station to a folder that the
laptop trainer can read. It checks every part against its `manifest.sha256` before copying, skips
incomplete, corrupt or purged parts (and reports them), and is incremental. It is read-only at the
source, never deletes anything, and uses plain file copy only. It has no network code.

Status: built and tested with fake evidence built by the real storage code. **Not yet run on the
Pi or against a real SMB share.** The steps below are the plan; run them by hand.

## 1. On the Pi

Run from the repo root, with the station stopped or idle (a part being saved is never picked up
until its folder is published, so a running station is safe).

```bash
cd /home/pi/vision-station   # adjust to the install path

# Dry run first: verifies and counts, writes nothing at the destination
python3 -m tools.export_evidence --source /home/pi/vision-station/data --dest /mnt/evidence_share --dry-run

# Export everything, or only from a date on
python3 -m tools.export_evidence --source /home/pi/vision-station/data --dest /mnt/evidence_share
python3 -m tools.export_evidence --source /home/pi/vision-station/data --dest /mnt/evidence_share --since 2026-10-20
```

`--source` is the folder that holds `store/` (or `store/` itself). Use the evidence root from your
station config. `--dest` is the destination folder. It becomes `DEST/store/...` with the same layout,
plus `DEST/export_report.json`.

Exit codes: `0` nothing corrupt or failed, `1` a part is corrupt or failed (read the printed lines),
`2` bad arguments or unusable source or destination.

Rerun the same command any time. Parts already at the destination with an identical manifest are
skipped. A destination manifest that differs from the source is never overwritten; that part is
reported as `failed` and needs a person to look at it.

## 2. Choosing the destination

Pick one. Do not put credentials in a file. Keep mount options in the OS config or ask the IT owner.

| Option | When to use | Notes |
|---|---|---|
| SMB share mounted on the Pi (e.g. `/mnt/evidence_share`) | Pi writes straight to the Windows share | Check the mount before running. If it is not mounted, the tool refuses to create the folder; this stops the export writing to the Pi's own disk. |
| USB drive | Air-gapped or no network | Mount it, run the export, unmount before removing. |
| Local staging folder | Then copy by hand or by scp/rsync | Useful for a dry check. |

### Mounting an SMB share on the Pi (document only)

Using `cifs-utils`. Replace the server, share, user and the credentials file path. Create the
credentials file yourself with mode 600; never type a password into the command line or into this
repo.

```bash
sudo apt install cifs-utils
sudo mkdir -p /mnt/evidence_share
# /etc/smb-evidence.cred contains two lines: username=... and password=...   (chmod 600, root only)
sudo mount -t cifs //FILESERVER/Quality /mnt/evidence_share -o credentials=/etc/smb-evidence.cred,uid=pi,vers=3.0
```

Use a `/etc/fstab` line only if the IT owner agrees. A line like this keeps the share from mounting
silently into an empty folder: add `x-systemd.automount` and `nofail` as needed.

### Copying with scp or rsync instead (document only)

If the share is not mounted, export to a local staging folder and then send it:

```bash
python3 -m tools.export_evidence --source /home/pi/vision-station/data --dest /home/pi/evidence_staging
rsync -av --ignore-existing /home/pi/evidence_staging/ user@laptop:/c/VisionStation/evidence/
# or
scp -r /home/pi/evidence_staging/store user@laptop:/c/VisionStation/evidence/
```

`rsync -av` (without `--delete`) never removes anything on the laptop. Do not add `--delete`.

## 3. On the Windows laptop: side check

The trainer reads the evidence folder named in `trainer.json` (see `docs/weekly_training.md`,
section 2). That is the folder that holds `store\`. In PowerShell:

```powershell
# The folder exists and has parts
Get-ChildItem C:\VisionStation\evidence\store -Recurse -Filter manifest.sha256 | Measure-Object

# Every part folder checks as ok (run from the vision-station folder).
# Read-only: it prints one line per part that is not ok, then the totals.
py -c "import pathlib; from tools.export_evidence import check_part; root = pathlib.Path(r'C:\VisionStation\evidence\store'); parts = [p.parent for p in root.rglob('part.json')]; bad = [(str(p), c.status, c.problems) for p in parts for c in [check_part(p)] if c.status != 'ok']; [print(b) for b in bad]; print('parts:', len(parts), 'not ok:', len(bad))"
```

If the last line says `not ok: 0`, the copy is intact. If the folder shows placeholder (online-only) files
under OneDrive, the trainer's run script refuses to start until they are available offline (see the
weekly runbook, section 2 step 6).

## 4. Report

`export_report.json` (in the destination root) holds: `schema`, `generated_utc`, `dry_run`,
`source`, `dest`, `since`, `counts` (found, copied, already_present, incomplete, corrupt, purged,
failed, skipped, out_of_range, temp_ignored) and `problems` (one entry per part that was not
copied, with its reasons). It contains no passwords or tokens and no image data. Keep it with the
evidence for the record.

## 5. Limits

- The check at the destination is by manifest bytes, not by re-hashing the copied files. An export
  run hashes each file as it copies it, so the copy itself is verified.
- Parts that are still being written are not published yet (the station renames the finished part
  into place), so they are not picked up.
- `--since` is by the date folder (UTC day) in the store path.
