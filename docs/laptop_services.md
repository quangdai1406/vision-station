# Laptop services: what runs on the one laptop

Audience: the engineer who sets up the laptop. Commands are PowerShell, run from the
vision-station folder (for example `C:\vision-station`).

**Status: not yet run on the laptop.** The backup and task scripts are built and checked by
static tests only. The inference service does not exist yet (plan task L1). Nothing here has
been run on Windows.

The laptop hosts four things that must not fight for the same resources:

| Component | What it does | How it starts at boot | Load it causes |
|---|---|---|---|
| PostgreSQL | MES database | Windows service installed by the PostgreSQL installer; set Startup type to **Automatic** in services.msc. Service name: `[Engineer to define: service name]` | low |
| MES (Node/Express) | inspector web app and machine API | `[Engineer to define: Windows service or scheduled task "At startup", whichever the engineer picks]` | low |
| Inference service | scores pictures on the GPU during station use (plan task L1, not built) | `[Engineer to define: Windows service or scheduled task, once L1 exists]` | GPU, during inspections |
| Weekly trainer | `python -m trainer weekly`, Sunday 06:00 | scheduled task `VisionStation Weekly Training` (install_weekly_task.ps1) | GPU and CPU, Sunday only |
| Nightly backup | `pg_dump` of the MES database, 22:00 by default | scheduled task `VisionStation MES Nightly Backup` (install_backup_task.ps1) | disk and some CPU, once a day |

The MES and PostgreSQL services should run whether or not anyone is logged on. Choose the
method per service. Do not use a method that needs a password typed into a file, a command
line or this document. Where Task Scheduler needs an account password, `schtasks` asks for it
on the console, as the weekly task does. The engineer records the account names (not the
passwords) in the laptop's setup record.

## 1. Load protection

- **Sunday 06:00 training runs at below-normal priority** (`run_weekly_training.ps1`). The
  trainer's output is copied into the transcript when the run ends.
- **Training refuses to start while the inference service is busy.** Set
  `inference_health_url` in `trainer.json` to the service's health URL. The script calls it
  with a 5-second timeout and reads a `busy` boolean. Busy: exit code `4`, nothing trained.
  Unreachable: training continues and the transcript says so. The URL is never printed.
- **Backup at 22:00, training at 06:00 Sunday**: the backup window does not overlap the
  training. `install_backup_task.ps1` refuses times between 05:00 and 08:00.
- The 22:00 time and the 2-hour backup limit are **engineering defaults**, not Quality or IT
  requirements. Change them with `-Time` or in the script's settings.

## 2. Nightly backup

### Setup (once)

1. Pick a backup folder on this laptop, not in the evidence store. Record it in the setup
   record. `[Engineer to define: backup folder]`
2. Copy the example and fill it in. Keep it outside the repo:

   ```powershell
   New-Item -ItemType Directory -Force "$env:USERPROFILE\VisionStation" | Out-Null
   Copy-Item .\scripts\windows\backup.example.json "$env:USERPROFILE\VisionStation\backup.json"
   notepad "$env:USERPROFILE\VisionStation\backup.json"
   ```

   Replace `backup_folder`, `database`, `pg_dump` and `keep_last`. `keep_last` is the number of
   nightly dumps to keep. It has no default; the engineer chooses it.
3. Give the account that runs the backup its PostgreSQL connection settings **by name only**:
   the environment variables `PGHOST`, `PGPORT` and `PGUSER`, and `PGPASSFILE`, which points at
   the PostgreSQL password file. By default that file is `%APPDATA%\postgresql\pgpass.conf`.
   The engineer creates the password file by hand, with the account's own password in it. It
   is never put in `backup.json`, in a script, in a command line or in the task.
4. Test once by hand. It writes one dump and logs the result:

   ```powershell
   .\scripts\windows\backup_postgres.ps1 -Config "$env:USERPROFILE\VisionStation\backup.json"
   echo "exit code: $LASTEXITCODE"
   ```

   Expected: exit code `0`, one new `<database>-<UTC time>Z.dump` in the backup folder, and a
   log in `<backup folder>\logs`.
5. Register the nightly task. Dry run first:

   ```powershell
   .\scripts\windows\install_backup_task.ps1 -WhatIf
   .\scripts\windows\install_backup_task.ps1
   ```

   To change the time: `.\scripts\windows\install_backup_task.ps1 -Time 21:30`. To remove it:
   `.\scripts\windows\install_backup_task.ps1 -Uninstall`.

### What the backup does

- Runs `pg_dump --format=custom --no-password` into `<database>-<UTC time>Z.dump.partial`. It
  renames the file to `.dump` only when `pg_dump` exits `0` and the file is not empty.
- `--no-password` makes `pg_dump` fail rather than wait for a prompt.
- Keeps the newest `keep_last` dumps of that database. Old dumps are removed **only after the
  new dump is verified**, and only files named `<database>-*Z.dump` in the backup folder.
- On failure it writes `BACKUP FAILED: ...` to the console and the log, and exits with code `5`
  (`pg_dump` failed or the dump could not be verified) or `3` (settings or preflight). A
  `.partial` file may be left behind; delete it by hand after reading the log.
- A dump is a copy of the database for recovery. It is not evidence and is not the evidence
  store. The backup never touches the evidence folder.

### RESTORE CHECK (monthly, by hand)

Once a month, check that the newest dump is a readable archive. Put the backup folder and
database name in the first line:

```powershell
$newest = Get-ChildItem -LiteralPath '<backup folder>' -Filter '<database>-*Z.dump' -File |
    Sort-Object Name -Descending | Select-Object -First 1
pg_restore --list $newest.FullName > $null
"pg_restore --list exit code: $LASTEXITCODE"
```

Expected: `exit code: 0`. Write the date, the file name and the result in the setup record.
`pg_restore --list` reads only the table of contents. It proves the file is a valid custom-format
archive, not that the data restores. A full restore into a scratch database is a separate
exercise: `[Engineer to define: scratch database and who runs it]`.

## 3. Nothing here deletes evidence

- The backup script removes only old MES dumps it made itself, and only after a new dump is
  verified. It never touches the evidence store, the models folder or the reports folder.
- The weekly training never deletes evidence and never purges. Deleting used pictures is a
  manual step by the owner (see `docs/weekly_training.md`, section 7).
- No script in this folder promotes a model or purges evidence.

## 4. Open decisions and unproven items

- **Backup copy off this laptop.** Both the database and the backup live on one machine. Whether
  a second copy is kept elsewhere is `[Engineer to define]` and needs the owner's decision (see
  the second-laptop question in `docs/weekly_training.md`, section 9).
- **Service and task names** for the MES and the inference service: `[Engineer to define]`.
- **Unproven on this laptop:** the backup against the real database; the restore check; that the
  busy check reads the inference service's `busy` value (the inference service's health reply is
  not built yet, so the key name is an assumption until L1 exists); the effect of BelowNormal on
  training time and on inference latency; the boot order of the services.
