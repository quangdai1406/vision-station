# Weekly training on the laptop: runbook

Audience: the engineer who sets up and maintains the laptop, and the person who reviews
the Monday report. Commands are for PowerShell, run from the vision-station folder
(for example `C:\vision-station`).

**Status: built and tested with fake extractors; not yet run on the laptop.** The scripts in
`scripts\windows\` and the trainer commands (`python -m trainer weekly`, `promote`,
`rollback`, `versions`, `purge`, `selftest`) exist. The PowerShell scripts have not been run on
Windows, and the torch/GPU code has not been run on a GPU. Nothing is trained, promoted or
deleted. Read section 9 before relying on this.

Thresholds and sample counts are not in this document. They belong to Quality and are shown
as `[Quality to define]`.

## 1. Weekly flow

```
 Stations (4 cameras each, top and bottom)
     |  save evidence per part (part.json + PNG images, hashed)
     v
 SharePoint folder  .../Quality/Automotive/VisionStation/
     |- evidence/   <- stations' saved evidence
     |- models/     <- candidate and approved model versions
     |- reports/    <- training reports for Quality
     |
     |  OneDrive client, folder set to "Always keep on this device"
     v
 Laptop (Windows 10/11, GPU)
     |
     |  Sunday 06:00 local time: scheduled task "VisionStation Weekly Training"
     |  run_weekly_training.ps1:
     |    1. read trainer.json, refuse if any value is a placeholder
     |    2. evidence folder exists, is not empty, has no online-only files
     |    3. nvidia-smi check (warning only)
     |    4. python -m trainer weekly  ->  CANDIDATE model in models/ + report in reports/
     |    5. transcript log in the logs folder (last 26 kept), exit code
     v
 Monday: Quality reviews the report in reports/ (approves or rejects the candidate)
     |
     |  Quality approves in writing (no approval, no promotion)
     v
 Named person runs   python -m trainer promote  
     |
     v
 Station picks up the approved model version
     |
     |  Optional, rare: manual purge of old evidence
     v
 python -m trainer purge  (dry run first, then the confirm step)  <- separate, by hand
```

The scheduled task never promotes and never purges.

## 2. First-time laptop setup

1. **Windows.** Keep Windows 10 or 11 up to date. Turn off "sleep" on AC power during the
   weekend window if the 06:00 run must happen on time (see section 6, laptop asleep).
2. **NVIDIA driver.** Install the current NVIDIA driver. Check with `nvidia-smi`. It should
   list the 8 GB GPU.
3. **Python 3.** Install 64-bit Python 3 from python.org (the `py` launcher comes with it).
4. **PyTorch with CUDA.** Install the PyTorch build that matches the driver's CUDA version,
   from the install selector on pytorch.org. **This step is up to the engineer and is
   UNPROVEN on this laptop.** Check it:

   ```powershell
   py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   If `cuda.is_available()` prints `False`, the trainer cannot use the GPU. Do not run the
   weekly task until this prints `True`.
5. **vision-station code.** Copy or clone the folder to `C:\vision-station`. Install the
   tools dependencies: `py -m pip install numpy opencv-python`.
6. **OneDrive.** Sign in to the account that can see the SharePoint Quality site. Open the
   `VisionStation` folder, then right-click `evidence`, `models` and `reports` and choose
   **Always keep on this device**. Wait until the status icons show "available offline" on
   every file. The check in step 2 of the run script fails if any file is still a placeholder.
7. **Config.** Copy the example and fill it in. Keep the file outside the repo:

   ```powershell
   New-Item -ItemType Directory -Force "$env:USERPROFILE\VisionStation" | Out-Null
   Copy-Item .\scripts\windows\trainer.example.json "$env:USERPROFILE\VisionStation\trainer.json"
   notepad "$env:USERPROFILE\VisionStation\trainer.json"
   ```

   Every value that starts with `[` must be replaced. `min_good`, `min_holdout` and
   `holdout_fraction` come from Quality. `holdout_fraction` is the share of the labelled data
   kept back for testing; it must be a number strictly between 0 and 1. The trainer's
   `weekly` command requires it (no default). Do not invent any of the three. Never put a
   password or token in this file.
8. **Self-test. Required before the first weekly run.** Run:

   ```powershell
   py -m trainer selftest
   ```

   **Paste the full output into the laptop's setup record and send it to the engineering
   lead.** Do not continue if it fails. 
9. **Run the script once by hand.** Only after step 8 passes and `trainer.json` has no
   placeholders. This runs the real trainer step, so it can write a candidate model. Until the
   trainer command exists, the trainer step fails and that is expected. Read the log:

   ```powershell
   .\scripts\windows\run_weekly_training.ps1 -Config "$env:USERPROFILE\VisionStation\trainer.json"
   ```

10. **Register the task.** Check the dry run first, then register:

    ```powershell
    .\scripts\windows\install_weekly_task.ps1 -WhatIf
    .\scripts\windows\install_weekly_task.ps1
    ```

    The second command asks for the Windows account name, then schtasks asks for its password
    on the console. The script never reads or stores the password; Task Scheduler keeps it.
11. **Power settings.** Set the laptop not to sleep on AC power at the weekend, or accept
    that the run waits for wake (`StartWhenAvailable` runs it when the laptop wakes).
    The task runs on AC only, so keep the charger connected.

## 3. What to check on Monday morning

1. **Task result.** In PowerShell:

   ```powershell
   Get-ScheduledTask -TaskName 'VisionStation Weekly Training' | Get-ScheduledTaskInfo
   ```

   `LastTaskResult` should be `0`. A value of `3` is a preflight failure (config, evidence or
   OneDrive). Any other value is the trainer's own exit code.
2. **Newest log.** Open the newest `weekly-*.log` in the logs folder. Check the start time,
   the evidence file count, the GPU line and the final `Trainer exit code: 0`.
3. **Report.** Open the newest folder in `reports\`. Send it to Quality. Do not summarise it
   as a pass or fail. The report is an input for Quality.
4. **Candidate.** A new candidate is in `models\`. It is NOT used by any station until it is
   approved and promoted.
5. **Missed run.** If the laptop was asleep at 06:00, the task starts at wake (StartWhenAvailable).
   Check the log time against the time you woke the laptop.

## 4. Approval and promotion (people, not the schedule)

- Quality reviews the report and decides in writing whether the candidate is approved.
  Thresholds and minimum counts: `[Quality to define]`.
- A named person runs the promotion command once Quality has approved:

  ```powershell
  py -m trainer promote --help
  ```

- The station picks up the approved version. Check the station's model version on its
  screen or in the part record (`model_version`).

## 5. Rolling back

- A candidate that Quality rejects is left in `models\` with its report. Mark it rejected in
  the report folder. Do not delete it.
- To roll back an approved model, a named person re-points the stations to the previous
  approved version with the promotion command (planned). Keep the previous approved model
  folder in `models\` until a newer one has run for a full week without a problem.
- The scheduled task never changes which model the station uses, so disabling the task does
  not roll anything back. To stop weekly training: `.\scripts\windows\install_weekly_task.ps1 -Uninstall`.

## 6. Failure table

| What you see | Cause | What to do |
|---|---|---|
| `LastTaskResult` is not 0 and there is no log | The task did not start (account password changed, task disabled) | Check the task in Task Scheduler. If the password changed, run the install script again. |
| Log says `config: '...' is still a placeholder` | `trainer.json` not finished | Replace the named value. Quality owns `min_good` and `min_holdout`. |
| Log says `evidence folder is empty` | OneDrive not synced, or wrong path | Check `store` in the config. Check the SharePoint folder is synced. |
| Log says `N of M evidence files are online-only` | OneDrive placeholder files (the client has not downloaded them) | Set the folder to **Always keep on this device**, wait for sync, run again. Nothing was trained. |
| Laptop asleep at 06:00 | No wake, or the power plan blocks wake | Run starts at wake (StartWhenAvailable). Set the power plan to allow wake timers, or keep the laptop on AC over the weekend. |
| Laptop on battery | Task is set to AC only | Plug in. The run does not start on battery and stops if the power source changes to battery. |
| `torch` missing, or `ModuleNotFoundError` in the log | PyTorch not installed for this Python | Install PyTorch with CUDA for the `python` in the config, then run `py -m trainer selftest` again. |
| `torch.cuda.is_available()` is `False` | CUDA build or driver mismatch | Fix the driver or reinstall the matching PyTorch. Do not train on CPU without an approved decision. |
| `CUDA out of memory` in the log | Batch or input size too large for 8 GB | Lower the batch size or `extractor.input_hw` in the trainer settings. Use a size the engineer has tested on this laptop. The value is an engineering setting, not a Quality limit. |
| `nvidia-smi not found` warning | Driver not installed, or not on PATH | Reinstall the NVIDIA driver. The run continues only if the trainer can still use the device in the config. |
| Exit code `3` with `FAILED:` line | A preflight check failed. Nothing was trained | Read the `FAILED:` line in the log and fix that item. |
| Trainer reports counts below `min_good` or `min_holdout` | Not enough labelled parts yet | Expected until Quality's minimums are met. Quality decides whether to collect more. No candidate should be produced. (Check the trainer's actual output once it exists.) |

## 7. Purging old evidence (manual, separate, OFF by default)

Evidence is never deleted by the weekly task. Purging is a separate command that an engineer
runs by hand, after Quality agrees which evidence may go (retention rules: `[Quality to define]`).

1. Dry run first. It lists what would be removed and deletes nothing (planned command):

   ```powershell
   py -m trainer purge --help
   ```

2. Only after the dry run is reviewed and approved in writing, run the confirm step that the
   command's help text names. Never add a purge to the scheduled task.

## 8. What is UNPROVEN

- The trainer commands `weekly`, `promote`, `purge` and `selftest` do not exist yet.
- CUDA PyTorch on this laptop (i7 10th gen, 8 GB NVIDIA GPU): install, speed and memory.
- Whether the batch and input size fit in 8 GB for the chosen extractor.
- Task Scheduler wake and StartWhenAvailable behaviour on this laptop's power plan.
- That the OneDrive "online-only" flags are seen correctly on this laptop's OneDrive client.
  Test it once: set a file to online-only on purpose and run the evidence check.
- A full run on real evidence. The first run is a supervised run, not an unattended one.
- The time a full weekly run takes. Measure it on the first two runs.

## 9. Open decisions

1. **Trainer commands.** Who builds `weekly`, `promote`, `purge` and `selftest`, and when.
   The scripts call `weekly` with the arguments in `run_weekly_training.ps1`. The trainer side
   must match.
2. **Extractor config format.** The run script writes `extractor-config.json` with an
   `extractor` section to the logs folder. The trainer must accept it. Confirm the format.
3. **Account for the task.** Which Windows account runs the task, and who changes its
   password. A password change stops the task silently until the install script runs again.
4. **Who approves and who promotes.** Named people for Quality approval and for the
   promotion command. Record the names here: `[Quality to define]`.
5. **Retention.** How long evidence and candidate models are kept, and who approves a purge.
   `[Quality to define]`.
6. **Minimum counts and holdout share.** `min_good`, `min_holdout` and `holdout_fraction`
   (must be strictly between 0 and 1; no default). `[Quality to define]`.
7. **Missed runs.** Whether a run that starts late (laptop woke at 10:00) is acceptable, or
   whether Quality needs the Monday report to say when the run actually started.
8. **Second laptop or backup.** Whether a second machine is needed if this one is off.
