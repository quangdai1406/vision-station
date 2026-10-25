# run_weekly_training.ps1
# Weekly training run for the vision-station laptop (Windows 10/11, i7 10th gen, 32 GB RAM,
# 8 GB NVIDIA GPU). Scheduled by install_weekly_task.ps1 for every Sunday at 06:00 local time.
#
# In order, this script:
#   1. reads the settings from the config JSON (-Config),
#   2. checks that the evidence folder exists, is not empty and is not online-only
#      (OneDrive placeholder files), so it never trains on a partial set,
#   3. checks the GPU with nvidia-smi when it is installed,
#   4. runs the trainer: python -m trainer weekly (it writes a candidate model and a report),
#   5. writes a transcript log (the last 26 are kept) and exits with the trainer's exit code.
#
# Exit codes: 3 = preflight check failed (nothing was trained); any other code is the
# trainer's own exit code.
#
# This script never approves or deploys a model, and it never deletes evidence.
# Approval is a named person's manual step, taken after Quality has reviewed the report.
#
# ---------------------------------------------------------------------------
# PURGE SWITCH: OFF. This script has no purge step and must not gain one by default.
# Deleting evidence is a separate MANUAL command that an engineer runs by hand: dry run
# first, then the confirm step. It is never run from this scheduled task.
# See docs/weekly_training.md, section "Purging old evidence".
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [string]$Config = (Join-Path $env:USERPROFILE 'VisionStation\trainer.json')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# OneDrive "online-only" attribute bits. A file with either bit set is a placeholder whose
# content is not on this laptop yet.
$RECALL_ON_DATA_ACCESS = 0x00400000
$RECALL_ON_OPEN        = 0x00040000
$LOG_KEEP              = 26

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$exitCode = 0
$transcriptStarted = $false

function Step([string]$Message) { Write-Host ""; Write-Host "== $Message" }
function Warn([string]$Message) { Write-Host "WARNING: $Message" }

# Returns the string value of a key, expands %VARS%, and refuses a missing, empty or
# placeholder value. Only the key name is printed, never a value.
function Get-RequiredValue($Obj, [string]$Key, [string]$Prefix = '') {
    $name = if ($Prefix) { "$Prefix.$Key" } else { $Key }
    $prop = $Obj.PSObject.Properties[$Key]
    if (-not $prop -or $null -eq $prop.Value -or [string]::IsNullOrWhiteSpace([string]$prop.Value)) {
        throw "config: '$name' is missing or empty in $Config"
    }
    $value = [string]$prop.Value
    if ($value -match '^\[.*\]$') {
        throw "config: '$name' is still a placeholder. Replace it in $Config before the weekly run can start."
    }
    return [Environment]::ExpandEnvironmentVariables($value)
}

function Get-Section($Obj, [string]$Key) {
    $prop = $Obj.PSObject.Properties[$Key]
    if (-not $prop -or $null -eq $prop.Value) {
        throw "config: section '$Key' is missing in $Config"
    }
    return $prop.Value
}

try {
    # 1. Settings -------------------------------------------------------------
    Step 'Reading settings'
    if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
        throw "config file not found: $Config. Copy scripts\windows\trainer.example.json there and fill it in."
    }
    $cfg = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json

    $store   = Get-RequiredValue $cfg 'store'
    $models  = Get-RequiredValue $cfg 'models'
    $reports = Get-RequiredValue $cfg 'reports'
    $logs    = Get-RequiredValue $cfg 'logs'
    $python  = Get-RequiredValue $cfg 'python'

    $extractor = Get-Section $cfg 'extractor'
    $extractorSettings = [ordered]@{
        arch         = Get-RequiredValue $extractor 'arch' 'extractor'
        input_hw     = Get-RequiredValue $extractor 'input_hw' 'extractor'
        weights_path = Get-RequiredValue $extractor 'weights_path' 'extractor'
        device       = Get-RequiredValue $extractor 'device' 'extractor'
    }

    # No defaults for these two: a missing value stops the run.
    $minGood    = Get-RequiredValue $cfg 'min_good'
    $minHoldout = Get-RequiredValue $cfg 'min_holdout'
    if ($minGood -notmatch '^\d+$')    { throw "config: 'min_good' must be a whole number" }
    if ($minHoldout -notmatch '^\d+$') { throw "config: 'min_holdout' must be a whole number" }

    # holdout_fraction: Quality owns it, no default. Must be a number strictly between 0 and 1.
    $holdoutRaw = Get-RequiredValue $cfg 'holdout_fraction'
    $holdoutValue = 0.0
    $holdoutOk = [double]::TryParse($holdoutRaw, [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture, [ref]$holdoutValue)
    if (-not $holdoutOk -or $holdoutValue -le 0 -or $holdoutValue -ge 1) {
        throw "config: 'holdout_fraction' must be a number greater than 0 and less than 1"
    }
    $holdoutFraction = $holdoutValue.ToString([Globalization.CultureInfo]::InvariantCulture)
    Write-Host "  settings ok (values not printed here; see the config file)"

    # Log folder and rotation: keep the last 26 transcripts, including this run.
    New-Item -ItemType Directory -Force -Path $logs | Out-Null
    $oldLogs = @(Get-ChildItem -LiteralPath $logs -Filter 'weekly-*.log' -File |
        Sort-Object LastWriteTime -Descending)
    $oldLogs | Select-Object -Skip ($LOG_KEEP - 1) | Remove-Item -Force

    $logFile = Join-Path $logs ('weekly-{0}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    Start-Transcript -LiteralPath $logFile -Force | Out-Null
    $transcriptStarted = $true
    Write-Host "Weekly training started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') on $env:COMPUTERNAME"

    # 2. Evidence folder ------------------------------------------------------
    Step 'Checking the evidence folder'
    if (-not (Test-Path -LiteralPath $store -PathType Container)) {
        throw "evidence folder not found: $store. Is the SharePoint folder synced by OneDrive?"
    }
    $files = @(Get-ChildItem -LiteralPath $store -Recurse -File)
    if ($files.Count -eq 0) {
        throw "evidence folder is empty: $store. Nothing to train on."
    }
    $online = @($files | Where-Object {
        (([int]$_.Attributes) -band ($RECALL_ON_DATA_ACCESS -bor $RECALL_ON_OPEN)) -ne 0
    })
    if ($online.Count -gt 0) {
        $sample = ($online | Select-Object -First 5 | ForEach-Object { '  ' + $_.Name }) -join "`n"
        throw ("{0} of {1} evidence files are online-only (OneDrive placeholders).`n" +
               "Right-click the evidence folder in OneDrive, choose 'Always keep on this device', " +
               "wait for the sync to finish, then run again. Not training on a partial set.`n" +
               "First few:`n{2}") -f $online.Count, $files.Count, $sample
    }
    Write-Host ("  {0} evidence files, all on this device" -f $files.Count)

    foreach ($dir in @($models, $reports)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }

    # 3. GPU ------------------------------------------------------------------
    Step 'GPU check'
    $smi = Get-Command -Name 'nvidia-smi' -ErrorAction SilentlyContinue
    if (-not $smi) {
        Warn 'nvidia-smi not found: no NVIDIA driver visible. The trainer will use the device in the config or fail.'
    } else {
        try {
            $gpuInfo = & $smi.Source '--query-gpu=name,memory.total,driver_version' '--format=csv,noheader' 2>$null
            if ($LASTEXITCODE -eq 0 -and $gpuInfo) {
                $gpuInfo | ForEach-Object { Write-Host "  GPU: $_" }
            } else {
                Warn 'nvidia-smi ran but returned no GPU data.'
            }
        } catch {
            Warn "nvidia-smi failed: $($_.Exception.Message)"
        }
    }

    # 4. Trainer ----------------------------------------------------------------
    Step 'Running the trainer'
    # The trainer reads the extractor section from its own JSON file; write it to the logs folder.
    $extractorFile = Join-Path $logs 'extractor-config.json'
    $extractorSettings | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath $extractorFile -Encoding UTF8

    $trainerArgs = @(
        '-m', 'trainer', 'weekly',
        '--store', $store,
        '--models', $models,
        '--reports', $reports,
        '--min-good', $minGood,
        '--min-holdout', $minHoldout,
        '--holdout-fraction', $holdoutFraction,
        '--extractor-config', $extractorFile
    )
    Push-Location -LiteralPath $RepoRoot
    try {
        & $python @trainerArgs
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    Write-Host "Trainer exit code: $exitCode"
}
catch {
    Write-Host "FAILED: $($_.Exception.Message)"
    $exitCode = 3
}
finally {
    if ($transcriptStarted) {
        Write-Host "Log: $logFile"
        Stop-Transcript | Out-Null
    }
}

exit $exitCode
