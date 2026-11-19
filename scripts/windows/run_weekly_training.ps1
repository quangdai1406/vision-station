
[CmdletBinding()]
param(
    [string]$Config = (Join-Path $env:USERPROFILE 'VisionStation\trainer.json')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RECALL_ON_DATA_ACCESS = 0x00400000
$RECALL_ON_OPEN        = 0x00040000
$LOG_KEEP              = 26

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$exitCode = 0
$transcriptStarted = $false

function Step([string]$Message) { Write-Host ""; Write-Host "== $Message" }
function Warn([string]$Message) { Write-Host "WARNING: $Message" }

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
    $hwProp = $extractor.PSObject.Properties['input_hw']
    $hwRaw = if ($hwProp) { @($hwProp.Value) } else { @() }
    if ($hwRaw.Count -ne 2 -or ($hwRaw | Where-Object { $_ -isnot [int] -and $_ -isnot [long] })) {
        throw "config: 'extractor.input_hw' must be a list of two whole numbers, for example [480, 640] (still a placeholder?)"
    }
    $inputHw = [int[]]@($hwRaw[0], $hwRaw[1])
    $extractorSettings = [ordered]@{
        arch         = Get-RequiredValue $extractor 'arch' 'extractor'
        input_hw     = $inputHw
        weights_path = Get-RequiredValue $extractor 'weights_path' 'extractor'
        device       = Get-RequiredValue $extractor 'device' 'extractor'
    }

    $minGood    = Get-RequiredValue $cfg 'min_good'
    $minHoldout = Get-RequiredValue $cfg 'min_holdout'
    if ($minGood -notmatch '^\d+$')    { throw "config: 'min_good' must be a whole number" }
    if ($minHoldout -notmatch '^\d+$') { throw "config: 'min_holdout' must be a whole number" }

    $holdoutRaw = Get-RequiredValue $cfg 'holdout_fraction'
    $holdoutValue = 0.0
    $holdoutOk = [double]::TryParse($holdoutRaw, [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture, [ref]$holdoutValue)
    if (-not $holdoutOk -or $holdoutValue -le 0 -or $holdoutValue -gt 0.9) {
        throw "config: 'holdout_fraction' must be a number greater than 0 and at most 0.9"
    }
    $holdoutFraction = $holdoutValue.ToString([Globalization.CultureInfo]::InvariantCulture)
    Write-Host "  settings ok (values not printed here; see the config file)"

    New-Item -ItemType Directory -Force -Path $logs | Out-Null
    $oldLogs = @(Get-ChildItem -LiteralPath $logs -Filter 'weekly-*.log' -File |
        Sort-Object LastWriteTime -Descending)
    $oldLogs | Select-Object -Skip ($LOG_KEEP - 1) | Remove-Item -Force

    $logFile = Join-Path $logs ('weekly-{0}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    Start-Transcript -LiteralPath $logFile -Force | Out-Null
    $transcriptStarted = $true
    Write-Host "Weekly training started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') on $env:COMPUTERNAME"

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

    Step 'Inference service check'
    $inferenceBusy = $false
    $healthProp = $cfg.PSObject.Properties['inference_health_url']
    if (-not $healthProp -or [string]::IsNullOrWhiteSpace([string]$healthProp.Value)) {
        Write-Host '  inference_health_url not set: busy check skipped'
    } else {
        $healthUrl = Get-RequiredValue $cfg 'inference_health_url'
        try {
            $health = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 5
            if (-not $health.PSObject.Properties['busy'] -or $health.busy -isnot [bool]) {
                Warn 'inference service answer has no boolean busy; continuing with training.'
            } elseif ($health.busy) {
                $inferenceBusy = $true
                Write-Host '  inference service reports busy'
            } else {
                Write-Host '  inference service idle'
            }
        } catch {
            Warn "inference service unreachable ($($_.Exception.GetType().Name)); continuing with training."
        }
    }

    if ($inferenceBusy) {
        Write-Host 'REFUSED: the inference service reports a request in progress. Training did not start.'
        Write-Host 'Start the task again when the service is idle, or run the script by hand.'
        $exitCode = 4
    } else {
        Step 'Running the trainer'
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
        $trainerLine = ($trainerArgs | ForEach-Object {
            $a = [string]$_
            if ($a -match '\s') { '"' + $a + '"' } else { $a }
        }) -join ' '
        $stdoutFile = Join-Path $logs 'trainer-stdout.txt'
        $stderrFile = Join-Path $logs 'trainer-stderr.txt'

        $proc = Start-Process -FilePath $python -ArgumentList $trainerLine -WorkingDirectory $RepoRoot `
            -NoNewWindow -PassThru -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
        $proc.PriorityClass = [Diagnostics.ProcessPriorityClass]::BelowNormal
        $null = $proc.Handle
        $proc.WaitForExit()
        $exitCode = $proc.ExitCode

        foreach ($f in @($stdoutFile, $stderrFile)) {
            if (Test-Path -LiteralPath $f -PathType Leaf) {
                Get-Content -LiteralPath $f -Encoding UTF8 | ForEach-Object { Write-Host $_ }
            }
        }
        Write-Host "Trainer exit code: $exitCode"
    }
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
