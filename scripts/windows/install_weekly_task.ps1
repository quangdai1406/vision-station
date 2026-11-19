
[CmdletBinding()]
param(
    [string]$Config = (Join-Path $env:USERPROFILE 'VisionStation\trainer.json'),
    [switch]$Uninstall,
    [switch]$WhatIf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$TaskName  = 'VisionStation Weekly Training'
$RunScript = Join-Path $PSScriptRoot 'run_weekly_training.ps1'
$WorkDir   = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($Uninstall) {
    if (-not $existing) {
        Write-Host "Task '$TaskName' is not registered. Nothing to remove."
        exit 0
    }
    if ($WhatIf) {
        Write-Host "DRY RUN: would remove the scheduled task '$TaskName'. Nothing changed."
        exit 0
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed the scheduled task '$TaskName'."
    exit 0
}

if (-not (Test-Path -LiteralPath $RunScript -PathType Leaf)) {
    throw "run script not found: $RunScript"
}
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    Write-Host "WARNING: config not found: $Config. The task will fail until it exists (copy scripts\windows\trainer.example.json and fill it in)."
}

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -File "{0}" -Config "{1}"' -f $RunScript, $Config) `
    -WorkingDirectory $WorkDir

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At '06:00'

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries:$false `
    -DontStopIfGoingOnBatteries:$false

$state = if ($existing) { 'replace the existing task' } else { 'create a new task' }

if ($WhatIf) {
    Write-Host "DRY RUN: would $state named '$TaskName'. Nothing changed. Details:"
    Write-Host "  action:    powershell.exe -File `"$RunScript`" -Config `"$Config`""
    Write-Host "  folder:    $WorkDir"
    Write-Host "  trigger:   weekly, Sunday, 06:00 local time"
    Write-Host "  account:   the account you enter when asked (run whether logged on or not), run level Limited"
    Write-Host "  settings:  StartWhenAvailable, WakeToRun, 6 h time limit, do not start a second instance, AC power only"
    exit 0
}

$account = Read-Host "Windows account the task runs as (DOMAIN\user). schtasks will then ask for its password"
if ([string]::IsNullOrWhiteSpace($account)) {
    throw 'no account given; the task was not changed'
}

& schtasks.exe /Create /F /TN $TaskName /SC WEEKLY /D SUN /ST 06:00 /RL LIMITED /RU $account /RP * /TR 'powershell.exe'
if ($LASTEXITCODE -ne 0) {
    throw "schtasks could not create the task (exit code $LASTEXITCODE); nothing else was changed"
}

$principal = New-ScheduledTaskPrincipal -UserId $account -LogonType Password -RunLevel Limited
Set-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null

Write-Host "Registered '$TaskName': every Sunday at 06:00 local time."
Write-Host "Check it with: Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
