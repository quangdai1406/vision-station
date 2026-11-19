
[CmdletBinding()]
param(
    [string]$Config = (Join-Path $env:USERPROFILE 'VisionStation\backup.json'),
    [ValidatePattern('^([01]\d|2[0-3]):[0-5]\d$')]
    [string]$Time = '22:00',
    [switch]$Uninstall,
    [switch]$WhatIf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$TaskName  = 'VisionStation MES Nightly Backup'
$RunScript = Join-Path $PSScriptRoot 'backup_postgres.ps1'
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

$at = [TimeSpan]::Parse($Time)
if ($at -ge [TimeSpan]::FromHours(5) -and $at -lt [TimeSpan]::FromHours(8)) {
    throw "-Time $Time is inside the 05:00-08:00 window used by the Sunday 06:00 weekly training. Choose another time."
}

if (-not (Test-Path -LiteralPath $RunScript -PathType Leaf)) {
    throw "backup script not found: $RunScript"
}
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    Write-Host "WARNING: config not found: $Config. The task will fail until it exists (copy scripts\windows\backup.example.json and fill it in)."
}

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument ('-NoProfile -ExecutionPolicy Bypass -File "{0}" -Config "{1}"' -f $RunScript, $Config) `
    -WorkingDirectory $WorkDir

$trigger = New-ScheduledTaskTrigger -Daily -At $Time

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries:$false `
    -DontStopIfGoingOnBatteries:$false

$state = if ($existing) { 'replace the existing task' } else { 'create a new task' }

if ($WhatIf) {
    Write-Host "DRY RUN: would $state named '$TaskName'. Nothing changed. Details:"
    Write-Host "  action:    powershell.exe -File `"$RunScript`" -Config `"$Config`""
    Write-Host "  folder:    $WorkDir"
    Write-Host "  trigger:   daily at $Time local time (engineering default 22:00)"
    Write-Host "  account:   the account you enter when asked (run whether logged on or not), run level Limited"
    Write-Host "  settings:  StartWhenAvailable, 2 h time limit, do not start a second instance, AC power only"
    exit 0
}

$account = Read-Host "Windows account the backup task runs as (DOMAIN\user). schtasks will then ask for its password"
if ([string]::IsNullOrWhiteSpace($account)) {
    throw 'no account given; the task was not changed'
}

& schtasks.exe /Create /F /TN $TaskName /SC DAILY /ST $Time /RL LIMITED /RU $account /RP * /TR 'powershell.exe'
if ($LASTEXITCODE -ne 0) {
    throw "schtasks could not create the task (exit code $LASTEXITCODE); nothing else was changed"
}

$principal = New-ScheduledTaskPrincipal -UserId $account -LogonType Password -RunLevel Limited
Set-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal | Out-Null

Write-Host "Registered '$TaskName': every day at $Time local time."
Write-Host "Check it with: Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
