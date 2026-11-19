
[CmdletBinding()]
param(
    [string]$Config = (Join-Path $env:USERPROFILE 'VisionStation\backup.json')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$exitCode = 0
$transcriptStarted = $false
$logFile = $null

function Step([string]$Message) { Write-Host ""; Write-Host "== $Message" }

function Get-RequiredValue($Obj, [string]$Key) {
    $prop = $Obj.PSObject.Properties[$Key]
    if (-not $prop -or $null -eq $prop.Value -or [string]::IsNullOrWhiteSpace([string]$prop.Value)) {
        throw "config: '$Key' is missing or empty in $Config"
    }
    $value = [string]$prop.Value
    if ($value -match '^\[.*\]$') {
        throw "config: '$Key' is still a placeholder. Replace it in $Config before the backup can run."
    }
    return [Environment]::ExpandEnvironmentVariables($value)
}

try {
    Step 'Reading settings'
    if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
        throw "config file not found: $Config. Copy scripts\windows\backup.example.json there and fill it in."
    }
    $cfg = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json

    $backupFolder = Get-RequiredValue $cfg 'backup_folder'
    $database     = Get-RequiredValue $cfg 'database'
    $pgDump       = Get-RequiredValue $cfg 'pg_dump'
    $keepRaw      = Get-RequiredValue $cfg 'keep_last'
    if ($keepRaw -notmatch '^\d+$' -or [int]$keepRaw -lt 1) {
        throw "config: 'keep_last' must be a whole number of 1 or more"
    }
    $keepLast = [int]$keepRaw
    if ($database -notmatch '^[A-Za-z0-9_]+$') {
        throw "config: 'database' may contain only letters, digits and underscore"
    }
    Write-Host '  settings ok (values not printed here; see the config file)'

    $logsFolder = Join-Path $backupFolder 'logs'
    New-Item -ItemType Directory -Force -Path $logsFolder | Out-Null
    $logFile = Join-Path $logsFolder ('backup-{0}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    Start-Transcript -LiteralPath $logFile -Force | Out-Null
    $transcriptStarted = $true
    Write-Host "MES backup started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') on $env:COMPUTERNAME"

    Step 'Running pg_dump'
    $stamp   = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
    $name    = '{0}-{1}Z.dump' -f $database, $stamp
    $final   = Join-Path $backupFolder $name
    $partial = "$final.partial"

    $exitCode = 5
    & $pgDump --format=custom --no-password --file $partial $database
    if ($LASTEXITCODE -ne 0) {
        throw "pg_dump failed with exit code $LASTEXITCODE. No new dump was kept. See the output above."
    }
    $size = (Get-Item -LiteralPath $partial).Length
    if ($size -le 0) {
        throw "pg_dump reported success but the dump file is empty ($name.partial). No new dump was kept."
    }
    Move-Item -LiteralPath $partial -Destination $final
    Write-Host ("  dump written: {0} ({1} bytes)" -f $name, $size)

    Step 'Keeping the newest dumps'
    $dumps = @(Get-ChildItem -LiteralPath $backupFolder -Filter ('{0}-*Z.dump' -f $database) -File |
        Sort-Object Name -Descending)
    $old = @($dumps | Select-Object -Skip $keepLast)
    foreach ($o in $old) {
        Write-Host "  removing old dump: $($o.Name)"
        Remove-Item -LiteralPath $o.FullName -Force
    }
    Write-Host ("  {0} dump(s) kept" -f [Math]::Min($dumps.Count, $keepLast))

    $exitCode = 0
}
catch {
    if ($exitCode -eq 0) { $exitCode = 3 }
    Write-Host "BACKUP FAILED: $($_.Exception.Message)"
}
finally {
    if ($transcriptStarted) {
        Write-Host "Log: $logFile"
        Stop-Transcript | Out-Null
    }
}

exit $exitCode
