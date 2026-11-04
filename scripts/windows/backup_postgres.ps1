# backup_postgres.ps1
# Nightly backup of the MES PostgreSQL database with pg_dump (custom format, one file per run).
# Scheduled by install_backup_task.ps1 every day at 22:00 local time (engineering default).
#
# In order, this script:
#   1. reads the settings from the backup JSON (-Config). A value that starts with '[' is a
#      placeholder and stops the run (exit 3). Nothing is dumped until every value is set.
#   2. writes a transcript log to <backup_folder>\logs,
#   3. runs pg_dump in custom format into <database>-<UTC timestamp>.dump.partial, then renames
#      it to .dump only when pg_dump exited 0 and the file is not empty,
#   4. keeps the newest keep_last dumps and removes older dumps of this database (only after a
#      new dump has been verified). Nothing else in the folder is touched.
#
# Exit codes: 0 = dump written and verified; 3 = settings or preflight failed (no dump was
# attempted); 5 = pg_dump failed or the dump could not be verified (no dump was kept, see the
# log; a .partial file may remain and can be deleted by hand).
#
# Connection: pg_dump reads the standard PostgreSQL settings from the environment of the
# account that runs the task: PGHOST, PGPORT, PGUSER and PGPASSFILE (the password file, by
# default %APPDATA%\postgresql\pgpass.conf). This script never takes a password on the command
# line, never reads a password variable and never writes one to a file or to the log.
# --no-password makes pg_dump fail instead of waiting for a prompt.
#
# This script does not delete evidence, does not restore anything and does not change the MES.

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

# Returns the string value of a key, expands %VARS%, and refuses a missing, empty or
# placeholder value. Only the key name is printed, never a value.
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
    # 1. Settings -------------------------------------------------------------
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

    # 2. Folder and transcript ------------------------------------------------
    $logsFolder = Join-Path $backupFolder 'logs'
    New-Item -ItemType Directory -Force -Path $logsFolder | Out-Null
    $logFile = Join-Path $logsFolder ('backup-{0}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    Start-Transcript -LiteralPath $logFile -Force | Out-Null
    $transcriptStarted = $true
    Write-Host "MES backup started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') on $env:COMPUTERNAME"

    # 3. pg_dump ---------------------------------------------------------------
    Step 'Running pg_dump'
    $stamp   = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss')
    $name    = '{0}-{1}Z.dump' -f $database, $stamp
    $final   = Join-Path $backupFolder $name
    $partial = "$final.partial"

    $exitCode = 5   # any failure from here on is a dump failure
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

    # 4. Retention -------------------------------------------------------------
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
