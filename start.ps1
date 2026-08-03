[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "[Startup failed] Virtual environment not found: $Python" -ForegroundColor Red
    Write-Host "Run: .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

Write-Host "Starting Work Assistant..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor DarkGray
& $Python "server.py"
$ServerExitCode = $LASTEXITCODE

if ($ServerExitCode -ne 0) {
    Write-Host "Work Assistant exited with code: $ServerExitCode" -ForegroundColor Red
}

exit $ServerExitCode
