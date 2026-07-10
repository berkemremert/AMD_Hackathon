# Rescue script for the Windows AF_UNIX stale-socket bug that crashes Docker Desktop
# at startup with errors like:
#   "initializing Inference manager / Secrets Engine: listening on unix://...:
#    remove ...: The file cannot be accessed by the system."
#
# Stale Unix-socket files from a previous Docker session become undeletable through
# the Windows file API, so the backend dies trying to recreate them. Deleting them
# from inside WSL works. Run this, then wait for the whale icon to go green.

$ErrorActionPreference = "Continue"

Write-Host "Stopping Docker Desktop..."
Get-Process "Docker Desktop", "com.docker.backend", "docker" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

$socketDirs = @(
    "$env:LOCALAPPDATA\Docker\run",
    "$env:LOCALAPPDATA\docker-secrets-engine"
)

foreach ($dir in $socketDirs) {
    if (-not (Test-Path $dir)) { continue }
    $wslPath = "/mnt/" + $dir.Substring(0, 1).ToLower() + ($dir.Substring(2) -replace "\\", "/")
    Write-Host "Clearing stale sockets in $dir"
    wsl -d Ubuntu -- rm -rf $wslPath
    if (Test-Path $dir) {
        # WSL deletion can silently miss; fall back to renaming the folder aside.
        $aside = "$dir.stale.$(Get-Date -Format yyyyMMdd-HHmmss)"
        try { Rename-Item $dir $aside -Force; Write-Host "  moved aside to $aside" } catch {}
    }
}

Write-Host "Starting Docker Desktop..."
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

$deadline = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $deadline) {
    docker info *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker engine is up: $(docker info --format '{{.ServerVersion}}')"
        exit 0
    }
    Start-Sleep -Seconds 5
}

Write-Host "Engine did not come up within 5 minutes. Last backend log lines:"
Get-Content "$env:LOCALAPPDATA\Docker\log\host\com.docker.backend.exe.log" -Tail 10
exit 1
