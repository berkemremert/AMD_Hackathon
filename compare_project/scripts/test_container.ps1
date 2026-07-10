# Run the built Track 1 container exactly the way the judges will: read
# /input/tasks.json, write /output/results.json, using the Fireworks env.
# Edit input\tasks.json (or pass -Tasks <file>) to try your own prompts.
#
#   powershell -ExecutionPolicy Bypass -File scripts/test_container.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/test_container.ps1 -Tasks mytasks.json
#
# Note: local testing uses serverless models (minimax/kimi) because the
# real Gemma models aren't serverless. The container LOGIC is identical to
# what the harness runs; only the model IDs differ (the harness injects its
# own ALLOWED_MODELS, which our policy routes to just the same).
param(
    [string]$Tasks = "data/sample_tasks.json",
    [string]$Image = "bastion:track1",
    [string]$Models = "accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code",
    [int]$Workers = 2
)

New-Item -ItemType Directory -Force -Path input, output | Out-Null
# Task files may contain // and /* */ comments (handy for templates). The
# container needs strict JSON, so strip comments here before handing it over.
$raw = Get-Content $Tasks -Raw
$raw = [regex]::Replace($raw, '/\*[\s\S]*?\*/', '')
$raw = ($raw -split "`r?`n" | Where-Object { $_ -notmatch '^\s*//' }) -join "`n"
Set-Content -Path input\tasks.json -Value $raw -Encoding utf8
$key = [Environment]::GetEnvironmentVariable("FIREWORKS_API_KEY", "User")

Write-Host "Running $Image on $Tasks ..." -ForegroundColor Yellow
docker run --rm `
    -e FIREWORKS_API_KEY=$key `
    -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" `
    -e ALLOWED_MODELS=$Models `
    -e FRUGAL_WORKERS=$Workers `
    -v ${PWD}\input:/input -v ${PWD}\output:/output `
    $Image
$code = $LASTEXITCODE
Write-Host "`ncontainer exit code: $code  (0 = success)" -ForegroundColor $(if ($code -eq 0) { "Green" } else { "Red" })

# Validate the output contract and print answers.
try {
    $results = Get-Content output\results.json -Raw | ConvertFrom-Json
    $empty = @($results | Where-Object { -not $_.answer -or $_.answer.Trim() -eq "" }).Count
    Write-Host ("output: {0} rows | valid JSON | empty answers: {1}" -f $results.Count, $empty) -ForegroundColor $(if ($empty -eq 0) { "Green" } else { "Red" })
    foreach ($r in $results) {
        Write-Host ""
        Write-Host ("[{0}]" -f $r.task_id) -ForegroundColor Cyan
        Write-Host $r.answer
    }
}
catch {
    Write-Host "!! Could not read/parse output\results.json - this would fail as INVALID_RESULTS_SCHEMA" -ForegroundColor Red
}
