param(
    [string]$Tasks = "data/eval_tasks.json",
    [string]$OutDir = "reports",
    [int]$Workers = 4,
    [switch]$Remote,
    [int]$Limit = 0,
    [double]$Delay = 0
)

$ErrorActionPreference = "Stop"

$argsList = @(
    "-m", "frugalrouter.eval_runner",
    "--tasks", $Tasks,
    "--out-dir", $OutDir,
    "--workers", "$Workers"
)

if (-not $Remote) {
    $argsList += "--no-remote"
}

if ($Limit -gt 0) {
    $argsList += @("--limit", "$Limit")
}

if ($Delay -gt 0) {
    $argsList += @("--delay", "$Delay")
}

python @argsList

Write-Host ""
Write-Host "Wrote:"
Write-Host "  $OutDir\eval_report.json"
Write-Host "  $OutDir\eval_results.json"

