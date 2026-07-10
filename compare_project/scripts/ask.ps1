# Ask Bastion a single prompt and see the route, model, tokens, and answer.
#
#   powershell -ExecutionPolicy Bypass -File scripts/ask.ps1 "What is 25% of 80? Return only the number."
#   powershell -ExecutionPolicy Bypass -File scripts/ask.ps1 "Explain ROCm in one sentence." -Model accounts/fireworks/models/kimi-k2p7-code
#   powershell -ExecutionPolicy Bypass -File scripts/ask.ps1 "Calculate 6 * 7." -NoRemote   # local shortcuts only, zero tokens
#
# Default model is minimax-m3 (serverless). Gemma models are deployment-only,
# so point -Model at a running deployment reference to test those.
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Prompt,
    [string]$Model = "accounts/fireworks/models/minimax-m3",
    [switch]$NoRemote
)

$env:FIREWORKS_API_KEY = [Environment]::GetEnvironmentVariable("FIREWORKS_API_KEY", "User")
$env:FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
$env:ALLOWED_MODELS = $Model

# Output must be .jsonl: the .json writer strips to the official {task_id, answer}
# shape, whereas .jsonl preserves the route/model/token debug fields.
$taskFile = Join-Path $env:TEMP "frugal_ask_task.json"
$outFile = Join-Path $env:TEMP "frugal_ask_result.jsonl"
$obj = @{ task_id = "ask"; prompt = $Prompt } | ConvertTo-Json -Compress -Depth 5
Set-Content -Path $taskFile -Value "[$obj]" -Encoding utf8

$cliArgs = @("-m", "frugalrouter.cli", "--tasks", $taskFile, "--output", $outFile, "--debug-output")
if ($NoRemote) { $cliArgs += "--no-remote" }
python @cliArgs | Out-Null

$r = Get-Content $outFile | Where-Object { $_.Trim() } | Select-Object -First 1 | ConvertFrom-Json
$u = $r.usage
$total = [int]$u.prompt_tokens + [int]$u.completion_tokens

Write-Host ""
Write-Host "PROMPT    : $Prompt"
Write-Host "-----------------------------------------------------------"
Write-Host "ANSWER    : $($r.answer)"
Write-Host "-----------------------------------------------------------"
Write-Host ("ROUTE     : {0}   (category: {1}, used_remote: {2})" -f $r.route, $r.category, $r.used_remote)
Write-Host ("MODEL     : {0}" -f $u.model)
Write-Host ("TOKENS    : {0} total   ({1} prompt + {2} completion)" -f $total, $u.prompt_tokens, $u.completion_tokens)
Write-Host ("LATENCY   : {0} ms" -f $u.latency_ms)
if ($r.fallback_reason) { Write-Host ("NOTE      : {0}" -f $r.fallback_reason) }
Write-Host ""
