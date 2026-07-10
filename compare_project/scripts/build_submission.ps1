# Build (and optionally publish) the Track 1 submission image.
#
# Build + tag only:
#   powershell -ExecutionPolicy Bypass -File scripts/build_submission.ps1 -Registry docker.io/kgotsomsiza
#
# Build, tag, and push publicly (only when ready to submit):
#   powershell -ExecutionPolicy Bypass -File scripts/build_submission.ps1 -Registry docker.io/kgotsomsiza -Push
param(
    [Parameter(Mandatory = $true)][string]$Registry,
    [string]$Name = "bastion",
    [string]$Tag = "track1",
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$image = "$Registry/$Name`:$Tag"

# Stamp the exact source commit into the image so any digest self-identifies.
# The :track1 tag is mutable; the commit label + digest are the durable identity.
$commit = (git rev-parse --short HEAD).Trim()
$fullCommit = (git rev-parse HEAD).Trim()
# Only tracked changes matter: the Dockerfile packages tracked dirs only, so
# untracked artifacts (slides, scratch files) must not flag the build as dirty.
$dirty = (git status --porcelain --untracked-files=no)
if ($dirty) {
    Write-Warning "Working tree is DIRTY. Build image <-> commit will NOT be exact. Commit first for clean version tracking."
    $commit = "$commit-dirty"
}
$builtAt = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")

Write-Host "Running tests first..."
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Tests failed; not building." }

Write-Host "Building linux/amd64 image $image (commit $commit)"
docker buildx build --platform linux/amd64 `
    --label "bastion.commit=$commit" `
    --label "org.opencontainers.image.revision=$fullCommit" `
    --label "bastion.built=$builtAt" `
    -t $image -t "$Name`:$Tag" .
if ($LASTEXITCODE -ne 0) { throw "Docker build failed." }

Write-Host "Verifying offline container contract..."
$smokeDir = Join-Path $env:TEMP "bastion-smoke"
New-Item -ItemType Directory -Force -Path "$smokeDir\input", "$smokeDir\output" | Out-Null
Copy-Item data\sample_tasks.json "$smokeDir\input\tasks.json" -Force
docker run --rm -e FRUGAL_ALLOW_REMOTE=0 -v "$smokeDir\input:/input" -v "$smokeDir\output:/output" $image
if ($LASTEXITCODE -ne 0) { throw "Container smoke test failed (exit $LASTEXITCODE)." }
if (-not (Test-Path "$smokeDir\output\results.json")) { throw "Container did not write results.json." }
Write-Host "Contract OK: reads /input/tasks.json, writes /output/results.json, exits 0."

if ($Push) {
    Write-Host "Pushing $image (this makes the image publicly pullable)..."
    docker push $image
    if ($LASTEXITCODE -ne 0) { throw "Push failed." }
    $digest = (docker buildx imagetools inspect $image --format '{{.Manifest.Digest}}' 2>$null)
    Write-Host ""
    Write-Host "==================== SUBMISSION LEDGER ENTRY ===================="
    Write-Host "  image  : $image"
    Write-Host "  commit : $commit"
    Write-Host "  digest : $digest"
    Write-Host "  builtAt: $builtAt"
    Write-Host "  -> Add a new V<n> row to AGENT_CONTEXT.local.md with these values,"
    Write-Host "     mark it LIVE / AWAITING SCORE, and clear the previous LIVE marker."
    Write-Host "     Title the lablab resubmission with V<n> so the score maps cleanly."
    Write-Host "================================================================"
} else {
    Write-Host "Not pushing (-Push not set). Image is built and tagged locally as $image (commit $commit)"
}
