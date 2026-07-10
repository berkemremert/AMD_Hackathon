# Pre-download the Linux wheels that Dockerfile.local installs offline.
# Run this on a machine with a reliable connection (Docker Desktop's build
# network on some setups is flaky); the vendored wheels then let the image
# build with no network at all.
#
#   powershell -ExecutionPolicy Bypass -File scripts/fetch_wheels.ps1
$ErrorActionPreference = "Stop"
Remove-Item -Recurse -Force wheels -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path wheels | Out-Null

python -m pip download `
  --only-binary=:all: --python-version 3.11 --implementation cp --abi cp311 `
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 `
  -d wheels --timeout 600 --retries 15 `
  llama-cpp-python numpy diskcache jinja2 MarkupSafe typing-extensions `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

Write-Host "`nWheels in ./wheels:"
Get-ChildItem wheels | ForEach-Object { "  {0}  ({1:N1} MB)" -f $_.Name, ($_.Length / 1MB) }
