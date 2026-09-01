# Fetch the AMD NPU kernel binaries (.xclbin) phlegm drives but does NOT
# redistribute (see NOTICE.md). They live in FastFlowLM's public repo and are
# proprietary, patent-pending, free to use only under FLM's revenue-capped
# terms: https://github.com/ROCm/FastFlowLM/blob/main/TERMS.md — by fetching
# them you accept those terms.
#
# Usage:
#   pwsh -File tools/get-kernels.ps1              # layer + lm_head (what the engine uses)
#   pwsh -File tools/get-kernels.ps1 -All         # also the op-level prefill kernels
#   pwsh -File tools/get-kernels.ps1 -OutDir D:\kernels
# Then point the engine at them:  $env:FLM_XCLBIN_DIR = "<OutDir>"
param(
  [string]$Model = "Qwen3.6-35B-A3B-NPU2",
  # PINNED to the FLM release our captured kernel ELFs pair with. Upstream
  # main already carries different bytes (v1.0.3) — do not bump this without
  # re-capturing the ELFs against the new engine.
  [string]$Ref = "v1.0.2",
  [string]$OutDir = "",
  [switch]$All
)
$ErrorActionPreference = "Stop"
$base = "https://raw.githubusercontent.com/ROCm/FastFlowLM/$Ref/src/xclbins/$Model"
if (-not $OutDir) { $OutDir = Join-Path $PSScriptRoot "..\kernels\$Model" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$files = @("layer.xclbin", "lm_head.xclbin")
if ($All) { $files += @("mm.xclbin", "dequant_mm.xclbin", "attn.xclbin", "GateDeltaNet_prefill.xclbin", "conv.xclbin") }

Write-Host "Fetching $($files.Count) kernel(s) for $Model -> $OutDir"
Write-Host "These binaries are proprietary (FLM TERMS.md, revenue-capped free use)."
foreach ($f in $files) {
  $dst = Join-Path $OutDir $f
  Invoke-WebRequest -Uri "$base/$f" -OutFile $dst
  Write-Host ("  {0,-28} {1,10:N0} B" -f $f, (Get-Item $dst).Length)
}
Write-Host "Done. Set `$env:FLM_XCLBIN_DIR = `"$((Resolve-Path $OutDir).Path)`" before running the engine."
