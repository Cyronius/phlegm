# Drives one FLM tensor-data (xrt::bo) capture run.
#   - swaps a model/config variant into the fixed model.q4nx/config.json names
#   - arms FLM_BO_CAPTURE_DIR (+ optional FLM_BO_DUMP_MAX bytes/sync)
#   - starts server -> one prompt -> stop
# Does NOT restore the .orig files; the caller restores after the last run.
# Usage:
#   pwsh -File bo_capture.ps1 -Variant 6Li3 -CaptureDir C:\caps\bo_i3 [-DumpMax 0] [-MaxTokens 8]
param(
  [Parameter(Mandatory=$true)][string]$Variant,      # 8Li4 | 6Li3
  [Parameter(Mandatory=$true)][string]$CaptureDir,
  [long]$DumpMax = 0,
  [int]$Port = 52625,
  [int]$MaxTokens = 8,
  [string]$Prompt = "Say hi.",
  [string]$Tag = "qwen3.6-moe:35b-a3b",
  [string]$FlmDir = "C:\flm-test",
  [string]$ModelRoot = "C:\Users\josha\.flm",
  [string]$ModelDir = "C:\Users\josha\.flm\models\Qwen3.6-35B-A3B-NPU2"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $CaptureDir | Out-Null
Get-ChildItem -Path $CaptureDir -File -ErrorAction SilentlyContinue | Remove-Item -Force

# --- swap the variant into the fixed load names -----------------------------
$srcModel  = Join-Path $ModelDir "model_$Variant.q4nx"
$srcConfig = Join-Path $ModelDir "config_$Variant.json"
if (-not (Test-Path $srcModel))  { throw "missing $srcModel" }
if (-not (Test-Path $srcConfig)) { throw "missing $srcConfig" }
Write-Host "[bo] swapping variant $Variant into model.q4nx/config.json"
Copy-Item $srcModel  (Join-Path $ModelDir "model.q4nx")  -Force
Copy-Item $srcConfig (Join-Path $ModelDir "config.json") -Force

# --- env --------------------------------------------------------------------
$env:FLM_MODEL_PATH      = $ModelRoot          # the .flm ROOT (FLM appends \models)
$env:FLM_BO_CAPTURE_DIR  = $CaptureDir
$env:FLM_BO_DUMP_MAX     = "$DumpMax"
Remove-Item Env:\FLM_SEQ_CAPTURE_DIR -ErrorAction SilentlyContinue  # bo plane only

Write-Host "[bo] starting server (variant=$Variant dumpMax=$DumpMax cap=$CaptureDir)"
$p = Start-Process -FilePath (Join-Path $FlmDir "flm.exe") `
     -ArgumentList @("serve", $Tag, "--port", "$Port", "--quiet") `
     -WorkingDirectory $FlmDir -PassThru -WindowStyle Hidden

$base = "http://127.0.0.1:$Port"
$ready = $false
for ($i = 0; $i -lt 180; $i++) {
  if ($p.HasExited) { Write-Host "[bo] server exited early (code $($p.ExitCode))"; break }
  try { Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 3 | Out-Null; $ready = $true; break }
  catch { Start-Sleep -Seconds 2 }
}
if (-not $ready) {
  Write-Host "[bo] server never became ready; killing"
  if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
  exit 1
}
Write-Host "[bo] server ready after ~$($i*2)s; sending prompt"

$body = @{ model=$Tag; messages=@(@{role="user"; content=$Prompt}); max_tokens=$MaxTokens; stream=$false } | ConvertTo-Json -Depth 5
try {
  $r = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec 300
  Write-Host "[bo] response: $($r.choices[0].message.content)"
} catch { Write-Host "[bo] chat request failed: $($_.Exception.Message)" }

Start-Sleep -Seconds 1
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
$rows = 0
$tf = Join-Path $CaptureDir "bo_trace.tsv"
if (Test-Path $tf) { $rows = (Get-Content $tf).Count }
$dumps = (Get-ChildItem -Path $CaptureDir -Filter *.bo -ErrorAction SilentlyContinue).Count
Write-Host "[bo] done. variant=$Variant  sync rows=$rows  byte-dumps=$dumps  -> $CaptureDir"
