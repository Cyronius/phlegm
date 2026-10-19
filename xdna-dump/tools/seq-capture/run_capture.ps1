# Drives one FLM capture run: start server (capture armed) -> one short prompt -> stop.
# Usage: pwsh -File run_capture.ps1 -CaptureDir C:\caps\i4 [-Port 52625] [-MaxTokens 8]
param(
  [Parameter(Mandatory=$true)][string]$CaptureDir,
  [int]$Port = 52625,
  [int]$MaxTokens = 8,
  [string]$Tag = "qwen3.6-moe:35b-a3b",
  [string]$FlmDir = "C:\flm-test",
  [string]$ModelPath = "C:\Users\josha\.flm\models"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $CaptureDir | Out-Null
Get-ChildItem -Path $CaptureDir -File -ErrorAction SilentlyContinue | Remove-Item -Force

$env:FLM_MODEL_PATH      = $ModelPath
$env:FLM_SEQ_CAPTURE_DIR = $CaptureDir

Write-Host "[cap] starting server (tag=$Tag port=$Port capture=$CaptureDir)"
$p = Start-Process -FilePath (Join-Path $FlmDir "flm.exe") `
     -ArgumentList @("serve", $Tag, "--port", "$Port", "--quiet") `
     -WorkingDirectory $FlmDir -PassThru -WindowStyle Hidden

$base = "http://127.0.0.1:$Port"
$ready = $false
for ($i = 0; $i -lt 180; $i++) {
  if ($p.HasExited) { Write-Host "[cap] server exited early (code $($p.ExitCode))"; break }
  try {
    Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 3 | Out-Null
    $ready = $true; break
  } catch { Start-Sleep -Seconds 2 }
}
if (-not $ready) {
  Write-Host "[cap] server never became ready; killing"
  if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
  exit 1
}
Write-Host "[cap] server ready after ~$($i*2)s; sending prompt"

$body = @{
  model    = $Tag
  messages = @(@{ role = "user"; content = "Say hi." })
  max_tokens = $MaxTokens
  stream   = $false
} | ConvertTo-Json -Depth 5

try {
  $r = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post `
        -ContentType "application/json" -Body $body -TimeoutSec 300
  $txt = $r.choices[0].message.content
  Write-Host "[cap] response: $txt"
} catch {
  Write-Host "[cap] chat request failed: $($_.Exception.Message)"
}

Start-Sleep -Seconds 1
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
$n = (Get-ChildItem -Path $CaptureDir -Filter *.seq -ErrorAction SilentlyContinue).Count
Write-Host "[cap] done. captured $n sequences in $CaptureDir"
