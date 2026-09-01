# Drives one FLM prefill capture run on the BASE model (no variant swap).
# Arms BOTH shim planes: seq (.seq ctrlcode blobs) and bo/event (events.tsv,
# elf_*.bin, NNNNNN.bo dumps up to -DumpMax bytes/sync; 0 = metadata only).
# Server stdout/stderr go to <CaptureDir>\server_out.log / server_err.log so
# FLM's own prefill/decode timings are kept.
# Usage:
#   pwsh -File prefill_capture.ps1 -CaptureDir C:/caps/pf_t11 -Prompt "Say hi." -DumpMax 4194304
param(
  [Parameter(Mandatory=$true)][string]$CaptureDir,
  [Parameter(Mandatory=$true)][string]$Prompt,
  [long]$DumpMax = 0,
  [ValidateSet("seq","both")][string]$Planes = "both",
  [int]$Port = 52625,
  [int]$MaxTokens = 8,
  [int]$RequestTimeoutSec = 7200,
  [string]$Tag = "qwen3.6-moe:35b-a3b",
  [string]$FlmDir = "C:\flm-test",
  [string]$ModelRoot = "C:\Users\josha\.flm"
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $CaptureDir | Out-Null
Get-ChildItem -Path $CaptureDir -File -ErrorAction SilentlyContinue | Remove-Item -Force

$env:FLM_MODEL_PATH     = $ModelRoot          # the .flm ROOT (FLM appends \models)
$env:FLM_SEQ_CAPTURE_DIR = $CaptureDir
if ($Planes -eq "both") {
  $env:FLM_BO_CAPTURE_DIR = $CaptureDir
  $env:FLM_BO_DUMP_MAX    = "$DumpMax"
  $env:FLM_BO_RUNARG_MAX  = "8388608"   # skip hashing >8MB run args (512MB pools stall prefill)
} else {
  Remove-Item Env:\FLM_BO_CAPTURE_DIR -ErrorAction SilentlyContinue
  Remove-Item Env:\FLM_BO_DUMP_MAX   -ErrorAction SilentlyContinue
}

$outLog = Join-Path $CaptureDir "server_out.log"
$errLog = Join-Path $CaptureDir "server_err.log"
Write-Host "[pf] starting server (dumpMax=$DumpMax cap=$CaptureDir)"
$p = Start-Process -FilePath (Join-Path $FlmDir "flm.exe") `
     -ArgumentList @("serve", $Tag, "--port", "$Port") `
     -WorkingDirectory $FlmDir -PassThru -WindowStyle Hidden `
     -RedirectStandardOutput $outLog -RedirectStandardError $errLog

$base = "http://127.0.0.1:$Port"
$ready = $false
for ($i = 0; $i -lt 300; $i++) {
  if ($p.HasExited) { Write-Host "[pf] server exited early (code $($p.ExitCode))"; break }
  try { Invoke-RestMethod -Uri "$base/v1/models" -TimeoutSec 3 | Out-Null; $ready = $true; break }
  catch { Start-Sleep -Seconds 2 }
}
if (-not $ready) {
  Write-Host "[pf] server never became ready; killing"
  if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
  exit 1
}
Write-Host "[pf] server ready after ~$($i*2)s; sending prompt: $Prompt"

$body = @{ model=$Tag; messages=@(@{role="user"; content=$Prompt}); max_tokens=$MaxTokens; stream=$false } | ConvertTo-Json -Depth 5
$t0 = Get-Date
try {
  $r = Invoke-RestMethod -Uri "$base/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body -TimeoutSec $RequestTimeoutSec
  $dt = ((Get-Date) - $t0).TotalSeconds
  Write-Host "[pf] request took $([math]::Round($dt,2))s; response: $($r.choices[0].message.content)"
  if ($r.usage) { Write-Host "[pf] usage: prompt=$($r.usage.prompt_tokens) completion=$($r.usage.completion_tokens)" }
  $r | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $CaptureDir "response.json")
} catch { Write-Host "[pf] chat request failed: $($_.Exception.Message)" }

Start-Sleep -Seconds 1
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
$rows = 0
$tf = Join-Path $CaptureDir "bo_trace.tsv"
if (Test-Path $tf) { $rows = (Get-Content $tf).Count }
$dumps = (Get-ChildItem -Path $CaptureDir -Filter *.bo -ErrorAction SilentlyContinue).Count
$seqs  = (Get-ChildItem -Path $CaptureDir -Filter *.seq -ErrorAction SilentlyContinue).Count
$elfs  = (Get-ChildItem -Path $CaptureDir -Filter elf_*.bin -ErrorAction SilentlyContinue).Count
Write-Host "[pf] done. sync rows=$rows  byte-dumps=$dumps  seqs=$seqs  elfs=$elfs  -> $CaptureDir"
