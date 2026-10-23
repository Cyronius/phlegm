param(
    [Parameter(Mandatory = $true)][string]$ReferenceRoot,
    [string]$OutDir,
    [string]$XrtInclude,
    [string]$XrtLib,
    [string]$Vcvars = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"
)
$ErrorActionPreference = 'Stop'
$adapterRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $adapterRoot '../..'))
if (-not $OutDir) { $OutDir = Join-Path $workspaceRoot 'target/reference-bench' }
if (-not $XrtInclude) { $XrtInclude = Join-Path $workspaceRoot 'phlegm-core/deps/XRT/src/runtime_src/core/include' }
if (-not $XrtLib) { $XrtLib = Join-Path $workspaceRoot 'phlegm-core/xrt-shim/out/xrt_coreutil.lib' }
function FullPath([string]$Value) {
    $result = [IO.Path]::GetFullPath($Value)
    # These paths enter a generated batch file. Reject shell expansion syntax.
    if ($result -match '["&|<>^%!\r\n]') { throw "Unsupported shell character in path: $result" }
    return $result
}
$ReferenceRoot = FullPath $ReferenceRoot
$OutDir = FullPath $OutDir
$XrtInclude = FullPath $XrtInclude
$XrtLib = FullPath $XrtLib
$Vcvars = FullPath $Vcvars
$adapter = FullPath (Join-Path $adapterRoot 'bridge.cpp')
$engine = Join-Path $ReferenceRoot 'src/open_qwen36'
foreach ($path in @($Vcvars, $XrtLib, "$XrtInclude/xrt/xrt_device.h", "$engine/core.cpp", "$engine/gguf_file.cpp")) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing build input: $path" }
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$binary = Join-Path $OutDir 'phlegm-reference-bench.exe'
$metadata = [IO.Path]::ChangeExtension($binary, 'build.json')
# Invalidate old provenance before compiling, so a failed rebuild cannot leave a
# newly named executable paired with stale metadata.
if (Test-Path -LiteralPath $metadata) { Remove-Item -LiteralPath $metadata }

function SourceFiles {
    $files = @{}
    $files['bridge/bridge.cpp'] = (Get-FileHash -LiteralPath $adapter -Algorithm SHA256).Hash.ToLowerInvariant()
    foreach ($file in Get-ChildItem -LiteralPath $engine -File -Recurse | Where-Object { $_.Extension -in @('.cpp', '.hpp') }) {
        $relative = $file.FullName.Substring($engine.Length + 1).Replace('\', '/')
        $files["engine/$relative"] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    foreach ($base in @(
        @{ Path = "$ReferenceRoot/src/include/nlohmann"; Prefix = 'json' },
        @{ Path = $XrtInclude; Prefix = 'xrt' }
    )) {
        foreach ($file in Get-ChildItem -LiteralPath $base.Path -File -Recurse | Where-Object { $_.Extension -in @('.h', '.hpp') }) {
            $relative = $file.FullName.Substring($base.Path.Length + 1).Replace('\', '/')
            $files["$($base.Prefix)/$relative"] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $files['harness/stream_patch.hpp'] = (Get-FileHash -LiteralPath "$ReferenceRoot/open_kernels/harness/stream_patch.hpp" -Algorithm SHA256).Hash.ToLowerInvariant()
    $files['link/xrt_coreutil.lib'] = (Get-FileHash -LiteralPath $XrtLib -Algorithm SHA256).Hash.ToLowerInvariant()
    return $files
}
function ManifestHash($Files) {
    $names = [string[]]@($Files.Keys)
    [Array]::Sort($names, [StringComparer]::Ordinal)
    $text = [Text.StringBuilder]::new()
    foreach ($name in $names) { [void]$text.Append($name).Append([char]0).Append($Files[$name]).Append("`n") }
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($text.ToString())))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}
$before = SourceFiles
$sourceHash = ManifestHash $before
$commit = & git -c "safe.directory=$($ReferenceRoot.Replace('\', '/'))" -C $ReferenceRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $commit -notmatch '^[0-9a-f]{40}$') { throw 'Cannot identify reference engine commit' }
$status = & git -c "safe.directory=$($ReferenceRoot.Replace('\', '/'))" -C $ReferenceRoot status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0) { throw 'Cannot read reference engine Git status' }
$dirty = -not [string]::IsNullOrWhiteSpace(($status -join "`n"))
$sources = @('core.cpp', 'block_host.cpp', 'pools.cpp', 'manifest.cpp', 'q4nx_file.cpp', 'gguf_file.cpp') |
    ForEach-Object { '"' + (Join-Path $engine $_) + '"' }
$compile = @"
@echo off
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "$Vcvars" >nul
if errorlevel 1 exit /b 1
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS /DDISABLE_ABI_CHECK=1 /bigobj /openmp /arch:AVX2 /I "$XrtInclude" /I "$ReferenceRoot/src" /I "$ReferenceRoot/src/include" /I "$ReferenceRoot/open_kernels/harness" $($sources -join ' ') "$adapter" "$XrtLib" /Fe:"$binary" /Fo:.\ /link /DELAYLOAD:VCOMP140.DLL delayimp.lib
exit /b %errorlevel%
"@
$batch = Join-Path $OutDir 'build.cmd'
[IO.File]::WriteAllText($batch, $compile, [Text.Encoding]::ASCII)
Push-Location $OutDir
try {
    & cmd.exe /d /c $batch
    if ($LASTEXITCODE -ne 0) { throw "Reference bridge compile failed ($LASTEXITCODE)" }
} finally { Pop-Location }
if ((ManifestHash (SourceFiles)) -ne $sourceHash) { throw 'Build inputs changed during compilation; rebuild required' }
$build = [ordered]@{
    protocol = 1
    engine = 'openflowlm-cpp-reference'
    source_commit = $commit.Trim()
    source_dirty = $dirty
    source_tree_sha256 = $sourceHash
    adapter_sha256 = $before['bridge/bridge.cpp']
    binary_sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    compiler_flags = 'MSVC /O2 /MD /std:c++17 /openmp /arch:AVX2 /DELAYLOAD:VCOMP140.DLL'
    source_files = $before
}
[IO.File]::WriteAllText($metadata, ($build | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
Write-Output "Built $binary"
Write-Output "Provenance $metadata"
