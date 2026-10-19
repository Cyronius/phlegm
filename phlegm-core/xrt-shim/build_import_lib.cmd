@echo off
REM Make out\xrt_coreutil.lib from the system xrt_coreutil.dll (2026-10-19); XRT ships no import lib for it.
setlocal
cd /d "%~dp0"
if "%VCVARS64%"=="" set VCVARS64=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
call "%VCVARS64%" >nul
set XRTDLL=C:\Windows\System32\xrt_coreutil.dll

if not exist out mkdir out
dumpbin /nologo /exports "%XRTDLL%" > out\xrt_exports.txt
python gen_import_lib.py out\xrt_exports.txt out\xrt_coreutil.def
lib /nologo /def:out\xrt_coreutil.def /out:out\xrt_coreutil.lib /machine:x64
if errorlevel 1 (echo [xrt-shim] import lib FAILED & exit /b 1)
echo [xrt-shim] OK -^> out\xrt_coreutil.lib
