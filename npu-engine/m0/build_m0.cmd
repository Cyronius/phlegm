@echo off
REM Build the M0 probe: import lib from the system xrt_coreutil.dll + compile.
setlocal
cd /d "%~dp0"
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS% >nul
set XRTINC=..\deps\XRT\src\runtime_src\core\include
set XRTDLL=C:\Windows\System32\xrt_coreutil.dll

if not exist out mkdir out
REM 1) mangled-export import lib for the real XRT runtime
dumpbin /nologo /exports "%XRTDLL%" > out\xrt_exports.txt
python gen_import_lib.py out\xrt_exports.txt out\xrt_coreutil.def
lib /nologo /def:out\xrt_coreutil.def /out:out\xrt_coreutil.lib /machine:x64
if errorlevel 1 (echo [m0] import-lib FAILED & exit /b 1)

REM 2) compile the probe
cl /nologo /EHsc /std:c++17 /Zc:__cplusplus /I "%XRTINC%" m0_probe.cpp out\xrt_coreutil.lib ^
   /Fe:out\m0_probe.exe /Fo:out\
if errorlevel 1 (echo [m0] compile FAILED & exit /b 1)
echo [m0] OK -^> out\m0_probe.exe
