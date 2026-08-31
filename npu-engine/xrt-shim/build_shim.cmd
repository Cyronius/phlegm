@echo off
REM Standalone build of the extern "C" xrt-shim into a static lib.
REM
REM Normally you do NOT need to run this: `cargo build --features npu` invokes
REM npu-engine/build.rs which performs these same steps into OUT_DIR and links
REM the result. This script exists for a manual/prebuilt-lib workflow and as
REM executable documentation of the compile command.
REM
REM Output: xrt-shim\out\xrt_shim.lib (+ .obj).
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set XRTINC=..\deps\XRT\src\runtime_src\core\include

if not exist out mkdir out
cl /nologo /c /EHsc /std:c++17 /Zc:__cplusplus /O2 /I "%XRTINC%" xrt_shim.cpp /Fo:out\xrt_shim.obj
if errorlevel 1 (echo [shim] compile FAILED & exit /b 1)
lib /nologo /out:out\xrt_shim.lib out\xrt_shim.obj
if errorlevel 1 (echo [shim] lib FAILED & exit /b 1)
echo [shim] OK -^> out\xrt_shim.lib
