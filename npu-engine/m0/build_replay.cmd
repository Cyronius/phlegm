@echo off
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set XRTINC=..\deps\XRT\src\runtime_src\core\include
if not exist out\xrt_coreutil.lib (echo run build_m0.cmd first & exit /b 1)
cl /nologo /EHsc /std:c++17 /Zc:__cplusplus /I "%XRTINC%" m0_replay.cpp out\xrt_coreutil.lib ^
   /Fe:out\m0_replay.exe /Fo:out\
if errorlevel 1 (echo [replay] compile FAILED & exit /b 1)
echo [replay] OK -^> out\m0_replay.exe
