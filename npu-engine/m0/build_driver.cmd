@echo off
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set XRTINC=..\deps\XRT\src\runtime_src\core\include
cl /nologo /EHsc /std:c++17 /Zc:__cplusplus /I "%XRTINC%" decode_driver.cpp out\xrt_coreutil.lib /Fe:out\decode_driver.exe /Fo:out\
if errorlevel 1 (echo [driver] FAILED & exit /b 1)
echo [driver] OK
