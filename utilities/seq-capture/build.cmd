@echo off
REM Build the xrt_coreutil capture proxy. No third-party deps -- MSVC + Win SDK only.
REM Output: out\xrt_coreutil.dll  (rename the real DLL to xrt_coreutil_orig.dll to use it)
REM
REM Two-step link:
REM   1. capi.lib -- an import library for the 62 undecorated C-API names, bound to
REM      xrt_coreutil_orig.dll. (Undecorated names can't be .def-forwarded; they are
REM      re-exported through this import lib instead.)
REM   2. the proxy DLL -- 478 mangled C++ exports forward straight to
REM      xrt_coreutil_orig.dll; xrt::elf::elf is aliased to our capture thunk.
setlocal
cd /d "%~dp0"
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo [build] vcvars64.bat not found -- edit VCVARS in this script to your VS path.
  exit /b 1
)
call %VCVARS% >nul
if not exist out mkdir out

lib /nologo /def:xrt_coreutil_orig.capi.def /name:xrt_coreutil_orig.dll ^
    /out:out\capi.lib /machine:x64
if errorlevel 1 (echo [build] import-lib step FAILED & exit /b 1)

cl /nologo /LD /O2 /EHsc /std:c++17 xrt_shim.cpp out\capi.lib ^
   /Fe:out\xrt_coreutil.dll /Fo:out\ ^
   /link /DEF:xrt_coreutil.def /OUT:out\xrt_coreutil.dll
if errorlevel 1 (echo [build] link step FAILED & exit /b 1)
echo [build] OK -^> out\xrt_coreutil.dll
