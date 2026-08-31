# Builds xrt_coreutil.dll -- a capture proxy for FastFlowLM's closed NPU engines.
# Output MUST be named exactly "xrt_coreutil.dll" and dropped next to flm.exe,
# with the real DLL renamed to "xrt_coreutil_orig.dll" (see README.md).
#
# Requires Microsoft Detours. Easiest: vcpkg install detours:x64-windows and
# configure with -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake_minimum_required(VERSION 3.20)
project(xrt_seq_capture CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(detours CONFIG REQUIRED)

add_library(xrt_coreutil SHARED
    xrt_shim.cpp
    xrt_coreutil.def)          # .def forwards all 541 real exports

set_target_properties(xrt_coreutil PROPERTIES
    OUTPUT_NAME "xrt_coreutil"
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out")

target_link_libraries(xrt_coreutil PRIVATE ${DETOURS_LIBRARY})
# vcpkg's detours target is usually "Microsoft::Detours"; fall back if needed.
if(TARGET Microsoft::Detours)
    target_link_libraries(xrt_coreutil PRIVATE Microsoft::Detours)
endif()

# The .def path is relative to the source dir.
target_sources(xrt_coreutil PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/xrt_coreutil.def)
