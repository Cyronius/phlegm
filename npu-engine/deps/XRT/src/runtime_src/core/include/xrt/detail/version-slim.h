// Minimal stand-in for XRT's build-generated version-slim.h (not in git).
// Only xrt/detail/abi.h consumes these; values affect the compile-time ABI tag.
#ifndef xrt_detail_version_slim_h
#define xrt_detail_version_slim_h
#define XRT_VERSION_MAJOR 2
#define XRT_VERSION_MINOR 20
#define XRT_VERSION_CODE  (XRT_VERSION_MAJOR * 1000 + XRT_VERSION_MINOR)
#define XRT_MAJOR(code)   ((code) / 1000)
#define XRT_MINOR(code)   ((code) % 1000)
#endif
