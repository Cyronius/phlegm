/* xrt_shim.h - thin extern "C" wrapper over XRT's C++ ELF/module/ext::kernel
 * flow, exposing exactly what the Rust decode driver needs.
 *
 * XRT's modern C++ API (xrt::device -> register_xclbin -> hw_context -> elf ->
 * module -> ext::kernel -> run, plus xrt::runlist and xrt::ext::bo) is heavily
 * templated and has no clean C bindings. Rather than reimplement XRT in Rust we
 * wrap the exact calls decode_driver.cpp makes behind opaque void* handles and
 * FFI to that from Rust.
 *
 * Error model: every fallible call is noexcept-wrapped. Constructors return
 * NULL on failure; int-returning calls return <0 on failure; the message is
 * retrievable via xrtsh_last_error() (thread-local). run/runlist wait returns
 * the ert_cmd_state (4 == completed) or <0 on exception.
 */
#ifndef XRT_SHIM_H
#define XRT_SHIM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *xrtsh_dev;
typedef void *xrtsh_ctx;
typedef void *xrtsh_kernel;
typedef void *xrtsh_bo;
typedef void *xrtsh_run;
typedef void *xrtsh_runlist;

/* Last error message on the calling thread (never NULL; "" if none). */
const char *xrtsh_last_error(void);

/* device[index]. NULL on failure. */
xrtsh_dev xrtsh_device_open(int index);
/* Writes device name into out (cap bytes, NUL-terminated). Returns length, or
 * -1 on failure. */
int xrtsh_device_name(xrtsh_dev dev, char *out, int cap);
void xrtsh_device_free(xrtsh_dev dev);

/* xrt::xclbin(path) -> register_xclbin -> hw_context. NULL on failure. */
xrtsh_ctx xrtsh_hwctx_create(xrtsh_dev dev, const char *xclbin_path);
void xrtsh_hwctx_free(xrtsh_ctx ctx);

/* xrt::elf(path) -> module -> ext::kernel(ctx, module, "MLIR_AIE").
 * NULL on failure. */
xrtsh_kernel xrtsh_kernel_create(xrtsh_ctx ctx, const char *elf_path);
/* Classic flow (IRON/mlir-aie designs): xrt::kernel(ctx, name) with the NPU
 * instruction stream passed per run as a cacheable BO at arg 1 and its
 * 32-bit word count at arg 2. NULL on failure. */
xrtsh_kernel xrtsh_kernel_create_xclbin(xrtsh_ctx ctx, const char *kernel_name);
/* Instruction-stream BO for a classic kernel (cacheable, group_id(1)); the
 * caller fills it via xrtsh_bo_write + xrtsh_bo_sync like any other BO. */
xrtsh_bo xrtsh_bo_create_instr(xrtsh_dev dev, xrtsh_kernel k, size_t size);
void xrtsh_kernel_free(xrtsh_kernel k);

/* xrt::ext::bo(dev, size). Caller passes the already-padded size. NULL on
 * failure. */
xrtsh_bo xrtsh_bo_create(xrtsh_dev dev, size_t size);
/* Host-mapped pointer (bo.map<uint8_t*>()). NULL on failure. */
void *xrtsh_bo_map(xrtsh_bo bo);
/* dir: 1 = HOST->DEVICE, 0 = DEVICE->HOST. Returns 0 ok, <0 on failure. */
int xrtsh_bo_sync(xrtsh_bo bo, int to_device);
/* memcpy helpers over the mapped pointer (no sync). Return 0 / <0. */
int xrtsh_bo_write(xrtsh_bo bo, const void *src, size_t n, size_t off);
int xrtsh_bo_read(xrtsh_bo bo, void *dst, size_t n, size_t off);
void xrtsh_bo_free(xrtsh_bo bo);

/* xrt::run(kernel). NULL on failure. */
xrtsh_run xrtsh_run_create(xrtsh_kernel k);
int xrtsh_run_set_arg_int(xrtsh_run r, int idx, int val);
int xrtsh_run_set_arg_bo(xrtsh_run r, int idx, xrtsh_bo bo);
/* The run's XRT-allocated control scratchpad BO (ELF-flow kernels whose
 * instruction stream contains create_scratchpad). NULL if absent/failure.
 * The returned handle is a new ShimBo sharing the underlying xrt::bo. */
xrtsh_bo xrtsh_run_scratchpad_bo(xrtsh_run r);
/* Device address of a BO (xrt::bo::address()). 0 on failure. */
unsigned long long xrtsh_bo_address(xrtsh_bo bo);
int xrtsh_run_start(xrtsh_run r);
/* Returns ert_cmd_state (4 == COMPLETED) or <0 on exception. */
int xrtsh_run_wait(xrtsh_run r);
void xrtsh_run_free(xrtsh_run r);

/* xrt::runlist(ctx). NULL on failure. Runs added must outlive the wait. */
xrtsh_runlist xrtsh_runlist_create(xrtsh_ctx ctx);
int xrtsh_runlist_add(xrtsh_runlist rl, xrtsh_run r);
int xrtsh_runlist_execute(xrtsh_runlist rl);
/* Returns 0 on success, <0 on exception (timeout / device error). */
int xrtsh_runlist_wait(xrtsh_runlist rl);
void xrtsh_runlist_free(xrtsh_runlist rl);

#ifdef __cplusplus
}
#endif

#endif /* XRT_SHIM_H */
