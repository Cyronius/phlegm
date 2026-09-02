// xrt_shim.cpp - extern "C" wrapper over XRT's C++ ELF/module/ext::kernel flow.
// Mirrors exactly the XRT calls decode_driver.cpp makes; see xrt_shim.h.
#include "xrt_shim.h"

#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"
#include "xrt/experimental/xrt_elf.h"
#include "xrt/experimental/xrt_module.h"
#include "xrt/experimental/xrt_ext.h"

#include <cstring>
#include <stdexcept>
#include <memory>
#include <string>

// Thread-local last-error string.
static thread_local std::string g_err;
static void set_err(const char *what) { g_err = what ? what : "unknown"; }
static void clear_err() { g_err.clear(); }

extern "C" const char *xrtsh_last_error(void) { return g_err.c_str(); }

// Concrete handle types behind the opaque void* pointers.
struct ShimDevice {
    xrt::device dev;
};
struct ShimCtx {
    xrt::hw_context ctx;
};
struct ShimKernel {
    // ELF flow (FLM's captured kernels): elf -> module -> ext::kernel.
    std::shared_ptr<xrt::elf> elf;
    std::shared_ptr<xrt::module> mod;
    std::shared_ptr<xrt::ext::kernel> kern;
    // Classic flow (mlir-aie xclbin + insts.bin): plain xrt::kernel.
    std::shared_ptr<xrt::kernel> classic;
};
struct ShimBo {
    // xrt::ext::bo is-a xrt::bo; holding the base lets instruction BOs
    // (plain xrt::bo with a kernel group id) share the same handle type.
    xrt::bo bo;
};
struct ShimRun {
    xrt::run run;
};
struct ShimRunlist {
    xrt::runlist rl;
};

#define GUARD_PTR(BODY)                                                        \
    try {                                                                      \
        clear_err();                                                           \
        BODY                                                                   \
    } catch (const std::exception &e) {                                        \
        set_err(e.what());                                                     \
        return nullptr;                                                        \
    } catch (...) {                                                            \
        set_err("unknown C++ exception");                                      \
        return nullptr;                                                        \
    }

#define GUARD_INT(BODY)                                                        \
    try {                                                                      \
        clear_err();                                                           \
        BODY                                                                   \
    } catch (const std::exception &e) {                                        \
        set_err(e.what());                                                     \
        return -1;                                                             \
    } catch (...) {                                                            \
        set_err("unknown C++ exception");                                      \
        return -1;                                                             \
    }

extern "C" {

xrtsh_dev xrtsh_device_open(int index) {
    GUARD_PTR({
        auto *d = new ShimDevice{xrt::device(static_cast<unsigned int>(index))};
        return static_cast<xrtsh_dev>(d);
    })
}

int xrtsh_device_name(xrtsh_dev dev, char *out, int cap) {
    GUARD_INT({
        auto *d = static_cast<ShimDevice *>(dev);
        std::string name = d->dev.get_info<xrt::info::device::name>();
        int n = (int)name.size();
        if (out && cap > 0) {
            int c = n < cap - 1 ? n : cap - 1;
            std::memcpy(out, name.data(), c);
            out[c] = '\0';
        }
        return n;
    })
}

void xrtsh_device_free(xrtsh_dev dev) { delete static_cast<ShimDevice *>(dev); }

xrtsh_ctx xrtsh_hwctx_create(xrtsh_dev dev, const char *xclbin_path) {
    GUARD_PTR({
        auto *d = static_cast<ShimDevice *>(dev);
        xrt::xclbin xcl{std::string(xclbin_path)};
        auto uuid = d->dev.register_xclbin(xcl);
        auto *c = new ShimCtx{xrt::hw_context(d->dev, uuid)};
        return static_cast<xrtsh_ctx>(c);
    })
}

void xrtsh_hwctx_free(xrtsh_ctx ctx) { delete static_cast<ShimCtx *>(ctx); }

xrtsh_kernel xrtsh_kernel_create(xrtsh_ctx ctx, const char *elf_path) {
    GUARD_PTR({
        auto *c = static_cast<ShimCtx *>(ctx);
        auto *k = new ShimKernel();
        k->elf = std::make_shared<xrt::elf>(std::string(elf_path));
        k->mod = std::make_shared<xrt::module>(*k->elf);
        k->kern = std::make_shared<xrt::ext::kernel>(c->ctx, *k->mod, "MLIR_AIE");
        return static_cast<xrtsh_kernel>(k);
    })
}

xrtsh_kernel xrtsh_kernel_create_xclbin(xrtsh_ctx ctx, const char *kernel_name) {
    GUARD_PTR({
        auto *c = static_cast<ShimCtx *>(ctx);
        auto *k = new ShimKernel();
        k->classic = std::make_shared<xrt::kernel>(c->ctx, std::string(kernel_name));
        return static_cast<xrtsh_kernel>(k);
    })
}

xrtsh_bo xrtsh_bo_create_instr(xrtsh_dev dev, xrtsh_kernel k, size_t size) {
    GUARD_PTR({
        auto *d = static_cast<ShimDevice *>(dev);
        auto *kern = static_cast<ShimKernel *>(k);
        if (!kern->classic) throw std::runtime_error("instr bo needs a classic (xclbin) kernel");
        auto *b = new ShimBo{xrt::bo(d->dev, size, xrt::bo::flags::cacheable, kern->classic->group_id(1))};
        return static_cast<xrtsh_bo>(b);
    })
}

void xrtsh_kernel_free(xrtsh_kernel k) { delete static_cast<ShimKernel *>(k); }

xrtsh_bo xrtsh_bo_create(xrtsh_dev dev, size_t size) {
    GUARD_PTR({
        auto *d = static_cast<ShimDevice *>(dev);
        auto *b = new ShimBo{xrt::ext::bo(d->dev, size)};
        return static_cast<xrtsh_bo>(b);
    })
}

void *xrtsh_bo_map(xrtsh_bo bo) {
    GUARD_PTR({
        auto *b = static_cast<ShimBo *>(bo);
        return static_cast<void *>(b->bo.map<uint8_t *>());
    })
}

int xrtsh_bo_sync(xrtsh_bo bo, int to_device) {
    GUARD_INT({
        auto *b = static_cast<ShimBo *>(bo);
        b->bo.sync(to_device ? XCL_BO_SYNC_BO_TO_DEVICE
                             : XCL_BO_SYNC_BO_FROM_DEVICE);
        return 0;
    })
}

int xrtsh_bo_write(xrtsh_bo bo, const void *src, size_t n, size_t off) {
    GUARD_INT({
        auto *b = static_cast<ShimBo *>(bo);
        auto *h = b->bo.map<uint8_t *>();
        std::memcpy(h + off, src, n);
        return 0;
    })
}

int xrtsh_bo_read(xrtsh_bo bo, void *dst, size_t n, size_t off) {
    GUARD_INT({
        auto *b = static_cast<ShimBo *>(bo);
        auto *h = b->bo.map<uint8_t *>();
        std::memcpy(dst, h + off, n);
        return 0;
    })
}

void xrtsh_bo_free(xrtsh_bo bo) { delete static_cast<ShimBo *>(bo); }

xrtsh_run xrtsh_run_create(xrtsh_kernel k) {
    GUARD_PTR({
        auto *kern = static_cast<ShimKernel *>(k);
        auto *r = kern->classic ? new ShimRun{xrt::run(*kern->classic)}
                                : new ShimRun{xrt::run(*kern->kern)};
        return static_cast<xrtsh_run>(r);
    })
}

int xrtsh_run_set_arg_int(xrtsh_run r, int idx, int val) {
    GUARD_INT({
        static_cast<ShimRun *>(r)->run.set_arg(idx, val);
        return 0;
    })
}

int xrtsh_run_set_arg_bo(xrtsh_run r, int idx, xrtsh_bo bo) {
    GUARD_INT({
        static_cast<ShimRun *>(r)->run.set_arg(idx, static_cast<ShimBo *>(bo)->bo);
        return 0;
    })
}

int xrtsh_run_start(xrtsh_run r) {
    GUARD_INT({
        static_cast<ShimRun *>(r)->run.start();
        return 0;
    })
}

int xrtsh_run_wait(xrtsh_run r) {
    GUARD_INT({
        ert_cmd_state st = static_cast<ShimRun *>(r)->run.wait();
        return static_cast<int>(st);
    })
}

void xrtsh_run_free(xrtsh_run r) { delete static_cast<ShimRun *>(r); }

xrtsh_runlist xrtsh_runlist_create(xrtsh_ctx ctx) {
    GUARD_PTR({
        auto *c = static_cast<ShimCtx *>(ctx);
        auto *rl = new ShimRunlist{xrt::runlist(c->ctx)};
        return static_cast<xrtsh_runlist>(rl);
    })
}

int xrtsh_runlist_add(xrtsh_runlist rl, xrtsh_run r) {
    GUARD_INT({
        static_cast<ShimRunlist *>(rl)->rl.add(static_cast<ShimRun *>(r)->run);
        return 0;
    })
}

int xrtsh_runlist_execute(xrtsh_runlist rl) {
    GUARD_INT({
        static_cast<ShimRunlist *>(rl)->rl.execute();
        return 0;
    })
}

int xrtsh_runlist_wait(xrtsh_runlist rl) {
    GUARD_INT({
        static_cast<ShimRunlist *>(rl)->rl.wait();
        return 0;
    })
}

void xrtsh_runlist_free(xrtsh_runlist rl) { delete static_cast<ShimRunlist *>(rl); }

} // extern "C"
