// xrt_coreutil.dll capture proxy for FastFlowLM's closed NPU engines.
//
// Why this exists: the model engines (e.g. qwen3_6_moe_npu.dll) ship as prebuilt
// DLLs; their per-layer scheduling is baked in and cannot be observed by editing
// the open headers. But every engine reaches the NPU through one dynamic import:
// xrt_coreutil.dll (confirmed from the engine's PE import table). This DLL is
// built named "xrt_coreutil.dll" and placed next to flm.exe.
//
//   * xrt_coreutil.def forwards ~540 exports to the RENAMED real DLL
//     (xrt_coreutil_orig.dll).
//   * The exports that carry the engine's work are aliased in the .def to the
//     extern "C" thunks below, which observe then forward to the real DLL.
//
// TWO capture planes, independently armed:
//
//   1. Op sequence (Tier-0). Hook xrt::elf::elf(const void*, size_t) -- mangled
//      ??0elf@xrt@@QEAA@PEBX_K@Z. One control-code blob is built per op, in
//      submission order, so the dumped corpus is the engine's per-layer op
//      sequence. Armed by FLM_SEQ_CAPTURE_DIR. (Result of the first study:
//      interval-3 and interval-4 emit the SAME op sequence -- the defect is not
//      here, it's in the tensor DATA the CPU computes.)
//
//   2. Tensor data path (this file's addition). The engine imports exactly three
//      xrt::bo entry points that move data across the CPU<->NPU boundary:
//        * ?map@bo@xrt@@QEAAPEAXXZ            void* bo::map()
//        * ?sync@bo@xrt@@QEAAXW4...@_K1@Z     bo::sync(dir, size, offset)
//        * ??1bo@xrt@@QEAA@XZ                 bo::~bo()
//      There is no bo::write import -- so the engine maps a buffer once, writes
//      tensor bytes into the mapped host pointer, then sync()s a sub-range to the
//      device. map() and sync() are member calls, so the bo object's `this`
//      pointer (rcx on x64) links "which host address" to "which bytes synced".
//      We record this->host_ptr at map(), and at each sync() dump/hash the exact
//      bytes crossing the boundary. ~bo() evicts the entry so a recycled `this`
//      address can't alias a dead buffer. Armed by FLM_BO_CAPTURE_DIR.
//
// No Detours / no third-party deps: a plain proxy DLL. Renaming the real DLL to
// xrt_coreutil_orig.dll is what keeps the thunks from re-entering themselves (the
// real code lives in a different module).
//
// Both planes are silent pass-throughs when their env var is unset. Build: see
// build.cmd / README.md.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

namespace {

// ---- shared init -----------------------------------------------------------

std::atomic<bool> g_init{false};
std::mutex        g_init_mtx;
HMODULE           g_real_mod = nullptr;

// ---- op-sequence plane (elf ctor) ------------------------------------------

// xrt::elf::elf(void* this, const void* buf, uint64_t n) -- this=rcx, buf=rdx,
// n=r8. A ctor returns `this` in rax; we mirror that as a void* return.
using elf_ctor_t = void* (*)(void*, const void*, uint64_t);
std::atomic<elf_ctor_t> g_real_elf{nullptr};
std::string             g_seq_dir;    // FLM_SEQ_CAPTURE_DIR ("" = disabled)
std::atomic<uint32_t>   g_seq_index{0};
std::mutex              g_seq_trace_mtx;

// ---- tensor-data plane (bo map/sync/dtor) ----------------------------------

using bo_map_t  = void* (*)(void*);                        // bo::map()
using bo_sync_t = void  (*)(void*, int, uint64_t, uint64_t); // bo::sync(dir,sz,off)
using bo_dtor_t = void  (*)(void*);                        // bo::~bo()
std::atomic<bo_map_t>  g_real_bo_map{nullptr};
std::atomic<bo_sync_t> g_real_bo_sync{nullptr};
std::atomic<bo_dtor_t> g_real_bo_dtor{nullptr};

std::string           g_bo_dir;                 // FLM_BO_CAPTURE_DIR ("" = off)
uint64_t              g_bo_dump_max = 0;         // FLM_BO_DUMP_MAX bytes/sync
uint64_t              g_bo_runarg_max = ~0ULL;   // FLM_BO_RUNARG_MAX: hash/dump run args only up to this size (hashing 512MB pools per submit stalls prefill ~100x)
std::atomic<uint32_t> g_bo_index{0};
std::mutex            g_bo_trace_mtx;
std::mutex            g_bo_map_mtx;
std::unordered_map<void*, void*> g_bo_hostptr;  // bo `this` -> mapped host ptr

// ---- run-correlation plane (kernel run args + submit) ----------------------
// Links each kernel run to its ELF (temporal order) and its data buffers
// (arg index -> bo `this`), so one op reconstructs as {elf, set_args, start,
// syncs}. Armed by FLM_BO_CAPTURE_DIR (same as the bo plane).
using run_setarg_bo_t = void (*)(void*, int, const void*);  // run::set_arg_at_index(idx, const bo&)
using run_start_t     = void (*)(void*);                    // run::start()
std::atomic<run_setarg_bo_t> g_real_run_setarg_bo{nullptr};
std::atomic<run_start_t>     g_real_run_start{nullptr};

// object-graph ctors: run -> kernel -> module -> elf (each ctor returns `this`).
using module_ctor_t = void* (*)(void*, const void*);                    // module(const elf&)
using extkern_ctor_t = void* (*)(void*, const void*, const void*, const void*); // ext::kernel(hwctx, module, name)
using run_ctor_t    = void* (*)(void*, const void*);                    // run(const kernel&)
using bo_size_t     = uint64_t (*)(const void*);                        // bo::size() const
std::atomic<module_ctor_t>  g_real_module_ctor{nullptr};
std::atomic<extkern_ctor_t> g_real_extkern_ctor{nullptr};
std::atomic<run_ctor_t>     g_real_run_ctor{nullptr};
std::atomic<bo_size_t>      g_real_bo_size{nullptr};
// run `this` -> ordered (argindex, bo ptr) bound so far, so run::start can dump
// the exact buffers this run executes on.
std::mutex g_run_args_mtx;
std::unordered_map<void*, std::vector<std::pair<int, void*>>> g_run_args;
// content-addressed dedup: a buffer's bytes are written once as blob_<size>_<hash>.bin
// (the ~512 MB weight pool is bound to every op -- dedup keeps it to one file).
std::mutex g_blob_mtx;
std::unordered_map<uint64_t, uint8_t> g_blob_done;  // (size^hash) -> 1

// One monotonic counter across elf / bo-sync / set_arg / start -> a single
// ordered events.tsv the analyzer replays.
std::atomic<uint64_t> g_event{0};
std::mutex            g_event_mtx;
std::atomic<uint32_t> g_elfcap_index{0};

// XCL_BO_SYNC_BO_TO_DEVICE = 0 (host->NPU), _FROM_DEVICE = 1 (NPU->host).
constexpr int SYNC_TO_DEVICE = 0;

uint64_t fnv1a(const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

void event_log(const char* line) {
    if (g_bo_dir.empty()) return;
    std::lock_guard<std::mutex> lk(g_event_mtx);
    FILE* f = nullptr;
    if (fopen_s(&f, (g_bo_dir + "\\events.tsv").c_str(), "ab") == 0 && f) {
        fputs(line, f); fclose(f);
    }
}

// Dump each ELF blob (the control code) into the capture dir + log an ELF event,
// so the replayer has the exact bytes FLM fed to xrt::elf for this op.
void elf_capture(void* self, const void* buf, uint64_t n) {
    if (g_bo_dir.empty() || !buf || !n) return;
    uint32_t idx = g_elfcap_index.fetch_add(1);
    char path[MAX_PATH];
    _snprintf_s(path, sizeof(path), _TRUNCATE, "%s\\elf_%06u.bin", g_bo_dir.c_str(), idx);
    FILE* f = nullptr;
    if (fopen_s(&f, path, "wb") == 0 && f) { fwrite(buf, 1, (size_t)n, f); fclose(f); }
    char line[192];
    _snprintf_s(line, sizeof(line), _TRUNCATE, "%llu\tELF\t%06u\t%llu\t%016llx\t%p\n",
                (unsigned long long)g_event.fetch_add(1), idx,
                (unsigned long long)n, (unsigned long long)fnv1a(buf, (size_t)n), self);
    event_log(line);
}

void init() {
    std::lock_guard<std::mutex> lk(g_init_mtx);
    if (g_init.load()) return;

    g_real_mod = LoadLibraryA("xrt_coreutil_orig.dll");
    if (!g_real_mod) {
        OutputDebugStringA("[xrt-shim] cannot load xrt_coreutil_orig.dll");
        g_init.store(true);
        return;
    }
    auto sym = [&](const char* n) { return GetProcAddress(g_real_mod, n); };

    g_real_elf.store(reinterpret_cast<elf_ctor_t>(sym("??0elf@xrt@@QEAA@PEBX_K@Z")));
    g_real_bo_map.store(reinterpret_cast<bo_map_t>(sym("?map@bo@xrt@@QEAAPEAXXZ")));
    g_real_bo_sync.store(reinterpret_cast<bo_sync_t>(
        sym("?sync@bo@xrt@@QEAAXW4xclBOSyncDirection@@_K1@Z")));
    g_real_bo_dtor.store(reinterpret_cast<bo_dtor_t>(sym("??1bo@xrt@@QEAA@XZ")));
    g_real_run_setarg_bo.store(reinterpret_cast<run_setarg_bo_t>(
        sym("?set_arg_at_index@run@xrt@@AEAAXHAEBVbo@2@@Z")));
    g_real_run_start.store(reinterpret_cast<run_start_t>(sym("?start@run@xrt@@QEAAXXZ")));
    g_real_module_ctor.store(reinterpret_cast<module_ctor_t>(sym("??0module@xrt@@QEAA@AEBVelf@1@@Z")));
    g_real_extkern_ctor.store(reinterpret_cast<extkern_ctor_t>(
        sym("??0kernel@ext@xrt@@QEAA@AEBVhw_context@2@AEBVmodule@2@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z")));
    g_real_run_ctor.store(reinterpret_cast<run_ctor_t>(sym("??0run@xrt@@QEAA@AEBVkernel@1@@Z")));
    g_real_bo_size.store(reinterpret_cast<bo_size_t>(sym("?size@bo@xrt@@QEBA_KXZ")));
    if (!g_real_elf.load() || !g_real_bo_map.load() ||
        !g_real_bo_sync.load() || !g_real_bo_dtor.load() ||
        !g_real_run_setarg_bo.load() || !g_real_run_start.load())
        OutputDebugStringA("[xrt-shim] a real xrt export failed to resolve");

    if (const char* d = getenv("FLM_SEQ_CAPTURE_DIR")) g_seq_dir = d;
    if (!g_seq_dir.empty()) {
        CreateDirectoryA(g_seq_dir.c_str(), NULL);
        OutputDebugStringA("[xrt-shim] op-sequence capture armed");
    }
    if (const char* d = getenv("FLM_BO_CAPTURE_DIR")) g_bo_dir = d;
    if (!g_bo_dir.empty()) {
        CreateDirectoryA(g_bo_dir.c_str(), NULL);
        if (const char* m = getenv("FLM_BO_DUMP_MAX")) g_bo_dump_max = strtoull(m, nullptr, 10);
        if (const char* m = getenv("FLM_BO_RUNARG_MAX")) g_bo_runarg_max = strtoull(m, nullptr, 10);
        OutputDebugStringA("[xrt-shim] tensor-data capture armed");
    }
    g_init.store(true);
}

void seq_dump(const void* buf, uint64_t n) {
    if (g_seq_dir.empty() || !buf || !n) return;
    uint32_t idx = g_seq_index.fetch_add(1);
    char path[MAX_PATH];
    _snprintf_s(path, sizeof(path), _TRUNCATE, "%s\\%06u.seq", g_seq_dir.c_str(), idx);
    FILE* f = nullptr;
    if (fopen_s(&f, path, "wb") == 0 && f) { fwrite(buf, 1, (size_t)n, f); fclose(f); }
    char line[256];
    _snprintf_s(line, sizeof(line), _TRUNCATE, "%06u\telf\t%llu\t%016llx\n",
                idx, (unsigned long long)n, (unsigned long long)fnv1a(buf, (size_t)n));
    std::lock_guard<std::mutex> lk(g_seq_trace_mtx);
    if (fopen_s(&f, (g_seq_dir + "\\trace.tsv").c_str(), "ab") == 0 && f) {
        fputs(line, f); fclose(f);
    }
}

// Dump/hash the exact bytes a sync() moves across the boundary. `host` is the
// mapped base for this bo (may be null if map() wasn't observed); the synced
// region is [host+offset, host+offset+n).
void bo_capture(void* self, int dir, uint64_t n, uint64_t offset, void* host) {
    if (g_bo_dir.empty()) return;
    uint32_t idx = g_bo_index.fetch_add(1);
    const uint8_t* region = host ? static_cast<uint8_t*>(host) + offset : nullptr;
    uint64_t hash = region && n ? fnv1a(region, (size_t)n) : 0;

    int dumped = 0;
    if (region && n && g_bo_dump_max && n <= g_bo_dump_max) {
        char path[MAX_PATH];
        _snprintf_s(path, sizeof(path), _TRUNCATE, "%s\\%06u.bo", g_bo_dir.c_str(), idx);
        FILE* f = nullptr;
        if (fopen_s(&f, path, "wb") == 0 && f) {
            fwrite(region, 1, (size_t)n, f); fclose(f); dumped = 1;
        }
    }
    char line[256];
    _snprintf_s(line, sizeof(line), _TRUNCATE, "%06u\t%s\t%llu\t%llu\t%016llx\t%d\n",
                idx, dir == SYNC_TO_DEVICE ? "H2D" : "D2H",
                (unsigned long long)n, (unsigned long long)offset,
                (unsigned long long)hash, dumped);
    { std::lock_guard<std::mutex> lk(g_bo_trace_mtx);
      FILE* f = nullptr;
      if (fopen_s(&f, (g_bo_dir + "\\bo_trace.tsv").c_str(), "ab") == 0 && f) {
          fputs(line, f); fclose(f);
      } }
    // unified event: link this sync's bytes (bo idx) to the bo `self` pointer.
    char ev[160];
    _snprintf_s(ev, sizeof(ev), _TRUNCATE, "%llu\t%s\t%p\t%06u\t%llu\t%016llx\n",
                (unsigned long long)g_event.fetch_add(1),
                dir == SYNC_TO_DEVICE ? "H2D" : "D2H", self, idx,
                (unsigned long long)n, (unsigned long long)hash);
    event_log(ev);
}

} // namespace

// ---- exported thunks (aliased from xrt_coreutil.def) -----------------------

extern "C" void* flmcap_elf_ctor(void* self, const void* buf, uint64_t n) {
    if (!g_init.load()) init();
    seq_dump(buf, n);          // op-sequence plane (FLM_SEQ_CAPTURE_DIR)
    elf_capture(self, buf, n); // unified event plane (FLM_BO_CAPTURE_DIR)
    elf_ctor_t real = g_real_elf.load();
    return real ? real(self, buf, n) : self;
}

// module(const elf&) -- link module `this` -> elf ptr.
extern "C" void* flmcap_module_ctor(void* self, const void* elf) {
    if (!g_init.load()) init();
    if (!g_bo_dir.empty()) {
        char ev[96];
        _snprintf_s(ev, sizeof(ev), _TRUNCATE, "%llu\tMODULE\t%p\t%p\n",
                    (unsigned long long)g_event.fetch_add(1), self, elf);
        event_log(ev);
    }
    module_ctor_t real = g_real_module_ctor.load();
    return real ? real(self, elf) : self;
}

// ext::kernel(hw_context, module, name) -- link kernel `this` -> module ptr.
extern "C" void* flmcap_extkern_ctor(void* self, const void* ctx, const void* mod, const void* name) {
    if (!g_init.load()) init();
    if (!g_bo_dir.empty()) {
        char ev[96];
        _snprintf_s(ev, sizeof(ev), _TRUNCATE, "%llu\tKERNEL\t%p\t%p\n",
                    (unsigned long long)g_event.fetch_add(1), self, mod);
        event_log(ev);
    }
    extkern_ctor_t real = g_real_extkern_ctor.load();
    return real ? real(self, ctx, mod, name) : self;
}

// run(const kernel&) -- link run `this` -> kernel ptr.
extern "C" void* flmcap_run_ctor(void* self, const void* kernel) {
    if (!g_init.load()) init();
    if (!g_bo_dir.empty()) {
        char ev[96];
        _snprintf_s(ev, sizeof(ev), _TRUNCATE, "%llu\tRUN\t%p\t%p\n",
                    (unsigned long long)g_event.fetch_add(1), self, kernel);
        event_log(ev);
    }
    run_ctor_t real = g_real_run_ctor.load();
    return real ? real(self, kernel) : self;
}

// run::set_arg_at_index(int index, const xrt::bo& bo) -- this=rcx, index=edx,
// bo=r8. Records which buffer is bound to which kernel arg for this run.
extern "C" void flmcap_run_setarg_bo(void* run, int index, const void* bo) {
    if (!g_init.load()) init();
    if (!g_bo_dir.empty()) {
        char ev[128];
        _snprintf_s(ev, sizeof(ev), _TRUNCATE, "%llu\tSETARG\t%p\t%d\t%p\n",
                    (unsigned long long)g_event.fetch_add(1), run, index, bo);
        event_log(ev);
        std::lock_guard<std::mutex> lk(g_run_args_mtx);
        g_run_args[run].push_back({index, const_cast<void*>(bo)});
    }
    run_setarg_bo_t real = g_real_run_setarg_bo.load();
    if (real) real(run, index, bo);
}

// run::start() -- submit marker. Dump the EXACT bytes of each bound buffer NOW
// (inputs are written, kernel about to execute) -> deterministic op inputs.
extern "C" void flmcap_run_start(void* run) {
    if (!g_init.load()) init();
    if (!g_bo_dir.empty()) {
        uint64_t sev = g_event.fetch_add(1);
        char ev[96];
        _snprintf_s(ev, sizeof(ev), _TRUNCATE, "%llu\tSTART\t%p\n", (unsigned long long)sev, run);
        event_log(ev);
        std::vector<std::pair<int, void*>> args;
        { std::lock_guard<std::mutex> lk(g_run_args_mtx);
          auto it = g_run_args.find(run);
          if (it != g_run_args.end()) args = it->second; }
        bo_size_t bosize = g_real_bo_size.load();
        for (auto& a : args) {
            void* host = nullptr;
            { std::lock_guard<std::mutex> lk(g_bo_map_mtx);
              auto h = g_bo_hostptr.find(a.second);
              if (h != g_bo_hostptr.end()) host = h->second; }
            uint64_t sz = bosize ? bosize(a.second) : 0;
            uint64_t hash = (host && sz && sz <= g_bo_runarg_max) ? fnv1a(host, (size_t)sz) : 0;
            int dumped = 0;
            if (host && sz && sz <= g_bo_runarg_max && g_bo_dump_max && sz <= g_bo_dump_max) {
                uint64_t key = sz ^ (hash * 1099511628211ULL);
                bool need;
                { std::lock_guard<std::mutex> lk(g_blob_mtx); need = g_blob_done.emplace(key, 1).second; }
                dumped = 1;  // content is available as blob_<size>_<hash>.bin either way
                if (need) {
                    char path[MAX_PATH];
                    _snprintf_s(path, sizeof(path), _TRUNCATE, "%s\\blob_%llu_%016llx.bin",
                                g_bo_dir.c_str(), (unsigned long long)sz, (unsigned long long)hash);
                    FILE* f = nullptr;
                    if (fopen_s(&f, path, "wb") == 0 && f) { fwrite(host, 1, (size_t)sz, f); fclose(f); }
                }
            }
            char rl[160];
            _snprintf_s(rl, sizeof(rl), _TRUNCATE, "%llu\tRUNARG\t%llu\t%d\t%p\t%llu\t%016llx\t%d\n",
                        (unsigned long long)g_event.fetch_add(1), (unsigned long long)sev,
                        a.first, a.second, (unsigned long long)sz, (unsigned long long)hash, dumped);
            event_log(rl);
        }
        { std::lock_guard<std::mutex> lk(g_run_args_mtx); g_run_args.erase(run); }
    }
    run_start_t real = g_real_run_start.load();
    if (real) real(run);
}

extern "C" void* flmcap_bo_map(void* self) {
    if (!g_init.load()) init();
    bo_map_t real = g_real_bo_map.load();
    void* host = real ? real(self) : nullptr;
    if (!g_bo_dir.empty() && host) {
        std::lock_guard<std::mutex> lk(g_bo_map_mtx);
        g_bo_hostptr[self] = host;
    }
    return host;
}

extern "C" void flmcap_bo_sync(void* self, int dir, uint64_t n, uint64_t offset) {
    if (!g_init.load()) init();
    void* host = nullptr;
    if (!g_bo_dir.empty()) {
        std::lock_guard<std::mutex> lk(g_bo_map_mtx);
        auto it = g_bo_hostptr.find(self);
        if (it != g_bo_hostptr.end()) host = it->second;
    }
    // Host->device: the engine has already written the bytes; capture before the
    // flush. Device->host: capture after the real sync has filled the buffer.
    if (!g_bo_dir.empty() && dir == SYNC_TO_DEVICE) bo_capture(self, dir, n, offset, host);
    bo_sync_t real = g_real_bo_sync.load();
    if (real) real(self, dir, n, offset);
    if (!g_bo_dir.empty() && dir != SYNC_TO_DEVICE) bo_capture(self, dir, n, offset, host);
}

extern "C" void flmcap_bo_dtor(void* self) {
    if (!g_init.load()) init();
    if (!g_bo_dir.empty()) {
        std::lock_guard<std::mutex> lk(g_bo_map_mtx);
        g_bo_hostptr.erase(self);
    }
    bo_dtor_t real = g_real_bo_dtor.load();
    if (real) real(self);
}

BOOL APIENTRY DllMain(HMODULE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) init();
    return TRUE;
}
