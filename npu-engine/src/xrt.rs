//! Safe Rust bindings over the `xrt-shim` C ABI (see `xrt-shim/xrt_shim.h`).
//!
//! The shim wraps XRT's modern C++ ELF/module/ext::kernel flow; here we present
//! it as RAII handles. Failures surface the shim's thread-local error string as
//! `Err(String)`. Buffer sizes passed to [`Device::bo`] must already be padded
//! to the 1 MB alignment the XDNA runtime expects (see [`padup`]).

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;

#[allow(non_camel_case_types)]
type handle = *mut c_void;

extern "C" {
    fn xrtsh_last_error() -> *const c_char;

    fn xrtsh_device_open(index: c_int) -> handle;
    fn xrtsh_device_name(dev: handle, out: *mut c_char, cap: c_int) -> c_int;
    fn xrtsh_device_free(dev: handle);

    fn xrtsh_hwctx_create(dev: handle, xclbin_path: *const c_char) -> handle;
    fn xrtsh_hwctx_free(ctx: handle);

    fn xrtsh_kernel_create(ctx: handle, elf_path: *const c_char) -> handle;
    fn xrtsh_kernel_free(k: handle);

    fn xrtsh_bo_create(dev: handle, size: usize) -> handle;
    fn xrtsh_bo_map(bo: handle) -> *mut c_void;
    fn xrtsh_bo_sync(bo: handle, to_device: c_int) -> c_int;
    fn xrtsh_bo_write(bo: handle, src: *const c_void, n: usize, off: usize) -> c_int;
    fn xrtsh_bo_read(bo: handle, dst: *mut c_void, n: usize, off: usize) -> c_int;
    fn xrtsh_bo_free(bo: handle);

    fn xrtsh_run_create(k: handle) -> handle;
    fn xrtsh_run_set_arg_int(r: handle, idx: c_int, val: c_int) -> c_int;
    fn xrtsh_run_set_arg_bo(r: handle, idx: c_int, bo: handle) -> c_int;
    fn xrtsh_run_start(r: handle) -> c_int;
    fn xrtsh_run_wait(r: handle) -> c_int;
    fn xrtsh_run_free(r: handle);

    fn xrtsh_runlist_create(ctx: handle) -> handle;
    fn xrtsh_runlist_add(rl: handle, r: handle) -> c_int;
    fn xrtsh_runlist_execute(rl: handle) -> c_int;
    fn xrtsh_runlist_wait(rl: handle) -> c_int;
    fn xrtsh_runlist_free(rl: handle);
}

/// `ert_cmd_state` value for a completed run.
pub const STATE_COMPLETED: i32 = 4;

pub type Result<T> = std::result::Result<T, String>;

fn last_error() -> String {
    unsafe {
        let p = xrtsh_last_error();
        if p.is_null() {
            return "unknown error".to_string();
        }
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

fn cpath(p: &str) -> Result<CString> {
    CString::new(p).map_err(|_| "path contains NUL".to_string())
}

/// Round `n` up to the 1 MB BO alignment.
pub fn padup(n: usize) -> usize {
    let a = 1024 * 1024;
    (n + a - 1) / a * a
}

pub struct Device(handle);
pub struct Context(handle);
pub struct Kernel(handle);
pub struct Bo {
    h: handle,
    size: usize,
}
pub struct Run(handle);
pub struct Runlist(handle);

impl Device {
    pub fn open(index: i32) -> Result<Device> {
        let h = unsafe { xrtsh_device_open(index) };
        if h.is_null() {
            return Err(last_error());
        }
        Ok(Device(h))
    }

    pub fn name(&self) -> String {
        let mut buf = [0u8; 256];
        let n = unsafe { xrtsh_device_name(self.0, buf.as_mut_ptr() as *mut c_char, 256) };
        if n < 0 {
            return String::new();
        }
        let len = (n as usize).min(255);
        String::from_utf8_lossy(&buf[..len]).into_owned()
    }

    /// Register an xclbin and create a hw_context bound to it.
    pub fn hwctx(&self, xclbin_path: &Path) -> Result<Context> {
        let p = cpath(&xclbin_path.to_string_lossy())?;
        let h = unsafe { xrtsh_hwctx_create(self.0, p.as_ptr()) };
        if h.is_null() {
            return Err(last_error());
        }
        Ok(Context(h))
    }

    /// Allocate an `xrt::ext::bo`. `size` is padded up to 1 MB.
    pub fn bo(&self, size: usize) -> Result<Bo> {
        let padded = padup(size);
        let h = unsafe { xrtsh_bo_create(self.0, padded) };
        if h.is_null() {
            return Err(last_error());
        }
        Ok(Bo { h, size: padded })
    }
}

impl Context {
    /// elf(path) -> module -> ext::kernel(ctx, module, "MLIR_AIE").
    pub fn kernel(&self, elf_path: &Path) -> Result<Kernel> {
        let p = cpath(&elf_path.to_string_lossy())?;
        let h = unsafe { xrtsh_kernel_create(self.0, p.as_ptr()) };
        if h.is_null() {
            return Err(last_error());
        }
        Ok(Kernel(h))
    }

    pub fn runlist(&self) -> Result<Runlist> {
        let h = unsafe { xrtsh_runlist_create(self.0) };
        if h.is_null() {
            return Err(last_error());
        }
        Ok(Runlist(h))
    }
}

impl Bo {
    pub fn size(&self) -> usize {
        self.size
    }

    /// Zero the whole (padded) buffer, then copy `data` into the front.
    pub fn init(&mut self, data: &[u8]) -> Result<()> {
        // Zero via the mapped pointer.
        let p = unsafe { xrtsh_bo_map(self.h) };
        if p.is_null() {
            return Err(last_error());
        }
        unsafe { std::ptr::write_bytes(p as *mut u8, 0, self.size) };
        let n = data.len().min(self.size);
        self.write(&data[..n], 0)
    }

    pub fn write(&mut self, data: &[u8], off: usize) -> Result<()> {
        let rc = unsafe {
            xrtsh_bo_write(self.h, data.as_ptr() as *const c_void, data.len(), off)
        };
        if rc < 0 {
            return Err(last_error());
        }
        Ok(())
    }

    pub fn read(&self, dst: &mut [u8], off: usize) -> Result<()> {
        let rc =
            unsafe { xrtsh_bo_read(self.h, dst.as_mut_ptr() as *mut c_void, dst.len(), off) };
        if rc < 0 {
            return Err(last_error());
        }
        Ok(())
    }

    pub fn sync_to_device(&self) -> Result<()> {
        if unsafe { xrtsh_bo_sync(self.h, 1) } < 0 {
            return Err(last_error());
        }
        Ok(())
    }

    pub fn sync_from_device(&self) -> Result<()> {
        if unsafe { xrtsh_bo_sync(self.h, 0) } < 0 {
            return Err(last_error());
        }
        Ok(())
    }
}

impl Kernel {
    pub fn run(&self) -> Result<Run> {
        let h = unsafe { xrtsh_run_create(self.0) };
        if h.is_null() {
            return Err(last_error());
        }
        Ok(Run(h))
    }
}

impl Run {
    pub fn set_arg_int(&mut self, idx: i32, val: i32) -> Result<()> {
        if unsafe { xrtsh_run_set_arg_int(self.0, idx, val) } < 0 {
            return Err(last_error());
        }
        Ok(())
    }

    pub fn set_arg_bo(&mut self, idx: i32, bo: &Bo) -> Result<()> {
        if unsafe { xrtsh_run_set_arg_bo(self.0, idx, bo.h) } < 0 {
            return Err(last_error());
        }
        Ok(())
    }

    pub fn start(&mut self) -> Result<()> {
        if unsafe { xrtsh_run_start(self.0) } < 0 {
            return Err(last_error());
        }
        Ok(())
    }

    /// Returns the `ert_cmd_state` (4 == completed).
    pub fn wait(&mut self) -> Result<i32> {
        let st = unsafe { xrtsh_run_wait(self.0) };
        if st < 0 {
            return Err(last_error());
        }
        Ok(st)
    }

    fn raw(&self) -> handle {
        self.0
    }
}

impl Runlist {
    /// The run must outlive the runlist's `wait` (XRT holds a reference).
    pub fn add(&mut self, run: &Run) -> Result<()> {
        if unsafe { xrtsh_runlist_add(self.0, run.raw()) } < 0 {
            return Err(last_error());
        }
        Ok(())
    }

    pub fn execute(&mut self) -> Result<()> {
        if unsafe { xrtsh_runlist_execute(self.0) } < 0 {
            return Err(last_error());
        }
        Ok(())
    }

    pub fn wait(&mut self) -> Result<()> {
        if unsafe { xrtsh_runlist_wait(self.0) } < 0 {
            return Err(last_error());
        }
        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { xrtsh_device_free(self.0) }
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe { xrtsh_hwctx_free(self.0) }
    }
}
impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { xrtsh_kernel_free(self.0) }
    }
}
impl Drop for Bo {
    fn drop(&mut self) {
        unsafe { xrtsh_bo_free(self.h) }
    }
}
impl Drop for Run {
    fn drop(&mut self) {
        unsafe { xrtsh_run_free(self.0) }
    }
}
impl Drop for Runlist {
    fn drop(&mut self) {
        unsafe { xrtsh_runlist_free(self.0) }
    }
}
