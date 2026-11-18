//! Resident FLUX.2 sampler: one worker keeps the weights and the NPU until the model is unloaded.
use crate::image::{Flux2ImageConfig, ImageConfig, ImageRequest};
use crate::npu_lock::NpuLock;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(600);
// A first request may pack weights (~2 min) and build the kernels before it samples.
const REPLY_TIMEOUT: Duration = Duration::from_secs(1800);
const EXIT_GRACE: Duration = Duration::from_secs(15);

/// An image model held by the server; only FLUX.2 keeps a resident worker.
pub struct ImageModel {
    pub cfg: ImageConfig,
    resident: Option<Mutex<Flux2Resident>>,
}

impl ImageModel {
    pub fn per_request(cfg: ImageConfig) -> Self {
        Self { cfg, resident: None }
    }

    pub fn load(cfg: ImageConfig) -> Result<Self, String> {
        let resident = match &cfg {
            ImageConfig::Flux2(flux) => Some(Mutex::new(Flux2Resident::start(flux.clone())?)),
            _ => None,
        };
        Ok(Self { cfg, resident })
    }

    pub fn generate(&self, request: &ImageRequest, out: &Path) -> Result<PathBuf, String> {
        let Some(resident) = &self.resident else {
            return crate::image::generate(&self.cfg, request, out);
        };
        let resolution = self.cfg.validate_request(request)?;
        if request.projected_conditioning.is_some() {
            return Err("projected conditioning is only available through `phlegm image`".into());
        }
        if out.exists() {
            return Err(format!("image output directory already exists: {}", out.display()));
        }
        resident.lock().unwrap_or_else(|e| e.into_inner()).generate(request, out, resolution)
    }
}

struct Flux2Resident {
    cfg: Flux2ImageConfig,
    worker: Option<Worker>,
    lock: Option<NpuLock>,
}

impl Flux2Resident {
    fn start(cfg: Flux2ImageConfig) -> Result<Self, String> {
        let lock = NpuLock::acquire("phlegm images")?;
        let worker = Worker::spawn(&cfg)?;
        Ok(Self { cfg, worker: Some(worker), lock: Some(lock) })
    }

    fn generate(&mut self, request: &ImageRequest, out: &Path, resolution: u32) -> Result<PathBuf, String> {
        if let Err(e) = self.lock.as_ref().ok_or("NPU lock was lost").map_err(String::from)
            .and_then(|l| l.ensure_held()) {
            self.worker = None;
            return Err(e);
        }
        if self.worker.is_none() {
            self.worker = Some(Worker::spawn(&self.cfg)?);
        }
        let message = json!({"prompt": request.prompt, "size": resolution,
            "steps": request.steps.unwrap_or(4), "seed": request.seed,
            "image": request.image, "out": out});
        let reply = match self.worker.as_mut().unwrap().request(&message) {
            Ok(reply) => reply,
            Err(e) => {
                // The next request starts a fresh worker instead of reusing a broken one.
                self.worker = None;
                return Err(format!("FLUX.2 worker: {e}"));
            }
        };
        if reply.get("ok").and_then(Value::as_bool) != Some(true) {
            return Err(reply.get("error").and_then(Value::as_str)
                .unwrap_or("FLUX.2 worker failed").to_string());
        }
        let image = reply.get("image").and_then(Value::as_str).map(PathBuf::from)
            .unwrap_or_else(|| out.join("image.png"));
        if !image.is_file() {
            return Err("FLUX.2 worker did not produce a PNG".into());
        }
        Ok(image)
    }
}

impl Drop for Flux2Resident {
    fn drop(&mut self) {
        // The worker must exit before the lock is released.
        drop(self.worker.take());
        drop(self.lock.take());
    }
}

struct Worker {
    child: Child,
    input: Option<ChildStdin>,
    replies: Receiver<Result<Value, String>>,
    #[cfg(windows)]
    _job: job::Job,
}

impl Worker {
    fn spawn(cfg: &Flux2ImageConfig) -> Result<Self, String> {
        let mut command = crate::image::flux2_base_command(cfg, "Phlegm FLUX.2 klein resident worker", 60)?;
        command.arg("--serve").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit());
        crate::bench::reference::die_with_parent(&mut command);
        let mut child = command.spawn().map_err(|e| format!("starting FLUX.2 worker: {e}"))?;
        #[cfg(windows)]
        let job = match job::Job::assign(&child) {
            Ok(job) => job,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e);
            }
        };
        let input = child.stdin.take().ok_or("missing FLUX.2 worker stdin")?;
        let output = child.stdout.take().ok_or("missing FLUX.2 worker stdout")?;
        let (tx, replies) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(output);
            loop {
                let mut line = Vec::new();
                let value = match (&mut reader).take(1_000_001).read_until(b'\n', &mut line) {
                    Ok(0) => Err("worker exited".to_string()),
                    Ok(_) if line.len() > 1_000_000 => Err("worker reply exceeds 1 MB".to_string()),
                    Ok(_) => serde_json::from_slice(&line).map_err(|e| format!("invalid worker reply: {e}")),
                    Err(e) => Err(format!("reading worker: {e}")),
                };
                let failed = value.is_err();
                if tx.send(value).is_err() || failed {
                    break;
                }
            }
        });
        let mut worker = Self { child, input: Some(input), replies,
                                #[cfg(windows)] _job: job };
        let hello = worker.receive(STARTUP_TIMEOUT).map_err(|e| format!("FLUX.2 worker startup: {e}"))?;
        if hello.get("ok").and_then(Value::as_bool) != Some(true)
            || hello.get("protocol").and_then(Value::as_u64) != Some(1) {
            let why = hello.get("error").and_then(Value::as_str).unwrap_or("unsupported protocol");
            worker.input = None;
            return Err(format!("FLUX.2 worker startup: {why}"));
        }
        Ok(worker)
    }

    fn receive(&mut self, timeout: Duration) -> Result<Value, String> {
        self.replies.recv_timeout(timeout).map_err(|e| format!("no reply ({e})"))?
    }

    fn request(&mut self, value: &Value) -> Result<Value, String> {
        let input = self.input.as_mut().ok_or("worker input is closed")?;
        serde_json::to_writer(&mut *input, value).map_err(|e| e.to_string())?;
        input.write_all(b"\n").and_then(|_| input.flush()).map_err(|e| format!("writing request: {e}"))?;
        self.receive(REPLY_TIMEOUT)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Closing stdin lets the worker free its hardware context itself; kill only if it hangs.
        drop(self.input.take());
        let deadline = Instant::now() + EXIT_GRACE;
        while Instant::now() < deadline {
            if !matches!(self.child.try_wait(), Ok(None)) {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// `--keep-alive`: omitted = 600 s idle, `-1` = never unload when idle, `0` = unload at once.
pub fn keep_alive_option(value: Option<&str>) -> Result<Option<Duration>, String> {
    match value.map(str::trim) {
        None => Ok(Some(Duration::from_secs(600))),
        Some("-1") => Ok(None),
        Some(v) => v.parse::<u64>().map(|s| Some(Duration::from_secs(s))).map_err(|_|
            "--keep-alive must be -1 (never unload when idle) or a nonnegative number of seconds".into()),
    }
}

/// Windows has no parent-death signal, so a kill-on-close job ends the worker tree with the server.
#[cfg(windows)]
mod job {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;

    type Handle = *mut c_void;
    const EXTENDED_LIMIT_INFORMATION: i32 = 9;
    const KILL_ON_JOB_CLOSE: u32 = 0x2000;

    #[repr(C)]
    #[derive(Default)]
    struct BasicLimits {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct ExtendedLimits {
        basic: BasicLimits,
        io_counters: [u64; 6],
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateJobObjectW(attributes: *mut c_void, name: *const u16) -> Handle;
        fn SetInformationJobObject(job: Handle, class: i32, info: *mut c_void, length: u32) -> i32;
        fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
        fn CloseHandle(handle: Handle) -> i32;
    }

    pub struct Job(Handle);
    // SAFETY: a job handle is a kernel object handle, usable from any thread.
    unsafe impl Send for Job {}

    impl Job {
        pub fn assign(child: &std::process::Child) -> Result<Self, String> {
            // SAFETY: plain Win32 calls on handles owned here or by `child`, with a correctly sized struct.
            unsafe {
                let handle = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
                if handle.is_null() {
                    return Err(format!("CreateJobObject: {}", std::io::Error::last_os_error()));
                }
                let job = Job(handle);
                let mut limits = ExtendedLimits::default();
                limits.basic.limit_flags = KILL_ON_JOB_CLOSE;
                if SetInformationJobObject(job.0, EXTENDED_LIMIT_INFORMATION,
                                           &mut limits as *mut _ as *mut c_void,
                                           std::mem::size_of::<ExtendedLimits>() as u32) == 0 {
                    return Err(format!("SetInformationJobObject: {}", std::io::Error::last_os_error()));
                }
                if AssignProcessToJobObject(job.0, child.as_raw_handle() as Handle) == 0 {
                    return Err(format!("AssignProcessToJobObject: {}", std::io::Error::last_os_error()));
                }
                Ok(job)
            }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // SAFETY: the handle came from CreateJobObjectW and is closed once.
            unsafe { CloseHandle(self.0); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_alive_defaults_to_ten_minutes_and_minus_one_never_unloads() {
        assert_eq!(keep_alive_option(None).unwrap(), Some(Duration::from_secs(600)));
        assert_eq!(keep_alive_option(Some("-1")).unwrap(), None);
        assert_eq!(keep_alive_option(Some("0")).unwrap(), Some(Duration::ZERO));
        assert_eq!(keep_alive_option(Some("90")).unwrap(), Some(Duration::from_secs(90)));
        for bad in ["-2", "ten", "1.5", ""] {
            assert!(keep_alive_option(Some(bad)).unwrap_err().contains("--keep-alive"));
        }
    }
}
