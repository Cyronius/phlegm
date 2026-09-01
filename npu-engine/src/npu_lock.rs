//! Exclusive lockfile over the shared physical NPU device, used by every
//! NPU-touching backend (previously duplicated in `generate_5li3` and
//! `generate_l30`; consolidated here). Mirrors `l30_run_npu.py`'s
//! `acquire_lock`/`release_lock`: create-exclusive with retry, stale-lock
//! removal after ~40 minutes, a 30-minute acquire timeout, and release on
//! `Drop` (so an early `?`-return still releases it).

use std::io::Write;
use std::path::Path;

pub struct NpuLock;

impl NpuLock {
    const PATH: &'static str = "C:/code/FastFlowLM/npu-engine/.npu.lock";
    const STALE_SECS: u64 = 40 * 60;
    const TIMEOUT_SECS: u64 = 30 * 60;

    pub fn acquire(tag: &str) -> Result<NpuLock, String> {
        let path = Path::new(Self::PATH);
        let start = std::time::Instant::now();
        loop {
            match std::fs::OpenOptions::new().create_new(true).write(true).open(path) {
                Ok(mut f) => {
                    let _ = write!(f, "{}", std::process::id());
                    return Ok(NpuLock);
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if let Ok(meta) = std::fs::metadata(path) {
                        if let Ok(age) = meta
                            .modified()
                            .and_then(|m| m.elapsed().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)))
                        {
                            if age.as_secs() > Self::STALE_SECS {
                                eprintln!("{tag}: removing stale NPU lock (age {}s)", age.as_secs());
                                let _ = std::fs::remove_file(path);
                                continue;
                            }
                        }
                    }
                    if start.elapsed().as_secs() > Self::TIMEOUT_SECS {
                        return Err("could not acquire NPU lock: timed out (device busy)".to_string());
                    }
                    eprintln!("{tag}: NPU busy (lock held) -- waiting...");
                    std::thread::sleep(std::time::Duration::from_secs(5));
                }
                Err(e) => return Err(format!("acquire NPU lock: {e}")),
            }
        }
    }
}

impl Drop for NpuLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(Self::PATH);
    }
}
