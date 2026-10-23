//! Exclusive lockfile over the shared physical NPU device, used by every
//! NPU-touching backend (previously duplicated in `generate_5li3` and
//! `generate_l30`; consolidated here). Mirrors `l30_run_npu.py`'s
//! `acquire_lock`/`release_lock`: create-exclusive with retry, stale-lock
//! removal after ~40 minutes, a 30-minute acquire timeout, and release on
//! `Drop`. A heartbeat keeps long benchmark suites from appearing stale.

use std::io::Write;
use std::{path::PathBuf, sync::mpsc, thread::JoinHandle, time::Duration};

pub struct NpuLock {
    path: PathBuf,
    owner: String,
    stop: mpsc::Sender<()>,
    heartbeat: Option<JoinHandle<()>>,
}

impl NpuLock {
    // One lockfile per machine: every process that drives the NPU must agree on it.
    fn path() -> std::path::PathBuf {
        std::env::var_os("PHLEGM_NPU_LOCK")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("phlegm-npu.lock"))
    }

    const STALE_SECS: u64 = 40 * 60;
    const TIMEOUT_SECS: u64 = 30 * 60;

    pub fn acquire(tag: &str) -> Result<NpuLock, String> {
        let path_buf = Self::path();
        let path = path_buf.as_path();
        let start = std::time::Instant::now();
        loop {
            match std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(path)
            {
                Ok(mut f) => {
                    if let Err(e) = write!(f, "{}", std::process::id()).and_then(|_| f.flush()) {
                        let _ = std::fs::remove_file(path);
                        return Err(format!("writing NPU lock: {e}"));
                    }
                    return Ok(Self::start(path_buf, Duration::from_secs(60)));
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if let Ok(meta) = std::fs::metadata(path) {
                        if let Ok(age) = meta.modified().and_then(|m| {
                            m.elapsed()
                                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
                        }) {
                            if age.as_secs() > Self::STALE_SECS {
                                eprintln!(
                                    "{tag}: removing stale NPU lock (age {}s)",
                                    age.as_secs()
                                );
                                let _ = std::fs::remove_file(path);
                                continue;
                            }
                        }
                    }
                    if start.elapsed().as_secs() > Self::TIMEOUT_SECS {
                        return Err(
                            "could not acquire NPU lock: timed out (device busy)".to_string()
                        );
                    }
                    eprintln!("{tag}: NPU busy (lock held) -- waiting...");
                    std::thread::sleep(std::time::Duration::from_secs(5));
                }
                Err(e) => return Err(format!("acquire NPU lock: {e}")),
            }
        }
    }

    fn start(path: PathBuf, interval: Duration) -> Self {
        let owner = std::process::id().to_string();
        let (stop, receiver) = mpsc::channel();
        let lease_path = path.clone();
        let lease_owner = owner.clone();
        let heartbeat = std::thread::spawn(move || {
            while let Err(mpsc::RecvTimeoutError::Timeout) = receiver.recv_timeout(interval) {
                if std::fs::read_to_string(&lease_path).ok().as_ref() != Some(&lease_owner) {
                    break;
                }
                let renewed = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&lease_path)
                    .and_then(|mut f| f.write_all(lease_owner.as_bytes()).and_then(|_| f.flush()));
                if renewed.is_err() {
                    break;
                }
            }
        });
        Self {
            path,
            owner,
            stop,
            heartbeat: Some(heartbeat),
        }
    }

    pub fn ensure_held(&self) -> Result<(), String> {
        if std::fs::read_to_string(&self.path).ok().as_ref() != Some(&self.owner) {
            return Err("NPU lock ownership was lost; discard the run".into());
        }
        Ok(())
    }
}

impl Drop for NpuLock {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(worker) = self.heartbeat.take() {
            let _ = worker.join();
        }
        if self.ensure_held().is_ok() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    fn fixture() -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/bench-lock-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "{}.lock",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, std::process::id().to_string()).unwrap();
        path
    }

    #[test]
    fn heartbeat_renews_old_lock_and_drop_releases_it() {
        let path = fixture();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_times(std::fs::FileTimes::new().set_modified(UNIX_EPOCH))
            .unwrap();
        drop(file);
        let lock = NpuLock::start(path.clone(), Duration::from_millis(10));
        let start = Instant::now();
        while std::fs::metadata(&path).unwrap().modified().unwrap() == UNIX_EPOCH {
            assert!(
                start.elapsed() < Duration::from_secs(2),
                "heartbeat did not renew lease"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        lock.ensure_held().unwrap();
        drop(lock);
        assert!(!path.exists());
    }

    #[test]
    fn lost_lock_is_reported_and_another_owner_is_not_deleted() {
        let path = fixture();
        let lock = NpuLock::start(path.clone(), Duration::from_secs(60));
        std::fs::write(&path, "another owner").unwrap();
        assert!(lock.ensure_held().is_err());
        drop(lock);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "another owner");
    }
}
