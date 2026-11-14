//! One resident model, with a lease held through the entire response (including SSE).
use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

struct State<T> {
    busy: bool,
    next_ticket: u64,
    waiting: VecDeque<u64>,
    resident: Option<(String, T)>,
    status: Option<(String, &'static str)>,
    idle_since: Instant,
}
pub(super) struct Slot<T> {
    state: Mutex<State<T>>,
    ready: Condvar,
}
pub(super) struct Lease<'a, T> {
    slot: &'a Slot<T>,
    pub resident: Option<(String, T)>,
}
impl<T> Slot<T> {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                busy: false,
                next_ticket: 0,
                waiting: VecDeque::new(),
                resident: None,
                status: None,
                idle_since: Instant::now(),
            }),
            ready: Condvar::new(),
        }
    }
    pub fn acquire(&self, timeout: Duration) -> Result<Lease<'_, T>, String> {
        let mut state = self.state.lock().unwrap();
        let ticket = state.next_ticket;
        state.next_ticket = state.next_ticket.wrapping_add(1);
        state.waiting.push_back(ticket);
        let blocked = |s: &mut State<T>| s.busy || s.waiting.front() != Some(&ticket);
        let (mut state, _) = self
            .ready
            .wait_timeout_while(state, timeout, blocked)
            .unwrap();
        if blocked(&mut state) {
            state.waiting.retain(|t| *t != ticket);
            self.ready.notify_all();
            return Err("model queue timeout; retry later".into());
        }
        state.waiting.pop_front();
        state.busy = true;
        Ok(Lease {
            slot: self,
            resident: state.resident.take(),
        })
    }
    pub fn status(&self) -> Option<(String, &'static str)> {
        self.state.lock().unwrap().status.clone()
    }
    pub fn expire(&self, keep_alive: Duration) {
        let mut state = self.state.lock().unwrap();
        if !state.busy && state.waiting.is_empty() && state.idle_since.elapsed() >= keep_alive {
            // Destruction must complete before another lease can load a model.
            drop(state.resident.take());
            state.status = None;
        }
    }
}
impl<T> Lease<'_, T> {
    pub fn select(
        &mut self,
        id: &str,
        load: impl FnOnce() -> Result<T, String>,
    ) -> Result<(), String> {
        if self
            .resident
            .as_ref()
            .is_some_and(|(current, _)| current == id)
        {
            return Ok(());
        }
        self.slot.state.lock().unwrap().status = None;
        // Do not keep an Arc to the old backend: its child must be reaped and its lock
        // released before starting the replacement, including on a failed load.
        drop(self.resident.take());
        self.slot.state.lock().unwrap().status = Some((id.into(), "loading"));
        match load() {
            Ok(model) => {
                self.resident = Some((id.into(), model));
                self.slot.state.lock().unwrap().status = Some((id.into(), "loaded"));
                Ok(())
            }
            Err(e) => {
                self.slot.state.lock().unwrap().status = None;
                Err(e)
            }
        }
    }
}
impl<T> Drop for Lease<'_, T> {
    fn drop(&mut self) {
        let mut state = self.slot.state.lock().unwrap();
        if std::thread::panicking() {
            drop(self.resident.take());
            state.status = None;
        } else {
            state.resident = self.resident.take();
        }
        state.idle_since = Instant::now();
        state.busy = false;
        self.slot.ready.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{mpsc, Arc};

    #[test]
    fn timed_out_front_waiter_does_not_block_later_requests() {
        let slot = Arc::new(Slot::<()>::new());
        let active = slot.acquire(Duration::ZERO).unwrap();
        let (expired, timeout) = mpsc::channel();
        let queued = slot.clone();
        let first = std::thread::spawn(move || {
            let failed = queued.acquire(Duration::from_millis(100)).is_err();
            expired.send(failed).unwrap();
        });
        assert!(timeout.recv_timeout(Duration::from_secs(3)).unwrap());
        let queued = slot.clone();
        let (finished, result) = mpsc::channel();
        let second = std::thread::spawn(move || {
            let _lease = queued.acquire(Duration::from_secs(3)).unwrap();
            finished.send(()).unwrap();
        });
        assert!(result.try_recv().is_err());
        drop(active);
        result.recv_timeout(Duration::from_secs(3)).unwrap();
        first.join().unwrap();
        second.join().unwrap();
    }
}
