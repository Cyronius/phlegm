"""Wait for a quiet host/device, then run one command under an advisory lock.

The timeout bounds waiting only. A launched command is always allowed to finish.
All cooperating NPU jobs must use the same lock path.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time


def log(message):
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    print(f'[{stamp}] {message}', flush=True)


def read_load():
    return float(Path('/proc/loadavg').read_text().split()[0])


def process_starttime(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # comm is parenthesized and may contain spaces or parentheses.
    fields = stat.rsplit(')', 1)[1].split()
    return None if fields[0] in ('Z', 'X') else int(fields[19])


def original_processes_running(watched, reader=process_starttime):
    return [pid for pid, start in watched.items()
            if start is not None and reader(pid) == start]


def read_pids():
    result = subprocess.run(['fuser', '/dev/accel/accel0'], capture_output=True,
                            text=True, timeout=2)
    if result.returncode not in (0, 1) or (result.returncode == 1 and result.stderr.strip()):
        raise RuntimeError(f'fuser failed: {result.stderr.strip()}')
    return result.stdout.split()


def idle_status(max_load, load_reader=read_load, pid_reader=read_pids):
    load = load_reader()
    if not math.isfinite(load) or load > max_load:
        return False, f'host load {load:.2f} exceeds {max_load:.2f}'
    pids = pid_reader()
    if pids:
        return False, 'NPU clients: ' + ' '.join(map(str, pids))
    return True, f'host load {load:.2f}; no NPU clients'


def wait_for_idle(lock, max_load, stable_seconds, timeout, *, probe=None,
                  clock=time.monotonic, sleep=time.sleep, report=log):
    probe = probe or (lambda: idle_status(max_load))
    deadline = clock() + timeout
    stable_since = None
    last_state = None
    while clock() < deadline:
        cycle_start = clock()
        idle, reason = probe()
        now = clock()
        state = 'idle' if idle else 'busy'
        if state != last_state:
            report(('Quiet: ' if idle else 'Waiting: ') + reason)
            last_state = state
        if not idle:
            stable_since = None
        elif stable_since is None:
            stable_since = now
        if idle and now - stable_since >= stable_seconds and now < deadline:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if last_state != 'locked':
                    report('Waiting: another cooperating job holds the lock')
                    last_state = 'locked'
            else:
                idle, reason = probe()  # Recheck while holding the lock.
                if idle and clock() < deadline:
                    report('Idle interval satisfied; command lock acquired')
                    return True
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                stable_since = None
                report('Readiness changed before launch: ' + reason)
        remaining = deadline - clock()
        if remaining > 0:
            delay = min(remaining, max(0.0, 5.0 - (clock() - cycle_start)))
            if stable_since is not None:
                until_stable = stable_seconds - (clock() - stable_since)
                if until_stable > 0:
                    delay = min(delay, until_stable)
            sleep(delay)
    report('Timed out waiting; no command was launched')
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', required=True, type=Path)
    parser.add_argument('--max-load', type=float, default=2.0)
    parser.add_argument('--stable-seconds', type=float, default=30.0)
    parser.add_argument('--timeout', type=float, default=3600.0)
    parser.add_argument('--wait-pid', type=int, action='append', default=[],
                        help='wait for this original process to end (repeatable; PID reuse ignored)')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('provide a command after --')
    if any(pid <= 0 for pid in args.wait_pid):
        parser.error('--wait-pid must be a positive integer')
    if any(not math.isfinite(v) or v < 0 for v in (args.max_load, args.stable_seconds, args.timeout)):
        parser.error('load, stable interval and waiting timeout must be finite and nonnegative')
    log(f'Waiting up to {args.timeout:g}s for load <= {args.max_load:g} and '
        f'{args.stable_seconds:g}s without NPU clients; lock {args.lock}')
    try:
        watched = {pid: process_starttime(pid) for pid in args.wait_pid}
        for pid, start in watched.items():
            log(f'Watching PID {pid}' if start is not None else f'PID {pid} already absent or exited')

        def probe():
            running = original_processes_running(watched)
            if running:
                return False, 'waiting for original PID(s): ' + ' '.join(map(str, running))
            return idle_status(args.max_load)

        with args.lock.open('a') as lock:
            if not wait_for_idle(lock, args.max_load, args.stable_seconds, args.timeout,
                                 probe=probe):
                return 124
            log('Starting: ' + shlex.join(command))
            child = subprocess.Popen(command, start_new_session=True)
            while True:
                try:
                    status = child.wait()  # No timeout and no signals sent to the child.
                    break
                except KeyboardInterrupt:
                    log('Command already launched; continuing to wait without interrupting it')
            log(f'Command exited with status {status}')
            return status if status >= 0 else 128 - status
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        log(f'Launcher error: {error}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
