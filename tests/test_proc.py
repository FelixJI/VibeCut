"""Bounded process management with real small subprocess fixtures (AC5).

All fixtures are tiny python -c children; the suite verifies cancellation,
timeout, descendant-stdout-holding, non-zero exits and tree cleanup. It never
kills anything by process name and never touches unrelated processes.
"""

import ctypes
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.core.proc import run_bounded  # noqa: E402
from vibecut.core import proc as proc_module  # noqa: E402

PY = sys.executable
IS_WINDOWS = sys.platform == "win32"


def pid_alive(pid: int) -> bool:
    if IS_WINDOWS:
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        from ctypes import wintypes
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.GetExitCodeProcess.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        try:
            code = ctypes.c_uint32()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    status = Path(f"/proc/{pid}/stat")
    if status.exists() and status.read_text().rsplit(")", 1)[1].split()[0] == "Z":
        return False  # A reparented zombie has exited and cannot hold pipes or run code.
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_normal_run_captures_output():
    res = run_bounded([PY, "-c", "import sys; print('hello'); print('err', file=sys.stderr)"],
                      timeout_s=15.0)
    assert res.returncode == 0
    assert not res.killed and not res.timed_out and not res.cancelled
    assert res.pipes_drained
    assert b"hello" in res.stdout_tail and b"err" in res.stderr_tail


def test_nonzero_exit_reported():
    res = run_bounded([PY, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"], timeout_s=15.0)
    assert res.returncode == 3
    assert not res.killed
    assert b"boom" in res.stderr_tail


def test_timeout_kills_within_bound():
    t0 = time.monotonic()
    res = run_bounded([PY, "-c", "import time; time.sleep(30)"], timeout_s=0.8)
    elapsed = time.monotonic() - t0
    assert res.timed_out and res.killed
    assert res.returncode is not None  # we still observed the real exit
    assert elapsed < 10.0  # well inside the default total budget


def test_cancel_event_stops_promptly():
    cancel = threading.Event()
    holder = {"res": None}

    def run():
        holder["res"] = run_bounded([PY, "-c", "import time; time.sleep(30)"],
                                    timeout_s=30.0, cancel_event=cancel)

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.4)
    cancel.set()
    t.join(timeout=10.0)
    assert not t.is_alive(), "cancel must reclaim the runner within the 10s budget"
    res = holder["res"]
    assert res.cancelled and res.killed
    assert res.returncode is not None


def test_descendant_holding_stdout_does_not_block_return(tmp_path):
    """Direct child exits; a grandchild keeps the pipe open. run_bounded must
    return after the direct child's real exit, bounded, and the grandchild
    must be dead afterwards (Job Object kill-on-close)."""
    pidfile = tmp_path / "grandchild.pid"
    script = (
        "import subprocess,sys,pathlib\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],\n"
        "    stdout=sys.stdout, stderr=sys.stderr)\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid))\n"
        "print('parent-done', flush=True)\n"
    )
    t0 = time.monotonic()
    res = run_bounded([PY, "-c", script], timeout_s=15.0)
    elapsed = time.monotonic() - t0
    assert res.returncode == 0, res.stderr_tail
    assert elapsed < 10.0, f"must not wait for pipe-holding descendants (took {elapsed:.1f}s)"
    assert res.pipes_drained
    assert b"parent-done" in res.stdout_tail
    # wait for pidfile then verify the grandchild was killed via the job object
    deadline = time.monotonic() + 5.0
    while not pidfile.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    gpid = int(pidfile.read_text())
    deadline = time.monotonic() + 5.0
    while pid_alive(gpid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not pid_alive(gpid), "grandchild must be terminated with our job object"


def test_bounded_tail_keeps_memory_small():
    res = run_bounded([PY, "-c", "print('x' * 2_000_000)"], timeout_s=20.0, tail_limit=4096)
    assert res.returncode == 0
    assert len(res.stdout_tail) <= 4096
    assert res.stdout_tail.endswith(b"\n")


def test_argv_only_no_shell():
    # a shell metacharacter is passed as a literal argument and cannot execute
    res = run_bounded([PY, "-c", "import sys; sys.exit(0 if len(sys.argv)==2 else 5)",
                       "a; rm -rf / #"], timeout_s=15.0)
    assert res.returncode == 0


@pytest.mark.parametrize("cancel", [False, True])
def test_fast_spawn_tree_cleanup_preserves_unrelated_process(tmp_path, cancel):
    unrelated = subprocess.Popen([PY, "-c", "import time; time.sleep(60)"])
    pidfile = tmp_path / "owned.pid"
    script = (
        "import subprocess,sys,pathlib,time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(pidfile.with_suffix('.tmp'))!r}).write_text(str(p.pid))\n"
        "print('spawned', flush=True)\n"
        # Publish readiness only after both PID and stdout are completely written.
        f"pathlib.Path({str(pidfile.with_suffix('.tmp'))!r}).replace({str(pidfile)!r})\n"
        "time.sleep(60)\n"
    )
    event = threading.Event()
    watcher = None
    try:
        if cancel:
            def cancel_after_spawn():
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                event.set()
            watcher = threading.Thread(target=cancel_after_spawn)
            watcher.start()
        started = time.monotonic()
        res = run_bounded([PY, "-c", script], timeout_s=5 if cancel else 0.8,
                          cancel_event=event)
        assert time.monotonic() - started < 10
        assert res.killed and res.returncode is not None
        assert res.cancelled if cancel else res.timed_out
        assert b"spawned" in res.stdout_tail
        assert res.pipes_drained
        pid = int(pidfile.read_text())
        deadline = time.monotonic() + 3
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not pid_alive(pid), "owned descendant survived tree cleanup"
        assert unrelated.poll() is None, "cleanup touched an unrelated process"
    finally:
        if watcher is not None:
            watcher.join(timeout=5)
        unrelated.kill()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize("stage", ["assign", "resume", "reader", "monitor"])
def test_setup_and_runtime_failures_reap_child(monkeypatch, tmp_path, stage):
    if stage == "resume" and not IS_WINDOWS:
        pytest.skip("Windows suspended-process resume")
    created = []
    real_popen = subprocess.Popen

    def track_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        created.append(child)
        return child

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(proc_module.subprocess, "Popen", track_popen)
    if stage == "assign":
        monkeypatch.setattr(proc_module, "_assign_job", fail)
    elif stage == "resume":
        monkeypatch.setattr(proc_module, "_resume_process", fail)
    elif stage == "reader":
        monkeypatch.setattr(proc_module.threading.Thread, "start", fail)
    else:
        from types import SimpleNamespace
        monkeypatch.setattr(proc_module, "time", SimpleNamespace(monotonic=time.monotonic, sleep=fail))
    marker = tmp_path / "executed"
    script = f"from pathlib import Path; import time; Path({str(marker)!r}).touch(); time.sleep(60)"
    with pytest.raises(RuntimeError, match="injected failure"):
        run_bounded([PY, "-c", script], timeout_s=10)
    assert len(created) == 1
    assert created[0].returncode is not None
    assert not pid_alive(created[0].pid)
    if IS_WINDOWS and stage in ("assign", "resume"):
        assert not marker.exists(), "user code ran before ownership/resume completed"


def test_job_creation_failure_does_not_spawn(monkeypatch):
    def fail():
        raise proc_module.ProcessError("injected job failure")

    def unexpected_spawn(*args, **kwargs):
        pytest.fail("spawned before job creation succeeded")

    monkeypatch.setattr(proc_module, "_make_job_object", fail)
    monkeypatch.setattr(proc_module.subprocess, "Popen", unexpected_spawn)
    with pytest.raises(proc_module.ProcessError, match="injected job failure"):
        run_bounded([PY, "-c", "pass"], timeout_s=1)


def test_large_stdout_file_keeps_stderr_bounded_and_caller_handle_open(tmp_path):
    path = tmp_path / "controlled-output"
    script = "import sys; sys.stderr.buffer.write(b'e'*2000000); sys.stdout.buffer.write(b'x'*2000000)"
    with path.open("xb") as output:
        result = run_bounded([PY, "-c", script], timeout_s=10, tail_limit=4096, stdout_file=output)
        assert not output.closed
    assert result.returncode == 0 and result.pipes_drained
    assert result.stdout_tail == b"" and result.stderr_tail == b"e" * 4096
    assert path.read_bytes() == b"x" * 2000000


def test_runtime_failure_reaps_already_spawned_descendant(monkeypatch, tmp_path):
    pidfile = tmp_path / "spawned.pid"
    script = (
        "import subprocess,sys,pathlib,time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(pidfile.with_suffix('.tmp'))!r}).write_text(str(p.pid))\n"
        # Existence must mean the PID file has been written and closed.
        f"pathlib.Path({str(pidfile.with_suffix('.tmp'))!r}).replace({str(pidfile)!r})\n"
        "time.sleep(60)\n"
    )
    event = threading.Event()
    created = []
    real_popen = subprocess.Popen

    def track_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        created.append(child)
        return child

    def fail_after_spawn():
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pidfile.exists(), "fixture failed to spawn descendant"
        assert pid_alive(int(pidfile.read_text())), "descendant was not alive before failure"
        raise RuntimeError("injected failure after spawn")

    monkeypatch.setattr(proc_module.subprocess, "Popen", track_popen)
    monkeypatch.setattr(event, "is_set", fail_after_spawn)
    with pytest.raises(RuntimeError, match="injected failure after spawn"):
        run_bounded([PY, "-c", script], timeout_s=10, cancel_event=event)
    assert created[0].returncode is not None and not pid_alive(created[0].pid)
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 3
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not pid_alive(pid), "runtime failure left an owned descendant alive"


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows Job Object ownership")
def test_windows_process_is_suspended_until_job_assignment(monkeypatch, tmp_path):
    marker = tmp_path / "executed"
    real_assign = proc_module._assign_job
    observed = []

    def check_assign(job, child):
        time.sleep(0.15)
        assert not marker.exists(), "user code executed before job assignment"
        real_assign(job, child)
        k32 = proc_module._kernel32
        from ctypes import wintypes
        k32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        k32.IsProcessInJob.restype = wintypes.BOOL
        inside = wintypes.BOOL()
        assert k32.IsProcessInJob(int(child._handle), job, ctypes.byref(inside))
        assert inside.value
        observed.append(child.pid)

    monkeypatch.setattr(proc_module, "_assign_job", check_assign)
    script = f"from pathlib import Path; Path({str(marker)!r}).touch(); print('owned')"
    res = run_bounded([PY, "-c", script], timeout_s=5)
    assert observed and marker.exists() and res.stdout_tail.strip() == b"owned"


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows Job Object handle cleanup")
def test_job_configuration_failure_closes_handle(monkeypatch):
    closed = []
    real_close = proc_module._kernel32.CloseHandle

    def close(handle):
        closed.append(handle)
        return real_close(handle)

    monkeypatch.setattr(proc_module._kernel32, "CloseHandle", close)
    monkeypatch.setattr(proc_module._kernel32, "SetInformationJobObject", lambda *args: False)
    with pytest.raises(proc_module.ProcessError, match="SetInformationJobObject"):
        proc_module._make_job_object()
    assert len(closed) == 1


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows startup handle cleanup")
def test_popen_failure_closes_job(monkeypatch):
    closed = []
    real_close = proc_module._kernel32.CloseHandle

    def close(handle):
        closed.append(handle)
        return real_close(handle)

    monkeypatch.setattr(proc_module._kernel32, "CloseHandle", close)
    with pytest.raises(OSError):
        run_bounded(["Z:/vibecut-proc-nonexistent/executable.exe"], timeout_s=1)
    assert len(closed) == 1


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows native API failure cleanup")
@pytest.mark.parametrize("api, closed_count", [
    ("AssignProcessToJobObject", 1), ("CreateToolhelp32Snapshot", 1),
    ("OpenThread", 2), ("ResumeThread", 3), ("TerminateJobObject", 3),
])
def test_windows_api_failures_close_owned_handles(monkeypatch, tmp_path, api, closed_count):
    created, closed = [], []
    real_popen = subprocess.Popen
    real_close = proc_module._kernel32.CloseHandle

    def track_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        created.append(child)
        return child

    def close(handle):
        closed.append(handle)
        return real_close(handle)

    failure = {"CreateToolhelp32Snapshot": ctypes.c_void_p(-1).value,
               "ResumeThread": 0xFFFFFFFF}.get(api, 0)
    monkeypatch.setattr(proc_module.subprocess, "Popen", track_popen)
    monkeypatch.setattr(proc_module._kernel32, "CloseHandle", close)
    monkeypatch.setattr(proc_module._kernel32, api, lambda *args: failure)
    marker = tmp_path / "executed"
    script = f"from pathlib import Path; Path({str(marker)!r}).touch(); print('done')"
    with pytest.raises(proc_module.ProcessError, match=api):
        run_bounded([PY, "-c", script], timeout_s=5)
    assert len(created) == 1 and created[0].returncode is not None
    assert not pid_alive(created[0].pid)
    assert created[0]._handle.closed
    assert len(closed) == closed_count
    if api != "TerminateJobObject":
        assert not marker.exists(), "user code ran after a startup API failed"


@pytest.mark.parametrize("kwargs", [
    {"timeout_s": float("nan")}, {"timeout_s": float("inf")},
    {"timeout_s": 1, "poll_interval_s": 0},
    {"timeout_s": 1, "hard_kill_after_s": 0},
    {"timeout_s": 1, "pipe_join_s": -1},
    {"timeout_s": 1, "tail_limit": -1},
])
def test_invalid_bounds_are_rejected(kwargs):
    with pytest.raises(proc_module.ProcessError):
        run_bounded([PY, "-c", "pass"], **kwargs)


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX process-group signal escalation")
def test_posix_sigterm_ignoring_tree_is_killed(tmp_path):
    pidfile = tmp_path / "ignoring.pid"
    script = (
        "import signal,subprocess,sys,pathlib,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "p = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid))\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    started = time.monotonic()
    res = run_bounded([PY, "-c", script], timeout_s=0.8, hard_kill_after_s=0.3)
    assert res.timed_out and res.returncode == -9 and res.pipes_drained
    assert time.monotonic() - started < 5
    pid = int(pidfile.read_text())
    deadline = time.monotonic() + 3
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not pid_alive(pid)
