"""Bounded child-process management.

Invariants (task #2):
  * argv only - never a shell command string, never shell=True.
  * every run has an overall timeout; a cancellation event is honoured.
  * stdout/stderr are drained concurrently into bounded tails (no deadlock,
    no unbounded memory); tail reads are limited.
  * we wait for the *actual* process exit; if a descendant keeps the pipe
    open we still return promptly (bounded join) with ``pipes_drained=False``.
  * we only ever terminate processes we own: on Windows the child is placed
    in a dedicated Job Object with KILL_ON_JOB_CLOSE (kills the whole tree,
    never by name); on POSIX a new session is created and the process group
    is signalled.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import BinaryIO, Mapping, Optional, Sequence

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]

    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _THREADENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]

    for _name, _args, _result in (
        ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ("CreateToolhelp32Snapshot", [wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
        ("Thread32First", [wintypes.HANDLE, ctypes.POINTER(_THREADENTRY32)], wintypes.BOOL),
        ("Thread32Next", [wintypes.HANDLE, ctypes.POINTER(_THREADENTRY32)], wintypes.BOOL),
        ("OpenThread", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        ("ResumeThread", [wintypes.HANDLE], wintypes.DWORD),
    ):
        _fn = getattr(_kernel32, _name)
        _fn.argtypes = _args
        _fn.restype = _result


class ProcessError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcResult:
    argv: tuple[str, ...]
    returncode: Optional[int]
    killed: bool
    timed_out: bool
    cancelled: bool
    pipes_drained: bool
    duration_s: float
    stdout_tail: bytes
    stderr_tail: bytes


def _make_job_object() -> Optional[int]:
    """Create a Job Object with kill-on-close; returns handle or None."""
    if not IS_WINDOWS:
        return None
    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ProcessError(f"CreateJobObjectW failed: {ctypes.get_last_error()}")
    try:
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise ProcessError(f"SetInformationJobObject failed: {ctypes.get_last_error()}")
    except BaseException:
        _kernel32.CloseHandle(job)
        raise
    return job


def _assign_job(job: Optional[int], proc: subprocess.Popen) -> None:
    if job is None:
        return
    ok = _kernel32.AssignProcessToJobObject(job, int(proc._handle))  # noqa: SLF001
    if not ok:
        raise ProcessError(f"AssignProcessToJobObject failed: {ctypes.get_last_error()}")


def _resume_process(proc: subprocess.Popen) -> None:
    """Resume the sole initial thread only after the suspended process is owned."""
    snapshot = _kernel32.CreateToolhelp32Snapshot(0x00000004, 0)  # TH32CS_SNAPTHREAD
    if snapshot == ctypes.c_void_p(-1).value:
        raise ProcessError(f"CreateToolhelp32Snapshot failed: {ctypes.get_last_error()}")
    try:
        entry = _THREADENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        found = _kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while found:
            if entry.th32OwnerProcessID == proc.pid:
                thread = _kernel32.OpenThread(0x0002, False, entry.th32ThreadID)
                if not thread:
                    raise ProcessError(f"OpenThread failed: {ctypes.get_last_error()}")
                try:
                    if _kernel32.ResumeThread(thread) == 0xFFFFFFFF:
                        raise ProcessError(f"ResumeThread failed: {ctypes.get_last_error()}")
                    return
                finally:
                    _kernel32.CloseHandle(thread)
            found = _kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        raise ProcessError("suspended process initial thread not found")
    finally:
        _kernel32.CloseHandle(snapshot)


def _terminate_job_tree(job: Optional[int], proc: subprocess.Popen, *, force: bool = False) -> None:
    """Terminate every process we own (job tree on Windows, process group on POSIX)."""
    if IS_WINDOWS:
        if job:
            if not _kernel32.TerminateJobObject(job, 1):
                raise ProcessError(f"TerminateJobObject failed: {ctypes.get_last_error()}")
        else:
            proc.kill()
    else:
        import signal

        try:
            # start_new_session makes pid the stable group id, even after leader exit.
            os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass


class _TailBuffer:
    """Bounded byte buffer keeping only the last ``limit`` bytes."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._buf = bytearray()
        self._lock = threading.Lock()

    def extend(self, data: bytes) -> None:
        with self._lock:
            self._buf += data
            if len(self._buf) > self.limit:
                del self._buf[: len(self._buf) - self.limit]

    def value(self) -> bytes:
        with self._lock:
            return bytes(self._buf)


def run_bounded(
    argv: Sequence[str],
    *,
    timeout_s: float,
    cancel_event: Optional[threading.Event] = None,
    tail_limit: int = 65536,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    poll_interval_s: float = 0.05,
    pipe_join_s: float = 2.0,
    hard_kill_after_s: float = 3.0,
    stdout_file: Optional[BinaryIO] = None,
) -> ProcResult:
    """Run ``argv`` with bounded lifetime, bounded pipe memory and tree kill.

    ``timeout_s`` bounds execution; cleanup has at most two
    ``hard_kill_after_s`` waits and one shared ``pipe_join_s`` drain budget.
    Descendants are reclaimed even if the direct child has already exited.
    ``stdout_file`` redirects large output to a caller-owned binary file;
    its handle remains caller-owned and stdout_tail is empty in that mode.
    """
    argv = tuple(str(a) for a in argv)
    if not argv:
        raise ProcessError("empty argv")
    for name, value in (("timeout_s", timeout_s), ("poll_interval_s", poll_interval_s),
                        ("hard_kill_after_s", hard_kill_after_s)):
        if not math.isfinite(value) or value <= 0:
            raise ProcessError(f"{name} must be finite and positive")
    if not math.isfinite(pipe_join_s) or pipe_join_s < 0:
        raise ProcessError("pipe_join_s must be finite and non-negative")
    if not isinstance(tail_limit, int) or tail_limit < 0:
        raise ProcessError("tail_limit must be a non-negative integer")
    popen_kwargs: dict = dict(
        stdout=stdout_file if stdout_file is not None else subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        cwd=cwd,
        env=dict(env) if env is not None else None,
        bufsize=0,
    )
    creationflags = 0
    start_new_session = False
    if IS_WINDOWS:
        creationflags = subprocess.CREATE_NO_WINDOW | 0x00000004  # CREATE_SUSPENDED
    else:
        start_new_session = True
    start = time.monotonic()
    out_buf, err_buf = _TailBuffer(tail_limit), _TailBuffer(tail_limit)

    def _pump(stream, buf: _TailBuffer) -> None:
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    timed_out = False
    cancelled = False
    killed = False
    returncode: Optional[int] = None
    deadline = start + timeout_s
    job = None
    proc = None
    readers: list[threading.Thread] = []
    try:
        job = _make_job_object()
        proc = subprocess.Popen(argv, creationflags=creationflags,
                                start_new_session=start_new_session, **popen_kwargs)
        _assign_job(job, proc)
        if IS_WINDOWS:
            _resume_process(proc)
        for stream, buf in ((proc.stdout, out_buf), (proc.stderr, err_buf)):
            if stream is None:
                continue
            reader = threading.Thread(target=_pump, args=(stream, buf), daemon=True)
            reader.start()
            readers.append(reader)
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(min(poll_interval_s, max(0, deadline - time.monotonic())))
        if cancelled or timed_out:
            killed = True
            _terminate_job_tree(job, proc)
            try:
                returncode = proc.wait(timeout=hard_kill_after_s)
            except subprocess.TimeoutExpired:
                pass  # The finally block escalates and observes the real exit.
    finally:
        try:
            if proc is not None:
                try:
                    _terminate_job_tree(job, proc, force=True)
                finally:
                    # Assignment/resume/reader failures must also reap a suspended child.
                    if proc.poll() is None:
                        proc.kill()
                    returncode = proc.wait(timeout=hard_kill_after_s)
        finally:
            if job is not None:
                _kernel32.CloseHandle(job)
            drain_deadline = time.monotonic() + pipe_join_s
            for reader in readers:
                reader.join(timeout=max(0, drain_deadline - time.monotonic()))
            if proc is not None:
                try:
                    for stream in (proc.stdout, proc.stderr):
                        # A live reader owns its stream; cross-thread close can hang.
                        if stream is not None and not any(t.is_alive() for t in readers):
                            stream.close()
                finally:
                    if IS_WINDOWS:
                        proc._handle.Close()  # noqa: SLF001

    pipes_drained = all(not reader.is_alive() for reader in readers)

    return ProcResult(
        argv=argv,
        returncode=returncode,
        killed=killed,
        timed_out=timed_out,
        cancelled=cancelled,
        pipes_drained=pipes_drained,
        duration_s=time.monotonic() - start,
        stdout_tail=out_buf.value(),
        stderr_tail=err_buf.value(),
    )
