"""Project persistence: atomicity, revision guards, cross-process locking."""

import json
import subprocess
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.core.project import (  # noqa: E402
    Project,
    ProjectFormatError,
    ProjectLockedError,
    StaleRevisionError,
)


def make_project(tmp_path: Path) -> Project:
    p = Project.create(tmp_path / "proj", fps=Fraction(30000, 1001))
    p.save()
    return p


def test_create_load_roundtrip(tmp_path):
    p = make_project(tmp_path)
    loaded = Project.load(p.path)
    assert loaded.project_id == p.project_id
    assert loaded.revision == 1  # creation save persists revision 1
    assert loaded.fps == Fraction(30000, 1001)


def test_save_increments_revision_and_is_atomic(tmp_path):
    p = make_project(tmp_path)
    for expected in (2, 3, 4):
        rev = p.save()
        assert rev == expected
    assert Project.load(p.path).revision == 4


def test_interrupted_save_leaves_last_complete_version(tmp_path):
    """A writer that dies between temp-write and replace must not corrupt."""
    p = make_project(tmp_path)
    before = p.path.read_text(encoding="utf-8")
    script = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, r"{Path(__file__).resolve().parents[1] / 'src'}")
        import vibecut.core.contracts as c
        real_replace = os.replace
        def boom(a, b):
            os._exit(9)  # crash exactly between tmp write and replace
        c.os.replace = boom
        from vibecut.core.project import Project
        p = Project.load(r"{p.path}")
        try:
            p.save()
        finally:
            os._exit(9)
        """
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=30)
    assert res.returncode != 0
    # the on-disk file is still the last complete version
    loaded = Project.load(p.path)
    assert loaded.revision == 1
    assert p.path.read_text(encoding="utf-8") == before
    # leftover tmp files never confuse the loader
    leftovers = list(p.path.parent.glob(p.path.name + ".tmp-*"))
    assert loaded.revision == 1
    assert isinstance(leftovers, list)


def test_corrupt_file_is_reported_not_crashing(tmp_path):
    p = make_project(tmp_path)
    p.path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ProjectFormatError):
        Project.load(p.path)


def test_stale_writer_rejected(tmp_path):
    """Two writers load revision 1; the second saver wins, the first is rejected."""
    p = make_project(tmp_path)
    a = Project.load(p.path)
    b = Project.load(p.path)
    b.save()
    with pytest.raises(StaleRevisionError):
        a.save()
    assert Project.load(p.path).revision == 2


def test_cross_process_lock_blocks_second_writer(tmp_path):
    p = make_project(tmp_path)
    holder = textwrap.dedent(
        f"""
        import os, sys, time
        sys.path.insert(0, r"{Path(__file__).resolve().parents[1] / 'src'}")
        from vibecut.core.project import Project
        pr = Project.load(r"{p.path}")
        with pr.exclusive_lock(timeout_s=10):
            print("locked", flush=True)
            time.sleep(3.0)
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", holder], stdout=subprocess.PIPE)
    try:
        line = proc.stdout.readline()  # wait until the child holds the lock
        assert b"locked" in line
        second = Project.load(p.path)
        t0 = __import__("time").monotonic()
        with pytest.raises(ProjectLockedError):
            second.exclusive_lock(timeout_s=1.0).__enter__()
        assert __import__("time").monotonic() - t0 < 5.0
    finally:
        proc.wait(timeout=30)
    # after release, locking works again (serialised, not deadlocked)
    with Project.load(p.path).exclusive_lock(timeout_s=5.0):
        pass


def test_lockfile_created_next_to_project(tmp_path):
    p = make_project(tmp_path)
    with p.exclusive_lock(timeout_s=2.0):
        assert p.lock_path.exists()


def test_job_owner_can_save_without_relocking(tmp_path):
    p = make_project(tmp_path)
    with p.exclusive_lock(timeout_s=0):
        p.check_revision()
        assert p.save() == 2
        with p.exclusive_lock(timeout_s=0):
            assert p.save() == 3
    assert Project.load(p.path).revision == 3


def test_failed_save_keeps_in_memory_and_disk_revision(tmp_path, monkeypatch):
    import vibecut.core.project as module
    p = make_project(tmp_path)
    before = p.path.read_bytes()
    def fail(*args):
        raise OSError("injected disk full")
    monkeypatch.setattr(module, "write_json_atomic", fail)
    with pytest.raises(OSError):
        p.save()
    assert p.revision == 1 and p.path.read_bytes() == before


def test_save_does_not_replace_unreadable_current_revision(tmp_path):
    p = make_project(tmp_path)
    p.path.write_bytes(b"broken externally")
    with pytest.raises(ProjectFormatError):
        p.save()
    assert p.path.read_bytes() == b"broken externally" and p.revision == 1
