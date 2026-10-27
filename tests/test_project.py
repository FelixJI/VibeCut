"""Project persistence: atomicity, revision guards, cross-process locking."""

import json
import subprocess
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.core.contracts import (  # noqa: E402
    CONTRACT_VERSION,
    CameraPath,
    CameraView,
    ColorContract,
    EditPlan,
    NormalizedAsset,
    TimeMap,
)
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
    """A writer that dies between temp-write and replace must not corrupt.

    The child patches the real ``os.replace`` on the ``os`` module (the
    attribute that ``write_json_atomic`` resolves at call time), verifies the
    temp file holds the *complete* new payload, then hard-exits 9. Any other
    exit path (unhandled exception, save returning normally) fails this test
    with distinct evidence instead of "any nonzero exit counts as success".
    """
    p = make_project(tmp_path)
    before = p.path.read_text(encoding="utf-8")
    script = textwrap.dedent(
        f"""
        import json, os, sys
        from pathlib import Path
        sys.path.insert(0, r"{Path(__file__).resolve().parents[1] / 'src'}")
        from vibecut.core.project import Project
        proj = Project.load(r"{p.path}")

        def boom(src, dst):
            # crash exactly between the complete tmp write and the replace
            data = Path(src).read_text(encoding="utf-8")
            payload = json.loads(data)  # must parse end-to-end: no torn write
            assert payload["revision"] == 2, payload["revision"]
            assert payload["project_schema_version"] == 1
            assert payload["contract_schema_version"] == 2
            assert Path(src).parent == Path(dst).parent  # same-directory tmp
            print("TEMP_VERIFIED revision=%d bytes=%d" % (payload["revision"], len(data)), flush=True)
            os._exit(9)

        os.replace = boom  # patch the real os.replace resolved inside write_json_atomic
        returned = proj.save()
        print("UNREACHED save returned %r" % (returned,), flush=True)
        os._exit(3)  # boom never fired: fail loudly, not as a fake success
        """
    )
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=30)
    assert res.returncode == 9  # died exactly inside os.replace, nowhere else
    assert res.stderr == b""  # no unhandled exception was masked
    assert b"TEMP_VERIFIED revision=2" in res.stdout  # tmp was complete pre-replace
    assert b"UNREACHED" not in res.stdout  # save() never completed
    # the on-disk file is still the last complete version and fully loadable
    assert p.path.read_text(encoding="utf-8") == before
    loaded = Project.load(p.path)
    assert loaded.revision == 1
    # exactly one complete-but-uncommitted temp is left behind
    leftovers = sorted(p.path.parent.glob(p.path.name + ".tmp-*"))
    assert len(leftovers) == 1
    crashed = json.loads(leftovers[0].read_text(encoding="utf-8"))
    assert crashed["revision"] == 2
    assert crashed["project_schema_version"] == 1


def test_load_rejects_unknown_root_contract_schema_version(tmp_path):
    """contract_schema_version=999 + future fields must not load (and so can
    never be silently re-saved as version 2 with the future fields dropped)."""
    p = make_project(tmp_path)
    raw = json.loads(p.path.read_text(encoding="utf-8"))
    raw["contract_schema_version"] = 999
    raw["future_field"] = {"unknown": True}
    mutated = json.dumps(raw, indent=1, sort_keys=True)
    p.path.write_text(mutated, encoding="utf-8")
    with pytest.raises(ProjectFormatError):
        Project.load(p.path)
    assert p.path.read_text(encoding="utf-8") == mutated  # rejected load leaves the file untouched


def test_load_rejects_missing_root_contract_schema_version(tmp_path):
    p = make_project(tmp_path)
    raw = json.loads(p.path.read_text(encoding="utf-8"))
    del raw["contract_schema_version"]
    p.path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ProjectFormatError):
        Project.load(p.path)


def _make_plan(camera_version=CONTRACT_VERSION) -> EditPlan:
    cam = CameraPath(
        schema_version=camera_version,
        keys=((Fraction(0), CameraView()),),
        provenance="user",
    )
    return EditPlan(CONTRACT_VERSION, "v1:ab", Fraction(0), Fraction(1), Fraction(30), cam, True)


def _asset_json(schema_version=CONTRACT_VERSION) -> dict:
    return {
        "schema_version": schema_version,
        "asset_id": "v1:xy",
        "kind": "proxy",
        "source_id": "v1:ab",
        "path": "p.mp4",
        "time_map": {"asset_start_source_rel": "0/1", "fps": "30/1"},
        "color": {"primaries": "bt709", "transfer": "bt709", "matrix": "bt709", "range": "tv", "bit_depth": 8},
        "stabilization_provenance": "none",
    }


def _source_json(schema_version=CONTRACT_VERSION) -> dict:
    fp = "ab" * 32
    return {
        "schema_version": schema_version,
        "source_id": f"v1:{fp[:16]}",
        "path": "C:/x/y.mp4",
        "content_fingerprint": fp,
        "size_bytes": 123,
        "container": "mp4",
        "projection": "equirect",
        "stereo": "mono",
        "color": {"primaries": "bt709", "transfer": "bt709", "matrix": "bt709", "range": "tv", "bit_depth": 8},
        "video": {"index": 0, "codec": "h264", "width": 1920, "height": 960, "start_pts": 0,
                  "time_base": "1/15360", "fps": "30/1", "r_frame_rate": "30/1", "nb_frames": 90},
        "audio": None,
        "vfr": False,
        "declared": True,
        "stabilization_provenance": "none",
        "timeline_origin": "0",
    }


def test_load_rejects_nested_unknown_contract_versions(tmp_path):
    """Unknown versions nested inside sources/assets/plans(camera)/jobs are
    rejected at load, and the file is left exactly as it was."""
    p = make_project(tmp_path)
    p.plans["cut1"] = _make_plan()
    p.assets.append(NormalizedAsset(
        CONTRACT_VERSION, "v1:xy", "proxy", "v1:ab", "p.mp4",
        TimeMap(Fraction(0), Fraction(30)),
        ColorContract("bt709", "bt709", "bt709", "tv", 8),
    ))
    p.save()
    good = json.loads(p.path.read_text(encoding="utf-8"))
    good["sources"] = [_source_json()]
    good["jobs"] = [{"schema_version": CONTRACT_VERSION, "job_id": "j1", "project_revision": p.revision,
                     "status": "succeeded", "at": "2026-01-01T00:00:00+00:00", "artifact": "out.mp4", "error": None}]
    baseline = json.dumps(good, sort_keys=True)
    p.path.write_text(baseline, encoding="utf-8")
    loaded = Project.load(p.path)  # baseline with all current versions loads fine
    assert loaded.revision == p.revision

    def _reject(mutate) -> None:
        raw = json.loads(baseline)
        mutate(raw)
        mutated = json.dumps(raw, sort_keys=True)
        p.path.write_text(mutated, encoding="utf-8")
        with pytest.raises(ProjectFormatError):
            Project.load(p.path)
        assert p.path.read_text(encoding="utf-8") == mutated  # file untouched by the rejection

    _reject(lambda raw: raw["sources"].__setitem__(0, _source_json(schema_version=999)))
    _reject(lambda raw: raw["assets"].__setitem__(0, _asset_json(schema_version=999)))
    _reject(lambda raw: raw["plans"]["cut1"].__setitem__("camera", {**raw["plans"]["cut1"]["camera"], "schema_version": 999}))
    _reject(lambda raw: raw["plans"]["cut1"].__setitem__("schema_version", 999))
    _reject(lambda raw: raw["jobs"].__setitem__(0, {**raw["jobs"][0], "schema_version": 999}))


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
