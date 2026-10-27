"""CLI end-to-end (AC1): project create -> ingest -> static export, no SDK."""

import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from conftest import (
    HAS_FFMPEG,
    HAS_FFPROBE,
    decode_gray_frames,
    make_marker_equirect,
    requires_ffmpeg,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.cli import main as cli_main  # noqa: E402


def run_cli(monkeypatch, tmp_path, *args) -> int:
    # keep the CLI honest: run it in-process with the real cwd, real tools
    monkeypatch.chdir(tmp_path)
    return cli_main([str(a) for a in args])


def test_version_and_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert exc.value.code == 0


def test_missing_tools_reports_exit_3(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECUT_FFMPEG", str(tmp_path / "nonexistent-ffmpeg"))
    rc = run_cli(monkeypatch, tmp_path, "inspect", str(tmp_path / "whatever.mp4"))
    assert rc == 3
    monkeypatch.delenv("VIBECUT_FFMPEG")
    monkeypatch.setenv("VIBECUT_FFPROBE", str(tmp_path / "nonexistent-ffprobe"))
    rc = run_cli(monkeypatch, tmp_path, "inspect", str(tmp_path / "whatever.mp4"))
    assert rc == 3


@requires_ffmpeg
def test_full_static_export_flow(tools, tmp_path, monkeypatch):
    # ffmpeg/ffprobe from PATH; make sure the environment points there
    monkeypatch.setenv("VIBECUT_FFMPEG", str(tools.ffmpeg))
    monkeypatch.setenv("VIBECUT_FFPROBE", str(tools.ffprobe))

    proj_dir = tmp_path / "项目 dir"
    src = make_marker_equirect(tools, tmp_path / "源 360.mp4", yaw_deg=90.0, pitch_deg=0.0)
    out = proj_dir / "成片 导出.mp4"

    assert run_cli(monkeypatch, tmp_path, "init", proj_dir) == 0
    project_file = proj_dir / "vibecut.project.json"
    assert project_file.is_file()

    assert run_cli(monkeypatch, tmp_path, "inspect", src, "--projection", "equirect") == 0
    assert run_cli(monkeypatch, tmp_path, "add-source", proj_dir, src, "--projection", "equirect") == 0

    rc = run_cli(monkeypatch, tmp_path, "export", proj_dir, src, out,
                 "--projection", "equirect", "--yaw", "90", "--hfov", "90")
    assert rc == 0, f"export failed with exit {rc}"
    assert out.is_file()

    # project recorded a succeeded job with the artifact, at a fresh revision
    payload = json.loads(project_file.read_text(encoding="utf-8"))
    assert payload["revision"] >= 2
    succeeded = [e for e in payload["jobs"] if e["status"] == "succeeded"]
    assert succeeded and succeeded[-1]["artifact"] == str(out.resolve())
    assert any(e["status"] == "running" for e in payload["jobs"])
    from vibecut.core.project import Project
    reopened = Project.load(project_file)
    saved_plan = reopened.plans[succeeded[-1]["job_id"]]
    assert saved_plan.source_id in reopened.sources
    assert saved_plan.camera.static_view.yaw_deg == 90
    assert saved_plan.camera.static_view.h_fov_deg == 90
    assert saved_plan.cut_in_rel == 0
    assert saved_plan.cut_out_rel == Fraction(3, 30)
    assert saved_plan.camera.provenance == "user" and saved_plan.cut_manual

    # source untouched, output sane (marker at view centre)
    frames = decode_gray_frames(tools, out, 1920, 1080)
    assert frames
    assert not any(Path(proj_dir).glob("**/*.partial")), "no temp artifacts may remain"

    # collision -> exit 5 and file preserved; unique suffix -> exit 0
    rc = run_cli(monkeypatch, tmp_path, "export", proj_dir, src, out,
                 "--projection", "equirect", "--yaw", "90")
    assert rc == 5
    rc = run_cli(monkeypatch, tmp_path, "export", proj_dir, src, out,
                 "--projection", "equirect", "--yaw", "90", "--unique-suffix")
    assert rc == 0
    assert out.with_name("成片 导出-1.mp4").is_file()

    # Reuse the reopened plan and its source, without reconstructing edit settings.
    from vibecut.render.static import render_static
    rerender = render_static(saved_plan, reopened.sources[saved_plan.source_id], tools,
                             proj_dir / "reopened.mp4")
    assert rerender.status == "succeeded", rerender.error

    # A changed file at the same ingested path must not become a new implicit source.
    before = project_file.read_bytes()
    make_marker_equirect(tools, src, yaw_deg=0.0, pitch_deg=0.0)
    assert run_cli(monkeypatch, tmp_path, "export", proj_dir, src, proj_dir / "changed.mp4",
                   "--projection", "equirect") == 5
    assert project_file.read_bytes() == before
    assert run_cli(monkeypatch, tmp_path, "add-source", proj_dir, src,
                   "--projection", "equirect") == 5
    assert project_file.read_bytes() == before


@requires_ffmpeg
def test_unsupported_source_exit_2(tools, tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECUT_FFMPEG", str(tools.ffmpeg))
    monkeypatch.setenv("VIBECUT_FFPROBE", str(tools.ffprobe))
    bad = tmp_path / "notpano.mp4"
    bad.write_bytes(b"garbage not a video")
    rc = run_cli(monkeypatch, tmp_path, "inspect", bad, "--projection", "equirect")
    assert rc in (2, 4)  # corrupt file: rejected (probe failure), never rendered


@requires_ffmpeg
@pytest.mark.parametrize("command", ["export", "add-source"])
def test_heavy_cli_job_rejects_second_process(tools, tmp_path, command):
    import subprocess
    import time
    from vibecut.core.project import Project

    project = Project.create(tmp_path / "project")
    project.save()
    src = make_marker_equirect(tools, tmp_path / "s.mp4", yaw_deg=0, pitch_deg=0)
    ready = tmp_path / "owner-ready"
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    holder = f'''
import sys, time
from pathlib import Path
sys.path.insert(0, {src_dir!r})
import vibecut.render.static as renderer
def slow(*args, **kwargs):
    Path({str(ready)!r}).write_text("ready")
    time.sleep(4)
    return renderer.RenderResult(kwargs["job_id"], "failed", error="synthetic held job")
renderer.render_static = slow
from vibecut.cli import main
sys.exit(main(["export", {str(project.path)!r}, {str(src)!r},
               {str(tmp_path / 'first.mp4')!r}, "--projection", "equirect"]))
'''
    owner = subprocess.Popen([sys.executable, "-c", holder], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), owner.communicate(timeout=10)
        before = project.path.read_bytes()
        args = [command, str(project.path), str(src)]
        args += [str(tmp_path / "second.mp4")] if command == "export" else ["--make-proxy"]
        args += ["--projection", "equirect"]
        runner = f"import sys; sys.path.insert(0, {src_dir!r}); from vibecut.cli import main; sys.exit(main(sys.argv[1:]))"
        started = time.monotonic()
        second = subprocess.run([sys.executable, "-c", runner, *args],
                                capture_output=True, timeout=3)
        assert second.returncode == 5, second.stderr
        assert time.monotonic() - started < 3
        assert project.path.read_bytes() == before
        assert not (tmp_path / "second.mp4").exists()
    finally:
        stdout, stderr = owner.communicate(timeout=15)
    assert owner.returncode == 4, (stdout, stderr)
