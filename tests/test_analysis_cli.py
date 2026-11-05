"""CLI analyze command: minimal surface, exit codes, noclobber --out, offline."""

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from conftest import make_luma_step_video, requires_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis_offline_server import OfflineLM, observe_payload

from vibecut.cli import main as cli_main


def run_cli(monkeypatch, tmp_path, *args) -> int:
    monkeypatch.chdir(tmp_path)
    return cli_main([str(a) for a in args])


def test_analyze_help_lists_options():
    with pytest.raises(SystemExit) as exc:
        cli_main(["analyze", "--help"])
    assert exc.value.code == 0


def test_analyze_rejects_public_endpoint(monkeypatch, tmp_path):
    rc = run_cli(monkeypatch, tmp_path, "analyze", tmp_path / "nope", tmp_path / "x.mp4",
                 "--focus", "人物优先", "--endpoint", "http://8.8.8.8:1234")
    assert rc == 2  # usage: non-loopback endpoint rejected before any connection


def test_analyze_rejects_bad_focus(monkeypatch, tmp_path):
    rc = run_cli(monkeypatch, tmp_path, "analyze", tmp_path / "nope", tmp_path / "x.mp4",
                 "--focus", "   ")
    assert rc == 2


@requires_ffmpeg
def test_analyze_ai_unavailable_exit(tools, tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECUT_FFMPEG", str(tools.ffmpeg))
    monkeypatch.setenv("VIBECUT_FFPROBE", str(tools.ffprobe))
    src = make_luma_step_video(tools, tmp_path / "pano.mp4", width=640, height=320,
                               fps=Fraction(30), frames=45, step_frame=20)
    lm = OfflineLM(model_state="not-loaded").start()
    try:
        rc = run_cli(monkeypatch, tmp_path, "init", tmp_path / "proj")
        assert rc == 0
        rc = run_cli(monkeypatch, tmp_path, "analyze", tmp_path / "proj", src,
                     "--focus", "人物优先", "--projection", "equirect",
                     "--endpoint", lm.base_url, "--samples", "2")
        assert rc == 4
        assert len(lm.chat_calls) == 0
    finally:
        lm.stop()


@requires_ffmpeg
def test_analyze_offline_end_to_end_and_noclobber(tools, tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECUT_FFMPEG", str(tools.ffmpeg))
    monkeypatch.setenv("VIBECUT_FFPROBE", str(tools.ffprobe))
    src = make_luma_step_video(tools, tmp_path / "pano.mp4", width=640, height=320,
                               fps=Fraction(30), frames=60, step_frame=30)
    lm = OfflineLM().start()
    try:
        assert run_cli(monkeypatch, tmp_path, "init", tmp_path / "proj") == 0
        from analysis_offline_server import observe_payload

        views = _views()
        chunks = [views[j:j + 8] for j in range(0, len(views), 8)]
        for i in range(2):
            for ch in chunks:
                lm.chat_script.append({"content": observe_payload([f"s{i:02d}." + v.view_id for v in ch])})
        lm.chat_script.append({"content": {"selections": []}})
        out = tmp_path / "analysis.json"
        rc = run_cli(monkeypatch, tmp_path, "analyze", tmp_path / "proj", src,
                     "--focus", "人物优先，剪出2秒", "--projection", "equirect",
                     "--endpoint", lm.base_url, "--samples", "2", "--out", out)
        assert rc == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["status"] in ("succeeded", "partial")
        assert payload["plan"]["status"] == "empty"
        assert payload["metering"]["request_count"] == 5  # 4 observe chunks + 1 select
        assert payload["candidates"]
        # never overwrite an existing result file
        rc2 = run_cli(monkeypatch, tmp_path, "analyze", tmp_path / "proj", src,
                      "--focus", "人物优先", "--projection", "equirect",
                      "--endpoint", lm.base_url, "--samples", "2", "--out", out)
        assert rc2 == 5
        # analysis wrote no project revision: the file is exactly as init left it
        project_path = tmp_path / "proj" / "vibecut.project.json"
        project = json.loads(project_path.read_text(encoding="utf-8"))
        assert project["revision"] == 1  # init saves revision 1; analyze must not bump it
        assert project["plans"] == {} and project["jobs"] == []
    finally:
        lm.stop()


def _views():
    from vibecut.analysis.views import default_view_set

    return default_view_set()



def test_noclobber_link_failure_never_exposes_partial_json(tmp_path, monkeypatch):
    import os
    from vibecut.cli import _write_result_noclobber
    out = tmp_path / "result.json"
    def fail(*args):
        raise PermissionError("filesystem denies atomic publication")
    monkeypatch.setattr(os, "link", fail)
    with pytest.raises(PermissionError):
        _write_result_noclobber(out, {"complete": True})
    assert not out.exists()
    assert not list(tmp_path.iterdir())


def test_analysis_holds_project_lock_and_rejects_stale_delivery(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from vibecut import cli
    from vibecut.core.project import Project, ProjectLockedError
    import vibecut.analysis.service as service
    project = Project.create(tmp_path / "project")
    project.save()
    out = tmp_path / "late.json"
    monkeypatch.setattr(cli, "_resolve_tools", lambda: object())
    monkeypatch.setattr(cli, "_probe_or_exit", lambda *args: (SimpleNamespace(source_id="test", path="source.mp4"), 0))
    def late(*args, **kwargs):
        other = Project.load(project.path)
        with pytest.raises(ProjectLockedError):
            with other.exclusive_lock(timeout_s=0):
                pass
        payload = json.loads(project.path.read_text(encoding="utf-8"))
        payload["revision"] += 1
        project.path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(to_json=lambda: {"status": "succeeded"})
    monkeypatch.setattr(service, "analyze", late)
    assert cli_main(["analyze", str(project.path), "source.mp4", "--focus", "人物", "--out", str(out)]) == 5
    assert not out.exists()
