"""Cache key semantics: content-bound, config-bound, revision-agnostic."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.core.cache import Cache, normalized_key, render_key  # noqa: E402


def base_parts(**over):
    parts = dict(
        source_fingerprint="ab" * 32,
        source_projection="equirect",
        tool_identity="toolhash123",
        kind="proxy",
        color={"primaries": "bt709", "transfer": "bt709", "matrix": "bt709", "range": "tv", "bit_depth": 8},
    )
    parts.update(over)
    return parts


def test_same_content_different_filename_same_key():
    # identity is content, not path: filenames never appear in keys
    k1 = normalized_key(**base_parts())
    k2 = normalized_key(**base_parts())
    assert k1 == k2


def test_different_content_same_name_different_key():
    k1 = normalized_key(**base_parts(source_fingerprint="cd" * 32))
    assert k1 != normalized_key(**base_parts())


def test_tool_or_color_change_invalidates():
    base = normalized_key(**base_parts())
    assert normalized_key(**base_parts(tool_identity="other")) != base
    assert normalized_key(**base_parts(color={**base_parts()["color"], "bit_depth": 10})) != base


def test_render_key_config_sensitivity_and_revision_insensitivity():
    n = normalized_key(**base_parts())
    k1 = render_key(normalized=n, config_identity={"hfov": 90, "cut": "0/1"})
    assert render_key(normalized=n, config_identity={"hfov": 90, "cut": "0/1"}) == k1
    assert render_key(normalized=n, config_identity={"hfov": 60, "cut": "0/1"}) != k1
    assert render_key(normalized=n, config_identity={"hfov": 90, "cut": "1/2"}) != k1


def test_render_config_identity_is_revision_free():
    """The identity the renderer actually builds must contain no revision/lock bookkeeping."""
    import json as _json
    from fractions import Fraction

    from vibecut.core.contracts import CONTRACT_VERSION, CameraPath, CameraView, EditPlan
    from vibecut.render.static import StaticRenderConfig, render_config_identity

    def plan_with(**cam):
        cam_path = CameraPath(schema_version=CONTRACT_VERSION,
                              keys=((Fraction(0), CameraView(**cam)),), provenance="user")
        return EditPlan(CONTRACT_VERSION, "v1:ab", Fraction(0), Fraction(1), Fraction(30), cam_path, True)

    p1 = plan_with(yaw_deg=0.0, h_fov_deg=90.0)
    p2 = plan_with(yaw_deg=0.0, h_fov_deg=90.0)
    p2 = EditPlan.from_json(p2.to_json())  # same content
    ident1 = render_config_identity(StaticRenderConfig(), p1)
    ident2 = render_config_identity(StaticRenderConfig(), p2)
    assert ident1 == ident2
    blob = _json.dumps(ident1).lower()
    assert "revision" not in blob, "edit/project revision must never be key material"
    assert "manual" not in blob, "lock flags are semantics, not output identity"
    # a camera change DOES change identity
    p3 = plan_with(yaw_deg=45.0, h_fov_deg=90.0)
    assert render_config_identity(StaticRenderConfig(), p3) != ident1


def test_put_atomic_commit_and_validation_abort(tmp_path):
    cache = Cache(tmp_path / "cache")
    key = "k1" * 16
    payload_written = []

    def writer(target: Path) -> None:
        target.write_bytes(b"hello")

    def valid(target: Path) -> None:
        assert target.read_bytes() == b"hello"

    path = cache.put_atomic(key, meta={"a": 1}, payload_writer=writer, validator=valid)
    assert cache.has(key)
    meta, payload = cache.get(key)
    assert meta["a"] == 1 and meta["has_payload"] is True
    assert payload is not None and payload.is_file()
    assert payload.read_bytes() == b"hello"

    # validator failure -> nothing committed, no partial hit
    def bad_writer(target: Path) -> None:
        target.write_bytes(b"partial")

    def reject(target: Path) -> None:
        raise RuntimeError("invalid artifact")

    with pytest.raises(RuntimeError):
        cache.put_atomic("k2" * 16, meta={}, payload_writer=bad_writer, validator=reject)
    assert not cache.has("k2" * 16)
    # partial tmp never matches get
    assert cache.get("k2" * 16) is None


def test_meta_only_entries(tmp_path):
    cache = Cache(tmp_path / "cache")
    cache.put_atomic("m" * 32, meta={"kind": "render-manifest"})
    assert cache.has("m" * 32)
    hit = cache.get("m" * 32)
    assert hit is not None
    meta, payload = hit
    assert meta["kind"] == "render-manifest"
    assert payload is None


def test_corrupt_meta_treated_as_absent(tmp_path):
    cache = Cache(tmp_path / "cache")
    cache.put_atomic("c" * 32, meta={"ok": True}, payload_writer=lambda t: t.write_bytes(b"x"))
    meta_path = cache._meta_path("c" * 32)
    meta_path.write_text("{broken", encoding="utf-8")
    assert cache.get("c" * 32) is None


def test_put_atomic_validator_failure_preserves_previous_entry(tmp_path):
    """Failure injection: a rejected payload must not disturb a valid entry."""
    cache = Cache(tmp_path / "cache")
    key = "p" * 32
    cache.put_atomic(key, meta={"v": 1}, payload_writer=lambda t: t.write_bytes(b"good"))
    good_meta, good_payload = cache.get(key)

    def bad_writer(target: Path) -> None:
        target.write_bytes(b"bad-partial")

    with pytest.raises(RuntimeError):
        cache.put_atomic(key, meta={"v": 2}, payload_writer=bad_writer,
                         validator=lambda t: (_ for _ in ()).throw(RuntimeError("invalid")))
    meta2, payload2 = cache.get(key)
    assert meta2 == good_meta and payload2 == good_payload
    assert payload2.read_bytes() == b"good"
    # no temp litter that a future get could mistake for an entry
    assert not list(cache.root.glob(f"{key}.tmp-*"))


def test_put_atomic_meta_failure_yields_miss_or_old_entry(tmp_path, monkeypatch):
    """Failure injection: if the meta replace fails after the payload commit,
    the entry must read as a MISS (no meta) or stay at the previous valid
    entry - never a half-committed mismatch."""
    import vibecut.core.cache as cache_mod

    cache = Cache(tmp_path / "cache")
    key = "m" * 32
    cache.put_atomic(key, meta={"v": 1}, payload_writer=lambda t: t.write_bytes(b"old"))

    real_replace = os.replace
    failed = {"done": False}

    def flaky_replace(src, dst):
        if str(dst).endswith(".json") and not failed["done"]:  # meta destination only
            failed["done"] = True
            raise OSError("injected meta commit failure")
        return real_replace(src, dst)

    monkeypatch.setattr(cache_mod.os, "replace", flaky_replace)
    with pytest.raises(OSError):
        cache.put_atomic(key, meta={"v": 2}, payload_writer=lambda t: t.write_bytes(b"new"))
    monkeypatch.undo()
    hit = cache.get(key)
    assert hit is not None, "previous valid entry must survive a meta-commit failure"
    meta, payload = hit
    assert meta["v"] == 1  # old authoritative meta never half-overwritten
    assert payload.read_bytes() == b"old"


def test_reader_uses_one_manifest_snapshot(tmp_path, monkeypatch):
    cache = Cache(tmp_path / "cache")
    old = cache.put_atomic("same", {"v": 1}, lambda t: t.write_bytes(b"old"))
    old_meta = cache.get("same")[0]
    new = cache.put_atomic("same", {"v": 2}, lambda t: t.write_bytes(b"new"))
    new_meta = cache.get("same")[0]
    reads = iter([old_meta, new_meta])
    monkeypatch.setattr(cache, "_read_meta", lambda key: next(reads))
    meta, payload = cache.get("same")
    assert meta["v"] == 1 and payload == old and payload.read_bytes() == b"old"
    assert new.read_bytes() == b"new"


def test_tool_configuration_is_part_of_cache_identity(monkeypatch):
    from types import SimpleNamespace
    from vibecut.media.ffmpeg import FFmpegTools
    tools = FFmpegTools(Path("ffmpeg"), Path("ffprobe"))
    configuration = ["--enable-gpl"]
    def run(self, argv, **kwargs):
        text = "ffmpeg version same-release\nconfiguration: " + configuration[0]
        return SimpleNamespace(returncode=0, stdout_tail=text.encode())
    monkeypatch.setattr(FFmpegTools, "run", run)
    old = tools.identity()
    configuration[0] = "--enable-gpl --enable-version3"
    new = tools.identity()
    assert old != new and "configuration:" in new
    assert normalized_key(**base_parts(tool_identity=old)) != normalized_key(**base_parts(tool_identity=new))
