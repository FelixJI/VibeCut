"""Dynamic FFmpeg graph construction; artifact lifecycle stays in render.static."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from vibecut.core.geometry import v360_args
from vibecut.core.sequence import SequencePlan


def _filter_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "'\\''") + "'"


def probe_backends(tools, directory, timeout_s, cancel_event):
    """Actual tiny H.264 decode/encode attempts, never encoder-list guessing."""
    from vibecut.render.static import ArtifactProcessError
    seed = directory / "capability.mp4"
    runs = []

    def attempt(name, args):
        r = tools.run(tools.ffmpeg_argv(*args), timeout_s=min(timeout_s, 15), cancel_event=cancel_event)
        ok = r.returncode == 0 and not (r.cancelled or r.timed_out or r.stderr_tail.strip())
        runs.append((name, f"{'available' if ok else 'unavailable'} seconds={r.duration_s:.4f} "
                     + r.stderr_tail.decode('utf-8', 'replace').strip()[-400:]))
        if r.cancelled:
            raise ArtifactProcessError("capability probe cancelled")
        return ok

    if not attempt("cpu-seed", ["-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=0.1", "-c:v", "libx264", "-y", str(seed)]):
        raise ArtifactProcessError("CPU capability seed failed")
    decode = attempt("nvdec-probe", ["-hwaccel", "cuda", "-i", str(seed), "-f", "null", "-"])
    encode = attempt("nvenc-probe", ["-i", str(seed), "-c:v", "h264_nvenc", "-f", "null", "-"])
    attempt("cpu-v360-probe", ["-i", str(seed), "-vf", "v360=e:flat:w=128:h=72:rorder=ypr:reset_rot=1", "-f", "null", "-"])
    return decode, encode, runs


def build_sequence_argv(sources, plan: SequencePlan, config, tmp_out: Path, directory: Path, *, nvdec=False, nvenc=False):
    from vibecut.render.static import _dec
    args = ["-xerror", "-err_detect", "explode", "-copyts"]
    graph = []
    for i, clip in enumerate(plan.clips):
        src = sources[clip.source_id]
        if nvdec:
            args += ["-hwaccel", "cuda"]
        args += ["-noautorotate", "-i", src.path]
        start, end = src.cut_window_absolute(clip.cut_in_rel, clip.cut_out_rel)
        frames = int(clip.duration_rel * plan.project_fps)
        commands = directory / f"camera-{i}.txt"
        with commands.open("w", encoding="utf-8", newline="\n") as fh:
            for n in range(frames):
                t = Fraction(n) / plan.project_fps
                view = clip.path.evaluate(clip.cut_in_rel + t)
                opts = v360_args(view, config.out_width, config.out_height)
                # reset_rot=1 makes each command an absolute orientation; rorder is initialization-only.
                fh.write(_dec(t) + " " + ", ".join(f"v360@cam{i} {k} {opts[k]}" for k in ("yaw", "pitch", "roll", "h_fov", "v_fov")) + ";\n")
        opts = v360_args(clip.path.evaluate(clip.cut_in_rel), config.out_width, config.out_height)
        projection = f"v360@cam{i}=e:flat:rorder=ypr:reset_rot=1:" + ":".join(f"{k}={opts[k]}" for k in ("w", "h", "yaw", "pitch", "roll", "h_fov", "v_fov"))
        graph.append(f"[{i}:{src.video.index}]trim=start={_dec(start)}:end={_dec(end)},"
                     f"setpts=PTS-({_dec(start)})/TB,fps={plan.project_fps}:start_time=0,trim=end_frame={frames},"
                     f"settb=1/{plan.project_fps.numerator},setpts=N*{plan.project_fps.denominator},"
                     f"sendcmd=f={_filter_path(commands)},{projection},format=yuv420p,setsar=1[v{i}]")
        if plan.audio:
            if src.audio:
                graph.append(f"[{i}:{src.audio.index}]atrim=start={_dec(start)}:end={_dec(end)},"
                             f"asetpts=PTS-({_dec(start)})/TB,aresample=48000:async=1:first_pts=0,"
                             f"aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=duration={_dec(clip.duration_rel)}[a{i}]")
            else:
                graph.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={_dec(clip.duration_rel)},asetpts=PTS-STARTPTS[a{i}]")
    inputs = "".join(f"[v{i}]" + (f"[a{i}]" if plan.audio else "") for i in range(len(plan.clips)))
    graph.append(f"{inputs}concat=n={len(plan.clips)}:v=1:a={int(plan.audio)}[v]" + ("[a]" if plan.audio else ""))
    args += ["-filter_complex", ";".join(graph), "-map", "[v]"]
    if plan.audio:
        args += ["-map", "[a]", "-c:a", "aac", "-b:a", config.audio_bitrate, "-ar", "48000", "-ac", "2"]
    args += ["-c:v", "h264_nvenc" if nvenc else "libx264"]
    if nvenc:
        args += ["-preset", "p4", "-rc", "vbr", "-cq", str(config.crf), "-b:v", "0"]
    else:
        args += ["-crf", str(config.crf), "-preset", config.preset,
                 "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709"]
    return tuple(args + ["-r", str(plan.project_fps), "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
                         "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv", "-map_metadata", "-1",
                         "-metadata:s:v:0", "rotate=0", "-movflags", "+faststart", "-f", "mp4", "-y", str(tmp_out)])


def render_sequence(plan, sources, tools, out_path, config=None, **kwargs):
    """Preview and final use this same path; only resolution/quality differ."""
    from vibecut.render.static import StaticRenderConfig, render_static
    if config is None:
        w, h = (1920, 1080) if plan.aspect == "16:9" else (1080, 1920)
        config = StaticRenderConfig(out_width=w, out_height=h, backend="auto")
    return render_static(plan, sources, tools, out_path, config, **kwargs)
