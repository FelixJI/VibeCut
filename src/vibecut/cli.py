"""VibeCut CLI.

Commands:
  init        create a project directory
  inspect     probe a source and print its MediaSource contract
  add-source  ingest a probed source into a project (fingerprint baseline)
  export      run the static fixed-view export and record job events

Exit codes:
  0 success | 2 usage/unsupported source | 3 ffmpeg/ffprobe missing
  4 render/validation failure | 5 conflict (output exists / source changed / lock)

Heavy layers (media, render) are imported inside command handlers so the CLI
core stays dependency-free.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from fractions import Fraction
from pathlib import Path
from typing import Optional, Sequence

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_TOOLS = 3
EXIT_RENDER = 4
EXIT_CONFLICT = 5


def _parse_time(text: str) -> Fraction:
    return Fraction(text)  # accepts "1.5", "3/2", "30000/1001"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vibecut", description="VibeCut static panorama->flat pipeline")
    parser.add_argument("--version", action="version", version=f"vibecut {__import__('vibecut').__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create a project")
    p_init.add_argument("directory", type=Path)
    p_init.add_argument("--fps", type=Fraction, default=Fraction(30000, 1001))

    p_inspect = sub.add_parser("inspect", help="probe a source file")
    p_inspect.add_argument("input", type=Path)
    p_inspect.add_argument("--projection", choices=["equirect"], default=None,
                           help="declare the projection if the file carries no spherical metadata")
    p_inspect.add_argument("--json", action="store_true", help="print the raw MediaSource JSON")

    p_add = sub.add_parser("add-source", help="ingest a probed source into a project")
    p_add.add_argument("project", type=Path, help="path to vibecut.project.json (or its directory)")
    p_add.add_argument("input", type=Path)
    p_add.add_argument("--projection", choices=["equirect"], default=None)
    p_add.add_argument("--make-proxy", action="store_true", help="also build the cached proxy")

    p_export = sub.add_parser("export", help="static fixed-view export")
    p_export.add_argument("project", type=Path)
    p_export.add_argument("input", type=Path, help="source file (must be ingested or will be probed on the fly)")
    p_export.add_argument("output", type=Path, help="target .mp4 (never overwritten)")
    p_export.add_argument("--in", dest="cut_in", type=_parse_time, default=Fraction(0), metavar="SEC",
                          help="source-relative cut-in seconds (0 = first displayed frame)")
    p_export.add_argument("--out", dest="cut_out", type=_parse_time, default=None, metavar="SEC",
                          help="source-relative cut-out seconds (default: whole source)")
    p_export.add_argument("--yaw", type=float, default=0.0, help="canonical yaw degrees (0=equirect centre, +right)")
    p_export.add_argument("--pitch", type=float, default=0.0, help="canonical pitch degrees (+up)")
    p_export.add_argument("--roll", type=float, default=0.0, help="canonical roll degrees (+clockwise)")
    p_export.add_argument("--hfov", type=float, default=90.0, help="horizontal FOV degrees")
    p_export.add_argument("--fps", type=Fraction, default=None, help="project fps override (default: project fps)")
    p_export.add_argument("--no-audio", action="store_true")
    p_export.add_argument("--projection", choices=["equirect"], default=None)
    p_export.add_argument("--unique-suffix", action="store_true",
                          help="on output collision create output-N.mp4 instead of failing")
    p_export.add_argument("--timeout", type=float, default=600.0, help="render timeout seconds")
    return parser


def _load_project(project_path: Path):
    from vibecut.core.project import PROJECT_FILENAME, Project, ProjectError

    path = Path(project_path)
    if path.is_dir():
        path = path / PROJECT_FILENAME
    try:
        return Project.load(path)
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def _resolve_tools():
    from vibecut.media.ffmpeg import FFmpegMissingError, FFmpegTools

    try:
        return FFmpegTools.discover()
    except FFmpegMissingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def _cmd_init(args: argparse.Namespace) -> int:
    from vibecut.core.project import Project, ProjectError

    try:
        project = Project.create(args.directory, fps=args.fps)
        revision = project.save()
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFLICT
    print(f"created {project.path} (project_id={project.project_id[:8]}, revision={revision})")
    return EXIT_OK


def _probe_or_exit(input_path: Path, tools, projection: Optional[str]):
    """Returns (source, None) or (None, exit_code) - never raises SystemExit."""
    from vibecut.media.probe import ProbeError, UnsupportedSourceError, probe_source

    try:
        return probe_source(input_path, tools, declared_projection=projection), None
    except UnsupportedSourceError as exc:
        print(f"unsupported source: {exc}", file=sys.stderr)
        return None, EXIT_USAGE
    except ProbeError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return None, EXIT_RENDER


def _cmd_inspect(args: argparse.Namespace) -> int:
    tools = _resolve_tools()
    if tools is None:
        return EXIT_TOOLS
    source, code = _probe_or_exit(args.input, tools, args.projection)
    if source is None:
        return code
    payload = source.to_json()
    print(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) if args.json else _summarize(payload))
    return EXIT_OK


def _summarize(payload: dict) -> str:
    v = payload["video"]
    a = payload.get("audio")
    color = payload["color"]
    audio_txt = f", audio {a['codec']} {a['sample_rate']}Hz start_pts={a['start_pts']}" if a else ", no audio"
    return (
        f"{payload['source_id']} {Path(payload['path']).name}\n"
        f"  {payload['projection']} {payload['stereo']} {payload['container']}, {v['codec']} "
        f"{v['width']}x{v['height']} @ {v['fps']} fps (tbn {v['time_base']}, start_pts {v['start_pts']}){audio_txt}\n"
        f"  color: {color['primaries']}/{color['transfer']}/{color['matrix']} {color['range']} {color['bit_depth']}bit, "
        f"fingerprint {payload['content_fingerprint'][:16]}"
    )


def _cmd_add_source(args: argparse.Namespace) -> int:
    from vibecut.core.cache import Cache
    from vibecut.media.probe import ArtifactValidationError, ProbeError
    from vibecut.media.proxy import ensure_proxy
    from vibecut.render.static import ExportError, SourceChangedError

    tools = _resolve_tools()
    if tools is None:
        return EXIT_TOOLS
    project = _load_project(args.project)
    with project.exclusive_lock(timeout_s=0):
        project.check_revision()
        source, code = _probe_or_exit(args.input, tools, args.projection)
        if source is None:
            return code
        project.add_source(source)
        if args.make_proxy:
            cache = Cache(project.path.parent / ".vibecut-cache")
            try:
                asset = ensure_proxy(source, tools, cache)
            except SourceChangedError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return EXIT_CONFLICT
            except (ExportError, ArtifactValidationError, ProbeError, OSError) as exc:
                print(f"proxy failed: {exc}", file=sys.stderr)
                return EXIT_RENDER
            project.assets = [a for a in project.assets if a.source_id != source.source_id] + [asset]
        revision = project.save()
    print(f"added {source.source_id} (project revision {revision})")
    return EXIT_OK


def _cmd_export(args: argparse.Namespace) -> int:
    tools = _resolve_tools()
    if tools is None:
        return EXIT_TOOLS
    project = _load_project(args.project)
    with project.exclusive_lock(timeout_s=0):
        project.check_revision()
        return _export_locked(args, project, tools)


def _export_locked(args, project, tools) -> int:
    import vibecut.core.jobs as jobs
    from vibecut.core.cache import Cache
    from vibecut.core.contracts import CONTRACT_VERSION, CameraPath, CameraView, EditPlan
    from vibecut.core.project import ProjectError, StaleRevisionError
    from vibecut.render.static import StaticRenderConfig, render_static

    source, code = _probe_or_exit(args.input, tools, args.projection)
    if source is None:
        return code
    project.add_source(source)

    cut_out = args.cut_out
    if cut_out is None:
        nb = source.video.nb_frames
        cut_out = Fraction(nb) / source.video.fps if nb else Fraction(1)
    plan = EditPlan(
        schema_version=CONTRACT_VERSION,
        source_id=source.source_id,
        cut_in_rel=args.cut_in,
        cut_out_rel=cut_out,
        project_fps=args.fps or project.fps,
        camera=CameraPath(schema_version=CONTRACT_VERSION, keys=((Fraction(0), CameraView(
            yaw_deg=args.yaw, pitch_deg=args.pitch, roll_deg=args.roll, h_fov_deg=args.hfov,
        )),)),
        audio=not args.no_audio,
    )

    job_id = jobs.new_job_id()
    project.plans[job_id] = plan
    project.save()
    project.add_job_event(jobs.make_event(job_id, project.revision, "queued"))
    project.add_job_event(jobs.make_event(job_id, project.revision, "running"))
    cancel_event = threading.Event()
    config = StaticRenderConfig(timeout_s=args.timeout)
    cache = Cache(project.path.parent / ".vibecut-cache")
    result = render_static(
        plan, source, tools, args.output, config,
        on_exists="unique" if args.unique_suffix else "fail",
        cache=cache, cancel_event=cancel_event, job_id=job_id,
    )
    try:
        if result.status == "succeeded":
            project.add_job_event(jobs.make_event(
                job_id, project.revision, "succeeded", artifact=str(result.artifact)
            ))
            project.save()
            print(f"succeeded: {result.artifact} (job {job_id[:8]})")
            return EXIT_OK
        project.add_job_event(jobs.make_event(
            job_id, project.revision, result.status, error=result.error
        ))
        project.save()
    except StaleRevisionError:
        pass  # late event: never overwrite a newer project revision
    except ProjectError as exc:
        print(f"warning: could not persist project state: {exc}", file=sys.stderr)

    if result.status == "cancelled":
        print("cancelled", file=sys.stderr)
        return EXIT_RENDER
    err = result.error or "unknown failure"
    print(f"render failed: {err}", file=sys.stderr)
    if isinstance_source_changed(err) or isinstance_output_exists(err):
        return EXIT_CONFLICT
    return EXIT_RENDER


def isinstance_source_changed(err: str) -> bool:
    return err.startswith("SourceChangedError")


def isinstance_output_exists(err: str) -> bool:
    return err.startswith("OutputExistsError")


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vibecut.core.project import ProjectError

    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "init": _cmd_init,
        "inspect": _cmd_inspect,
        "add-source": _cmd_add_source,
        "export": _cmd_export,
    }
    try:
        return handlers[args.command](args)
    except ProjectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFLICT
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as exc:
        print(f"I/O failed: {exc}", file=sys.stderr)
        return EXIT_RENDER
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_RENDER


if __name__ == "__main__":
    sys.exit(main())
