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
    p_sequence = sub.add_parser("render-sequence", help="render a bound dynamic sequence JSON")
    p_sequence.add_argument("project", type=Path)
    p_sequence.add_argument("sequence", type=Path)
    p_sequence.add_argument("output", type=Path)
    p_sequence.add_argument("--preview", action="store_true", help="same evaluator, 640x360 / 360x640")
    p_sequence.add_argument("--cpu", action="store_true", help="disable actual NVDEC/NVENC probes")
    p_sequence.add_argument("--unique-suffix", action="store_true")
    p_sequence.add_argument("--timeout", type=float, default=600.0)
    p_an = sub.add_parser("analyze", help="AI candidate analysis (local loopback model only, task #5)")
    p_an.add_argument("project", type=Path, help="path to vibecut.project.json (or its directory)")
    p_an.add_argument("input", type=Path, help="synthetic/canonical equirect source")
    p_an.add_argument("--focus", required=True, help="中文重点描述（人物/风景/时长/排除项）")
    p_an.add_argument("--person-priority", action="store_true", help="人物优先")
    p_an.add_argument("--scenery-priority", action="store_true", help="风景优先")
    p_an.add_argument("--target-duration", type=_parse_time, default=None, metavar="SEC",
                      help="目标总时长（秒，允许误差 ≤1s）")
    p_an.add_argument("--endpoint", default="http://127.0.0.1:1234",
                      help="本地回环模型服务（仅 127.0.0.1/::1/localhost，禁止公网）")
    p_an.add_argument("--model", default=None, help="模型 ID（默认用户已选 qwen3.8-27b@q4_k_xl，探测核实）")
    p_an.add_argument("--samples", type=int, default=12, help="全片均匀采样点数（1..64）")
    p_an.add_argument("--projection", choices=["equirect"], default=None)
    p_an.add_argument("--max-requests", type=int, default=32, help="模型请求预算（含重试）")
    p_an.add_argument("--timeout", type=float, default=300.0, help="单次模型请求超时秒")
    p_an.add_argument("--json", action="store_true", help="stdout 输出完整 JSON 结果")
    p_an.add_argument("--allow-jit-load", action="store_true",
                      help="模型未加载时允许首个推理请求触发 JIT 加载（需显式授权 GPU 窗口；默认拒绝无人值守加载）")
    p_an.add_argument("--out", type=Path, default=None,
                      help="把完整 JSON 写入一个新文件（原子写入，绝不覆盖已有文件）")
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
    return _record_result(project, result, job_id)


def _cmd_sequence(args):
    import vibecut.core.jobs as jobs
    from vibecut.core.cache import Cache
    from vibecut.core.sequence import SequencePlan
    from vibecut.render.dynamic import render_sequence
    from vibecut.render.static import StaticRenderConfig

    tools = _resolve_tools()
    if tools is None:
        return EXIT_TOOLS
    project = _load_project(args.project)
    with project.exclusive_lock(timeout_s=0):
        project.check_revision()
        try:
            plan = SequencePlan.from_json(json.loads(args.sequence.read_text(encoding="utf-8")))
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError(f"invalid sequence JSON: {exc}") from exc
        plan.validate_sources(project.sources)
        job_id = jobs.new_job_id()
        project.sequences[job_id] = plan
        project.save()
        for status in ("queued", "running"):
            project.add_job_event(jobs.make_event(job_id, project.revision, status))
        width, height = (640, 360) if args.preview else (1920, 1080)
        if plan.aspect == "9:16":
            width, height = height, width
        config = StaticRenderConfig(out_width=width, out_height=height, crf=26 if args.preview else 18,
                                    timeout_s=args.timeout, backend="cpu" if args.cpu else "auto", preview=args.preview)
        result = render_sequence(plan, project.sources, tools, args.output, config,
                                 on_exists="unique" if args.unique_suffix else "fail",
                                 cache=Cache(project.path.parent / ".vibecut-cache"), job_id=job_id)
        for status, note in result.events:
            print(f"{status}: {note}")
        return _record_result(project, result, job_id)


def _record_result(project, result, job_id):
    import vibecut.core.jobs as jobs
    from vibecut.core.project import ProjectError, StaleRevisionError

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


def _write_result_noclobber(out: Path, payload: dict) -> bool:
    """Publish a result JSON atomically without ever overwriting.

    Write to a unique temp file in the target directory, fsync it, then
    publish via ``os.link`` (an atomic no-clobber create): readers either see
    the previous file or the complete new one - never a half-written JSON.
    Returns False when the target already exists.
    """
    import os
    import tempfile

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=str(out.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, out)
            return True
        except FileExistsError:
            return False
    finally:
        tmp.unlink(missing_ok=True)


def _cmd_analyze(args: argparse.Namespace) -> int:
    import threading

    from vibecut.analysis import AnalysisConfig, parse_focus
    from vibecut.analysis.errors import AnalysisError
    from vibecut.analysis.focus import FocusError
    from vibecut.analysis.service import DEFAULT_MODEL, analyze
    from vibecut.analysis.transport import validate_endpoint
    from vibecut.core.cache import Cache

    tools = _resolve_tools()
    if tools is None:
        return EXIT_TOOLS
    try:
        validate_endpoint(args.endpoint)
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    priority = "person" if args.person_priority else ("scenery" if args.scenery_priority else None)
    try:
        focus = parse_focus(args.focus, priority=priority, target_duration_s=args.target_duration)
    except FocusError as exc:
        print(f"error: --focus: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if args.out is not None and Path(args.out).exists():
        # fail fast: never spend GPU time just to discover the target exists
        print(f"error: output already exists (never overwritten): {args.out}", file=sys.stderr)
        return EXIT_CONFLICT

    config = AnalysisConfig(
        endpoint=args.endpoint, model_id=args.model if args.model else DEFAULT_MODEL,
        samples=args.samples, max_model_requests=args.max_requests, model_timeout_s=args.timeout,
        allow_jit_load=args.allow_jit_load,
    )

    project = _load_project(args.project)
    # Hold the project job lock from probe through revision-checked publication.
    with project.exclusive_lock(timeout_s=0):
        project.check_revision()
        source, code = _probe_or_exit(args.input, tools, args.projection)
        if source is None:
            return code
        for existing in project.sources.values():
            if Path(existing.path).resolve() == Path(source.path).resolve() \
                    and existing.content_fingerprint != source.content_fingerprint:
                print("error: source content changed since ingest", file=sys.stderr)
                return EXIT_CONFLICT
        asset = next((a for a in project.assets if a.source_id == source.source_id and a.kind == "proxy"), None)

        cache = Cache(Path(project.path).parent / ".vibecut-cache")

        cancel_event = threading.Event()
        try:
            result = analyze(source, asset, tools, cache, config, focus, cancel_event=cancel_event)
        except KeyboardInterrupt:
            cancel_event.set()
            print("cancelled", file=sys.stderr)
            return EXIT_RENDER

        project.check_revision()
        payload = result.to_json()
        if args.out is not None and not _write_result_noclobber(Path(args.out), payload):
            print(f"error: output already exists (never overwritten): {args.out}", file=sys.stderr)
            return EXIT_CONFLICT

        if args.json:
            import json as _json

            print(_json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True))
        else:
            _print_analysis_summary(result)

    if result.status == "ai-unavailable":
        diag = result.diagnostics[0] if result.diagnostics else {}
        print(f"AI 不可用（stage={diag.get('stage')} code={diag.get('code')}）：{diag.get('message', '')}", file=sys.stderr)
        print("可人工审阅候选窗口后手动剪辑；本工具不提供规则兑底冒充 AI。", file=sys.stderr)
        return EXIT_RENDER
    if result.status in ("cancelled", "failed"):
        for d in result.diagnostics:
            print(f"analysis {d.get('status', d.get('stage'))}: {d}", file=sys.stderr)
        return EXIT_RENDER
    return EXIT_OK


def _print_analysis_summary(result) -> None:
    m = result.metering
    plan = result.plan
    if plan is not None:
        print(f"选片状态: {plan.status}  片段 {len(plan.segments)} 段  "
              f"总时长 {float(plan.total_duration_s):.1f}s / 目标 {float(plan.target_duration_s):.1f}s")
        if plan.shortfall_reason:
            print(f"  说明: {plan.shortfall_reason}")
        for s in plan.segments:
            ev = ", ".join(e.candidate_id for e in s.evidence)
            print(f"  {s.segment_id}  {float(s.cut_in_rel):.2f}s-{float(s.cut_out_rel):.2f}s  "
                  f"view={s.view_id}  相关度 {s.relevance:.2f}  证据[{ev}]  理由: {s.reason[:60]}")
    print(f"候选窗口 {len(result.candidates)} 个（供人工审阅，含非 AI 廉价特征来源）")
    print(f"计量: 推理请求 {m.request_count} 次（重试 {m.retry_count}），探测 {sum(r.get('kind') == 'probe' for r in m.requests)} 次，模型耗时 {m.model_time_s:.1f}s，"
          f"缓存 cheap={m.cache.get('cheap_features')} 观察命中={m.cache.get('observation_batches_hit', 0)} "
          f"未命中={m.cache.get('observation_batches_miss', 0)} 选片={m.cache.get('selection')}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vibecut.core.project import ProjectError

    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "init": _cmd_init,
        "inspect": _cmd_inspect,
        "add-source": _cmd_add_source,
        "export": _cmd_export,
        "analyze": _cmd_analyze,
        "render-sequence": _cmd_sequence,
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
