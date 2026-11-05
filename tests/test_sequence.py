"""Dynamic contracts and real synthetic-only FFmpeg end-to-end checks."""
import json
import math
import threading
import time
from dataclasses import replace
from fractions import Fraction as F

import pytest
from conftest import (requires_ffmpeg, run_ffmpeg, run_ffprobe_json, make_luma_step_video,
                      make_tone_audio, decode_gray_frames, first_bright_frame_index, audio_onset_seconds,
                      make_marker_equirect, centroid_of_bright)
from vibecut.core.cache import Cache
from vibecut.core.contracts import CONTRACT_VERSION, CameraPath, CameraView, ContractError
from vibecut.core.project import Project
from vibecut.core.sequence import BoundCameraPath, SequenceClip, SequencePlan
from vibecut.media.probe import probe_source
from vibecut.render.dynamic import render_sequence
from vibecut.render.static import StaticRenderConfig, render_config_identity


def test_dense_camera_lookup_does_not_scan_all_keys():
    class CountedTime(F):
        comparisons = 0
        def __lt__(self, other):
            type(self).comparisons += 1
            return super().__lt__(other)
        def __gt__(self, other):
            type(self).comparisons += 1
            return super().__gt__(other)
    keys = tuple((CountedTime(n,30),CameraView(n/20)) for n in range(3601))
    camera = path("asset",F(120),keys=keys)
    CountedTime.comparisons=0
    for n in range(3600):
        assert abs(camera.evaluate(F(2*n+1,60)).yaw_deg-(n+.5)/20)<1e-8
    assert CountedTime.comparisons < 3600*30
    print("DENSE_LOOKUP_EVIDENCE comparisons",CountedTime.comparisons,"queries",3600)


@pytest.mark.parametrize("mode",["cancel","timeout"])
def test_command_preparation_is_interruptible_after_it_started(tmp_path,monkeypatch,mode):
    from types import SimpleNamespace
    from vibecut.render.dynamic import build_sequence_argv
    from vibecut.render.static import ArtifactProcessError
    keys=tuple((F(n,30),CameraView(n/20)) for n in range(3601))
    camera=path("asset",F(120),keys=keys)
    plan=SequencePlan((SequenceClip("asset",F(0),F(120),camera),),F(30),audio=False)
    # The graph builder consumes verified metadata only; it never opens media or runs FFmpeg.
    source=SimpleNamespace(path="synthetic.mp4",video=SimpleNamespace(index=0),audio=None,
                           cut_window_absolute=lambda start,end:(start,end))
    cancel=threading.Event()
    original=BoundCameraPath.evaluate
    evaluations=0
    def observe(self,source_rel):
        nonlocal evaluations
        evaluations+=1
        value=original(self,source_rel)
        if evaluations==1 and mode=="cancel":
            cancel.set()  # preparation has evaluated a frame; stop synchronously before the next
        return value
    monkeypatch.setattr(BoundCameraPath,"evaluate",observe)
    started=time.monotonic()
    with pytest.raises(ArtifactProcessError,match="cancelled.*preparation" if mode=="cancel" else "preparation timed out"):
        build_sequence_argv({"asset":source},plan,StaticRenderConfig(timeout_s=.01 if mode=="timeout" else 5),
                            tmp_path/"out.mp4",tmp_path,cancel_event=cancel)
    elapsed=time.monotonic()-started
    assert 0 < evaluations < 3600
    assert elapsed < 1
    assert not (tmp_path/"out.mp4").exists()
    print("PREPARATION_EVIDENCE",mode,"evaluated_frames",evaluations,"elapsed",elapsed)


def path(asset, duration, *, aspect="16:9", keys=None, cuts=()):
    keys = keys or ((F(0), CameraView()), (duration, CameraView()))
    return BoundCameraPath(asset, aspect, CameraPath(CONTRACT_VERSION, tuple(keys)), tuple(cuts))


def sequence(source, *, start=F(0), end=None, aspect="16:9", keys=None, cuts=(), fps=None, audio=True):
    duration = F(source.video.nb_frames) / source.video.fps
    p = path(source.source_id, duration, aspect=aspect, keys=keys, cuts=cuts)
    return SequencePlan((SequenceClip(source.source_id, start, end or duration, p),), fps or source.video.fps, aspect, audio)


def test_evaluator_binding_cut_seam_and_schema(tmp_path):
    p = path("asset", F(1), keys=((F(0), CameraView(178)), (F(1, 2), CameraView(-178)), (F(1), CameraView(90))), cuts=(F(1),))
    assert p.evaluate(F(1, 4)).yaw_deg == 180
    assert p.evaluate(F(3, 4)).yaw_deg == -178
    assert p.evaluate(F(1)).yaw_deg == 90
    assert BoundCameraPath.from_json(p.to_json()) == p
    for values in ({"geometry_revision": 99}, {"cuts": (F(1, 3),)}, {"aspect": "1:1"}, {"cuts": (F(1), F(1))}):
        with pytest.raises(ContractError):
            replace(p, **values)
    for view in (CameraView(float("nan")), CameraView(h_fov_deg=float("inf")), CameraView(pitch_deg=91), CameraView(h_fov_deg=180)):
        with pytest.raises(ContractError):
            path("asset", F(1), keys=((F(0), view), (F(1), view)))
    with pytest.raises(ContractError):
        SequenceClip("other", F(0), F(1), p)
    clip = SequenceClip("asset", F(0), F(1), p)
    with pytest.raises(ContractError):
        SequencePlan((clip,), F(30000, 1001))
    plan = SequencePlan((clip,), F(30))
    project = Project.create(tmp_path)
    project.sequences["one"] = plan
    project.save()
    assert Project.load(project.path).sequences["one"] == plan
    assert render_config_identity(StaticRenderConfig(), plan) != render_config_identity(StaticRenderConfig(out_width=640), plan)


@requires_ffmpeg
def test_dynamic_smoke_backends_and_no_clobber(tools, tmp_path):
    src = make_luma_step_video(tools, tmp_path / "input.mp4", width=320, height=160, fps=F(30), frames=12, step_frame=6)
    source = probe_source(src, tools, declared_projection="equirect")
    plan = sequence(source)
    config = StaticRenderConfig(out_width=320, out_height=180, backend="auto")
    cache = Cache(tmp_path / "cache")
    out = tmp_path / "result.mp4"
    result = render_sequence(plan, {source.source_id: source}, tools, out, config, cache=cache)
    print("BACKEND_EVIDENCE", result.events)
    assert result.status == "succeeded", result.error
    assert any(name == "nvdec-probe" for name, _ in result.events)
    assert any(name == "nvenc-probe" for name, _ in result.events)
    assert any(name == "decoded" for name, _ in result.events)
    assert cache.has(result.cache_key)
    original = out.read_bytes()
    collision = render_sequence(plan, {source.source_id: source}, tools, out, config)
    assert collision.status == "failed" and "OutputExistsError" in collision.error
    assert out.read_bytes() == original
    cancel = threading.Event(); cancel.set()
    failed = render_sequence(plan, {source.source_id: source}, tools, tmp_path / "cancel.mp4", config, cache=cache, cancel_event=cancel)
    assert failed.status == "cancelled" and failed.artifact is None and failed.cache_key is None
    # RTX 4090 accepts 160x90; 32x18 was actually probed to fail NVENC initialization.
    cut_plan = sequence(source, start=F(3,30), end=F(9,30),
                        keys=((F(0),CameraView(0)),(F(12,30),CameraView(40,10,5,70))))
    small_config = replace(config, out_width=32, out_height=18)
    small_out = tmp_path / "small.mp4"
    small = render_sequence(cut_plan, {source.source_id: source}, tools, small_out, small_config)
    assert small.status == "succeeded", small.error
    if any(name == "nvenc-probe" and note.startswith("available") for name, note in small.events):
        assert any(name == "cpu-fallback" for name, _ in small.events)
        assert any("InitializeEncoder failed" in note for name,note in small.events if name=="cpu-fallback")
    cpu = render_sequence(cut_plan, {source.source_id: source}, tools, tmp_path / "small-cpu.mp4",
                          replace(small_config, backend="cpu"))
    assert cpu.status == "succeeded", cpu.error
    fallback_frames = decode_gray_frames(tools, small_out, 32, 18)
    assert len(fallback_frames) == 6
    assert first_bright_frame_index(fallback_frames,32,18)==3
    assert fallback_frames == decode_gray_frames(tools, cpu.artifact, 32, 18)
    preserved = small_out.read_bytes()
    conflict = render_sequence(cut_plan,{source.source_id:source},tools,small_out,small_config)
    assert conflict.status=="failed" and "OutputExistsError" in conflict.error
    assert small_out.read_bytes()==preserved
    print("FALLBACK_EVIDENCE", small.events)


@requires_ffmpeg
def test_sequence_nonzero_pts_fractional_fps_mixed_audio(tools, tmp_path):
    fps = F(30000, 1001)
    silent = make_luma_step_video(tools, tmp_path / "silent.mp4", width=320, height=160, fps=fps, frames=30, step_frame=9, ts_offset=1.5)
    video = make_luma_step_video(tools, tmp_path / "video.mp4", width=320, height=160, fps=fps, frames=30, step_frame=9, ts_offset=1.5)
    tone = make_tone_audio(tools, tmp_path / "tone.m4a", seconds=2, onset_s=float(F(9)/fps))
    av = tmp_path / "av.mp4"
    mux = run_ffmpeg(tools, ["-copyts", "-i", str(video), "-itsoffset", "1.5", "-i", str(tone),
                             "-map", "0:v", "-map", "1:a", "-c", "copy", "-y", str(av)])
    assert mux.returncode == 0, mux.stderr
    sources = [probe_source(p, tools, declared_projection="equirect") for p in (silent, av)]
    assert all(s.first_video_frame_seconds > 1 for s in sources)
    start, end = F(3)/fps, F(18)/fps
    clips = tuple(SequenceClip(s.source_id, start, end, path(s.source_id, F(30)/fps)) for s in sources)
    plan = SequencePlan(clips, fps)
    out = tmp_path / "joined.mp4"
    result = render_sequence(plan, {s.source_id: s for s in sources}, tools, out,
                             StaticRenderConfig(out_width=320, out_height=180))
    assert result.status == "succeeded", result.error
    frames = decode_gray_frames(tools, out, 320, 180)
    assert len(frames) == 30
    assert abs(first_bright_frame_index(frames[:15], 320, 180) - 6) <= 1
    assert abs(first_bright_frame_index(frames[15:], 320, 180) - 6) <= 1
    onset = audio_onset_seconds(tools, out)
    expected = float(F(21)/fps)
    print("SYNC_EVIDENCE", "onset", onset, "expected", expected, "error", abs(onset-expected))
    assert abs(onset - expected) <= 0.080
    streams = run_ffprobe_json(tools, out)["streams"]
    a = next(s for s in streams if s["codec_type"] == "audio")
    assert a["sample_rate"] == "48000" and a["channels"] == 2


def _world_ray(view, x, y, w, h):
    """Independent Ry Rx Rz reference, image axes x-right/y-down/z-forward."""
    focal = w / (2 * math.tan(math.radians(view.h_fov_deg) / 2))
    x, y, z = (x + .5 - w/2) / focal, (y + .5 - h/2) / focal, 1
    yaw, pitch, roll = map(math.radians, (view.yaw_deg, view.pitch_deg, view.roll_deg))
    x, y = math.cos(roll)*x-math.sin(roll)*y, math.sin(roll)*x+math.cos(roll)*y
    y, z = math.cos(pitch)*y-math.sin(pitch)*z, math.sin(pitch)*y+math.cos(pitch)*z
    x, z = math.cos(yaw)*x+math.sin(yaw)*z, -math.sin(yaw)*x+math.cos(yaw)*z
    norm = math.sqrt(x*x+y*y+z*z)
    return tuple(a/norm for a in (x,y,z))


def _direction_source(tools, out, views, aspect):
    w,h=1440,720
    ow,oh=(320,180) if aspect=="16:9" else (180,320)
    raw=bytearray()
    for view in views:
        image=bytearray([16])*(w*h)
        for fx,fy,value in ((.5,.5,235),(.7,.7,170)):
            target=_world_ray(view,ow*fx-.5,oh*fy-.5,ow,oh)
            latitude=math.asin(-target[1]); longitude=math.atan2(target[0],target[2])
            radius=math.radians(1.5)
            lo=max(0,int((.5-(latitude+radius)/math.pi)*h)-1)
            hi=min(h,int((.5-(latitude-radius)/math.pi)*h)+2)
            for y in range(lo,hi):
                lat=(.5-(y+.5)/h)*math.pi
                for x in range(w):
                    lon=((x+.5)/w-.5)*2*math.pi
                    dot=math.cos(lat)*math.sin(lon)*target[0]-math.sin(lat)*target[1]+math.cos(lat)*math.cos(lon)*target[2]
                    if dot>=math.cos(radius): image[y*w+x]=value
        raw.extend(bytes(image)+bytes([128])*(w*h//2))
    r=run_ffmpeg(tools,["-f","rawvideo","-pix_fmt","yuv420p","-s",f"{w}x{h}","-r","10","-i","-",
                       "-frames:v",str(len(views)),
                       "-c:v","libx264","-crf","0","-x264-params","colorprim=bt709:transfer=bt709:colormatrix=bt709",
                       "-colorspace","bt709","-color_range","tv","-y",str(out)],stdin_data=bytes(raw))
    assert r.returncode == 0, r.stderr
    return probe_source(out,tools,declared_projection="equirect")


def _decode_y(tools,out,w,h):
    r=run_ffmpeg(tools,["-i",str(out),"-f","rawvideo","-pix_fmt","yuv420p","-"])
    assert r.returncode==0,r.stderr
    size=w*h*3//2
    return [r.stdout[n:n+w*h] for n in range(0,len(r.stdout),size)]


@requires_ffmpeg
@pytest.mark.parametrize("aspect",["16:9","9:16"])
def test_dynamic_independent_geometry_preview_final_and_seek(tools,tmp_path,aspect):
    views=[CameraView(0),CameraView(90),CameraView(180),CameraView(-90),CameraView(30,20,17),CameraView(30,20,17),
           CameraView(30,20,17,65),CameraView(178),CameraView(-178),CameraView(90,90,23,80),CameraView(-90,-90,-23,80),CameraView()]
    source=_direction_source(tools,tmp_path/"one.mp4",views,aspect)
    keys=tuple((F(n,10),v) for n,v in enumerate(views))+((F(12,10),views[-1]),)
    plan=sequence(source,aspect=aspect,keys=keys,cuts=tuple(F(n,10) for n in range(1,12)))
    sizes=[(320,180),(1920,1080)] if aspect=="16:9" else [(180,320),(1080,1920)]
    max_errors=[]
    for j,(w,h) in enumerate(sizes):
        out=tmp_path/f"geometry{j}.mp4"
        result=render_sequence(plan,{source.source_id:source},tools,out,StaticRenderConfig(out_width=w,out_height=h,crf=0))
        assert result.status=="succeeded",result.error
        frames=_decode_y(tools,out,w,h)
        assert len(frames)==12
        errors=[]
        for n in range(12):
            for fx,fy in ((.5,.5),(.7,.7)):
                x,y=w*fx-.5,h*fy-.5
                samples=[(xx,yy,max(0,frames[n][yy*w+xx]-25))
                         for yy in range(max(0,int(y-h*.08)),min(h,int(y+h*.08)+1))
                         for xx in range(max(0,int(x-w*.08)),min(w,int(x+w*.08)+1))]
                total=sum(value for _,_,value in samples)
                assert total>0, f"missing world marker frame {n} at {fx},{fy}"
                cx=sum(xx*value for xx,_,value in samples)/total
                cy=sum(yy*value for _,yy,value in samples)/total
                focal=w/(2*math.tan(math.radians(views[n].h_fov_deg)/2))
                error=math.degrees(math.atan(math.hypot(cx-x,cy-y)/focal))
                errors.append(error)
        max_errors.append(max(errors))
        assert max(errors)<=1, f"{aspect} independent direction error {max(errors)}"
    print("GEOMETRY_EVIDENCE",aspect,"preview/final max angular errors",max_errors)
    # Cropping/re-rendering must evaluate source-relative time, including cut sides.
    clip=replace(plan.clips[0],cut_in_rel=F(4,10),cut_out_rel=F(10,10))
    cropped=replace(plan,clips=(clip,))
    w,h=sizes[0]
    seek_out=tmp_path/"seek.mp4"
    result=render_sequence(cropped,{source.source_id:source},tools,seek_out,StaticRenderConfig(out_width=w,out_height=h,crf=0))
    assert result.status=="succeeded",result.error
    baseline=decode_gray_frames(tools,tmp_path/"geometry0.mp4",w,h)[4:10]
    seek=decode_gray_frames(tools,seek_out,w,h)
    assert seek==baseline


@requires_ffmpeg
def test_10bit_sdr_real_conversion_and_proxy(tools,tmp_path):
    from vibecut.media.proxy import ensure_proxy
    src=tmp_path/"tenbit.mp4"
    r=run_ffmpeg(tools,["-f","lavfi","-i","nullsrc=s=320x160:r=30:d=0.2,format=yuv420p10le,geq=lum=64+876*X/W:cb=512:cr=512",
                       "-c:v","libx264","-crf","0","-pix_fmt","yuv420p10le",
                       "-x264-params","colorprim=bt709:transfer=bt709:colormatrix=bt709","-colorspace","bt709","-color_range","tv","-y",str(src)])
    assert r.returncode==0,r.stderr
    source=probe_source(src,tools,declared_projection="equirect")
    assert source.color.bit_depth==10
    out=tmp_path/"eightbit.mp4"
    result=render_sequence(sequence(source),{source.source_id:source},tools,out,StaticRenderConfig(out_width=320,out_height=180,crf=0))
    assert result.status=="succeeded",result.error
    v=next(s for s in run_ffprobe_json(tools,out)["streams"] if s["codec_type"]=="video")
    assert v["pix_fmt"]=="yuv420p" and v["color_transfer"]=="bt709"
    raw=run_ffmpeg(tools,["-i",str(out),"-frames:v","1","-f","rawvideo","-pix_fmt","yuv420p","-"]).stdout[:320*180]
    # Independent ramp reference at the centre ray: original limited-range 10bit ~502 -> 8bit ~125.5.
    assert abs(raw[90*320+160]-126)<=3
    asset=ensure_proxy(source,tools,Cache(tmp_path/"cache"),width=320,height=160)
    assert asset.color.bit_depth==8
    assert run_ffprobe_json(tools,asset.path)["streams"][0]["pix_fmt"]=="yuv420p"


@requires_ffmpeg
def test_seam_short_arc_actual_frames(tools, tmp_path):
    src = make_marker_equirect(tools, tmp_path / "seam.mp4", yaw_deg=180, pitch_deg=0, frames=9, span_deg=2)
    source = probe_source(src, tools, declared_projection="equirect")
    plan = sequence(source, keys=((F(0), CameraView(178)), (F(3,10), CameraView(-178))))
    out = tmp_path / "seam-flat.mp4"
    result = render_sequence(plan, {source.source_id: source}, tools, out, StaticRenderConfig(out_width=640, out_height=360, crf=0))
    assert result.status == "succeeded", result.error
    frames = decode_gray_frames(tools, out, 640, 360)
    errors = []
    for n, frame in enumerate(frames):
        cx, cy = centroid_of_bright([frame],640,360,window=.4)
        actual = math.degrees(math.atan((cx+.5-320)/320))
        expected = 2 - 4*n/9
        errors.append(abs(actual-expected))
    assert max(errors) <= 1
    print("SEAM_EVIDENCE", max(errors))


@requires_ffmpeg
@pytest.mark.parametrize("failure", ["missing-filter", "cancel", "disk-full"])
def test_dynamic_failures_publish_nothing(tools,tmp_path,monkeypatch,failure):
    import errno
    import vibecut.render.dynamic as dynamic
    import vibecut.render.static as renderer
    src=make_luma_step_video(tools,tmp_path/"input.mp4",width=320,height=160,fps=F(30),frames=60,step_frame=30)
    source=probe_source(src,tools,declared_projection="equirect")
    plan=sequence(source)
    cache=Cache(tmp_path/"cache")
    cancel=threading.Event()
    timer=None
    if failure=="missing-filter":
        real=dynamic.build_sequence_argv
        def bad(*args,**kw):
            return tuple(a.replace("v360@cam0=", "missing_filter@cam0=") for a in real(*args,**kw))
        monkeypatch.setattr(dynamic,"build_sequence_argv",bad)
    elif failure=="disk-full":
        def full(*args):
            raise OSError(errno.ENOSPC,"injected full output volume")
        monkeypatch.setattr(renderer,"commit_artifact",full)
    else:
        timer=threading.Timer(.1,cancel.set); timer.start()
    try:
        out=tmp_path/"failure.mp4"
        result=render_sequence(plan,{source.source_id:source},tools,out,
                               StaticRenderConfig(),cache=cache,cancel_event=cancel)
    finally:
        if timer: timer.cancel(); timer.join()
    assert result.status==("cancelled" if failure=="cancel" else "failed")
    assert result.artifact is None and not out.exists()
    assert not list(tmp_path.glob("*.partial")) and not list((tmp_path/"cache").glob("*.json"))
    assert all(name!="committed" for name,_ in result.events)


@requires_ffmpeg
def test_sequence_cli_persists_and_rerenders(tools,tmp_path,monkeypatch):
    from vibecut.cli import main
    monkeypatch.setenv("VIBECUT_FFMPEG",str(tools.ffmpeg))
    monkeypatch.setenv("VIBECUT_FFPROBE",str(tools.ffprobe))
    src=make_marker_equirect(tools,tmp_path/"source.mp4",yaw_deg=0,pitch_deg=0,frames=6)
    source=probe_source(src,tools,declared_projection="equirect")
    project=Project.create(tmp_path/"project"); project.add_source(source); project.save()
    plan=sequence(source)
    data=tmp_path/"sequence.json"; data.write_text(json.dumps(plan.to_json()),encoding="utf-8")
    assert main(["render-sequence",str(project.path),str(data),str(tmp_path/"cli.mp4"),"--preview","--cpu"])==0
    reopened=Project.load(project.path)
    event=next(e for e in reopened.jobs if e.status=="succeeded")
    assert reopened.sequences[event.job_id]==plan
    config=StaticRenderConfig(out_width=640,out_height=360,crf=26,preview=True)
    cached=render_sequence(reopened.sequences[event.job_id],reopened.sources,tools,tmp_path/"cached.mp4",config,
                           cache=Cache(project.path.parent/".vibecut-cache"))
    assert cached.status=="succeeded",cached.error
    assert any(name=="cache-hit" for name,_ in cached.events)
    assert decode_gray_frames(tools,tmp_path/"cached.mp4",640,360)==decode_gray_frames(tools,tmp_path/"cli.mp4",640,360)
    data.write_text('{"schema_version":99}',encoding="utf-8")
    assert main(["render-sequence",str(project.path),str(data),str(tmp_path/"invalid.mp4"),"--cpu"])==2


@requires_ffmpeg
def test_dynamic_output_path_survives_both_filter_escape_layers(tools, tmp_path):
    directory = tmp_path / "中文 single'quote space,semi;[brackets]"
    directory.mkdir()
    src = make_marker_equirect(tools, directory / "source's 空格,;[].mp4", yaw_deg=0, pitch_deg=0, frames=6)
    source = probe_source(src, tools, declared_projection="equirect")
    out = directory / "out's 空格,;[].mp4"
    result = render_sequence(sequence(source), {source.source_id: source}, tools, out,
                             StaticRenderConfig(out_width=320, out_height=180))
    assert result.status == "succeeded", result.error
    assert len(decode_gray_frames(tools, out, 320, 180)) == 6
    assert not list(directory.glob(".vibecut-camera-*"))


@requires_ffmpeg
def test_repeated_preview_hits_reuse_one_immutable_cache_payload(tools,tmp_path):
    src=make_marker_equirect(tools,tmp_path/"source.mp4",yaw_deg=0,pitch_deg=0,frames=6)
    source=probe_source(src,tools,declared_projection="equirect")
    plan=sequence(source)
    cache=Cache(tmp_path/"cache")
    config=StaticRenderConfig(out_width=320,out_height=180,preview=True)
    first=render_sequence(plan,{source.source_id:source},tools,tmp_path/"preview0.mp4",config,cache=cache)
    assert first.status=="succeeded",first.error
    initial={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in cache.root.iterdir()}
    manifest=(cache.root/f"{first.cache_key}.json").read_bytes()
    assert len(initial)==2
    for n in range(1,4):
        hit=render_sequence(plan,{source.source_id:source},tools,tmp_path/f"preview{n}.mp4",config,cache=cache)
        assert hit.status=="succeeded",hit.error
        assert any(name=="cache-hit" for name,_ in hit.events)
        assert {p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in cache.root.iterdir()}==initial
        assert (cache.root/f"{first.cache_key}.json").read_bytes()==manifest
    assert cache.invalidate(first.cache_key)
    assert list(cache.root.iterdir())==[]


@requires_ffmpeg
def test_render_cancel_during_command_preparation_leaves_no_artifact_or_cache(tools,tmp_path,monkeypatch):
    src=make_luma_step_video(tools,tmp_path/"source.mp4",width=320,height=160,fps=F(30),frames=60,step_frame=30)
    source=probe_source(src,tools,declared_projection="equirect")
    # Dense output sampling makes preparation observable without producing a long input fixture.
    keys=tuple((F(n,1800),CameraView(n/20)) for n in range(3601))
    plan=sequence(source,keys=keys,fps=F(1800))
    cancel=threading.Event()
    original=BoundCameraPath.evaluate
    count=0
    timer=None
    def observe(self,t):
        nonlocal count,timer
        count+=1
        if count==1:
            timer=threading.Timer(.01,cancel.set);timer.start()
        return original(self,t)
    monkeypatch.setattr(BoundCameraPath,"evaluate",observe)
    cache=Cache(tmp_path/"cache")
    try:
        result=render_sequence(plan,{source.source_id:source},tools,tmp_path/"cancel.mp4",
                               StaticRenderConfig(),cancel_event=cancel,cache=cache)
    finally:
        if timer:timer.cancel();timer.join()
    assert result.status=="cancelled" and "command preparation" in result.error
    assert 0 < count < 3600
    assert not (tmp_path/"cancel.mp4").exists()
    assert not list(tmp_path.glob("*.partial")) and not list(tmp_path.glob(".vibecut-camera-*"))
    assert not list(cache.root.glob("*.json"))


@requires_ffmpeg
def test_mixed_source_fps_and_distinct_origins_keep_av_boundaries(tools, tmp_path):
    output_fps = F(30000, 1001)
    sources = []
    source_fps = (F(24), output_fps)
    origins = (F(3, 2), F(7, 2))
    # Each independently clocked source starts dark/silent and changes at its sixth displayed frame.
    for n, (fps, origin) in enumerate(zip(source_fps, origins)):
        video = make_luma_step_video(tools, tmp_path / f"video{n}.mp4", width=320, height=160,
                                     fps=fps, frames=48 if n == 0 else 60, step_frame=6,
                                     ts_offset=float(origin))
        tone = make_tone_audio(tools, tmp_path / f"tone{n}.m4a", seconds=3, onset_s=float(F(6)/fps))
        av = tmp_path / f"av{n}.mp4"
        mux = run_ffmpeg(tools, ["-copyts", "-i", str(video), "-itsoffset", str(float(origin)), "-i", str(tone),
                                "-map", "0:v", "-map", "1:a", "-c", "copy", "-y", str(av)])
        assert mux.returncode == 0, mux.stderr
        sources.append(probe_source(av, tools, declared_projection="equirect"))
    assert sources[0].video.fps == 24 and sources[1].video.fps == output_fps
    for source, origin in zip(sources, origins):
        assert abs(source.first_video_frame_seconds - origin) <= source.video.time_base
    length = F(15) / output_fps
    # Different source-relative cut-ins plus different absolute origins exercise both mappings.
    starts = (F(2)/24, F(3)/output_fps)
    clips = tuple(SequenceClip(source.source_id, start, start + length,
                               path(source.source_id, F(source.video.nb_frames)/source.video.fps))
                  for source, start in zip(sources, starts))
    plan = SequencePlan(clips, output_fps)
    out = tmp_path / "mixed-clocks.mp4"
    result = render_sequence(plan, {s.source_id: s for s in sources}, tools, out,
                             StaticRenderConfig(out_width=320, out_height=180, crf=0))
    assert result.status == "succeeded", result.error
    frames = decode_gray_frames(tools, out, 320, 180)
    assert len(frames) == 30
    observed = []
    for n, (fps, cut_in) in enumerate(zip(source_fps, starts)):
        index = first_bright_frame_index(frames[n*15:(n+1)*15], 320, 180)
        expected = (F(6)/fps - cut_in) * output_fps
        assert index is not None and abs(F(index) - expected) <= 1
        # Decode the actual final output's audio within each concatenated segment.
        segment = tmp_path / f"audio-boundary{n}.mp4"
        split = run_ffmpeg(tools, ["-i", str(out), "-ss", str(float(n*length)), "-t", str(float(length)),
                                  "-map", "0:a:0", "-c:a", "aac", "-y", str(segment)])
        assert split.returncode == 0, split.stderr
        audio = audio_onset_seconds(tools, segment)
        expected_audio = float(F(6)/fps-cut_in)
        video_time = index/float(output_fps)
        assert abs(audio - expected_audio) <= .080
        assert abs(audio - video_time) <= .080
        observed.append({"source_fps": str(fps), "source_origin": str(sources[n].first_video_frame_seconds),
                         "video_frame": index, "expected_frame": float(expected), "audio_onset": audio,
                         "expected_audio": expected_audio})
    print("MIXED_CLOCK_EVIDENCE", observed)
