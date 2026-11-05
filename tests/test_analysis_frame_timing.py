"""Actual decoded-frame identity, packet PTS, fractional fps and source-relative origin."""
from fractions import Fraction as F
import pytest
from conftest import _encode_gray_video, decode_gray_frames, requires_ffmpeg
from vibecut.analysis.extract import extract_perspective_jpeg, probe_asset_timing
from vibecut.analysis.errors import ExtractionError
from vibecut.analysis.sampling import plan_sampling
from vibecut.analysis.views import ViewSpec


@requires_ffmpeg
@pytest.mark.parametrize("fps,count,index,offset", [
    (F(30000,1001),16,7,None),
    (F(30000,1001),16,15,1.5),
    (F(24000,1001),96,77,1.5),
])
def test_actual_jpeg_is_exact_requested_display_frame(tools,tmp_path,fps,count,index,offset):
    # A single bright frame distinguishes identity despite lossy JPEG level rounding.
    levels = [200 if i == index else 16 for i in range(count)]
    source = _encode_gray_video(tools,tmp_path/"numbered.mp4",[bytes([v])*(64*32) for v in levels],64,32,fps,crf=1,ts_offset=offset)
    timing = probe_asset_timing(tools,source,expected_fps=fps,expected_frames=count)
    if offset:
        assert timing.first_pts * timing.time_base > 1
    reference = [sum(f)/len(f) for f in decode_gray_frames(tools,source,64,32)]
    commands = []
    class RecordingTools:
        def __getattr__(self,name):
            return getattr(tools,name)
        def run(self,argv,**kwargs):
            commands.append(argv)
            return tools.run(argv,**kwargs)
    time = F(index)/fps
    jpg = extract_perspective_jpeg(RecordingTools(),str(source),time,ViewSpec("forward",0,0,width=64,height=36),
                                   tmp_path,cancel_event=None,timing=timing)
    image = decode_gray_frames(tools,jpg,64,36)[0]
    actual_mean = sum(image)/len(image)
    actual_index = min(range(count),key=lambda i:abs(reference[i]-actual_mean))
    assert actual_index == index
    command = commands[-1]
    assert command.index("-ss") < command.index("-i") and "-copyts" in command
    if index/fps > 2:
        assert int(command[command.index("-ss")+1]) >= 1  # actual input-side seek, never a full-file per-image decode
    target_pts = timing.first_pts + int(time/timing.time_base)
    assert f"trim=start_pts={target_pts}:end_pts={target_pts+1}" in command[command.index("-vf")+1]
    with pytest.raises(ExtractionError,match="not a verified displayed frame"):
        extract_perspective_jpeg(tools,str(source),time+F(1,2)/fps,ViewSpec("forward",0,0),tmp_path,
                                 cancel_event=None,timing=timing)


def test_non_frame_midpoint_is_recorded_as_actual_cfr_time_and_tail_is_retained():
    fps = F(30000,1001)
    sample = plan_sampling(F(15)/fps,samples=1,verified_fps=fps)
    assert sample.times == (F(7)/fps,)  # original bin midpoint was frame 7.5, not a displayed frame
    dense = plan_sampling(F(16)/fps,samples=64,verified_fps=fps)
    assert dense.times == tuple(F(i)/fps for i in range(16))
    assert dense.times[-1] == F(15)/fps < dense.duration_s
    short = plan_sampling(F(2)/fps,samples=64,verified_fps=fps)
    assert short.times == (F(0),F(1)/fps)
