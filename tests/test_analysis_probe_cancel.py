"""Real blocked probe processes and mid-scan cancellation, never a model call."""
import json
import builtins
from types import SimpleNamespace as NS
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction as F
from pathlib import Path

import pytest
from vibecut.analysis.extract import probe_asset_timing
from vibecut.analysis.errors import ModelCancelledError, ExtractionError
from vibecut.core.proc import run_bounded
from vibecut.media import probe


@pytest.mark.parametrize("phase", ["metadata", "packets"])
@pytest.mark.parametrize("mode", ["cancel", "timeout"])
def test_asset_probe_aborts_blocked_child_and_cleans_temp(tmp_path,monkeypatch,phase,mode):
    ready = tmp_path / "ready"
    cancel = threading.Event()
    children = []
    original_popen, original_mkstemp = subprocess.Popen, tempfile.mkstemp
    def spawn(*args,**kwargs):
        child = original_popen(*args,**kwargs)
        children.append(child)
        original_poll = child.poll
        def poll():
            if mode == "cancel" and ready.exists():
                cancel.set()  # the targeted real process has started and announced readiness
            return original_poll()
        child.poll = poll
        return child
    def temporary(*args,**kwargs):
        kwargs["dir"] = tmp_path
        return original_mkstemp(*args,**kwargs)
    monkeypatch.setattr(subprocess,"Popen",spawn)
    monkeypatch.setattr(tempfile,"mkstemp",temporary)
    metadata = {"streams":[{"codec_type":"video","index":0,"time_base":"1/1","avg_frame_rate":"1/1","start_pts":0}],
                "format":{"start_time":"0"}}
    class Tools:
        def ffprobe_argv(self,*args):
            packets = "-show_packets" in args
            blocked = packets == (phase == "packets")
            if blocked:
                script = f"from pathlib import Path; import time; Path({str(ready)!r}).touch(); time.sleep(20)"
            elif packets:
                script = "print('pts=0|flags=K_\\npts=1|flags=__')"
            else:
                script = "print(" + repr(json.dumps(metadata)) + ")"
            return (sys.executable,"-c",script)
        def run(self,argv,**kwargs):
            assert kwargs["cancel_event"] is cancel
            if phase == "packets":
                # Isolate the packet-process deadline from unrelated Python metadata startup.
                return NS(returncode=0,cancelled=False,timed_out=False,stderr_tail=b"",stdout_tail=json.dumps(metadata).encode())
            return run_bounded(argv,**kwargs)
    start = time.monotonic()
    with pytest.raises(ModelCancelledError if mode == "cancel" else ExtractionError) as raised:
        probe_asset_timing(Tools(),"synthetic.mp4",expected_fps=F(1),expected_frames=2,
                           cancel_event=cancel,timeout_s=5 if mode == "cancel" else .8)
    assert raised.value.stage == "asset-timing"
    assert raised.value.code == ("cancelled" if mode == "cancel" else "timeout")
    assert ready.exists() and time.monotonic()-start < 3
    assert children and all(child.returncode is not None for child in children)
    assert not list(tmp_path.glob("vibecut-ffprobe-*"))


def test_cancel_while_scanning_packets_stops_before_full_scan_and_cleans_file(tmp_path,monkeypatch):
    packet_file = tmp_path / "packets.txt"
    packet_file.write_text("".join(f"pts={i}|flags=__\n" for i in range(100)),encoding="utf-8")
    cancel, scanned = threading.Event(), []
    original_open = builtins.open
    def opening(path,*args,**kwargs):
        stream = original_open(path,*args,**kwargs)
        if Path(path) != packet_file:
            return stream
        class InterruptedReader:
            def __enter__(self): return self
            def __exit__(self,*args): stream.close()
            def __iter__(self):
                for index,line in enumerate(stream):
                    scanned.append(index)
                    if index == 1:
                        cancel.set()
                    yield line
        return InterruptedReader()
    monkeypatch.setattr(builtins,"open",opening)
    def prepared(*args,**kwargs):
        assert kwargs["cancel_event"] is cancel
        return packet_file
    monkeypatch.setattr(probe,"_run_ffprobe_to_tempfile",prepared)
    class Tools:
        def ffprobe_argv(self,*args): return ("unused",)
    with pytest.raises(probe.ProbeCancelledError):
        probe._video_packet_pts(Tools(),Path("synthetic.mp4"),0,cancel_event=cancel)
    assert scanned == [0,1] and not packet_file.exists()



def test_asset_probe_cancellation_preserves_analysis_cancelled_diagnostic(tmp_path,monkeypatch):
    import vibecut.analysis.extract as extract
    import vibecut.media.probe as media_probe
    from vibecut.analysis.service import analyze,AnalysisConfig
    from vibecut.analysis.focus import parse_focus
    from vibecut.core.cache import Cache
    cancel = threading.Event()
    def interrupted(*args,**kwargs):
        assert kwargs["cancel_event"] is cancel
        raise ModelCancelledError("asset probe cancelled",stage="asset-timing")
    monkeypatch.setattr(extract,"probe_asset_timing",interrupted)
    monkeypatch.setattr(media_probe,"content_fingerprint",lambda path:"fp")
    source=NS(source_id="source",path="synthetic",content_fingerprint="fp",video=NS(fps=F(1),nb_frames=2))
    asset=NS(source_id="source",kind="proxy",path="synthetic",asset_id="asset",time_map=NS(fps=F(1),asset_start_source_rel=F(0)))
    result=analyze(source,asset,None,Cache(tmp_path),AnalysisConfig(),parse_focus("人物"),cancel_event=cancel)
    assert result.status=="cancelled" and result.plan is None
    assert result.diagnostics[0]["stage"]=="asset-timing" and result.diagnostics[0]["code"]=="cancelled"
