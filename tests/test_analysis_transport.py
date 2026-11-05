"""Transport network boundary: loopback-only, no redirects, refused/timeout."""

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis_offline_server import OfflineLM

from vibecut.analysis.errors import AIUnavailableError, EndpointError, ModelCancelledError, ModelTimeoutError
from vibecut.analysis.transport import LoopbackTransport, validate_endpoint


@pytest.mark.parametrize("url", [
    "http://8.8.8.8:1234",
    "http://api.openai.com/v1",
    "https://127.0.0.1:1234",
    "http://user:pass@127.0.0.1:1234",
    "http://127.0.0.1:1234/path?x=1",
    "http://10.0.0.1:1234",
    "http://localhost.evil.com:1234",
    "ftp://127.0.0.1:1234",
])
def test_non_loopback_or_invalid_endpoints_rejected(url):
    with pytest.raises(EndpointError):
        validate_endpoint(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:1234", "http://localhost:1234", "http://[::1]:1234"])
def test_loopback_endpoints_accepted(url):
    assert validate_endpoint(url).startswith("http://")


def test_no_socket_opened_for_forbidden_host():
    # validation happens before any connection attempt
    with pytest.raises(EndpointError):
        LoopbackTransport("http://93.184.216.34:1")


def test_connection_refused_maps_to_not_started():
    t = LoopbackTransport("http://127.0.0.1:1", request_timeout_s=0.6)
    with pytest.raises(AIUnavailableError) as exc:
        t.request("GET", "/api/v0/models")
    assert exc.value.code == "not-started"


def test_redirect_never_followed():
    lm = OfflineLM().start()
    try:
        lm.chat_script.append({"redirect": 302, "redirect_location": "http://169.254.169.254/latest/meta-data"})
        t = LoopbackTransport(lm.base_url, request_timeout_s=5.0)
        with pytest.raises(EndpointError) as exc:
            t.request("POST", "/v1/chat/completions", body=b"{}")
        assert exc.value.code == "redirect-forbidden"
    finally:
        lm.stop()


def test_timeout_bounded():
    lm = OfflineLM().start()
    try:
        lm.chat_script.append({"delay": 1.2, "content": {"selections": []}})
        t = LoopbackTransport(lm.base_url, request_timeout_s=0.5)
        with pytest.raises(ModelTimeoutError):
            t.request("POST", "/v1/chat/completions", body=b"{}")
    finally:
        lm.stop()


def test_cancel_before_request_immediate():
    lm = OfflineLM().start()
    try:
        t = LoopbackTransport(lm.base_url, request_timeout_s=5.0)
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(ModelCancelledError):
            t.request("GET", "/api/v0/models", cancel_event=cancel)
        assert lm.chat_calls == []
    finally:
        lm.stop()


def test_response_size_bounded():
    lm = OfflineLM().start()
    try:
        big = "x" * (9 * 1024 * 1024)
        lm.chat_script.append({"raw": '{"pad": "' + big + '"}'})
        t = LoopbackTransport(lm.base_url, request_timeout_s=10.0)
        with pytest.raises(EndpointError) as exc:
            t.request("POST", "/v1/chat/completions", body=b"{}")
        assert exc.value.code == "response-too-large"
    finally:
        lm.stop()


def test_get_json_probe_payload():
    lm = OfflineLM().start()
    try:
        t = LoopbackTransport(lm.base_url, request_timeout_s=5.0)
        status, data = t.get_json("/api/v0/models")
        assert status == 200 and isinstance(data, dict)
        ids = [m["id"] for m in data["data"]]
        assert "qwen3.8-27b@q4_k_xl" in ids
        assert any(m.get("state") == "loaded" and "vlm" in str(m.get("type", "")) for m in data["data"])
    finally:
        lm.stop()


@pytest.mark.parametrize("phase", ["headers", "body"])
@pytest.mark.parametrize("abort", ["deadline", "cancel"])
def test_inflight_headers_and_slow_body_abort_promptly(phase, abort):
    lm = OfflineLM().start()
    cancel = threading.Event()
    timer = None
    try:
        lm.chat_script.append({"delay": 1.5, "content": {"selections": []}} if phase == "headers"
                              else {"raw": "x" * 200, "drip_s": 0.02})
        timeout = 0.2 if abort == "deadline" else 5.0
        transport = LoopbackTransport(lm.base_url, request_timeout_s=timeout)
        if abort == "cancel":
            timer = threading.Timer(0.2, cancel.set)
            timer.start()
        started = time.monotonic()
        expected = ModelTimeoutError if abort == "deadline" else ModelCancelledError
        with pytest.raises(expected):
            transport.request("POST", "/v1/chat/completions", b"{}", cancel_event=cancel)
        assert time.monotonic() - started < 1.0
        assert len(lm.chat_calls) == 1
    finally:
        if timer:
            timer.cancel()
        lm.stop()


def test_post_is_not_repeated_after_request_reset(monkeypatch):
    import http.client
    from vibecut.analysis.errors import ModelAPIError

    lm = OfflineLM().start()
    original = http.client.HTTPConnection.request
    attempts = []
    def accepted_then_reset(connection, *args, **kwargs):
        attempts.append(1)
        original(connection, *args, **kwargs)
        raise ConnectionResetError("reset after submission")
    monkeypatch.setattr(http.client.HTTPConnection, "request", accepted_then_reset)
    try:
        with pytest.raises(ModelAPIError):
            LoopbackTransport(lm.base_url, request_timeout_s=0.4).request("POST", "/v1/chat/completions", b"{}")
        assert len(attempts) == 1
    finally:
        lm.stop()
