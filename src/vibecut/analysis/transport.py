"""Loopback-only HTTP transport for the local model service.

Network boundary (task #5 contract):
  * only explicit loopback endpoints (http://127.0.0.1:PORT, http://[::1]:PORT,
    http://localhost:PORT) may be connected to - everything else is rejected
    before any socket is opened;
  * redirects are NEVER followed (a 3xx reply is an error);
  * there is no cloud fallback path anywhere in this module;
  * requests are bounded by an overall deadline and a max response size.

Cancellation is honoured before connecting and during the
response; a server that hangs mid-body is bounded by the request deadline.
"""

from __future__ import annotations

import http.client
import json
import select
import socket
import threading
import time
from urllib.parse import urlsplit

from vibecut.analysis.errors import (
    AIUnavailableError,
    EndpointError,
    ModelAPIError,
    ModelCancelledError,
    ModelTimeoutError,
)

ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_CONNECT_TIMEOUT_S = 0.5
_CHUNK = 65536

ForbiddenRedirect = EndpointError  # readability alias


def validate_endpoint(base_url: str) -> str:
    """Validate and normalize an explicit loopback base URL (no path/query)."""
    try:
        parts = urlsplit(base_url)
    except ValueError as exc:
        raise EndpointError(f"cannot parse endpoint {base_url!r}", code="bad-endpoint") from exc
    if parts.scheme != "http":
        raise EndpointError("endpoint must be http:// (loopback only)", code="not-http")
    host = (parts.hostname or "").lower()
    if parts.username or parts.password:
        raise EndpointError("endpoint must not carry credentials", code="credentials")
    if parts.query or parts.fragment:
        raise EndpointError("endpoint must not carry query/fragment", code="query")
    if parts.path not in ("", "/"):
        raise EndpointError("endpoint must be a bare base URL (no path)", code="path")
    if host not in ALLOWED_LOOPBACK_HOSTS:
        raise EndpointError(
            f"endpoint host {host!r} is not an explicit loopback address "
            f"({', '.join(sorted(ALLOWED_LOOPBACK_HOSTS))}); cloud/public endpoints are forbidden",
            code="not-loopback",
        )
    try:
        port = parts.port
    except ValueError as exc:
        raise EndpointError("invalid endpoint port", code="bad-port") from exc
    if port is None:
        raise EndpointError("endpoint needs an explicit port", code="no-port")
    host_str = f"[{host}]" if ":" in host else host
    return f"http://{host_str}:{port}"


class _BoundedSocket(socket.socket):
    """Poll reads/writes so Windows cancellation never depends on shutdown()."""

    def __init__(self, original, check_abort):
        super().__init__(original.family, original.type, original.proto, fileno=original.detach())
        self.setblocking(False)
        self.check_abort = check_abort

    def _ready(self, *, writing=False):
        while True:
            self.check_abort()
            readers, writers, _ = select.select([] if writing else [self], [self] if writing else [], [], 0.02)
            if readers or writers:
                self.check_abort()
                return

    def recv_into(self, buffer, nbytes=0, flags=0):
        while True:
            self._ready()
            try:
                return super().recv_into(buffer, nbytes, flags)
            except BlockingIOError:
                continue

    def sendall(self, data, flags=0):
        remaining = memoryview(data)
        while remaining:
            self._ready(writing=True)
            try:
                sent = self.send(remaining[:65536], flags)
            except BlockingIOError:
                continue
            if sent == 0:
                raise ConnectionResetError("model socket closed during send")
            remaining = remaining[sent:]


class LoopbackTransport:
    """Minimal http.client transport; never follows redirects, never retries."""

    def __init__(self, base_url: str, *, request_timeout_s: float = 120.0) -> None:
        self.base_url = validate_endpoint(base_url)
        self.request_timeout_s = float(request_timeout_s)
        parts = urlsplit(self.base_url)
        self._host = parts.hostname or ""
        self._port = parts.port or 80

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        *,
        cancel_event: threading.Event | None = None,
        timeout_s: float | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        if cancel_event is not None and cancel_event.is_set():
            raise ModelCancelledError("cancelled before request")
        if not path.startswith("/"):
            raise EndpointError("request path must start with /", code="path")
        timeout = self.request_timeout_s if timeout_s is None else min(self.request_timeout_s, timeout_s)
        deadline = time.monotonic() + timeout
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "Content-Length": str(len(body)) if body is not None else "0"}
        # localhost is resolved to an explicit loopback, never via DNS/hosts overrides.
        host = "127.0.0.1" if self._host == "localhost" else self._host
        conn = http.client.HTTPConnection(host, self._port, timeout=min(timeout, _CONNECT_TIMEOUT_S))
        connected = False
        resp = None

        def check_abort():
            if cancel_event is not None and cancel_event.is_set():
                raise ModelCancelledError("cancelled during model request")
            if time.monotonic() >= deadline:
                raise ModelTimeoutError(f"model request deadline exceeded: {path}")

        try:
            check_abort()
            conn.connect()
            connected = True
            conn.sock = _BoundedSocket(conn.sock, check_abort)
            check_abort()
            # Never retry request(): a reset may happen after a POST was accepted.
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            check_abort()
            if resp.status in REDIRECT_STATUSES:
                raise EndpointError(
                    f"redirects are forbidden: {resp.status} -> {resp.getheader('Location')!r}",
                    code="redirect-forbidden",
                )
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = resp.read(_CHUNK)
                check_abort()
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_RESPONSE_BYTES:
                    raise EndpointError("response exceeds bounded size", code="response-too-large")
                chunks.append(chunk)
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}, b"".join(chunks)
        except socket.timeout as exc:
            check_abort()
            if not connected:
                raise AIUnavailableError("loopback model service did not accept a connection", code="not-started") from exc
            raise ModelTimeoutError(f"model request timed out: {path}") from exc
        except (OSError, http.client.HTTPException) as exc:
            check_abort()
            if isinstance(exc, ConnectionRefusedError):
                raise AIUnavailableError(
                    f"loopback model service not started: {self.base_url}", code="not-started"
                ) from exc
            raise ModelAPIError("loopback model connection failed", code="connection-error") from exc
        finally:
            if resp is not None:
                resp.close()
            conn.close()

    def get_json(self, path: str, *, cancel_event: threading.Event | None = None, timeout_s: float | None = None) -> tuple[int, object]:
        status, _headers, body = self.request("GET", path, cancel_event=cancel_event, timeout_s=timeout_s)
        if not body:
            return status, None
        try:
            return status, json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return status, {"_raw": body[:200].decode("utf-8", "replace")}

    def post_json(
        self, path: str, payload: dict, *, cancel_event: threading.Event | None = None, timeout_s: float | None = None
    ) -> tuple[int, object]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        status, _headers, raw = self.request("POST", path, body, cancel_event=cancel_event, timeout_s=timeout_s)
        try:
            return status, (json.loads(raw.decode("utf-8")) if raw else None)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ModelAPIError(
                f"non-JSON response from model service ({len(raw)} bytes)", code="bad-json"
            ) from exc
