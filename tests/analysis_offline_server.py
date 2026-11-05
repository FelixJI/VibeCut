"""Offline loopback fixture server standing in for LM Studio (tests only).

A tiny ``http.server`` bound to 127.0.0.1 on an ephemeral port serving
scripted canned responses for ``/api/v0/models`` (LM Studio native, with
state/type) and ``/v1/chat/completions`` (+ the portable ``/v1/models``).
No external network is ever touched; prompts/images are counted, never
logged, satisfying the no-leakage policy of the offline fixtures.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "qwen3.8-27b@q4_k_xl"


def _content_response(payload: dict, *, model: str = MODEL_ID) -> dict:
    return {
        "id": "chatcmpl-fixture",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 128, "completion_tokens": 64},
    }


class OfflineLM:
    """Scripted loopback model service."""

    def __init__(self, *, model_state: str = "loaded", vision: bool = True, model_id: str = MODEL_ID) -> None:
        self.model_state = model_state
        self.vision = vision
        self.model_id = model_id
        self.chat_script: list[dict] = []  # each: {"content": dict} | {"raw": str} | {"status": int} | {"delay": float}
        self.chat_calls: list[dict] = []  # metering-only records (counts/sizes, no content)
        self.on_served = None  # optional callable(n_served_chat_calls) after each response
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _send(self, status: int, payload: dict, headers: dict | None = None) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if self.path == "/api/v0/models":
                    self._send(200, {"data": [
                        {"id": outer.model_id, "state": outer.model_state, "type": "vlm" if outer.vision else "llm"},
                        {"id": "other-model", "state": "not-loaded", "type": "llm"},
                    ]})
                    return
                if self.path == "/v1/models":
                    self._send(200, {"data": [{"id": outer.model_id}]})
                    return
                self._send(404, {"error": {"message": "not found"}})

            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b""
                if self.path != "/v1/chat/completions":
                    self._send(404, {"error": {"message": "not found"}})
                    return
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    payload = {}
                messages = payload.get("messages", [])
                images = sum(
                    1 for m in messages if isinstance(m.get("content"), list)
                    for part in m["content"] if isinstance(part, dict) and part.get("type") == "image_url"
                )
                record = {"path": self.path, "bytes": len(raw), "images": images,
                          "model": payload.get("model"), "response_format": bool(payload.get("response_format"))}
                with outer._lock:
                    outer.chat_calls.append(record)
                    script = outer.chat_script.pop(0) if outer.chat_script else {"content": {"observations": []}}
                if script.get("delay"):
                    import time

                    time.sleep(script["delay"])
                if "status" in script and script["status"] != 200:
                    self._send(script["status"], {"error": {"message": script.get("message", "boom")}})
                    return
                if "raw" in script:
                    body = script["raw"].encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    try:
                        if script.get("drip_s"):
                            import time
                            for byte in body:
                                self.wfile.write(bytes([byte]))
                                self.wfile.flush()
                                time.sleep(script["drip_s"])
                        else:
                            self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                    return
                if "redirect" in script:
                    self.send_response(script.get("redirect", 302))
                    self.send_header("Location", script["redirect_location"])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self._send(200, _content_response(script["content"], model=script.get("model", payload.get("model", outer.model_id))))
                if outer.on_served is not None:
                    try:
                        outer.on_served(len(outer.chat_calls))
                    except Exception:
                        pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://127.0.0.1:{port}"

    def start(self) -> "OfflineLM":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


def observe_payload(img_ids: list[str], *, scene: str = "synthetic scene", quality: str = "good",
                    motion: str = "slight", subjects: list[str] | None = None) -> dict:
    subjects = subjects if subjects is not None else ["synthetic marker"]
    return {"observations": [
        {"img_id": i, "subjects": subjects, "scene": scene, "quality": quality, "motion_hint": motion}
        for i in img_ids
    ]}


def select_payload(items: list[dict]) -> dict:
    return {"selections": items}
