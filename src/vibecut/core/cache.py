"""Derived-asset cache with content-addressed keys.

Key semantics (invariant): cache identity is bound to the SOURCE CONTENT
(via MediaSource.content_fingerprint), the TOOL identity (ffmpeg/ffprobe
build), the pipeline CONFIG and the colour-space contract. It is never bound
to the edit revision and never to a filename:
  * same content under a new filename -> same key (it *is* the same asset);
  * different content under the same filename -> different key;
  * bumping the project/edit revision alone -> unchanged keys (normalised
    masters are not needlessly invalidated);
  * changing render config (camera, cut window, encoding) only invalidates
    render outputs, not the normalised master; changing tool or colour
    contract invalidates both.

Entries are committed atomically: payload is written to a temp file,
validated by the caller-supplied validator, then ``os.replace``d. A cancelled
or crashed producer leaves only ``.tmp-*`` files which ``get`` never matches,
so partial caches cannot be hit.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional


class CacheError(RuntimeError):
    pass


def _hash_parts(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(json.dumps(part, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def normalized_key(*, source_fingerprint: str, source_projection: str, tool_identity: str, kind: str, color: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    """Cache key for a normalised asset (canonical master / proxy).

    Bound to source content + tool + projection + colour contract (+ the
    generating configuration, e.g. proxy dimensions); NOT to the edit plan
    or project revision.
    """
    return _hash_parts(
        {"namespace": "normalized", "schema": 1},
        source_fingerprint,
        source_projection,
        tool_identity,
        kind,
        color,
        config or {},
    )


def render_key(*, normalized: str, config_identity: dict[str, Any]) -> str:
    """Cache key for a rendered artifact: normalised key + render config."""
    return _hash_parts({"namespace": "render", "schema": 1}, normalized, config_identity)


class Cache:
    """File-backed cache storing one JSON metadata doc + one binary payload per key."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _meta_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def _payload_path(self, key: str, meta: Optional[dict[str, Any]] = None) -> Path:
        if meta is None:
            meta = self._read_meta(key) or {}
        name = meta.get("payload_name", f"{key}.bin")
        if not isinstance(name, str) or Path(name).name != name:
            raise CacheError("invalid cache payload reference")
        return self.root / name

    def has(self, key: str) -> bool:
        """True if a committed entry exists (payload too, when one was written)."""
        meta = self._read_meta(key)
        if meta is None:
            return False
        if meta.get("has_payload"):
            return self._payload_path(key, meta).is_file()
        return True

    def _read_meta(self, key: str) -> Optional[dict[str, Any]]:
        try:
            meta = json.loads(self._meta_path(key).read_text(encoding="utf-8"))
            return meta if isinstance(meta, dict) else None
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, OSError):
            return None

    def get(self, key: str) -> Optional[tuple[dict[str, Any], Optional[Path]]]:
        """Return (meta, payload_path|None) or None. Never matches partial writes."""
        meta = self._read_meta(key)
        if meta is None:
            return None
        payload = self._payload_path(key, meta) if meta.get("has_payload") else None
        if payload is not None and not payload.is_file():
            return None  # payload vanished -> treat as absent
        return meta, payload

    def put_atomic(
        self,
        key: str,
        meta: dict[str, Any],
        payload_writer: Optional[Callable[[Path], None]] = None,
        *,
        validator: Optional[Callable[[Path], None]] = None,
    ) -> Optional[Path]:
        """Write payload via ``payload_writer(tmp)`` (optional), validate, commit.

        A unique immutable payload is written and validated first. One atomic
        manifest replacement publishes its filename together with metadata.
        Failure before that replacement preserves the previous complete entry.
        A crash can leave an unreferenced payload, which is never a cache hit.
        Old generations remain available to readers that already opened them.

        Entries without a payload_writer are manifest-only.
        """
        meta = {**meta, "has_payload": payload_writer is not None}
        self.root.mkdir(parents=True, exist_ok=True)
        committed = False
        payload: Optional[Path] = None
        tmp: Optional[Path] = None
        if payload_writer is not None:
            fd, tmp_name = tempfile.mkstemp(prefix=f"{key}.payload-", suffix=".bin", dir=str(self.root))
            os.close(fd)
            tmp = Path(tmp_name)
        meta_tmp: Optional[Path] = None
        try:
            if tmp is not None:
                payload_writer(tmp)
                if validator is not None:
                    validator(tmp)
                # One manifest replace publishes both metadata and immutable payload.
                payload = tmp
                meta["payload_name"] = payload.name
            meta_fd, meta_tmp_name = tempfile.mkstemp(prefix=f"{key}.meta-", dir=str(self.root))
            meta_tmp = Path(meta_tmp_name)
            with os.fdopen(meta_fd, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=1, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(meta_tmp, self._meta_path(key))
            committed = True
        finally:
            if tmp is not None and not committed:
                tmp.unlink(missing_ok=True)
            if meta_tmp is not None and meta_tmp.exists():
                meta_tmp.unlink(missing_ok=True)
        return payload

    def invalidate(self, key: str) -> bool:
        """Remove an entry (used on abort so partial/failed results never persist)."""
        existed = self.has(key)
        payload = self._payload_path(key)
        self._meta_path(key).unlink(missing_ok=True)
        payload.unlink(missing_ok=True)
        for tmp in self.root.glob(f"{key}.tmp-*"):
            tmp.unlink(missing_ok=True)
        for tmp in self.root.glob(f"{key}.meta-*"):
            tmp.unlink(missing_ok=True)
        return existed
