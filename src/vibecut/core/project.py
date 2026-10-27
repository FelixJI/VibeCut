"""Project persistence: schema/revision, atomic replace, single-writer lock.

Layout: a project directory contains ``vibecut.project.json`` plus a
``vibecut.project.json.lock`` sidecar used for cross-process mutual exclusion.

Safety model:
  * saves are written to a temp file *in the same directory*, fsynced, then
    ``os.replace``d - an interrupted save never corrupts the last complete
    version;
  * the on-disk revision is re-checked under the lock right before replace;
    a writer holding a stale revision is rejected (``StaleRevisionError``);
  * concurrent writers are serialised by the lock and the second writer then
    fails the revision check (never silently clobbering).
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import threading
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Optional

from vibecut.core.contracts import (
    CONTRACT_VERSION,
    EditPlan,
    JobEvent,
    MediaSource,
    NormalizedAsset,
    frac_str,
    parse_frac,
    write_json_atomic,
)

PROJECT_SCHEMA_VERSION = 1
PROJECT_FILENAME = "vibecut.project.json"
MAX_JOB_EVENTS = 200


class ProjectError(RuntimeError):
    pass


class ProjectFormatError(ProjectError):
    pass


class StaleRevisionError(ProjectError):
    pass


class ProjectLockedError(ProjectError):
    pass


class Project:
    """In-memory view of a project file; mutations are saved via :meth:`save`."""

    def __init__(
        self,
        path: Path,
        *,
        project_id: str,
        revision: int,
        fps: Fraction,
        sources: dict[str, MediaSource],
        assets: list[NormalizedAsset],
        plans: dict[str, EditPlan],
        jobs: list[JobEvent],
    ) -> None:
        self.path = Path(path)
        self.project_id = project_id
        self.revision = revision
        self.fps = fps
        self.sources = dict(sources)
        self.assets = list(assets)
        self.plans = dict(plans)
        self.jobs = list(jobs)
        self._loaded_revision = revision
        self._lock_owner: Optional[int] = None
        self._lock_fd: Optional[int] = None

    # -- construction ----------------------------------------------------

    @classmethod
    def create(cls, directory: Path, fps: Fraction = Fraction(30000, 1001)) -> "Project":
        path = Path(directory) / PROJECT_FILENAME
        if path.exists():
            raise ProjectError(f"project file already exists: {path}")
        return cls(
            path=path,
            project_id=uuid.uuid4().hex,
            revision=0,
            fps=fps,
            sources={},
            assets=[],
            plans={},
            jobs=[],
        )

    @classmethod
    def load(cls, path: Path) -> "Project":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProjectError(f"project file not found: {path}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProjectFormatError(f"corrupt project file {path}: {exc}") from exc
        if raw.get("project_schema_version") != PROJECT_SCHEMA_VERSION:
            raise ProjectFormatError(
                f"unsupported project schema_version {raw.get('project_schema_version')!r}"
            )
        # The payload's contract_schema_version pins the versioned contract
        # objects nested inside (sources/assets/plans/jobs). Loading a file
        # from an unknown contract schema and re-saving would silently
        # downgrade it (and drop future fields), so reject it up front.
        if raw.get("contract_schema_version") != CONTRACT_VERSION:
            raise ProjectFormatError(
                f"unsupported contract_schema_version {raw.get('contract_schema_version')!r}"
            )
        try:
            project = cls(
                path=path,
                project_id=str(raw["project_id"]),
                revision=int(raw["revision"]),
                fps=parse_frac(raw["fps"], name="fps"),
                sources={s["source_id"]: MediaSource.from_json(s) for s in raw.get("sources", [])},
                assets=[NormalizedAsset.from_json(a) for a in raw.get("assets", [])],
                plans={k: EditPlan.from_json(v) for k, v in raw.get("plans", {}).items()},
                jobs=[JobEvent.from_json(e) for e in raw.get("jobs", [])],
            )
        except (KeyError, ValueError) as exc:
            raise ProjectFormatError(f"invalid project content in {path}: {exc}") from exc
        project._loaded_revision = project.revision
        return project

    # -- serialisation ---------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        return {
            "project_schema_version": PROJECT_SCHEMA_VERSION,
            "contract_schema_version": CONTRACT_VERSION,
            "project_id": self.project_id,
            "revision": self.revision,
            "fps": frac_str(self.fps),
            "sources": [s.to_json() for s in self.sources.values()],
            "assets": [a.to_json() for a in self.assets],
            "plans": {k: p.to_json() for k, p in self.plans.items()},
            "jobs": [e.to_json() for e in self.jobs[-MAX_JOB_EVENTS:]],
        }

    # -- locking ---------------------------------------------------------

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    @contextlib.contextmanager
    def exclusive_lock(self, timeout_s: float = 5.0) -> Iterator[int]:
        """Hold the cross-process project write lock for a bounded time."""
        if self._lock_owner == threading.get_ident():
            yield self._lock_fd
            return
        lock_path = self.lock_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
        deadline = time.monotonic() + timeout_s
        acquired = False
        try:
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise ProjectLockedError(
                            f"project {self.path} is locked by another writer"
                        )
                    time.sleep(0.05)
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                os.write(fd, uuid.uuid4().hex[:8].encode("ascii"))
            except OSError:
                pass
            self._lock_owner = threading.get_ident()
            self._lock_fd = fd
            yield fd
        finally:
            if acquired:
                self._lock_owner = None
                self._lock_fd = None
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)

    # -- mutation & save ---------------------------------------------------

    def save(self) -> int:
        """Atomically persist the project, guarding against stale revisions.

        The on-disk file is re-read under the exclusive lock; if its revision
        has advanced past ``_loaded_revision``, this writer is stale and is
        rejected. Returns the new on-disk revision.
        """
        with self.exclusive_lock():
            self.check_revision()
            payload = self.to_payload()
            new_revision = self._loaded_revision + 1
            payload["revision"] = new_revision
            write_json_atomic(self.path, payload)
            self.revision = new_revision
            self._loaded_revision = new_revision
        return new_revision

    def check_revision(self) -> None:
        """Call under the project lock before beginning a long-running job."""
        on_disk = _read_on_disk_revision(self.path)
        if on_disk is not None and on_disk != self._loaded_revision:
            raise StaleRevisionError(
                f"project moved from revision {self._loaded_revision} to {on_disk}; refusing stale write"
            )

    # -- domain helpers ----------------------------------------------------

    def add_source(self, source: MediaSource) -> None:
        for existing in self.sources.values():
            if Path(existing.path).resolve() == Path(source.path).resolve():
                if existing.content_fingerprint != source.content_fingerprint:
                    raise ProjectError(f"source content changed since ingest: {source.path}")
        if source.source_id in self.sources:
            existing = self.sources[source.source_id]
            if existing.content_fingerprint != source.content_fingerprint:
                raise ProjectError("source_id collision with different fingerprint")
            if Path(existing.path).resolve() != Path(source.path).resolve():
                raise ProjectError("source relocation is unsupported; use the ingested path or a new project")
        self.sources[source.source_id] = source

    def add_job_event(self, event: JobEvent) -> None:
        if event.project_revision != self.revision:
            raise StaleRevisionError(
                f"job event revision {event.project_revision} != project revision {self.revision}; late event rejected"
            )
        self.jobs.append(event)
        if len(self.jobs) > MAX_JOB_EVENTS:
            del self.jobs[:-MAX_JOB_EVENTS]


def _read_on_disk_revision(path: Path) -> Optional[int]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return int(raw.get("revision", -1))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ProjectFormatError(f"cannot read current project revision: {path}") from exc
