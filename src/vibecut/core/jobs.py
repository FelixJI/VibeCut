"""Job lifecycle: event construction and revision-guarded publication.

Ordering contract (invariant): a ``succeeded`` event may only be published
after the artifact has been (a) written to a same-directory temp file,
(b) produced by a child process that actually exited, (c) probe-validated and
(d) committed without overwriting anything. ``render.static`` enforces this;
this module only provides guards so late/stale events are rejected.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from vibecut.core.contracts import CONTRACT_VERSION, JobEvent
from vibecut.core.project import Project, StaleRevisionError


def new_job_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def make_event(
    job_id: str, project_revision: int, status: str, *, artifact: str | None = None, error: str | None = None
) -> JobEvent:
    return JobEvent(
        schema_version=CONTRACT_VERSION,
        job_id=job_id,
        project_revision=project_revision,
        status=status,
        at=now_iso(),
        artifact=artifact,
        error=error,
    )


def publish(project: Project, event: JobEvent) -> None:
    """Attach ``event`` to ``project`` if it is not stale.

    Raises ``StaleRevisionError`` for events produced against an older project
    revision so they can never overwrite newer state.
    """
    project.add_job_event(event)
