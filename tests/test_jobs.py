"""Job event publication: revision guard against late events."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vibecut.core import jobs  # noqa: E402
from vibecut.core.project import Project, StaleRevisionError  # noqa: E402


def test_publish_attaches_event_to_current_revision(tmp_path):
    p = Project.create(tmp_path / "j")
    p.save()
    ev = jobs.make_event("job1", 1, "queued")
    jobs.publish(p, ev)
    assert p.jobs[-1] is ev


def test_late_event_with_old_revision_rejected(tmp_path):
    p = Project.create(tmp_path / "j")
    p.save()  # revision 1
    p.save()  # revision 2 (project moved on)
    late = jobs.make_event("job1", 1, "succeeded", artifact="x.mp4")
    with pytest.raises(StaleRevisionError):
        jobs.publish(p, late)
    # nothing from the late event landed
    assert all(e.job_id != "job1" for e in p.jobs)


def test_event_ordering_contract_documents_success_after_validation(tmp_path):
    """The succeeded status is only meaningful after artifact validation.

    render.static is responsible for only returning success post-validation;
    here we assert the event schema distinguishes statuses and carries the
    artifact field for succeeded only when provided.
    """
    ev = jobs.make_event("job2", 5, "succeeded", artifact="a.mp4")
    assert ev.artifact == "a.mp4"
    ev2 = jobs.make_event("job2", 5, "failed", error="boom")
    assert ev2.error == "boom" and ev2.artifact is None
    assert ev.job_id == ev2.job_id
