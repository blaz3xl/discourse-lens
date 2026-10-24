import sqlite3

import pytest

from discourselens import Project
from discourselens.errors import DocumentError, EvidenceValidationError
from discourselens.types import EvidenceRole


def test_completed_analysis_run_freezes_findings_evidence_and_lineage(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "The policy changed. A counterexample follows.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "immutability-test"})
    first_span = project.create_span(revision, 0, len("The policy changed."))
    second_span = project.create_span(
        revision,
        len("The policy changed. "),
        len("The policy changed. A counterexample follows."),
    )
    first_evidence = project.textual_evidence(first_span)
    second_evidence = project.textual_evidence(second_span)
    finding_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "policy-change",
        "The source describes a policy change.",
    )
    source_finding_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "counterexample",
        "A separate source passage provides context.",
    )
    late_source_finding_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "late-source",
        "This source must not be linked after finalization.",
    )
    project.attach_evidence(finding_id, first_evidence, EvidenceRole.SUPPORT)
    project.attach_evidence(source_finding_id, second_evidence, EvidenceRole.CONTEXT)
    project.link_finding_source(finding_id, source_finding_id, "SYNTHESIZES")

    project.finalize_run(run_id)

    finding = project.get_finding(finding_id)
    completed_at = project.db.execute(
        "SELECT completed_at FROM analysis_runs WHERE id=?", (run_id,)
    ).fetchone()["completed_at"]
    assert finding["finalized_at"] == completed_at
    with pytest.raises(DocumentError, match="immutable or unknown"):
        project.finalize_run(run_id)
    with pytest.raises(EvidenceValidationError, match="immutable analysis run"):
        project.create_finding(run_id, "fixture", "1.0.0", "late", "Must be rejected.")
    with pytest.raises(EvidenceValidationError, match="immutable analysis run"):
        project.attach_evidence(finding_id, second_evidence, EvidenceRole.COUNTEREVIDENCE)
    with pytest.raises(EvidenceValidationError, match="immutable analysis run"):
        project.link_finding_source(finding_id, late_source_finding_id, "SYNTHESIZES")

    with pytest.raises(sqlite3.IntegrityError, match="immutable analysis run"):
        project.db.execute(
            "INSERT INTO finding_evidence VALUES(?,?,?,?)",
            (finding_id, second_evidence, EvidenceRole.COUNTEREVIDENCE, None),
        )
    with pytest.raises(sqlite3.IntegrityError, match="immutable analysis run"):
        project.db.execute(
            "UPDATE findings SET observation=? WHERE id=?",
            ("silently rewritten", finding_id),
        )
    with pytest.raises(sqlite3.IntegrityError, match="immutable analysis run"):
        project.db.execute("DELETE FROM findings WHERE id=?", (finding_id,))

    # SQLite leaves the surrounding transaction open after a rejected statement.
    # Clear it before exercising the application-level append-only review API.
    project.db.rollback()

    # Human review is an append-only layer and remains available after generation.
    review_id = project.review_finding(finding_id, "MODIFIED", "Researcher review.")
    assert project.finding_reviews(finding_id)[0]["id"] == review_id
    project.close()


def test_pending_analysis_run_can_be_completed_after_findings_are_created(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "Evidence remains available.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "pending"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "observation", "An observation."
    )

    project.finalize_run(run_id)

    assert project.get_finding(finding_id)["finalized_at"] is not None
    project.close()
