import sqlite3

import pytest

from discourselens import Project
from discourselens.errors import (
    DocumentError,
    EvidenceValidationError,
    ScopeViolationError,
    SourceSpanError,
)
from discourselens.types import EvidenceRole


def test_codebook_annotations_are_canonical_span_bound_and_scope_limited(tmp_path):
    project = Project.init(tmp_path / "project")
    _, first = project.import_text("first.txt", "Migration needs public deliberation.")
    _, second = project.import_text("second.txt", "Elephants live in Africa.")
    codebook = project.create_codebook("actor analysis", "Researcher-controlled descriptive codes")
    actor = project.create_code(codebook, "collective actor", "A collective social actor.")
    span = project.create_span(first, 0, 9)
    annotation = project.annotate_span(actor, span, "Initial human coding.")

    in_scope = project.annotations_for_scope(project.snapshot([first]))
    out_scope = project.annotations_for_scope(project.snapshot([second]))

    assert in_scope[0]["id"] == annotation
    assert in_scope[0]["span_id"] == span
    assert in_scope[0]["label"] == "collective actor"
    assert out_scope == []


def test_code_parent_and_annotation_integrity_are_enforced(tmp_path):
    project = Project.init(tmp_path / "project")
    first_book = project.create_codebook("one")
    second_book = project.create_codebook("two")
    parent = project.create_code(first_book, "parent")
    _, revision = project.import_text("text.txt", "Canonical text.")

    with pytest.raises(DocumentError, match="same codebook"):
        project.create_code(second_book, "child", parent_code_id=parent)
    with pytest.raises(DocumentError, match="same codebook"):
        project.create_code(first_book, "orphan", parent_code_id="")
    assert project.db.execute("SELECT 1 FROM codes WHERE label='orphan'").fetchone() is None
    with pytest.raises(SourceSpanError, match="unknown span"):
        project.annotate_span(parent, "unknown")

    span = project.create_span(revision, 0, 9)
    project.annotate_span(parent, span)
    with pytest.raises(DocumentError, match="already attached"):
        project.annotate_span(parent, span)


def test_annotation_rejects_span_with_missing_document_revision(tmp_path):
    project = Project.init(tmp_path / "project")
    codebook = project.create_codebook("analysis")
    code = project.create_code(codebook, "observation")
    _, revision = project.import_text("text.txt", "Canonical text.")
    span = project.create_span(revision, 0, 9)

    # Simulate a dangling span left by a damaged or externally edited project DB.
    project.db.execute("DELETE FROM document_revisions WHERE id=?", (revision,))
    project.db.commit()

    with pytest.raises(SourceSpanError, match="unknown span"):
        project.annotate_span(code, span)
    assert project.db.execute("SELECT 1 FROM annotations").fetchone() is None


def test_coding_scope_rejects_lineage_to_a_missing_document_revision(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("text.txt", "Canonical text.")
    scope = project.snapshot([revision])
    codebook = project.create_codebook("analysis")
    code = project.create_code(codebook, "observation")
    span = project.create_span(revision, 0, 9)
    project.annotate_span(code, span)
    project.create_research_memo(scope, "Selection rationale", "The selected revision.")

    # A damaged project database can lose a revision while retaining scopes/spans.
    project.db.execute("DELETE FROM document_revisions WHERE id=?", (revision,))
    project.db.commit()

    with pytest.raises(ScopeViolationError, match="unknown document revision"):
        project.annotations_for_scope(scope)
    with pytest.raises(ScopeViolationError, match="unknown document revision"):
        project.research_memos(scope)
    with pytest.raises(ScopeViolationError, match="unknown document revision"):
        project.create_research_memo(scope, "Later memo", "Must not persist.")
    assert project.db.execute("SELECT COUNT(*) FROM research_memos").fetchone()[0] == 1
    project.close()


def test_research_memo_is_anchored_to_one_immutable_scope(tmp_path):
    project = Project.init(tmp_path / "project")
    _, first = project.import_text("first.txt", "First selected text.")
    _, second = project.import_text("second.txt", "Later text.")
    scope = project.snapshot([first])
    memo = project.create_research_memo(scope, "Selection rationale", "Initial comparison set.")

    assert project.research_memos(scope)[0]["id"] == memo
    assert project.research_memos(project.snapshot([second])) == []
    with pytest.raises(DocumentError, match="title and content"):
        project.create_research_memo(scope, "", "missing title")


def test_finding_review_history_keeps_append_order_for_timestamp_ties(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project = Project.init(project_root)
    _, revision = project.import_text("source.txt", "Review history remains auditable.")
    run = project.create_run(project.snapshot([revision]), {"profile": "review-history"})
    finding = project.create_finding(run, "fixture", "1.0.0", "observation", "A finding.")
    project.finalize_run(run)

    review_ids = iter(("z-first", "a-second"))
    with monkeypatch.context() as patch:
        patch.setattr("discourselens.project.uid", lambda: next(review_ids))
        patch.setattr("discourselens.project.now", lambda: "2026-01-01T00:00:00+00:00")
        first = project.review_finding(finding, "ACCEPTED", "First decision.")
        second = project.review_finding(finding, "REJECTED", "Later decision.")
    project.close()

    database_path = project_root / ".discourselens" / "project.sqlite"
    vacuumed = sqlite3.connect(database_path)
    try:
        vacuumed.execute("VACUUM")
    finally:
        vacuumed.close()

    reopened = Project.open(project_root)
    history = reopened.finding_reviews(finding)
    assert [row["id"] for row in history] == [first, second]
    assert [row["status"] for row in history] == ["ACCEPTED", "REJECTED"]
    assert [row["review_order"] for row in history] == [1, 2]
    assert reopened.get_finding(finding)["review_status"] == "REJECTED"
    reopened.close()


def test_finding_review_status_comes_from_history_without_rewriting_generated_content(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "The generated finding stays unchanged.")
    run = project.create_run(project.snapshot([revision]), {"profile": "review-history"})
    evidence = project.textual_evidence(project.create_span(revision, 0, 12))
    finding = project.create_finding(
        run, "fixture", "1.0.0", "observation", "The generated observation."
    )
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    project.finalize_run(run)

    original_finding = dict(project.get_finding(finding))
    original_evidence = [dict(row) for row in project.finding_evidence(finding)]
    for status in ("ACCEPTED", "MODIFIED", "REJECTED"):
        project.review_finding(finding, status, f"{status.lower()} review")

    reviews = project.finding_reviews(finding)
    assert [row["status"] for row in reviews] == ["ACCEPTED", "MODIFIED", "REJECTED"]
    assert [row["review_order"] for row in reviews] == [1, 2, 3]
    assert [dict(row) for row in project.finding_evidence(finding)] == original_evidence
    after_reviews = dict(project.get_finding(finding))
    assert {
        key: value for key, value in after_reviews.items() if key != "review_status"
    } == {key: value for key, value in original_finding.items() if key != "review_status"}

    # The stored status is only a legacy/cache field; the audit history is authoritative.
    project.db.execute("UPDATE findings SET review_status='ACCEPTED' WHERE id=?", (finding,))
    project.db.commit()
    assert project.get_finding(finding)["review_status"] == "REJECTED"

    with pytest.raises(EvidenceValidationError, match="unknown finding"):
        project.review_finding("missing-finding", "ACCEPTED")
    assert project.finding_reviews(finding)[-1]["status"] == "REJECTED"
    project.close()


@pytest.mark.parametrize("limit", [True, 1.5, "1", 0, 1_001])
def test_finding_review_pages_reject_invalid_limits(tmp_path, limit):
    project = Project.init(tmp_path / "invalid-review-limit")
    _, revision = project.import_text("source.txt", "Review history.")
    run = project.create_run(project.snapshot([revision]), {"profile": "review-history"})
    finding = project.create_finding(run, "fixture", "1.0.0", "observation", "A finding.")
    project.finalize_run(run)

    with pytest.raises(EvidenceValidationError, match="finding review limit"):
        project.finding_reviews(finding, limit=limit)

    project.close()
