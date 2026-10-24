import sqlite3
from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError
from discourselens.scope_retrieval import retrieve_framework_context
from discourselens.vectors import VectorChunk, build_vector_index, search_vectors


def test_framework_retrieval_rejects_seed_outside_metadata_selected_scope(tmp_path):
    project = Project.init(tmp_path / "project")
    selected_document, _ = project.import_text(
        "selected.txt", "The selected report discusses policy framing."
    )
    excluded_document, excluded_revision = project.import_text(
        "excluded.txt", "The excluded report discusses xenolexicon."
    )
    project.set_document_metadata(selected_document, "collection", "selected")
    project.set_document_metadata(excluded_document, "collection", "excluded")
    scope_id, revisions = project.snapshot_by_document_metadata({"collection": "selected"})
    assert revisions == project.scope_revisions(scope_id)

    start = project.revision_text(excluded_revision).index("xenolexicon")
    span_id = project.create_span(excluded_revision, start, start + len("xenolexicon"))
    evidence_id = project.textual_evidence(span_id)
    seed = project.db.execute(
        "SELECT id,span_id FROM evidence_artifacts WHERE id=?", (evidence_id,)
    ).fetchone()

    with pytest.raises(ScopeViolationError, match="seed evidence is outside"):
        retrieve_framework_context(
            project,
            scope_id,
            [seed],
            phenomenon="policy",
            observation="framing",
        )


def test_framework_retrieval_excludes_every_paragraph_overlapped_by_seed_span(tmp_path):
    project = Project.init(tmp_path / "project")
    text = (
        "Xenolexicon appears in the first seed paragraph.\n\n"
        "Xenolexicon appears in the second seed paragraph.\n\n"
        "Xenolexicon appears in additional context."
    )
    _, revision = project.import_text("selected.txt", text)
    scope_id = project.snapshot([revision])
    seed_end = text.index("\n\nXenolexicon")
    second_paragraph_end = text.index("\n\n", seed_end + 2)
    span_id = project.create_span(revision, 0, second_paragraph_end)
    evidence_id = project.textual_evidence(span_id)
    seed = project.db.execute(
        "SELECT id,span_id FROM evidence_artifacts WHERE id=?", (evidence_id,)
    ).fetchone()

    result = retrieve_framework_context(
        project,
        scope_id,
        [seed],
        phenomenon="xenolexicon",
        observation="",
        query_texts=("xenolexicon",),
    )

    paragraphs = project.db.execute(
        "SELECT id,start_offset,end_offset,text FROM paragraphs "
        "WHERE revision_id=? ORDER BY start_offset",
        (revision,),
    ).fetchall()
    overlapped_ids = {
        row["id"]
        for row in paragraphs
        if row["start_offset"] < second_paragraph_end and row["end_offset"] > 0
    }
    assert len(overlapped_ids) > 1
    candidate_ids = [candidate.paragraph_id for candidate in result.candidates]
    assert not overlapped_ids.intersection(candidate_ids)
    assert candidate_ids == [
        row["id"]
        for row in paragraphs
        if row["id"] not in overlapped_ids and "xenolexicon" in row["text"].casefold()
    ]


def test_framework_retrieval_reports_output_truncation_separately_from_scan_cap(tmp_path):
    project = Project.init(tmp_path / "project")
    text = (
        "Migration policy passage alpha.\n\n"
        "Migration policy passage beta.\n\n"
        "Migration policy passage gamma.\n\n"
        "Migration policy passage delta."
    )
    _, revision = project.import_text("selected.txt", text)
    scope_id = project.snapshot([revision])

    truncated = retrieve_framework_context(
        project,
        scope_id,
        [],
        phenomenon="",
        observation="",
        query_texts=("migration",),
        limit=2,
    )
    assert truncated.inspected_count == 4
    assert len(truncated.candidates) == 2
    assert truncated.output_limit_reached is True
    assert truncated.candidate_limit_reached is False

    complete = retrieve_framework_context(
        project,
        scope_id,
        [],
        phenomenon="",
        observation="",
        query_texts=("migration",),
        limit=4,
    )
    assert complete.inspected_count == 4
    assert len(complete.candidates) == 4
    assert complete.output_limit_reached is False
    assert complete.candidate_limit_reached is False


def test_framework_retrieval_rejects_boolean_limit(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("selected.txt", "Migration discourse discusses framing.")
    scope_id = project.snapshot([revision])

    with pytest.raises(ValueError, match="framework context limit"):
        retrieve_framework_context(
            project,
            scope_id,
            [],
            phenomenon="migration discourse",
            observation="policy framing",
            query_texts=("migration discourse",),
            limit=True,
        )


def test_framework_retrieval_rejects_search_response_outside_scope(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "project")
    _, selected = project.import_text("selected.txt", "Selected text discusses migration.")
    _, outside = project.import_text("outside.txt", "Xenolexicon appears outside scope.")
    scope_id = project.snapshot([selected])
    leaked = project.db.execute(
        "SELECT f.revision_id,f.paragraph_id,f.text,p.start_offset,p.end_offset "
        "FROM revision_fts f JOIN paragraphs p ON p.id=f.paragraph_id "
        "WHERE f.revision_id=?",
        (outside,),
    ).fetchone()
    monkeypatch.setattr(
        project,
        "search_paragraphs",
        lambda query, selected_scope, *, limit: [leaked],
    )

    with pytest.raises(ScopeViolationError, match="outside the explicit scope"):
        retrieve_framework_context(
            project,
            scope_id,
            [],
            phenomenon="xenolexicon",
            observation="outside language",
            query_texts=("xenolexicon",),
        )


def test_framework_retrieval_rejects_oversized_search_response(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("selected.txt", "Migration discourse discusses framing.")
    scope_id = project.snapshot([revision])
    row = project.db.execute(
        "SELECT f.revision_id,f.paragraph_id,f.text,p.start_offset,p.end_offset "
        "FROM revision_fts f JOIN paragraphs p ON p.id=f.paragraph_id "
        "WHERE f.revision_id=?",
        (revision,),
    ).fetchone()
    monkeypatch.setattr(
        project,
        "search_paragraphs",
        lambda query, selected_scope, *, limit: [row] * 1_001,
    )

    with pytest.raises(ProjectError, match="unbounded results"):
        retrieve_framework_context(
            project,
            scope_id,
            [],
            phenomenon="migration discourse",
            observation="policy framing",
            query_texts=("migration discourse",),
        )


def test_lexical_fts_search_methods_keep_a_b_scope_on_syntax_errors(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision_a = project.import_text("a.txt", "Shared marker Aonly context.")
    _, revision_b = project.import_text("b.txt", "Shared marker Bonly neighbor.")
    scope_a = project.snapshot([revision_a])
    scope_b = project.snapshot([revision_b])

    assert [row["revision_id"] for row in project.search("shared", scope_a)] == [revision_a]
    assert [row["revision_id"] for row in project.search_paragraphs("shared", scope_a, limit=5)] == [
        revision_a
    ]
    assert project.search("Bonly", scope_a) == []
    assert project.search_paragraphs("Bonly", scope_a, limit=5) == []
    assert [row["revision_id"] for row in project.search_exact("Bonly neighbor", scope_a)] == []
    assert project.search_exact_page("Bonly neighbor", scope_a)["rows"] == []
    assert [row["revision_id"] for row in project.search("Bonly", scope_b)] == [revision_b]

    for malformed in ('shared AND (', '"shared'):
        with pytest.raises(sqlite3.OperationalError):
            project.search(malformed, scope_a)
        with pytest.raises(sqlite3.OperationalError):
            project.search_paragraphs(malformed, scope_a, limit=5)
        assert project.search_exact(malformed, scope_a) == []
        assert project.search_exact_page(malformed, scope_a)["rows"] == []

    with pytest.raises(ScopeViolationError, match="unknown scope"):
        project.search("shared", "not-a-scope")


def test_large_scope_lexical_and_annotation_reads_respect_sqlite_variable_limit(tmp_path):
    project = Project.init(tmp_path / "project")
    revisions = []
    for index in range(1_000):
        text = (
            "ScopeLexeme exact phrase appears."
            if index == 0
            else "Unindexed fixture."
        )
        _, revision = project.import_text(
            f"plain-{index}.txt", text, create_paragraphs=index == 0
        )
        revisions.append(revision)
    _, selected_revision = project.import_text(
        "selected.txt", "ScopeLexeme exact phrase appears.", create_paragraphs=True
    )
    revisions.append(selected_revision)
    _, excluded_revision = project.import_text(
        "excluded.txt", "Outscopelexeme appears in the exact phrase.", create_paragraphs=True
    )
    codebook = project.create_codebook("large-scope")
    code = project.create_code(codebook, "selected")
    span = project.create_span(selected_revision, 0, len("ScopeLexeme"))
    annotation = project.annotate_span(code, span)
    corpus_id = project.create_corpus("large-scope")
    for revision in revisions:
        project.add_revision_to_corpus(corpus_id, revision)

    project.db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    scope_a = project.snapshot(revisions)
    scope_b = project.snapshot([excluded_revision])
    subcorpus_id = project.create_subcorpus(corpus_id, "all selected", revisions)
    subcorpus_scope = project.subcorpus_snapshot(subcorpus_id)
    assert set(project.scope_revisions(subcorpus_scope)) == set(revisions)

    matching_revisions = sorted((revisions[0], selected_revision))
    assert [row["revision_id"] for row in project.search("ScopeLexeme", scope_a)] == matching_revisions
    assert [
        row["revision_id"]
        for row in project.search_paragraphs("ScopeLexeme", scope_a, limit=5)
    ] == matching_revisions
    assert [
        row["revision_id"] for row in project.search_exact("ScopeLexeme exact phrase", scope_a)
    ] == matching_revisions
    exact_page = project.search_exact_page("ScopeLexeme exact phrase", scope_a, limit=1)
    assert exact_page["total_count"] == 2
    exact_rows = cast(list[sqlite3.Row], exact_page["rows"])
    assert [row["revision_id"] for row in exact_rows] == [matching_revisions[0]]
    next_exact_page = project.search_exact_page("ScopeLexeme exact phrase", scope_a, offset=1)
    next_exact_rows = cast(list[sqlite3.Row], next_exact_page["rows"])
    assert [row["revision_id"] for row in next_exact_rows] == [matching_revisions[1]]
    assert project.search("Outscopelexeme", scope_a) == []
    assert [row["revision_id"] for row in project.search("Outscopelexeme", scope_b)] == [
        excluded_revision
    ]
    assert [row["id"] for row in project.annotations_for_scope(scope_a)] == [annotation]
    project.close()


def test_vector_search_uses_metadata_scope_before_exposing_candidates(tmp_path):
    project = Project.init(tmp_path / "project")
    selected_document, selected_revision = project.import_text("selected.txt", "Selected text.")
    excluded_document, excluded_revision = project.import_text("excluded.txt", "Excluded text.")
    project.set_document_metadata(selected_document, "collection", "selected")
    project.set_document_metadata(excluded_document, "collection", "excluded")
    scope_id, revisions = project.snapshot_by_document_metadata({"collection": "selected"})
    assert revisions == [selected_revision]

    selected_index = build_vector_index(
        project,
        selected_revision,
        [VectorChunk("selected-chunk", 0, len("Selected text."), (0.0, 1.0))],
    )
    excluded_index = build_vector_index(
        project,
        excluded_revision,
        [VectorChunk("excluded-chunk", 0, len("Excluded text."), (1.0, 0.0))],
    )

    hits = search_vectors(project, scope_id, [selected_index], (1.0, 0.0))
    assert [hit.revision_id for hit in hits] == [selected_revision]
    with pytest.raises(ScopeViolationError, match="outside the explicit scope"):
        search_vectors(project, scope_id, [excluded_index], (1.0, 0.0))
    with pytest.raises(ScopeViolationError, match="outside the explicit scope"):
        search_vectors(project, scope_id, [selected_index, excluded_index], (1.0, 0.0))


@pytest.mark.parametrize(
    ("offset", "limit"),
    [(True, 1), (0, True), (1.5, 1), (0, 1.5), (2**63, 1)],
)
def test_analysis_run_summary_page_rejects_invalid_sqlite_pagination_bounds(
    tmp_path, offset, limit
):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("selected.txt", "Selected text.")
    run_id = project.create_run(project.snapshot([revision]), {})

    with pytest.raises(ProjectError, match="analysis run page"):
        project.analysis_run_summary_page(run_id, offset=offset, limit=limit)
