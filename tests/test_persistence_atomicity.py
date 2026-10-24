import sqlite3

import pytest

from discourselens import Project
from discourselens.types import CoverageDescriptor, CoverageMode


def _retrieval_fixture(project: Project) -> tuple[str, tuple[str, str], tuple[str, str]]:
    _, revision_id = project.import_text("retrieval.txt", "alpha beta")
    scope_id = project.snapshot([revision_id])
    analysis_run_id = project.create_run(scope_id, {"profile": "atomicity-fixture"})
    skill_run_id = project.create_skill_run(
        analysis_run_id,
        scope_id,
        "fixture.retrieval",
        "1.0.0",
        "fixture.executor",
        "fixture-digest",
    )
    coverage_id = project.coverage(
        CoverageDescriptor(coverage_mode=CoverageMode.TARGETED, scope_snapshot_id=scope_id)
    )
    retrieval_run_id = project.create_retrieval_run(
        analysis_run_id=analysis_run_id,
        skill_run_id=skill_run_id,
        scope_id=scope_id,
        retrieval_plan_ref="fixture/retrieval-v1",
        query_terms=["alpha"],
        selection_policy="fixture candidates",
        candidate_count=2,
        inspected_count=2,
        returned_count=2,
        candidate_limit_reached=False,
        coverage_id=coverage_id,
    )
    spans = (
        project.create_span(revision_id, 0, 5),
        project.create_span(revision_id, 6, 10),
    )
    hits = (
        project.add_retrieval_hit(retrieval_run_id, spans[0], "fixture", 1, 1.0),
        project.add_retrieval_hit(retrieval_run_id, spans[1], "fixture", 2, 0.5),
    )
    return retrieval_run_id, spans, hits


def test_failed_span_batch_rolls_back_prefix_and_preserves_outer_transaction(tmp_path):
    root = tmp_path / "project"
    project = Project.init(root)
    try:
        _, revision_id = project.import_text("source.txt", "alpha beta")
        project.db.execute(
            "CREATE TRIGGER reject_second_span BEFORE INSERT ON source_spans "
            "WHEN NEW.start_offset=6 "
            "BEGIN SELECT RAISE(ABORT,'fixture span failure'); END"
        )
        project.db.commit()

        # This caller-owned change must survive the failed operation and be
        # independently committable afterward.
        project.db.execute(
            "UPDATE project_meta SET value='true' WHERE key='offline_mode'"
        )
        with pytest.raises(sqlite3.IntegrityError, match="fixture span failure"):
            project.create_spans(revision_id, [(0, 5), (6, 10)])

        assert project.db.in_transaction
        assert project.db.execute(
            "SELECT value FROM project_meta WHERE key='offline_mode'"
        ).fetchone()[0] == "true"
        assert project.db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == 0
        project.db.commit()
        project.close()

        project = Project.open(root)
        assert project.offline_mode
        assert project.db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == 0
    finally:
        project.close()


def test_failed_span_projection_rolls_back_span_and_segments_with_outer_transaction(
    tmp_path,
):
    root = tmp_path / "project"
    project = Project.init(root)
    try:
        _, revision_id = project.import_text("source.txt", "alpha\nbeta")
        project.db.executemany(
            "INSERT INTO pages(id,revision_id,page_index,page_label,width,height,rotation,"
            "canonical_start,canonical_end,text) VALUES(?,?,?,?,?,?,?,?,?,?)",
            [
                ("page-1", revision_id, 0, "1", None, None, 0, 0, 5, "alpha"),
                ("page-2", revision_id, 1, "2", None, None, 0, 6, 10, "beta"),
            ],
        )
        project.db.execute(
            "UPDATE paragraphs SET page_id='page-1' WHERE revision_id=? AND ordinal=0",
            (revision_id,),
        )
        project.db.execute(
            "UPDATE paragraphs SET page_id='page-2' WHERE revision_id=? AND ordinal=1",
            (revision_id,),
        )
        project.db.execute(
            "CREATE TRIGGER reject_second_segment BEFORE INSERT ON source_span_segments "
            "WHEN NEW.page_number='page-2' "
            "BEGIN SELECT RAISE(ABORT,'fixture projection failure'); END"
        )
        project.db.commit()

        project.db.execute(
            "UPDATE project_meta SET value='true' WHERE key='offline_mode'"
        )
        with pytest.raises(sqlite3.IntegrityError, match="fixture projection failure"):
            project.create_span(revision_id, 0, 10)

        assert project.db.in_transaction
        assert project.db.execute(
            "SELECT value FROM project_meta WHERE key='offline_mode'"
        ).fetchone()[0] == "true"
        assert project.db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == 0
        assert project.db.execute("SELECT COUNT(*) FROM source_span_segments").fetchone()[0] == 0
        project.db.commit()
        project.close()
        project = Project.open(root)
        assert project.offline_mode
        assert project.db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == 0
        assert project.db.execute("SELECT COUNT(*) FROM source_span_segments").fetchone()[0] == 0
    finally:
        project.close()


def test_failed_run_finalization_rolls_back_finding_prefix_and_preserves_outer_transaction(
    tmp_path,
):
    root = tmp_path / "project"
    project = Project.init(root)
    try:
        _, revision_id = project.import_text("source.txt", "canonical evidence")
        run_id = project.create_run(project.snapshot([revision_id]), {"profile": "fixture"})
        finding_id = project.create_finding(run_id, "fixture", "1.0.0", "topic", "Observed.")
        project.db.execute(
            "CREATE TRIGGER reject_run_completion BEFORE UPDATE OF status ON analysis_runs "
            "WHEN NEW.status='COMPLETED' "
            "BEGIN SELECT RAISE(ABORT,'fixture completion failure'); END"
        )
        project.db.commit()

        project.db.execute("UPDATE project_meta SET value='true' WHERE key='offline_mode'")
        with pytest.raises(sqlite3.IntegrityError, match="fixture completion failure"):
            project.finalize_run(run_id)

        assert project.db.in_transaction
        assert project.db.execute(
            "SELECT status FROM analysis_runs WHERE id=?", (run_id,)
        ).fetchone()[0] == "PENDING"
        project.db.commit()
        assert project.offline_mode
        assert project.db.execute(
            "SELECT finalized_at FROM findings WHERE id=?", (finding_id,)
        ).fetchone()[0] is None
        project.close()

        project = Project.open(root)
        assert project.offline_mode
        assert project.db.execute(
            "SELECT status FROM analysis_runs WHERE id=?", (run_id,)
        ).fetchone()[0] == "PENDING"
        assert project.db.execute(
            "SELECT finalized_at FROM findings WHERE id=?", (finding_id,)
        ).fetchone()[0] is None
    finally:
        project.close()


def test_failed_run_failure_transition_rolls_back_skill_prefix_and_preserves_outer_transaction(
    tmp_path,
):
    root = tmp_path / "project"
    project = Project.init(root)
    try:
        _, revision_id = project.import_text("source.txt", "canonical evidence")
        scope_id = project.snapshot([revision_id])
        run_id = project.create_run(scope_id, {"profile": "fixture"})
        project.start_run(run_id)
        skill_run_id = project.create_skill_run(
            run_id, scope_id, "fixture.skill", "1.0.0", "fixture.executor", "fixture-digest"
        )
        project.db.execute(
            "CREATE TRIGGER reject_run_failure BEFORE UPDATE OF status ON analysis_runs "
            "WHEN NEW.status='FAILED' "
            "BEGIN SELECT RAISE(ABORT,'fixture failure transition'); END"
        )
        project.db.commit()

        project.db.execute("UPDATE project_meta SET value='true' WHERE key='offline_mode'")
        with pytest.raises(sqlite3.IntegrityError, match="fixture failure transition"):
            project.fail_run(run_id, RuntimeError("fixture"))

        assert project.db.in_transaction
        project.db.commit()
        assert project.offline_mode
        assert project.db.execute(
            "SELECT status FROM analysis_runs WHERE id=?", (run_id,)
        ).fetchone()[0] == "RUNNING"
        assert project.db.execute(
            "SELECT status FROM skill_runs WHERE id=?", (skill_run_id,)
        ).fetchone()[0] == "RUNNING"
        project.close()

        project = Project.open(root)
        assert project.offline_mode
        assert project.db.execute(
            "SELECT status FROM analysis_runs WHERE id=?", (run_id,)
        ).fetchone()[0] == "RUNNING"
        assert project.db.execute(
            "SELECT status FROM skill_runs WHERE id=?", (skill_run_id,)
        ).fetchone()[0] == "RUNNING"
    finally:
        project.close()


def test_failed_retrieval_hit_batch_rolls_back_prefix_and_preserves_outer_transaction(
    tmp_path,
):
    root = tmp_path / "project"
    project = Project.init(root)
    try:
        retrieval_run_id, spans, _ = _retrieval_fixture(project)
        revision_id = project.db.execute(
            "SELECT revision_id FROM source_spans WHERE id=?", (spans[0],)
        ).fetchone()[0]
        batch_spans = (
            project.create_span(revision_id, 0, 2),
            project.create_span(revision_id, 6, 8),
        )
        project.db.execute(
            "CREATE TRIGGER reject_second_retrieval_hit BEFORE INSERT ON retrieval_hits "
            "WHEN NEW.rank=4 "
            "BEGIN SELECT RAISE(ABORT,'fixture retrieval-hit failure'); END"
        )
        project.db.commit()

        project.db.execute("UPDATE project_meta SET value='true' WHERE key='offline_mode'")
        with pytest.raises(sqlite3.IntegrityError, match="fixture retrieval-hit failure"):
            project.add_retrieval_hits(retrieval_run_id, batch_spans, "fixture", 3, 1.0)

        assert project.db.in_transaction
        assert project.db.execute(
            "SELECT COUNT(*) FROM retrieval_hits WHERE retrieval_run_id=?",
            (retrieval_run_id,),
        ).fetchone()[0] == 2
        project.db.commit()
        assert project.offline_mode
        project.close()

        project = Project.open(root)
        assert project.offline_mode
        assert project.db.execute(
            "SELECT COUNT(*) FROM retrieval_hits WHERE retrieval_run_id=?",
            (retrieval_run_id,),
        ).fetchone()[0] == 2
    finally:
        project.close()


def test_failed_reranking_batch_preserves_outer_transaction_and_no_prefix(tmp_path):
    root = tmp_path / "project"
    project = Project.init(root)
    try:
        retrieval_run_id, _, hits = _retrieval_fixture(project)
        project.db.execute(
            "CREATE TRIGGER reject_second_reranking_hit BEFORE INSERT ON reranking_hits "
            "WHEN NEW.rank=2 "
            "BEGIN SELECT RAISE(ABORT,'fixture reranking-hit failure'); END"
        )
        project.db.commit()

        project.db.execute("UPDATE project_meta SET value='true' WHERE key='offline_mode'")
        with pytest.raises(sqlite3.IntegrityError, match="fixture reranking-hit failure"):
            project.record_reranking(
                retrieval_run_id,
                provider_id="fixture-reranker",
                credential_alias=None,
                model="fixture-model",
                query="alpha",
                top_n=2,
                effective_top_n=2,
                candidate_count=2,
                ranked_hits=[(hits[0], 1, 0.9), (hits[1], 2, 0.8)],
            )

        assert project.db.in_transaction
        assert project.db.execute("SELECT COUNT(*) FROM reranking_runs").fetchone()[0] == 0
        project.db.commit()
        assert project.offline_mode
        assert project.db.execute("SELECT COUNT(*) FROM reranking_hits").fetchone()[0] == 0
        project.close()

        project = Project.open(root)
        assert project.offline_mode
        assert project.db.execute("SELECT COUNT(*) FROM reranking_runs").fetchone()[0] == 0
        assert project.db.execute("SELECT COUNT(*) FROM reranking_hits").fetchone()[0] == 0
    finally:
        project.close()
