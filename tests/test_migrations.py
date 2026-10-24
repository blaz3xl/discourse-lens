import json
import sqlite3

import pytest

from discourselens import Project
from discourselens.errors import MigrationError
from discourselens.migrations import SCHEMA_VERSION, migrate
from discourselens.types import (
    ClaimType,
    CoverageDescriptor,
    CoverageMode,
    EvidenceRole,
)


def _migrate_fixture(db: sqlite3.Connection) -> None:
    """Commit fixture setup before invoking the migration transaction."""
    db.commit()
    migrate(db)


def test_legacy_schema_migrates_to_current_version_with_required_columns():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(1,'old')")
    db.execute("CREATE TABLE findings(id TEXT PRIMARY KEY)")
    db.execute("INSERT INTO findings VALUES('legacy-finding')")

    _migrate_fixture(db)

    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION
    columns = {row[1] for row in db.execute("PRAGMA table_info(findings)")}
    assert {
        "analysis_run_id",
        "skill_run_id",
        "coverage_id",
        "created_at",
        "finalized_at",
    } <= columns
    assert db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='finding_reviews'"
    ).fetchone()
    legacy_finding = db.execute(
        "SELECT id,analysis_run_id,created_at FROM findings WHERE id='legacy-finding'"
    ).fetchone()
    assert legacy_finding[0] == "legacy-finding"
    assert legacy_finding[1] == "legacy"
    assert isinstance(legacy_finding[2], str) and legacy_finding[2]
    skill_columns = {row[1] for row in db.execute("PRAGMA table_info(skill_runs)")}
    assert {"started_at", "completed_at", "error_type", "input_finding_id"} <= skill_columns


def test_v34_database_migrates_reranking_candidate_and_return_counts():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(34,'prior')")
    db.execute("CREATE TABLE retrieval_hits(id TEXT,retrieval_run_id TEXT)")
    db.execute("INSERT INTO retrieval_hits VALUES('hit-1','retrieval'),('hit-2','retrieval')")
    db.execute("CREATE TABLE reranking_hits(id TEXT,reranking_run_id TEXT)")
    db.execute("INSERT INTO reranking_hits VALUES('ranked-1','rr')")
    db.execute(
        "CREATE TABLE reranking_runs(id TEXT PRIMARY KEY,retrieval_run_id TEXT NOT NULL,"
        "provider_id TEXT NOT NULL,model TEXT NOT NULL,query TEXT NOT NULL,top_n INTEGER,"
        "candidate_count INTEGER NOT NULL,created_at TEXT NOT NULL)"
    )
    db.execute(
        "INSERT INTO reranking_runs VALUES('rr','retrieval','jina-reranker','model','query',1,1,'old')"
    )

    _migrate_fixture(db)

    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION
    table_names = {
        row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"reranking_runs", "reranking_hits"} <= table_names
    reranking_columns = {row[1] for row in db.execute("PRAGMA table_info(reranking_runs)")}
    assert {"credential_alias", "effective_top_n", "returned_count"} <= reranking_columns
    assert db.execute(
        "SELECT id,credential_alias,effective_top_n,candidate_count,returned_count "
        "FROM reranking_runs"
    ).fetchone() == (
        "rr",
        None,
        1,
        2,
        1,
    )


def test_v38_migration_backfills_completed_finding_finalization_and_preserves_lineage(tmp_path):
    root = tmp_path / "project"
    project = Project.init(root)
    _, revision = project.import_text("source.txt", "Evidence is preserved.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(scope_id, {"profile": "migration-v38"})
    span_id = project.create_span(revision, 0, len("Evidence"))
    evidence_id = project.textual_evidence(span_id)
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "preservation", "Historical finding."
    )
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    completed_at = project.db.execute(
        "SELECT completed_at FROM analysis_runs WHERE id=?", (run_id,)
    ).fetchone()["completed_at"]

    trigger_rows = project.db.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN "
        "('findings_insert_requires_active_run','findings_content_update_requires_active_run',"
        "'findings_delete_requires_active_run',"
        "'finding_evidence_insert_requires_active_run','finding_evidence_update_requires_active_run',"
        "'finding_evidence_delete_requires_active_run','finding_lineage_insert_requires_active_run',"
        "'finding_lineage_update_requires_active_run','finding_lineage_delete_requires_active_run')"
    ).fetchall()
    assert len(trigger_rows) == 9
    for row in trigger_rows:
        project.db.execute(f'DROP TRIGGER "{row["name"]}"')
    project.db.execute("UPDATE schema_migrations SET version=38")
    project.db.execute("UPDATE findings SET finalized_at=NULL WHERE id=?", (finding_id,))
    project.db.commit()
    project.close()

    migrated = Project.open(root)
    try:
        assert (
            migrated.db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
            == SCHEMA_VERSION
        )
        assert migrated.get_finding(finding_id)["finalized_at"] == completed_at
        assert migrated.finding_evidence(finding_id)[0]["id"] == evidence_id
        assert migrated.span_text(span_id) == "Evidence"
        assert migrated.scope_revisions(scope_id) == [revision]
    finally:
        migrated.close()


def test_v39_migration_preserves_append_order_for_existing_finding_reviews():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(39,'prior')")
    db.execute(
        "CREATE TABLE finding_reviews(id TEXT PRIMARY KEY,finding_id TEXT NOT NULL,"
        "status TEXT NOT NULL,note TEXT,created_at TEXT NOT NULL)"
    )
    db.executemany(
        "INSERT INTO finding_reviews VALUES(?,?,?,?,?)",
        [
            ("z-first", "finding-a", "ACCEPTED", "First decision", "same-time"),
            ("other-finding", "finding-b", "MODIFIED", "Separate history", "same-time"),
            ("a-second", "finding-a", "REJECTED", "Later decision", "same-time"),
        ],
    )
    db.commit()

    _migrate_fixture(db)

    assert db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION
    assert db.execute(
        "SELECT id,finding_id,status,note,created_at,review_order FROM finding_reviews "
        "WHERE finding_id='finding-a' ORDER BY review_order"
    ).fetchall() == [
        ("z-first", "finding-a", "ACCEPTED", "First decision", "same-time", 1),
        ("a-second", "finding-a", "REJECTED", "Later decision", "same-time", 2),
    ]
    assert db.execute(
        "SELECT id,review_order FROM finding_reviews WHERE finding_id='finding-b'"
    ).fetchone() == ("other-finding", 1)


def test_v46_upgrade_adds_observation_candidate_storage_without_rewriting_findings(tmp_path):
    root = tmp_path / "project"
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "Historic candidate wording.")
    run_id = project.create_run(project.snapshot([revision_id]), {"profile": "migration-v46"})
    finding_id = project.create_finding(
        run_id,
        "fixture.framework",
        "1.0.0",
        "historic observation",
        "Candidate description supplied by the model: historic text.",
    )
    evidence_id = project.textual_evidence(project.create_span(revision_id, 0, 7))
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    finding_provenance_before = project.db.execute(
        "SELECT analysis_run_id,created_at,finalized_at FROM findings WHERE id=?",
        (finding_id,),
    ).fetchone()

    for trigger in (
        "finding_observation_candidates_insert_requires_active_run",
        "finding_observation_candidates_update_requires_active_run",
        "finding_observation_candidates_delete_requires_active_run",
        "findings_delete_observation_candidates",
    ):
        project.db.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
    project.db.execute("DROP TABLE finding_observation_candidates")
    project.db.execute("UPDATE schema_migrations SET version=46")
    project.db.commit()
    project.close()

    migrated = Project.open(root)
    try:
        assert migrated.db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION
        assert migrated.get_finding(finding_id)["observation"] == (
            "Candidate description supplied by the model: historic text."
        )
        assert tuple(
            migrated.db.execute(
                "SELECT analysis_run_id,created_at,finalized_at FROM findings WHERE id=?",
                (finding_id,),
            ).fetchone()
        ) == tuple(finding_provenance_before)
        assert migrated.db.execute(
            "SELECT status FROM analysis_runs WHERE id=?", (run_id,)
        ).fetchone()["status"] == "COMPLETED"
        assert migrated.finding_observation_candidates(finding_id) == []
        assert migrated.finding_evidence(finding_id)[0]["id"] == evidence_id
    finally:
        migrated.close()


def test_v36_database_adds_ocr_provenance_without_rewriting_revisions():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(36,'fixture')")
    db.execute(
        "CREATE TABLE document_revisions(id TEXT PRIMARY KEY,document_id TEXT NOT NULL,"
        "source_hash TEXT NOT NULL,canonical_hash TEXT NOT NULL,extractor TEXT NOT NULL,"
        "coordinate_scheme TEXT NOT NULL,text_path TEXT NOT NULL,original_path TEXT,mime_type TEXT,"
        "source_size INTEGER,extraction_status TEXT NOT NULL,created_at TEXT NOT NULL)"
    )
    db.execute(
        "INSERT INTO document_revisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "revision-v36",
            "document",
            "source-hash",
            "canonical-hash",
            "pymupdf@1.28.2",
            "unicode-codepoint-half-open-v1",
            "source/revision.txt",
            "source/revision.original",
            "application/pdf",
            128,
            "SUCCESS",
            "fixture-time",
        ),
    )

    _migrate_fixture(db)

    revision = db.execute(
        "SELECT id,canonical_hash,extractor,ocr_language,ocr_dpi FROM document_revisions"
    ).fetchone()
    assert revision == (
        "revision-v36",
        "canonical-hash",
        "pymupdf@1.28.2",
        None,
        None,
    )
    assert db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION


def test_v48_document_revision_identity_migration_preserves_rows_and_expands_key():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(48,'prior')")
    db.execute(
        "CREATE TABLE document_revisions(id TEXT PRIMARY KEY,document_id TEXT NOT NULL,"
        "source_hash TEXT NOT NULL,canonical_hash TEXT NOT NULL,extractor TEXT NOT NULL,"
        "coordinate_scheme TEXT NOT NULL,text_path TEXT NOT NULL,original_path TEXT,mime_type TEXT,"
        "source_size INTEGER,extraction_status TEXT NOT NULL DEFAULT 'SUCCESS',"
        "created_at TEXT NOT NULL,ocr_language TEXT,ocr_dpi INTEGER,"
        "UNIQUE(document_id,source_hash,canonical_hash,extractor,coordinate_scheme))"
    )
    db.execute(
        "INSERT INTO document_revisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "revision-v48",
            "document-1",
            "source-hash",
            "canonical-hash",
            "plain-text@1",
            "unicode-codepoint-half-open-v1",
            "source/revision.txt",
            "source/revision.original",
            "text/plain",
            16,
            "SUCCESS",
            "fixture-time",
            None,
            None,
        ),
    )

    _migrate_fixture(db)

    revision = db.execute(
        "SELECT id,source_hash,canonical_hash,extractor,extraction_manifest_json,structure_hash "
        "FROM document_revisions"
    ).fetchone()
    assert revision[:4] == (
        "revision-v48",
        "source-hash",
        "canonical-hash",
        "plain-text@1",
    )
    assert json.loads(revision[4])["extractor_identity"] == "plain-text@1"
    assert revision[5] == "canonical-hash"
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION

    # Identical source/text/extractor can coexist when the structural map differs.
    db.execute(
        "INSERT INTO document_revisions "
        "(id,document_id,source_hash,canonical_hash,extractor,coordinate_scheme,text_path,"
        "extraction_manifest_json,structure_hash,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            "revision-v49-layout-2",
            "document-1",
            "source-hash",
            "canonical-hash",
            "plain-text@1",
            "unicode-codepoint-half-open-v1",
            "source/revision-2.txt",
            '{"manifest_version":"test-v2"}',
            "different-structure-hash",
            "fixture-time-2",
        ),
    )
    assert db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 2


def test_v37_database_adds_streamed_comparison_storage_without_rewriting_runs():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(37,'fixture')")
    db.execute(
        "CREATE TABLE comparison_runs(id TEXT PRIMARY KEY,left_scope_id TEXT NOT NULL,"
        "right_scope_id TEXT NOT NULL,metric TEXT NOT NULL,calculation_version TEXT NOT NULL,"
        "config_json TEXT NOT NULL,result_json TEXT NOT NULL,result_hash TEXT NOT NULL,"
        "created_at TEXT NOT NULL)"
    )
    db.execute(
        "INSERT INTO comparison_runs VALUES(?,?,?,?,?,?,?,?,?)",
        (
            "comparison-v37",
            "left-scope",
            "right-scope",
            "frequency",
            "fixture-v1",
            "{}",
            "[]",
            "legacy-hash",
            "fixture-time",
        ),
    )

    _migrate_fixture(db)

    assert db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION
    run = db.execute(
        "SELECT id,result_hash,storage_kind FROM comparison_runs WHERE id='comparison-v37'"
    ).fetchone()
    assert run == ("comparison-v37", "legacy-hash", "json-v1")
    assert db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='comparison_result_rows'"
    ).fetchone() == ("comparison_result_rows",)


def test_v23_model_execution_rows_migrate_with_empty_parameter_snapshot():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(23,'old')")
    db.execute("CREATE TABLE model_executions(id TEXT PRIMARY KEY)")
    db.execute("INSERT INTO model_executions(id) VALUES('prior-run')")

    _migrate_fixture(db)

    columns = {row[1] for row in db.execute("PRAGMA table_info(model_executions)")}
    assert "parameters_json" in columns
    assert "capabilities_json" in columns
    assert "endpoint" in columns
    assert (
        db.execute("SELECT parameters_json FROM model_executions WHERE id='prior-run'").fetchone()[
            0
        ]
        == "{}"
    )
    assert (
        db.execute(
            "SELECT capabilities_json FROM model_executions WHERE id='prior-run'"
        ).fetchone()[0]
        == "{}"
    )
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION


def test_v44_model_execution_migration_adds_nullable_endpoint(tmp_path):
    project = Project.init(tmp_path / "endpoint-migration")
    try:
        project.db.execute("UPDATE schema_migrations SET version=44")
        project.db.execute(
            "INSERT INTO model_executions(id,provider_id,credential_alias,model,request_hash,"
            "started_at) VALUES('legacy','openai','default','model','hash','now')"
        )
        project.db.commit()
        project.close()

        migrated = Project.open(tmp_path / "endpoint-migration")
        try:
            assert migrated.db.execute(
                "SELECT endpoint FROM model_executions WHERE id='legacy'"
            ).fetchone()[0] is None
            assert migrated.db.execute(
                "SELECT max(version) FROM schema_migrations"
            ).fetchone()[0] == SCHEMA_VERSION
        finally:
            migrated.close()
    except Exception:
        project.close()
        raise


def test_v45_model_execution_migration_adds_nullable_skill_run_link(tmp_path):
    project = Project.init(tmp_path / "skill-run-migration")
    try:
        project.db.execute("UPDATE schema_migrations SET version=45")
        project.db.execute(
            "INSERT INTO model_executions(id,provider_id,credential_alias,model,request_hash,"
            "started_at) VALUES('legacy','openai','default','model','hash','now')"
        )
        project.db.commit()
        project.close()

        migrated = Project.open(tmp_path / "skill-run-migration")
        try:
            assert migrated.db.execute(
                "SELECT skill_run_id FROM model_executions WHERE id='legacy'"
            ).fetchone()[0] is None
            assert migrated.db.execute(
                "SELECT max(version) FROM schema_migrations"
            ).fetchone()[0] == SCHEMA_VERSION
        finally:
            migrated.close()
    except Exception:
        project.close()
        raise


def test_v25_reasoning_profiles_migrate_additively():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(25,'old')")
    db.execute(
        "CREATE TABLE model_presets(id TEXT PRIMARY KEY,name TEXT,provider_id TEXT,"
        "credential_alias TEXT,model TEXT,reasoning TEXT,parameters_json TEXT,created_at TEXT)"
    )
    db.execute(
        "INSERT INTO model_presets VALUES('preset','review','anthropic','lab','claude-old',"
        "'high','{}','old')"
    )
    db.execute(
        "CREATE TABLE model_executions(id TEXT PRIMARY KEY,analysis_run_id TEXT,"
        "provider_id TEXT,credential_alias TEXT,model TEXT,reasoning TEXT,request_hash TEXT,"
        "response_hash TEXT,error_type TEXT,input_tokens INTEGER,output_tokens INTEGER,"
        "reasoning_tokens INTEGER,latency_ms INTEGER,started_at TEXT,completed_at TEXT,"
        "parameters_json TEXT,capabilities_json TEXT)"
    )
    db.execute(
        "INSERT INTO model_executions(id,provider_id,credential_alias,model,reasoning,request_hash) "
        "VALUES('execution','anthropic','lab','claude-old','high','hash')"
    )

    _migrate_fixture(db)

    preset_columns = {row[1] for row in db.execute("PRAGMA table_info(model_presets)")}
    execution_columns = {row[1] for row in db.execute("PRAGMA table_info(model_executions)")}
    assert {"reasoning_mode", "reasoning_budget_tokens"} <= preset_columns
    assert {"reasoning_mode", "reasoning_budget_tokens"} <= execution_columns
    assert db.execute(
        "SELECT reasoning,reasoning_mode,reasoning_budget_tokens FROM model_presets "
        "WHERE id='preset'"
    ).fetchone() == ("high", None, None)
    assert db.execute(
        "SELECT reasoning,reasoning_mode,reasoning_budget_tokens FROM model_executions "
        "WHERE id='execution'"
    ).fetchone() == ("high", None, None)
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION


def test_v26_skill_runs_gain_skip_reason_without_mutating_history():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(26,'old')")
    db.execute(
        "CREATE TABLE skill_runs(id TEXT PRIMARY KEY,analysis_run_id TEXT,skill_id TEXT,"
        "skill_version TEXT,scope_id TEXT,status TEXT,executor TEXT,config_hash TEXT,"
        "coverage_id TEXT,input_finding_id TEXT,started_at TEXT,completed_at TEXT,error_type TEXT)"
    )
    db.execute("INSERT INTO skill_runs(id,status) VALUES('prior-skill-run','COMPLETED')")

    _migrate_fixture(db)

    columns = {row[1] for row in db.execute("PRAGMA table_info(skill_runs)")}
    assert "skip_reason" in columns
    assert db.execute(
        "SELECT status,skip_reason FROM skill_runs WHERE id='prior-skill-run'"
    ).fetchone() == ("COMPLETED", None)
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION


def test_v27_derived_artifact_registry_gains_integrity_hash_without_losing_rows():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(27,'old')")
    db.execute(
        "CREATE TABLE artifact_registry(id TEXT PRIMARY KEY,revision_id TEXT NOT NULL,"
        "artifact_type TEXT NOT NULL,backend TEXT NOT NULL,backend_version TEXT,model TEXT,"
        "model_version TEXT,schema_version TEXT NOT NULL,content_hash TEXT NOT NULL,"
        "relative_path TEXT NOT NULL,created_at TEXT NOT NULL)"
    )
    db.execute(
        "INSERT INTO artifact_registry VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "artifact",
            "revision",
            "surface_tokens",
            "pyarrow",
            None,
            None,
            None,
            "surface-token-v1",
            "legacy-fingerprint",
            "artifacts/nlp/a.parquet",
            "old",
        ),
    )

    _migrate_fixture(db)

    columns = {row[1] for row in db.execute("PRAGMA table_info(artifact_registry)")}
    assert "file_hash" in columns
    assert db.execute(
        "SELECT content_hash,relative_path,file_hash FROM artifact_registry WHERE id='artifact'"
    ).fetchone() == ("legacy-fingerprint", "artifacts/nlp/a.parquet", None)
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION


def test_v28_source_span_segments_gain_explicit_bbox_projection_provenance():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(28,'old')")
    db.execute(
        "CREATE TABLE source_span_segments(id TEXT PRIMARY KEY,span_id TEXT NOT NULL,"
        "paragraph_id TEXT,page_number INTEGER,start_offset INTEGER NOT NULL,"
        "end_offset INTEGER NOT NULL,bounding_boxes_json TEXT)"
    )
    db.execute(
        "INSERT INTO source_span_segments VALUES('segment','span','paragraph','page',1,8,'[[1,2,3,4]]')"
    )

    _migrate_fixture(db)

    row = db.execute(
        "SELECT id,bounding_boxes_json,bbox_status,bbox_projection_version "
        "FROM source_span_segments WHERE id='segment'"
    ).fetchone()
    assert row == ("segment", "[[1,2,3,4]]", "APPROXIMATE", "paragraph-block-v1")
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION


def test_v29_analysis_runs_gain_immutable_runtime_provenance():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(29,'old')")
    db.execute(
        "CREATE TABLE analysis_runs(id TEXT PRIMARY KEY,scope_id TEXT NOT NULL,status TEXT NOT NULL,"
        "config_json TEXT NOT NULL,config_hash TEXT NOT NULL,started_at TEXT NOT NULL,completed_at TEXT)"
    )
    db.execute(
        "INSERT INTO analysis_runs VALUES('historical','scope','COMPLETED','{}','hash','then','then')"
    )

    _migrate_fixture(db)

    row = db.execute(
        "SELECT status,software_version,project_format_version,database_schema_version "
        "FROM analysis_runs WHERE id='historical'"
    ).fetchone()
    assert row == ("COMPLETED", "unknown", 0, 0)
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION


def test_future_schema_is_rejected_without_changing_database_objects(tmp_path):
    database = tmp_path / "future.sqlite"
    db = sqlite3.connect(database)
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(?, 'future')", (SCHEMA_VERSION + 1,))
    db.execute("CREATE TABLE future_only(id TEXT PRIMARY KEY, payload BLOB NOT NULL)")
    db.commit()
    before = db.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type,name"
    ).fetchall()

    with pytest.raises(MigrationError, match="newer than this software"):
        migrate(db)

    after = db.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type,name"
    ).fetchall()
    assert after == before
    assert db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == (
        SCHEMA_VERSION + 1
    )
    db.close()


@pytest.mark.parametrize("has_migration_table", [False, True], ids=["missing", "empty"])
def test_unversioned_existing_schema_is_rejected_without_marking_current(
    tmp_path, has_migration_table
):
    database = tmp_path / "unversioned.sqlite"
    db = sqlite3.connect(database)
    if has_migration_table:
        db.execute(
            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
    db.execute("CREATE TABLE findings(id TEXT PRIMARY KEY, legacy_payload TEXT NOT NULL)")
    db.execute("INSERT INTO findings VALUES('historical','preserve me')")
    db.commit()
    before = db.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type,name"
    ).fetchall()

    with pytest.raises(MigrationError, match="schema objects but has no applied schema version"):
        migrate(db)

    after = db.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type,name"
    ).fetchall()
    assert after == before
    assert db.execute("SELECT * FROM findings").fetchone() == ("historical", "preserve me")
    if has_migration_table:
        assert db.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 0
    db.close()

    reopened = sqlite3.connect(database)
    assert reopened.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type,name"
    ).fetchall() == before
    assert reopened.execute("SELECT * FROM findings").fetchone() == (
        "historical",
        "preserve me",
    )
    reopened.close()


def test_failed_migration_rolls_back_schema_and_data_changes(tmp_path):
    database = tmp_path / "partial.sqlite"
    db = sqlite3.connect(database)
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(39, 'prior')")
    db.execute(
        "CREATE TABLE finding_reviews(id TEXT PRIMARY KEY,finding_id TEXT NOT NULL,"
        "status TEXT NOT NULL,note TEXT,created_at TEXT NOT NULL)"
    )
    db.execute(
        "INSERT INTO finding_reviews VALUES('review','finding','ACCEPTED',NULL,'prior')"
    )
    db.execute(
        "CREATE TRIGGER reject_review_backfill BEFORE UPDATE ON finding_reviews "
        "BEGIN SELECT RAISE(ABORT,'fixture migration failure'); END"
    )
    db.commit()
    before = db.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type,name"
    ).fetchall()

    with pytest.raises(MigrationError, match="fixture migration failure"):
        migrate(db)

    after = db.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type,name"
    ).fetchall()
    assert after == before
    assert db.execute("SELECT version FROM schema_migrations").fetchone()[0] == 39
    assert db.execute("SELECT id,status FROM finding_reviews").fetchone() == (
        "review",
        "ACCEPTED",
    )
    assert "review_order" not in {
        row[1] for row in db.execute("PRAGMA table_info(finding_reviews)")
    }
    db.close()

    reopened = sqlite3.connect(database)
    assert reopened.execute("SELECT version FROM schema_migrations").fetchone()[0] == 39
    assert reopened.execute("SELECT id,status FROM finding_reviews").fetchone() == (
        "review",
        "ACCEPTED",
    )
    reopened.execute("DROP TRIGGER reject_review_backfill")
    reopened.commit()
    migrate(reopened)
    assert reopened.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == (
        SCHEMA_VERSION
    )
    assert reopened.execute(
        "SELECT id,status,review_order FROM finding_reviews"
    ).fetchone() == ("review", "ACCEPTED", 1)
    migrated_schema = reopened.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type,name"
    ).fetchall()
    reopened.close()

    idempotent_reopen = sqlite3.connect(database)
    migrate(idempotent_reopen)
    assert idempotent_reopen.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type,name"
    ).fetchall() == migrated_schema
    idempotent_reopen.close()


def test_migration_rejects_active_transaction_without_committing_caller_changes():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(39, 'prior')")
    db.execute(
        "CREATE TABLE finding_reviews(id TEXT PRIMARY KEY,finding_id TEXT NOT NULL,"
        "status TEXT NOT NULL,note TEXT,created_at TEXT NOT NULL)"
    )
    db.execute("CREATE TABLE caller_changes(value TEXT NOT NULL)")
    db.commit()

    before = db.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type,name"
    ).fetchall()
    db.execute("INSERT INTO caller_changes VALUES('pending')")

    with pytest.raises(MigrationError, match="transaction is active"):
        migrate(db)

    after = db.execute(
        "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
        "ORDER BY type,name"
    ).fetchall()
    assert after == before
    assert db.in_transaction
    assert db.execute("SELECT version FROM schema_migrations").fetchone()[0] == 39
    db.rollback()
    assert db.execute("SELECT COUNT(*) FROM caller_changes").fetchone()[0] == 0
    db.close()


def test_opening_current_project_repeatedly_is_idempotent(tmp_path):
    root = tmp_path / "current-project"
    project = Project.init(root)
    project_id = project.project_id
    schema = [
        tuple(row)
        for row in project.db.execute(
            "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
            "ORDER BY type,name"
        ).fetchall()
    ]
    project.close()

    for _ in range(2):
        reopened = Project.open(root)
        try:
            assert reopened.project_id == project_id
            assert [
                row["version"]
                for row in reopened.db.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                ).fetchall()
            ] == [SCHEMA_VERSION]
            assert [
                tuple(row)
                for row in reopened.db.execute(
                    "SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' "
                    "ORDER BY type,name"
                ).fetchall()
            ] == schema
        finally:
            reopened.close()


def test_v30_adds_retrieval_lineage_tables():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(30,'old')")

    _migrate_fixture(db)

    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"retrieval_runs", "retrieval_hits"} <= tables
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION


def test_v31_adds_finding_lineage_for_controlled_synthesis():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(31,'old')")

    _migrate_fixture(db)

    assert db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='finding_lineage'"
    ).fetchone()
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION


def test_v32_adds_vector_embedding_provenance_without_mutating_legacy_artifacts():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(32,'old')")
    db.execute(
        "CREATE TABLE artifact_registry(id TEXT PRIMARY KEY,revision_id TEXT NOT NULL,"
        "artifact_type TEXT NOT NULL,backend TEXT NOT NULL,backend_version TEXT,model TEXT,"
        "model_version TEXT,schema_version TEXT NOT NULL,content_hash TEXT NOT NULL,"
        "relative_path TEXT NOT NULL,created_at TEXT NOT NULL,file_hash TEXT)"
    )
    db.execute(
        "INSERT INTO artifact_registry VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "legacy-index",
            "revision",
            "vector_index",
            "sqlite-vec",
            None,
            None,
            None,
            "vector-index-v1:768",
            "derived",
            "artifacts/vectors/legacy.sqlite",
            "old",
            "file-hash",
        ),
    )

    _migrate_fixture(db)

    columns = {row[1] for row in db.execute("PRAGMA table_info(artifact_registry)")}
    expected = {
        "embedding_provider_id",
        "embedding_model",
        "embedding_dimension",
        "chunker_version",
        "embedding_input_type",
    }
    assert expected <= columns
    assert db.execute(
        "SELECT schema_version,file_hash,embedding_provider_id,embedding_model "
        "FROM artifact_registry WHERE id='legacy-index'"
    ).fetchone() == ("vector-index-v1:768", "file-hash", None, None)
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION


def test_v40_migrates_to_scope_and_analysis_run_immutability_guards():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    db.execute("INSERT INTO schema_migrations VALUES(40,'prior')")

    _migrate_fixture(db)

    triggers = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
    }
    assert {
        "scope_snapshots_immutable_update",
        "scope_snapshots_immutable_delete",
        "scope_snapshots_immutable_insert_conflict",
        "coverage_descriptors_immutable_update",
        "coverage_descriptors_immutable_delete",
        "coverage_descriptors_immutable_insert_conflict",
        "analysis_runs_provenance_immutable",
        "analysis_runs_provenance_immutable_insert_conflict",
        "analysis_runs_lifecycle_transition",
        "analysis_runs_immutable_delete",
    } <= triggers
    assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == SCHEMA_VERSION


def test_scope_and_coverage_rows_reject_sqlite_mutation(tmp_path):
    project = Project.init(tmp_path / "append-only-scopes")
    _, revision = project.import_text("source.txt", "Evidence.")
    scope_id = project.snapshot([revision])
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXAMPLE_BASED,
            scope_snapshot_id=scope_id,
        )
    )

    with pytest.raises(sqlite3.IntegrityError, match="ScopeSnapshot records are immutable"):
        project.db.execute(
            "UPDATE scope_snapshots SET revision_ids_json='[]' WHERE id=?", (scope_id,)
        )
    with pytest.raises(sqlite3.IntegrityError, match="ScopeSnapshot records are immutable"):
        project.db.execute("DELETE FROM scope_snapshots WHERE id=?", (scope_id,))
    with pytest.raises(sqlite3.IntegrityError, match="CoverageDescriptor records are immutable"):
        project.db.execute(
            "UPDATE coverage_descriptors SET payload_json='{}' WHERE id=?", (coverage_id,)
        )
    with pytest.raises(sqlite3.IntegrityError, match="CoverageDescriptor records are immutable"):
        project.db.execute("DELETE FROM coverage_descriptors WHERE id=?", (coverage_id,))

    project.db.commit()
    assert project.scope_revisions(scope_id) == [revision]
    assert project.coverage_descriptor(coverage_id).scope_snapshot_id == scope_id
    project.close()


def test_v48_migration_makes_completed_skill_run_provenance_immutable(tmp_path):
    root = tmp_path / "skill-run-provenance-migration"
    project = Project.init(root)
    try:
        _, revision = project.import_text("source.txt", "Evidence is preserved.")
        _, outside_revision = project.import_text("outside.txt", "Outside evidence.")
        scope_id = project.snapshot([revision])
        other_scope_id = project.snapshot([outside_revision])
        run_id = project.create_run(scope_id, {"profile": "skill-run-provenance"})
        skill_run_id = project.create_skill_run(
            run_id,
            scope_id,
            "fixture.skill",
            "1.0.0",
            "fixture.executor",
            "fixture-config-hash",
        )
        coverage_id = project.coverage(
            CoverageDescriptor(
                coverage_mode=CoverageMode.TARGETED,
                scope_snapshot_id=scope_id,
            )
        )
        project.set_running_skill_coverage(skill_run_id, coverage_id)
        project.complete_skill_run(skill_run_id)
        project.finalize_run(run_id)

        skill_triggers = (
            "skill_runs_immutable_insert_conflict",
            "skill_runs_provenance_immutable",
            "skill_runs_coverage_update_requires_running",
            "skill_runs_lifecycle_transition",
            "skill_runs_immutable_delete",
        )
        for trigger in skill_triggers:
            project.db.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
        project.db.execute("UPDATE schema_migrations SET version=47")
        project.db.commit()
        project.close()

        migrated = Project.open(root)
        try:
            assert (
                migrated.db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
                == SCHEMA_VERSION
            )
            installed = {
                row[0]
                for row in migrated.db.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                ).fetchall()
            }
            assert set(skill_triggers) <= installed
            assert tuple(
                migrated.db.execute(
                    "SELECT status,coverage_id FROM skill_runs WHERE id=?", (skill_run_id,)
                ).fetchone()
            ) == ("COMPLETED", coverage_id)

            alternate_coverage_id = migrated.coverage(
                CoverageDescriptor(
                    coverage_mode=CoverageMode.TARGETED,
                    scope_snapshot_id=other_scope_id,
                )
            )
            rewrites = (
                ("id", "rewritten-skill-run-id"),
                ("skill_id", "rewritten.skill"),
                ("config_hash", "rewritten-config-hash"),
                ("scope_id", other_scope_id),
                ("coverage_id", alternate_coverage_id),
            )
            for column, value in rewrites:
                with pytest.raises(sqlite3.IntegrityError, match="SkillRun"):
                    migrated.db.execute(
                        f"UPDATE skill_runs SET {column}=? WHERE id=?",
                        (value, skill_run_id),
                    )
                migrated.db.rollback()
        finally:
            migrated.close()
    except Exception:
        project.close()
        raise


def test_sql_rebind_or_replace_cannot_revalidate_finding(tmp_path):
    project = Project.init(tmp_path / "immutable-run-scope")
    selected_text = "Every selected item uses apples."
    _, selected_revision = project.import_text("selected.txt", selected_text)
    _, contradictory_revision = project.import_text("other.txt", "This item does not use apples.")
    original_scope = project.snapshot([selected_revision])
    broadened_scope = project.snapshot([selected_revision, contradictory_revision])
    run_id = project.create_run(original_scope, {})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=original_scope,
            population_definition="all documents in the immutable ScopeSnapshot",
            population_count=1,
            inspection_count=1,
            completion_criterion="every document inspected",
            termination_reason="completed deterministic enumeration",
        )
    )
    finding_id = project.create_finding(
        run_id,
        "fixture",
        "1.0.0",
        "universal claim",
        "Every selected item uses apples.",
        claim_types=[ClaimType.UNIVERSAL],
        coverage_id=coverage_id,
    )
    evidence_id = project.textual_evidence(
        project.create_span(selected_revision, 0, len(selected_text))
    )
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    assert project.validate_finding(finding_id).valid

    descriptor = json.loads(project.coverage_descriptor(coverage_id).model_dump_json())
    descriptor["scope_snapshot_id"] = broadened_scope
    descriptor["population_count"] = 2
    descriptor["inspection_count"] = 2
    with pytest.raises(sqlite3.IntegrityError, match="identity and provenance are immutable"):
        project.db.execute(
            "UPDATE analysis_runs SET scope_id=? WHERE id=?", (broadened_scope, run_id)
        )
    with pytest.raises(sqlite3.IntegrityError, match="CoverageDescriptor records are immutable"):
        project.db.execute(
            "UPDATE coverage_descriptors SET payload_json=? WHERE id=?",
            (json.dumps(descriptor), coverage_id),
        )
    assert project.db.execute("PRAGMA recursive_triggers").fetchone()[0] == 0
    with pytest.raises(sqlite3.IntegrityError, match="ScopeSnapshot records are immutable"):
        project.db.execute(
            "INSERT OR REPLACE INTO scope_snapshots(id,revision_ids_json) VALUES(?,?)",
            (original_scope, json.dumps([selected_revision, contradictory_revision])),
        )
    with pytest.raises(sqlite3.IntegrityError, match="CoverageDescriptor records are immutable"):
        project.db.execute(
            "INSERT OR REPLACE INTO coverage_descriptors(id,payload_json) VALUES(?,?)",
            (coverage_id, json.dumps(descriptor)),
        )
    run = project.db.execute("SELECT * FROM analysis_runs WHERE id=?", (run_id,)).fetchone()
    with pytest.raises(
        sqlite3.IntegrityError, match="analysis run identity and provenance are immutable"
    ):
        project.db.execute(
            "INSERT OR REPLACE INTO analysis_runs("
            "id,scope_id,status,config_json,config_hash,started_at,completed_at,"
            "software_version,project_format_version,database_schema_version) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                run["id"],
                broadened_scope,
                run["status"],
                run["config_json"],
                run["config_hash"],
                run["started_at"],
                run["completed_at"],
                run["software_version"],
                run["project_format_version"],
                run["database_schema_version"],
            ),
        )
    project.db.commit()

    assert project.validate_finding(finding_id).valid
    assert project.scope_revisions(original_scope) == [selected_revision]
    assert project.coverage_descriptor(coverage_id).scope_snapshot_id == original_scope
    project.close()


def test_analysis_run_lifecycle_transitions_remain_allowed(tmp_path):
    project = Project.init(tmp_path / "run-lifecycle")
    _, revision = project.import_text("source.txt", "Evidence.")
    run_id = project.create_run(project.snapshot([revision]), {})

    project.start_run(run_id)
    assert project.db.execute(
        "SELECT status FROM analysis_runs WHERE id=?", (run_id,)
    ).fetchone()[0] == "RUNNING"
    project.fail_run(run_id, RuntimeError("fixture failure"))
    assert project.db.execute(
        "SELECT status FROM analysis_runs WHERE id=?", (run_id,)
    ).fetchone()[0] == "FAILED"
    project.resume_run(run_id)
    resumed = project.db.execute(
        "SELECT status,completed_at FROM analysis_runs WHERE id=?", (run_id,)
    ).fetchone()
    assert resumed["status"] == "RUNNING"
    assert resumed["completed_at"] is None
    project.finalize_run(run_id)
    assert project.db.execute(
        "SELECT status,completed_at FROM analysis_runs WHERE id=?", (run_id,)
    ).fetchone()[0] == "COMPLETED"
    with pytest.raises(sqlite3.IntegrityError, match="invalid analysis run lifecycle transition"):
        project.db.execute(
            "UPDATE analysis_runs SET status='RUNNING',completed_at=NULL WHERE id=?", (run_id,)
        )
    with pytest.raises(sqlite3.IntegrityError, match="history is immutable"):
        project.db.execute("DELETE FROM analysis_runs WHERE id=?", (run_id,))

    project.db.commit()
    project.close()


def test_finding_lineage_cannot_be_reassigned_into_completed_run(tmp_path):
    project = Project.init(tmp_path / "lineage-immutability")
    project.db.execute("DROP TRIGGER finding_lineage_update_requires_active_target_run")
    project.db.commit()
    migrate(project.db)

    _, revision_id = project.import_text("source.txt", "Evidence.")
    scope_id = project.snapshot([revision_id])

    active_run = project.create_run(scope_id, {"fixture": "active"})
    active_target = project.create_finding(
        active_run, "fixture", "1.0.0", "active-target", "Active target."
    )
    source_finding = project.create_finding(
        active_run, "fixture", "1.0.0", "source", "Source finding."
    )

    completed_run = project.create_run(scope_id, {"fixture": "completed"})
    completed_target = project.create_finding(
        completed_run, "fixture", "1.0.0", "completed-target", "Completed target."
    )
    project.finalize_run(completed_run)
    assert project.db.execute(
        "SELECT status FROM analysis_runs WHERE id=?", (completed_run,)
    ).fetchone()[0] == "COMPLETED"

    project.db.execute(
        "INSERT INTO finding_lineage(finding_id,source_finding_id,relation) VALUES(?,?,?)",
        (active_target, source_finding, "derived_from"),
    )
    project.db.commit()

    with pytest.raises(sqlite3.IntegrityError, match="cannot change lineage"):
        project.db.execute(
            "UPDATE finding_lineage SET finding_id=? WHERE finding_id=?",
            (completed_target, active_target),
        )
    assert tuple(
        project.db.execute(
            "SELECT finding_id,source_finding_id FROM finding_lineage"
        ).fetchone()
    ) == (active_target, source_finding)
    project.db.commit()
    project.close()
