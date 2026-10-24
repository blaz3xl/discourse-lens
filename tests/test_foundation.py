import hashlib
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import DocumentError, ScopeViolationError, SourceSpanError
from discourselens.skills import eligibility, parse_manifest
from discourselens.types import CoverageDescriptor, CoverageMode, EvidenceRole
from discourselens.versioning import software_version


def test_end_to_end_and_portability(tmp_path):
    project = Project.init(tmp_path / "one")
    _, revision = project.import_text("speech.txt", "Apples are green.\nElephants live in Africa.")
    span = project.create_span(revision, 0, 17)
    evidence = project.textual_evidence(span)
    assert project.span_text(span) == "Apples are green."
    assert "speech.txt" in project.render_citation(evidence)
    scope = project.snapshot([revision])
    assert project.search("apples", scope)
    run = project.create_run(scope, {"profile": "fixture"})
    project.finalize_run(run)
    project.close()
    shutil.copytree(tmp_path / "one", tmp_path / "two")
    reopened = Project.open(tmp_path / "two")
    assert reopened.span_text(span) == "Apples are green."
    assert reopened.render_citation(evidence)
    reopened.close()


def test_analysis_run_summary_retains_scope_and_never_returns_raw_config_json(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Evidence remains local.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})

    summary = project.analysis_run_summary(run)

    assert summary["revision_ids"] == [revision]
    run_data = summary["analysis_run"]
    assert isinstance(run_data, dict)
    assert run_data["config"] == {"profile": "fixture"}
    assert run_data["config_json"] is None
    assert run_data["software_version"] == software_version()
    assert project.list_analysis_runs()[0]["id"] == run


def test_analysis_run_configuration_rejects_secret_keys_but_allows_tokenizer_metadata(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Evidence remains local.")
    scope = project.snapshot([revision])

    run = project.create_run(scope, {"tokenizer": "corpus.TOKEN", "credential_alias": "lab"})

    assert run
    with pytest.raises(DocumentError, match="must not contain secrets"):
        project.create_run(scope, {"secret_reference": "DL_KEY"})
    for field in ("token", "authorization", "api-token", "credentials"):
        with pytest.raises(DocumentError, match="must not contain secrets"):
            project.create_run(scope, {field: "sensitive-value"})


def test_analysis_run_rejects_unknown_scope_before_persisting(tmp_path):
    project = Project.init(tmp_path / "p")

    with pytest.raises(ScopeViolationError, match="unknown scope"):
        project.create_run("missing-scope", {})

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


def test_model_execution_rejects_unknown_analysis_run_before_persisting(tmp_path):
    project = Project.init(tmp_path / "p")

    with pytest.raises(DocumentError, match="unknown analysis run"):
        project.record_model_execution(
            "fixture-provider",
            "fixture-alias",
            "fixture-model",
            None,
            "request-hash",
            "response-hash",
            analysis_run_id="missing-run",
        )

    assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0


def test_revision_source_reads_verify_persisted_hashes(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text(
        "speech.txt", "Canonical text.", source_bytes=b"original source bytes"
    )
    row = project.db.execute(
        "SELECT text_path,original_path FROM document_revisions WHERE id=?", (revision,)
    ).fetchone()

    (project.hidden / row["text_path"]).write_text("Changed canonical text.", encoding="utf-8")
    with pytest.raises(DocumentError, match="canonical text integrity check"):
        project.revision_text(revision)
    with pytest.raises(DocumentError, match="canonical text integrity check"):
        project.create_span(revision, 0, 7)

    (project.hidden / row["text_path"]).write_text("Canonical text.", encoding="utf-8")
    (project.hidden / row["original_path"]).write_bytes(b"changed source bytes")
    with pytest.raises(DocumentError, match="original source integrity check"):
        project.original_bytes(revision)


def test_failed_duplicate_revision_insert_removes_uncommitted_source_files(tmp_path):
    project = Project.init(tmp_path / "p")
    document_id, _ = project.import_text("speech.txt", "Same canonical source.")
    source_directory = project.hidden / "source"
    existing_files = set(source_directory.iterdir())

    with pytest.raises(sqlite3.IntegrityError):
        project.import_text("speech.txt", "Same canonical source.", document_id=document_id)

    assert set(source_directory.iterdir()) == existing_files
    assert project.db.execute("SELECT COUNT(*) FROM document_revisions").fetchone()[0] == 1


def test_concurrent_project_metadata_initialization_keeps_one_identity(tmp_path):
    project = Project.init(tmp_path / "p")
    database_path = project.hidden / "project.sqlite"
    project.close()

    setup = sqlite3.connect(database_path)
    setup.execute("DELETE FROM project_meta")
    setup.commit()
    setup.close()

    barrier = threading.Barrier(2)

    def initialize_metadata() -> None:
        connection = sqlite3.connect(database_path, timeout=5)

        class BarrierConnection:
            def execute(self, sql, parameters=()):
                cursor = connection.execute(sql, parameters)
                if sql == "SELECT 1 FROM project_meta WHERE key='project_id'":
                    barrier.wait(timeout=5)
                return cursor

            def executemany(self, sql, parameters):
                if sql.startswith("INSERT OR IGNORE INTO project_meta"):
                    barrier.wait(timeout=5)
                return connection.executemany(sql, parameters)

            def commit(self) -> None:
                connection.commit()

        initializer = Project.__new__(Project)
        initializer.db = cast(sqlite3.Connection, BarrierConnection())
        try:
            initializer._ensure_project_metadata()
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(initialize_metadata) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)

    verification = sqlite3.connect(database_path)
    metadata = dict(verification.execute("SELECT key,value FROM project_meta").fetchall())
    verification.close()
    assert set(metadata) == {"project_id", "offline_mode"}
    assert metadata["project_id"]
    assert metadata["offline_mode"] == "false"


def test_historical_secret_bearing_run_is_rejected_by_inspection_and_exports(tmp_path):
    import json

    from discourselens.bundle import export_bundle
    from discourselens.errors import EvidenceValidationError
    from discourselens.reporting import analysis_payload

    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Evidence remains local.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    unsafe = json.dumps({"api-token": "sensitive-value"}, sort_keys=True)
    # Exercise the read-side integrity check after simulating an external tamper.
    project.db.execute("DROP TRIGGER analysis_runs_provenance_immutable")
    project.db.execute(
        "UPDATE analysis_runs SET config_json=?,config_hash=? WHERE id=?",
        (unsafe, hashlib.sha256(unsafe.encode()).hexdigest(), run),
    )
    project.db.commit()

    with pytest.raises(DocumentError, match="contains secret data"):
        project.analysis_run_summary(run)
    with pytest.raises(EvidenceValidationError, match="provenance is invalid"):
        analysis_payload(project, run)
    with pytest.raises(DocumentError, match="contains secret data"):
        export_bundle(project, run, tmp_path / "bundle")


def test_offsets_and_scope(tmp_path):
    project = Project.init(tmp_path / "p")
    _, a = project.import_text("a.txt", "Apples are green.")
    _, b = project.import_text("b.txt", "Elephants live in Africa.")
    with pytest.raises(SourceSpanError):
        project.create_span(a, 0, 100)
    assert not project.search("elephants", project.snapshot([a]))
    assert project.search("elephants", project.snapshot([b]))
    assert not project.search_exact("Elephants live", project.snapshot([a]))
    assert project.search_exact("Elephants live", project.snapshot([b]))
    project.close()


def test_coverage_and_manifest():
    with pytest.raises(ValueError):
        CoverageDescriptor(coverage_mode=CoverageMode.EXHAUSTIVE, scope_snapshot_id="s")
    manifest = parse_manifest(
        {
            "id": "fixture.exact",
            "version": "1.0.0",
            "framework": "fixture",
            "execution_kind": "DETERMINISTIC",
            "input_schema_ref": "i",
            "output_schema_ref": "o",
            "executor": "builtin.fixture",
            "retrieval_plan_ref": "retrieval/r",
            "documentation_ref": "d",
            "methodology_refs": ["fixture"],
            "required_capabilities": ["lexical"],
        }
    )
    assert not eligibility(manifest, set()).runnable
    assert eligibility(manifest, {"lexical"}).runnable


def test_finding_lineage_survives_reopen_and_revision(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision_a = project.import_text("source.txt", "Café is open.")
    span = project.create_span(revision_a, 0, 4)
    evidence = project.textual_evidence(span)
    run = project.create_run(project.snapshot([revision_a]), {})
    finding = project.create_finding(run, "fixture", "1.0.0", "example", "Café occurs")
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    _, revision_b = project.import_text("source.txt", "Café is closed.", extractor="plain-text@2")
    assert revision_a != revision_b
    project.close()
    project = Project.open(tmp_path / "p")
    linked = project.finding_evidence(finding)
    assert linked[0]["id"] == evidence
    assert project.span_text(linked[0]["span_id"]) == "Café"
    project.close()


def test_document_metadata_is_structured_and_does_not_change_revision(tmp_path):
    project = Project.init(tmp_path / "p")
    document, revision = project.import_text("speech.txt", "Evidence persists.")
    project.set_document_metadata(document, "year", 2025)
    project.set_document_metadata(document, "tags", ["speech", "public"])
    assert project.document_metadata(document) == {"tags": ["speech", "public"], "year": 2025}
    assert project.revision_text(revision) == "Evidence persists."
    project.close()


def test_metadata_selection_freezes_latest_matching_revisions_without_future_expansion(tmp_path):
    project = Project.init(tmp_path / "p")
    document_a, revision_a = project.import_text("a.txt", "Apples are green.")
    document_b, _ = project.import_text("b.txt", "Elephants live in Africa.")
    project.set_document_metadata(document_a, "genre", "speech")
    project.set_document_metadata(document_a, "year", 2025)
    project.set_document_metadata(document_b, "genre", "news")
    project.set_document_metadata(document_b, "year", 2025)

    scope_id, selected = project.snapshot_by_document_metadata({"genre": "speech", "year": 2025})
    assert selected == [revision_a]
    project.set_document_metadata(document_b, "genre", "speech")
    assert project.scope_revisions(scope_id) == [revision_a]

    with pytest.raises(ScopeViolationError, match="matched no document revisions"):
        project.snapshot_by_document_metadata({"year": "2025"})
    project.close()


def test_project_identity_and_offline_mode_are_portable_across_reopen_and_copy(tmp_path):
    from shutil import copytree

    project = Project.init(tmp_path / "original")
    project.set_offline_mode(True)
    identity = project.project_id
    project.close()
    reopened = Project.open(tmp_path / "original")
    assert reopened.project_id == identity and reopened.offline_mode
    reopened.close()
    copytree(tmp_path / "original", tmp_path / "copied")
    copied = Project.open(tmp_path / "copied")
    assert copied.project_id == identity and copied.offline_mode
    copied.close()


def test_sentences_are_persisted_with_exact_canonical_offsets(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("speech.txt", "Café migración. Otra frase!")
    text = project.revision_text(revision)
    sentences = project.db.execute(
        "SELECT * FROM sentences WHERE revision_id=? ORDER BY ordinal", (revision,)
    ).fetchall()
    assert [row["text"] for row in sentences] == ["Café migración.", "Otra frase!"]
    assert all(text[row["start_offset"] : row["end_offset"]] == row["text"] for row in sentences)
