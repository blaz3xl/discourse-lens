import re
import unicodedata

import pytest

from discourselens import Project
from discourselens.errors import (
    ArtifactIntegrityError,
    EvidenceValidationError,
    ScopeViolationError,
)
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import PARSED_TOKEN_SCHEMA_VERSION, build_parsed_token_artifact
from discourselens.types import CoverageMode


class _EvidenceLemmaBackend:
    language = "en"
    backend_id = "lemma-evidence-fixture"
    backend_version = "1.2.3"
    model_version = "4.5.6"

    def __init__(self, model_id="fixture-model-v1"):
        self.model_id = model_id

    def parse(self, text):
        tokens = []
        for index, match in enumerate(re.finditer(r"\S+", text)):
            surface = match.group()
            lexical = surface.strip(".,!?;:()[]{}\"'“”‘’")
            lemma = unicodedata.normalize("NFC", lexical.casefold())
            tokens.append(
                ParsedToken(
                    surface,
                    lemma,
                    "NOUN",
                    None,
                    None,
                    "ROOT" if index == 0 else "dep",
                    match.start(),
                    match.end(),
                )
            )
        return tokens


def _artifact(project, revision, backend=None):
    return build_parsed_token_artifact(project, revision, backend or _EvidenceLemmaBackend())


def test_lemma_kwic_evidence_persists_one_canonical_example_and_survives_reopen(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    root = tmp_path / "lemma-kwic-evidence"
    project = Project.init(root)
    canonical = "The cafe\u0301,\t café!"
    _, revision = project.import_text("unicode.txt", canonical)
    _, other_revision = project.import_text("other.txt", "café elsewhere.")
    artifacts = [
        _artifact(project, revision),
        _artifact(project, other_revision),
    ]
    artifact_ids = [item.artifact_id for item in artifacts]
    scope = project.snapshot([revision, other_revision])
    occurrence = project.corpus_lemma_kwic(
        scope, artifact_ids, "café", window=0, limit=10
    )["rows"][0]

    result = project.corpus_lemma_kwic_evidence(
        scope,
        artifact_ids,
        revision,
        "café",
        occurrence["start_offset"],
        occurrence["end_offset"],
    )

    assert result["scope_id"] == scope
    assert result["revision_id"] == revision
    assert result["artifact_ids"] == sorted(artifact_ids)
    assert result["exact_text"] == "cafe\u0301,"
    assert canonical[result["start_offset"] : result["end_offset"]] == result["exact_text"]
    assert result["coverage_mode"] == CoverageMode.EXAMPLE_BASED.value
    assert result["backend"] == "lemma-evidence-fixture"
    assert result["backend_version"] == "1.2.3"
    assert result["model"] == "fixture-model-v1"
    assert result["model_version"] == "4.5.6"
    assert result["schema_version"] == PARSED_TOKEN_SCHEMA_VERSION
    assert result["field_coverage"]["status"] == "SUPPORTED"
    assert result["citation"] == "[unicode.txt · ¶1]"

    evidence = project.evidence_content(result["evidence_id"])
    assert evidence["acquisition_method"] == "corpus_lemma_kwic"
    assert evidence["scope_snapshot_id"] == scope
    assert evidence["artifact_ids"] == sorted(artifact_ids)
    assert evidence["lemma"] == "café"
    assert evidence["coverage_descriptor_id"] == result["coverage_id"]
    assert evidence["coverage_mode"] == CoverageMode.EXAMPLE_BASED.value
    assert evidence["backend_version"] == "1.2.3"
    assert evidence["model_version"] == "4.5.6"
    assert evidence["schema_version"] == PARSED_TOKEN_SCHEMA_VERSION
    field_coverage = evidence.get("field_coverage")
    assert isinstance(field_coverage, dict)
    assert field_coverage.get("status") == "SUPPORTED"
    result_semantics = evidence["result_semantics"]
    assert isinstance(result_semantics, str)
    assert "one parser-annotated example" in result_semantics
    coverage = project.coverage_descriptor(result["coverage_id"])
    assert coverage.scope_snapshot_id == scope
    assert coverage.coverage_mode is CoverageMode.EXAMPLE_BASED
    assert coverage.population_count is None
    assert coverage.inspection_count == 1

    project.close()
    reopened = Project.open(root)
    try:
        assert reopened.evidence_content(result["evidence_id"]) == evidence
        assert reopened.coverage_descriptor(result["coverage_id"]).coverage_mode is (
            CoverageMode.EXAMPLE_BASED
        )
        assert reopened.render_citation(result["evidence_id"]) == result["citation"]
    finally:
        reopened.close()


def test_lemma_kwic_evidence_rejects_unknown_incomplete_and_incompatible_artifacts(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / "lemma-kwic-evidence-artifacts")
    _, first_revision = project.import_text("first.txt", "run.")
    _, second_revision = project.import_text("second.txt", "run.")
    scope = project.snapshot([first_revision, second_revision])
    first_artifact = _artifact(project, first_revision)
    second_artifact = _artifact(
        project,
        second_revision,
        _EvidenceLemmaBackend(model_id="different-model-v2"),
    )

    with pytest.raises(ScopeViolationError, match="unknown or outside the explicit scope"):
        project.corpus_lemma_kwic_evidence(
            scope, ["unknown-artifact"], first_revision, "run", 0, 4
        )
    with pytest.raises(ScopeViolationError, match="one compatible artifact per scoped revision"):
        project.corpus_lemma_kwic_evidence(
            scope, [first_artifact.artifact_id], first_revision, "run", 0, 4
        )
    with pytest.raises(ScopeViolationError, match="one compatible parser and model"):
        project.corpus_lemma_kwic_evidence(
            scope,
            [first_artifact.artifact_id, second_artifact.artifact_id],
            first_revision,
            "run",
            0,
            4,
        )


def test_lemma_kwic_evidence_rejects_unknown_out_of_scope_and_wrong_lemma(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / "lemma-kwic-evidence-scope")
    _, included_revision = project.import_text("included.txt", "running.")
    _, excluded_revision = project.import_text("excluded.txt", "run.")
    artifact = _artifact(project, included_revision)
    scope = project.snapshot([included_revision])
    unknown_revision = "unknown-revision"

    for revision_id in (excluded_revision, unknown_revision):
        with pytest.raises(EvidenceValidationError, match="not a lemma KWIC hit in this scope"):
            project.corpus_lemma_kwic_evidence(
                scope, [artifact.artifact_id], revision_id, "run", 0, 7
            )
    with pytest.raises(EvidenceValidationError, match="not a lemma KWIC hit in this scope"):
        project.corpus_lemma_kwic_evidence(
            scope, [artifact.artifact_id], included_revision, "run", 0, 8
        )


@pytest.mark.parametrize(
    ("start_offset", "end_offset"),
    [(-1, 3), (3, 3), (0, 9), (True, 3)],
)
def test_lemma_kwic_evidence_rejects_offsets_outside_revision_or_token_boundary(
    tmp_path, start_offset, end_offset
):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / f"lemma-kwic-evidence-offset-{start_offset}-{end_offset}")
    _, revision = project.import_text("text.txt", "run now.")
    artifact = _artifact(project, revision)
    scope = project.snapshot([revision])

    with pytest.raises(EvidenceValidationError, match="selected offsets"):
        project.corpus_lemma_kwic_evidence(
            scope,
            [artifact.artifact_id],
            revision,
            "run",
            start_offset,
            end_offset,
        )


def test_lemma_kwic_evidence_rejects_tampered_artifact_before_persistence(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / "lemma-kwic-evidence-tamper")
    _, revision = project.import_text("text.txt", "run.")
    artifact = _artifact(project, revision)
    scope = project.snapshot([revision])
    path = project.hidden / artifact.relative_path
    path.write_bytes(path.read_bytes() + b"tamper")
    before = project.db.execute("SELECT count(*) FROM evidence_artifacts").fetchone()[0]

    with pytest.raises(ArtifactIntegrityError, match="integrity check failed"):
        project.corpus_lemma_kwic_evidence(
            scope, [artifact.artifact_id], revision, "run", 0, 4
        )

    after = project.db.execute("SELECT count(*) FROM evidence_artifacts").fetchone()[0]
    assert after == before
