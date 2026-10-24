from __future__ import annotations

import unicodedata

import pytest

from discourselens import Project
from discourselens.corpus import SURFACE_NORMALIZATION_V2, SURFACE_TOKENIZER_V2
from discourselens.errors import ArtifactIntegrityError, ScopeViolationError
from discourselens.nlp_artifacts import SURFACE_TOKEN_SCHEMA_VERSION


@pytest.fixture
def surface_frequency_project(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / "surface-frequency-page")
    _, first_revision = project.import_text("first.txt", "alpha beta alpha gamma")
    _, second_revision = project.import_text("second.txt", "alpha delta")
    _, outside_revision = project.import_text("outside.txt", "outside outside")
    first_artifact = project.build_surface_token_artifact(first_revision)
    second_artifact = project.build_surface_token_artifact(second_revision)
    outside_artifact = project.build_surface_token_artifact(outside_revision)
    scope_id = project.snapshot([first_revision, second_revision])
    try:
        yield (
            project,
            scope_id,
            [first_revision, second_revision],
            [first_artifact.artifact_id, second_artifact.artifact_id],
            outside_artifact.artifact_id,
            first_artifact,
        )
    finally:
        project.close()


def test_surface_frequency_page_ranks_normalized_counts_and_paginates(
    surface_frequency_project,
):
    project, scope_id, revisions, artifact_ids, *_ = surface_frequency_project

    first = project.corpus_surface_frequency_page(
        scope_id, artifact_ids, field="normalized", limit=2
    )
    second = project.corpus_surface_frequency_page(
        scope_id, artifact_ids, field="normalized", offset=2, limit=2
    )

    assert first == {
        "scope_id": scope_id,
        "revision_ids": revisions,
        "artifact_ids": sorted(artifact_ids),
        "artifact_schema_version": SURFACE_TOKEN_SCHEMA_VERSION,
        "calculation_version": "surface-token-artifact-frequency-v2",
        "tokenizer": SURFACE_TOKENIZER_V2,
        "normalization": SURFACE_NORMALIZATION_V2,
        "unicode_data_version": unicodedata.unidata_version,
        "field": "normalized",
        "offset": 0,
        "limit": 2,
        "total_values": 4,
        "has_more": True,
        "rows": [
            {"token": "alpha", "count": 3},
            {"token": "beta", "count": 1},
        ],
    }
    assert second["rows"] == [
        {"token": "delta", "count": 1},
        {"token": "gamma", "count": 1},
    ]
    assert second["has_more"] is False


def test_surface_frequency_page_ranks_revision_document_frequency(
    surface_frequency_project,
):
    project, scope_id, _, artifact_ids, *_ = surface_frequency_project

    page = project.corpus_surface_frequency_page(
        scope_id, artifact_ids, field="document_frequency", limit=5
    )

    assert page["field"] == "document_frequency"
    assert page["rows"] == [
        {"token": "alpha", "count": 2},
        {"token": "beta", "count": 1},
        {"token": "delta", "count": 1},
        {"token": "gamma", "count": 1},
    ]
    assert page["total_values"] == 4
    assert page["has_more"] is False


def test_v2_surface_artifacts_normalize_composed_and_decomposed_unicode_like_core(
    tmp_path,
):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / "surface-frequency-unicode-v2")
    _, revision = project.import_text(
        "unicode.txt", "cafe\u0301 migration café migration CAFÉ"
    )
    artifact = project.build_surface_token_artifact(revision)
    scope_id = project.snapshot([revision])

    page = project.corpus_surface_frequency_page(
        scope_id, [artifact.artifact_id], field="normalized", limit=10
    )

    assert page["rows"] == [
        {"token": "café", "count": 3},
        {"token": "migration", "count": 2},
    ]
    assert page["total_values"] == 2
    assert page["tokenizer"] == SURFACE_TOKENIZER_V2
    assert page["normalization"] == SURFACE_NORMALIZATION_V2
    assert page["unicode_data_version"] == unicodedata.unidata_version
    assert project.corpus_frequency(scope_id) == {"café": 3, "migration": 2}
    assert project.corpus_ngrams(scope_id, 1) == {("café",): 3, ("migration",): 2}
    assert project.corpus_collocations(scope_id, "café", window=1) == {"migration": 4}
    project.close()


@pytest.mark.parametrize(
    ("field", "offset", "limit"),
    [
        ("unknown", 0, 10),
        ("normalized", -1, 10),
        ("normalized", True, 10),
        ("normalized", 0, 0),
        ("normalized", 0, 1_001),
    ],
)
def test_surface_frequency_page_rejects_invalid_requests(
    surface_frequency_project, field, offset, limit
):
    project, scope_id, _, artifact_ids, *_ = surface_frequency_project

    with pytest.raises(ScopeViolationError):
        project.corpus_surface_frequency_page(
            scope_id,
            artifact_ids,
            field=field,
            offset=offset,
            limit=limit,
        )


def test_surface_frequency_page_requires_complete_scoped_artifacts(
    surface_frequency_project,
):
    project, scope_id, _, artifact_ids, outside_artifact_id, *_ = surface_frequency_project

    with pytest.raises(ScopeViolationError, match="one compatible artifact per scoped revision"):
        project.corpus_surface_frequency_page(
            scope_id, artifact_ids[:1], field="normalized"
        )
    with pytest.raises(ScopeViolationError):
        project.corpus_surface_frequency_page(
            scope_id,
            [artifact_ids[0], outside_artifact_id],
            field="normalized",
        )


def test_surface_frequency_page_checks_artifact_hashes(surface_frequency_project):
    project, scope_id, _, artifact_ids, _, first_artifact = surface_frequency_project
    artifact_path = project.hidden / first_artifact.relative_path
    artifact_path.write_bytes(b"corrupted artifact")

    with pytest.raises(ArtifactIntegrityError, match="integrity check"):
        project.corpus_surface_frequency_page(
            scope_id, artifact_ids, field="normalized"
        )
