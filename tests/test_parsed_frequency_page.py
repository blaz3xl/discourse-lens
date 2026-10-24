from __future__ import annotations

import re

import pytest

from discourselens import Project
from discourselens.errors import ScopeViolationError
from discourselens.nlp import ParsedToken


class _ParsedFixture:
    language = "en"
    backend_id = "parsed-frequency-fixture"
    model_id = "parsed-frequency-fixture-v1"

    def parse(self, text: str) -> list[ParsedToken]:
        result: list[ParsedToken] = []
        for index, match in enumerate(re.finditer(r"\w+", text)):
            surface = match.group()
            lemma = surface.casefold().rstrip(".")
            result.append(
                ParsedToken(
                    surface=surface,
                    lemma=lemma,
                    pos="VERB" if lemma == "beta" else "NOUN",
                    morphology=None,
                    dependency_head=None if index == 0 else 0,
                    dependency_relation="ROOT" if index == 0 else "obj",
                    char_start=match.start(),
                    char_end=match.end(),
                )
            )
        return result


@pytest.fixture
def parsed_frequency_project(tmp_path):
    project = Project.init(tmp_path / "parsed-frequency")
    _, selected_revision = project.import_text("selected.txt", "Alpha alpha beta gamma beta.")
    _, other_revision = project.import_text("other.txt", "Alpha outside.")
    selected_artifact = project.build_parsed_token_artifact(selected_revision, _ParsedFixture())
    other_artifact = project.build_parsed_token_artifact(other_revision, _ParsedFixture())
    scope_id = project.snapshot([selected_revision])
    try:
        yield project, scope_id, selected_revision, selected_artifact, other_artifact
    finally:
        project.close()


def test_sdk_parsed_frequency_page_is_stable_bounded_and_scope_explicit(
    parsed_frequency_project,
):
    project, scope_id, revision, selected_artifact, _ = parsed_frequency_project

    first_page = project.corpus_parsed_frequency_page(
        scope_id, [selected_artifact.artifact_id], field="lemma", limit=2
    )
    second_page = project.corpus_parsed_frequency_page(
        scope_id, [selected_artifact.artifact_id], field="lemma", offset=2, limit=2
    )

    assert first_page["scope_id"] == scope_id
    assert first_page["revision_ids"] == [revision]
    assert first_page["artifact_ids"] == [selected_artifact.artifact_id]
    assert first_page["field"] == "lemma"
    assert first_page["offset"] == 0
    assert first_page["limit"] == 2
    assert first_page["total_values"] == 3
    assert first_page["has_more"] is True
    assert first_page["rows"] == [
        {"value": "alpha", "count": 2},
        {"value": "beta", "count": 2},
    ]
    assert first_page["coverage"]["tokenization"]["status"] == "LIMITED"
    assert first_page["field_coverage"]["status"] == "LIMITED"
    assert second_page["rows"] == [{"value": "gamma", "count": 1}]
    assert second_page["has_more"] is False


@pytest.mark.parametrize(
    ("field", "expected_rows"),
    [
        ("pos", [{"value": "NOUN", "count": 3}, {"value": "VERB", "count": 2}]),
        ("dependency_relation", [{"value": "obj", "count": 4}, {"value": "ROOT", "count": 1}]),
    ],
)
def test_sdk_parsed_frequency_page_supports_all_declared_fields(
    parsed_frequency_project, field, expected_rows
):
    project, scope_id, _, artifact, _ = parsed_frequency_project

    result = project.corpus_parsed_frequency_page(
        scope_id, [artifact.artifact_id], field=field, limit=10
    )

    assert result["field"] == field
    assert result["rows"] == expected_rows
    assert result["has_more"] is False
    assert result["field_coverage"]["status"] == "LIMITED"


def test_sdk_parsed_frequency_page_rejects_artifacts_outside_scope(parsed_frequency_project):
    project, scope_id, _, selected_artifact, other_artifact = parsed_frequency_project

    with pytest.raises(ScopeViolationError):
        project.corpus_parsed_frequency_page(
            scope_id,
            [selected_artifact.artifact_id, other_artifact.artifact_id],
            field="lemma",
        )


@pytest.mark.parametrize(
    ("field", "offset", "limit"),
    [
        ("unknown", 0, 10),
        ("lemma", -1, 10),
        ("lemma", True, 10),
        ("lemma", 0, 0),
        ("lemma", 0, 1_001),
    ],
)
def test_sdk_parsed_frequency_page_rejects_invalid_requests(
    parsed_frequency_project, field, offset, limit
):
    project, scope_id, _, artifact, _ = parsed_frequency_project

    with pytest.raises(ScopeViolationError):
        project.corpus_parsed_frequency_page(
            scope_id,
            [artifact.artifact_id],
            field=field,
            offset=offset,
            limit=limit,
        )
