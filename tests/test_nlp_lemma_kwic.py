import re
import unicodedata

import pytest

from discourselens import Project
from discourselens.errors import ArtifactIntegrityError, ScopeViolationError
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import build_parsed_token_artifact


class _LemmaBackend:
    language = "en"
    backend_id = "lemma-kwic-fixture"
    backend_version = "1.0"
    model_version = "1.0"

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
    return build_parsed_token_artifact(project, revision, backend or _LemmaBackend())


def test_lemma_kwic_uses_exact_lemma_and_canonical_context(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / "lemma-kwic-canonical")
    canonical = "\tCafe\u0301,\t café! \n\nRUN   run."
    _, revision = project.import_text("unicode.txt", canonical)
    artifact = _artifact(project, revision)
    scope = project.snapshot([revision])

    page = project.corpus_lemma_kwic(
        scope, [artifact.artifact_id], "café", window=1, limit=10
    )

    first_start = canonical.index("Cafe\u0301,")
    second_start = canonical.index("café!")
    assert page["total_count"] == 2
    assert page["lemma"] == "café"
    assert [row["keyword"] for row in page["rows"]] == ["Cafe\u0301,", "café!"]
    assert [row["start_offset"] for row in page["rows"]] == [first_start, second_start]
    assert all(
        canonical[row["start_offset"] : row["end_offset"]] == row["keyword"]
        for row in page["rows"]
    )
    assert page["rows"][0]["left"] == "\t"
    assert page["rows"][0]["right"] == "\t café!"
    assert page["rows"][1]["left"] == "Cafe\u0301,\t "
    assert page["rows"][1]["right"] == " "
    assert all(row["revision_id"] == revision for row in page["rows"])
    assert page["rows"][0]["paragraph_id"] == page["rows"][1]["paragraph_id"]

    case_mismatch = project.corpus_lemma_kwic(
        scope, [artifact.artifact_id], "CAFÉ", window=1
    )
    assert case_mismatch["total_count"] == 0
    assert case_mismatch["rows"] == []


def test_lemma_kwic_pages_are_stable_and_isolated_to_the_snapshot(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / "lemma-kwic-pages")
    _, first_revision = project.import_text("first.txt", "run  RUN.")
    _, second_revision = project.import_text("second.txt", "run.")
    _, outside_revision = project.import_text("outside.txt", "run run run.")
    first_artifact = _artifact(project, first_revision)
    second_artifact = _artifact(project, second_revision)
    outside_artifact = _artifact(project, outside_revision)
    scope = project.snapshot([second_revision, first_revision])
    explicit_ids = [second_artifact.artifact_id, first_artifact.artifact_id]

    first_page = project.corpus_lemma_kwic(
        scope, explicit_ids, "run", window=0, offset=0, limit=1
    )
    second_page = project.corpus_lemma_kwic(
        scope, explicit_ids, "run", window=0, offset=1, limit=1
    )
    third_page = project.corpus_lemma_kwic(
        scope, explicit_ids, "run", window=0, offset=2, limit=1
    )

    assert first_page["total_count"] == second_page["total_count"] == 3
    assert first_page["has_more"] and second_page["has_more"]
    assert not third_page["has_more"]
    assert first_page["rows"][0]["revision_id"] == second_revision
    assert second_page["rows"][0]["revision_id"] == first_revision
    assert third_page["rows"][0]["revision_id"] == first_revision
    assert first_page["rows"][0]["keyword"] == "run."
    assert first_page["artifact_ids"] == sorted(explicit_ids)

    with pytest.raises(ScopeViolationError, match="outside the explicit scope"):
        project.corpus_lemma_kwic(
            scope,
            [first_artifact.artifact_id, outside_artifact.artifact_id],
            "run",
        )


def test_lemma_kwic_requires_complete_compatible_artifacts(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / "lemma-kwic-completeness")
    _, first_revision = project.import_text("first.txt", "run.")
    _, second_revision = project.import_text("second.txt", "run.")
    scope = project.snapshot([first_revision, second_revision])
    first_artifact = _artifact(project, first_revision)
    second_artifact = _artifact(
        project,
        second_revision,
        _LemmaBackend(model_id="different-model-v2"),
    )

    with pytest.raises(ScopeViolationError, match="one compatible artifact per scoped revision"):
        project.corpus_lemma_kwic(scope, [first_artifact.artifact_id], "run")
    with pytest.raises(ScopeViolationError, match="one compatible parser and model"):
        project.corpus_lemma_kwic(
            scope,
            [first_artifact.artifact_id, second_artifact.artifact_id],
            "run",
        )


def test_lemma_kwic_rejects_corrupt_artifact_and_empty_result_is_valid(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / "lemma-kwic-integrity")
    _, revision = project.import_text("text.txt", "nothing here.")
    artifact = _artifact(project, revision)
    scope = project.snapshot([revision])

    empty = project.corpus_lemma_kwic(scope, [artifact.artifact_id], "missing")
    assert empty["total_count"] == 0
    assert empty["rows"] == []
    assert not empty["has_more"]
    assert "do not establish raw-text" in empty["result_semantics"]

    path = project.hidden / artifact.relative_path
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ArtifactIntegrityError, match="integrity check failed"):
        project.corpus_lemma_kwic(scope, [artifact.artifact_id], "nothing")


@pytest.mark.parametrize(
    ("window", "offset", "limit"),
    [(-1, 0, 1), (101, 0, 1), (0, -1, 1), (0, 0, 0), (0, 0, 1001), (True, 0, 1)],
)
def test_lemma_kwic_rejects_unbounded_or_non_integer_parameters(
    tmp_path, window, offset, limit
):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project = Project.init(tmp_path / f"lemma-kwic-bounds-{window}-{offset}-{limit}")
    _, revision = project.import_text("text.txt", "run.")
    artifact = _artifact(project, revision)
    scope = project.snapshot([revision])

    with pytest.raises(ScopeViolationError, match="lemma KWIC"):
        project.corpus_lemma_kwic(
            scope,
            [artifact.artifact_id],
            "run",
            window=window,
            offset=offset,
            limit=limit,
        )
