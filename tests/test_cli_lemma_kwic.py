import json
import re

import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import build_parsed_token_artifact


@pytest.fixture
def project_with_revisions(tmp_path):
    root = tmp_path / "research"
    project = Project.init(root)
    _, first = project.import_text("first.txt", "Migrants support migration.")
    _, second = project.import_text("second.txt", "A migrant speaks.")
    project.close()
    return root, first, second


def test_lemma_kwic_forwards_explicit_scope_and_page_parameters(
    monkeypatch, project_with_revisions
):
    root, first, second = project_with_revisions
    captured = {}

    def fake_lemma_kwic(self, scope_id, artifact_ids, lemma, *, window, offset, limit):
        captured.update(
            scope_id=scope_id,
            revision_ids=self.scope_revisions(scope_id),
            artifact_ids=artifact_ids,
            lemma=lemma,
            window=window,
            offset=offset,
            limit=limit,
        )
        return {
            "scope_id": scope_id,
            "lemma": lemma,
            "window": window,
            "offset": offset,
            "limit": limit,
            "total_count": 4,
            "has_more": True,
            "rows": [
                {
                    "revision_id": first,
                    "paragraph_id": "paragraph-1",
                    "start_offset": 9,
                    "end_offset": 15,
                    "left": "Migrants support",
                    "keyword": "migrant",
                    "lemma": lemma,
                    "right": "migration.",
                }
            ],
        }

    monkeypatch.setattr(Project, "corpus_lemma_kwic", fake_lemma_kwic, raising=False)
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "lemma-kwic",
            str(root),
            "migrant",
            "--revisions",
            f"{first}, {second}",
            "--artifacts",
            "artifact-first,artifact-second",
            "--window",
            "3",
            "--offset",
            "2",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    page = json.loads(result.stdout)
    assert captured["revision_ids"] == [first, second]
    assert captured["artifact_ids"] == ["artifact-first", "artifact-second"]
    assert captured["lemma"] == "migrant"
    assert (captured["window"], captured["offset"], captured["limit"]) == (3, 2, 1)
    assert page["scope_id"] == captured["scope_id"]
    assert page["rows"][0]["keyword"] == "migrant"
    assert page["rows"][0]["revision_id"] == first


def test_lemma_kwic_uses_only_selected_revisions_and_parsed_artifacts(tmp_path):
    class Backend:
        language = "en"
        backend_id = "fixture"
        model_id = "fixture-v1"

        def parse(self, text):
            lemmas = {
                "A": "a",
                "Migrants": "migrant",
                "Migrant": "migrant",
                "migrant": "migrant",
                "migration": "migration",
                "only": "only",
                "speaks": "speak",
                "support": "support",
            }
            return [
                ParsedToken(
                    surface,
                    lemmas[surface],
                    "NOUN",
                    None,
                    None,
                    "ROOT",
                    match.start(),
                    match.end(),
                )
                for match in re.finditer(r"\w+", text)
                if (surface := match.group()) in lemmas
            ]

    root = tmp_path / "research"
    project = Project.init(root)
    _, first = project.import_text("first.txt", "Migrants support migration.")
    _, second = project.import_text("second.txt", "A migrant speaks.")
    _, outside = project.import_text("outside.txt", "Migrant only.")
    backend = Backend()
    first_artifact = build_parsed_token_artifact(project, first, backend)
    second_artifact = build_parsed_token_artifact(project, second, backend)
    outside_artifact = build_parsed_token_artifact(project, outside, backend)
    project.close()

    runner = CliRunner()
    common = [
        "corpus",
        "lemma-kwic",
        str(root),
        "migrant",
        "--revisions",
        f"{first},{second}",
        "--artifacts",
        f"{first_artifact.artifact_id},{second_artifact.artifact_id}",
        "--window",
        "1",
        "--limit",
        "1",
    ]
    first_page = runner.invoke(app, [*common, "--offset", "0"])
    second_page = runner.invoke(app, [*common, "--offset", "1"])
    wrong_scope = runner.invoke(
        app,
        [
            "corpus",
            "lemma-kwic",
            str(root),
            "migrant",
            "--revisions",
            first,
            "--artifacts",
            second_artifact.artifact_id,
        ],
    )

    assert first_page.exit_code == second_page.exit_code == 0
    first_payload = json.loads(first_page.stdout)
    second_payload = json.loads(second_page.stdout)
    assert first_payload["total_count"] == second_payload["total_count"] == 2
    assert first_payload["has_more"] is True
    assert second_payload["has_more"] is False
    assert {
        row["revision_id"]
        for row in first_payload["rows"] + second_payload["rows"]
    } == {first, second}
    assert set(first_payload["artifact_ids"]) == {
        first_artifact.artifact_id,
        second_artifact.artifact_id,
    }
    second_revision_row = next(
        row
        for row in first_payload["rows"] + second_payload["rows"]
        if row["revision_id"] == second
    )
    assert second_revision_row["left"] == "A "
    assert second_revision_row["keyword"] == "migrant"
    assert second_revision_row["right"] == " speaks"
    assert outside not in {
        row["revision_id"] for row in first_payload["rows"] + second_payload["rows"]
    }
    assert outside_artifact.artifact_id not in first_payload["artifact_ids"]
    assert wrong_scope.exit_code == 2
    assert "Traceback" not in wrong_scope.output


def test_lemma_kwic_evidence_forwards_exact_selection_and_returns_persisted_record(
    monkeypatch, project_with_revisions
):
    root, first, second = project_with_revisions
    captured = {}

    def fake_lemma_kwic_evidence(
        self, scope_id, artifact_ids, revision_id, lemma, start_offset, end_offset
    ):
        captured.update(
            scope_id=scope_id,
            revision_ids=self.scope_revisions(scope_id),
            artifact_ids=artifact_ids,
            revision_id=revision_id,
            lemma=lemma,
            start_offset=start_offset,
            end_offset=end_offset,
        )
        return {
            "scope_id": scope_id,
            "revision_id": revision_id,
            "lemma": lemma,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "exact_text": "Migrants",
            "coverage_id": "coverage-1",
            "evidence_id": "evidence-1",
            "citation": "[first.txt · ¶1]",
        }

    monkeypatch.setattr(
        Project, "corpus_lemma_kwic_evidence", fake_lemma_kwic_evidence, raising=False
    )
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "lemma-kwic-evidence",
            str(root),
            "migrant",
            "--revisions",
            f"{first}, {second}",
            "--artifacts",
            "artifact-first,artifact-second",
            "--revision-id",
            first,
            "--start-offset",
            "0",
            "--end-offset",
            "8",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert captured["revision_ids"] == [first, second]
    assert captured["artifact_ids"] == ["artifact-first", "artifact-second"]
    assert captured["revision_id"] == first
    assert captured["lemma"] == "migrant"
    assert (captured["start_offset"], captured["end_offset"]) == (0, 8)
    assert payload["exact_text"] == "Migrants"
    assert payload["evidence_id"] == "evidence-1"
    assert payload["citation"] == "[first.txt · ¶1]"


def test_lemma_kwic_evidence_validates_required_scope_and_offsets(project_with_revisions):
    root, first, second = project_with_revisions
    runner = CliRunner()
    common = [
        "corpus",
        "lemma-kwic-evidence",
        str(root),
        "migrant",
        "--revisions",
        first,
        "--artifacts",
        "artifact-first",
        "--revision-id",
        first,
        "--start-offset",
        "0",
        "--end-offset",
        "1",
    ]
    missing_revisions = runner.invoke(
        app,
        [
            "corpus",
            "lemma-kwic-evidence",
            str(root),
            "migrant",
            "--artifacts",
            "artifact-first",
            "--revision-id",
            first,
            "--start-offset",
            "0",
            "--end-offset",
            "1",
        ],
    )
    missing_artifacts = runner.invoke(
        app, [*common[:6], *common[8:]]
    )
    empty_revisions = runner.invoke(
        app, [*common[:4], "--revisions", " , ", *common[6:]]
    )
    empty_artifacts = runner.invoke(
        app, [*common[:4], "--revisions", first, "--artifacts", " , ", *common[8:]]
    )
    mismatched_revision = runner.invoke(
        app, [*common[:9], second, *common[10:]]
    )
    reversed_offsets = runner.invoke(
        app,
        [*common[:10], "--start-offset", "2", "--end-offset", "1"],
    )
    negative_start = runner.invoke(
        app,
        [*common[:10], "--start-offset", "-1", "--end-offset", "1"],
    )
    zero_end = runner.invoke(
        app,
        [*common[:10], "--start-offset", "0", "--end-offset", "0"],
    )

    assert missing_revisions.exit_code == 2
    assert "--revisions" in missing_revisions.output
    assert missing_artifacts.exit_code == 2
    assert "--artifacts" in missing_artifacts.output
    assert empty_revisions.exit_code == 2
    assert "must include at least one" in empty_revisions.output
    assert empty_artifacts.exit_code == 2
    assert "must include at least one" in empty_artifacts.output
    assert mismatched_revision.exit_code == 2
    assert "must be included" in mismatched_revision.output
    assert reversed_offsets.exit_code == 2
    assert "greater than" in reversed_offsets.output
    assert negative_start.exit_code == zero_end.exit_code == 2
    assert "--start-offset" in negative_start.output
    assert "--end-offset" in zero_end.output


def test_lemma_kwic_evidence_persists_selected_artifact_occurrence(tmp_path):
    class Backend:
        language = "en"
        backend_id = "fixture"
        model_id = "fixture-v1"

        def parse(self, text):
            return [
                ParsedToken(
                    match.group(),
                    (
                        "migrant"
                        if match.group() in {"Migrants", "Migrant", "migrant"}
                        else match.group().casefold()
                    ),
                    "NOUN",
                    None,
                    None,
                    "ROOT",
                    match.start(),
                    match.end(),
                )
                for match in re.finditer(r"\w+", text)
            ]

    root = tmp_path / "research"
    project = Project.init(root)
    _, selected = project.import_text("speech.txt", "Migrants support migration.")
    _, outside = project.import_text("outside.txt", "Migrant appears outside this scope.")
    backend = Backend()
    selected_artifact = build_parsed_token_artifact(project, selected, backend)
    outside_artifact = build_parsed_token_artifact(project, outside, backend)
    project.close()

    runner = CliRunner()
    page = runner.invoke(
        app,
        [
            "corpus",
            "lemma-kwic",
            str(root),
            "migrant",
            "--revisions",
            selected,
            "--artifacts",
            selected_artifact.artifact_id,
        ],
    )
    assert page.exit_code == 0, page.output
    row = json.loads(page.stdout)["rows"][0]

    evidence_result = runner.invoke(
        app,
        [
            "corpus",
            "lemma-kwic-evidence",
            str(root),
            "migrant",
            "--revisions",
            selected,
            "--artifacts",
            selected_artifact.artifact_id,
            "--revision-id",
            row["revision_id"],
            "--start-offset",
            str(row["start_offset"]),
            "--end-offset",
            str(row["end_offset"]),
        ],
    )

    assert evidence_result.exit_code == 0, evidence_result.output
    evidence = json.loads(evidence_result.stdout)
    assert evidence["revision_id"] == selected
    assert evidence["artifact_ids"] == [selected_artifact.artifact_id]
    assert evidence["lemma"] == "migrant"
    assert evidence["exact_text"] == "Migrants"
    assert evidence["evidence_id"]
    assert evidence["coverage_id"]
    assert evidence["coverage_mode"] == "EXAMPLE_BASED"
    assert evidence["citation"] == "[speech.txt · ¶1]"
    assert outside_artifact.artifact_id not in evidence["artifact_ids"]


@pytest.mark.parametrize(
    ("output_format", "expected"),
    [
        ("jsonl", '"revision_id"'),
        ("csv", "revision_id,paragraph_id,start_offset,end_offset,left,keyword,lemma,right"),
        ("table", "revision_id"),
    ],
)
def test_lemma_kwic_renders_structured_rows_in_supported_formats(
    monkeypatch, project_with_revisions, output_format, expected
):
    root, first, _ = project_with_revisions

    def fake_lemma_kwic(self, scope_id, artifact_ids, lemma, *, window, offset, limit):
        return {
            "scope_id": scope_id,
            "lemma": lemma,
            "window": window,
            "offset": offset,
            "limit": limit,
            "total_count": 1,
            "has_more": False,
            "rows": [
                {
                    "revision_id": first,
                    "paragraph_id": "paragraph-1",
                    "start_offset": 9,
                    "end_offset": 15,
                    "left": "Migrants support",
                    "keyword": "migrant",
                    "lemma": lemma,
                    "right": "migration.",
                }
            ],
        }

    monkeypatch.setattr(Project, "corpus_lemma_kwic", fake_lemma_kwic, raising=False)
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "lemma-kwic",
            str(root),
            "migrant",
            "--revisions",
            first,
            "--artifacts",
            "artifact-first",
            "--format",
            output_format,
        ],
    )

    assert result.exit_code == 0, result.output
    assert expected in result.stdout
    assert "has_more=false" in result.stderr
    if output_format == "jsonl":
        row = json.loads(result.stdout)
        assert row["keyword"] == "migrant"


def test_lemma_kwic_requires_nonempty_explicit_revision_and_artifact_lists(
    project_with_revisions,
):
    root, first, _ = project_with_revisions
    runner = CliRunner()
    missing_revisions = runner.invoke(
        app,
        [
            "corpus",
            "lemma-kwic",
            str(root),
            "migrant",
            "--artifacts",
            "artifact-first",
        ],
    )
    missing_artifacts = runner.invoke(
        app,
        ["corpus", "lemma-kwic", str(root), "migrant", "--revisions", first],
    )
    empty_revisions = runner.invoke(
        app,
        [
            "corpus",
            "lemma-kwic",
            str(root),
            "migrant",
            "--revisions",
            " , ",
            "--artifacts",
            "artifact-first",
        ],
    )
    empty_artifacts = runner.invoke(
        app,
        [
            "corpus",
            "lemma-kwic",
            str(root),
            "migrant",
            "--revisions",
            first,
            "--artifacts",
            " , ",
        ],
    )

    assert missing_revisions.exit_code == 2
    assert "--revisions" in missing_revisions.output
    assert missing_artifacts.exit_code == 2
    assert "--artifacts" in missing_artifacts.output
    assert empty_revisions.exit_code == 2
    assert "must include at least one" in empty_revisions.output
    assert "revision ID" in empty_revisions.output
    assert empty_artifacts.exit_code == 2
    assert "must include at least one" in empty_artifacts.output
    assert "parsed-token artifact ID" in empty_artifacts.output


def test_lemma_kwic_pagination_bounds_are_validated_by_cli(project_with_revisions):
    root, first, _ = project_with_revisions
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "lemma-kwic",
            str(root),
            "migrant",
            "--revisions",
            first,
            "--artifacts",
            "artifact-first",
            "--limit",
            "1001",
        ],
    )

    assert result.exit_code == 2
    assert "--limit" in result.output


def test_lemma_kwic_is_discoverable_and_documents_required_scope_inputs():
    runner = CliRunner()
    corpus_help = runner.invoke(app, ["corpus", "--help"])
    command_help = runner.invoke(
        app, ["corpus", "lemma-kwic", "--help"], terminal_width=140
    )

    assert corpus_help.exit_code == command_help.exit_code == 0
    assert "lemma-kwic" in corpus_help.output
    assert "--revisions" in command_help.output
    assert "--artifacts" in command_help.output
    assert "one for each" in command_help.output


def test_lemma_kwic_evidence_help_requires_offsets_and_has_no_text_argument():
    result = CliRunner().invoke(
        app,
        ["corpus", "lemma-kwic-evidence", "--help"],
        terminal_width=140,
    )

    assert result.exit_code == 0
    assert "--revisions" in result.output
    assert "--artifacts" in result.output
    assert "--revision-id" in result.output
    assert "--start-offset" in result.output
    assert "--end-offset" in result.output
    assert "--text" not in result.output
