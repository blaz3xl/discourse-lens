from __future__ import annotations

import csv
import io
import json
import re
from typing import cast

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import build_parsed_token_artifact


class _ParsedFrequencyBackend:
    language = "en"
    backend_id = "cli-parsed-frequency-fixture"
    backend_version = "1.0"
    model_id = "fixture-model"
    model_version = "1.0"

    def parse(self, text: str) -> list[ParsedToken]:
        result = []
        for index, match in enumerate(re.finditer(r"\w+", text)):
            surface = match.group()
            lemma = surface.casefold()
            pos = {
                "alpha": "NOUN",
                "beta": "VERB",
                "gamma": "ADJ",
                "delta": "NOUN",
                "eta": "ADV",
            }.get(lemma, "NOUN")
            relation = "ROOT" if index == 0 else ("obj" if lemma == "beta" else "dep")
            result.append(
                ParsedToken(
                    surface,
                    lemma,
                    pos,
                    None,
                    None if index == 0 else 0,
                    relation,
                    match.start(),
                    match.end(),
                )
            )
        return result


def _create_project(root):
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "Alpha alpha beta gamma")
    _, second_revision = project.import_text("second.txt", "Alpha delta beta eta")
    backend = _ParsedFrequencyBackend()
    first_artifact = build_parsed_token_artifact(project, first_revision, backend)
    second_artifact = build_parsed_token_artifact(project, second_revision, backend)
    scope_id = project.snapshot([first_revision, second_revision])
    _, later_revision = project.import_text("later.txt", "outsidesentinel alpha")
    later_artifact = build_parsed_token_artifact(project, later_revision, backend)
    project.close()
    return (
        first_revision,
        second_revision,
        scope_id,
        [first_artifact.artifact_id, second_artifact.artifact_id],
        later_revision,
        later_artifact.artifact_id,
    )


def _invoke(root, scope_id: str, artifact_ids: list[str], field: str, *options: str):
    return CliRunner().invoke(
        app,
        [
            "corpus",
            "parsed-frequency-scope",
            str(root),
            scope_id,
            "--artifacts",
            ",".join(artifact_ids),
            "--field",
            field,
            *options,
        ],
    )


def _scope_ids(project: Project) -> list[str]:
    return [
        str(row[0])
        for row in project.db.execute("SELECT id FROM scope_snapshots ORDER BY id").fetchall()
    ]


def _artifact_counts(project: Project) -> dict[str, int]:
    return {
        table: int(project.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("scope_snapshots", "artifact_registry")
    }


def test_parsed_frequency_scope_matches_core_pages_for_all_fields_without_snapshot(
    tmp_path, monkeypatch
):
    root = tmp_path / "parsed-frequency-scope"
    (
        first_revision,
        second_revision,
        scope_id,
        artifact_ids,
        later_revision,
        later_artifact_id,
    ) = _create_project(root)

    original_open = Project.open
    with original_open(root, read_only=True) as project:
        expected_pages = {
            (field, offset): dict(
                project.corpus_parsed_frequency_page(
                    scope_id,
                    artifact_ids,
                    field=field,
                    offset=offset,
                    limit=2,
                )
            )
            for field in ("lemma", "pos", "dependency_relation")
            for offset in (0, 2, 4)
        }
        expected_all = {
            field: dict(
                project.corpus_parsed_frequency_page(
                    scope_id,
                    artifact_ids,
                    field=field,
                    limit=100,
                )
            )
            for field in ("lemma", "pos", "dependency_relation")
        }
        scope_members_before = project.scope_revisions(scope_id)
        scope_ids_before = _scope_ids(project)
        counts_before = _artifact_counts(project)

    opened_read_only: list[bool] = []

    def tracked_open(path, *, read_only: bool = False):
        opened_read_only.append(read_only)
        return original_open(path, read_only=read_only)

    def unexpected_snapshot(*_args, **_kwargs):
        raise AssertionError("an existing-scope parsed-frequency read must not snapshot")

    monkeypatch.setattr(cli_module.Project, "open", staticmethod(tracked_open))
    monkeypatch.setattr(Project, "snapshot", unexpected_snapshot)
    database = root / ".discourselens" / "project.sqlite"
    database_before = database.read_bytes()

    cli_pages: dict[tuple[str, int], dict[str, object]] = {}
    for field in ("lemma", "pos", "dependency_relation"):
        for offset in (0, 2, 4):
            result = _invoke(
                root,
                scope_id,
                artifact_ids,
                field,
                "--offset",
                str(offset),
                "--limit",
                "2",
            )
            assert result.exit_code == 0, result.output
            payload = cast(dict[str, object], json.loads(result.stdout))
            expected = expected_pages[(field, offset)]
            assert payload == {
                **expected,
                "rows": [
                    {"token": row["value"], "count": row["count"]}
                    for row in cast(list[dict[str, object]], expected["rows"])
                ],
            }
            cli_pages[(field, offset)] = payload

    assert opened_read_only == [True] * 9
    assert database.read_bytes() == database_before
    for field in ("lemma", "pos", "dependency_relation"):
        pages = [cli_pages[(field, offset)] for offset in (0, 2, 4)]
        assert all(page["scope_id"] == scope_id for page in pages)
        assert all(page["revision_ids"] == [
            first_revision,
            second_revision,
        ] for page in pages)
        assert all(page["artifact_ids"] == sorted(artifact_ids) for page in pages)
        for page in pages:
            page_offset = cast(int, page["offset"])
            page_total = cast(int, page["total_values"])
            rows = cast(list[dict[str, object]], page["rows"])
            assert page["has_more"] == (page_offset + len(rows) < page_total)
        page_rows = [
            row
            for page in pages
            for row in cast(list[dict[str, object]], page["rows"])
        ]
        page_tokens = [row["token"] for row in page_rows]
        assert len(page_tokens) == len(set(page_tokens))
        assert "outsidesentinel" not in page_tokens
        assert page_rows == [
            {"token": row["value"], "count": row["count"]}
            for row in cast(list[dict[str, object]], expected_all[field]["rows"])
        ]

    with original_open(root, read_only=True) as verified:
        assert verified.scope_revisions(scope_id) == scope_members_before == [
            first_revision,
            second_revision,
        ]
        assert later_revision not in verified.scope_revisions(scope_id)
        assert later_artifact_id not in artifact_ids
        assert _scope_ids(verified) == scope_ids_before
        assert _artifact_counts(verified) == counts_before


@pytest.mark.parametrize("output_format", ["jsonl", "csv", "table"])
def test_parsed_frequency_scope_preserves_row_formats(tmp_path, output_format):
    root = tmp_path / f"parsed-frequency-scope-{output_format}"
    _, _, scope_id, artifact_ids, _, _ = _create_project(root)

    result = _invoke(root, scope_id, artifact_ids, "lemma", "--format", output_format)

    assert result.exit_code == 0, result.output
    assert "token" in result.stdout
    assert "has_more" in result.stderr
    if output_format == "jsonl":
        assert json.loads(result.stdout.splitlines()[0])["token"]
    elif output_format == "csv":
        assert next(iter(csv.DictReader(io.StringIO(result.stdout))))["token"]


@pytest.mark.parametrize("artifact_selection", ["incomplete", "out-of-scope", "unknown-scope"])
def test_parsed_frequency_scope_rejects_invalid_scope_artifacts_without_writes(
    tmp_path, artifact_selection
):
    root = tmp_path / f"parsed-frequency-scope-invalid-{artifact_selection}"
    _, _, scope_id, artifact_ids, _, later_artifact_id = _create_project(root)
    original_open = Project.open
    with original_open(root, read_only=True) as project:
        counts_before = _artifact_counts(project)
        scope_ids_before = _scope_ids(project)
        scope_members_before = project.scope_revisions(scope_id)

    requested_scope = "missing-scope" if artifact_selection == "unknown-scope" else scope_id
    requested_artifacts = (
        artifact_ids[:1]
        if artifact_selection == "incomplete"
        else artifact_ids + [later_artifact_id]
        if artifact_selection == "out-of-scope"
        else artifact_ids
    )

    result = _invoke(root, requested_scope, requested_artifacts, "lemma")

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    with original_open(root, read_only=True) as verified:
        assert _artifact_counts(verified) == counts_before
        assert _scope_ids(verified) == scope_ids_before
        assert verified.scope_revisions(scope_id) == scope_members_before


@pytest.mark.parametrize(
    ("scope_id", "artifacts", "field", "options", "message"),
    [
        (" ", ["artifact"], "lemma", [], "scope ID cannot be blank"),
        ("scope", [], "lemma", [], "must include at least one"),
        ("scope", ["artifact", "artifact"], "lemma", [], "must not contain duplicates"),
        ("scope", ["artifact"], "unsupported", [], "field must be lemma, pos"),
        ("scope", ["artifact"], "lemma", ["--offset", "-1"], "--offset"),
        ("scope", ["artifact"], "lemma", ["--limit", "0"], "--limit"),
    ],
)
def test_parsed_frequency_scope_validates_args_before_open(
    tmp_path, monkeypatch, scope_id, artifacts, field, options, message
):
    root = tmp_path / "parsed-frequency-scope-cli-invalid"
    Project.init(root).close()
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid parsed-frequency arguments must fail before opening")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = _invoke(root, scope_id, artifacts, field, *options)

    assert result.exit_code == 2
    assert message in result.output
    assert "Traceback" not in result.output
    assert opened == []
