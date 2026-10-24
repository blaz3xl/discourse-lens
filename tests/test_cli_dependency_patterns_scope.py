from __future__ import annotations

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


class _DependencyBackend:
    language = "en"
    backend_id = "cli-dependency-pattern-fixture"
    backend_version = "1.0"
    model_id = "fixture-model"
    model_version = "1.0"

    def parse(self, text: str) -> list[ParsedToken]:
        tokens = []
        for index, match in enumerate(re.finditer(r"\w+", text)):
            surface = match.group()
            lemma = {
                "leader": "leader",
                "supports": "support",
                "migrant": "migrant",
                "migrants": "migrant",
            }.get(surface.casefold(), surface.casefold())
            if index == 0:
                dependency_head = None
                relation = "ROOT"
            elif lemma == "support":
                dependency_head = 0
                relation = "nsubj"
            elif lemma == "migrant":
                dependency_head = 0
                relation = "obj"
            else:
                dependency_head = 1
                relation = "obj"
            tokens.append(
                ParsedToken(
                    surface,
                    lemma,
                    "NOUN",
                    None,
                    dependency_head,
                    relation,
                    match.start(),
                    match.end(),
                )
            )
        return tokens


def _create_project(root):
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "Leader supports migrant speaker")
    _, second_revision = project.import_text("second.txt", "Leader supports migrant policy")
    backend = _DependencyBackend()
    first_artifact = build_parsed_token_artifact(project, first_revision, backend)
    second_artifact = build_parsed_token_artifact(project, second_revision, backend)
    scope_id = project.snapshot([first_revision, second_revision])
    _, later_revision = project.import_text("later.txt", "Outsider supports migrant")
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


def _invoke_scope(root, scope_id: str, artifact_ids: list[str], *options: str):
    return CliRunner().invoke(
        app,
        [
            "corpus",
            "dependency-patterns-scope",
            str(root),
            scope_id,
            "--artifacts",
            ",".join(artifact_ids),
            *options,
        ],
    )


def _scope_ids(project: Project) -> list[str]:
    return [
        str(row[0])
        for row in project.db.execute("SELECT id FROM scope_snapshots ORDER BY id").fetchall()
    ]


def _row_counts(project: Project) -> dict[str, int]:
    return {
        table: int(project.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("scope_snapshots", "artifact_registry")
    }


def test_dependency_patterns_scope_matches_core_pages_and_preserves_frozen_membership(
    tmp_path, monkeypatch
):
    root = tmp_path / "dependency-patterns-scope"
    (
        first_revision,
        second_revision,
        scope_id,
        artifact_ids,
        later_revision,
        later_artifact_id,
    ) = _create_project(root)
    page_specs = [
        ({}, 0, 2),
        ({}, 2, 2),
        ({}, 4, 2),
        ({"relation": "obj", "head_lemma": " LEADER ", "dependent_lemma": "migrant"}, 0, 1),
    ]

    original_open = Project.open
    with original_open(root, read_only=True) as project:
        expected_pages = [
            dict(
                project.corpus_dependency_patterns(
                    scope_id,
                    artifact_ids,
                    relation=filters.get("relation"),
                    head_lemma=filters.get("head_lemma"),
                    dependent_lemma=filters.get("dependent_lemma"),
                    offset=offset,
                    limit=limit,
                )
            )
            for filters, offset, limit in page_specs
        ]
        scope_members_before = project.scope_revisions(scope_id)
        scope_ids_before = _scope_ids(project)
        counts_before = _row_counts(project)

    opened_read_only: list[bool] = []

    def tracked_open(path, *, read_only: bool = False):
        opened_read_only.append(read_only)
        return original_open(path, read_only=read_only)

    def unexpected_snapshot(*_args, **_kwargs):
        raise AssertionError("an existing-scope dependency-pattern read must not snapshot")

    monkeypatch.setattr(cli_module.Project, "open", staticmethod(tracked_open))
    monkeypatch.setattr(Project, "snapshot", unexpected_snapshot)
    database = root / ".discourselens" / "project.sqlite"
    database_before = database.read_bytes()

    results = []
    for (filters, offset, limit), expected in zip(page_specs, expected_pages, strict=True):
        options: list[str] = ["--offset", str(offset), "--limit", str(limit)]
        for name, option in (
            ("relation", "--relation"),
            ("head_lemma", "--head-lemma"),
            ("dependent_lemma", "--dependent-lemma"),
        ):
            if name in filters:
                options.extend((option, filters[name]))
        result = _invoke_scope(root, scope_id, artifact_ids, *options)
        assert result.exit_code == 0, result.output
        payload = cast(dict[str, object], json.loads(result.stdout))
        assert payload == expected
        results.append(payload)

    assert opened_read_only == [True] * len(page_specs)
    assert database.read_bytes() == database_before
    for payload in results:
        assert payload["scope_id"] == scope_id
        assert payload["artifact_ids"] == sorted(artifact_ids)
        rows = cast(list[dict[str, object]], payload["rows"])
        assert all("head_lemma" in row and "relation" in row for row in rows)
        assert all("outsider" not in str(row.values()) for row in rows)

    assert results[0]["total_rows"] == results[1]["total_rows"] == results[2]["total_rows"] == 4
    assert [page["has_more"] for page in results[:3]] == [True, False, False]
    assert results[3]["filters"] == {
        "relation": "obj",
        "head_lemma": "leader",
        "dependent_lemma": "migrant",
    }
    assert results[3]["rows"] == [
        {
            "head_lemma": "leader",
            "relation": "obj",
            "dependent_lemma": "migrant",
            "frequency": 2,
            "document_frequency": 2,
        }
    ]
    assert later_artifact_id not in cast(list[str], results[0]["artifact_ids"])

    with original_open(root, read_only=True) as verified:
        assert verified.scope_revisions(scope_id) == scope_members_before == [
            first_revision,
            second_revision,
        ]
        assert later_revision not in verified.scope_revisions(scope_id)
        assert _scope_ids(verified) == scope_ids_before
        assert _row_counts(verified) == counts_before


@pytest.mark.parametrize("selection", ["incomplete", "out-of-scope", "unknown-scope"])
def test_dependency_patterns_scope_rejects_invalid_scope_artifacts_without_writes(
    tmp_path, selection
):
    root = tmp_path / f"dependency-patterns-scope-invalid-{selection}"
    _, _, scope_id, artifact_ids, _, later_artifact_id = _create_project(root)
    original_open = Project.open
    with original_open(root, read_only=True) as project:
        counts_before = _row_counts(project)
        scope_ids_before = _scope_ids(project)
        scope_members_before = project.scope_revisions(scope_id)

    requested_scope = "missing-scope" if selection == "unknown-scope" else scope_id
    requested_artifacts = (
        artifact_ids[:1]
        if selection == "incomplete"
        else artifact_ids + [later_artifact_id]
        if selection == "out-of-scope"
        else artifact_ids
    )

    result = _invoke_scope(root, requested_scope, requested_artifacts)

    assert result.exit_code == 2
    assert "Traceback" not in result.output
    with original_open(root, read_only=True) as verified:
        assert _row_counts(verified) == counts_before
        assert _scope_ids(verified) == scope_ids_before
        assert verified.scope_revisions(scope_id) == scope_members_before


@pytest.mark.parametrize(
    ("scope_id", "artifacts", "options", "message"),
    [
        (" ", ["artifact"], [], "scope ID must be trimmed"),
        (" padded-scope ", ["artifact"], [], "scope ID must be trimmed"),
        ("s" * 129, ["artifact"], [], "scope ID must be trimmed"),
        ("scope", [], [], "1..10000"),
        ("scope", ["x" * 129], [], "at most 128 characters"),
        ("scope", ["artifact", "artifact"], [], "must not contain duplicates"),
        ("scope", [f"artifact-{index}" for index in range(10_001)], [], "1..10000"),
        ("scope", ["artifact"], ["--relation", " "], "relation must be nonblank"),
        ("scope", ["artifact"], ["--relation", "x" * 257], "relation must be nonblank"),
        ("scope", ["artifact"], ["--head-lemma", " "], "head lemma must be nonblank"),
        ("scope", ["artifact"], ["--head-lemma", "x" * 257], "head lemma must be nonblank"),
        (
            "scope",
            ["artifact"],
            ["--dependent-lemma", " "],
            "dependent lemma must be nonblank",
        ),
        (
            "scope",
            ["artifact"],
            ["--dependent-lemma", "x" * 257],
            "dependent lemma must be nonblank",
        ),
        ("scope", ["artifact"], ["--offset", "-1"], "--offset"),
        ("scope", ["artifact"], ["--limit", "0"], "--limit"),
    ],
)
def test_dependency_patterns_scope_validates_arguments_before_open(
    tmp_path, monkeypatch, scope_id, artifacts, options, message
):
    root = tmp_path / "dependency-patterns-scope-cli-invalid"
    Project.init(root).close()
    opened: list[object] = []

    def unexpected_open(*args, **kwargs):
        opened.append((args, kwargs))
        raise AssertionError("invalid dependency-pattern arguments must fail before opening")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = _invoke_scope(root, scope_id, artifacts, *options)

    assert result.exit_code == 2
    assert message in result.output
    assert "Traceback" not in result.output
    assert opened == []


def test_legacy_dependency_patterns_revision_list_command_is_unchanged(tmp_path):
    root = tmp_path / "dependency-patterns-legacy"
    first_revision, second_revision, _, artifact_ids, _, _ = _create_project(root)

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "dependency-patterns",
            str(root),
            "--revisions",
            f"{first_revision},{second_revision}",
            "--artifacts",
            ",".join(artifact_ids),
            "--limit",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["scope_id"]
    assert payload["limit"] == 2
    assert payload["total_rows"] == 4
    assert payload["rows"][0] == {
        "head_lemma": "leader",
        "relation": "nsubj",
        "dependent_lemma": "support",
        "frequency": 2,
        "document_frequency": 2,
    }
