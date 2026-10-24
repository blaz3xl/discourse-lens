import asyncio
import json

import pytest
from typer.testing import CliRunner

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.cli import app
from discourselens.errors import ProjectError, ScopeViolationError
from discourselens.mcp_server import corpus_dependency_patterns
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import build_parsed_token_artifact


class _PatternBackend:
    language = "en"
    backend_id = "fixture"
    model_id = "pattern-pages-v1"

    def parse(self, text: str) -> list[ParsedToken]:
        head, dependent = text.split()
        return [
            ParsedToken(head, head.casefold(), "NOUN", None, None, "ROOT", 0, len(head)),
            ParsedToken(
                dependent,
                dependent.casefold(),
                "NOUN",
                None,
                0,
                "obj",
                len(head) + 1,
                len(text),
            ),
        ]


def _project_with_patterns(root):
    project = Project.init(root)
    revisions: list[str] = []
    artifact_ids: list[str] = []
    outside_artifact_id = ""
    for name, text in (
        ("rivers.txt", "Rivers carry"),
        ("citizens.txt", "Citizens demand"),
        ("outside.txt", "Visitors challenge"),
    ):
        _, revision_id = project.import_text(name, text)
        artifact = build_parsed_token_artifact(project, revision_id, _PatternBackend())
        if name == "outside.txt":
            outside_artifact_id = artifact.artifact_id
        else:
            revisions.append(revision_id)
            artifact_ids.append(artifact.artifact_id)
    return project, revisions, artifact_ids, outside_artifact_id


def _assert_stitched_pages(full, first, second) -> None:
    assert first["offset"] == 0
    assert first["limit"] == 1
    assert first["total_rows"] == full["total_rows"] == 2
    assert first["has_more"] is first["truncated"] is True
    assert second["offset"] == 1
    assert second["total_rows"] == full["total_rows"]
    assert second["has_more"] is second["truncated"] is False
    assert first["rows"] + second["rows"] == full["rows"]
    assert "visitors" not in json.dumps(first["rows"] + second["rows"])


def test_project_dependency_pattern_pages_stitch_and_reject_out_of_scope_artifacts(tmp_path):
    project, revisions, artifact_ids, outside_artifact_id = _project_with_patterns(
        tmp_path / "project-pages"
    )
    scope_id = project.snapshot(revisions)

    full = project.corpus_dependency_patterns(scope_id, artifact_ids)
    first = project.corpus_dependency_patterns(scope_id, artifact_ids, offset=0, limit=1)
    second = project.corpus_dependency_patterns(scope_id, artifact_ids, offset=1, limit=1)

    _assert_stitched_pages(full, first, second)
    assert full["candidate_count_lower_bound"] == 2
    assert full["truncated"] is False
    assert set(full["artifact_ids"]) == set(artifact_ids)
    try:
        project.corpus_dependency_patterns(
            scope_id, [*artifact_ids, outside_artifact_id], offset=0, limit=1
        )
    except ScopeViolationError as exc:
        assert "outside the explicit scope" in str(exc)
    else:
        raise AssertionError("out-of-scope parsed artifacts must be rejected")
    project.close()


def test_project_dependency_pattern_offset_rejects_values_outside_signed_64_bit_range(
    tmp_path,
):
    project, revisions, artifact_ids, _ = _project_with_patterns(tmp_path / "core-offset")
    scope_id = project.snapshot(revisions)

    with pytest.raises(ScopeViolationError, match="signed 64-bit range"):
        project.corpus_dependency_patterns(
            scope_id,
            artifact_ids,
            offset=0x7FFF_FFFF_FFFF_FFFF + 1,
        )
    project.close()


def test_cli_dependency_pattern_pages_stitch_within_explicit_revision_scope(tmp_path):
    root = tmp_path / "cli-pages"
    project, revisions, artifact_ids, _ = _project_with_patterns(root)
    scope_id = project.snapshot(revisions)
    full = project.corpus_dependency_patterns(scope_id, artifact_ids)
    project.close()

    base = [
        "corpus",
        "dependency-patterns",
        str(root),
        "--revisions",
        ",".join(revisions),
        "--artifacts",
        ",".join(artifact_ids),
    ]
    runner = CliRunner()
    first_result = runner.invoke(app, [*base, "--offset", "0", "--limit", "1"])
    second_result = runner.invoke(app, [*base, "--offset", "1", "--limit", "1"])

    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code == 0, second_result.output
    first = json.loads(first_result.stdout)
    second = json.loads(second_result.stdout)
    _assert_stitched_pages(full, first, second)
    assert set(first["artifact_ids"]) == set(artifact_ids)


def test_cli_dependency_pattern_offset_rejects_values_outside_signed_64_bit_range(tmp_path):
    root = tmp_path / "cli-offset"
    project, revisions, artifact_ids, _ = _project_with_patterns(root)
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "dependency-patterns",
            str(root),
            "--revisions",
            ",".join(revisions),
            "--artifacts",
            ",".join(artifact_ids),
            "--offset",
            str(0x7FFF_FFFF_FFFF_FFFF + 1),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "signed 64-bit range" in result.output


def test_mcp_dependency_pattern_pages_stitch_within_explicit_revision_scope(
    tmp_path, monkeypatch
):
    root = tmp_path / "mcp-pages"
    project, revisions, artifact_ids, _ = _project_with_patterns(root)
    scope_id = project.snapshot(revisions)
    full = project.corpus_dependency_patterns(scope_id, artifact_ids)
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    first = corpus_dependency_patterns(revisions, artifact_ids, offset=0, limit=1)
    second = corpus_dependency_patterns(revisions, artifact_ids, offset=1, limit=1)

    _assert_stitched_pages(full, first, second)
    assert set(first["artifact_ids"]) == set(artifact_ids)


def test_mcp_dependency_pattern_offset_schema_and_boundary_use_signed_64_bit_range(
    monkeypatch,
):
    tools = asyncio.run(mcp_module.server.list_tools())
    tool = next(item for item in tools if item.name == "corpus_dependency_patterns")
    schema = tool.model_dump(by_alias=True)["inputSchema"]
    assert schema["properties"]["offset"]["maximum"] == 0x7FFF_FFFF_FFFF_FFFF

    def unexpected_open():
        raise AssertionError("invalid paging must be rejected before opening the project")

    monkeypatch.setattr(mcp_module, "_open_project", unexpected_open)
    with pytest.raises(ProjectError, match="signed 64-bit range"):
        corpus_dependency_patterns(
            ["revision-id"],
            ["artifact-id"],
            offset=0x7FFF_FFFF_FFFF_FFFF + 1,
        )


def test_mcp_dependency_pattern_page_rejects_oversized_lemma_without_truncating_counts(
    monkeypatch,
):
    class OversizedProject:
        def snapshot(self, revision_ids: list[str]) -> str:
            assert revision_ids == ["revision-id"]
            return "scope-id"

        def close(self) -> None:
            pass

    def oversized_patterns(project, scope_id, artifact_ids, **kwargs):
        assert isinstance(project, OversizedProject)
        assert scope_id == "scope-id"
        assert artifact_ids == ["artifact-id"]
        return {
            "scope_id": scope_id,
            "artifact_ids": artifact_ids,
            "calculation_version": "parsed-dependency-patterns-v2",
            "filters": {"relation": None, "head_lemma": None, "dependent_lemma": None},
            "offset": kwargs["offset"],
            "limit": kwargs["limit"],
            "total_rows": 1,
            "has_more": False,
            "candidate_count_lower_bound": 1,
            "rows": [
                {
                    "head_lemma": "x" * mcp_module.MAX_MCP_DEPENDENCY_PATTERN_PAGE_BYTES,
                    "relation": "obj",
                    "dependent_lemma": "policy",
                    "frequency": 7,
                    "document_frequency": 1,
                }
            ],
            "truncated": False,
        }

    monkeypatch.setattr(mcp_module, "_open_project", OversizedProject)
    monkeypatch.setattr(mcp_module, "dependency_patterns", oversized_patterns)

    with pytest.raises(ProjectError, match="64 KiB"):
        corpus_dependency_patterns(["revision-id"], ["artifact-id"], limit=1)
