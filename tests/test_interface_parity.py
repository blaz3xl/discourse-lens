import asyncio
import json
import sqlite3
from typing import cast

import pytest
from typer.testing import CliRunner

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.cli import app
from discourselens.errors import ProjectError, SkillManifestError
from discourselens.mcp_server import (
    corpus_frequency as mcp_corpus_frequency,
)
from discourselens.mcp_server import (
    corpus_frequency_evidence as mcp_corpus_frequency_evidence,
)
from discourselens.mcp_server import (
    run_controlled_cda_review as mcp_run_controlled_cda_review,
)
from discourselens.mcp_server import (
    run_counterevidence_lexical as mcp_run_counterevidence_lexical,
)
from discourselens.mcp_server import (
    run_evidence_extraction as mcp_run_evidence_extraction,
)
from discourselens.mcp_server import (
    run_textual_descriptors as mcp_run_textual_descriptors,
)
from discourselens.mcp_server import (
    search_exact as mcp_search_exact,
)
from discourselens.mcp_server import (
    search_lexical as mcp_search_lexical,
)
from discourselens.types import EvidenceRole


def _comparable_page(page):
    """Ignore adapter envelope IDs; keep paging, rows and counts comparable."""
    comparable = {
        key: value for key, value in page.items() if key not in {"scope_id", "revision_id"}
    }
    if "rows" in comparable:
        comparable["rows"] = [dict(row) for row in comparable["rows"]]
    return comparable


def test_frequency_page_is_structurally_equivalent_across_sdk_cli_and_mcp(tmp_path, monkeypatch):
    root = tmp_path / "research"
    project = Project.init(root)
    _, in_scope_revision = project.import_text("a.txt", "Apples are green. Apples grow.")
    project.import_text("b.txt", "Elephants live in Africa.")
    scope_id = project.snapshot([in_scope_revision])
    sdk_page = project.corpus_frequency_page(scope_id, offset=1, limit=2)
    scope_revisions = project.scope_revisions(scope_id)
    project.close()

    cli_result = CliRunner().invoke(
        app,
        [
            "corpus",
            "frequency",
            str(root),
            in_scope_revision,
            "--format",
            "json",
            "--offset",
            "1",
            "--limit",
            "2",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_page = json.loads(cli_result.output)

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    mcp_page = mcp_corpus_frequency(in_scope_revision, offset=1, limit=2)

    assert _comparable_page(cli_page) == _comparable_page(sdk_page)
    assert _comparable_page(mcp_page) == _comparable_page(sdk_page)
    assert scope_revisions == [in_scope_revision]
    assert all(row["token"] != "elephants" for row in sdk_page["rows"])
    assert sdk_page["offset"] == 1
    assert sdk_page["limit"] == 2


@pytest.mark.parametrize(
    ("option", "value", "mcp_arguments", "error_message"),
    [
        ("--size", "1", {"size": 1}, "lexical bundle size"),
        ("--min-frequency", "0", {"min_frequency": 0}, "minimum bundle frequency"),
        (
            "--min-documents",
            "0",
            {"min_document_frequency": 0},
            "minimum document frequency",
        ),
        ("--limit", "1001", {"limit": 1_001}, "lexical bundle limit"),
    ],
)
def test_invalid_lexical_bundle_options_do_not_create_scope_snapshots(
    tmp_path, monkeypatch, option, value, mcp_arguments, error_message
):
    root = tmp_path / "invalid-lexical-bundles"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "the public must act")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    with pytest.raises(ProjectError, match=error_message):
        mcp_module.corpus_lexical_bundles([revision], **mcp_arguments)

    cli_result = CliRunner().invoke(
        app,
        [
            "corpus",
            "lexical-bundles",
            str(root),
            revision,
            option,
            value,
        ],
    )
    assert cli_result.exit_code == 2
    with Project.open(root) as unchanged:
        snapshot_count = unchanged.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0]
    assert snapshot_count == 0


def test_mcp_corpus_keyness_matches_core_and_keeps_two_frozen_scopes(tmp_path, monkeypatch):
    root = tmp_path / "keyness"
    project = Project.init(root)
    _, left_first = project.import_text("left-first.txt", "targetonly targetonly shared")
    _, left_second = project.import_text("left-second.txt", "targetonly targetsecondary")
    _, right_revision = project.import_text("right.txt", "referenceonly referenceonly shared")
    _, outside_revision = project.import_text(
        "outside.txt", "outsidesentinel outsidesentinel outsidesentinel"
    )
    left_scope = project.snapshot([left_first, left_second])
    right_scope = project.snapshot([right_revision])
    sdk_page = project.corpus_keyness_page(left_scope, right_scope, offset=1, limit=2)
    project.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    mcp_page = mcp_module.corpus_keyness(left_scope, right_scope, offset=1, limit=2)

    assert mcp_page["metric"] == sdk_page["metric"] == "keyness"
    assert mcp_page["offset"] == sdk_page["offset"] == 1
    assert mcp_page["limit"] == sdk_page["limit"] == 2
    assert mcp_page["total_rows"] == sdk_page["total_rows"]
    assert mcp_page["has_more"] == sdk_page["has_more"]
    assert mcp_page["rows"] == sdk_page["rows"]
    assert mcp_page["left_scope_id"] == left_scope
    assert mcp_page["right_scope_id"] == right_scope
    populations = cast(dict[str, dict[str, object]], mcp_page["populations"])
    coverage = cast(dict[str, object], mcp_page["coverage"])
    statistic = cast(dict[str, object], mcp_page["statistic"])
    rows = cast(list[dict[str, object]], mcp_page["rows"])
    left_revision_ids = cast(list[str], populations["left"]["revision_ids_preview"])
    right_revision_ids = cast(list[str], populations["right"]["revision_ids_preview"])
    assert left_revision_ids == [left_first, left_second]
    assert right_revision_ids == [right_revision]
    assert populations["left"]["revision_count"] == 2
    assert populations["right"]["revision_count"] == 1
    assert populations["left"]["revision_ids_truncated"] is False
    assert populations["right"]["revision_ids_truncated"] is False
    assert outside_revision not in left_revision_ids
    assert outside_revision not in right_revision_ids
    assert populations["left"]["surface_token_count"] == 5
    assert populations["right"]["surface_token_count"] == 3
    assert mcp_page["total_rows"] == 4
    assert coverage["mode"] == "EXHAUSTIVE"
    assert coverage["inspection_counts"] == {"left": 5, "right": 3}
    assert statistic["name"] == "signed_log_likelihood"
    assert statistic["calculation_version"] == "surface-comparison-v2"
    assert all(row["token"] != "outsidesentinel" for row in rows)
    assert "corpus_keyness" in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    assert database.read_bytes() == before


def test_mcp_compare_frequency_matches_core_with_bounded_frozen_scope_pages(tmp_path, monkeypatch):
    root = tmp_path / "compare-frequency"
    project = Project.init(root)
    _, left_first = project.import_text("left-first.txt", "leftonly leftonly common alpha beta")
    _, left_second = project.import_text("left-second.txt", "leftonly lower")
    _, right_revision = project.import_text("right.txt", "rightonly rightonly common gamma delta")
    _, outside_revision = project.import_text(
        "outside.txt", "outsideunique outsideunique outsideunique"
    )
    left_scope = project.snapshot([left_first, left_second])
    right_scope = project.snapshot([right_revision])
    sdk_full_page = project.corpus_compare_frequency_page(
        left_scope, right_scope, offset=0, limit=100
    )
    sdk_pages = [
        project.corpus_compare_frequency_page(left_scope, right_scope, offset=offset, limit=2)
        for offset in (0, 2, 4, 6)
    ]
    project.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", True)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    mcp_pages = [
        mcp_module.corpus_compare_frequency(left_scope, right_scope, offset=offset, limit=2)
        for offset in (0, 2, 4, 6)
    ]

    assert "corpus_compare_frequency" in mcp_module.READ_ONLY_MCP_TOOL_NAMES
    assert sdk_full_page["metric"] == "frequency"
    assert sdk_full_page["total_rows"] == 8
    assert [page["offset"] for page in mcp_pages] == [0, 2, 4, 6]
    assert [page["has_more"] for page in mcp_pages] == [True, True, True, False]
    assert all(
        len(cast(list[object], page["rows"])) == 2 and page["limit"] == 2 for page in mcp_pages
    )
    for sdk_page, mcp_page in zip(sdk_pages, mcp_pages, strict=True):
        assert mcp_page["metric"] == sdk_page["metric"] == "frequency"
        assert mcp_page["offset"] == sdk_page["offset"]
        assert mcp_page["limit"] == sdk_page["limit"] == 2
        assert mcp_page["total_rows"] == sdk_page["total_rows"] == 8
        assert mcp_page["has_more"] == sdk_page["has_more"]
        assert mcp_page["rows"] == sdk_page["rows"]
        assert mcp_page["left_scope_id"] == left_scope
        assert mcp_page["right_scope_id"] == right_scope

    first = mcp_pages[0]
    populations = cast(dict[str, dict[str, object]], first["populations"])
    coverage = cast(dict[str, object], first["coverage"])
    statistic = cast(dict[str, object], first["statistic"])
    assert populations["left"] == {
        "scope_id": left_scope,
        "revision_ids_preview": [left_first, left_second],
        "revision_ids_truncated": False,
        "revision_count": 2,
        "surface_token_count": 7,
    }
    assert populations["right"] == {
        "scope_id": right_scope,
        "revision_ids_preview": [right_revision],
        "revision_ids_truncated": False,
        "revision_count": 1,
        "surface_token_count": 5,
    }
    assert outside_revision not in cast(list[str], populations["left"]["revision_ids_preview"])
    assert outside_revision not in cast(list[str], populations["right"]["revision_ids_preview"])
    assert coverage["mode"] == "EXHAUSTIVE"
    assert coverage["inspection_counts"] == {"left": 7, "right": 5}
    assert statistic["name"] == "rate_difference"
    assert statistic["calculation_version"] == "surface-comparison-v2"
    assert [row for page in mcp_pages for row in cast(list[dict[str, object]], page["rows"])] == (
        sdk_full_page["rows"]
    )
    assert all(
        row["token"] != "outsideunique"
        for page in mcp_pages
        for row in cast(list[dict[str, object]], page["rows"])
    )

    with pytest.raises(ProjectError, match="unknown scope"):
        mcp_module.corpus_compare_frequency(left_first, right_scope, limit=2)
    with pytest.raises(ProjectError, match="unknown scope"):
        mcp_module.corpus_compare_frequency(left_scope, "unknown-scope", limit=2)
    assert database.read_bytes() == before
    with Project.open(root, read_only=True) as unchanged:
        assert unchanged.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == 2
        assert unchanged.db.execute("SELECT COUNT(*) FROM comparison_runs").fetchone()[0] == 0


def test_exact_search_page_preserves_scope_and_paragraph_provenance_across_adapters(
    tmp_path, monkeypatch
):
    root = tmp_path / "research"
    project = Project.init(root)
    _, revision = project.import_text(
        "speech.txt",
        "Migration appears in paragraph one.\n\n"
        "Migration appears in paragraph two.\n\n"
        "Migration appears in paragraph three.",
    )
    project.import_text("outside.txt", "Migration is outside the selected revision.")
    scope_id = project.snapshot([revision])
    sdk_page = project.search_exact_page("Migration", scope_id, offset=1, limit=1)
    project.close()

    cli_result = CliRunner().invoke(
        app,
        [
            "search",
            "exact",
            str(root),
            revision,
            "Migration",
            "--format",
            "json",
            "--offset",
            "1",
            "--limit",
            "1",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_page = json.loads(cli_result.output)

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    mcp_page = mcp_search_exact(revision, "Migration", offset=1, limit=1)

    assert _comparable_page(cli_page) == _comparable_page(sdk_page)
    assert _comparable_page(mcp_page) == _comparable_page(sdk_page)
    assert set(sdk_page) == {"offset", "limit", "total_count", "has_more", "rows"}
    adapter_envelope = {
        "scope_id",
        "revision_id",
        "offset",
        "limit",
        "total_count",
        "has_more",
        "rows",
    }
    assert set(cli_page) == adapter_envelope
    assert set(mcp_page) == adapter_envelope
    assert cli_page["revision_id"] == mcp_page["revision_id"] == revision
    reopened = Project.open(root)
    try:
        assert reopened.scope_revisions(scope_id) == [revision]
        assert reopened.scope_revisions(str(cli_page["scope_id"])) == [revision]
        assert reopened.scope_revisions(str(mcp_page["scope_id"])) == [revision]
    finally:
        reopened.close()

    assert sdk_page["total_count"] == 3
    assert sdk_page["offset"] == 1 and sdk_page["limit"] == 1
    assert sdk_page["has_more"] is True
    sdk_rows = cast(list[sqlite3.Row], sdk_page["rows"])
    assert len(sdk_rows) == 1
    row = dict(sdk_rows[0])
    assert row == dict(cast(list[dict[str, object]], cli_page["rows"])[0])
    assert row == dict(cast(list[dict[str, object]], mcp_page["rows"])[0])
    assert row["revision_id"] == revision
    assert row["text"] == "Migration appears in paragraph two."
    reopened = Project.open(root)
    try:
        paragraph = reopened.paragraph(str(row["paragraph_id"]))
        assert paragraph["revision_id"] == revision
        assert paragraph["text"] == row["text"]
    finally:
        reopened.close()


def test_lexical_search_page_matches_core_cli_mcp_and_is_scope_bounded(tmp_path, monkeypatch):
    root = tmp_path / "lexical-search"
    project = Project.init(root)
    _, first_revision = project.import_text(
        "first.txt", "Migration appears in one.\n\nMigration appears in two."
    )
    _, second_revision = project.import_text("second.txt", "Migration appears in three.")
    _, outside_revision = project.import_text("outside.txt", "Migration is out of scope.")
    scope_id = project.snapshot([first_revision, second_revision])

    def sdk_page(offset: int, limit: int) -> dict[str, object]:
        return project.search_lexical_page("migration", scope_id, offset=offset, limit=limit)

    sdk_first = sdk_page(0, 2)
    sdk_second = sdk_page(2, 2)
    project.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    cli_first_result = CliRunner().invoke(
        app,
        ["search", "lexical", str(root), scope_id, "migration", "--offset", "0", "--limit", "2"],
    )
    cli_second_result = CliRunner().invoke(
        app,
        ["search", "lexical", str(root), scope_id, "migration", "--offset", "2", "--limit", "2"],
    )
    assert cli_first_result.exit_code == 0, cli_first_result.output
    assert cli_second_result.exit_code == 0, cli_second_result.output
    cli_first = json.loads(cli_first_result.output)
    cli_second = json.loads(cli_second_result.output)

    mcp_first = mcp_search_lexical(scope_id, "migration", offset=0, limit=2)
    mcp_second = mcp_search_lexical(scope_id, "migration", offset=2, limit=2)
    assert cli_first == mcp_first == sdk_first
    assert cli_second == mcp_second == sdk_second
    assert sdk_first["has_more"] is True
    assert sdk_second["has_more"] is False
    assert len(cast(list[dict[str, object]], sdk_first["rows"])) == 2
    assert len(cast(list[dict[str, object]], sdk_second["rows"])) == 1
    assert outside_revision not in {
        row["revision_id"]
        for page in (sdk_first, sdk_second)
        for row in cast(list[dict[str, object]], page["rows"])
    }

    tool = next(
        item
        for item in asyncio.run(mcp_module.server.list_tools())
        if item.name == "search_lexical"
    )
    output_schema = tool.model_dump(by_alias=True).get("outputSchema")
    assert output_schema is not None
    assert output_schema["type"] == "object"
    assert {
        "scope_id",
        "offset",
        "limit",
        "has_more",
        "candidate_window_full",
        "rows",
    } <= set(output_schema["properties"])

    mcp_call = asyncio.run(
        mcp_module.server.call_tool(
            "search_lexical",
            {"scope_id": scope_id, "query": "migration", "offset": 0, "limit": 2},
        )
    )
    assert mcp_call.structured_content == sdk_first


def test_lexical_search_last_offset_at_candidate_cap_is_reachable_across_interfaces(
    tmp_path, monkeypatch
):
    root = tmp_path / "lexical-search-cap"
    project = Project.init(root)
    text = "\n\n".join(f"Migration result {index:04d}." for index in range(5_000))
    _, revision = project.import_text("many-matches.txt", text)
    scope_id = project.snapshot([revision])
    sdk_first = project.search_lexical_page("migration", scope_id, limit=1)
    sdk_last = project.search_lexical_page("migration", scope_id, offset=4_999, limit=1)
    project.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    cli_result = CliRunner().invoke(
        app,
        ["search", "lexical", str(root), scope_id, "migration", "--offset", "4999", "--limit", "1"],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_last = json.loads(cli_result.output)
    mcp_last = mcp_search_lexical(scope_id, "migration", offset=4_999, limit=1)

    assert sdk_first["candidate_window_full"] is True
    assert sdk_first["has_more"] is True
    assert sdk_last == cli_last == mcp_last
    assert sdk_last["offset"] == 4_999
    assert sdk_last["has_more"] is False
    assert sdk_last["candidate_window_full"] is True
    assert len(cast(list[dict[str, object]], sdk_last["rows"])) == 1


@pytest.mark.parametrize(("option", "value"), [("--offset", "5000"), ("--limit", "101")])
def test_cli_lexical_search_rejects_out_of_range_page_bounds(option, value):
    result = CliRunner().invoke(
        app,
        ["search", "lexical", "missing-project", "scope-id", "query", option, value],
    )
    assert result.exit_code == 2
    assert "Invalid value" in result.stderr
    assert "Traceback" not in result.stderr


def test_citable_frequency_workflow_is_equivalent_across_sdk_cli_and_mcp(tmp_path, monkeypatch):
    root = tmp_path / "research"
    project = Project.init(root)
    _, revision = project.import_text("in-scope.txt", "Apples are green. Apples grow.")
    project.import_text("outside.txt", "Apples appear outside the selected scope.")
    scope_id = project.snapshot([revision])
    sdk_result = project.corpus_frequency_evidence(scope_id, "Apples")
    project.close()

    cli_result = CliRunner().invoke(
        app,
        ["corpus", "frequency-evidence", str(root), revision, "Apples"],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_evidence = json.loads(cli_result.output)

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    mcp_result = mcp_corpus_frequency_evidence(revision, "Apples")

    for result in (sdk_result, cli_evidence, mcp_result):
        assert {key: result[key] for key in ("term", "count", "population_count")} == {
            "term": "apples",
            "count": 2,
            "population_count": 5,
        }
        assert result["evidence_id"] and result["coverage_id"]

    reopened = Project.open(root)
    try:
        for result in (sdk_result, cli_evidence, mcp_result):
            coverage = reopened.coverage_descriptor(str(result["coverage_id"]))
            assert coverage.coverage_mode.value == "EXHAUSTIVE"
            assert coverage.scope_snapshot_id == result["scope_id"]
            assert coverage.population_count == coverage.inspection_count == 5
            assert coverage.termination_state is not None
            assert coverage.termination_state.value == "COMPLETED"
            evidence = reopened.evidence_content(str(result["evidence_id"]))
            assert evidence["value"] == 2
            assert evidence["scope_snapshot_id"] == result["scope_id"]
            assert reopened.render_citation(str(result["evidence_id"])) == (
                "[corpus statistic · surface frequency for 'apples']"
            )
    finally:
        reopened.close()


def test_evidence_extraction_pipeline_is_equivalent_across_sdk_cli_and_mcp(tmp_path, monkeypatch):
    root = tmp_path / "research"
    project = Project.init(root)
    _, revision = project.import_text(
        "in-scope.txt",
        "Officials said the migration policy must change. The park has ten trees.",
    )
    project.import_text("outside.txt", "Officials said the migration policy must change.")
    sdk_result = project.extract_evidence([revision], languages=("en",), max_examples=12)
    project.close()

    cli_result = CliRunner().invoke(
        app,
        [
            "analyze",
            "evidence-extraction",
            str(root),
            revision,
            "--languages",
            "en",
            "--max-examples",
            "12",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_extraction = json.loads(cli_result.output)

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    mcp_result = mcp_run_evidence_extraction([revision], ["en"], max_examples=12)
    results = (sdk_result, cli_extraction, mcp_result)

    for result in results:
        assert {
            key: result[key] for key in ("population_count", "candidate_count", "selected_count")
        } == {"population_count": 2, "candidate_count": 1, "selected_count": 1}
        assert result["finding_ids"] and result["evidence_ids"]

    reopened = Project.open(root)
    try:
        normalized_findings = []
        for result in results:
            run_metadata = reopened.analysis_run_metadata(str(result["analysis_run_id"]))
            assert run_metadata["revision_ids"] == [revision]
            analysis_run = cast(dict[str, object], run_metadata["analysis_run"])
            assert analysis_run["status"] == "COMPLETED"
            coverage = reopened.coverage_descriptor(str(result["coverage_id"]))
            assert coverage.scope_snapshot_id == analysis_run["scope_id"]
            assert coverage.population_count == coverage.inspection_count == 2

            finding_ids = cast(list[str], result["finding_ids"])
            finding = reopened.get_finding(finding_ids[0])
            links = reopened.finding_evidence(str(finding["id"]))
            assert len(links) == 1
            link = links[0]
            normalized_findings.append(
                {
                    "phenomenon": finding["phenomenon"],
                    "observation": finding["observation"],
                    "evidence_text": reopened.span_text(str(link["span_id"])),
                    "evidence_role": link["role"],
                }
            )
        assert normalized_findings[0] == normalized_findings[1] == normalized_findings[2]
        assert normalized_findings[0]["evidence_text"] == (
            "Officials said the migration policy must change."
        )
    finally:
        reopened.close()


def test_textual_descriptors_are_equivalent_across_sdk_cli_json_and_mcp_call(tmp_path, monkeypatch):
    root = tmp_path / "research"
    selected_text = "Officials must preserve local evidence. Residents may request review."
    outside_text = "Officials should intervene elsewhere."
    project = Project.init(root)
    _, revision = project.import_text("in-scope.txt", selected_text)
    _, outside_revision = project.import_text("outside.txt", outside_text)
    sdk_result = cast(
        dict[str, object], project.analyze_textual_descriptors([revision], languages=("en",))
    )
    project.close()

    cli_response = CliRunner().invoke(
        app,
        [
            "analyze",
            "textual-descriptors",
            str(root),
            revision,
            "--languages",
            "en",
        ],
    )
    assert cli_response.exit_code == 0, cli_response.output
    cli_result = json.loads(cli_response.output)

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    mcp_result = mcp_run_textual_descriptors([revision], ["en"])

    results = (sdk_result, cli_result, mcp_result)
    reopened = Project.open(root)
    try:
        normalized_runs = []
        for result in results:
            metadata = reopened.analysis_run_metadata(str(result["analysis_run_id"]))
            run = cast(dict[str, object], metadata["analysis_run"])
            assert metadata["revision_ids"] == [revision]
            assert run["status"] == "COMPLETED"
            assert cast(dict[str, object], run["config"])["execution_kind"] == "DETERMINISTIC"

            skill_run = reopened.db.execute(
                "SELECT scope_id,status,coverage_id FROM skill_runs WHERE id=?",
                (result["skill_run_id"],),
            ).fetchone()
            assert skill_run is not None
            assert skill_run["scope_id"] == run["scope_id"]
            assert skill_run["status"] == "COMPLETED"
            coverage_id = str(result["coverage_id"])
            assert skill_run["coverage_id"] == coverage_id
            coverage = reopened.coverage_descriptor(coverage_id)
            assert coverage.coverage_mode.value == "EXHAUSTIVE"
            assert coverage.scope_snapshot_id == run["scope_id"]
            assert coverage.population_definition == (
                "Canonical Unicode code points in every language-eligible revision selected "
                "from the original ScopeSnapshot"
            )
            assert coverage.population_count == coverage.inspection_count == len(selected_text)
            assert coverage.selection_policy == (
                "Exhaustive deterministic scan of the defined canonical text population against "
                "the versioned modal-marker inventory; no sampling or candidate cap"
            )
            assert coverage.completion_criterion == (
                "Every canonical code point in every selected language-eligible revision was "
                "checked against the declared modal-marker patterns"
            )
            assert coverage.termination_state is not None
            assert coverage.termination_state.value == "COMPLETED"
            assert coverage.termination_reason == "completed"

            finding_ids = cast(list[str], result["finding_ids"])
            normalized_findings = []
            for finding_id in finding_ids:
                finding = reopened.get_finding(finding_id)
                assert finding["coverage_id"] == coverage_id
                links = reopened.finding_evidence(finding_id)
                normalized_evidence = []
                for link in links:
                    span = reopened.db.execute(
                        "SELECT revision_id FROM source_spans WHERE id=?", (link["span_id"],)
                    ).fetchone()
                    assert span is not None
                    normalized_evidence.append(
                        {
                            "text": reopened.span_text(str(link["span_id"])),
                            "role": str(link["role"]),
                            "revision_id": span["revision_id"],
                        }
                    )
                normalized_findings.append(
                    {
                        "phenomenon": finding["phenomenon"],
                        "observation": finding["observation"],
                        "evidence": normalized_evidence,
                    }
                )
            normalized_runs.append(
                {
                    "revision_ids": metadata["revision_ids"],
                    "coverage": {
                        "mode": coverage.coverage_mode.value,
                        "population_definition": coverage.population_definition,
                        "population_count": coverage.population_count,
                        "selection_policy": coverage.selection_policy,
                        "inspection_count": coverage.inspection_count,
                        "completion_criterion": coverage.completion_criterion,
                        "termination_reason": coverage.termination_reason,
                        "termination_state": (
                            coverage.termination_state.value
                            if coverage.termination_state is not None
                            else None
                        ),
                    },
                    "findings": normalized_findings,
                }
            )

        assert normalized_runs[0] == normalized_runs[1] == normalized_runs[2]
        assert normalized_runs[0] == {
            "revision_ids": [revision],
            "coverage": {
                "mode": "EXHAUSTIVE",
                "population_definition": (
                    "Canonical Unicode code points in every language-eligible revision selected "
                    "from the original ScopeSnapshot"
                ),
                "population_count": len(selected_text),
                "selection_policy": (
                    "Exhaustive deterministic scan of the defined canonical text population against "
                    "the versioned modal-marker inventory; no sampling or candidate cap"
                ),
                "inspection_count": len(selected_text),
                "completion_criterion": (
                    "Every canonical code point in every selected language-eligible revision was "
                    "checked against the declared modal-marker patterns"
                ),
                "termination_reason": "completed",
                "termination_state": "COMPLETED",
            },
            "findings": [
                {
                    "phenomenon": "modal_marker",
                    "observation": (
                        "The declared modal marker 'must' occurs in the language-eligible source "
                        "population selected from the original scope."
                    ),
                    "evidence": [
                        {"text": "must", "role": EvidenceRole.EXAMPLE, "revision_id": revision}
                    ],
                },
                {
                    "phenomenon": "modal_marker",
                    "observation": (
                        "The declared modal marker 'may' occurs in the language-eligible source "
                        "population selected from the original scope."
                    ),
                    "evidence": [
                        {"text": "may", "role": EvidenceRole.EXAMPLE, "revision_id": revision}
                    ],
                },
            ],
        }
        assert outside_revision != revision
        assert all(
            evidence["revision_id"] == revision and evidence["text"] not in outside_text
            for finding in normalized_runs[0]["findings"]
            for evidence in finding["evidence"]
        )
        assert all(
            "should" not in finding["observation"] for finding in normalized_runs[0]["findings"]
        )
    finally:
        reopened.close()


@pytest.mark.parametrize("operation", ["textual-descriptors", "evidence-extraction"])
def test_mcp_preserves_explicit_empty_language_selection_semantics(
    operation, tmp_path, monkeypatch
):
    root = tmp_path / "research"
    project = Project.init(root)
    _, revision = project.import_text(
        "fixture.txt",
        "Officials must preserve local evidence. Residents said policy harms families.",
    )

    with pytest.raises(SkillManifestError) as sdk_error:
        if operation == "textual-descriptors":
            project.analyze_textual_descriptors([revision], languages=())
        else:
            project.extract_evidence([revision], languages=())
    project.close()

    command = "textual-descriptors" if operation == "textual-descriptors" else "evidence-extraction"
    cli_result = CliRunner().invoke(
        app,
        ["analyze", command, str(root), revision, "--languages", ""],
    )
    assert cli_result.exit_code == 1
    assert sdk_error.value.args[0] in cli_result.stderr
    assert "Traceback" not in cli_result.stderr

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    with pytest.raises(SkillManifestError) as mcp_error:
        if operation == "textual-descriptors":
            mcp_run_textual_descriptors([revision], [])
        else:
            mcp_run_evidence_extraction([revision], [])
    assert str(mcp_error.value) == str(sdk_error.value)


def test_empty_framework_selector_matches_sdk_cli_and_mcp(tmp_path, monkeypatch):
    root = tmp_path / "research"
    project = Project.init(root)
    _, revision = project.import_text("fixture.txt", "A bounded local research passage.")
    try:
        with pytest.raises(ProjectError, match="requires at least one framework"):
            project.analyze_controlled_cda([revision], "missing-preset", frameworks=())
    finally:
        project.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    with pytest.raises(ProjectError, match="requires at least one framework"):
        mcp_run_controlled_cda_review([revision], "missing-preset", frameworks=[])

    cli_result = CliRunner().invoke(
        app,
        [
            "analyze",
            "controlled-cda-review",
            str(root),
            "--revisions",
            revision,
            "--preset",
            "missing-preset",
            "--frameworks",
            "",
        ],
    )
    assert cli_result.exit_code == 1
    assert "controlled CDA review requires at least one framework" in cli_result.stderr
    assert "unknown model preset" not in cli_result.stderr
    assert "Traceback" not in cli_result.stderr


@pytest.mark.parametrize(
    ("batch", "frameworks", "expected_message"),
    [
        (False, None, "unknown model preset"),
        (True, None, "unknown model preset"),
        (True, "", "controlled CDA review requires at least one framework"),
    ],
)
def test_cli_framework_defaults_distinguish_omitted_from_empty(
    batch, frameworks, expected_message, tmp_path
):
    root = tmp_path / "research"
    project = Project.init(root)
    _, revision = project.import_text("fixture.txt", "A bounded local research passage.")
    project.close()
    command = "batch-controlled-cda-review" if batch else "controlled-cda-review"
    arguments = [
        "analyze",
        command,
        str(root),
        "--revisions",
        revision,
        "--preset",
        "missing-preset",
    ]
    if frameworks is not None:
        arguments.extend(["--frameworks", frameworks])

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 1
    assert expected_message in result.stderr
    assert "Traceback" not in result.stderr


def test_counterevidence_search_uses_original_scope_across_sdk_cli_and_mcp(tmp_path, monkeypatch):
    root = tmp_path / "research"
    project = Project.init(root)
    _, supporting_revision = project.import_text(
        "supporting.txt", "Officials say the policy protects families."
    )
    _, counterevidence_revision = project.import_text(
        "counterevidence.txt", "Residents say the policy harms families."
    )
    _, outside_revision = project.import_text(
        "outside.txt", "A separate report says the policy harms families."
    )
    original_scope = project.snapshot([supporting_revision, counterevidence_revision])

    analysis_run_id = project.create_run(original_scope, {"profile": "fixture-finding"})
    project.start_run(analysis_run_id)
    support_text = "Officials say the policy protects families."
    support_span_id = project.create_span(supporting_revision, 0, len(support_text))
    support_evidence_id = project.textual_evidence(support_span_id)
    finding_id = project.create_finding(
        analysis_run_id,
        "fixture.supporting-observation",
        "1.0.0",
        "policy-protection",
        "Officials describe the policy as protective.",
        interpretation="The passage presents the policy positively.",
    )
    project.attach_evidence(finding_id, support_evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(analysis_run_id)

    sdk_result = project.counterevidence_review(finding_id, "policy harms families")
    project.close()

    cli_result = CliRunner().invoke(
        app,
        [
            "analyze",
            "counterevidence-lexical",
            str(root),
            finding_id,
            "policy harms families",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_result_data = json.loads(cli_result.output)

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    mcp_result = mcp_run_counterevidence_lexical(finding_id, "policy harms families")

    reopened = Project.open(root)
    try:
        comparable_results = []
        results = (
            cast(dict[str, object], sdk_result),
            cast(dict[str, object], cli_result_data),
            cast(dict[str, object], mcp_result),
        )
        for result in results:
            metadata = reopened.analysis_run_metadata(str(result["analysis_run_id"]))
            assert metadata["revision_ids"] == [supporting_revision, counterevidence_revision]
            run = cast(dict[str, object], metadata["analysis_run"])
            assert run["status"] == "COMPLETED"
            assert run["scope_id"] == original_scope
            assert result["candidate_count"] == result["returned_count"] == 1
            assert result["coverage_mode"] == "EXHAUSTIVE"
            candidate_span_ids = cast(list[str], result["candidate_span_ids"])
            assert len(candidate_span_ids) == 1

            span_id = candidate_span_ids[0]
            span = reopened.db.execute(
                "SELECT revision_id FROM source_spans WHERE id=?", (span_id,)
            ).fetchone()
            assert span is not None
            assert span["revision_id"] == counterevidence_revision
            assert span["revision_id"] != outside_revision
            comparable_results.append(reopened.span_text(span_id))

            coverage = reopened.coverage_descriptor(str(result["coverage_id"]))
            assert coverage.scope_snapshot_id == original_scope
            assert coverage.population_count == sum(
                len(reopened.revision_text(revision_id))
                for revision_id in (supporting_revision, counterevidence_revision)
            )

        assert comparable_results == [
            "policy harms families",
            "policy harms families",
            "policy harms families",
        ]
        # The finding is supported only by A; finding-support evidence is not the
        # candidate universe, while the original A+B scope remains authoritative.
        supporting_links = reopened.finding_evidence(finding_id)
        assert len(supporting_links) == 1
        assert supporting_links[0]["span_id"] == support_span_id
    finally:
        reopened.close()
