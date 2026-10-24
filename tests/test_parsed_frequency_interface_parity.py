from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from typer.testing import CliRunner

from discourselens import Project, mcp_server
from discourselens.cli import app
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import build_parsed_token_artifact


class _ParsedFrequencyContractBackend:
    language = "en"
    backend_id = "interface-parity-fixture"
    model_id = "interface-parity-fixture-v1"

    def __init__(self, *, partial: bool) -> None:
        self.partial = partial

    def parse(self, text: str) -> list[ParsedToken]:
        if text != "They speak.":
            raise AssertionError("unexpected local interface-parity fixture input")
        return [
            ParsedToken("They", "they", "PRON", None, None, "ROOT", 0, 4),
            ParsedToken(
                "speak",
                "speak",
                "" if self.partial else "VERB",
                None,
                None if self.partial else 0,
                None if self.partial else "dep",
                5,
                10,
            ),
            ParsedToken(
                ".",
                ".",
                "PUNCT",
                None,
                None if self.partial else 1,
                None if self.partial else "punct",
                10,
                11,
            ),
        ]


def _normalized_page(page: dict[str, Any]) -> dict[str, Any]:
    rows = page["rows"]
    assert isinstance(rows, list)
    normalized_rows = []
    for row in rows:
        assert isinstance(row, dict)
        value_key = "value" if "value" in row else "token"
        normalized_rows.append({"value": row[value_key], "count": row["count"]})
    return {
        "revision_ids": page["revision_ids"],
        "artifact_ids": page["artifact_ids"],
        "field": page["field"],
        "offset": page["offset"],
        "limit": page["limit"],
        "total_values": page["total_values"],
        "has_more": page["has_more"],
        "coverage": page["coverage"],
        "field_coverage": page["field_coverage"],
        "rows": normalized_rows,
    }


def _assert_no_sensitive_annotation_payload(value: Any) -> None:
    forbidden_keys = {
        "text",
        "surface",
        "token_id",
        "token_ids",
        "relative_path",
        "path",
        "char_start",
        "char_end",
    }
    if isinstance(value, dict):
        assert not forbidden_keys.intersection(value)
        for nested in value.values():
            _assert_no_sensitive_annotation_payload(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_sensitive_annotation_payload(nested)


@pytest.mark.parametrize(
    ("partial", "expected_status"), [(False, "SUPPORTED"), (True, "LIMITED")]
)
def test_parsed_frequency_coverage_has_project_cli_mcp_contract_parity(
    tmp_path, monkeypatch, partial: bool, expected_status: str
) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    project_root = tmp_path / f"parsed-frequency-{expected_status.lower()}"
    with Project.init(project_root) as project:
        _, revision_id = project.import_text("fixture.txt", "They speak.")
        artifact = build_parsed_token_artifact(
            project, revision_id, _ParsedFrequencyContractBackend(partial=partial)
        )
        sdk_scope_id = project.snapshot([revision_id])

        sdk_page = project.corpus_parsed_frequency_page(
            sdk_scope_id, [artifact.artifact_id], field="pos"
        )

        monkeypatch.setenv("DISCOURSELENS_PROJECT", str(project_root))
        cli_result = CliRunner().invoke(
            app,
            [
                "corpus",
                "parsed-frequency",
                str(project_root),
                "--revisions",
                revision_id,
                "--artifacts",
                artifact.artifact_id,
                "--field",
                "pos",
            ],
        )
        assert cli_result.exit_code == 0, cli_result.output
        cli_page = json.loads(cli_result.output)

        async def call_mcp():
            return await mcp_server.server.call_tool(
                "corpus_parsed_frequency",
                {
                    "revision_ids": [revision_id],
                    "artifact_ids": [artifact.artifact_id],
                    "field": "pos",
                },
            )

        mcp_result = asyncio.run(call_mcp())
        assert not mcp_result.is_error
        mcp_page = json.loads(mcp_result.content[0].text)

        normalized_sdk = _normalized_page(dict(sdk_page))
        normalized_cli = _normalized_page(cli_page)
        normalized_mcp = _normalized_page(mcp_page)
        assert normalized_sdk == normalized_cli == normalized_mcp
        assert normalized_sdk["revision_ids"] == [revision_id]
        assert normalized_sdk["artifact_ids"] == [artifact.artifact_id]
        assert normalized_sdk["field_coverage"]["status"] == expected_status
        assert normalized_sdk["coverage"]["pos"]["status"] == expected_status
        assert normalized_sdk["coverage"]["tokenization"]["status"] == "SUPPORTED"
        _assert_no_sensitive_annotation_payload(sdk_page)
        _assert_no_sensitive_annotation_payload(cli_page)
        _assert_no_sensitive_annotation_payload(mcp_page)
        assert "They speak." not in cli_result.output
        assert "They speak." not in mcp_result.content[0].text

        scope_ids = [sdk_page["scope_id"], cli_page["scope_id"], mcp_page["scope_id"]]
        for scope_id in scope_ids:
            assert project.scope_revisions(scope_id) == [revision_id]
