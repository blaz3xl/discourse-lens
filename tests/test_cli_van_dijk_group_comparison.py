from __future__ import annotations

import csv
import io
import json

import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def _comparison_fixture(root):
    selected_text = "Officials describe a shared public policy debate."
    second_text = "Residents discuss the same public policy debate."
    outside_text = "OUTSIDE_SCOPE_SENTINEL describes a separate event."
    with Project.init(root) as project:
        _, first_revision = project.import_text("first.txt", selected_text)
        _, second_revision = project.import_text("second.txt", second_text)
        _, outside_revision = project.import_text("outside.txt", outside_text)
        codebook_id = project.create_codebook("descriptive comparison")
        code_id = project.create_code(codebook_id, "collective action")
        first_span = project.create_span(first_revision, 0, len("Officials"))
        second_span = project.create_span(second_revision, 0, len("Residents"))
        outside_span = project.create_span(outside_revision, 0, len("OUTSIDE_SCOPE_SENTINEL"))
        first_annotation = project.annotate_span(code_id, first_span)
        second_annotation = project.annotate_span(code_id, second_span)
        outside_annotation = project.annotate_span(code_id, outside_span)
    return {
        "project_path": root,
        "first_revision": first_revision,
        "second_revision": second_revision,
        "outside_revision": outside_revision,
        "codebook_id": codebook_id,
        "first_annotation": first_annotation,
        "second_annotation": second_annotation,
        "outside_annotation": outside_annotation,
        "outside_text": outside_text,
    }


def _payload(fixture, *, first_annotation=None, second_annotation=None):
    return {
        "groups": [
            {
                "label": "Officials",
                "annotation_ids": [first_annotation or fixture["first_annotation"]],
            },
            {
                "label": "Residents",
                "annotation_ids": [second_annotation or fixture["second_annotation"]],
            },
        ],
        "shared_topic_event": "A shared policy debate",
        "comparability_basis": "Both sources address the same stated debate",
        "codebook_id": fixture["codebook_id"],
    }


def _write_input(tmp_path, payload, *, name="comparison.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _invoke(fixture, input_path, *extra_args):
    return CliRunner().invoke(
        app,
        [
            "skills",
            "van-dijk-group-comparison",
            str(fixture["project_path"]),
            "--revision",
            fixture["first_revision"],
            "--revision",
            fixture["second_revision"],
            "--input",
            str(input_path),
            *extra_args,
        ],
        terminal_width=160,
    )


def _normalized_output(result):
    plain = "".join(
        character for character in result.output if not 0x2500 <= ord(character) <= 0x257F
    )
    return " ".join(plain.split())


def test_van_dijk_comparison_json_uses_only_explicit_revisions_and_annotations(tmp_path):
    fixture = _comparison_fixture(tmp_path / "research")
    input_path = _write_input(tmp_path, _payload(fixture))

    result = _invoke(fixture, input_path, "--format", "json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["selected_annotation_count"] == 2
    assert payload["scope_snapshot_id"]
    assert [group["label"] for group in payload["groups"]] == ["Officials", "Residents"]
    returned_annotations = [
        annotation for group in payload["groups"] for annotation in group["annotations"]
    ]
    assert {item["annotation_id"] for item in returned_annotations} == {
        fixture["first_annotation"],
        fixture["second_annotation"],
    }
    assert {item["revision_id"] for item in returned_annotations} == {
        fixture["first_revision"],
        fixture["second_revision"],
    }
    assert fixture["outside_annotation"] not in {
        item["annotation_id"] for item in returned_annotations
    }
    assert fixture["outside_text"] not in result.stdout
    with Project.open(fixture["project_path"], read_only=True) as project:
        assert project.scope_revisions(payload["scope_snapshot_id"]) == [
            fixture["first_revision"],
            fixture["second_revision"],
        ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"groups":', "valid JSON"),
        (b'{"groups": [], "unexpected": "private-marker"}', "exactly groups"),
        (
            (
                b'{"groups":[{"label":"A"},{"label":"B","annotation_ids":[]}],'
                b'"shared_topic_event":"event","comparability_basis":"basis",'
                b'"codebook_id":"book"}'
            ),
            "exactly label and annotation_ids",
        ),
        (b"\xffinvalid-json", "valid UTF-8"),
        (b'{"groups": [], "groups": []}', "duplicate keys"),
    ],
)
def test_van_dijk_comparison_rejects_malformed_or_invalid_input_before_project_open(
    tmp_path, content, message
):
    input_path = tmp_path / "invalid.json"
    input_path.write_bytes(content)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "skills",
            "van-dijk-group-comparison",
            str(tmp_path / "not-a-project"),
            "--revision",
            "revision-id",
            "--input",
            str(input_path),
        ],
        terminal_width=160,
    )

    assert result.exit_code == 2
    assert message in _normalized_output(result)
    assert "Traceback" not in result.output
    assert "private-marker" not in result.output


def test_van_dijk_comparison_rejects_input_over_64_kib_before_project_open(tmp_path):
    input_path = tmp_path / "large.json"
    input_path.write_bytes(b"x" * (64 * 1024 + 1))

    result = CliRunner().invoke(
        app,
        [
            "skills",
            "van-dijk-group-comparison",
            str(tmp_path / "not-a-project"),
            "--revision",
            "revision-id",
            "--input",
            str(input_path),
        ],
        terminal_width=160,
    )

    assert result.exit_code == 2
    assert "65536-byte limit" in _normalized_output(result)
    assert "Traceback" not in result.output


def test_van_dijk_comparison_requires_nonempty_unique_revision_scope(tmp_path):
    fixture = _comparison_fixture(tmp_path / "research")
    input_path = _write_input(tmp_path, _payload(fixture))
    base = [
        "skills",
        "van-dijk-group-comparison",
        str(fixture["project_path"]),
        "--input",
        str(input_path),
    ]
    runner = CliRunner()

    missing = runner.invoke(app, base)
    empty = runner.invoke(app, [*base, "--revision", ""])
    duplicate = runner.invoke(
        app,
        [
            *base,
            "--revision",
            fixture["first_revision"],
            "--revision",
            fixture["first_revision"],
        ],
    )

    assert missing.exit_code == 2 and "--revision" in missing.output
    assert empty.exit_code == 2 and "nonempty" in empty.output
    assert duplicate.exit_code == 2 and "must not be duplicated" in duplicate.output
    assert all("Traceback" not in result.output for result in (missing, empty, duplicate))


def test_van_dijk_comparison_surfaces_annotation_scope_violation_cleanly(tmp_path):
    fixture = _comparison_fixture(tmp_path / "research")
    input_path = _write_input(
        tmp_path,
        _payload(fixture, second_annotation=fixture["outside_annotation"]),
    )

    result = _invoke(fixture, input_path)

    assert result.exit_code == 2
    assert "every selected annotation must be in the original ScopeSnapshot" in _normalized_output(
        result
    )
    assert fixture["outside_text"] not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("output_format", ["jsonl", "csv", "table"])
def test_van_dijk_comparison_row_formats_emit_group_and_annotation_rows_and_bounded_metadata(
    tmp_path, output_format
):
    fixture = _comparison_fixture(tmp_path / "research")
    input_path = _write_input(tmp_path, _payload(fixture))

    result = _invoke(fixture, input_path, "--format", output_format)

    assert result.exit_code == 0, result.output
    assert "van-dijk-group-comparison metadata " in result.stderr
    metadata_text = result.stderr.split("metadata ", 1)[1]
    metadata = json.loads(metadata_text)
    assert metadata["analysis_run_id"]
    assert metadata["scope_snapshot_id"]
    assert metadata["selected_annotation_count"] == 2
    assert metadata["group_count"] == 2
    assert "A shared policy debate" not in result.stderr
    assert "Officials" not in result.stderr

    if output_format == "jsonl":
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        assert [row["row_type"] for row in rows] == [
            "group",
            "annotation",
            "group",
            "annotation",
        ]
        assert rows[1]["annotation_id"] == fixture["first_annotation"]
    elif output_format == "csv":
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        assert [row["row_type"] for row in rows] == [
            "group",
            "annotation",
            "group",
            "annotation",
        ]
        assert rows[1]["annotation_id"] == fixture["first_annotation"]
    else:
        assert "row_type" in result.stdout
        assert "annotation" in result.stdout
        assert fixture["first_annotation"] in result.stdout
