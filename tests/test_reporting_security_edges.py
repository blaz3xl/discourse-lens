import os
import stat
from xml.etree import ElementTree
from zipfile import ZipFile

import pymupdf as fitz
import pytest

from discourselens import Project
from discourselens.errors import EvidenceValidationError
from discourselens.reporting import (
    analysis_payload,
    analysis_payload_page,
    comparison_payload,
    comparison_payload_page,
    render_analysis_html,
    render_analysis_markdown,
    render_finding_html,
    render_finding_markdown,
    write_analysis_report,
)
from discourselens.types import EvidenceRole


def _stringify_report_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(
            f"{_stringify_report_value(key)}\n{_stringify_report_value(item)}"
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify_report_value(item) for item in value)
    return str(value)


def _report_text(path, output_format):
    if output_format in {"markdown", "html", "json"}:
        return path.read_text(encoding="utf-8")
    if output_format == "pdf":
        with fitz.open(path) as document:
            return "\n".join(_stringify_report_value(page.get_text()) for page in document)
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    return "\n".join(item.text or "" for item in root.iter())


def _assert_safe_display_name(value, original_path, safe_name):
    serialized = _stringify_report_value(value)
    assert original_path not in serialized
    markdown_unescaped = serialized.replace(r"\-", "-").replace(r"\.", ".")
    assert safe_name in markdown_unescaped


def test_report_sanitizes_absolute_document_path_in_evidence_citation(tmp_path):
    project = Project.init(tmp_path / "project")
    display_name = str(tmp_path / "private-source.txt")
    safe_name = "private-source.txt"
    _, revision = project.import_text(display_name, "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {})
    evidence = project.textual_evidence(project.create_span(revision, 0, 6))
    finding = project.create_finding(run, "fixture", "1.0.0", "lexical choice", "Apples occurs")
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)

    renderers = (
        lambda: render_finding_markdown(project, finding),
        lambda: render_finding_html(project, finding),
        lambda: render_analysis_markdown(project, run),
        lambda: render_analysis_html(project, run),
        lambda: analysis_payload(project, run),
        lambda: analysis_payload_page(project, run, limit=1),
    )
    for render in renderers:
        _assert_safe_display_name(render(), display_name, safe_name)

    for output_format in ("markdown", "html", "json", "pdf", "docx"):
        destination = tmp_path / f"report.{output_format}"
        write_analysis_report(project, run, destination, output_format)
        assert destination.exists()
        _assert_safe_display_name(
            _report_text(destination, output_format), display_name, safe_name
        )


def test_report_sanitizes_absolute_document_path_in_annotation_location(tmp_path):
    project = Project.init(tmp_path / "project")
    display_name = str(tmp_path / "private-source.txt")
    safe_name = "private-source.txt"
    _, revision = project.import_text(display_name, "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {})
    codebook = project.create_codebook("descriptive")
    code = project.create_code(codebook, "fruit")
    project.annotate_span(code, project.create_span(revision, 0, 6), "Human annotation.")

    _assert_safe_display_name(analysis_payload(project, run), display_name, safe_name)
    _assert_safe_display_name(
        analysis_payload_page(project, run, limit=1), display_name, safe_name
    )


def test_report_rejects_duplicate_model_metadata_keys_before_export(tmp_path):
    project = Project.init(tmp_path / "duplicate-model-metadata")
    _, revision = project.import_text("source.txt", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {})
    execution_id = project.record_model_execution(
        provider_id="fixture-provider",
        credential_alias="fixture-alias",
        model="fixture-model",
        reasoning=None,
        request_hash="request-hash",
        response_hash=None,
        analysis_run_id=run,
        parameters={"metadata": {"temperature": 0.1}},
    )
    project.db.execute(
        "UPDATE model_executions SET parameters_json=? WHERE id=?",
        (
            (
                '{"metadata":{"api_key":"duplicate-key-hidden-secret"},'
                '"metadata":{"temperature":0.1}}'
            ),
            execution_id,
        ),
    )
    project.db.commit()

    with pytest.raises(EvidenceValidationError, match="model execution report metadata is invalid"):
        analysis_payload(project, run)


def test_comparison_report_rejects_secret_like_config_metadata(tmp_path):
    project = Project.init(tmp_path / "comparison-secret")
    _, left_revision = project.import_text("left.txt", "apples")
    _, right_revision = project.import_text("right.txt", "oranges")
    left_scope = project.snapshot([left_revision])
    right_scope = project.snapshot([right_revision])
    comparison_id = project.create_comparison_run(
        left_scope,
        right_scope,
        metric="frequency",
        calculation_version="fixture-v1",
        config={"api_key": "comparison-secret-sentinel"},
        rows=[{"token": "apples", "left_count": 1, "right_count": 0}],
    )

    with pytest.raises(
        EvidenceValidationError, match="comparison report configuration.*secret-like"
    ):
        comparison_payload(project, comparison_id)
    with pytest.raises(
        EvidenceValidationError, match="comparison report configuration.*secret-like"
    ):
        comparison_payload_page(project, comparison_id)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions are required")
def test_report_export_is_private_under_permissive_umask(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    destination = project.root / "reports" / "analysis.json"

    previous_umask = os.umask(0)
    try:
        write_analysis_report(project, run, destination, "json")
    finally:
        os.umask(previous_umask)

    assert destination.is_file()
    assert stat.S_IMODE(destination.stat().st_mode) & 0o077 == 0
