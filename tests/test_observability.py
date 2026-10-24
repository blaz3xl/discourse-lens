import json
import logging
from collections.abc import Iterator, Mapping
from io import StringIO

from typer.testing import CliRunner

from discourselens import Project, mcp_server
from discourselens.cli import app
from discourselens.observability import JsonEventFormatter, configure_cli_logging, log_event


def test_structured_event_allowlists_fields_and_uses_jsonl_stderr():
    output = StringIO()
    restore = configure_cli_logging(verbose=True, debug=False, stream=output)
    try:
        logger = logging.getLogger("discourselens.test")
        log_event(
            logger,
            "document.import.completed",
            document_id="00000000-0000-4000-8000-000000000001",
            revision_id="00000000-0000-4000-8000-000000000002",
            status="SUCCESS",
            byte_count=27,
            filename="private-research-title.pdf",
            text="RESEARCH-CONTENT-MUST-NOT-APPEAR",
            secret="fake-api-key-must-not-appear",
        )
    finally:
        restore()

    lines = output.getvalue().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "document.import.completed"
    assert event["document_id"] == "00000000-0000-4000-8000-000000000001"
    assert event["revision_id"] == "00000000-0000-4000-8000-000000000002"
    assert event["status"] == "SUCCESS"
    assert event["byte_count"] == 27
    assert "private-research-title" not in lines[0]
    assert "RESEARCH-CONTENT" not in lines[0]
    assert "fake-api-key" not in lines[0]


def test_malformed_identifiers_and_credential_shaped_exception_names_are_suppressed():
    output = StringIO()
    restore = configure_cli_logging(verbose=True, debug=False, stream=output)
    try:
        logger = logging.getLogger("discourselens.test.credential-shaped")
        log_event(
            logger,
            "document.import.completed",
            document_id="private-research-title",
            revision_id="sk-proj-not-a-real-key",
            status="SUCCESS",
            byte_count=8,
        )
        log_event(
            logger,
            "analysis.run.failed",
            analysis_run_id="00000000-0000-4000-8000-000000000003",
            error_type="ghp_not_a_real_token",
            status="FAILED",
        )
        log_event(
            logger,
            "analysis.run.failed",
            analysis_run_id="00000000-0000-4000-8000-000000000004",
            error_type="ModelProviderError",
            status="FAILED",
        )
    finally:
        restore()

    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert "document_id" not in events[0]
    assert "revision_id" not in events[0]
    assert events[0]["byte_count"] == 8
    assert events[1]["analysis_run_id"] == "00000000-0000-4000-8000-000000000003"
    assert "error_type" not in events[1]
    assert events[2]["error_type"] == "ModelProviderError"
    assert "private-research-title" not in output.getvalue()
    assert "sk-proj-not-a-real-key" not in output.getvalue()
    assert "ghp_not_a_real_token" not in output.getvalue()


def test_uuid_v1_identifiers_are_suppressed_but_generated_uuid_versions_remain_safe():
    output = StringIO()
    restore = configure_cli_logging(verbose=True, debug=False, stream=output)
    try:
        log_event(
            logging.getLogger("discourselens.test.uuid-version"),
            "document.import.completed",
            document_id="00000000-0000-1000-8000-001122334455",
            revision_id="00000000-0000-7000-8000-000000000001",
            status="SUCCESS",
            byte_count=12,
        )
    finally:
        restore()

    event = json.loads(output.getvalue())
    assert "document_id" not in event
    assert event["revision_id"] == "00000000-0000-7000-8000-000000000001"
    assert event["byte_count"] == 12
    assert "001122334455" not in output.getvalue()


def test_password_required_import_status_is_preserved_in_safe_event_fields():
    output = StringIO()
    restore = configure_cli_logging(verbose=True, debug=False, stream=output)
    try:
        log_event(
            logging.getLogger("discourselens.test.password-required"),
            "document.import.completed",
            status="PASSWORD_REQUIRED",
            byte_count=64,
        )
    finally:
        restore()

    event = json.loads(output.getvalue())
    assert event["status"] == "PASSWORD_REQUIRED"
    assert event["byte_count"] == 64


def test_lifecycle_statuses_are_event_specific_and_byte_counts_are_bounded():
    output = StringIO()
    restore = configure_cli_logging(verbose=True, debug=False, stream=output)
    try:
        logger = logging.getLogger("discourselens.test.lifecycle-contract")
        log_event(
            logger,
            "analysis.run.created",
            analysis_run_id="00000000-0000-4000-8000-000000000011",
            scope_id="00000000-0000-4000-8000-000000000012",
            status="RUNNING",
        )
        log_event(
            logger,
            "analysis.run.created",
            analysis_run_id="00000000-0000-4000-8000-000000000013",
            scope_id="00000000-0000-4000-8000-000000000014",
            status="PENDING",
        )
        log_event(
            logger,
            "document.import.completed",
            status="COMPLETED",
            byte_count=1 << 100,
        )
        log_event(
            logger,
            "document.import.completed",
            status="SUCCESS",
            byte_count=64,
        )
    finally:
        restore()

    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert "status" not in events[0]
    assert events[1]["status"] == "PENDING"
    assert "status" not in events[2]
    assert "byte_count" not in events[2]
    assert events[3]["status"] == "SUCCESS"
    assert events[3]["byte_count"] == 64


def test_json_formatter_does_not_trust_custom_level_names_or_message_content():
    record = logging.LogRecord(
        "discourselens.test.custom-level",
        987654321,
        __file__,
        1,
        "PRIVATE-LOG-MESSAGE",
        (),
        None,
    )
    record.levelname = "PRIVATE-CUSTOM-LEVEL-SECRET"
    record.discourselens_event = "analysis.run.started"
    record.discourselens_fields = {
        "analysis_run_id": "00000000-0000-4000-8000-000000000015",
        "status": "RUNNING",
        "text": "PRIVATE-CORPUS-TEXT",
    }

    formatted = JsonEventFormatter().format(record)
    event = json.loads(formatted)

    assert event["level"] == "OTHER"
    assert event["analysis_run_id"] == "00000000-0000-4000-8000-000000000015"
    assert event["status"] == "RUNNING"
    assert "PRIVATE" not in formatted


def test_json_formatter_suppresses_mapping_errors_without_leaking_to_stderr(capsys):
    class ExplodingFields(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError("PRIVATE-CORPUS-TEXT /private/research/source.pdf")

        def __len__(self) -> int:
            return 1

    output = StringIO()
    restore = configure_cli_logging(verbose=True, debug=False, stream=output)
    try:
        logging.getLogger("discourselens.test.malformed-fields").info(
            "PRIVATE-LOG-MESSAGE",
            extra={
                "discourselens_event": "analysis.run.started",
                "discourselens_fields": ExplodingFields(),
            },
            exc_info=(RuntimeError, RuntimeError("PRIVATE-EXCEPTION-DETAIL"), None),
        )
    finally:
        restore()

    captured = capsys.readouterr()
    lines = output.getvalue().splitlines()
    assert captured.out == ""
    assert captured.err == ""
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "analysis.run.started"
    assert "analysis_run_id" not in event
    assert "PRIVATE" not in output.getvalue()


def test_cli_verbose_logs_analysis_lifecycle_only_to_stderr(tmp_path):
    project_root = tmp_path / "research"
    project = Project.init(project_root)
    _, revision_id = project.import_text("speech.txt", "Officials may inspect PRIVATE-CORPUS-TEXT.")
    project.close()

    result = CliRunner().invoke(
        app,
        ["--verbose", "analyze", "textual-descriptors", str(project_root), revision_id],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["analysis_run_id"]
    events = [json.loads(line) for line in result.stderr.splitlines()]
    assert [event["event"] for event in events] == [
        "analysis.run.created",
        "analysis.run.started",
        "analysis.run.completed",
    ]
    assert "PRIVATE-CORPUS-TEXT" not in result.stderr
    assert all(event["analysis_run_id"] for event in events)


def test_cli_verbose_document_import_does_not_log_source_name_or_content(tmp_path):
    project_root = tmp_path / "research"
    Project.init(project_root).close()
    source = tmp_path / "private-title.txt"
    source.write_text("PRIVATE-RESEARCH-CONTENT", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["--verbose", "document", "import", str(project_root), str(source)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "SUCCESS"
    event = json.loads(result.stderr)
    assert event["event"] == "document.import.completed"
    assert event["byte_count"] == len("PRIVATE-RESEARCH-CONTENT")
    assert "private-title.txt" not in result.stderr
    assert "PRIVATE-RESEARCH-CONTENT" not in result.stderr


def test_cli_debug_adds_only_safe_import_preflight_metadata(tmp_path):
    project_root = tmp_path / "research"
    Project.init(project_root).close()
    source = tmp_path / "sensitive-name.txt"
    source.write_text("SENSITIVE-CORPUS-TEXT", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["--debug", "document", "import", str(project_root), str(source)],
    )

    assert result.exit_code == 0, result.output
    events = [json.loads(line) for line in result.stderr.splitlines()]
    assert [event["event"] for event in events] == [
        "document.import.started",
        "document.import.completed",
    ]
    assert events[0]["level"] == "DEBUG"
    assert events[0]["byte_count"] == len("SENSITIVE-CORPUS-TEXT")
    assert "sensitive-name.txt" not in result.stderr
    assert "SENSITIVE-CORPUS-TEXT" not in result.stderr


def test_mcp_verbose_diagnostics_stay_off_protocol_stdout(monkeypatch, capsys, tmp_path):
    project_root = tmp_path / "research"
    Project.init(project_root).close()
    monkeypatch.setattr(
        "sys.argv", ["discourselens-mcp", "--project", str(project_root), "--verbose"]
    )

    async def fake_stdio() -> None:
        log_event(
            logging.getLogger("discourselens.mcp_server.test"),
            "analysis.run.created",
            analysis_run_id="run-123",
            scope_id="scope-456",
            status="PENDING",
            prompt="PRIVATE-CORPUS-CONTENT",
        )

    monkeypatch.setattr(mcp_server.server, "run_stdio_async", fake_stdio)
    mcp_server.main()
    captured = capsys.readouterr()

    assert captured.out == ""
    event = json.loads(captured.err)
    assert event["event"] == "analysis.run.created"
    assert "PRIVATE-CORPUS-CONTENT" not in captured.err


def test_cli_is_quiet_by_default_and_debug_option_is_discoverable(tmp_path):
    project_root = tmp_path / "research"
    project = Project.init(project_root)
    _, revision_id = project.import_text("speech.txt", "Officials may respond.")
    project.close()
    runner = CliRunner()

    quiet = runner.invoke(
        app,
        ["analyze", "textual-descriptors", str(project_root), revision_id],
    )
    help_result = runner.invoke(app, ["--help"])

    assert quiet.exit_code == 0
    assert quiet.stderr == ""
    assert "--verbose" in help_result.stdout
    assert "--debug" in help_result.stdout
