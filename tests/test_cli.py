import csv
import hashlib
import io
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens import Project
from discourselens.cli import app
from discourselens.errors import ModelProviderError, ProjectError
from discourselens.models import (
    CredentialReference,
    ModelCapabilities,
    ModelCapabilityProfile,
    ModelPreset,
    ModelPrice,
)
from discourselens.nlp import ParsedToken
from discourselens.nlp_artifacts import (
    SURFACE_TOKEN_SCHEMA_VERSION,
    build_parsed_token_artifact,
    build_surface_token_artifact,
)
from discourselens.secrets import ReferenceSecretStore
from discourselens.types import EvidenceRole
from discourselens.vectors import (
    HybridHit,
    RetrievalSearchResult,
    VectorChunk,
    build_vector_index,
)
from discourselens.versioning import software_version


class _ParsedFixture:
    language = "en"
    backend_id = "fixture"
    model_id = "fixture-v1"

    def parse(self, text: str) -> list[ParsedToken]:
        return [ParsedToken("Evidence", "evidence", "NOUN", None, None, "ROOT", 0, 8)]


def test_init_cli(tmp_path):
    result = CliRunner().invoke(app, ["init", str(tmp_path / "research")])
    assert (
        result.exit_code == 0
        and (tmp_path / "research" / ".discourselens" / "project.sqlite").exists()
    )


def test_cli_version_matches_installed_distribution():
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout == f"DiscourseLens {software_version()}\n"
    assert result.stderr == ""


def test_document_import_help_matches_supported_formats():
    result = CliRunner().invoke(app, ["document", "import", "--help"])

    assert result.exit_code == 0
    assert "Markdown/HTML" in result.stdout
    assert "DOCX" in result.stdout
    assert "OCR is opt-in and local" in result.stdout


def test_cli_converts_database_open_failure_to_actionable_error(tmp_path):
    missing_project = tmp_path / "missing-project"

    result = CliRunner().invoke(
        app,
        ["project", "offline", str(missing_project), "--enabled"],
    )

    assert result.exit_code == 1
    assert "Could not access the project database" in result.output
    assert "safely" in result.output
    assert "Verify the project path, schema, and filesystem permissions." in result.output
    assert str(missing_project) not in result.output
    assert "Traceback" not in result.output
    assert "OperationalError" not in result.output


def test_cli_converts_core_domain_error_without_traceback(tmp_path):
    project = Project.init(tmp_path / "research")
    project.close()

    result = CliRunner().invoke(
        app,
        ["document", "show", str(tmp_path / "research"), "missing-revision"],
    )

    assert result.exit_code == 2
    assert "unknown revision" in result.output
    assert "Traceback" not in result.output


def test_missing_revision_uses_invalid_argument_exit_code_across_commands(tmp_path):
    root = tmp_path / "research"
    project = Project.init(root)
    project.close()

    result = CliRunner().invoke(app, ["corpus", "frequency", str(root), "missing-revision"])

    assert result.exit_code == 2
    assert "unknown document revision" in result.output
    assert "Traceback" not in result.output


def test_cli_redacts_absolute_paths_from_project_errors(monkeypatch, tmp_path):
    sensitive_model_path = tmp_path / "credential-fixture-secret" / "local-spacy-model"

    def fail_load_spacy(language: str, model_id: str):
        raise ProjectError(f"spaCy model {model_id!r} is not installed locally")

    monkeypatch.setattr(cli_module, "load_spacy", fail_load_spacy)
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "build-nlp",
            str(tmp_path / "research"),
            "missing-revision",
            "en",
            "--backend",
            "spacy",
            "--model",
            str(sensitive_model_path),
        ],
    )

    assert result.exit_code == 1
    assert "is not installed locally" in result.output
    assert str(sensitive_model_path) not in result.output
    assert "credential-fixture-secret" not in result.output
    assert "Traceback" not in result.output


def test_cli_build_nlp_emits_partial_field_coverage_immediately(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")

    class PartialBackend:
        language = "en"
        backend_id = "spacy"
        model_id = "fixture-en-v1"

        def parse(self, text: str) -> list[ParsedToken]:
            if text != "They speak.":
                raise AssertionError("unexpected CLI fixture input")
            return [
                ParsedToken("They", "they", "", None, None, "ROOT", 0, 4),
                ParsedToken("speak", "speak", "VERB", None, None, None, 5, 10),
                ParsedToken(".", ".", "PUNCT", None, None, None, 10, 11),
            ]

    root = tmp_path / "cli-parsed-field-coverage"
    project = Project.init(root)
    _, revision = project.import_text("source.txt", "They speak.")
    project.close()
    monkeypatch.setattr(cli_module, "load_spacy", lambda language, model: PartialBackend())

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "build-nlp",
            str(root),
            revision,
            "en",
            "--backend",
            "spacy",
            "--model",
            "fixture-en-v1",
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = json.loads(result.output)
    assert artifact["coverage"]["tokenization"]["status"] == "SUPPORTED"
    assert artifact["coverage"]["lemma"]["status"] == "SUPPORTED"
    assert artifact["coverage"]["pos"]["status"] == "LIMITED"
    assert artifact["coverage"]["dependency_parse"]["status"] == "LIMITED"


def test_project_error_redactor_masks_unc_paths_but_preserves_https_urls():
    unc_message = cli_module._safe_project_error_message(
        ProjectError("failed //server/share/key")
    )
    url_message = cli_module._safe_project_error_message(
        ProjectError("documentation at https://example.test/path")
    )

    assert unc_message == "failed <path>"
    assert url_message == "documentation at https://example.test/path"


def test_cli_redacts_api_secrets_in_provider_error_urls(monkeypatch, tmp_path):
    root = tmp_path / "provider-error-redaction"
    Project.init(root).close()
    key = "gemini-query-key-fixture-secret"
    token = "oauth-query-token-fixture-secret"
    signature = "signed-url-signature-fixture-secret"
    client_secret = "client-secret-fixture-value"
    private_key = "private-key-fixture-value"
    session_token = "session-token-fixture-value"
    prefixed_api_key = "prefixed-api-key-fixture-value"

    def fail_review(*_args, **_kwargs):
        raise ModelProviderError(
            "provider request failed at "
            f"https://api.example/review?key={key}&token={token}"
            f"&X-Amz-Signature={signature}&client_secret={client_secret}"
            f"&private_key={private_key}&session_token={session_token}"
            f"&x_api_key={prefixed_api_key}"
        )

    monkeypatch.setattr(cli_module, "review_finding_with_model", fail_review)
    result = CliRunner().invoke(
        app,
        ["analyze", "evidence-review", str(root), "finding-id", "preset-id"],
    )

    assert result.exit_code == 1
    assert "provider request failed" in result.output
    assert "https://api.example/review?key=<redacted>&token=<redacted>" in result.output
    assert "X-Amz-Signature=<redacted>" in result.output
    assert all(
        secret not in result.output
        for secret in (key, token, signature, client_secret, private_key, session_token, prefixed_api_key)
    )
    assert "client_secret=<redacted>" in result.output
    assert "private_key=<redacted>" in result.output
    assert "session_token=<redacted>" in result.output
    assert "x_api_key=<redacted>" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "diagnostic",
    [
        "managed project storage changed during database open",
        "read-only project database is unavailable",
        "project root is unavailable or unsafe",
    ],
)
def test_storage_project_errors_use_safe_actionable_database_message(diagnostic):
    message = cli_module._safe_project_error_message(ProjectError(diagnostic))

    assert message == (
        "Could not access the project database safely. Verify the project path, "
        "schema, and filesystem permissions."
    )


def test_cli_hides_paths_and_sanitizes_control_characters_in_os_errors(monkeypatch, tmp_path):
    def fail_init(path):
        raise PermissionError(f"permission denied for {path}\n\x1b[31mError: forged\x1b[0m")

    monkeypatch.setattr(cli_module.Project, "init", fail_init)

    sensitive_path = tmp_path / "credential-fixture-secret" / "research"
    result = CliRunner().invoke(app, ["init", str(sensitive_path)])

    assert result.exit_code == 1
    assert result.output == (
        "Error: Could not complete the filesystem operation. Verify the requested "
        "path and filesystem permissions.\n"
    )
    assert str(sensitive_path) not in result.output
    assert "credential-fixture-secret" not in result.output
    assert "\x1b" not in result.output
    assert "Traceback" not in result.output


def test_cli_table_output_escapes_terminal_controls_from_document_text(tmp_path):
    project = Project.init(tmp_path / "terminal-output")
    _, revision = project.import_text("terminal.txt", "alert \x1b[2J erase \u202e hidden")
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "search",
            "exact",
            str(tmp_path / "terminal-output"),
            revision,
            "alert",
            "--format",
            "table",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "\\x1b[2J" in result.output
    assert "\\u202e" in result.output
    assert "\x1b" not in result.output


def test_report_output_escapes_controls_only_when_written_to_a_terminal(monkeypatch):
    content = "first line\n\x1b[2J erase \u202e hidden\n"

    class TerminalCapture(io.StringIO):
        def isatty(self) -> bool:
            return True

    terminal = TerminalCapture()
    monkeypatch.setattr(cli_module.sys, "stdout", terminal)
    cli_module._echo_report_document(content)

    redirected = io.StringIO()
    monkeypatch.setattr(cli_module.sys, "stdout", redirected)
    cli_module._echo_report_document(content)

    assert terminal.getvalue() == "first line\n\\x1b[2J erase \\u202e hidden\n"
    # Click already strips ANSI sequences on redirected stdout; report markup is
    # otherwise left unescaped so a consumer can store its original text.
    assert redirected.getvalue() == "first line\n erase \u202e hidden\n"


def test_cli_document_dispersion_is_explicitly_scoped_and_citable(tmp_path):
    project = Project.init(tmp_path / "research")
    _, first = project.import_text("first.txt", "Migration policy.")
    _, second = project.import_text("second.txt", "Migration economy.")
    project.close()

    read = CliRunner().invoke(
        app,
        ["corpus", "dispersion", str(tmp_path / "research"), f"{first},{second}", "migration"],
    )
    assert read.exit_code == 0, read.output
    assert json.loads(read.output)["document_frequency"] == 2

    persisted = CliRunner().invoke(
        app,
        [
            "corpus",
            "dispersion-evidence",
            str(tmp_path / "research"),
            f"{first},{second}",
            "migration",
        ],
    )
    assert persisted.exit_code == 0, persisted.output
    assert json.loads(persisted.output)["evidence_id"]


def test_cli_lexical_bundles_uses_explicit_revision_selection(tmp_path):
    project = Project.init(tmp_path / "research")
    _, first = project.import_text("first.txt", "the public must act")
    _, second = project.import_text("second.txt", "the public must respond")
    project.import_text("outside.txt", "the public must disappear")
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "lexical-bundles",
            str(tmp_path / "research"),
            f"{first}, {second}",
            "--size",
            "3",
            "--min-frequency",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["scope_id"]
    assert payload["rows"] == [
        {"bundle": "the public must", "frequency": 2, "document_frequency": 2}
    ]


def test_cli_lexical_bundle_evidence_persists_exhaustive_count(tmp_path):
    root = tmp_path / "research"
    project = Project.init(root)
    _, first = project.import_text("a.txt", "the public must act")
    _, second = project.import_text("b.txt", "the public must respond")
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "lexical-bundle-evidence",
            str(root),
            f"{first},{second}",
            "the public must",
            "--size",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["count"] == 2
    assert payload["coverage_id"] and payload["evidence_id"]


def test_cli_kwic_returns_canonical_offsets(tmp_path):
    root = tmp_path / "research"
    project = Project.init(root)
    _, revision = project.import_text("speech.txt", "Café migración.")
    project.close()

    result = CliRunner().invoke(app, ["corpus", "kwic", str(root), revision, "migración"])

    assert result.exit_code == 0, result.output
    page = json.loads(result.output)
    assert page["total_count"] == 1 and page["has_more"] is False
    row = page["rows"][0]
    assert row["revision_id"] == revision
    assert row["start_offset"] == 5
    assert row["end_offset"] == 14

    evidence_result = CliRunner().invoke(
        app,
        [
            "corpus",
            "kwic-evidence",
            str(root),
            revision,
            "migración",
            str(row["start_offset"]),
            str(row["end_offset"]),
        ],
    )
    assert evidence_result.exit_code == 0, evidence_result.output
    evidence = json.loads(evidence_result.output)
    assert evidence["exact_text"] == "migración"
    assert evidence["evidence_id"]
    assert evidence["citation"] == "[speech.txt · ¶1]"


def test_cli_kwic_paginates_without_silent_truncation(tmp_path):
    root = tmp_path / "research"
    project = Project.init(root)
    _, revision = project.import_text("repeat.txt", "Migration one. Migration two.")
    project.close()

    first = CliRunner().invoke(
        app,
        ["corpus", "kwic", str(root), revision, "migration", "--limit", "1"],
    )
    second = CliRunner().invoke(
        app,
        [
            "corpus",
            "kwic",
            str(root),
            revision,
            "migration",
            "--limit",
            "1",
            "--offset",
            "1",
        ],
    )

    first_page = json.loads(first.output)
    second_page = json.loads(second.output)
    assert first.exit_code == second.exit_code == 0
    assert first_page["total_count"] == second_page["total_count"] == 2
    assert first_page["has_more"] is True
    assert second_page["has_more"] is False
    assert first_page["rows"][0]["start_offset"] == 0
    assert second_page["rows"][0]["start_offset"] == 15


def test_cli_dependency_patterns_use_complete_scoped_parsed_artifacts(tmp_path):
    class Backend:
        language = "es"
        backend_id = "fixture"
        model_id = "fixture-v1"

        def parse(self, text: str) -> list[ParsedToken]:
            return [
                ParsedToken("Café", "café", "NOUN", None, None, "ROOT", 0, 4),
                ParsedToken("migración", "migración", "NOUN", None, 0, "obj", 5, 14),
            ]

    root = tmp_path / "research"
    project = Project.init(root)
    _, revision = project.import_text("sample.txt", "Café migración")
    artifact = build_parsed_token_artifact(project, revision, Backend())
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "dependency-patterns",
            str(root),
            "--revisions",
            revision,
            "--artifacts",
            artifact.artifact_id,
            "--relation",
            "obj",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["rows"] == [
        {
            "head_lemma": "café",
            "relation": "obj",
            "dependent_lemma": "migración",
            "frequency": 1,
            "document_frequency": 1,
        }
    ]


def test_cli_help_exposes_only_supported_public_commands():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Evidence-first discourse and corpus analysis for reproducible research." in result.output
    assert "Configure safe local diagnostics" not in result.output
    assert "frequency-cmd" not in result.output
    assert "kwic-cmd" not in result.output
    assert "collocations-cmd" not in result.output
    assert "corpus" in result.output and "run" in result.output
    assert "--version" in result.output and "--verbose" in result.output


def test_skill_discovery_is_human_readable_by_default_and_machine_readable_on_request():
    runner = CliRunner()

    table = runner.invoke(app, ["skills", "list"])
    machine = runner.invoke(app, ["skills", "list", "--format", "json"])
    jsonl = runner.invoke(app, ["skills", "list", "--format", "jsonl"])

    assert table.exit_code == machine.exit_code == jsonl.exit_code == 0
    assert "builtin.evidence-extraction" in table.output
    assert "execution_kind" in table.output
    assert json.loads(machine.output)[0]["id"] == "builtin.textual-descriptors"
    assert '"id": "builtin.textual-descriptors"' in jsonl.output


def test_provider_discovery_table_does_not_include_credential_material():
    table = CliRunner().invoke(app, ["providers", "list"])
    machine = CliRunner().invoke(app, ["providers", "list", "--format", "json"])

    assert table.exit_code == machine.exit_code == 0
    assert "openai" in table.output and "anthropic" in table.output
    assert "credential" not in table.output.casefold()
    providers = json.loads(machine.output)
    assert any(item["provider_id"] == "anthropic" for item in providers)


def test_cli_rejects_unsupported_machine_output_format_as_parameter_error():
    result = CliRunner().invoke(app, ["providers", "list", "--format", "yaml"])

    assert result.exit_code == 2
    assert "--format" in result.output
    assert "format must be json, jsonl, csv, or table" in result.output
    assert "Traceback" not in result.output
    assert result.stdout == ""


def test_emit_preserves_valid_json_format_and_unicode(capsys):
    cli_module._emit({"token": "café", "count": 2})

    assert capsys.readouterr().out == '{"token": "café", "count": 2}\n'


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_cli_rejects_non_finite_json_without_partial_stdout(monkeypatch, non_finite):
    monkeypatch.setattr(
        cli_module,
        "supported_providers",
        lambda: [
            SimpleNamespace(
                provider_id="valid-first-row",
                transport="fixture",
                endpoint_policy="none",
                generation=True,
                embeddings=False,
                reranking=False,
            ),
            SimpleNamespace(
                provider_id="invalid-second-row",
                transport="fixture",
                endpoint_policy="none",
                generation=non_finite,
                embeddings=False,
                reranking=False,
            ),
        ],
    )

    result = CliRunner().invoke(app, ["providers", "list", "--format", "json"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "finite numeric values" in result.stderr
    assert "Traceback" not in result.output


def test_cli_language_capability_probe_requires_and_uses_explicit_backend_selection(
    monkeypatch, tmp_path
):
    captured: dict[str, object] = {}

    def inspect(language, *, backend, model, model_dir=None):
        captured.update(language=language, backend=backend, model=model, model_dir=model_dir)
        return [
            SimpleNamespace(
                language=language,
                capability="dependency_parse",
                backend=backend,
                model=model,
                operational_status="SUPPORTED",
                assurance_status="TESTED",
                limitation=None,
            )
        ]

    monkeypatch.setattr(cli_module, "inspect_local_backend", inspect)
    result = CliRunner().invoke(
        app,
        [
            "nlp",
            "capabilities",
            "ar",
            "--backend",
            "stanza",
            "--model",
            "ar-ud-fixture",
            "--model-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "language": "ar",
        "backend": "stanza",
        "model": "ar-ud-fixture",
        "model_dir": str(tmp_path),
    }
    assert json.loads(result.output)[0]["operational_status"] == "SUPPORTED"

    incomplete = CliRunner().invoke(app, ["nlp", "capabilities", "ar", "--backend", "stanza"])
    assert incomplete.exit_code != 0


def test_nlp_capability_error_hides_absolute_paths_and_secret_text(tmp_path, monkeypatch):
    from discourselens import nlp

    sensitive_path = tmp_path / "private-model-store" / "model"
    secret = "model-probe-secret-fixture"

    class _FailingBackend:
        language = "en"
        backend_id = "stanza"
        model_id = "fixture-model"

        def parse(self, text):
            raise OSError(
                f"could not read {sensitive_path}; Authorization: Bearer {secret}"
            )

    monkeypatch.setattr(nlp, "load_stanza", lambda language, model, model_dir: _FailingBackend())
    result = CliRunner().invoke(
        app,
        [
            "nlp",
            "capabilities",
            "en",
            "--backend",
            "stanza",
            "--model",
            "fixture-model",
            "--model-dir",
            str(sensitive_path),
        ],
    )

    assert result.exit_code == 2
    assert "selected local NLP backend failed its capability probe" in result.stderr
    assert "<path>" in result.stderr
    assert str(sensitive_path) not in result.output
    assert "private-model-store" not in result.output
    assert secret not in result.output
    assert "Traceback" not in result.output


def test_cli_lists_reviewed_provider_transports_without_project_or_network():
    result = CliRunner().invoke(app, ["providers", "list", "--format", "json"])
    assert result.exit_code == 0
    providers = {item["provider_id"]: item for item in json.loads(result.output)}
    assert providers["anthropic"]["transport"] == "anthropic-messages"
    assert providers["ollama"]["endpoint_policy"] == "required"
    assert providers["gemini-embeddings"]["embeddings"] is True
    assert providers["gemini-embeddings"]["generation"] is False
    assert providers["jina-embeddings"]["embeddings"] is True
    assert providers["jina-embeddings"]["generation"] is False


def test_cli_evidence_review_routes_explicit_reranker_credential_and_model(tmp_path, monkeypatch):
    root = tmp_path / "research"
    Project.init(root).close()
    captured_route: tuple[str, str] | None = None
    captured_kwargs: dict[str, object] = {}
    gateway = object()

    def build_gateway(project, provider_id, credential_alias):
        nonlocal captured_route
        captured_route = (provider_id, credential_alias)
        return gateway

    def review(project, finding, preset, **kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return SimpleNamespace(finding_id="reviewed")

    monkeypatch.setattr(cli_module, "build_reranking_gateway", build_gateway)
    monkeypatch.setattr(cli_module, "review_finding_with_model", review)
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "evidence-review",
            str(root),
            "finding-1",
            "review-preset",
            "--reranking-provider",
            "jina-reranker",
            "--reranking-credential",
            "university",
            "--reranking-model",
            "jina-reranker-v3.5",
            "--reranking-top-n",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured_route == ("jina-reranker", "university")
    assert captured_kwargs["reranking_gateway"] is gateway
    assert captured_kwargs["reranking_credential_alias"] == "university"
    assert captured_kwargs["reranking_model"] == "jina-reranker-v3.5"
    assert captured_kwargs["reranking_top_n"] == 5


def test_cli_stores_context_from_file_and_stdin_without_echoing_content(tmp_path):
    root = tmp_path / "context-project"
    Project.init(root).close()
    excerpt_file = tmp_path / "excerpt.txt"
    excerpt_file.write_text("A supplied context excerpt.", encoding="utf-8")

    file_result = CliRunner().invoke(
        app,
        ["evidence", "add-user-context", str(root), "field memo", str(excerpt_file)],
    )
    assert file_result.exit_code == 0, file_result.output
    file_payload = json.loads(file_result.output)
    assert file_payload["kind"] == "USER_CONTEXT"
    assert "A supplied context excerpt" not in file_result.output

    stdin_result = CliRunner().invoke(
        app,
        ["evidence", "add-external-source", str(root), "archive", "2026-09-29", "-"],
        input="An excerpt passed through stdin.",
    )
    assert stdin_result.exit_code == 0, stdin_result.output
    stdin_payload = json.loads(stdin_result.output)
    assert stdin_payload["kind"] == "EXTERNAL_SOURCE"
    assert "An excerpt passed through stdin" not in stdin_result.output


def test_cli_reports_invalid_utf8_prompt_as_a_parameter_error(tmp_path):
    root = tmp_path / "research"
    Project.init(root).close()
    prompt_file = tmp_path / "invalid-prompt.txt"
    prompt_file.write_bytes(bytes((0xFF,)))

    result = CliRunner().invoke(
        app,
        ["models", "preview-request", str(root), "unused-preset", str(prompt_file)],
    )

    assert result.exit_code == 2
    assert "prompt file must be valid UTF-8" in result.output
    assert "Traceback" not in result.output
    assert "UnicodeDecodeError" not in result.output


def test_cli_evidence_review_passes_explicit_context_ids(tmp_path, monkeypatch):
    root = tmp_path / "research"
    project = Project.init(root)
    external = project.external_source_evidence("archive", "Excerpt.", retrieved_at="today")
    user = project.user_context_evidence("memo", "Context.")
    project.close()
    captured: dict[str, object] = {}

    def review(project, finding, preset, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(finding_id="reviewed")

    monkeypatch.setattr(cli_module, "review_finding_with_model", review)
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "evidence-review",
            str(root),
            "finding-id",
            "preset",
            "--context-evidence-id",
            external,
            "--context-evidence-id",
            user,
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["context_evidence_ids"] == (external, user)


def test_cli_reranking_rejects_incomplete_provider_selection(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "evidence-review",
            str(tmp_path / "missing-project"),
            "finding",
            "preset",
            "--reranking-provider",
            "jina-reranker",
        ],
    )
    assert result.exit_code != 0
    assert "reranking requires" in result.output


def test_cli_batch_cda_passes_explicit_reranking_route(tmp_path, monkeypatch):
    root = tmp_path / "research"
    Project.init(root).close()
    gateway = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "build_reranking_gateway",
        lambda project, provider_id, credential_alias: gateway,
    )

    def batch(project, revisions, preset, **kwargs):
        captured["revisions"] = revisions
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(cli_module, "run_batch_controlled_cda_review", batch)
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "batch-controlled-cda-review",
            str(root),
            "--revisions",
            "rev-a,rev-b",
            "--preset",
            "review",
            "--reranking-provider",
            "jina-reranker",
            "--reranking-credential",
            "lab",
            "--reranking-model",
            "jina-reranker-v3.5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["revisions"] == ["rev-a", "rev-b"]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["reranking_gateway"] is gateway
    assert kwargs["reranking_provider_id"] == "jina-reranker"
    assert kwargs["reranking_credential_alias"] == "lab"
    assert kwargs["reranking_model"] == "jina-reranker-v3.5"


def test_cli_prints_bundled_skill_methodology_without_a_project():
    result = CliRunner().invoke(app, ["skills", "docs", "builtin.evidence-extraction"])
    assert result.exit_code == 0, result.output
    assert "# Descriptive evidence extraction" in result.output
    assert "deterministic candidate-selection skill" in result.output


def test_cli_persists_explicit_anthropic_reasoning_mode_and_budget(tmp_path):
    root = tmp_path / "research"
    project = Project.init(root)
    project.save_credential_reference(CredentialReference("anthropic", "lab", "DL_ANTHROPIC"))
    project.close()
    runner = CliRunner()

    capabilities = runner.invoke(
        app,
        [
            "models",
            "set-capabilities",
            str(root),
            "anthropic",
            "claude-manual-fixture",
            "--capabilities-json",
            (
                '{"reasoning_efforts":["high"],"reasoning_modes":["manual"],'
                '"generation_parameters":["max_tokens"]}'
            ),
            "--source-reference",
            "fixture://anthropic-model-contract",
        ],
    )
    assert capabilities.exit_code == 0, capabilities.output

    saved = runner.invoke(
        app,
        [
            "models",
            "save-preset",
            str(root),
            "review",
            "anthropic",
            "lab",
            "claude-manual-fixture",
            "--reasoning",
            "high",
            "--reasoning-mode",
            "manual",
            "--reasoning-budget-tokens",
            "2048",
            "--parameters-json",
            '{"max_tokens":8192}',
        ],
    )
    assert saved.exit_code == 0, saved.output
    listed = runner.invoke(app, ["models", "list-presets", str(root), "--format", "json"])
    assert listed.exit_code == 0, listed.output
    preset = json.loads(listed.output)[0]
    assert preset["reasoning_mode"] == "manual"
    assert preset["reasoning_budget_tokens"] == 2048


def test_cli_rejects_whitespace_capability_values_as_parameter_error(tmp_path):
    root = tmp_path / "research"
    project = Project.init(root)
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "models",
            "set-capabilities",
            str(root),
            "fixture-provider",
            "fixture-model",
            "--capabilities-json",
            '{"reasoning_efforts":[" "]}',
        ],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "--capabilities-json" in result.stderr
    assert "non-empty strings" in result.stderr
    assert "Traceback" not in result.stderr
    reopened = Project.open(root)
    try:
        assert reopened.model_capability_profile("fixture-provider", "fixture-model") is None
    finally:
        reopened.close()


def test_model_preset_discovery_is_readable_and_never_resolves_secrets(tmp_path, monkeypatch):
    root = tmp_path / "research"
    project = Project.init(root)
    project.save_credential_reference(CredentialReference("openai-compatible", "lab", "DL_LAB_KEY"))
    project.save_model_preset(
        ModelPreset("analysis", "openai-compatible", "lab", "local-research-model", "high")
    )
    project.close()
    monkeypatch.setenv("DL_LAB_KEY", "secret-fixture-that-must-not-be-printed")
    runner = CliRunner()

    table = runner.invoke(app, ["models", "list-presets", str(root)])
    machine = runner.invoke(app, ["models", "list-presets", str(root), "--format", "json"])

    assert table.exit_code == machine.exit_code == 0
    assert "analysis" in table.output and "local-research-model" in table.output
    assert "lab" in table.output and "secret-fixture-that-must-not-be-printed" not in table.output
    record = json.loads(machine.output)[0]
    assert record["credential_alias"] == "lab"
    assert "secret-fixture-that-must-not-be-printed" not in machine.output


def test_cli_selects_metadata_into_a_frozen_scope(tmp_path):
    project = Project.init(tmp_path / "research")
    document, revision = project.import_text("speech.txt", "Evidence must remain local.")
    project.set_document_metadata(document, "genre", "speech")
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "document",
            "select-metadata",
            str(tmp_path / "research"),
            "--filters-json",
            '{"genre":"speech"}',
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.output)["revision_ids"] == [revision]


def test_cli_exports_analysis_report_as_pdf(tmp_path):
    project_root = tmp_path / "research"
    project = Project.init(project_root)
    _, revision = project.import_text("speech.txt", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    evidence = project.textual_evidence(project.create_span(revision, 0, 6))
    finding = project.create_finding(run, "fixture", "1.0.0", "lexicon", "Apples occurs.")
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    project.close()

    destination = tmp_path / "report.pdf"
    result = CliRunner().invoke(
        app,
        ["report", "export", str(project_root), run, str(destination), "--format", "pdf"],
    )
    assert result.exit_code == 0
    assert destination.read_bytes().startswith(b"%PDF")


def test_cli_lists_and_reads_declarative_analysis_profiles(tmp_path):
    runner = CliRunner()
    root = tmp_path / "research"
    Project.init(root).close()
    profile_file = tmp_path / "focused.json"
    profile_file.write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "kind": "controlled-cda",
                "frameworks": ["fairclough-text"],
                "languages": ["en"],
            }
        ),
        encoding="utf-8",
    )

    saved = runner.invoke(
        app, ["profiles", "save-controlled", str(root), "focused", str(profile_file)]
    )
    assert saved.exit_code == 0
    listed_table = runner.invoke(app, ["profiles", "list", str(root)])
    listed = runner.invoke(app, ["profiles", "list", str(root), "--format", "json"])
    listed_jsonl = runner.invoke(app, ["profiles", "list", str(root), "--format", "jsonl"])
    assert listed_table.exit_code == listed.exit_code == listed_jsonl.exit_code == 0
    assert "name" in listed_table.output and "focused" in listed_table.output
    assert json.loads(listed.output)[0]["name"] == "focused"
    assert '"name": "focused"' in listed_jsonl.output
    info = runner.invoke(app, ["profiles", "info", str(root), "focused"])
    assert info.exit_code == 0
    assert json.loads(info.output)["frameworks"] == ["fairclough-text"]


def test_profile_list_empty_project_renders_clear_table_state(tmp_path):
    root = tmp_path / "research"
    Project.init(root).close()

    result = CliRunner().invoke(app, ["profiles", "list", str(root)])

    assert result.exit_code == 0
    assert result.output == "No results.\n"


def test_grouped_cli_import_inspect_and_corpus_operations(tmp_path):
    runner = CliRunner()
    project = tmp_path / "research"
    assert runner.invoke(app, ["project", "init", str(project)]).exit_code == 0
    source = tmp_path / "note.txt"
    source.write_text("Evidence must stay local. Evidence remains local.", encoding="utf-8")

    imported = runner.invoke(app, ["document", "import", str(project), str(source)])
    assert imported.exit_code == 0
    payload = json.loads(imported.output)
    revision = payload["revision_id"]

    inspected = runner.invoke(app, ["document", "show", str(project), revision])
    assert inspected.exit_code == 0
    assert json.loads(inspected.output)["mime_type"] == "text/plain"

    frequency = runner.invoke(app, ["corpus", "frequency", str(project), revision])
    assert frequency.exit_code == 0
    frequency_payload = json.loads(frequency.output)
    assert {"token": "evidence", "count": 2} in frequency_payload["rows"]
    assert frequency_payload["total_tokens"] == 7
    assert frequency_payload["has_more"] is False

    frequency_evidence = runner.invoke(
        app, ["corpus", "frequency-evidence", str(project), revision, "Evidence"]
    )
    assert frequency_evidence.exit_code == 0
    assert json.loads(frequency_evidence.output)["count"] == 2

    csv_frequency = runner.invoke(
        app, ["corpus", "frequency", str(project), revision, "--format", "csv"]
    )
    assert csv_frequency.exit_code == 0
    assert csv_frequency.output.startswith("token,count")

    paged_csv = runner.invoke(
        app,
        [
            "corpus",
            "frequency",
            str(project),
            revision,
            "--format",
            "csv",
            "--offset",
            "1",
            "--limit",
            "1",
        ],
    )
    assert paged_csv.exit_code == 0
    assert paged_csv.output.splitlines()[:2] == ["token,count", "local,2"]
    assert "has_more=true" in paged_csv.output
    collocations = runner.invoke(
        app,
        ["corpus", "collocations", str(project), revision, "evidence", "--format", "jsonl"],
    )
    assert collocations.exit_code == 0
    assert {json.loads(line)["token"] for line in collocations.output.splitlines()} >= {
        "must",
        "remains",
    }

    association = runner.invoke(
        app,
        [
            "corpus",
            "collocation-analysis",
            str(project),
            revision,
            "evidence",
            "--min-frequency",
            "1",
        ],
    )
    assert association.exit_code == 0, association.output
    association_payload = json.loads(association.output)
    assert association_payload["calculation_version"] == "surface-window-association-v2"
    assert association_payload["population_tokens"] > 0
    assert "not significance tests" in association_payload["interpretation_note"]

    association_evidence = runner.invoke(
        app,
        [
            "corpus",
            "collocation-evidence",
            str(project),
            revision,
            "evidence",
            "must",
            "--min-frequency",
            "1",
        ],
    )
    assert association_evidence.exit_code == 0, association_evidence.output
    evidence_payload = json.loads(association_evidence.output)
    assert evidence_payload["evidence_id"]
    assert evidence_payload["coverage_id"]

    ngram_table = runner.invoke(
        app,
        ["corpus", "ngrams", str(project), revision, "--size", "2", "--format", "table"],
    )
    assert ngram_table.exit_code == 0
    assert "ngram" in ngram_table.output and "evidence must" in ngram_table.output

    exact = runner.invoke(app, ["search", "exact", str(project), revision, "Evidence must"])
    assert exact.exit_code == 0
    exact_page = json.loads(exact.output)
    assert len(exact_page["rows"]) == 1
    assert exact_page["total_count"] == 1
    assert exact_page["has_more"] is False

    analysis = runner.invoke(
        app,
        ["analyze", "textual-descriptors", str(project), revision, "--languages", "en"],
    )
    assert analysis.exit_code == 0
    assert json.loads(analysis.output)["finding_ids"]

    extraction = runner.invoke(
        app,
        ["analyze", "evidence-extraction", str(project), revision, "--languages", "en"],
    )
    assert extraction.exit_code == 0
    payload = json.loads(extraction.output)
    assert payload["candidate_count"] >= 1
    assert payload["finding_ids"]

    artifact = runner.invoke(app, ["corpus", "build-tokens", str(project), revision])
    assert artifact.exit_code == 0
    artifact_payload = json.loads(artifact.output)
    artifact_id = artifact_payload["artifact_id"]
    assert artifact_payload["schema_version"] == SURFACE_TOKEN_SCHEMA_VERSION
    normalized = runner.invoke(
        app, ["corpus", "normalized-frequency", str(project), revision, artifact_id]
    )
    assert normalized.exit_code == 0
    normalized_payload = json.loads(normalized.output)
    assert {"token": "evidence", "count": 2} in normalized_payload["rows"]
    assert normalized_payload["field"] == "normalized"
    assert normalized_payload["calculation_version"] == "surface-token-artifact-frequency-v2"
    assert normalized_payload["tokenizer"] == "unicode-surface-tokenizer-v2"
    assert normalized_payload["normalization"] == "NFC-after-casefold-v1"
    assert normalized_payload["total_values"] == 5
    normalized_csv = runner.invoke(
        app,
        [
            "corpus",
            "normalized-frequency",
            str(project),
            revision,
            artifact_id,
            "--format",
            "csv",
            "--offset",
            "1",
            "--limit",
            "1",
        ],
    )
    assert normalized_csv.exit_code == 0
    assert normalized_csv.output.splitlines()[:2] == ["token,count", "local,2"]
    assert "has_more=true" in normalized_csv.output
    direct_project = Project.open(project)
    parsed_artifact = build_parsed_token_artifact(direct_project, revision, _ParsedFixture())
    direct_project.close()
    parsed_frequency = runner.invoke(
        app,
        [
            "corpus",
            "parsed-frequency",
            str(project),
            "--revisions",
            revision,
            "--artifacts",
            parsed_artifact.artifact_id,
            "--field",
            "lemma",
            "--format",
            "csv",
        ],
    )
    assert parsed_frequency.exit_code == 0
    assert parsed_frequency.output.splitlines()[:2] == ["token,count", "evidence,1"]
    assert "has_more=false" in parsed_frequency.output

    corpus = runner.invoke(app, ["corpus", "create", str(project), "sample"])
    assert corpus.exit_code == 0
    corpus_id = json.loads(corpus.output)["corpus_id"]
    assert runner.invoke(app, ["corpus", "add", str(project), corpus_id, revision]).exit_code == 0
    assert (
        json.loads(runner.invoke(app, ["corpus", "list", str(project)]).output)[0]["revision_count"]
        == 1
    )
    subcorpus = runner.invoke(
        app,
        [
            "corpus",
            "create-subcorpus",
            str(project),
            corpus_id,
            "only-document",
            "--revisions",
            revision,
        ],
    )
    assert subcorpus.exit_code == 0

    assert (
        json.loads(runner.invoke(app, ["corpus", "list-subcorpora", str(project)]).output)[0][
            "revision_count"
        ]
        == 1
    )

    other = tmp_path / "other.txt"
    other.write_text("Different evidence.", encoding="utf-8")
    other_revision = json.loads(
        runner.invoke(app, ["document", "import", str(project), str(other)]).output
    )["revision_id"]
    compared = runner.invoke(
        app,
        [
            "corpus",
            "compare-frequency",
            str(project),
            revision,
            other_revision,
            "--limit",
            "2",
        ],
    )
    assert compared.exit_code == 0
    compared_payload = json.loads(compared.output)
    assert len(compared_payload["rows"]) == 2
    assert compared_payload["total_rows"] == 6
    assert compared_payload["has_more"] is True

    keyness_page = runner.invoke(
        app,
        [
            "corpus",
            "keyness",
            str(project),
            revision,
            other_revision,
            "--offset",
            "1",
            "--limit",
            "1",
        ],
    )
    assert keyness_page.exit_code == 0
    assert len(json.loads(keyness_page.output)["rows"]) == 1

    persisted_comparison = runner.invoke(
        app,
        ["corpus", "compare-documents", str(project), revision, other_revision, "--format", "csv"],
    )
    assert persisted_comparison.exit_code == 0
    assert persisted_comparison.output.startswith("token,left_count,right_count")

    persisted_page = runner.invoke(
        app,
        [
            "corpus",
            "compare-documents",
            str(project),
            revision,
            other_revision,
            "--limit",
            "2",
            "--offset",
            "2",
        ],
    )
    assert persisted_page.exit_code == 0
    comparison_page = json.loads(persisted_page.output)
    assert len(comparison_page["rows"]) == 2
    assert comparison_page["offset"] == 2
    assert comparison_page["total_rows"] == 6
    assert comparison_page["has_more"] is True

    persisted_json = runner.invoke(
        app,
        ["corpus", "compare-documents", str(project), revision, other_revision],
    )
    comparison_id = json.loads(persisted_json.output)["comparison_id"]
    comparison_page = runner.invoke(
        app,
        [
            "report",
            "comparison",
            str(project),
            comparison_id,
            "--limit",
            "1",
            "--offset",
            "1",
        ],
    )
    assert comparison_page.exit_code == 0
    assert "Showing rows 2–" in comparison_page.output

    comparative_finding = runner.invoke(
        app,
        [
            "corpus",
            "comparison-finding",
            str(project),
            comparison_id,
            "evidence",
            "--comparability-basis",
            "Both generated texts are sampled from the same declared genre.",
        ],
    )
    assert comparative_finding.exit_code == 0, comparative_finding.output
    finding_payload = json.loads(comparative_finding.output)
    assert finding_payload["validation"] == {
        "valid": True,
        "errors": [],
        "semantic_entailment_assessed": False,
    }
    assert finding_payload["finding_id"]
    assert finding_payload["analysis_run_id"]


def test_cli_named_corpora_comparison_can_create_comparative_finding(tmp_path):
    runner = CliRunner()
    project_path = tmp_path / "named-corpus-finding"
    project = Project.init(project_path)
    _, left_revision = project.import_text("left.txt", "migration policy migration")
    _, right_revision = project.import_text("right.txt", "migration public discourse")
    left_corpus = project.create_corpus("left")
    right_corpus = project.create_corpus("right")
    project.add_revision_to_corpus(left_corpus, left_revision)
    project.add_revision_to_corpus(right_corpus, right_revision)
    project.close()

    comparison = runner.invoke(
        app,
        ["corpus", "compare-named", str(project_path), left_corpus, right_corpus],
    )
    assert comparison.exit_code == 0, comparison.output
    comparison_id = json.loads(comparison.output)["comparison_id"]

    finding = runner.invoke(
        app,
        [
            "corpus",
            "comparison-finding",
            str(project_path),
            comparison_id,
            "migration",
            "--comparability-basis",
            "The two generated corpora use the same declared genre and period.",
        ],
    )
    assert finding.exit_code == 0, finding.output
    payload = json.loads(finding.output)
    assert payload["validation"] == {
        "valid": True,
        "errors": [],
        "semantic_entailment_assessed": False,
    }

    with Project.open(project_path) as reopened:
        run = reopened.analysis_run_metadata(str(payload["analysis_run_id"]))
        assert run["revision_ids"] == sorted([left_revision, right_revision])
        assert reopened.validate_finding(str(payload["finding_id"])).valid


def test_cli_named_corpora_comparison_supports_machine_and_human_formats(tmp_path):
    runner = CliRunner()
    project_path = tmp_path / "named-corpus-formats"
    project = Project.init(project_path)
    _, left_revision = project.import_text("left.txt", "migration migration policy")
    _, right_revision = project.import_text("right.txt", "migration public discourse")
    left_corpus = project.create_corpus("left")
    right_corpus = project.create_corpus("right")
    project.add_revision_to_corpus(left_corpus, left_revision)
    project.add_revision_to_corpus(right_corpus, right_revision)
    project.close()

    common = ["corpus", "compare-named", str(project_path), left_corpus, right_corpus]
    json_result = runner.invoke(app, [*common, "--format", "json", "--limit", "1"])
    assert json_result.exit_code == 0, json_result.output
    json_payload = json.loads(json_result.output)
    assert json_payload["total_rows"] == 4
    assert len(json_payload["rows"]) == 1
    assert json_payload["has_more"] is True

    jsonl_result = runner.invoke(
        app, [*common, "--format", "jsonl", "--offset", "1", "--limit", "2"]
    )
    assert jsonl_result.exit_code == 0, jsonl_result.output
    jsonl_rows = [json.loads(line) for line in jsonl_result.output.splitlines()]
    assert [row["token"] for row in jsonl_rows] == ["migration", "policy"]
    assert all(row["comparison_id"] for row in jsonl_rows)
    assert all(row["left_scope_id"] and row["right_scope_id"] for row in jsonl_rows)
    assert all(row["has_more"] is True for row in jsonl_rows)

    csv_result = runner.invoke(app, [*common, "--format", "csv", "--offset", "1", "--limit", "2"])
    assert csv_result.exit_code == 0, csv_result.output
    csv_rows = list(csv.DictReader(csv_result.output.splitlines()))
    assert [row["token"] for row in csv_rows] == ["migration", "policy"]
    assert all(row["comparison_id"] for row in csv_rows)
    assert all(row["left_scope_id"] and row["right_scope_id"] for row in csv_rows)
    next_page = runner.invoke(
        app,
        [
            "report",
            "comparison",
            str(project_path),
            csv_rows[0]["comparison_id"],
            "--offset",
            "2",
            "--limit",
            "1",
        ],
    )
    assert next_page.exit_code == 0, next_page.output
    assert "Showing rows 3–" in next_page.output

    table_result = runner.invoke(
        app, [*common, "--format", "table", "--offset", "1", "--limit", "2"]
    )
    assert table_result.exit_code == 0, table_result.output
    assert "migration" in table_result.output and "comparison_id" in table_result.output


def test_cli_search_exact_csv_is_byte_faithful_by_default_and_opt_in_safe(tmp_path):
    runner = CliRunner()
    project_root = tmp_path / "formula-corpus"
    project = Project.init(project_root)
    _, revision = project.import_text("formula.txt", "=1+1 formula-leading corpus text")
    project.close()

    common = ["search", "exact", str(project_root), revision, "formula-leading"]
    default_csv = runner.invoke(app, [*common, "--format", "csv"])
    safe_csv = runner.invoke(
        app,
        [*common, "--format", "csv", "--spreadsheet-safe-csv"],
    )
    json_with_csv_safety = runner.invoke(
        app,
        [*common, "--format", "json", "--spreadsheet-safe-csv"],
    )
    csv_help = runner.invoke(app, ["search", "exact", "--help"])
    json_only_help = runner.invoke(app, ["corpus", "frequency-evidence", "--help"])

    assert default_csv.exit_code == 0, default_csv.output
    assert safe_csv.exit_code == 0, safe_csv.output
    assert next(csv.DictReader(default_csv.stdout.splitlines()))["text"].startswith("=1+1")
    assert next(csv.DictReader(safe_csv.stdout.splitlines()))["text"].startswith("'=1+1")
    assert json_with_csv_safety.exit_code != 0
    assert "Invalid value for --spreadsheet-safe-csv" in json_with_csv_safety.output
    assert "--format csv" in json_with_csv_safety.output
    assert "--spreadsheet-safe-csv" in csv_help.output
    assert "--spreadsheet-safe-csv" not in json_only_help.output


def test_cli_document_frequency_is_paginated_with_scope_and_artifact_provenance(tmp_path):
    runner = CliRunner()
    root = tmp_path / "frequency-project"
    project = Project.init(root)
    _, first = project.import_text("first.txt", "apple apple banana")
    _, second = project.import_text("second.txt", "apple cherry")
    first_artifact = build_surface_token_artifact(project, first)
    second_artifact = build_surface_token_artifact(project, second)
    project.close()

    result = runner.invoke(
        app,
        [
            "corpus",
            "document-frequency",
            str(root),
            f"{first},{second}",
            f"{first_artifact.artifact_id},{second_artifact.artifact_id}",
            "--offset",
            "1",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload["revision_ids"]) == {first, second}
    assert set(payload["artifact_ids"]) == {
        first_artifact.artifact_id,
        second_artifact.artifact_id,
    }
    assert payload["field"] == "document_frequency"
    assert payload["rows"] == [{"token": "banana", "count": 1}]
    assert payload["total_values"] == 3
    assert payload["has_more"] is True


def test_cli_renders_evidence_bound_finding_and_exports_bundle(tmp_path):
    project_root = tmp_path / "research"
    project = Project.init(project_root)
    _, revision = project.import_text("speech.txt", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {"profile": "test"})
    evidence = project.textual_evidence(project.create_span(revision, 0, 6))
    finding = project.create_finding(run, "test", "1.0.0", "lexicon", "Apples occurs.")
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    project.finalize_run(run)
    project.close()

    runner = CliRunner()
    report = runner.invoke(app, ["report", "finding", str(project_root), finding])
    assert report.exit_code == 0
    assert r"\[speech\.txt · ¶1\]" in report.output

    validation = runner.invoke(app, ["finding", "validate", str(project_root), finding])
    assert validation.exit_code == 0
    assert json.loads(validation.output)["valid"]

    exported = runner.invoke(
        app,
        ["export", "research-bundle", str(project_root), run, str(tmp_path / "bundle")],
    )
    assert exported.exit_code == 0
    assert (tmp_path / "bundle" / "manifest.json").exists()

    verified = runner.invoke(
        app, ["export", "verify-bundle", str(tmp_path / "bundle" / "manifest.json")]
    )
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.output)["integrity"] == "VERIFIED_SHA256"


def test_cli_bundle_full_canonical_text_requires_opt_in_but_keeps_cited_excerpt(tmp_path):
    project_root = tmp_path / "bundle-text-project"
    project = Project.init(project_root)
    source_text = "Private source corpus text."
    _, revision = project.import_text("source.txt", source_text)
    run = project.create_run(project.snapshot([revision]), {"profile": "test"})
    evidence = project.textual_evidence(project.create_span(revision, 0, len(source_text)))
    finding = project.create_finding(run, "fixture", "1.0.0", "text", "Source text exists.")
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    project.start_run(run)
    project.finalize_run(run)
    project.close()

    runner = CliRunner()
    default_output = tmp_path / "default-bundle"
    opted_output = tmp_path / "opted-bundle"
    default_export = runner.invoke(
        app,
        ["export", "research-bundle", str(project_root), run, str(default_output)],
    )
    opted_export = runner.invoke(
        app,
        [
            "export",
            "research-bundle",
            str(project_root),
            run,
            str(opted_output),
            "--include-canonical-text",
        ],
    )
    help_result = runner.invoke(app, ["export", "research-bundle", "--help"])

    assert default_export.exit_code == 0, default_export.output
    assert opted_export.exit_code == 0, opted_export.output
    default_payload = json.loads((default_output / "manifest.json").read_text(encoding="utf-8"))
    opted_payload = json.loads((opted_output / "manifest.json").read_text(encoding="utf-8"))
    assert default_payload["canonical_texts"] == []
    assert opted_payload["canonical_texts"] == [
        {
            "revision_id": revision,
            "sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "text": source_text,
        }
    ]
    help_text = " ".join(help_result.output.replace("│", " ").split())
    assert "--include-canonical-text" in help_text
    assert "Off by default for privacy" in help_text
    assert "cited excerpts are still present" in help_text
    assert "64 MiB" in help_text
    assert default_payload["evidence"][finding][0]["text"] == source_text


def test_cli_lists_and_inspects_immutable_analysis_runs(tmp_path):
    project_root = tmp_path / "research"
    project = Project.init(project_root)
    _, revision = project.import_text("speech.txt", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    project.close()
    runner = CliRunner()

    listed = runner.invoke(app, ["run", "list", str(project_root)])
    inspected = runner.invoke(app, ["run", "inspect", str(project_root), run])

    assert json.loads(listed.output)[0]["id"] == run
    payload = json.loads(inspected.output)
    assert payload["revision_ids"] == [revision]
    assert payload["analysis_run"]["config"] == {"profile": "fixture"}


def test_cli_runs_counterevidence_against_existing_finding_scope(tmp_path):
    project_root = tmp_path / "research"
    project = Project.init(project_root)
    _, revision = project.import_text("speech.txt", "Claim. Exception appears.")
    run = project.create_run(project.snapshot([revision]), {"profile": "test"})
    evidence = project.textual_evidence(project.create_span(revision, 0, 5))
    finding = project.create_finding(run, "fixture", "1.0.0", "claim", "Claim occurs.")
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    project.finalize_run(run)
    project.close()

    result = CliRunner().invoke(
        app, ["analyze", "counterevidence-lexical", str(project_root), finding, "Exception"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["candidate_span_ids"]
    assert payload["retrieval_run_id"]
    assert payload["candidate_count"] == payload["returned_count"] == 1
    assert payload["has_more"] is False


def test_cli_counterevidence_continues_the_same_persisted_retrieval_run(tmp_path):
    project_root = tmp_path / "research"
    project = Project.init(project_root)
    _, revision = project.import_text("speech.txt", "Claim. Exception one. Exception two.")
    run = project.create_run(project.snapshot([revision]), {"profile": "test"})
    evidence = project.textual_evidence(project.create_span(revision, 0, 5))
    finding = project.create_finding(run, "fixture", "1.0.0", "claim", "Claim occurs.")
    project.attach_evidence(finding, evidence, EvidenceRole.SUPPORT)
    project.finalize_run(run)
    project.close()

    runner = CliRunner()
    first_page = runner.invoke(
        app,
        [
            "analyze",
            "counterevidence-lexical",
            str(project_root),
            finding,
            "Exception",
            "--limit",
            "1",
        ],
    )
    assert first_page.exit_code == 0
    first_payload = json.loads(first_page.output)
    assert first_payload["candidate_count"] == 2
    assert first_payload["returned_count"] == 1
    assert first_payload["has_more"] is True

    second_page = runner.invoke(
        app,
        [
            "analyze",
            "counterevidence-lexical",
            str(project_root),
            finding,
            "Exception",
            "--retrieval-run",
            first_payload["retrieval_run_id"],
            "--offset",
            "1",
            "--limit",
            "1",
        ],
    )
    assert second_page.exit_code == 0
    second_payload = json.loads(second_page.output)
    assert second_payload["candidate_count"] == 2
    assert second_payload["returned_count"] == 1
    assert second_payload["offset"] == 1
    assert second_payload["has_more"] is False
    assert second_payload["retrieval_run_id"] == first_payload["retrieval_run_id"]
    assert second_payload["analysis_run_id"] == first_payload["analysis_run_id"]


def test_cli_stores_credential_reference_and_model_preset(tmp_path):
    runner = CliRunner()
    project = tmp_path / "research"
    runner.invoke(app, ["project", "init", str(project)])
    created = runner.invoke(
        app,
        [
            "providers",
            "add-credential",
            str(project),
            "openai-compatible",
            "lab",
            "--secret-env",
            "DL_LAB_KEY",
            "--endpoint",
            "http://localhost:11434/v1",
        ],
    )
    assert created.exit_code == 0
    preset = runner.invoke(
        app,
        [
            "models",
            "save-preset",
            str(project),
            "local",
            "openai-compatible",
            "lab",
            "llama",
            "--reasoning",
            "low",
            "--parameters-json",
            '{"temperature":0.2}',
        ],
    )
    assert preset.exit_code == 0
    listed = runner.invoke(app, ["models", "list-presets", str(project), "--format", "json"])
    assert json.loads(listed.output)[0]["name"] == "local"
    assert json.loads(json.loads(listed.output)[0]["parameters_json"]) == {"temperature": 0.2}
    declared = runner.invoke(
        app,
        [
            "models",
            "set-capabilities",
            str(project),
            "openai-compatible",
            "llama",
            "--capabilities-json",
            '{"generation_parameters":["temperature"],"system_prompt":false}',
            "--assurance",
            "UNASSESSED",
        ],
    )
    assert declared.exit_code == 0, declared.output
    capability_list = runner.invoke(
        app, ["models", "list-capabilities", str(project), "--format", "json"]
    )
    assert capability_list.exit_code == 0
    profile = json.loads(capability_list.output)[0]
    assert profile["model"] == "llama"
    assert json.loads(profile["capabilities_json"])["generation_parameters"] == ["temperature"]
    assert json.loads(profile["capabilities_json"])["system_prompt"] is False


def test_provider_credential_cli_only_exposes_non_secret_reference_options(tmp_path):
    root = tmp_path / "research"
    Project.init(root).close()
    runner = CliRunner()

    help_result = runner.invoke(app, ["providers", "add-credential", "--help"])
    assert help_result.exit_code == 0
    assert "--secret-reference" in help_result.output
    assert "--secret-env" in help_result.output
    assert "--api-key" not in help_result.output
    assert "--value" not in help_result.output

    raw_key = "sk-proj-cli-argv-fixture-secret"
    for forbidden_option in ("--api-key", "--value"):
        result = runner.invoke(
            app,
            [
                "providers",
                "add-credential",
                str(root),
                "openai",
                "personal",
                forbidden_option,
                raw_key,
            ],
        )
        assert result.exit_code != 0
        assert raw_key not in result.output

    positional_secret = runner.invoke(
        app,
        [
            "providers",
            "add-credential",
            str(root),
            "openai",
            "personal",
            raw_key,
        ],
    )
    assert positional_secret.exit_code != 0
    assert raw_key not in positional_secret.output
    assert "unexpected positional argument" in positional_secret.output


def test_provider_credential_cli_rejects_credentials_in_endpoint_url(tmp_path):
    root = tmp_path / "research"
    Project.init(root).close()
    secret = "sk-proj-endpoint-query-fixture-secret"

    result = CliRunner().invoke(
        app,
        [
            "providers",
            "add-credential",
            str(root),
            "openai-compatible",
            "lab",
            "--secret-reference",
            "DL_LAB_KEY",
            "--endpoint",
            f"https://user:password@api.example/v1?api_key={secret}#fragment",
        ],
    )

    assert result.exit_code != 0
    assert "without credentials" in result.output
    assert "query, or fragment" in result.output
    assert secret not in result.output
    assert "password" not in result.output
    project = Project.open(root)
    try:
        assert project.list_credential_references() == []
    finally:
        project.close()


def test_provider_credential_listing_never_echoes_endpoint_urls(tmp_path):
    root = tmp_path / "research"
    project = Project.init(root)
    endpoint = "https://api.example/v1"
    project.save_credential_reference(
        CredentialReference("openai-compatible", "lab", "DL_LAB_KEY", endpoint)
    )
    project.close()

    result = CliRunner().invoke(app, ["providers", "list-credentials", str(root)])

    assert result.exit_code == 0, result.output
    row = json.loads(result.output)[0]
    assert row["endpoint_configured"] is True
    assert "endpoint" not in row
    assert endpoint not in result.output


def test_provider_credential_cli_supports_multiple_named_aliases(tmp_path):
    root = tmp_path / "research"
    Project.init(root).close()
    runner = CliRunner()
    results = [
        runner.invoke(
            app,
            [
                "providers",
                "add-credential",
                str(root),
                "openai-compatible",
                alias,
                "--secret-reference",
                reference,
                "--endpoint",
                "http://localhost:11434/v1",
            ],
        )
        for alias, reference in (
            ("personal", "DL_PERSONAL_KEY"),
            ("university", "DL_UNIVERSITY_KEY"),
        )
    ]
    listed = runner.invoke(app, ["providers", "list-credentials", str(root)])

    assert all(result.exit_code == 0 for result in results)
    assert listed.exit_code == 0
    assert {row["credential_alias"] for row in json.loads(listed.output)} == {
        "personal",
        "university",
    }
    assert "DL_PERSONAL_KEY" in listed.output
    assert "DL_UNIVERSITY_KEY" in listed.output


def test_cli_accepts_keyring_reference_without_exposing_secret_value(tmp_path, monkeypatch):
    root = tmp_path / "research"
    Project.init(root).close()
    secret_value = "keyring-secret-fixture-must-not-be-printed"

    class FixtureKeyring(ModuleType):
        def __init__(self) -> None:
            super().__init__("keyring")

        def get_password(self, service: str, account: str) -> str | None:
            if (service, account) == ("discourselens", "anthropic/university"):
                return secret_value
            return None

    keyring_module = FixtureKeyring()
    monkeypatch.setitem(sys.modules, "keyring", keyring_module)
    reference = "keyring:discourselens:anthropic/university"
    replacement_reference = "keyring:discourselens:anthropic/university-replacement"

    runner = CliRunner()
    created = runner.invoke(
        app,
        [
            "providers",
            "add-credential",
            str(root),
            "anthropic",
            "university",
            "--secret-reference",
            reference,
        ],
    )
    replaced = runner.invoke(
        app,
        [
            "providers",
            "add-credential",
            str(root),
            "anthropic",
            "university",
            "--secret-reference",
            replacement_reference,
        ],
        input=f"{secret_value}\n",
    )
    listed = runner.invoke(app, ["providers", "list-credentials", str(root)])
    rejected = runner.invoke(
        app,
        [
            "providers",
            "add-credential",
            str(root),
            "anthropic",
            "university",
            "--secret-reference",
            secret_value,
        ],
    )

    assert created.exit_code == replaced.exit_code == listed.exit_code == 0
    assert rejected.exit_code != 0
    assert all(
        secret_value not in result.output for result in (created, replaced, listed, rejected)
    )
    rows = json.loads(listed.output)
    assert len(rows) == 1
    assert rows[0]["secret_reference"] == replacement_reference
    assert ReferenceSecretStore().get(reference) == secret_value


def test_cli_stores_no_auth_only_for_explicit_loopback_compatible_endpoint(tmp_path):
    root = tmp_path / "research"
    Project.init(root).close()
    created = CliRunner().invoke(
        app,
        [
            "providers",
            "add-credential",
            str(root),
            "ollama",
            "local",
            "--no-auth",
            "--endpoint",
            "http://localhost:11434/v1",
        ],
    )
    assert created.exit_code == 0, created.output
    project = Project.open(root)
    try:
        credential = project.credential_reference("ollama", "local")
        assert credential["secret_reference"] == "no-auth"
        assert credential["endpoint"] == "http://localhost:11434/v1"
    finally:
        project.close()

    rejected = CliRunner().invoke(
        app,
        [
            "providers",
            "add-credential",
            str(root),
            "ollama",
            "remote",
            "--no-auth",
            "--endpoint",
            "https://remote.example/v1",
        ],
    )
    assert rejected.exit_code != 0
    assert "loopback" in rejected.output


def test_cli_requires_exactly_one_authentication_mode(tmp_path):
    root = tmp_path / "research"
    Project.init(root).close()
    base = ["providers", "add-credential", str(root), "ollama", "local"]

    missing = CliRunner().invoke(app, base)
    conflicting = CliRunner().invoke(
        app,
        [
            *base,
            "--no-auth",
            "--secret-reference",
            "DL_LOCAL_KEY",
            "--endpoint",
            "http://localhost/v1",
        ],
    )

    assert missing.exit_code != 0
    assert conflicting.exit_code != 0
    assert "choose exactly one" in missing.output
    assert "choose exactly one" in conflicting.output


def test_cli_rejects_raw_provider_key_instead_of_persisting_or_echoing_it(tmp_path):
    root = tmp_path / "research"
    Project.init(root).close()
    raw_key = "sk-proj-fixture-raw-key-value"

    result = CliRunner().invoke(
        app,
        [
            "providers",
            "add-credential",
            str(root),
            "openai",
            "university",
            "--secret-reference",
            raw_key,
        ],
    )

    assert result.exit_code != 0
    assert raw_key not in result.output
    assert "never provide the secret value" in result.output
    project = Project.open(root)
    try:
        assert project.list_credential_references() == []
    finally:
        project.close()


def test_model_capability_and_routing_discovery_has_readable_and_machine_formats(tmp_path):
    root = tmp_path / "research"
    project = Project.init(root)
    project.save_credential_reference(
        CredentialReference("openai-compatible", "lab", "DL_ROUTING_KEY")
    )
    project.save_model_preset(
        ModelPreset("review-local", "openai-compatible", "lab", "fixture-model")
    )
    project.save_model_capability_profile(
        ModelCapabilityProfile(
            "openai-compatible",
            "fixture-model",
            ModelCapabilities(
                system_prompt=False,
                generation_parameters=("temperature",),
            ),
            assurance="TESTED",
            source_reference="fixture://model-contract",
        )
    )
    project.save_model_routing_profile(
        "research-route", {"review": "review-local"}, fallback_preset="review-local"
    )
    project.close()
    runner = CliRunner()

    capability_table = runner.invoke(app, ["models", "list-capabilities", str(root)])
    capability_json = runner.invoke(
        app, ["models", "list-capabilities", str(root), "--format", "json"]
    )
    routing_table = runner.invoke(app, ["models", "list-routing", str(root)])
    routing_json = runner.invoke(app, ["models", "list-routing", str(root), "--format", "json"])

    assert all(
        result.exit_code == 0
        for result in (capability_table, capability_json, routing_table, routing_json)
    )
    assert (
        "TESTED" in capability_table.output and '"system_prompt":false' in capability_table.output
    )
    capability = json.loads(capability_json.output)[0]
    assert capability["source_reference"] == "fixture://model-contract"
    assert "review → review-local" in routing_table.output
    assert "review-local" in routing_table.output
    assert json.loads(routing_json.output)[0]["routes_json"] == '{"review": "review-local"}'


def test_cli_previews_model_request_without_printing_prompt_by_default(tmp_path):
    runner = CliRunner()
    project = tmp_path / "research"
    runner.invoke(app, ["project", "init", str(project)])
    runner.invoke(
        app,
        [
            "providers",
            "add-credential",
            str(project),
            "openai-compatible",
            "remote",
            "--secret-env",
            "DL_REMOTE",
            "--endpoint",
            "https://api.example.test/v1",
        ],
    )
    runner.invoke(
        app,
        ["models", "save-preset", str(project), "review", "openai-compatible", "remote", "model-a"],
    )
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("private research excerpt", encoding="utf-8")
    result = runner.invoke(
        app, ["models", "preview-request", str(project), "review", str(prompt_file)]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["prompt_characters"] == len("private research excerpt")
    assert "private research excerpt" not in result.output
    assert "DL_REMOTE" not in result.output


def test_cli_previews_paragraph_embedding_without_text_or_secret_by_default(tmp_path, monkeypatch):
    project_root = tmp_path / "research"
    project = Project.init(project_root)
    _, revision = project.import_text("notes.txt", "First private paragraph.\nSecond paragraph.")
    project.save_credential_reference(
        CredentialReference("gemini-embeddings", "lab", "DL_GEMINI_PREVIEW")
    )
    project.save_model_preset(
        ModelPreset("embed", "gemini-embeddings", "lab", "gemini-embedding-2")
    )
    project.close()
    monkeypatch.setenv("DL_GEMINI_PREVIEW", "never-display-this-secret")

    result = CliRunner().invoke(
        app,
        [
            "vectors",
            "preview-paragraphs",
            str(project_root),
            "embed",
            revision,
            "--offset",
            "1",
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["selected_count"] == 1
    assert payload["network_request_made"] is False
    assert payload["secret_values_resolved"] is False
    assert payload["batches"][0]["request_body"] is None
    assert "First private paragraph." not in result.output
    assert "Second paragraph." not in result.output
    assert "never-display-this-secret" not in result.output
    assert "DL_GEMINI_PREVIEW" not in result.output


def test_cli_hybrid_search_rejects_boolean_vector_components(tmp_path):
    project_root = tmp_path / "research"
    project = Project.init(project_root)
    _, revision = project.import_text("notes.txt", "Apples are green.")
    paragraph_id = project.db.execute(
        "SELECT id FROM paragraphs WHERE revision_id=?", (revision,)
    ).fetchone()[0]
    artifact_id = build_vector_index(
        project, revision, [VectorChunk(paragraph_id, 0, 17, (1.0, 0.0))]
    )
    project.close()

    result = CliRunner().invoke(
        app,
        [
            "vectors",
            "hybrid-search",
            str(project_root),
            revision,
            artifact_id,
            "Apples",
            "--vector-json",
            "[true,false]",
        ],
    )

    assert result.exit_code != 0
    assert "vector_json must be a JSON array of finite numbers" in result.output


def test_cli_hybrid_search_with_preset_reads_query_file_and_freezes_scope(tmp_path, monkeypatch):
    project_root = tmp_path / "research"
    project = Project.init(project_root)
    _, revision = project.import_text("notes.txt", "Apples are green.")
    expected_scope = project.snapshot([revision])
    project.close()

    query_file = tmp_path / "query.txt"
    query_file.write_text("private migration query", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_hybrid_search_with_preset(
        project, preset, scope_id, artifact_ids, query_text, **kwargs
    ):
        del project
        captured.update(
            {
                "preset": preset,
                "scope_id": scope_id,
                "artifact_ids": artifact_ids,
                "query_text": query_text,
                **kwargs,
            }
        )
        return RetrievalSearchResult(
            scope_id,
            "fixture-run-id",
            [
                HybridHit(
                    revision,
                    "paragraph-id",
                    1,
                    1,
                    0.03,
                    artifact_id="artifact-a",
                )
            ],
        )

    monkeypatch.setattr(cli_module, "hybrid_search_with_preset", fake_hybrid_search_with_preset)
    result = CliRunner().invoke(
        app,
        [
            "vectors",
            "hybrid-search-preset",
            str(project_root),
            "embed",
            revision,
            "artifact-a, artifact-b",
            "--query-file",
            str(query_file),
            "--limit",
            "4",
            "--rrf-constant",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["retrieval_run_id"] == "fixture-run-id"
    assert payload["hits"] == [
        {
            "revision_id": revision,
            "unit_id": "paragraph-id",
            "artifact_id": "artifact-a",
            "lexical_rank": 1,
            "vector_rank": 1,
            "fusion_score": 0.03,
        }
    ]
    captured_scope = captured.pop("scope_id", None)
    assert payload["scope_id"] == captured_scope
    assert captured == {
        "preset": "embed",
        "artifact_ids": ["artifact-a", "artifact-b"],
        "query_text": "private migration query",
        "limit": 4,
        "rrf_constant": 20,
    }
    assert isinstance(captured_scope, str) and captured_scope != expected_scope
    with Project.open(project_root, read_only=True) as reopened:
        assert reopened.scope_revisions(captured_scope) == [revision]
    assert "private migration query" not in result.output


def test_model_price_cli_sets_and_lists_explicit_versioned_rates(tmp_path):
    root = tmp_path / "priced-project"
    Project.init(root).close()
    runner = CliRunner()

    saved = runner.invoke(
        app,
        [
            "models",
            "price",
            "set",
            str(root),
            "openai-compatible",
            "fixture-model",
            "--input-per-million-tokens",
            "1.25",
            "--output-per-million-tokens",
            "5",
            "--currency",
            "USD",
            "--source-reference",
            "https://provider.example/pricing",
            "--rate-version",
            "2026-01",
            "--rate-date",
            "2026-01-10",
            "--reasoning-per-million-tokens",
            "12.5",
        ],
    )
    assert saved.exit_code == 0, saved.output
    assert json.loads(saved.output)["rate_version"] == "2026-01"

    listed = runner.invoke(
        app,
        ["models", "price", "list", str(root), "--format", "json"],
    )
    assert listed.exit_code == 0, listed.output
    record = json.loads(listed.output)[0]
    assert record["input_per_million_tokens"] == "1.25"
    assert record["output_per_million_tokens"] == "5"
    assert record["reasoning_per_million_tokens"] == "12.5"
    assert record["currency"] == "USD"
    assert record["source_reference"] == "https://provider.example/pricing"
    assert record["rate_date"] == "2026-01-10"

    table = runner.invoke(app, ["models", "price", "list", str(root)])
    assert table.exit_code == 0, table.output
    assert "fixture-model" in table.output
    assert "2026-01" in table.output
    assert "https://provider.example/pricing" in table.output


def test_model_price_cli_rejects_secret_bearing_source_references(tmp_path):
    root = tmp_path / "secret-price-source"
    Project.init(root).close()

    result = CliRunner().invoke(
        app,
        [
            "models",
            "price",
            "set",
            str(root),
            "openai-compatible",
            "fixture-model",
            "--input-per-million-tokens",
            "1",
            "--output-per-million-tokens",
            "2",
            "--currency",
            "USD",
            "--source-reference",
            "https://provider.example/pricing?api_key=do-not-print-this",
            "--rate-version",
            "v1",
            "--rate-date",
            "2026-01-10",
        ],
    )

    assert result.exit_code == 2
    assert "secret-like values" in result.output
    assert "do-not-print-this" not in result.output
    with Project.open(root, read_only=True) as project:
        assert project.db.execute("SELECT count(*) FROM model_prices").fetchone()[0] == 0


def test_model_costs_cli_lists_persisted_snapshots_without_current_rate_lookup(tmp_path):
    root = tmp_path / "cost-ledger"
    project = Project.init(root)
    first_price = ModelPrice(
        provider_id="openai-compatible",
        model="fixture-model",
        input_per_million_tokens="1",
        output_per_million_tokens="2",
        currency="USD",
        source_reference="https://provider.example/pricing-v1",
        rate_version="v1",
        rate_date="2026-01-01",
    )
    project.save_model_price(first_price)
    priced_execution_id = project.record_model_execution(
        "openai-compatible",
        "fixture",
        "fixture-model",
        None,
        "request-hash",
        "response-hash",
        input_tokens=1_000,
        output_tokens=500,
        model_price=first_price,
    )
    unpriced_execution_id = project.record_model_execution(
        "openai-compatible",
        "fixture",
        "fixture-model",
        None,
        "unpriced-request-hash",
        "unpriced-response-hash",
        input_tokens=1_000,
        output_tokens=500,
    )
    project.save_model_price(
        ModelPrice(
            provider_id="openai-compatible",
            model="fixture-model",
            input_per_million_tokens="100",
            output_per_million_tokens="200",
            currency="USD",
            source_reference="https://provider.example/pricing-v2",
            rate_version="v2",
            rate_date="2026-02-01",
        )
    )
    project.close()

    listed = CliRunner().invoke(
        app,
        ["models", "costs", str(root), "--format", "json"],
    )

    assert listed.exit_code == 0, listed.output
    payload = json.loads(listed.output)
    assert payload["total_count"] == 2
    assert payload["has_more"] is False
    records = {record["execution_id"]: record for record in payload["rows"]}
    assert records[priced_execution_id]["estimated_cost"] == "0.002"
    assert records[priced_execution_id]["estimated_cost_currency"] == "USD"
    assert records[priced_execution_id]["model_price_snapshot"]["rate_version"] == "v1"
    assert (
        records[priced_execution_id]["model_price_snapshot"]["source_reference"]
        == "https://provider.example/pricing-v1"
    )
    assert records[unpriced_execution_id]["estimated_cost"] is None
    assert records[unpriced_execution_id]["model_price_snapshot"] is None
    assert "200" not in json.dumps(records[priced_execution_id]["model_price_snapshot"])

    table = CliRunner().invoke(app, ["models", "costs", str(root)])
    assert table.exit_code == 0, table.output
    assert "0.002" in table.output
    assert "USD" in table.output
    assert "v1" in table.output


@pytest.mark.parametrize("snapshot_kind", ["malformed", "unknown_secret_key"])
def test_model_costs_cli_validates_and_normalizes_price_snapshots(
    tmp_path, snapshot_kind
):
    root = tmp_path / f"corrupt-cost-snapshot-{snapshot_kind}"
    project = Project.init(root)
    price = ModelPrice(
        provider_id="openai-compatible",
        model="fixture-model",
        input_per_million_tokens="1",
        output_per_million_tokens="2",
        currency="USD",
        source_reference="https://provider.example/pricing",
        rate_version="v1",
        rate_date="2026-01-01",
    )
    execution_id = project.record_model_execution(
        "openai-compatible",
        "fixture",
        "fixture-model",
        None,
        "request-hash",
        "response-hash",
        input_tokens=1_000,
        output_tokens=500,
        model_price=price,
    )
    secret_value = "unknown-snapshot-secret-must-not-echo"
    if snapshot_kind == "malformed":
        stored_snapshot = json.dumps({"rate_unit": "unknown", "api_key": secret_value})
    else:
        stored_snapshot = json.dumps({**price.to_snapshot(), "api_key": secret_value})
    project.db.execute(
        "UPDATE model_executions SET model_price_json=? WHERE id=?",
        (stored_snapshot, execution_id),
    )
    project.db.commit()
    project.close()

    result = CliRunner().invoke(app, ["models", "costs", str(root), "--format", "json"])

    assert secret_value not in result.output
    assert "api_key" not in result.output
    if snapshot_kind == "malformed":
        assert result.exit_code == 1
        assert "stored model price snapshot is invalid" in result.output
    else:
        assert result.exit_code == 0, result.output
        record = json.loads(result.output)["rows"][0]
        assert record["model_price_snapshot"] == price.to_snapshot()


def test_model_costs_cli_returns_bounded_pages_and_jsonl_page_metadata(tmp_path):
    root = tmp_path / "paged-cost-ledger"
    project = Project.init(root)
    execution_ids: list[str] = []
    for index in range(3):
        execution_id = project.record_model_execution(
            "openai-compatible",
            "fixture",
            "fixture-model",
            None,
            f"request-hash-{index}",
            f"response-hash-{index}",
        )
        execution_ids.append(execution_id)
        project.db.execute(
            "UPDATE model_executions SET started_at=? WHERE id=?",
            (f"2026-01-01T00:00:0{index}+00:00", execution_id),
        )
    project.db.commit()
    project.close()

    runner = CliRunner()
    first = runner.invoke(
        app,
        ["models", "costs", str(root), "--format", "json", "--limit", "2", "--offset", "0"],
    )
    second = runner.invoke(
        app,
        ["models", "costs", str(root), "--format", "json", "--limit", "2", "--offset", "2"],
    )

    assert first.exit_code == second.exit_code == 0, first.output + second.output
    first_page = json.loads(first.output)
    second_page = json.loads(second.output)
    assert first_page["offset"] == 0
    assert first_page["limit"] == 2
    assert first_page["total_count"] == 3
    assert first_page["has_more"] is True
    assert [row["execution_id"] for row in first_page["rows"]] == execution_ids[:2]
    assert second_page["offset"] == 2
    assert second_page["total_count"] == 3
    assert second_page["has_more"] is False
    assert [row["execution_id"] for row in second_page["rows"]] == execution_ids[2:]
    assert "estimated_cost" in first_page["rows"][0]
    assert "model_price_snapshot" in first_page["rows"][0]

    jsonl = runner.invoke(
        app,
        ["models", "costs", str(root), "--format", "jsonl", "--limit", "1", "--offset", "1"],
    )
    assert jsonl.exit_code == 0, jsonl.output
    lines = [json.loads(line) for line in jsonl.output.splitlines()]
    assert lines[0]["_page"] == {
        "offset": 1,
        "limit": 1,
        "total_count": 3,
        "has_more": True,
    }
    assert lines[1]["execution_id"] == execution_ids[1]
    assert "estimated_cost" in lines[1]
    assert "model_price_snapshot" in lines[1]

    csv_page = runner.invoke(
        app,
        ["models", "costs", str(root), "--format", "csv", "--limit", "1", "--offset", "2"],
    )
    assert csv_page.exit_code == 0, csv_page.output
    csv_rows = list(csv.DictReader(io.StringIO(csv_page.output)))
    assert len(csv_rows) == 1
    assert csv_rows[0]["execution_id"] == execution_ids[2]


@pytest.mark.parametrize("invalid_limit", ["0", "1001"])
def test_model_costs_cli_rejects_out_of_range_limit(tmp_path, invalid_limit):
    root = tmp_path / f"invalid-cost-limit-{invalid_limit}"
    Project.init(root).close()

    result = CliRunner().invoke(
        app,
        ["models", "costs", str(root), "--limit", invalid_limit],
    )

    assert result.exit_code == 2
    assert "--limit" in result.output
    assert "Traceback" not in result.output
