from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app
from discourselens.errors import DocumentError, ProjectError
from discourselens.models import CredentialReference, ModelPreset
from discourselens.secrets import (
    EnvironmentSecretStore,
    is_credential_like_alias,
    validate_credential_alias,
    validate_secret_reference,
)


def _fixture_value(*parts: str) -> str:
    return "".join(parts)


def _environment_reference() -> str:
    prefix = "DL"
    purpose = "TEST"
    field = "KEY"
    return f"{prefix}_{purpose}_{field}"


@pytest.mark.parametrize("alias", ["university", "lab", "project-grant", "local.v2"])
def test_short_human_credential_aliases_are_valid(alias: str) -> None:
    assert not is_credential_like_alias(alias)
    validate_credential_alias(alias)


@pytest.mark.parametrize(
    "value",
    [
        "sk-proj-example-openai-key",
        "sk-ant-api03-example-anthropic-key",
        "gsk_example-groq-key-123",
        "xai-example-key-123",
        "sk-or-v1-example-openrouter-key",
        "jina_example-jina-key",
        "AIzaSyExampleGoogleKey12345678901234567890",
        "ya29.example-google-oauth-token",
        _fixture_value("AK", "IA", "IOSFODNN7EXAMPLE"),
        _fixture_value("gh", "p_", "examplegithubtoken123456789"),
        _fixture_value("github", "_pat_", "examplegithubtoken123456"),
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.signature00000000",
        "https://api.example.test/v1?api_key=example-secret",
        "Bearer opaque-example-secret",
        "api_key=opaque-example-secret",
        _fixture_value("xox", "b-", "fixture-slack-token"),
        "Wv9K8mQ2rT6xB4nP7cL1dF5hJ3sA0uYz",
    ],
)
def test_credential_like_aliases_are_detected_and_rejected_without_echo(value: str) -> None:
    assert is_credential_like_alias(value)

    with pytest.raises(ProjectError) as error:
        validate_credential_alias(value)

    assert value not in str(error.value)
    assert str(error.value) == "credential alias must be a short human-readable label"


@pytest.mark.parametrize(
    "reference",
    [
        "keyring:discourselens:anthropic/university",
        "keyring:discourselens:lab",
        "OPENAI_API_KEY",
        "JINA_API_KEY",
    ],
)
def test_safe_keyring_account_paths_and_environment_references_remain_supported(
    reference: str,
) -> None:
    validate_secret_reference(reference)


@pytest.mark.parametrize(
    "reference",
    [
        "keyring:service:account?api_key=opaque-secret",
        "keyring:service:https://user:password@vault.example/token",
        "keyring:service:Bearer opaque-secret",
        "keyring:service:sk-ant-api03-example-secret",
        f"keyring:service:{_fixture_value('xox', 'b-', 'fixture-slack-token')}",
        "keyring:service//account",
        "keyring:service:../account",
        "keyring:service:/account",
        "keyring:service:account/",
    ],
)
def test_unsafe_keyring_references_fail_without_echo(reference: str) -> None:
    with pytest.raises(ProjectError) as error:
        validate_secret_reference(reference)

    assert reference not in str(error.value)


def test_opaque_environment_reference_is_rejected_but_named_environment_references_work():
    opaque_value = _fixture_value("Wv9K8mQ2rT6xB4nP7cL1dF5hJ3sA0uYz")
    with pytest.raises(ProjectError) as error:
        validate_secret_reference(opaque_value)

    assert opaque_value not in str(error.value)
    validate_secret_reference("OPENAI_API_KEY")


def test_environment_reference_and_standard_alias_flow_still_works(tmp_path, monkeypatch):
    reference = _environment_reference()
    monkeypatch.setenv(reference, "fixture-secret-value")
    project = Project.init(tmp_path / "credential-flow")
    try:
        credential_id = project.save_credential_reference(
            CredentialReference("fixture-provider", "project-grant", reference)
        )
        preset_id = project.save_model_preset(
            ModelPreset("review", "fixture-provider", "project-grant", "fixture-model")
        )

        assert credential_id
        assert preset_id
        assert project.credential_reference("fixture-provider", "project-grant")[
            "secret_reference"
        ] == reference
        assert project.list_credential_references()[0]["credential_alias"] == "project-grant"
        assert project.model_preset("review")["credential_alias"] == "project-grant"
        assert project.list_model_presets()[0]["credential_alias"] == "project-grant"
        assert EnvironmentSecretStore().get(reference) == "fixture-secret-value"
    finally:
        project.close()


def test_credential_alias_rejections_do_not_write_project_database(tmp_path):
    project = Project.init(tmp_path / "reject-before-write")
    try:
        before = project.db.total_changes
        invalid = "sk-proj-example-openai-key"

        with pytest.raises(ProjectError):
            project.save_credential_reference(
                cast(
                    CredentialReference,
                    SimpleNamespace(
                        provider_id="fixture-provider",
                        alias=invalid,
                        secret_reference=_environment_reference(),
                        endpoint=None,
                    ),
                )
            )
        assert project.db.total_changes == before
        assert project.list_credential_references() == []

        with pytest.raises(ProjectError):
            project.save_model_preset(
                cast(
                    ModelPreset,
                    SimpleNamespace(
                        name="review",
                        provider_id="fixture-provider",
                        credential_alias=invalid,
                        model="fixture-model",
                        parameters=None,
                    ),
                )
            )
        assert project.db.total_changes == before
        assert project.db.execute("SELECT COUNT(*) FROM model_presets").fetchone()[0] == 0

        with pytest.raises(DocumentError):
            project.save_analysis_profile(
                "unsafe",
                {
                    "frameworks": ["fairclough-text"],
                    "routing": ({"credential_alias": invalid},),
                },
            )
        assert project.db.total_changes == before
        assert project.db.execute("SELECT COUNT(*) FROM analysis_profiles").fetchone()[0] == 0

        _, revision_id = project.import_text("fixture.txt", "A fixture document.")
        scope_id = project.snapshot([revision_id])
        before_run = project.db.total_changes
        with pytest.raises(DocumentError):
            project.create_run(
                scope_id,
                {"routing": {"stages": ({"credential_alias": invalid},)}},
            )
        assert project.db.total_changes == before_run
        assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0

        before_ledger = project.db.total_changes
        with pytest.raises(DocumentError):
            project.record_model_execution(
                "fixture-provider", invalid, "fixture-model", None, "request", None
            )
        with pytest.raises(ProjectError):
            project.record_reranking(
                "missing-retrieval-run",
                provider_id="fixture-provider",
                credential_alias=invalid,
                model="fixture-model",
                query="fixture query",
                top_n=1,
                effective_top_n=1,
                candidate_count=1,
                ranked_hits=[("missing-hit", 1, 0.5)],
            )
        assert project.db.total_changes == before_ledger
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
        assert project.db.execute("SELECT COUNT(*) FROM reranking_runs").fetchone()[0] == 0
    finally:
        project.close()


@pytest.mark.parametrize(
    ("metadata_key", "metadata_value"),
    [
        ("credentialAlias", "sk-proj-example-openai-key"),
        ("secretReference", "DL_TEST_KEY"),
        ("clientSecret", "fixture-value"),
        ("refreshToken", "fixture-value"),
        ("privateKey", "fixture-value"),
    ],
)
def test_camel_case_secret_metadata_is_rejected_before_profile_or_run_persistence(
    tmp_path, metadata_key: str, metadata_value: str
):
    project = Project.init(tmp_path / f"camel-case-{metadata_key}")
    try:
        _, revision_id = project.import_text("fixture.txt", "A fixture document.")
        scope_id = project.snapshot([revision_id])
        before = project.db.total_changes

        with pytest.raises(DocumentError):
            project.save_analysis_profile(
                "unsafe",
                {"frameworks": ["fairclough-text"], metadata_key: metadata_value},
            )
        assert project.db.total_changes == before
        assert project.db.execute("SELECT COUNT(*) FROM analysis_profiles").fetchone()[0] == 0

        with pytest.raises(DocumentError):
            project.create_run(scope_id, {metadata_key: metadata_value})
        assert project.db.total_changes == before
        assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0

        with pytest.raises(DocumentError) as preset_error:
            project.save_model_preset(
                ModelPreset(
                    "unsafe-preset",
                    "fixture-provider",
                    "lab",
                    "fixture-model",
                    parameters={metadata_key: metadata_value},
                )
            )
        assert metadata_value not in str(preset_error.value)
        assert project.db.total_changes == before
        assert project.db.execute("SELECT COUNT(*) FROM model_presets").fetchone()[0] == 0
    finally:
        project.close()


@pytest.mark.parametrize(
    "legacy_key",
    [
        "credentialAlias",
        "clientSecret",
        "client_secret",
        "refreshToken",
        "refresh_token",
        "privateKey",
        "private_key",
    ],
)
def test_preset_reads_and_cli_listing_reject_sensitive_legacy_metadata(tmp_path, legacy_key):
    root = tmp_path / "legacy-camel-preset"
    project = Project.init(root)
    project.save_credential_reference(
        CredentialReference("fixture-provider", "lab", _environment_reference())
    )
    project.save_model_preset(
        ModelPreset("review", "fixture-provider", "lab", "fixture-model")
    )
    legacy_value = (
        _fixture_value("gh", "p_", "legacyReadToken123456")
        if legacy_key == "credentialAlias"
        else "legacy-sensitive-marker"
    )
    project.db.execute(
        "UPDATE model_presets SET parameters_json=? WHERE name=?",
        (json.dumps({legacy_key: legacy_value}), "review"),
    )
    project.db.commit()

    with pytest.raises(DocumentError) as read_error:
        project.model_preset("review")
    assert legacy_value not in str(read_error.value)
    with pytest.raises(DocumentError) as list_error:
        project.list_model_presets()
    assert legacy_value not in str(list_error.value)
    project.close()

    result = CliRunner().invoke(
        app, ["models", "list-presets", str(root), "--format", "json"]
    )
    assert result.exit_code != 0
    assert "stored model preset metadata is unsafe" in result.output
    assert legacy_value not in result.output


@pytest.mark.parametrize(
    ("alias", "secret_reference"),
    [
        ("sk-proj-legacy-secret", "DL_TEST_CREDENTIAL"),
        ("lab", "sk-ant-api03-legacy-secret"),
    ],
)
def test_credential_listing_and_read_reject_contaminated_legacy_rows(
    tmp_path, alias: str, secret_reference: str
):
    project = Project.init(tmp_path / f"legacy-{len(alias)}")
    try:
        project.save_credential_reference(
            CredentialReference("fixture-provider", "lab", _environment_reference())
        )
        project.db.execute(
            "UPDATE provider_credentials SET credential_alias=?,secret_reference=? "
            "WHERE provider_id=?",
            (alias, secret_reference, "fixture-provider"),
        )
        project.db.commit()

        with pytest.raises(DocumentError) as listing_error:
            project.list_credential_references()
        assert alias not in str(listing_error.value)
        assert secret_reference not in str(listing_error.value)

        with pytest.raises(ProjectError) as read_error:
            project.credential_reference("fixture-provider", "lab")
        assert alias not in str(read_error.value)
        assert secret_reference not in str(read_error.value)
    finally:
        project.close()


def test_model_execution_ledger_rejects_contaminated_legacy_alias(tmp_path):
    project = Project.init(tmp_path / "legacy-model-ledger")
    try:
        _, revision_id = project.import_text("fixture.txt", "A fixture document.")
        scope_id = project.snapshot([revision_id])
        run_id = project.create_run(scope_id, {"profile": "fixture"})
        execution_id = project.record_model_execution(
            "fixture-provider", "lab", "fixture-model", None, "request", None,
            analysis_run_id=run_id,
        )
        secret_alias = _fixture_value("gh", "p_", "legacyGitHubSecret123456")
        project.db.execute(
            "UPDATE model_executions SET credential_alias=? WHERE id=?",
            (secret_alias, execution_id),
        )
        project.db.commit()

        with pytest.raises(DocumentError) as error:
            project.analysis_run_summary(run_id)
        assert secret_alias not in str(error.value)
    finally:
        project.close()
