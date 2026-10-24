from __future__ import annotations

import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app
from discourselens.models import CredentialReference
from discourselens.secrets import is_credential_like_alias


def test_cli_rejects_secret_like_credential_alias_without_persisting_or_echoing_it(tmp_path):
    root = tmp_path / "credential-alias"
    Project.init(root).close()
    suspicious_alias = "sk-proj-cli-alias-fixture-secret"
    assert is_credential_like_alias(suspicious_alias)

    result = CliRunner().invoke(
        app,
        [
            "providers",
            "add-credential",
            str(root),
            "openai",
            suspicious_alias,
            "--secret-reference",
            "DL_OPENAI_KEY",
        ],
    )

    assert result.exit_code != 0
    assert suspicious_alias not in result.output
    project = Project.open(root)
    try:
        assert project.list_credential_references() == []
    finally:
        project.close()


def test_cli_rejects_credential_like_endpoint_hostname_without_echoing_it(tmp_path):
    root = tmp_path / "credential-hostname"
    Project.init(root).close()
    sentinel = "sk-proj-cli-host-fixture-secret"
    result = CliRunner().invoke(
        app,
        [
            "providers",
            "add-credential",
            str(root),
            "openai-compatible",
            "university",
            "--secret-reference",
            "DL_OPENAI_KEY",
            "--endpoint",
            f"https://{sentinel}.example/v1",
        ],
    )

    assert result.exit_code != 0
    assert sentinel not in result.output
    project = Project.open(root)
    try:
        assert project.list_credential_references() == []
    finally:
        project.close()


def test_cli_credential_and_preset_listings_show_labels_without_resolving_key_material(
    tmp_path, monkeypatch
):
    root = tmp_path / "credential-listing"
    Project.init(root).close()
    resolved_key = "sk-proj-environment-value-fixture-secret"
    monkeypatch.setenv("DL_OPENAI_UNIVERSITY_KEY", resolved_key)
    runner = CliRunner()

    added = runner.invoke(
        app,
        [
            "providers",
            "add-credential",
            str(root),
            "openai",
            "university",
            "--secret-reference",
            "DL_OPENAI_UNIVERSITY_KEY",
        ],
    )
    saved = runner.invoke(
        app,
        [
            "models",
            "save-preset",
            str(root),
            "analysis",
            "openai",
            "university",
            "fixture-model",
        ],
    )
    credentials = runner.invoke(app, ["providers", "list-credentials", str(root)])
    presets = runner.invoke(app, ["models", "list-presets", str(root), "--format", "json"])

    assert added.exit_code == saved.exit_code == credentials.exit_code == presets.exit_code == 0
    assert "university" in credentials.output
    assert "DL_OPENAI_UNIVERSITY_KEY" in credentials.output
    assert "university" in presets.output
    assert resolved_key not in added.output
    assert resolved_key not in saved.output
    assert resolved_key not in credentials.output
    assert resolved_key not in presets.output
    credential_rows = json.loads(credentials.output)
    preset_rows = json.loads(presets.output)
    assert credential_rows[0]["credential_alias"] == "university"
    assert preset_rows[0]["credential_alias"] == "university"


def test_cli_rejects_secret_like_camel_case_credential_alias_parameter_before_persistence(
    tmp_path,
):
    root = tmp_path / "camel-case-credential-alias"
    project = Project.init(root)
    project.save_credential_reference(
        CredentialReference("openai", "university", "DL_OPENAI_UNIVERSITY_KEY")
    )
    project.close()

    suspicious_alias = "sk-proj-camel-case-parameter-fixture-secret"
    assert is_credential_like_alias(suspicious_alias)
    parameters = json.dumps({"credentialAlias": suspicious_alias})
    runner = CliRunner()
    saved = runner.invoke(
        app,
        [
            "models",
            "save-preset",
            str(root),
            "analysis",
            "openai",
            "university",
            "fixture-model",
            "--parameters-json",
            parameters,
        ],
    )
    listed = runner.invoke(app, ["models", "list-presets", str(root), "--format", "json"])

    assert saved.exit_code != 0
    assert suspicious_alias not in saved.output
    assert listed.exit_code == 0
    assert json.loads(listed.output) == []
    assert suspicious_alias not in listed.output
