from __future__ import annotations

import builtins
import json
import sys
from types import ModuleType

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app
from discourselens.secrets import ReferenceSecretStore


class FixtureKeyring(ModuleType):
    def __init__(self) -> None:
        super().__init__("keyring")
        self.entries: dict[tuple[str, str], str] = {}
        self.set_calls: list[tuple[str, str, str]] = []
        self.delete_calls: list[tuple[str, str]] = []

    def get_password(self, service: str, account: str) -> str | None:
        return self.entries.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.set_calls.append((service, account, value))
        self.entries[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        self.delete_calls.append((service, account))
        if (service, account) not in self.entries:
            raise KeyError("entry does not exist")
        del self.entries[(service, account)]


def test_store_key_hides_secret_and_saved_reference_resolves_without_project_storage(
    tmp_path, monkeypatch
):
    secret = "provider-fixture-secret-c03a"
    environment_secret = "provider-fixture-environment-secret"
    fixture = FixtureKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fixture)
    monkeypatch.setenv("OPENAI_API_KEY", environment_secret)
    runner = CliRunner()

    stored = runner.invoke(
        app,
        ["providers", "store-key", "openai", "university"],
        input=f"{secret}\n",
    )

    assert stored.exit_code == 0, stored.output
    assert secret not in stored.output
    assert environment_secret not in stored.output
    metadata = json.loads(stored.output.splitlines()[-1])
    reference = "keyring:discourselens:openai/university"
    assert metadata == {
        "provider_id": "openai",
        "credential_alias": "university",
        "secret_reference": reference,
        "stored": True,
    }
    assert fixture.entries[("discourselens", "openai/university")] == secret

    root = tmp_path / "research"
    Project.init(root).close()
    added = runner.invoke(
        app,
        [
            "providers",
            "add-credential",
            str(root),
            "openai",
            "university",
            "--secret-reference",
            reference,
        ],
    )

    assert added.exit_code == 0, added.output
    assert secret not in added.output
    assert ReferenceSecretStore().get(reference) == secret
    assert all(secret.encode() not in path.read_bytes() for path in root.rglob("*") if path.is_file())


def test_store_key_refuses_existing_entry_without_overwriting_or_echoing(monkeypatch):
    existing = "existing-fixture-secret"
    attempted = "replacement-fixture-secret"
    fixture = FixtureKeyring()
    fixture.entries[("discourselens", "anthropic/lab")] = existing
    monkeypatch.setitem(sys.modules, "keyring", fixture)

    result = CliRunner().invoke(
        app,
        ["providers", "store-key", "anthropic", "lab"],
        input=f"{attempted}\n",
    )

    assert result.exit_code != 0
    assert "already exists" in result.output
    assert existing not in result.output
    assert attempted not in result.output
    assert fixture.entries[("discourselens", "anthropic/lab")] == existing
    assert fixture.set_calls == []


def test_store_key_reports_missing_optional_keyring_extra_safely(monkeypatch):
    original_import = builtins.__import__

    def block_keyring(name, *args, **kwargs):
        if name == "keyring":
            raise ModuleNotFoundError("test backend unavailable", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "keyring", raising=False)
    monkeypatch.setattr(builtins, "__import__", block_keyring)

    result = CliRunner().invoke(app, ["providers", "store-key", "openai", "lab"])

    assert result.exit_code != 0
    assert "discourselens[secrets]" in result.output
    assert "test backend unavailable" not in result.output


def test_store_key_sanitizes_unavailable_backend_errors(monkeypatch):
    class UnavailableKeyring(FixtureKeyring):
        def get_password(self, service: str, account: str) -> str | None:
            raise RuntimeError("backend credential fixture should not escape")

    monkeypatch.setitem(sys.modules, "keyring", UnavailableKeyring())

    result = CliRunner().invoke(app, ["providers", "store-key", "openai", "lab"])

    assert result.exit_code != 0
    assert "configure a working keyring backend" in result.output
    assert "backend credential fixture" not in result.output


def test_store_key_rejects_secret_as_extra_argument_without_echoing_it(monkeypatch):
    fixture = FixtureKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fixture)
    secret = "provider-fixture-argument-secret"

    result = CliRunner().invoke(
        app,
        ["providers", "store-key", "openai", "lab", secret],
    )

    assert result.exit_code != 0
    assert "unexpected argument" in result.output
    assert secret not in result.output
    assert fixture.set_calls == []


def test_remove_key_confirms_then_deletes_only_derived_entry_without_echoing_secret(monkeypatch):
    secret = "existing-fixture-secret"
    fixture = FixtureKeyring()
    fixture.entries[("discourselens", "openai/university")] = secret
    fixture.entries[("another-service", "openai/university")] = "unrelated-fixture-secret"
    monkeypatch.setitem(sys.modules, "keyring", fixture)

    result = CliRunner().invoke(
        app,
        ["providers", "remove-key", "openai", "university"],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert "Remove keyring entry" in result.output
    assert secret not in result.output
    assert "unrelated-fixture-secret" not in result.output
    assert json.loads(result.output.splitlines()[-1]) == {
        "provider_id": "openai",
        "credential_alias": "university",
        "removed": True,
        "status": "removed",
    }
    assert fixture.entries == {("another-service", "openai/university"): "unrelated-fixture-secret"}
    assert fixture.delete_calls == [("discourselens", "openai/university")]


def test_remove_key_cancellation_keeps_existing_entry(monkeypatch):
    secret = "existing-fixture-secret"
    fixture = FixtureKeyring()
    fixture.entries[("discourselens", "anthropic/lab")] = secret
    monkeypatch.setitem(sys.modules, "keyring", fixture)

    result = CliRunner().invoke(
        app,
        ["providers", "remove-key", "anthropic", "lab"],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert "cancelled" in result.output
    assert secret not in result.output
    assert fixture.entries[("discourselens", "anthropic/lab")] == secret
    assert fixture.delete_calls == []


def test_remove_key_yes_skips_prompt_and_deletes(monkeypatch):
    secret = "existing-fixture-secret"
    fixture = FixtureKeyring()
    fixture.entries[("discourselens", "openai/lab")] = secret
    monkeypatch.setitem(sys.modules, "keyring", fixture)

    result = CliRunner().invoke(app, ["providers", "remove-key", "openai", "lab", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Remove keyring entry" not in result.output
    assert secret not in result.output
    assert fixture.entries == {}
    assert fixture.delete_calls == [("discourselens", "openai/lab")]


def test_remove_key_reports_missing_entry_without_prompt_or_deletion(monkeypatch):
    fixture = FixtureKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fixture)

    result = CliRunner().invoke(app, ["providers", "remove-key", "openai", "lab"])

    assert result.exit_code == 0, result.output
    assert "not_found" in result.output
    assert "Remove keyring entry" not in result.output
    assert fixture.delete_calls == []


def test_remove_key_rejects_secret_or_reference_as_an_extra_argument(monkeypatch):
    fixture = FixtureKeyring()
    fixture.entries[("discourselens", "openai/lab")] = "existing-fixture-secret"
    monkeypatch.setitem(sys.modules, "keyring", fixture)
    runner = CliRunner()

    for forbidden in (
        "sk-proj-cli-argument-fixture-secret",
        "keyring:discourselens:openai/lab",
    ):
        result = runner.invoke(
            app,
            ["providers", "remove-key", "openai", "lab", forbidden],
        )

        assert result.exit_code != 0
        assert "unexpected argument" in result.output
        assert forbidden not in result.output
        assert fixture.entries[("discourselens", "openai/lab")] == "existing-fixture-secret"

    assert fixture.delete_calls == []


def test_remove_key_reports_missing_optional_extra_safely(monkeypatch):
    original_import = builtins.__import__

    def block_keyring(name, *args, **kwargs):
        if name == "keyring":
            raise ModuleNotFoundError("keyring fixture unavailable", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "keyring", raising=False)
    monkeypatch.setattr(builtins, "__import__", block_keyring)

    result = CliRunner().invoke(app, ["providers", "remove-key", "openai", "lab"])

    assert result.exit_code != 0
    assert "discourselens[secrets]" in result.output
    assert "keyring fixture unavailable" not in result.output


def test_remove_key_sanitizes_backend_failures(monkeypatch):
    class BrokenKeyring(FixtureKeyring):
        def get_password(self, service: str, account: str) -> str | None:
            raise RuntimeError("backend echoed fixture-secret")

    monkeypatch.setitem(sys.modules, "keyring", BrokenKeyring())

    result = CliRunner().invoke(app, ["providers", "remove-key", "openai", "lab"])

    assert result.exit_code != 0
    assert "configure a working keyring backend" in result.output
    assert "backend echoed fixture-secret" not in result.output


def test_remove_key_sanitizes_delete_errors(monkeypatch):
    secret = "existing-fixture-secret"

    class DeleteFailureKeyring(FixtureKeyring):
        def delete_password(self, service: str, account: str) -> None:
            raise RuntimeError("backend delete failed for fixture-secret")

    fixture = DeleteFailureKeyring()
    fixture.entries[("discourselens", "openai/lab")] = secret
    monkeypatch.setitem(sys.modules, "keyring", fixture)

    result = CliRunner().invoke(
        app,
        ["providers", "remove-key", "openai", "lab", "--yes"],
    )

    assert result.exit_code != 0
    assert "could not be removed" in result.output
    assert secret not in result.output
    assert "fixture-secret" not in result.output
    assert fixture.entries[("discourselens", "openai/lab")] == secret


def test_removed_key_can_be_stored_again(monkeypatch):
    old_secret = "old-fixture-secret"
    new_secret = "new-fixture-secret"
    fixture = FixtureKeyring()
    fixture.entries[("discourselens", "openai/lab")] = old_secret
    monkeypatch.setitem(sys.modules, "keyring", fixture)
    runner = CliRunner()

    removed = runner.invoke(app, ["providers", "remove-key", "openai", "lab", "--yes"])
    stored = runner.invoke(
        app,
        ["providers", "store-key", "openai", "lab"],
        input=f"{new_secret}\n",
    )

    assert removed.exit_code == stored.exit_code == 0
    assert old_secret not in removed.output
    assert new_secret not in stored.output
    assert fixture.entries[("discourselens", "openai/lab")] == new_secret
