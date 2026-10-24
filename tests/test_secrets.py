import sys
import traceback
from types import ModuleType
from urllib.parse import quote

import pytest

from discourselens import Project
from discourselens.errors import DocumentError, OptionalDependencyError, ProjectError
from discourselens.execution import preview_model_preset_request, snapshot_model_preset
from discourselens.models import CredentialReference, ModelPreset
from discourselens.network_policy import is_loopback_endpoint
from discourselens.secrets import (
    NO_AUTH_REFERENCE,
    EnvironmentSecretStore,
    KeyringSecretStore,
    ReferenceSecretStore,
    has_credential_like_endpoint_host,
    validate_secret_reference,
)


def _fixture_value(*parts: str) -> str:
    return "".join(parts)


def test_environment_secret_store_never_persists_value(monkeypatch):
    monkeypatch.setenv("DL_TEST_KEY", "secret")
    assert EnvironmentSecretStore().get("DL_TEST_KEY") == "secret"
    with pytest.raises(ProjectError):
        EnvironmentSecretStore().get("MISSING")


def test_reference_secret_store_keeps_legacy_environment_references_compatible(monkeypatch):
    monkeypatch.setenv("DL_TEST_KEY", "secret")
    assert ReferenceSecretStore().get("DL_TEST_KEY") == "secret"


def test_reference_secret_store_resolves_app_prefixed_provider_alias_exactly(monkeypatch):
    sentinel = "provider-fixture-value-71a9"
    reference = "DISCOURSELENS_PROVIDER_A_API_KEY"
    monkeypatch.setenv(reference, sentinel)

    assert ReferenceSecretStore().get(reference) == sentinel


@pytest.mark.parametrize(
    ("store", "reference"),
    [
        (EnvironmentSecretStore(), "sk-proj-fixture-secret"),
        (ReferenceSecretStore(), "sk-proj-fixture-secret"),
        (EnvironmentSecretStore(), "jina_fixture_secret"),
        (ReferenceSecretStore(), "jina_fixture_secret"),
        (KeyringSecretStore(), "keyring:discourselens:jina_fixture_secret"),
        (ReferenceSecretStore(), "keyring:discourselens:jina_fixture_secret"),
        (KeyringSecretStore(), "keyring:discourselens:sk-ant-api03-fixture-secret"),
        (ReferenceSecretStore(), "keyring:discourselens:sk-ant-api03-fixture-secret"),
        (
            KeyringSecretStore(),
            f"keyring:discourselens:{_fixture_value('xox', 'b-', 'fixture-slack-token')}",
        ),
        (
            ReferenceSecretStore(),
            f"keyring:discourselens:{_fixture_value('xox', 'b-', 'fixture-slack-token')}",
        ),
    ],
)
def test_secret_stores_reject_raw_secret_values_without_echoing_them(store, reference):
    with pytest.raises(ProjectError) as error:
        store.get(reference)

    assert reference not in str(error.value)


def test_keyring_reference_rejects_space_prefixed_raw_token_before_lookup(monkeypatch):
    secret_value = "sk-ant-api03-fixture-secret"
    reference = f"keyring:discourselens: {secret_value}"
    lookups: list[tuple[str, str]] = []
    keyring_module = ModuleType("keyring")

    def get_password(service: str, account: str) -> None:
        lookups.append((service, account))

    keyring_module.get_password = get_password  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", keyring_module)

    with pytest.raises(ProjectError) as error:
        KeyringSecretStore().get(reference)

    assert secret_value not in str(error.value)
    assert lookups == []


def test_missing_keyring_extra_has_actionable_optional_dependency_error(monkeypatch):
    import builtins

    reference = "keyring:discourselens:fixture-account"
    original_import = builtins.__import__

    def block_keyring(name, *args, **kwargs):
        if name == "keyring":
            raise ModuleNotFoundError("keyring intentionally blocked by test", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_keyring)

    with pytest.raises(OptionalDependencyError) as error:
        KeyringSecretStore().get(reference)

    assert "discourselens[secrets]" in str(error.value)
    assert reference not in str(error.value)


@pytest.mark.parametrize("failure_mode", ["missing", "backend_exception"])
def test_keyring_lookup_failure_does_not_echo_reference_or_exception(
    monkeypatch, failure_mode
):
    credential = f"opaque-fake-bearer-{len(failure_mode)}"
    account = f"https://vault.example.invalid/token?api_key={credential}"
    reference = f"keyring:discourselens:{account}"
    keyring_module = ModuleType("keyring")

    def get_password(service: str, account_name: str) -> None:
        if failure_mode == "backend_exception":
            raise RuntimeError(f"backend rejected {account_name}")

    keyring_module.get_password = get_password  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "keyring", keyring_module)

    with pytest.raises(ProjectError) as error:
        KeyringSecretStore().get(reference)

    rendered_error = "".join(traceback.format_exception(error.value))
    assert credential not in rendered_error
    assert account not in rendered_error
    assert "backend rejected" not in rendered_error


@pytest.mark.parametrize(
    "reference",
    [
        "sk-proj-fixture-secret",
        "sk-ant-api03-fixture-secret",
        "jina_fixture_secret",
        "AIzaSy-fixture-secret",
        "gsk_fixture-secret",
        "not a reference",
        "keyring::account",
        "keyring:service:",
        "keyring:discourselens:sk-ant-api03-fixture-secret",
        f"keyring:discourselens:{_fixture_value('xox', 'b-', 'fixture-slack-token')}",
    ],
)
def test_project_rejects_raw_or_malformed_secret_references(tmp_path, reference):
    project = Project.init(tmp_path / "project")
    try:
        with pytest.raises(ProjectError):
            project.save_credential_reference(
                CredentialReference("fixture-provider", "lab", reference)
            )
        assert project.list_credential_references() == []
    finally:
        project.close()


@pytest.mark.parametrize(
    ("provider_id", "secret_reference", "endpoint", "sensitive_value"),
    [
        (
            "fixture-provider",
            "DL_TEST_KEY",
            f"https://api.example/v1/{_fixture_value('sk', '-proj-', 'endpoint-fixture-secret')}",
            _fixture_value("sk", "-proj-", "endpoint-fixture-secret"),
        ),
        (
            "openai-compatible",
            NO_AUTH_REFERENCE,
            f"http://localhost:11434/v1/{_fixture_value('sk', '-proj-', 'local-fixture-secret')}",
            _fixture_value("sk", "-proj-", "local-fixture-secret"),
        ),
        (
            "fixture-provider",
            "DL_TEST_KEY",
            "https://api.example/v1/%2573k%252Dproj%252Dpath%252Dsentinel",
            "path-sentinel",
        ),
        (
            "openai-compatible",
            NO_AUTH_REFERENCE,
            "http://localhost:11434/v1/%2573k%252Dproj%252Dpath%252Dsentinel",
            "path-sentinel",
        ),
        (
            "openai-compatible",
            NO_AUTH_REFERENCE,
            "http://localhost:11434/v1?token=query-fixture-secret",
            "query-fixture-secret",
        ),
        (
            "openai-compatible",
            NO_AUTH_REFERENCE,
            "http://localhost:11434/v1#token=fragment-fixture-secret",
            "fragment-fixture-secret",
        ),
        (
            "openai-compatible",
            NO_AUTH_REFERENCE,
            "http://localhost:11434/v1?",
            "empty-query-marker",
        ),
        (
            "openai-compatible",
            NO_AUTH_REFERENCE,
            "http://localhost:11434/v1#",
            "empty-fragment-marker",
        ),
        (
            "fixture-provider",
            "DL_TEST_KEY",
            "https://sk-proj-endpoint-host-fixture-secret.example/v1",
            "sk-proj-endpoint-host-fixture-secret",
        ),
        (
            "fixture-provider",
            "DL_TEST_KEY",
            "https://api.sk-ant-api03-endpoint-host-fixture-secret.example/v1",
            "sk-ant-api03-endpoint-host-fixture-secret",
        ),
        (
            "fixture-provider",
            "DL_TEST_KEY",
            "https://sk%2Dproj%2Dencoded%2Dhost%2Dfixture%2Dsecret.example/v1",
            "sk-proj-encoded-host-fixture-secret",
        ),
    ],
)
def test_credential_save_rejects_sensitive_endpoint_metadata_without_writes(
    tmp_path, provider_id, secret_reference, endpoint, sensitive_value
):
    project = Project.init(tmp_path / "unsafe-endpoint")
    try:
        before = project.db.total_changes
        with pytest.raises(ProjectError) as error:
            project.save_credential_reference(
                CredentialReference(provider_id, "lab", secret_reference, endpoint)
            )

        assert sensitive_value not in str(error.value)
        assert endpoint not in str(error.value)
        assert str(error.value) == "provider endpoint must not contain credential material"
        assert project.db.total_changes == before
        assert project.list_credential_references() == []
    finally:
        project.close()


def test_credential_save_fails_closed_when_endpoint_path_exceeds_decode_limit(tmp_path):
    encoded_path = "/"
    for _ in range(9):
        encoded_path = quote(encoded_path, safe="")
    endpoint = f"https://api.example/v1/{encoded_path}"
    project = Project.init(tmp_path / "deeply-encoded-endpoint")
    try:
        before = project.db.total_changes
        with pytest.raises(ProjectError) as error:
            project.save_credential_reference(
                CredentialReference("fixture-provider", "lab", "DL_TEST_KEY", endpoint)
            )

        assert endpoint not in str(error.value)
        assert str(error.value) == "provider endpoint must not contain credential material"
        assert project.db.total_changes == before
        assert project.list_credential_references() == []
    finally:
        project.close()


def test_preview_and_snapshot_reject_legacy_credential_like_endpoint_host(tmp_path):
    sentinel = "sk-proj-preview-host-fixture-secret"
    project = Project.init(tmp_path / "legacy-endpoint-host")
    try:
        project.save_credential_reference(
            CredentialReference(
                "openai-compatible", "lab", "DL_TEST_KEY", "https://provider.example/v1"
            )
        )
        project.save_model_preset(
            ModelPreset("review", "openai-compatible", "lab", "fixture-model")
        )
        project.db.execute(
            "UPDATE provider_credentials SET endpoint=? WHERE provider_id=? AND credential_alias=?",
            (f"https://{sentinel}.example/v1", "openai-compatible", "lab"),
        )
        project.db.commit()

        for invoke in (
            lambda: snapshot_model_preset(project, "review"),
            lambda: preview_model_preset_request(project, "review", "fixture prompt"),
        ):
            with pytest.raises(DocumentError) as error:
                invoke()
            assert sentinel not in str(error.value)
    finally:
        project.close()


def test_analysis_run_config_rejects_credential_like_endpoint_host_without_echoing(tmp_path):
    sentinel = "sk-proj-run-snapshot-host-fixture-secret"
    project = Project.init(tmp_path / "unsafe-run-endpoint")
    try:
        _, revision_id = project.import_text("source.txt", "A local fixture.")
        scope_id = project.snapshot([revision_id])
        before = project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0]

        with pytest.raises(DocumentError) as error:
            project.create_run(
                scope_id,
                {"framework_preset_snapshots": {"review": {"endpoint": f"https://{sentinel}.example/v1"}}},
            )

        assert sentinel not in str(error.value)
        assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == before
    finally:
        project.close()


def test_credential_like_endpoint_host_detection_preserves_safe_host_identity():
    assert not has_credential_like_endpoint_host("https://api.example.com/v1")
    assert not has_credential_like_endpoint_host("https://localhost:11434/v1")
    assert has_credential_like_endpoint_host("https://sk-proj-host-fixture-secret.example/v1")


def test_safe_no_auth_loopback_endpoint_remains_supported(tmp_path):
    project = Project.init(tmp_path / "safe-no-auth-endpoint")
    try:
        credential_id = project.save_credential_reference(
            CredentialReference(
                "openai-compatible",
                "local",
                NO_AUTH_REFERENCE,
                "http://localhost:11434/v1",
            )
        )

        assert credential_id
        assert project.credential_reference("openai-compatible", "local")[
            "secret_reference"
        ] == NO_AUTH_REFERENCE
    finally:
        project.close()


@pytest.mark.parametrize(
    "reference",
    [
        "DL_TEST_KEY",
        "DISCOURSELENS_PROVIDER_A_API_KEY",
        "JINA_API_KEY",
        "keyring:discourselens:lab",
    ],
)
def test_secret_reference_contract_accepts_supported_reference_forms(reference):
    validate_secret_reference(reference)


@pytest.mark.parametrize("reference", [None, 1, [], {}])
def test_secret_reference_contract_rejects_non_text_inputs(reference):
    with pytest.raises(ProjectError, match="secret reference"):
        validate_secret_reference(reference)


@pytest.mark.parametrize("reference", [None, 1])
def test_reference_secret_store_normalizes_non_text_inputs(reference):
    with pytest.raises(ProjectError, match="secret reference"):
        ReferenceSecretStore().get(reference)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1/v1",
        "https://[::1]:443/v1",
    ],
)
def test_loopback_endpoint_policy_accepts_explicit_loopback_urls(endpoint):
    assert is_loopback_endpoint(endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:not-a-port/v1",
        "http://localhost@remote.example/v1",
        "http://192.168.1.5:11434/v1",
        "https://remote.example/v1",
        "file:///localhost/model",
    ],
)
def test_loopback_endpoint_policy_rejects_malformed_or_non_loopback_urls(endpoint):
    assert not is_loopback_endpoint(endpoint)
