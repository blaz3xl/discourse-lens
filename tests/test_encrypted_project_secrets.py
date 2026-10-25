from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.models import CredentialReference
from discourselens.secrets import ReferenceSecretStore
from discourselens.web import create_app

WEB_TOKEN = "e" * 64


def test_project_secret_is_encrypted_and_survives_reopen(tmp_path, monkeypatch):
    encryption_key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("DISCOURSELENS_CREDENTIALS_ENCRYPTION_KEY", encryption_key)
    root = tmp_path / "project"
    project = Project.init(root)
    secret = "provider-key-for-fixture-only"
    reference = project.save_encrypted_provider_secret(
        "openai-compatible", "researcher", secret
    )
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "researcher", reference, "http://127.0.0.1:8080/v1"
        )
    )
    assert ReferenceSecretStore(project).get(reference) == secret
    stored = project.db.execute(
        "SELECT ciphertext FROM encrypted_secrets WHERE secret_reference=?", (reference,)
    ).fetchone()[0]
    assert secret.encode() not in stored
    project.close()

    reopened = Project.open(root)
    try:
        assert ReferenceSecretStore(reopened).get(reference) == secret
    finally:
        reopened.close()

def test_wrong_deployment_key_cannot_decrypt_project_secret(tmp_path, monkeypatch):
    root = tmp_path / "project"
    monkeypatch.setenv(
        "DISCOURSELENS_CREDENTIALS_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii")
    )
    project = Project.init(root)
    reference = project.save_encrypted_provider_secret("openai", "test", "secret-value")
    project.close()

    monkeypatch.setenv(
        "DISCOURSELENS_CREDENTIALS_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii")
    )
    reopened = Project.open(root)
    try:
        try:
            reopened.resolve_encrypted_secret(reference)
        except ProjectError as exc:
            assert "could not be decrypted" in str(exc)
        else:
            raise AssertionError("a mismatched deployment key decrypted a provider secret")
    finally:
        reopened.close()


def test_web_provider_settings_never_return_the_raw_key(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "DISCOURSELENS_CREDENTIALS_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii")
    )
    root = tmp_path / "project"
    Project.init(root).close()
    with TestClient(create_app(root, WEB_TOKEN)) as client:
        headers = {"Authorization": f"Bearer {WEB_TOKEN}"}
        catalog = client.get("/api/provider-catalog", headers=headers)
        assert catalog.status_code == 200
        assert any(row["provider_id"] == "openai-compatible" for row in catalog.json()["providers"])

        secret = "sk-test-this-is-not-a-real-provider-secret"
        response = client.post(
            "/api/provider-credentials",
            headers=headers,
            json={
                "provider_id": "openai-compatible",
                "alias": "teun-test",
                "secret": secret,
                "endpoint": "http://127.0.0.1:8080/v1",
            },
        )
        assert response.status_code == 201
        assert secret not in response.text

        listed = client.get("/api/provider-credentials", headers=headers)
        assert listed.status_code == 200
        assert listed.json()["items"] == [
            {
                "provider_id": "openai-compatible",
                "alias": "teun-test",
                "endpoint": "http://127.0.0.1:8080/v1",
                "encrypted": True,
                "model_presets": [],
            }
        ]
        assert secret not in listed.text

        with Project.open(root) as project:
            saved = project.credential_reference("openai-compatible", "teun-test")
            assert saved["secret_reference"].startswith("project-secret:")
            assert ReferenceSecretStore(project).get(saved["secret_reference"]) == secret

        deleted = client.delete(
            "/api/provider-credentials/openai-compatible/teun-test", headers=headers
        )
        assert deleted.status_code == 204
        assert client.get("/api/provider-credentials", headers=headers).json()["items"] == []
