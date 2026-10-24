from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.execution import execute_model_preset
from discourselens.models import CredentialReference, ModelPreset
from discourselens.secrets import ReferenceSecretStore


def test_saved_openai_compatible_preset_uses_alias_and_records_safe_usage_ledger(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = "compatible-integration-prompt-marker-4c91"
    system_prompt = "compatible-system-prompt-marker-2026"
    completion = "compatible-completion-marker-78a2"
    selected_secret = "selected-compatible-secret-marker-0e32"
    other_secret = "other-compatible-secret-must-not-be-used"
    selected_reference = "DL_COMPATIBLE_SELECTED_KEY"
    other_reference = "DL_COMPATIBLE_OTHER_KEY"
    monkeypatch.setenv(selected_reference, selected_secret)
    monkeypatch.setenv(other_reference, other_secret)
    for variable in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    received: list[tuple[str, str | None, dict[str, object]]] = []

    class _CaptureHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = cast(dict[str, object], json.loads(self.rfile.read(content_length)))
            received.append((self.path, self.headers.get("Authorization"), body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [
                            {"message": {"content": completion}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 19, "completion_tokens": 7},
                    }
                ).encode("utf-8")
            )

        def log_message(self, format: str, *args: object) -> None:
            return

    http_server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    server_thread = Thread(target=http_server.serve_forever, daemon=True)
    server_thread.start()
    endpoint = f"http://127.0.0.1:{http_server.server_port}/compatible/v1"
    project = Project.init(tmp_path / "compatible-saved-execution")
    resolved_references: list[str] = []
    original_get = ReferenceSecretStore.get

    def record_secret_lookup(store: ReferenceSecretStore, reference: str) -> str:
        resolved_references.append(reference)
        return original_get(store, reference)

    monkeypatch.setattr(ReferenceSecretStore, "get", record_secret_lookup)
    try:
        project.save_credential_reference(
            CredentialReference("openai-compatible", "selected", selected_reference, endpoint)
        )
        project.save_credential_reference(
            CredentialReference("openai-compatible", "other", other_reference, endpoint)
        )
        project.save_model_preset(
            ModelPreset(
                "compatible-review",
                "openai-compatible",
                "selected",
                "fixture-compatible-model",
                parameters={"max_tokens": 31, "temperature": 0.2},
            )
        )

        result = execute_model_preset(
            project,
            "compatible-review",
            prompt,
            system_prompt=system_prompt,
        )

        assert result.response.text == completion
        assert result.response.provider == "openai-compatible"
        assert result.response.model == "fixture-compatible-model"
        assert result.response.usage is not None
        assert result.response.usage.input_tokens == 19
        assert result.response.usage.output_tokens == 7
        assert resolved_references == [selected_reference]
        assert len(received) == 1
        request_path, authorization, body = received[0]
        assert request_path == "/compatible/v1/chat/completions"
        assert authorization == f"Bearer {selected_secret}"
        assert body == {
            "model": "fixture-compatible-model",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 31,
            "temperature": 0.2,
        }

        ledger = project.db.execute("SELECT * FROM model_executions").fetchone()
        assert ledger is not None
        assert ledger["id"] == result.execution_id
        assert ledger["provider_id"] == "openai-compatible"
        assert ledger["credential_alias"] == "selected"
        assert ledger["endpoint"] == endpoint
        assert ledger["model"] == "fixture-compatible-model"
        assert ledger["input_tokens"] == 19
        assert ledger["output_tokens"] == 7
        assert ledger["error_type"] is None
        assert ledger["request_hash"] == hashlib.sha256(
            json.dumps(
                {"user_prompt": prompt, "system_prompt": system_prompt},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        assert ledger["response_hash"] == hashlib.sha256(completion.encode()).hexdigest()

        serialized_ledger = json.dumps(dict(ledger), sort_keys=True)
        for private_value in (
            prompt,
            system_prompt,
            completion,
            selected_secret,
            other_secret,
            selected_reference,
            other_reference,
        ):
            assert private_value not in serialized_ledger
    finally:
        project.close()
        http_server.shutdown()
        server_thread.join(timeout=5)
        http_server.server_close()


def test_offline_saved_openai_compatible_preset_blocks_before_secret_transport_or_ledger(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project.init(tmp_path / "compatible-offline")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible",
            "selected",
            "DL_COMPATIBLE_OFFLINE_KEY",
            "https://provider.example/v1",
        )
    )
    project.save_model_preset(
        ModelPreset("compatible-review", "openai-compatible", "selected", "fixture-model")
    )
    project.set_offline_mode(True)
    secret_lookups: list[str] = []
    transport_calls: list[bool] = []

    def unexpected_secret_lookup(
        _store: ReferenceSecretStore, reference: str
    ) -> str:
        secret_lookups.append(reference)
        pytest.fail("offline compatible execution must not resolve a secret")

    def unexpected_transport(*_args: object, **_kwargs: object) -> None:
        transport_calls.append(True)
        pytest.fail("offline compatible execution must not open provider transport")

    monkeypatch.setattr(ReferenceSecretStore, "get", unexpected_secret_lookup)
    monkeypatch.setattr("discourselens.providers._open_provider_request", unexpected_transport)
    try:
        with pytest.raises(ProjectError, match="offline mode blocks remote model execution"):
            execute_model_preset(project, "compatible-review", "offline prompt")

        assert secret_lookups == []
        assert transport_calls == []
        assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0
    finally:
        project.close()
