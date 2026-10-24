from __future__ import annotations

import json
from types import SimpleNamespace

import click
import pytest
from typer.testing import CliRunner

import discourselens.cli as cli_module
from discourselens.cli import app
from discourselens.errors import ModelProviderError, ProjectError
from discourselens.models import GenerationImage


class _ProjectStub:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _project_stub(monkeypatch) -> _ProjectStub:
    project = _ProjectStub()
    monkeypatch.setattr(cli_module.Project, "open", lambda _path: project)
    return project


def test_model_generate_image_help_lists_supported_adapter_families():
    result = CliRunner().invoke(app, ["models", "generate", "--help"])
    output = " ".join(result.output.split())

    assert result.exit_code == 0, result.output
    assert "buffered Gemini" in output
    # Rich wraps the option description at the terminal width and inserts
    # table borders between the two words, so check both adjacent fragments.
    assert "OpenAI Chat" in output
    assert "Completions, or Anthropic Messages" in output
    assert "Anthropic Messages" in output
    assert "exact model" in output
    assert "confirmed" in output
    assert "individually." in output
    assert "Gemini only" not in output


def test_model_generate_streams_only_explicit_text_output(tmp_path, monkeypatch):
    project = _project_stub(monkeypatch)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fixture prompt", encoding="utf-8")
    observed: list[tuple[object, str, str]] = []

    def stream(_project, preset: str, text: str):
        observed.append((_project, preset, text))
        yield "discourse"
        yield " lens"

    monkeypatch.setattr(cli_module, "execute_model_preset_stream", stream)
    result = CliRunner().invoke(
        app,
        [
            "models",
            "generate",
            "research",
            "analysis",
            "--prompt-file",
            str(prompt),
            "--stream",
            "--format",
            "text",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "discourse lens\n"
    assert observed == [(project, "analysis", "fixture prompt")]
    assert project.closed


def test_model_generate_stream_rejects_json_before_opening_project(tmp_path, monkeypatch):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fixture prompt", encoding="utf-8")
    opened = False

    def unexpected_open(_path):
        nonlocal opened
        opened = True
        raise AssertionError("format validation must precede project opening")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = CliRunner().invoke(
        app,
        ["models", "generate", "research", "analysis", "--prompt-file", str(prompt), "--stream"],
    )

    assert result.exit_code == 2
    assert "--stream requires --format text" in result.output
    assert not opened


def test_model_generate_keeps_buffered_json_default(tmp_path, monkeypatch):
    project = _project_stub(monkeypatch)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fixture prompt", encoding="utf-8")
    monkeypatch.setattr(
        cli_module,
        "execute_model_preset",
        lambda *_args, **_kwargs: SimpleNamespace(
            execution_id="execution-fixture",
            response=SimpleNamespace(provider="fixture", model="model-a", text="complete"),
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "execute_model_preset_stream",
        lambda *_args, **_kwargs: pytest.fail("default output must stay buffered"),
    )

    result = CliRunner().invoke(
        app,
        ["models", "generate", "research", "analysis", "--prompt-file", str(prompt)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "execution_id": "execution-fixture",
        "provider": "fixture",
        "model": "model-a",
        "text": "complete",
    }
    assert project.closed


def test_model_generate_parses_one_local_image_without_live_api(tmp_path, monkeypatch):
    project = _project_stub(monkeypatch)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fixture prompt", encoding="utf-8")
    image_path = tmp_path / "fixture.png"
    image_bytes = b"\x89PNG\r\n\x1a\ncli-image-fixture"
    image_path.write_bytes(image_bytes)
    observed: list[GenerationImage | None] = []

    def buffered(_project, _preset: str, _prompt: str, *, image: GenerationImage | None = None):
        observed.append(image)
        return SimpleNamespace(
            execution_id="execution-fixture",
            response=SimpleNamespace(provider="gemini", model="fixture-model", text="done"),
        )

    monkeypatch.setattr(cli_module, "execute_model_preset", buffered)
    result = CliRunner().invoke(
        app,
        [
            "models",
            "generate",
            "research",
            "analysis",
            "--prompt-file",
            str(prompt),
            "--image",
            str(image_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["text"] == "done"
    assert len(observed) == 1
    assert observed[0] is not None
    assert observed[0].mime_type == "image/png"
    assert observed[0].data == image_bytes
    assert str(image_path) not in result.stdout
    assert project.closed


def test_model_generate_rejects_image_streaming_before_opening_project(tmp_path, monkeypatch):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fixture prompt", encoding="utf-8")
    opened = False

    def unexpected_open(_path):
        nonlocal opened
        opened = True
        raise AssertionError("image streaming validation must precede project opening")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = CliRunner().invoke(
        app,
        [
            "models",
            "generate",
            "research",
            "analysis",
            "--prompt-file",
            str(prompt),
            "--image",
            "https://example.test/image.png",
            "--stream",
            "--format",
            "text",
        ],
    )

    assert result.exit_code == 2
    assert "--image is not supported with --stream" in result.output
    assert not opened


def test_model_generate_rejects_repeated_image_option_before_opening_project(
    tmp_path, monkeypatch
):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fixture prompt", encoding="utf-8")
    opened = False

    def unexpected_open(_path):
        nonlocal opened
        opened = True
        raise AssertionError("repeated image validation must precede project opening")

    monkeypatch.setattr(cli_module.Project, "open", unexpected_open)
    result = CliRunner().invoke(
        app,
        [
            "models",
            "generate",
            "research",
            "analysis",
            "--prompt-file",
            str(prompt),
            "--image",
            "first.png",
            "--image",
            "second.png",
        ],
    )

    assert result.exit_code == 2
    assert "--image can be supplied only once" in result.output
    assert not opened


def test_model_generate_redacts_stream_failure_and_preserves_partial_output(tmp_path, monkeypatch):
    project = _project_stub(monkeypatch)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fixture prompt", encoding="utf-8")
    secret = "streaming-secret-fixture-value"

    def failed_stream(*_args, **_kwargs):
        yield "partial"
        raise ModelProviderError(f"Authorization: Bearer {secret}")

    monkeypatch.setattr(cli_module, "execute_model_preset_stream", failed_stream)
    result = CliRunner().invoke(
        app,
        [
            "models",
            "generate",
            "research",
            "analysis",
            "--prompt-file",
            str(prompt),
            "--stream",
            "--format",
            "text",
        ],
    )

    assert result.exit_code == 1
    assert "partial" in result.stdout
    assert "Error:" in result.stderr
    assert secret not in result.output
    assert "<redacted>" in result.stderr
    assert project.closed


def test_model_generate_reports_offline_policy_failure_without_traceback(tmp_path, monkeypatch):
    project = _project_stub(monkeypatch)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fixture prompt", encoding="utf-8")

    def offline_stream(*_args, **_kwargs):
        raise ProjectError("project offline mode blocks remote model execution")
        yield "unreachable"

    monkeypatch.setattr(cli_module, "execute_model_preset_stream", offline_stream)
    result = CliRunner().invoke(
        app,
        [
            "models",
            "generate",
            "research",
            "analysis",
            "--prompt-file",
            str(prompt),
            "--stream",
            "--format",
            "text",
        ],
    )

    assert result.exit_code == 1
    assert "project offline mode blocks remote model execution" in result.stderr
    assert "Traceback" not in result.output
    assert project.closed


def test_model_generate_closes_stream_if_stdout_write_fails(tmp_path, monkeypatch):
    project = _project_stub(monkeypatch)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("fixture prompt", encoding="utf-8")
    closed = False

    class _Stream:
        def __iter__(self):
            yield "chunk"

        def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(cli_module, "execute_model_preset_stream", lambda *_args: _Stream())
    original_echo = click.echo

    def broken_stdout(message=None, *, err=False, **kwargs):
        if not err:
            raise OSError("simulated closed output pipe")
        return original_echo(message, err=err, **kwargs)

    monkeypatch.setattr(cli_module.click, "echo", broken_stdout)
    result = CliRunner().invoke(
        app,
        [
            "models",
            "generate",
            "research",
            "analysis",
            "--prompt-file",
            str(prompt),
            "--stream",
            "--format",
            "text",
        ],
    )

    assert result.exit_code == 1
    assert "filesystem operation" in result.stderr
    assert closed
    assert project.closed
