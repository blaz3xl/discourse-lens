from __future__ import annotations

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from discourselens import Project, mcp_server
from discourselens.errors import DocumentError
from discourselens.nlp import ParsedToken


class _ParsedFixture:
    language = "en"
    backend_id = "spacy"
    model_id = "fixture-en-v1"

    def parse(self, text: str) -> list[ParsedToken]:
        assert text == "Apples matter."
        return [ParsedToken("Apples", "apple", "NOUN", None, None, "ROOT", 0, 6)]


def _open_fixture_project(root, *, language: str | None = "en"):
    project = Project.init(root)
    document_id, revision_id = project.import_text("source.txt", "Apples matter.")
    if language is not None:
        project.set_document_metadata(document_id, "language", language)
    return project, revision_id


@pytest.mark.parametrize("failure", ["unknown_revision", "missing_language"])
def test_route_validates_revision_and_language_before_loading_and_closes_project(
    tmp_path, monkeypatch, failure: str
) -> None:
    root = tmp_path / f"route-validation-{failure}"
    project, valid_revision = _open_fixture_project(
        root, language=None if failure == "missing_language" else "en"
    )
    events: list[str] = []
    original_close = project.close
    monkeypatch.setattr(mcp_server, "_open_project", lambda: project)

    def close() -> None:
        events.append("close")
        original_close()

    def unexpected_loader(*_args: object) -> None:
        events.append("loader")
        pytest.fail("invalid revision/language must be rejected before loading NLP code")

    monkeypatch.setattr(project, "close", close)
    monkeypatch.setattr(mcp_server, "load_spacy", unexpected_loader)
    monkeypatch.setattr(mcp_server, "load_stanza", unexpected_loader)

    if failure == "unknown_revision":
        revision_id = "revision-that-does-not-exist"
        expected_error: type[Exception] = DocumentError
    else:
        revision_id = valid_revision
        expected_error = ToolError

    with pytest.raises(expected_error):
        mcp_server.build_parsed_token_artifact(revision_id, "spacy", "fixture-en-v1")

    assert events == ["close"]


def test_successful_explicit_route_closes_project_after_artifact_creation(
    tmp_path, monkeypatch
) -> None:
    project, revision_id = _open_fixture_project(tmp_path / "route-success")
    events: list[str] = []
    original_close = project.close
    original_build = project.build_parsed_token_artifact
    monkeypatch.setattr(mcp_server, "_open_project", lambda: project)

    def close() -> None:
        events.append("close")
        original_close()

    def build(revision: str, backend: _ParsedFixture) -> object:
        events.append("persist")
        return original_build(revision, backend)

    def load(language: str, model: str) -> _ParsedFixture:
        events.append("load:spacy")
        assert (language, model) == ("en", "fixture-en-v1")
        return _ParsedFixture()

    monkeypatch.setattr(project, "close", close)
    monkeypatch.setattr(project, "build_parsed_token_artifact", build)
    monkeypatch.setattr(mcp_server, "load_spacy", load)

    result = mcp_server.build_parsed_token_artifact(
        revision_id, "spacy", "fixture-en-v1"
    )

    assert result["revision_id"] == revision_id
    assert result["artifact_type"] == "parsed_tokens"
    assert events == ["load:spacy", "persist", "close"]


@pytest.mark.parametrize("backend", ["spacy", "stanza"])
def test_selected_backend_failure_never_falls_back_or_downloads(
    tmp_path, monkeypatch, backend: str
) -> None:
    project, revision_id = _open_fixture_project(tmp_path / f"no-fallback-{backend}")
    events: list[str] = []
    original_close = project.close
    monkeypatch.setattr(mcp_server, "_open_project", lambda: project)

    def close() -> None:
        events.append("close")
        original_close()

    def fail_spacy(language: str, model: str) -> object:
        events.append("load:spacy")
        raise RuntimeError("offline fixture: no installed model")

    def fail_stanza(language: str, model: str, model_dir: str) -> object:
        events.append("load:stanza")
        raise RuntimeError("offline fixture: no installed model")

    monkeypatch.setattr(project, "close", close)
    monkeypatch.setattr(mcp_server, "load_spacy", fail_spacy)
    monkeypatch.setattr(mcp_server, "load_stanza", fail_stanza)

    if backend == "spacy":
        selected_event = "load:spacy"
        fallback_event = "load:stanza"
        invoke_selected = lambda: mcp_server.build_parsed_token_artifact(
            revision_id, "spacy", "fixture-en-v1"
        )
    else:
        monkeypatch.setenv(
            mcp_server.MCP_STANZA_MODEL_DIR_ENV, str(tmp_path / "models")
        )
        selected_event = "load:stanza"
        fallback_event = "load:spacy"
        invoke_selected = lambda: mcp_server.build_parsed_token_artifact(
            revision_id, "stanza", "fixture-en-v1"
        )

    with pytest.raises(ToolError, match="no downloads or fallback were attempted"):
        invoke_selected()

    assert events == [selected_event, "close"]
    assert fallback_event not in events
