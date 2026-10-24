"""Focused Core contract tests for evidence-grounded Project.ask_question."""

from __future__ import annotations

import json
from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import (
    DocumentError,
    EvidenceValidationError,
    ModelResponseError,
    ProjectError,
)
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
)


class _QuestionProvider:
    provider_id = "openai-compatible"

    def __init__(self, response_factory=None):
        self.response_factory = response_factory
        self.calls = 0

    def capabilities(self, model: str) -> ModelCapabilities:
        del model
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        marker = "UNTRUSTED_RESEARCH_DATA_JSON:\n"
        research = json.loads(request.prompt.split(marker, 1)[1])
        records = cast(list[dict[str, object]], research["evidence_records"])
        record = next(item for item in records if item.get("evidence_id") and item.get("excerpt"))
        response = (
            self.response_factory(str(record["evidence_id"]), str(record["excerpt"]))
            if self.response_factory
            else {
                "interpretation": "The passage may foreground institutional action.",
                "alternative_interpretation": "It may simply report an administrative event.",
                "evidence_ids": [record["evidence_id"]],
                "counterevidence_ids": [],
                "limitations": ["The retrieval was targeted and inspected selected passages only."],
            }
        )
        # The framework review contract requires an observation tied to exact
        # text. Core validates both the ID and verbatim excerpt.
        response.setdefault(
            "observations",
            [
                {
                    "evidence_id": record["evidence_id"],
                    "excerpt": record["excerpt"],
                    "description": "The passage explicitly names the social actor.",
                }
            ],
        )
        return GenerationResponse(json.dumps(response), self.provider_id, request.model)


def _project_with_preset(tmp_path, monkeypatch, *, name: str = "research") -> Project:
    monkeypatch.setenv("DL_ASK_TEST_KEY", "not-a-real-secret")
    project = Project.init(tmp_path / name)
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "fixture", "DL_ASK_TEST_KEY", "http://127.0.0.1:9/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("fixture-review", "openai-compatible", "fixture", "fixture-model")
    )
    return project


def _import(project: Project, name: str, text: str) -> str:
    _, revision_id = project.import_text(name, text)
    return revision_id


def test_ask_question_returns_validated_answer_and_core_citations(tmp_path, monkeypatch):
    project = _project_with_preset(tmp_path, monkeypatch)
    provider = _QuestionProvider()
    from discourselens import interpretation

    original_execute = interpretation.execute_model_preset

    def execute_with_fixture_provider(project, preset_name, prompt, **kwargs):
        kwargs["provider_factory"] = lambda endpoint, reference, provider_id: provider
        return original_execute(project, preset_name, prompt, **kwargs)

    monkeypatch.setattr(interpretation, "execute_model_preset", execute_with_fixture_provider)
    revision_id = _import(
        project,
        "speech.txt",
        "The council welcomed migrants after the meeting.",
    )

    result = project.ask_question(
        [revision_id],
        "What did the council do about migrants?",
        "fairclough-text",
        "fixture-review",
    )

    assert result["status"] == "answered"
    assert result["question"] == "What did the council do about migrants?"
    assert result["framework_id"] == "fairclough-text"
    assert result["scope_snapshot_id"]
    assert result["candidate_analysis_run_id"]
    assert result["candidate_skill_run_id"]
    assert result["candidate_finding_id"]
    assert result["candidate_retrieval_run_id"]
    assert result["answer_analysis_run_id"]
    assert result["answer_finding_id"]
    assert result["model_execution_id"]
    assert result["answer_coverage_id"]
    assert "TARGETED" in json.dumps(result["answer_coverage"])
    assert result["answer_interpretation"]
    assert result["alternative_interpretation"]
    assert "TARGETED" in json.dumps(result["coverage"])
    assert provider.calls == 1
    assert project.validate_finding(str(result["answer_finding_id"])).valid

    evidence = cast(list[dict[str, object]], result["evidence"])
    assert evidence
    assert all(item["evidence_id"] and item["citation"] for item in evidence)
    assert evidence[0]["excerpt"] == "The council welcomed migrants after the meeting."
    assert evidence[0]["citation"] == "[speech.txt · ¶1]"
    assert evidence[0]["role"] == "EXAMPLE"


def test_ask_question_zero_hits_returns_no_answer_and_never_calls_model(tmp_path, monkeypatch):
    project = _project_with_preset(tmp_path, monkeypatch, name="no-candidates")
    revision_id = _import(project, "speech.txt", "The apples are green.")
    calls = 0

    def forbidden_review(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("model review must not run without lexical candidates")

    monkeypatch.setattr(Project, "review_finding_with_model", forbidden_review)
    result = project.ask_question(
        [revision_id], "Where do elephants live?", "fairclough-text", "fixture-review"
    )

    assert result["status"] == "no_candidates"
    assert result["answer_analysis_run_id"] is None
    assert result["answer_finding_id"] is None
    assert result["model_execution_id"] is None
    assert result["answer_coverage_id"] is None
    assert result["answer_coverage"] is None
    assert result["answer_interpretation"] is None
    assert result["evidence"] == []
    assert any("does not establish" in item for item in result["limitations"])
    assert calls == 0
    assert project.db.execute("SELECT COUNT(*) FROM model_executions").fetchone()[0] == 0


def test_ask_question_never_retrieves_outside_explicit_revision_scope(tmp_path, monkeypatch):
    project = _project_with_preset(tmp_path, monkeypatch, name="scope-isolation")
    provider = _QuestionProvider()
    from discourselens import interpretation

    original_execute = interpretation.execute_model_preset

    def execute_with_fixture_provider(project, preset_name, prompt, **kwargs):
        kwargs["provider_factory"] = lambda endpoint, reference, provider_id: provider
        return original_execute(project, preset_name, prompt, **kwargs)

    monkeypatch.setattr(interpretation, "execute_model_preset", execute_with_fixture_provider)
    revision_a = _import(project, "a.txt", "Apples are green.")
    revision_b = _import(project, "b.txt", "Elephants live in Africa.")

    scoped_miss = project.ask_question(
        [revision_a], "Where do elephants live?", "fairclough-text", "fixture-review"
    )
    scoped_hit = project.ask_question(
        [revision_b], "Where do elephants live?", "fairclough-text", "fixture-review"
    )

    assert scoped_miss["status"] == "no_candidates"
    assert scoped_hit["status"] == "answered"
    for item in cast(list[dict[str, object]], scoped_hit["evidence"]):
        evidence = project.db.execute(
            "SELECT s.revision_id FROM evidence_artifacts e JOIN source_spans s ON s.id=e.span_id "
            "WHERE e.id=?",
            (item["evidence_id"],),
        ).fetchone()
        assert evidence["revision_id"] == revision_b
    assert provider.calls == 1


@pytest.mark.parametrize(
    ("response_factory", "error_type", "error_match"),
    [
        (
            lambda evidence_id, excerpt: {
                "interpretation": "This may describe a social actor.",
                "alternative_interpretation": "It may be a neutral report.",
                "evidence_ids": ["EV-NOT-PERSISTED"],
                "counterevidence_ids": [],
                "limitations": ["Targeted evidence only."],
            },
            EvidenceValidationError,
            "unknown evidence ID",
        ),
        (
            lambda evidence_id, excerpt: {
                "interpretation": "This appears on p. 9.",
                "alternative_interpretation": "It may be a neutral report.",
                "evidence_ids": [evidence_id],
                "counterevidence_ids": [],
                "limitations": ["Targeted evidence only."],
            },
            ModelResponseError,
            "inline citations",
        ),
    ],
)
def test_ask_question_rejects_unknown_or_inline_model_citations(
    tmp_path, monkeypatch, response_factory, error_type, error_match
):
    project = _project_with_preset(tmp_path, monkeypatch, name="invalid-model-answer")
    provider = _QuestionProvider(response_factory)
    from discourselens import interpretation

    original_execute = interpretation.execute_model_preset

    def execute_with_fixture_provider(project, preset_name, prompt, **kwargs):
        kwargs["provider_factory"] = lambda endpoint, reference, provider_id: provider
        return original_execute(project, preset_name, prompt, **kwargs)

    monkeypatch.setattr(interpretation, "execute_model_preset", execute_with_fixture_provider)
    revision_id = _import(project, "speech.txt", "The council welcomed migrants.")

    with pytest.raises(error_type, match=error_match):
        project.ask_question(
            [revision_id], "What happened to migrants?", "fairclough-text", "fixture-review"
        )

    assert provider.calls == 1
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM findings WHERE skill_id='builtin.fairclough-text'"
        ).fetchone()[0]
        == 0
    )


@pytest.mark.parametrize(
    ("revision_ids", "question", "framework", "preset", "limit"),
    [
        ([], "migrants", "fairclough-text", "fixture-review", 8),
        (["missing-revision"], "migrants", "fairclough-text", "fixture-review", 8),
        (["REVISION"], "  ", "fairclough-text", "fixture-review", 8),
        (["REVISION"], "migrants", "critical-reviewer", "fixture-review", 8),
        (["REVISION"], "migrants", "fairclough-text", "missing-preset", 8),
        (["REVISION"], "migrants", "fairclough-text", "fixture-review", True),
    ],
)
def test_ask_question_invalid_inputs_fail_before_creating_runs(
    tmp_path, monkeypatch, revision_ids, question, framework, preset, limit
):
    project = _project_with_preset(tmp_path, monkeypatch, name="invalid-inputs")
    valid_revision = _import(project, "speech.txt", "The council welcomed migrants.")
    revision_ids = [valid_revision] if revision_ids == ["REVISION"] else revision_ids
    before = project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0]

    with pytest.raises((ProjectError, DocumentError)):
        project.ask_question(
            revision_ids,
            question,
            framework,
            preset,
            limit=limit,
        )

    after = project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0]
    assert after == before
