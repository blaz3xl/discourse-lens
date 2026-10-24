from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def _result(*, status: str = "answered") -> dict[str, object]:
    return {
        "status": status,
        "question": "How is responsibility represented?",
        "framework_id": "fairclough-text",
        "scope_snapshot_id": "scope-a",
        "candidate_analysis_run_id": "candidate-run",
        "candidate_skill_run_id": "candidate-skill",
        "candidate_retrieval_run_id": "retrieval-run",
        "candidate_finding_id": "candidate-finding",
        "answer_analysis_run_id": "answer-run",
        "answer_finding_id": "finding-a",
        "model_execution_id": "execution-a",
        "coverage": {"coverage_mode": "TARGETED", "inspection_count": 2},
        "answer_interpretation": "The passage backgrounds the agent.",
        "alternative_interpretation": "The agent may be recoverable from surrounding context.",
        "limitations": ["Only targeted passages were reviewed."],
        "evidence": [
            {
                "evidence_id": "evidence-a",
                "excerpt": "Several mistakes were made.",
                "citation": "[speech.pdf · p. 2 · ¶3]",
                "role": "SUPPORT",
            }
        ],
    }


def _invoke(runner: CliRunner, root: Path, *extra: str):
    return runner.invoke(
        app,
        [
            "ask",
            str(root),
            "How is responsibility represented?",
            "--revision",
            "rev-a",
            "--framework",
            "fairclough-text",
            "--preset",
            "research",
            *extra,
        ],
    )


def test_ask_help_describes_scoped_framework_review() -> None:
    result = CliRunner().invoke(app, ["ask", "--help"])

    assert result.exit_code == 0, result.output
    assert "--revision" in result.output
    assert "--framework" in result.output
    assert "--preset" in result.output
    assert "--limit" in result.output
    assert "--retrieval" in result.output
    assert "--embedding-preset" in result.output
    assert "--embedding-artifac" in result.output
    assert "Output format:" in result.output


def test_ask_text_formats_core_answer_citations_and_targeted_limitations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ask-project"
    Project.init(root).close()
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        Project,
        "ask_question",
        lambda self, revisions, question, framework, preset, *, limit=8, **kwargs: (
            calls.append((revisions, question, framework, preset, limit, kwargs)) or _result()
        ),
        raising=False,
    )

    result = _invoke(CliRunner(), root, "--limit", "5")

    assert result.exit_code == 0, result.output
    assert "The passage backgrounds the agent." in result.stdout
    assert "[speech.pdf · p. 2 · ¶3]" in result.stdout
    assert "Several mistakes were made." in result.stdout
    assert "Targeted retrieval" in result.stdout
    assert "does not establish absence" not in result.stdout
    assert "Only targeted passages were reviewed." in result.stdout
    assert calls == [
        (
            ["rev-a"],
            "How is responsibility represented?",
            "fairclough-text",
            "research",
            5,
            {"retrieval_mode": "lexical"},
        )
    ]
    assert "Candidate run: candidate-run" in result.stdout
    assert "Answer run: answer-run" in result.stdout
    assert "Answer finding: finding-a" in result.stdout


def test_ask_hybrid_passes_embedding_options_to_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ask-hybrid-project"
    Project.init(root).close()
    calls: list[dict[str, object]] = []

    def capture_ask(
        self: Project, *args: object, **kwargs: object
    ) -> dict[str, object]:
        calls.append(kwargs)
        return _result()

    monkeypatch.setattr(Project, "ask_question", capture_ask, raising=False)

    result = _invoke(
        CliRunner(),
        root,
        "--retrieval",
        "hybrid",
        "--embedding-preset",
        "embed",
        "--embedding-artifacts",
        "artifact-a, artifact-b",
        "--format",
        "json",
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "limit": 8,
            "retrieval_mode": "hybrid",
            "embedding_preset_name": "embed",
            "embedding_artifact_ids": ["artifact-a", "artifact-b"],
        }
    ]


def test_ask_json_preserves_structured_core_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ask-json-project"
    Project.init(root).close()
    expected = _result()
    monkeypatch.setattr(
        Project,
        "ask_question",
        lambda self, *args, **kwargs: expected,
        raising=False,
    )

    result = _invoke(CliRunner(), root, "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == expected


def test_ask_zero_candidates_has_safe_wording_and_never_renders_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ask-empty-project"
    Project.init(root).close()
    no_candidates = _result(status="no_candidates")
    no_candidates["answer_interpretation"] = "THIS ANSWER MUST NOT BE SHOWN"
    monkeypatch.setattr(
        Project,
        "ask_question",
        lambda self, *args, **kwargs: no_candidates,
        raising=False,
    )

    result = _invoke(CliRunner(), root)

    assert result.exit_code == 0, result.output
    assert "No examples were found under the retrieval plan." in result.stdout
    assert "does not establish that no such examples exist in the scope" in result.stdout
    assert "THIS ANSWER MUST NOT BE SHOWN" not in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        [
            "ask",
            "PROJECT",
            "   ",
            "--revision",
            "rev-a",
            "--framework",
            "fairclough-text",
            "--preset",
            "research",
        ],
        [
            "ask",
            "PROJECT",
            "question",
            "--revision",
            "rev-a",
            "--framework",
            "fairclough-text",
            "--preset",
            "   ",
        ],
        [
            "ask",
            "PROJECT",
            "question",
            "--revision",
            "rev-a",
            "--framework",
            "critical-reviewer",
            "--preset",
            "research",
        ],
        [
            "ask",
            "PROJECT",
            "question",
            "--revision",
            " rev-a ",
            "--revision",
            "rev-a",
            "--framework",
            "fairclough-text",
            "--preset",
            "research",
        ],
        [
            "ask",
            "PROJECT",
            "question",
            "--revision",
            "rev-a",
            "--framework",
            "fairclough-text",
            "--preset",
            "research",
            "--retrieval",
            "hybrid",
            "--embedding-artifacts",
            "artifact-a",
        ],
        [
            "ask",
            "PROJECT",
            "question",
            "--revision",
            "rev-a",
            "--framework",
            "fairclough-text",
            "--preset",
            "research",
            "--retrieval",
            "hybrid",
            "--embedding-preset",
            "embed",
        ],
        [
            "ask",
            "PROJECT",
            "question",
            "--revision",
            "rev-a",
            "--framework",
            "fairclough-text",
            "--preset",
            "research",
            "--retrieval",
            "hybrid",
            "--embedding-preset",
            "embed",
            "--embedding-artifacts",
            "artifact-a,,artifact-b",
        ],
        [
            "ask",
            "PROJECT",
            "question",
            "--revision",
            "rev-a",
            "--framework",
            "fairclough-text",
            "--preset",
            "research",
            "--retrieval",
            "hybrid",
            "--embedding-preset",
            "embed",
            "--embedding-artifacts",
            "artifact-a,artifact-a",
        ],
        [
            "ask",
            "PROJECT",
            "question",
            "--revision",
            "rev-a",
            "--framework",
            "fairclough-text",
            "--preset",
            "research",
            "--embedding-preset",
            "embed",
        ],
    ],
)
def test_ask_rejects_invalid_inputs_before_opening_project(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []

    def unexpected_open(path: Path, *args: Any, **kwargs: Any) -> Project:
        opened.append(path)
        raise AssertionError("invalid input must be rejected before opening the project")

    monkeypatch.setattr(Project, "open", unexpected_open)
    arguments = [str(Path("unused-project")) if item == "PROJECT" else item for item in arguments]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 2, result.output
    assert opened == []


def test_ask_closes_project_when_core_call_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ask-close-project"
    Project.init(root).close()
    closed: list[bool] = []
    original_close = Project.close

    def tracked_close(self: Project) -> None:
        closed.append(True)
        original_close(self)

    def failing_ask(self: Project, *args: Any, **kwargs: Any) -> dict[str, object]:
        raise RuntimeError("expected mocked failure")

    monkeypatch.setattr(Project, "close", tracked_close)
    monkeypatch.setattr(Project, "ask_question", failing_ask, raising=False)

    result = _invoke(CliRunner(), root)

    assert result.exit_code != 0
    assert closed == [True]
