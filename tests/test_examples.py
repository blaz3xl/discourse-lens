import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from discourselens import Project, verify_research_bundle

_EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples/offline_corpus_workflow.py"
_EXAMPLE_SPEC = importlib.util.spec_from_file_location("offline_corpus_workflow", _EXAMPLE_PATH)
assert _EXAMPLE_SPEC is not None and _EXAMPLE_SPEC.loader is not None
_EXAMPLE_MODULE = importlib.util.module_from_spec(_EXAMPLE_SPEC)
_EXAMPLE_SPEC.loader.exec_module(_EXAMPLE_MODULE)
run_offline_workflow = _EXAMPLE_MODULE.run_offline_workflow

_SCOPE_QUESTION_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples/offline_scope_question_workflow.py"
)
_SCOPE_QUESTION_EXAMPLE_SPEC = importlib.util.spec_from_file_location(
    "offline_scope_question_workflow", _SCOPE_QUESTION_EXAMPLE_PATH
)
assert _SCOPE_QUESTION_EXAMPLE_SPEC is not None
assert _SCOPE_QUESTION_EXAMPLE_SPEC.loader is not None
_SCOPE_QUESTION_EXAMPLE_MODULE = importlib.util.module_from_spec(
    _SCOPE_QUESTION_EXAMPLE_SPEC
)
_SCOPE_QUESTION_EXAMPLE_SPEC.loader.exec_module(_SCOPE_QUESTION_EXAMPLE_MODULE)
run_offline_scope_question_workflow = (
    _SCOPE_QUESTION_EXAMPLE_MODULE.run_offline_scope_question_workflow
)


def test_offline_corpus_workflow_creates_and_verifies_portable_bundle(tmp_path):
    output_directory = tmp_path / "offline-example"
    output_directory.mkdir()

    result = run_offline_workflow(output_directory)

    selected_revision = result["corpus_scope_revision_ids"][0]
    assert len(result["corpus_scope_revision_ids"]) == 2
    assert result["analysis_scope_id"] != result["corpus_scope_id"]
    assert result["migration_frequency"] == 3
    assert len(result["migration_kwic"]) == 3
    assert {row["keyword"] for row in result["migration_kwic"]} == {"migration"}
    assert result["analysis_revision_ids"] == [selected_revision]
    assert len(result["finding_ids"]) == 1
    assert result["evidence_texts"] == ["must", "must"]

    manifest_path = output_directory / "research-bundle" / "manifest.json"
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [row["id"] for row in bundle["document_revisions"]] == [selected_revision]
    assert verify_research_bundle(manifest_path)["integrity"] == "VERIFIED_SHA256"
    assert result["verification"]["status"] == "VALID"
    with Project.open(output_directory / "project") as project:
        assert project.offline_mode is True


def test_offline_corpus_workflow_command_prints_verified_summary(tmp_path, capsys):
    output_directory = tmp_path / "offline-command"

    assert _EXAMPLE_MODULE.main([str(output_directory)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["verification"]["status"] == "VALID"
    assert summary["verification"]["integrity"] == "VERIFIED_SHA256"
    assert (output_directory / "research-bundle" / "manifest.json").is_file()


@pytest.mark.parametrize(
    ("script_name", "expected_summary"),
    [
        (
            "offline_corpus_workflow.py",
            {"migration_frequency": 3, "verification_status": "VALID"},
        ),
        (
            "offline_scope_question_workflow.py",
            {"candidate_count": 2, "coverage_mode": "TARGETED"},
        ),
    ],
)
def test_offline_examples_execute_as_cli_and_print_json(
    tmp_path, script_name, expected_summary
):
    repository_root = Path(__file__).resolve().parents[1]
    script_path = repository_root / "examples" / script_name
    output_directory = tmp_path / "example-output"
    environment = os.environ.copy()
    source_path = str(repository_root / "src")
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_path, current_pythonpath) if value
    )

    completed = subprocess.run(
        [sys.executable, str(script_path), str(output_directory)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["output_directory"] == str(output_directory)
    for key, value in expected_summary.items():
        if key == "verification_status":
            assert summary["verification"]["status"] == value
            assert summary["verification"]["integrity"] == "VERIFIED_SHA256"
            manifest_path = output_directory / "research-bundle" / "manifest.json"
            assert verify_research_bundle(manifest_path)["integrity"] == "VERIFIED_SHA256"
        else:
            assert summary[key] == value
    with Project.open(output_directory / "project", read_only=True) as project:
        assert project.offline_mode is True


@pytest.mark.parametrize(
    ("workflow", "output_name"),
    [
        (run_offline_workflow, "corpus-output"),
        (run_offline_scope_question_workflow, "question-output"),
    ],
)
def test_offline_workflows_close_the_project_after_return(
    tmp_path, monkeypatch, workflow, output_name
):
    initialized_projects = []
    original_init = Project.init

    def track_project_init(cls, root):
        project = original_init(root)
        initialized_projects.append(project)
        return project

    monkeypatch.setattr(Project, "init", classmethod(track_project_init))

    workflow(tmp_path / output_name)

    assert len(initialized_projects) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        initialized_projects[0].db.execute("SELECT 1")


def test_offline_corpus_workflow_refuses_nonempty_output_directory_without_overwriting(
    tmp_path,
):
    output_directory = tmp_path / "existing-output"
    output_directory.mkdir()
    existing_file = output_directory / "keep.txt"
    existing_file.write_text("preserve this file", encoding="utf-8")

    with pytest.raises(ValueError, match="new or empty"):
        run_offline_workflow(output_directory)

    assert existing_file.read_text(encoding="utf-8") == "preserve this file"
    assert sorted(path.name for path in output_directory.iterdir()) == ["keep.txt"]


def test_offline_scope_question_workflow_persists_targeted_retrieval_lineage(tmp_path):
    output_directory = tmp_path / "scope-question-example"
    output_directory.mkdir()

    result = run_offline_scope_question_workflow(output_directory)

    assert len(result["revision_ids"]) == 2
    assert result["candidate_count"] == 2
    assert result["coverage_mode"] == "TARGETED"
    assert set(result["evidence_texts"]) == {
        "Officials describe housing support as essential for families.",
        "Residents say housing support must remain available.",
    }

    with Project.open(output_directory / "project", read_only=True) as project:
        metadata = project.analysis_run_summary(result["analysis_run_id"])
        run = metadata["analysis_run"]
        assert isinstance(run, Mapping)
        config = run["config"]
        assert isinstance(config, Mapping)
        assert run["status"] == "COMPLETED"
        assert metadata["revision_ids"] == result["revision_ids"]
        assert run["scope_id"] == result["scope_snapshot_id"]
        assert config["framework_id"] == "fairclough-text"
        assert config["question"] == "How is housing support described?"
        retrieval_runs = cast(list[dict[str, object]], metadata["retrieval_runs"])
        assert len(retrieval_runs) == 1
        assert retrieval_runs[0]["id"] == result["retrieval_run_id"]
        assert retrieval_runs[0]["skill_run_id"] == result["skill_run_id"]
        assert retrieval_runs[0]["scope_id"] == result["scope_snapshot_id"]

        scope_revisions = project.scope_revisions(result["scope_snapshot_id"])
        assert scope_revisions == result["revision_ids"]
        assert len(project.list_analysis_runs()) == 1


def test_offline_scope_question_workflow_refuses_nonempty_output_directory_without_overwriting(
    tmp_path,
):
    output_directory = tmp_path / "existing-scope-output"
    output_directory.mkdir()
    existing_file = output_directory / "keep.txt"
    existing_file.write_text("preserve this file", encoding="utf-8")

    with pytest.raises(ValueError, match="new or empty"):
        run_offline_scope_question_workflow(output_directory)

    assert existing_file.read_text(encoding="utf-8") == "preserve this file"
    assert sorted(path.name for path in output_directory.iterdir()) == ["keep.txt"]
