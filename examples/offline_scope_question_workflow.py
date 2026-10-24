"""Run targeted, researcher-directed lexical retrieval on synthetic local text."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

from discourselens import Project
from discourselens.types import CoverageMode


class WorkflowSummary(TypedDict):
    output_directory: str
    project_directory: str
    revision_ids: list[str]
    scope_snapshot_id: str
    analysis_run_id: str
    skill_run_id: str
    retrieval_run_id: str
    finding_id: str
    candidate_count: int
    coverage_mode: str
    evidence_texts: list[str]


_PASSAGES = (
    (
        "officials.txt",
        "Officials describe housing support as essential for families.",
    ),
    (
        "residents.txt",
        "Residents say housing support must remain available.",
    ),
)
_OUTSIDE_PASSAGE = "outside.txt", "Housing support is described as a burden."
_QUESTION = "How is housing support described?"


def _prepare_output_directory(path: str | Path) -> Path:
    output_directory = Path(path).expanduser().absolute()
    if output_directory.exists():
        if not output_directory.is_dir() or next(output_directory.iterdir(), None) is not None:
            raise ValueError("output directory must be new or empty")
    else:
        output_directory.mkdir(parents=True)
    return output_directory


def run_offline_scope_question_workflow(output_path: str | Path) -> WorkflowSummary:
    """Run one explicit targeted retrieval without a model or network provider."""
    output_directory = _prepare_output_directory(output_path)
    project_directory = output_directory / "project"

    with Project.init(project_directory) as project:
        project.set_offline_mode(True)
        revision_ids = [
            project.import_text(name, text)[1]
            for name, text in _PASSAGES
        ]
        project.import_text(*_OUTSIDE_PASSAGE)

        result = project.run_scope_question_candidates(
            revision_ids,
            "fairclough-text",
            _QUESTION,
            limit=8,
        )
        finding_id = result["finding_id"]
        if finding_id is None or result["candidate_count"] == 0:
            raise RuntimeError("the sample question did not return a lexical candidate")

        if result["coverage"].coverage_mode is not CoverageMode.TARGETED:
            raise RuntimeError("scope-question retrieval did not retain targeted coverage")
        if not project.validate_finding(finding_id).valid:
            raise RuntimeError("the targeted finding failed structural evidence validation")

        scope_snapshot_id = result["scope_snapshot_id"]
        if project.scope_revisions(scope_snapshot_id) != revision_ids:
            raise RuntimeError("targeted retrieval escaped its explicit revision scope")

        evidence_texts = [
            project.span_text(str(row["span_id"]))
            for row in project.finding_evidence(finding_id)
        ]
        run_metadata = project.analysis_run_summary(result["analysis_run_id"])
        retrieval_runs = cast(list[dict[str, object]], run_metadata["retrieval_runs"])
        matching_retrievals = [
            retrieval
            for retrieval in retrieval_runs
            if retrieval["id"] == result["retrieval_run_id"]
        ]
        if len(matching_retrievals) != 1:
            raise RuntimeError("retrieval run provenance is missing or ambiguous")
        retrieval = matching_retrievals[0]
        if (
            retrieval["analysis_run_id"] != result["analysis_run_id"]
            or retrieval["scope_id"] != scope_snapshot_id
            or retrieval["skill_run_id"] != result["skill_run_id"]
        ):
            raise RuntimeError("retrieval lineage does not match its analysis scope")

        return {
            "output_directory": str(output_directory),
            "project_directory": str(project_directory),
            "revision_ids": revision_ids,
            "scope_snapshot_id": scope_snapshot_id,
            "analysis_run_id": result["analysis_run_id"],
            "skill_run_id": result["skill_run_id"],
            "retrieval_run_id": result["retrieval_run_id"],
            "finding_id": finding_id,
            "candidate_count": result["candidate_count"],
            "coverage_mode": result["coverage"].coverage_mode.value,
            "evidence_texts": evidence_texts,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_directory",
        type=Path,
        help="new or empty directory for the project created by this offline example",
    )
    arguments = parser.parse_args(argv)
    try:
        summary = run_offline_scope_question_workflow(arguments.output_directory)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
