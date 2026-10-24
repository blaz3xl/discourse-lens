"""Run a fully local corpus and evidence workflow on synthetic text."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

from discourselens import Project, verify_research_bundle


class WorkflowSummary(TypedDict):
    output_directory: str
    project_directory: str
    corpus_scope_id: str
    corpus_scope_revision_ids: list[str]
    migration_frequency: int
    migration_kwic: list[dict[str, object]]
    analysis_run_id: str
    analysis_scope_id: str
    analysis_revision_ids: list[str]
    finding_ids: list[str]
    evidence_texts: list[str]
    bundle_manifest: str
    verification: dict[str, object]


_SELECTED_TEXT = (
    "Officials must discuss migration in public deliberation. "
    "Residents must ask whether migration policy meets local needs."
)
_COMPARISON_TEXT = "Earlier residents discussed migration timelines and seasonal services."


def _prepare_output_directory(path: str | Path) -> Path:
    output_directory = Path(path).expanduser().absolute()
    if output_directory.exists():
        if not output_directory.is_dir() or next(output_directory.iterdir(), None) is not None:
            raise ValueError("output directory must be new or empty")
    else:
        output_directory.mkdir(parents=True)
    return output_directory


def run_offline_workflow(output_path: str | Path) -> WorkflowSummary:
    """Create a portable, verified research bundle in a new or empty directory."""
    output_directory = _prepare_output_directory(output_path)
    project_directory = output_directory / "project"
    bundle_directory = output_directory / "research-bundle"

    with Project.init(project_directory) as project:
        project.set_offline_mode(True)
        _, selected_revision = project.import_text("selected.txt", _SELECTED_TEXT)
        _, comparison_revision = project.import_text("comparison.txt", _COMPARISON_TEXT)

        corpus_scope_id = project.snapshot([selected_revision, comparison_revision])
        corpus_scope_revision_ids = project.scope_revisions(corpus_scope_id)
        frequencies = project.corpus_frequency(corpus_scope_id)
        kwic_rows = project.corpus_kwic(corpus_scope_id, "migration", window=4)

        analysis = project.analyze_textual_descriptors([selected_revision], languages=("en",))
        analysis_run_id = analysis["analysis_run_id"]
        analysis_metadata = project.analysis_run_metadata(analysis_run_id)
        analysis_run = cast(dict[str, object], analysis_metadata["analysis_run"])
        analysis_scope_id = str(analysis_run["scope_id"])
        analysis_revision_ids = cast(
            list[str], analysis_metadata["revision_ids"]
        )
        finding_ids = analysis["finding_ids"]
        evidence_texts: list[str] = []
        for finding_id in finding_ids:
            if not project.validate_finding(finding_id).valid:
                raise RuntimeError(f"deterministic finding {finding_id} failed validation")
            for evidence in project.finding_evidence(finding_id):
                span_id = evidence["span_id"]
                if not span_id:
                    raise RuntimeError(f"finding {finding_id} has non-textual evidence")
                evidence_texts.append(project.span_text(str(span_id)))

        if analysis_revision_ids != [selected_revision]:
            raise RuntimeError("deterministic analysis escaped its selected revision scope")

        bundle_manifest = project.export_research_bundle(analysis_run_id, bundle_directory)
        verification = verify_research_bundle(bundle_manifest)
        if verification["status"] != "VALID" or verification["integrity"] != "VERIFIED_SHA256":
            raise RuntimeError("exported research bundle did not pass integrity verification")

        return {
            "output_directory": str(output_directory),
            "project_directory": str(project_directory),
            "corpus_scope_id": corpus_scope_id,
            "corpus_scope_revision_ids": corpus_scope_revision_ids,
            "migration_frequency": int(frequencies.get("migration", 0)),
            "migration_kwic": [
                {
                    "revision_id": row["revision_id"],
                    "start_offset": row["start_offset"],
                    "end_offset": row["end_offset"],
                    "left": row["left"],
                    "keyword": row["keyword"],
                    "right": row["right"],
                }
                for row in kwic_rows
            ],
            "analysis_run_id": analysis_run_id,
            "analysis_scope_id": analysis_scope_id,
            "analysis_revision_ids": analysis_revision_ids,
            "finding_ids": finding_ids,
            "evidence_texts": evidence_texts,
            "bundle_manifest": str(bundle_manifest),
            "verification": verification,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_directory",
        type=Path,
        help="new or empty directory for the project and verified research bundle",
    )
    arguments = parser.parse_args(argv)
    try:
        summary = run_offline_workflow(arguments.output_directory)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
