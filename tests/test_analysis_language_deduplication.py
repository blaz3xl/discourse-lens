import json

from discourselens import Project
from discourselens.analysis import run_batch_textual_descriptors_analysis


def _analysis_config(project: Project, analysis_run_id: str) -> dict[str, object]:
    row = project.db.execute(
        "SELECT config_json FROM analysis_runs WHERE id=?", (analysis_run_id,)
    ).fetchone()
    assert row is not None
    return json.loads(row["config_json"])


def _finding_evidence_texts(project: Project, finding_ids: list[str]) -> list[list[str]]:
    return [
        [project.span_text(str(row["span_id"])) for row in project.finding_evidence(finding_id)]
        for finding_id in finding_ids
    ]


def test_textual_descriptors_deduplicate_languages_and_preserve_first_order(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision_id = project.import_text("source.txt", "El gobierno debe actuar. We may wait.")

    unique = project.analyze_textual_descriptors([revision_id], languages=("es", "en"))
    repeated = project.analyze_textual_descriptors(
        [revision_id], languages=("es", " en ", "es")
    )

    assert _analysis_config(project, repeated["analysis_run_id"])["languages"] == ["es", "en"]
    assert _finding_evidence_texts(project, repeated["finding_ids"]) == _finding_evidence_texts(
        project, unique["finding_ids"]
    )
    assert _finding_evidence_texts(project, repeated["finding_ids"]) == [["debe"], ["may"]]


def test_textual_descriptors_canonicalize_case_variants(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision_id = project.import_text("source.txt", "Officials must act.")

    result = project.analyze_textual_descriptors([revision_id], languages=("en", "EN"))

    assert _analysis_config(project, result["analysis_run_id"])["languages"] == ["en"]
    assert _finding_evidence_texts(project, result["finding_ids"]) == [["must"]]


def test_evidence_extraction_deduplicates_cue_languages_and_rationale(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision_id = project.import_text("source.txt", "We must act.")

    result = project.extract_evidence([revision_id], languages=("en", "en"))

    assert _analysis_config(project, result["analysis_run_id"])["languages"] == ["en"]
    assert result["candidate_count"] == 1
    evidence = project.finding_evidence(result["finding_ids"][0])
    assert len(evidence) == 1
    assert evidence[0]["rationale"] == (
        "Deterministic candidate cues: en:modality, en:actor_reference"
    )


def test_batch_textual_descriptors_use_deduplicated_language_config(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision_id = project.import_text("source.txt", "Officials must act.")

    results = run_batch_textual_descriptors_analysis(
        project, [revision_id], languages=("en", "en")
    )

    assert len(results) == 1
    result = results[0]
    assert _analysis_config(project, result["analysis_run_id"])["languages"] == ["en"]
    assert _finding_evidence_texts(project, result["finding_ids"]) == [["must"]]
