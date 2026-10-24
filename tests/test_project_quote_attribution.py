from collections.abc import Mapping

import pytest

from discourselens import Project
from discourselens.errors import ProjectError


def test_project_quote_attribution_uses_explicit_scope_and_persists_run_lineage(tmp_path):
    project = Project.init(tmp_path / "project")
    _, english = project.import_text("english.txt", 'The minister said, “We will return.”')
    _, spanish = project.import_text("spanish.txt", 'La ministra dijo: «Volveremos pronto.»')
    project.import_text("outside.txt", 'Officials said “Leave now.”')

    languages = {english: "en", spanish: "es"}
    result = project.analyze_quote_attribution_candidates(
        [english, spanish], revision_languages=languages
    )

    assert result["candidate_count"] == 2
    candidates = {candidate["revision_id"]: candidate for candidate in result["candidates"]}
    assert set(candidates) == {english, spanish}
    assert candidates[english]["attribution_cues"] == ["said"]
    assert candidates[english]["attribution_cue_languages"] == ["en"]
    assert candidates[spanish]["attribution_cues"] == ["dijo"]
    assert candidates[spanish]["attribution_cue_languages"] == ["es"]

    metadata = project.analysis_run_metadata(result["analysis_run_id"])
    run = metadata["analysis_run"]
    assert isinstance(run, Mapping)
    assert metadata["revision_ids"] == [english, spanish]
    assert run["scope_id"] == result["scope_snapshot_id"]
    assert run["status"] == "COMPLETED"
    run_config = run["config"]
    assert isinstance(run_config, Mapping)
    assert run_config["revision_languages"] == languages
    assert project.scope_revisions(result["scope_snapshot_id"]) == [english, spanish]

    skill_run = project.db.execute(
        "SELECT analysis_run_id,scope_id,status FROM skill_runs WHERE id=?",
        (result["skill_run_id"],),
    ).fetchone()
    assert skill_run["analysis_run_id"] == result["analysis_run_id"]
    assert skill_run["scope_id"] == result["scope_snapshot_id"]
    assert skill_run["status"] == "COMPLETED"
    finding = project.get_finding(result["finding_ids"][0])
    assert finding["analysis_run_id"] == result["analysis_run_id"]


@pytest.mark.parametrize("invalid_scope", ["empty", "duplicate", "unknown"])
def test_project_quote_attribution_rejects_invalid_revision_scope_without_run(
    tmp_path, invalid_scope
):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("valid.txt", 'The minister said, “We will return.”')
    revision_ids = {
        "empty": [],
        "duplicate": [revision, revision],
        "unknown": ["missing-revision"],
    }[invalid_scope]

    with pytest.raises(ProjectError):
        project.analyze_quote_attribution_candidates(revision_ids)

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
