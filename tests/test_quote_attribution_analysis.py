import pytest

from discourselens import Project
from discourselens.analysis import run_quote_attribution_candidate_analysis
from discourselens.errors import ProjectError


def test_quote_attribution_analysis_persists_scoped_run_and_candidates(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text(
        "speech.txt", "The minister said, “We will return.” The report noted the statement."
    )

    result = run_quote_attribution_candidate_analysis(
        project, [revision], revision_languages={revision: "en"}
    )

    assert result["analysis_run_id"]
    assert result["scope_snapshot_id"]
    assert result["candidates"]
    assert result["attribution_candidate_count"] == 1
    assert all(item["quote_evidence_id"] in result["evidence_ids"] for item in result["candidates"])
    run = project.db.execute(
        "SELECT scope_id,status FROM analysis_runs WHERE id=?", (result["analysis_run_id"],)
    ).fetchone()
    assert run["scope_id"] == result["scope_snapshot_id"]
    assert run["status"] == "COMPLETED"
    assert project.scope_revisions(run["scope_id"]) == [revision]


def test_quote_attribution_without_language_only_inventories_quote_marks(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("speech.txt", 'The minister said, "We will return."')

    result = run_quote_attribution_candidate_analysis(project, [revision])

    assert result["candidate_count"] == 1
    assert result["attribution_candidate_count"] == 0
    assert result["candidates"][0]["attribution_cues"] == []


@pytest.mark.parametrize(
    "revision_ids,languages",
    [([], None), (["missing-revision"], None), (["revision-a", "revision-a"], None)],
)
def test_quote_attribution_rejects_invalid_scope_before_creating_run(
    tmp_path, revision_ids, languages
):
    project = Project.init(tmp_path / "project")
    with pytest.raises(ProjectError):
        run_quote_attribution_candidate_analysis(
            project, revision_ids, revision_languages=languages
        )
    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


def test_quote_attribution_requires_complete_supported_language_map(tmp_path):
    project = Project.init(tmp_path / "project")
    _, first = project.import_text("first.txt", 'A said “one”.')
    _, second = project.import_text("second.txt", 'B said “two”.')

    with pytest.raises(ProjectError, match="cover exactly"):
        run_quote_attribution_candidate_analysis(
            project, [first, second], revision_languages={first: "en"}
        )
    with pytest.raises(ProjectError, match="only en and es"):
        run_quote_attribution_candidate_analysis(project, [first], revision_languages={first: "ar"})

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
