import json
from collections.abc import Mapping

import pytest

from discourselens import Project
from discourselens.errors import ProjectError, ScopeViolationError, SkillManifestError
from discourselens.skills import VanDijkGroupComparisonGroupInput


def test_project_group_comparison_persists_scope_and_selected_annotation_lineage(tmp_path):
    project = Project.init(tmp_path / "project")
    _, resident_revision = project.import_text(
        "residents.txt", "Residents described the road as unsafe."
    )
    _, official_revision = project.import_text(
        "officials.txt", "Officials announced a safety review."
    )
    _, outside_revision = project.import_text("outside.txt", "The issue remains unresolved.")

    codebook_id = project.create_codebook("Road safety", "Researcher controlled codes")
    concern_code = project.create_code(
        codebook_id, "Negative evaluation", "A negative evaluation in the selected span."
    )
    action_code = project.create_code(
        codebook_id, "Institutional action", "An institution is described acting."
    )
    resident_text = "Residents described the road as unsafe."
    official_text = "Officials announced a safety review."
    resident_annotation = project.annotate_span(
        concern_code,
        project.create_span(resident_revision, 0, len(resident_text)),
    )
    unselected_annotation = project.annotate_span(
        action_code,
        project.create_span(resident_revision, 0, len(resident_text)),
        memo="present in scope, not selected",
    )
    official_annotation = project.annotate_span(
        action_code,
        project.create_span(official_revision, 0, len(official_text)),
    )
    outside_annotation = project.annotate_span(
        concern_code,
        project.create_span(outside_revision, 0, len("The issue remains unresolved.")),
    )

    result = project.prepare_van_dijk_group_comparison(
        [resident_revision, official_revision],
        groups=[
            {"label": "Resident accounts", "annotation_ids": [resident_annotation]},
            {"label": "Official statements", "annotation_ids": [official_annotation]},
        ],
        shared_topic_event="the road safety announcement",
        comparability_basis="both groups address the same announced safety review",
        codebook_id=codebook_id,
    )

    run_metadata = project.analysis_run_metadata(result["analysis_run_id"])
    analysis_run = run_metadata["analysis_run"]
    assert isinstance(analysis_run, Mapping)
    run_config = analysis_run["config"]
    assert isinstance(run_config, Mapping)
    assert run_metadata["revision_ids"] == [resident_revision, official_revision]
    assert analysis_run["scope_id"] == result["scope_snapshot_id"]
    assert analysis_run["status"] == "COMPLETED"
    assert run_config["groups"] == [
        {"label": "Resident accounts", "annotation_ids": [resident_annotation]},
        {"label": "Official statements", "annotation_ids": [official_annotation]},
    ]
    assert run_config["shared_topic_event"] == "the road safety announcement"
    assert run_config["comparability_basis"] == (
        "both groups address the same announced safety review"
    )
    assert run_config["codebook_id"] == codebook_id

    skill_run = project.db.execute(
        "SELECT analysis_run_id,scope_id,status FROM skill_runs WHERE id=?",
        (result["skill_run_id"],),
    ).fetchone()
    assert skill_run["analysis_run_id"] == result["analysis_run_id"]
    assert skill_run["scope_id"] == result["scope_snapshot_id"]
    assert skill_run["status"] == "COMPLETED"

    selected_ids = [resident_annotation, official_annotation]
    output_annotations = [
        annotation
        for group in result["groups"]
        for annotation in group["annotations"]
    ]
    assert [annotation["annotation_id"] for annotation in output_annotations] == selected_ids
    assert unselected_annotation not in {
        annotation["annotation_id"] for annotation in output_annotations
    }
    assert outside_annotation not in {
        annotation["annotation_id"] for annotation in output_annotations
    }
    assert [annotation["text"] for annotation in output_annotations] == [
        resident_text,
        official_text,
    ]
    assert [group["selected_annotation_count"] for group in result["groups"]] == [1, 1]

    finding = project.get_finding(result["finding_id"])
    assert finding["analysis_run_id"] == result["analysis_run_id"]
    assert finding["skill_run_id"] == result["skill_run_id"]
    assert json.loads(finding["claim_types_json"]) == ["EXAMPLE_BASED"]
    assert project.validate_finding(result["finding_id"]).valid

    attached_evidence = project.finding_evidence(result["finding_id"])
    assert {row["id"] for row in attached_evidence} == set(result["evidence_ids"])
    assert {row["role"] for row in attached_evidence} == {"EXAMPLE"}
    assert len(attached_evidence) == len(selected_ids)
    evidence_by_id = {row["id"]: row for row in attached_evidence}
    for group in result["groups"]:
        for annotation in group["annotations"]:
            evidence = evidence_by_id[annotation["evidence_id"]]
            assert annotation["evidence_id"] == evidence["id"]
            assert project.span_text(evidence["span_id"]) == annotation["text"]
            assert project.evidence_content(evidence["id"]) == {
                "annotation_id": annotation["annotation_id"],
                "codebook_id": codebook_id,
                "codebook_name": "Road safety",
                "code_id": annotation["code_id"],
                "code_label": annotation["code_label"],
                "comparability_basis": "both groups address the same announced safety review",
                "group_label": group["label"],
                "scope_snapshot_id": result["scope_snapshot_id"],
                "shared_topic_event": "the road safety announcement",
                "skill_id": "builtin.van-dijk-group-comparison",
                "skill_version": "1.0.0",
            }


@pytest.mark.parametrize(
    ("revision_ids", "message"),
    [
        ([], "explicit revision scope"),
        (["known", "known"], "must not contain duplicates"),
        (["missing-revision"], "unknown document revision"),
    ],
)
def test_project_group_comparison_rejects_invalid_revision_scope_before_run(
    tmp_path, revision_ids, message
):
    project = Project.init(tmp_path / "project")
    _, known_revision = project.import_text("known.txt", "A deterministic fixture span.")
    codebook_id = project.create_codebook("Fixture codebook")
    code_id = project.create_code(codebook_id, "Fixture code")
    annotation_id = project.annotate_span(
        code_id,
        project.create_span(known_revision, 0, len("A deterministic fixture span.")),
    )
    # Use a nonempty selection so empty-scope validation is the reason for rejection.
    groups: list[VanDijkGroupComparisonGroupInput] = [
        {"label": "A", "annotation_ids": [annotation_id]},
        {"label": "B", "annotation_ids": []},
    ]

    with pytest.raises(ProjectError, match=message):
        project.prepare_van_dijk_group_comparison(
            revision_ids,
            groups=groups,
            shared_topic_event="fixture event",
            comparability_basis="same fixture event",
            codebook_id=codebook_id,
        )

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM skill_runs").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM findings").fetchone()[0] == 0


def test_project_group_comparison_rejects_annotation_outside_requested_revisions(tmp_path):
    project = Project.init(tmp_path / "project")
    _, included_revision = project.import_text("included.txt", "Included source span.")
    _, excluded_revision = project.import_text("excluded.txt", "Excluded source span.")
    codebook_id = project.create_codebook("Fixture codebook")
    code_id = project.create_code(codebook_id, "Fixture code")
    included_annotation = project.annotate_span(
        code_id,
        project.create_span(included_revision, 0, len("Included source span.")),
    )
    excluded_annotation = project.annotate_span(
        code_id,
        project.create_span(excluded_revision, 0, len("Excluded source span.")),
    )

    with pytest.raises(ScopeViolationError, match="original ScopeSnapshot"):
        project.prepare_van_dijk_group_comparison(
            [included_revision],
            groups=[
                {"label": "Included", "annotation_ids": [included_annotation]},
                {"label": "Excluded", "annotation_ids": [excluded_annotation]},
            ],
            shared_topic_event="fixture event",
            comparability_basis="same fixture event",
            codebook_id=codebook_id,
        )

    failed_run = project.db.execute(
        "SELECT id,scope_id,status FROM analysis_runs"
    ).fetchone()
    assert failed_run is not None
    assert failed_run["status"] == "FAILED"
    assert project.scope_revisions(failed_run["scope_id"]) == [included_revision]
    assert project.db.execute("SELECT COUNT(*) FROM skill_runs").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM findings").fetchone()[0] == 0


def test_project_group_comparison_rejects_unknown_and_duplicate_annotation_ids(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "A deterministic coded span.")
    codebook_id = project.create_codebook("Fixture codebook")
    code_id = project.create_code(codebook_id, "Fixture code")
    annotation_id = project.annotate_span(
        code_id,
        project.create_span(revision, 0, len("A deterministic coded span.")),
    )

    with pytest.raises(SkillManifestError, match="duplicate annotation ID"):
        project.prepare_van_dijk_group_comparison(
            [revision],
            groups=[
                {"label": "A", "annotation_ids": [annotation_id, annotation_id]},
                {"label": "B", "annotation_ids": []},
            ],
            shared_topic_event="fixture event",
            comparability_basis="same fixture event",
            codebook_id=codebook_id,
        )

    failed_run = project.db.execute(
        "SELECT id,scope_id,status FROM analysis_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert failed_run["status"] == "FAILED"
    assert project.scope_revisions(failed_run["scope_id"]) == [revision]
    assert project.db.execute("SELECT COUNT(*) FROM skill_runs").fetchone()[0] == 0

    with pytest.raises(ScopeViolationError, match="original ScopeSnapshot"):
        project.prepare_van_dijk_group_comparison(
            [revision],
            groups=[
                {"label": "A", "annotation_ids": ["unknown-annotation"]},
                {"label": "B", "annotation_ids": []},
            ],
            shared_topic_event="fixture event",
            comparability_basis="same fixture event",
            codebook_id=codebook_id,
        )

    runs = project.db.execute(
        "SELECT scope_id,status FROM analysis_runs ORDER BY started_at, id"
    ).fetchall()
    assert len(runs) == 2
    assert all(run["status"] == "FAILED" for run in runs)
    assert all(project.scope_revisions(run["scope_id"]) == [revision] for run in runs)
    assert project.db.execute("SELECT COUNT(*) FROM skill_runs").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM findings").fetchone()[0] == 0


def test_project_group_comparison_rejects_all_empty_groups_before_run(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "No annotations selected.")
    codebook_id = project.create_codebook("Fixture codebook")

    with pytest.raises(ProjectError, match="at least one selected annotation"):
        project.prepare_van_dijk_group_comparison(
            [revision],
            groups=[
                {"label": "A", "annotation_ids": []},
                {"label": "B", "annotation_ids": []},
            ],
            shared_topic_event="fixture event",
            comparability_basis="same fixture event",
            codebook_id=codebook_id,
        )

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM skill_runs").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM findings").fetchone()[0] == 0
