import pytest

from discourselens import Project
from discourselens.bundle import export_bundle
from discourselens.errors import DocumentError, EvidenceValidationError, ProjectError


def test_finding_creation_rejects_skill_run_from_another_analysis_run(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        _, revision_id = project.import_text("source.txt", "A bounded source passage.")
        scope_id = project.snapshot([revision_id])
        source_run_id = project.create_run(scope_id, {"profile": "source"})
        finding_run_id = project.create_run(scope_id, {"profile": "finding"})
        skill_run_id = project.create_skill_run(
            source_run_id,
            scope_id,
            "fixture.skill",
            "1.0.0",
            "fixture.executor",
            "fixture-config-hash",
        )

        with pytest.raises(EvidenceValidationError, match="skill run provenance"):
            project.create_finding(
                finding_run_id,
                "fixture.skill",
                "1.0.0",
                "source observation",
                "The source contains a bounded passage.",
                skill_run_id=skill_run_id,
            )

        matching_skill_run_id = project.create_skill_run(
            finding_run_id,
            scope_id,
            "fixture.skill",
            "1.0.0",
            "fixture.executor",
            "fixture-config-hash",
        )
        finding_id = project.create_finding(
            finding_run_id,
            "fixture.skill",
            "1.0.0",
            "source observation",
            "The source contains a bounded passage.",
            skill_run_id=matching_skill_run_id,
        )

        assert project.get_finding(finding_id)["skill_run_id"] == matching_skill_run_id
    finally:
        project.close()


def test_model_execution_rejects_skill_run_from_another_analysis_run(tmp_path):
    with Project.init(tmp_path / "project") as project:
        _, revision_id = project.import_text("source.txt", "A bounded source passage.")
        scope_id = project.snapshot([revision_id])
        source_run_id = project.create_run(scope_id, {"profile": "source"})
        execution_run_id = project.create_run(scope_id, {"profile": "execution"})
        skill_run_id = project.create_skill_run(
            source_run_id,
            scope_id,
            "fixture.skill",
            "1.0.0",
            "fixture.executor",
            "fixture-config-hash",
        )

        with pytest.raises(DocumentError, match="same analysis run"):
            project.record_model_execution(
                provider_id="fixture-provider",
                credential_alias="fixture",
                model="fixture-model",
                reasoning=None,
                request_hash="request-hash",
                response_hash="response-hash",
                analysis_run_id=execution_run_id,
                skill_run_id=skill_run_id,
            )

        execution_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash="response-hash",
            analysis_run_id=source_run_id,
            skill_run_id=skill_run_id,
        )
        assert project.db.execute(
            "SELECT skill_run_id FROM model_executions WHERE id=?", (execution_id,)
        ).fetchone()[0] == skill_run_id


@pytest.mark.parametrize("invalid_link", ["dangling", "cross-run"])
def test_model_execution_exports_reject_corrupt_skill_run_links(tmp_path, invalid_link):
    with Project.init(tmp_path / f"corrupt-{invalid_link}") as project:
        _, revision_id = project.import_text("source.txt", "A bounded source passage.")
        scope_id = project.snapshot([revision_id])
        analysis_run_id = project.create_run(scope_id, {"profile": "execution"})
        foreign_run_id = project.create_run(scope_id, {"profile": "foreign"})
        skill_run_id = project.create_skill_run(
            analysis_run_id,
            scope_id,
            "fixture.skill",
            "1.0.0",
            "fixture.executor",
            "fixture-config-hash",
        )
        foreign_skill_run_id = project.create_skill_run(
            foreign_run_id,
            scope_id,
            "fixture.skill",
            "1.0.0",
            "fixture.executor",
            "fixture-config-hash",
        )
        execution_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash="response-hash",
            analysis_run_id=analysis_run_id,
            skill_run_id=skill_run_id,
        )
        invalid_skill_run_id = (
            "missing-skill-run" if invalid_link == "dangling" else foreign_skill_run_id
        )
        project.db.execute(
            "UPDATE model_executions SET skill_run_id=? WHERE id=?",
            (invalid_skill_run_id, execution_id),
        )
        project.db.commit()

        for summary in (
            lambda: project.analysis_run_summary(analysis_run_id),
            lambda: project.analysis_run_summary_page(analysis_run_id, limit=1),
        ):
            with pytest.raises(DocumentError, match="model execution skill run reference"):
                summary()

        destination = tmp_path / f"bundle-{invalid_link}"
        with pytest.raises(
            ProjectError,
            match="research bundle model execution references an invalid skill run",
        ):
            export_bundle(project, analysis_run_id, destination)
        assert not destination.exists()
