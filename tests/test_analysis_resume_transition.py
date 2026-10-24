import json

import pytest

from discourselens import Project
from discourselens import analysis as analysis_module
from discourselens.analysis import resume_controlled_cda_review, run_controlled_cda_review
from discourselens.interpretation import review_finding_with_model
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
)
from discourselens.types import EvidenceRole


class _FailingProvider:
    provider_id = "openai-compatible"

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise RuntimeError("simulated review failure")


class _InterruptingProvider(_FailingProvider):
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise KeyboardInterrupt("simulated review interruption")


class _EvidenceEchoProvider:
    provider_id = "openai-compatible"

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = json.loads(request.prompt.split("UNTRUSTED_RESEARCH_DATA_JSON:\n", 1)[1])
        record = payload["evidence_records"][0]
        response: dict[str, object] = {
            "interpretation": "The construction is interpreted from the cited passage.",
            "alternative_interpretation": "The wording may be read another way.",
            "evidence_ids": [record["evidence_id"]],
            "limitations": ["A single passage does not establish prevalence."],
        }
        if "source_findings" not in payload:
            response["observations"] = [
                {
                    "evidence_id": record["evidence_id"],
                    "excerpt": record["excerpt"],
                    "description": "The passage contains the cited construction.",
                }
            ]
        return GenerationResponse(
            json.dumps(response),
            self.provider_id,
            request.model,
        )


def test_resume_transition_interruption_is_recorded_failed_and_remains_resumable(
    tmp_path, monkeypatch
):
    with Project.init(tmp_path / "resume-transition") as project:
        document_id, revision_id = project.import_text("selected.txt", "We must act.")
        project.set_document_metadata(document_id, "language", "en")
        project.save_credential_reference(
            CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
        )
        project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))

        with pytest.raises(RuntimeError, match="simulated review failure"):
            run_controlled_cda_review(
                project,
                [revision_id],
                "review",
                frameworks=("fairclough-text",),
                provider_factory=lambda endpoint, reference, provider_id: _FailingProvider(),
            )
        run_id = str(
            project.db.execute("SELECT id FROM analysis_runs WHERE status='FAILED'").fetchone()[
                "id"
            ]
        )

        resume_run = project.resume_run

        def interrupt_after_transition(analysis_run_id: str) -> None:
            resume_run(analysis_run_id)
            raise KeyboardInterrupt("simulated interruption after RUNNING transition")

        monkeypatch.setattr(project, "resume_run", interrupt_after_transition)
        with pytest.raises(KeyboardInterrupt, match="after RUNNING transition"):
            resume_controlled_cda_review(project, run_id)

        assert (
            project.db.execute("SELECT status FROM analysis_runs WHERE id=?", (run_id,)).fetchone()[
                "status"
            ]
            == "FAILED"
        )

        monkeypatch.setattr(project, "resume_run", resume_run)
        result = resume_controlled_cda_review(
            project,
            run_id,
            provider_factory=lambda endpoint, reference, provider_id: _EvidenceEchoProvider(),
        )
        assert result["analysis_run_id"] == run_id
        assert result["synthesis_finding_id"] is not None
        assert (
            project.db.execute("SELECT status FROM analysis_runs WHERE id=?", (run_id,)).fetchone()[
                "status"
            ]
            == "COMPLETED"
        )


def test_initial_run_start_interruption_is_persisted_as_failed(tmp_path, monkeypatch):
    with Project.init(tmp_path / "initial-run-transition") as project:
        _, revision_id = project.import_text("selected.txt", "We must act.")
        start_run = project.start_run

        def interrupt_after_transition(analysis_run_id: str) -> None:
            start_run(analysis_run_id)
            raise KeyboardInterrupt("simulated interruption after RUNNING transition")

        monkeypatch.setattr(project, "start_run", interrupt_after_transition)
        with pytest.raises(KeyboardInterrupt, match="after RUNNING transition"):
            analysis_module.run_exact_phrase_analysis(project, [revision_id], "must")

        run = project.db.execute("SELECT status FROM analysis_runs").fetchone()
        assert run["status"] == "FAILED"


def test_execution_keyboard_interrupt_is_persisted_as_failed(tmp_path, monkeypatch):
    with Project.init(tmp_path / "execution-interruption") as project:
        _, revision_id = project.import_text("selected.txt", "We must act.")

        def interrupt_during_execution(*args, **kwargs):
            raise KeyboardInterrupt("simulated execution interruption")

        monkeypatch.setattr(analysis_module, "run_exact_phrase_fixture", interrupt_during_execution)
        with pytest.raises(KeyboardInterrupt, match="execution interruption"):
            analysis_module.run_exact_phrase_analysis(project, [revision_id], "must")

        run = project.db.execute("SELECT status FROM analysis_runs").fetchone()
        assert run["status"] == "FAILED"


def test_standalone_interpretation_interruption_is_persisted_as_failed(tmp_path):
    with Project.init(tmp_path / "standalone-interpretation-interruption") as project:
        _, revision_id = project.import_text("selected.txt", "A bounded source passage.")
        source_run_id = project.create_run(
            project.snapshot([revision_id]), {"profile": "source"}
        )
        evidence_id = project.textual_evidence(
            project.create_span(revision_id, 0, len("A bounded source passage."))
        )
        finding_id = project.create_finding(
            source_run_id,
            "fixture.source",
            "1.0.0",
            "source observation",
            "The source contains a bounded passage.",
        )
        project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
        project.finalize_run(source_run_id)
        project.save_credential_reference(
            CredentialReference(
                "openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1"
            )
        )
        project.save_model_preset(
            ModelPreset("review", "openai-compatible", "fixture", "fixture")
        )

        with pytest.raises(KeyboardInterrupt, match="simulated review interruption"):
            review_finding_with_model(
                project,
                finding_id,
                "review",
                provider_factory=lambda endpoint, reference, provider_id: _InterruptingProvider(),
            )

        review_run = project.db.execute(
            "SELECT id,status FROM analysis_runs WHERE id<>?", (source_run_id,)
        ).fetchone()
        assert review_run is not None
        assert review_run["status"] == "FAILED"
        skill_run = project.db.execute(
            "SELECT status,error_type FROM skill_runs WHERE analysis_run_id=?",
            (review_run["id"],),
        ).fetchone()
        assert skill_run["status"] == "FAILED"
        assert skill_run["error_type"] == "KeyboardInterrupt"
        assert project.db.execute(
            "SELECT status FROM analysis_runs WHERE id=?", (source_run_id,)
        ).fetchone()["status"] == "COMPLETED"


def test_standalone_interpretation_skill_start_interruption_is_persisted_as_failed(
    tmp_path, monkeypatch
):
    with Project.init(tmp_path / "standalone-skill-start-interruption") as project:
        _, revision_id = project.import_text("selected.txt", "A bounded source passage.")
        source_run_id = project.create_run(project.snapshot([revision_id]), {"profile": "source"})
        evidence_id = project.textual_evidence(
            project.create_span(revision_id, 0, len("A bounded source passage."))
        )
        finding_id = project.create_finding(
            source_run_id,
            "fixture.source",
            "1.0.0",
            "source observation",
            "The source contains a bounded passage.",
        )
        project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
        project.finalize_run(source_run_id)
        project.save_credential_reference(
            CredentialReference(
                "openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1"
            )
        )
        project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))

        create_skill_run = project.create_skill_run

        def interrupt_after_skill_start(*args, **kwargs):
            create_skill_run(*args, **kwargs)
            raise KeyboardInterrupt("simulated interruption after SkillRun creation")

        monkeypatch.setattr(project, "create_skill_run", interrupt_after_skill_start)
        with pytest.raises(KeyboardInterrupt, match="after SkillRun creation"):
            review_finding_with_model(
                project,
                finding_id,
                "review",
                provider_factory=lambda endpoint, reference, provider_id: _EvidenceEchoProvider(),
            )

        review_run = project.db.execute(
            "SELECT id,status FROM analysis_runs WHERE id<>?", (source_run_id,)
        ).fetchone()
        assert review_run is not None
        assert review_run["status"] == "FAILED"
        skill_run = project.db.execute(
            "SELECT status,error_type FROM skill_runs WHERE analysis_run_id=?",
            (review_run["id"],),
        ).fetchone()
        assert skill_run["status"] == "FAILED"
        assert skill_run["error_type"] == "KeyboardInterrupt"
