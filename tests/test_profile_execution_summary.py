import json
import re

import pytest

import discourselens.analysis as analysis_module
import discourselens.nlp as nlp_module
import discourselens.profiles as profiles_module
from discourselens import Project
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
)


class _Token:
    def __init__(self, surface: str, index: int, start: int, end: int) -> None:
        self.surface = surface
        self.lemma = surface.casefold()
        self.pos = "PROPN" if surface.casefold() in {"minister", "gobierno"} else "VERB"
        self.morphology = None
        self.dependency_head = None if index == 0 else 0
        self.dependency_relation = "root" if index == 0 else "dep"
        self.char_start = start
        self.char_end = end


class _FixtureBackend:
    backend_id = "spacy"

    def __init__(self, language: str, model: str) -> None:
        self.language = language
        self.model_id = model

    def parse(self, text: str) -> list[_Token]:
        return [
            _Token(match.group(), index, match.start(), match.end())
            for index, match in enumerate(re.finditer(r"\w+|[^\w\s]", text))
        ]


class _FixtureProvider:
    provider_id = "openai-compatible"

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = json.loads(request.prompt.split("UNTRUSTED_RESEARCH_DATA_JSON:\n", 1)[1])
        records = payload["evidence_records"]
        response = {
            "interpretation": "Evidence-bound fixture reading.",
            "alternative_interpretation": "A plausible alternative fixture reading.",
            "evidence_ids": [records[0]["evidence_id"]],
        }
        if "source_findings" not in payload:
            response["observations"] = [
                {
                    "evidence_id": records[0]["evidence_id"],
                    "excerpt": records[0]["excerpt"],
                    "description": "The passage contains an observable wording feature.",
                }
            ]
        return GenerationResponse(
            json.dumps(response),
            self.provider_id,
            request.model,
        )


class _FailOnceProvider(_FixtureProvider):
    def __init__(self) -> None:
        self.failed = False

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.failed:
            self.failed = True
            raise RuntimeError("fixture transient failure")
        return super().generate(request)


def _prepare_project(project: Project) -> tuple[str, str]:
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "fixture", "DL_FIXTURE_KEY", "http://fixture.test/v1"
        )
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    english_document, english_revision = project.import_text(
        "english.txt", "The Minister must act today."
    )
    spanish_document, spanish_revision = project.import_text(
        "spanish.txt", "El gobierno no debe actuar."
    )
    project.set_document_metadata(english_document, "language", "en")
    project.set_document_metadata(spanish_document, "language", "es")
    project.save_analysis_profile(
        "mixed-local-candidates",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en", "es"],
            "default_preset": "review",
            "nlp_routes": {
                "en": {"backend": "spacy", "model": "fixture-en"},
                "es": {"backend": "spacy", "model": "fixture-es"},
            },
            "local_preprocessors": ["social_actor_candidates"],
        },
    )
    return english_revision, spanish_revision


def _mixed_preflight(languages, required_capabilities, *, routes=None):
    assert required_capabilities == {"pos"}
    assert routes is not None
    return {
        language: {
            "runnable": language == "en",
            "limitations": [] if language == "en" else ["pos: UNASSESSED"],
            "required_capabilities": ["pos"],
            "statuses": {"pos": "SUPPORTED" if language == "en" else "UNASSESSED"},
            "backend": routes[language]["backend"],
            "model": routes[language]["model"],
            "model_dir_env": None,
            "route_source": "profile",
        }
        for language in languages
    }


def _install_local_fixtures(monkeypatch) -> None:
    monkeypatch.setattr(profiles_module, "local_execution_preflight", _mixed_preflight)
    monkeypatch.setattr(
        nlp_module,
        "load_spacy",
        lambda language, model: _FixtureBackend(language, model),
    )


def test_mixed_language_execution_summary_is_safe_and_matches_after_resume(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "project")
    english_revision, spanish_revision = _prepare_project(project)
    _install_local_fixtures(monkeypatch)
    revisions = [english_revision, spanish_revision]

    result = analysis_module.run_saved_controlled_profile(
        project,
        revisions,
        "mixed-local-candidates",
        provider_factory=lambda *_args, **_kwargs: _FixtureProvider(),
    )

    expected = {
        "status": "PARTIAL",
        "operations": {
            "social_actor_candidates": {
                "status": "PARTIAL",
                "required_capability": "pos",
                "processed_revision_ids": [english_revision],
                "skipped_revision_ids": [spanish_revision],
                "skipped_revisions": {
                    spanish_revision: {
                        "language": "es",
                        "capability": "pos",
                        "status": "UNASSESSED",
                        "limitations": ["pos: UNASSESSED"],
                    }
                },
            }
        },
    }
    assert result["local_nlp_execution_summary"] == expected
    first_run = project.db.execute(
        "SELECT scope_id FROM analysis_runs WHERE id=?", (result["analysis_run_id"],)
    ).fetchone()
    assert project.scope_revisions(first_run["scope_id"]) == revisions

    failing_provider = _FailOnceProvider()
    with pytest.raises(RuntimeError, match="fixture transient failure"):
        analysis_module.run_saved_controlled_profile(
            project,
            revisions,
            "mixed-local-candidates",
            provider_factory=lambda *_args, **_kwargs: failing_provider,
        )
    failed_run = project.db.execute(
        "SELECT id,scope_id,config_json FROM analysis_runs WHERE status='FAILED' "
        "ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    saved_config = json.loads(failed_run["config_json"])
    assert saved_config["local_nlp_execution_plan"]["status"] == "PARTIAL"
    assert project.scope_revisions(failed_run["scope_id"]) == revisions

    resumed = analysis_module.resume_controlled_cda_review(
        project,
        failed_run["id"],
        provider_factory=lambda *_args, **_kwargs: failing_provider,
    )

    assert resumed["local_nlp_execution_summary"] == result["local_nlp_execution_summary"]
    resumed_run = project.db.execute(
        "SELECT scope_id FROM analysis_runs WHERE id=?", (failed_run["id"],)
    ).fetchone()
    assert project.scope_revisions(resumed_run["scope_id"]) == revisions
    serialized_summary = json.dumps(resumed["local_nlp_execution_summary"], sort_keys=True)
    assert "DL_FIXTURE_KEY" not in serialized_summary
    assert "fixture.test" not in serialized_summary


def test_pipeline_reports_not_configured_when_profile_has_no_local_preprocessors(
    tmp_path,
):
    project = Project.init(tmp_path / "project")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "fixture", "DL_FIXTURE_KEY", "http://fixture.test/v1"
        )
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    document_id, revision_id = project.import_text("english.txt", "The Minister must act.")
    project.set_document_metadata(document_id, "language", "en")
    project.save_analysis_profile(
        "lexical-only",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "default_preset": "review",
        },
    )

    result = analysis_module.run_saved_controlled_profile(
        project,
        [revision_id],
        "lexical-only",
        provider_factory=lambda *_args, **_kwargs: _FixtureProvider(),
    )

    assert result["local_nlp_execution_summary"] == {
        "status": "NOT_CONFIGURED",
        "operations": {},
    }


def test_execution_summary_redacts_route_diagnostics():
    plan = analysis_module._build_local_nlp_execution_plan(
        ["revision-es"],
        {"revision-es": "es"},
        {
            "es": {
                "statuses": {"pos": "UNASSESSED"},
                "limitations": [
                    "pos: UNASSESSED",
                    "local NLP route: DL_PRIVATE_HOME=/private/model token=secret-value",
                ],
            }
        },
        ("social_actor_candidates",),
    )

    summary = analysis_module._local_nlp_execution_summary(plan)
    serialized_summary = json.dumps(summary, sort_keys=True)

    assert "DL_PRIVATE_HOME" not in serialized_summary
    assert "/private/model" not in serialized_summary
    assert "secret-value" not in serialized_summary
    assert summary["operations"]["social_actor_candidates"]["skipped_revisions"][
        "revision-es"
    ]["limitations"] == [
        "pos: UNASSESSED",
        "local NLP route: configured route was unavailable or incomplete",
    ]
