import json
import re
import traceback
from dataclasses import dataclass
from typing import Any, cast

import pytest

import discourselens.analysis as analysis_module
import discourselens.nlp as nlp_module
import discourselens.profiles as profiles_module
from discourselens import Project
from discourselens.errors import ProjectError
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
)
from discourselens.profiles import ControlledCdaProfile


@dataclass(frozen=True)
class _Token:
    surface: str
    lemma: str
    pos: str
    morphology: str | None
    dependency_head: int | None
    dependency_relation: str | None
    char_start: int
    char_end: int


class _FixtureBackend:
    def __init__(self, backend_id="spacy", model_id="fixture-en", language="en"):
        self.backend_id = backend_id
        self.model_id = model_id
        self.language = language

    def parse(self, text: str) -> list[_Token]:
        tokens = []
        for index, match in enumerate(re.finditer(r"\w+|[^\w\s]", text)):
            surface = match.group()
            tokens.append(
                _Token(
                    surface=surface,
                    lemma=surface.casefold(),
                    pos="PROPN" if surface == "Minister" else "VERB",
                    morphology=None,
                    dependency_head=None if index == 0 else 0,
                    dependency_relation="root" if index == 0 else "dep",
                    char_start=match.start(),
                    char_end=match.end(),
                )
            )
        return tokens


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


def _eligible_preflight(self, languages=None):
    selected = languages or tuple(self.languages)
    return {
        language: {
            "runnable": True,
            "limitations": [],
            "required_capabilities": ["pos"],
            "statuses": {"pos": "SUPPORTED"},
            "backend": self.nlp_routes[language].backend,
            "model": self.nlp_routes[language].model,
            "model_dir_env": self.nlp_routes[language].model_dir_env,
            "route_source": "profile",
        }
        for language in selected
    }


def _saved_profile(
    project,
    *,
    local_preprocessors=None,
    backend="spacy",
    model="fixture-en",
    model_dir_env=None,
    default_preset="review",
):
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "fixture", "DL_FIXTURE_KEY", "http://fixture.test/v1"
        )
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_analysis_profile(
        "local-candidates",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "default_preset": default_preset,
            "nlp_routes": {
                "en": {
                    "backend": backend,
                    "model": model,
                    **({"model_dir_env": model_dir_env} if model_dir_env else {}),
                }
            },
            "local_preprocessors": local_preprocessors or ["social_actor_candidates"],
        },
    )


def test_saved_profile_builds_and_consumes_local_nlp_artifact_in_same_analysis_run(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "project")
    document_id, revision_id = project.import_text("speech.txt", "Minister spoke.")
    project.set_document_metadata(document_id, "language", "en")
    _saved_profile(project)
    monkeypatch.setattr(ControlledCdaProfile, "local_preflight", _eligible_preflight)
    monkeypatch.setattr(nlp_module, "load_spacy", lambda language, model: _FixtureBackend())

    result = analysis_module.run_saved_controlled_profile(
        project, [revision_id], "local-candidates"
    )

    assert result["local_nlp_skill_run_ids"]
    assert result["local_nlp_finding_ids"]
    skill_run = project.db.execute(
        "SELECT analysis_run_id,skill_id,status FROM skill_runs WHERE id=?",
        (result["local_nlp_skill_run_ids"][0],),
    ).fetchone()
    assert skill_run["analysis_run_id"] == result["analysis_run_id"]
    assert skill_run["skill_id"] == "builtin.social-actor-candidates"
    assert skill_run["status"] == "COMPLETED"

    summary = cast(dict[str, Any], project.analysis_run_summary(result["analysis_run_id"]))
    run_config = cast(dict[str, Any], summary["analysis_run"]["config"])
    assert run_config["local_preprocessors"] == ["social_actor_candidates"]
    artifact_ids = run_config["local_nlp_artifacts"][revision_id]
    assert len(artifact_ids) == 1
    artifact = project.db.execute(
        "SELECT artifact_type,backend,model FROM artifact_registry WHERE id=?",
        (artifact_ids[0],),
    ).fetchone()
    assert tuple(artifact) == ("parsed_tokens", "spacy", "fixture-en")
    local_nlp = run_config["declared_profile"]["local_nlp"]
    assert local_nlp["document_languages"] == {revision_id: "en"}
    assert local_nlp["artifacts"][0]["artifact_id"] == artifact_ids[0]


def test_saved_agency_profile_pins_dependency_parse_to_its_artifact(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "project")
    document_id, revision_id = project.import_text("speech.txt", "Minister spoke.")
    project.set_document_metadata(document_id, "language", "en")
    _saved_profile(project, local_preprocessors=["agency_candidates"])

    def agency_supported_preflight(self, languages=None):
        selected = languages or tuple(self.languages)
        return {
            language: {
                "runnable": True,
                "limitations": [],
                "required_capabilities": ["dependency_parse"],
                "statuses": {"dependency_parse": "SUPPORTED"},
                "backend": self.nlp_routes[language].backend,
                "model": self.nlp_routes[language].model,
                "model_dir_env": self.nlp_routes[language].model_dir_env,
                "route_source": "profile",
            }
            for language in selected
        }

    monkeypatch.setattr(ControlledCdaProfile, "local_preflight", agency_supported_preflight)
    monkeypatch.setattr(nlp_module, "load_spacy", lambda language, model: _FixtureBackend())

    result = analysis_module.run_saved_controlled_profile(
        project,
        [revision_id],
        "local-candidates",
        provider_factory=lambda *_args, **_kwargs: _FixtureProvider(),
    )

    summary = cast(dict[str, Any], project.analysis_run_summary(result["analysis_run_id"]))
    run = summary["analysis_run"]
    config = cast(dict[str, Any], run["config"])
    artifact_id = config["local_nlp_artifacts"][revision_id][0]
    snapshot = config["agency_dependency_parse_preflight"][revision_id]
    assert snapshot["status"] == "SUPPORTED"
    assert snapshot["language"] == "en"
    assert snapshot["artifact_id"] == artifact_id
    assert snapshot["backend"] == "spacy"
    assert snapshot["model"] == "fixture-en"
    skill = project.db.execute(
        "SELECT status FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.agency-candidates'",
        (result["analysis_run_id"],),
    ).fetchone()
    assert skill["status"] == "COMPLETED"


def test_invalid_default_preset_fails_before_local_nlp_artifact_registration(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "project")
    document_id, revision_id = project.import_text("speech.txt", "Minister spoke.")
    project.set_document_metadata(document_id, "language", "en")
    _saved_profile(project, default_preset="missing-review-preset")
    monkeypatch.setattr(ControlledCdaProfile, "local_preflight", _eligible_preflight)
    monkeypatch.setattr(nlp_module, "load_spacy", lambda language, model: _FixtureBackend())

    with pytest.raises(ProjectError, match="unknown model preset"):
        analysis_module.run_saved_controlled_profile(
            project, [revision_id], "local-candidates"
        )

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='parsed_tokens'"
        ).fetchone()[0]
        == 0
    )


def test_missing_framework_route_fails_before_local_nlp_artifact_registration(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "project")
    document_id, revision_id = project.import_text("speech.txt", "Minister spoke.")
    project.set_document_metadata(document_id, "language", "en")
    _saved_profile(project)
    project.save_model_routing_profile("synthesis-only", {"synthesis": "review"})
    routed_profile = project.analysis_profile("local-candidates")
    routed_profile["routing_profile"] = "synthesis-only"
    project.save_analysis_profile("missing-framework-route", routed_profile)
    monkeypatch.setattr(ControlledCdaProfile, "local_preflight", _eligible_preflight)
    monkeypatch.setattr(nlp_module, "load_spacy", lambda language, model: _FixtureBackend())

    with pytest.raises(ProjectError, match="no preset for framework 'fairclough-text'"):
        analysis_module.run_saved_controlled_profile(
            project, [revision_id], "missing-framework-route"
        )

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='parsed_tokens'"
        ).fetchone()[0]
        == 0
    )


def test_missing_reranking_credential_fails_before_local_nlp_artifact_registration(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "project")
    document_id, revision_id = project.import_text("speech.txt", "Minister spoke.")
    project.set_document_metadata(document_id, "language", "en")
    _saved_profile(project)
    reranked_profile = project.analysis_profile("local-candidates")
    reranked_profile.update(
        {
            "reranking_provider_id": "jina-reranker",
            "reranking_credential_alias": "missing-reranker",
            "reranking_model": "fixture-reranker-v1",
        }
    )
    project.save_analysis_profile("missing-reranker-credential", reranked_profile)
    loaded_languages = []
    monkeypatch.setattr(ControlledCdaProfile, "local_preflight", _eligible_preflight)
    monkeypatch.setattr(
        nlp_module,
        "load_spacy",
        lambda language, model: loaded_languages.append(language) or _FixtureBackend(),
    )

    with pytest.raises(ProjectError, match="unknown provider credential alias"):
        analysis_module.run_saved_controlled_profile(
            project, [revision_id], "missing-reranker-credential"
        )

    assert loaded_languages == []
    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='parsed_tokens'"
        ).fetchone()[0]
        == 0
    )


@pytest.mark.parametrize(
    "blocked_status", ["LIMITED", "EXPERIMENTAL", "UNASSESSED", "UNSUPPORTED"]
)
def test_all_blocked_local_operation_is_skipped_while_cda_stages_continue(
    tmp_path, monkeypatch, blocked_status
):
    project = Project.init(tmp_path / "project")
    document_id, revision_id = project.import_text("speech.txt", "Minister must act.")
    project.set_document_metadata(document_id, "language", "en")
    _saved_profile(project, local_preprocessors=["agency_candidates"])
    observed = {}

    def blocked_preflight(languages, required_capabilities, *, routes=None):
        observed["languages"] = languages
        observed["required_capabilities"] = required_capabilities
        return {
            "en": {
                "runnable": False,
                "limitations": [f"dependency_parse: {blocked_status}"],
                "required_capabilities": ["dependency_parse"],
                "statuses": {"dependency_parse": blocked_status},
                "backend": "spacy",
                "model": "fixture-en",
                "model_dir_env": None,
                "route_source": "profile",
            }
        }

    def forbidden_model_load(language, model):
        pytest.fail("local model loaded despite an ineligible capability preflight")

    monkeypatch.setattr(profiles_module, "local_execution_preflight", blocked_preflight)
    monkeypatch.setattr(nlp_module, "load_spacy", forbidden_model_load)

    result = analysis_module.run_saved_controlled_profile(
        project,
        [revision_id],
        "local-candidates",
        provider_factory=lambda *_args, **_kwargs: _FixtureProvider(),
    )

    assert observed == {
        "languages": ("en",),
        "required_capabilities": {"dependency_parse"},
    }
    assert result["seed_finding_ids"]
    assert result["reviewed_finding_ids"]
    assert any(
        "agency_candidates" in warning and blocked_status in warning
        for warning in result["warnings"]
    )
    run = project.db.execute(
        "SELECT scope_id,status,config_json FROM analysis_runs WHERE id=?",
        (result["analysis_run_id"],),
    ).fetchone()
    assert run["status"] == "COMPLETED"
    assert project.scope_revisions(run["scope_id"]) == [revision_id]
    config = json.loads(run["config_json"])
    assert config["local_nlp_artifacts"] == {}
    operation = config["local_nlp_execution_plan"]["operations"]["agency_candidates"]
    assert operation["status"] == "SKIPPED"
    assert operation["processed_revision_ids"] == []
    assert operation["skipped_revision_ids"] == [revision_id]
    skipped = project.db.execute(
        "SELECT status,skip_reason,scope_id FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.agency-candidates'",
        (result["analysis_run_id"],),
    ).fetchone()
    assert skipped["status"] == "SKIPPED"
    assert skipped["scope_id"] == run["scope_id"]
    assert f"dependency_parse={blocked_status}" in skipped["skip_reason"]
    assert "no fallback" in skipped["skip_reason"]
    assert project.db.execute(
        "SELECT COUNT(*) FROM artifact_registry WHERE artifact_type='parsed_tokens'"
    ).fetchone()[0] == 0
    completed_stages = {
        row["skill_id"]: row["status"]
        for row in project.db.execute(
            "SELECT skill_id,status FROM skill_runs WHERE analysis_run_id=?",
            (result["analysis_run_id"],),
        ).fetchall()
    }
    assert completed_stages["builtin.textual-descriptors"] == "COMPLETED"
    assert completed_stages["builtin.evidence-extraction"] == "COMPLETED"
    assert completed_stages["builtin.fairclough-text"] == "COMPLETED"


def test_mixed_language_profile_runs_supported_revision_and_reports_partial_scope(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "project")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "fixture", "DL_FIXTURE_KEY", "http://fixture.test/v1"
        )
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    english_document, english_revision = project.import_text("english.txt", "Minister must act.")
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
            "counterevidence_queries": {"descriptive_candidate_inventory": "no"},
        },
    )
    observed = {}
    loaded_languages = []

    def mixed_preflight(languages, required_capabilities, *, routes=None):
        assert routes is not None
        observed["languages"] = languages
        observed["required_capabilities"] = required_capabilities
        observed["routes"] = routes
        return {
            language: {
                "runnable": language == "en",
                "limitations": [] if language == "en" else ["pos: UNSUPPORTED"],
                "required_capabilities": ["pos"],
                "statuses": {"pos": "SUPPORTED" if language == "en" else "UNSUPPORTED"},
                "backend": routes[language]["backend"],
                "model": routes[language]["model"],
                "model_dir_env": None,
                "route_source": "profile",
            }
            for language in languages
        }

    def load_local_model(language, model):
        loaded_languages.append(language)
        return _FixtureBackend(model_id=model, language=language)

    monkeypatch.setattr(profiles_module, "local_execution_preflight", mixed_preflight)
    monkeypatch.setattr(nlp_module, "load_spacy", load_local_model)

    result = analysis_module.run_saved_controlled_profile(
        project,
        [english_revision, spanish_revision],
        "mixed-local-candidates",
        provider_factory=lambda *_args, **_kwargs: _FixtureProvider(),
    )

    assert observed["languages"] == ("en", "es")
    assert observed["required_capabilities"] == {"pos"}
    assert {
        language: route["model"] for language, route in observed["routes"].items()
    } == {"en": "fixture-en", "es": "fixture-es"}
    assert loaded_languages == ["en"]
    assert result["seed_finding_ids"]
    assert result["reviewed_finding_ids"]
    assert any(
        "social_actor_candidates" in warning and "UNSUPPORTED" in warning
        for warning in result["warnings"]
    )
    run = project.db.execute(
        "SELECT scope_id,status,config_json FROM analysis_runs WHERE id=?",
        (result["analysis_run_id"],),
    ).fetchone()
    assert run["status"] == "COMPLETED"
    assert project.scope_revisions(run["scope_id"]) == [english_revision, spanish_revision]
    config = json.loads(run["config_json"])
    assert set(config["local_nlp_artifacts"]) == {english_revision}
    operation = config["local_nlp_execution_plan"]["operations"]["social_actor_candidates"]
    assert operation["status"] == "PARTIAL"
    assert operation["processed_revision_ids"] == [english_revision]
    assert operation["skipped_revision_ids"] == [spanish_revision]
    assert len(result["local_nlp_skill_run_ids"]) == 2
    local_rows = project.db.execute(
        "SELECT id,status,skip_reason,scope_id FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.social-actor-candidates' ORDER BY status",
        (result["analysis_run_id"],),
    ).fetchall()
    assert {row["status"] for row in local_rows} == {"COMPLETED", "SKIPPED"}
    assert all(row["scope_id"] == run["scope_id"] for row in local_rows)
    skipped_row = next(row for row in local_rows if row["status"] == "SKIPPED")
    assert spanish_revision in skipped_row["skip_reason"]
    assert "pos=UNSUPPORTED" in skipped_row["skip_reason"]

    extraction_coverage = project.coverage_descriptor(
        project.db.execute(
            "SELECT coverage_id FROM skill_runs WHERE analysis_run_id=? "
            "AND skill_id='builtin.evidence-extraction'",
            (result["analysis_run_id"],),
        ).fetchone()[0]
    )
    assert extraction_coverage.scope_snapshot_id == run["scope_id"]
    counterevidence_statuses = list(result["counterevidence_statuses"].values())
    assert counterevidence_statuses
    assert all(status["status"] == "COMPLETED" for status in counterevidence_statuses)
    for status in counterevidence_statuses:
        coverage_id = status["coverage_id"]
        assert coverage_id is not None
        assert project.coverage_descriptor(coverage_id).scope_snapshot_id == run["scope_id"]
    hit_revisions = set()
    for status in counterevidence_statuses:
        hit_revisions.update(
            row["revision_id"]
            for row in project.db.execute(
                "SELECT s.revision_id FROM retrieval_hits h JOIN source_spans s "
                "ON s.id=h.source_span_id WHERE h.retrieval_run_id=?",
                (status["retrieval_run_id"],),
            ).fetchall()
        )
    assert spanish_revision in hit_revisions


def test_local_model_path_is_rejected_without_echoing_path_in_traceback(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision_id = project.import_text("speech.txt", "Minister spoke.")
    private_model_path = str(tmp_path / "private-model-directory")
    _saved_profile(project, model=private_model_path)

    with pytest.raises(ProjectError, match="analysis profile is invalid") as exc_info:
        analysis_module.run_saved_controlled_profile(
            project, [revision_id], "local-candidates"
        )

    rendered_traceback = "".join(traceback.format_exception(exc_info.value))
    assert private_model_path not in rendered_traceback
    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


def test_profile_local_preprocessors_require_explicit_document_language(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision_id = project.import_text("speech.txt", "Minister spoke.")
    _saved_profile(project)

    with pytest.raises(ProjectError, match="explicit document metadata language"):
        analysis_module.run_saved_controlled_profile(project, [revision_id], "local-candidates")


def test_profile_local_preprocessors_require_an_exact_language_route(tmp_path):
    project = Project.init(tmp_path / "project")
    document_id, revision_id = project.import_text("speech.txt", "Minister spoke.")
    project.set_document_metadata(document_id, "language", "es")
    _saved_profile(project)

    with pytest.raises(ProjectError, match="explicit local NLP route.*es"):
        analysis_module.run_saved_controlled_profile(project, [revision_id], "local-candidates")


def test_profile_local_stanza_route_requires_its_configured_resource_environment(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "project")
    document_id, revision_id = project.import_text("speech.txt", "Minister spoke.")
    project.set_document_metadata(document_id, "language", "en")
    _saved_profile(
        project, backend="stanza", model="en-fixture", model_dir_env="DL_MISSING_STANZA_HOME"
    )
    monkeypatch.delenv("DL_MISSING_STANZA_HOME", raising=False)

    with pytest.raises(ProjectError, match="requires environment variable 'DL_MISSING_STANZA_HOME'"):
        analysis_module.run_saved_controlled_profile(project, [revision_id], "local-candidates")

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


def test_local_preprocessors_add_their_capability_requirements(monkeypatch):
    observed = {}

    def capture(languages, required_capabilities, *, routes=None):
        observed["languages"] = languages
        observed["required"] = required_capabilities
        observed["routes"] = routes
        return {}

    monkeypatch.setattr(profiles_module, "local_execution_preflight", capture)
    profile = ControlledCdaProfile.model_validate(
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "nlp_routes": {"en": {"backend": "spacy", "model": "en_fixture"}},
            "local_preprocessors": ["social_actor_candidates", "agency_candidates"],
        }
    )

    profile.local_preflight(("en",))

    assert observed["languages"] == ("en",)
    assert observed["required"] == {"pos", "dependency_parse"}


def test_local_nlp_operation_plan_round_trips_sorted_json_keys_without_reordering_scope():
    revision_ids = ["revision-z", "revision-b", "revision-a"]
    plan = analysis_module._build_local_nlp_execution_plan(
        revision_ids,
        {"revision-z": "es", "revision-b": "en", "revision-a": "es"},
        {
            "en": {"statuses": {"pos": "SUPPORTED"}, "limitations": []},
            "es": {"statuses": {"pos": "UNASSESSED"}, "limitations": []},
        },
        ("social_actor_candidates",),
    )

    saved = json.loads(json.dumps(plan, sort_keys=True))
    restored = analysis_module._validated_local_nlp_execution_plan(
        saved, ("social_actor_candidates",), revision_ids
    )

    operation = restored["operations"]["social_actor_candidates"]
    assert operation["processed_revision_ids"] == ["revision-b"]
    assert operation["skipped_revision_ids"] == ["revision-z", "revision-a"]


def test_local_nlp_artifact_provenance_does_not_capture_model_directory(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "project")
    document_id, revision_id = project.import_text("speech.txt", "Minister spoke.")
    project.set_document_metadata(document_id, "language", "en")
    _saved_profile(
        project, backend="stanza", model="en-fixture", model_dir_env="DL_LOCAL_STANZA_HOME"
    )
    monkeypatch.setattr(ControlledCdaProfile, "local_preflight", _eligible_preflight)
    private_model_dir = str(tmp_path / "private-stanza-resources")
    monkeypatch.setenv("DL_LOCAL_STANZA_HOME", private_model_dir)
    monkeypatch.setattr(
        nlp_module,
        "load_stanza",
        lambda language, model, model_dir: _FixtureBackend("stanza", "en-fixture"),
    )

    result = analysis_module.run_saved_controlled_profile(
        project, [revision_id], "local-candidates"
    )
    summary = cast(dict[str, Any], project.analysis_run_summary(result["analysis_run_id"]))
    run_config = cast(dict[str, Any], summary["analysis_run"]["config"])
    serialized = json.dumps(run_config, sort_keys=True)
    assert private_model_dir not in serialized
    assert "DL_LOCAL_STANZA_HOME" in serialized
    assert "DL_FIXTURE_KEY" not in serialized
