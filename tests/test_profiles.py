import hashlib
import json
import re
import traceback

import pytest

import discourselens.analysis as analysis_module
from discourselens import Project
from discourselens.analysis import resume_controlled_cda_review, run_saved_controlled_profile
from discourselens.errors import ProjectError
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelGateway,
    ModelPreset,
    RerankResult,
)
from discourselens.profiles import ControlledCdaProfile, LocalNlpRoute


class _Provider:
    provider_id = "openai-compatible"

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = json.loads(request.prompt.split("UNTRUSTED_RESEARCH_DATA_JSON:\n", 1)[1])
        records = payload["evidence_records"]
        response = {
            "interpretation": "Evidence-bound reading.",
            "alternative_interpretation": "Alternative reading.",
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


class _FailOnceProvider(_Provider):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("fixture transient failure")
        return super().generate(request)


class _FixtureReranker:
    def __init__(self, provider_id="jina-reranker"):
        self.provider_id = provider_id

    def rerank(self, model, query, documents, *, top_n=None):
        indexes = list(reversed(range(len(documents))))
        if top_n is not None:
            indexes = indexes[:top_n]
        return [RerankResult(index, float(len(documents) - index)) for index in indexes]


def test_saved_controlled_profile_is_validated_and_runs_only_known_frameworks(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "We must act.")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    profile = {
        "version": "1.0.0",
        "kind": "controlled-cda",
        "frameworks": ["fairclough-text"],
        "languages": ["en"],
        "default_preset": "review",
    }
    project.save_analysis_profile("focused", profile)
    assert project.list_analysis_profiles()[0]["name"] == "focused"
    assert project.list_analysis_profiles()[0]["version"] == "1.0.0"

    result = run_saved_controlled_profile(
        project,
        [revision],
        "focused",
        provider_factory=lambda endpoint, reference, provider: _Provider(),
    )
    config = json.loads(
        project.db.execute(
            "SELECT config_json FROM analysis_runs WHERE id=?", (result["analysis_run_id"],)
        ).fetchone()[0]
    )
    assert config["frameworks"] == ["fairclough-text"]
    canonical_profile = ControlledCdaProfile.model_validate(profile).model_dump(mode="json")
    assert {
        key: value
        for key, value in config["declared_profile"].items()
        if key != "language_preflight"
    } == {
        "name": "focused",
        "version": "1.0.0",
        "payload_hash": hashlib.sha256(
            json.dumps(
                canonical_profile, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "payload": canonical_profile,
    }
    assert config["declared_profile"]["language_preflight"]["en"]["runnable"]
    assert config["declared_profile"]["language_preflight"]["en"]["required_capabilities"] == []

    project.save_analysis_profile("unsafe", {**profile, "frameworks": ["made-up"]})
    with pytest.raises(ProjectError, match="unknown framework"):
        run_saved_controlled_profile(project, [revision], "unsafe", preset_name="review")


def test_saved_profile_preset_override_respects_model_routing_precedence(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "We must act.")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    presets = {
        "profile-default": "profile-default-model",
        "per-call-override": "per-call-model",
        "routed-framework": "framework-model",
        "routed-fallback": "fallback-model",
    }
    for preset_name, model in presets.items():
        project.save_model_preset(
            ModelPreset(preset_name, "openai-compatible", "fixture", model)
        )

    project.save_analysis_profile(
        "single-preset",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "default_preset": "profile-default",
        },
    )
    single_result = run_saved_controlled_profile(
        project,
        [revision],
        "single-preset",
        preset_name="per-call-override",
        provider_factory=lambda endpoint, reference, provider: _Provider(),
    )
    single_config = json.loads(
        project.db.execute(
            "SELECT config_json FROM analysis_runs WHERE id=?",
            (single_result["analysis_run_id"],),
        ).fetchone()[0]
    )
    assert single_config["preset_name"] == "per-call-override"
    assert single_config["framework_presets"] == {
        "fairclough-text": "per-call-override",
        "synthesis": "per-call-override",
    }
    assert single_config["framework_preset_snapshots"]["fairclough-text"]["model"] == (
        "per-call-model"
    )
    assert single_config["framework_preset_snapshots"]["synthesis"]["model"] == (
        "per-call-model"
    )

    project.save_model_routing_profile(
        "partial-routing", {"fairclough-text": "routed-framework"}, "routed-fallback"
    )
    project.save_analysis_profile(
        "routed-profile",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "routing_profile": "partial-routing",
            "default_preset": "profile-default",
        },
    )
    routed_result = run_saved_controlled_profile(
        project,
        [revision],
        "routed-profile",
        preset_name="per-call-override",
        provider_factory=lambda endpoint, reference, provider: _Provider(),
    )
    routed_config = json.loads(
        project.db.execute(
            "SELECT config_json FROM analysis_runs WHERE id=?",
            (routed_result["analysis_run_id"],),
        ).fetchone()[0]
    )
    assert routed_config["preset_name"] == "per-call-override"
    assert routed_config["framework_presets"] == {
        "fairclough-text": "routed-framework",
        "synthesis": "routed-fallback",
    }
    assert routed_config["framework_preset_snapshots"]["fairclough-text"]["preset_name"] == (
        "routed-framework"
    )
    assert routed_config["framework_preset_snapshots"]["fairclough-text"]["model"] == (
        "framework-model"
    )
    assert routed_config["framework_preset_snapshots"]["synthesis"]["preset_name"] == (
        "routed-fallback"
    )
    assert routed_config["framework_preset_snapshots"]["synthesis"]["model"] == (
        "fallback-model"
    )


def test_saved_profile_validation_error_does_not_leak_payload_values_in_traceback(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "A sentence.")
    secret = "sk-profile-validation-secret"
    project.save_analysis_profile(
        "invalid-sensitive-profile",
        {
            "frameworks": ["fairclough-text"],
            "default_preset": "review",
            "reranking_top_n": secret,
        },
    )

    with pytest.raises(ProjectError, match="analysis profile is invalid") as exc_info:
        run_saved_controlled_profile(project, [revision], "invalid-sensitive-profile")

    rendered_traceback = "".join(traceback.format_exception(exc_info.value))
    assert secret not in rendered_traceback


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("frameworks", ["sk-profile-semantic-secret"], "unknown framework"),
        (
            "required_capabilities",
            ["sk-profile-semantic-secret"],
            "unknown language capability",
        ),
        (
            "counterevidence_queries",
            {"sk-profile-semantic-secret": "exception"},
            "supported evidence-extraction finding phenomenon",
        ),
    ],
)
def test_saved_profile_semantic_validation_does_not_leak_payload_values(
    tmp_path, field, value, diagnostic
):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "A sentence.")
    project.save_analysis_profile(
        "invalid-semantic-profile",
        {"frameworks": ["fairclough-text"], field: value},
    )

    with pytest.raises(ProjectError, match=diagnostic) as exc_info:
        run_saved_controlled_profile(project, [revision], "invalid-semantic-profile")

    rendered_traceback = "".join(traceback.format_exception(exc_info.value))
    assert "sk-profile-semantic-secret" not in rendered_traceback
    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


def test_saved_profile_eligibility_error_does_not_leak_declared_language(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "A sentence.")
    project.save_analysis_profile(
        "ineligible-sensitive-profile",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["sk-profile-semantic-secret"],
            "required_capabilities": ["pos"],
            "default_preset": "unused",
        },
    )

    with pytest.raises(ProjectError, match="not eligible") as exc_info:
        run_saved_controlled_profile(project, [revision], "ineligible-sensitive-profile")

    rendered_traceback = "".join(traceback.format_exception(exc_info.value))
    assert "sk-profile-semantic-secret" not in rendered_traceback
    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


def test_saved_profile_can_use_fully_routed_models_without_a_default_preset(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "We must act.")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("text-review", "openai-compatible", "fixture", "text"))
    project.save_model_preset(
        ModelPreset("synthesis-review", "openai-compatible", "fixture", "synthesis")
    )
    project.save_model_routing_profile(
        "fully-routed",
        {"fairclough-text": "text-review", "synthesis": "synthesis-review"},
    )
    project.save_analysis_profile(
        "routed",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "routing_profile": "fully-routed",
        },
    )
    models: list[str] = []

    class _RecordingProvider(_Provider):
        def generate(self, request: GenerationRequest) -> GenerationResponse:
            models.append(request.model)
            if len(models) == 1:
                raise RuntimeError("fixture failure before routed run completes")
            return super().generate(request)

    with pytest.raises(RuntimeError, match="before routed run completes"):
        run_saved_controlled_profile(
            project,
            [revision],
            "routed",
            provider_factory=lambda endpoint, reference, provider: _RecordingProvider(),
        )
    failed_run_id = project.db.execute(
        "SELECT id FROM analysis_runs WHERE status='FAILED'"
    ).fetchone()["id"]
    result = resume_controlled_cda_review(
        project,
        failed_run_id,
        provider_factory=lambda endpoint, reference, provider: _RecordingProvider(),
    )

    config = json.loads(
        project.db.execute(
            "SELECT config_json FROM analysis_runs WHERE id=?", (result["analysis_run_id"],)
        ).fetchone()[0]
    )
    assert models == ["text", "text", "synthesis"]
    assert config["preset_name"] is None
    assert config["framework_presets"] == {
        "fairclough-text": "text-review",
        "synthesis": "synthesis-review",
    }


def test_profile_research_question_runs_scoped_retrieval_without_cue_seeds(tmp_path):
    project = Project.init(tmp_path / "project")
    _, selected_revision = project.import_text(
        "selected.txt",
        "The procurement committee selected the contractor after reviewing the bids.\n\n"
        "The procurement committee selected the supplier after a review.\n\n"
        "The procurement committee selected the provider after deliberation.",
    )
    _, out_of_scope_revision = project.import_text(
        "other.txt",
        "The procurement committee selected the contractor after reviewing the bids.",
    )
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_analysis_profile(
        "procurement-question",
        {
            "version": "1.0.0",
            "kind": "controlled-cda",
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "research_questions": {
                "fairclough-text": "How did the procurement committee select the contractor?"
            },
            "question_candidate_limit": 1,
            "default_preset": "review",
        },
    )

    result = run_saved_controlled_profile(
        project,
        [selected_revision],
        "procurement-question",
        provider_factory=lambda endpoint, reference, provider: _Provider(),
    )

    assert result["seed_finding_ids"] == []
    assert len(result["reviewed_finding_ids"]) == 1
    candidate_finding_id = result["scope_question_finding_ids"]["fairclough-text"]
    candidate_finding = project.get_finding(candidate_finding_id)
    assert json.loads(candidate_finding["claim_types_json"]) == ["EXAMPLE_BASED"]
    assert (
        project.coverage_descriptor(candidate_finding["coverage_id"]).coverage_mode.value
        == "TARGETED"
    )
    assert project.finding_evidence(candidate_finding_id)

    retrieval_run_id = result["scope_question_skill_run_ids"]
    assert len(retrieval_run_id) == 1
    retrieval = project.db.execute(
        "SELECT scope_id,coverage_id,returned_count,selection_policy FROM retrieval_runs WHERE skill_run_id=?",
        (retrieval_run_id[0],),
    ).fetchone()
    assert project.scope_revisions(retrieval["scope_id"]) == [selected_revision]
    assert retrieval["returned_count"] == 1
    assert "return at most 1" in retrieval["selection_policy"]
    assert project.coverage_descriptor(retrieval["coverage_id"]).coverage_mode.value == "TARGETED"
    hits = project.db.execute(
        "SELECT s.revision_id FROM retrieval_hits h JOIN source_spans s "
        "ON s.id=h.source_span_id WHERE h.retrieval_run_id=?",
        (
            project.db.execute(
                "SELECT id FROM retrieval_runs WHERE skill_run_id=?", (retrieval_run_id[0],)
            ).fetchone()["id"],
        ),
    ).fetchall()
    assert hits
    assert len(hits) == 1
    assert {row["revision_id"] for row in hits} == {selected_revision}
    assert out_of_scope_revision not in {row["revision_id"] for row in hits}
    run_summary = project.analysis_run_summary(result["analysis_run_id"])
    analysis_run = run_summary["analysis_run"]
    assert isinstance(analysis_run, dict)
    run_config = analysis_run["config"]
    assert isinstance(run_config, dict)
    assert run_config["question_candidate_limit"] == 1


def test_arabic_profile_skips_english_spanish_cues_but_runs_scoped_question(tmp_path):
    project = Project.init(tmp_path / "project")
    document_id, revision = project.import_text(
        "arabic.txt", "صورت اللجنة قرار الهجرة بوصفه استجابة لازمة."
    )
    project.set_document_metadata(document_id, "language", "ar")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_analysis_profile(
        "arabic-question",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["ar"],
            "research_questions": {"fairclough-text": "كيف صورت اللجنة قرار الهجرة؟"},
            "default_preset": "review",
        },
    )

    result = run_saved_controlled_profile(
        project,
        [revision],
        "arabic-question",
        provider_factory=lambda endpoint, reference, provider: _Provider(),
    )

    assert result["seed_finding_ids"] == []
    assert len(result["reviewed_finding_ids"]) == 1
    assert result["scope_question_finding_ids"]
    assert any(
        "no configured language-matched rules for: ar" in item for item in result["warnings"]
    )
    skipped_cues = project.db.execute(
        "SELECT skill_id,skip_reason FROM skill_runs WHERE analysis_run_id=? "
        "AND status='SKIPPED' AND skill_id IN ('builtin.textual-descriptors', 'builtin.evidence-extraction')",
        (result["analysis_run_id"],),
    ).fetchall()
    assert {row["skill_id"] for row in skipped_cues} == {
        "builtin.textual-descriptors",
        "builtin.evidence-extraction",
    }
    assert all(
        row["skip_reason"] == "NO_LANGUAGE_MATCHED_SUPPORTED_CUE_INVENTORY" for row in skipped_cues
    )
    retrieval = project.db.execute(
        "SELECT rr.id,rr.scope_id FROM retrieval_runs rr JOIN skill_runs sr "
        "ON sr.id=rr.skill_run_id WHERE sr.analysis_run_id=? "
        "AND sr.skill_id='builtin.scope-question-candidates'",
        (result["analysis_run_id"],),
    ).fetchone()
    assert project.scope_revisions(retrieval["scope_id"]) == [revision]
    hit = project.db.execute(
        "SELECT s.revision_id FROM retrieval_hits h JOIN source_spans s "
        "ON s.id=h.source_span_id WHERE h.retrieval_run_id=?",
        (retrieval["id"],),
    ).fetchone()
    assert hit["revision_id"] == revision


def test_multilingual_profile_runs_cues_only_on_explicitly_supported_revisions(tmp_path):
    project = Project.init(tmp_path / "project")
    english_document, english_revision = project.import_text("english.txt", "We must act now.")
    arabic_document, arabic_revision = project.import_text(
        "arabic.txt", "صورت اللجنة قرار الهجرة بوصفه استجابة لازمة."
    )
    project.set_document_metadata(english_document, "language", "en")
    project.set_document_metadata(arabic_document, "language", "ar")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_analysis_profile(
        "mixed-languages",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en", "ar"],
            "default_preset": "review",
        },
    )

    result = run_saved_controlled_profile(
        project,
        [english_revision, arabic_revision],
        "mixed-languages",
        provider_factory=lambda endpoint, reference, provider: _Provider(),
    )

    assert result["seed_finding_ids"]
    assert any("only 1 of 2 scoped revisions" in item for item in result["warnings"])
    summary = project.analysis_run_summary(result["analysis_run_id"])
    analysis_run = summary["analysis_run"]
    assert isinstance(analysis_run, dict)
    config = analysis_run["config"]
    assert isinstance(config, dict)
    cue_plan = config["cue_language_plan"]
    assert isinstance(cue_plan, dict)
    assert cue_plan["revision_ids"] == [english_revision]
    assert cue_plan["revision_languages"] == {english_revision: "en"}
    seed_evidence = project.finding_evidence(result["seed_finding_ids"][0])
    evidence_revisions = {
        project.db.execute(
            "SELECT revision_id FROM source_spans WHERE id=?", (row["span_id"],)
        ).fetchone()["revision_id"]
        for row in seed_evidence
    }
    assert evidence_revisions == {english_revision}
    extraction = project.db.execute(
        "SELECT coverage_id FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.evidence-extraction' AND status='COMPLETED'",
        (result["analysis_run_id"],),
    ).fetchone()
    population = project.coverage_descriptor(extraction["coverage_id"])
    assert population.population_definition is not None
    assert "language-eligible revisions" in population.population_definition


def test_mixed_language_profile_requires_document_language_metadata_for_cue_selection(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("unlabeled.txt", "We must act now.")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_analysis_profile(
        "mixed-unlabeled",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en", "ar"],
            "default_preset": "review",
        },
    )

    result = run_saved_controlled_profile(project, [revision], "mixed-unlabeled")

    assert result["seed_finding_ids"] == []
    assert any("lacks per-document language metadata" in item for item in result["warnings"])
    skipped = project.db.execute(
        "SELECT skill_id,skip_reason FROM skill_runs WHERE analysis_run_id=? "
        "AND status='SKIPPED' AND skill_id IN ('builtin.textual-descriptors', 'builtin.evidence-extraction')",
        (result["analysis_run_id"],),
    ).fetchall()
    assert {row["skill_id"] for row in skipped} == {
        "builtin.textual-descriptors",
        "builtin.evidence-extraction",
    }
    assert all(
        row["skip_reason"] == "LANGUAGE_ASSIGNMENT_REQUIRED_FOR_PARTIAL_CUE_EXECUTION"
        for row in skipped
    )


def test_single_declared_language_assignment_is_frozen_and_warned(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("unlabeled.txt", "La migración requiere atención.")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_analysis_profile(
        "single-language",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "default_preset": "review",
        },
    )

    result = run_saved_controlled_profile(project, [revision], "single-language")

    summary = project.analysis_run_summary(result["analysis_run_id"])
    run = summary["analysis_run"]
    assert isinstance(run, dict)
    config = run["config"]
    assert isinstance(config, dict)
    plan = config["cue_language_plan"]
    assert isinstance(plan, dict)
    assert plan["revision_ids"] == [revision]
    assert plan["revision_languages"] == {revision: "en"}
    assert any(
        "explicit assumption, not as detected language" in item for item in result["warnings"]
    )


def test_multilingual_supported_profile_skips_unlabeled_cue_scope(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("unlabeled.txt", "We must act. Debemos esperar.")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_analysis_profile(
        "mixed-supported",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en", "es"],
            "default_preset": "review",
        },
    )

    result = run_saved_controlled_profile(project, [revision], "mixed-supported")

    assert result["seed_finding_ids"] == []
    assert any("lacks per-document language metadata" in item for item in result["warnings"])
    skipped = project.db.execute(
        "SELECT skip_reason FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.evidence-extraction' AND status='SKIPPED'",
        (result["analysis_run_id"],),
    ).fetchone()
    assert skipped["skip_reason"] == "LANGUAGE_ASSIGNMENT_REQUIRED_FOR_PARTIAL_CUE_EXECUTION"


def test_profile_rejects_duplicate_frameworks():
    profile = ControlledCdaProfile(
        frameworks=["fairclough-text", "fairclough-text"],
        languages=["en"],
    )

    with pytest.raises(ProjectError, match="must not contain duplicates"):
        profile.validate_method_contract()


@pytest.mark.parametrize("field", ["routing_profile", "default_preset"])
@pytest.mark.parametrize("value", ["", "   "])
def test_profile_rejects_explicit_blank_model_route_references(field, value):
    with pytest.raises(ValueError, match="model-route references must not be blank"):
        ControlledCdaProfile.model_validate(
            {"frameworks": ["fairclough-text"], "default_preset": "review", field: value}
        )


def test_profile_preserves_explicit_model_route_references_after_whitespace_normalization():
    profile = ControlledCdaProfile.model_validate(
        {
            "frameworks": ["fairclough-text"],
            "routing_profile": "  discourse-review-v2  ",
            "default_preset": "  university-review  ",
        }
    )

    assert profile.routing_profile == "discourse-review-v2"
    assert profile.default_preset == "university-review"


def test_cue_language_plan_is_frozen_when_resuming_partial_profile_run(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "project")
    english_document, english_revision = project.import_text("english.txt", "We must act now.")
    arabic_document, arabic_revision = project.import_text(
        "arabic.txt", "صورت اللجنة قرار الهجرة بوصفه استجابة لازمة."
    )
    project.set_document_metadata(english_document, "language", "en")
    project.set_document_metadata(arabic_document, "language", "ar")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_analysis_profile(
        "resumable-mixed-language",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en", "ar"],
            "default_preset": "review",
        },
    )
    original_extraction = analysis_module.run_evidence_extraction

    def fail_extraction(context, languages, **kwargs):
        del context, languages, kwargs
        raise RuntimeError("fixture failure between cue stages")

    monkeypatch.setattr(analysis_module, "run_evidence_extraction", fail_extraction)
    with pytest.raises(RuntimeError, match="between cue stages"):
        run_saved_controlled_profile(
            project,
            [english_revision, arabic_revision],
            "resumable-mixed-language",
            provider_factory=lambda endpoint, reference, provider: _Provider(),
        )

    run = project.db.execute(
        "SELECT id,config_json FROM analysis_runs WHERE status='FAILED'"
    ).fetchone()
    assert run is not None
    config_before = json.loads(run["config_json"])
    frozen_plan = config_before["cue_language_plan"]
    assert frozen_plan["revision_ids"] == [english_revision]
    assert frozen_plan["revision_languages"] == {english_revision: "en"}

    project.set_document_metadata(english_document, "language", "ar")
    monkeypatch.setattr(analysis_module, "run_evidence_extraction", original_extraction)
    result = resume_controlled_cda_review(
        project,
        run["id"],
        provider_factory=lambda endpoint, reference, provider: _Provider(),
    )

    assert result["seed_finding_ids"]
    evidence_rows = project.finding_evidence(result["seed_finding_ids"][0])
    evidence_revisions = {
        project.db.execute(
            "SELECT revision_id FROM source_spans WHERE id=?", (row["span_id"],)
        ).fetchone()["revision_id"]
        for row in evidence_rows
    }
    assert evidence_revisions == {english_revision}
    summary_after = project.analysis_run_summary(run["id"])
    analysis_run_after = summary_after["analysis_run"]
    assert isinstance(analysis_run_after, dict)
    config_after = analysis_run_after["config"]
    assert isinstance(config_after, dict)
    assert config_after["cue_language_plan"] == frozen_plan


def test_profile_research_question_zero_hits_is_not_a_corpus_absence_claim(tmp_path):
    project = Project.init(tmp_path / "project")
    document_id, revision = project.import_text(
        "report.txt", "The procurement committee selected the contractor after reviewing bids."
    )
    project.set_document_metadata(document_id, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_analysis_profile(
        "unmatched-question",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "research_questions": {"fairclough-text": "How are whales classified?"},
            "default_preset": "review",
        },
    )

    result = run_saved_controlled_profile(project, [revision], "unmatched-question")

    assert result["seed_finding_ids"] == []
    assert result["scope_question_finding_ids"] == {}
    assert result["reviewed_finding_ids"] == []
    assert result["synthesis_finding_id"] is None
    assert "no corpus-absence claim" in result["warnings"][0]
    retrieval = project.db.execute(
        "SELECT rr.coverage_id,rr.returned_count FROM retrieval_runs rr "
        "JOIN skill_runs sr ON sr.id=rr.skill_run_id "
        "WHERE sr.analysis_run_id=? AND sr.skill_id='builtin.scope-question-candidates'",
        (result["analysis_run_id"],),
    ).fetchone()
    assert retrieval["returned_count"] == 0
    coverage = project.coverage_descriptor(retrieval["coverage_id"])
    assert coverage.coverage_mode.value == "TARGETED"
    assert coverage.negative_result_status.value == "NO_MATCHES_UNDER_PLAN"
    assert coverage.selection_policy is not None
    assert (
        project.db.execute(
            "SELECT count(*) FROM findings WHERE analysis_run_id=? AND claim_types_json LIKE '%ABSENCE%'",
            (result["analysis_run_id"],),
        ).fetchone()[0]
        == 0
    )


def test_scope_question_retrieval_is_reused_when_analysis_run_resumes(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text(
        "selected.txt",
        "The procurement committee selected the contractor after reviewing the bids.\n\n"
        "The procurement committee selected the supplier after a review.\n\n"
        "The procurement committee selected the provider after deliberation.",
    )
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_analysis_profile(
        "resumable-question",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "research_questions": {
                "fairclough-text": "How did the procurement committee select the contractor?"
            },
            "question_candidate_limit": 1,
            "default_preset": "review",
        },
    )
    provider = _FailOnceProvider()
    provider_factory = lambda endpoint, reference, provider_id: provider

    with pytest.raises(RuntimeError, match="fixture transient failure"):
        run_saved_controlled_profile(
            project,
            [revision],
            "resumable-question",
            provider_factory=provider_factory,
        )

    run = project.db.execute("SELECT id FROM analysis_runs WHERE status='FAILED'").fetchone()
    assert run is not None
    before = project.db.execute(
        "SELECT sr.id,rr.id AS retrieval_id FROM skill_runs sr "
        "JOIN retrieval_runs rr ON rr.skill_run_id=sr.id "
        "WHERE sr.analysis_run_id=? AND sr.skill_id='builtin.scope-question-candidates' "
        "AND sr.status='COMPLETED'",
        (run["id"],),
    ).fetchall()
    assert len(before) == 1
    run_summary = project.analysis_run_summary(run["id"])
    analysis_run = run_summary["analysis_run"]
    assert isinstance(analysis_run, dict)
    run_config = analysis_run["config"]
    assert isinstance(run_config, dict)
    assert run_config["question_candidate_limit"] == 1

    result = resume_controlled_cda_review(project, run["id"], provider_factory=provider_factory)

    after = project.db.execute(
        "SELECT sr.id,rr.id AS retrieval_id FROM skill_runs sr "
        "JOIN retrieval_runs rr ON rr.skill_run_id=sr.id "
        "WHERE sr.analysis_run_id=? AND sr.skill_id='builtin.scope-question-candidates' "
        "AND sr.status='COMPLETED'",
        (run["id"],),
    ).fetchall()
    assert result["scope_question_finding_ids"]
    assert [tuple(row) for row in after] == [tuple(row) for row in before]
    retrieval = project.db.execute(
        "SELECT returned_count FROM retrieval_runs WHERE id=?", (before[0]["retrieval_id"],)
    ).fetchone()
    assert retrieval["returned_count"] == 1


@pytest.mark.parametrize("limit", [0, 13])
def test_profile_rejects_out_of_range_question_candidate_limit(limit):
    with pytest.raises(ValueError):
        ControlledCdaProfile.model_validate(
            {
                "frameworks": ["fairclough-text"],
                "question_candidate_limit": limit,
            }
        )


def test_profile_rejects_boolean_question_candidate_limit():
    with pytest.raises(ValueError):
        ControlledCdaProfile.model_validate(
            {
                "frameworks": ["fairclough-text"],
                "question_candidate_limit": True,
            }
        )


@pytest.mark.parametrize("top_n", [True, 1.0])
def test_profile_rejects_boolean_or_float_reranking_top_n(top_n):
    with pytest.raises(ValueError):
        ControlledCdaProfile.model_validate(
            {
                "frameworks": ["fairclough-text"],
                "reranking_provider_id": "jina-reranker",
                "reranking_credential_alias": "university",
                "reranking_model": "fixture-reranker-v1",
                "reranking_top_n": top_n,
            }
        )


@pytest.mark.parametrize(
    "questions, message",
    [
        ({"van-dijk-ideology": "Who is represented?"}, "selected interpretive frameworks"),
        ({"critical-reviewer": "Try to refute the result"}, "selected interpretive frameworks"),
        ({"fairclough-text": "   "}, "research_questions requires"),
    ],
)
def test_profile_rejects_invalid_research_question_targets(questions, message):
    payload = {
        "frameworks": ["fairclough-text", "critical-reviewer"],
        "languages": ["en"],
        "research_questions": questions,
    }
    try:
        profile = ControlledCdaProfile.model_validate(payload)
    except ValueError as exc:
        assert message in str(exc)
    else:
        with pytest.raises(ProjectError, match=message):
            profile.validate_method_contract()


def test_profile_rejects_research_question_keys_that_collide_after_trimming():
    with pytest.raises(ValueError, match="duplicate normalized"):
        ControlledCdaProfile.model_validate(
            {
                "frameworks": ["fairclough-text"],
                "research_questions": {
                    "fairclough-text": "How is authority represented?",
                    " fairclough-text ": "Who is granted authority?",
                },
            }
        )


def test_profile_normalizes_counterevidence_target_and_query():
    profile = ControlledCdaProfile.model_validate(
        {
            "frameworks": ["fairclough-text"],
            "counterevidence_queries": {" descriptive_candidate_inventory ": "  exception  "},
        }
    )

    profile.validate_method_contract()
    assert profile.counterevidence_queries == {"descriptive_candidate_inventory": "exception"}


def test_profile_rejects_counterevidence_query_for_unknown_seed_phenomenon():
    profile = ControlledCdaProfile.model_validate(
        {
            "frameworks": ["fairclough-text"],
            "counterevidence_queries": {"descriptive_candidate_inventroy": "exception"},
        }
    )

    with pytest.raises(ProjectError, match="supported evidence-extraction"):
        profile.validate_method_contract()


def test_profile_rejects_counterevidence_keys_that_collide_after_normalization():
    with pytest.raises(ValueError, match="duplicate normalized"):
        ControlledCdaProfile.model_validate(
            {
                "frameworks": ["fairclough-text"],
                "counterevidence_queries": {
                    "descriptive_candidate_inventory": "exception",
                    " descriptive_candidate_inventory ": "counterexample",
                },
            }
        )


def test_saved_profile_rejects_declared_unassessed_local_capability_before_a_run(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "A sentence.")
    project.save_analysis_profile(
        "arabic-dependency",
        {
            "version": "1.0.0",
            "kind": "controlled-cda",
            "frameworks": ["fairclough-text"],
            "languages": ["ar"],
            "required_capabilities": ["dependency_parse"],
            "default_preset": "unused",
        },
    )

    with pytest.raises(ProjectError, match="not eligible"):
        run_saved_controlled_profile(project, [revision], "arabic-dependency")
    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


def test_saved_profile_snapshots_selected_local_nlp_route_without_machine_path(
    tmp_path, monkeypatch
):
    import discourselens.languages as language_module
    from discourselens.languages import probe_backend_capabilities
    from discourselens.nlp import ParsedToken

    class LocalBackend:
        language = "en"
        backend_id = "stanza"
        model_id = "en-ud"
        pipeline = type(
            "Pipeline", (), {"processors": {"tokenize": object(), "depparse": object()}}
        )()

        def parse(self, text: str):
            return [
                ParsedToken(
                    match.group(),
                    match.group(),
                    "NOUN",
                    "Case=Nom",
                    None if index == 0 else 0,
                    "root" if index == 0 else "dep",
                    match.start(),
                    match.end(),
                )
                    for index, match in enumerate(re.finditer(r"\w+|[^\w\s]", text))
            ]

    selected_resources: list[str] = []

    def inspect(language, *, backend, model, model_dir=None):
        selected_resources.append(str(model_dir))
        assert (language, backend, model) == ("en", "stanza", "en-ud")
        return probe_backend_capabilities(LocalBackend())

    monkeypatch.setattr(language_module, "inspect_local_backend", inspect)
    monkeypatch.setenv("DL_TEST_STANZA_HOME", str(tmp_path / "private-models"))
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "A local fixture.")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_analysis_profile(
        "english-routed",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "required_capabilities": ["dependency_parse"],
            "nlp_routes": {
                "en": {
                    "backend": "stanza",
                    "model": "en-ud",
                    "model_dir_env": "DL_TEST_STANZA_HOME",
                }
            },
            "default_preset": "review",
        },
    )

    result = run_saved_controlled_profile(
        project,
        [revision],
        "english-routed",
        provider_factory=lambda endpoint, reference, provider: _Provider(),
    )

    summary = project.analysis_run_summary(result["analysis_run_id"])
    analysis_run = summary["analysis_run"]
    assert isinstance(analysis_run, dict)
    config = analysis_run["config"]
    assert isinstance(config, dict)
    declared_profile = config["declared_profile"]
    assert isinstance(declared_profile, dict)
    language_preflight = declared_profile["language_preflight"]
    assert isinstance(language_preflight, dict)
    preflight = language_preflight["en"]
    assert isinstance(preflight, dict)
    assert selected_resources == [str(tmp_path / "private-models")]
    assert preflight["runnable"]
    assert preflight["backend"] == "stanza"
    assert preflight["model"] == "en-ud"
    assert preflight["model_dir_env"] == "DL_TEST_STANZA_HOME"
    assert preflight["statuses"]["dependency_parse"] == "SUPPORTED"
    assert str(tmp_path) not in json.dumps(config)


@pytest.mark.parametrize(
    "route",
    [
        {"backend": "spacy", "model": "."},
        {"backend": "spacy", "model": ".."},
        {"backend": "spacy", "model": ""},
        {"backend": "spacy", "model": "   "},
        {"backend": "spacy", "model": "a" * 129},
        {"backend": "spacy", "model": "en_core_web_sm", "model_dir_env": "MODEL_HOME"},
        {
            "backend": "stanza",
            "model": "ar-padt",
            "model_dir_env": "A" * 129,
        },
        {"backend": "spacy", "model": "/private/local/spacy-model"},
        {"backend": "spacy", "model": "relative/model-directory"},
        {"backend": "spacy", "model": r"C:\private\spacy-model"},
        {"backend": "stanza", "model": "ar-padt"},
        {"backend": "unknown", "model": "x"},
        {"backend": "stanza", "model": "ar-padt", "model_dir_env": "bad-name"},
        {"backend": "stanza", "model": "ar-padt", "model_dir": "/private/models"},
    ],
)
def test_saved_profile_rejects_invalid_local_nlp_route(route):
    with pytest.raises((ProjectError, ValueError)):
        profile = ControlledCdaProfile.model_validate(
            {
                "frameworks": ["fairclough-text"],
                "languages": ["ar"],
                "nlp_routes": {"ar": route},
            }
        )
        profile.validate_method_contract()


@pytest.mark.parametrize(
    ("backend", "model", "model_dir_env"),
    [
        ("spacy", "en_core_web_sm", None),
        ("stanza", "ar-padt", "STANZA_HOME"),
        ("stanza", "en-ud", "LOCAL_NLP_HOME"),
    ],
)
def test_local_nlp_route_accepts_bounded_portable_model_ids(backend, model, model_dir_env):
    route = LocalNlpRoute(
        backend=backend,
        model=model,
        model_dir_env=model_dir_env,
    )

    assert route.model == model


def test_saved_profile_rejects_local_route_for_undeclared_language():
    profile = ControlledCdaProfile.model_validate(
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "nlp_routes": {
                "ar": {"backend": "stanza", "model": "ar-padt", "model_dir_env": "STANZA_HOME"}
            },
        }
    )
    with pytest.raises(ProjectError, match="declared languages"):
        profile.validate_method_contract()


def test_saved_profile_rejects_unknown_route_fields_instead_of_silently_ignoring_them():
    with pytest.raises(ValueError):
        ControlledCdaProfile.model_validate(
            {
                "frameworks": ["fairclough-text"],
                "languages": ["ar"],
                "nlp_route": {
                    "backend": "stanza",
                    "model": "ar-padt",
                    "model_dir_env": "STANZA_HOME",
                },
            }
        )


@pytest.mark.parametrize(
    ("reranker_provider_id", "secret_reference"),
    [("jina-reranker", "JINA_TEST"), ("cohere-reranker", "COHERE_TEST")],
)
def test_saved_controlled_profile_reranks_framework_context_with_selected_alias(
    tmp_path, monkeypatch, reranker_provider_id, secret_reference
):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text(
        "source.txt",
        "We must act.\n\nThe review attached new candidate examples for the policy.",
    )
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    project.save_credential_reference(
        CredentialReference(reranker_provider_id, "university", secret_reference)
    )
    reranker = _FixtureReranker(reranker_provider_id)
    monkeypatch.setattr(
        analysis_module,
        "build_reranking_gateway",
        lambda project, provider_id, credential_alias: ModelGateway(
            {}, rerankers={provider_id: reranker}
        ),
    )
    project.save_analysis_profile(
        "reranked",
        {
            "frameworks": ["fairclough-text"],
            "languages": ["en"],
            "default_preset": "review",
            "reranking_provider_id": reranker_provider_id,
            "reranking_credential_alias": "university",
            "reranking_model": "fixture-reranker-v1",
            "reranking_top_n": 1,
        },
    )

    result = run_saved_controlled_profile(
        project,
        [revision],
        "reranked",
        provider_factory=lambda endpoint, reference, provider: _Provider(),
    )

    summary = project.analysis_run_summary(result["analysis_run_id"])
    run_summary = summary["analysis_run"]
    assert isinstance(run_summary, dict)
    config = run_summary["config"]
    assert isinstance(config, dict)
    assert config["reranking"] == {
        "provider_id": reranker_provider_id,
        "credential_alias": "university",
        "credential_reference_hash": hashlib.sha256(secret_reference.encode()).hexdigest(),
        "model": "fixture-reranker-v1",
        "top_n": 1,
    }
    assert secret_reference not in json.dumps(config["reranking"])
    reranking_runs = summary["reranking_runs"]
    assert isinstance(reranking_runs, list) and reranking_runs
    assert all(item["credential_alias"] == "university" for item in reranking_runs)
    assert all(item["model"] == "fixture-reranker-v1" for item in reranking_runs)
    assert all(item["candidate_count"] >= item["returned_count"] for item in reranking_runs)
    assert all(item["effective_top_n"] == 1 for item in reranking_runs)
