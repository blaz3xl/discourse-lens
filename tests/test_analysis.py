import json

import pytest

from discourselens import Project
from discourselens import analysis as analysis_module
from discourselens import skills as skills_module
from discourselens.analysis import (
    CONTROLLED_CDA_PROMPT_SNAPSHOT_VERSION,
    DEFAULT_CDA_FRAMEWORKS,
    _controlled_cda_prompt_snapshots,
    _controlled_cda_synthesis_prompt_snapshot,
    resume_controlled_cda_review,
    run_batch_controlled_cda_review,
    run_batch_textual_descriptors_analysis,
    run_controlled_cda_review,
    run_exact_phrase_analysis,
)
from discourselens.bundle import export_bundle
from discourselens.errors import ModelProviderError, ModelResponseError, ProjectError
from discourselens.execution import build_reranking_gateway
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelGateway,
    ModelPreset,
    RerankResult,
)
from discourselens.reporting import analysis_payload


class _EvidenceEchoProvider:
    provider_id = "openai-compatible"

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = json.loads(request.prompt.split("UNTRUSTED_RESEARCH_DATA_JSON:\n", 1)[1])
        records = payload["evidence_records"]
        response = {
            "interpretation": "The construction is interpreted only from the cited record.",
            "alternative_interpretation": "The construction may have a neutral reading.",
            "evidence_ids": [records[0]["evidence_id"]],
        }
        if "source_findings" not in payload:
            response["observations"] = [
                {
                    "evidence_id": records[0]["evidence_id"],
                    "excerpt": records[0]["excerpt"],
                    "description": "The passage contains the cited construction.",
                }
            ]
        return GenerationResponse(
            json.dumps(response),
            self.provider_id,
            request.model,
        )


class _FixtureReranker:
    provider_id = "fixture-reranker"

    def rerank(self, model, query, documents, *, top_n=None):
        indexes = list(reversed(range(len(documents))))
        if top_n is not None:
            indexes = indexes[:top_n]
        return [RerankResult(index, float(len(documents) - index)) for index in indexes]


class _CounterevidenceAwareProvider(_EvidenceEchoProvider):
    def __init__(self) -> None:
        self.counterevidence_ids: list[str] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = json.loads(request.prompt.split("UNTRUSTED_RESEARCH_DATA_JSON:\n", 1)[1])
        candidates = [
            item for item in payload["evidence_records"] if item.get("counterevidence_candidate")
        ]
        self.counterevidence_ids.extend(item["evidence_id"] for item in candidates)
        if candidates:
            candidate = candidates[0]
            return GenerationResponse(
                json.dumps(
                    {
                        "observations": [
                            {
                                "evidence_id": candidate["evidence_id"],
                                "excerpt": candidate["excerpt"],
                                "description": "The retrieved passage contains a relevant wording feature.",
                            }
                        ],
                        "interpretation": "The selected query candidate may challenge the reading.",
                        "alternative_interpretation": "It may not contradict the claim in context.",
                        "evidence_ids": [candidate["evidence_id"]],
                        "counterevidence_ids": [candidate["evidence_id"]],
                    }
                ),
                self.provider_id,
                request.model,
            )
        return super().generate(request)


class _MalformedReviewProvider(_EvidenceEchoProvider):
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse("not valid JSON", self.provider_id, request.model)


class _RecordingEvidenceProvider(_EvidenceEchoProvider):
    def __init__(self, models: list[str]) -> None:
        self.models = models

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.models.append(request.model)
        return super().generate(request)


class _FailOnCallProvider(_EvidenceEchoProvider):
    def __init__(self, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated transient provider failure")
        return super().generate(request)


class _FailForPromptMarkerProvider(_EvidenceEchoProvider):
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.marker in request.prompt:
            raise RuntimeError("simulated document-specific provider failure")
        return super().generate(request)


def test_controlled_analysis_is_evidence_bound(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    result = run_exact_phrase_analysis(project, [revision], "Apples")
    assert len(result["finding_ids"]) == 1
    finding = project.get_finding(result["finding_ids"][0])
    assert finding["analysis_run_id"] == result["analysis_run_id"]
    assert project.finding_evidence(finding["id"])


def test_direct_controlled_analysis_rejects_duplicate_frameworks(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "The evidence is limited.")

    with pytest.raises(ProjectError, match="must not contain duplicates"):
        run_controlled_cda_review(
            project,
            [revision],
            "unused",
            frameworks=("fairclough-text", "fairclough-text"),
        )


def test_batch_analysis_creates_one_run_per_revision(tmp_path):
    project = Project.init(tmp_path / "p")
    _, first = project.import_text("one.txt", "We must test.")
    _, second = project.import_text("two.txt", "They should test.")
    results = run_batch_textual_descriptors_analysis(project, [first, second], ("en",))
    assert len(results) == 2
    first_scope = project.db.execute(
        "SELECT scope_id FROM analysis_runs WHERE id=?", (results[0]["analysis_run_id"],)
    ).fetchone()[0]
    second_scope = project.db.execute(
        "SELECT scope_id FROM analysis_runs WHERE id=?", (results[1]["analysis_run_id"],)
    ).fetchone()[0]
    assert project.scope_revisions(first_scope) == [first]
    assert project.scope_revisions(second_scope) == [second]


def test_batch_textual_interrupt_marks_only_current_run_failed(monkeypatch, tmp_path):
    project = Project.init(tmp_path / "p")
    _, first = project.import_text("one.txt", "We must test.")
    _, second = project.import_text("two.txt", "They should test.")
    original_runner = analysis_module.run_textual_descriptors

    def interrupt_second(context, languages):
        if context.project.scope_revisions(context.scope_id) == [second]:
            raise KeyboardInterrupt("simulated document interruption")
        return original_runner(context, languages)

    monkeypatch.setattr(analysis_module, "run_textual_descriptors", interrupt_second)

    with pytest.raises(KeyboardInterrupt, match="simulated document interruption"):
        run_batch_textual_descriptors_analysis(project, [first, second], ("en",))

    rows = project.db.execute(
        "SELECT id,scope_id,status FROM analysis_runs"
    ).fetchall()
    runs_by_scope = {tuple(project.scope_revisions(row["scope_id"])): row for row in rows}
    assert set(runs_by_scope) == {(first,), (second,)}
    assert runs_by_scope[(first,)]["status"] == "COMPLETED"
    assert runs_by_scope[(second,)]["status"] == "FAILED"


def test_batch_analysis_rejects_duplicate_revision_before_creating_runs(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("one.txt", "We must test.")

    with pytest.raises(ProjectError, match="must not contain duplicates"):
        run_batch_textual_descriptors_analysis(project, [revision, revision], ("en",))
    with pytest.raises(ProjectError, match="must not contain duplicates"):
        run_batch_controlled_cda_review(project, [revision, revision], "missing-preset")

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


def test_batch_controlled_cda_creates_isolated_runs_per_document(tmp_path):
    project = Project.init(tmp_path / "p")
    context = "The review attached new candidate examples for the policy."
    first_document, first = project.import_text("one.txt", f"We must act.\n\n{context}")
    second_document, second = project.import_text("two.txt", f"They should wait.\n\n{context}")
    project.set_document_metadata(first_document, "language", "en")
    project.set_document_metadata(second_document, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    reranker = _FixtureReranker()

    results = run_batch_controlled_cda_review(
        project,
        [first, second],
        "review",
        frameworks=("fairclough-text",),
        provider_factory=lambda endpoint, reference, provider: _EvidenceEchoProvider(),
        reranking_gateway=ModelGateway({}, rerankers={reranker.provider_id: reranker}),
        reranking_provider_id=reranker.provider_id,
        reranking_credential_alias="research",
        reranking_model="fixture-v1",
    )

    assert len(results) == 2
    scopes = [
        project.db.execute(
            "SELECT scope_id FROM analysis_runs WHERE id=?", (item["analysis_run_id"],)
        ).fetchone()[0]
        for item in results
    ]
    assert project.scope_revisions(scopes[0]) == [first]
    assert project.scope_revisions(scopes[1]) == [second]
    for result, revision in zip(results, (first, second), strict=True):
        summary = project.analysis_run_summary(result["analysis_run_id"])
        reranking_runs = summary["reranking_runs"]
        assert isinstance(reranking_runs, list) and reranking_runs
        assert all(row["credential_alias"] == "research" for row in reranking_runs)
        for reranking_run in reranking_runs:
            retrieval = project.db.execute(
                "SELECT scope_id FROM retrieval_runs WHERE id=?",
                (reranking_run["retrieval_run_id"],),
            ).fetchone()
            assert project.scope_revisions(retrieval["scope_id"]) == [revision]


def test_batch_failure_isolates_runs_and_resume_keeps_original_scope(tmp_path):
    project = Project.init(tmp_path / "p")
    first_document, first = project.import_text("one.txt", "We must act.")
    second_document, second = project.import_text(
        "two.txt", "We must act FAILMARK now.\n\nA second sentence provides context."
    )
    project.set_document_metadata(first_document, "language", "en")
    project.set_document_metadata(second_document, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))

    with pytest.raises(RuntimeError, match="document-specific provider failure"):
        run_batch_controlled_cda_review(
            project,
            [first, second],
            "review",
            frameworks=("fairclough-text",),
            provider_factory=lambda endpoint, reference, provider: _FailForPromptMarkerProvider(
                "FAILMARK"
            ),
        )

    rows = project.db.execute(
        "SELECT id,scope_id,status FROM analysis_runs ORDER BY started_at,id"
    ).fetchall()
    runs_by_scope = {tuple(project.scope_revisions(row["scope_id"])): row for row in rows}
    first_run = runs_by_scope[(first,)]
    failed_run = runs_by_scope[(second,)]
    assert first_run["status"] == "COMPLETED"
    assert failed_run["status"] == "FAILED"

    for run_row, expected_revision in ((first_run, first), (failed_run, second)):
        finding_ids = project.db.execute(
            "SELECT id FROM findings WHERE analysis_run_id=?", (run_row["id"],)
        ).fetchall()
        for finding_row in finding_ids:
            for evidence_row in project.finding_evidence(finding_row["id"]):
                if evidence_row["span_id"]:
                    span = project.db.execute(
                        "SELECT revision_id FROM source_spans WHERE id=?",
                        (evidence_row["span_id"],),
                    ).fetchone()
                    assert span["revision_id"] == expected_revision

    original_scope_id = failed_run["scope_id"]
    later_document, later_revision = project.import_text(
        "later.txt", "A later document is out of scope."
    )
    project.set_document_metadata(later_document, "language", "en")
    resumed = resume_controlled_cda_review(
        project,
        failed_run["id"],
        provider_factory=lambda endpoint, reference, provider: _EvidenceEchoProvider(),
    )

    assert project.scope_revisions(original_scope_id) == [second]
    assert project.scope_revisions(
        project.db.execute(
            "SELECT scope_id FROM analysis_runs WHERE id=?", (resumed["analysis_run_id"],)
        ).fetchone()["scope_id"]
    ) == [second]
    assert project.db.execute(
        "SELECT status FROM analysis_runs WHERE id=?", (failed_run["id"],)
    ).fetchone()["status"] == "COMPLETED"
    assert project.scope_revisions(original_scope_id) != [later_revision]


def test_controlled_cda_review_keeps_one_scope_and_versions_each_framework(tmp_path):
    project = Project.init(tmp_path / "p")
    selected_document, selected = project.import_text(
        "selected.txt", "We must act. A neutral exception is recorded."
    )
    project.set_document_metadata(selected_document, "language", "en")
    project.import_text("outside.txt", "We must not enter this review.")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    provider = _CounterevidenceAwareProvider()

    result = run_controlled_cda_review(
        project,
        [selected],
        "review",
        counterevidence_queries={"descriptive_candidate_inventory": "exception"},
        provider_factory=lambda endpoint, reference, provider_id: provider,
    )

    assert len(result["seed_finding_ids"]) == 1
    expected_interpretations = len(DEFAULT_CDA_FRAMEWORKS) - 1
    assert len(result["reviewed_finding_ids"]) == expected_interpretations
    assert len(result["critical_review_finding_ids"]) == expected_interpretations
    synthesis_id = result["synthesis_finding_id"]
    assert synthesis_id
    synthesis_finding = project.get_finding(synthesis_id)
    assert synthesis_finding["skill_id"] == "builtin.synthesis"
    assert project.finding_sources(synthesis_id)
    assert {row["source_finding_id"] for row in project.finding_sources(synthesis_id)} == set(
        result["critical_review_finding_ids"]
    )
    assert project.finding_evidence(synthesis_id)
    bundle = json.loads(
        export_bundle(project, result["analysis_run_id"], tmp_path / "bundle").read_text()
    )
    assert any(item["finding_id"] == synthesis_id for item in bundle["finding_lineage"])
    assert len(result["counterevidence_skill_run_ids"]) == 1
    assert (
        project.db.execute(
            "SELECT status FROM analysis_runs WHERE id=?", (result["analysis_run_id"],)
        ).fetchone()[0]
        == "COMPLETED"
    )
    for finding_id in result["reviewed_finding_ids"]:
        finding = project.get_finding(finding_id)
        assert finding["analysis_run_id"] == result["analysis_run_id"]
        assert finding["skill_id"].removeprefix("builtin.") in DEFAULT_CDA_FRAMEWORKS
        assert project.finding_evidence(finding_id)
    for finding_id in result["critical_review_finding_ids"]:
        finding = project.get_finding(finding_id)
        assert finding["skill_id"] == "builtin.critical-reviewer"
        assert project.finding_evidence(finding_id)
    status = result["counterevidence_statuses"][result["seed_finding_ids"][0]]
    assert status["status"] == "COMPLETED"
    assert status["candidate_count"] == 1
    assert status["coverage_id"]
    automatic = result["alternative_candidate_statuses"][result["seed_finding_ids"][0]]
    assert automatic["status"] == "COMPLETED"
    assert automatic["coverage_mode"] == "TARGETED"
    assert len(automatic["retrieval_run_ids"]) == len(result["critical_review_finding_ids"])
    assert "does not establish absence" in str(automatic["reason"])
    hits = project.db.execute(
        "SELECT h.source_span_id FROM retrieval_hits h WHERE h.retrieval_run_id=?",
        (status["retrieval_run_id"],),
    ).fetchall()
    assert [project.span_text(item["source_span_id"]) for item in hits] == ["exception"]
    assert all(
        item["role"] != "COUNTEREVIDENCE"
        for item in project.finding_evidence(result["seed_finding_ids"][0])
    )
    assert provider.counterevidence_ids
    critical_links = project.finding_evidence(result["critical_review_finding_ids"][0])
    assert any(item["role"] == "COUNTEREVIDENCE" for item in critical_links)
    assert any(
        project.span_text(item["span_id"]) == "exception"
        for item in critical_links
        if item["role"] == "COUNTEREVIDENCE"
    )


def test_controlled_cda_resume_reuses_completed_reviews_after_synthesis_failure(tmp_path):
    project = Project.init(tmp_path / "p")
    speech_document, revision = project.import_text("speech.txt", "We must act.")
    project.set_document_metadata(speech_document, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    failing_provider = _FailOnCallProvider(fail_on_call=3)

    with pytest.raises(RuntimeError, match="simulated transient"):
        run_controlled_cda_review(
            project,
            [revision],
            "review",
            frameworks=("fairclough-text", "critical-reviewer"),
            provider_factory=lambda endpoint, reference, provider: failing_provider,
        )

    run_id = project.db.execute(
        "SELECT id FROM analysis_runs ORDER BY started_at DESC,id DESC LIMIT 1"
    ).fetchone()["id"]
    assert (
        project.db.execute("SELECT status FROM analysis_runs WHERE id=?", (run_id,)).fetchone()[
            "status"
        ]
        == "FAILED"
    )

    assert (
        project.db.execute(
            "SELECT status FROM skill_runs WHERE analysis_run_id=? AND skill_id='builtin.synthesis'",
            (run_id,),
        ).fetchone()["status"]
        == "FAILED"
    )

    resumed_provider = _FailOnCallProvider(fail_on_call=99)
    result = resume_controlled_cda_review(
        project,
        run_id,
        provider_factory=lambda endpoint, reference, provider: resumed_provider,
    )

    assert resumed_provider.calls == 1
    assert result["synthesis_finding_id"]
    assert (
        project.db.execute("SELECT status FROM analysis_runs WHERE id=?", (run_id,)).fetchone()[
            "status"
        ]
        == "COMPLETED"
    )
    stages = project.db.execute(
        "SELECT skill_id,status FROM skill_runs WHERE analysis_run_id=? ORDER BY rowid", (run_id,)
    ).fetchall()
    assert [row["status"] for row in stages if row["skill_id"] == "builtin.fairclough-text"] == [
        "COMPLETED"
    ]
    assert [row["status"] for row in stages if row["skill_id"] == "builtin.critical-reviewer"] == [
        "COMPLETED"
    ]
    assert [row["status"] for row in stages if row["skill_id"] == "builtin.synthesis"] == [
        "FAILED",
        "COMPLETED",
    ]


def test_resume_alternative_candidate_lineage_excludes_failed_critical_review(tmp_path):
    project = Project.init(tmp_path / "p")
    document_id, revision = project.import_text(
        "speech.txt", "We must act. A neutral reading offers another context."
    )
    project.set_document_metadata(document_id, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    failing_provider = _FailOnCallProvider(fail_on_call=2)

    with pytest.raises(RuntimeError, match="simulated transient"):
        run_controlled_cda_review(
            project,
            [revision],
            "review",
            frameworks=("fairclough-text", "critical-reviewer"),
            provider_factory=lambda endpoint, reference, provider: failing_provider,
        )

    run_id = project.db.execute(
        "SELECT id FROM analysis_runs WHERE status='FAILED'"
    ).fetchone()["id"]
    seed_id = project.db.execute(
        "SELECT f.id FROM findings f JOIN skill_runs s ON s.id=f.skill_run_id "
        "WHERE f.analysis_run_id=? AND s.skill_id='builtin.evidence-extraction' "
        "AND s.status='COMPLETED' ORDER BY f.created_at,f.id LIMIT 1",
        (run_id,),
    ).fetchone()["id"]
    failed_alternative_retrievals = project.db.execute(
        "SELECT COUNT(*) FROM retrieval_runs r JOIN skill_runs s ON s.id=r.skill_run_id "
        "WHERE r.analysis_run_id=? AND r.retrieval_plan_ref=? "
        "AND s.skill_id='builtin.critical-reviewer' AND s.status='FAILED'",
        (run_id, "retrieval/counterevidence-alternative-lexical-v1"),
    ).fetchone()[0]
    assert failed_alternative_retrievals == 1

    result = resume_controlled_cda_review(
        project,
        run_id,
        provider_factory=lambda endpoint, reference, provider: _EvidenceEchoProvider(),
    )

    automatic = result["alternative_candidate_statuses"][seed_id]
    assert automatic["status"] == "COMPLETED"
    assert len(automatic["retrieval_run_ids"]) == len(result["critical_review_finding_ids"]) == 1
    returned_statuses = project.db.execute(
        "SELECT DISTINCT s.status FROM retrieval_runs r JOIN skill_runs s ON s.id=r.skill_run_id "
        "WHERE r.id=?",
        (automatic["retrieval_run_ids"][0],),
    ).fetchall()
    assert [row["status"] for row in returned_statuses] == ["COMPLETED"]


def test_resume_rejects_changed_reranking_credential_reference(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text(
        "selected.txt", "We must act.\n\nSome related exception context is also present."
    )
    document_id = project.db.execute("SELECT id FROM documents").fetchone()[0]
    project.set_document_metadata(document_id, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "university", "DL_MODEL", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "university", "fixture"))
    project.save_credential_reference(CredentialReference("jina-reranker", "lab", "JINA_ORIGINAL"))

    class LocalJinaFixtureReranker(_FixtureReranker):
        provider_id = "jina-reranker"

    with pytest.raises(ModelResponseError):
        run_controlled_cda_review(
            project,
            [revision],
            "review",
            frameworks=("fairclough-text",),
            provider_factory=lambda endpoint, reference, provider: _MalformedReviewProvider(),
            reranking_gateway=ModelGateway(
                {}, rerankers={"jina-reranker": LocalJinaFixtureReranker()}
            ),
            reranking_provider_id="jina-reranker",
            reranking_credential_alias="lab",
            reranking_model="fixture-reranker",
            reranking_top_n=1,
        )

    run_row = project.db.execute(
        "SELECT id,config_json FROM analysis_runs WHERE status='FAILED' "
        "ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    run_config = json.loads(run_row["config_json"])
    assert run_config["reranking"]["credential_reference_hash"]
    assert "JINA_ORIGINAL" not in run_row["config_json"]
    project.save_credential_reference(
        CredentialReference("jina-reranker", "lab", "JINA_REPLACEMENT")
    )
    provider_calls: list[str | None] = []

    with pytest.raises(ModelProviderError, match="refusing implicit credential substitution"):
        resume_controlled_cda_review(
            project,
            run_row["id"],
            provider_factory=lambda endpoint, reference, provider: (
                provider_calls.append(reference) or _EvidenceEchoProvider()
            ),
        )

    assert provider_calls == []
    assert (
        project.db.execute(
            "SELECT status FROM analysis_runs WHERE id=?", (run_row["id"],)
        ).fetchone()["status"]
        == "FAILED"
    )


def test_controlled_run_rejects_gateway_built_before_credential_remap(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("selected.txt", "We must act.")
    document_id = project.db.execute("SELECT id FROM documents").fetchone()[0]
    project.set_document_metadata(document_id, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "lab", "DL_MODEL", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "lab", "fixture"))
    project.save_credential_reference(CredentialReference("jina-reranker", "lab", "JINA_OLD"))
    reranking_gateway = build_reranking_gateway(project, "jina-reranker", "lab")
    project.save_credential_reference(CredentialReference("jina-reranker", "lab", "JINA_NEW"))

    with pytest.raises(ModelProviderError, match="gateway credential does not match"):
        run_controlled_cda_review(
            project,
            [revision],
            "review",
            frameworks=("fairclough-text",),
            provider_factory=lambda endpoint, reference, provider: _EvidenceEchoProvider(),
            reranking_gateway=reranking_gateway,
            reranking_provider_id="jina-reranker",
            reranking_credential_alias="lab",
            reranking_model="fixture-reranker",
        )

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


def test_controlled_cda_persists_skipped_frameworks_when_seed_extraction_is_empty(tmp_path):
    project = Project.init(tmp_path / "p")
    report_document, revision = project.import_text("report.txt", "A quiet afternoon followed.")
    project.set_document_metadata(report_document, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))

    def unexpected_provider(*_args):
        raise AssertionError("no model should run without descriptive seed evidence")

    result = run_controlled_cda_review(
        project,
        [revision],
        "review",
        frameworks=("fairclough-text", "critical-reviewer"),
        provider_factory=unexpected_provider,
    )

    assert result["seed_finding_ids"] == []
    assert result["reviewed_finding_ids"] == []
    assert result["critical_review_finding_ids"] == []
    assert result["synthesis_finding_id"] is None
    assert any("no corpus-absence claim" in warning for warning in result["warnings"])
    skipped = project.db.execute(
        "SELECT skill_id,status,skip_reason FROM skill_runs "
        "WHERE analysis_run_id=? ORDER BY skill_id",
        (result["analysis_run_id"],),
    ).fetchall()
    assert [(row["skill_id"], row["status"]) for row in skipped] == [
        ("builtin.critical-reviewer", "SKIPPED"),
        ("builtin.evidence-extraction", "COMPLETED"),
        ("builtin.fairclough-text", "SKIPPED"),
        ("builtin.synthesis", "SKIPPED"),
        ("builtin.textual-descriptors", "COMPLETED"),
    ]
    assert all(
        row["skip_reason"]
        == (
            "NO_VALIDATED_REVIEW_FINDINGS"
            if row["skill_id"] == "builtin.synthesis"
            else "NO_DESCRIPTIVE_SEED_EVIDENCE"
        )
        for row in skipped
        if row["status"] == "SKIPPED"
    )
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM model_executions WHERE analysis_run_id=?",
            (result["analysis_run_id"],),
        ).fetchone()[0]
        == 0
    )


def test_controlled_cda_warns_when_one_framework_has_no_candidate_evidence(tmp_path):
    project = Project.init(tmp_path / "p")
    document, revision = project.import_text(
        "report.txt", "The institutional responsibility remains unclear."
    )
    project.set_document_metadata(document, "language", "fr")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))

    result = run_controlled_cda_review(
        project,
        [revision],
        "review",
        frameworks=("fairclough-text", "van-leeuwen-social-actors"),
        languages=("fr",),
        research_questions={"fairclough-text": "institutional responsibility"},
        provider_factory=lambda endpoint, reference, provider: _EvidenceEchoProvider(),
    )

    skipped = project.db.execute(
        "SELECT status,skip_reason FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.van-leeuwen-social-actors'",
        (result["analysis_run_id"],),
    ).fetchone()
    assert skipped is not None
    assert (skipped["status"], skipped["skip_reason"]) == (
        "SKIPPED",
        "NO_SEED_OR_QUESTION_CANDIDATE_EVIDENCE",
    )
    assert (
        project.db.execute(
            "SELECT status FROM analysis_runs WHERE id=?", (result["analysis_run_id"],)
        ).fetchone()["status"]
        == "COMPLETED"
    )
    assert result["reviewed_finding_ids"]
    assert all(
        project.get_finding(finding_id)["skill_id"] == "builtin.fairclough-text"
        for finding_id in result["reviewed_finding_ids"]
    )
    assert result["synthesis_finding_id"]
    assert any(
        "van-leeuwen-social-actors" in warning
        and "skipped" in warning
        and "synthesis does not cover" in warning
        for warning in result["warnings"]
    )
    assert {
        row["source_finding_id"]
        for row in project.finding_sources(result["synthesis_finding_id"])
    } == set(result["reviewed_finding_ids"])


def test_controlled_cda_exposes_counterevidence_that_was_not_run(tmp_path):
    project = Project.init(tmp_path / "p")
    selected_document, selected = project.import_text("selected.txt", "We must act.")
    project.set_document_metadata(selected_document, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))

    result = run_controlled_cda_review(
        project,
        [selected],
        "review",
        frameworks=("fairclough-text",),
        provider_factory=lambda endpoint, reference, provider: _EvidenceEchoProvider(),
    )

    assert len(result["counterevidence_skill_run_ids"]) == 1
    seed = result["seed_finding_ids"][0]
    status = result["counterevidence_statuses"][seed]
    assert status["status"] == "NOT_RUN"
    assert status["coverage_id"] is None
    assert status["skill_run_id"] == result["counterevidence_skill_run_ids"][0]
    assert "no counterevidence" in str(status["reason"])
    automatic = result["alternative_candidate_statuses"][seed]
    assert automatic["status"] == "NOT_RUN"
    assert automatic["retrieval_run_ids"] == []
    assert "Critical reviewer was not selected" in str(automatic["reason"])
    skipped = project.db.execute(
        "SELECT status,skip_reason FROM skill_runs WHERE id=?", (status["skill_run_id"],)
    ).fetchone()
    assert skipped["status"] == "SKIPPED"
    assert skipped["skip_reason"] == "NO_RESEARCHER_PROVIDED_CONTRAST_QUERY"


@pytest.mark.parametrize(
    "queries, message",
    [
        ({"descriptive_candidate_inventroy": "exception"}, "no current evidence-extraction"),
        ({"descriptive_candidate_inventory": "  "}, "must be non-empty"),
        ({"descriptive_candidate_inventory": 42}, "must be strings"),
    ],
)
def test_controlled_cda_rejects_unusable_counterevidence_queries_before_run(
    tmp_path, queries, message
):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("source.txt", "We must act.")

    with pytest.raises(ProjectError, match=message):
        run_controlled_cda_review(
            project,
            [revision],
            "not-needed-because-validation-precedes-routing",
            frameworks=("fairclough-text",),
            counterevidence_queries=queries,
        )

    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


def test_controlled_cda_records_candidate_limit_and_resumes_without_rescanning(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "p")
    document, revision = project.import_text(
        "selected.txt", "We must act. Exception one. Exception two. Exception three."
    )
    project.set_document_metadata(document, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    monkeypatch.setattr(skills_module, "COUNTEREVIDENCE_MAX_CANDIDATES", 2)
    failing_provider = _FailOnCallProvider(fail_on_call=2)

    with pytest.raises(RuntimeError, match="simulated transient provider failure"):
        run_controlled_cda_review(
            project,
            [revision],
            "review",
            frameworks=("fairclough-text",),
            counterevidence_queries={"descriptive_candidate_inventory": "Exception"},
            provider_factory=lambda endpoint, reference, provider: failing_provider,
        )

    run_id = project.db.execute(
        "SELECT id FROM analysis_runs ORDER BY started_at DESC,id DESC LIMIT 1"
    ).fetchone()["id"]
    limited_run = project.db.execute(
        "SELECT id,status,skip_reason FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.counterevidence-lexical'",
        (run_id,),
    ).fetchone()
    assert limited_run["status"] == "SKIPPED"
    assert limited_run["skip_reason"].startswith("RETRIEVAL_LIMIT_REACHED:")
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM retrieval_runs WHERE skill_run_id=?", (limited_run["id"],)
        ).fetchone()[0]
        == 0
    )

    resumed_provider = _FailOnCallProvider(fail_on_call=99)
    result = resume_controlled_cda_review(
        project,
        run_id,
        provider_factory=lambda endpoint, reference, provider: resumed_provider,
    )

    assert resumed_provider.calls == 1
    assert result["counterevidence_statuses"][result["seed_finding_ids"][0]]["status"] == (
        "LIMIT_REACHED"
    )
    assert any("RETRIEVAL_LIMIT_REACHED" in warning for warning in result["warnings"])
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM skill_runs WHERE analysis_run_id=? "
            "AND skill_id='builtin.counterevidence-lexical'",
            (run_id,),
        ).fetchone()[0]
        == 1
    )
    assert (
        project.db.execute("SELECT status FROM analysis_runs WHERE id=?", (run_id,)).fetchone()[0]
        == "COMPLETED"
    )


def test_controlled_cda_review_resolves_and_persists_framework_model_routing(tmp_path):
    project = Project.init(tmp_path / "p")
    selected_document, revision = project.import_text("selected.txt", "We must act.")
    project.set_document_metadata(selected_document, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(
        ModelPreset("default", "openai-compatible", "fixture", "default-model")
    )
    project.save_model_preset(ModelPreset("text", "openai-compatible", "fixture", "text-model"))
    project.save_model_routing_profile("routing", {"fairclough-text": "text"}, "default")
    models: list[str] = []

    result = run_controlled_cda_review(
        project,
        [revision],
        "default",
        frameworks=("fairclough-text",),
        routing_profile="routing",
        provider_factory=lambda endpoint, reference, provider: _RecordingEvidenceProvider(models),
    )

    config = json.loads(
        project.db.execute(
            "SELECT config_json FROM analysis_runs WHERE id=?", (result["analysis_run_id"],)
        ).fetchone()[0]
    )
    assert models == ["text-model", "default-model"]
    assert config["framework_presets"] == {"fairclough-text": "text", "synthesis": "default"}
    snapshot = config["framework_preset_snapshots"]["fairclough-text"]
    assert snapshot["model"] == "text-model"
    assert snapshot["credential_alias"] == "fixture"
    assert "DL_TEST" not in json.dumps(snapshot)
    assert config["framework_preset_snapshots"]["synthesis"]["model"] == "default-model"
    assert config["preset_snapshot_version"] == 2


def test_controlled_cda_review_honors_each_route_and_snapshots_provider_credentials(tmp_path):
    project = Project.init(tmp_path / "p")
    selected_document, revision = project.import_text("selected.txt", "We must act.")
    project.set_document_metadata(selected_document, "language", "en")
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "university", "DL_UNIVERSITY_KEY", "http://review.test/v1"
        )
    )
    project.save_credential_reference(
        CredentialReference(
            "anthropic-compatible", "lab", "DL_LAB_KEY", "http://synthesis.test/v1"
        )
    )
    project.save_model_preset(
        ModelPreset("review", "openai-compatible", "university", "review-model")
    )
    project.save_model_preset(
        ModelPreset("synthesis", "anthropic-compatible", "lab", "synthesis-model")
    )
    project.save_model_routing_profile(
        "fully-routed",
        {"fairclough-text": "review", "synthesis": "synthesis"},
    )
    requests: list[tuple[str, str | None, str]] = []
    models: list[str] = []

    def provider_factory(endpoint, reference, provider_id):
        requests.append((endpoint, reference, provider_id))
        provider = _RecordingEvidenceProvider(models)
        provider.provider_id = provider_id
        return provider

    result = run_controlled_cda_review(
        project,
        [revision],
        "unused-default",
        frameworks=("fairclough-text",),
        routing_profile="fully-routed",
        provider_factory=provider_factory,
    )

    assert models == ["review-model", "synthesis-model"]
    assert requests == [
        ("http://review.test/v1", "DL_UNIVERSITY_KEY", "openai-compatible"),
        ("http://synthesis.test/v1", "DL_LAB_KEY", "anthropic-compatible"),
    ]
    config = json.loads(
        project.db.execute(
            "SELECT config_json FROM analysis_runs WHERE id=?", (result["analysis_run_id"],)
        ).fetchone()[0]
    )
    assert config["framework_presets"] == {
        "fairclough-text": "review",
        "synthesis": "synthesis",
    }
    assert config["framework_preset_snapshots"]["fairclough-text"]["provider_id"] == (
        "openai-compatible"
    )
    assert config["framework_preset_snapshots"]["fairclough-text"]["credential_alias"] == (
        "university"
    )
    assert config["framework_preset_snapshots"]["synthesis"]["provider_id"] == (
        "anthropic-compatible"
    )
    assert config["framework_preset_snapshots"]["synthesis"]["credential_alias"] == "lab"
    serialized = json.dumps(config)
    assert "DL_UNIVERSITY_KEY" not in serialized
    assert "DL_LAB_KEY" not in serialized
    ledger = project.db.execute(
        "SELECT me.provider_id,me.credential_alias,me.model,me.skill_run_id,sr.skill_id,"
        "sr.analysis_run_id FROM model_executions me LEFT JOIN skill_runs sr "
        "ON sr.id=me.skill_run_id WHERE me.analysis_run_id=? ORDER BY me.rowid",
        (result["analysis_run_id"],),
    ).fetchall()
    linked_ledger = [
        (
            row["provider_id"],
            row["credential_alias"],
            row["model"],
            row["skill_id"],
            row["analysis_run_id"],
        )
        for row in ledger
    ]
    assert linked_ledger == [
        (
            "openai-compatible",
            "university",
            "review-model",
            "builtin.fairclough-text",
            result["analysis_run_id"],
        ),
        (
            "anthropic-compatible",
            "lab",
            "synthesis-model",
            "builtin.synthesis",
            result["analysis_run_id"],
        ),
    ]
    assert all(row["skill_run_id"] for row in ledger)
    expected_skill_run_ids = {row["skill_run_id"] for row in ledger}
    snapshots = config["framework_preset_snapshots"]
    assert isinstance(snapshots, dict)
    for row, stage in zip(ledger, ("fairclough-text", "synthesis"), strict=True):
        stage_snapshot = snapshots[stage]
        assert row["provider_id"] == stage_snapshot["provider_id"]
        assert row["model"] == stage_snapshot["model"]

    summary = project.analysis_run_summary(result["analysis_run_id"])
    summary_executions = summary["model_executions"]
    assert isinstance(summary_executions, list)
    assert {item["skill_run_id"] for item in summary_executions} == expected_skill_run_ids
    report = analysis_payload(project, result["analysis_run_id"])
    assert {item["skill_run_id"] for item in report["model_executions"]} == expected_skill_run_ids
    bundle = json.loads(
        export_bundle(project, result["analysis_run_id"], tmp_path / "bundle").read_text()
    )
    assert {item["skill_run_id"] for item in bundle["model_executions"]} == expected_skill_run_ids


@pytest.mark.parametrize(
    ("routes", "fallback", "expected_error"),
    [
        (
            {"fairclough_text": "review", "synthesis": "default"},
            "default",
            "unknown stage route",
        ),
        ({"synthesis": "default"}, None, "no preset for framework"),
    ],
)
def test_controlled_cda_review_rejects_invalid_or_missing_stage_routes_before_provider_calls(
    tmp_path, routes, fallback, expected_error
):
    project = Project.init(tmp_path / "p")
    selected_document, revision = project.import_text("selected.txt", "We must act.")
    project.set_document_metadata(selected_document, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_FIXTURE", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("default", "openai-compatible", "fixture", "default-model"))
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "review-model"))
    project.save_model_routing_profile("invalid-routing", routes, fallback)
    provider_calls: list[str] = []

    with pytest.raises(ProjectError, match=expected_error):
        run_controlled_cda_review(
            project,
            [revision],
            "default",
            frameworks=("fairclough-text",),
            routing_profile="invalid-routing",
            provider_factory=lambda endpoint, reference, provider_id: (
                provider_calls.append(provider_id) or _EvidenceEchoProvider()
            ),
        )

    assert provider_calls == []
    assert project.db.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0] == 0


def test_controlled_cda_review_rejects_critical_reviewer_without_prior_interpretation(tmp_path):
    project = Project.init(tmp_path / "p")
    selected_document, revision = project.import_text("selected.txt", "We must act.")
    project.set_document_metadata(selected_document, "language", "en")

    with pytest.raises(ProjectError, match="requires at least one interpretive"):
        run_controlled_cda_review(project, [revision], "review", frameworks=("critical-reviewer",))


def test_controlled_cda_failure_preserves_completed_steps_and_marks_failed_skill(tmp_path):
    project = Project.init(tmp_path / "p")
    selected_document, revision = project.import_text("selected.txt", "We must act.")
    project.set_document_metadata(selected_document, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))

    with pytest.raises(ModelResponseError):
        run_controlled_cda_review(
            project,
            [revision],
            "review",
            provider_factory=lambda endpoint, reference, provider: _MalformedReviewProvider(),
        )

    run = project.db.execute(
        "SELECT status FROM analysis_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    skill_rows = project.db.execute(
        "SELECT skill_id,status,error_type FROM skill_runs ORDER BY started_at,id"
    ).fetchall()
    assert run["status"] == "FAILED"
    completed_ids = {row["skill_id"] for row in skill_rows if row["status"] == "COMPLETED"}
    assert "builtin.textual-descriptors" in completed_ids
    assert "builtin.evidence-extraction" in completed_ids
    assert skill_rows[-1]["status"] == "FAILED"
    assert skill_rows[-1]["error_type"] == "ModelResponseError"


def test_controlled_cda_interruption_is_resumable_without_replaying_completed_stages(tmp_path):
    project = Project.init(tmp_path / "p")
    document, revision = project.import_text("selected.txt", "We must act.")
    project.set_document_metadata(document, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))

    class InterruptOnceProvider(_EvidenceEchoProvider):
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            self.calls += 1
            if self.calls <= 2:
                raise KeyboardInterrupt("simulated process interruption")
            return super().generate(request)

    provider = InterruptOnceProvider()
    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        run_controlled_cda_review(
            project,
            [revision],
            "review",
            frameworks=("fairclough-text",),
            provider_factory=lambda endpoint, reference, provider_id: provider,
        )

    run_id = project.db.execute(
        "SELECT id FROM analysis_runs WHERE status='FAILED'"
    ).fetchone()["id"]
    assert project.db.execute(
        "SELECT COUNT(*) FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.textual-descriptors'",
        (run_id,),
    ).fetchone()[0] == 1
    assert project.db.execute(
        "SELECT COUNT(*) FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.evidence-extraction'",
        (run_id,),
    ).fetchone()[0] == 1

    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        resume_controlled_cda_review(
            project,
            run_id,
            provider_factory=lambda endpoint, reference, provider_id: provider,
        )
    assert project.db.execute(
        "SELECT status FROM analysis_runs WHERE id=?", (run_id,)
    ).fetchone()[0] == "FAILED"

    resumed = resume_controlled_cda_review(
        project,
        run_id,
        provider_factory=lambda endpoint, reference, provider_id: provider,
    )

    assert resumed["analysis_run_id"] == run_id
    assert resumed["synthesis_finding_id"] is not None
    assert provider.calls >= 3
    assert project.db.execute(
        "SELECT status FROM analysis_runs WHERE id=?", (run_id,)
    ).fetchone()[0] == "COMPLETED"
    assert project.db.execute(
        "SELECT COUNT(*) FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.textual-descriptors'",
        (run_id,),
    ).fetchone()[0] == 1
    assert project.db.execute(
        "SELECT COUNT(*) FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.evidence-extraction'",
        (run_id,),
    ).fetchone()[0] == 1


def test_resume_controlled_cda_reuses_completed_stages_after_model_failure(tmp_path):
    project = Project.init(tmp_path / "p")
    selected_document, revision = project.import_text("selected.txt", "We must act.")
    project.set_document_metadata(selected_document, "language", "en")
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://fixture.test/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    endpoints: list[str] = []

    with pytest.raises(ModelResponseError):
        run_controlled_cda_review(
            project,
            [revision],
            "review",
            provider_factory=lambda endpoint, reference, provider: _MalformedReviewProvider(),
        )
    failed_run = project.db.execute(
        "SELECT id FROM analysis_runs WHERE status='FAILED'"
    ).fetchone()["id"]

    # A retry must use the model/parameters/endpoint resolved when this run began,
    # not mutable project presets or credential routing edited after the failure.
    project.save_model_preset(
        ModelPreset(
            "review",
            "openai-compatible",
            "fixture",
            "changed-model",
            parameters={"temperature": 0.4},
        )
    )
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "http://changed.test/v1")
    )

    class EndpointRecordingProvider(_EvidenceEchoProvider):
        def __init__(self, model_log, parameter_log) -> None:
            self.models = model_log
            self.parameters = parameter_log

        def generate(self, request: GenerationRequest) -> GenerationResponse:
            self.models.append(request.model)
            self.parameters.append(request.parameters or {})
            return super().generate(request)

    models: list[str] = []
    parameters: list[dict[str, object]] = []

    def provider_factory(endpoint, reference, provider):
        endpoints.append(endpoint)
        return EndpointRecordingProvider(models, parameters)

    resumed = resume_controlled_cda_review(
        project,
        failed_run,
        provider_factory=provider_factory,
    )

    assert resumed["analysis_run_id"] == failed_run
    assert (
        project.db.execute("SELECT status FROM analysis_runs WHERE id=?", (failed_run,)).fetchone()[
            "status"
        ]
        == "COMPLETED"
    )
    descriptor_runs = project.db.execute(
        "SELECT COUNT(*) FROM skill_runs WHERE analysis_run_id=? AND skill_id='builtin.textual-descriptors'",
        (failed_run,),
    ).fetchone()[0]
    assert descriptor_runs == 1
    skipped_counterevidence = project.db.execute(
        "SELECT status,skip_reason FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.counterevidence-lexical'",
        (failed_run,),
    ).fetchall()
    assert len(skipped_counterevidence) == 1
    assert skipped_counterevidence[0]["status"] == "SKIPPED"
    assert skipped_counterevidence[0]["skip_reason"] == "NO_RESEARCHER_PROVIDED_CONTRAST_QUERY"
    assert len(resumed["reviewed_finding_ids"]) == len(DEFAULT_CDA_FRAMEWORKS) - 1
    assert len(resumed["critical_review_finding_ids"]) == len(DEFAULT_CDA_FRAMEWORKS) - 1
    assert endpoints and set(endpoints) == {"http://fixture.test/v1"}
    assert set(models) == {"fixture"}
    assert parameters and all(item == {} for item in parameters)
    executions = project.db.execute(
        "SELECT model,parameters_json FROM model_executions WHERE analysis_run_id=?",
        (failed_run,),
    ).fetchall()
    assert executions and {row["model"] for row in executions} == {"fixture"}
    assert {row["parameters_json"] for row in executions} == {"{}"}


def test_resume_rejects_legacy_run_without_model_configuration_snapshot(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("selected.txt", "We must act.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(
        scope_id,
        {
            "profile": "general-cda-controlled-v3",
            "preset_name": "mutable-preset",
            "frameworks": ["fairclough-text"],
            "framework_prompt_snapshot_version": CONTROLLED_CDA_PROMPT_SNAPSHOT_VERSION,
            "framework_prompt_snapshots": _controlled_cda_prompt_snapshots(("fairclough-text",)),
            "synthesis_prompt_snapshot_version": CONTROLLED_CDA_PROMPT_SNAPSHOT_VERSION,
            "synthesis_prompt_snapshot": _controlled_cda_synthesis_prompt_snapshot(),
            "languages": ["en"],
            "framework_presets": {
                "fairclough-text": "mutable-preset",
                "synthesis": "mutable-preset",
            },
            "framework_preset_snapshots": {},
        },
    )
    project.start_run(run_id)
    project.fail_run(run_id, RuntimeError("fixture failure"))

    with pytest.raises(ProjectError, match="lacks immutable model preset snapshots"):
        resume_controlled_cda_review(project, run_id)

    assert (
        project.db.execute("SELECT status FROM analysis_runs WHERE id=?", (run_id,)).fetchone()[0]
        == "FAILED"
    )


def test_resume_rejects_changed_credential_reference_without_provider_call(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("selected.txt", "We must act.")
    project.set_document_metadata(
        project.db.execute("SELECT id FROM documents").fetchone()[0], "language", "en"
    )
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "university", "DL_ORIGINAL", "http://fixture.test/v1"
        )
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "university", "fixture"))

    with pytest.raises(ModelResponseError):
        run_controlled_cda_review(
            project,
            [revision],
            "review",
            frameworks=("fairclough-text",),
            provider_factory=lambda endpoint, reference, provider: _MalformedReviewProvider(),
        )
    run_id = project.db.execute(
        "SELECT id FROM analysis_runs WHERE status='FAILED' ORDER BY started_at DESC LIMIT 1"
    ).fetchone()["id"]
    project.save_credential_reference(
        CredentialReference(
            "openai-compatible", "university", "DL_REPLACEMENT", "http://fixture.test/v1"
        )
    )
    provider_calls: list[str | None] = []

    with pytest.raises(ModelProviderError, match="refusing implicit credential substitution"):
        resume_controlled_cda_review(
            project,
            run_id,
            provider_factory=lambda endpoint, reference, provider: (
                provider_calls.append(reference) or _EvidenceEchoProvider()
            ),
        )

    assert provider_calls == []
    assert (
        project.db.execute("SELECT status FROM analysis_runs WHERE id=?", (run_id,)).fetchone()[
            "status"
        ]
        == "FAILED"
    )
