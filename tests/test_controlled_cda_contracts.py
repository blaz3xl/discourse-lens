"""Offline contracts for the public controlled CDA workflow."""

import json
from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import EvidenceValidationError, ModelResponseError
from discourselens.models import (
    CredentialReference,
    GenerationRequest,
    GenerationResponse,
    ModelCapabilities,
    ModelPreset,
)


class _FixtureProvider:
    """Deterministic model boundary that never opens a network connection."""

    provider_id = "openai-compatible"

    def __init__(
        self,
        *,
        fail_first_synthesis: bool = False,
        invent_first_review_citation: bool = False,
        invent_synthesis_citation: bool = False,
    ) -> None:
        self.fail_first_synthesis = fail_first_synthesis
        self.invent_first_review_citation = invent_first_review_citation
        self.invent_synthesis_citation = invent_synthesis_citation
        self.payloads: list[dict[str, object]] = []
        self.failed_synthesis = False

    def capabilities(self, model: str) -> ModelCapabilities:
        del model
        return ModelCapabilities()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        marker = "UNTRUSTED_RESEARCH_DATA_JSON:\n"
        payload = cast(dict[str, object], json.loads(request.prompt.split(marker, 1)[1]))
        self.payloads.append(payload)
        evidence_records = cast(list[dict[str, object]], payload["evidence_records"])
        assert evidence_records
        is_synthesis = "source_findings" in payload
        if is_synthesis and self.fail_first_synthesis and not self.failed_synthesis:
            self.failed_synthesis = True
            raise RuntimeError("fixture synthesis interruption")

        evidence_id = str(evidence_records[0]["evidence_id"])
        if self.invent_first_review_citation and not is_synthesis:
            evidence_id = "provider-invented-evidence-id"
        response = {
            "interpretation": (
                "A synthesis with a fabricated citation [selected.txt · p. 999]."
                if is_synthesis and self.invent_synthesis_citation
                else "A bounded fixture interpretation of the supplied evidence."
            ),
            "alternative_interpretation": "The evidence may also admit a neutral reading.",
            "evidence_ids": [evidence_id],
            "counterevidence_ids": [],
            "limitations": ["Fixture model output is limited to the supplied records."],
        }
        if not is_synthesis and request.system_prompt and "Each observation must contain" in request.system_prompt:
            observation_record = next(
                item
                for item in evidence_records
                if isinstance(item.get("evidence_id"), str)
                and isinstance(item.get("excerpt"), str)
            )
            response["observations"] = [
                {
                    "evidence_id": observation_record["evidence_id"],
                    "excerpt": observation_record["excerpt"],
                    "description": "The passage contains a modal marker.",
                }
            ]
        return GenerationResponse(json.dumps(response), self.provider_id, request.model)


def _configure_fake_model(project: Project, monkeypatch, provider: _FixtureProvider) -> None:
    project.save_credential_reference(
        CredentialReference("openai-compatible", "fixture", "DL_TEST", "https://fixture.invalid/v1")
    )
    project.save_model_preset(ModelPreset("review", "openai-compatible", "fixture", "fixture"))
    monkeypatch.setattr(
        "discourselens.execution.OpenAICompatibleProvider",
        lambda *_args, **_kwargs: provider,
    )


def test_public_controlled_cda_reopens_and_resumes_with_frozen_scope_and_citations(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    provider = _FixtureProvider(fail_first_synthesis=True)
    with Project.init(root) as project:
        selected_document, selected_revision = project.import_text(
            "selected.txt",
            "Officials must respond to migration. The committee should explain exceptions.",
        )
        project.set_document_metadata(selected_document, "language", "en")
        project.import_text("outside.txt", "OUTSIDE_MARKER Officials must mention elephants.")
        _configure_fake_model(project, monkeypatch, provider)

        with pytest.raises(RuntimeError, match="fixture synthesis interruption"):
            project.analyze_controlled_cda(
                [selected_revision], "review", frameworks=("fairclough-text",)
            )

        run = project.db.execute(
            "SELECT id,scope_id,status FROM analysis_runs ORDER BY started_at DESC,id DESC LIMIT 1"
        ).fetchone()
        assert run is not None
        analysis_run_id = str(run["id"])
        scope_id = str(run["scope_id"])
        assert run["status"] == "FAILED"
        assert project.scope_revisions(scope_id) == [selected_revision]
        before = project.db.execute(
            "SELECT skill_id,status FROM skill_runs WHERE analysis_run_id=? ORDER BY rowid",
            (analysis_run_id,),
        ).fetchall()
        assert [
            (row["skill_id"], row["status"])
            for row in before
            if row["skill_id"] == "builtin.fairclough-text"
        ] == [("builtin.fairclough-text", "COMPLETED")]
        assert [
            (row["skill_id"], row["status"])
            for row in before
            if row["skill_id"] == "builtin.synthesis"
        ] == [("builtin.synthesis", "FAILED")]

    with Project.open(root) as reopened:
        late_document, late_revision = reopened.import_text(
            "late.txt", "LATE_MARKER Officials must discuss zebras."
        )
        reopened.set_document_metadata(late_document, "language", "en")
        result = reopened.resume_controlled_cda(analysis_run_id)

        assert result["analysis_run_id"] == analysis_run_id
        assert result["seed_finding_ids"]
        assert len(result["reviewed_finding_ids"]) == 1
        assert result["synthesis_finding_id"] is not None
        assert reopened.scope_revisions(scope_id) == [selected_revision]
        assert reopened.scope_revisions(
            str(
                reopened.db.execute(
                    "SELECT scope_id FROM analysis_runs WHERE id=?", (analysis_run_id,)
                ).fetchone()["scope_id"]
            )
        ) == [selected_revision]

        review_payloads = [
            payload for payload in provider.payloads if "source_findings" not in payload
        ]
        synthesis_payloads = [
            payload for payload in provider.payloads if "source_findings" in payload
        ]
        assert len(review_payloads) == 1
        assert len(synthesis_payloads) == 2
        assert all(
            "OUTSIDE_MARKER" not in json.dumps(payload) and "LATE_MARKER" not in json.dumps(payload)
            for payload in provider.payloads
        )

        finding_ids = [
            str(row["id"])
            for row in reopened.db.execute(
                "SELECT id FROM findings WHERE analysis_run_id=?", (analysis_run_id,)
            ).fetchall()
        ]
        for finding_id in finding_ids:
            assert reopened.validate_finding(finding_id).valid
            for evidence in reopened.finding_evidence(finding_id):
                if evidence["span_id"] is None:
                    continue
                source = reopened.db.execute(
                    "SELECT revision_id FROM source_spans WHERE id=?", (evidence["span_id"],)
                ).fetchone()
                assert source is not None
                assert source["revision_id"] == selected_revision

        synthesis_id = str(result["synthesis_finding_id"])
        source_ids = {
            str(row["source_finding_id"]) for row in reopened.finding_sources(synthesis_id)
        }
        cited_source_evidence = {
            str(evidence["id"])
            for source_id in source_ids
            for evidence in reopened.finding_evidence(source_id)
        }
        synthesis_evidence = {
            str(evidence["id"]) for evidence in reopened.finding_evidence(synthesis_id)
        }
        assert source_ids == set(result["reviewed_finding_ids"])
        assert synthesis_evidence <= cited_source_evidence
        assert (
            reopened.db.execute(
                "SELECT status FROM analysis_runs WHERE id=?", (analysis_run_id,)
            ).fetchone()["status"]
            == "COMPLETED"
        )
        retrieval_scopes = reopened.db.execute(
            "SELECT DISTINCT scope_id FROM retrieval_runs WHERE analysis_run_id=?",
            (analysis_run_id,),
        ).fetchall()
        assert all(
            reopened.scope_revisions(str(row["scope_id"])) == [selected_revision]
            for row in retrieval_scopes
        )
        assert late_revision not in reopened.scope_revisions(scope_id)


def test_public_controlled_cda_rejects_provider_citations_outside_supplied_evidence(
    tmp_path, monkeypatch
):
    provider = _FixtureProvider(invent_first_review_citation=True)
    with Project.init(tmp_path / "project") as project:
        document_id, revision = project.import_text("selected.txt", "We must respond to migration.")
        project.set_document_metadata(document_id, "language", "en")
        _configure_fake_model(project, monkeypatch, provider)

        with pytest.raises(EvidenceValidationError, match="unknown evidence ID"):
            project.analyze_controlled_cda([revision], "review", frameworks=("fairclough-text",))

        run = project.db.execute(
            "SELECT id,status FROM analysis_runs ORDER BY started_at DESC,id DESC LIMIT 1"
        ).fetchone()
        assert run is not None
        assert run["status"] == "FAILED"
        assert (
            project.db.execute(
                "SELECT COUNT(*) FROM findings WHERE analysis_run_id=? "
                "AND skill_id='builtin.fairclough-text'",
                (run["id"],),
            ).fetchone()[0]
            == 0
        )
        assert len(provider.payloads) == 1


def test_public_controlled_cda_rejects_inline_citation_in_synthesis(tmp_path, monkeypatch):
    provider = _FixtureProvider(invent_synthesis_citation=True)
    with Project.init(tmp_path / "project") as project:
        document_id, revision = project.import_text("selected.txt", "We must respond to migration.")
        project.set_document_metadata(document_id, "language", "en")
        _configure_fake_model(project, monkeypatch, provider)

        with pytest.raises(ModelResponseError, match="must not contain inline citations"):
            project.analyze_controlled_cda([revision], "review", frameworks=("fairclough-text",))

        run = project.db.execute(
            "SELECT id,status FROM analysis_runs ORDER BY started_at DESC,id DESC LIMIT 1"
        ).fetchone()
        assert run is not None
        assert run["status"] == "FAILED"
        assert (
            project.db.execute(
                "SELECT COUNT(*) FROM findings WHERE analysis_run_id=? "
                "AND skill_id='builtin.synthesis'",
                (run["id"],),
            ).fetchone()[0]
            == 0
        )


def test_controlled_cda_critical_reviewer_receives_unverified_description_separately(
    tmp_path, monkeypatch
):
    provider = _FixtureProvider()
    with Project.init(tmp_path / "structured-observation-pipeline") as project:
        document, revision = project.import_text("selected.txt", "Officials must respond.")
        project.set_document_metadata(document, "language", "en")
        _configure_fake_model(project, monkeypatch, provider)

        result = project.analyze_controlled_cda(
            [revision],
            "review",
            frameworks=("fairclough-text", "critical-reviewer"),
        )

        review_payloads = [payload for payload in provider.payloads if "source_findings" not in payload]
        assert len(review_payloads) == 2
        first_review_finding = project.get_finding(result["reviewed_finding_ids"][0])
        first_review_evidence_id = project.finding_evidence(
            result["reviewed_finding_ids"][0]
        )[0]["id"]
        assert review_payloads[1]["finding_observation"] == (
            "Core-validated verbatim textual excerpts:\n"
            f"[{first_review_evidence_id}] “Officials must respond.”"
        )
        assert "The passage contains a modal marker." not in str(
            review_payloads[1]["finding_observation"]
        )
        assert review_payloads[1]["unverified_observation_candidates"] == [
            {
                "candidate_id": project.finding_observation_candidates(
                    result["reviewed_finding_ids"][0]
                )[0]["id"],
                "evidence_id": first_review_evidence_id,
                "excerpt": "Officials must respond.",
                "description": "The passage contains a modal marker.",
                "status": "UNVERIFIED",
            }
        ]
        assert "The passage contains a modal marker." not in json.dumps(
            review_payloads[1]["evidence_records"]
        )
        critical_finding = project.get_finding(result["critical_review_finding_ids"][0])
        assert critical_finding["observation"] == review_payloads[1]["finding_observation"]
        assert first_review_finding["observation"] == review_payloads[1]["finding_observation"]
