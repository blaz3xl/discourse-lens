"""Export guards for historical credential-like alias values."""

import hashlib
import json

import pytest

from discourselens import Project
from discourselens.bundle import export_bundle, verify_bundle
from discourselens.errors import DocumentError, EvidenceValidationError, ProjectError
from discourselens.reporting import (
    analysis_payload,
    analysis_payload_page,
    write_analysis_report,
)
from discourselens.types import CoverageDescriptor, CoverageMode


def _summary_run_config(summary: dict[str, object]) -> dict[str, object]:
    analysis_run = summary.get("analysis_run")
    assert isinstance(analysis_run, dict)
    config = analysis_run.get("config")
    assert isinstance(config, dict)
    return config


def _create_reranking(project, tmp_path, credential_alias):
    _, revision = project.import_text("source.txt", "Migration policy. Migration discourse.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(scope_id, {"profile": "credential-export-fixture"})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope_id,
            population_definition="Two fixture passages",
        )
    )
    skill_run_id = project.create_skill_run(
        run_id,
        scope_id,
        "fixture.reranking",
        "1.0.0",
        "test",
        "fixture",
        coverage_id=coverage_id,
    )
    retrieval_id = project.create_retrieval_run(
        analysis_run_id=run_id,
        skill_run_id=skill_run_id,
        scope_id=scope_id,
        retrieval_plan_ref="retrieval/test-v1",
        query_terms=["Migration"],
        selection_policy="fixture-candidates",
        candidate_count=1,
        inspected_count=1,
        returned_count=1,
        candidate_limit_reached=False,
        coverage_id=coverage_id,
    )
    span_id = project.create_span(revision, 0, 9)
    hit_id = project.add_retrieval_hit(retrieval_id, span_id, "lexical", 1, 1.0)
    reranking_id = project.record_reranking(
        retrieval_id,
        provider_id="fixture-reranker",
        credential_alias=credential_alias,
        model="reranker-test",
        query="Migration",
        top_n=1,
        effective_top_n=1,
        candidate_count=1,
        ranked_hits=[(hit_id, 1, 0.9)],
    )
    return run_id, reranking_id


def _assert_exports_reject_alias(project, run_id, tmp_path, secret):
    report_destination = tmp_path / "reports" / "analysis.md"
    with pytest.raises(EvidenceValidationError) as report_error:
        write_analysis_report(project, run_id, report_destination, "markdown")
    assert secret not in str(report_error.value)
    assert any(
        indicator in str(report_error.value).casefold()
        for indicator in ("analysis run", "credential alias", "metadata", "provenance")
    )
    assert not report_destination.exists()
    assert not report_destination.parent.exists()

    bundle_destination = tmp_path / "bundle"
    with pytest.raises(ProjectError) as bundle_error:
        export_bundle(project, run_id, bundle_destination)
    assert secret not in str(bundle_error.value)
    assert any(
        indicator in str(bundle_error.value).casefold()
        for indicator in (
            "research bundle",
            "analysis run",
            "configuration",
            "credential alias",
            "metadata",
            "provenance",
        )
    )
    assert not bundle_destination.exists()


@pytest.mark.parametrize("secret", ["sk-proj-decoy-MUST-NOT-EXPORT"])
def test_report_and_bundle_reject_contaminated_legacy_run_configuration(tmp_path, secret):
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        run_id = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
        config_json = json.dumps({"profile": "fixture", "credential_alias": secret})
        project.db.execute("DROP TRIGGER analysis_runs_provenance_immutable")
        project.db.execute(
            "UPDATE analysis_runs SET config_json=?,config_hash=? WHERE id=?",
            (config_json, hashlib.sha256(config_json.encode("utf-8")).hexdigest(), run_id),
        )
        project.db.commit()

        _assert_exports_reject_alias(project, run_id, tmp_path, secret)
    finally:
        project.close()


def test_report_and_bundle_reject_credential_like_endpoint_host_in_run_snapshot(tmp_path):
    sentinel = "sk-proj-export-host-fixture-secret"
    endpoint = f"https://{sentinel}.example/v1"
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        config = {
            "framework_preset_snapshots": {
                "review": {"provider_id": "openai-compatible", "endpoint": endpoint}
            }
        }
        run_id = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
        project.db.execute("DROP TRIGGER analysis_runs_provenance_immutable")
        project.db.execute(
            "UPDATE analysis_runs SET config_json=?,config_hash=? WHERE id=?",
            (
                json.dumps(config, sort_keys=True),
                hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
                run_id,
            ),
        )
        project.db.commit()

        report_destination = tmp_path / "reports" / "analysis.json"
        with pytest.raises(EvidenceValidationError) as report_error:
            write_analysis_report(project, run_id, report_destination, "json")
        assert sentinel not in str(report_error.value)
        assert str(report_error.value) == "analysis run provenance is invalid"
        assert not report_destination.exists()

        bundle_destination = tmp_path / "bundle"
        with pytest.raises(ProjectError) as bundle_error:
            export_bundle(project, run_id, bundle_destination)
        assert sentinel not in str(bundle_error.value)
        assert str(bundle_error.value) == "analysis run configuration contains secret data"
        assert not bundle_destination.exists()
    finally:
        project.close()


def test_report_and_bundle_reject_contaminated_legacy_model_execution_alias(tmp_path):
    secret = "Bearer decoy-MUST-NOT-EXPORT"
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        run_id = project.create_run(project.snapshot([revision]), {})
        project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="university",
            model="fixture-model",
            reasoning=None,
            request_hash=hashlib.sha256(b"request").hexdigest(),
            response_hash=hashlib.sha256(b"response").hexdigest(),
            analysis_run_id=run_id,
        )
        project.db.execute(
            "UPDATE model_executions SET credential_alias=? WHERE analysis_run_id=?",
            (secret, run_id),
        )
        project.db.commit()

        _assert_exports_reject_alias(project, run_id, tmp_path, secret)
    finally:
        project.close()


@pytest.mark.parametrize(
    "credential_field",
    ["clientAccessToken", "apikey", "clientsecret", "privatekey", "sessiontoken"],
)
def test_model_execution_summaries_reject_secret_metadata_in_historical_rows(
    tmp_path, credential_field
):
    secret = "Bearer capability-fixture-secret"
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        run_id = project.create_run(
            project.snapshot([revision]),
            {
                "credential_alias": "university",
                "tokenizer": "default",
                "token_count": 4,
            },
        )
        execution_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="university",
            model="fixture-model",
            reasoning=None,
            request_hash=hashlib.sha256(b"request").hexdigest(),
            response_hash=hashlib.sha256(b"response").hexdigest(),
            analysis_run_id=run_id,
            capabilities={"effective": {"text": True}, "source": "adapter-default"},
        )
        summary = project.analysis_run_summary(run_id)
        config = _summary_run_config(summary)
        assert config["credential_alias"] == "university"
        assert config["tokenizer"] == "default"
        assert config["token_count"] == 4
        executions = summary.get("model_executions")
        assert isinstance(executions, list) and executions
        execution = executions[0]
        assert isinstance(execution, dict)
        assert execution.get("capabilities_json") == json.dumps(
            {"effective": {"text": True}, "source": "adapter-default"}, sort_keys=True
        )
        project.db.execute(
            "UPDATE model_executions SET capabilities_json=? WHERE id=?",
            (json.dumps({"provider": {credential_field: secret}}), execution_id),
        )
        project.db.commit()

        for summary in (
            lambda: project.analysis_run_summary(run_id),
            lambda: project.analysis_run_summary_page(run_id, limit=1),
        ):
            with pytest.raises(DocumentError, match="stored model execution metadata") as error:
                summary()
            assert secret not in str(error.value)
        stored = project.db.execute(
            "SELECT capabilities_json FROM model_executions WHERE id=?", (execution_id,)
        ).fetchone()
        assert stored["capabilities_json"] == json.dumps(
            {"provider": {credential_field: secret}}
        )
    finally:
        project.close()


def test_run_configuration_accepts_only_valid_credential_reference_fingerprints(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        fingerprint = "a" * 64
        run_id = project.create_run(
            project.snapshot([revision]), {"credential_reference_hash": fingerprint}
        )
        assert _summary_run_config(project.analysis_run_summary(run_id))[
            "credential_reference_hash"
        ] == fingerprint
        absent_hash_run = project.create_run(
            project.snapshot([revision]), {"credential_reference_hash": None}
        )
        assert _summary_run_config(project.analysis_run_summary(absent_hash_run))[
            "credential_reference_hash"
        ] is None
    finally:
        project.close()


@pytest.mark.parametrize(
    "fingerprint",
    ["DL_OPENAI", "a" * 63, "g" * 64],
)
def test_run_configuration_rejects_raw_or_malformed_credential_reference_hashes(
    tmp_path, fingerprint
):
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        with pytest.raises(DocumentError, match="configuration must not contain secrets"):
            project.create_run(
                project.snapshot([revision]),
                {"credential_reference_hash": fingerprint},
            )
    finally:
        project.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("credentialReferenceHash", "a" * 64),
        ("credential-reference-hash", "a" * 64),
        ("credentialReferenceHash", None),
        ("credential-reference-hash", None),
    ],
)
def test_run_configuration_rejects_normalized_credential_fingerprint_aliases(
    tmp_path, field, value
):
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        with pytest.raises(DocumentError, match="configuration must not contain secrets"):
            project.create_run(
                project.snapshot([revision]),
                {field: value},
            )
    finally:
        project.close()


def test_run_configuration_accepts_only_valid_canonical_credential_alias_fields(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        run_id = project.create_run(
            project.snapshot([revision]),
            {"credential_alias": "university", "reranking_credential_alias": "lab"},
        )
        config = _summary_run_config(project.analysis_run_summary(run_id))
        assert config["credential_alias"] == "university"
        assert config["reranking_credential_alias"] == "lab"
    finally:
        project.close()


@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("credential_alias", "sk-proj-secret-fixture"),
        ("reranking_credential_alias", "sk-proj-secret-fixture"),
        ("credential_alias", "Bearer secret-fixture"),
        ("reranking_credential_alias", "Bearer secret-fixture"),
    ],
)
def test_run_configuration_rejects_unsafe_credential_alias_fields(tmp_path, field, alias):
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        with pytest.raises(DocumentError, match="configuration must not contain secrets"):
            project.create_run(project.snapshot([revision]), {field: alias})
    finally:
        project.close()


def test_counterevidence_query_identifiers_are_not_treated_as_config_field_names(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        run_id = project.create_run(
            project.snapshot([revision]),
            {
                "counterevidence_queries": {
                    "api_key": "find counterevidence",
                    "sessiontoken": "look for alternate explanations",
                    "sk-profile-semantic-secret": "check claimed exception",
                }
            },
        )
        config = _summary_run_config(project.analysis_run_summary(run_id))
        assert config["counterevidence_queries"] == {
            "api_key": "find counterevidence",
            "sessiontoken": "look for alternate explanations",
            "sk-profile-semantic-secret": "check claimed exception",
        }
    finally:
        project.close()


def test_counterevidence_query_values_still_screen_nested_config_objects(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        with pytest.raises(DocumentError, match="configuration must not contain secrets"):
            project.create_run(
                project.snapshot([revision]),
                {
                    "counterevidence_queries": {
                        "api_key": {"provider_options": {"api_key": "fixture-secret"}}
                    }
                },
            )
    finally:
        project.close()


def test_report_and_bundle_reject_contaminated_legacy_reranking_alias(tmp_path):
    secret = "sk-proj-reranking-decoy-MUST-NOT-EXPORT"
    project = Project.init(tmp_path / "project")
    try:
        run_id, reranking_id = _create_reranking(project, tmp_path, "university")
        project.db.execute(
            "UPDATE reranking_runs SET credential_alias=? WHERE id=?",
            (secret, reranking_id),
        )
        project.db.commit()

        _assert_exports_reject_alias(project, run_id, tmp_path, secret)
        with pytest.raises(EvidenceValidationError, match="suspicious credential alias"):
            analysis_payload_page(project, run_id, limit=1)
    finally:
        project.close()


def test_bundle_verifier_rejects_alias_inside_serialized_execution_metadata(tmp_path):
    secret = "Bearer serialized-decoy-MUST-NOT-EXPORT"
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        run_id = project.create_run(project.snapshot([revision]), {})
        project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="university",
            model="fixture-model",
            reasoning=None,
            request_hash=hashlib.sha256(b"request").hexdigest(),
            response_hash=hashlib.sha256(b"response").hexdigest(),
            analysis_run_id=run_id,
        )
        manifest = export_bundle(project, run_id, tmp_path / "bundle" / "original")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["model_executions"][0]["parameters_json"] = json.dumps(
            {"provider": {"credential_alias": secret}}
        )
        unsigned_payload = {key: value for key, value in payload.items() if key != "integrity"}
        payload["integrity"]["digest"] = hashlib.sha256(
            json.dumps(
                unsigned_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(ProjectError, match="suspicious credential alias") as error:
            verify_bundle(manifest)
        assert secret not in str(error.value)
    finally:
        project.close()


def test_bundle_verifier_rejects_credential_like_endpoint_host_in_serialized_snapshot(tmp_path):
    sentinel = "sk-proj-bundle-host-fixture-secret"
    project = Project.init(tmp_path / "project")
    try:
        _, revision = project.import_text("source.txt", "Evidence stays local.")
        run_id = project.create_run(project.snapshot([revision]), {})
        manifest = export_bundle(project, run_id, tmp_path / "bundle" / "original")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["analysis_run"]["config"] = {
            "framework_preset_snapshots": {
                "review": {
                    "provider_id": "openai-compatible",
                    "endpoint": f"https://{sentinel}.example/v1",
                }
            }
        }
        unsigned_payload = {key: value for key, value in payload.items() if key != "integrity"}
        payload["integrity"]["digest"] = hashlib.sha256(
            json.dumps(
                unsigned_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        manifest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(ProjectError) as error:
            verify_bundle(manifest)
        assert "endpoint" in str(error.value)
        assert sentinel not in str(error.value)
    finally:
        project.close()


def test_safe_credential_alias_exports_still_pass(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        run_id, _ = _create_reranking(project, tmp_path, "university")
        project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="university",
            model="fixture-model",
            reasoning=None,
            request_hash=hashlib.sha256(b"request").hexdigest(),
            response_hash=hashlib.sha256(b"response").hexdigest(),
            analysis_run_id=run_id,
        )
        report = analysis_payload(project, run_id)
        destination = write_analysis_report(project, run_id, tmp_path / "safe-report.json", "json")
        manifest = export_bundle(project, run_id, tmp_path / "safe-bundle")

        assert any(
            execution["credential_alias"] == "university"
            for execution in report["model_executions"]
        )
        assert "university" in destination.read_text(encoding="utf-8")
        assert "university" in manifest.read_text(encoding="utf-8")
        assert verify_bundle(manifest)["status"] == "VALID"
    finally:
        project.close()


def test_optional_missing_reranking_alias_exports_still_pass(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        run_id, _ = _create_reranking(project, tmp_path, None)
        report = analysis_payload(project, run_id)
        manifest = export_bundle(project, run_id, tmp_path / "bundle")

        assert report["reranking_runs"][0]["credential_alias"] is None
        assert verify_bundle(manifest)["status"] == "VALID"
    finally:
        project.close()
