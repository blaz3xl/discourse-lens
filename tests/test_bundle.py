import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import discourselens.bundle as bundle_module
from discourselens import Project, verify_research_bundle
from discourselens.analysis import run_exact_phrase_analysis
from discourselens.bundle import (
    _contains_absolute_host_path,
    _contains_secret_metadata_key,
    export_bundle,
    verify_bundle,
)
from discourselens.corpus import compare_documents
from discourselens.errors import ProjectError
from discourselens.models import ModelPrice
from discourselens.reporting import analysis_payload
from discourselens.types import ClaimType, CoverageDescriptor, CoverageMode, EvidenceRole


def _write_with_valid_bundle_digest(path, payload):
    unsigned = {key: value for key, value in payload.items() if key != "integrity"}
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "digest": hashlib.sha256(canonical).hexdigest(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _remove_v14_bundle_fields(payload):
    """Remove v14+ lineage additions to represent a pre-v14 bundle."""
    payload.pop("finding_evidence_spans", None)
    payload.pop("canonical_texts", None)
    for evidence_rows in payload.get("evidence", {}).values():
        for evidence in evidence_rows:
            evidence.pop("source_revision_id", None)
            evidence.pop("rationale", None)


def _canonical_text_preflight_base_size(payload):
    """Measure the empty-text lower bound used before opt-in source reads."""
    preflight_payload = dict(payload)
    revision_ids = sorted(
        {span["revision_id"] for span in preflight_payload["finding_evidence_spans"]}
    )
    preflight_payload["finding_evidence_spans"] = [
        {**span, "canonical_text_length": 0}
        for span in preflight_payload["finding_evidence_spans"]
    ]
    preflight_payload["canonical_texts"] = [
        {"revision_id": revision_id, "sha256": "0" * 64, "text": ""}
        for revision_id in revision_ids
    ]
    preflight_payload["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "digest": "0" * 64,
    }
    return len(
        json.dumps(
            preflight_payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def test_bundle_exports_canonical_lineage_without_secrets(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    manifest = export_bundle(project, run, tmp_path / "bundle")
    data = json.loads(manifest.read_text())
    assert (
        data["analysis_run"]["id"] == run
        and data["findings"]
        and "secret" not in manifest.read_text().casefold()
    )
    assert data["bundle_version"] == 16
    assert data["canonical_texts"] == []
    assert data["human_reviews"] == {finding["id"]: [] for finding in data["findings"]}
    assert data["analysis_run"]["software_version"]
    assert data["analysis_run"]["database_schema_version"] > 0
    assert data["document_revisions"][0]["canonical_hash"]
    assert data["document_revisions"][0]["extraction_manifest_json"]
    assert data["document_revisions"][0]["structure_hash"]
    assert data["skill_runs"] and data["model_executions"] == []
    result = verify_research_bundle(manifest)
    assert result["status"] == "VALID"
    assert result["integrity"] == "VERIFIED_SHA256"
    assert result["analysis_run_id"] == run


def test_bundle_preserves_document_revision_provenance_and_integrity(tmp_path):
    project = Project.init(tmp_path / "revision-provenance")
    _, revision = project.import_text("source.txt", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    stored = project.db.execute(
        "SELECT extraction_manifest_json,structure_hash FROM document_revisions WHERE id=?",
        (revision,),
    ).fetchone()

    manifest = export_bundle(project, run, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    exported = payload["document_revisions"][0]

    assert payload["bundle_version"] == 16
    assert exported["extraction_manifest_json"] == stored["extraction_manifest_json"]
    assert exported["structure_hash"] == stored["structure_hash"]
    assert verify_bundle(manifest)["status"] == "VALID"

    exported["structure_hash"] = "f" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProjectError, match="integrity check failed"):
        verify_bundle(manifest)


@pytest.mark.parametrize("boundary", ["export", "verify"])
def test_bundle_rejects_secret_bearing_revision_extraction_manifest(tmp_path, boundary):
    project = Project.init(tmp_path / f"unsafe-revision-manifest-{boundary}")
    _, revision = project.import_text("source.txt", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    secret_manifest = json.dumps(
        {"api_key": None}, separators=(",", ":"), sort_keys=True
    )

    if boundary == "export":
        project.db.execute(
            "UPDATE document_revisions SET extraction_manifest_json=? WHERE id=?",
            (secret_manifest, revision),
        )
        with pytest.raises(ProjectError, match="secret-bearing document extraction metadata"):
            export_bundle(project, run, tmp_path / "bundle")
    else:
        manifest = export_bundle(project, run, tmp_path / "bundle")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["document_revisions"][0]["extraction_manifest_json"] = secret_manifest
        _write_with_valid_bundle_digest(manifest, payload)
        with pytest.raises(ProjectError, match="secret-bearing document extraction metadata"):
            verify_bundle(manifest)


def test_bundle_manifest_does_not_disclose_temporary_host_paths(tmp_path):
    host_root = tmp_path.resolve()
    source = host_root / "private-source" / "source.txt"
    source.parent.mkdir()
    source.write_text("Officials discuss migration.", encoding="utf-8")
    project = Project.init(host_root / "private-project")
    try:
        imported = project.import_document(source)
        assert imported.revision_id is not None
        run = run_exact_phrase_analysis(project, [imported.revision_id], "migration")[
            "analysis_run_id"
        ]
        manifest = export_bundle(project, run, host_root / "bundle")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    finally:
        project.close()

    def strings(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from strings(key)
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)
        elif isinstance(value, str):
            yield value

    forbidden_root = host_root.as_posix().casefold()
    assert all(
        forbidden_root not in value.replace("\\", "/").casefold() for value in strings(payload)
    )
    result = verify_bundle(manifest)
    assert result["status"] == "VALID"
    assert result["source_location_verification"] == "NOT_VERIFIED_CANONICAL_TEXT_NOT_INCLUDED"


def test_bundle_preserves_model_cost_snapshots_and_accepts_nullable_costs(tmp_path):
    project = Project.init(tmp_path / "model-costs")
    try:
        _, revision = project.import_text("a.txt", "Evidence remains local.")
        run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
        price = ModelPrice(
            provider_id="fixture-provider",
            model="fixture-model",
            input_per_million_tokens="1.25",
            output_per_million_tokens="2.5",
            currency="USD",
            source_reference="https://prices.example/models/fixture-model",
            rate_version="v1",
            rate_date="2026-01-01",
        )
        priced_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash="response-hash",
            analysis_run_id=run,
            input_tokens=1_000,
            output_tokens=500,
            model_price=price,
            endpoint="https://provider.example/v1",
        )
        unpriced_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="unpriced-request-hash",
            response_hash="unpriced-response-hash",
            analysis_run_id=run,
            input_tokens=1_000,
            output_tokens=500,
        )

        manifest = export_bundle(project, run, tmp_path / "bundle")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        executions = {item["id"]: item for item in payload["model_executions"]}
        assert executions[priced_id]["model_price_json"] == json.dumps(
            price.to_snapshot(), sort_keys=True, separators=(",", ":")
        )
        assert executions[priced_id]["endpoint"] == "https://provider.example/v1"
        assert executions[priced_id]["estimated_cost"] == "0.0025"
        assert executions[priced_id]["estimated_cost_currency"] == "USD"
        assert executions[unpriced_id]["model_price_json"] is None
        assert executions[unpriced_id]["estimated_cost"] is None
        assert executions[unpriced_id]["estimated_cost_currency"] is None
        assert verify_bundle(manifest)["status"] == "VALID"

        # Fields added after older bundle versions were published stay optional.
        for execution in payload["model_executions"]:
            execution.pop("model_price_json")
            execution.pop("estimated_cost")
            execution.pop("estimated_cost_currency")
        _write_with_valid_bundle_digest(manifest, payload)
        assert verify_bundle(manifest)["status"] == "VALID"
    finally:
        project.close()


@pytest.mark.parametrize("boundary", ["export", "verify"])
@pytest.mark.parametrize(
    "unsafe_endpoint",
    [
        "https://user:password@provider.example/v1",
        "https://provider.example/v1?token=secret",
        "https://provider.example/v1#fragment",
    ],
)
def test_bundle_rejects_credential_material_in_model_execution_endpoint(
    tmp_path, boundary, unsafe_endpoint
):
    project = Project.init(tmp_path / f"unsafe-endpoint-{boundary}")
    try:
        _, revision = project.import_text("source.txt", "Selected evidence.")
        run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
        execution_id = project.record_model_execution(
            provider_id="fixture",
            credential_alias="fixture-alias",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash=None,
            analysis_run_id=run,
            endpoint="https://provider.example/v1",
        )
        if boundary == "export":
            project.db.execute(
                "UPDATE model_executions SET endpoint=? WHERE id=?",
                (unsafe_endpoint, execution_id),
            )
            project.db.commit()
            with pytest.raises(
                ProjectError,
                match="analysis run contains credential-like provider endpoint metadata",
            ) as caught:
                export_bundle(project, run, tmp_path / "bundle-export")
        else:
            manifest = export_bundle(project, run, tmp_path / "bundle-verify")
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["model_executions"][0]["endpoint"] = unsafe_endpoint
            _write_with_valid_bundle_digest(manifest, payload)
            with pytest.raises(
                ProjectError,
                match="research bundle contains credential-like provider endpoint metadata",
            ) as caught:
                verify_bundle(manifest)
        assert unsafe_endpoint not in str(caught.value)
    finally:
        project.close()


def test_bundle_export_replaces_manifest_symlink_without_overwriting_target(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    destination = tmp_path / "bundle"
    destination.mkdir()
    outside_target = tmp_path / "outside.json"
    outside_target.write_text("preserve this file", encoding="utf-8")
    manifest = destination / "manifest.json"
    try:
        manifest.symlink_to(outside_target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    exported = export_bundle(project, run, destination)

    assert not exported.is_symlink()
    assert outside_target.read_text(encoding="utf-8") == "preserve this file"
    assert verify_bundle(exported)["integrity"] == "VERIFIED_SHA256"


def test_bundle_export_rejects_destination_directory_symlink(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    outside_manifest = outside_directory / "manifest.json"
    outside_manifest.write_text("preserve this file", encoding="utf-8")
    destination = tmp_path / "bundle"
    try:
        destination.symlink_to(outside_directory, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(ProjectError, match="destination.*symlink"):
        export_bundle(project, run, destination)

    assert outside_manifest.read_text(encoding="utf-8") == "preserve this file"


def test_bundle_export_does_not_follow_destination_symlink_swap(tmp_path, monkeypatch):
    if not bundle_module._directory_fd_write_supported():
        pytest.skip("pinned directory-relative writes are unavailable")

    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    destination = tmp_path / "bundle"
    destination.mkdir()
    displaced = tmp_path / "bundle-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_manifest = outside / "manifest.json"
    outside_manifest.write_text("preserve external manifest", encoding="utf-8")

    real_open = os.open
    swapped = False

    def swap_destination_before_open(path, flags, *args, **kwargs):
        nonlocal swapped
        candidate = Path(path)
        if not swapped and (candidate == destination or candidate.parent == destination):
            destination.rename(displaced)
            destination.symlink_to(outside, target_is_directory=True)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    # Monkeypatching os.open changes its identity in supports_dir_fd; preserve the
    # platform capability decision while injecting the race at the actual open.
    monkeypatch.setattr(bundle_module, "_directory_fd_write_supported", lambda: True)
    monkeypatch.setattr(bundle_module.os, "open", swap_destination_before_open)
    try:
        with pytest.raises(ProjectError, match="destination.*(unsafe|changed)"):
            export_bundle(project, run, destination)
        assert swapped
        assert outside_manifest.read_text(encoding="utf-8") == "preserve external manifest"
        assert not list(outside.glob(".manifest-*"))
    finally:
        if destination.is_symlink():
            destination.unlink()
        if displaced.exists():
            displaced.rename(destination)


@pytest.mark.parametrize(
    ("location", "message"),
    [("run_config", "secret-bearing"), ("model_parameters", "secret-bearing")],
)
def test_bundle_verifier_rejects_secret_metadata_in_resigned_provenance(
    tmp_path, location, message
):
    project = Project.init(tmp_path / f"secret-{location}")
    _, revision = project.import_text("source.txt", "Selected evidence.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    if location == "model_parameters":
        project.record_model_execution(
            provider_id="fixture",
            credential_alias="fixture-alias",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash=None,
            analysis_run_id=run,
        )
    manifest = export_bundle(project, run, tmp_path / f"bundle-{location}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if location == "run_config":
        payload["analysis_run"]["config"]["provider"] = {"apiKey": "x"}
    else:
        payload["model_executions"][0]["parameters_json"] = json.dumps(
            {"provider": {"bearerToken": "y"}}
        )
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match=message):
        verify_bundle(manifest)


@pytest.mark.parametrize("boundary", ["export", "verify"])
def test_bundle_rejects_duplicate_model_metadata_keys_that_hide_secret_fields(tmp_path, boundary):
    project = Project.init(tmp_path / boundary)
    _, revision = project.import_text("source.txt", "Selected evidence.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    execution = project.record_model_execution(
        provider_id="fixture",
        credential_alias="fixture-alias",
        model="fixture-model",
        reasoning=None,
        request_hash="request-hash",
        response_hash=None,
        analysis_run_id=run,
    )
    destination = tmp_path / f"bundle-{boundary}"
    if boundary == "export":
        project.db.execute(
            "UPDATE model_executions SET parameters_json=? WHERE id=?",
            ('{"provider":{"apiKey":"x"},"provider":{}}', execution),
        )
        project.db.commit()

        with pytest.raises(ProjectError, match="model execution metadata is invalid"):
            export_bundle(project, run, destination)

        assert not (destination / "manifest.json").exists()
        return

    manifest = export_bundle(project, run, destination)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["model_executions"][0]["parameters_json"] = '{"provider":{"apiKey":"x"},"provider":{}}'
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match="model execution metadata is invalid"):
        verify_bundle(manifest)


def test_bundle_exports_findings_in_stable_created_at_id_order(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "stable-finding-order")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(scope_id, {"profile": "stable-finding-order"})
    evidence_id = project.user_context_evidence("fixture", "Researcher context.")

    generated_ids = iter(("z-first", "a-second"))
    monkeypatch.setattr("discourselens.project.uid", lambda: next(generated_ids))
    monkeypatch.setattr("discourselens.project.now", lambda: "2026-09-30T12:00:00+00:00")
    inserted_ids = []
    for observation in ("Inserted first.", "Inserted second."):
        finding_id = project.create_finding(run_id, "fixture", "1.0.0", "stable order", observation)
        project.attach_evidence(finding_id, evidence_id, EvidenceRole.CONTEXT)
        inserted_ids.append(finding_id)

    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert inserted_ids == ["z-first", "a-second"]
    assert [finding["id"] for finding in payload["findings"]] == ["a-second", "z-first"]


def test_bundle_exports_and_verifies_retrieval_run_coverage_reference(tmp_path):
    project = Project.init(tmp_path / "retrieval-coverage-export")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(scope_id, {"profile": "retrieval-coverage-export"})
    skill_coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope_id,
            population_definition="The skill's declared fixture population.",
        )
    )
    retrieval_coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope_id,
            population_definition="The retrieval operation's fixture population.",
        )
    )
    skill_run_id = project.create_skill_run(
        run_id,
        scope_id,
        "fixture.retrieval-coverage",
        "1.0.0",
        "test",
        "fixture",
        coverage_id=skill_coverage_id,
    )
    retrieval_run_id = project.create_retrieval_run(
        analysis_run_id=run_id,
        skill_run_id=skill_run_id,
        scope_id=scope_id,
        retrieval_plan_ref="retrieval/coverage-fixture",
        query_terms=["Apples"],
        selection_policy="fixture",
        candidate_count=0,
        inspected_count=0,
        returned_count=0,
        candidate_limit_reached=False,
        coverage_id=retrieval_coverage_id,
    )

    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["retrieval_runs"][0]["id"] == retrieval_run_id
    assert {item["id"] for item in payload["coverage_descriptors"]} == {
        skill_coverage_id,
        retrieval_coverage_id,
    }
    assert verify_bundle(manifest)["integrity"] == "VERIFIED_SHA256"

    payload["coverage_descriptors"] = [
        item for item in payload["coverage_descriptors"] if item["id"] != retrieval_coverage_id
    ]
    _write_with_valid_bundle_digest(manifest, payload)
    with pytest.raises(ProjectError, match="retrieval_runs.*missing coverage descriptor"):
        verify_bundle(manifest)


@pytest.mark.parametrize("boundary", ["export", "verify"])
@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"api_key": "coverage-secret-sentinel"}, "secret-bearing coverage descriptor"),
        ({"cache_path": "/home/researcher/.cache/model"}, "absolute host paths"),
    ],
)
def test_bundle_rejects_sensitive_coverage_descriptor_metadata(
    tmp_path, boundary, metadata, message
):
    project = Project.init(tmp_path / f"coverage-metadata-{boundary}")
    _, revision = project.import_text("source.txt", "Selected evidence.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(scope_id, {"profile": "coverage-metadata-fixture"})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope_id,
            population_definition="A bounded fixture population.",
        )
    )
    project.create_skill_run(
        run_id,
        scope_id,
        "fixture.coverage-metadata",
        "1.0.0",
        "test",
        "fixture",
        coverage_id=coverage_id,
    )
    if boundary == "export":
        raw_descriptor = project.db.execute(
            "SELECT payload_json FROM coverage_descriptors WHERE id=?", (coverage_id,)
        ).fetchone()[0]
        descriptor = json.loads(raw_descriptor)
        descriptor.update(metadata)
        project.db.execute("DROP TRIGGER coverage_descriptors_immutable_update")
        project.db.execute(
            "UPDATE coverage_descriptors SET payload_json=? WHERE id=?",
            (json.dumps(descriptor), coverage_id),
        )
        project.db.commit()

        with pytest.raises(ProjectError, match=message):
            export_bundle(project, run_id, tmp_path / "bundle")
        return

    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    descriptor = json.loads(payload["coverage_descriptors"][0]["payload_json"])
    descriptor.update(metadata)
    payload["coverage_descriptors"][0]["payload_json"] = json.dumps(descriptor)
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match=message):
        verify_bundle(manifest)


def test_bundle_export_batches_large_scope_selections_under_sqlite_limit(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "large-scope-export")
    _, revision = project.import_text("source.txt", "Selected evidence.")
    revision_row = project.db.execute(
        "SELECT * FROM document_revisions WHERE id=?", (revision,)
    ).fetchone()

    variable_limit = min(project.db.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER), 999)
    revision_count = variable_limit + 1
    revision_columns = tuple(revision_row.keys())
    revision_placeholders = ",".join("?" for _ in revision_columns)
    revision_insert = (
        f"INSERT INTO document_revisions ({','.join(revision_columns)}) "
        f"VALUES ({revision_placeholders})"
    )
    revision_ids = [revision]
    revision_rows = []
    for index in range(1, revision_count):
        cloned_revision = dict(revision_row)
        cloned_revision["id"] = f"bulk-revision-{index:04d}"
        cloned_revision["source_hash"] = hashlib.sha256(f"source-{index}".encode()).hexdigest()
        cloned_revision["canonical_hash"] = hashlib.sha256(
            f"canonical-{index}".encode()
        ).hexdigest()
        revision_rows.append(tuple(cloned_revision[column] for column in revision_columns))
        revision_ids.append(cloned_revision["id"])
    project.db.executemany(revision_insert, revision_rows)

    scope_id = "bulk-selection-scope"
    project.db.execute(
        "INSERT INTO scope_snapshots VALUES(?,?)", (scope_id, json.dumps(revision_ids))
    )
    run_id = project.create_run(scope_id, {"profile": "large-scope-export"})
    span_id = project.create_span(revision, 0, 8)

    seed_coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope_id,
            population_definition="Synthetic bounded-export fixture population.",
        )
    )
    descriptor_payload = project.db.execute(
        "SELECT payload_json FROM coverage_descriptors WHERE id=?", (seed_coverage_id,)
    ).fetchone()[0]
    coverage_ids = [seed_coverage_id]
    coverage_rows = []
    skill_run_rows = []
    timestamp = "2026-09-30T12:00:00+00:00"
    for index in range(1, revision_count):
        coverage_id = f"bulk-coverage-{index:04d}"
        coverage_ids.append(coverage_id)
        coverage_rows.append((coverage_id, descriptor_payload))
        skill_run_rows.append(
            (
                f"bulk-skill-run-{index:04d}",
                run_id,
                "fixture.large-scope",
                "1.0.0",
                scope_id,
                "COMPLETED",
                "fixture",
                "fixture-config-hash",
                coverage_id,
                None,
                timestamp,
                timestamp,
                None,
                None,
            )
        )
    skill_run_rows.append(
        (
            "bulk-skill-run-seed",
            run_id,
            "fixture.large-scope",
            "1.0.0",
            scope_id,
            "COMPLETED",
            "fixture",
            "fixture-config-hash",
            seed_coverage_id,
            None,
            timestamp,
            timestamp,
            None,
            None,
        )
    )
    project.db.executemany("INSERT INTO coverage_descriptors VALUES(?,?)", coverage_rows)
    project.db.executemany(
        "INSERT INTO skill_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", skill_run_rows
    )

    codebook_rows = []
    code_rows = []
    annotation_rows = []
    for index in range(revision_count):
        codebook_id = f"bulk-codebook-{index:04d}"
        code_id = f"bulk-code-{index:04d}"
        codebook_rows.append((codebook_id, f"Fixture {index:04d}", None, timestamp))
        code_rows.append((code_id, codebook_id, None, "Selected", None, timestamp))
        annotation_rows.append((f"bulk-annotation-{index:04d}", code_id, span_id, None, timestamp))
    project.db.executemany("INSERT INTO codebooks VALUES(?,?,?,?)", codebook_rows)
    project.db.executemany("INSERT INTO codes VALUES(?,?,?,?,?,?)", code_rows)
    project.db.executemany("INSERT INTO annotations VALUES(?,?,?,?,?)", annotation_rows)
    project.db.commit()

    previous_limit = project.db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, variable_limit)
    try:
        # This is the same oversized explicit-ID query the exporter previously built.
        placeholders = ",".join("?" for _ in revision_ids)
        with pytest.raises(sqlite3.OperationalError, match="too many SQL variables"):
            project.db.execute(
                f"SELECT id FROM document_revisions WHERE id IN ({placeholders})",
                revision_ids,
            )

        manifest = export_bundle(project, run_id, tmp_path / "bundle")
        first_bytes = manifest.read_bytes()
        payload = json.loads(first_bytes)
        assert len(payload["document_revisions"]) == revision_count
        assert {item["id"] for item in payload["coverage_descriptors"]} == set(coverage_ids)
        assert len(payload["qualitative_coding"]["annotations"]) == revision_count
        assert len(payload["qualitative_coding"]["codes"]) == revision_count
        assert len(payload["qualitative_coding"]["codebooks"]) == revision_count
        assert len(first_bytes) <= 64 * 1024 * 1024
        assert verify_bundle(manifest)["integrity"] == "VERIFIED_SHA256"

        export_bundle(project, run_id, manifest.parent)
        assert manifest.read_bytes() == first_bytes

        monkeypatch.setattr("discourselens.bundle.MAX_BUNDLE_MANIFEST_BYTES", len(first_bytes) - 1)
        with pytest.raises(ProjectError, match="configured size limit"):
            export_bundle(project, run_id, manifest.parent)
        assert manifest.read_bytes() == first_bytes
    finally:
        project.db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)


def test_bundle_verification_detects_content_corruption(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    manifest = export_bundle(project, run, tmp_path / "bundle")
    data = json.loads(manifest.read_text())
    data["analysis_run"]["status"] = "TAMPERED"
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ProjectError, match="integrity check failed"):
        verify_bundle(manifest)


def test_bundle_export_enforces_manifest_size_limit(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "oversized-export")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    destination = tmp_path / "bundle"
    monkeypatch.setattr("discourselens.bundle.MAX_BUNDLE_MANIFEST_BYTES", 128)

    with pytest.raises(ProjectError, match="configured size limit"):
        export_bundle(project, run, destination)

    assert not (destination / "manifest.json").exists()


def test_bundle_verifier_bounds_manifest_read(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "oversized-verification")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    manifest = export_bundle(project, run, tmp_path / "bundle")
    monkeypatch.setattr(
        "discourselens.bundle.MAX_BUNDLE_MANIFEST_BYTES", manifest.stat().st_size - 1
    )

    with pytest.raises(ProjectError, match="configured size limit"):
        verify_bundle(manifest)


@pytest.mark.parametrize("location", ["value", "key"])
def test_bundle_verifier_rejects_host_paths_in_resigned_run_metadata(tmp_path, location):
    project = Project.init(tmp_path / "verify-host-path")
    _, revision = project.import_text("source.txt", "Selected evidence.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    manifest = export_bundle(project, run, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if location == "value":
        payload["analysis_run"]["config"]["cache_path"] = "/home/researcher/.cache/model"
    else:
        payload["analysis_run"]["config"]["/home/researcher/.cache/model"] = "private-model"
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match="absolute host paths"):
        verify_bundle(manifest)


def test_bundle_path_guard_ignores_absolute_http_urls():
    assert not _contains_absolute_host_path({"endpoint": "https://example.test/model"})


@pytest.mark.parametrize(
    "mutate_raw",
    [
        lambda raw: raw.replace(
            '"bundle_version": 16,',
            '"bundle_version": 16,\n  "bundle_version": 16,',
            1,
        ),
        lambda raw: raw.replace("{", '{"non_finite": NaN,', 1),
    ],
    ids=["duplicate-key", "non-standard-number"],
)
def test_bundle_verifier_rejects_ambiguous_or_non_json_input(tmp_path, mutate_raw):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    manifest = export_bundle(project, run, tmp_path / "bundle")
    raw = manifest.read_text(encoding="utf-8")
    manifest.write_text(mutate_raw(raw), encoding="utf-8")

    with pytest.raises(ProjectError, match="unreadable or invalid JSON"):
        verify_bundle(manifest)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("finding_lineage"), "finding_lineage"),
        (lambda payload: payload.pop("qualitative_coding"), "qualitative_coding"),
        (lambda payload: payload.pop("human_reviews"), "human_reviews"),
        (
            lambda payload: payload["qualitative_coding"].update({"memos": {}}),
            "memos.*must be an array",
        ),
    ],
)
def test_bundle_verifier_rejects_incomplete_v12_contract_even_with_valid_digest(
    tmp_path, mutate, message
):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    manifest = export_bundle(project, run, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutate(payload)
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match=message):
        verify_bundle(manifest)


@pytest.mark.parametrize("version", [11, 12])
def test_bundle_verifier_rejects_unsupported_top_level_fields_with_valid_digest(tmp_path, version):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    manifest = export_bundle(project, run, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_version"] = version
    if version == 11:
        payload.pop("human_reviews")
    payload["unrecognized_extension"] = {"scope": "outside-the-contract"}
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match="unsupported top-level fields"):
        verify_bundle(manifest)


def test_bundle_verifier_rejects_invalid_project_format_version(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    manifest = export_bundle(project, run, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["project_format_version"] = 0
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match="project format version"):
        verify_bundle(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_hash", "not-a-sha256"),
        ("canonical_hash", "a" * 63),
        ("source_hash", None),
    ],
)
def test_bundle_verifier_rejects_missing_or_malformed_revision_hashes(tmp_path, field, value):
    project = Project.init(tmp_path / "malformed-document-hash")
    _, revision = project.import_text("a.txt", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    manifest = export_bundle(project, run, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["document_revisions"][0][field] = value
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match="document revision hash"):
        verify_bundle(manifest)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["analysis_run"].update({"id": ""}), "analysis-run provenance"),
        (lambda payload: payload["findings"][0].update({"id": ""}), "finding without a valid ID"),
    ],
)
def test_bundle_verifier_rejects_empty_provenance_ids_even_with_valid_digest(
    tmp_path, mutate, message
):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    manifest = export_bundle(project, run, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutate(payload)
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match=message):
        verify_bundle(manifest)


def test_bundle_verifier_rejects_dangling_finding_lineage_even_with_valid_digest(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    manifest = export_bundle(project, run, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["finding_lineage"] = [
        {
            "finding_id": payload["findings"][0]["id"],
            "source_finding_id": "missing-source-finding",
            "relation": "SYNTHESIZES",
        }
    ]
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match="finding_lineage.*invalid reference"):
        verify_bundle(manifest)


def test_bundle_verifier_rejects_model_execution_with_dangling_skill_run(tmp_path):
    project = Project.init(tmp_path / "model-skill-run-link")
    try:
        _, revision = project.import_text("source.txt", "Selected evidence.")
        scope_id = project.snapshot([revision])
        run_id = project.create_run(scope_id, {"profile": "execution-link"})
        skill_run_id = project.create_skill_run(
            run_id,
            scope_id,
            "fixture.review",
            "1.0.0",
            "fixture.executor",
            "fixture-config-hash",
        )
        project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash="response-hash",
            analysis_run_id=run_id,
            skill_run_id=skill_run_id,
        )
        manifest = export_bundle(project, run_id, tmp_path / "bundle")
        assert verify_bundle(manifest)["status"] == "VALID"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["model_executions"][0]["skill_run_id"] = "missing-skill-run"
        _write_with_valid_bundle_digest(manifest, payload)

        with pytest.raises(ProjectError, match="model execution.*invalid skill run"):
            verify_bundle(manifest)
    finally:
        project.close()


def test_bundle_verification_keeps_structural_support_for_v10_without_claiming_integrity(
    tmp_path,
):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    manifest = export_bundle(project, run, tmp_path / "bundle")
    data = json.loads(manifest.read_text())
    data["bundle_version"] = 10
    data.pop("integrity")
    data.pop("human_reviews")
    _remove_v14_bundle_fields(data)
    manifest.write_text(json.dumps(data), encoding="utf-8")

    result = verify_bundle(manifest)

    assert result["status"] == "VALID"
    assert result["integrity"] == "NOT_AVAILABLE_LEGACY"


@pytest.mark.parametrize(
    ("location", "metadata", "message"),
    [
        ("run_config", {"provider": {"apiKey": "fixture-secret"}}, "secret-bearing"),
        (
            "run_config",
            {"cache_path": "/home/researcher/.cache/model"},
            "absolute host paths",
        ),
        (
            "model_parameters",
            {"provider": {"bearerToken": "fixture-secret"}},
            "secret-bearing",
        ),
        (
            "model_parameters",
            {"cache_path": "/home/researcher/.cache/model"},
            "absolute host paths",
        ),
    ],
)
def test_v10_bundle_verifier_rejects_secret_or_host_path_metadata(
    tmp_path, location, metadata, message
):
    project = Project.init(tmp_path / f"legacy-v10-{location}")
    _, revision = project.import_text("source.txt", "Selected evidence.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    if location == "model_parameters":
        project.record_model_execution(
            provider_id="fixture",
            credential_alias="fixture-alias",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash=None,
            analysis_run_id=run_id,
        )

    manifest = export_bundle(project, run_id, tmp_path / f"bundle-{location}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_version"] = 10
    payload.pop("integrity")
    payload.pop("human_reviews")
    _remove_v14_bundle_fields(payload)
    if location == "run_config":
        payload["analysis_run"]["config"].update(metadata)
    else:
        payload["model_executions"][0]["parameters_json"] = json.dumps(metadata)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProjectError, match=message):
        verify_bundle(manifest)


def test_v11_bundle_remains_valid_and_integrity_checked(tmp_path):
    project = Project.init(tmp_path / "legacy-v11")
    _, revision = project.import_text("a", "Apples are green.")
    run = run_exact_phrase_analysis(project, [revision], "Apples")["analysis_run_id"]
    manifest = export_bundle(project, run, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_version"] = 11
    payload.pop("human_reviews")
    _remove_v14_bundle_fields(payload)
    payload["qualitative_coding"].pop("source_spans")
    _write_with_valid_bundle_digest(manifest, payload)

    result = verify_bundle(manifest)

    assert result["status"] == "VALID"
    assert result["bundle_version"] == 11
    assert result["integrity"] == "VERIFIED_SHA256"


def test_v12_bundle_remains_valid_without_v13_source_span_inventory(tmp_path):
    project = Project.init(tmp_path / "legacy-v12")
    _, revision = project.import_text("a", "Apples are green.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(scope_id, {"profile": "legacy-coding"})
    codebook_id = project.create_codebook("descriptive")
    code_id = project.create_code(codebook_id, "fruit")
    project.annotate_span(code_id, project.create_span(revision, 0, 6))
    project.record_model_execution(
        provider_id="fixture",
        credential_alias="fixture-alias",
        model="fixture-model",
        reasoning=None,
        request_hash="legacy-request",
        response_hash="legacy-response",
        analysis_run_id=run_id,
        endpoint="https://provider.example/v1",
    )
    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_version"] = 12
    _remove_v14_bundle_fields(payload)
    payload["qualitative_coding"].pop("source_spans")
    for execution in payload["model_executions"]:
        execution.pop("endpoint")
    _write_with_valid_bundle_digest(manifest, payload)

    result = verify_bundle(manifest)

    assert result["status"] == "VALID"
    assert result["bundle_version"] == 12
    assert result["integrity"] == "VERIFIED_SHA256"


def test_v13_bundle_remains_valid_without_v14_finding_evidence_span_inventory(tmp_path):
    project = Project.init(tmp_path / "legacy-v13")
    _, revision = project.import_text("source.txt", "Apples are green.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "legacy-v13"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical observation", "Apples occurs."
    )
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 6))
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_version"] = 13
    _remove_v14_bundle_fields(payload)
    _write_with_valid_bundle_digest(manifest, payload)

    result = verify_bundle(manifest)

    assert result["status"] == "VALID"
    assert result["bundle_version"] == 13
    assert result["source_location_verification"] == "NOT_VERIFIED_CANONICAL_TEXT_NOT_INCLUDED"


def test_v14_exports_deduplicated_finding_source_spans_only_for_textual_evidence(tmp_path):
    project = Project.init(tmp_path / "finding-span-inventory")
    _, revision = project.import_text("selected.txt", "Apples are green. Pears are green.")
    _, outside_revision = project.import_text("outside.txt", "Elephants live in Africa.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "span-inventory"})
    span_id = project.create_span(revision, 0, 6)
    evidence_id = project.textual_evidence(span_id)
    for index in range(2):
        finding_id = project.create_finding(
            run_id, "fixture", "1.0.0", f"finding {index}", "The selected word occurs."
        )
        project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    unused_span_id = project.create_span(revision, 18, 23)
    project.create_span(outside_revision, 0, 9)
    project.finalize_run(run_id)

    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["bundle_version"] == 16
    assert payload["finding_evidence_spans"] == [
        {
            "id": span_id,
            "revision_id": revision,
            "start_offset": 0,
            "end_offset": 6,
            "text_hash": hashlib.sha256(b"Apples").hexdigest(),
            "canonical_text_length": len("Apples are green. Pears are green."),
        }
    ]
    assert unused_span_id not in {item["id"] for item in payload["finding_evidence_spans"]}
    for rows in payload["evidence"].values():
        assert rows[0]["source_span_id"] == span_id
        assert rows[0]["source_revision_id"] == revision
        assert rows[0]["rationale"] is None
    result = verify_bundle(manifest)
    assert result["status"] == "VALID"
    assert result["source_location_verification"] == "NOT_VERIFIED_CANONICAL_TEXT_NOT_INCLUDED"

    schema_dir = Path(__file__).parents[1] / "schemas"
    schemas = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in schema_dir.glob("*.json")
    }
    registry = Registry()
    for schema in schemas.values():
        if "$id" in schema:
            registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    Draft202012Validator(schemas["research-bundle-v16.json"], registry=registry).validate(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["evidence"][next(iter(payload["evidence"]))][0].update(
                {"source_span_id": "missing-span"}
            ),
            "textual evidence references a missing source span",
        ),
        (
            lambda payload: (
                payload["finding_evidence_spans"][0].update({"revision_id": "outside-revision"})
                or payload["evidence"][next(iter(payload["evidence"]))][0].update(
                    {"source_revision_id": "outside-revision"}
                )
            ),
            "out-of-scope revision",
        ),
        (
            lambda payload: payload["finding_evidence_spans"][0].update(
                {"end_offset": payload["finding_evidence_spans"][0]["canonical_text_length"] + 1}
            ),
            "invalid finding evidence source span",
        ),
        (
            lambda payload: payload["evidence"][next(iter(payload["evidence"]))][0].update(
                {"text": "Apple"}
            ),
            "span length does not match its excerpt",
        ),
        (
            lambda payload: payload["evidence"][next(iter(payload["evidence"]))][0].update(
                {"text": "Pears?"}
            ),
            "text does not match its source span",
        ),
    ],
)
def test_v14_verifier_rejects_inconsistent_finding_source_span_lineage(tmp_path, mutate, message):
    project = Project.init(tmp_path / "invalid-finding-span-lineage")
    _, revision = project.import_text("selected.txt", "Apples are green.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "span-validation"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical", "The selected word occurs."
    )
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 6))
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutate(payload)
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match=message):
        verify_bundle(manifest)


def test_v14_finding_source_span_inventory_rejects_duplicate_and_unreferenced_ids(tmp_path):
    project = Project.init(tmp_path / "duplicate-finding-span")
    _, revision = project.import_text("source.txt", "Apples are green.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "span-duplicate"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical", "The selected word occurs."
    )
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 6))
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    original = json.loads(manifest.read_text(encoding="utf-8"))

    duplicated = json.loads(json.dumps(original))
    duplicated["finding_evidence_spans"].append(duplicated["finding_evidence_spans"][0])
    _write_with_valid_bundle_digest(manifest, duplicated)
    with pytest.raises(ProjectError, match="invalid finding evidence source span"):
        verify_bundle(manifest)

    unreferenced = json.loads(json.dumps(original))
    extra = dict(unreferenced["finding_evidence_spans"][0])
    extra["id"] = "unreferenced-span"
    unreferenced["finding_evidence_spans"].append(extra)
    _write_with_valid_bundle_digest(manifest, unreferenced)
    with pytest.raises(ProjectError, match="do not match textual evidence"):
        verify_bundle(manifest)


def test_bundle_verification_reports_location_limit_and_non_applicability(tmp_path):
    project = Project.init(tmp_path / "source-location-verification")
    _, revision = project.import_text("source.txt", "Apples are green.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "location-status"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical", "The researcher supplied context."
    )
    context_id = project.user_context_evidence("memo", "Context not tied to a source span.")
    project.attach_evidence(finding_id, context_id, EvidenceRole.CONTEXT)
    project.finalize_run(run_id)

    context_bundle = export_bundle(project, run_id, tmp_path / "context-bundle")
    assert verify_bundle(context_bundle)["source_location_verification"] == "NOT_APPLICABLE"

    text_run_id = project.create_run(project.snapshot([revision]), {"profile": "location-status"})
    textual_finding = project.create_finding(
        text_run_id, "fixture", "1.0.0", "lexical", "A textual claim."
    )
    textual_evidence = project.textual_evidence(project.create_span(revision, 0, 6))
    project.attach_evidence(textual_finding, textual_evidence, EvidenceRole.SUPPORT)
    project.finalize_run(text_run_id)
    textual_bundle = export_bundle(project, text_run_id, tmp_path / "textual-bundle")
    payload = json.loads(textual_bundle.read_text(encoding="utf-8"))
    assert payload["canonical_texts"] == []
    payload["finding_evidence_spans"][0]["start_offset"] = 1
    payload["finding_evidence_spans"][0]["end_offset"] = 7
    _write_with_valid_bundle_digest(textual_bundle, payload)

    limited = verify_bundle(textual_bundle)
    assert limited["status"] == "VALID"
    assert limited["source_location_verification"] == ("NOT_VERIFIED_CANONICAL_TEXT_NOT_INCLUDED")


def test_v16_opt_in_canonical_text_verifies_exact_cited_source_location(tmp_path):
    project = Project.init(tmp_path / "canonical-text-opt-in")
    _, revision = project.import_text("source.txt", "Apples are green.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "canonical-opt-in"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical", "The source contains the selected word."
    )
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 6))
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)

    manifest = export_bundle(
        project, run_id, tmp_path / "bundle", include_canonical_text=True
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["bundle_version"] == 16
    assert payload["canonical_texts"] == [
        {
            "revision_id": revision,
            "sha256": hashlib.sha256(b"Apples are green.").hexdigest(),
            "text": "Apples are green.",
        }
    ]
    assert {item["revision_id"] for item in payload["canonical_texts"]} == {revision}
    assert verify_bundle(manifest)["source_location_verification"] == "VERIFIED_CANONICAL_TEXT"


def test_v16_canonical_text_rejects_same_length_shifted_source_location(tmp_path):
    project = Project.init(tmp_path / "canonical-text-offset-check")
    _, revision = project.import_text("source.txt", "Apples are green.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "canonical-offset"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical", "The source contains the selected word."
    )
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 6))
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    manifest = export_bundle(
        project, run_id, tmp_path / "bundle", include_canonical_text=True
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["finding_evidence_spans"][0].update({"start_offset": 1, "end_offset": 7})
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match="does not match cited evidence location"):
        verify_bundle(manifest)


@pytest.mark.parametrize("malformation", ["bad-hash", "unknown-revision", "unreferenced-revision"])
def test_v16_verifier_rejects_malformed_or_unreferenced_canonical_text(
    tmp_path, malformation
):
    project = Project.init(tmp_path / f"canonical-text-{malformation}")
    _, revision = project.import_text("selected.txt", "Apples are green.")
    _, other_revision = project.import_text("unreferenced.txt", "Pears are green.")
    run_id = project.create_run(
        project.snapshot([revision, other_revision]), {"profile": "canonical-text-validation"}
    )
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical", "The selected source contains Apples."
    )
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 6))
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    manifest = export_bundle(
        project, run_id, tmp_path / "bundle", include_canonical_text=True
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if malformation == "bad-hash":
        payload["canonical_texts"][0]["sha256"] = "0" * 64
        expected_error = "does not match its revision hash"
    elif malformation == "unknown-revision":
        payload["canonical_texts"][0]["revision_id"] = "outside-scope"
        expected_error = "out-of-scope revision"
    else:
        payload["canonical_texts"].append(
            {
                "revision_id": other_revision,
                "sha256": hashlib.sha256(b"Pears are green.").hexdigest(),
                "text": "Pears are green.",
            }
        )
        expected_error = "do not match cited revisions"
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match=expected_error):
        verify_bundle(manifest)


def test_v14_bundle_remains_verifiable_without_v15_canonical_text_field(tmp_path):
    project = Project.init(tmp_path / "legacy-v14")
    _, revision = project.import_text("source.txt", "Apples are green.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "legacy-v14"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical", "The source contains the selected word."
    )
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 6))
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_version"] = 14
    payload.pop("canonical_texts")
    for document_revision in payload["document_revisions"]:
        document_revision.pop("extraction_manifest_json")
        document_revision.pop("structure_hash")
    _write_with_valid_bundle_digest(manifest, payload)

    result = verify_bundle(manifest)

    assert result["bundle_version"] == 14
    assert result["source_location_verification"] == "NOT_VERIFIED_CANONICAL_TEXT_NOT_INCLUDED"


def test_v15_bundle_remains_verifiable_with_canonical_text(tmp_path):
    project = Project.init(tmp_path / "legacy-v15")
    _, revision = project.import_text("source.txt", "Apples are green.")
    run_id = project.create_run(project.snapshot([revision]), {"profile": "legacy-v15"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical", "The source contains the selected word."
    )
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 6))
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    manifest = export_bundle(
        project, run_id, tmp_path / "bundle", include_canonical_text=True
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["bundle_version"] = 15
    for document_revision in payload["document_revisions"]:
        document_revision.pop("extraction_manifest_json")
        document_revision.pop("structure_hash")
    _write_with_valid_bundle_digest(manifest, payload)

    result = verify_bundle(manifest)

    assert result["bundle_version"] == 15
    assert result["source_location_verification"] == "VERIFIED_CANONICAL_TEXT"


def test_canonical_text_opt_in_remains_under_manifest_size_limit(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "canonical-text-size-bound")
    source_text = "x" * 1_000_000
    _, revision = project.import_text("source.txt", source_text)
    run_id = project.create_run(project.snapshot([revision]), {"profile": "canonical-size"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical", "The source contains the selected character."
    )
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 1))
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    default_manifest = export_bundle(project, run_id, tmp_path / "default-bundle")
    default_size = default_manifest.stat().st_size
    monkeypatch.setattr("discourselens.bundle.MAX_BUNDLE_MANIFEST_BYTES", default_size)

    with pytest.raises(ProjectError, match="configured size limit"):
        export_bundle(
            project,
            run_id,
            tmp_path / "opt-in-bundle",
            include_canonical_text=True,
        )


def test_oversized_single_cited_revision_is_rejected_before_text_read(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "canonical-text-single-preflight")
    source_text = "x" * 8192
    _, revision = project.import_text("source.txt", source_text)
    run_id = project.create_run(project.snapshot([revision]), {"profile": "single-preflight"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical", "The source contains cited text."
    )
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 1))
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)

    ordinary_manifest = export_bundle(project, run_id, tmp_path / "ordinary-bundle")
    preflight_base_size = _canonical_text_preflight_base_size(
        json.loads(ordinary_manifest.read_text(encoding="utf-8"))
    )
    remaining = len(source_text.encode("utf-8")) // 2
    assert len(source_text.encode("utf-8")) > remaining
    monkeypatch.setattr(
        bundle_module,
        "MAX_BUNDLE_MANIFEST_BYTES",
        preflight_base_size + remaining,
    )

    revision_text_calls = []

    def unexpected_revision_text_read(revision_id):
        revision_text_calls.append(revision_id)
        raise AssertionError("revision text was read before bundle size preflight")

    monkeypatch.setattr(project, "revision_text", unexpected_revision_text_read)
    with pytest.raises(ProjectError, match="canonical text.*configured size limit"):
        export_bundle(
            project,
            run_id,
            tmp_path / "oversized-single-bundle",
            include_canonical_text=True,
        )

    assert revision_text_calls == []


def test_combined_cited_revision_sizes_are_preflighted_before_any_text_read(
    tmp_path, monkeypatch
):
    project = Project.init(tmp_path / "canonical-text-combined-preflight")
    source_texts = ["a" * 100_000, "b" * 100_000]
    revisions = [
        project.import_text(f"source-{index}.txt", text)[1]
        for index, text in enumerate(source_texts)
    ]
    run_id = project.create_run(project.snapshot(revisions), {"profile": "combined-preflight"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical", "The sources contain cited text."
    )
    for revision in revisions:
        evidence_id = project.textual_evidence(project.create_span(revision, 0, 1))
        project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)

    ordinary_manifest = export_bundle(project, run_id, tmp_path / "ordinary-bundle")
    preflight_base_size = _canonical_text_preflight_base_size(
        json.loads(ordinary_manifest.read_text(encoding="utf-8"))
    )
    remaining = 150_000
    raw_sizes = [len(text.encode("utf-8")) for text in source_texts]
    assert all(size < remaining for size in raw_sizes)
    assert sum(raw_sizes) > remaining
    monkeypatch.setattr(
        bundle_module,
        "MAX_BUNDLE_MANIFEST_BYTES",
        preflight_base_size + remaining,
    )

    revision_text_calls = []

    def unexpected_revision_text_read(revision_id):
        revision_text_calls.append(revision_id)
        raise AssertionError("revision text was read before bundle size preflight")

    monkeypatch.setattr(project, "revision_text", unexpected_revision_text_read)
    with pytest.raises(ProjectError, match="canonical text.*configured size limit"):
        export_bundle(
            project,
            run_id,
            tmp_path / "oversized-combined-bundle",
            include_canonical_text=True,
        )

    assert revision_text_calls == []


def test_bundle_exports_tied_timestamp_review_history_in_durable_order(tmp_path, monkeypatch):
    project = Project.init(tmp_path / "review-history")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(scope_id, {"profile": "review-export"})
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 6))
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical observation", "Apples occurs."
    )
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)

    generated_ids = iter(("z-first", "a-second"))
    monkeypatch.setattr("discourselens.project.uid", lambda: next(generated_ids))
    monkeypatch.setattr("discourselens.project.now", lambda: "2026-09-29T12:00:00+00:00")
    first_id = project.review_finding(finding_id, "ACCEPTED", "First decision.")
    second_id = project.review_finding(finding_id, "REJECTED", "Later decision.")
    assert first_id != second_id

    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    reviews = payload["human_reviews"][finding_id]
    assert [item["id"] for item in reviews] == [first_id, second_id]
    assert [item["status"] for item in reviews] == ["ACCEPTED", "REJECTED"]
    assert [item["review_order"] for item in reviews] == [1, 2]
    assert reviews[0]["created_at"] == reviews[1]["created_at"]
    assert verify_bundle(manifest)["integrity"] == "VERIFIED_SHA256"


def test_bundle_verifier_detects_review_tampering_and_rejects_invalid_shape(tmp_path):
    project = Project.init(tmp_path / "review-tamper")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(scope_id, {"profile": "review-tamper"})
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 6))
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical observation", "Apples occurs."
    )
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    project.review_finding(finding_id, "ACCEPTED", "Reviewed.")
    manifest = export_bundle(project, run_id, tmp_path / "bundle")

    tampered = json.loads(manifest.read_text(encoding="utf-8"))
    tampered["human_reviews"][finding_id][0]["status"] = "REJECTED"
    manifest.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ProjectError, match="integrity check failed"):
        verify_bundle(manifest)

    malformed = json.loads(export_bundle(project, run_id, tmp_path / "bundle-2").read_text())
    malformed["human_reviews"][finding_id][0]["review_order"] = "1"
    _write_with_valid_bundle_digest(manifest, malformed)
    with pytest.raises(ProjectError, match="human_reviews.*invalid review"):
        verify_bundle(manifest)


def test_bundle_verifier_requires_contiguous_review_order_even_with_valid_digest(tmp_path):
    project = Project.init(tmp_path / "review-order")
    _, revision = project.import_text("speech.txt", "Apples are green.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(scope_id, {"profile": "review-order"})
    evidence_id = project.textual_evidence(project.create_span(revision, 0, 6))
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "lexical observation", "Apples occurs."
    )
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    project.finalize_run(run_id)
    project.review_finding(finding_id, "ACCEPTED", "Reviewed.")
    project.review_finding(finding_id, "MODIFIED", "Updated.")
    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["human_reviews"][finding_id][1]["review_order"] = 3
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match="human_reviews.*invalid review"):
        verify_bundle(manifest)


def test_bundle_preserves_comparative_coverage_and_bilateral_evidence(tmp_path):
    project = Project.init(tmp_path / "comparative-bundle")
    _, left = project.import_text("left.txt", "Migration policy. Migration is debated.")
    _, right = project.import_text("right.txt", "Migration discourse. Migration changes.")
    analysis_scope = project.snapshot([left, right])
    run = project.create_run(analysis_scope, {"profile": "comparative-fixture"})
    comparison = compare_documents(project, left, right)
    result = project.corpus_comparison_evidence(
        run,
        str(comparison["comparison_id"]),
        "migration",
        comparability_basis="Both generated fixture texts represent the same declared genre.",
    )
    finding = project.create_finding(
        run,
        "fixture.comparison",
        "1.0.0",
        "relative frequency",
        "The selected token has a different rate in the two populations.",
        claim_types=[ClaimType.COMPARATIVE],
        coverage_id=str(result["coverage_id"]),
    )
    project.attach_evidence(finding, str(result["evidence_id"]), EvidenceRole.SUPPORT)
    manifest = export_bundle(project, run, tmp_path / "bundle")
    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert [item["id"] for item in data["coverage_descriptors"]] == [result["coverage_id"]]
    exported_evidence = data["evidence"][finding][0]
    assert "left_scope_snapshot_id" in exported_evidence["text"]
    assert "right_scope_snapshot_id" in exported_evidence["text"]
    assert str(comparison["comparison_id"]) in exported_evidence["citation"]
    assert verify_research_bundle(manifest)["status"] == "VALID"


def test_reranking_provenance_is_versioned_and_exported_without_secrets(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Migration policy. Migration discourse.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(scope_id, {"profile": "reranking-export-fixture"})
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=scope_id,
            population_definition="Two fixture passages",
        )
    )
    skill_run_id = project.create_skill_run(
        run_id, scope_id, "fixture.reranking", "1.0.0", "test", "fixture", coverage_id=coverage_id
    )
    retrieval_id = project.create_retrieval_run(
        analysis_run_id=run_id,
        skill_run_id=skill_run_id,
        scope_id=scope_id,
        retrieval_plan_ref="retrieval/test-v1",
        query_terms=["Migration"],
        selection_policy="fixture-candidates",
        candidate_count=2,
        inspected_count=2,
        returned_count=2,
        candidate_limit_reached=False,
        coverage_id=coverage_id,
    )
    spans = [project.create_span(revision, 0, 9), project.create_span(revision, 19, 28)]
    hit_ids = [
        project.add_retrieval_hit(retrieval_id, span_id, "lexical", rank, 1.0 / rank)
        for rank, span_id in enumerate(spans, 1)
    ]
    reranking_id = project.record_reranking(
        retrieval_id,
        provider_id="jina-reranker",
        credential_alias="university",
        model="reranker-test",
        query="Migration",
        top_n=7,
        effective_top_n=2,
        candidate_count=2,
        ranked_hits=[(hit_id, rank, 0.9 - rank / 100) for rank, hit_id in enumerate(hit_ids, 1)],
    )

    report_data = analysis_payload(project, run_id)
    bundle_data = json.loads(export_bundle(project, run_id, tmp_path / "bundle").read_text())

    for payload in (report_data, bundle_data):
        reranking = next(item for item in payload["reranking_runs"] if item["id"] == reranking_id)
        assert reranking["top_n"] == 7
        assert reranking["effective_top_n"] == 2
        assert reranking["candidate_count"] == 2
        assert reranking["returned_count"] == 2
        assert len(payload["reranking_hits"]) == 2
        serialized = json.dumps(payload)
        assert "university" in serialized
        assert "api_key" not in serialized.casefold()


def test_bundle_retains_scope_provenance_for_context_only_finding(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    run = project.create_run(project.snapshot([revision]), {"profile": "context"})
    finding = project.create_finding(run, "fixture", "1.0.0", "context", "Context supplied.")
    context = project.user_context_evidence("memo", "Researcher context.")
    from discourselens.types import EvidenceRole

    project.attach_evidence(finding, context, EvidenceRole.CONTEXT)
    manifest = export_bundle(project, run, tmp_path / "bundle")
    data = json.loads(manifest.read_text())
    assert [item["id"] for item in data["document_revisions"]] == [revision]
    assert data["evidence"][finding][0]["kind"] == "USER_CONTEXT"


def test_bundle_exports_only_documents_and_coding_from_the_run_scope(tmp_path):
    project = Project.init(tmp_path / "scoped-bundle")
    selected_text = "Selected document evidence."
    unselected_text = "UNSELECTED-DOCUMENT-TEXT-BAIT"
    _, selected_revision = project.import_text("selected.txt", selected_text)
    _, unselected_revision = project.import_text("unselected.txt", unselected_text)
    selected_scope = project.snapshot([selected_revision])
    unselected_scope = project.snapshot([unselected_revision])
    run_id = project.create_run(selected_scope, {"profile": "scoped-export"})
    finding_id = project.create_finding(
        run_id, "fixture", "1.0.0", "selected evidence", "A scoped observation."
    )
    evidence_id = project.textual_evidence(
        project.create_span(selected_revision, 0, len(selected_text))
    )
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    codebook_id = project.create_codebook("scope fixture")
    code_id = project.create_code(codebook_id, "unselected code")
    project.annotate_span(
        code_id,
        project.create_span(unselected_revision, 0, len(unselected_text)),
        "UNSELECTED-ANNOTATION-BAIT",
    )
    project.create_research_memo(unselected_scope, "Unselected", "UNSELECTED-MEMO-BAIT")

    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    raw = manifest.read_text(encoding="utf-8")
    payload = json.loads(raw)

    assert [item["id"] for item in payload["document_revisions"]] == [selected_revision]
    assert selected_text in raw
    assert unselected_text not in raw
    assert "UNSELECTED-ANNOTATION-BAIT" not in raw
    assert "UNSELECTED-MEMO-BAIT" not in raw


@pytest.mark.parametrize(
    "absolute_path",
    [
        "/",
        "/home/researcher/.cache/private-model",
        "//server/share/private-model",
        r"C:\Users\researcher\private-model",
        "file:///Users/researcher/private-model",
    ],
)
@pytest.mark.parametrize("location", ["run_config", "run_config_key", "model_parameters"])
def test_bundle_export_rejects_absolute_host_paths_in_metadata(tmp_path, absolute_path, location):
    project = Project.init(tmp_path / f"host-path-{location}")
    _, revision = project.import_text("source.txt", "Selected evidence.")
    config = {"profile": "path-export-fixture"}
    if location == "run_config":
        config["cache_path"] = absolute_path
    elif location == "run_config_key":
        config[absolute_path] = "private-model"
    run_id = project.create_run(project.snapshot([revision]), config)
    if location == "model_parameters":
        project.record_model_execution(
            provider_id="fixture",
            credential_alias="decoy-alias",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash=None,
            analysis_run_id=run_id,
            parameters={"cache_path": absolute_path},
        )
    destination = tmp_path / f"bundle-{location}"

    with pytest.raises(ProjectError, match="absolute host paths"):
        export_bundle(project, run_id, destination)

    assert not (destination / "manifest.json").exists()


def test_bundle_includes_scope_bound_human_coding_and_memos(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "Apples are green.")
    scope = project.snapshot([revision])
    run = project.create_run(scope, {"profile": "coding"})
    codebook = project.create_codebook("descriptive")
    code = project.create_code(codebook, "fruit")
    span = project.create_span(revision, 0, 6)
    annotation = project.annotate_span(code, span, "Human coding note.")
    memo = project.create_research_memo(scope, "Memo", "Scope-bound memo.")

    data = json.loads(export_bundle(project, run, tmp_path / "bundle").read_text())

    coding = data["qualitative_coding"]
    assert coding["annotations"][0]["id"] == annotation
    assert coding["codes"][0]["id"] == code
    assert coding["codebooks"][0]["id"] == codebook
    assert coding["memos"][0]["id"] == memo


def test_bundle_verifier_rejects_dangling_qualitative_annotation_code_reference(tmp_path):
    project = Project.init(tmp_path / "dangling-coding-reference")
    _, revision = project.import_text("a.txt", "Apples are green.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(scope_id, {"profile": "coding-reference"})
    codebook_id = project.create_codebook("descriptive")
    code_id = project.create_code(codebook_id, "fruit")
    span_id = project.create_span(revision, 0, 6)
    project.annotate_span(code_id, span_id, "Human coding note.")
    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["qualitative_coding"]["annotations"][0]["code_id"] = "missing-code"
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match="qualitative coding.*annotation.*code"):
        verify_bundle(manifest)


def test_bundle_verifier_rejects_dangling_qualitative_annotation_span_reference(tmp_path):
    project = Project.init(tmp_path / "dangling-coding-span-reference")
    _, revision = project.import_text("a.txt", "Apples are green.")
    scope_id = project.snapshot([revision])
    run_id = project.create_run(scope_id, {"profile": "coding-span-reference"})
    codebook_id = project.create_codebook("descriptive")
    code_id = project.create_code(codebook_id, "fruit")
    span_id = project.create_span(revision, 0, 6)
    project.annotate_span(code_id, span_id, "Human coding note.")
    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["qualitative_coding"]["annotations"][0]["span_id"] = "missing-span"
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match="qualitative coding.*annotation.*span"):
        verify_bundle(manifest)


def test_bundle_verifier_rejects_qualitative_span_from_an_unselected_revision(tmp_path):
    project = Project.init(tmp_path / "out-of-scope-coding-span")
    _, selected_revision = project.import_text("selected.txt", "Apples are green.")
    _, outside_revision = project.import_text("outside.txt", "Elephants are distant.")
    scope_id = project.snapshot([selected_revision])
    run_id = project.create_run(scope_id, {"profile": "coding-span-scope"})
    codebook_id = project.create_codebook("descriptive")
    code_id = project.create_code(codebook_id, "fruit")
    span_id = project.create_span(selected_revision, 0, 6)
    project.annotate_span(code_id, span_id, "Human coding note.")
    manifest = export_bundle(project, run_id, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["qualitative_coding"]["source_spans"][0]["revision_id"] = outside_revision
    _write_with_valid_bundle_digest(manifest, payload)

    with pytest.raises(ProjectError, match="source span belongs to an out-of-scope revision"):
        verify_bundle(manifest)


def test_bundle_export_rejects_secret_bearing_historical_model_metadata(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Evidence remains local.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    execution = project.record_model_execution(
        provider_id="fixture",
        credential_alias="research",
        model="fixture-model",
        reasoning=None,
        request_hash="request-hash",
        response_hash="response-hash",
        analysis_run_id=run,
        parameters={"temperature": 0.1},
    )
    secret = "must-not-be-exported-fixture"
    project.db.execute(
        "UPDATE model_executions SET parameters_json=? WHERE id=?",
        (json.dumps({"nested": [{"api-token": secret}]}), execution),
    )
    project.db.commit()

    destination = tmp_path / "bundle"
    with pytest.raises(ProjectError, match="secret-bearing model execution metadata"):
        export_bundle(project, run, destination)

    assert not (destination / "manifest.json").exists()


@pytest.mark.parametrize("boundary", ["export", "verify"])
def test_bundle_rejects_secret_bearing_model_price_source_reference(tmp_path, boundary):
    project = Project.init(tmp_path / f"secret-price-{boundary}")
    try:
        _, revision = project.import_text("a.txt", "Evidence remains local.")
        run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
        safe_price = ModelPrice(
            provider_id="fixture-provider",
            model="fixture-model",
            input_per_million_tokens="1",
            output_per_million_tokens="2",
            currency="USD",
            source_reference="https://prices.example/pricing-v1",
            rate_version="v1",
            rate_date="2026-01-01",
        )
        execution_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash="response-hash",
            analysis_run_id=run,
            input_tokens=1,
            output_tokens=1,
            model_price=safe_price,
        )
        destination = tmp_path / f"bundle-{boundary}"
        unsafe_snapshot = {
            **safe_price.to_snapshot(),
            "source_reference": (
                "https://prices.example/pricing?X-Amz-Signature=do-not-export-this"
            ),
        }

        if boundary == "export":
            project.db.execute(
                "UPDATE model_executions SET model_price_json=? WHERE id=?",
                (json.dumps(unsafe_snapshot), execution_id),
            )
            project.db.commit()
            with pytest.raises(ProjectError, match="model price snapshot is invalid"):
                export_bundle(project, run, destination)
            assert not (destination / "manifest.json").exists()
            return

        manifest = export_bundle(project, run, destination)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["model_executions"][0]["model_price_json"] = json.dumps(
            unsafe_snapshot, sort_keys=True, separators=(",", ":")
        )
        _write_with_valid_bundle_digest(manifest, payload)
        with pytest.raises(ProjectError, match="model price snapshot is invalid"):
            verify_bundle(manifest)
    finally:
        project.close()


def test_bundle_verifier_rejects_model_price_snapshot_with_unknown_fields(tmp_path):
    project = Project.init(tmp_path / "unknown-price-field")
    try:
        _, revision = project.import_text("a.txt", "Evidence remains local.")
        run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
        price = ModelPrice(
            provider_id="fixture-provider",
            model="fixture-model",
            input_per_million_tokens="1",
            output_per_million_tokens="2",
            currency="USD",
            source_reference="https://prices.example/pricing-v1",
            rate_version="v1",
            rate_date="2026-01-01",
        )
        project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash="response-hash",
            analysis_run_id=run,
            input_tokens=1,
            output_tokens=1,
            model_price=price,
        )
        manifest = export_bundle(project, run, tmp_path / "bundle")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        snapshot = json.loads(payload["model_executions"][0]["model_price_json"])
        snapshot["unrecognized"] = "must be rejected"
        payload["model_executions"][0]["model_price_json"] = json.dumps(snapshot)
        _write_with_valid_bundle_digest(manifest, payload)

        with pytest.raises(ProjectError, match="model price snapshot is invalid"):
            verify_bundle(manifest)
    finally:
        project.close()


@pytest.mark.parametrize("boundary", ["export", "verify"])
@pytest.mark.parametrize(
    ("snapshot_provider", "snapshot_model"),
    [("fixture-provider", "other-model"), ("other-provider", "fixture-model")],
)
def test_bundle_rejects_valid_price_snapshot_for_a_different_execution(
    tmp_path, boundary, snapshot_provider, snapshot_model
):
    project = Project.init(
        tmp_path / f"mismatched-price-{boundary}-{snapshot_provider}-{snapshot_model}"
    )
    try:
        _, revision = project.import_text("a.txt", "Evidence remains local.")
        run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
        execution_price = ModelPrice(
            provider_id="fixture-provider",
            model="fixture-model",
            input_per_million_tokens="1.25",
            output_per_million_tokens="2.5",
            currency="USD",
            source_reference="https://prices.example/fixture-model",
            rate_version="v1",
            rate_date="2026-01-01",
        )
        other_model_price = ModelPrice(
            provider_id=snapshot_provider,
            model=snapshot_model,
            input_per_million_tokens="1.25",
            output_per_million_tokens="2.5",
            currency="USD",
            source_reference="https://prices.example/other-model",
            rate_version="v1",
            rate_date="2026-01-01",
        )
        execution_id = project.record_model_execution(
            provider_id="fixture-provider",
            credential_alias="fixture",
            model="fixture-model",
            reasoning=None,
            request_hash="request-hash",
            response_hash="response-hash",
            analysis_run_id=run,
            input_tokens=1_000,
            output_tokens=500,
            model_price=execution_price,
        )
        stored_cost = project.db.execute(
            "SELECT estimated_cost FROM model_executions WHERE id=?", (execution_id,)
        ).fetchone()[0]
        assert stored_cost == "0.0025"
        destination = tmp_path / f"bundle-{boundary}"

        if boundary == "export":
            project.db.execute(
                "UPDATE model_executions SET model_price_json=? WHERE id=?",
                (
                    json.dumps(
                        other_model_price.to_snapshot(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    execution_id,
                ),
            )
            project.db.commit()
            with pytest.raises(ProjectError, match="does not match model execution"):
                export_bundle(project, run, destination)
            assert (
                project.db.execute(
                    "SELECT estimated_cost FROM model_executions WHERE id=?", (execution_id,)
                ).fetchone()[0]
                == stored_cost
            )
            assert not (destination / "manifest.json").exists()
            return

        manifest = export_bundle(project, run, destination)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        execution = payload["model_executions"][0]
        execution["model_price_json"] = json.dumps(
            other_model_price.to_snapshot(), sort_keys=True, separators=(",", ":")
        )
        assert execution["estimated_cost"] == stored_cost
        _write_with_valid_bundle_digest(manifest, payload)
        with pytest.raises(ProjectError, match="does not match model execution"):
            verify_bundle(manifest)
    finally:
        project.close()


def test_bundle_export_rejects_secret_bearing_historical_run_configuration(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Evidence remains local.")
    run = project.create_run(project.snapshot([revision]), {"profile": "fixture"})
    config_json = json.dumps(
        {"profile": "fixture", "provider": {"vendorApiKey": "redacted"}},
        sort_keys=True,
    )
    # Exercise the export-side validation after simulating external DB tampering.
    project.db.execute("DROP TRIGGER analysis_runs_provenance_immutable")
    project.db.execute(
        "UPDATE analysis_runs SET config_json=?,config_hash=? WHERE id=?",
        (config_json, hashlib.sha256(config_json.encode("utf-8")).hexdigest(), run),
    )
    project.db.commit()
    destination = tmp_path / "bundle"

    with pytest.raises(ProjectError, match="secret-bearing configuration metadata"):
        export_bundle(project, run, destination)

    assert not destination.exists()


@pytest.mark.parametrize(
    "secret_key",
    [
        "apiKey",
        "ApiKey",
        "APIKey",
        "apiToken",
        "apiTokens",
        "private_key",
        "privateKey",
        "PrivateKey",
        "openai_api_key",
        "anthropicApiKey",
        "vendor_api_key",
        "vendorAPIKey",
        "x-api-key",
        "bearerToken",
        "bearerTokens",
        "oauthAccessToken",
        "oauthAccessTokens",
        "refresh_tokens",
        "secretKey",
    ],
)
def test_secret_metadata_key_guard_recognizes_common_spellings(secret_key):
    assert _contains_secret_metadata_key({"metadata": [{secret_key: "redacted"}]})


@pytest.mark.parametrize(
    "secret_key",
    ["credential", "credentialReference", "credential_ref", "CredentialValue"],
)
def test_secret_metadata_key_guard_rejects_credential_references(secret_key):
    assert _contains_secret_metadata_key({"metadata": {secret_key: "redacted"}})


def test_secret_metadata_key_guard_allows_reproducibility_aliases():
    assert not _contains_secret_metadata_key({"credential_alias": "university"})


@pytest.mark.parametrize(
    "usage_key",
    [
        "reasoning_tokens",
        "token_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_tokens",
    ],
)
def test_secret_metadata_key_guard_allows_token_usage_metadata(usage_key):
    assert not _contains_secret_metadata_key({"usage": {usage_key: 128}})
