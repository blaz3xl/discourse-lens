import hashlib
import json
import sqlite3
import unicodedata
from typing import cast

import pytest

import discourselens.interpretation as interpretation_module
from discourselens import Project
from discourselens.errors import SourceSpanError
from discourselens.models import CredentialReference, ModelPreset
from discourselens.nlp import ParsedToken
from discourselens.types import ClaimType, EvidenceRole


class _DependencyBackend:
    language = "es"
    backend_id = "fixture"
    model_id = "fixture-v1"

    def parse(self, text: str) -> list[ParsedToken]:
        return [
            ParsedToken("Café", "café", "NOUN", None, None, "ROOT", 0, 4),
            ParsedToken("migración", "migración", "NOUN", None, 0, "obj", 5, 14),
        ]


def test_python_sdk_span_offsets_require_exact_integer_coordinates(tmp_path):
    with Project.init(tmp_path / "span-offsets") as project:
        _, revision = project.import_text("unicode.txt", "Café migration")
        valid_span = project.create_span(revision, 0, 4)
        initial_count = project.db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0]

        # bool subclasses int and floats compare like integer offsets, but neither
        # belongs in the canonical Unicode-code-point coordinate contract.
        invalid_ranges: tuple[tuple[object, object], ...] = (
            (False, 4),
            (0, True),
            (0.0, 4.0),
            (1.5, 4),
        )
        for start, end in invalid_ranges:
            with pytest.raises(SourceSpanError, match="invalid half-open offsets"):
                project.create_span(revision, cast(int, start), cast(int, end))

        # Validate all inputs before range deduplication/writes: (False, 2) is
        # tuple-equal to (0, 2) in Python and must not be silently accepted.
        with pytest.raises(SourceSpanError, match="invalid half-open offsets"):
            project.create_spans(revision, [(0, 2), (False, 2)])

        assert project.span_text(valid_span) == "Café"
        assert project.db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == initial_count


def test_python_sdk_project_context_import_scope_and_frequency(tmp_path):
    root = tmp_path / "research"
    source = tmp_path / "speech.txt"
    source.write_text(
        "Migration policy matters. Migration changes public debate.", encoding="utf-8"
    )

    with Project.init(root) as project:
        imported = project.import_document(source)
        assert imported.status == "SUCCESS"
        assert imported.document_id is not None
        assert imported.revision_id is not None
        project.import_text("outside.txt", "Elephants live in Africa.")

        scope_id = project.snapshot([imported.revision_id])
        counts = project.corpus_frequency(scope_id)
        assert counts["migration"] == 2
        assert "elephants" not in counts
        lines = project.corpus_kwic(scope_id, "migration", window=2)
        assert len(lines) == 2
        assert {line["revision_id"] for line in lines} == {imported.revision_id}
        first_page = project.corpus_kwic_page(scope_id, "migration", window=2, limit=1)
        next_page = project.corpus_kwic_page(scope_id, "migration", window=2, offset=1, limit=1)
        assert first_page["total_count"] == 2 and first_page["has_more"]
        assert next_page["total_count"] == 2 and not next_page["has_more"]
        assert first_page["rows"][0] == lines[0]
        assert next_page["rows"][0] == lines[1]
        associations = project.corpus_collocation_analysis(
            scope_id, "migration", window=2, min_frequency=1
        )
        assert associations["focus_frequency"] == 2
        assert associations["scope_id"] == scope_id
        association_evidence = project.corpus_collocation_evidence(
            scope_id, "migration", "policy", window=2, min_frequency=1
        )
        assert association_evidence["scope_id"] == scope_id
        assert association_evidence["evidence_id"]
        assert project.corpus_collocations(scope_id, "migration", window=2)["policy"] == 2
        assert project.corpus_ngrams(scope_id, 2)[("migration", "policy")] == 1
        bundles = project.corpus_lexical_bundles(scope_id, size=2, min_frequency=1)
        assert bundles["scope_id"] == scope_id
        assert any(row["bundle"] == "migration policy" for row in bundles["rows"])
        bundle_evidence = project.corpus_lexical_bundle_evidence(
            scope_id, "migration policy", size=2
        )
        assert bundle_evidence["count"] == 1
        assert project.evidence_content(bundle_evidence["evidence_id"])["value"] == 1
        statistic = project.corpus_frequency_evidence(scope_id, "migration")
        assert statistic["count"] == 2
        assert (
            project.coverage_descriptor(str(statistic["coverage_id"])).scope_snapshot_id == scope_id
        )

    try:
        project.db.execute("SELECT 1")
    except sqlite3.ProgrammingError as exc:
        assert "closed" in str(exc).casefold()
    else:
        raise AssertionError("Project context manager must close its SQLite connection")


def test_python_sdk_dispersion_is_scope_bound_and_matches_cli_mcp(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from discourselens.cli import app
    from discourselens.mcp_server import corpus_dispersion as mcp_dispersion
    from discourselens.mcp_server import corpus_dispersion_evidence as mcp_dispersion_evidence

    root = tmp_path / "dispersion-sdk"
    project = Project.init(root)
    _, first_revision = project.import_text("first.txt", "Migration policy migration.")
    _, second_revision = project.import_text("second.txt", "Migration matters.")
    _, outside_revision = project.import_text("outside.txt", "Migration outside the selection.")
    scope_id = project.snapshot([first_revision, second_revision])

    sdk_result = project.corpus_dispersion(scope_id, "MIGRATION")
    assert sdk_result["scope_id"] == scope_id
    assert sdk_result["term"] == "migration"
    assert sdk_result["document_count"] == 2
    assert sdk_result["document_frequency"] == 2
    assert sdk_result["total_occurrences"] == 3
    assert {row["revision_id"] for row in sdk_result["documents"]} == {
        first_revision,
        second_revision,
    }

    cli_result = CliRunner().invoke(
        app,
        ["corpus", "dispersion", str(root), f"{first_revision},{second_revision}", "MIGRATION"],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_payload = json.loads(cli_result.output)
    mcp_page = project.corpus_dispersion_page(scope_id, "MIGRATION")
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    mcp_payload = mcp_dispersion([first_revision, second_revision], "MIGRATION")
    assert {key: value for key, value in cli_payload.items() if key != "scope_id"} == {
        key: value for key, value in sdk_result.items() if key != "scope_id"
    }
    assert {key: value for key, value in mcp_payload.items() if key != "scope_id"} == {
        key: value for key, value in mcp_page.items() if key != "scope_id"
    }

    sdk_evidence = project.corpus_dispersion_evidence(scope_id, "migration")
    mcp_evidence = mcp_dispersion_evidence([first_revision, second_revision], "migration")
    assert sdk_evidence["total_occurrences"] == mcp_evidence["total_occurrences"] == 3
    assert sdk_evidence["documents"] == mcp_evidence["documents"]
    assert sdk_evidence["scope_id"] == scope_id
    assert sdk_evidence["evidence_id"]
    assert mcp_evidence["evidence_id"]
    assert {row["revision_id"] for row in sdk_evidence["documents"]} == {
        first_revision,
        second_revision,
    }
    sdk_metadata = project.evidence_content(str(sdk_evidence["evidence_id"]))
    assert sdk_metadata["scope_snapshot_id"] == scope_id
    provenance = sdk_metadata["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["documents"] == sdk_evidence["documents"]
    cli_evidence_result = CliRunner().invoke(
        app,
        [
            "corpus",
            "dispersion-evidence",
            str(root),
            f"{first_revision},{second_revision}",
            "migration",
        ],
    )
    assert cli_evidence_result.exit_code == 0, cli_evidence_result.output
    cli_evidence = json.loads(cli_evidence_result.output)
    assert cli_evidence["documents"] == sdk_evidence["documents"]
    assert cli_evidence["total_occurrences"] == sdk_evidence["total_occurrences"]
    assert cli_evidence["evidence_id"]
    assert outside_revision not in {row["revision_id"] for row in cli_evidence["documents"]}
    project.close()


def test_python_sdk_named_corpus_comparison_matches_cli_mcp(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import discourselens.mcp_server as mcp_module
    from discourselens.cli import app

    root = tmp_path / "named-corpus-comparison-sdk"
    with Project.init(root) as project:
        _, left_first = project.import_text("left-first.txt", "targetonly targetonly shared")
        _, left_second = project.import_text("left-second.txt", "targetonly secondary")
        _, right_revision = project.import_text("right.txt", "referenceonly referenceonly shared")
        left_corpus = project.create_corpus("left")
        right_corpus = project.create_corpus("right")
        project.add_revision_to_corpus(left_corpus, left_first)
        project.add_revision_to_corpus(left_corpus, left_second)
        project.add_revision_to_corpus(right_corpus, right_revision)

        sdk_result = project.compare_named_corpora(
            left_corpus, right_corpus, metric="keyness", offset=0, limit=1
        )
        assert sdk_result["left_corpus_id"] == left_corpus
        assert sdk_result["right_corpus_id"] == right_corpus
        assert project.scope_revisions(str(sdk_result["left_scope_id"])) == sorted(
            [left_first, left_second]
        )
        assert project.scope_revisions(str(sdk_result["right_scope_id"])) == [right_revision]
        stored = project.comparison_run(str(sdk_result["comparison_id"]))
        assert stored["left_scope_id"] == sdk_result["left_scope_id"]
        assert stored["right_scope_id"] == sdk_result["right_scope_id"]
        stored_page = project.comparison_rows_page(
            str(sdk_result["comparison_id"]), offset=0, limit=1
        )
        assert stored_page["rows"] == sdk_result["rows"]
        assert sdk_result["has_more"]

    cli_result = CliRunner().invoke(
        app,
        [
            "corpus",
            "compare-named",
            str(root),
            left_corpus,
            right_corpus,
            "--metric",
            "keyness",
            "--limit",
            "1",
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    cli_payload = json.loads(cli_result.output)
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    mcp_payload = mcp_module.compare_named_corpora(
        left_corpus, right_corpus, metric="keyness", limit=1
    )

    for payload in (cli_payload, mcp_payload):
        assert payload["metric"] == sdk_result["metric"] == "keyness"
        assert payload["left_corpus_id"] == left_corpus
        assert payload["right_corpus_id"] == right_corpus
        assert payload["rows"] == sdk_result["rows"]
        assert payload["total_rows"] == sdk_result["total_rows"]
        assert payload["has_more"] == sdk_result["has_more"]


def test_python_sdk_unicode_frequency_kwic_and_evidence_preserve_source_offsets(tmp_path):
    source_text = "🐦 cafe\u0301—CAFÉ"

    with Project.init(tmp_path / "unicode-sdk") as project:
        _, included_revision = project.import_text("included.txt", source_text)
        project.import_text("outside.txt", "café elephants")
        scope_id = project.snapshot([included_revision])

        counts = project.corpus_frequency(scope_id)
        assert counts == {"café": 2}
        statistic = project.corpus_frequency_evidence(scope_id, "cafe\u0301")
        assert statistic["scope_id"] == scope_id
        assert statistic["term"] == "café"
        assert statistic["count"] == 2
        statistic_metadata = project.evidence_content(str(statistic["evidence_id"]))
        statistic_provenance = statistic_metadata["provenance"]
        assert isinstance(statistic_provenance, dict)
        assert statistic_provenance["calculation_version"] == "surface-frequency-v2"
        assert statistic_provenance["tokenizer"] == "unicode-surface-tokenizer-v2"
        assert statistic_provenance["normalization"] == "NFC-after-casefold-v1"
        assert statistic_provenance["unicode_data_version"] == unicodedata.unidata_version
        assert statistic_provenance["term"] == "café"
        assert statistic_metadata["scope_snapshot_id"] == scope_id

        rows = project.corpus_kwic(scope_id, "CAFE\u0301", window=0)
        assert [row["keyword"] for row in rows] == ["cafe\u0301", "CAFÉ"]
        assert [(row["start_offset"], row["end_offset"]) for row in rows] == [(2, 7), (8, 12)]
        assert {row["revision_id"] for row in rows} == {included_revision}
        assert all(
            source_text[row["start_offset"] : row["end_offset"]] == row["keyword"] for row in rows
        )
        page = project.corpus_kwic_page(scope_id, "café", window=0, limit=1)
        assert page["scope_id"] == scope_id
        assert page["total_count"] == 2
        assert page["rows"] == rows[:1]

        citation = project.corpus_kwic_evidence(
            scope_id,
            included_revision,
            "café",
            rows[0]["start_offset"],
            rows[0]["end_offset"],
        )
        assert citation["scope_id"] == scope_id
        assert citation["exact_text"] == "cafe\u0301"
        assert (
            source_text[citation["start_offset"] : citation["end_offset"]] == citation["exact_text"]
        )
        assert citation["citation"] == "[included.txt · ¶1]"
        kwic_metadata = project.evidence_content(citation["evidence_id"])
        assert kwic_metadata["acquisition_method"] == "corpus_kwic"
        assert kwic_metadata["scope_snapshot_id"] == scope_id
        assert kwic_metadata["calculation_version"] == "surface-kwic-v2"
        assert kwic_metadata["tokenizer"] == "unicode-surface-tokenizer-v2"
        assert kwic_metadata["normalization"] == "NFC-after-casefold-v1"
        assert kwic_metadata["unicode_data_version"] == unicodedata.unidata_version

        run_id = project.create_run(scope_id, {"fixture": "unicode-sdk"})
        frequency_finding = project.create_finding(
            run_id,
            "fixture",
            "1.0.0",
            "normalized frequency",
            "The selected scope contains two canonically equivalent surface forms.",
            claim_types=[ClaimType.QUANTIFIED],
            coverage_id=str(statistic["coverage_id"]),
        )
        project.attach_evidence(
            frequency_finding, str(statistic["evidence_id"]), EvidenceRole.SUPPORT
        )
        kwic_finding = project.create_finding(
            run_id, "fixture", "1.0.0", "Unicode excerpt", "The source excerpt is preserved."
        )
        project.attach_evidence(kwic_finding, citation["evidence_id"], EvidenceRole.SUPPORT)

        assert project.validate_finding(frequency_finding).valid
        assert project.validate_finding(kwic_finding).valid


def test_python_sdk_model_review_accepts_explicit_context_ids(tmp_path, monkeypatch):
    from discourselens.interpretation import InterpretationResult

    project = Project.init(tmp_path / "review-sdk")
    captured: dict[str, object] = {}

    def fake_review(project, finding_id, preset_name, **kwargs):
        captured.update(kwargs)
        return InterpretationResult("run-id", "review-id", "execution-id")

    monkeypatch.setattr(interpretation_module, "review_finding_with_model", fake_review)
    result = project.review_finding_with_model(
        "source-id",
        "review-preset",
        framework="fairclough-text",
        context_evidence_ids=("external-id", "user-context-id"),
    )

    assert result.finding_id == "review-id"
    assert captured == {
        "framework": "fairclough-text",
        "context_evidence_ids": ("external-id", "user-context-id"),
    }


def test_python_sdk_previews_embedding_request_without_secret_or_network(tmp_path):
    root = tmp_path / "research"
    with Project.init(root) as project:
        _, revision = project.import_text("notes.txt", "First paragraph.\nSecond paragraph.")
        project.save_credential_reference(
            CredentialReference("gemini-embeddings", "lab", "DL_GEMINI_PREVIEW")
        )
        project.save_model_preset(
            ModelPreset("embed", "gemini-embeddings", "lab", "gemini-embedding-2")
        )

    with Project.open(root, read_only=True) as project:
        preview = project.preview_paragraph_embedding_request("embed", revision, offset=1, limit=1)

    assert preview.total_paragraphs == 2
    assert preview.selected_count == 1
    assert preview.network_request_made is False
    assert preview.secret_values_resolved is False
    assert preview.batches[0].request_body is None
    assert preview.batches[0].inputs[0].prepared_text is None


def test_python_sdk_previews_generation_request_without_secret_or_network(tmp_path, monkeypatch):
    from discourselens.models import ModelCapabilities, ModelCapabilityProfile

    monkeypatch.delenv("DL_OPENAI_PREVIEW_SECRET", raising=False)

    def no_network(*args, **kwargs):
        del args, kwargs
        raise AssertionError("preview must never make a network request")

    monkeypatch.setattr("discourselens.providers._open_provider_request", no_network)
    with Project.init(tmp_path / "research") as project:
        project.save_credential_reference(
            CredentialReference("openai", "university", "DL_OPENAI_PREVIEW_SECRET")
        )
        project.save_model_preset(ModelPreset("analysis", "openai", "university", "fixture-model"))
        project.save_model_capability_profile(
            ModelCapabilityProfile(
                "openai",
                "fixture-model",
                ModelCapabilities(
                    generation_parameters=(
                        "temperature",
                        "max_tokens",
                        "top_p",
                        "stop",
                        "frequency_penalty",
                        "presence_penalty",
                        "seed",
                    )
                ),
                assurance="TESTED",
            )
        )
        preview = project.preview_model_preset_request("analysis", "A private research question")

    assert preview.provider_id == "openai"
    assert preview.credential_alias == "university"
    assert preview.endpoint == "https://api.openai.com/v1"
    assert preview.model == "fixture-model"
    assert preview.allowed is True
    assert preview.prompt_characters == len("A private research question")
    assert preview.prompt_hash
    assert not hasattr(preview, "prompt")
    assert "DL_OPENAI_PREVIEW_SECRET" not in repr(preview)
    assert "A private research question" not in repr(preview)


def test_python_sdk_generate_with_preset_uses_core_image_offline_and_capability_checks(
    tmp_path, monkeypatch
):
    import base64
    from email.message import Message
    from typing import Self, cast
    from urllib.request import Request

    import pytest

    from discourselens.errors import ModelProviderError, ProjectError
    from discourselens.models import GenerationImage, ModelCapabilities, ModelCapabilityProfile

    secret_reference = "DL_SDK_GEMINI_KEY"
    monkeypatch.setenv(secret_reference, "sdk-generation-test-secret")
    image_path = tmp_path / "research-image.png"
    image_bytes = b"\x89PNG\r\n\x1a\nlocal-fixture-image"
    image_path.write_bytes(image_bytes)
    image = GenerationImage.from_file(image_path)
    prompt = "Describe this local research image"
    response_body = json.dumps(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "Image description"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3},
        }
    ).encode("utf-8")

    class _Response:
        def __init__(self) -> None:
            self.headers = Message()

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            return response_body if size < 0 else response_body[:size]

    requests: list[object] = []

    def open_request(request: Request, *, timeout: float) -> _Response:
        assert timeout > 0
        assert request.data is not None
        requests.append(json.loads(cast(bytes, request.data).decode("utf-8")))
        return _Response()

    monkeypatch.setattr("discourselens.providers._open_provider_request", open_request)
    root = tmp_path / "generation-sdk"
    with Project.init(root) as project:
        project.save_credential_reference(
            CredentialReference("gemini", "research", secret_reference)
        )
        project.save_model_preset(
            ModelPreset(
                "analysis",
                "gemini",
                "research",
                "gemini-sdk-fixture",
                parameters={"temperature": 0.2, "max_output_tokens": 128},
            )
        )
        project.save_model_capability_profile(
            ModelCapabilityProfile(
                "gemini",
                "gemini-sdk-fixture",
                ModelCapabilities(
                    system_prompt=True,
                    vision=True,
                    generation_parameters=("temperature", "max_output_tokens"),
                ),
                assurance="TESTED",
            )
        )

        result = project.generate_with_preset("analysis", prompt, image=image)
        assert result.response.text == "Image description"
        assert result.response.provider == "gemini"
        assert result.response.model == "gemini-sdk-fixture"
        assert len(requests) == 1
        assert base64.b64encode(image_bytes).decode("ascii") in json.dumps(requests[0])

        ledger = project.db.execute(
            "SELECT * FROM model_executions WHERE id=?", (result.execution_id,)
        ).fetchone()
        assert ledger is not None
        capabilities = json.loads(ledger["capabilities_json"])
        assert capabilities["request_image"] == image.metadata()
        ledger_text = json.dumps(dict(ledger))
        assert prompt not in ledger_text
        assert "Image description" not in ledger_text
        assert image_path.name not in ledger_text
        assert base64.b64encode(image_bytes).decode("ascii") not in ledger_text

        project.set_offline_mode(True)
        with pytest.raises(ProjectError, match="offline mode"):
            project.generate_with_preset("analysis", "offline prompt")
        assert len(requests) == 1

        project.set_offline_mode(False)
        project.save_model_capability_profile(
            ModelCapabilityProfile(
                "gemini",
                "gemini-sdk-fixture",
                ModelCapabilities(
                    system_prompt=True,
                    vision=False,
                    generation_parameters=("temperature", "max_output_tokens"),
                ),
                assurance="TESTED",
            )
        )
        with pytest.raises(ModelProviderError, match="vision support"):
            project.generate_with_preset("analysis", prompt, image=image)
        assert len(requests) == 1

    with Project.open(root, read_only=True) as project, pytest.raises(
        ProjectError, match="read-only"
    ):
        project.generate_with_preset("analysis", prompt)


def test_python_sdk_reopens_portable_project_and_uses_same_revision(tmp_path):
    root = tmp_path / "research"
    source = tmp_path / "notes.md"
    source.write_text("Café migration.", encoding="utf-8")

    with Project.init(root) as project:
        imported = project.import_document(source)
        project_id = project.project_id

    with Project.open(root) as reopened:
        assert reopened.project_id == project_id
        assert imported.revision_id is not None
        assert reopened.revision_text(imported.revision_id) == "Café migration."


def test_python_sdk_inspects_explicit_local_language_backend_without_download(
    tmp_path, monkeypatch
):
    import discourselens.languages as language_module
    from discourselens.languages import LanguageCapability

    requested: dict[str, object] = {}

    def inspect(language, *, backend, model, model_dir=None):
        requested.update(language=language, backend=backend, model=model, model_dir=model_dir)
        return (
            LanguageCapability(language, "dependency_parse", backend, model, "SUPPORTED", "TESTED"),
        )

    monkeypatch.setattr(language_module, "inspect_local_backend", inspect)
    with Project.init(tmp_path / "research") as project:
        result = project.inspect_language_capabilities(
            "ar", backend="stanza", model="ar-ud", model_dir=tmp_path / "resources"
        )

    assert requested == {
        "language": "ar",
        "backend": "stanza",
        "model": "ar-ud",
        "model_dir": str(tmp_path / "resources"),
    }
    assert result[0].assurance_status == "TESTED"


def test_python_sdk_dependency_patterns_use_registered_artifact_scope(tmp_path):
    with Project.init(tmp_path / "research") as project:
        _, revision = project.import_text("sample.txt", "Café migración")
        artifact = project.build_parsed_token_artifact(revision, _DependencyBackend())
        result = project.corpus_dependency_patterns(
            project.snapshot([revision]), [artifact.artifact_id], relation="obj"
        )
        assert result["rows"][0]["dependent_lemma"] == "migración"


def test_python_sdk_builds_portable_surface_artifact_through_project_core(tmp_path):
    from pyarrow import parquet

    with Project.init(tmp_path / "research") as project:
        _, revision = project.import_text("sample.txt", "Café migración")
        artifact = project.build_surface_token_artifact(revision)
        table = parquet.read_table(project.hidden / artifact.relative_path)
        assert table.column("surface").to_pylist() == ["Café", "migración"]
        assert artifact.revision_id == revision


def test_python_sdk_bundle_full_text_requires_opt_in_but_keeps_cited_excerpt(tmp_path):
    source_text = "Private source corpus text."
    with Project.init(tmp_path / "research") as project:
        _, revision = project.import_text("source.txt", source_text)
        run_id = project.create_run(project.snapshot([revision]), {"profile": "bundle-test"})
        evidence_id = project.textual_evidence(
            project.create_span(revision, 0, len(source_text))
        )
        finding_id = project.create_finding(
            run_id, "fixture", "1.0.0", "text", "Source text exists."
        )
        project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
        project.start_run(run_id)
        project.finalize_run(run_id)

        default_manifest = project.export_research_bundle(
            run_id, tmp_path / "default-bundle"
        )
        opted_manifest = project.export_research_bundle(
            run_id,
            tmp_path / "opted-bundle",
            include_canonical_text=True,
        )

    default_payload = json.loads(default_manifest.read_text(encoding="utf-8"))
    opted_payload = json.loads(opted_manifest.read_text(encoding="utf-8"))
    assert default_payload["canonical_texts"] == []
    assert default_payload["evidence"][finding_id][0]["text"] == source_text
    assert opted_payload["canonical_texts"] == [
        {
            "revision_id": revision,
            "sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "text": source_text,
        }
    ]


def test_python_sdk_runs_deterministic_analysis_with_saved_scope(tmp_path):
    source = tmp_path / "speech.txt"
    source.write_text("Officials must respond. The public must know.", encoding="utf-8")

    with Project.init(tmp_path / "research") as project:
        imported = project.import_document(source)
        assert imported.revision_id is not None

        descriptors = project.analyze_textual_descriptors([imported.revision_id], languages=("en",))
        extraction = project.extract_evidence(
            [imported.revision_id], languages=("en",), max_examples=8
        )

        assert descriptors["finding_ids"]
        assert extraction["finding_ids"]
        assert project.validate_finding(descriptors["finding_ids"][0]).valid
        report = project.export_analysis_report(
            descriptors["analysis_run_id"], tmp_path / "report.json", format="json"
        )
        report_data = json.loads(report.read_text(encoding="utf-8"))
        assert report_data["analysis_run"]["id"] == descriptors["analysis_run_id"]
        bundle = project.export_research_bundle(
            descriptors["analysis_run_id"], tmp_path / "research-bundle"
        )
        assert (
            json.loads(bundle.read_text(encoding="utf-8"))["analysis_run"]["id"]
            == (descriptors["analysis_run_id"])
        )
        for result in (descriptors, extraction):
            run = project.analysis_run_summary(result["analysis_run_id"])
            run_metadata = run["analysis_run"]
            assert isinstance(run_metadata, dict)
            assert run_metadata["status"] == "COMPLETED"
            assert run["revision_ids"] == [imported.revision_id]


def test_python_sdk_pages_saved_counterevidence_without_rescanning(tmp_path):
    with Project.init(tmp_path / "research") as project:
        _, revision = project.import_text("speech.txt", "Claim. Exception one. Exception two.")
        analysis_run_id = project.create_run(project.snapshot([revision]), {"profile": "test"})
        evidence_id = project.textual_evidence(project.create_span(revision, 0, 5))
        finding_id = project.create_finding(
            analysis_run_id, "fixture", "1.0.0", "claim", "Claim occurs."
        )
        project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
        project.finalize_run(analysis_run_id)

        first = project.counterevidence_review(finding_id, "Exception", limit=1)
        retrieval_run_count = project.db.execute("SELECT COUNT(*) FROM retrieval_runs").fetchone()[
            0
        ]
        second = project.counterevidence_review(
            finding_id,
            "Exception",
            offset=1,
            limit=1,
            retrieval_run_id=first["retrieval_run_id"],
        )

        assert first["candidate_count"] == second["candidate_count"] == 2
        assert first["returned_count"] == second["returned_count"] == 1
        assert first["retrieval_run_id"] == second["retrieval_run_id"]
        assert first["analysis_run_id"] == second["analysis_run_id"]
        assert project.db.execute("SELECT COUNT(*) FROM retrieval_runs").fetchone()[0] == (
            retrieval_run_count
        )


def test_python_sdk_vector_workflow_uses_explicit_scope_and_preset(tmp_path, monkeypatch):
    import discourselens.vectors as vectors_module
    from discourselens.vectors import HybridHit, VectorHit

    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    hybrid_result = [HybridHit("revision-id", "paragraph-id", 1, 2, 0.032)]
    semantic_result = [VectorHit("artifact-id", "revision-id", "paragraph-id", 0, 8, 0.1)]

    def record_build(*args, **kwargs):
        calls.append(("build", args, kwargs))
        return "artifact-id"

    def record_semantic(*args, **kwargs):
        calls.append(("semantic", args, kwargs))
        return semantic_result

    def record_hybrid(*args, **kwargs):
        calls.append(("hybrid", args, kwargs))
        return hybrid_result

    monkeypatch.setattr(vectors_module, "build_paragraph_embedding_index", record_build)
    monkeypatch.setattr(vectors_module, "semantic_search_with_preset", record_semantic)
    monkeypatch.setattr(vectors_module, "hybrid_search_with_preset", record_hybrid)

    with Project.init(tmp_path / "research") as project:
        assert project.build_paragraph_embedding_index("embed", "revision-id") == "artifact-id"
        scope_id = "frozen-scope-id"
        assert (
            project.semantic_search_with_preset(
                "embed", scope_id, ["artifact-id"], "private query"
            )
            == semantic_result
        )
        assert (
            project.hybrid_search_with_preset(
                "embed",
                scope_id,
                ["artifact-id"],
                "private query",
                limit=4,
                rrf_constant=20,
            )
            == hybrid_result
        )

    assert calls[0] == ("build", (project, "embed", "revision-id"), {"embedding_provider": None})
    assert calls[1] == (
        "semantic",
        (project, "embed", scope_id, ["artifact-id"], "private query"),
        {"embedding_provider": None, "limit": 10},
    )
    assert calls[2] == (
        "hybrid",
        (project, "embed", scope_id, ["artifact-id"], "private query"),
        {"embedding_provider": None, "limit": 4, "rrf_constant": 20},
    )
