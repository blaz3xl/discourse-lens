from __future__ import annotations

import json

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project

# Keep the write-capable tool inventory independent of the production allowlist:
# adding a newly registered mutator to read-only mode must fail this contract.
REGISTERED_MUTATING_TOOLS = frozenset(
    {
        "add_external_source_evidence",
        "add_revision_to_corpus",
        "add_user_context_evidence",
        "ask_question",
        "annotate_span",
        "build_paragraph_embedding_index",
        "build_parsed_token_artifact",
        "build_surface_token_artifact",
        "compare_documents",
        "compare_named_corpora",
        "corpus_artifact_frequency_evidence",
        "corpus_collocation_analysis",
        "corpus_collocation_evidence",
        "corpus_collocation_evidence_for_scope",
        "corpus_collocations",
        "corpus_dependency_patterns",
        "corpus_dispersion",
        "corpus_dispersion_evidence",
        "corpus_document_frequency",
        "corpus_frequency",
        "corpus_frequency_evidence",
        "corpus_frequency_evidence_for_scope",
        "corpus_kwic",
        "corpus_kwic_evidence",
        "corpus_kwic_evidence_for_scope",
        "corpus_lemma_kwic_evidence",
        "corpus_lexical_bundle_evidence",
        "corpus_lexical_bundle_evidence_for_scope",
        "corpus_lexical_bundles",
        "corpus_ngrams",
        "corpus_normalized_frequency",
        "corpus_parsed_frequency",
        "create_code",
        "create_codebook",
        "create_comparative_finding",
        "create_corpus",
        "create_research_memo",
        "create_subcorpus",
        "export_research_bundle",
        "fetch_web_source",
        "import_document",
        "import_documents",
        "prepare_van_dijk_group_comparison",
        "record_finding_review",
        "reprocess_document",
        "research_web_context",
        "resume_controlled_cda_review",
        "review_finding_with_model",
        "run_agency_candidates",
        "run_batch_controlled_cda_review",
        "run_batch_saved_analysis_profile",
        "run_batch_textual_descriptors",
        "run_controlled_cda_review",
        "run_counterevidence_lexical",
        "run_evidence_extraction",
        "run_quote_attribution_candidates",
        "run_saved_analysis_profile",
        "run_scope_question_candidates",
        "run_social_actor_candidates",
        "run_textual_descriptors",
        "save_controlled_analysis_profile",
        "save_model_routing_profile",
        "search_exact",
        "search_hybrid",
        "search_hybrid_with_preset",
        "search_semantic_with_preset",
        "search_vector",
        "select_documents_by_metadata",
        "set_document_metadata",
        "snapshot_corpus",
        "snapshot_subcorpus",
    }
)

NON_PROJECT_READ_TOOLS = frozenset({"language_capabilities"})
EXPECTED_READ_ONLY_EXACT_SEARCH_TOOLS = frozenset({"search_exact_for_scope"})
EXPECTED_READ_ONLY_SCOPE_CORPUS_TOOLS = frozenset(
    {
        "corpus_frequency_for_scope",
        "corpus_dispersion_for_scope",
        "corpus_lexical_bundles_for_scope",
        "corpus_kwic_for_scope",
        "corpus_collocations_for_scope",
        "corpus_ngrams_for_scope",
    }
)


def test_read_only_cli_hides_registered_mutators_and_keeps_corpus_read_available(
    tmp_path, monkeypatch
):
    root = tmp_path / "read-only-project"
    project = Project.init(root)
    project.create_corpus("visible corpus")
    _, revision_id = project.import_text("source.txt", "A scoped source.")
    scope_id = project.snapshot([revision_id])
    project.close()
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    observed: dict[str, object] = {}

    async def inspect_read_only_server() -> None:
        assert mcp_module._MCP_READ_ONLY is True
        registered_names = set(mcp_module._FULL_MCP_TOOLS)
        assert registered_names == (
            REGISTERED_MUTATING_TOOLS
            | NON_PROJECT_READ_TOOLS
            | mcp_module.READ_ONLY_MCP_TOOL_NAMES
        )
        assert REGISTERED_MUTATING_TOOLS.isdisjoint(mcp_module.READ_ONLY_MCP_TOOL_NAMES)

        exposed_names = {tool.name for tool in await mcp_module.server.list_tools()}
        assert exposed_names == mcp_module.READ_ONLY_MCP_TOOL_NAMES
        assert EXPECTED_READ_ONLY_EXACT_SEARCH_TOOLS <= exposed_names
        assert EXPECTED_READ_ONLY_SCOPE_CORPUS_TOOLS <= exposed_names
        assert REGISTERED_MUTATING_TOOLS.isdisjoint(exposed_names)
        scope_tool = next(
            tool for tool in await mcp_module.server.list_tools() if tool.name == "get_scope_revisions"
        )
        scope_schema = scope_tool.model_dump(by_alias=True)["inputSchema"]
        scope_properties = scope_schema["properties"]
        assert scope_properties["offset"]["minimum"] == 0
        assert scope_properties["offset"]["maximum"] == 0x7FFF_FFFF_FFFF_FFFF
        assert scope_properties["limit"]["minimum"] == 1
        assert scope_properties["limit"]["maximum"] == 100

        for name in sorted(REGISTERED_MUTATING_TOOLS):
            with pytest.raises(Exception, match=f"Unknown tool: {name}"):
                await mcp_module.server.call_tool(name, {})

        result = await mcp_module.server.call_tool("list_corpora", {"offset": 0, "limit": 10})
        assert not result.is_error
        assert len(result.content) == 1
        payload = json.loads(result.content[0].text)
        assert payload["total"] == 1
        assert payload["items"][0]["name"] == "visible corpus"
        observed["read_result"] = payload

        scope_result = await mcp_module.server.call_tool(
            "get_scope_revisions",
            {"scope_id": scope_id, "offset": 0, "limit": 10},
        )
        assert not scope_result.is_error
        assert len(scope_result.content) == 1
        scope_payload = json.loads(scope_result.content[0].text)
        assert scope_payload["revision_ids"] == [revision_id]
        assert scope_payload["revision_count"] == 1
        assert scope_payload["has_more"] is False
        observed["scope_result"] = scope_payload

        frequency_result = await mcp_module.server.call_tool(
            "corpus_frequency_for_scope",
            {"scope_id": scope_id, "offset": 0, "limit": 10},
        )
        assert not frequency_result.is_error
        frequency_payload = json.loads(frequency_result.content[0].text)
        assert frequency_payload["scope_id"] == scope_id
        assert frequency_payload["total_tokens"] == 3
        observed["frequency_result"] = frequency_payload

        kwic_result = await mcp_module.server.call_tool(
            "corpus_kwic_for_scope",
            {"scope_id": scope_id, "query": "scoped", "window": 1},
        )
        assert not kwic_result.is_error
        kwic_payload = json.loads(kwic_result.content[0].text)
        assert kwic_payload["scope_id"] == scope_id
        assert kwic_payload["total_count"] == 1
        observed["kwic_result"] = kwic_payload

        collocations_result = await mcp_module.server.call_tool(
            "corpus_collocations_for_scope",
            {
                "scope_id": scope_id,
                "query": "scoped",
                "window": 1,
                "offset": 0,
                "limit": 10,
            },
        )
        assert not collocations_result.is_error
        collocations_payload = json.loads(collocations_result.content[0].text)
        assert collocations_payload["scope_id"] == scope_id
        assert collocations_payload["query"] == "scoped"
        assert collocations_payload["window"] == 1
        assert collocations_payload["total_values"] == 2
        observed["collocations_result"] = collocations_payload

        ngram_result = await mcp_module.server.call_tool(
            "corpus_ngrams_for_scope",
            {"scope_id": scope_id, "size": 1, "offset": 0, "limit": 10},
        )
        assert not ngram_result.is_error
        ngram_payload = json.loads(ngram_result.content[0].text)
        assert ngram_payload["scope_id"] == scope_id
        assert ngram_payload["size"] == 1
        assert ngram_payload["total_values"] == 3
        observed["ngram_result"] = ngram_payload

        exact_result = await mcp_module.server.call_tool(
            "search_exact_for_scope",
            {"scope_id": scope_id, "phrase": "scoped", "offset": 0, "limit": 10},
        )
        assert not exact_result.is_error
        exact_payload = json.loads(exact_result.content[0].text)
        assert exact_payload["scope_id"] == scope_id
        assert exact_payload["total_count"] == 1
        assert exact_payload["has_more"] is False
        assert exact_payload["rows"][0]["revision_id"] == revision_id
        observed["exact_result"] = exact_payload

    monkeypatch.setattr(mcp_module.server, "run_stdio_async", inspect_read_only_server)
    monkeypatch.setattr(
        "sys.argv", ["discourselens-mcp", "--project", str(root), "--read-only"]
    )
    mcp_module.main()

    assert observed["read_result"]
    assert observed["scope_result"]
    assert observed["frequency_result"]
    assert observed["kwic_result"]
    assert observed["collocations_result"]
    assert observed["ngram_result"]
    assert observed["exact_result"]
    assert database.read_bytes() == before
