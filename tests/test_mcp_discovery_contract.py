from __future__ import annotations

import asyncio
import json

from discourselens import Project
from discourselens.mcp_server import (
    READ_ONLY_MCP_TOOL_NAMES,
    get_analysis_profile,
    list_analysis_profiles,
    project_summary_resource,
    server,
)


def test_mcp_discovery_surfaces_and_read_schemas_are_registered() -> None:
    resources = asyncio.run(server.list_resources())
    templates = asyncio.run(server.list_resource_templates())
    prompts = asyncio.run(server.list_prompts())
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}

    assert {resource.uri for resource in resources} == {
        "discourselens://methodology/epistemic-policy",
        "discourselens://project/summary",
    }
    assert {template.uri_template for template in templates} == {
        "discourselens://skills/{skill_id}/methodology",
        "discourselens://findings/{finding_id}",
    }
    assert {
        "critical-evidence-review",
        "full-cda",
        "social-actor-analysis",
        "dha-analysis",
        "corpus-comparison",
        "critical-review-finding",
    } == {prompt.name for prompt in prompts}

    discovery_names = {
        "list_skills",
        "get_skill",
        "list_analysis_profiles",
        "get_analysis_profile",
    }
    assert discovery_names <= tools.keys()
    assert discovery_names <= READ_ONLY_MCP_TOOL_NAMES
    assert not any(name.startswith("run_") for name in discovery_names)

    schemas = {
        name: tool.model_dump(by_alias=True)["inputSchema"]
        for name, tool in tools.items()
        if name in discovery_names
    }
    assert schemas["list_skills"]["properties"] == {}
    assert schemas["list_analysis_profiles"]["properties"] == {}
    assert schemas["get_skill"]["required"] == ["skill_id"]
    assert set(schemas["get_skill"]["properties"]) == {"skill_id"}
    assert schemas["get_analysis_profile"]["required"] == ["name"]
    assert set(schemas["get_analysis_profile"]["properties"]) == {"name"}


def test_project_summary_is_counts_only_and_does_not_publish_corpus_text(
    tmp_path, monkeypatch
) -> None:
    marker = "PRIVATE_DISCOVERY_FIXTURE_9a4b"
    root = tmp_path / "research"
    project = Project.init(root)
    _, _ = project.import_text(f"{marker}.txt", f"corpus text {marker}")
    project.create_corpus(f"{marker} corpus")
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    response = project_summary_resource()

    assert json.loads(response) == {"documents": 1, "revisions": 1, "corpora": 1}
    assert marker not in response


def test_profile_discovery_returns_metadata_and_reads_one_declared_profile(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "profile-project"
    project = Project.init(root)
    profile_id = project.save_analysis_profile(
        "bounded-review",
        {
            "version": "1.0.0",
            "kind": "controlled-cda",
            "frameworks": ["fairclough-text"],
        },
    )
    project.close()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))

    listing = list_analysis_profiles()
    selected = get_analysis_profile("bounded-review")

    assert len(listing) == 1
    assert listing[0]["profile_id"] == profile_id
    assert listing[0]["name"] == "bounded-review"
    assert listing[0]["version"] == "1.0.0"
    assert listing[0]["kind"] == "controlled-cda"
    assert set(listing[0]) == {"profile_id", "name", "version", "kind", "created_at"}
    assert selected == {
        "version": "1.0.0",
        "kind": "controlled-cda",
        "frameworks": ["fairclough-text"],
    }
