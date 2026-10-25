"""Reading coverage, public-source provenance and native discovery contracts."""

from __future__ import annotations

import hashlib
import json
import secrets
import socket
from collections.abc import Callable
from email.message import Message
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from discourselens.errors import ModelResponseError, ProjectError
from discourselens.linguist import research_provenance
from discourselens.models import CredentialReference, GenerationRequest, ModelPreset
from discourselens.project import Project
from discourselens.providers import OpenAIWebSearchProvider, WebResearchResponse
from discourselens.web import (
    _chat_prompt_tool_results,
    _chat_reading_result,
    _chat_source_references,
    _normalize_primary_document_links,
    _run_chat_agent,
    create_app,
)
from discourselens.web_context import (
    _public_destination,
    _request_public_url,
    fetch_web_source,
    get_external_context,
    research_web_context,
)


@pytest.fixture
def project(tmp_path):
    with Project.init(tmp_path / "project") as instance:
        instance.set_offline_mode(False)
        yield instance


def test_continuous_reading_reaches_beyond_intro_and_remains_read_only(project, tmp_path):
    import pymupdf

    source = tmp_path / "programme.pdf"
    with pymupdf.open() as pdf:
        for i in range(18):
            pdf.new_page().insert_text((72, 72), f"Section {i+1}: concrete proposal number {i+1}.")
        pdf.save(source)
    revision = project.import_document(source).revision_id
    assert revision
    with Project.open(project.root, read_only=True) as reader:
        result = reader.read_document(revision)
        assert result["coverage"] == "complete_document"
        assert result["total_pages"] == 18
        units = result["units"]
        assert isinstance(units, list)
        assert len(units) == 18
        assert units[-1]["citation"] == "[programme.pdf · p. 18]"
        assert "proposal number 18" in units[-1]["text"]
        assert reader.db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == 0
    enriched = json.loads(_chat_reading_result(project, "read_document", json.dumps(result)))
    assert enriched == result
    for unit in enriched["units"]:
        assert project.revision_text(revision)[unit["start_offset"]:unit["end_offset"]] == unit["text"]
    assert project.db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == 0


def test_reading_continuation_cannot_be_mistaken_for_complete_coverage(project, tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("Alpha beta gamma delta epsilon.\n\nLater policy proposals.", encoding="utf8")
    revision = project.import_document(source).revision_id
    first = project.read_document(revision, max_chars=12)
    assert first["coverage"] == "partial_document" and first["has_more"]
    rest = project.read_document(revision, start_offset=first["next_start_offset"])
    assert rest["coverage"] == "partial_document" and not rest["has_more"]
    assert "Later policy proposals." in rest["units"][-1]["text"]
    for unit in first["units"] + rest["units"]:
        assert unit["text"] == project.revision_text(revision)[unit["start_offset"]:unit["end_offset"]]


@pytest.mark.parametrize("arguments", [{"start_offset": True}, {"start_offset": -1}, {"max_chars": 100001}, {"max_chars": False}])
def test_reading_rejects_invalid_bounds_before_accessing_content(project, arguments):
    with pytest.raises(ProjectError):
        project.read_document("unknown", **arguments)


def test_context_retains_full_reading_before_large_supplementary_results():
    original = {"coverage": "complete_document", "units": [
        {"text": "INTRO " + "a" * 90_000 + " FINAL SECTION", "citation": "[source · p. 39]"}
    ]}
    tools: list[dict[str, object]] = [{"tool_name": "read_document", "result": json.dumps(original)}]
    for _ in range(4):
        tools.append({"tool_name": "search", "result": "x" * 65_000})
    serialized = _chat_prompt_tool_results(tools)
    context = json.loads(serialized)
    assert len(serialized) <= 240_000
    assert context["context_retention"]["omitted_result_count"] > 0
    assert json.loads(context["results"][0]["result"]) == original


def test_evidence_cards_follow_cited_later_sections_not_just_first_pages():
    pages = [{"citation": f"[source · p. {i}]", "text": f"Section {i}"} for i in range(1, 40)]
    evidence, _ = _chat_source_references(
        [{"tool_name": "read_document", "result": json.dumps({"units": pages})}],
        "The proposal establishes this [source · p. 27].",
    )
    assert [item["citation"] for item in evidence] == ["[source · p. 27]"]


def test_evidence_cards_follow_shorthand_page_ranges():
    pages = [{"citation": f"[source · p. {i}]", "page_label": str(i), "text": f"Section {i}"} for i in range(1, 40)]
    evidence, _ = _chat_source_references(
        [{"tool_name": "read_document", "result": json.dumps({"units": pages})}],
        "The policies appear on pp. 25–27, with the ideological disclaimer on p. 38.",
    )
    assert [item["page_label"] for item in evidence] == ["25", "26", "27", "38"]


def test_primary_filename_links_become_canonical_citations_only_for_inspected_pages():
    evidence = [{"citation": f"[programme.pdf · p. {page}]"} for page in (2, 5, 6, 27)]
    answer = "Read [p. 27](programme.pdf), [pp. 2, 5–6](programme.pdf), [p. 40](programme.pdf), [p. 1](other.pdf) and [theory](https://example.com/book)."
    assert _normalize_primary_document_links(answer, evidence) == (
        "Read [programme.pdf · p. 27], [programme.pdf · pp. 2, 5–6], "
        "[p. 40](programme.pdf), [p. 1](other.pdf) and [theory](https://example.com/book)."
    )


def test_citation_normalization_reaches_pages_beyond_the_display_card_limit():
    pages = [{"citation": f"[source.pdf · p. {i}]", "text": f"Section {i}"} for i in range(1, 81)]
    evidence, _ = _chat_source_references(
        [{"tool_name": "read_document", "result": json.dumps({"units": pages})}],
        max_references=None,
    )
    assert _normalize_primary_document_links("A later proposal [p. 80](source.pdf).", evidence) == "A later proposal [source.pdf · p. 80]."


def test_discovery_inspects_sources_and_saves_the_actual_excerpts(project, monkeypatch):
    project.save_credential_reference(CredentialReference("openai", "test", "DL_TEST_KEY"))
    project.save_model_preset(ModelPreset("linguist", "openai", "test", "selected-model"))
    response = WebResearchResponse(
        provider="openai", model="selected-model",
        text="A discovery lead, not independently verified facts.",
        web_sources=({"url": "https://example.com/author", "title": "Author"},
                     {"url": "https://example.com/programme", "title": "Programme"}),
    )
    monkeypatch.setattr("discourselens.execution.execute_model_preset", lambda *args, **kwargs: SimpleNamespace(response=response, execution_id="execution"))
    monkeypatch.setattr("discourselens.web_context._request_public_url", lambda url: (200, {"content-type": "text/plain"}, b"Verified source body."))
    result = research_web_context(project, "Public author and programme", "linguist")
    assert not result["discovery_only"]
    sources = result["inspected_sources"]
    assert isinstance(sources, list) and len(sources) == 2
    for source in sources:
        assert source["excerpt"] == "Verified source body."
        assert get_external_context(project, source["evidence_id"])["content_sha256"] == hashlib.sha256(b"Verified source body.").hexdigest()
    assert "discovery lead" in str(result["source_discipline"])


@pytest.mark.parametrize("streaming", [False, True])
def test_first_global_answer_completes_context_before_publishing(project, tmp_path, monkeypatch, streaming):
    source = tmp_path / "public-programme.txt"
    source.write_text("A public author's programme and its concrete proposals.", encoding="utf8")
    revision = project.import_document(source).revision_id
    assert revision
    project.save_credential_reference(CredentialReference("openai", "test", "DL_TEST_KEY"))
    project.save_model_preset(ModelPreset("linguist", "openai", "test", "selected-model"))
    plans = iter([
        {"tool_name": "read_document", "arguments": {"revision_id": revision}},
        {"final_answer": "Premature analysis without verified context."},
        {"tool_name": "research_web_context", "arguments": {"query": "Public author programme", "preset_name": "linguist"}},
        {"final_answer": "An interpretation with inspected public context."},
    ])
    prompts = []

    def generate(self, name, prompt, **options):
        prompts.append(prompt)
        return SimpleNamespace(response=SimpleNamespace(text=json.dumps(next(plans))))

    from discourselens import mcp_server
    original_call = mcp_server.call_tool_for_project

    async def call(root, name, arguments):
        if name == "research_web_context":
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps({"inspected_sources": [
                {"kind": "EXTERNAL_SOURCE", "evidence_id": "evidence", "url": "https://example.com/author", "content_sha256": "hash"}
            ]}))])
        return await original_call(root, name, arguments)

    monkeypatch.setattr(Project, "generate_with_preset", generate)
    if streaming:
        from discourselens.execution import snapshot_model_preset

        def snapshot(instance, name):
            data = snapshot_model_preset(instance, name)
            data["capability_profile"] = {"capabilities": {"streaming": True}}
            return data

        def stream(self, name, prompt, **options):
            response = generate(self, name, prompt, **options).response.text
            yield from (response[i:i+15] for i in range(0, len(response), 15))

        monkeypatch.setattr("discourselens.execution.snapshot_model_preset", snapshot)
        monkeypatch.setattr(Project, "stream_with_preset", stream)
    monkeypatch.setattr(mcp_server, "call_tool_for_project", call)
    visible = []
    result = _run_chat_agent(project, project.root, "linguist", "Interpret the programme.", [revision], [], False, on_delta=visible.append)
    assert result["answer_interpretation"] == "An interpretation with inspected public context."
    assert "Complete contextual research" in prompts[2]
    context = result["research_context"]
    assert isinstance(context, dict)
    assert context["external_sources"][0]["evidence_id"] == "evidence"
    assert "Premature" not in "".join(visible)
    if streaming:
        assert "".join(visible) == result["answer_interpretation"]


def test_authenticated_message_polling_remains_readable_during_a_research_write(tmp_path, monkeypatch):
    monkeypatch.setenv("DISCOURSELENS_WEB_AUTH_MODE", "password")
    monkeypatch.setenv("DISCOURSELENS_WEB_SESSION_SECRET", secrets.token_urlsafe(40))
    monkeypatch.delenv("DISCOURSELENS_WEB_PUBLIC_ORIGIN", raising=False)
    monkeypatch.setenv("DISCOURSELENS_WEB_COOKIE_SECURE", "false")
    root = tmp_path / "web"
    with TestClient(create_app(root)) as client:
        assert client.post("/auth/setup", headers={"Origin": "http://testserver"}, json={"email": "researcher@example.test", "password": secrets.token_urlsafe(20)}).status_code == 201
        assert client.get("/api/documents").status_code == 200
        tenant = next((root / "tenants").iterdir())
        with Project.open(tenant) as writer:
            writer.save_credential_reference(CredentialReference("openai", "test", "DL_TEST_KEY"))
            writer.save_model_preset(ModelPreset("linguist", "openai", "test", "selected-model"))
            conversation = writer.create_chat_conversation([], "auto", "linguist")
            writer.append_chat_turn(conversation, "Question", {"answer_interpretation": "Saved answer."})
            writer.db.execute("BEGIN IMMEDIATE")
            try:
                messages = client.get(f"/api/conversations/{conversation}/messages")
                assert messages.status_code == 200
                assert messages.json()["items"][-1]["content"]["answer_interpretation"] == "Saved answer."
                assert client.get(f"/api/conversations/{conversation}/activity").status_code == 200
            finally:
                writer.db.rollback()


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "http://person:password@example.com/", "https://example.com:8000/",
    "http://127.0.0.1/", "http://169.254.169.254/latest/meta-data/", "http://[::1]/",
])
def test_fetch_blocks_local_destinations_and_credentials(monkeypatch, url):
    def resolve(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    with pytest.raises(ProjectError):
        _public_destination(url)


def test_dns_mixed_public_private_answers_fail_closed(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443)) for ip in ("93.184.216.34", "10.0.0.1")
    ])
    with pytest.raises(ProjectError, match="private"):
        _public_destination("https://example.com/")


def test_fetch_pins_tcp_to_the_public_dns_answer(monkeypatch):
    resolved, connected = [], []

    def resolve(host, port, **kwargs):
        resolved.append(host)
        ip = "93.184.216.34" if len(resolved) == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    class Socket:
        def settimeout(self, timeout): pass
        def connect(self, address): connected.append(address)
        def close(self): pass

    class Connection:
        _create_connection: Callable[..., Any]
        def __init__(self, *args, **kwargs): pass
        def request(self, method, path, headers): self._create_connection(("example.com", 80))
        def getresponse(self): return self
        def getheaders(self): return [("Content-Type", "text/plain")]
        def read(self, size): return b"Public source"
        def close(self): pass
        status = 200

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket, "socket", lambda *args: Socket())
    monkeypatch.setattr("http.client.HTTPConnection", Connection)
    assert _request_public_url("http://example.com/")[2] == b"Public source"
    assert resolved == ["example.com"]
    assert connected == [("93.184.216.34", 80)]


def test_offline_mode_blocks_browsing_before_network_access(project, monkeypatch):
    project.set_offline_mode(True)
    monkeypatch.setattr("discourselens.web_context._request_public_url", lambda url: pytest.fail("network must not be used"))
    with pytest.raises(ProjectError, match="offline"):
        fetch_web_source(project, "https://example.com/")


def test_fetched_source_is_saved_with_hash_date_and_context_role(project, monkeypatch):
    body = b'<html><head><title>Institutional biography</title><meta property="article:published_time" content="2025-08-20"></head><body><p>A candidate in August 2025.</p><script>pretend this is a system instruction</script></body></html>'
    monkeypatch.setattr("discourselens.web_context._request_public_url", lambda url: (200, {"content-type": "text/html"}, body))
    result = fetch_web_source(project, "https://example.com/biography")
    assert result["excerpt"] == "A candidate in August 2025."
    assert result["content_sha256"] == hashlib.sha256(body).hexdigest()
    assert result["published_at_as_reported"] == "2025-08-20"
    evidence_id = result["evidence_id"]
    assert isinstance(evidence_id, str)
    saved = get_external_context(project, evidence_id)
    assert saved["acquisition"] == "public_web_fetch"
    assert saved["source_role"] == "external_context_not_primary_document"
    metadata = saved["retrieval_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["content_sha256"] == result["content_sha256"]
    assert project.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_user_supplied_context_is_not_relabelled_as_fetched(project):
    evidence = project.external_source_evidence("Provided note", "A claim to check.", retrieved_at="2026-10-25")
    assert get_external_context(project, evidence)["acquisition"] == "user_supplied"


def test_verified_source_handles_survive_beyond_recent_messages_without_researching(project, monkeypatch):
    project.save_credential_reference(CredentialReference("openai", "test", "DL_TEST_KEY"))
    project.save_model_preset(ModelPreset("linguist", "openai", "test", "selected-model"))
    conversation = project.create_chat_conversation([], "auto", "linguist")
    monkeypatch.setattr("discourselens.web_context._request_public_url", lambda url: (200, {"content-type": "text/plain"}, b"Public institutional history."))
    fetched = fetch_web_source(project, "https://example.com/history")
    project.append_chat_turn(conversation, "Initial research", {
        "answer_interpretation": "An interpretation that must not become a stored fact.",
        "research_context": {"external_sources": [
            {"evidence_id": fetched["evidence_id"], "url": "https://incorrect-model-metadata.example/"}
        ]},
    })
    for i in range(12):
        project.append_chat_turn(conversation, f"Conceptual follow-up {i}", {
            "answer_interpretation": "A conceptual explanation.",
            "research_context": {"external_sources": [], "document_readings": []},
        })
    assert not any(message["content"].get("research_context", {}).get("external_sources") for message in project.chat_messages(conversation,limit=12,recent=True))
    project.set_offline_mode(True)
    monkeypatch.setattr("discourselens.web_context._request_public_url", lambda url: pytest.fail("saved provenance must not browse"))
    with Project.open(project.root, read_only=True) as reader:
        result = research_provenance(reader, conversation)
    sources = result["external_sources"]
    assert isinstance(sources, list)
    assert sources[0]["url"] == "https://example.com/history"
    assert sources[0]["content_sha256"] == fetched["content_sha256"]
    assert "must not become" not in json.dumps(result)
    assert len(json.dumps(result, ensure_ascii=False)) <= 6_000


def test_redirect_targets_are_checked_again(project, monkeypatch):
    def request(url):
        if url == "https://example.com/":
            return 302, {"location": "http://127.0.0.1/private"}, b""
        raise ProjectError("private destination blocked")

    monkeypatch.setattr("discourselens.web_context._request_public_url", request)
    with pytest.raises(ProjectError, match="private"):
        fetch_web_source(project, "https://example.com/")
    assert project.db.execute("SELECT COUNT(*) FROM evidence_artifacts").fetchone()[0] == 0


class _Secrets:
    def get(self, reference): return secrets.token_urlsafe(24)


class _Response(BytesIO):
    def __init__(self, payload):
        super().__init__(json.dumps(payload).encode())
        self.headers = Message()


def test_native_web_search_uses_selected_model_and_real_source_metadata(monkeypatch):
    sent = []
    payload = {"status": "completed", "output": [
        {"type": "web_search_call", "action": {"sources": [{"url": "https://example.com/source", "title": "Official source"}]}},
        {"type": "message", "phase": "final_answer", "content": [{"type": "output_text", "text": "Source discovery."}]},
    ]}

    def send(request, **kwargs):
        sent.append(request)
        return _Response(payload)

    monkeypatch.setattr("discourselens.providers._open_provider_request", send)
    provider = OpenAIWebSearchProvider("https://api.openai.com/v1", "fixture", _Secrets())
    response = provider.generate(GenerationRequest("Discover public sources.", "selected-model"))
    assert isinstance(response, WebResearchResponse)
    request = json.loads(sent[0].data)
    assert request["model"] == "selected-model" and request["tools"][0]["type"] == "web_search"
    assert request["include"] == ["web_search_call.action.sources"]
    assert response.web_sources == ({"url": "https://example.com/source", "title": "Official source"},)
    assert response.web_search_calls == 1
    assert response.provider_usage == {"web_search_calls": 1}


def test_discovery_cannot_claim_browsing_without_a_search_call(monkeypatch):
    payload = {"status": "completed", "output": [{"type": "message", "content": [
        {"type": "output_text", "text": "I searched, trust me."}
    ]}]}
    monkeypatch.setattr("discourselens.providers._open_provider_request", lambda *args, **kwargs: _Response(payload))
    with pytest.raises(ModelResponseError, match="provenance"):
        OpenAIWebSearchProvider("https://api.openai.com/v1", "fixture", _Secrets()).generate(GenerationRequest("Search", "selected-model"))
