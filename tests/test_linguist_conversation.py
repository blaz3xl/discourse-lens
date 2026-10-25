"""Regressions for usable tools, source grounding and sustained discussion."""

import json
import secrets
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from discourselens.linguist import conversation_context
from discourselens.models import CredentialReference, ModelPreset
from discourselens.project import Project
from discourselens.web import (
    _CHAT_AGENT_PLAN_SCHEMA,
    _agent_mcp_catalog,
    _chat_prompt_tool_catalog,
    _chat_reading_result,
    _decode_chat_agent_plan,
    _run_chat_agent,
    _validate_agent_tool_scope,
    create_app,
)


def test_structured_actions_keep_native_values_without_json_inside_strings():
    plan = {
        "kind": "tool", "tool_name": "example", "final_answer": "",
        "arguments": [
            {"name": "question", "value": '¿Por qué dice "libertad"?'},
            {"name": "revision_ids", "value": ["one", "two"]},
            {"name": "limit", "value": 3},
            {"name": "metadata", "value": {"entries": [
                {"name": "reviewed", "value": False}, {"name": "note", "value": None}
            ]}},
        ],
    }
    jsonschema.validate(plan, _CHAT_AGENT_PLAN_SCHEMA)
    assert _decode_chat_agent_plan(json.dumps(plan), structured=True) == {
        "tool_name": "example", "arguments": {
            "question": '¿Por qué dice "libertad"?', "revision_ids": ["one", "two"],
            "limit": 3, "metadata": {"reviewed": False, "note": None},
        },
    }


def test_structured_actions_reject_duplicate_argument_names():
    plan = {"kind": "tool", "tool_name": "get_page", "final_answer": "", "arguments": [
        {"name": "page_index", "value": 0}, {"name": "page_index", "value": 4}
    ]}
    with pytest.raises(json.JSONDecodeError, match="duplicated"):
        _decode_chat_agent_plan(json.dumps(plan), structured=True)


@pytest.fixture
def research_project(tmp_path: Path):
    with Project.init(tmp_path / "project") as project:
        source = tmp_path / "programme.txt"
        source.write_text("Nosotros protegeremos a las familias.\n", encoding="utf-8")
        imported = project.import_document(source)
        project.save_credential_reference(CredentialReference("openai", "test", "DL_TEST_KEY"))
        project.save_model_preset(ModelPreset("linguist", "openai", "test", "gpt-example"))
        yield project, imported.revision_id


def test_catalog_preserves_executable_argument_schemas():
    _, catalog = _agent_mcp_catalog()
    selected = {name: catalog[name] for name in ("get_page", "run_scope_question_candidates")}
    compact = {item["name"]: item["arguments"] for item in json.loads(
        _chat_prompt_tool_catalog(selected)
    )}
    jsonschema.validate({"revision_id": "revision", "page_index": 0}, compact["get_page"])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"revision_id": "revision", "page_index": "one"}, compact["get_page"])
    assert compact["get_page"]["properties"]["max_chars"]["default"] == 8_000
    jsonschema.validate(
        {"revision_ids": ["revision"], "framework_id": "argumentation", "question": "familias"},
        compact["run_scope_question_candidates"],
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {"revision_ids": ["revision"], "framework_id": "made-up", "question": "familias"},
            compact["run_scope_question_candidates"],
        )


def test_catalog_keeps_nested_properties_and_named_definitions():
    schema = {
        "type": "object",
        "$defs": {"Choice": {"type": "string", "enum": ["a", "b"]}},
        "properties": {
            "options": {
                "type": "object",
                "properties": {"choice": {"$ref": "#/$defs/Choice"}},
                "required": ["choice"],
                "additionalProperties": False,
            }
        },
        "required": ["options"],
    }
    result = json.loads(_chat_prompt_tool_catalog({
        "example": {"name": "example", "description": "test", "arguments": schema}
    }))[0]["arguments"]
    jsonschema.validate({"options": {"choice": "a"}}, result)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"options": {"choice": "c"}}, result)


def test_recent_messages_keep_the_latest_turns_in_chronological_order(research_project):
    project, revision_id = research_project
    conversation_id = project.create_chat_conversation([revision_id], "auto", "linguist")
    for index in range(9):
        project.append_chat_turn(conversation_id, f"Question {index}", {
            "answer_interpretation": f"Interpretation {index}"
        })
    assert [item["ordinal"] for item in project.chat_messages(conversation_id, limit=4)] == [0, 1, 2, 3]
    recent = project.chat_messages(conversation_id, limit=12, recent=True)
    assert [item["ordinal"] for item in recent] == list(range(6, 18))
    context = conversation_context(recent)
    assert context[-2] == {"role": "user", "content": "Question 8"}
    assert context[-1] == {"role": "assistant", "content": "Interpretation 8"}


def test_memory_preserves_the_argument_before_large_tool_outputs():
    answer = "My objection concerns a presupposition, not a demonstrated shared belief."
    messages = [{"role": "assistant", "content": {
        "answer_interpretation": answer,
        "evidence": [{"citation": "Programme, p. 2", "excerpt": "Evidence " * 10_000}],
        "tool_results": [{
            "tool_name": "get_page", "arguments": {"revision_id": "r", "page_index": 1},
            "result": "Technical output " * 10_000,
        }],
    }}, {"role": "user", "content": {"text": "Explain that distinction."}}]
    context = conversation_context(messages)
    assert context[0]["content"].startswith(answer)
    assert '"page_index": 1' in context[0]["content"]
    assert "Technical output" not in context[0]["content"]
    assert context[-1]["content"] == "Explain that distinction."


def test_memory_is_bounded_and_prioritizes_the_newest_exchange():
    messages = [{"role": "user", "content": {"text": str(index) + "x" * 8_000}}
                for index in range(20)]
    context = conversation_context(messages)
    assert sum(len(item["content"]) for item in context) <= 32_000
    assert context[-1]["content"].startswith("19")
    assert context[0]["content"].startswith("12")


def test_indirect_passage_handles_cannot_escape_attached_sources(research_project, tmp_path):
    project, allowed_revision = research_project
    other = tmp_path / "private.txt"
    other.write_text("An unrelated private source.", encoding="utf-8")
    other_revision = project.import_document(other).revision_id
    paragraph = project.db.execute(
        "SELECT id FROM paragraphs WHERE revision_id=?", (other_revision,)
    ).fetchone()[0]
    span = project.create_span(other_revision, 0, 10)
    for arguments in ({"paragraph_id": paragraph}, {"span_id": span}):
        with pytest.raises(HTTPException) as error:
            _validate_agent_tool_scope(project, arguments, [allowed_revision])
        assert error.value.status_code == 422


def test_direct_paragraph_read_gets_a_canonical_citation(research_project):
    project, revision_id = research_project
    paragraph = dict(project.db.execute(
        "SELECT * FROM paragraphs WHERE revision_id=?", (revision_id,)
    ).fetchone())
    result = json.loads(_chat_reading_result(project, "get_paragraph", json.dumps(paragraph)))
    assert result["citation"] == "[programme.txt · ¶1]"
    assert project.revision_text(revision_id)[result["start_offset"]:result["end_offset"]] == paragraph["text"]
    assert project.db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == 0
    assert result["excerpt"] == paragraph["text"]


def test_pdf_read_uses_core_page_label_and_exact_source_span(tmp_path):
    import pymupdf

    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "We will protect families.")
    pdf.set_page_labels([{"startpage": 0, "prefix": "A-", "style": "D", "firstpagenum": 7}])
    pdf.save(tmp_path / "source.pdf")
    pdf.close()
    with Project.init(tmp_path / "pdf-project") as project:
        revision_id = project.import_document(tmp_path / "source.pdf").revision_id
        assert revision_id is not None
        page = dict(project.page(revision_id, 0))
        result = json.loads(_chat_reading_result(project, "get_page", json.dumps(page)))
        assert result["citation"] == f"[source.pdf · p. {page['page_label']}]"
        assert project.revision_text(revision_id)[result["start_offset"]:result["end_offset"]] == page["text"]
        assert project.db.execute("SELECT COUNT(*) FROM source_spans").fetchone()[0] == 0


def test_agent_supplies_source_map_and_preserves_tool_handles(research_project, monkeypatch):
    project, revision_id = research_project
    paragraph_id = project.db.execute(
        "SELECT id FROM paragraphs WHERE revision_id=?", (revision_id,)
    ).fetchone()[0]
    plans = iter([
        {"tool_name": "get_paragraph", "arguments": {"paragraph_id": paragraph_id}},
        {"final_answer": "La forma ‘protegeremos’ expresa un compromiso futuro."},
    ])
    prompts = []

    def generate(self, name, prompt, **options):
        prompts.append(prompt)
        return SimpleNamespace(response=SimpleNamespace(text=json.dumps(next(plans))))

    monkeypatch.setattr(Project, "generate_with_preset", generate)
    result = _run_chat_agent(project, project.root, "linguist", "Guíame en la lectura.",
                             [revision_id], [], False)
    assert "programme.txt" in prompts[0]
    assert '"paragraph_count": 1' in prompts[0]
    tool_results, evidence = result["tool_results"], result["evidence"]
    assert isinstance(tool_results, list) and isinstance(evidence, list)
    source = json.loads(tool_results[0]["result"])
    assert source["id"] == paragraph_id
    assert source["revision_id"] == revision_id
    assert source["revision_id"] == evidence[0]["revision_id"]
    assert evidence[0]["citation"] == "[programme.txt · ¶1]"


def test_web_followup_after_six_turns_receives_the_latest_argument(research_project, monkeypatch):
    project, revision_id = research_project
    conversation_id = project.create_chat_conversation([revision_id], "auto", "linguist")
    for index in range(8):
        project.append_chat_turn(conversation_id, f"Argument {index}", {
            "answer_interpretation": f"Reply {index}"
        })
    captured = []

    def agent(project, root, preset, question, revisions, context, embeddings, **options):
        captured.extend(context)
        return {"answer_interpretation": "Let's examine that distinction.", "evidence": []}

    monkeypatch.setattr("discourselens.web._run_chat_agent", agent)
    token = secrets.token_urlsafe(48)
    with TestClient(create_app(project.root, token)) as client:
        response = client.post(f"/api/conversations/{conversation_id}/messages", json={
            "question": "Discutamos tu última respuesta.", "model_egress_acknowledged": True,
        }, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 201
    assert captured[-2] == {"role": "user", "content": "Argument 7"}
    assert captured[-1] == {"role": "assistant", "content": "Reply 7"}
    assert not any(item["content"] == "Argument 0" for item in captured)
