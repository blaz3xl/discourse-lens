"""A preamble must not be concatenated with a schema-constrained action."""

import json
from io import BytesIO

import pytest

from discourselens.errors import ModelResponseError
from discourselens.providers import (
    _iter_openai_responses_structured_stream_text,
    _iter_openai_stream_text,
    _openai_responses_output_text,
)


def _message(text, phase=None):
    return {"type": "message", "role": "assistant", "phase": phase,
            "content": [{"type": "output_text", "text": text}]}


def test_completed_response_excludes_intermediate_plans():
    final = '{"kind":"tool","tool_name":"get_page","arguments":[],"final_answer":""}'
    payload = {"output": [_message("intermediate invalid JSON", "commentary"),
                          _message(final, "final_answer")]}
    assert _openai_responses_output_text(payload) == final


def test_completed_response_retains_legacy_unphased_messages():
    assert _openai_responses_output_text({"output": [_message("Hola "), _message("mundo")]}) == "Hola mundo"


def test_structured_stream_uses_only_the_completed_answer():
    final = '{"final_answer":"Una lectura razonada.","kind":"final_answer","tool_name":"","arguments":[]}'
    completed = {"type": "response.completed", "response": {
        "status": "completed", "output": [_message("Draft", "commentary"), _message(final, "final_answer")]
    }}
    events = [
        {"type": "response.output_text.delta", "delta": "Draft"}, completed,
    ]
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    assert list(_iter_openai_responses_structured_stream_text(BytesIO(body.encode()))) == [final]


def test_commentary_alone_is_not_a_completed_structured_answer():
    event = {"type": "response.completed", "response": {
        "status": "completed", "output": [_message("I will read the source.", "commentary")]
    }}
    body = "data: " + json.dumps(event) + "\n\n"
    with pytest.raises(ModelResponseError, match="no text output"):
        list(_iter_openai_responses_structured_stream_text(BytesIO(body.encode())))


def test_chat_completions_usage_trailer_is_metadata_not_a_text_delta():
    events = [
        {"choices": [{"delta": {"content": "Answer"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 80, "completion_tokens": 30}},
    ]
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
    usage = []
    assert list(_iter_openai_stream_text(BytesIO(body.encode()), on_usage=usage.append)) == ["Answer"]
    assert len(usage) == 1 and usage[0].input_tokens == 80 and usage[0].output_tokens == 30


def test_incomplete_chat_stream_does_not_invent_final_usage():
    body = b'data: {"choices":[{"delta":{"content":"Partial"},"finish_reason":null}]}\n\n'
    usage = []
    with pytest.raises(ModelResponseError):
        list(_iter_openai_stream_text(BytesIO(body), on_usage=usage.append))
    assert usage == []
