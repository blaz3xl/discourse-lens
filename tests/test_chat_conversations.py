from pathlib import Path

import pytest

from discourselens.errors import DocumentError
from discourselens.migrations import SCHEMA_VERSION
from discourselens.models import CredentialReference, ModelPreset
from discourselens.project import Project


def _project_with_revision(root: Path) -> tuple[Project, str]:
    project = Project.init(root)
    source = root.parent / "chat-source.txt"
    source.write_text("A minister described the reform as necessary.\n", encoding="utf-8")
    revision_id = project.import_document(source).revision_id
    assert revision_id is not None
    project.save_credential_reference(CredentialReference("openai", "fixture", "DL_CHAT_KEY"))
    project.save_model_preset(ModelPreset("chat-model", "openai", "fixture", "gpt-example"))
    return project, revision_id


def test_conversation_scope_messages_and_delete_are_persisted(tmp_path: Path):
    project, revision_id = _project_with_revision(tmp_path / "project")
    try:
        conversation_id = project.create_chat_conversation(
            [revision_id], "fairclough-text", "chat-model"
        )
        conversation = project.chat_conversation(conversation_id)
        assert conversation["revision_ids"] == [revision_id]
        assert conversation["framework_id"] == "fairclough-text"
        assert conversation["preset_name"] == "chat-model"
        assert project.list_chat_conversations()[0]["id"] == conversation_id

        answer = {
            "status": "answered",
            "question": "Why is the reform called necessary?",
            "framework_id": "fairclough-text",
            "candidate_analysis_run_id": "candidate-run",
            "candidate_finding_id": "candidate-finding",
            "answer_analysis_run_id": "answer-run",
            "answer_finding_id": "answer-finding",
            "answer_interpretation": "The wording presents necessity as given.",
            "alternative_interpretation": "It may refer to a documented constraint.",
            "evidence": [],
        }
        user_id, assistant_id = project.append_chat_turn(conversation_id, answer["question"], answer)
        messages = project.chat_messages(conversation_id)
        assert [item["id"] for item in messages] == [user_id, assistant_id]
        assert [item["role"] for item in messages] == ["user", "assistant"]
        assert messages[0]["content"] == {"text": answer["question"]}
        assistant_content = messages[1]["content"]
        assert isinstance(assistant_content, dict)
        assert assistant_content["answer_interpretation"] == answer["answer_interpretation"]
        assert messages[1]["analysis_run_id"] == "answer-run"
        assert project.chat_conversation(conversation_id)["title"] == answer["question"]

        project.delete_chat_conversation(conversation_id)
        assert project.list_chat_conversations() == []
        with pytest.raises(DocumentError, match="unknown chat conversation"):
            project.chat_conversation(conversation_id)
    finally:
        project.close()


def test_chat_turn_rejects_oversized_or_unserializable_messages(tmp_path: Path):
    project, revision_id = _project_with_revision(tmp_path / "project")
    try:
        conversation_id = project.create_chat_conversation(
            [revision_id], "fairclough-text", "chat-model"
        )
        with pytest.raises(DocumentError, match="8000"):
            project.append_chat_turn(conversation_id, "x" * 8_001, {})
        with pytest.raises(ValueError):
            project.append_chat_turn(conversation_id, "valid question", {"value": float("nan")})
        assert project.chat_messages(conversation_id) == []
    finally:
        project.close()


def test_hybrid_conversation_rejects_non_string_artifact_ids(tmp_path: Path):
    project, revision_id = _project_with_revision(tmp_path / "project")
    try:
        with pytest.raises(DocumentError, match="unique embedding artifact IDs"):
            project.create_chat_conversation(
                [revision_id],
                "fairclough-text",
                "chat-model",
                retrieval_mode="hybrid",
                embedding_preset_name="embed",
                embedding_artifact_ids=[{}],  # type: ignore[list-item]
            )
    finally:
        project.close()


def test_schema_v50_migrates_to_chat_conversations(tmp_path: Path):
    project = Project.init(tmp_path / "project")
    project.db.execute("DROP TABLE chat_messages")
    project.db.execute("DROP TABLE chat_conversations")
    project.db.execute("DELETE FROM schema_migrations WHERE version=?", (SCHEMA_VERSION,))
    project.db.execute("INSERT INTO schema_migrations(version,applied_at) VALUES(50,'test')")
    project.db.commit()
    project.close()

    migrated = Project.open(tmp_path / "project")
    try:
        version = migrated.db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        assert version == SCHEMA_VERSION
        tables = {
            row[0]
            for row in migrated.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"chat_conversations", "chat_messages"} <= tables
    finally:
        migrated.close()
