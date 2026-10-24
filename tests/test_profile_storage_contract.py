import json

import pytest

from discourselens import Project
from discourselens.errors import DocumentError
from discourselens.profiles import ControlledCdaProfile


def test_save_and_read_profile_materializes_pydantic_metadata_defaults(tmp_path):
    payload: dict[str, object] = {"frameworks": ["fairclough-text"]}
    with Project.init(tmp_path / "project") as project:
        profile_id = project.save_analysis_profile("minimal", payload)

        stored_payload = json.loads(
            project.db.execute(
                "SELECT payload_json FROM analysis_profiles WHERE id=?", (profile_id,)
            ).fetchone()["payload_json"]
        )
        assert stored_payload["version"] == "1.0.0"
        assert stored_payload["kind"] == "controlled-cda"

        listed = project.list_analysis_profiles()
        assert len(listed) == 1
        assert listed[0]["profile_id"] == profile_id
        assert listed[0]["name"] == "minimal"
        assert listed[0]["version"] == "1.0.0"
        assert listed[0]["kind"] == "controlled-cda"
        assert set(listed[0]) == {"profile_id", "name", "version", "kind", "created_at"}
        saved_payload = project.analysis_profile("minimal")
        assert saved_payload == stored_payload

        profile = ControlledCdaProfile.model_validate(saved_payload)
        preflight = profile.local_preflight()
        assert profile.frameworks == ["fairclough-text"]
        assert set(preflight) == {"en", "es"}


def test_legacy_profile_without_metadata_defaults_is_normalized_on_read(tmp_path):
    payload: dict[str, object] = {"frameworks": ["fairclough-text"]}
    with Project.init(tmp_path / "project") as project:
        profile_id = project.save_analysis_profile("legacy", payload)
        project.db.execute(
            "UPDATE analysis_profiles SET payload_json=? WHERE id=?",
            (json.dumps(payload, sort_keys=True), profile_id),
        )
        project.db.commit()

        assert project.analysis_profile("legacy") == {
            "frameworks": ["fairclough-text"],
            "version": "1.0.0",
            "kind": "controlled-cda",
        }
        listed = project.list_analysis_profiles()
        assert listed[0]["version"] == "1.0.0"
        assert listed[0]["kind"] == "controlled-cda"


def test_explicit_invalid_profile_metadata_is_not_replaced_by_defaults(tmp_path):
    with Project.init(tmp_path / "project") as project:
        project.save_analysis_profile(
            "invalid", {"frameworks": ["fairclough-text"], "version": None}
        )

        with pytest.raises(DocumentError, match="analysis profile is corrupt"):
            project.list_analysis_profiles()
