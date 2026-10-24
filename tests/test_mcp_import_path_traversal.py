from __future__ import annotations

import pytest

import discourselens.mcp_server as mcp_module
from discourselens import Project
from discourselens.errors import ProjectError


@pytest.mark.parametrize("entrypoint", ["single", "batch"])
def test_mcp_import_rejects_relative_parent_traversal_outside_import_root(
    tmp_path, monkeypatch, entrypoint
):
    project_root = tmp_path / "project"
    Project.init(project_root).close()
    import_root = tmp_path / "inbox"
    import_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("must stay outside the project", encoding="utf-8")
    traversing_path = import_root / ".." / outside.name

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(project_root))
    monkeypatch.setenv("DISCOURSELENS_IMPORT_ROOT", str(import_root))
    monkeypatch.setattr(mcp_module, "_MCP_READ_ONLY", False)

    if entrypoint == "single":
        with pytest.raises(ProjectError, match="outside the configured import root"):
            mcp_module.import_document(str(traversing_path))
    else:
        result = mcp_module.import_documents([str(traversing_path)])
        assert result["success_count"] == 0
        assert result["failed_count"] == 1
        items = result["items"]
        assert isinstance(items, list)
        assert items[0]["failure_code"] == "OUTSIDE_IMPORT_ROOT"

    with Project.open(project_root, read_only=True) as project:
        assert project.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
