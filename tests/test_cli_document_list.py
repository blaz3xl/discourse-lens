from __future__ import annotations

import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def _create_document_history(root):
    project = Project.init(root)
    alpha_document, alpha_revision = project.import_text("alpha.txt", "Alpha secret source contents.")
    zeta_document, old_zeta_revision = project.import_text("zeta.txt", "Old zeta secret contents.")
    _, latest_zeta_revision = project.import_text(
        "zeta.txt", "Latest zeta secret contents.", document_id=zeta_document
    )
    project.close()
    return alpha_document, alpha_revision, zeta_document, old_zeta_revision, latest_zeta_revision


def test_document_list_is_bounded_ordered_metadata_with_latest_revision(tmp_path):
    root = tmp_path / "document-list-project"
    _, alpha_revision, _, old_zeta_revision, latest_zeta_revision = _create_document_history(
        root
    )
    runner = CliRunner()

    default_page = runner.invoke(app, ["document", "list", str(root)])
    explicit_page = runner.invoke(
        app,
        ["document", "list", str(root), "--offset", "1", "--limit", "1"],
    )

    assert default_page.exit_code == 0, default_page.output
    assert explicit_page.exit_code == 0, explicit_page.output
    payload = json.loads(default_page.output)
    assert payload["offset"] == 0
    assert payload["limit"] == 100
    assert payload["total"] == 2
    assert payload["has_more"] is False
    items = payload["items"]
    assert [item["display_name"] for item in items] == ["alpha.txt", "zeta.txt"]
    assert items[0]["latest_revision_id"] == alpha_revision
    assert items[1]["latest_revision_id"] == latest_zeta_revision
    assert old_zeta_revision not in {item["latest_revision_id"] for item in items}
    assert all("text" not in item and "path" not in item for item in items)
    assert "secret source contents" not in default_page.output
    assert "Old zeta secret contents" not in default_page.output
    assert "Latest zeta secret contents" not in default_page.output
    assert str(root) not in default_page.output

    page = json.loads(explicit_page.output)
    assert page["items"] == items[1:2]
    assert page["offset"] == 1
    assert page["limit"] == 1
    assert page["total"] == 2
    assert page["has_more"] is False


def test_document_list_bounds_offset_and_limit(tmp_path):
    root = tmp_path / "document-list-bounds"
    _create_document_history(root)
    runner = CliRunner()

    invalid_offset = runner.invoke(app, ["document", "list", str(root), "--offset", "-1"])
    excessive_offset = runner.invoke(
        app,
        ["document", "list", str(root), "--offset", "9223372036854775808"],
    )
    zero_limit = runner.invoke(app, ["document", "list", str(root), "--limit", "0"])
    excessive_limit = runner.invoke(app, ["document", "list", str(root), "--limit", "1001"])
    max_offset = runner.invoke(
        app,
        ["document", "list", str(root), "--offset", "9223372036854775807"],
    )

    assert invalid_offset.exit_code == 2
    assert excessive_offset.exit_code == 2
    assert zero_limit.exit_code == 2
    assert excessive_limit.exit_code == 2
    assert max_offset.exit_code == 0, max_offset.output
    payload = json.loads(max_offset.output)
    assert payload["offset"] == 0x7FFF_FFFF_FFFF_FFFF
    assert payload["limit"] == 100
    assert payload["total"] == 2
    assert payload["items"] == []
    assert payload["has_more"] is False
