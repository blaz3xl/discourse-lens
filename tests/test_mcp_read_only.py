from __future__ import annotations

import asyncio
import sqlite3
from typing import cast

import pytest

from discourselens import Project, mcp_server
from discourselens.errors import ProjectError
from discourselens.migrations import SCHEMA_VERSION
from discourselens.versioning import software_version


def test_mcp_version_is_available_without_project_argument(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["discourselens-mcp", "--version"])

    with pytest.raises(SystemExit) as exit_info:
        mcp_server.main()

    captured = capsys.readouterr()
    assert exit_info.value.code == 0
    assert captured.out == f"DiscourseLens MCP {software_version()}\n"
    assert captured.err == ""


def test_read_only_project_rejects_database_and_file_mutations(tmp_path):
    root = tmp_path / "project"
    writable = Project.init(root)
    project_id = writable.project_id
    writable.close()

    project = Project.open(root, read_only=True)
    try:
        assert project.read_only is True
        assert project.project_id == project_id
        assert project.db.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            project.db.execute("INSERT INTO project_meta(key,value) VALUES('blocked','true')")
        with pytest.raises(ProjectError, match="read-only"):
            project.set_offline_mode(True)
        with pytest.raises(ProjectError, match="read-only"):
            project.import_text("blocked.txt", "must not be written")
        assert not list((root / ".discourselens" / "source").glob("*"))
    finally:
        project.close()


def test_read_only_project_does_not_migrate_an_outdated_schema(tmp_path):
    root = tmp_path / "project"
    project = Project.init(root)
    project.db.execute("UPDATE schema_migrations SET version=?", (SCHEMA_VERSION - 1,))
    project.db.commit()
    project.close()

    with pytest.raises(ProjectError, match="requires schema version"):
        Project.open(root, read_only=True)

    db = sqlite3.connect(
        f"{(root / '.discourselens' / 'project.sqlite').as_uri()}?mode=ro", uri=True
    )
    try:
        assert db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == (
            SCHEMA_VERSION - 1
        )
    finally:
        db.close()


def test_read_only_research_memo_listing_is_paged_and_bounds_text(tmp_path, monkeypatch):
    root = tmp_path / "project"
    project = Project.init(root)
    _, revision_id = project.import_text("source.txt", "A sentence.")
    scope_id = project.snapshot([revision_id])
    memo_content = "x" * (mcp_server.MAX_RESEARCH_MEMO_TEXT_CHARS + 1_000)
    memo_title = "t" * (mcp_server.MAX_RESEARCH_MEMO_TITLE_CHARS + 20)
    for index in range(mcp_server.MAX_RESEARCH_MEMO_PAGE_SIZE + 1):
        project.create_research_memo(scope_id, f"{memo_title}-{index}", memo_content)
    project.close()

    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    previous_read_only = mcp_server._MCP_READ_ONLY
    mcp_server._MCP_READ_ONLY = True
    try:
        first_page = mcp_server.list_research_memos(scope_id)
        second_page = mcp_server.list_research_memos(
            scope_id, offset=mcp_server.MAX_RESEARCH_MEMO_PAGE_SIZE
        )
    finally:
        mcp_server._MCP_READ_ONLY = previous_read_only

    assert (
        first_page["total_count"]
        == second_page["total_count"]
        == (mcp_server.MAX_RESEARCH_MEMO_PAGE_SIZE + 1)
    )
    assert first_page["has_more"] is True
    assert second_page["has_more"] is False
    first_rows = cast(list[dict[str, object]], first_page["rows"])
    second_rows = cast(list[dict[str, object]], second_page["rows"])
    assert len(first_rows) == mcp_server.MAX_RESEARCH_MEMO_PAGE_SIZE
    assert len(second_rows) == 1
    for memo in first_rows + second_rows:
        content = memo["content"]
        content_characters = memo["content_characters"]
        title = memo["title"]
        title_characters = memo["title_characters"]
        assert isinstance(content, str)
        assert isinstance(content_characters, int)
        assert isinstance(title, str)
        assert isinstance(title_characters, int)
        assert len(content) == mcp_server.MAX_RESEARCH_MEMO_TEXT_CHARS
        assert content_characters == len(memo_content)
        assert memo["content_truncated"] is True
        assert len(title) == mcp_server.MAX_RESEARCH_MEMO_TITLE_CHARS
        assert title_characters > len(title)
        assert memo["title_truncated"] is True
    assert database.read_bytes() == before


def test_mcp_read_only_finding_views_do_not_lazily_write_pdf_citations(tmp_path, monkeypatch):
    import json

    import pymupdf as fitz

    from discourselens.documents import import_pdf
    from discourselens.types import EvidenceRole

    root = tmp_path / "project"
    source = tmp_path / "paper.pdf"
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Persisted PDF evidence.")
    pdf.save(source)
    pdf.close()

    project = Project.init(root)
    imported = import_pdf(project, source)
    assert imported.revision_id is not None
    scope_id = project.snapshot([imported.revision_id])
    run_id = project.create_run(scope_id, {"fixture": "read-only-resource"})
    finding_id = project.create_finding(
        run_id, "fixture.method", "1.0.0", "agency-backgrounding", "Observed evidence."
    )
    span_id = project.create_spans(
        imported.revision_id, [(0, len("Persisted PDF"))], project_segments=False
    )[0]
    evidence_id = project.textual_evidence(span_id)
    project.attach_evidence(finding_id, evidence_id, EvidenceRole.SUPPORT)
    codebook_id = project.create_codebook("read-only-citations")
    code_id = project.create_code(codebook_id, "persisted quote")
    project.annotate_span(code_id, span_id)
    assert project.db.execute("SELECT COUNT(*) FROM source_span_segments").fetchone()[0] == 0
    project.close()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    previous_read_only = mcp_server._MCP_READ_ONLY
    mcp_server._MCP_READ_ONLY = True
    try:
        finding = json.loads(mcp_server.finding_resource(finding_id))
        report = mcp_server.get_analysis_report(run_id, output_format="json")
    finally:
        mcp_server._MCP_READ_ONLY = previous_read_only

    assert finding["evidence"][0]["citation"] == "[paper.pdf · p. 1]"
    assert report["format"] == "json"
    report_payload = report["report"]
    assert isinstance(report_payload, dict)
    qualitative_coding = report_payload["qualitative_coding"]
    assert isinstance(qualitative_coding, dict)
    annotations = qualitative_coding["annotations"]
    assert isinstance(annotations, list) and annotations
    assert isinstance(annotations[0], dict)
    assert annotations[0]["citation"] == "[paper.pdf · p. 1]"
    with Project.open(root, read_only=True) as unchanged:
        assert unchanged.db.execute("SELECT COUNT(*) FROM source_span_segments").fetchone()[0] == 0


def test_mcp_read_only_then_writable_start_restores_full_server_surface(monkeypatch, tmp_path):
    root = tmp_path / "project"
    Project.init(root).close()
    custom_import_root = tmp_path / "custom-import-root"
    custom_import_root.mkdir()

    monkeypatch.setenv("DISCOURSELENS_PROJECT", str(root))
    monkeypatch.delenv("DISCOURSELENS_IMPORT_ROOT", raising=False)
    observed: list[tuple[bool, set[str], set[str]]] = []
    import_roots: list[str | None] = []

    async def fake_stdio() -> None:
        read_only = mcp_server._MCP_READ_ONLY
        import_roots.append(mcp_server.os.environ.get("DISCOURSELENS_IMPORT_ROOT"))
        opened = mcp_server._open_project()
        try:
            assert opened.read_only is read_only
            assert opened.project_id
        finally:
            opened.close()
        tools = {tool.name for tool in await mcp_server.server.list_tools()}
        prompts = {prompt.name for prompt in await mcp_server.server.list_prompts()}
        observed.append((read_only, tools, prompts))
        if read_only:
            assert tools == mcp_server.READ_ONLY_MCP_TOOL_NAMES
            assert prompts == set()
            with pytest.raises(Exception, match="Unknown tool: import_document"):
                await mcp_server.server.call_tool("import_document", {})

    monkeypatch.setattr(mcp_server.server, "run_stdio_async", fake_stdio)

    monkeypatch.setattr(
        "sys.argv",
        [
            "discourselens-mcp",
            "--project",
            str(root),
            "--import-root",
            str(custom_import_root),
            "--read-only",
        ],
    )
    mcp_server.main()

    # Shutdown must also leave the singleton reusable before any writable
    # restart is attempted.
    after_read_only_tools = {tool.name for tool in asyncio.run(mcp_server.server.list_tools())}
    after_read_only_prompts = {
        prompt.name for prompt in asyncio.run(mcp_server.server.list_prompts())
    }
    assert mcp_server._MCP_READ_ONLY is False
    assert "import_document" in after_read_only_tools
    assert {"full-cda", "critical-evidence-review"} <= after_read_only_prompts

    # Start a second server lifecycle in the same process without restoring any
    # private MCP state from the test. The full tool/prompt surface must return.
    monkeypatch.setattr("sys.argv", ["discourselens-mcp", "--project", str(root)])
    mcp_server.main()

    assert import_roots == [str(custom_import_root), None]
    assert len(observed) == 2
    assert observed[0][0] is True
    assert observed[0][1] == mcp_server.READ_ONLY_MCP_TOOL_NAMES
    assert observed[0][2] == set()
    assert observed[1][0] is False
    assert mcp_server.READ_ONLY_MCP_TOOL_NAMES < observed[1][1]
    assert {"import_document", "create_comparative_finding"} <= observed[1][1]
    assert {"full-cda", "critical-evidence-review"} <= observed[1][2]
    assert mcp_server._MCP_READ_ONLY is False

    tools = asyncio.run(mcp_server.server.list_tools())
    prompts = asyncio.run(mcp_server.server.list_prompts())
    assert {tool.name for tool in tools} == observed[1][1]
    assert {prompt.name for prompt in prompts} == observed[1][2]
