from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


class CodebookItem(TypedDict):
    id: str
    name: str
    description: str | None
    created_at: str


class CodeItem(TypedDict):
    id: str
    codebook_id: str
    parent_code_id: str | None
    label: str
    definition: str | None
    created_at: str


class CodebookPage(TypedDict):
    items: list[CodebookItem]
    offset: int
    limit: int
    total: int
    has_more: bool


class CodePage(TypedDict):
    items: list[CodeItem]
    offset: int
    limit: int
    total: int
    has_more: bool


def _create_codebook_pages(
    root: Path,
) -> tuple[str, str, str, CodebookPage, CodebookPage, CodePage, CodePage]:
    project = Project.init(root)
    first_codebook = project.create_codebook("Alpha", "First codebook.")
    project.create_codebook("Beta", "Second codebook.")
    parent_code = project.create_code(first_codebook, "A parent")
    child_code = project.create_code(first_codebook, "B child", parent_code_id=parent_code)
    unused_code = project.create_code(first_codebook, "C unused")
    first_codebooks = cast(CodebookPage, project.codebook_pages(offset=0, limit=1))
    second_codebooks = cast(CodebookPage, project.codebook_pages(offset=1, limit=1))
    first_codes = cast(CodePage, project.codes_for_codebook_page(first_codebook, offset=0, limit=2))
    second_codes = cast(
        CodePage, project.codes_for_codebook_page(first_codebook, offset=2, limit=2)
    )
    project.close()
    return (
        first_codebook,
        child_code,
        unused_code,
        first_codebooks,
        second_codebooks,
        first_codes,
        second_codes,
    )


def test_coding_codebook_listings_return_bounded_core_pages_read_only(tmp_path):
    root = tmp_path / "codebook-pages"
    (
        first_codebook,
        child_code,
        unused_code,
        expected_codebooks_first,
        expected_codebooks_second,
        expected_codes_first,
        expected_codes_second,
    ) = _create_codebook_pages(root)
    database = root / ".discourselens" / "project.sqlite"
    before = database.read_bytes()
    runner = CliRunner()

    codebooks_first = runner.invoke(
        app, ["coding", "list-codebooks", str(root), "--offset", "0", "--limit", "1"]
    )
    codebooks_second = runner.invoke(
        app, ["coding", "list-codebooks", str(root), "--offset", "1", "--limit", "1"]
    )
    codes_first = runner.invoke(
        app,
        ["coding", "list-codes", str(root), first_codebook, "--offset", "0", "--limit", "2"],
    )
    codes_second = runner.invoke(
        app,
        ["coding", "list-codes", str(root), first_codebook, "--offset", "2", "--limit", "2"],
    )

    for result in (codebooks_first, codebooks_second, codes_first, codes_second):
        assert result.exit_code == 0, result.output
    assert json.loads(codebooks_first.output) == expected_codebooks_first
    assert json.loads(codebooks_second.output) == expected_codebooks_second
    assert json.loads(codes_first.output) == expected_codes_first
    assert json.loads(codes_second.output) == expected_codes_second

    listed_codes = expected_codes_first["items"] + expected_codes_second["items"]
    child = next(item for item in listed_codes if item["id"] == child_code)
    assert child["parent_code_id"] is not None
    assert any(item["id"] == unused_code for item in listed_codes)
    assert all("parent_code_id" in item for item in listed_codes)
    assert database.read_bytes() == before


def test_coding_codebook_listings_default_to_one_bounded_page(tmp_path):
    root = tmp_path / "codebook-default-page"
    first_codebook, _, _, *_ = _create_codebook_pages(root)
    runner = CliRunner()

    codebooks = runner.invoke(app, ["coding", "list-codebooks", str(root)])
    codes = runner.invoke(app, ["coding", "list-codes", str(root), first_codebook])

    assert codebooks.exit_code == 0, codebooks.output
    assert codes.exit_code == 0, codes.output
    assert json.loads(codebooks.output)["limit"] == 100
    assert json.loads(codes.output)["limit"] == 100


def test_coding_codebook_listings_reject_invalid_bounds_and_unknown_ids(tmp_path):
    root = tmp_path / "codebook-invalid"
    first_codebook, *_ = _create_codebook_pages(root)
    runner = CliRunner()

    invalid_offset = runner.invoke(app, ["coding", "list-codebooks", str(root), "--offset", "-1"])
    zero_limit = runner.invoke(app, ["coding", "list-codebooks", str(root), "--limit", "0"])
    excessive_limit = runner.invoke(
        app, ["coding", "list-codes", str(root), first_codebook, "--limit", "101"]
    )
    unknown_codebook = runner.invoke(app, ["coding", "list-codes", str(root), "missing-codebook"])

    assert invalid_offset.exit_code == 2
    assert zero_limit.exit_code == 2
    assert excessive_limit.exit_code == 2
    assert unknown_codebook.exit_code == 1
    assert "unknown codebook" in unknown_codebook.output
