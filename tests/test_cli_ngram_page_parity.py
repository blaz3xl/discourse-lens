import json

from typer.testing import CliRunner

from discourselens import Project
from discourselens.cli import app


def test_cli_ngram_page_uses_core_tie_order_without_changing_full_output(tmp_path):
    root = tmp_path / "ngram-page-order"
    project = Project.init(root)
    _, revision = project.import_text("ties.txt", "zeta beta zeta alpha zeta")
    scope_id = project.snapshot([revision])
    sdk_page = project.corpus_ngram_page(scope_id, 2, limit=4)
    project.close()

    command = ["corpus", "ngrams", str(root), revision, "--size", "2"]
    runner = CliRunner()
    full = runner.invoke(app, command)
    page = runner.invoke(app, [*command, "--offset", "0", "--limit", "4"])

    assert full.exit_code == 0, full.output
    assert page.exit_code == 0, page.output
    full_rows = json.loads(full.stdout)
    page_payload = json.loads(page.stdout)
    assert [row[0] for row in full_rows] == [
        "zeta beta",
        "beta zeta",
        "zeta alpha",
        "alpha zeta",
    ]
    assert [row["ngram"] for row in page_payload["rows"]] == [
        row["ngram"] for row in sdk_page["rows"]
    ] == ["alpha zeta", "beta zeta", "zeta alpha", "zeta beta"]
    assert page_payload["total"] == sdk_page["total_values"] == 4


def test_cli_ngram_page_uses_nfc_normalized_surface_tokens(tmp_path):
    root = tmp_path / "ngram-page-nfc"
    project = Project.init(root)
    _, revision = project.import_text("nfd.txt", "cafe\u0301 policy")
    project.close()

    result = CliRunner().invoke(
        app,
        ["corpus", "ngrams", str(root), revision, "--size", "2", "--offset", "0", "--limit", "5"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["rows"] == [{"ngram": "café policy", "count": 1}]
