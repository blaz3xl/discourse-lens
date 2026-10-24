from pathlib import Path

README = Path(__file__).parents[1] / "README.md"


def test_quickstart_keeps_optional_nlp_commands_behind_their_prerequisites():
    readme = README.read_text(encoding="utf-8")
    quickstart = readme.split("## Quick start", maxsplit=1)[1].split("```", maxsplit=2)[1]
    nlp_section = readme.split("Local NLP is optional", maxsplit=1)[1].split(
        "`document import-many`", maxsplit=1
    )[0]

    assert "discourselens[nlp]" not in readme.split("## Python SDK", maxsplit=1)[0].split(
        "Local NLP is optional", maxsplit=1
    )[0]
    assert "corpus build-nlp" not in quickstart
    assert "nlp capabilities" not in quickstart
    assert 'uv tool install --force "discourselens[corpus,vectors,mcp,nlp]"' in nlp_section
    assert "including both `corpus` and `nlp`" in nlp_section
    assert "already installed the base tool above" in nlp_section
    assert "does not install" in nlp_section
    assert "language models or Stanza resource files" in nlp_section
    assert "Install the selected model/resource" in nlp_section
    assert "corpus build-nlp" in nlp_section
    assert "nlp capabilities" in nlp_section
