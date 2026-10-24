import subprocess
import sys
import tomllib
from pathlib import Path


def test_mcp_sdk_is_an_optional_project_extra():
    project_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    base_dependencies = [item.casefold() for item in project["dependencies"]]
    mcp_dependencies = [item.casefold() for item in project["optional-dependencies"]["mcp"]]

    assert not any(item.startswith("mcp") for item in base_dependencies)
    assert any(item.startswith("mcp>=") for item in mcp_dependencies)


def test_core_and_cli_import_without_optional_dependencies():
    root = Path(__file__).resolve().parents[1]
    script = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(root / "src")!r})

OPTIONAL_IMPORT_ROOTS = {{
    'duckdb', 'pyarrow', 'spacy', 'stanza', 'sqlite_vec', 'keyring', 'mcp',
}}

class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in OPTIONAL_IMPORT_ROOTS:
            raise ModuleNotFoundError(
                f'{{fullname}} intentionally blocked by optional-dependency test',
                name=fullname,
            )
        return None

sys.meta_path.insert(0, BlockOptionalDependencies())
from discourselens import Project
import discourselens.cli
import discourselens.project
assert Project is not None

from discourselens.errors import OptionalDependencyError
try:
    import discourselens.mcp_server
except OptionalDependencyError as exc:
    assert 'discourselens[mcp]' in str(exc)
else:
    raise AssertionError('MCP adapter imported although the optional dependency was blocked')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
