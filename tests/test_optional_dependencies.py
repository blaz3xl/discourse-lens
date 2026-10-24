import subprocess
import sys
from pathlib import Path

_OPTIONAL_DEPENDENCY_ASSERTIONS = """
import site
import sysconfig
import tempfile

from discourselens.errors import OptionalDependencyError

RAW_PATHS = {
    tempfile.gettempdir(),
    *site.getsitepackages(),
    site.getusersitepackages(),
    sysconfig.get_paths().get('purelib', ''),
    sysconfig.get_paths().get('platlib', ''),
}

def expect_optional_dependency(extra_hint, operation):
    try:
        operation()
    except OptionalDependencyError as exc:
        assert extra_hint in str(exc), str(exc)
    except Exception as exc:
        raise AssertionError(
            f'expected OptionalDependencyError for {extra_hint}, got {type(exc).__name__}'
        ) from exc
    else:
        raise AssertionError(f'operation unexpectedly succeeded without {extra_hint}')

def expect_nlp_optional_dependency(operation):
    try:
        operation()
    except OptionalDependencyError as exc:
        message = str(exc)
        assert 'discourselens[nlp]' in message, message
        for raw_path in RAW_PATHS:
            if raw_path and len(raw_path) > 1:
                assert raw_path not in message, message
    except Exception as exc:
        raise AssertionError(
            'expected OptionalDependencyError for discourselens[nlp], '
            f'got {type(exc).__name__}'
        ) from exc
    else:
        raise AssertionError('operation unexpectedly succeeded without discourselens[nlp]')
"""


def _run_without_optional_dependencies(body: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(project_root / "src")!r})

OPTIONAL_IMPORT_ROOTS = {{
    'duckdb', 'pyarrow', 'spacy', 'stanza', 'sqlite_vec', 'keyring', 'mcp',
}}

# Keep distribution metadata intact: the NLP loaders can see an installed
# distribution while this finder still blocks its module imports.
class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in OPTIONAL_IMPORT_ROOTS:
            raise ModuleNotFoundError(
                f'{{fullname}} intentionally blocked by optional-dependency test',
                name=fullname,
            )
        return None

sys.meta_path.insert(0, BlockOptionalDependencies())
{_OPTIONAL_DEPENDENCY_ASSERTIONS}
{body}
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_spacy_extra_reports_sanitized_install_hint():
    _run_without_optional_dependencies(
        """
from discourselens.nlp import load_spacy

expect_nlp_optional_dependency(lambda: load_spacy('en', 'en_core_web_sm'))
"""
    )


def test_missing_stanza_extra_reports_sanitized_install_hint():
    _run_without_optional_dependencies(
        """
from discourselens.nlp import load_stanza

expect_nlp_optional_dependency(lambda: load_stanza('en', 'en', '/unused'))
"""
    )


def test_missing_pyarrow_corpus_extra_reports_install_hint():
    _run_without_optional_dependencies(
        """
from discourselens.nlp_artifacts import _pyarrow

expect_optional_dependency('discourselens[corpus]', _pyarrow)
"""
    )


def test_missing_duckdb_corpus_extra_reports_install_hint():
    _run_without_optional_dependencies(
        """
from discourselens.nlp_artifacts import _duckdb

expect_optional_dependency('discourselens[corpus]', _duckdb)
"""
    )


def test_missing_sqlite_vec_extra_reports_install_hint():
    _run_without_optional_dependencies(
        """
from discourselens.vectors import _sqlite_vec

expect_optional_dependency('discourselens[vectors]', _sqlite_vec)
"""
    )


def test_missing_keyring_extra_reports_install_hint():
    _run_without_optional_dependencies(
        """
from discourselens.secrets import KeyringSecretStore

expect_optional_dependency(
    'discourselens[secrets]',
    lambda: KeyringSecretStore().get('keyring:discourselens:fixture-account'),
)
"""
    )
