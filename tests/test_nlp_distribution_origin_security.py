"""Regression tests for NLP distributions discovered outside interpreter package roots."""

from __future__ import annotations

import re
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest

from discourselens import nlp
from discourselens.errors import ProjectError

_NLP_MODULES = (
    "spacy",
    "stanza",
    "stanza.pipeline",
    "stanza.pipeline.core",
)


@pytest.fixture
def isolated_nlp_import_state(monkeypatch: pytest.MonkeyPatch) -> None:
    names = set(_NLP_MODULES)
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.delitem(nlp._VERIFIED_MODULES, name, raising=False)
    monkeypatch.setattr(nlp, "_VERIFIED_EXECUTED", nlp._VERIFIED_EXECUTED - names)


def _write_distribution(root: Path, name: str, files: dict[str, str]) -> None:
    for relative_path, source in files.items():
        package_file = root / relative_path
        package_file.parent.mkdir(parents=True, exist_ok=True)
        package_file.write_text(source, encoding="utf-8")

    normalized_name = re.sub(r"[-_.]+", "_", name)
    dist_info_name = f"{normalized_name}-1.0.dist-info"
    dist_info = root / dist_info_name
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n",
        encoding="utf-8",
    )
    record_entries = [f"{path},," for path in sorted(files)]
    record_entries.extend(
        [
            f"{dist_info_name}/METADATA,,",
            f"{dist_info_name}/RECORD,,",
        ]
    )
    (dist_info / "RECORD").write_text("\n".join(record_entries) + "\n", encoding="utf-8")


def _marker_source(marker: Path) -> str:
    return (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
    )


@pytest.mark.parametrize("backend", ("spacy", "stanza"))
def test_loader_rejects_paired_shadow_distribution_before_initializer_runs(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_nlp_import_state: None,
) -> None:
    shadow_root = tmp_path / f"{backend}-shadow-site"
    marker = tmp_path / f"{backend}-initializer-executed"

    if backend == "spacy":
        distribution_name = "spacy"
        files = {"spacy/__init__.py": _marker_source(marker)}
    else:
        distribution_name = "stanza"
        files = {
            "stanza/__init__.py": _marker_source(marker),
            "stanza/pipeline/__init__.py": "# hostile Stanza pipeline fixture\n",
            "stanza/pipeline/core.py": "class DownloadMethod:\n    NONE = object()\n",
        }

    _write_distribution(shadow_root, distribution_name, files)
    monkeypatch.syspath_prepend(str(shadow_root))

    # Confirm importlib.metadata sees the paired fake distribution at the front
    # of sys.path; only its non-interpreter-owned location should make it fail.
    discovered = distribution(distribution_name)
    assert Path(str(discovered.locate_file(""))).resolve() == shadow_root.resolve()
    resolved_shadow = shadow_root.resolve()
    assert not any(
        root == resolved_shadow or root in resolved_shadow.parents
        for root in nlp._approved_install_roots()
    )

    with pytest.raises(ProjectError, match="outside approved interpreter package roots"):
        if backend == "spacy":
            nlp.load_spacy("en", "en_core_web_sm")
        else:
            nlp.load_stanza("en", "en-padt", str(tmp_path / "resources"))

    assert not marker.exists()
