"""Direct loader regressions for sanitizing trusted package import failures."""

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
    "en_core_web_sm",
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


def _write_distribution(
    root: Path,
    name: str,
    files: dict[str, str],
    *,
    entry_points: str | None = None,
) -> None:
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
    record_entries.append(f"{dist_info_name}/METADATA,,")
    if entry_points is not None:
        (dist_info / "entry_points.txt").write_text(entry_points, encoding="utf-8")
        record_entries.append(f"{dist_info_name}/entry_points.txt,,")
    record_entries.append(f"{dist_info_name}/RECORD,,")
    (dist_info / "RECORD").write_text("\n".join(record_entries) + "\n", encoding="utf-8")


def _raising_initializer(marker: Path, private_path: Path) -> str:
    return (
        "from pathlib import Path\n"
        "from discourselens.errors import ProjectError\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
        f"raise ProjectError('private package error at {private_path}')\n"
    )


def _marker_initializer(marker: Path) -> str:
    return (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
    )


def _trust_fixture_root(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    resolved_root = root.resolve()
    monkeypatch.setattr(nlp, "_approved_install_roots", lambda: (resolved_root,))
    monkeypatch.syspath_prepend(str(resolved_root))


def test_spacy_initializer_error_is_sanitized_after_distribution_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_nlp_import_state: None,
) -> None:
    trusted_root = tmp_path / "trusted-spacy-site"
    trusted_root.mkdir()
    marker = tmp_path / "spacy-initializer-reached"
    private_path = tmp_path / "private-models" / "spacy-secret"
    model_module = "en_core_web_sm"

    _write_distribution(
        trusted_root,
        "spacy",
        {"spacy/__init__.py": _raising_initializer(marker, private_path)},
    )
    _write_distribution(
        trusted_root,
        model_module,
        {f"{model_module}/__init__.py": "# synthetic installed model package\n"},
        entry_points=f"[spacy_models]\n{model_module} = {model_module}\n",
    )
    _trust_fixture_root(trusted_root, monkeypatch)

    assert Path(str(distribution("spacy").locate_file(""))).resolve() == trusted_root.resolve()
    assert (
        Path(str(distribution(model_module).locate_file(""))).resolve()
        == trusted_root.resolve()
    )

    with pytest.raises(ProjectError, match="spaCy package could not be imported safely") as exc:
        nlp.load_spacy("en", model_module)

    assert marker.read_text(encoding="utf-8") == "executed"
    assert str(private_path) not in str(exc.value)
    assert str(tmp_path) not in str(exc.value)


def test_spacy_model_initializer_error_is_sanitized_after_distribution_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_nlp_import_state: None,
) -> None:
    trusted_root = tmp_path / "trusted-spacy-model-site"
    trusted_root.mkdir()
    root_marker = tmp_path / "spacy-root-initializer-reached"
    model_marker = tmp_path / "spacy-model-initializer-reached"
    private_path = tmp_path / "private-models" / "model-secret"
    model_module = "en_core_web_sm"

    _write_distribution(
        trusted_root,
        "spacy",
        {"spacy/__init__.py": _marker_initializer(root_marker)},
    )
    _write_distribution(
        trusted_root,
        model_module,
        {f"{model_module}/__init__.py": _raising_initializer(model_marker, private_path)},
        entry_points=f"[spacy_models]\n{model_module} = {model_module}\n",
    )
    _trust_fixture_root(trusted_root, monkeypatch)

    assert Path(str(distribution("spacy").locate_file(""))).resolve() == trusted_root.resolve()
    assert (
        Path(str(distribution(model_module).locate_file(""))).resolve()
        == trusted_root.resolve()
    )

    with pytest.raises(
        ProjectError,
        match="spaCy model package could not be imported safely",
    ) as exc:
        nlp.load_spacy("en", model_module)

    assert root_marker.read_text(encoding="utf-8") == "executed"
    assert model_marker.read_text(encoding="utf-8") == "executed"
    assert str(private_path) not in str(exc.value)
    assert str(tmp_path) not in str(exc.value)


def test_stanza_initializer_error_is_sanitized_after_distribution_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_nlp_import_state: None,
) -> None:
    trusted_root = tmp_path / "trusted-stanza-site"
    trusted_root.mkdir()
    marker = tmp_path / "stanza-initializer-reached"
    private_path = tmp_path / "private-models" / "stanza-secret"

    _write_distribution(
        trusted_root,
        "stanza",
        {
            "stanza/__init__.py": _raising_initializer(marker, private_path),
            "stanza/pipeline/__init__.py": "# synthetic installed Stanza pipeline\n",
            "stanza/pipeline/core.py": (
                "class DownloadMethod:\n"
                "    NONE = object()\n"
            ),
        },
    )
    _trust_fixture_root(trusted_root, monkeypatch)

    assert Path(str(distribution("stanza").locate_file(""))).resolve() == trusted_root.resolve()

    with pytest.raises(ProjectError, match="Stanza package could not be imported safely") as exc:
        nlp.load_stanza("en", "en-padt", str(private_path))

    assert marker.read_text(encoding="utf-8") == "executed"
    assert str(private_path) not in str(exc.value)
    assert str(tmp_path) not in str(exc.value)
