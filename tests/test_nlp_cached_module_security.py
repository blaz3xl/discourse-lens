"""Regression tests for cached optional NLP modules and import provenance."""

from __future__ import annotations

import importlib
import re
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import pytest

from discourselens import nlp
from discourselens.errors import ProjectError

_MODULE_NAMES = (
    "spacy",
    "en_core_web_sm",
    "stanza",
    "stanza.pipeline",
    "stanza.pipeline.core",
)


@pytest.fixture
def isolated_verified_module_state():
    """Keep the loader's process-global trust registry isolated per test."""
    names = set(_MODULE_NAMES)
    previous_modules = {name: sys.modules.get(name) for name in names}
    previous_registry = {name: nlp._VERIFIED_MODULES.get(name) for name in names}
    previous_executed = set(nlp._VERIFIED_EXECUTED)
    try:
        for name in names:
            sys.modules.pop(name, None)
            nlp._VERIFIED_MODULES.pop(name, None)
            nlp._VERIFIED_EXECUTED.discard(name)
        yield
    finally:
        for name in names:
            sys.modules.pop(name, None)
            nlp._VERIFIED_MODULES.pop(name, None)
        for name, module in previous_modules.items():
            if module is not None:
                sys.modules[name] = module
        for name, provenance in previous_registry.items():
            if provenance is not None:
                nlp._VERIFIED_MODULES[name] = provenance
        nlp._VERIFIED_EXECUTED.clear()
        nlp._VERIFIED_EXECUTED.update(previous_executed)


def _cached_module(module_name: str, origin: Path) -> ModuleType:
    module = ModuleType(module_name)
    module.__file__ = str(origin)
    module.__spec__ = ModuleSpec(module_name, loader=None, origin=str(origin))
    return module


@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_truthful_cached_origin_is_not_trusted_without_loader_registry(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_verified_module_state: None,
):
    origin = tmp_path / module_name.replace(".", "/") / "__init__.py"
    origin.parent.mkdir(parents=True, exist_ok=True)
    origin.write_text("# synthetic installed origin\n", encoding="utf-8")
    cached = _cached_module(module_name, origin)
    monkeypatch.setitem(sys.modules, module_name, cached)

    with pytest.raises(ProjectError, match="already loaded from an unknown origin"):
        nlp._cached_module_with_verified_origin(module_name, origin, "test backend")


@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_cached_origin_is_accepted_only_for_exact_verified_executed_object(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_verified_module_state: None,
):
    origin = tmp_path / module_name.replace(".", "/") / "__init__.py"
    origin.parent.mkdir(parents=True, exist_ok=True)
    origin.write_text("# synthetic installed origin\n", encoding="utf-8")
    trusted = _cached_module(module_name, origin)
    monkeypatch.setitem(sys.modules, module_name, trusted)
    monkeypatch.setitem(nlp._VERIFIED_MODULES, module_name, (trusted, origin.resolve()))
    monkeypatch.setattr(nlp, "_VERIFIED_EXECUTED", {*nlp._VERIFIED_EXECUTED, module_name})

    assert nlp._cached_module_with_verified_origin(module_name, origin.resolve(), "backend") is trusted

    # A second object cannot inherit trust by copying both plausible path fields.
    replacement = _cached_module(module_name, origin)
    monkeypatch.setitem(sys.modules, module_name, replacement)
    with pytest.raises(ProjectError, match="already loaded from an unknown origin"):
        nlp._cached_module_with_verified_origin(module_name, origin.resolve(), "backend")


def _write_distribution(
    root: Path,
    name: str,
    files: dict[str, str],
    *,
    entry_points: str | None = None,
) -> None:
    for relative, contents in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")
    normalized_name = re.sub(r"[-_.]+", "_", name)
    dist_info = root / f"{normalized_name}-1.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8"
    )
    entry_points_path = f"{normalized_name}-1.0.dist-info/entry_points.txt"
    if entry_points is not None:
        (dist_info / "entry_points.txt").write_text(entry_points, encoding="utf-8")
    entries = [f"{relative},," for relative in sorted(files)]
    entries.extend(
        [
            f"{normalized_name}-1.0.dist-info/METADATA,,",
            f"{normalized_name}-1.0.dist-info/RECORD,,",
        ]
    )
    if entry_points is not None:
        entries.append(f"{entry_points_path},,")
    (dist_info / "RECORD").write_text("\n".join(entries) + "\n", encoding="utf-8")


def _approve_temp_install_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    approved_root = root.resolve()
    monkeypatch.setattr(nlp, "_approved_install_roots", lambda: (approved_root,))


def _hostile_cached_module(module_name: str, origin: Path, marker: Path) -> ModuleType:
    module = _cached_module(module_name, origin)

    def mark_executed(*_args: object, **_kwargs: object) -> None:
        marker.write_text(f"{module_name} executed\n", encoding="utf-8")

    module.hostile = mark_executed  # type: ignore[attr-defined]
    return module


def _write_marker_source(marker: Path, label: str) -> str:
    return (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text({(label + ' executed')!r}, encoding='utf-8')\n"
    )


def test_preseeded_spacy_root_cannot_reach_hostile_loader_callable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_verified_module_state: None,
):
    site_packages = tmp_path / "site-packages"
    marker = tmp_path / "executed.txt"
    _write_distribution(
        site_packages,
        "spacy",
        {"spacy/__init__.py": _write_marker_source(marker, "trusted disk root")},
    )
    _write_distribution(
        site_packages,
        "en_core_web_sm",
        {"en_core_web_sm/__init__.py": "# registered model package fixture\n"},
        entry_points="[spacy_models]\nen_core_web_sm = en_core_web_sm\n",
    )
    _approve_temp_install_root(monkeypatch, site_packages)
    monkeypatch.syspath_prepend(str(site_packages))
    importlib.invalidate_caches()

    hostile = _hostile_cached_module("spacy", site_packages / "spacy/__init__.py", marker)

    def hostile_is_package(_model: str) -> bool:
        hostile.hostile()  # type: ignore[attr-defined]
        return True

    hostile.util = type("Util", (), {"is_package": staticmethod(hostile_is_package)})()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", hostile)

    with pytest.raises(ProjectError, match="already loaded from an unknown origin"):
        nlp.load_spacy("en", "en_core_web_sm")

    assert not marker.exists()


def test_preseeded_spacy_model_cannot_reach_spacy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_verified_module_state: None,
):
    site_packages = tmp_path / "site-packages"
    marker = tmp_path / "executed.txt"
    _write_distribution(site_packages, "spacy", {"spacy/__init__.py": "# trusted root\n"})
    _write_distribution(
        site_packages,
        "en_core_web_sm",
        {"en_core_web_sm/__init__.py": _write_marker_source(marker, "model disk source")},
        entry_points="[spacy_models]\nen_core_web_sm = en_core_web_sm\n",
    )
    _approve_temp_install_root(monkeypatch, site_packages)
    monkeypatch.syspath_prepend(str(site_packages))
    importlib.invalidate_caches()

    spacy_module = nlp._load_verified_package_module("spacy", "spacy", "spaCy")
    calls: list[str] = []
    spacy_module.util = type(  # type: ignore[attr-defined]
        "Util", (), {"is_package": staticmethod(lambda _model: True)}
    )()

    def load_model(model: str) -> object:
        calls.append(model)
        marker.write_text("spacy.load invoked\n", encoding="utf-8")
        return object()

    spacy_module.load = load_model  # type: ignore[attr-defined]
    hostile_model = _hostile_cached_module(
        "en_core_web_sm", site_packages / "en_core_web_sm/__init__.py", marker
    )
    monkeypatch.setitem(sys.modules, "en_core_web_sm", hostile_model)

    with pytest.raises(ProjectError, match="already loaded from an unknown origin"):
        nlp.load_spacy("en", "en_core_web_sm")

    assert calls == []
    assert not marker.exists()


@pytest.mark.parametrize("module_name", ("stanza", "stanza.pipeline", "stanza.pipeline.core"))
def test_preseeded_stanza_module_is_rejected_before_any_initializer_runs(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_verified_module_state: None,
):
    site_packages = tmp_path / "site-packages"
    marker = tmp_path / "executed.txt"
    files = {
        "stanza/__init__.py": _write_marker_source(marker, "stanza root"),
        "stanza/pipeline/__init__.py": _write_marker_source(marker, "stanza pipeline"),
        "stanza/pipeline/core.py": _write_marker_source(marker, "stanza core"),
    }
    _write_distribution(site_packages, "stanza", files)
    _approve_temp_install_root(monkeypatch, site_packages)
    monkeypatch.syspath_prepend(str(site_packages))
    importlib.invalidate_caches()

    relative_origins = {
        "stanza": "stanza/__init__.py",
        "stanza.pipeline": "stanza/pipeline/__init__.py",
        "stanza.pipeline.core": "stanza/pipeline/core.py",
    }
    origin = site_packages / relative_origins[module_name]
    monkeypatch.setitem(sys.modules, module_name, _hostile_cached_module(module_name, origin, marker))

    with pytest.raises(ProjectError, match="already loaded from an unknown origin"):
        nlp._load_verified_stanza_modules()

    assert not marker.exists()


def test_exact_registered_module_identity_is_required_even_when_paths_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_verified_module_state: None,
):
    module_name = "stanza.pipeline.core"
    origin = tmp_path / "stanza/pipeline/core.py"
    origin.parent.mkdir(parents=True)
    origin.write_text("# verified source\n", encoding="utf-8")
    trusted = _cached_module(module_name, origin)
    impostor = _cached_module(module_name, origin)
    monkeypatch.setitem(sys.modules, module_name, impostor)
    monkeypatch.setitem(nlp._VERIFIED_MODULES, module_name, (trusted, origin.resolve()))
    monkeypatch.setattr(nlp, "_VERIFIED_EXECUTED", {*nlp._VERIFIED_EXECUTED, module_name})

    with pytest.raises(ProjectError, match="already loaded from an unknown origin"):
        nlp._cached_module_with_verified_origin(module_name, origin.resolve(), "Stanza core")
