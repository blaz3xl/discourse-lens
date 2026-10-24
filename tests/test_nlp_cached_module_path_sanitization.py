"""Regression tests for safe inspection of verified cached NLP modules."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from discourselens import nlp
from discourselens.errors import ProjectError


def test_cached_verified_module_does_not_use_dynamic_path_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "_discourselens_dynamic_path_probe"
    origin = tmp_path / "verified-package" / "__init__.py"
    origin.parent.mkdir()
    origin.write_text("# synthetic verified origin\n", encoding="utf-8")
    resolved_origin = origin.resolve(strict=True)
    private_path = tmp_path / "private" / "cache-sentinel"
    dynamic_lookups: list[str] = []
    module = ModuleType(module_name)
    module.__dict__.pop("__file__", None)
    module.__dict__.pop("__spec__", None)

    def dynamic_attribute(name: str) -> object:
        dynamic_lookups.append(name)
        raise ProjectError(f"private cached module path: {private_path}")

    module.__getattr__ = dynamic_attribute  # type: ignore[attr-defined]
    assert "__file__" not in module.__dict__
    assert "__spec__" not in module.__dict__

    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(nlp._VERIFIED_MODULES, module_name, (module, resolved_origin))
    monkeypatch.setattr(
        nlp,
        "_VERIFIED_EXECUTED",
        nlp._VERIFIED_EXECUTED | {module_name},
    )

    with pytest.raises(ProjectError, match="already loaded from an unknown origin") as exc:
        nlp._cached_module_with_verified_origin(module_name, resolved_origin, "spaCy")

    assert dynamic_lookups == []
    assert str(private_path) not in str(exc.value)
    assert str(tmp_path) not in str(exc.value)
