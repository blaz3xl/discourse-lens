"""Regression tests for bypassing monkeypatchable ModuleSpec dispatch."""

from __future__ import annotations

from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import pytest

from discourselens import nlp


def test_module_origin_extraction_uses_base_spec_namespace_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "_discourselens_spec_dispatch_probe"
    origin = tmp_path / "verified-package" / "__init__.py"
    origin.parent.mkdir()
    origin.write_text("# synthetic module origin\n", encoding="utf-8")
    origin_text = str(origin.resolve(strict=True))
    marker = tmp_path / "module-spec-dict-accessed"
    spec = ModuleSpec(module_name, loader=None, origin=origin_text)
    module = ModuleType(module_name)
    module.__file__ = origin_text
    module.__spec__ = spec

    original_getattribute = ModuleSpec.__getattribute__

    def tracked_getattribute(instance: ModuleSpec, name: str) -> object:
        if name == "__dict__":
            marker.write_text("dispatched", encoding="utf-8")
        return original_getattribute(instance, name)

    monkeypatch.setattr(ModuleSpec, "__getattribute__", tracked_getattribute)

    assert nlp._module_origin_values(module) == (origin_text, origin_text)
    assert not marker.exists()
