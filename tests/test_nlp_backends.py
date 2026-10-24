import os
import re
import sys
from importlib.machinery import ModuleSpec, PathFinder
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from discourselens import nlp as nlp_module
from discourselens.errors import ProjectError
from discourselens.languages import inspect_local_backend, probe_backend_capabilities
from discourselens.nlp import (
    ParsedToken,
    SpacyBackend,
    StanzaBackend,
    load_spacy,
    load_stanza,
)


def _install_fake_distribution(
    root: Path,
    distribution_name: str,
    package_files: dict[str, str],
    *,
    entry_points: str | None = None,
) -> None:
    """Create an importable package with a minimal RECORD-backed distribution."""
    for relative_path, source in package_files.items():
        package_file = root / relative_path
        package_file.parent.mkdir(parents=True, exist_ok=True)
        package_file.write_text(source, encoding="utf-8")
    normalized_name = re.sub(r"[-_.]+", "_", distribution_name)
    distribution_dir = root / f"{normalized_name}-1.0.dist-info"
    distribution_dir.mkdir()
    (distribution_dir / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution_name}\nVersion: 1.0\n",
        encoding="utf-8",
    )
    entry_points_path = f"{normalized_name}-1.0.dist-info/entry_points.txt"
    if entry_points is not None:
        (distribution_dir / "entry_points.txt").write_text(entry_points, encoding="utf-8")
    record_entries = [f"{path},," for path in sorted(package_files)]
    record_entries.extend(
        [
            f"{normalized_name}-1.0.dist-info/METADATA,,",
            f"{normalized_name}-1.0.dist-info/RECORD,,",
        ]
    )
    if entry_points is not None:
        record_entries.append(f"{entry_points_path},,")
    (distribution_dir / "RECORD").write_text("\n".join(record_entries) + "\n", encoding="utf-8")


def _trust_fixture_root(monkeypatch, root: Path) -> None:
    approved_roots = nlp_module._approved_install_roots()
    fixture_root = root.resolve(strict=True)
    monkeypatch.setattr(
        nlp_module,
        "_approved_install_roots",
        lambda: (*approved_roots, fixture_root),
    )


def _install_fake_spacy_packages(
    root: Path, monkeypatch, model_distribution: str = "en_core_web_sm"
) -> None:
    _install_fake_distribution(root, "spacy", {"spacy/__init__.py": "# trusted spaCy fixture\n"})
    module_name = model_distribution.replace("-", "_")
    _install_fake_distribution(
        root,
        model_distribution,
        {f"{module_name}/__init__.py": "# trusted local model fixture\n"},
        entry_points=f"[spacy_models]\n{module_name} = {module_name}\n",
    )
    _trust_fixture_root(monkeypatch, root)


def _reset_verified_modules(monkeypatch, *module_names: str) -> None:
    for module_name in module_names:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
        monkeypatch.delitem(nlp_module._VERIFIED_MODULES, module_name, raising=False)
    monkeypatch.setattr(
        nlp_module,
        "_VERIFIED_EXECUTED",
        nlp_module._VERIFIED_EXECUTED - set(module_names),
    )


def _fake_spacy_module(pipeline: object, *, is_package=lambda _name: True):
    module = nlp_module._load_verified_package_module("spacy", "spacy", "spaCy")
    module.util = SimpleNamespace(is_package=is_package)  # type: ignore[attr-defined]
    module.load = lambda _name: pipeline  # type: ignore[attr-defined]
    return module


def _install_fake_stanza_distribution(root: Path, monkeypatch) -> None:
    files = {
        "stanza/__init__.py": "from stanza.pipeline.core import DownloadMethod\n",
        "stanza/pipeline/__init__.py": "# trusted Stanza pipeline fixture\n",
        "stanza/pipeline/core.py": "class DownloadMethod:\n    NONE = object()\n",
    }
    _install_fake_distribution(root, "stanza", files)
    _trust_fixture_root(monkeypatch, root)


def _load_fake_stanza(monkeypatch, root: Path, pipeline_callback):
    root.mkdir(parents=True, exist_ok=True)
    _install_fake_stanza_distribution(root, monkeypatch)
    _reset_verified_modules(
        monkeypatch, "stanza", "stanza.pipeline", "stanza.pipeline.core"
    )
    monkeypatch.syspath_prepend(str(root))
    stanza, core = nlp_module._load_verified_stanza_modules()
    stanza.Pipeline = pipeline_callback  # type: ignore[attr-defined]
    return stanza, core


def test_parsed_token_preserves_canonical_unicode_offsets():
    token = ParsedToken(
        surface="Café",
        lemma="café",
        pos="NOUN",
        morphology=None,
        dependency_head=None,
        dependency_relation="ROOT",
        char_start=0,
        char_end=4,
    )
    assert "Café migración"[token.char_start : token.char_end] == token.surface


def test_stanza_loader_selects_model_package_and_disables_downloads(monkeypatch, tmp_path):
    calls: dict[str, object] = {}
    sentinel = SimpleNamespace(lang="ar")

    def pipeline(language: str, **options: object):
        calls["language"] = language
        calls.update(options)
        return sentinel

    _, core = _load_fake_stanza(monkeypatch, tmp_path / "site-packages", pipeline)
    DownloadMethod = core.DownloadMethod  # type: ignore[attr-defined]

    backend = load_stanza("ar", "ar-padt", str(tmp_path))

    assert backend.model_id == "ar-padt"
    assert backend.pipeline is sentinel
    assert calls["language"] == "ar"
    assert calls["package"] == "padt"
    assert calls["model_dir"] == str(tmp_path)
    assert calls["download_method"] is DownloadMethod.NONE
    assert calls["processors"] == "tokenize,pos,lemma,depparse,ner"


@pytest.mark.parametrize("reported_language", ["es", None])
def test_stanza_loader_rejects_unverified_or_mismatched_model_language(
    monkeypatch, tmp_path, reported_language
):
    _load_fake_stanza(
        monkeypatch,
        tmp_path / "site-packages",
        lambda *_args, **_kwargs: (
            SimpleNamespace(lang=reported_language)
            if reported_language is not None
            else SimpleNamespace()
        ),
    )

    with pytest.raises(ProjectError, match="Stanza model"):
        load_stanza("ar", "ar-padt", str(tmp_path))


def test_stanza_loader_reports_offline_model_failure(monkeypatch, tmp_path):
    def fail_pipeline(*_args, **_kwargs):
        raise OSError("not installed")

    _load_fake_stanza(monkeypatch, tmp_path / "site-packages", fail_pipeline)

    with pytest.raises(ProjectError, match="no downloads were attempted"):
        load_stanza("ar", "ar-padt", str(tmp_path))


@pytest.mark.parametrize(("requested", "reported"), [("es", "en"), ("en", None)])
def test_spacy_loader_rejects_unverified_or_mismatched_model_language(
    monkeypatch, tmp_path, requested, reported
):
    model_id = "en_fixture_model"
    package_root = tmp_path / "site-packages"
    _install_fake_spacy_packages(package_root, monkeypatch, model_id)
    monkeypatch.syspath_prepend(str(package_root))
    _reset_verified_modules(monkeypatch, "spacy", model_id)
    _fake_spacy_module(SimpleNamespace(lang=reported))

    with pytest.raises(ProjectError, match="spaCy model"):
        load_spacy(requested, model_id)


def test_spacy_loader_accepts_an_installed_package_matching_explicit_language(
    monkeypatch, tmp_path
):
    model_id = "en_core_web_sm"
    package_root = tmp_path / "site-packages"
    _install_fake_spacy_packages(package_root, monkeypatch, model_id)
    monkeypatch.syspath_prepend(str(package_root))
    _reset_verified_modules(monkeypatch, "spacy", model_id)
    pipeline = SimpleNamespace(lang="en")
    loaded_models: list[str] = []

    def load(model_id: str):
        loaded_models.append(model_id)
        return pipeline

    spacy_module = _fake_spacy_module(pipeline)
    spacy_module.load = load  # type: ignore[attr-defined]

    selected = load_spacy("en-US", model_id)

    assert isinstance(selected, SpacyBackend)
    assert selected.language == "en-US"
    assert selected.pipeline is pipeline
    assert loaded_models == [model_id]


def test_spacy_loader_normalizes_hyphenated_distribution_names(monkeypatch, tmp_path):
    model_id = "en-core-web-sm"
    module_name = "en_core_web_sm"
    package_root = tmp_path / "site-packages"
    _install_fake_spacy_packages(package_root, monkeypatch, model_id)
    monkeypatch.syspath_prepend(str(package_root))
    _reset_verified_modules(monkeypatch, "spacy", module_name)
    pipeline = SimpleNamespace(lang="en")
    loaded_models: list[str] = []
    spacy_module = _fake_spacy_module(pipeline)
    spacy_module.load = lambda name: (loaded_models.append(name), pipeline)[1]  # type: ignore[attr-defined]

    backend = load_spacy("en", model_id)

    assert backend.model_id == model_id
    assert loaded_models == [module_name]


def test_spacy_loader_rejects_shadow_spacy_module_without_executing_it(monkeypatch, tmp_path):
    model_id = "en_core_web_sm"
    installed_root = tmp_path / "installed-site"
    _install_fake_spacy_packages(installed_root, monkeypatch, model_id)
    marker = tmp_path / "spacy-shadow-executed"
    shadow_root = tmp_path / "shadow-site"
    shadow_root.mkdir()
    (shadow_root / "spacy.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(installed_root))
    monkeypatch.syspath_prepend(str(shadow_root))
    _reset_verified_modules(monkeypatch, "spacy", model_id)

    with pytest.raises(ProjectError, match="does not originate"):
        load_spacy("en", model_id)

    assert not marker.exists()


def test_spacy_loader_sanitizes_project_error_from_package_initializer(monkeypatch, tmp_path):
    model_id = "en_core_web_sm"
    package_root = tmp_path / "site-packages"
    _install_fake_spacy_packages(package_root, monkeypatch, model_id)
    marker = tmp_path / "spacy-initializer-ran"
    private_path = tmp_path / "private" / "model-cache"
    (package_root / "spacy" / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from discourselens.errors import ProjectError\n"
        f"Path({str(marker)!r}).write_text('ran')\n"
        f"raise ProjectError({str(private_path)!r})\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(package_root))
    _reset_verified_modules(monkeypatch, "spacy", model_id)

    with pytest.raises(ProjectError, match="package could not be imported safely") as exc_info:
        load_spacy("en", model_id)

    assert marker.read_text(encoding="utf-8") == "ran"
    assert str(private_path) not in str(exc_info.value)
    assert "spacy" not in sys.modules


def test_spacy_loader_does_not_dispatch_module_getattr_for_origin_check(monkeypatch, tmp_path):
    model_id = "en_core_web_sm"
    package_root = tmp_path / "site-packages"
    _install_fake_spacy_packages(package_root, monkeypatch, model_id)
    marker = tmp_path / "dynamic-origin-lookup-ran"
    private_path = tmp_path / "private" / "model-cache"
    (package_root / "spacy" / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from discourselens.errors import ProjectError\n"
        "del __file__\n"
        "def __getattr__(name):\n"
        f"    Path({str(marker)!r}).write_text(name)\n"
        f"    raise ProjectError({str(private_path)!r})\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(package_root))
    _reset_verified_modules(monkeypatch, "spacy", model_id)

    with pytest.raises(ProjectError, match="changed its verified origin during import") as exc_info:
        load_spacy("en", model_id)

    assert not marker.exists()
    assert str(private_path) not in str(exc_info.value)
    assert "spacy" not in sys.modules


def test_spacy_loader_rejects_unapproved_shadow_distribution_before_execution(
    monkeypatch, tmp_path
):
    shadow_root = tmp_path / "untrusted-pythonpath"
    marker = tmp_path / "untrusted-distribution-executed"
    spacy_source = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
    )
    _install_fake_distribution(shadow_root, "spacy", {"spacy/__init__.py": spacy_source})
    _install_fake_distribution(
        shadow_root,
        "en_core_web_sm",
        {"en_core_web_sm/__init__.py": "# untrusted model\n"},
    )
    monkeypatch.syspath_prepend(str(shadow_root))
    _reset_verified_modules(monkeypatch, "spacy", "en_core_web_sm")

    assert Path(str(distribution("spacy").locate_file(""))).resolve() == shadow_root.resolve()
    assert shadow_root.resolve() not in nlp_module._approved_install_roots()
    with pytest.raises(ProjectError, match="outside approved interpreter package roots"):
        load_spacy("en", "en_core_web_sm")

    assert not marker.exists()


@pytest.mark.parametrize(
    "entry_points",
    [
        None,
        "[spacy_models]\nother_pipeline = unrelated_nlp_package\n",
        "[spacy_models]\nunrelated_nlp_package = other_module\n",
    ],
)
def test_spacy_loader_rejects_unrelated_package_before_model_initializer(
    monkeypatch, tmp_path, entry_points
):
    package_root = tmp_path / "site-packages"
    marker = tmp_path / "unrelated-package-executed"
    model_module = "unrelated_nlp_package"
    _install_fake_distribution(
        package_root,
        "spacy",
        {"spacy/__init__.py": "# trusted fake spaCy distribution\n"},
    )
    model_source = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
    )
    _install_fake_distribution(
        package_root,
        model_module,
        {f"{model_module}/__init__.py": model_source},
        entry_points=entry_points,
    )
    _trust_fixture_root(monkeypatch, package_root)
    monkeypatch.syspath_prepend(str(package_root))
    _reset_verified_modules(monkeypatch, "spacy", model_module)

    with pytest.raises(ProjectError, match="not registered as a spaCy model"):
        load_spacy("en", model_module)

    assert not marker.exists()
    assert "spacy" not in sys.modules
    assert model_module not in sys.modules


def test_spacy_loader_rejects_unregistered_module_even_when_it_claims_installed_origin(
    monkeypatch, tmp_path
):
    model_id = "en_core_web_sm"
    package_root = tmp_path / "site-packages"
    _install_fake_spacy_packages(package_root, monkeypatch, model_id)
    monkeypatch.syspath_prepend(str(package_root))
    _reset_verified_modules(monkeypatch, "spacy", model_id)
    spec = PathFinder.find_spec("spacy", [str(package_root)])
    assert spec is not None and spec.origin is not None
    fake_spacy = ModuleType("spacy")
    fake_spacy.__file__ = spec.origin
    fake_spacy.__spec__ = spec
    calls: list[str] = []
    fake_spacy.util = SimpleNamespace(is_package=lambda _name: True)  # type: ignore[attr-defined]
    fake_spacy.load = lambda name: calls.append(name)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)

    with pytest.raises(ProjectError, match="already loaded from an unknown origin"):
        load_spacy("en", model_id)

    assert calls == []


def test_spacy_loader_rejects_preloaded_model_even_when_it_claims_installed_origin(
    monkeypatch, tmp_path
):
    model_id = "en_core_web_sm"
    package_root = tmp_path / "site-packages"
    _install_fake_spacy_packages(package_root, monkeypatch, model_id)
    monkeypatch.syspath_prepend(str(package_root))
    _reset_verified_modules(monkeypatch, "spacy", model_id)
    load_calls: list[str] = []
    spacy_module = _fake_spacy_module(SimpleNamespace(lang="en"))
    spacy_module.load = lambda name: load_calls.append(name)  # type: ignore[attr-defined]
    model_spec = PathFinder.find_spec(model_id, [str(package_root)])
    assert model_spec is not None and model_spec.origin is not None
    preloaded_model = ModuleType(model_id)
    preloaded_model.__file__ = model_spec.origin
    preloaded_model.__spec__ = model_spec
    monkeypatch.setitem(sys.modules, model_id, preloaded_model)

    with pytest.raises(ProjectError, match="already loaded from an unknown origin"):
        load_spacy("en", model_id)

    assert load_calls == []


def test_spacy_loader_rejects_same_named_shadow_package_before_loading(monkeypatch, tmp_path):
    model_id = "en_core_web_sm"
    installed_root = tmp_path / "installed-site"
    _install_fake_spacy_packages(installed_root, monkeypatch, model_id)
    shadow_root = tmp_path / "shadow-site"
    shadow_package = shadow_root / model_id
    shadow_package.mkdir(parents=True)
    marker = tmp_path / "model-shadow-executed"
    (shadow_package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(installed_root))
    monkeypatch.syspath_prepend(str(shadow_root))
    _reset_verified_modules(monkeypatch, "spacy", model_id)

    load_calls: list[str] = []
    _fake_spacy_module(SimpleNamespace(lang="en")).load = lambda name: load_calls.append(name)  # type: ignore[attr-defined]

    with pytest.raises(ProjectError, match="does not originate"):
        load_spacy("en", model_id)

    assert load_calls == []
    assert not marker.exists()


def test_stanza_loader_rejects_shadow_root_before_executing_initializer(monkeypatch, tmp_path):
    installed_root = tmp_path / "installed-site"
    _install_fake_stanza_distribution(installed_root, monkeypatch)
    marker = tmp_path / "stanza-shadow-executed"
    shadow_root = tmp_path / "shadow-site"
    shadow_package = shadow_root / "stanza"
    shadow_package.mkdir(parents=True)
    (shadow_package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(installed_root))
    monkeypatch.syspath_prepend(str(shadow_root))
    _reset_verified_modules(monkeypatch, "stanza", "stanza.pipeline", "stanza.pipeline.core")

    with pytest.raises(ProjectError, match="does not originate"):
        load_stanza("en", "en-padt", str(tmp_path))

    assert not marker.exists()


@pytest.mark.parametrize("module_name", ["stanza", "stanza.pipeline", "stanza.pipeline.core"])
def test_stanza_loader_rejects_unregistered_modules_claiming_installed_origins(
    monkeypatch, tmp_path, module_name
):
    package_root = tmp_path / "site-packages"
    _install_fake_stanza_distribution(package_root, monkeypatch)
    marker = tmp_path / "stanza-initializer-executed"
    initializer = package_root / "stanza" / "__init__.py"
    initializer.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n"
        "from stanza.pipeline.core import DownloadMethod\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(package_root))
    _reset_verified_modules(monkeypatch, "stanza", "stanza.pipeline", "stanza.pipeline.core")
    search_roots = {
        "stanza": package_root,
        "stanza.pipeline": package_root / "stanza",
        "stanza.pipeline.core": package_root / "stanza" / "pipeline",
    }
    spec = PathFinder.find_spec(module_name, [str(search_roots[module_name])])
    assert spec is not None and spec.origin is not None
    fake_module = ModuleType(module_name)
    fake_module.__file__ = spec.origin
    fake_module.__spec__ = spec
    if spec.submodule_search_locations is not None:
        fake_module.__path__ = list(spec.submodule_search_locations)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    with pytest.raises(ProjectError, match="already loaded from an unknown origin"):
        load_stanza("en", "en-padt", str(tmp_path))

    assert not marker.exists()


def test_stanza_loader_rejects_cached_shadow_core_without_executing_it(monkeypatch, tmp_path):
    package_root = tmp_path / "site-packages"
    _install_fake_stanza_distribution(package_root, monkeypatch)
    _reset_verified_modules(monkeypatch, "stanza", "stanza.pipeline", "stanza.pipeline.core")
    monkeypatch.syspath_prepend(str(package_root))
    calls: list[bool] = []
    stanza, _ = nlp_module._load_verified_stanza_modules()
    stanza.Pipeline = lambda *_args, **_kwargs: calls.append(True)  # type: ignore[attr-defined]

    marker = tmp_path / "stanza-core-shadow-executed"
    shadow_core = tmp_path / "shadow-site" / "stanza" / "pipeline" / "core.py"
    shadow_core.parent.mkdir(parents=True)
    shadow_core.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    shadow_module = ModuleType("stanza.pipeline.core")
    shadow_module.__file__ = str(shadow_core)
    shadow_module.__spec__ = ModuleSpec("stanza.pipeline.core", loader=None, origin=str(shadow_core))
    monkeypatch.setitem(sys.modules, "stanza.pipeline.core", shadow_module)

    with pytest.raises(ProjectError, match="already loaded from an unknown origin"):
        load_stanza("en", "en-padt", str(tmp_path))

    assert not marker.exists()
    assert calls == []


def test_spacy_loader_never_loads_a_same_named_cwd_directory(monkeypatch, tmp_path):
    model_id = "discourselens_cwd_only_model_64b3a2e9"
    package_dir = tmp_path / model_id
    package_dir.mkdir()
    marker = tmp_path / "cwd-model-executed"
    (package_dir / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    with pytest.raises(PackageNotFoundError):
        distribution(model_id)
    monkeypatch.chdir(tmp_path)
    _reset_verified_modules(monkeypatch, "spacy", model_id)

    with pytest.raises(ProjectError, match="spaCy model is not installed as a package"):
        load_spacy("en", model_id)

    assert not marker.exists()
    assert "spacy" not in sys.modules


def test_spacy_loader_rejects_model_identifier_for_another_language(monkeypatch):
    load_calls: list[str] = []
    spacy_module = ModuleType("spacy")

    def load(model_id: str):
        load_calls.append(model_id)
        # A damaged alias/registry could resolve a Spanish-named package to a
        # pipeline whose own metadata claims English. Both signals must agree.
        return SimpleNamespace(lang="en")

    spacy_module.load = load  # type: ignore[attr-defined]
    spacy_module.util = SimpleNamespace(is_package=lambda _model: True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", spacy_module)

    with pytest.raises(ProjectError, match="model identifier language"):
        load_spacy("en", "es_core_news_sm")

    assert load_calls == []


def test_spacy_backend_maps_the_root_token_to_a_root_head():
    root = SimpleNamespace(
        i=1,
        text="runs",
        lemma_="run",
        pos_="VERB",
        morph="VerbForm=Fin",
        dep_="ROOT",
        idx=6,
        is_space=False,
    )
    root.head = root
    subject = SimpleNamespace(
        i=0,
        text="She",
        lemma_="she",
        pos_="PRON",
        morph="Case=Nom",
        dep_="nsubj",
        idx=0,
        is_space=False,
        head=root,
    )

    class Pipeline:
        def __call__(self, text: str):
            assert text == "She runs"
            return [subject, root]

    tokens = SpacyBackend("en", "fixture-en", Pipeline()).parse("She runs")

    assert tokens[0].dependency_head == 1
    assert tokens[1].dependency_relation == "ROOT"
    assert tokens[1].dependency_head is None


@pytest.mark.parametrize(
    ("language", "model"),
    [("en", "en_core_web_sm"), ("es", "es_core_news_sm")],
)
def test_installed_spacy_models_pass_the_real_local_probe(monkeypatch, language, model):
    try:
        distribution("spacy")
        distribution(model)
    except PackageNotFoundError:
        pytest.skip(f"local spaCy model {model} is not installed")

    module_name = model.replace("-", "_")
    _reset_verified_modules(monkeypatch, "spacy", module_name)
    capabilities = inspect_local_backend(language, backend="spacy", model=model)
    statuses = {item.capability: item for item in capabilities}

    for name in ("sentence_segmentation", "tokenization", "lemma", "pos", "dependency_parse"):
        assert statuses[name].operational_status == "SUPPORTED"
        assert statuses[name].assurance_status == "TESTED"


def test_explicit_local_stanza_package_passes_the_real_probe_without_downloads(monkeypatch):
    try:
        distribution("stanza")
    except PackageNotFoundError:
        pytest.skip("Stanza is not installed")
    model_dir = os.environ.get("DISCOURSELENS_TEST_STANZA_RESOURCES")
    if not model_dir or not (Path(model_dir) / "resources.json").is_file():
        pytest.skip("set DISCOURSELENS_TEST_STANZA_RESOURCES to an existing local resource dir")

    _reset_verified_modules(
        monkeypatch, "stanza", "stanza.pipeline", "stanza.pipeline.core"
    )
    capabilities = inspect_local_backend("ar", backend="stanza", model="ar-padt", model_dir=model_dir)
    statuses = {item.capability: item for item in capabilities}

    assert statuses["dependency_parse"].operational_status == "SUPPORTED"
    assert statuses["dependency_parse"].assurance_status == "TESTED"
    # This explicit Stanza package does not provide an NER component. The
    # separate `default` package in the feasibility spike is a different route.
    assert statuses["named_entities"].operational_status == "UNSUPPORTED"


def test_stanza_backend_does_not_substitute_surface_for_missing_lemma():
    class Pipeline:
        def __init__(self):
            self.processors = {
                "tokenize": object(),
                "pos": object(),
                "lemma": object(),
                "depparse": object(),
            }

        def __call__(self, text: str):
            words = [
                SimpleNamespace(
                    id=index,
                    text=match.group(),
                    lemma=None,
                    upos="NOUN",
                    feats="Case=Nom",
                    head=0 if index == 1 else 1,
                    deprel="root" if index == 1 else "dep",
                    start_char=match.start(),
                    end_char=match.end(),
                )
                for index, match in enumerate(re.finditer(r"\w+|[^\w\s]", text), start=1)
            ]
            return SimpleNamespace(sentences=[SimpleNamespace(words=words)])

    backend = StanzaBackend("en", "fixture-en", Pipeline())
    tokens = backend.parse("Officials may discuss migration policy.")

    assert tokens
    assert all(token.lemma == "" for token in tokens)
    capabilities = {item.capability: item for item in probe_backend_capabilities(backend)}
    assert capabilities["lemma"].operational_status == "LIMITED"


def test_stanza_backend_requires_canonical_offsets_for_every_word():
    text = "Café migration"
    words = [
        SimpleNamespace(
            id=1,
            text="Café",
            lemma="café",
            upos="NOUN",
            feats=None,
            head=0,
            deprel="root",
            start_char=0,
            end_char=4,
        ),
        SimpleNamespace(
            id=2,
            text="migration",
            lemma="migration",
            upos="NOUN",
            feats=None,
            head=1,
            deprel="dep",
            start_char=None,
            end_char=None,
        ),
    ]

    class Pipeline:
        def __call__(self, source: str):
            assert source == text
            return SimpleNamespace(sentences=[SimpleNamespace(words=words)])

    backend = StanzaBackend("en", "fixture-en", Pipeline())

    with pytest.raises(ProjectError, match="offsets that do not match the source text"):
        backend.parse(text)


def test_stanza_backend_preserves_unicode_codepoint_offsets():
    text = "🧭 Cafe\u0301 migration"
    words = [
        SimpleNamespace(
            id=1,
            text="🧭",
            lemma="🧭",
            upos="SYM",
            feats=None,
            head=2,
            deprel="dep",
            start_char=0,
            end_char=1,
        ),
        SimpleNamespace(
            id=2,
            text="Cafe\u0301",
            lemma="café",
            upos="NOUN",
            feats="Case=Nom",
            head=0,
            deprel="root",
            start_char=2,
            end_char=7,
        ),
        SimpleNamespace(
            id=1,
            text="migration",
            lemma="migration",
            upos="NOUN",
            feats=None,
            head=0,
            deprel="root",
            start_char=8,
            end_char=17,
        ),
    ]

    class Pipeline:
        def __call__(self, source: str):
            assert source == text
            return SimpleNamespace(
                sentences=[
                    SimpleNamespace(words=words[:2]),
                    SimpleNamespace(words=words[2:]),
                ]
            )

    tokens = StanzaBackend("en", "fixture-en", Pipeline()).parse(text)

    assert [(token.char_start, token.char_end) for token in tokens] == [
        (0, 1),
        (2, 7),
        (8, 17),
    ]
    assert all(text[token.char_start : token.char_end] == token.surface for token in tokens)
    assert [token.dependency_head for token in tokens] == [1, None, None]
