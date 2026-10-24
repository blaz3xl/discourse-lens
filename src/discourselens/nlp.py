"""Optional local NLP backends with explicit language/model selection.

No model is downloaded automatically.  A caller must select a local backend and
model that has already been installed, which keeps import and corpus workflows
offline by default and prevents a silent quality downgrade.
"""

from __future__ import annotations

import os
import site
import sys
import sysconfig
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib.machinery import ModuleSpec, PathFinder
from importlib.metadata import Distribution, PackageNotFoundError
from importlib.util import module_from_spec
from pathlib import Path
from types import ModuleType
from typing import Protocol

from .errors import OptionalDependencyError, ProjectError

# Import resolution is not a Python sandbox. This registry only distinguishes
# modules installed through this verified loader from pre-populated sys.modules
# entries that can lie about __file__ and __spec__.origin.
_VERIFIED_MODULES: dict[str, tuple[ModuleType, Path]] = {}
_VERIFIED_EXECUTED: set[str] = set()


def _module_origin_values(module: ModuleType) -> tuple[str, str] | None:
    """Read module origin fields without invoking module-level ``__getattr__``."""
    try:
        namespace = ModuleType.__getattribute__(module, "__dict__")
        file_origin = namespace.get("__file__")
        module_spec = namespace.get("__spec__")
        if type(module_spec) is not ModuleSpec:
            return None
        spec_namespace = object.__getattribute__(module_spec, "__dict__")
        spec_origin = spec_namespace.get("origin")
    except Exception:  # noqa: BLE001 - origin metadata is not a diagnostic channel.
        return None
    if not isinstance(file_origin, str) or not isinstance(spec_origin, str):
        return None
    return file_origin, spec_origin


def _approved_install_roots() -> tuple[Path, ...]:
    """Return interpreter-owned package roots, never arbitrary sys.path entries."""
    candidates: list[str] = []
    try:
        site_packages = site.getsitepackages()
    except Exception:  # noqa: BLE001 - site configuration errors fail closed at use.
        site_packages = []
    candidates.extend(path for path in site_packages if isinstance(path, str))
    if site.ENABLE_USER_SITE is True:
        try:
            user_site = site.getusersitepackages()
        except Exception:  # noqa: BLE001 - site configuration errors fail closed at use.
            user_site = None
        if isinstance(user_site, str):
            candidates.append(user_site)
    for key in ("purelib", "platlib"):
        configured = sysconfig.get_path(key)
        if isinstance(configured, str):
            candidates.append(configured)

    roots: list[Path] = []
    for candidate in candidates:
        try:
            root = Path(os.path.normcase(str(Path(candidate).resolve(strict=True))))
        except OSError:
            continue
        if root.is_dir() and root not in roots:
            roots.append(root)
    return tuple(roots)


def _require_approved_install_path(path: object, display_name: str) -> Path:
    try:
        resolved = Path(os.path.normcase(str(Path(str(path)).resolve(strict=True))))
    except Exception:  # noqa: BLE001 - metadata paths are untrusted diagnostics.
        raise ProjectError(f"{display_name} is outside approved interpreter package roots") from None
    if not any(root == resolved or root in resolved.parents for root in _approved_install_roots()):
        raise ProjectError(f"{display_name} is outside approved interpreter package roots")
    return resolved


@dataclass(frozen=True)
class ParsedToken:
    surface: str
    lemma: str
    pos: str
    morphology: str | None
    dependency_head: int | None
    dependency_relation: str | None
    char_start: int
    char_end: int


class NlpBackend(Protocol):
    @property
    def language(self) -> str: ...

    @property
    def backend_id(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def parse(self, text: str) -> list[ParsedToken]: ...


@dataclass(frozen=True)
class SpacyBackend:
    language: str
    model_id: str
    pipeline: object
    backend_id: str = "spacy"

    def parse(self, text: str) -> list[ParsedToken]:
        # spaCy is optional, so keep its types out of the production import path.
        document = self.pipeline(text)  # type: ignore[operator]
        tokens = [token for token in document if not token.is_space]
        index_by_spacy_token = {token.i: index for index, token in enumerate(tokens)}
        return [
            ParsedToken(
                surface=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                morphology=str(token.morph) or None,
                dependency_head=(
                    None
                    if token.dep_.casefold() == "root"
                    else index_by_spacy_token.get(token.head.i)
                ),
                dependency_relation=token.dep_ or None,
                char_start=token.idx,
                char_end=token.idx + len(token.text),
            )
            for token in tokens
        ]


@dataclass(frozen=True)
class StanzaBackend:
    language: str
    model_id: str
    pipeline: object
    backend_id: str = "stanza"

    def parse(self, text: str) -> list[ParsedToken]:
        document = self.pipeline(text)  # type: ignore[operator]
        result: list[ParsedToken] = []
        offset = 0
        for sentence in document.sentences:
            words = sentence.words
            by_stanza_id = {word.id: offset + index for index, word in enumerate(words)}
            for index, word in enumerate(words):
                start, end = _canonical_token_span(
                    text,
                    word.text,
                    word.start_char,
                    word.end_char,
                    backend="Stanza",
                )
                result.append(
                    ParsedToken(
                        surface=word.text,
                        # Preserve an unavailable annotation as unavailable. Falling
                        # back to the surface makes capability probes report a
                        # lemmatizer even when the selected Stanza package omitted
                        # that processor.
                        lemma=word.lemma or "",
                        pos=word.upos or "",
                        morphology=word.feats,
                        dependency_head=by_stanza_id.get(word.head),
                        dependency_relation=word.deprel,
                        char_start=start,
                        char_end=end,
                    )
                )
            offset += len(words)
        return result


def _canonical_token_span(
    text: str,
    surface: str,
    start: object,
    end: object,
    *,
    backend: str,
) -> tuple[int, int]:
    """Require token offsets to identify the exact surface in the parsed text."""
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end <= start
        or end > len(text)
        or text[start:end] != surface
    ):
        raise ProjectError(f"{backend} returned token offsets that do not match the source text")
    return start, end


def _spacy_model_language_code(model_id: str) -> str | None:
    """Extract a language prefix only from conventional spaCy package names."""
    model_name = model_id.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].casefold()
    for marker in ("_core_", "_dep_", "_ent_wiki_", "_sent_ud_"):
        if marker in model_name:
            language_code = model_name.split(marker, 1)[0]
            if language_code.isalpha() and len(language_code) in {2, 3}:
                return language_code
    return None


def _installed_distribution(
    distribution_name: str, display_name: str, *, optional_dependency: bool = False
) -> Distribution:
    try:
        distribution = importlib_metadata.distribution(distribution_name)
        _require_approved_install_path(
            distribution.locate_file(""), f"{display_name} distribution"
        )
        return distribution
    except PackageNotFoundError:
        if optional_dependency:
            raise OptionalDependencyError(
                f"{display_name} backend requires discourselens[nlp]"
            ) from None
        raise ProjectError(f"{display_name} is not installed as a package") from None
    except ProjectError:
        raise
    except Exception:  # noqa: BLE001 - metadata failures may include local paths.
        raise ProjectError(f"{display_name} package origin could not be verified") from None


def _optional_nlp_import_error(display_name: str) -> OptionalDependencyError:
    backend = "spaCy" if display_name.startswith("spaCy") else "Stanza"
    return OptionalDependencyError(f"{backend} backend requires discourselens[nlp]")


def _require_spacy_model_entry_point(
    distribution: Distribution, model_module: str
) -> None:
    """Require spaCy's non-executing package marker before importing a model.

    spaCy's `is_package()` checks whether any distribution with this name is
    installed. Model packages generated by spaCy instead advertise themselves
    through an exact `spacy_models` entry point.
    """
    try:
        matches = [
            entry_point
            for entry_point in distribution.entry_points
            if entry_point.group == "spacy_models"
            and entry_point.name == model_module
            and entry_point.value == model_module
        ]
    except Exception:  # noqa: BLE001 - parse metadata without importing its target.
        raise ProjectError("spaCy model distribution metadata could not be verified") from None
    if len(matches) != 1:
        raise ProjectError("selected distribution is not registered as a spaCy model")


def _verified_distribution_spec(
    module_name: str,
    distribution: Distribution,
    distribution_path: str,
    display_name: str,
    *,
    search_path: list[str] | None = None,
) -> tuple[ModuleSpec, Path]:
    distribution_files = distribution.files
    if distribution_files is None or distribution_path not in {
        Path(str(item)).as_posix() for item in distribution_files
    }:
        raise ProjectError(f"{display_name} distribution does not declare the required module file")
    try:
        spec = PathFinder.find_spec(module_name, search_path)
        if spec is None or not isinstance(spec.origin, str):
            raise ProjectError(f"{display_name} package origin could not be verified")
        expected_origin = _require_approved_install_path(
            distribution.locate_file(distribution_path), display_name
        )
        actual_origin = Path(str(spec.origin)).resolve(strict=True)
    except ProjectError:
        raise
    except Exception:  # noqa: BLE001 - import-resolution failures may include host paths.
        raise ProjectError(f"{display_name} package origin could not be verified") from None
    if actual_origin != expected_origin:
        raise ProjectError(f"{display_name} module does not originate from its installed distribution")
    return spec, expected_origin


def _cached_module_with_verified_origin(
    module_name: str, expected_origin: Path, display_name: str
) -> ModuleType | None:
    if module_name in sys.modules:
        existing = sys.modules[module_name]
        if not isinstance(existing, ModuleType):
            raise ProjectError(f"{display_name} module is already loaded from an unknown origin")
        provenance = _VERIFIED_MODULES.get(module_name)
        if (
            provenance is None
            or provenance[0] is not existing
            or provenance[1] != expected_origin
            or module_name not in _VERIFIED_EXECUTED
        ):
            raise ProjectError(f"{display_name} module is already loaded from an unknown origin")
        existing_origins = _module_origin_values(existing)
        if existing_origins is None:
            raise ProjectError(f"{display_name} module is already loaded from an unknown origin")
        try:
            if any(Path(origin).resolve(strict=True) != expected_origin for origin in existing_origins):
                raise ProjectError(f"{display_name} module is already loaded from an unknown origin")
        except ProjectError:
            raise
        except Exception:  # noqa: BLE001 - cached module paths are untrusted diagnostics.
            raise ProjectError(f"{display_name} module is already loaded from an unknown origin") from None
        return existing
    return None


def _prepare_verified_spec(
    spec: ModuleSpec, expected_origin: Path, display_name: str
) -> ModuleType:
    module_name = spec.name
    if module_name in sys.modules:
        existing = sys.modules[module_name]
        provenance = _VERIFIED_MODULES.get(module_name)
        if (
            isinstance(existing, ModuleType)
            and provenance is not None
            and provenance[0] is existing
            and provenance[1] == expected_origin
            and module_name not in _VERIFIED_EXECUTED
        ):
            return existing
        cached = _cached_module_with_verified_origin(module_name, expected_origin, display_name)
        assert cached is not None
        return cached
    module: ModuleType | None = None
    try:
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        _VERIFIED_MODULES[module_name] = (module, expected_origin)
    except Exception:  # noqa: BLE001 - import setup errors may include source paths.
        if module is not None and sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        _VERIFIED_MODULES.pop(module_name, None)
        _VERIFIED_EXECUTED.discard(module_name)
        raise ProjectError(f"{display_name} package could not be prepared safely") from None
    assert module is not None
    return module


def _forget_verified_module(module_name: str, module: ModuleType) -> None:
    if sys.modules.get(module_name) is module:
        del sys.modules[module_name]
    _VERIFIED_MODULES.pop(module_name, None)
    _VERIFIED_EXECUTED.discard(module_name)


def _execute_verified_spec(
    spec: ModuleSpec,
    module: ModuleType,
    expected_origin: Path,
    display_name: str,
    *,
    optional_dependency: bool = False,
) -> ModuleType:
    if spec.loader is None:
        raise ProjectError(f"{display_name} package has no import loader")
    if sys.modules.get(spec.name) is not module:
        _forget_verified_module(spec.name, module)
        raise ProjectError(f"{display_name} module changed during verified import")
    provenance = _VERIFIED_MODULES.get(spec.name)
    if provenance is None or provenance[0] is not module or provenance[1] != expected_origin:
        _forget_verified_module(spec.name, module)
        raise ProjectError(f"{display_name} module changed during verified import")

    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        _forget_verified_module(spec.name, module)
        if optional_dependency:
            raise _optional_nlp_import_error(display_name) from None
        raise ProjectError(f"{display_name} package could not be imported safely") from None
    except Exception:  # noqa: BLE001 - package import errors may include source paths.
        _forget_verified_module(spec.name, module)
        raise ProjectError(f"{display_name} package could not be imported safely") from None

    try:
        loaded_origins = _module_origin_values(module)
        if loaded_origins is None or any(
            Path(origin).resolve(strict=True) != expected_origin for origin in loaded_origins
        ):
            raise ProjectError(f"{display_name} module changed its verified origin during import")
        _VERIFIED_EXECUTED.add(spec.name)
    except ProjectError:
        _forget_verified_module(spec.name, module)
        raise
    except Exception:  # noqa: BLE001 - package import errors may include source paths.
        _forget_verified_module(spec.name, module)
        raise ProjectError(f"{display_name} package could not be imported safely") from None
    return module


def _load_verified_spec(
    spec: ModuleSpec,
    expected_origin: Path,
    display_name: str,
    *,
    optional_dependency: bool = False,
) -> ModuleType:
    module = _prepare_verified_spec(spec, expected_origin, display_name)
    if spec.name in _VERIFIED_EXECUTED:
        return module
    return _execute_verified_spec(
        spec,
        module,
        expected_origin,
        display_name,
        optional_dependency=optional_dependency,
    )


def _load_verified_package_module(
    module_name: str,
    distribution_name: str,
    display_name: str,
    *,
    optional_dependency: bool = False,
) -> ModuleType:
    """Load a package only when its module origin matches its installed distribution."""
    distribution = _installed_distribution(
        distribution_name, display_name, optional_dependency=optional_dependency
    )
    initializer = f"{module_name}/__init__.py"
    spec, expected_origin = _verified_distribution_spec(
        module_name, distribution, initializer, display_name
    )
    return _load_verified_spec(
        spec,
        expected_origin,
        display_name,
        optional_dependency=optional_dependency,
    )


def _load_verified_stanza_modules() -> tuple[ModuleType, ModuleType]:
    """Preflight and load Stanza and its DownloadMethod module from one distribution."""
    distribution = _installed_distribution("stanza", "Stanza", optional_dependency=True)
    stanza_spec, stanza_origin = _verified_distribution_spec(
        "stanza", distribution, "stanza/__init__.py", "Stanza"
    )
    stanza_paths = stanza_spec.submodule_search_locations
    if stanza_paths is None:
        raise ProjectError("Stanza package origin could not be verified")
    stanza_search_path = list(stanza_paths)
    pipeline_spec, pipeline_origin = _verified_distribution_spec(
        "stanza.pipeline",
        distribution,
        "stanza/pipeline/__init__.py",
        "Stanza pipeline",
        search_path=stanza_search_path,
    )
    pipeline_paths = pipeline_spec.submodule_search_locations
    if pipeline_paths is None:
        raise ProjectError("Stanza pipeline origin could not be verified")
    core_spec, core_origin = _verified_distribution_spec(
        "stanza.pipeline.core",
        distribution,
        "stanza/pipeline/core.py",
        "Stanza pipeline core",
        search_path=list(pipeline_paths),
    )

    # Check all cached origins before executing the Stanza package: its initializer
    # may itself import the pipeline modules.
    _cached_module_with_verified_origin("stanza", stanza_origin, "Stanza")
    _cached_module_with_verified_origin("stanza.pipeline", pipeline_origin, "Stanza pipeline")
    _cached_module_with_verified_origin("stanza.pipeline.core", core_origin, "Stanza pipeline core")
    specs = (
        (stanza_spec, stanza_origin, "Stanza"),
        (pipeline_spec, pipeline_origin, "Stanza pipeline"),
        (core_spec, core_origin, "Stanza pipeline core"),
    )
    new_modules: list[tuple[str, ModuleType]] = []
    modules: dict[str, ModuleType] = {}
    try:
        for spec, expected_origin, display_name in specs:
            was_cached = spec.name in sys.modules
            module = _prepare_verified_spec(spec, expected_origin, display_name)
            modules[spec.name] = module
            if not was_cached:
                new_modules.append((spec.name, module))
        # Prepare every verified package before running initializers: stanza's root
        # initializer imports stanza.pipeline.core through Python's normal importer.
        for name, expected_origin, display_name in (
            ("stanza.pipeline.core", core_origin, "Stanza pipeline core"),
            ("stanza.pipeline", pipeline_origin, "Stanza pipeline"),
            ("stanza", stanza_origin, "Stanza"),
        ):
            spec = next(spec for spec, _, _ in specs if spec.name == name)
            if name not in _VERIFIED_EXECUTED:
                _execute_verified_spec(
                    spec,
                    modules[name],
                    expected_origin,
                    display_name,
                    optional_dependency=True,
                )
    except Exception:
        for name, module in new_modules:
            if sys.modules.get(name) is module:
                del sys.modules[name]
            _VERIFIED_MODULES.pop(name, None)
            _VERIFIED_EXECUTED.discard(name)
        raise
    stanza = modules["stanza"]
    core = modules["stanza.pipeline.core"]
    return stanza, core


def load_spacy(language: str, model_id: str) -> SpacyBackend:
    expected_language = language.casefold().split("-")[0].split("_")[0]
    model_language_code = _spacy_model_language_code(model_id)
    if model_language_code is not None and model_language_code != expected_language:
        raise ProjectError(
            "spaCy model identifier language does not match the selected document language"
        )
    model_module = model_id.replace("-", "_")
    if not model_module.isidentifier():
        raise ProjectError("spaCy model identifier does not map to an installed package module")
    # Validate the distribution marker and both filesystem origins before any
    # selected model code can run. spaCy's own is_package() accepts any installed
    # distribution, not only registered pipeline packages.
    try:
        model_distribution = importlib_metadata.distribution(model_id)
    except PackageNotFoundError:
        # An uninstalled conventional backend/model request should explain how to
        # install the optional backend. A same-named CWD package remains a model
        # error and is never considered an installed model.
        if Path.cwd().joinpath(model_module).exists():
            raise ProjectError("spaCy model is not installed as a package") from None
        _load_verified_package_module(
            "spacy", "spacy", "spaCy", optional_dependency=True
        )
        raise ProjectError("spaCy model is not installed as a package") from None
    except Exception:  # noqa: BLE001 - metadata diagnostics may include local paths.
        raise ProjectError("spaCy model package origin could not be verified") from None
    _require_approved_install_path(
        model_distribution.locate_file(""), "spaCy model distribution"
    )
    _require_spacy_model_entry_point(model_distribution, model_module)
    model_spec, model_origin = _verified_distribution_spec(
        model_module,
        model_distribution,
        f"{model_module}/__init__.py",
        "spaCy model",
    )
    spacy = _load_verified_package_module(
        "spacy", "spacy", "spaCy", optional_dependency=True
    )
    _load_verified_spec(model_spec, model_origin, "spaCy model")
    try:
        pipeline = spacy.load(model_module)
    except Exception:  # noqa: BLE001 - model package diagnostics may expose local paths.
        raise ProjectError("installed spaCy model package could not be loaded") from None
    model_language = getattr(pipeline, "lang", None)
    if not isinstance(model_language, str) or not model_language:
        raise ProjectError("spaCy model does not expose a verifiable language")
    if model_language.casefold() != expected_language:
        raise ProjectError("spaCy model language does not match the selected document language")
    return SpacyBackend(language, model_id, pipeline)


def load_stanza(language: str, model_id: str, model_dir: str) -> StanzaBackend:
    stanza, stanza_core = _load_verified_stanza_modules()
    try:
        package = model_id
        for prefix in (f"{language}-", f"{language}_"):
            if package.startswith(prefix):
                package = package[len(prefix) :]
                break
        download_method = stanza_core.DownloadMethod
        pipeline = stanza.Pipeline(
            language,
            model_dir=model_dir,
            package=package,
            processors="tokenize,pos,lemma,depparse,ner",
            download_method=download_method.NONE,
            use_gpu=False,
            verbose=False,
        )
    except Exception:  # noqa: BLE001 - Stanza diagnostics may include local paths.
        raise ProjectError("Stanza model is unavailable; no downloads were attempted") from None
    model_language = getattr(pipeline, "lang", None)
    if not isinstance(model_language, str) or not model_language:
        raise ProjectError("Stanza model does not expose a verifiable language")
    if model_language.casefold() != language.casefold():
        raise ProjectError("Stanza model language does not match the selected document language")
    return StanzaBackend(language, model_id, pipeline)
