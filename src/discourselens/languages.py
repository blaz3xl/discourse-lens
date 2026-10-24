"""Explicit language-capability gating; no capability is assumed supported."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from .errors import ProjectError
from .nlp import NlpBackend

OPERATIONAL_SUPPORTED = "SUPPORTED"
OPERATIONAL_LIMITED = "LIMITED"
OPERATIONAL_UNSUPPORTED = "UNSUPPORTED"
OPERATIONAL_UNASSESSED = "UNASSESSED"
ASSURANCE_TESTED = "TESTED"
ASSURANCE_UNASSESSED = "UNASSESSED"


@dataclass(frozen=True)
class LanguageCapability:
    """One concrete local backend capability, not a claim of CDS validation."""

    language: str
    capability: str
    backend: str | None
    model: str | None
    operational_status: str
    assurance_status: str
    limitation: str | None = None


LOCAL_SPACY_MODELS = {"en": "en_core_web_sm", "es": "es_core_news_sm"}
CAPABILITY_PROBE_SAMPLES = {
    "en": "Officials may discuss migration policy. Researchers reviewed the text.",
    "es": "Las autoridades podrían revisar la política migratoria. Las investigadoras analizaron el texto.",
    "ar": "قد تناقش السلطات سياسة الهجرة. راجع الباحثون النص.",
}
NLP_CAPABILITIES = (
    "sentence_segmentation",
    "tokenization",
    "lemma",
    "pos",
    "morphology",
    "dependency_parse",
    "named_entities",
)
LANGUAGE_INDEPENDENT_CAPABILITIES = ("exact_search", "surface_frequency", "kwic")
KNOWN_CAPABILITIES = frozenset(NLP_CAPABILITIES + LANGUAGE_INDEPENDENT_CAPABILITIES)


@dataclass(frozen=True)
class CapabilityDecision:
    runnable: bool
    limitations: tuple[str, ...]


class LocalPreflight(TypedDict):
    runnable: bool
    limitations: list[str]
    required_capabilities: list[str]
    statuses: dict[str, str]
    backend: str | None
    model: str | None
    model_dir_env: str | None
    route_source: str


def evaluate_capabilities(statuses: dict[str, str], required: set[str]) -> CapabilityDecision:
    missing = [name for name in sorted(required) if statuses.get(name) != "SUPPORTED"]
    return CapabilityDecision(
        not missing, tuple(f"{name}: {statuses.get(name, 'UNASSESSED')}" for name in missing)
    )


def _has_complete_tokenization(backend_tokens: Sequence[object], sample: str) -> bool:
    """Check that probe tokens cover the fixture once, in source order."""
    if not backend_tokens:
        return False

    covered = [False] * len(sample)
    previous_end = 0
    for token in backend_tokens:
        start = getattr(token, "char_start", None)
        end = getattr(token, "char_end", None)
        surface = getattr(token, "surface", None)
        if (
            type(start) is not int
            or type(end) is not int
            or not isinstance(surface, str)
            or start < previous_end
            or start < 0
            or end <= start
            or end > len(sample)
            or sample[start:end] != surface
            or any(char.isspace() for char in sample[start:end])
        ):
            return False
        for index in range(start, end):
            if covered[index]:
                return False
            covered[index] = True
        previous_end = end
    return all(char.isspace() or covered[index] for index, char in enumerate(sample))


def _has_valid_dependency_tree(tokens: Sequence[object]) -> bool:
    """Check basic dependency structure without making a linguistic-quality claim."""
    if not tokens:
        return False

    parents: list[int | None] = []
    for index, token in enumerate(tokens):
        relation = getattr(token, "dependency_relation", None)
        head = getattr(token, "dependency_head", None)
        if not isinstance(relation, str) or not relation.strip():
            return False
        if relation.casefold() == "root":
            if head is not None:
                return False
            parents.append(None)
        elif type(head) is int and 0 <= head < len(tokens) and head != index:
            parents.append(head)
        else:
            return False

    if not any(parent is None for parent in parents):
        return False
    for index in range(len(parents)):
        visited: set[int] = set()
        current: int | None = index
        while current is not None:
            if current in visited:
                return False
            visited.add(current)
            current = parents[current]
    return True


def local_language_capabilities(language: str) -> tuple[LanguageCapability, ...]:
    """Probe the configured default spaCy model locally; never download or use network.

    The result concerns mechanical NLP preprocessing. It deliberately does not
    mark any CDS framework skill as methodologically validated.
    """
    normalized = language.casefold()
    base = [
        LanguageCapability(
            normalized,
            capability,
            None,
            None,
            OPERATIONAL_SUPPORTED,
            ASSURANCE_TESTED,
            "Language-independent deterministic text operation.",
        )
        for capability in LANGUAGE_INDEPENDENT_CAPABILITIES
    ]
    model = LOCAL_SPACY_MODELS.get(normalized)
    if not model:
        return tuple(
            base
            + [
                LanguageCapability(
                    normalized,
                    capability,
                    None,
                    None,
                    OPERATIONAL_UNASSESSED,
                    ASSURANCE_UNASSESSED,
                    "No maintained local NLP model is configured for this language.",
                )
                for capability in NLP_CAPABILITIES
            ]
        )
    try:
        from .nlp import load_spacy

        selected = load_spacy(normalized, model)
    except ProjectError as exc:
        return tuple(
            base
            + [
                LanguageCapability(
                    normalized,
                    capability,
                    "spacy",
                    model,
                    OPERATIONAL_UNASSESSED,
                    ASSURANCE_UNASSESSED,
                    str(exc),
                )
                for capability in NLP_CAPABILITIES
            ]
        )
    try:
        return probe_backend_capabilities(selected)
    except ValueError as exc:
        return tuple(
            base
            + [
                LanguageCapability(
                    normalized,
                    capability,
                    "spacy",
                    model,
                    OPERATIONAL_UNASSESSED,
                    ASSURANCE_UNASSESSED,
                    str(exc),
                )
                for capability in NLP_CAPABILITIES
            ]
        )


def probe_backend_capabilities(backend: NlpBackend) -> tuple[LanguageCapability, ...]:
    """Run a tiny local parse and report observable backend capabilities.

    This measures whether the selected backend exposes usable token annotations;
    it is not a linguistic-quality evaluation or validation of CDA methodology.
    """
    language = backend.language.casefold().split("-")[0].split("_")[0]
    sample = CAPABILITY_PROBE_SAMPLES.get(language)
    if sample is None:
        base = [
            LanguageCapability(
                language,
                capability,
                None,
                None,
                OPERATIONAL_SUPPORTED,
                ASSURANCE_TESTED,
                "Language-independent deterministic text operation.",
            )
            for capability in LANGUAGE_INDEPENDENT_CAPABILITIES
        ]
        return tuple(
            base
            + [
                LanguageCapability(
                    language,
                    capability,
                    backend.backend_id,
                    backend.model_id,
                    OPERATIONAL_UNASSESSED,
                    ASSURANCE_UNASSESSED,
                    "No language-matched local capability fixture is configured.",
                )
                for capability in NLP_CAPABILITIES
            ]
        )
    try:
        tokens = backend.parse(sample)
    except Exception as exc:
        raise ValueError(f"selected local NLP backend failed its capability probe: {exc}") from exc
    base = [
        LanguageCapability(
            backend.language.casefold(),
            capability,
            None,
            None,
            OPERATIONAL_SUPPORTED,
            ASSURANCE_TESTED,
            "Language-independent deterministic text operation.",
        )
        for capability in LANGUAGE_INDEPENDENT_CAPABILITIES
    ]
    tokenization_ok = _has_complete_tokenization(tokens, sample)
    pipe_names: set[str] = set()
    component_inventory_available = False
    pipeline = getattr(backend, "pipeline", None)
    raw_pipes = getattr(pipeline, "pipe_names", None)
    if isinstance(raw_pipes, (list, tuple, set)):
        component_inventory_available = True
        pipe_names.update(str(item) for item in raw_pipes)
    processors = getattr(pipeline, "processors", None)
    if isinstance(processors, dict):
        component_inventory_available = True
        pipe_names.update(str(item) for item in processors)
    model = backend.model_id
    fields = {
        "tokenization": tokenization_ok,
        "lemma": tokenization_ok and all(bool(token.lemma.strip()) for token in tokens),
        "pos": tokenization_ok and all(bool(token.pos.strip()) for token in tokens),
        "morphology": tokenization_ok and any(token.morphology for token in tokens),
        "dependency_parse": tokenization_ok and _has_valid_dependency_tree(tokens),
        "sentence_segmentation": tokenization_ok
        and bool(
            pipe_names & {"parser", "senter", "sentencizer"}
            or (backend.backend_id == "stanza" and "tokenize" in pipe_names)
        ),
        "named_entities": tokenization_ok and bool(pipe_names & {"ner"}),
    }
    component_requirements = {
        "sentence_segmentation": {"parser", "senter", "sentencizer", "tokenize"},
        "tokenization": {"tokenize"},
        "lemma": {"lemmatizer", "lemma"},
        "pos": {"tagger", "morphologizer", "pos"},
        "morphology": {"tagger", "morphologizer", "pos"},
        "dependency_parse": {"parser", "depparse"},
        "named_entities": {"ner"},
    }
    backend_components = {
        "spacy": {
            "sentence_segmentation": {"parser", "senter", "sentencizer"},
            "tokenization": set(),  # spaCy's tokenizer is intrinsic to the pipeline.
            "lemma": {"lemmatizer"},
            "pos": {"tagger", "morphologizer"},
            "morphology": {"tagger", "morphologizer"},
            "dependency_parse": {"parser"},
            "named_entities": {"ner"},
        },
        "stanza": component_requirements,
    }
    known_requirements = backend_components.get(backend.backend_id.casefold())
    result = list(base)
    for capability in NLP_CAPABILITIES:
        available = fields[capability]
        required_components = (
            known_requirements.get(capability, set())
            if known_requirements is not None
            else set()
        )
        # If a known backend exposes its component inventory, distinguish an
        # absent component from a component that failed to produce an annotation
        # for this probe. For other backend implementations, observed output is
        # the only defensible capability signal.
        missing_component = (
            component_inventory_available
            and bool(required_components)
            and not (pipe_names & required_components)
        )
        if missing_component:
            status = OPERATIONAL_UNSUPPORTED
            limitation = "The selected backend/model does not include a required pipeline component."
        elif available:
            status = OPERATIONAL_SUPPORTED
            limitation = None
        elif not tokenization_ok and capability in {
            "tokenization",
            "lemma",
            "pos",
            "morphology",
            "dependency_parse",
            "sentence_segmentation",
            "named_entities",
        }:
            status = OPERATIONAL_LIMITED
            limitation = "The selected backend did not expose complete token coverage in the local probe."
        else:
            status = OPERATIONAL_LIMITED
            limitation = "The selected backend did not expose this annotation in the local probe."
        result.append(
            LanguageCapability(
                language,
                capability,
                backend.backend_id,
                model,
                status,
                ASSURANCE_TESTED,
                limitation,
            )
        )
    return tuple(result)


def inspect_local_backend(
    language: str,
    *,
    backend: str,
    model: str,
    model_dir: str | None = None,
) -> tuple[LanguageCapability, ...]:
    """Load and probe one explicitly selected, already-installed local NLP backend."""
    from .nlp import load_spacy, load_stanza

    normalized = language.casefold()
    if not model.strip():
        raise ValueError("an explicit local NLP model identifier is required")
    if backend == "spacy":
        if model_dir is not None:
            raise ValueError("--model-dir is only applicable to Stanza")
        selected = load_spacy(normalized, model)
    elif backend == "stanza":
        if model_dir is None:
            raise ValueError("an explicit local Stanza --model-dir is required")
        selected = load_stanza(normalized, model, model_dir)
    else:
        raise ValueError("backend must be spacy or stanza")
    if (
        selected.language.casefold() != normalized
        or selected.backend_id != backend
        or selected.model_id != model
    ):
        raise ValueError("loaded backend identity does not match the explicit selection")
    return probe_backend_capabilities(selected)


def capability_statuses(capabilities: tuple[LanguageCapability, ...]) -> dict[str, str]:
    """Convert a profile to the strict eligibility input expected by skills."""
    return {item.capability: item.operational_status for item in capabilities}


def local_execution_preflight(
    languages: tuple[str, ...],
    required_capabilities: set[str],
    routes: Mapping[str, Mapping[str, str | None]] | None = None,
) -> dict[str, LocalPreflight]:
    """Evaluate explicit local preprocessing requirements without changing execution state.

    The result is designed for immutable AnalysisRun provenance: it records the
    concrete capability status observed when a profile was started. It does not
    claim theoretical validation of any CDS interpretation.
    """
    result: dict[str, LocalPreflight] = {}
    selected_routes = {
        language.casefold().strip(): route for language, route in (routes or {}).items()
    }
    for normalized_language in sorted({language.casefold().strip() for language in languages}):
        language = normalized_language
        route = selected_routes.get(normalized_language)
        route_error: str | None = None
        if route is None:
            capabilities = local_language_capabilities(normalized_language)
            nlp_capability = next(
                (item for item in capabilities if item.capability in NLP_CAPABILITIES), None
            )
            backend = nlp_capability.backend if nlp_capability is not None else None
            model = nlp_capability.model if nlp_capability is not None else None
            model_dir_env = None
            route_source = "default"
        else:
            backend = route["backend"]
            model = route["model"]
            model_dir_env = route.get("model_dir_env")
            route_source = "profile"
            model_dir = os.environ.get(model_dir_env) if model_dir_env else None
            if backend == "stanza" and not model_dir:
                route_error = (
                    f"selected stanza route requires environment variable {model_dir_env!r}"
                )
                capabilities = ()
            else:
                try:
                    capabilities = inspect_local_backend(
                        normalized_language,
                        backend=str(backend),
                        model=str(model),
                        model_dir=model_dir,
                    )
                except (ProjectError, ValueError, OSError):
                    # Provider paths and local install details may contain private absolute
                    # locations. Keep run provenance useful without persisting those values.
                    route_error = (
                        "selected local NLP route could not be loaded; verify the optional "
                        "dependency, installed model, and local resource configuration"
                    )
                    capabilities = ()
        if route_error is not None:
            capabilities = tuple(
                LanguageCapability(
                    normalized_language,
                    capability,
                    str(backend) if backend else None,
                    str(model) if model else None,
                    OPERATIONAL_UNASSESSED,
                    ASSURANCE_UNASSESSED,
                    route_error,
                )
                for capability in NLP_CAPABILITIES
            ) + tuple(
                LanguageCapability(
                    normalized_language,
                    capability,
                    None,
                    None,
                    OPERATIONAL_SUPPORTED,
                    ASSURANCE_TESTED,
                    "Language-independent deterministic text operation.",
                )
                for capability in LANGUAGE_INDEPENDENT_CAPABILITIES
            )
        statuses = capability_statuses(capabilities)
        decision = evaluate_capabilities(statuses, required_capabilities)
        limitations = list(decision.limitations)
        limitations.extend(
            f"{item.capability} assurance: UNASSESSED"
            for item in capabilities
            if item.capability in required_capabilities
            and item.operational_status == OPERATIONAL_SUPPORTED
            and item.assurance_status == ASSURANCE_UNASSESSED
        )
        if route_error is not None:
            limitations.append(f"local NLP route: {route_error}")
        result[language] = {
            # A configured local route is optional unless this operation actually
            # requires an NLP capability. Keep its load failure visible below, but
            # do not let an unused/broken model route block lexical-only analysis.
            # Required NLP capabilities are already UNASSESSED on route failure,
            # so evaluate_capabilities still fails closed for those operations.
            "runnable": decision.runnable,
            "limitations": limitations,
            "required_capabilities": sorted(required_capabilities),
            "statuses": statuses,
            "backend": backend,
            "model": model,
            "model_dir_env": model_dir_env,
            "route_source": route_source,
        }
    return result
