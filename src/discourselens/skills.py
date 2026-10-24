import hashlib
import importlib.resources
import json
import re
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, TypedDict, cast

from .artifact_integrity import verify_project_artifact
from .errors import (
    EvidenceValidationError,
    RetrievalLimitError,
    ScopeViolationError,
    SkillManifestError,
)
from .frameworks import FRAMEWORK_REVIEWS
from .models import EmbeddingProvider
from .project import Project
from .skill_contracts import (
    EVIDENCE_EXTRACTION_FINDING_PHENOMENON,
)
from .types import (
    ClaimType,
    CoverageDescriptor,
    CoverageMode,
    CoverageTerminationState,
    EvidenceRole,
    ExecutionEligibility,
    NegativeResultStatus,
    SkillManifest,
)


class FixtureSkillResult(TypedDict):
    skill_run_id: str
    evidence_ids: list[str]


class TextualDescriptorResult(TypedDict):
    skill_run_id: str
    finding_ids: list[str]
    evidence_ids: list[str]
    coverage_id: str


class CandidateSkillResult(TypedDict):
    skill_run_id: str
    finding_ids: list[str]
    evidence_ids: list[str]


class EvidenceExtractionResult(TypedDict):
    skill_run_id: str
    finding_ids: list[str]
    evidence_ids: list[str]
    coverage_id: str
    population_count: int
    candidate_count: int
    selected_count: int


class VanDijkGroupComparisonCodeCount(TypedDict):
    code_id: str
    label: str
    selected_annotation_count: int


class VanDijkGroupComparisonGroupInput(TypedDict):
    label: str
    annotation_ids: list[str]


class VanDijkGroupComparisonAnnotation(TypedDict):
    annotation_id: str
    code_id: str
    code_label: str
    code_definition: str | None
    span_id: str
    revision_id: str
    start_offset: int
    end_offset: int
    text: str
    evidence_id: str
    citation: str


class VanDijkGroupComparisonGroup(TypedDict):
    label: str
    selected_annotation_count: int
    coding_status: Literal["CODED", "UNKNOWN"]
    code_counts: list[VanDijkGroupComparisonCodeCount]
    annotations: list[VanDijkGroupComparisonAnnotation]


class VanDijkGroupComparisonResult(TypedDict):
    skill_run_id: str
    finding_id: str
    coverage_id: str
    scope_snapshot_id: str
    artifact_type: str
    shared_topic_event: str
    comparability_basis: str
    codebook_id: str
    codebook_name: str
    count_basis: Literal["SELECTED_ANNOTATIONS"]
    claim_semantics: Literal["EXAMPLE_BASED"]
    selected_annotation_count: int
    groups: list[VanDijkGroupComparisonGroup]
    evidence_ids: list[str]
    limitations: list[str]


class QuoteAttributionCandidate(TypedDict):
    candidate_id: str
    revision_id: str
    quote_start: int
    quote_end: int
    opening_marker: str
    closing_marker: str
    quote_evidence_id: str
    context_evidence_id: str
    attribution_cues: list[str]
    attribution_cue_languages: list[str]
    attribution_cue_evidence_ids: list[str]


class QuoteAttributionCandidatesResult(TypedDict):
    skill_run_id: str
    finding_ids: list[str]
    evidence_ids: list[str]
    coverage_id: str
    population_count: int
    candidate_count: int
    attribution_candidate_count: int
    unmatched_marker_count: int
    ambiguous_marker_count: int
    candidates: list[QuoteAttributionCandidate]


class ScopeQuestionResult(TypedDict):
    skill_run_id: str
    finding_id: str | None
    retrieval_run_id: str
    coverage_id: str
    candidate_count: int
    retrieval_mode: Literal["lexical", "hybrid"]


class _EvidenceCandidate(TypedDict):
    revision_id: str
    start: int
    end: int
    cues: list[str]


class _QuotationSpan(TypedDict):
    start: int
    end: int
    opening_marker: str
    closing_marker: str


class _QuotationFrame(TypedDict):
    family: str
    start: int
    opening_marker: str
    group_markers: list[int]
    group_ambiguous: bool


class _AttributionCueMatch(TypedDict):
    language: str
    surface: str
    start: int
    end: int


class _QuotationInventoryItem(TypedDict):
    quote: _QuotationSpan
    context_start: int
    context_end: int
    cues: list[_AttributionCueMatch]
    candidate_id: str
    quote_span_index: int
    context_span_index: int
    cue_span_indices: list[int]


class CounterEvidenceResult(TypedDict):
    skill_run_id: str
    retrieval_run_id: str
    retrieval_hit_ids: list[str]
    candidate_span_ids: list[str]
    coverage_id: str
    coverage_mode: str
    population_count: int
    candidate_count: int
    offset: int
    limit: int
    returned_count: int
    has_more: bool
    query_interpretation: str


class DependencyParseSnapshot(TypedDict):
    """Capability decision pinned to one parsed artifact and its revision."""

    revision_id: str
    artifact_id: str
    language: str | None
    status: str
    backend: str | None
    model: str | None
    limitations: list[str]


@dataclass(frozen=True)
class SkillContext:
    project: Project
    analysis_run_id: str
    scope_id: str
    dependency_parse_preflight: Mapping[str, DependencyParseSnapshot] | None = None


@dataclass(frozen=True)
class SkillDefinition:
    """A reviewed built-in method contract plus its trusted executor reference."""

    manifest: SkillManifest
    executor: Callable[..., object]


@dataclass(frozen=True)
class SkillDependencyPlan:
    """Dependency closure in deterministic topological order and parallel layers."""

    closure: tuple[str, ...]
    layers: tuple[tuple[str, ...], ...]


class SkillRegistry:
    """Explicit registry only; it never imports project-supplied Python or shell code."""

    def __init__(self) -> None:
        self._definitions: dict[str, SkillDefinition] = {}
        self._claim_contracts: dict[str, tuple[str, frozenset[ClaimType]]] = {}

    def register(self, definition: SkillDefinition) -> None:
        skill_id = definition.manifest.id
        if skill_id in self._definitions:
            raise SkillManifestError(f"duplicate built-in skill {skill_id!r}")
        self._definitions[skill_id] = definition
        self._claim_contracts[skill_id] = (
            definition.manifest.version,
            frozenset(definition.manifest.allowed_claims),
        )

    def get(self, skill_id: str) -> SkillDefinition:
        try:
            return self._definitions[skill_id]
        except KeyError as exc:
            raise SkillManifestError(f"unknown built-in skill {skill_id!r}") from exc

    def manifests(self) -> tuple[SkillManifest, ...]:
        return tuple(item.manifest for item in self._definitions.values())

    def plan_dependencies(self, requested_skill_ids: Iterable[str]) -> SkillDependencyPlan:
        """Plan registered skills and their transitive dependencies.

        Skill IDs are sorted within each execution layer, making the result
        independent of registration order and the input iterable's ordering.
        Dependencies may refer to skills registered later; references are
        checked when planning, after registration is complete.
        """
        requested = sorted(set(requested_skill_ids))
        for skill_id in requested:
            self.get(skill_id)

        state: dict[str, int] = {}
        stack: list[str] = []
        closure: set[str] = set()

        def visit(skill_id: str) -> None:
            visit_state = state.get(skill_id, 0)
            if visit_state == 2:
                return
            if visit_state == 1:
                cycle_start = stack.index(skill_id)
                cycle = (*stack[cycle_start:], skill_id)
                raise SkillManifestError(
                    "built-in skill dependency cycle detected: " + " -> ".join(cycle)
                )

            state[skill_id] = 1
            stack.append(skill_id)
            definition = self._definitions[skill_id]
            for dependency_id in sorted(set(definition.manifest.dependencies)):
                if dependency_id not in self._definitions:
                    raise SkillManifestError(
                        f"built-in skill {skill_id!r} depends on unknown built-in skill "
                        f"{dependency_id!r}"
                    )
                visit(dependency_id)
            stack.pop()
            state[skill_id] = 2
            closure.add(skill_id)

        for skill_id in requested:
            visit(skill_id)

        remaining = set(closure)
        completed: set[str] = set()
        layers: list[tuple[str, ...]] = []
        while remaining:
            ready = tuple(
                sorted(
                    skill_id
                    for skill_id in remaining
                    if set(self._definitions[skill_id].manifest.dependencies) <= completed
                )
            )
            if not ready:
                # DFS above reports reachable cycles with their path. Keep this
                # guard so the planner remains safe if graph construction changes.
                raise SkillManifestError("built-in skill dependency cycle detected")
            layers.append(ready)
            completed.update(ready)
            remaining.difference_update(ready)

        return SkillDependencyPlan(
            closure=tuple(skill_id for layer in layers for skill_id in layer),
            layers=tuple(layers),
        )

    def allowed_claims_for(self, skill_id: str, skill_version: str) -> frozenset[ClaimType] | None:
        """Return the registered version's immutable claim contract, if known.

        Unknown IDs remain available to custom skills, whose contracts are
        validated by the general evidence and coverage policy. A registered
        built-in ID with an unknown version cannot borrow the current version's
        claim contract.
        """
        contract = self._claim_contracts.get(skill_id)
        if contract is None:
            return None
        registered_version, allowed_claims = contract
        if skill_version != registered_version:
            raise SkillManifestError(
                f"registered built-in skill {skill_id!r} has version "
                f"{registered_version}, not {skill_version}"
            )
        return allowed_claims


TEXTUAL_DESCRIPTORS_MANIFEST = {
    "id": "builtin.textual-descriptors",
    "version": "1.0.0",
    "framework": "descriptive-text-analysis",
    "execution_kind": "DETERMINISTIC",
    "input_schema_ref": "schemas/textual-descriptors-input-v1.json",
    "output_schema_ref": "schemas/textual-descriptors-output-v1.json",
    "executor": "builtin.textual_descriptors",
    "methodology_refs": ["docs/methodology/textual-descriptors-v1.md"],
    "retrieval_plan_ref": "retrieval/exhaustive-token-pattern-v1",
    "documentation_ref": "docs/methodology/textual-descriptors-v1.md",
    "allowed_claims": ["OBSERVATIONAL"],
    "prohibited_claims": ["author_intention", "psychological_state", "political_affiliation"],
}

EVIDENCE_EXTRACTION_MANIFEST = {
    "id": "builtin.evidence-extraction",
    "version": "1.0.0",
    "framework": "descriptive-evidence-inventory",
    "execution_kind": "DETERMINISTIC",
    "input_schema_ref": "schemas/evidence-extraction-input-v1.json",
    "output_schema_ref": "schemas/evidence-extraction-output-v1.json",
    "executor": "builtin.evidence_extraction",
    "methodology_refs": ["docs/methodology/evidence-extraction-v1.md"],
    "required_capabilities": ["canonical_sentences", "textual_evidence"],
    "retrieval_plan_ref": "retrieval/exhaustive-sentence-candidate-scan-v1",
    "documentation_ref": "docs/methodology/evidence-extraction-v1.md",
    "allowed_claims": ["OBSERVATIONAL", "EXAMPLE_BASED"],
    "prohibited_claims": [
        "author_intention",
        "psychological_state",
        "political_affiliation",
        "automatic_ideology_classification",
        "absence_of_discursive_phenomena",
    ],
}

QUOTE_ATTRIBUTION_CANDIDATES_MANIFEST = {
    "id": "builtin.quote-attribution-candidates",
    "version": "1.0.0",
    "framework": "quotation-and-attribution-candidate-inventory",
    "execution_kind": "DETERMINISTIC",
    "input_schema_ref": "schemas/quote-attribution-candidates-input-v1.json",
    "output_schema_ref": "schemas/quote-attribution-candidates-output-v1.json",
    "executor": "builtin.quote_attribution_candidates",
    "methodology_refs": ["docs/methodology/quote-attribution-candidates-v1.md"],
    "required_capabilities": ["canonical_text", "textual_evidence"],
    "retrieval_plan_ref": "retrieval/scoped-quotation-mark-inventory-v1",
    "documentation_ref": "docs/methodology/quote-attribution-candidates-v1.md",
    "allowed_claims": [],
    "prohibited_claims": [
        "source_authenticity",
        "speaker_identity",
        "endorsement",
        "influence",
        "author_intention",
        "plagiarism",
    ],
}

SCOPE_QUESTION_CANDIDATES_MANIFEST = {
    "id": "builtin.scope-question-candidates",
    "version": "1.0.0",
    "framework": "researcher-directed-evidence-retrieval",
    "execution_kind": "DETERMINISTIC",
    "input_schema_ref": "schemas/scope-question-candidates-input-v1.json",
    "output_schema_ref": "schemas/scope-question-candidates-output-v1.json",
    "executor": "builtin.scope_question_candidates",
    "methodology_refs": ["docs/methodology/scope-question-candidates-v1.md"],
    "retrieval_plan_ref": "retrieval/framework-context-lexical-v1",
    "documentation_ref": "docs/methodology/scope-question-candidates-v1.md",
    "allowed_claims": ["EXAMPLE_BASED"],
    "prohibited_claims": ["corpus_wide_absence", "prevalence", "universal_claim"],
}

SCOPE_QUESTION_HYBRID_CANDIDATES_MANIFEST = {
    **SCOPE_QUESTION_CANDIDATES_MANIFEST,
    "id": "builtin.scope-question-hybrid-candidates",
    "execution_kind": "HYBRID",
    "input_schema_ref": "schemas/scope-question-hybrid-candidates-input-v1.json",
    "output_schema_ref": "schemas/scope-question-hybrid-candidates-output-v1.json",
    "executor": "builtin.scope_question_hybrid_candidates",
    "required_capabilities": ["semantic_search"],
    "retrieval_plan_ref": "retrieval/hybrid-rrf-preset-v1",
}

COUNTEREVIDENCE_LEXICAL_MANIFEST = {
    "id": "builtin.counterevidence-lexical",
    "version": "2.0.0",
    "framework": "evidence-review",
    "execution_kind": "DETERMINISTIC",
    "input_schema_ref": "schemas/counterevidence-lexical-input-v1.json",
    "output_schema_ref": "schemas/counterevidence-lexical-output-v2.json",
    "executor": "builtin.counterevidence_lexical",
    "methodology_refs": ["docs/methodology/counterevidence-lexical-v2.md"],
    "retrieval_plan_ref": "retrieval/exhaustive-literal-match-candidates-v1",
    "documentation_ref": "docs/methodology/counterevidence-lexical-v2.md",
    "allowed_claims": ["EXAMPLE_BASED"],
    "prohibited_claims": [
        "author_intention",
        "psychological_state",
        "political_affiliation",
        "corpus_wide_absence_from_targeted_search",
    ],
}

SOCIAL_ACTOR_CANDIDATES_MANIFEST = {
    "id": "builtin.social-actor-candidates",
    "version": "1.0.0",
    "framework": "van-leeuwen-social-actors",
    "execution_kind": "DETERMINISTIC",
    "input_schema_ref": "schemas/social-actor-candidates-input-v1.json",
    "output_schema_ref": "schemas/social-actor-candidates-output-v1.json",
    "executor": "builtin.social_actor_candidates",
    "methodology_refs": ["docs/methodology/social-actor-candidates-v1.md"],
    "required_capabilities": ["pos"],
    "retrieval_plan_ref": "retrieval/parsed-token-scope-v1",
    "documentation_ref": "docs/methodology/social-actor-candidates-v1.md",
    "allowed_claims": ["OBSERVATIONAL"],
    "prohibited_claims": [
        "author_intention",
        "psychological_state",
        "political_affiliation",
        "automatic_van_leeuwen_interpretation",
    ],
}

AGENCY_CANDIDATES_MANIFEST = {
    "id": "builtin.agency-candidates",
    "version": "1.0.0",
    "framework": "fairclough-textual-analysis",
    "execution_kind": "DETERMINISTIC",
    "input_schema_ref": "schemas/agency-candidates-input-v1.json",
    "output_schema_ref": "schemas/agency-candidates-output-v1.json",
    "executor": "builtin.agency_candidates",
    "methodology_refs": ["docs/methodology/agency-candidates-v1.md"],
    "required_capabilities": ["dependency_parse"],
    "retrieval_plan_ref": "retrieval/parsed-dependency-scope-v1",
    "documentation_ref": "docs/methodology/agency-candidates-v1.md",
    "allowed_claims": ["OBSERVATIONAL"],
    "prohibited_claims": [
        "author_intention",
        "psychological_state",
        "automatic_agency_interpretation",
    ],
}

SYNTHESIS_MANIFEST = {
    "id": "builtin.synthesis",
    "version": "1.0.0",
    "framework": "evidence-constrained-synthesis",
    "execution_kind": "LLM_ASSISTED",
    "input_schema_ref": "schemas/synthesis-executor-input-v1.json",
    "output_schema_ref": "schemas/synthesis-executor-output-v1.json",
    "model_input_schema_ref": "schemas/synthesis-model-input-v1.json",
    "model_output_schema_ref": "schemas/evidence-bound-review-model-output-v1.json",
    "executor": "builtin.evidence_constrained_synthesis",
    "methodology_refs": ["docs/methodology/synthesis-v1.md"],
    "required_capabilities": ["validated_findings", "textual_evidence"],
    "retrieval_plan_ref": "retrieval/validated-findings-only-v1",
    "prompt_refs": ["evidence-constrained-synthesis-v1"],
    "documentation_ref": "docs/methodology/synthesis-v1.md",
    "allowed_claims": ["INTERPRETIVE"],
    "prohibited_claims": [
        "new_uncited_phenomenon",
        "author_intention",
        "psychological_state",
        "political_affiliation",
        "corpus_wide_absence_from_examples",
    ],
}

VAN_DIJK_GROUP_COMPARISON_MANIFEST = {
    "id": "builtin.van-dijk-group-comparison",
    "version": "1.0.0",
    "framework": "van-dijk-group-comparison-preparation",
    "execution_kind": "DETERMINISTIC",
    "input_schema_ref": "schemas/van-dijk-group-comparison-input-v1.json",
    "output_schema_ref": "schemas/van-dijk-group-comparison-output-v1.json",
    "executor": "builtin.van_dijk_group_comparison",
    "methodology_refs": [
        "docs/methodology/van-dijk-group-comparison-v1.md",
        "docs/methodology/qualitative-coding-v1.md",
    ],
    "retrieval_plan_ref": "retrieval/explicit-annotation-id-selection-v1",
    "documentation_ref": "docs/methodology/van-dijk-group-comparison-v1.md",
    "allowed_claims": ["EXAMPLE_BASED"],
    "prohibited_claims": [
        "political_identity",
        "ideology",
        "motivation",
        "corpus_prevalence",
        "universal_prevalence",
        "absence",
        "ideology_score",
        "asymmetry_score",
    ],
}


def _framework_review_manifest(framework_id: str) -> dict[str, object]:
    """Produce the manifest for one reviewed, trusted framework executor."""
    profile = FRAMEWORK_REVIEWS[framework_id]
    methodology_version = "v4" if framework_id == "critical-reviewer" else "v3"
    methodology_ref = profile.methodology_ref or (
        f"docs/methodology/framework-reviews-{methodology_version}.md"
    )
    prompt_ref = profile.prompt_ref or (
        f"framework-review-instructions-{methodology_version}:{profile.id}"
    )
    return {
        "id": f"builtin.{profile.id}",
        "version": profile.version,
        "framework": profile.id,
        "execution_kind": "LLM_ASSISTED",
        "input_schema_ref": "schemas/evidence-bound-review-executor-input-v1.json",
        "output_schema_ref": "schemas/evidence-bound-review-executor-output-v1.json",
        "model_input_schema_ref": "schemas/evidence-bound-review-model-input-v2.json",
        "model_output_schema_ref": "schemas/framework-review-v1.json",
        "executor": "builtin.evidence_bound_review",
        "methodology_refs": [methodology_ref],
        "retrieval_plan_ref": (
            "retrieval/critical-review-context-and-alternative-v1"
            if framework_id == "critical-reviewer"
            else "retrieval/framework-context-lexical-v1"
        ),
        "prompt_refs": [prompt_ref],
        "documentation_ref": methodology_ref,
        "allowed_claims": ["INTERPRETIVE"],
        "prohibited_claims": [
            "author_intention",
            "psychological_state",
            "political_affiliation",
            "fabricated_citation",
        ],
    }


FRAMEWORK_REVIEW_MANIFESTS = tuple(
    _framework_review_manifest(identifier) for identifier in FRAMEWORK_REVIEWS
)

MODAL_MARKERS = {
    "en": ("must", "should", "may", "might", "can", "will"),
    "es": ("debe", "deben", "debería", "deberian", "puede", "pueden", "podría"),
}

# These patterns are intentionally compiled from reviewed built-in constants,
# never from a document, MCP request, project config, or model response.
MODAL_PATTERNS = {
    language: re.compile(
        r"\b(?:" + "|".join(re.escape(marker) for marker in markers) + r")\b", re.IGNORECASE
    )
    for language, markers in MODAL_MARKERS.items()
}

# Reviewed, deliberately small candidate cues. A hit selects a whole sentence for
# inspection; it does not classify the sentence or make a methodological claim.
EVIDENCE_CUE_PATTERNS = {
    "en": {
        "modality": MODAL_PATTERNS["en"],
        "actor_reference": re.compile(
            r"\b(?:we|our|they|their|government|council|police|residents|migrants|"
            r"workers|citizens|officials|community|communities)\b",
            re.IGNORECASE,
        ),
        "reported_speech": re.compile(
            r"\b(?:said|says|stated|claims?|argued|according to|reported)\b|[\"“‘]",
            re.IGNORECASE,
        ),
        "negation": re.compile(r"\b(?:not|no|never|neither|without|cannot|can't)\b", re.IGNORECASE),
        "passive_surface": re.compile(
            r"\b(?:is|are|was|were|be|been|being)\s+(?:not\s+)?[a-z]{3,}ed\b", re.IGNORECASE
        ),
    },
    "es": {
        "modality": MODAL_PATTERNS["es"],
        "actor_reference": re.compile(
            r"\b(?:nosotros|nosotras|nuestro|nuestra|ellos|ellas|su|gobierno|consejo|"
            r"policía|residentes|migrantes|trabajadores|ciudadanos|autoridades|comunidad)\b",
            re.IGNORECASE,
        ),
        "reported_speech": re.compile(
            r"\b(?:dijo|dice|afirmó|afirma|sostuvo|según|informó|declaró)\b|[\"“‘]",
            re.IGNORECASE,
        ),
        "negation": re.compile(r"\b(?:no|nunca|jamás|ningún|ninguna|sin|tampoco)\b", re.IGNORECASE),
        "passive_surface": re.compile(
            r"\b(?:fue|fueron|es|son|ha sido|han sido)\s+[\wáéíóúñ]+(?:ado|ada|ados|adas|ido|ida|idos|idas)\b",
            re.IGNORECASE,
        ),
    },
}

EVIDENCE_EXTRACTION_LIMIT = 48
COUNTEREVIDENCE_MAX_CANDIDATES = 10_000
_SQLITE_BIND_BATCH_SIZE = 900
_SKILL_CURSOR_BATCH_SIZE = 512

QUOTE_OPENING_PAIRS = {
    "“": "”",
    "„": "“",
    "«": "»",
    "‹": "›",
}
QUOTE_CLOSING_PAIRS = {closing: opening for opening, closing in QUOTE_OPENING_PAIRS.items()}
QUOTE_ATTRIBUTION_CUES = {
    "en": re.compile(
        r"(?<!\w)(?:according\s+to|said|says|stated|states|claimed|claims|argued|"
        r"argues|reported|reports|wrote|writes|told|announced)(?!\w)",
        re.IGNORECASE,
    ),
    "es": re.compile(
        r"(?<!\w)(?:según|dijo|dice|afirmó|afirma|sostuvo|sostiene|informó|informa|"
        r"declaró|declara|escribió|escribe|contó|cuenta)(?!\w)",
        re.IGNORECASE,
    ),
}


def _sqlite_bind_batches(values: list[str]) -> Iterator[list[str]]:
    """Yield ID groups that fit below SQLite's bind-variable ceiling."""
    for offset in range(0, len(values), _SQLITE_BIND_BATCH_SIZE):
        yield values[offset : offset + _SQLITE_BIND_BATCH_SIZE]


def _cursor_rows(cursor) -> Iterator:
    """Yield query rows in bounded batches instead of materializing full results."""
    while batch := cursor.fetchmany(_SKILL_CURSOR_BATCH_SIZE):
        yield from batch


def _agency_sentence_ranges(
    project: Project, revision_ids: list[str]
) -> Iterator[tuple[str, int, int]]:
    """Stream sentence coordinates in the same stable order as agency candidates."""
    for revision_batch in _sqlite_bind_batches(revision_ids):
        placeholders = ",".join("?" for _ in revision_batch)
        cursor = project.db.execute(
            "SELECT revision_id,start_offset,end_offset FROM sentences "
            f"WHERE revision_id IN ({placeholders}) ORDER BY revision_id,start_offset",
            revision_batch,
        )
        for sentence in _cursor_rows(cursor):
            yield (
                str(sentence["revision_id"]),
                int(sentence["start_offset"]),
                int(sentence["end_offset"]),
            )


def _skill_scoped_artifacts(
    project: Project, scope_id: str, artifact_ids: list[str], expected_type: str
):
    """Validate artifact groups without exceeding SQLite's parameter limit."""
    from .nlp_artifacts import _scoped_artifacts

    if not artifact_ids:
        raise ScopeViolationError("token analytics requires explicit artifact IDs")
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ScopeViolationError("artifact IDs must not be duplicated")

    artifacts = []
    seen_revisions: set[str] = set()
    for artifact_batch in _sqlite_bind_batches(artifact_ids):
        batch = _scoped_artifacts(project, scope_id, artifact_batch, expected_type)
        for artifact in batch:
            revision_id = str(artifact["revision_id"])
            if revision_id in seen_revisions:
                raise ScopeViolationError(
                    "select at most one token artifact per document revision; "
                    "artifact versions cannot be combined"
                )
            seen_revisions.add(revision_id)
            artifacts.append(artifact)
    return artifacts


def _literal_occurrences(text: str, matcher: re.Pattern[str]) -> Iterator[re.Match[str]]:
    """Yield all case-insensitive literal matches, including overlapping ones.

    ``Pattern.finditer`` advances to the end of each match, which silently omits
    overlapping occurrences. That is not exhaustive coverage for substring
    searches, so resume one Unicode code point after each match's start.
    """
    start = 0
    while start < len(text):
        match = matcher.search(text, start)
        if match is None:
            return
        yield match
        start = match.start() + 1


def _ascii_quote_direction(text: str, position: int, stack: list[_QuotationFrame]) -> str | None:
    """Classify a straight double quote only where its local boundary is clear."""
    previous = text[position - 1] if position else ""
    following = text[position + 1] if position + 1 < len(text) else ""
    opening_boundary = not previous or previous.isspace() or previous in "([{<—–:;,.!?"
    closing_boundary = not following or following.isspace() or following in ".,!?;:)]}—–"
    has_content_after = bool(following) and not following.isspace()
    has_content_before = bool(previous) and not previous.isspace()

    if stack and stack[-1]["family"] == '"':
        if has_content_before and closing_boundary:
            return "close"
        if opening_boundary and has_content_after:
            return "open"
        return None
    if opening_boundary and has_content_after:
        return "open"
    if has_content_before and closing_boundary:
        return "close"
    return None


def _scan_quotation_spans(text: str) -> tuple[list[_QuotationSpan], int, int]:
    """Find non-empty, visibly paired quote spans and count excluded markers.

    Every nested or structurally ambiguous group is discarded as one candidate
    group. Straight double quotes use conservative local-boundary checks because
    the same glyph marks both sides. Blank-line paragraph boundaries resynchronize
    scanning after unmatched openers so one damaged passage does not consume a
    later paragraph.
    """
    spans: list[_QuotationSpan] = []
    unmatched_markers = 0
    ambiguous_markers: set[int] = set()
    paragraph_ranges = [
        (match.end(), match.start()) for match in re.finditer(r"\n[\t \r]*\n", text)
    ]
    segment_start = 0

    def scan_segment(start_offset: int, end_offset: int) -> None:
        nonlocal unmatched_markers
        segment = text[start_offset:end_offset]
        stack: list[_QuotationFrame] = []

        def record_closure(frame: _QuotationFrame, closing_marker: str, end: int) -> None:
            if stack:
                return
            if frame["group_ambiguous"]:
                ambiguous_markers.update(
                    start_offset + marker_position for marker_position in frame["group_markers"]
                )
                return
            if segment[frame["start"] + len(frame["opening_marker"]) : end].strip():
                spans.append(
                    _QuotationSpan(
                        start=start_offset + frame["start"],
                        end=start_offset + end + len(closing_marker),
                        opening_marker=frame["opening_marker"],
                        closing_marker=closing_marker,
                    )
                )

        position = 0
        while position < len(segment):
            marker = segment[position]
            if marker == '"':
                direction = _ascii_quote_direction(segment, position, stack)
                if direction is None:
                    if stack:
                        stack[0]["group_ambiguous"] = True
                        stack[0]["group_markers"].append(position)
                    else:
                        ambiguous_markers.add(start_offset + position)
                    unmatched_markers += 1
                elif direction == "open":
                    if stack:
                        stack[0]["group_ambiguous"] = True
                        stack[0]["group_markers"].append(position)
                        group_markers = stack[0]["group_markers"]
                    else:
                        group_markers = [position]
                    stack.append(
                        _QuotationFrame(
                            family='"',
                            start=position,
                            opening_marker=marker,
                            group_markers=group_markers,
                            group_ambiguous=bool(stack),
                        )
                    )
                elif stack and stack[-1]["family"] == '"':
                    root = stack[0]
                    root["group_markers"].append(position)
                    frame = stack.pop()
                    record_closure(frame, marker, position)
                else:
                    if stack:
                        stack[0]["group_ambiguous"] = True
                        stack[0]["group_markers"].append(position)
                    unmatched_markers += 1
                position += 1
                continue

            if marker in QUOTE_OPENING_PAIRS:
                if stack:
                    stack[0]["group_ambiguous"] = True
                    stack[0]["group_markers"].append(position)
                    group_markers = stack[0]["group_markers"]
                else:
                    group_markers = [position]
                stack.append(
                    _QuotationFrame(
                        family=marker,
                        start=position,
                        opening_marker=marker,
                        group_markers=group_markers,
                        group_ambiguous=bool(stack),
                    )
                )
                position += 1
                continue

            if marker in QUOTE_CLOSING_PAIRS:
                matching_index = next(
                    (
                        index
                        for index in range(len(stack) - 1, -1, -1)
                        if stack[index]["family"] == QUOTE_CLOSING_PAIRS[marker]
                    ),
                    None,
                )
                if matching_index is None:
                    unmatched_markers += 1
                    if stack:
                        stack[0]["group_ambiguous"] = True
                        stack[0]["group_markers"].append(position)
                else:
                    root = stack[0]
                    root["group_markers"].append(position)
                    if matching_index != len(stack) - 1:
                        root["group_ambiguous"] = True
                        while len(stack) > matching_index:
                            frame = stack.pop()
                            if not stack:
                                record_closure(frame, marker, position)
                    else:
                        frame = stack.pop()
                        record_closure(frame, marker, position)
                position += 1
                continue
            position += 1

        if stack:
            root = stack[0]
            unmatched_markers += len(stack)
            ambiguous_markers.update(start_offset + item for item in root["group_markers"])

    for break_start, break_end in paragraph_ranges:
        scan_segment(segment_start, break_start)
        segment_start = break_end
    scan_segment(segment_start, len(text))
    return spans, unmatched_markers, len(ambiguous_markers)


def _quotation_context_bounds(
    project: Project, revision_id: str, quote_start: int, quote_end: int, text: str
) -> tuple[int, int]:
    rows = project.db.execute(
        "SELECT start_offset,end_offset FROM sentences "
        "WHERE revision_id=? AND start_offset<? AND end_offset>? "
        "ORDER BY ordinal,start_offset,id",
        (revision_id, quote_end, quote_start),
    ).fetchall()
    if rows:
        context_start = int(rows[0]["start_offset"])
        context_end = int(rows[-1]["end_offset"])
    else:
        context_start = max(0, quote_start - 160)
        context_end = min(len(text), quote_end + 160)

    # Some sentence splitters leave a quote-final punctuation mark attached to
    # its closing delimiter and merge the following sentence. Split only at
    # quote-final punctuation so initials and ordinary abbreviations do not
    # arbitrarily truncate context.
    quoted_boundary = re.compile(r"(?:[.!?][\"”»›]|[\"”»›][.!?])(?=\s|$)")
    prior_boundaries = list(quoted_boundary.finditer(text, context_start, quote_start))
    if prior_boundaries:
        context_start = prior_boundaries[-1].end()
        while context_start < quote_start and text[context_start].isspace():
            context_start += 1

    quote_surface = text[max(context_start, quote_start - 2) : quote_end]
    if re.search(r"(?:[.!?][\"”»›]|[\"”»›][.!?])$", quote_surface):
        context_end = quote_end
    else:
        following_boundary = quoted_boundary.search(text, quote_end, context_end)
        if following_boundary:
            context_end = following_boundary.end()
    return context_start, context_end


def _nearby_attribution_cues(
    text: str,
    context_start: int,
    context_end: int,
    quote_start: int,
    quote_end: int,
    language: str | None,
) -> list[_AttributionCueMatch]:
    if language is None:
        return []
    pattern = QUOTE_ATTRIBUTION_CUES[language]
    candidates: list[tuple[int, _AttributionCueMatch]] = []
    for match in pattern.finditer(text, context_start, context_end):
        if match.end() <= quote_start:
            distance = quote_start - match.end()
        elif match.start() >= quote_end:
            distance = match.start() - quote_end
        else:
            continue
        if distance <= 160:
            candidates.append(
                (
                    distance,
                    _AttributionCueMatch(
                        language=language,
                        surface=match.group(0),
                        start=match.start(),
                        end=match.end(),
                    ),
                )
            )
    if not candidates:
        return []
    _distance, nearest = min(
        candidates,
        key=lambda item: (item[0], item[1]["start"]),
    )
    return [nearest]


def parse_manifest(payload: dict) -> SkillManifest:
    try:
        return SkillManifest.model_validate(payload)
    except Exception as exc:
        raise SkillManifestError(str(exc)) from exc


def _scoped_revision_selection(
    project: Project,
    scope_id: str,
    revision_ids: tuple[str, ...] | None,
    revision_languages: Mapping[str, str] | None,
    supported_languages: tuple[str, ...],
) -> tuple[list[str], dict[str, str] | None]:
    original = project.scope_revisions(scope_id)
    selected = original if revision_ids is None else list(revision_ids)
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(original):
        raise SkillManifestError("skill revision selection must be a non-empty subset of its scope")
    if revision_languages is None:
        return selected, None
    language_map = dict(revision_languages)
    if set(language_map) != set(selected) or any(
        language not in supported_languages for language in language_map.values()
    ):
        raise SkillManifestError(
            "skill revision language mapping does not match its supported scope"
        )
    return selected, language_map


def eligibility(
    manifest: SkillManifest, capabilities: set[str] | Mapping[str, str]
) -> ExecutionEligibility:
    """Evaluate the manifest against names or explicit operational statuses.

    A plain set remains useful for isolated deterministic tests. Production
    language capability checks should pass a status mapping so LIMITED,
    UNASSESSED and UNSUPPORTED never silently become usable.
    """
    if isinstance(capabilities, set):
        missing = [item for item in manifest.required_capabilities if item not in capabilities]
    else:
        missing = [
            item for item in manifest.required_capabilities if capabilities.get(item) != "SUPPORTED"
        ]
    return ExecutionEligibility(
        runnable=not missing,
        missing_capabilities=missing,
        warnings=["required capability unavailable or not assessed"] if missing else [],
    )


def builtin_manifest(skill_id: str) -> SkillManifest:
    return BUILTIN_SKILLS.get(skill_id).manifest


def skill_resource_text(reference: str) -> str:
    """Read a reviewed methodology/schema resource referenced by a built-in manifest.

    Logical references are intentionally restricted to the two bundled resource
    trees. This works from an installed wheel as well as a source checkout and
    never treats a manifest reference as an arbitrary filesystem path.
    """
    logical_path = PurePosixPath(reference)
    windows_path = PureWindowsPath(reference)
    parts = logical_path.parts
    if (
        logical_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in parts
        or ".." in windows_path.parts
        or not (
            (len(parts) == 2 and parts[0] == "schemas")
            or (len(parts) == 3 and parts[:2] == ("docs", "methodology"))
        )
    ):
        raise SkillManifestError("resource reference is outside bundled skill resources")

    if parts[0] == "schemas":
        tree = "schemas"
    else:
        tree = "methodology"
    name = logical_path.name
    if not name or PurePosixPath(name).name != name or PureWindowsPath(name).name != name:
        raise SkillManifestError("invalid bundled resource reference")

    resource = importlib.resources.files("discourselens").joinpath("_resources", tree, name)
    if resource.is_file():
        return resource.read_text(encoding="utf-8")

    # Source-tree fallback keeps local development usable before a wheel build.
    package_root = Path(__file__).resolve().parents[2]
    source_path = package_root / ("schemas" if tree == "schemas" else "docs/methodology") / name
    if source_path.is_file():
        return source_path.read_text(encoding="utf-8")
    raise SkillManifestError(f"bundled skill resource not found: {reference}")


def skill_documentation(skill_id: str) -> str:
    """Return the primary methodology document for a reviewed built-in skill."""
    manifest = builtin_manifest(skill_id)
    return skill_resource_text(manifest.documentation_ref)


def run_textual_descriptors(
    context: SkillContext,
    languages: tuple[str, ...] = ("en", "es"),
    *,
    revision_ids: tuple[str, ...] | None = None,
    revision_languages: Mapping[str, str] | None = None,
) -> TextualDescriptorResult:
    """Record descriptive modal-marker observations with canonical evidence only.

    This deliberately does not assign ideology, intent, or a discourse interpretation.
    The executor exhaustively scans declared forms in the immutable scope snapshot.
    """
    project = context.project
    unsupported_languages = sorted(set(languages) - set(MODAL_MARKERS))
    if unsupported_languages:
        raise SkillManifestError(
            "textual descriptors has no declared modal-marker capability for: "
            + ", ".join(unsupported_languages)
        )
    markers = tuple(marker for language in languages for marker in MODAL_MARKERS[language])
    if not markers:
        raise SkillManifestError(
            "textual descriptors requires at least one supported marker language"
        )
    selected_revisions, language_map = _scoped_revision_selection(
        project, context.scope_id, revision_ids, revision_languages, languages
    )
    config = json.dumps(
        {
            "languages": languages,
            "markers": markers,
            "revision_ids": selected_revisions,
            "revision_languages": language_map,
        },
        sort_keys=True,
    )
    spans_by_marker: dict[str, list[tuple[str, int, int]]] = {}
    population_count = 0
    for revision_id in selected_revisions:
        text = project.revision_text(revision_id)
        population_count += len(text)
        active_languages = (language_map[revision_id],) if language_map else languages
        for language in active_languages:
            for match in MODAL_PATTERNS[language].finditer(text):
                marker = match.group(0).casefold()
                spans_by_marker.setdefault(marker, []).append(
                    (revision_id, match.start(), match.end())
                )

    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=context.scope_id,
            population_definition=(
                "Canonical Unicode code points in every language-eligible revision selected "
                "from the original ScopeSnapshot"
            ),
            population_count=population_count,
            selection_policy=(
                "Exhaustive deterministic scan of the defined canonical text population against "
                "the versioned modal-marker inventory; no sampling or candidate cap"
            ),
            inspection_count=population_count,
            completion_criterion=(
                "Every canonical code point in every selected language-eligible revision was "
                "checked against the declared modal-marker patterns"
            ),
            termination_reason="completed",
            termination_state=CoverageTerminationState.COMPLETED,
            negative_result_status=(
                NegativeResultStatus.NOT_APPLICABLE
                if spans_by_marker
                else NegativeResultStatus.NO_MATCHES_UNDER_PLAN
            ),
        )
    )
    run_id = project.create_skill_run(
        context.analysis_run_id,
        context.scope_id,
        "builtin.textual-descriptors",
        "1.0.0",
        "builtin.textual_descriptors",
        hashlib.sha256(config.encode()).hexdigest(),
        coverage_id=coverage_id,
    )

    finding_ids: list[str] = []
    evidence_ids: list[str] = []
    for marker, marker_spans in spans_by_marker.items():
        marker_evidence = [
            project.textual_evidence(project.create_span(revision_id, start, end))
            for revision_id, start, end in marker_spans
        ]
        finding = project.create_finding(
            context.analysis_run_id,
            "builtin.textual-descriptors",
            "1.0.0",
            "modal_marker",
            (
                f"The declared modal marker {marker!r} occurs in the language-eligible "
                "source population selected from the original scope."
            ),
            skill_run_id=run_id,
            claim_types=[ClaimType.OBSERVATIONAL],
            evidence_sufficiency="SUFFICIENT",
            limitations=[
                (
                    "Coverage is exhaustive only for the declared modal-marker inventory and the "
                    "language-eligible revision population; it does not characterize other forms "
                    "of modality or the function of a marker in context."
                )
            ],
            coverage_id=coverage_id,
        )
        for evidence in marker_evidence:
            project.attach_evidence(finding, evidence, EvidenceRole.EXAMPLE)
        finding_ids.append(finding)
        evidence_ids.extend(marker_evidence)
    project.complete_skill_run(run_id)
    return {
        "skill_run_id": run_id,
        "finding_ids": finding_ids,
        "evidence_ids": evidence_ids,
        "coverage_id": coverage_id,
    }


def run_evidence_extraction(
    context: SkillContext,
    languages: tuple[str, ...] = ("en", "es"),
    *,
    max_examples: int = EVIDENCE_EXTRACTION_LIMIT,
    revision_ids: tuple[str, ...] | None = None,
    revision_languages: Mapping[str, str] | None = None,
) -> EvidenceExtractionResult:
    """Build a bounded, descriptive inventory from the original sentence scope.

    The matcher inspects every persisted sentence in the immutable ScopeSnapshot,
    but only candidate examples (not negative or prevalence claims) are exposed to
    downstream interpretation. Selection is deterministic and round-robin across
    revisions so an early document cannot consume the entire evidence budget.
    """
    project = context.project
    unsupported_languages = sorted(set(languages) - set(EVIDENCE_CUE_PATTERNS))
    if unsupported_languages:
        raise SkillManifestError(
            "evidence extraction has no declared cue inventory for: "
            + ", ".join(unsupported_languages)
        )
    if not languages or max_examples < 1 or max_examples > 256:
        raise SkillManifestError(
            "evidence extraction requires languages and a 1..256 example limit"
        )

    selected_revisions, language_map = _scoped_revision_selection(
        project, context.scope_id, revision_ids, revision_languages, languages
    )
    rows_by_revision: dict[str, list[_EvidenceCandidate]] = {}
    population_count = 0
    candidate_count = 0
    for revision_id in selected_revisions:
        rows = project.db.execute(
            "SELECT revision_id,start_offset,end_offset,text FROM sentences "
            "WHERE revision_id=? ORDER BY ordinal,start_offset,id",
            (revision_id,),
        ).fetchall()
        population_count += len(rows)
        candidates: list[_EvidenceCandidate] = []
        for row in rows:
            matched_cues = [
                f"{language}:{cue}"
                for language in ((language_map[revision_id],) if language_map else languages)
                for cue, pattern in EVIDENCE_CUE_PATTERNS[language].items()
                if pattern.search(row["text"])
            ]
            if matched_cues:
                candidates.append(
                    _EvidenceCandidate(
                        revision_id=str(row["revision_id"]),
                        start=int(row["start_offset"]),
                        end=int(row["end_offset"]),
                        cues=matched_cues,
                    )
                )
        rows_by_revision[revision_id] = candidates
        candidate_count += len(candidates)

    # Round-robin stable selection across documents/revisions.
    selected: list[_EvidenceCandidate] = []
    offsets = {revision_id: 0 for revision_id in selected_revisions}
    while len(selected) < min(max_examples, candidate_count):
        made_progress = False
        for revision_id in selected_revisions:
            index = offsets[revision_id]
            candidates = rows_by_revision[revision_id]
            if index < len(candidates):
                selected.append(candidates[index])
                offsets[revision_id] += 1
                made_progress = True
                if len(selected) >= max_examples:
                    break
        if not made_progress:
            break

    full_scope_revisions = project.scope_revisions(context.scope_id)
    population_definition = (
        "Persisted canonical sentences in each revision named by the original ScopeSnapshot"
        if selected_revisions == full_scope_revisions
        else "Persisted canonical sentences in the language-eligible revisions selected from the original ScopeSnapshot"
    )
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.BROAD,
            scope_snapshot_id=context.scope_id,
            population_definition=population_definition,
            population_count=population_count,
            selection_policy=(
                "exhaustive deterministic scan of the defined sentences against versioned descriptive cue patterns; "
                f"retain at most {max_examples} candidate examples"
            ),
            inspection_count=population_count,
            completion_criterion=(
                "Every persisted sentence in the defined language-eligible population was checked against the versioned built-in cue patterns"
            ),
            termination_reason=(
                "example_limit_reached" if candidate_count > len(selected) else "completed"
            ),
            negative_result_status=(
                NegativeResultStatus.NO_MATCHES_UNDER_PLAN
                if candidate_count == 0
                else NegativeResultStatus.NOT_APPLICABLE
            ),
        )
    )
    configuration = json.dumps(
        {
            "languages": list(languages),
            "cue_inventory": "descriptive-candidate-cues-v1",
            "max_examples": max_examples,
            "scope_id": context.scope_id,
            "revision_ids": selected_revisions,
            "revision_languages": language_map,
        },
        sort_keys=True,
    )
    skill_run_id = project.create_skill_run(
        context.analysis_run_id,
        context.scope_id,
        "builtin.evidence-extraction",
        "1.0.0",
        "builtin.evidence_extraction",
        hashlib.sha256(configuration.encode()).hexdigest(),
        coverage_id=coverage_id,
    )
    finding_ids: list[str] = []
    evidence_ids: list[str] = []
    if selected:
        limitations = [
            "Cue rules are a bounded selection aid and can miss relevant passages or flag irrelevant ones.",
            "The attached set may be capped; it cannot establish absence, prevalence, or universality.",
            "Candidate labels do not classify agency, social actors, ideology, or author intention.",
        ]
        if selected_revisions != full_scope_revisions:
            limitations.append(
                "The cue inventory was restricted to revisions with an explicitly supported language assignment; other revisions were not inspected by this skill."
            )
        finding_id = project.create_finding(
            context.analysis_run_id,
            "builtin.evidence-extraction",
            "1.0.0",
            EVIDENCE_EXTRACTION_FINDING_PHENOMENON,
            (
                f"The declared descriptive cue inventory flagged {candidate_count} of "
                f"{population_count} scoped sentences; {len(selected)} candidate examples "
                "are attached for review. A cue match is not an interpretation."
            ),
            skill_run_id=skill_run_id,
            claim_types=[],
            evidence_sufficiency="LIMITED",
            interpretation_confidence="LOW",
            limitations=limitations,
            coverage_id=coverage_id,
        )
        for candidate in selected:
            span_id = project.create_span(
                candidate["revision_id"], candidate["start"], candidate["end"]
            )
            evidence_id = project.textual_evidence(span_id)
            project.attach_evidence(
                finding_id,
                evidence_id,
                EvidenceRole.EXAMPLE,
                "Deterministic candidate cues: " + ", ".join(candidate["cues"]),
            )
            evidence_ids.append(evidence_id)
        finding_ids.append(finding_id)
    project.complete_skill_run(skill_run_id)
    return {
        "skill_run_id": skill_run_id,
        "finding_ids": finding_ids,
        "evidence_ids": evidence_ids,
        "coverage_id": coverage_id,
        "population_count": population_count,
        "candidate_count": candidate_count,
        "selected_count": len(selected),
    }


def run_quote_attribution_candidates(
    context: SkillContext,
    *,
    revision_languages: Mapping[str, str] | None = None,
) -> QuoteAttributionCandidatesResult:
    """Inventory visibly paired quotation marks in the original ScopeSnapshot.

    Nearby reporting words are retained as lexical candidates only when callers
    assign each scoped revision to a supported language. Quote-mark detection is
    language-neutral; unsupported or unassigned languages never receive an
    English/Spanish attribution-cue scan by assumption.
    """
    project = context.project
    original_revisions = project.scope_revisions(context.scope_id)
    language_map: dict[str, str] | None = None
    if revision_languages is not None:
        unsupported_languages = sorted(
            set(revision_languages.values()) - set(QUOTE_ATTRIBUTION_CUES)
        )
        if unsupported_languages:
            raise SkillManifestError(
                "quote-attribution candidates have no reviewed cue inventory for: "
                + ", ".join(unsupported_languages)
            )
        selected_revisions, language_map = _scoped_revision_selection(
            project,
            context.scope_id,
            None,
            revision_languages,
            tuple(QUOTE_ATTRIBUTION_CUES),
        )
    else:
        selected_revisions = original_revisions

    population_count = 0
    candidate_count = 0
    attribution_candidate_count = 0
    unmatched_marker_count = 0
    ambiguous_marker_count = 0
    for revision_id in selected_revisions:
        text = project.revision_text(revision_id)
        population_count += len(text)
        quotations, unmatched, ambiguous = _scan_quotation_spans(text)
        candidate_count += len(quotations)
        unmatched_marker_count += unmatched
        ambiguous_marker_count += ambiguous
        language = language_map[revision_id] if language_map else None
        for quotation in quotations:
            context_start, context_end = _quotation_context_bounds(
                project,
                revision_id,
                quotation["start"],
                quotation["end"],
                text,
            )
            if _nearby_attribution_cues(
                text,
                context_start,
                context_end,
                quotation["start"],
                quotation["end"],
                language,
            ):
                attribution_candidate_count += 1

    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=context.scope_id,
            population_definition=(
                "Canonical Unicode code points in every revision named by the original ScopeSnapshot"
            ),
            population_count=population_count,
            selection_policy=(
                "Exhaustive deterministic scan of the original scoped text for the versioned paired quote-mark inventory; "
                "nested and ambiguous groups are excluded from candidate spans"
            ),
            inspection_count=population_count,
            completion_criterion=(
                "Every canonical code point in every revision in the original ScopeSnapshot was checked against the declared quote-mark inventory"
            ),
            termination_reason="completed",
            termination_state=CoverageTerminationState.COMPLETED,
            negative_result_status=(
                NegativeResultStatus.NO_MATCHES_UNDER_PLAN
                if candidate_count == 0
                else NegativeResultStatus.NOT_APPLICABLE
            ),
        )
    )
    configuration = json.dumps(
        {
            "scope_id": context.scope_id,
            "quote_inventory": "paired-quote-mark-inventory-v1",
            "attribution_cue_inventory": "english-spanish-reporting-cues-v1",
            "revision_languages": language_map,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    skill_run_id = project.create_skill_run(
        context.analysis_run_id,
        context.scope_id,
        "builtin.quote-attribution-candidates",
        "1.0.0",
        "builtin.quote_attribution_candidates",
        hashlib.sha256(configuration.encode("utf-8")).hexdigest(),
        coverage_id=coverage_id,
    )

    candidates: list[QuoteAttributionCandidate] = []
    evidence_ids: list[str] = []
    for revision_id in selected_revisions:
        text = project.revision_text(revision_id)
        quotations, _unmatched, _ambiguous = _scan_quotation_spans(text)
        language = language_map[revision_id] if language_map else None
        inventory_items: list[_QuotationInventoryItem] = []
        span_ranges: list[tuple[int, int]] = []
        for quotation in quotations:
            context_start, context_end = _quotation_context_bounds(
                project,
                revision_id,
                quotation["start"],
                quotation["end"],
                text,
            )
            cues = _nearby_attribution_cues(
                text,
                context_start,
                context_end,
                quotation["start"],
                quotation["end"],
                language,
            )
            candidate_id = (
                "quote-"
                + hashlib.sha256(
                    f"{revision_id}\0{quotation['start']}\0{quotation['end']}".encode()
                ).hexdigest()[:24]
            )
            quote_span_index = len(span_ranges)
            span_ranges.append((quotation["start"], quotation["end"]))
            context_span_index = len(span_ranges)
            span_ranges.append((context_start, context_end))
            cue_span_indices: list[int] = []
            for cue in cues:
                cue_span_indices.append(len(span_ranges))
                span_ranges.append((cue["start"], cue["end"]))
            inventory_items.append(
                _QuotationInventoryItem(
                    quote=quotation,
                    context_start=context_start,
                    context_end=context_end,
                    cues=cues,
                    candidate_id=candidate_id,
                    quote_span_index=quote_span_index,
                    context_span_index=context_span_index,
                    cue_span_indices=cue_span_indices,
                )
            )

        if not span_ranges:
            continue
        span_ids = project.create_spans(revision_id, span_ranges, canonical_text=text)
        for item in inventory_items:
            quotation = item["quote"]
            candidate_id = item["candidate_id"]
            base_provenance: dict[str, object] = {
                "method": "quote-attribution-candidates-v1",
                "scope_snapshot_id": context.scope_id,
                "candidate_id": candidate_id,
                "quote_start": quotation["start"],
                "quote_end": quotation["end"],
            }
            quote_evidence_id = project.textual_evidence(
                span_ids[item["quote_span_index"]],
                provenance={**base_provenance, "candidate_part": "quotation_span"},
            )
            context_evidence_id = project.textual_evidence(
                span_ids[item["context_span_index"]],
                provenance={**base_provenance, "candidate_part": "sentence_context"},
            )
            cue_evidence_ids: list[str] = []
            for cue, cue_span_index in zip(item["cues"], item["cue_span_indices"], strict=True):
                cue_evidence_ids.append(
                    project.textual_evidence(
                        span_ids[cue_span_index],
                        provenance={
                            **base_provenance,
                            "candidate_part": "nearby_attribution_cue",
                            "cue_language": cue["language"],
                            "cue_start": cue["start"],
                            "cue_end": cue["end"],
                        },
                    )
                )
            evidence_ids.extend([quote_evidence_id, context_evidence_id, *cue_evidence_ids])
            candidates.append(
                QuoteAttributionCandidate(
                    candidate_id=candidate_id,
                    revision_id=revision_id,
                    quote_start=quotation["start"],
                    quote_end=quotation["end"],
                    opening_marker=quotation["opening_marker"],
                    closing_marker=quotation["closing_marker"],
                    quote_evidence_id=quote_evidence_id,
                    context_evidence_id=context_evidence_id,
                    attribution_cues=[cue["surface"] for cue in item["cues"]],
                    attribution_cue_languages=[cue["language"] for cue in item["cues"]],
                    attribution_cue_evidence_ids=cue_evidence_ids,
                )
            )

    if (
        len(candidates) != candidate_count
        or sum(bool(candidate["attribution_cues"]) for candidate in candidates)
        != attribution_candidate_count
    ):
        raise SkillManifestError("quotation inventory changed while the skill was running")

    finding_ids: list[str] = []
    if candidates:
        limitations = [
            "Quotation marks identify textual candidates only; they do not verify an original source or establish who spoke.",
            "Nearby reporting words are lexical proximity candidates only; they do not establish that a speaker endorsed the quoted wording or identify an authentic source.",
            "The reviewed reporting-cue inventory covers English and Spanish only and runs only for revisions with an explicit supported language assignment.",
            "This inventory does not detect unquoted reported speech, citations, source reuse, influence, plagiarism, or author intent.",
            "Only non-empty, visibly paired quotation spans are returned; OCR damage and unsupported typographic forms can be missed.",
        ]
        if language_map is None:
            limitations.append(
                "No per-revision language assignment was supplied, so no language-specific attribution cues were scanned."
            )
        if unmatched_marker_count:
            limitations.append(
                f"{unmatched_marker_count} unmatched quote marker(s) were excluded from quotation candidates."
            )
        if ambiguous_marker_count:
            limitations.append(
                f"{ambiguous_marker_count} marker(s) in nested or otherwise ambiguous quote groups were excluded."
            )
        finding_id = project.create_finding(
            context.analysis_run_id,
            "builtin.quote-attribution-candidates",
            "1.0.0",
            "quotation_attribution_candidate_inventory",
            (
                f"The declared quote-mark inventory yielded {candidate_count} closed quotation-span candidate(s); "
                f"{attribution_candidate_count} had at least one nearby reviewed reporting-cue candidate. "
                f"The scan also recorded {unmatched_marker_count} unmatched marker(s) and "
                f"{ambiguous_marker_count} marker(s) in ambiguous groups. These are candidate signals only."
            ),
            skill_run_id=skill_run_id,
            claim_types=[],
            evidence_sufficiency="LIMITED",
            interpretation_confidence="LOW",
            limitations=limitations,
            coverage_id=coverage_id,
        )
        for candidate in candidates:
            project.attach_evidence(
                finding_id,
                candidate["quote_evidence_id"],
                EvidenceRole.EXAMPLE,
                "Closed, non-empty quotation-mark span candidate; source and speaker remain unverified.",
            )
            project.attach_evidence(
                finding_id,
                candidate["context_evidence_id"],
                EvidenceRole.CONTEXT,
                "Sentence context retained to support later review of the quotation construction.",
            )
            for cue_surface, cue_evidence_id, cue_language in zip(
                candidate["attribution_cues"],
                candidate["attribution_cue_evidence_ids"],
                candidate["attribution_cue_languages"],
                strict=True,
            ):
                project.attach_evidence(
                    finding_id,
                    cue_evidence_id,
                    EvidenceRole.CONTEXT,
                    f"Nearby {cue_language} reporting-cue candidate {cue_surface!r}; lexical proximity does not establish attribution.",
                )
        finding_ids.append(finding_id)

    project.complete_skill_run(skill_run_id)
    return {
        "skill_run_id": skill_run_id,
        "finding_ids": finding_ids,
        "evidence_ids": evidence_ids,
        "coverage_id": coverage_id,
        "population_count": population_count,
        "candidate_count": candidate_count,
        "attribution_candidate_count": attribution_candidate_count,
        "unmatched_marker_count": unmatched_marker_count,
        "ambiguous_marker_count": ambiguous_marker_count,
        "candidates": candidates,
    }


def run_scope_question_candidates(
    context: SkillContext,
    framework_id: str,
    question: str,
    *,
    limit: int = 12,
    retrieval_mode: Literal["lexical", "hybrid"] = "lexical",
    embedding_preset_name: str | None = None,
    embedding_artifact_ids: list[str] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> ScopeQuestionResult:
    """Retrieve bounded lexical examples for one framework question in original scope.

    This stage is independent of cue-derived seed findings. Its TARGETED coverage
    and exact retrieval lineage are persisted; zero results never means absence.
    """
    normalized_question = question.strip()
    if not normalized_question:
        raise SkillManifestError("scope research question must not be empty")
    if framework_id not in FRAMEWORK_REVIEWS or framework_id == "critical-reviewer":
        raise SkillManifestError("scope question must target a selected interpretive framework")
    if retrieval_mode not in {"lexical", "hybrid"}:
        raise SkillManifestError("scope question retrieval mode must be lexical or hybrid")
    if retrieval_mode == "hybrid":
        return _run_scope_question_hybrid_candidates(
            context,
            framework_id,
            normalized_question,
            limit=limit,
            embedding_preset_name=embedding_preset_name,
            embedding_artifact_ids=embedding_artifact_ids,
            embedding_provider=embedding_provider,
        )
    if embedding_preset_name is not None or embedding_artifact_ids is not None:
        raise SkillManifestError("embedding configuration is only valid for hybrid retrieval")
    if embedding_provider is not None:
        raise SkillManifestError("an embedding provider is only valid for hybrid retrieval")

    from .scope_retrieval import FRAMEWORK_CONTEXT_PLAN, retrieve_framework_context

    configuration = {
        "framework_id": framework_id,
        "question": normalized_question,
        "retrieval_plan": FRAMEWORK_CONTEXT_PLAN,
        "limit": limit,
        "scope_id": context.scope_id,
    }
    config_hash = hashlib.sha256(
        json.dumps(configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    prior = context.project.db.execute(
        "SELECT id,coverage_id FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.scope-question-candidates' AND skill_version='1.0.0' "
        "AND executor='builtin.scope_question_candidates' AND config_hash=? "
        "AND status='COMPLETED' ORDER BY started_at,id LIMIT 1",
        (context.analysis_run_id, config_hash),
    ).fetchone()
    if prior:
        finding = context.project.db.execute(
            "SELECT id FROM findings WHERE skill_run_id=? ORDER BY created_at,id LIMIT 1",
            (prior["id"],),
        ).fetchone()
        retrieval = context.project.db.execute(
            "SELECT id,returned_count FROM retrieval_runs WHERE skill_run_id=? "
            "ORDER BY created_at,id LIMIT 1",
            (prior["id"],),
        ).fetchone()
        if retrieval is None or not prior["coverage_id"]:
            raise SkillManifestError("completed scope-question stage is missing provenance")
        return {
            "skill_run_id": str(prior["id"]),
            "finding_id": str(finding["id"]) if finding else None,
            "retrieval_run_id": str(retrieval["id"]),
            "coverage_id": str(prior["coverage_id"]),
            "candidate_count": int(retrieval["returned_count"]),
            "retrieval_mode": "lexical",
        }

    retrieval = retrieve_framework_context(
        context.project,
        context.scope_id,
        [],
        phenomenon=framework_id,
        observation=normalized_question,
        query_texts=(normalized_question,),
        limit=limit,
    )
    revision_ids = context.project.scope_revisions(context.scope_id)
    population_count = 0
    for revision_batch in _sqlite_bind_batches(revision_ids):
        placeholders = ",".join("?" for _ in revision_batch)
        population_count += int(
            context.project.db.execute(
                f"SELECT count(*) FROM paragraphs WHERE revision_id IN ({placeholders})",
                revision_batch,
            ).fetchone()[0]
        )
    termination = (
        "candidate_limit_reached"
        if retrieval.candidate_limit_reached
        else "no_searchable_terms"
        if not retrieval.query_terms
        else "retrieval_completed"
    )
    selection_policy = (
        f"researcher question lexical terms; scoped paragraph FTS; rank by term overlap; "
        f"return at most {limit}; no seed-evidence restriction"
    )
    negative_result_status = (
        NegativeResultStatus.NO_SEARCHABLE_TERMS
        if not retrieval.query_terms
        else NegativeResultStatus.NO_MATCHES_UNDER_PLAN
        if not retrieval.candidates
        else NegativeResultStatus.NOT_APPLICABLE
    )
    coverage_id = context.project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.TARGETED,
            scope_snapshot_id=context.scope_id,
            population_definition=(
                "Persisted paragraphs in the immutable ScopeSnapshot; this retrieval inspects only "
                "query-matched lexical candidates."
            ),
            population_count=population_count,
            selection_policy=selection_policy,
            inspection_count=retrieval.inspected_count,
            termination_reason=termination,
            negative_result_status=negative_result_status,
        )
    )
    skill_run_id = context.project.create_skill_run(
        context.analysis_run_id,
        context.scope_id,
        "builtin.scope-question-candidates",
        "1.0.0",
        "builtin.scope_question_candidates",
        config_hash,
        coverage_id=coverage_id,
    )
    retrieval_run_id = context.project.create_retrieval_run(
        analysis_run_id=context.analysis_run_id,
        skill_run_id=skill_run_id,
        scope_id=context.scope_id,
        retrieval_plan_ref=FRAMEWORK_CONTEXT_PLAN,
        query_terms=list(retrieval.query_terms),
        selection_policy=selection_policy,
        candidate_count=retrieval.inspected_count,
        inspected_count=retrieval.inspected_count,
        returned_count=len(retrieval.candidates),
        candidate_limit_reached=retrieval.candidate_limit_reached,
        coverage_id=coverage_id,
    )

    evidence_ids: list[str] = []
    for rank, candidate in enumerate(retrieval.candidates, start=1):
        span_row = context.project.db.execute(
            "SELECT id FROM source_spans WHERE revision_id=? AND start_offset=? AND end_offset=? "
            "ORDER BY id LIMIT 1",
            (candidate.revision_id, candidate.start, candidate.end),
        ).fetchone()
        span_id = (
            str(span_row["id"])
            if span_row
            else context.project.create_span(candidate.revision_id, candidate.start, candidate.end)
        )
        evidence_row = context.project.db.execute(
            "SELECT id FROM evidence_artifacts WHERE kind='TEXTUAL' AND span_id=? "
            "ORDER BY created_at,id LIMIT 1",
            (span_id,),
        ).fetchone()
        evidence_id = (
            str(evidence_row["id"])
            if evidence_row
            else context.project.textual_evidence(
                span_id,
                provenance={
                    "acquisition": "scope-question-targeted-retrieval",
                    "retrieval_plan_ref": FRAMEWORK_CONTEXT_PLAN,
                },
            )
        )
        context.project.add_retrieval_hit(
            retrieval_run_id,
            span_id,
            "scoped_fts_term_overlap",
            rank,
            candidate.score,
        )
        evidence_ids.append(evidence_id)

    finding_id: str | None = None
    if evidence_ids:
        finding_id = context.project.create_finding(
            context.analysis_run_id,
            "builtin.scope-question-candidates",
            "1.0.0",
            f"{framework_id}_question_candidates",
            (
                f"A bounded lexical search for the researcher question {normalized_question!r} "
                f"returned {len(evidence_ids)} passage candidate(s) in the original ScopeSnapshot. "
                "These examples are inputs for interpretation, not an exhaustive answer."
            ),
            skill_run_id=skill_run_id,
            claim_types=[ClaimType.EXAMPLE_BASED],
            evidence_sufficiency="LIMITED",
            interpretation_confidence="LOW",
            limitations=[
                "Targeted lexical retrieval may miss relevant passages and does not establish prevalence or absence.",
                "A lexical match is a candidate for framework review, not a methodological interpretation.",
            ],
            coverage_id=coverage_id,
        )
        for evidence_id in evidence_ids:
            context.project.attach_evidence(
                finding_id,
                evidence_id,
                EvidenceRole.EXAMPLE,
                "Targeted lexical candidate for the selected framework research question.",
            )
    context.project.complete_skill_run(skill_run_id)
    return {
        "skill_run_id": skill_run_id,
        "finding_id": finding_id,
        "retrieval_run_id": retrieval_run_id,
        "coverage_id": coverage_id,
        "candidate_count": len(evidence_ids),
        "retrieval_mode": "lexical",
    }


def _run_scope_question_hybrid_candidates(
    context: SkillContext,
    framework_id: str,
    question: str,
    *,
    limit: int,
    embedding_preset_name: str | None,
    embedding_artifact_ids: list[str] | None,
    embedding_provider: EmbeddingProvider | None,
) -> ScopeQuestionResult:
    """Run top-k lexical/vector candidate retrieval with linked ask provenance."""
    if not isinstance(embedding_preset_name, str) or not embedding_preset_name.strip():
        raise SkillManifestError("hybrid retrieval requires an embedding preset")
    if (
        type(embedding_artifact_ids) is not list
        or not embedding_artifact_ids
        or len(embedding_artifact_ids) > 100
        or any(type(item) is not str or not item.strip() for item in embedding_artifact_ids)
        or len(embedding_artifact_ids) != len(set(embedding_artifact_ids))
    ):
        raise SkillManifestError(
            "hybrid retrieval requires unique explicit vector artifact IDs"
        )
    normalized_preset = embedding_preset_name.strip()
    artifact_ids = [item.strip() for item in embedding_artifact_ids]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise SkillManifestError("hybrid vector artifact IDs must be unique")

    from .vectors import hybrid_search_with_preset

    retrieval_plan = "retrieval/hybrid-rrf-preset-v1"
    configuration = {
        "framework_id": framework_id,
        "question": question,
        "retrieval_plan": retrieval_plan,
        "limit": limit,
        "scope_id": context.scope_id,
        "embedding_preset_name": normalized_preset,
        "embedding_artifact_ids": artifact_ids,
        "retrieval_mode": "hybrid",
    }
    config_hash = hashlib.sha256(
        json.dumps(configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    skill_id = "builtin.scope-question-hybrid-candidates"
    executor = "builtin.scope_question_hybrid_candidates"
    prior = context.project.db.execute(
        "SELECT id,coverage_id FROM skill_runs WHERE analysis_run_id=? AND skill_id=? "
        "AND skill_version='1.0.0' AND executor=? AND config_hash=? "
        "AND status='COMPLETED' ORDER BY started_at,id LIMIT 1",
        (context.analysis_run_id, skill_id, executor, config_hash),
    ).fetchone()
    if prior:
        finding = context.project.db.execute(
            "SELECT id FROM findings WHERE skill_run_id=? ORDER BY created_at,id LIMIT 1",
            (prior["id"],),
        ).fetchone()
        retrieval = context.project.db.execute(
            "SELECT id,returned_count,coverage_id FROM retrieval_runs WHERE skill_run_id=? "
            "ORDER BY created_at,id LIMIT 1",
            (prior["id"],),
        ).fetchone()
        if retrieval is None or not retrieval["coverage_id"]:
            raise SkillManifestError("completed hybrid scope-question stage is missing provenance")
        return {
            "skill_run_id": str(prior["id"]),
            "finding_id": str(finding["id"]) if finding else None,
            "retrieval_run_id": str(retrieval["id"]),
            "coverage_id": str(retrieval["coverage_id"]),
            "candidate_count": int(retrieval["returned_count"]),
            "retrieval_mode": "hybrid",
        }

    # Validate the selected preset before any query text can leave the process.
    context.project.model_preset(normalized_preset)
    skill_run_id = context.project.create_skill_run(
        context.analysis_run_id,
        context.scope_id,
        skill_id,
        "1.0.0",
        executor,
        config_hash,
    )
    hits = hybrid_search_with_preset(
        context.project,
        normalized_preset,
        context.scope_id,
        artifact_ids,
        question,
        embedding_provider,
        limit=limit,
        analysis_run_id=context.analysis_run_id,
        skill_run_id=skill_run_id,
        require_complete_scope=True,
    )
    retrieval_row = context.project.db.execute(
        "SELECT coverage_id,returned_count FROM retrieval_runs WHERE id=? AND analysis_run_id=? "
        "AND skill_run_id=? AND scope_id=?",
        (hits.retrieval_run_id, context.analysis_run_id, skill_run_id, context.scope_id),
    ).fetchone()
    if retrieval_row is None or not retrieval_row["coverage_id"]:
        raise ScopeViolationError("hybrid candidate retrieval lost its linked scope provenance")
    coverage_id = str(retrieval_row["coverage_id"])
    context.project.set_running_skill_coverage(skill_run_id, coverage_id)
    retrieval_rows = context.project.db.execute(
        "SELECT source_span_id FROM retrieval_hits WHERE retrieval_run_id=? ORDER BY rank,id",
        (hits.retrieval_run_id,),
    ).fetchall()
    if len(retrieval_rows) != len(hits):
        raise ScopeViolationError("hybrid candidate retrieval hit lineage is incomplete")

    evidence_ids: list[str] = []
    for row in retrieval_rows:
        span_id = str(row["source_span_id"])
        existing = context.project.db.execute(
            "SELECT id FROM evidence_artifacts WHERE kind='TEXTUAL' AND span_id=? "
            "ORDER BY created_at,id LIMIT 1",
            (span_id,),
        ).fetchone()
        evidence_ids.append(
            str(existing["id"])
            if existing
            else context.project.textual_evidence(
                span_id,
                provenance={
                    "acquisition": "scope-question-hybrid-retrieval",
                    "retrieval_plan_ref": retrieval_plan,
                },
            )
        )

    finding_id: str | None = None
    if evidence_ids:
        finding_id = context.project.create_finding(
            context.analysis_run_id,
            skill_id,
            "1.0.0",
            f"{framework_id}_question_candidates",
            (
                f"A bounded hybrid lexical/vector search for the researcher question {question!r} "
                f"returned {len(evidence_ids)} passage candidate(s) in the original ScopeSnapshot. "
                "These examples are inputs for interpretation, not an exhaustive answer."
            ),
            skill_run_id=skill_run_id,
            claim_types=[ClaimType.EXAMPLE_BASED],
            evidence_sufficiency="LIMITED",
            interpretation_confidence="LOW",
            limitations=[
                "Hybrid lexical/vector top-k retrieval may miss relevant passages and does not establish prevalence or absence.",
                "Retrieved passages are candidate examples for framework review, not an exhaustive answer.",
            ],
            coverage_id=coverage_id,
        )
        for evidence_id in evidence_ids:
            context.project.attach_evidence(
                finding_id,
                evidence_id,
                EvidenceRole.EXAMPLE,
                "Targeted hybrid lexical/vector candidate for the selected framework research question.",
            )
    context.project.complete_skill_run(skill_run_id)
    return {
        "skill_run_id": skill_run_id,
        "finding_id": finding_id,
        "retrieval_run_id": hits.retrieval_run_id,
        "coverage_id": coverage_id,
        "candidate_count": len(evidence_ids),
        "retrieval_mode": "hybrid",
    }


def run_scope_question_hybrid_candidates(
    context: SkillContext,
    framework_id: str,
    question: str,
    *,
    embedding_preset_name: str,
    embedding_artifact_ids: list[str],
    limit: int = 12,
    embedding_provider: EmbeddingProvider | None = None,
) -> ScopeQuestionResult:
    """Registered hybrid executor requiring an explicit embedding configuration."""
    return run_scope_question_candidates(
        context,
        framework_id,
        question,
        limit=limit,
        retrieval_mode="hybrid",
        embedding_preset_name=embedding_preset_name,
        embedding_artifact_ids=embedding_artifact_ids,
        embedding_provider=embedding_provider,
    )


def run_exact_phrase_fixture(
    project, analysis_run_id: str, scope_id: str, phrase: str
) -> FixtureSkillResult:
    """Internal test skill: deterministic phrase evidence, not a CDS methodology skill."""
    if not phrase:
        raise SkillManifestError("phrase is required")
    run_id = project.create_skill_run(
        analysis_run_id,
        scope_id,
        "fixture.exact-phrase",
        "1.0.0",
        "builtin.fixture.exact_phrase",
        phrase,
    )
    import json

    revision_ids = json.loads(
        project.db.execute(
            "SELECT revision_ids_json FROM scope_snapshots WHERE id=?", (scope_id,)
        ).fetchone()[0]
    )
    evidence = []
    for revision_id in revision_ids:
        text = project.revision_text(revision_id)
        start = 0
        while (found := text.casefold().find(phrase.casefold(), start)) >= 0:
            span = project.create_span(revision_id, found, found + len(phrase))
            evidence.append(project.textual_evidence(span))
            start = found + len(phrase)
    project.complete_skill_run(run_id)
    return {"skill_run_id": run_id, "evidence_ids": evidence}


def run_counterevidence_lexical(
    context: SkillContext,
    finding_id: str,
    phrase: str,
    *,
    offset: int = 0,
    limit: int = 100,
) -> CounterEvidenceResult:
    """Retrieve literal counterevidence candidates from the original run scope.

    The phrase is a researcher-provided review query. Hits are candidates only:
    this executor never asserts that they refute a finding and never turns zero
    targeted hits into a corpus-wide absence claim.
    """
    if not phrase.strip():
        raise SkillManifestError("counterevidence phrase is required")
    if (
        type(offset) is not int
        or not 0 <= offset <= 0x7FFF_FFFF_FFFF_FFFF
        or type(limit) is not int
        or not 1 <= limit <= 1_000
    ):
        raise SkillManifestError(
            "counterevidence page requires an integer offset in SQLite range "
            "and integer limit 1..1000"
        )
    finding = context.project.get_finding(finding_id)
    run = context.project.db.execute(
        "SELECT scope_id FROM analysis_runs WHERE id=?", (finding["analysis_run_id"],)
    ).fetchone()
    if not run or run["scope_id"] != context.scope_id:
        raise SkillManifestError(
            "counterevidence must receive the finding's original ScopeSnapshot"
        )
    revisions = context.project.scope_revisions(context.scope_id)
    matcher = re.compile(re.escape(phrase), re.IGNORECASE)
    # Count first so the immutable coverage/retrieval records contain exact
    # population and hit totals. Read one revision at a time; the second pass
    # streams its occurrence offsets through bounded persistence batches.
    population_count = 0
    candidate_count = 0
    for revision_id in revisions:
        text = context.project.revision_text(revision_id)
        population_count += len(text)
        for _ in _literal_occurrences(text, matcher):
            candidate_count += 1
            if candidate_count > COUNTEREVIDENCE_MAX_CANDIDATES:
                raise RetrievalLimitError(
                    "counterevidence query exceeded the per-run safety limit of "
                    f"{COUNTEREVIDENCE_MAX_CANDIDATES:,} occurrences; refine the literal "
                    "query before retrying",
                    limit=COUNTEREVIDENCE_MAX_CANDIDATES,
                )
        del text
    coverage_id = context.project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=context.scope_id,
            population_definition=(
                "Unicode code points in canonical text for every revision in the original ScopeSnapshot"
            ),
            population_count=population_count,
            selection_policy=(
                "case-insensitive literal occurrence scan over all canonical text in the original ScopeSnapshot"
            ),
            inspection_count=population_count,
            completion_criterion=(
                "A case-insensitive literal matcher inspected the complete canonical text of every scoped revision"
            ),
            termination_reason="completed",
            negative_result_status=(
                NegativeResultStatus.NO_MATCHES_UNDER_PLAN
                if not candidate_count
                else NegativeResultStatus.NOT_APPLICABLE
            ),
        )
    )
    configuration = json.dumps(
        {
            "finding_id": finding_id,
            "phrase": phrase,
            "retrieval": "exhaustive-literal-match-candidates-v1",
        },
        sort_keys=True,
    )
    skill_run_id = context.project.create_skill_run(
        context.analysis_run_id,
        context.scope_id,
        "builtin.counterevidence-lexical",
        "2.0.0",
        "builtin.counterevidence_lexical",
        hashlib.sha256(configuration.encode()).hexdigest(),
        input_finding_id=finding_id,
        coverage_id=coverage_id,
    )
    retrieval_run_id = context.project.create_retrieval_run(
        analysis_run_id=context.analysis_run_id,
        skill_run_id=skill_run_id,
        scope_id=context.scope_id,
        retrieval_plan_ref="retrieval/exhaustive-literal-match-candidates-v1",
        query_terms=[phrase],
        selection_policy=(
            "exhaustive case-insensitive literal occurrences; hits are candidates, not validated refutations"
        ),
        candidate_count=candidate_count,
        inspected_count=candidate_count,
        returned_count=candidate_count,
        candidate_limit_reached=False,
        coverage_id=coverage_id,
    )
    rank = 0
    for revision_id in revisions:
        text = context.project.revision_text(revision_id)
        ranges = ((match.start(), match.end()) for match in _literal_occurrences(text, matcher))
        for span_ids in context.project.create_span_batches(
            revision_id, ranges, canonical_text=text
        ):
            context.project.add_retrieval_hits(
                retrieval_run_id,
                span_ids,
                "exhaustive_literal_candidate",
                rank + 1,
                1.0,
            )
            rank += len(span_ids)
        del text
    if rank != candidate_count:
        raise SkillManifestError("counterevidence scan changed while the retrieval was running")
    page_rows = context.project.db.execute(
        "SELECT id,source_span_id FROM retrieval_hits WHERE retrieval_run_id=? "
        "ORDER BY rank LIMIT ? OFFSET ?",
        (retrieval_run_id, limit, offset),
    ).fetchall()
    context.project.complete_skill_run(skill_run_id)
    candidate_span_ids = [str(row["source_span_id"]) for row in page_rows]
    retrieval_hit_ids = [str(row["id"]) for row in page_rows]
    return {
        "skill_run_id": skill_run_id,
        "retrieval_run_id": retrieval_run_id,
        "retrieval_hit_ids": retrieval_hit_ids,
        "candidate_span_ids": candidate_span_ids,
        "coverage_id": coverage_id,
        "coverage_mode": CoverageMode.EXHAUSTIVE,
        "population_count": population_count,
        "candidate_count": candidate_count,
        "offset": offset,
        "limit": limit,
        "returned_count": len(page_rows),
        "has_more": offset + len(page_rows) < candidate_count,
        "query_interpretation": (
            "Exact literal occurrences are retrieval candidates only, not validated refutations; "
            "zero matches do not establish absence of counterevidence."
        ),
    }


def run_social_actor_candidates(
    context: SkillContext, artifact_ids: list[str]
) -> CandidateSkillResult:
    """Expose POS-tagged actor candidates for later human/hybrid analysis.

    A noun/proper-noun/pronoun is not automatically categorized as an included,
    activated, collectivized or otherwise interpreted social actor. This skill only
    preserves an auditable candidate set with exact textual evidence.
    """
    from .nlp_artifacts import _duckdb

    artifacts = _skill_scoped_artifacts(
        context.project, context.scope_id, artifact_ids, "parsed_tokens"
    )
    artifact_revision_ids = {str(item["revision_id"]) for item in artifacts}
    configuration = json.dumps({"artifact_ids": sorted(artifact_ids)}, sort_keys=True)
    skill_run_id = context.project.create_skill_run(
        context.analysis_run_id,
        context.scope_id,
        "builtin.social-actor-candidates",
        "1.0.0",
        "builtin.social_actor_candidates",
        hashlib.sha256(configuration.encode()).hexdigest(),
    )
    duckdb = _duckdb()
    paths = [
        str(verify_project_artifact(context.project.root, item["relative_path"], item["file_hash"]))
        for item in artifacts
    ]
    connection = duckdb.connect()
    findings: list[str] = []
    evidence_ids: list[str] = []
    try:
        cursor = connection.execute(
            "SELECT revision_id,surface,pos,char_start,char_end FROM read_parquet(?) "
            "WHERE revision_id IN (SELECT unnest(?)) "
            "AND pos IN ('NOUN','PROPN','PRON') ORDER BY revision_id,char_start",
            [paths, sorted(artifact_revision_ids)],
        )
        for revision_id, surface, pos, start, end in _cursor_rows(cursor):
            revision_key = str(revision_id)
            if revision_key not in artifact_revision_ids:
                raise ScopeViolationError(
                    "parsed token artifact returned a social-actor candidate outside the "
                    "selected original ScopeSnapshot revisions"
                )
            span = context.project.create_span(revision_key, int(start), int(end))
            evidence = context.project.textual_evidence(span)
            finding = context.project.create_finding(
                context.analysis_run_id,
                "builtin.social-actor-candidates",
                "1.0.0",
                "social_actor_candidate",
                f"The parser tagged {surface!r} as {pos}, a candidate social-actor expression.",
                skill_run_id=skill_run_id,
                claim_types=[],
                limitations=[
                    "Parser candidate only; it does not classify social-actor representation or author intent."
                ],
            )
            context.project.attach_evidence(finding, evidence, EvidenceRole.EXAMPLE)
            findings.append(finding)
            evidence_ids.append(evidence)
    finally:
        connection.close()
    context.project.complete_skill_run(skill_run_id)
    return {"skill_run_id": skill_run_id, "finding_ids": findings, "evidence_ids": evidence_ids}


def run_agency_candidates(context: SkillContext, artifact_ids: list[str]) -> CandidateSkillResult:
    """Record dependency-parser agency candidates without attributing intent.

    The dependency relation is a parser observation. It can support later analysis
    of activation/passivation and transitivity, but is not itself an explanation of
    why an author selected a construction.
    """
    from .nlp_artifacts import _duckdb

    artifacts = _skill_scoped_artifacts(
        context.project, context.scope_id, artifact_ids, "parsed_tokens"
    )
    artifact_revision_ids = {str(item["revision_id"]) for item in artifacts}
    preflight = context.dependency_parse_preflight
    if not isinstance(preflight, Mapping):
        raise SkillManifestError(
            "agency candidates require an explicit dependency_parse capability snapshot; "
            "status is UNASSESSED"
        )
    snapshot_by_revision: dict[str, DependencyParseSnapshot] = {}
    for artifact in artifacts:
        revision_id = str(artifact["revision_id"])
        snapshot = preflight.get(revision_id)
        if not isinstance(snapshot, Mapping):
            raise SkillManifestError(
                "agency candidates require a dependency_parse snapshot for every artifact; "
                "status is UNASSESSED"
            )
        raw_status = snapshot.get("status")
        status = (
            raw_status
            if isinstance(raw_status, str)
            and raw_status in {"SUPPORTED", "LIMITED", "UNSUPPORTED", "UNASSESSED"}
            else "UNASSESSED"
        )
        identity_matches = (
            snapshot.get("revision_id") == revision_id
            and snapshot.get("artifact_id") == str(artifact["id"])
            and isinstance(snapshot.get("language"), str)
            and bool(str(snapshot.get("language")).strip())
            and isinstance(snapshot.get("backend"), str)
            and isinstance(snapshot.get("model"), str)
            and snapshot.get("backend") == artifact["backend"]
            and snapshot.get("model") == artifact["model"]
        )
        raw_limitations = snapshot.get("limitations", [])
        if not isinstance(raw_limitations, list) or not all(
            isinstance(item, str) for item in raw_limitations
        ):
            identity_matches = False
        if not identity_matches:
            status = "UNASSESSED"
        if status != "SUPPORTED":
            raise SkillManifestError(
                "agency candidates are not eligible: dependency_parse="
                f"{status} for a scoped parser artifact"
            )
        snapshot_by_revision[revision_id] = {
            "revision_id": revision_id,
            "artifact_id": str(artifact["id"]),
            "language": str(snapshot["language"]),
            "status": "SUPPORTED",
            "backend": str(snapshot["backend"]),
            "model": str(snapshot["model"]),
            "limitations": [item for item in raw_limitations if isinstance(item, str)],
        }
    configuration = json.dumps({"artifact_ids": sorted(artifact_ids)}, sort_keys=True)
    skill_run_id = context.project.create_skill_run(
        context.analysis_run_id,
        context.scope_id,
        "builtin.agency-candidates",
        "1.0.0",
        "builtin.agency_candidates",
        hashlib.sha256(configuration.encode()).hexdigest(),
    )
    duckdb = _duckdb()
    paths = [
        str(verify_project_artifact(context.project.root, item["relative_path"], item["file_hash"]))
        for item in artifacts
    ]
    revision_ids = sorted(artifact_revision_ids)
    sentence_rows = iter(_agency_sentence_ranges(context.project, revision_ids))
    next_sentence: tuple[str, int, int] | None = None
    active_sentence_revision: str | None = None
    active_sentence: tuple[int, int] | None = None
    connection = duckdb.connect()
    findings: list[str] = []
    evidence_ids: list[str] = []
    try:
        cursor = connection.execute(
            "WITH tokens AS (SELECT * FROM read_parquet(?)) "
            "SELECT dependent.revision_id,dependent.surface,dependent.dependency_relation,"
            "dependent.char_start,dependent.char_end,head.surface,head.char_start,head.char_end "
            "FROM tokens dependent JOIN tokens head "
            "ON dependent.revision_id=head.revision_id AND dependent.dependency_head=head.token_index "
            "WHERE dependent.revision_id IN (SELECT unnest(?)) "
            "AND dependent.dependency_relation IN ('nsubj','nsubj:pass','csubj','obj','obl') "
            "ORDER BY dependent.revision_id,dependent.char_start",
            [paths, revision_ids],
        )
        next_sentence = next(sentence_rows, None)
        for (
            revision_id,
            dependent,
            relation,
            start,
            end,
            head,
            head_start,
            head_end,
        ) in _cursor_rows(cursor):
            revision_key = str(revision_id)
            if revision_key not in artifact_revision_ids:
                raise ScopeViolationError(
                    "parsed token artifact returned an agency candidate outside the selected "
                    "original ScopeSnapshot revisions"
                )
            token_start = int(start)
            while next_sentence is not None and (
                next_sentence[0] < revision_key
                or (next_sentence[0] == revision_key and next_sentence[1] <= token_start)
            ):
                active_sentence_revision = next_sentence[0]
                active_sentence = (next_sentence[1], next_sentence[2])
                next_sentence = next(sentence_rows, None)

            sentence_range = (
                active_sentence
                if active_sentence_revision == revision_key
                and active_sentence is not None
                and token_start < active_sentence[1]
                else None
            )
            evidence_start = (
                sentence_range[0]
                if sentence_range is not None
                else min(token_start, int(head_start))
            )
            evidence_end = (
                sentence_range[1] if sentence_range is not None else max(int(end), int(head_end))
            )
            span = context.project.create_span(revision_id, evidence_start, evidence_end)
            evidence = context.project.textual_evidence(span)
            finding = context.project.create_finding(
                context.analysis_run_id,
                "builtin.agency-candidates",
                "1.0.0",
                "dependency_agency_candidate",
                f"The parser links {dependent!r} to predicate {head!r} with relation {relation!r}.",
                skill_run_id=skill_run_id,
                limitations=[
                    *snapshot_by_revision[revision_key]["limitations"],
                    "Parser candidate only; agency/backgrounding interpretation and author intent require contextual review.",
                ],
            )
            context.project.attach_evidence(finding, evidence, EvidenceRole.EXAMPLE)
            findings.append(finding)
            evidence_ids.append(evidence)
    finally:
        connection.close()
    context.project.complete_skill_run(skill_run_id)
    return {"skill_run_id": skill_run_id, "finding_ids": findings, "evidence_ids": evidence_ids}


def run_framework_review(
    context: SkillContext,
    finding_id: str,
    preset_name: str,
    framework_id: str,
    *,
    provider_factory=None,
    preset_snapshot: dict[str, object] | None = None,
    reranking_gateway=None,
    reranking_provider_id: str | None = None,
    reranking_credential_alias: str | None = None,
    reranking_model: str | None = None,
    reranking_top_n: int | None = None,
):
    """Execute a registered framework profile through the trusted review boundary.

    The local import avoids a module cycle: interpretation consumes the registry's
    framework metadata but is not itself a plugin surface. The called executor
    validates that the source finding and this skill run share one ScopeSnapshot.
    """
    if framework_id not in FRAMEWORK_REVIEWS:
        raise SkillManifestError(f"unknown framework review {framework_id!r}")
    from .interpretation import review_finding_with_model

    result = review_finding_with_model(
        context.project,
        finding_id,
        preset_name,
        framework=framework_id,
        analysis_run_id=context.analysis_run_id,
        provider_factory=provider_factory,
        preset_snapshot=preset_snapshot,
        reranking_gateway=reranking_gateway,
        reranking_provider_id=reranking_provider_id,
        reranking_credential_alias=reranking_credential_alias,
        reranking_model=reranking_model,
        reranking_top_n=reranking_top_n,
    )
    validation = context.project.validate_finding(result.finding_id)
    if not validation.valid:
        raise EvidenceValidationError(
            "framework review produced an invalid finding: " + "; ".join(validation.errors)
        )
    return result


def run_synthesis(
    context: SkillContext,
    finding_ids: list[str],
    preset_name: str,
    preset_snapshot: dict[str, object],
    *,
    provider_factory=None,
) -> dict[str, str]:
    """Synthesize only findings already persisted and valid in this run."""
    from .synthesis import synthesize_findings

    return synthesize_findings(
        context.project,
        context.analysis_run_id,
        finding_ids,
        preset_name,
        preset_snapshot,
        provider_factory=provider_factory,
    )


def run_van_dijk_group_comparison(
    context: SkillContext,
    *,
    groups: list[Mapping[str, object]],
    shared_topic_event: str,
    comparability_basis: str,
    codebook_id: str,
) -> VanDijkGroupComparisonResult:
    """Prepare a descriptive comparison from exactly two explicit annotation lists.

    This executor does no retrieval or interpretation. It validates every selected
    annotation against the analysis run's original ScopeSnapshot and counts only
    the annotation IDs the researcher supplied.
    """
    project = context.project
    if not isinstance(groups, list) or len(groups) != 2:
        raise SkillManifestError("group comparison requires exactly two labeled groups")
    if not isinstance(shared_topic_event, str) or not shared_topic_event.strip():
        raise SkillManifestError("group comparison requires a shared topic or event")
    if not isinstance(comparability_basis, str) or not comparability_basis.strip():
        raise SkillManifestError("group comparison requires a comparability basis")
    if not isinstance(codebook_id, str) or not codebook_id.strip():
        raise SkillManifestError("group comparison requires one codebook ID")

    normalized_groups: list[VanDijkGroupComparisonGroupInput] = []
    group_labels: list[str] = []
    all_annotation_ids: list[str] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or set(group) != {"label", "annotation_ids"}:
            raise SkillManifestError(
                "each comparison group must contain only label and annotation_ids"
            )
        label = group["label"]
        annotation_ids = group["annotation_ids"]
        if not isinstance(label, str) or not label.strip():
            raise SkillManifestError(f"comparison group {index + 1} requires a label")
        if not isinstance(annotation_ids, list) or any(
            not isinstance(annotation_id, str) or not annotation_id.strip()
            for annotation_id in annotation_ids
        ):
            raise SkillManifestError(f"comparison group {label!r} requires an annotation ID list")
        if len(annotation_ids) != len(set(annotation_ids)):
            raise SkillManifestError(
                f"comparison group {label!r} contains a duplicate annotation ID"
            )
        normalized_label = label.strip()
        group_labels.append(normalized_label)
        selected_ids = list(cast(list[str], annotation_ids))
        normalized_groups.append({"label": normalized_label, "annotation_ids": selected_ids})
        all_annotation_ids.extend(selected_ids)
    if len({label.casefold() for label in group_labels}) != 2:
        raise SkillManifestError("comparison group labels must be distinct")
    if len(all_annotation_ids) != len(set(all_annotation_ids)):
        raise SkillManifestError("an annotation ID cannot be assigned to both comparison groups")
    if not all_annotation_ids:
        raise SkillManifestError(
            "group comparison requires at least one selected annotation across both groups"
        )

    scope_revision_ids = project.scope_revisions(context.scope_id)
    scope_revision_set = set(scope_revision_ids)
    codebook = project.db.execute(
        "SELECT id,name FROM codebooks WHERE id=?", (codebook_id.strip(),)
    ).fetchone()
    if codebook is None:
        raise SkillManifestError("group comparison references an unknown codebook")
    codebook_key = str(codebook["id"])
    code_rows = project.db.execute(
        "SELECT id,label,definition FROM codes WHERE codebook_id=? "
        "ORDER BY label COLLATE NOCASE,id",
        (codebook_key,),
    ).fetchall()
    code_by_id = {str(row["id"]): row for row in code_rows}

    scoped_annotations = {
        str(row["id"]): row for row in project.annotations_for_scope(context.scope_id)
    }
    selected_rows_by_group: list[list[sqlite3.Row]] = []
    for normalized_group in normalized_groups:
        selected_rows: list[sqlite3.Row] = []
        for annotation_id in normalized_group["annotation_ids"]:
            row = scoped_annotations.get(annotation_id)
            if row is None:
                raise ScopeViolationError(
                    "every selected annotation must be in the original ScopeSnapshot"
                )
            revision_id = str(row["revision_id"])
            if revision_id not in scope_revision_set:
                raise ScopeViolationError(
                    "every selected annotation span must be in the original ScopeSnapshot"
                )
            if str(row["codebook_id"]) != codebook_key:
                raise SkillManifestError(
                    "every selected annotation must belong to the declared codebook"
                )
            if str(row["code_id"]) not in code_by_id:
                raise SkillManifestError(
                    "every selected annotation code must belong to the declared codebook"
                )
            selected_rows.append(row)
        selected_rows_by_group.append(selected_rows)

    selected_annotation_count = len(all_annotation_ids)
    selection_config = {
        "scope_snapshot_id": context.scope_id,
        "groups": normalized_groups,
        "shared_topic_event": shared_topic_event.strip(),
        "comparability_basis": comparability_basis.strip(),
        "codebook_id": codebook_key,
    }
    configuration = json.dumps(
        selection_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXAMPLE_BASED,
            scope_snapshot_id=context.scope_id,
            population_definition=(
                "The distinct annotation IDs explicitly selected by the researcher across "
                "the two labeled groups"
            ),
            population_count=selected_annotation_count,
            selection_policy=(
                "Use exactly the supplied annotation IDs; do not retrieve, sample, or infer "
                "additional annotations"
            ),
            inspection_count=selected_annotation_count,
            completion_criterion=(
                "Validate every supplied annotation, its codebook, and its SourceSpan against "
                "the original ScopeSnapshot"
            ),
            termination_reason="all_selected_annotations_validated",
            termination_state=CoverageTerminationState.COMPLETED,
            negative_result_status=NegativeResultStatus.NOT_APPLICABLE,
        )
    )
    skill_run_id = project.create_skill_run(
        context.analysis_run_id,
        context.scope_id,
        "builtin.van-dijk-group-comparison",
        "1.0.0",
        "builtin.van_dijk_group_comparison",
        hashlib.sha256(configuration.encode("utf-8")).hexdigest(),
        coverage_id=coverage_id,
    )

    limitations = [
        "Group labels, shared topic/event, and comparability basis are researcher supplied and are not independently verified.",
        "All totals and code counts describe only the explicitly selected annotation records, not corpus frequency or prevalence.",
        "An empty annotation selection is UNKNOWN coding and does not establish absence from the corpus.",
        "A zero count for a code describes only the selected annotation list and does not establish absence.",
        "This deterministic preparation does not infer political identity, ideology, motivation, universal prevalence, or an ideology/asymmetry score.",
    ]
    summary_lines = [
        "Descriptive preparation for a researcher-defined two-group comparison.",
        f"Researcher-supplied shared topic/event: {shared_topic_event.strip()}",
        f"Researcher-supplied comparability basis: {comparability_basis.strip()}",
        f"Declared codebook: {codebook['name']} ({codebook_key})",
        (
            f"Counts cover exactly {selected_annotation_count} explicitly selected "
            "annotation record(s); they are not corpus-frequency or prevalence estimates."
        ),
    ]
    for normalized_group, selected_rows in zip(
        normalized_groups, selected_rows_by_group, strict=True
    ):
        label = str(normalized_group["label"])
        group_status = "CODED" if selected_rows else "UNKNOWN"
        summary_lines.append(
            f"Group {json.dumps(label, ensure_ascii=False)}: "
            f"{len(selected_rows)} selected annotation record(s); coding status {group_status}."
        )
        if selected_rows:
            code_counts: dict[str, int] = {code_id: 0 for code_id in code_by_id}
            for row in selected_rows:
                code_counts[str(row["code_id"])] += 1
            for code_id, count in code_counts.items():
                code = code_by_id[code_id]
                summary_lines.append(f"  {code['label']}: {count} selected annotation record(s).")
        else:
            summary_lines.append(
                "  Code distribution: UNKNOWN because no annotations were selected."
            )

    finding_id = project.create_finding(
        context.analysis_run_id,
        "builtin.van-dijk-group-comparison",
        "1.0.0",
        "qualitative_group_comparison_preparation",
        "\n".join(summary_lines),
        skill_run_id=skill_run_id,
        claim_types=[ClaimType.EXAMPLE_BASED],
        evidence_sufficiency="LIMITED",
        interpretation_confidence="LOW",
        limitations=limitations,
        coverage_id=coverage_id,
    )

    evidence_ids: list[str] = []
    output_groups: list[VanDijkGroupComparisonGroup] = []
    for normalized_group, selected_rows in zip(
        normalized_groups, selected_rows_by_group, strict=True
    ):
        label = str(normalized_group["label"])
        output_annotations: list[VanDijkGroupComparisonAnnotation] = []
        if selected_rows:
            code_counts: dict[str, int] = {code_id: 0 for code_id in code_by_id}
            for row in selected_rows:
                code_counts[str(row["code_id"])] += 1
            output_code_counts: list[VanDijkGroupComparisonCodeCount] = [
                {
                    "code_id": code_id,
                    "label": str(code_by_id[code_id]["label"]),
                    "selected_annotation_count": count,
                }
                for code_id, count in code_counts.items()
            ]
            for row in selected_rows:
                annotation_id = str(row["id"])
                code_id = str(row["code_id"])
                span_id = str(row["span_id"])
                code = code_by_id[code_id]
                evidence_id = project.textual_evidence(
                    span_id,
                    provenance={
                        "skill_id": "builtin.van-dijk-group-comparison",
                        "skill_version": "1.0.0",
                        "scope_snapshot_id": context.scope_id,
                        "annotation_id": annotation_id,
                        "codebook_id": codebook_key,
                        "codebook_name": str(codebook["name"]),
                        "code_id": code_id,
                        "code_label": str(code["label"]),
                        "group_label": label,
                        "shared_topic_event": shared_topic_event.strip(),
                        "comparability_basis": comparability_basis.strip(),
                    },
                )
                citation = project.render_citation(evidence_id)
                project.attach_evidence(
                    finding_id,
                    evidence_id,
                    EvidenceRole.EXAMPLE,
                    (
                        f"Selected annotation {annotation_id} for researcher-labeled group "
                        f"{label!r}; the attached text is the annotation's canonical SourceSpan."
                    ),
                )
                evidence_ids.append(evidence_id)
                output_annotations.append(
                    {
                        "annotation_id": annotation_id,
                        "code_id": code_id,
                        "code_label": str(code["label"]),
                        "code_definition": (
                            str(code["definition"]) if code["definition"] is not None else None
                        ),
                        "span_id": span_id,
                        "revision_id": str(row["revision_id"]),
                        "start_offset": int(row["start_offset"]),
                        "end_offset": int(row["end_offset"]),
                        "text": project.span_text(span_id),
                        "evidence_id": evidence_id,
                        "citation": citation,
                    }
                )
        else:
            output_code_counts = []
        output_groups.append(
            {
                "label": label,
                "selected_annotation_count": len(selected_rows),
                "coding_status": "CODED" if selected_rows else "UNKNOWN",
                "code_counts": output_code_counts,
                "annotations": output_annotations,
            }
        )

    validation = project.validate_finding(finding_id)
    if not validation.valid:
        raise EvidenceValidationError(
            "group comparison preparation produced an invalid finding: "
            + "; ".join(validation.errors)
        )
    project.complete_skill_run(skill_run_id)
    return {
        "skill_run_id": skill_run_id,
        "finding_id": finding_id,
        "coverage_id": coverage_id,
        "scope_snapshot_id": context.scope_id,
        "artifact_type": "descriptive_qualitative_group_comparison_preparation",
        "shared_topic_event": shared_topic_event.strip(),
        "comparability_basis": comparability_basis.strip(),
        "codebook_id": codebook_key,
        "codebook_name": str(codebook["name"]),
        "count_basis": "SELECTED_ANNOTATIONS",
        "claim_semantics": "EXAMPLE_BASED",
        "selected_annotation_count": selected_annotation_count,
        "groups": output_groups,
        "evidence_ids": evidence_ids,
        "limitations": limitations,
    }


BUILTIN_SKILLS = SkillRegistry()
BUILTIN_SKILLS.register(
    SkillDefinition(parse_manifest(TEXTUAL_DESCRIPTORS_MANIFEST), run_textual_descriptors)
)
BUILTIN_SKILLS.register(
    SkillDefinition(parse_manifest(EVIDENCE_EXTRACTION_MANIFEST), run_evidence_extraction)
)
BUILTIN_SKILLS.register(
    SkillDefinition(
        parse_manifest(QUOTE_ATTRIBUTION_CANDIDATES_MANIFEST),
        run_quote_attribution_candidates,
    )
)
BUILTIN_SKILLS.register(
    SkillDefinition(
        parse_manifest(SCOPE_QUESTION_CANDIDATES_MANIFEST), run_scope_question_candidates
    )
)
BUILTIN_SKILLS.register(
    SkillDefinition(
        parse_manifest(SCOPE_QUESTION_HYBRID_CANDIDATES_MANIFEST),
        run_scope_question_hybrid_candidates,
    )
)
BUILTIN_SKILLS.register(
    SkillDefinition(parse_manifest(COUNTEREVIDENCE_LEXICAL_MANIFEST), run_counterevidence_lexical)
)
BUILTIN_SKILLS.register(
    SkillDefinition(parse_manifest(SOCIAL_ACTOR_CANDIDATES_MANIFEST), run_social_actor_candidates)
)
BUILTIN_SKILLS.register(
    SkillDefinition(parse_manifest(AGENCY_CANDIDATES_MANIFEST), run_agency_candidates)
)
for framework_manifest in FRAMEWORK_REVIEW_MANIFESTS:
    BUILTIN_SKILLS.register(
        SkillDefinition(parse_manifest(framework_manifest), run_framework_review)
    )
BUILTIN_SKILLS.register(SkillDefinition(parse_manifest(SYNTHESIS_MANIFEST), run_synthesis))
BUILTIN_SKILLS.register(
    SkillDefinition(
        parse_manifest(VAN_DIJK_GROUP_COMPARISON_MANIFEST), run_van_dijk_group_comparison
    )
)
