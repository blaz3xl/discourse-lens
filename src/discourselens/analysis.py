"""Controlled analysis orchestration over registered, evidence-bound steps."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Mapping
from typing import Literal, TypedDict, cast, overload

from .errors import (
    DocumentError,
    EvidenceValidationError,
    ModelProviderError,
    ProjectError,
    RetrievalLimitError,
)
from .execution import (
    ProviderFactory,
    build_reranking_gateway,
    credential_reference_fingerprint,
    snapshot_model_preset,
)
from .frameworks import FRAMEWORK_REVIEWS
from .interpretation import (
    FRAMEWORK_REVIEW_PROMPT_TEMPLATE_REF,
    FRAMEWORK_REVIEW_PROMPT_TEMPLATE_VERSION,
    framework_review_prompt_template_sha256,
)
from .languages import (
    OPERATIONAL_LIMITED,
    OPERATIONAL_SUPPORTED,
    OPERATIONAL_UNASSESSED,
    OPERATIONAL_UNSUPPORTED,
    local_execution_preflight,
)
from .models import EmbeddingProvider, ModelGateway
from .profiles import ControlledCdaProfile
from .provider_catalog import is_reviewed_reranking_provider
from .scope_retrieval import MAX_FRAMEWORK_HITS
from .skill_contracts import EVIDENCE_EXTRACTION_FINDING_PHENOMENA
from .skills import (
    EVIDENCE_CUE_PATTERNS,
    MODAL_MARKERS,
    DependencyParseSnapshot,
    QuoteAttributionCandidatesResult,
    SkillContext,
    VanDijkGroupComparisonGroupInput,
    VanDijkGroupComparisonResult,
    builtin_manifest,
    run_agency_candidates,
    run_counterevidence_lexical,
    run_evidence_extraction,
    run_exact_phrase_fixture,
    run_framework_review,
    run_quote_attribution_candidates,
    run_scope_question_candidates,
    run_social_actor_candidates,
    run_synthesis,
    run_textual_descriptors,
    run_van_dijk_group_comparison,
)
from .synthesis import SYNTHESIS_PROMPT_REF, synthesis_prompt_template_sha256
from .types import CoverageDescriptor, CoverageMode, EvidenceRole, RunStatus


class AnalysisResult(TypedDict):
    analysis_run_id: str
    skill_run_id: str
    finding_ids: list[str]


class TextualDescriptorAnalysisResult(AnalysisResult):
    coverage_id: str


class EvidenceExtractionAnalysisResult(TypedDict):
    analysis_run_id: str
    skill_run_id: str
    finding_ids: list[str]
    evidence_ids: list[str]
    coverage_id: str
    population_count: int
    candidate_count: int
    selected_count: int


class QuoteAttributionAnalysisResult(QuoteAttributionCandidatesResult):
    analysis_run_id: str
    scope_snapshot_id: str


class ScopeQuestionAnalysisResult(TypedDict):
    analysis_run_id: str
    scope_snapshot_id: str
    skill_run_id: str
    retrieval_run_id: str
    finding_id: str | None
    evidence_ids: list[str]
    coverage_id: str
    candidate_count: int
    coverage: CoverageDescriptor
    retrieval_mode: Literal["lexical", "hybrid"]


class AskQuestionEvidence(TypedDict):
    evidence_id: str
    role: str
    citation: str
    excerpt: str
    excerpt_truncated: bool


class AskQuestionResult(TypedDict):
    status: Literal["answered", "no_candidates"]
    question: str
    framework_id: str
    retrieval_mode: Literal["lexical", "hybrid"]
    scope_snapshot_id: str
    candidate_analysis_run_id: str
    candidate_skill_run_id: str
    candidate_finding_id: str | None
    candidate_retrieval_run_id: str
    coverage_id: str
    coverage: dict[str, object]
    answer_coverage_id: str | None
    answer_coverage: dict[str, object] | None
    answer_analysis_run_id: str | None
    answer_finding_id: str | None
    model_execution_id: str | None
    answer_interpretation: str | None
    alternative_interpretation: str | None
    limitations: list[str]
    evidence: list[AskQuestionEvidence]
    evidence_total: int
    evidence_truncated: bool


class VanDijkGroupComparisonAnalysisResult(VanDijkGroupComparisonResult):
    analysis_run_id: str


class CounterEvidenceAnalysisResult(TypedDict):
    analysis_run_id: str
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


class CounterEvidenceStatus(TypedDict):
    status: str
    query: str | None
    skill_run_id: str | None
    coverage_id: str | None
    retrieval_run_id: str | None
    candidate_count: int | None
    reason: str | None


class AlternativeCandidateStatus(TypedDict):
    status: str
    retrieval_run_ids: list[str]
    candidate_count: int
    coverage_mode: str | None
    scope_id: str
    reason: str | None


class ControlledCdaResult(TypedDict):
    """Result of an explicit, evidence-bound multi-stage review pipeline."""

    analysis_run_id: str
    seed_finding_ids: list[str]
    reviewed_finding_ids: list[str]
    critical_review_finding_ids: list[str]
    counterevidence_skill_run_ids: list[str]
    counterevidence_statuses: dict[str, CounterEvidenceStatus]
    alternative_candidate_statuses: dict[str, AlternativeCandidateStatus]
    synthesis_finding_id: str | None
    local_nlp_skill_run_ids: list[str]
    local_nlp_finding_ids: list[str]
    local_nlp_execution_summary: LocalNlpExecutionSummary
    scope_question_finding_ids: dict[str, str]
    scope_question_skill_run_ids: list[str]
    warnings: list[str]


class SavedProfileBatchCompleted(TypedDict):
    revision_id: str
    status: Literal["completed"]
    result: ControlledCdaResult


class SavedProfileBatchError(TypedDict):
    type: str
    message: str


class SavedProfileBatchFailed(TypedDict):
    revision_id: str
    status: Literal["failed"]
    analysis_run_id: str | None
    error: SavedProfileBatchError


SavedProfileBatchItem = SavedProfileBatchCompleted | SavedProfileBatchFailed


class CueLanguageExecutionPlan(TypedDict):
    languages: list[str]
    revision_ids: list[str] | None
    revision_languages: dict[str, str] | None
    warnings: list[str]
    skip_reason: str | None


class LocalNlpSkippedRevision(TypedDict):
    language: str
    capability: str
    status: str
    limitations: list[str]


class LocalNlpOperationPlan(TypedDict):
    status: str
    required_capability: str
    eligible_revision_ids: list[str]
    processed_revision_ids: list[str]
    skipped_revision_ids: list[str]
    skipped_revisions: dict[str, LocalNlpSkippedRevision]


class LocalNlpExecutionPlan(TypedDict):
    status: str
    operations: dict[str, LocalNlpOperationPlan]
    warnings: list[str]


class LocalNlpOperationExecutionSummary(TypedDict):
    status: str
    required_capability: str
    processed_revision_ids: list[str]
    skipped_revision_ids: list[str]
    skipped_revisions: dict[str, LocalNlpSkippedRevision]


class LocalNlpExecutionSummary(TypedDict):
    status: str
    operations: dict[str, LocalNlpOperationExecutionSummary]


DEFAULT_CDA_FRAMEWORKS = tuple(FRAMEWORK_REVIEWS)
CONTROLLED_CDA_PROMPT_SNAPSHOT_VERSION = 2
MAX_ASK_QUESTION_EVIDENCE_ITEMS = 24
MAX_ASK_QUESTION_EXCERPT_CHARS = 2_000


def _controlled_cda_prompt_snapshots(frameworks: tuple[str, ...]) -> list[dict[str, str]]:
    """Resolve stable references and content digests for selected built-in prompts."""
    snapshots: list[dict[str, str]] = []
    for framework_id in frameworks:
        profile = FRAMEWORK_REVIEWS.get(framework_id)
        if profile is None:
            raise ProjectError(f"unknown framework review {framework_id!r}")
        manifest = builtin_manifest(f"builtin.{framework_id}")
        if len(manifest.prompt_refs) != 1:
            raise ProjectError(
                f"framework review {framework_id!r} does not resolve to one built-in prompt reference"
            )
        snapshots.append(
            {
                "framework_id": framework_id,
                "framework_version": profile.version,
                "skill_id": manifest.id,
                "skill_version": manifest.version,
                "prompt_ref": manifest.prompt_refs[0],
                "prompt_template_ref": FRAMEWORK_REVIEW_PROMPT_TEMPLATE_REF,
                "prompt_template_version": str(FRAMEWORK_REVIEW_PROMPT_TEMPLATE_VERSION),
                "prompt_template_sha256": framework_review_prompt_template_sha256(profile),
            }
        )
    return snapshots


def _controlled_cda_synthesis_prompt_snapshot() -> dict[str, str]:
    """Resolve the single prompt reference used by the built-in synthesis skill."""
    manifest = builtin_manifest("builtin.synthesis")
    if manifest.prompt_refs != [SYNTHESIS_PROMPT_REF]:
        raise ProjectError(
            "built-in synthesis prompt reference does not match its registered skill manifest"
        )
    return {
        "framework_id": manifest.framework,
        "framework_version": manifest.version,
        "skill_id": manifest.id,
        "skill_version": manifest.version,
        "prompt_ref": SYNTHESIS_PROMPT_REF,
        "prompt_template_sha256": synthesis_prompt_template_sha256(),
    }


def _resolve_framework_presets(
    project,
    frameworks: tuple[str, ...],
    default_preset: str | None,
    routing_profile: str | None,
) -> dict[str, str]:
    """Resolve every stage before execution; missing routing is a hard error."""
    if not routing_profile:
        if not default_preset:
            raise ProjectError(
                "controlled CDA requires a default preset or a complete routing profile"
            )
        project.model_preset(default_preset)
        return {
            **{framework: default_preset for framework in frameworks},
            "synthesis": default_preset,
        }
    routes, fallback = project.model_routing_profile(routing_profile)
    unknown_stages = sorted(set(routes) - (set(FRAMEWORK_REVIEWS) | {"synthesis"}))
    if unknown_stages:
        raise ProjectError(
            f"routing profile {routing_profile!r} contains unknown stage route(s): "
            + ", ".join(unknown_stages)
        )
    resolved: dict[str, str] = {}
    for framework in frameworks:
        selected = routes.get(framework, fallback)
        if not selected:
            raise ProjectError(
                f"routing profile {routing_profile!r} has no preset for framework {framework!r}"
            )
        resolved[framework] = selected
    synthesis_preset = routes.get("synthesis", fallback)
    if not synthesis_preset:
        raise ProjectError(
            f"routing profile {routing_profile!r} requires a synthesis route or fallback preset"
        )
    resolved["synthesis"] = synthesis_preset
    return resolved


def _completed_seed_findings(project, analysis_run_id: str) -> list[str]:
    rows = project.db.execute(
        """
        SELECT f.id FROM findings f JOIN skill_runs s ON s.id=f.skill_run_id
        WHERE f.analysis_run_id=? AND s.skill_id='builtin.evidence-extraction'
          AND s.status='COMPLETED'
        ORDER BY f.created_at,f.id
        """,
        (analysis_run_id,),
    ).fetchall()
    return [row["id"] for row in rows]


def _base_language_code(value: str) -> str:
    return value.casefold().replace("_", "-").split("-", maxsplit=1)[0]


def _ordered_unique_languages(languages: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize language codes and keep each code's first occurrence."""
    return tuple(dict.fromkeys(language.strip().casefold() for language in languages))


def _normalize_counterevidence_queries(value: object) -> dict[str, str]:
    """Reject query mappings that the current seed-finding inventory cannot use."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ProjectError("counterevidence queries must be a phenomenon-to-query mapping")
    normalized: dict[str, str] = {}
    for raw_phenomenon, raw_query in value.items():
        if not isinstance(raw_phenomenon, str) or not isinstance(raw_query, str):
            raise ProjectError("counterevidence phenomena and queries must be strings")
        phenomenon = raw_phenomenon.strip()
        query = raw_query.strip()
        if not phenomenon or not query:
            raise ProjectError("counterevidence phenomena and queries must be non-empty")
        if phenomenon in normalized:
            raise ProjectError("counterevidence queries contain duplicate normalized phenomena")
        normalized[phenomenon] = query
    unknown = sorted(set(normalized) - EVIDENCE_EXTRACTION_FINDING_PHENOMENA)
    if unknown:
        raise ProjectError(
            "counterevidence queries target no current evidence-extraction finding phenomenon: "
            + ", ".join(unknown)
        )
    return normalized


def _cue_language_execution_plan(
    project, scope_id: str, declared_languages: tuple[str, ...]
) -> CueLanguageExecutionPlan:
    """Choose only cue rules that can be assigned to the scoped documents safely."""
    declared = tuple(
        dict.fromkeys(_base_language_code(item.strip()) for item in declared_languages)
    )
    if not declared or any(not item for item in declared):
        raise ProjectError("controlled CDA requires at least one non-empty language code")
    supported = set(MODAL_MARKERS).intersection(EVIDENCE_CUE_PATTERNS)
    unsupported = set(declared) - supported
    revisions = project.scope_revisions(scope_id)
    assignments: dict[str, str] = {}
    missing_assignments: list[str] = []
    for revision_id in revisions:
        row = project.db.execute(
            "SELECT document_id FROM document_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        if row is None:
            raise ProjectError("language planning found an unknown scoped revision")
        language = project.document_metadata(str(row["document_id"])).get("language")
        if not isinstance(language, str) or not language.strip():
            missing_assignments.append(revision_id)
        else:
            assignments[revision_id] = _base_language_code(language.strip())

    warnings: list[str] = []
    if unsupported:
        warnings.append(
            "Cue-based textual descriptors and evidence extraction have no configured "
            "language-matched rules for: " + ", ".join(sorted(unsupported)) + "."
        )

    eligible = {
        revision_id: language
        for revision_id, language in assignments.items()
        if language in declared and language in supported
    }
    outside_profile = sorted(set(assignments.values()) - set(declared))
    if outside_profile:
        warnings.append(
            "Cue-based stages excluded documents whose declared language is outside the "
            "profile: " + ", ".join(outside_profile) + "."
        )

    if missing_assignments:
        # A single declared, mechanically supported language can be treated as a
        # project/profile assertion for unlabelled revisions, but record that
        # assumption in the frozen per-revision plan. A multilingual profile does
        # not reveal which rule set belongs to an unlabelled document.
        singleton_is_consistent = (
            len(declared) == 1
            and declared[0] in supported
            and all(language == declared[0] for language in assignments.values())
        )
        if singleton_is_consistent:
            eligible.update({revision_id: declared[0] for revision_id in missing_assignments})
            warnings.append(
                "Per-document language metadata is missing for some scoped revisions; the "
                f"single profile-declared language {declared[0]!r} is applied to those revisions "
                "as an explicit assumption, not as detected language."
            )
        else:
            warnings.append(
                f"Cue-based stages were skipped for {len(missing_assignments)} revision(s) because "
                "the scope lacks per-document language metadata and the profile does not identify "
                "one safe language assignment for them."
            )

    excluded = len(revisions) - len(eligible)
    if excluded and eligible:
        warnings.append(
            f"Cue-based stages will inspect only {len(eligible)} of {len(revisions)} "
            "scoped revisions with an eligible language assignment."
        )
    languages = sorted(set(eligible.values()))
    if eligible:
        return {
            "languages": languages,
            "revision_ids": list(eligible),
            "revision_languages": eligible,
            "warnings": warnings,
            "skip_reason": None,
        }

    skip_reason = (
        "LANGUAGE_ASSIGNMENT_REQUIRED_FOR_PARTIAL_CUE_EXECUTION"
        if missing_assignments
        else "NO_LANGUAGE_MATCHED_SUPPORTED_CUE_INVENTORY"
    )
    return {
        "languages": [],
        "revision_ids": [],
        "revision_languages": {},
        "warnings": warnings,
        "skip_reason": skip_reason,
    }


def _evidence_extraction_completed(project, analysis_run_id: str) -> bool:
    return bool(
        project.db.execute(
            "SELECT 1 FROM skill_runs WHERE analysis_run_id=? "
            "AND skill_id='builtin.evidence-extraction' AND status IN ('COMPLETED','SKIPPED') LIMIT 1",
            (analysis_run_id,),
        ).fetchone()
    )


def _record_cue_skill_skip(
    project,
    run_id: str,
    scope_id: str,
    skill_id: str,
    executor: str,
    plan: CueLanguageExecutionPlan,
) -> None:
    reason = plan["skip_reason"]
    if not reason:
        raise ProjectError("cue skill skip requires a language-plan reason")
    config_hash = hashlib.sha256(
        json.dumps({"cue_language_plan": plan, "skip_reason": reason}, sort_keys=True).encode()
    ).hexdigest()
    existing = project.db.execute(
        "SELECT 1 FROM skill_runs WHERE analysis_run_id=? AND skill_id=? "
        "AND config_hash=? AND status='SKIPPED' LIMIT 1",
        (run_id, skill_id, config_hash),
    ).fetchone()
    if existing:
        return
    skill_run_id = project.create_skill_run(
        run_id,
        scope_id,
        skill_id,
        "1.0.0",
        executor,
        config_hash,
    )
    project.skip_skill_run(skill_run_id, reason)


def _completed_review_finding(
    project, analysis_run_id: str, skill_id: str, input_finding_id: str
) -> str | None:
    row = project.db.execute(
        """
        SELECT f.id FROM findings f JOIN skill_runs s ON s.id=f.skill_run_id
        WHERE f.analysis_run_id=? AND s.skill_id=? AND s.input_finding_id=?
          AND s.status='COMPLETED'
        ORDER BY f.created_at,f.id LIMIT 1
        """,
        (analysis_run_id, skill_id, input_finding_id),
    ).fetchone()
    return str(row["id"]) if row else None


def _completed_synthesis_finding(project, analysis_run_id: str) -> str | None:
    row = project.db.execute(
        "SELECT f.id FROM findings f JOIN skill_runs s ON s.id=f.skill_run_id "
        "WHERE f.analysis_run_id=? AND f.skill_id='builtin.synthesis' AND s.status='COMPLETED' "
        "ORDER BY f.created_at,f.id LIMIT 1",
        (analysis_run_id,),
    ).fetchone()
    return str(row["id"]) if row else None


def _completed_counterevidence_skill(
    project, analysis_run_id: str, input_finding_id: str, phrase: str
) -> str | None:
    configuration = json.dumps(
        {
            "finding_id": input_finding_id,
            "phrase": phrase,
            "retrieval": "exhaustive-literal-match-candidates-v1",
        },
        sort_keys=True,
    )
    row = project.db.execute(
        """
        SELECT id FROM skill_runs
        WHERE analysis_run_id=? AND skill_id='builtin.counterevidence-lexical'
          AND input_finding_id=? AND config_hash=? AND status='COMPLETED'
        ORDER BY started_at,id LIMIT 1
        """,
        (analysis_run_id, input_finding_id, hashlib.sha256(configuration.encode()).hexdigest()),
    ).fetchone()
    return str(row["id"]) if row else None


def _retrieval_run_for_skill(project, skill_run_id: str) -> str | None:
    row = project.db.execute(
        "SELECT id FROM retrieval_runs WHERE skill_run_id=? ORDER BY created_at,id LIMIT 1",
        (skill_run_id,),
    ).fetchone()
    return str(row["id"]) if row else None


def _retrieval_hit_count(project, retrieval_run_id: str | None) -> int | None:
    if retrieval_run_id is None:
        return None
    row = project.db.execute(
        "SELECT COUNT(*) AS count FROM retrieval_hits WHERE retrieval_run_id=?",
        (retrieval_run_id,),
    ).fetchone()
    return int(row["count"])


def _alternative_candidate_statuses(
    project,
    run_id: str,
    scope_id: str,
    seed_finding_ids: list[str],
    frameworks: tuple[str, ...],
) -> dict[str, AlternativeCandidateStatus]:
    """Report the separate automatic alternative-reading retrieval per seed."""
    if "critical-reviewer" not in frameworks:
        return {
            seed_id: {
                "status": "NOT_RUN",
                "retrieval_run_ids": [],
                "candidate_count": 0,
                "coverage_mode": None,
                "scope_id": scope_id,
                "reason": "Critical reviewer was not selected; no automatic alternative-reading search ran.",
            }
            for seed_id in seed_finding_ids
        }
    statuses: dict[str, AlternativeCandidateStatus] = {}
    for seed_id in seed_finding_ids:
        rows = project.db.execute(
            "SELECT r.id,r.returned_count FROM retrieval_runs r "
            "JOIN skill_runs critical_skill ON critical_skill.id=r.skill_run_id "
            "JOIN findings source_finding ON source_finding.id=critical_skill.input_finding_id "
            "JOIN skill_runs source_skill ON source_skill.id=source_finding.skill_run_id "
            "WHERE r.analysis_run_id=? AND r.scope_id=? "
            "AND r.retrieval_plan_ref='retrieval/counterevidence-alternative-lexical-v1' "
            "AND critical_skill.skill_id='builtin.critical-reviewer' "
            "AND critical_skill.status='COMPLETED' AND source_skill.status='COMPLETED' "
            "AND source_skill.input_finding_id=? ORDER BY r.created_at,r.id",
            (run_id, scope_id, seed_id),
        ).fetchall()
        statuses[seed_id] = {
            "status": "COMPLETED" if rows else "NOT_RUN",
            "retrieval_run_ids": [str(row["id"]) for row in rows],
            "candidate_count": sum(int(row["returned_count"]) for row in rows),
            "coverage_mode": "TARGETED" if rows else None,
            "scope_id": scope_id,
            "reason": (
                "Targeted alternative-reading candidates were searched; an empty result does not establish absence."
                if rows
                else "No completed critical-reviewer alternative-reading retrieval was recorded."
            ),
        }
    return statuses


def _skipped_counterevidence_skill(
    project, analysis_run_id: str, input_finding_id: str
) -> str | None:
    configuration = json.dumps(
        {
            "finding_id": input_finding_id,
            "retrieval": "exhaustive-literal-match-candidates-v1",
            "skip_reason": "NO_RESEARCHER_PROVIDED_CONTRAST_QUERY",
        },
        sort_keys=True,
    )
    row = project.db.execute(
        "SELECT id FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.counterevidence-lexical' AND input_finding_id=? "
        "AND config_hash=? AND status='SKIPPED' ORDER BY started_at,id LIMIT 1",
        (analysis_run_id, input_finding_id, hashlib.sha256(configuration.encode()).hexdigest()),
    ).fetchone()
    return str(row["id"]) if row else None


def _counterevidence_limit_skip(
    project,
    analysis_run_id: str,
    scope_id: str,
    input_finding_id: str,
    phrase: str,
    *,
    limit: int | None = None,
) -> tuple[str, str] | None:
    """Reuse or persist an explicit, resumable skip for an over-limit query."""
    existing = project.db.execute(
        "SELECT id,skip_reason FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.counterevidence-lexical' AND input_finding_id=? "
        "AND status='SKIPPED' AND skip_reason LIKE 'RETRIEVAL_LIMIT_REACHED:%' "
        "ORDER BY started_at,id LIMIT 1",
        (analysis_run_id, input_finding_id),
    ).fetchone()
    if existing:
        return str(existing["id"]), str(existing["skip_reason"])
    if limit is None:
        return None

    reason = (
        f"RETRIEVAL_LIMIT_REACHED: more than {limit:,} literal occurrences; "
        "no candidate spans or retrieval hits were persisted. Refine the query."
    )
    configuration = json.dumps(
        {
            "finding_id": input_finding_id,
            "phrase": phrase,
            "retrieval": "exhaustive-literal-match-candidates-v1",
            "limit": limit,
            "skip_reason": "RETRIEVAL_LIMIT_REACHED",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    skill_run_id = project.create_skill_run(
        analysis_run_id,
        scope_id,
        "builtin.counterevidence-lexical",
        "2.0.0",
        "builtin.counterevidence_lexical",
        hashlib.sha256(configuration.encode("utf-8")).hexdigest(),
        input_finding_id=input_finding_id,
    )
    project.skip_skill_run(skill_run_id, reason)
    return skill_run_id, reason


def _record_no_seed_skips(project, run_id: str, scope_id: str, frameworks: tuple[str, ...]) -> None:
    """Persist configured analysis stages that cannot run without seed evidence."""
    reason = "NO_DESCRIPTIVE_SEED_EVIDENCE"
    for framework in frameworks:
        skill_id = f"builtin.{framework}"
        config_hash = hashlib.sha256(
            json.dumps({"framework": framework, "skip_reason": reason}, sort_keys=True).encode()
        ).hexdigest()
        existing = project.db.execute(
            "SELECT 1 FROM skill_runs WHERE analysis_run_id=? AND skill_id=? "
            "AND config_hash=? AND status='SKIPPED' LIMIT 1",
            (run_id, skill_id, config_hash),
        ).fetchone()
        if existing:
            continue
        skill_run_id = project.create_skill_run(
            run_id,
            scope_id,
            skill_id,
            FRAMEWORK_REVIEWS[framework].version,
            "builtin.evidence_bound_review",
            config_hash,
        )
        project.skip_skill_run(skill_run_id, reason)
    config_hash = hashlib.sha256(
        json.dumps({"skip_reason": "NO_VALIDATED_REVIEW_FINDINGS"}, sort_keys=True).encode()
    ).hexdigest()
    existing = project.db.execute(
        "SELECT 1 FROM skill_runs WHERE analysis_run_id=? AND skill_id='builtin.synthesis' "
        "AND config_hash=? AND status='SKIPPED' LIMIT 1",
        (run_id, config_hash),
    ).fetchone()
    if not existing:
        synthesis_run_id = project.create_skill_run(
            run_id,
            scope_id,
            "builtin.synthesis",
            "1.0.0",
            "builtin.evidence_constrained_synthesis",
            config_hash,
        )
        project.skip_skill_run(synthesis_run_id, "NO_VALIDATED_REVIEW_FINDINGS")


def _record_framework_skip(project, run_id: str, scope_id: str, framework: str) -> None:
    """Persist a framework omission when neither seed source found candidates."""
    reason = "NO_SEED_OR_QUESTION_CANDIDATE_EVIDENCE"
    skill_id = f"builtin.{framework}"
    config_hash = hashlib.sha256(
        json.dumps({"framework": framework, "skip_reason": reason}, sort_keys=True).encode()
    ).hexdigest()
    existing = project.db.execute(
        "SELECT 1 FROM skill_runs WHERE analysis_run_id=? AND skill_id=? "
        "AND config_hash=? AND status='SKIPPED' LIMIT 1",
        (run_id, skill_id, config_hash),
    ).fetchone()
    if existing:
        return
    skill_run_id = project.create_skill_run(
        run_id,
        scope_id,
        skill_id,
        FRAMEWORK_REVIEWS[framework].version,
        "builtin.evidence_bound_review",
        config_hash,
    )
    project.skip_skill_run(skill_run_id, reason)


_LOCAL_PREPROCESSOR_CAPABILITIES = {
    "social_actor_candidates": "pos",
    "agency_candidates": "dependency_parse",
}
_LOCAL_OPERATIONAL_STATUSES = {
    OPERATIONAL_SUPPORTED,
    OPERATIONAL_LIMITED,
    OPERATIONAL_UNSUPPORTED,
    OPERATIONAL_UNASSESSED,
    "EXPERIMENTAL",
}
_DEPENDENCY_PARSE_STATUSES = {
    OPERATIONAL_SUPPORTED,
    OPERATIONAL_LIMITED,
    OPERATIONAL_UNSUPPORTED,
    OPERATIONAL_UNASSESSED,
}


def _build_local_nlp_execution_plan(
    revision_ids: list[str],
    revision_languages: Mapping[str, str],
    language_preflight: Mapping[str, Mapping[str, object]],
    local_preprocessors: tuple[str, ...],
) -> LocalNlpExecutionPlan:
    """Freeze per-operation eligibility without shrinking the AnalysisRun scope."""
    operations: dict[str, LocalNlpOperationPlan] = {}
    warnings: list[str] = []
    for preprocessor in local_preprocessors:
        capability = _LOCAL_PREPROCESSOR_CAPABILITIES.get(preprocessor)
        if capability is None:
            raise ProjectError(f"unknown saved local NLP preprocessor {preprocessor!r}")
        eligible: list[str] = []
        skipped: dict[str, LocalNlpSkippedRevision] = {}
        for revision_id in revision_ids:
            language = revision_languages.get(revision_id)
            preflight = language_preflight.get(language or "")
            if not isinstance(language, str) or not isinstance(preflight, Mapping):
                raise ProjectError("local NLP preflight does not cover every scoped language")
            raw_statuses = preflight.get("statuses", {})
            if not isinstance(raw_statuses, Mapping):
                raise ProjectError("local NLP preflight capability statuses are malformed")
            raw_status = raw_statuses.get(capability, OPERATIONAL_UNASSESSED)
            if not isinstance(raw_status, str) or raw_status not in _LOCAL_OPERATIONAL_STATUSES:
                raise ProjectError("local NLP preflight contains an unknown capability status")
            if raw_status == OPERATIONAL_SUPPORTED:
                eligible.append(revision_id)
                continue
            raw_limitations = preflight.get("limitations", [])
            if not isinstance(raw_limitations, list) or not all(
                isinstance(item, str) for item in raw_limitations
            ):
                raise ProjectError("local NLP preflight limitations are malformed")
            limitations = [
                item
                for item in raw_limitations
                if item.startswith((f"{capability}:", "local NLP route:"))
            ]
            if not limitations:
                limitations = [f"{capability}: {raw_status}"]
            skipped[revision_id] = {
                "language": language,
                "capability": capability,
                "status": raw_status,
                "limitations": limitations,
            }
        status = "COMPLETED" if not skipped else "PARTIAL" if eligible else "SKIPPED"
        operations[preprocessor] = {
            "status": status,
            "required_capability": capability,
            "eligible_revision_ids": eligible,
            "processed_revision_ids": list(eligible),
            "skipped_revision_ids": list(skipped),
            "skipped_revisions": skipped,
        }
        if skipped:
            rendered = "; ".join(
                f"{revision_id} ({item['language']}, {capability}={item['status']}; "
                f"{', '.join(item['limitations'])})"
                for revision_id, item in skipped.items()
            )
            warnings.append(
                f"Local NLP operation {preprocessor!r} is {status.lower()}: "
                f"{len(eligible)}/{len(revision_ids)} scoped revision(s) are eligible; "
                f"skipped {rendered}. No language or model fallback was attempted."
            )
    if not operations or all(item["status"] == "COMPLETED" for item in operations.values()):
        overall_status = "COMPLETED" if operations else "NOT_CONFIGURED"
    elif all(item["status"] == "SKIPPED" for item in operations.values()):
        overall_status = "SKIPPED"
    else:
        overall_status = "PARTIAL"
    return {"status": overall_status, "operations": operations, "warnings": warnings}


def _validated_local_nlp_execution_plan(
    raw_plan: object,
    local_preprocessors: tuple[str, ...],
    scoped_revision_ids: list[str],
) -> LocalNlpExecutionPlan:
    """Validate immutable operation partitions before resuming a saved run."""
    if not isinstance(raw_plan, dict):
        raise ProjectError("analysis run has malformed local NLP execution provenance")
    raw_operations = raw_plan.get("operations")
    raw_warnings = raw_plan.get("warnings")
    if (
        not isinstance(raw_operations, dict)
        or set(raw_operations) != set(local_preprocessors)
        or not isinstance(raw_warnings, list)
        or not all(isinstance(item, str) for item in raw_warnings)
    ):
        raise ProjectError("analysis run has malformed local NLP execution provenance")
    expected_scope = set(scoped_revision_ids)
    operations: dict[str, LocalNlpOperationPlan] = {}
    for preprocessor in local_preprocessors:
        expected_capability = _LOCAL_PREPROCESSOR_CAPABILITIES.get(preprocessor)
        raw_operation = raw_operations[preprocessor]
        if expected_capability is None or not isinstance(raw_operation, dict):
            raise ProjectError("analysis run has malformed local NLP operation provenance")
        raw_eligible = raw_operation.get("eligible_revision_ids")
        raw_processed = raw_operation.get("processed_revision_ids")
        raw_skipped_ids = raw_operation.get("skipped_revision_ids")
        raw_skipped = raw_operation.get("skipped_revisions")
        if (
            raw_operation.get("required_capability") != expected_capability
            or not isinstance(raw_eligible, list)
            or not all(isinstance(item, str) for item in raw_eligible)
            or len(raw_eligible) != len(set(raw_eligible))
            or not isinstance(raw_processed, list)
            or not all(isinstance(item, str) for item in raw_processed)
            or not isinstance(raw_skipped_ids, list)
            or not all(isinstance(item, str) for item in raw_skipped_ids)
            or not isinstance(raw_skipped, dict)
        ):
            raise ProjectError("analysis run has malformed local NLP operation provenance")
        eligible = list(raw_eligible)
        skipped: dict[str, LocalNlpSkippedRevision] = {}
        for revision_id, raw_item in raw_skipped.items():
            raw_status = raw_item.get("status") if isinstance(raw_item, dict) else None
            if (
                not isinstance(revision_id, str)
                or not isinstance(raw_item, dict)
                or not isinstance(raw_item.get("language"), str)
                or raw_item.get("capability") != expected_capability
                or not isinstance(raw_status, str)
                or raw_status not in _LOCAL_OPERATIONAL_STATUSES
                or raw_status == OPERATIONAL_SUPPORTED
                or not isinstance(raw_item.get("limitations"), list)
                or not all(isinstance(item, str) for item in raw_item["limitations"])
            ):
                raise ProjectError("analysis run has malformed local NLP skip provenance")
            skipped[revision_id] = {
                "language": raw_item["language"],
                "capability": expected_capability,
                "status": raw_status,
                "limitations": list(raw_item["limitations"]),
            }
        if set(eligible) & set(skipped) or set(eligible) | set(skipped) != expected_scope:
            raise ProjectError("analysis run local NLP operation does not match its original scope")
        if (
            raw_processed != eligible
            or len(raw_skipped_ids) != len(set(raw_skipped_ids))
            or set(raw_skipped_ids) != set(skipped)
        ):
            raise ProjectError("analysis run local NLP revision accounting is inconsistent")
        expected_status = "COMPLETED" if not skipped else "PARTIAL" if eligible else "SKIPPED"
        if raw_operation.get("status") != expected_status:
            raise ProjectError("analysis run local NLP operation status is inconsistent")
        operations[preprocessor] = {
            "status": expected_status,
            "required_capability": expected_capability,
            "eligible_revision_ids": eligible,
            "processed_revision_ids": eligible,
            "skipped_revision_ids": list(raw_skipped_ids),
            "skipped_revisions": skipped,
        }
    if not operations or all(item["status"] == "COMPLETED" for item in operations.values()):
        overall_status = "COMPLETED" if operations else "NOT_CONFIGURED"
    elif all(item["status"] == "SKIPPED" for item in operations.values()):
        overall_status = "SKIPPED"
    else:
        overall_status = "PARTIAL"
    if raw_plan.get("status") != overall_status:
        raise ProjectError("analysis run local NLP execution status is inconsistent")
    return {
        "status": overall_status,
        "operations": operations,
        "warnings": list(raw_warnings),
    }


def _local_nlp_execution_summary(
    execution_plan: LocalNlpExecutionPlan | None,
) -> LocalNlpExecutionSummary:
    """Project the frozen plan into a safe, result-facing execution summary."""
    if execution_plan is None:
        return {"status": "NOT_CONFIGURED", "operations": {}}

    operations: dict[str, LocalNlpOperationExecutionSummary] = {}
    for operation_name, operation in execution_plan["operations"].items():
        capability = operation["required_capability"]
        skipped_revisions: dict[str, LocalNlpSkippedRevision] = {}
        for revision_id, skipped in operation["skipped_revisions"].items():
            safe_limitations: list[str] = []
            for limitation in skipped["limitations"]:
                if limitation in {
                    f"{capability}: {skipped['status']}",
                    f"{capability} assurance: UNASSESSED",
                }:
                    safe_limitations.append(limitation)
                elif limitation.startswith("local NLP route:"):
                    # Route diagnostics can contain environment names or local paths.
                    safe_limitations.append(
                        "local NLP route: configured route was unavailable or incomplete"
                    )
            if not safe_limitations:
                safe_limitations.append(f"{capability}: {skipped['status']}")
            skipped_revisions[revision_id] = {
                "language": skipped["language"],
                "capability": capability,
                "status": skipped["status"],
                "limitations": list(dict.fromkeys(safe_limitations)),
            }
        operations[operation_name] = {
            "status": operation["status"],
            "required_capability": capability,
            "processed_revision_ids": list(operation["processed_revision_ids"]),
            "skipped_revision_ids": list(operation["skipped_revision_ids"]),
            "skipped_revisions": skipped_revisions,
        }
    return {"status": execution_plan["status"], "operations": operations}


def _record_local_preprocessor_skip(
    project,
    run_id: str,
    scope_id: str,
    preprocessor: str,
    operation: LocalNlpOperationPlan,
    artifact_ids: list[str],
) -> str:
    skill_id, executor_id = {
        "social_actor_candidates": (
            "builtin.social-actor-candidates",
            "builtin.social_actor_candidates",
        ),
        "agency_candidates": ("builtin.agency-candidates", "builtin.agency_candidates"),
    }[preprocessor]
    skipped = operation["skipped_revisions"]
    configuration = {
        "preprocessor": preprocessor,
        "status": operation["status"],
        "processed_revision_ids": operation["processed_revision_ids"],
        "skipped_revision_ids": operation["skipped_revision_ids"],
        "eligible_artifact_ids": sorted(artifact_ids),
        "skipped_revisions": skipped,
    }
    config_hash = hashlib.sha256(
        json.dumps(configuration, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    row = project.db.execute(
        "SELECT id FROM skill_runs WHERE analysis_run_id=? AND skill_id=? "
        "AND skill_version='1.0.0' AND executor=? AND config_hash=? AND status='SKIPPED' LIMIT 1",
        (run_id, skill_id, executor_id, config_hash),
    ).fetchone()
    if row is not None:
        return str(row["id"])
    details = "; ".join(
        f"{revision_id} ({item['language']}, {item['capability']}={item['status']}; "
        f"{', '.join(item['limitations'])})"
        for revision_id, item in skipped.items()
    )
    processed_ids = ", ".join(operation["processed_revision_ids"]) or "none"
    skipped_ids = ", ".join(operation["skipped_revision_ids"]) or "none"
    reason = (
        f"LOCAL_NLP_{operation['status']}: processed_revision_ids=[{processed_ids}]; "
        f"skipped_revision_ids=[{skipped_ids}]; limitations={details}; "
        "no fallback was attempted."
    )
    skill_run_id = project.create_skill_run(
        run_id, scope_id, skill_id, "1.0.0", executor_id, config_hash
    )
    project.skip_skill_run(skill_run_id, reason)
    return skill_run_id


def _execute_controlled_cda_steps(
    project,
    run_id: str,
    scope_id: str,
    framework_presets: Mapping[str, str],
    framework_preset_snapshots: Mapping[str, dict[str, object]],
    frameworks: tuple[str, ...],
    languages: tuple[str, ...],
    counterevidence_queries: Mapping[str, str] | None,
    provider_factory: ProviderFactory | None,
    reranking_gateway: ModelGateway | None = None,
    reranking_provider_id: str | None = None,
    reranking_credential_alias: str | None = None,
    reranking_model: str | None = None,
    reranking_top_n: int | None = None,
    local_preprocessors: tuple[str, ...] = (),
    local_nlp_artifacts: Mapping[str, list[str]] | None = None,
    research_questions: Mapping[str, str] | None = None,
    cue_language_plan: CueLanguageExecutionPlan | None = None,
    question_candidate_limit: int = MAX_FRAMEWORK_HITS,
    local_nlp_execution_plan: LocalNlpExecutionPlan | None = None,
    dependency_parse_preflight: Mapping[str, DependencyParseSnapshot] | None = None,
) -> ControlledCdaResult:
    """Execute or reuse only completed, provenance-matching controlled CDA stages."""
    context = SkillContext(
        project,
        run_id,
        scope_id,
        dependency_parse_preflight=dependency_parse_preflight,
    )
    cue_plan = cue_language_plan or _cue_language_execution_plan(project, scope_id, languages)
    warnings = list(cue_plan["warnings"])
    if local_nlp_execution_plan is not None:
        local_nlp_execution_plan = _validated_local_nlp_execution_plan(
            local_nlp_execution_plan,
            local_preprocessors,
            project.scope_revisions(scope_id),
        )
        warnings.extend(local_nlp_execution_plan["warnings"])
    local_nlp_execution_summary = _local_nlp_execution_summary(local_nlp_execution_plan)
    descriptor_exists = project.db.execute(
        "SELECT 1 FROM skill_runs WHERE analysis_run_id=? "
        "AND skill_id='builtin.textual-descriptors' AND status IN ('COMPLETED','SKIPPED') LIMIT 1",
        (run_id,),
    ).fetchone()
    if not descriptor_exists:
        if cue_plan["skip_reason"]:
            _record_cue_skill_skip(
                project,
                run_id,
                scope_id,
                "builtin.textual-descriptors",
                "builtin.textual_descriptors",
                cue_plan,
            )
        else:
            run_textual_descriptors(
                context,
                tuple(cue_plan["languages"]),
                revision_ids=(
                    tuple(cue_plan["revision_ids"])
                    if cue_plan["revision_ids"] is not None
                    else None
                ),
                revision_languages=cue_plan["revision_languages"],
            )
    seed_finding_ids = _completed_seed_findings(project, run_id)
    if not _evidence_extraction_completed(project, run_id):
        if cue_plan["skip_reason"]:
            _record_cue_skill_skip(
                project,
                run_id,
                scope_id,
                "builtin.evidence-extraction",
                "builtin.evidence_extraction",
                cue_plan,
            )
        else:
            extraction = run_evidence_extraction(
                context,
                tuple(cue_plan["languages"]),
                revision_ids=(
                    tuple(cue_plan["revision_ids"])
                    if cue_plan["revision_ids"] is not None
                    else None
                ),
                revision_languages=cue_plan["revision_languages"],
            )
            seed_finding_ids = extraction["finding_ids"]

    local_nlp_skill_run_ids: list[str] = []
    local_nlp_finding_ids: list[str] = []
    if local_preprocessors:
        artifact_map = local_nlp_artifacts or {}
        scope_revisions = project.scope_revisions(scope_id)
        eligible_revisions = (
            {
                revision_id
                for operation in local_nlp_execution_plan["operations"].values()
                for revision_id in operation["eligible_revision_ids"]
            }
            if local_nlp_execution_plan is not None
            else set(scope_revisions)
        )
        if set(artifact_map) != eligible_revisions or any(
            not isinstance(artifact_map[revision_id], list) or len(artifact_map[revision_id]) != 1
            for revision_id in eligible_revisions
        ):
            raise ProjectError(
                "saved local NLP routes require exactly one parsed artifact per eligible revision"
            )
        artifact_ids = [
            artifact_id
            for revision_id in scope_revisions
            for artifact_id in artifact_map.get(revision_id, [])
        ]
        from .nlp_artifacts import _scoped_artifacts

        if artifact_ids:
            _scoped_artifacts(project, scope_id, artifact_ids, "parsed_tokens")
        for preprocessor in local_preprocessors:
            runner = {
                "social_actor_candidates": run_social_actor_candidates,
                "agency_candidates": run_agency_candidates,
            }.get(preprocessor)
            if runner is None:
                raise ProjectError(f"unknown saved local NLP preprocessor {preprocessor!r}")
            operation = (
                local_nlp_execution_plan["operations"][preprocessor]
                if local_nlp_execution_plan is not None
                else None
            )
            operation_revision_ids = (
                operation["eligible_revision_ids"] if operation is not None else scope_revisions
            )
            operation_artifact_ids = [
                artifact_id
                for revision_id in operation_revision_ids
                for artifact_id in artifact_map.get(revision_id, [])
            ]
            if operation_artifact_ids:
                prior = _completed_local_preprocessor(
                    project, run_id, preprocessor, operation_artifact_ids
                )
                if prior is None:
                    result = runner(context, operation_artifact_ids)
                    local_nlp_skill_run_ids.append(result["skill_run_id"])
                    local_nlp_finding_ids.extend(result["finding_ids"])
                else:
                    skill_run_id, finding_ids = prior
                    local_nlp_skill_run_ids.append(skill_run_id)
                    local_nlp_finding_ids.extend(finding_ids)
            if operation is not None and operation["skipped_revisions"]:
                skipped_skill_run_id = _record_local_preprocessor_skip(
                    project,
                    run_id,
                    scope_id,
                    preprocessor,
                    operation,
                    operation_artifact_ids,
                )
                local_nlp_skill_run_ids.append(skipped_skill_run_id)

    scope_question_finding_ids: dict[str, str] = {}
    scope_question_skill_run_ids: list[str] = []
    for framework, question in (research_questions or {}).items():
        stage = run_scope_question_candidates(
            context, framework, question, limit=question_candidate_limit
        )
        scope_question_skill_run_ids.append(stage["skill_run_id"])
        if stage["finding_id"] is not None:
            scope_question_finding_ids[framework] = stage["finding_id"]

    if not seed_finding_ids and not scope_question_finding_ids:
        _record_no_seed_skips(project, run_id, scope_id, frameworks)
        return {
            "analysis_run_id": run_id,
            "seed_finding_ids": [],
            "reviewed_finding_ids": [],
            "critical_review_finding_ids": [],
            "counterevidence_skill_run_ids": [],
            "counterevidence_statuses": {},
            "alternative_candidate_statuses": {},
            "synthesis_finding_id": None,
            "local_nlp_skill_run_ids": local_nlp_skill_run_ids,
            "local_nlp_finding_ids": local_nlp_finding_ids,
            "local_nlp_execution_summary": local_nlp_execution_summary,
            "scope_question_finding_ids": scope_question_finding_ids,
            "scope_question_skill_run_ids": scope_question_skill_run_ids,
            "warnings": [
                *warnings,
                "No descriptive cue or configured research-question retrieval produced candidate evidence; framework stages were skipped and no corpus-absence claim was made.",
            ],
        }

    counterevidence_skill_run_ids: list[str] = []
    counterevidence_statuses: dict[str, CounterEvidenceStatus] = {}
    queries = counterevidence_queries or {}
    for seed_id in seed_finding_ids:
        seed = project.get_finding(seed_id)
        query = queries.get(seed["phenomenon"])
        if query:
            existing = _completed_counterevidence_skill(project, run_id, seed_id, query)
            if existing:
                counterevidence_skill_run_ids.append(existing)
                skill_run = project.db.execute(
                    "SELECT coverage_id FROM skill_runs WHERE id=?", (existing,)
                ).fetchone()
                counterevidence_statuses[seed_id] = {
                    "status": "COMPLETED",
                    "query": query,
                    "skill_run_id": existing,
                    "coverage_id": skill_run["coverage_id"] if skill_run else None,
                    "retrieval_run_id": _retrieval_run_for_skill(project, existing),
                    "candidate_count": _retrieval_hit_count(
                        project, _retrieval_run_for_skill(project, existing)
                    ),
                    "reason": None,
                }
            else:
                limited = _counterevidence_limit_skip(project, run_id, scope_id, seed_id, query)
                if limited:
                    skill_run_id, reason = limited
                    counterevidence_skill_run_ids.append(skill_run_id)
                    counterevidence_statuses[seed_id] = {
                        "status": "LIMIT_REACHED",
                        "query": query,
                        "skill_run_id": skill_run_id,
                        "coverage_id": None,
                        "retrieval_run_id": None,
                        "candidate_count": None,
                        "reason": reason,
                    }
                    warnings.append(
                        f"Counterevidence retrieval for finding {seed_id} was not run: {reason}"
                    )
                else:
                    try:
                        counter = run_counterevidence_lexical(
                            SkillContext(project, run_id, scope_id), seed_id, query
                        )
                    except RetrievalLimitError as exc:
                        limited = _counterevidence_limit_skip(
                            project,
                            run_id,
                            scope_id,
                            seed_id,
                            query,
                            limit=exc.limit,
                        )
                        if limited is None:
                            raise ProjectError("failed to persist counterevidence limit status")
                        skill_run_id, reason = limited
                        counterevidence_skill_run_ids.append(skill_run_id)
                        counterevidence_statuses[seed_id] = {
                            "status": "LIMIT_REACHED",
                            "query": query,
                            "skill_run_id": skill_run_id,
                            "coverage_id": None,
                            "retrieval_run_id": None,
                            "candidate_count": None,
                            "reason": reason,
                        }
                        warnings.append(
                            f"Counterevidence retrieval for finding {seed_id} was not run: {reason}"
                        )
                    else:
                        counterevidence_skill_run_ids.append(counter["skill_run_id"])
                        counterevidence_statuses[seed_id] = {
                            "status": "COMPLETED",
                            "query": query,
                            "skill_run_id": counter["skill_run_id"],
                            "coverage_id": counter["coverage_id"],
                            "retrieval_run_id": counter["retrieval_run_id"],
                            "candidate_count": counter["candidate_count"],
                            "reason": None,
                        }
        else:
            skipped_run_id = _skipped_counterevidence_skill(project, run_id, seed_id)
            if not skipped_run_id:
                skip_reason = "NO_RESEARCHER_PROVIDED_CONTRAST_QUERY"
                configuration = json.dumps(
                    {
                        "finding_id": seed_id,
                        "retrieval": "exhaustive-literal-match-candidates-v1",
                        "skip_reason": skip_reason,
                    },
                    sort_keys=True,
                )
                skipped_run_id = project.create_skill_run(
                    run_id,
                    scope_id,
                    "builtin.counterevidence-lexical",
                    "2.0.0",
                    "builtin.counterevidence_lexical",
                    hashlib.sha256(configuration.encode()).hexdigest(),
                    input_finding_id=seed_id,
                )
                project.skip_skill_run(skipped_run_id, skip_reason)
            counterevidence_skill_run_ids.append(skipped_run_id)
            counterevidence_statuses[seed_id] = {
                "status": "NOT_RUN",
                "query": None,
                "skill_run_id": skipped_run_id,
                "coverage_id": None,
                "retrieval_run_id": None,
                "candidate_count": None,
                "reason": (
                    "No researcher-provided contrast query; no counterevidence or corpus-absence conclusion was made."
                ),
            }

    interpretation_frameworks = tuple(
        framework for framework in frameworks if framework != "critical-reviewer"
    )
    reviewed_finding_ids: list[str] = []
    for framework in interpretation_frameworks:
        framework_inputs = list(seed_finding_ids)
        question_finding_id = scope_question_finding_ids.get(framework)
        if question_finding_id is not None:
            framework_inputs.append(question_finding_id)
        if not framework_inputs:
            _record_framework_skip(project, run_id, scope_id, framework)
            warnings.append(
                f"Framework {framework!r} (skill 'builtin.{framework}') was skipped because no "
                "seed or research-question candidate evidence was available; the synthesis does "
                "not cover this framework."
            )
            continue
        for input_finding_id in framework_inputs:
            existing = _completed_review_finding(
                project, run_id, f"builtin.{framework}", input_finding_id
            )
            if existing:
                reviewed_finding_ids.append(existing)
            else:
                result = run_framework_review(
                    SkillContext(project, run_id, scope_id),
                    input_finding_id,
                    framework_presets[framework],
                    framework,
                    provider_factory=provider_factory,
                    preset_snapshot=framework_preset_snapshots[framework],
                    reranking_gateway=reranking_gateway,
                    reranking_provider_id=reranking_provider_id,
                    reranking_credential_alias=reranking_credential_alias,
                    reranking_model=reranking_model,
                    reranking_top_n=reranking_top_n,
                )
                reviewed_finding_ids.append(result.finding_id)

    critical_review_finding_ids: list[str] = []
    if "critical-reviewer" in frameworks:
        for review_id in reviewed_finding_ids:
            existing = _completed_review_finding(
                project, run_id, "builtin.critical-reviewer", review_id
            )
            if existing:
                critical_review_finding_ids.append(existing)
            else:
                result = run_framework_review(
                    SkillContext(project, run_id, scope_id),
                    review_id,
                    framework_presets["critical-reviewer"],
                    "critical-reviewer",
                    provider_factory=provider_factory,
                    preset_snapshot=framework_preset_snapshots["critical-reviewer"],
                    reranking_gateway=reranking_gateway,
                    reranking_provider_id=reranking_provider_id,
                    reranking_credential_alias=reranking_credential_alias,
                    reranking_model=reranking_model,
                    reranking_top_n=reranking_top_n,
                )
                critical_review_finding_ids.append(result.finding_id)
    synthesis_sources = critical_review_finding_ids or reviewed_finding_ids
    synthesis_finding_id = _completed_synthesis_finding(project, run_id)
    if synthesis_finding_id is None and synthesis_sources:
        synthesis = run_synthesis(
            context,
            synthesis_sources,
            framework_presets["synthesis"],
            framework_preset_snapshots["synthesis"],
            provider_factory=provider_factory,
        )
        synthesis_finding_id = synthesis["finding_id"]
    alternative_statuses = _alternative_candidate_statuses(
        project, run_id, scope_id, seed_finding_ids, frameworks
    )
    return {
        "analysis_run_id": run_id,
        "seed_finding_ids": seed_finding_ids,
        "reviewed_finding_ids": reviewed_finding_ids,
        "critical_review_finding_ids": critical_review_finding_ids,
        "counterevidence_skill_run_ids": counterevidence_skill_run_ids,
        "counterevidence_statuses": counterevidence_statuses,
        "alternative_candidate_statuses": alternative_statuses,
        "synthesis_finding_id": synthesis_finding_id,
        "local_nlp_skill_run_ids": local_nlp_skill_run_ids,
        "local_nlp_finding_ids": local_nlp_finding_ids,
        "local_nlp_execution_summary": local_nlp_execution_summary,
        "scope_question_finding_ids": scope_question_finding_ids,
        "scope_question_skill_run_ids": scope_question_skill_run_ids,
        "warnings": warnings,
    }


def _completed_local_preprocessor(
    project, run_id: str, preprocessor: str, artifact_ids: list[str]
) -> tuple[str, list[str]] | None:
    skill_id, executor_id = {
        "social_actor_candidates": (
            "builtin.social-actor-candidates",
            "builtin.social_actor_candidates",
        ),
        "agency_candidates": ("builtin.agency-candidates", "builtin.agency_candidates"),
    }[preprocessor]
    configuration = json.dumps({"artifact_ids": sorted(artifact_ids)}, sort_keys=True)
    config_hash = hashlib.sha256(configuration.encode()).hexdigest()
    row = project.db.execute(
        "SELECT id FROM skill_runs WHERE analysis_run_id=? AND skill_id=? AND skill_version='1.0.0' "
        "AND executor=? AND config_hash=? AND status='COMPLETED' LIMIT 1",
        (run_id, skill_id, executor_id, config_hash),
    ).fetchone()
    if row is None:
        return None
    skill_run_id = str(row["id"])
    findings = project.db.execute(
        "SELECT id FROM findings WHERE skill_run_id=? ORDER BY created_at,id", (skill_run_id,)
    ).fetchall()
    return skill_run_id, [str(item["id"]) for item in findings]


def _revision_language_assignments(project, revision_ids: list[str]) -> dict[str, str]:
    """Require researcher-declared document language for local NLP routing."""
    assignments: dict[str, str] = {}
    for revision_id in revision_ids:
        row = project.db.execute(
            "SELECT document_id FROM document_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        if row is None:
            raise ProjectError("local NLP scope contains an unknown document revision")
        language = project.document_metadata(str(row["document_id"])).get("language")
        if not isinstance(language, str) or not language.strip():
            raise ProjectError(
                "local NLP preprocessors require explicit document metadata language for every revision"
            )
        assignments[revision_id] = language.casefold().strip()
    return assignments


def _agency_dependency_parse_preflight(
    project,
    scope_id: str,
    artifact_ids: list[str],
    *,
    profile_provenance: Mapping[str, object] | None = None,
) -> dict[str, DependencyParseSnapshot]:
    """Pin dependency-parse eligibility to the exact parser artifact being consumed.

    Direct artifact execution re-probes the artifact's explicit backend/model route;
    it never substitutes the host's default model. Saved-profile execution instead
    reuses that run's explicit route preflight and verifies it against both the
    registered artifact and the persisted artifact provenance.
    """
    from .nlp_artifacts import _scoped_artifacts

    if not artifact_ids:
        return {}
    artifacts = _scoped_artifacts(project, scope_id, artifact_ids, "parsed_tokens")
    profile_languages: Mapping[str, object] | None = None
    profile_preflight: Mapping[str, object] | None = None
    profile_artifact_rows: list[object] = []
    if profile_provenance is not None:
        raw_local_nlp = profile_provenance.get("local_nlp")
        raw_preflight = profile_provenance.get("language_preflight")
        if isinstance(raw_local_nlp, Mapping):
            raw_languages = raw_local_nlp.get("document_languages")
            raw_artifacts = raw_local_nlp.get("artifacts")
            if isinstance(raw_languages, Mapping):
                profile_languages = raw_languages
            if isinstance(raw_artifacts, list):
                profile_artifact_rows = raw_artifacts
        if isinstance(raw_preflight, Mapping):
            profile_preflight = raw_preflight

    snapshots: dict[str, DependencyParseSnapshot] = {}
    for artifact in artifacts:
        revision_id = str(artifact["revision_id"])
        artifact_id = str(artifact["id"])
        backend = artifact["backend"] if isinstance(artifact["backend"], str) else None
        model = artifact["model"] if isinstance(artifact["model"], str) else None
        language: str | None = None
        limitations: list[str] = []
        status = OPERATIONAL_UNASSESSED

        if profile_provenance is None:
            revision = project.db.execute(
                "SELECT document_id FROM document_revisions WHERE id=?", (revision_id,)
            ).fetchone()
            if revision is not None:
                raw_language = project.document_metadata(str(revision["document_id"])).get(
                    "language"
                )
                if isinstance(raw_language, str) and raw_language.strip():
                    language = raw_language.casefold().strip()
            if language is None:
                limitations.append(
                    "dependency_parse: UNASSESSED (artifact language provenance is unavailable)"
                )
            elif backend is None or model is None:
                limitations.append(
                    "dependency_parse: UNASSESSED (artifact parser identity is incomplete)"
                )
            else:
                try:
                    local_result = local_execution_preflight(
                        (language,),
                        {"dependency_parse"},
                        routes={language: {"backend": backend, "model": model}},
                    ).get(language)
                except (ProjectError, ValueError, OSError):
                    local_result = None
                if isinstance(local_result, Mapping):
                    raw_statuses = local_result.get("statuses", {})
                    raw_status = (
                        raw_statuses.get("dependency_parse", OPERATIONAL_UNASSESSED)
                        if isinstance(raw_statuses, Mapping)
                        else OPERATIONAL_UNASSESSED
                    )
                    route_matches = (
                        local_result.get("backend") == backend
                        and local_result.get("model") == model
                        and local_result.get("route_source") == "profile"
                    )
                    if route_matches and raw_status in _DEPENDENCY_PARSE_STATUSES:
                        status = str(raw_status)
                    else:
                        limitations.append(
                            "dependency_parse: UNASSESSED (exact parser route preflight did not match the artifact)"
                        )
                    raw_limitations = local_result.get("limitations", [])
                    if isinstance(raw_limitations, list):
                        limitations.extend(
                            item for item in raw_limitations if isinstance(item, str)
                        )
                else:
                    limitations.append(
                        "dependency_parse: UNASSESSED (exact parser route preflight is unavailable)"
                    )
        else:
            raw_language = profile_languages.get(revision_id) if profile_languages else None
            if isinstance(raw_language, str) and raw_language.strip():
                language = raw_language.casefold().strip()
            preflight_for_language = (
                profile_preflight.get(language) if profile_preflight and language else None
            )
            matching_artifact = next(
                (
                    item
                    for item in profile_artifact_rows
                    if isinstance(item, Mapping)
                    and item.get("artifact_id") == artifact_id
                    and item.get("revision_id") == revision_id
                ),
                None,
            )
            artifact_provenance_matches = isinstance(matching_artifact, Mapping) and all(
                matching_artifact.get(key) == artifact[key]
                for key in (
                    "backend",
                    "backend_version",
                    "model",
                    "model_version",
                    "schema_version",
                )
            )
            if (
                isinstance(preflight_for_language, Mapping)
                and artifact_provenance_matches
                and preflight_for_language.get("route_source") == "profile"
                and preflight_for_language.get("backend") == backend
                and preflight_for_language.get("model") == model
            ):
                raw_statuses = preflight_for_language.get("statuses", {})
                raw_status = (
                    raw_statuses.get("dependency_parse", OPERATIONAL_UNASSESSED)
                    if isinstance(raw_statuses, Mapping)
                    else OPERATIONAL_UNASSESSED
                )
                status = (
                    str(raw_status)
                    if raw_status in _DEPENDENCY_PARSE_STATUSES
                    else OPERATIONAL_UNASSESSED
                )
                raw_limitations = preflight_for_language.get("limitations", [])
                if isinstance(raw_limitations, list):
                    limitations.extend(item for item in raw_limitations if isinstance(item, str))
            else:
                limitations.append(
                    "dependency_parse: UNASSESSED (saved route could not be tied to this artifact)"
                )

        if status != OPERATIONAL_SUPPORTED and not limitations:
            limitations.append(f"dependency_parse: {status}")
        snapshots[revision_id] = {
            "revision_id": revision_id,
            "artifact_id": artifact_id,
            "language": language,
            "status": status,
            "backend": backend,
            "model": model,
            "limitations": sorted(set(limitations)),
        }
    return snapshots


def _validated_dependency_parse_preflight(
    raw_snapshot: object,
) -> dict[str, DependencyParseSnapshot]:
    """Validate a persisted artifact-bound capability snapshot before resuming."""
    if not isinstance(raw_snapshot, dict):
        raise ProjectError("analysis run has malformed dependency_parse capability provenance")
    snapshot: dict[str, DependencyParseSnapshot] = {}
    for revision_id, raw_item in raw_snapshot.items():
        if not isinstance(revision_id, str) or not isinstance(raw_item, Mapping):
            raise ProjectError("analysis run has malformed dependency_parse capability provenance")
        artifact_id = raw_item.get("artifact_id")
        item_revision = raw_item.get("revision_id")
        language = raw_item.get("language")
        status = raw_item.get("status")
        backend = raw_item.get("backend")
        model = raw_item.get("model")
        limitations = raw_item.get("limitations")
        if (
            item_revision != revision_id
            or not isinstance(artifact_id, str)
            or not artifact_id
            or (language is not None and not isinstance(language, str))
            or status not in _DEPENDENCY_PARSE_STATUSES
            or (backend is not None and not isinstance(backend, str))
            or (model is not None and not isinstance(model, str))
            or not isinstance(limitations, list)
            or not all(isinstance(item, str) for item in limitations)
        ):
            raise ProjectError("analysis run has malformed dependency_parse capability provenance")
        snapshot[revision_id] = {
            "revision_id": revision_id,
            "artifact_id": artifact_id,
            "language": language,
            "status": status,
            "backend": backend,
            "model": model,
            "limitations": list(limitations),
        }
    return snapshot


def _build_profile_nlp_artifacts(project, revision_ids: list[str], profile):
    """Build exact-route local parses and return immutable artifact provenance."""
    if not profile.local_preprocessors:
        return {}, {"document_languages": {}, "artifacts": []}
    from .nlp import load_spacy, load_stanza

    language_assignments = _revision_language_assignments(project, revision_ids)
    routes = {language.casefold().strip(): route for language, route in profile.nlp_routes.items()}
    backend_cache = {}
    artifacts_by_revision: dict[str, list[str]] = {}
    snapshots: list[dict[str, object]] = []
    for revision_id in revision_ids:
        language = language_assignments[revision_id]
        route = routes.get(language)
        if route is None:
            raise ProjectError(
                f"local NLP profile has no explicit route for document language {language!r}"
            )
        if language not in {item.casefold().strip() for item in profile.languages}:
            raise ProjectError(
                f"document language {language!r} is not declared by the saved profile"
            )
        if language not in backend_cache:
            try:
                if route.backend == "spacy":
                    backend = load_spacy(language, route.model)
                else:
                    assert route.model_dir_env is not None
                    model_dir = os.environ.get(route.model_dir_env)
                    if not model_dir:
                        raise ProjectError(
                            f"selected Stanza route requires environment variable {route.model_dir_env!r}"
                        )
                    backend = load_stanza(language, route.model, model_dir)
            except (ProjectError, OSError, ValueError):
                raise ProjectError(
                    f"selected local NLP route for {language!r} could not be loaded; no fallback was attempted"
                ) from None
            if (
                backend.language.casefold() != language
                or backend.backend_id != route.backend
                or backend.model_id != route.model
            ):
                raise ProjectError("loaded local NLP route does not match the saved profile")
            backend_cache[language] = backend
        artifact = project.build_parsed_token_artifact(revision_id, backend_cache[language])
        artifacts_by_revision[revision_id] = [artifact.artifact_id]
        row = project.db.execute(
            "SELECT backend,backend_version,model,model_version,schema_version,content_hash,file_hash "
            "FROM artifact_registry WHERE id=?",
            (artifact.artifact_id,),
        ).fetchone()
        if row is None:
            raise ProjectError("parsed NLP artifact was not registered")
        snapshots.append(
            {
                "revision_id": revision_id,
                "language": language,
                "artifact_id": artifact.artifact_id,
                **dict(row),
            }
        )
    return artifacts_by_revision, {
        "document_languages": language_assignments,
        "artifacts": snapshots,
    }


def run_controlled_cda_review(
    project,
    revision_ids: list[str],
    preset_name: str | None,
    *,
    frameworks: tuple[str, ...] = DEFAULT_CDA_FRAMEWORKS,
    languages: tuple[str, ...] = ("en", "es"),
    counterevidence_queries: Mapping[str, str] | None = None,
    research_questions: Mapping[str, str] | None = None,
    routing_profile: str | None = None,
    profile_provenance: Mapping[str, object] | None = None,
    provider_factory: ProviderFactory | None = None,
    reranking_gateway: ModelGateway | None = None,
    reranking_provider_id: str | None = None,
    reranking_credential_alias: str | None = None,
    reranking_model: str | None = None,
    reranking_top_n: int | None = None,
    local_preprocessors: tuple[str, ...] = (),
    local_nlp_artifacts: Mapping[str, list[str]] | None = None,
    local_nlp_execution_plan: LocalNlpExecutionPlan | None = None,
    question_candidate_limit: int = MAX_FRAMEWORK_HITS,
    _analysis_run_id_capture: list[str] | None = None,
) -> ControlledCdaResult:
    """Run a controlled, evidence-first CDA review over an explicit document scope.

    The deterministic descriptive pass is distinct from interpretation. Each
    framework review can retrieve additional targeted evidence in the immutable
    original scope, and a final synthesis consumes only validated findings from
    this run. A caller may supply phenomenon-specific lexical counterevidence
    queries; none is fabricated when a defensible contrary query is unavailable.
    """
    if not revision_ids:
        raise ProjectError("controlled CDA review requires at least one revision")
    unknown_frameworks = sorted(set(frameworks) - set(FRAMEWORK_REVIEWS))
    if unknown_frameworks:
        raise ProjectError(f"unknown framework review(s): {', '.join(unknown_frameworks)}")
    if not frameworks:
        raise ProjectError("controlled CDA review requires at least one framework")
    if not 1 <= question_candidate_limit <= MAX_FRAMEWORK_HITS:
        raise ProjectError(f"question candidate limit must be between 1 and {MAX_FRAMEWORK_HITS}")
    if len(frameworks) != len(set(frameworks)):
        raise ProjectError("controlled CDA framework selection must not contain duplicates")
    framework_prompt_snapshots = _controlled_cda_prompt_snapshots(frameworks)
    synthesis_prompt_snapshot = _controlled_cda_synthesis_prompt_snapshot()
    interpretation_frameworks = tuple(
        framework for framework in frameworks if framework != "critical-reviewer"
    )
    if "critical-reviewer" in frameworks and not interpretation_frameworks:
        raise ProjectError("critical-reviewer requires at least one interpretive framework review")
    selected_questions = dict(research_questions or {})
    invalid_question_frameworks = sorted(set(selected_questions) - set(interpretation_frameworks))
    if invalid_question_frameworks:
        raise ProjectError(
            "research questions must target selected interpretive frameworks: "
            + ", ".join(invalid_question_frameworks)
        )
    if any(
        not isinstance(question, str) or not question.strip()
        for question in selected_questions.values()
    ):
        raise ProjectError("research questions must be non-empty strings")
    selected_questions = {key: value.strip() for key, value in selected_questions.items()}
    selected_counterevidence_queries = _normalize_counterevidence_queries(counterevidence_queries)
    reranking_configured = (
        any(
            value is not None
            for value in (
                reranking_gateway,
                reranking_provider_id,
                reranking_credential_alias,
                reranking_model,
            )
        )
        or reranking_top_n is not None
    )
    if reranking_configured and not all(
        value is not None
        for value in (
            reranking_gateway,
            reranking_provider_id,
            reranking_credential_alias,
            reranking_model,
        )
    ):
        raise ProjectError(
            "controlled CDA reranking requires explicit provider, credential alias, and model"
        )

    framework_presets = _resolve_framework_presets(
        project, frameworks, preset_name, routing_profile
    )
    framework_preset_snapshots = {
        framework: snapshot_model_preset(project, selected_preset)
        for framework, selected_preset in framework_presets.items()
    }
    reranking_credential_reference_hash = None
    if reranking_gateway is not None and reranking_provider_id and reranking_credential_alias:
        try:
            reranking_credential_reference_hash = credential_reference_fingerprint(
                project, reranking_provider_id, reranking_credential_alias
            )
        except DocumentError:
            if is_reviewed_reranking_provider(reranking_provider_id):
                raise
        gateway_fingerprint = reranking_gateway.credential_reference_fingerprints.get(
            reranking_provider_id
        )
        if (
            gateway_fingerprint is not None
            and gateway_fingerprint != reranking_credential_reference_hash
        ):
            raise ModelProviderError(
                "reranking gateway credential does not match the selected project alias"
            )
    if local_nlp_execution_plan is not None:
        local_nlp_execution_plan = _validated_local_nlp_execution_plan(
            local_nlp_execution_plan, local_preprocessors, revision_ids
        )
    scope_id = project.snapshot(revision_ids)
    dependency_parse_preflight = (
        _agency_dependency_parse_preflight(
            project,
            scope_id,
            [
                artifact_id
                for revision_artifacts in (local_nlp_artifacts or {}).values()
                for artifact_id in revision_artifacts
            ],
            profile_provenance=profile_provenance,
        )
        if "agency_candidates" in local_preprocessors
        else None
    )
    cue_language_plan = _cue_language_execution_plan(project, scope_id, languages)
    configuration: dict[str, object] = {
        "profile": "general-cda-controlled-v3",
        "execution_order": [
            "textual-descriptors",
            "evidence-extraction",
            *(["local-nlp-preprocessors"] if local_preprocessors else []),
            *(["scope-question-candidates"] if selected_questions else []),
            "counterevidence",
            "framework-reviews",
            "critical-reviewer",
            "synthesis",
        ],
        "frameworks": list(frameworks),
        "framework_prompt_snapshot_version": CONTROLLED_CDA_PROMPT_SNAPSHOT_VERSION,
        "framework_prompt_snapshots": framework_prompt_snapshots,
        "synthesis_prompt_snapshot_version": CONTROLLED_CDA_PROMPT_SNAPSHOT_VERSION,
        "synthesis_prompt_snapshot": synthesis_prompt_snapshot,
        "languages": list(languages),
        "cue_language_plan": cue_language_plan,
        "preset_name": preset_name,
        "routing_profile": routing_profile,
        "framework_presets": framework_presets,
        "preset_snapshot_version": 2,
        "framework_preset_snapshots": framework_preset_snapshots,
        "counterevidence_queries": selected_counterevidence_queries,
        "research_questions": selected_questions,
        "question_candidate_limit": question_candidate_limit,
        "local_preprocessors": list(local_preprocessors),
        "local_nlp_artifacts": {
            key: list(value) for key, value in (local_nlp_artifacts or {}).items()
        },
        "reranking": (
            {
                "provider_id": reranking_provider_id,
                "credential_alias": reranking_credential_alias,
                "credential_reference_hash": reranking_credential_reference_hash,
                "model": reranking_model,
                "top_n": reranking_top_n,
            }
            if reranking_gateway is not None
            else None
        ),
    }
    if local_nlp_execution_plan is not None:
        configuration["local_nlp_execution_plan"] = local_nlp_execution_plan
    if dependency_parse_preflight is not None:
        configuration["agency_dependency_parse_preflight"] = dependency_parse_preflight
    if profile_provenance:
        configuration["declared_profile"] = dict(profile_provenance)
    run_id = project.create_run(scope_id, configuration)
    if _analysis_run_id_capture is not None:
        _analysis_run_id_capture.append(run_id)
    try:
        project.start_run(run_id)
        result = _execute_controlled_cda_steps(
            project,
            run_id,
            scope_id,
            framework_presets,
            framework_preset_snapshots,
            frameworks,
            languages,
            selected_counterevidence_queries,
            provider_factory,
            reranking_gateway,
            reranking_provider_id,
            reranking_credential_alias,
            reranking_model,
            reranking_top_n,
            local_preprocessors,
            local_nlp_artifacts,
            selected_questions,
            cue_language_plan,
            question_candidate_limit=question_candidate_limit,
            local_nlp_execution_plan=local_nlp_execution_plan,
            dependency_parse_preflight=dependency_parse_preflight,
        )
        project.finalize_run(run_id)
        return result
    except BaseException as exc:
        project.fail_run(run_id, exc)
        raise


def resume_controlled_cda_review(
    project, analysis_run_id: str, *, provider_factory: ProviderFactory | None = None
) -> ControlledCdaResult:
    """Resume a failed controlled CDA run without replaying completed skill runs."""
    run = project.db.execute(
        "SELECT scope_id,status,config_json FROM analysis_runs WHERE id=?", (analysis_run_id,)
    ).fetchone()
    if not run:
        raise ProjectError("unknown analysis run")
    if run["status"] != "FAILED":
        raise ProjectError("only a failed controlled CDA run can be resumed")
    cue_language_plan: CueLanguageExecutionPlan
    local_nlp_execution_plan: LocalNlpExecutionPlan | None = None
    dependency_parse_preflight: dict[str, DependencyParseSnapshot] | None = None
    try:
        run_summary = project.analysis_run_summary(analysis_run_id)
        config = run_summary["analysis_run"]["config"]
        if not isinstance(config, dict):
            raise TypeError
        if config.get("profile") != "general-cda-controlled-v3":
            raise ValueError
        raw_preset_name = config["preset_name"]
        if raw_preset_name is not None and not isinstance(raw_preset_name, str):
            raise ValueError
        preset_name = raw_preset_name
        frameworks = tuple(str(item) for item in config["frameworks"])
        raw_prompt_snapshot_version = config.get("framework_prompt_snapshot_version")
        raw_prompt_snapshots = config.get("framework_prompt_snapshots")
        if (
            type(raw_prompt_snapshot_version) is not int
            or raw_prompt_snapshot_version != CONTROLLED_CDA_PROMPT_SNAPSHOT_VERSION
            or not isinstance(raw_prompt_snapshots, list)
            or not all(isinstance(item, dict) for item in raw_prompt_snapshots)
        ):
            raise ProjectError(
                "analysis run lacks immutable framework prompt snapshots and cannot be safely resumed"
            )
        current_prompt_snapshots = _controlled_cda_prompt_snapshots(frameworks)
        if raw_prompt_snapshots != current_prompt_snapshots:
            raise ProjectError(
                "analysis run framework prompt snapshots differ from the current built-in "
                "instructions and cannot be safely resumed"
            )
        raw_synthesis_snapshot_version = config.get("synthesis_prompt_snapshot_version")
        raw_synthesis_snapshot = config.get("synthesis_prompt_snapshot")
        if (
            type(raw_synthesis_snapshot_version) is not int
            or raw_synthesis_snapshot_version != CONTROLLED_CDA_PROMPT_SNAPSHOT_VERSION
            or not isinstance(raw_synthesis_snapshot, dict)
        ):
            raise ProjectError(
                "analysis run lacks an immutable synthesis prompt snapshot and cannot be safely resumed"
            )
        if raw_synthesis_snapshot != _controlled_cda_synthesis_prompt_snapshot():
            raise ProjectError(
                "analysis run synthesis prompt snapshot differs from the current built-in "
                "instructions and cannot be safely resumed"
            )
        languages = tuple(str(item) for item in config["languages"])
        raw_cue_plan = config.get("cue_language_plan")
        if raw_cue_plan is None:
            cue_language_plan = _cue_language_execution_plan(project, run["scope_id"], languages)
            cue_language_plan["warnings"].append(
                "This historical run predates a frozen cue-language execution plan; its plan was reconstructed from current document metadata."
            )
        elif isinstance(raw_cue_plan, dict):
            raw_plan_languages = raw_cue_plan.get("languages")
            raw_revision_ids = raw_cue_plan.get("revision_ids")
            raw_revision_languages = raw_cue_plan.get("revision_languages")
            raw_warnings = raw_cue_plan.get("warnings")
            raw_skip_reason = raw_cue_plan.get("skip_reason")
            if (
                not isinstance(raw_plan_languages, list)
                or not all(isinstance(item, str) for item in raw_plan_languages)
                or (
                    raw_revision_ids is not None
                    and (
                        not isinstance(raw_revision_ids, list)
                        or not all(isinstance(item, str) for item in raw_revision_ids)
                    )
                )
                or (
                    raw_revision_languages is not None
                    and (
                        not isinstance(raw_revision_languages, dict)
                        or not all(
                            isinstance(key, str) and isinstance(value, str)
                            for key, value in raw_revision_languages.items()
                        )
                    )
                )
                or not isinstance(raw_warnings, list)
                or not all(isinstance(item, str) for item in raw_warnings)
                or (raw_skip_reason is not None and not isinstance(raw_skip_reason, str))
            ):
                raise ValueError
            cue_language_plan = CueLanguageExecutionPlan(
                languages=list(raw_plan_languages),
                revision_ids=list(raw_revision_ids) if raw_revision_ids is not None else None,
                revision_languages=(
                    dict(raw_revision_languages) if raw_revision_languages is not None else None
                ),
                warnings=list(raw_warnings),
                skip_reason=raw_skip_reason,
            )
        else:
            raise ValueError
        queries = _normalize_counterevidence_queries(config.get("counterevidence_queries", {}))
        raw_questions = config.get("research_questions", {})
        if not isinstance(raw_questions, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in raw_questions.items()
        ):
            raise ValueError
        research_questions = {str(key): str(value) for key, value in raw_questions.items()}
        raw_question_candidate_limit = config.get("question_candidate_limit", MAX_FRAMEWORK_HITS)
        if (
            type(raw_question_candidate_limit) is not int
            or not 1 <= raw_question_candidate_limit <= MAX_FRAMEWORK_HITS
        ):
            raise ValueError
        question_candidate_limit = raw_question_candidate_limit
        local_preprocessors = tuple(str(item) for item in config.get("local_preprocessors", []))
        raw_artifacts = config.get("local_nlp_artifacts", {})
        if not isinstance(raw_artifacts, dict) or not all(
            isinstance(key, str)
            and isinstance(value, list)
            and all(isinstance(item, str) for item in value)
            for key, value in raw_artifacts.items()
        ):
            raise ValueError
        local_nlp_artifacts = {
            str(key): [str(item) for item in value] for key, value in raw_artifacts.items()
        }
        if "local_nlp_execution_plan" in config:
            local_nlp_execution_plan = _validated_local_nlp_execution_plan(
                config["local_nlp_execution_plan"],
                local_preprocessors,
                project.scope_revisions(run["scope_id"]),
            )
        if "agency_dependency_parse_preflight" in config:
            dependency_parse_preflight = _validated_dependency_parse_preflight(
                config["agency_dependency_parse_preflight"]
            )
        framework_presets = {
            str(framework): str(preset)
            for framework, preset in config.get("framework_presets", {}).items()
        }
        framework_preset_snapshots = {
            str(framework): snapshot
            for framework, snapshot in config.get("framework_preset_snapshots", {}).items()
            if isinstance(snapshot, dict)
        }
        reranking_config = config.get("reranking")
        if reranking_config is None:
            reranking_provider_id = reranking_credential_alias = reranking_model = None
            reranking_top_n = None
            reranking_credential_reference_hash = None
        elif isinstance(reranking_config, dict):
            reranking_provider_id = str(reranking_config["provider_id"])
            reranking_credential_alias = str(reranking_config["credential_alias"])
            reranking_model = str(reranking_config["model"])
            raw_top_n = reranking_config.get("top_n")
            reranking_top_n = int(raw_top_n) if raw_top_n is not None else None
            raw_credential_hash = reranking_config.get("credential_reference_hash")
            if raw_credential_hash is not None and not isinstance(raw_credential_hash, str):
                raise ValueError
            reranking_credential_reference_hash = raw_credential_hash
            if (
                is_reviewed_reranking_provider(reranking_provider_id)
                and reranking_credential_reference_hash is None
            ):
                raise ValueError
        else:
            raise ValueError
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProjectError("analysis run lacks resumable controlled CDA configuration") from exc
    unknown = sorted(set(frameworks) - set(FRAMEWORK_REVIEWS))
    if unknown:
        raise ProjectError(
            f"analysis run references unknown framework review(s): {', '.join(unknown)}"
        )
    if not framework_presets:
        if preset_name is None:
            raise ProjectError("analysis run lacks a default preset for its unrouted stages")
        framework_presets = {
            **{framework: preset_name for framework in frameworks},
            "synthesis": preset_name,
        }
    expected_stages = set(frameworks) | {"synthesis"}
    if set(framework_presets) != expected_stages:
        raise ProjectError("analysis run lacks a preset for one or more framework/synthesis stages")
    snapshots_are_credential_pinned = all(
        isinstance(snapshot.get("credential_reference_hash"), str)
        and len(str(snapshot["credential_reference_hash"])) == 64
        for snapshot in framework_preset_snapshots.values()
    )
    if (
        config.get("preset_snapshot_version") != 2
        or set(framework_preset_snapshots) != set(expected_stages)
        or not snapshots_are_credential_pinned
    ):
        raise ProjectError(
            "analysis run lacks immutable model preset snapshots and cannot be safely resumed"
        )
    reranking_gateway = (
        build_reranking_gateway(
            project,
            reranking_provider_id,
            reranking_credential_alias,
            expected_credential_reference_hash=reranking_credential_reference_hash,
        )
        if reranking_provider_id and reranking_credential_alias
        else None
    )
    try:
        project.resume_run(analysis_run_id)
        result = _execute_controlled_cda_steps(
            project,
            analysis_run_id,
            run["scope_id"],
            framework_presets,
            framework_preset_snapshots,
            frameworks,
            languages,
            queries,
            provider_factory,
            reranking_gateway,
            reranking_provider_id,
            reranking_credential_alias,
            reranking_model,
            reranking_top_n,
            local_preprocessors,
            local_nlp_artifacts,
            research_questions,
            cue_language_plan,
            question_candidate_limit=question_candidate_limit,
            local_nlp_execution_plan=local_nlp_execution_plan,
            dependency_parse_preflight=dependency_parse_preflight,
        )
        project.finalize_run(analysis_run_id)
        return result
    except BaseException as exc:
        project.fail_run(analysis_run_id, exc)
        raise


def run_exact_phrase_analysis(project, revision_ids: list[str], phrase: str) -> AnalysisResult:
    scope = project.snapshot(revision_ids)
    analysis_run_id = project.create_run(
        scope, {"profile": "fixture.exact-phrase", "phrase": phrase}
    )
    try:
        project.start_run(analysis_run_id)
        result = run_exact_phrase_fixture(project, analysis_run_id, scope, phrase)
        evidence_ids = result["evidence_ids"]
        finding_ids = []
        for evidence_id in evidence_ids:
            finding = project.create_finding(
                analysis_run_id,
                "fixture.exact-phrase",
                "1.0.0",
                "exact_phrase",
                f"The phrase {phrase!r} occurs in the scoped source.",
            )
            project.attach_evidence(finding, evidence_id, EvidenceRole.EXAMPLE)
            finding_ids.append(finding)
        project.finalize_run(analysis_run_id)
        return {
            "analysis_run_id": analysis_run_id,
            "skill_run_id": result["skill_run_id"],
            "finding_ids": finding_ids,
        }
    except BaseException as exc:
        project.fail_run(analysis_run_id, exc)
        raise


def run_textual_descriptors_analysis(
    project, revision_ids: list[str], languages: tuple[str, ...] = ("en", "es")
) -> TextualDescriptorAnalysisResult:
    """Run the built-in deterministic descriptive textual-analysis skill."""
    languages = _ordered_unique_languages(languages)
    scope = project.snapshot(revision_ids)
    analysis_run_id = project.create_run(
        scope,
        {
            "profile": "builtin.textual-descriptors",
            "languages": languages,
            "execution_kind": "DETERMINISTIC",
        },
    )
    try:
        project.start_run(analysis_run_id)
        result = run_textual_descriptors(SkillContext(project, analysis_run_id, scope), languages)
        project.finalize_run(analysis_run_id)
        return {
            "analysis_run_id": analysis_run_id,
            "skill_run_id": result["skill_run_id"],
            "finding_ids": result["finding_ids"],
            "coverage_id": result["coverage_id"],
        }
    except BaseException as exc:
        project.fail_run(analysis_run_id, exc)
        raise


def run_evidence_extraction_analysis(
    project,
    revision_ids: list[str],
    languages: tuple[str, ...] = ("en", "es"),
    *,
    max_examples: int = 48,
) -> EvidenceExtractionAnalysisResult:
    """Run the deterministic candidate inventory as an independent saved analysis."""
    if not revision_ids:
        raise ProjectError("evidence extraction requires at least one document revision")
    languages = _ordered_unique_languages(languages)
    scope = project.snapshot(revision_ids)
    analysis_run_id = project.create_run(
        scope,
        {
            "profile": "builtin.evidence-extraction",
            "version": "1.0.0",
            "languages": list(languages),
            "max_examples": max_examples,
            "execution_kind": "DETERMINISTIC",
        },
    )
    try:
        project.start_run(analysis_run_id)
        result = run_evidence_extraction(
            SkillContext(project, analysis_run_id, scope),
            languages,
            max_examples=max_examples,
        )
        project.finalize_run(analysis_run_id)
        return {"analysis_run_id": analysis_run_id, **result}
    except BaseException as exc:
        project.fail_run(analysis_run_id, exc)
        raise


def run_quote_attribution_candidate_analysis(
    project,
    revision_ids: list[str],
    *,
    revision_languages: Mapping[str, str] | None = None,
) -> QuoteAttributionAnalysisResult:
    """Run the deterministic quotation inventory in one explicit revision scope.

    This public application operation deliberately exposes only the reviewed
    quote-candidate executor. It cannot load project-provided code or choose a
    different executor by name.
    """
    if not revision_ids:
        raise ProjectError("quote attribution requires at least one document revision")
    _validate_batch_revision_ids(revision_ids)
    scope = project.snapshot(revision_ids)
    scoped_revisions = project.scope_revisions(scope)
    if revision_languages is not None and set(revision_languages) != set(scoped_revisions):
        raise ProjectError("language assignments must cover exactly the selected revisions")
    if revision_languages is not None and not set(revision_languages.values()).issubset(
        {"en", "es"}
    ):
        raise ProjectError("quote-attribution language assignments support only en and es")

    language_snapshot = (
        dict(sorted(revision_languages.items())) if revision_languages is not None else None
    )
    analysis_run_id = project.create_run(
        scope,
        {
            "profile": "builtin.quote-attribution-candidates",
            "version": "1.0.0",
            "execution_kind": "DETERMINISTIC",
            "revision_languages": language_snapshot,
        },
    )
    try:
        project.start_run(analysis_run_id)
        result = run_quote_attribution_candidates(
            SkillContext(project, analysis_run_id, scope),
            revision_languages=language_snapshot,
        )
        project.finalize_run(analysis_run_id)
        return {
            "analysis_run_id": analysis_run_id,
            "scope_snapshot_id": scope,
            **result,
        }
    except BaseException as exc:
        project.fail_run(analysis_run_id, exc)
        raise


def run_scope_question_candidates_analysis(
    project,
    revision_ids: list[str],
    framework_id: str,
    question: str,
    *,
    limit: int = 12,
    retrieval_mode: Literal["lexical", "hybrid"] = "lexical",
    embedding_preset_name: str | None = None,
    embedding_artifact_ids: list[str] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> ScopeQuestionAnalysisResult:
    """Run bounded, researcher-directed lexical retrieval in an explicit scope.

    The immutable revision scope and all request parameters are validated before
    creating an analysis run. Results are targeted examples, and an empty result
    is not evidence that a question's subject is absent from the scope.
    """
    if not isinstance(revision_ids, list) or not revision_ids:
        raise ProjectError("scope question requires a non-empty revision scope")
    if any(
        not isinstance(revision_id, str) or not revision_id.strip() for revision_id in revision_ids
    ):
        raise ProjectError("scope question revision IDs must be non-empty strings")
    if len(revision_ids) != len(set(revision_ids)):
        raise ProjectError("scope question revision scope must not contain duplicates")
    if (
        not isinstance(framework_id, str)
        or framework_id not in FRAMEWORK_REVIEWS
        or framework_id == "critical-reviewer"
    ):
        raise ProjectError("scope question must target a registered interpretive framework")
    if not isinstance(question, str) or not question.strip():
        raise ProjectError("scope research question must not be empty")
    if type(limit) is not int or not 1 <= limit <= MAX_FRAMEWORK_HITS:
        raise ProjectError(f"scope question limit must be between 1 and {MAX_FRAMEWORK_HITS}")
    if retrieval_mode not in {"lexical", "hybrid"}:
        raise ProjectError("scope question retrieval mode must be lexical or hybrid")
    if retrieval_mode == "hybrid":
        if not isinstance(embedding_preset_name, str) or not embedding_preset_name.strip():
            raise ProjectError("hybrid scope question requires an embedding preset")
        if (
            type(embedding_artifact_ids) is not list
            or not embedding_artifact_ids
            or len(embedding_artifact_ids) > 100
            or any(type(item) is not str or not item.strip() for item in embedding_artifact_ids)
            or len(embedding_artifact_ids) != len(set(embedding_artifact_ids))
        ):
            raise ProjectError(
                "hybrid scope question requires unique explicit embedding artifact IDs"
            )
        if len({item.strip() for item in embedding_artifact_ids}) != len(embedding_artifact_ids):
            raise ProjectError("hybrid embedding artifact IDs must be unique")
    elif (
        embedding_preset_name is not None
        or embedding_artifact_ids is not None
        or embedding_provider is not None
    ):
        raise ProjectError("embedding options are only valid with hybrid retrieval")

    normalized_question = question.strip()
    scope = project.snapshot(revision_ids)
    embedding_configuration_snapshot: dict[str, object] | None = None
    if retrieval_mode == "hybrid":
        from .vectors import hybrid_retrieval_configuration_snapshot

        embedding_configuration_snapshot = hybrid_retrieval_configuration_snapshot(
            project,
            str(embedding_preset_name).strip(),
            scope,
            cast(list[str], embedding_artifact_ids),
            require_complete_scope=True,
        )
    analysis_run_id = project.create_run(
        scope,
        {
            "profile": (
                "builtin.scope-question-hybrid-candidates"
                if retrieval_mode == "hybrid"
                else "builtin.scope-question-candidates"
            ),
            "version": "1.0.0",
            "execution_kind": "HYBRID" if retrieval_mode == "hybrid" else "DETERMINISTIC",
            "framework_id": framework_id,
            "question": normalized_question,
            "limit": limit,
            "retrieval_mode": retrieval_mode,
            "embedding_preset_name": embedding_preset_name,
            "embedding_artifact_ids": embedding_artifact_ids,
            "embedding_configuration_snapshot": embedding_configuration_snapshot,
        },
    )
    try:
        project.start_run(analysis_run_id)
        result = run_scope_question_candidates(
            SkillContext(project, analysis_run_id, scope),
            framework_id,
            normalized_question,
            limit=limit,
            retrieval_mode=retrieval_mode,
            embedding_preset_name=embedding_preset_name,
            embedding_artifact_ids=embedding_artifact_ids,
            embedding_provider=embedding_provider,
        )
        evidence_rows = project.db.execute(
            "SELECT (SELECT e.id FROM evidence_artifacts e "
            "WHERE e.kind='TEXTUAL' AND e.span_id=h.source_span_id "
            "ORDER BY e.created_at,e.id LIMIT 1) AS evidence_id "
            "FROM retrieval_hits h WHERE h.retrieval_run_id=? ORDER BY h.rank,h.id",
            (result["retrieval_run_id"],),
        ).fetchall()
        evidence_ids = [str(row["evidence_id"]) for row in evidence_rows if row["evidence_id"]]
        if len(evidence_ids) != result["candidate_count"]:
            raise ProjectError("scope question retrieval is missing candidate evidence lineage")
        coverage = project.coverage_descriptor(result["coverage_id"])
        project.finalize_run(analysis_run_id)
        return {
            "analysis_run_id": analysis_run_id,
            "scope_snapshot_id": scope,
            **result,
            "evidence_ids": evidence_ids,
            "coverage": coverage,
            "retrieval_mode": retrieval_mode,
        }
    except BaseException as exc:
        project.fail_run(analysis_run_id, exc)
        raise


def ask_question_analysis(
    project,
    revision_ids: list[str],
    question: str,
    framework_id: str,
    preset_name: str,
    *,
    limit: int = 8,
    retrieval_mode: Literal["lexical", "hybrid"] = "lexical",
    embedding_preset_name: str | None = None,
    embedding_artifact_ids: list[str] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    conversation_context: list[dict[str, str]] | None = None,
) -> AskQuestionResult:
    """Answer one research question from targeted, explicitly scoped evidence.

    Retrieval and interpretation remain separate persisted runs. A zero-hit
    targeted search returns no answer and is explicitly not an absence result.
    """
    # Validate every caller-controlled argument before snapshot/run creation.
    if not isinstance(revision_ids, list) or not revision_ids:
        raise ProjectError("ask question requires a non-empty revision scope")
    if any(not isinstance(item, str) or not item.strip() for item in revision_ids):
        raise ProjectError("ask question revision IDs must be non-empty strings")
    if len(revision_ids) != len(set(revision_ids)):
        raise ProjectError("ask question revision scope must not contain duplicates")
    if not isinstance(question, str) or not question.strip():
        raise ProjectError("ask question text must not be empty")
    if conversation_context is not None:
        if type(conversation_context) is not list or len(conversation_context) > 12:
            raise ProjectError("conversation context must contain at most 12 messages")
        context_chars = 0
        for message in conversation_context:
            if (
                type(message) is not dict
                or set(message) != {"role", "content"}
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                or len(message["content"]) > 4_000
            ):
                raise ProjectError("conversation context message is invalid")
            context_chars += len(message["content"])
        if context_chars > 16_000:
            raise ProjectError("conversation context exceeds 16000 characters")
    if (
        not isinstance(framework_id, str)
        or framework_id not in FRAMEWORK_REVIEWS
        or framework_id == "critical-reviewer"
    ):
        raise ProjectError("ask question requires a registered interpretive framework")
    if not isinstance(preset_name, str) or not preset_name.strip():
        raise ProjectError("ask question requires a saved model preset")
    if type(limit) is not int or not 1 <= limit <= MAX_FRAMEWORK_HITS:
        raise ProjectError(f"ask question limit must be between 1 and {MAX_FRAMEWORK_HITS}")
    if retrieval_mode not in {"lexical", "hybrid"}:
        raise ProjectError("ask question retrieval mode must be lexical or hybrid")
    if retrieval_mode == "hybrid":
        if not isinstance(embedding_preset_name, str) or not embedding_preset_name.strip():
            raise ProjectError("hybrid ask requires an embedding preset")
        if (
            type(embedding_artifact_ids) is not list
            or not embedding_artifact_ids
            or len(embedding_artifact_ids) > 100
            or any(type(item) is not str or not item.strip() for item in embedding_artifact_ids)
            or len(embedding_artifact_ids) != len(set(embedding_artifact_ids))
        ):
            raise ProjectError("hybrid ask requires unique explicit embedding artifact IDs")
        if len({item.strip() for item in embedding_artifact_ids}) != len(embedding_artifact_ids):
            raise ProjectError("hybrid embedding artifact IDs must be unique")
    elif (
        embedding_preset_name is not None
        or embedding_artifact_ids is not None
        or embedding_provider is not None
    ):
        raise ProjectError("embedding options are only valid with hybrid retrieval")

    revision_ids = [item.strip() for item in revision_ids]
    if len(revision_ids) != len(set(revision_ids)):
        raise ProjectError("ask question revision scope must not contain duplicates")
    for revision_id in revision_ids:
        if not project.db.execute(
            "SELECT 1 FROM document_revisions WHERE id=?", (revision_id,)
        ).fetchone():
            raise ProjectError("ask question scope references an unknown document revision")
    normalized_question = question.strip()
    # This is read-only validation, deliberately before the candidate run.
    project.model_preset(preset_name.strip())

    candidates = run_scope_question_candidates_analysis(
        project,
        revision_ids,
        framework_id,
        normalized_question,
        limit=limit,
        retrieval_mode=retrieval_mode,
        embedding_preset_name=embedding_preset_name,
        embedding_artifact_ids=embedding_artifact_ids,
        embedding_provider=embedding_provider,
    )
    coverage = candidates["coverage"].model_dump(mode="json")
    if candidates["candidate_count"] == 0:
        if candidates["finding_id"] is not None or candidates["evidence_ids"]:
            raise EvidenceValidationError(
                "empty ask-question retrieval unexpectedly contains candidate evidence"
            )
        return {
            "question": normalized_question,
            "framework_id": framework_id,
            "retrieval_mode": retrieval_mode,
            "scope_snapshot_id": candidates["scope_snapshot_id"],
            "candidate_analysis_run_id": candidates["analysis_run_id"],
            "candidate_skill_run_id": candidates["skill_run_id"],
            "candidate_finding_id": candidates["finding_id"],
            "candidate_retrieval_run_id": candidates["retrieval_run_id"],
            "coverage_id": candidates["coverage_id"],
            "coverage": coverage,
            "answer_coverage_id": None,
            "answer_coverage": None,
            "status": "no_candidates",
            "answer_analysis_run_id": None,
            "answer_finding_id": None,
            "model_execution_id": None,
            "answer_interpretation": None,
            "alternative_interpretation": None,
            "limitations": [
                f"No examples were found under this targeted {retrieval_mode} retrieval plan; this does not establish that relevant examples are absent from the scope."
            ],
            "evidence": [],
            "evidence_total": 0,
            "evidence_truncated": False,
        }

    candidate_finding_id = candidates["finding_id"]
    if candidate_finding_id is None:
        raise EvidenceValidationError("ask-question candidates are missing their Finding")
    candidate_validation = project.validate_finding(candidate_finding_id)
    if not candidate_validation.valid:
        raise EvidenceValidationError(
            "ask-question candidate Finding is invalid: " + "; ".join(candidate_validation.errors)
        )

    if conversation_context is None:
        answer_run = project.review_finding_with_model(
            candidate_finding_id,
            preset_name.strip(),
            framework=framework_id,
        )
    else:
        answer_run = project.review_finding_with_model(
            candidate_finding_id,
            preset_name.strip(),
            framework=framework_id,
            conversation_context=conversation_context,
        )
    answer_validation = project.validate_finding(answer_run.finding_id)
    if not answer_validation.valid:
        raise EvidenceValidationError(
            "ask-question answer Finding is invalid: " + "; ".join(answer_validation.errors)
        )
    answer_finding = project.get_finding(answer_run.finding_id)
    answer_coverage_id = answer_finding["coverage_id"]
    if not isinstance(answer_coverage_id, str) or not answer_coverage_id:
        raise EvidenceValidationError("ask-question answer Finding is missing coverage provenance")
    answer_coverage = project.coverage_descriptor(answer_coverage_id).model_dump(mode="json")
    if answer_finding["analysis_run_id"] != answer_run.analysis_run_id:
        raise EvidenceValidationError("ask-question answer Finding has mismatched run provenance")
    run_row = project.db.execute(
        "SELECT scope_id,status FROM analysis_runs WHERE id=?", (answer_run.analysis_run_id,)
    ).fetchone()
    if (
        run_row is None
        or run_row["scope_id"] != candidates["scope_snapshot_id"]
        or run_row["status"] != RunStatus.COMPLETED
    ):
        raise EvidenceValidationError("ask-question answer run did not preserve the original scope")
    execution = project.db.execute(
        "SELECT analysis_run_id,error_type FROM model_executions WHERE id=?",
        (answer_run.model_execution_id,),
    ).fetchone()
    if (
        execution is None
        or execution["analysis_run_id"] != answer_run.analysis_run_id
        or execution["error_type"] is not None
    ):
        raise EvidenceValidationError("ask-question answer has invalid model execution provenance")

    try:
        limitations_value = json.loads(answer_finding["limitations_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError("ask-question answer limitations are corrupt") from exc
    if not isinstance(limitations_value, list) or any(
        not isinstance(item, str) for item in limitations_value
    ):
        raise EvidenceValidationError("ask-question answer limitations are invalid")

    evidence_rows = project.finding_evidence(answer_run.finding_id)
    evidence_total = len(evidence_rows)
    evidence: list[AskQuestionEvidence] = []
    for row in evidence_rows[:MAX_ASK_QUESTION_EVIDENCE_ITEMS]:
        if row["kind"] != "TEXTUAL" or not row["span_id"]:
            # Framework answers should be grounded in textual evidence. This
            # guard also prevents an unexpected evidence kind being rendered as
            # a textual excerpt by this API.
            raise EvidenceValidationError("ask-question answer contains non-textual evidence")
        excerpt = project.span_text(str(row["span_id"]))
        evidence.append(
            {
                "evidence_id": str(row["id"]),
                "role": str(row["role"]),
                "citation": project.render_citation(str(row["id"])),
                "excerpt": excerpt[:MAX_ASK_QUESTION_EXCERPT_CHARS],
                "excerpt_truncated": len(excerpt) > MAX_ASK_QUESTION_EXCERPT_CHARS,
            }
        )
    return {
        "question": normalized_question,
        "framework_id": framework_id,
        "retrieval_mode": retrieval_mode,
        "scope_snapshot_id": candidates["scope_snapshot_id"],
        "candidate_analysis_run_id": candidates["analysis_run_id"],
        "candidate_skill_run_id": candidates["skill_run_id"],
        "candidate_finding_id": candidates["finding_id"],
        "candidate_retrieval_run_id": candidates["retrieval_run_id"],
        "coverage_id": candidates["coverage_id"],
        "coverage": coverage,
        "answer_coverage_id": answer_coverage_id,
        "answer_coverage": answer_coverage,
        "status": "answered",
        "answer_analysis_run_id": answer_run.analysis_run_id,
        "answer_finding_id": answer_run.finding_id,
        "model_execution_id": answer_run.model_execution_id,
        "answer_interpretation": str(answer_finding["interpretation"] or ""),
        "alternative_interpretation": str(answer_finding["alternative_interpretation"] or ""),
        "limitations": list(dict.fromkeys(limitations_value)),
        "evidence": evidence,
        "evidence_total": evidence_total,
        "evidence_truncated": evidence_total > len(evidence),
    }


def run_van_dijk_group_comparison_analysis(
    project,
    revision_ids: list[str],
    *,
    groups: list[VanDijkGroupComparisonGroupInput],
    shared_topic_event: str,
    comparability_basis: str,
    codebook_id: str,
) -> VanDijkGroupComparisonAnalysisResult:
    """Prepare a researcher-labeled, evidence-linked two-group comparison."""
    if not revision_ids:
        raise ProjectError("Van Dijk group comparison requires an explicit revision scope")
    _validate_batch_revision_ids(revision_ids)
    if len(groups) != 2 or not any(group.get("annotation_ids") for group in groups):
        raise ProjectError(
            "group comparison requires two groups and at least one selected annotation"
        )

    scope = project.snapshot(revision_ids)
    analysis_run_id = project.create_run(
        scope,
        {
            "profile": "builtin.van-dijk-group-comparison",
            "version": "1.0.0",
            "execution_kind": "DETERMINISTIC",
            "groups": groups,
            "shared_topic_event": shared_topic_event,
            "comparability_basis": comparability_basis,
            "codebook_id": codebook_id,
        },
    )
    try:
        project.start_run(analysis_run_id)
        result = run_van_dijk_group_comparison(
            SkillContext(project, analysis_run_id, scope),
            groups=cast(list[Mapping[str, object]], groups),
            shared_topic_event=shared_topic_event,
            comparability_basis=comparability_basis,
            codebook_id=codebook_id,
        )
        project.finalize_run(analysis_run_id)
        return {"analysis_run_id": analysis_run_id, **result}
    except BaseException as exc:
        project.fail_run(analysis_run_id, exc)
        raise


def run_batch_textual_descriptors_analysis(
    project, revision_ids: list[str], languages: tuple[str, ...] = ("en", "es")
) -> list[TextualDescriptorAnalysisResult]:
    """Create a separate immutable AnalysisRun for every selected revision."""
    if not revision_ids:
        raise ValueError("batch analysis requires at least one revision")
    _validate_batch_revision_ids(revision_ids)
    languages = _ordered_unique_languages(languages)
    return [
        run_textual_descriptors_analysis(project, [revision_id], languages)
        for revision_id in revision_ids
    ]


def _validate_batch_revision_ids(revision_ids: list[str]) -> None:
    """Reject duplicate inputs before they can create repeated independent runs."""
    if len(set(revision_ids)) != len(revision_ids):
        raise ProjectError("batch analysis revision selection must not contain duplicates")


def run_batch_controlled_cda_review(
    project,
    revision_ids: list[str],
    preset_name: str | None,
    *,
    frameworks: tuple[str, ...] = DEFAULT_CDA_FRAMEWORKS,
    languages: tuple[str, ...] = ("en", "es"),
    counterevidence_queries: Mapping[str, str] | None = None,
    routing_profile: str | None = None,
    provider_factory: ProviderFactory | None = None,
    reranking_gateway: ModelGateway | None = None,
    reranking_provider_id: str | None = None,
    reranking_credential_alias: str | None = None,
    reranking_model: str | None = None,
    reranking_top_n: int | None = None,
    question_candidate_limit: int = MAX_FRAMEWORK_HITS,
) -> list[ControlledCdaResult]:
    """Run controlled CDA separately per revision, never on concatenated documents."""
    if not revision_ids:
        raise ProjectError("batch controlled CDA review requires at least one revision")
    _validate_batch_revision_ids(revision_ids)
    return [
        run_controlled_cda_review(
            project,
            [revision_id],
            preset_name,
            frameworks=frameworks,
            languages=languages,
            counterevidence_queries=counterevidence_queries,
            routing_profile=routing_profile,
            provider_factory=provider_factory,
            reranking_gateway=reranking_gateway,
            reranking_provider_id=reranking_provider_id,
            reranking_credential_alias=reranking_credential_alias,
            reranking_model=reranking_model,
            reranking_top_n=reranking_top_n,
            question_candidate_limit=question_candidate_limit,
        )
        for revision_id in revision_ids
    ]


def run_saved_controlled_profile(
    project,
    revision_ids: list[str],
    profile_name: str,
    *,
    preset_name: str | None = None,
    provider_factory: ProviderFactory | None = None,
    _analysis_run_id_capture: list[str] | None = None,
) -> ControlledCdaResult:
    """Execute a stored declarative profile through the trusted controlled pipeline."""
    try:
        payload = project.analysis_profile(profile_name)
        profile = ControlledCdaProfile.model_validate(payload)
    except ValueError:
        raise ProjectError("analysis profile is invalid") from None
    profile.validate_method_contract()
    language_assignments = (
        _revision_language_assignments(project, revision_ids) if profile.local_preprocessors else {}
    )
    route_languages = {language.casefold().strip() for language in profile.nlp_routes}
    missing_routes = sorted(set(language_assignments.values()) - route_languages)
    if missing_routes:
        raise ProjectError(
            "saved profile requires an explicit local NLP route for document language(s): "
            + ", ".join(missing_routes)
        )
    normalized_routes = {
        language.casefold().strip(): route for language, route in profile.nlp_routes.items()
    }
    for language in set(language_assignments.values()):
        route = normalized_routes[language]
        if route.backend == "stanza" and not os.environ.get(route.model_dir_env or ""):
            raise ProjectError(
                f"selected Stanza route requires environment variable {route.model_dir_env!r}"
            )
    preflight_languages = (
        tuple(sorted(set(language_assignments.values()))) if language_assignments else None
    )
    language_preflight = profile.local_preflight(preflight_languages)
    blocked_requirements: list[str] = []
    # Only explicitly declared profile-level requirements are run-wide gates.
    # Capabilities inferred from local_preprocessors are handled per operation below.
    declared_global_requirements = tuple(profile.required_capabilities)
    for data in language_preflight.values():
        if not isinstance(data, Mapping):
            raise ProjectError("analysis profile local preflight response is malformed")
        statuses = data.get("statuses", {})
        if not isinstance(statuses, Mapping):
            raise ProjectError("analysis profile local preflight capability statuses are malformed")
        for capability in declared_global_requirements:
            status = statuses.get(capability, OPERATIONAL_UNASSESSED)
            if not isinstance(status, str) or status not in _LOCAL_OPERATIONAL_STATUSES:
                raise ProjectError("analysis profile local preflight contains an unknown status")
            if status != OPERATIONAL_SUPPORTED:
                # Language identifiers come from the profile and can contain private
                # researcher-supplied data. Capability/status values are closed registries.
                blocked_requirements.append(f"{capability}: {status}")
    if blocked_requirements:
        raise ProjectError(
            "analysis profile is not eligible on this machine: "
            + "; ".join(sorted(set(blocked_requirements)))
        )
    local_nlp_execution_plan = (
        _build_local_nlp_execution_plan(
            revision_ids,
            language_assignments,
            language_preflight,
            tuple(profile.local_preprocessors),
        )
        if profile.local_preprocessors
        else None
    )
    selected_preset = preset_name if preset_name is not None else profile.default_preset
    if selected_preset is None and profile.routing_profile is None:
        raise ProjectError("analysis profile requires an explicit preset or default_preset")
    stage_presets = _resolve_framework_presets(
        project,
        tuple(profile.frameworks),
        selected_preset,
        profile.routing_profile,
    )
    # Saved-profile NLP artifacts are persistent derived data. Resolve every
    # framework and synthesis preset, including its immutable execution metadata,
    # before building any of those artifacts.
    for stage_preset in stage_presets.values():
        snapshot_model_preset(project, stage_preset)
    reranking_gateway = None
    if profile.reranking_provider_id:
        assert profile.reranking_credential_alias is not None
        reranking_gateway = build_reranking_gateway(
            project, profile.reranking_provider_id, profile.reranking_credential_alias
        )
    eligible_revision_ids = (
        [
            revision_id
            for revision_id in revision_ids
            if any(
                revision_id in operation["eligible_revision_ids"]
                for operation in local_nlp_execution_plan["operations"].values()
            )
        ]
        if local_nlp_execution_plan is not None
        else []
    )
    local_nlp_artifacts, local_nlp_provenance = _build_profile_nlp_artifacts(
        project, eligible_revision_ids, profile
    )
    if local_nlp_execution_plan is not None:
        local_nlp_provenance["document_languages"] = language_assignments
        local_nlp_provenance["execution_plan"] = local_nlp_execution_plan
    canonical_payload = profile.model_dump(mode="json")
    payload_hash = hashlib.sha256(
        json.dumps(
            canonical_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    profile_provenance: dict[str, object] = {
        "name": profile_name,
        "version": profile.version,
        "payload_hash": payload_hash,
        "payload": canonical_payload,
        "language_preflight": language_preflight,
    }
    if profile.local_preprocessors:
        profile_provenance["local_nlp"] = local_nlp_provenance
    return run_controlled_cda_review(
        project,
        revision_ids,
        selected_preset,
        frameworks=tuple(profile.frameworks),
        languages=tuple(profile.languages),
        counterevidence_queries=profile.counterevidence_queries,
        research_questions=profile.research_questions,
        routing_profile=profile.routing_profile,
        profile_provenance=profile_provenance,
        provider_factory=provider_factory,
        reranking_gateway=reranking_gateway,
        reranking_provider_id=profile.reranking_provider_id,
        reranking_credential_alias=profile.reranking_credential_alias,
        reranking_model=profile.reranking_model,
        reranking_top_n=profile.reranking_top_n,
        local_preprocessors=tuple(profile.local_preprocessors),
        local_nlp_artifacts=local_nlp_artifacts,
        local_nlp_execution_plan=local_nlp_execution_plan,
        question_candidate_limit=profile.question_candidate_limit,
        _analysis_run_id_capture=_analysis_run_id_capture,
    )


@overload
def run_batch_saved_controlled_profile(
    project,
    revision_ids: list[str],
    profile_name: str,
    *,
    preset_name: str | None = None,
    provider_factory: ProviderFactory | None = None,
    continue_on_error: Literal[False] = False,
) -> list[ControlledCdaResult]: ...


@overload
def run_batch_saved_controlled_profile(
    project,
    revision_ids: list[str],
    profile_name: str,
    *,
    preset_name: str | None = None,
    provider_factory: ProviderFactory | None = None,
    continue_on_error: Literal[True],
) -> list[SavedProfileBatchItem]: ...


@overload
def run_batch_saved_controlled_profile(
    project,
    revision_ids: list[str],
    profile_name: str,
    *,
    preset_name: str | None = None,
    provider_factory: ProviderFactory | None = None,
    continue_on_error: bool,
) -> list[ControlledCdaResult] | list[SavedProfileBatchItem]: ...


def run_batch_saved_controlled_profile(
    project,
    revision_ids: list[str],
    profile_name: str,
    *,
    preset_name: str | None = None,
    provider_factory: ProviderFactory | None = None,
    continue_on_error: bool = False,
) -> list[ControlledCdaResult] | list[SavedProfileBatchItem]:
    """Run a saved declarative profile independently for each document revision."""
    if not revision_ids:
        raise ProjectError("batch saved-profile analysis requires at least one revision")
    _validate_batch_revision_ids(revision_ids)
    for revision_id in revision_ids:
        if (
            project.db.execute(
                "SELECT 1 FROM document_revisions WHERE id=?", (revision_id,)
            ).fetchone()
            is None
        ):
            raise ProjectError(
                "batch saved-profile analysis references an unknown document revision"
            )

    if not continue_on_error:
        return [
            run_saved_controlled_profile(
                project,
                [revision_id],
                profile_name,
                preset_name=preset_name,
                provider_factory=provider_factory,
            )
            for revision_id in revision_ids
        ]

    outcomes: list[SavedProfileBatchItem] = []
    for revision_id in revision_ids:
        analysis_run_id_capture: list[str] = []
        try:
            result = run_saved_controlled_profile(
                project,
                [revision_id],
                profile_name,
                preset_name=preset_name,
                provider_factory=provider_factory,
                _analysis_run_id_capture=analysis_run_id_capture,
            )
        except Exception as exc:  # noqa: BLE001 - continue mode must isolate each revision failure.
            outcomes.append(
                {
                    "revision_id": revision_id,
                    "status": "failed",
                    "analysis_run_id": _captured_failed_profile_run_id(
                        project, analysis_run_id_capture, revision_id
                    ),
                    "error": {
                        "type": type(exc).__name__,
                        "message": "Saved-profile analysis failed for this revision.",
                    },
                }
            )
        else:
            outcomes.append({"revision_id": revision_id, "status": "completed", "result": result})
    return outcomes


def _captured_failed_profile_run_id(
    project, captured_run_ids: list[str], revision_id: str
) -> str | None:
    """Return this invocation's failed run ID without searching unrelated run history."""
    if len(captured_run_ids) != 1:
        return None
    run_id = captured_run_ids[0]
    try:
        row = project.db.execute(
            "SELECT status,scope_id FROM analysis_runs WHERE id=?", (run_id,)
        ).fetchone()
        if (
            row is not None
            and row["status"] == RunStatus.FAILED
            and project.scope_revisions(str(row["scope_id"])) == [revision_id]
        ):
            return run_id
    except (ProjectError, sqlite3.Error, KeyError, IndexError, TypeError, ValueError):
        # Failure metadata is best effort; never replace the original safe outcome.
        return None
    return None


def _counterevidence_page_from_run(
    project, finding_id: str, phrase: str, retrieval_run_id: str, *, offset: int, limit: int
) -> CounterEvidenceAnalysisResult:
    """Read a continuation page only from the completed matching exhaustive run."""
    finding = project.get_finding(finding_id)
    source_run = project.db.execute(
        "SELECT scope_id FROM analysis_runs WHERE id=?", (finding["analysis_run_id"],)
    ).fetchone()
    saved = project.db.execute(
        """
        SELECT rr.*,sr.analysis_run_id AS skill_analysis_run_id,sr.skill_id,sr.skill_version,
               sr.scope_id AS skill_scope_id,sr.status AS skill_status,
               sr.input_finding_id,ar.scope_id AS run_scope_id,ar.status AS run_status,
               ar.config_json,ar.config_hash
        FROM retrieval_runs rr
        JOIN skill_runs sr ON sr.id=rr.skill_run_id
        JOIN analysis_runs ar ON ar.id=rr.analysis_run_id
        WHERE rr.id=?
        """,
        (retrieval_run_id,),
    ).fetchone()
    if not source_run or not saved:
        raise ProjectError("unknown counterevidence retrieval run")
    try:
        config = json.loads(saved["config_json"])
        query_terms = json.loads(saved["query_terms_json"])
    except json.JSONDecodeError as exc:
        raise ProjectError("counterevidence retrieval provenance is corrupt") from exc
    if (
        not isinstance(config, dict)
        or hashlib.sha256(saved["config_json"].encode("utf-8")).hexdigest() != saved["config_hash"]
        or config.get("profile") != "counterevidence-lexical"
        or config.get("finding_id") != finding_id
        or config.get("phrase") != phrase
        or query_terms != [phrase]
        or saved["retrieval_plan_ref"] != "retrieval/exhaustive-literal-match-candidates-v1"
        or saved["skill_id"] != "builtin.counterevidence-lexical"
        or saved["input_finding_id"] != finding_id
        or saved["skill_analysis_run_id"] != saved["analysis_run_id"]
        or saved["skill_scope_id"] != saved["scope_id"]
        or saved["scope_id"] != source_run["scope_id"]
        or saved["run_scope_id"] != saved["scope_id"]
        or saved["skill_status"] != RunStatus.COMPLETED
        or saved["run_status"] != RunStatus.COMPLETED
    ):
        raise ProjectError(
            "retrieval run does not match this finding, phrase, and original ScopeSnapshot"
        )
    coverage = project.coverage_descriptor(saved["coverage_id"])
    if (
        coverage.scope_snapshot_id != saved["scope_id"]
        or coverage.coverage_mode != CoverageMode.EXHAUSTIVE
        or coverage.population_count is None
    ):
        raise ProjectError("counterevidence retrieval coverage is invalid")
    candidate_count = int(saved["candidate_count"])
    persisted_count = int(
        project.db.execute(
            "SELECT COUNT(*) FROM retrieval_hits WHERE retrieval_run_id=?",
            (retrieval_run_id,),
        ).fetchone()[0]
    )
    if (
        persisted_count != candidate_count
        or int(saved["inspected_count"]) != candidate_count
        or int(saved["returned_count"]) != candidate_count
    ):
        raise ProjectError("counterevidence retrieval result is incomplete")
    page_rows = project.db.execute(
        "SELECT h.id,h.source_span_id,h.rank,h.retrieval_method,s.revision_id,"
        "s.start_offset,s.end_offset,s.text_hash FROM retrieval_hits h "
        "LEFT JOIN source_spans s ON s.id=h.source_span_id "
        "WHERE h.retrieval_run_id=? ORDER BY h.rank LIMIT ? OFFSET ?",
        (retrieval_run_id, limit, offset),
    ).fetchall()
    scoped_revisions = set(project.scope_revisions(str(saved["scope_id"])))
    matcher = re.compile(re.escape(phrase), re.IGNORECASE)
    for page_index, row in enumerate(page_rows, start=1):
        revision_id = row["revision_id"]
        if (
            not isinstance(revision_id, str)
            or revision_id not in scoped_revisions
            or row["retrieval_method"] != "exhaustive_literal_candidate"
            or row["rank"] != offset + page_index
            or not isinstance(row["start_offset"], int)
            or not isinstance(row["end_offset"], int)
            or row["start_offset"] < 0
            or row["end_offset"] <= row["start_offset"]
        ):
            raise ProjectError("counterevidence retrieval contains an invalid scoped candidate")
        source_text = project.revision_text(revision_id)
        start, end = int(row["start_offset"]), int(row["end_offset"])
        excerpt = source_text[start:end]
        if (
            end > len(source_text)
            or hashlib.sha256(excerpt.encode("utf-8")).hexdigest() != row["text_hash"]
            or matcher.fullmatch(excerpt) is None
        ):
            raise ProjectError(
                "counterevidence retrieval candidate no longer matches canonical text"
            )
    return {
        "analysis_run_id": str(saved["analysis_run_id"]),
        "skill_run_id": str(saved["skill_run_id"]),
        "retrieval_run_id": retrieval_run_id,
        "retrieval_hit_ids": [str(row["id"]) for row in page_rows],
        "candidate_span_ids": [str(row["source_span_id"]) for row in page_rows],
        "coverage_id": str(saved["coverage_id"]),
        "coverage_mode": CoverageMode.EXHAUSTIVE,
        "population_count": int(coverage.population_count),
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


def run_counterevidence_review(
    project,
    finding_id: str,
    phrase: str,
    *,
    offset: int = 0,
    limit: int = 100,
    retrieval_run_id: str | None = None,
) -> CounterEvidenceAnalysisResult:
    """Exhaustively enumerate a literal query, then page its immutable retrieval run."""
    if offset < 0 or not 1 <= limit <= 1_000:
        raise ProjectError("counterevidence page requires non-negative offset and limit 1..1000")
    if not phrase.strip():
        raise ProjectError("counterevidence phrase is required")
    if retrieval_run_id is not None:
        return _counterevidence_page_from_run(
            project, finding_id, phrase, retrieval_run_id, offset=offset, limit=limit
        )
    finding = project.get_finding(finding_id)
    prior_run = project.db.execute(
        "SELECT scope_id FROM analysis_runs WHERE id=?", (finding["analysis_run_id"],)
    ).fetchone()
    if not prior_run:
        raise ValueError("finding references an unknown analysis run")
    # A review run snapshots the same immutable revision selection; it does not gain
    # documents imported after the finding was created.
    scope_id = prior_run["scope_id"]
    review_run = project.create_run(
        scope_id, {"profile": "counterevidence-lexical", "finding_id": finding_id, "phrase": phrase}
    )
    try:
        project.start_run(review_run)
        result = run_counterevidence_lexical(
            SkillContext(project, review_run, scope_id),
            finding_id,
            phrase,
            offset=offset,
            limit=limit,
        )
        project.finalize_run(review_run)
        return {"analysis_run_id": review_run, **result}
    except BaseException as exc:
        project.fail_run(review_run, exc)
        raise


def run_social_actor_candidate_analysis(
    project, revision_ids: list[str], artifact_ids: list[str]
) -> AnalysisResult:
    """Run the conservative local parser preprocessor for actor expressions."""
    scope = project.snapshot(revision_ids)
    analysis_run_id = project.create_run(
        scope,
        {
            "profile": "builtin.social-actor-candidates",
            "execution_kind": "DETERMINISTIC",
            "artifact_ids": sorted(artifact_ids),
        },
    )
    try:
        project.start_run(analysis_run_id)
        result = run_social_actor_candidates(
            SkillContext(project, analysis_run_id, scope), artifact_ids
        )
        project.finalize_run(analysis_run_id)
        return {
            "analysis_run_id": analysis_run_id,
            "skill_run_id": result["skill_run_id"],
            "finding_ids": result["finding_ids"],
        }
    except BaseException as exc:
        project.fail_run(analysis_run_id, exc)
        raise


def run_agency_candidate_analysis(
    project, revision_ids: list[str], artifact_ids: list[str]
) -> AnalysisResult:
    """Run dependency-based agency candidates under an explicit parser artifact scope."""
    scope = project.snapshot(revision_ids)
    dependency_parse_preflight = _agency_dependency_parse_preflight(project, scope, artifact_ids)
    analysis_run_id = project.create_run(
        scope,
        {
            "profile": "builtin.agency-candidates",
            "execution_kind": "DETERMINISTIC",
            "artifact_ids": sorted(artifact_ids),
            "agency_dependency_parse_preflight": dependency_parse_preflight,
        },
    )
    try:
        project.start_run(analysis_run_id)
        result = run_agency_candidates(
            SkillContext(
                project,
                analysis_run_id,
                scope,
                dependency_parse_preflight=dependency_parse_preflight,
            ),
            artifact_ids,
        )
        project.finalize_run(analysis_run_id)
        return {
            "analysis_run_id": analysis_run_id,
            "skill_run_id": result["skill_run_id"],
            "finding_ids": result["finding_ids"],
        }
    except BaseException as exc:
        project.fail_run(analysis_run_id, exc)
        raise
