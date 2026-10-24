"""Deterministic validation of persisted evidence lineage before presentation."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from .errors import (
    DocumentError,
    EvidenceValidationError,
    ProjectError,
    ScopeViolationError,
    SkillManifestError,
)
from .policy import claim_allowed, is_scope_revision_population_definition
from .types import ClaimType, CoverageDescriptor, CoverageMode, EvidenceRole

_COMPARISON_PROVENANCE_KEYS = frozenset(
    {
        "comparison_run_id",
        "comparison_metric",
        "comparison_result_hash",
        "left_scope_snapshot_id",
        "right_scope_snapshot_id",
        "left_revision_ids",
        "right_revision_ids",
        "left_population",
        "right_population",
        "selected_row",
        "comparability_basis",
    }
)
_BUNDLE_CALCULATION_PREFIX = "surface-lexical-bundles-v1-size-"
_CORE_STATISTIC_PROVENANCE_FIELDS = {
    "surface-frequency-v1": frozenset({"term", "tokenizer"}),
    "surface-frequency-v2": frozenset(
        {"term", "tokenizer", "normalization", "unicode_data_version"}
    ),
    "surface-document-dispersion-v1": frozenset(
        {
            "document_count",
            "document_frequency",
            "document_range",
            "documents",
            "population_tokens",
        }
    ),
    "surface-document-dispersion-v2": frozenset(
        {
            "document_count",
            "document_frequency",
            "document_range",
            "documents",
            "population_tokens",
            "tokenizer",
            "normalization",
            "unicode_data_version",
        }
    ),
    "surface-window-association-v2": frozenset(
        {
            "measure",
            "query",
            "collocate",
            "window",
            "min_frequency",
            "focus_frequency",
            "pair_frequency",
            "token_frequency",
            "pair_opportunities",
            "expected_frequency",
            "mutual_information",
            "t_score",
            "tokenizer",
        }
    ),
    "derived-token-artifact-frequency-v1": frozenset(
        {
            "artifact_ids",
            "artifact_type",
            "artifact_schema_version",
            "field",
            "query_value",
            "query_engine",
        }
    ),
    "derived-surface-token-frequency-v2": frozenset(
        {
            "artifact_ids",
            "artifact_type",
            "artifact_schema_version",
            "field",
            "query_value",
            "query_engine",
            "tokenizer",
            "normalization",
            "unicode_data_version",
        }
    ),
}
_ABSENCE_CALCULATION_VERSIONS = frozenset(
    {
        "surface-frequency-v1",
        "surface-frequency-v2",
        "surface-document-dispersion-v1",
        "surface-document-dispersion-v2",
        "derived-token-artifact-frequency-v1",
        "derived-surface-token-frequency-v2",
    }
)
@dataclass(frozen=True)
class FindingValidation:
    """Structural lineage/policy result; semantic entailment is never assessed here."""

    finding_id: str
    valid: bool
    errors: tuple[str, ...]
    semantic_entailment_assessed: Literal[False] = False


def validate_finding(project, finding_id: str) -> FindingValidation:
    """Check canonical evidence integrity and prevent cross-scope citations."""
    finding = project.get_finding(finding_id)
    run = project.db.execute(
        "SELECT scope_id FROM analysis_runs WHERE id=?", (finding["analysis_run_id"],)
    ).fetchone()
    errors: list[str] = []
    if not run:
        return FindingValidation(finding_id, False, ("finding references unknown analysis run",))
    if finding["skill_run_id"]:
        skill_run = project.db.execute(
            "SELECT analysis_run_id,scope_id,skill_id,skill_version FROM skill_runs WHERE id=?",
            (finding["skill_run_id"],),
        ).fetchone()
        if (
            not skill_run
            or skill_run["analysis_run_id"] != finding["analysis_run_id"]
            or skill_run["scope_id"] != run["scope_id"]
            or skill_run["skill_id"] != finding["skill_id"]
            or skill_run["skill_version"] != finding["skill_version"]
        ):
            errors.append("finding has invalid skill run provenance")
    scope_revisions = set(project.scope_revisions(run["scope_id"]))
    dangling_evidence = project.db.execute(
        "SELECT fe.evidence_id FROM finding_evidence fe "
        "LEFT JOIN evidence_artifacts e ON e.id=fe.evidence_id "
        "WHERE fe.finding_id=? AND e.id IS NULL",
        (finding_id,),
    ).fetchall()
    for dangling in dangling_evidence:
        errors.append(
            f"finding references unknown evidence artifact {dangling['evidence_id']}"
        )
    links = project.finding_evidence(finding_id)
    if not links:
        errors.append("finding has no evidence")
    for item in links:
        try:
            EvidenceRole(item["role"])
        except (TypeError, ValueError):
            errors.append(f"evidence {item['id']} has unsupported evidence role")
        if item["kind"] == "EXTERNAL_SOURCE":
            _validate_external_evidence(project, item["id"], errors)
            continue
        if item["kind"] == "USER_CONTEXT":
            _validate_user_context_evidence(project, item["id"], errors)
            continue
        if item["kind"] == "CORPUS_STATISTIC":
            _validate_corpus_statistic_evidence(
                project,
                item["id"],
                run["scope_id"],
                finding["analysis_run_id"],
                errors,
            )
            continue
        if item["kind"] != "TEXTUAL":
            if item["kind"] not in {"CORPUS_STATISTIC", "EXTERNAL_SOURCE", "USER_CONTEXT"}:
                errors.append(f"evidence {item['id']} has unsupported evidence kind")
            continue
        if not item["span_id"]:
            errors.append(f"textual evidence {item['id']} has no source span")
            continue
        span = project.db.execute(
            "SELECT revision_id,start_offset,end_offset,text_hash FROM source_spans WHERE id=?",
            (item["span_id"],),
        ).fetchone()
        if not span:
            errors.append(f"evidence {item['id']} references unknown source span")
            continue
        if span["revision_id"] not in scope_revisions:
            errors.append(f"evidence {item['id']} is outside the analysis scope")
            continue
        revision_text = project.revision_text(span["revision_id"])
        start_offset = span["start_offset"]
        end_offset = span["end_offset"]
        if (
            type(start_offset) is not int
            or type(end_offset) is not int
            or start_offset < 0
            or end_offset <= start_offset
            or end_offset > len(revision_text)
        ):
            errors.append(f"source span for evidence {item['id']} has invalid offsets")
            continue
        exact = revision_text[start_offset:end_offset]
        if hashlib.sha256(exact.encode()).hexdigest() != span["text_hash"]:
            errors.append(f"source span hash mismatch for evidence {item['id']}")
        _validate_kwic_textual_provenance(
            project,
            item["id"],
            run["scope_id"],
            span["revision_id"],
            start_offset,
            end_offset,
            exact,
            errors,
        )
    _validate_claim_contract(
        project, finding, run["scope_id"], finding["analysis_run_id"], links, errors
    )
    return FindingValidation(finding_id, not errors, tuple(errors))


def _has_linked_evidence(links, *, kinds: set[str], roles: set[str] | None = None) -> bool:
    return any(item["kind"] in kinds and (roles is None or item["role"] in roles) for item in links)


def _has_supporting_text_in_each_revision(project, links, scope_id: str) -> bool:
    scoped_revisions = set(project.scope_revisions(scope_id))
    supported_revisions: set[str] = set()
    for item in links:
        if item["kind"] != "TEXTUAL" or item["role"] != "SUPPORT" or not item["span_id"]:
            continue
        span = project.db.execute(
            "SELECT revision_id FROM source_spans WHERE id=?", (item["span_id"],)
        ).fetchone()
        if span and span["revision_id"] in scoped_revisions:
            supported_revisions.add(span["revision_id"])
    return bool(scoped_revisions) and scoped_revisions.issubset(supported_revisions)


def _validate_external_evidence(project, evidence_id: str, errors: list[str]) -> None:
    try:
        payload = project.evidence_content(evidence_id)
    except EvidenceValidationError as exc:
        errors.append(str(exc))
        return
    if not all(
        isinstance(payload.get(key), str) and payload[key].strip()
        for key in ("source", "excerpt", "retrieved_at")
    ):
        errors.append(f"external evidence {evidence_id} lacks required provenance")


def _validate_user_context_evidence(project, evidence_id: str, errors: list[str]) -> None:
    try:
        payload = project.evidence_content(evidence_id)
    except EvidenceValidationError as exc:
        errors.append(str(exc))
        return
    if not all(
        isinstance(payload.get(key), str) and payload[key].strip() for key in ("label", "content")
    ):
        errors.append(f"user context evidence {evidence_id} lacks label or content")


def _validate_kwic_textual_provenance(
    project,
    evidence_id: str,
    scope_id: str,
    revision_id: str,
    start_offset: int,
    end_offset: int,
    exact_text: str,
    errors: list[str],
) -> None:
    """Recompute v1 historical or v2 Unicode-aware corpus KWIC evidence."""
    try:
        provenance = project.evidence_content(evidence_id)
    except EvidenceValidationError as exc:
        errors.append(str(exc))
        return
    if provenance.get("acquisition_method") != "corpus_kwic":
        return

    from .corpus import (
        SURFACE_NORMALIZATION_V2,
        SURFACE_TOKENIZER_V2,
        TOKEN,
        _surface_form_v2,
        _tokenize_v2,
    )

    version = provenance.get("calculation_version")
    query = provenance.get("query")
    if not isinstance(query, str):
        errors.append(f"textual KWIC evidence {evidence_id} has invalid calculation provenance")
        return
    if version == "surface-kwic-v1":
        if provenance.get("tokenizer") != TOKEN.pattern:
            errors.append(f"textual KWIC evidence {evidence_id} has invalid calculation provenance")
            return
        query_tokens = TOKEN.findall(query)
        if len(query_tokens) != 1 or TOKEN.fullmatch(exact_text) is None:
            errors.append(
                f"textual KWIC evidence {evidence_id} does not match its versioned scope calculation"
            )
            return
        needle = query_tokens[0].casefold()

        def tokens_in(text: str):
            return ((match.group(0), match.start(), match.end()) for match in TOKEN.finditer(text))

        normalized = str.casefold
        expected_criterion = "all scope texts tokenized by corpus.TOKEN and enumerated"
    elif version == "surface-kwic-v2":
        query_tokens = list(_tokenize_v2(query))
        exact_tokens = list(_tokenize_v2(exact_text))
        if (
            provenance.get("tokenizer") != SURFACE_TOKENIZER_V2
            or provenance.get("normalization") != SURFACE_NORMALIZATION_V2
            or provenance.get("unicode_data_version") != unicodedata.unidata_version
            or len(query_tokens) != 1
            or len(exact_tokens) != 1
            or exact_tokens[0].start_offset != 0
            or exact_tokens[0].end_offset != len(exact_text)
        ):
            errors.append(
                f"textual KWIC evidence {evidence_id} has unsupported or invalid tokenizer provenance"
            )
            return
        needle = _surface_form_v2(query_tokens[0].surface)

        def tokens_in(text: str):
            return (
                (token.surface, token.start_offset, token.end_offset)
                for token in _tokenize_v2(text)
            )

        normalized = _surface_form_v2
        if normalized(exact_tokens[0].surface) != needle:
            errors.append(
                f"textual KWIC evidence {evidence_id} does not match its versioned scope calculation"
            )
            return
        expected_criterion = f"all scope texts tokenized by {SURFACE_TOKENIZER_V2} and enumerated"
    else:
        errors.append(f"textual KWIC evidence {evidence_id} has an unsupported calculation version")
        return

    try:
        evidence_scope_id = provenance.get("scope_snapshot_id")
        if not isinstance(evidence_scope_id, str) or not evidence_scope_id:
            raise EvidenceValidationError("KWIC evidence has no immutable scope")
        revisions = project.scope_revisions(evidence_scope_id)
        finding_revisions = project.scope_revisions(scope_id)
        if revision_id not in revisions or set(revisions) != set(finding_revisions):
            raise EvidenceValidationError("KWIC evidence is outside its immutable finding scope")
        token_population = 0
        candidate_count = 0
        selected_token_hit = False
        for scoped_revision_id in revisions:
            for surface, token_start, token_end in tokens_in(
                project.revision_text(scoped_revision_id)
            ):
                if (
                    scoped_revision_id == revision_id
                    and token_start == start_offset
                    and token_end == end_offset
                    and surface == exact_text
                ):
                    selected_token_hit = True
                token_population += 1
                candidate_count += normalized(surface) == needle
        coverage_id = provenance.get("coverage_descriptor_id")
        if not isinstance(coverage_id, str) or not coverage_id:
            raise EvidenceValidationError("KWIC evidence has no coverage descriptor")
        coverage = project.coverage_descriptor(coverage_id)
        if (
            not selected_token_hit
            or provenance.get("coverage_mode") != "EXHAUSTIVE"
            or type(provenance.get("candidate_count")) is not int
            or provenance.get("candidate_count") != candidate_count
            or type(provenance.get("inspected_count")) is not int
            or provenance.get("inspected_count") != token_population
            or coverage.scope_snapshot_id != evidence_scope_id
            or coverage.coverage_mode.value != "EXHAUSTIVE"
            or coverage.population_definition
            != "all surface-token positions in the immutable ScopeSnapshot"
            or coverage.population_count != token_population
            or (
                coverage.inspection_count != token_population
                if version == "surface-kwic-v2"
                else coverage.inspection_count not in (None, token_population)
            )
            or coverage.completion_criterion != expected_criterion
        ):
            raise EvidenceValidationError("KWIC evidence does not match its exhaustive population")
    except (
        DocumentError,
        EvidenceValidationError,
        ScopeViolationError,
        ValueError,
        TypeError,
    ) as exc:
        errors.append(f"textual KWIC evidence {evidence_id} cannot be verified: {exc}")


def _validate_corpus_statistic_evidence(
    project, evidence_id: str, scope_id: str, analysis_run_id: str, errors: list[str]
) -> None:
    try:
        payload = project.evidence_content(evidence_id)
    except EvidenceValidationError as exc:
        errors.append(str(exc))
        return
    if not isinstance(payload.get("label"), str) or not payload["label"].strip():
        errors.append(f"corpus statistic evidence {evidence_id} lacks a label")
    if (
        not isinstance(payload.get("population_definition"), str)
        or not payload["population_definition"].strip()
    ):
        errors.append(f"corpus statistic evidence {evidence_id} lacks a population definition")
    if payload.get("scope_snapshot_id") != scope_id:
        errors.append(f"corpus statistic evidence {evidence_id} is outside the analysis scope")
    value = payload.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"corpus statistic evidence {evidence_id} lacks a numeric value")
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"corpus statistic evidence {evidence_id} has a non-finite value")
    provenance = payload.get("provenance")
    if provenance is not None:
        _validate_corpus_statistic_provenance(
            project, evidence_id, scope_id, analysis_run_id, payload, provenance, errors
        )


def _has_zero_statistic_for_coverage(
    project,
    links,
    scope_id: str,
    coverage_id: str,
    coverage: CoverageDescriptor,
) -> bool:
    """Require a supporting exact zero statistic for the identical population."""
    for item in links:
        if item["kind"] != "CORPUS_STATISTIC" or item["role"] != "SUPPORT":
            continue
        try:
            payload = project.evidence_content(item["id"])
        except EvidenceValidationError:
            continue
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            continue
        calculation_version = _recognized_core_statistic_version(provenance)
        if (
            calculation_version == "derived-token-artifact-frequency-v1"
            and provenance.get("artifact_type") == "parsed_tokens"
        ):
            # Legacy parsed artifacts lack annotation-coverage provenance, so
            # their zeroes cannot establish corpus-wide absence.
            continue
        recorded_population_count = provenance.get("population_count")
        if (
            not _is_comparison_statistic(provenance)
            and (
                calculation_version in _ABSENCE_CALCULATION_VERSIONS
                or (
                    isinstance(calculation_version, str)
                    and calculation_version.startswith(_BUNDLE_CALCULATION_PREFIX)
                )
            )
            and payload.get("scope_snapshot_id") == scope_id
            and payload.get("population_definition") == coverage.population_definition
            and _is_exact_count(payload.get("value"))
            and payload.get("value") == 0
            and provenance.get("coverage_id") == coverage_id
            and isinstance(recorded_population_count, int)
            and not isinstance(recorded_population_count, bool)
            and recorded_population_count == coverage.population_count
        ):
            return True
    return False


def _has_quantified_statistic_for_coverage(
    project,
    links,
    scope_id: str,
    coverage_id: str,
    coverage: CoverageDescriptor,
) -> bool:
    """Require a versioned deterministic statistic for the exact finding population."""
    for item in links:
        if item["kind"] != "CORPUS_STATISTIC":
            continue
        try:
            payload = project.evidence_content(item["id"])
        except EvidenceValidationError:
            continue
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            continue
        if _is_comparison_statistic(provenance):
            continue
        calculation_version = _recognized_core_statistic_version(provenance)
        recorded_population_count = provenance.get("population_count")
        if (
            payload.get("scope_snapshot_id") == scope_id
            and payload.get("population_definition") == coverage.population_definition
            and provenance.get("coverage_id") == coverage_id
            and item["role"] == "SUPPORT"
            and calculation_version is not None
            and _valid_core_statistic_value(calculation_version, payload.get("value"))
            and isinstance(recorded_population_count, int)
            and not isinstance(recorded_population_count, bool)
            and recorded_population_count == coverage.population_count
        ):
            return True
    return False


def _recognized_core_statistic_version(provenance: dict[str, object]) -> str | None:
    """Recognize only versioned provenance emitted by deterministic Core producers.

    This is an allowlist, not proof that a statistic's free-form interpretation is
    entailed.  The detailed provenance shape and scope are checked independently.
    """
    version = provenance.get("calculation_version")
    if not isinstance(version, str) or not version.strip():
        return None
    required_fields = _CORE_STATISTIC_PROVENANCE_FIELDS.get(version)
    if required_fields is None and version.startswith(_BUNDLE_CALCULATION_PREFIX):
        suffix = version.removeprefix(_BUNDLE_CALCULATION_PREFIX)
        if not suffix.isdigit() or not 2 <= int(suffix) <= 10:
            return None
        if provenance.get("size") != int(suffix):
            return None
        required_fields = frozenset({"bundle", "size", "tokenizer"})
    if required_fields is None or not required_fields.issubset(provenance):
        return None
    if not _valid_core_calculation_details(version, provenance):
        return None
    population_count = provenance.get("population_count")
    coverage_id = provenance.get("coverage_id")
    if (
        not isinstance(population_count, int)
        or isinstance(population_count, bool)
        or population_count < 0
        or not isinstance(coverage_id, str)
        or not coverage_id.strip()
    ):
        return None
    return version


def _valid_core_calculation_details(version: str, provenance: dict[str, object]) -> bool:
    """Check the small producer-specific shape needed to trust a version ID."""
    if version == "surface-frequency-v1":
        return all(_nonblank_string(provenance.get(key)) for key in ("term", "tokenizer"))
    if version == "surface-frequency-v2":
        return all(
            _nonblank_string(provenance.get(key))
            for key in ("term", "tokenizer", "normalization", "unicode_data_version")
        )
    if version == "surface-document-dispersion-v1":
        return _valid_dispersion_summary(provenance)
    if version == "surface-document-dispersion-v2":
        return (
            all(
                _nonblank_string(provenance.get(key))
                for key in ("tokenizer", "normalization", "unicode_data_version")
            )
            and _valid_dispersion_summary(provenance)
        )
    if version.startswith(_BUNDLE_CALCULATION_PREFIX):
        return all(_nonblank_string(provenance.get(key)) for key in ("bundle", "tokenizer"))
    if version == "surface-window-association-v2":
        return (
            provenance.get("measure") == "mutual_information_log2"
            and all(
                _nonblank_string(provenance.get(key)) for key in ("query", "collocate", "tokenizer")
            )
            and all(
                _integer_at_least(provenance.get(key), 1)
                for key in (
                    "window",
                    "min_frequency",
                    "focus_frequency",
                    "pair_frequency",
                    "token_frequency",
                    "pair_opportunities",
                )
            )
            and all(
                _finite_number(provenance.get(key))
                for key in ("expected_frequency", "mutual_information", "t_score")
            )
        )
    if version == "derived-token-artifact-frequency-v1":
        artifact_ids = provenance.get("artifact_ids")
        return (
            isinstance(artifact_ids, list)
            and bool(artifact_ids)
            and all(isinstance(item, str) and bool(item.strip()) for item in artifact_ids)
            and _nonblank_string(provenance.get("artifact_type"))
            and _nonblank_string(provenance.get("artifact_schema_version"))
            and _nonblank_string(provenance.get("field"))
            and _nonblank_string(provenance.get("query_value"))
            and provenance.get("query_engine") == "duckdb"
        )
    if version == "derived-surface-token-frequency-v2":
        from .corpus import SURFACE_NORMALIZATION_V2, SURFACE_TOKENIZER_V2
        from .nlp_artifacts import SURFACE_TOKEN_SCHEMA_VERSION

        artifact_ids = provenance.get("artifact_ids")
        return (
            isinstance(artifact_ids, list)
            and bool(artifact_ids)
            and all(isinstance(item, str) and bool(item.strip()) for item in artifact_ids)
            and provenance.get("artifact_type") == "surface_tokens"
            and provenance.get("artifact_schema_version") == SURFACE_TOKEN_SCHEMA_VERSION
            and provenance.get("field") == "normalized"
            and _nonblank_string(provenance.get("query_value"))
            and provenance.get("query_engine") == "duckdb"
            and provenance.get("tokenizer") == SURFACE_TOKENIZER_V2
            and provenance.get("normalization") == SURFACE_NORMALIZATION_V2
            and provenance.get("unicode_data_version") == unicodedata.unidata_version
        )
    return False


def _valid_dispersion_summary(provenance: dict[str, object]) -> bool:
    documents = provenance.get("documents")
    document_count = provenance.get("document_count")
    document_frequency = provenance.get("document_frequency")
    document_range = provenance.get("document_range")
    population_tokens = provenance.get("population_tokens")
    return (
        isinstance(documents, list)
        and isinstance(document_count, int)
        and not isinstance(document_count, bool)
        and document_count > 0
        and len(documents) == document_count
        and isinstance(document_frequency, int)
        and not isinstance(document_frequency, bool)
        and 0 <= document_frequency <= document_count
        and _unit_interval_number(document_range)
        and isinstance(population_tokens, int)
        and not isinstance(population_tokens, bool)
        and population_tokens >= 0
        and all(
            isinstance(row, dict)
            and isinstance(row.get("revision_id"), str)
            and isinstance(row.get("occurrences"), int)
            and not isinstance(row.get("occurrences"), bool)
            and isinstance(row.get("token_count"), int)
            and not isinstance(row.get("token_count"), bool)
            and row["occurrences"] >= 0
            and row["token_count"] >= row["occurrences"]
            for row in documents
        )
    )


def _nonblank_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _integer_at_least(value: object, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _finite_number(value: object) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and math.isfinite(value)
    )


def _is_exact_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_core_statistic_value(version: str, value: object) -> bool:
    if version == "surface-window-association-v2":
        return _finite_number(value)
    return _is_exact_count(value)


def _unit_interval_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0 <= value <= 1
    return isinstance(value, float) and math.isfinite(value) and 0 <= value <= 1


def _is_comparison_statistic(provenance: object) -> bool:
    return isinstance(provenance, dict) and bool(
        _COMPARISON_PROVENANCE_KEYS.intersection(provenance)
    )


def _has_comparison_statistic_for_coverage(
    project, links, analysis_run_id: str, scope_id: str, coverage_id: str
) -> bool:
    for item in links:
        if item["kind"] != "CORPUS_STATISTIC" or item["role"] != "SUPPORT":
            continue
        try:
            payload = project.evidence_content(item["id"])
        except EvidenceValidationError:
            continue
        provenance = payload.get("provenance")
        if (
            payload.get("scope_snapshot_id") == scope_id
            and isinstance(provenance, dict)
            and provenance.get("comparison_run_id")
            and provenance.get("analysis_run_id") == analysis_run_id
            and provenance.get("coverage_id") == coverage_id
            and isinstance(provenance.get("comparability_basis"), str)
            and bool(provenance["comparability_basis"].strip())
        ):
            return True
    return False


def _validate_corpus_statistic_provenance(
    project,
    evidence_id: str,
    scope_id: str,
    analysis_run_id: str,
    payload: dict[str, object],
    provenance: object,
    errors: list[str],
) -> None:
    """Validate general calculation provenance and, when present, NLP artifacts."""
    if not isinstance(provenance, dict):
        errors.append(f"corpus statistic evidence {evidence_id} has invalid provenance")
        return
    coverage_id = provenance.get("coverage_id")
    if coverage_id is not None:
        if not isinstance(coverage_id, str) or not coverage_id:
            errors.append(
                f"corpus statistic evidence {evidence_id} has invalid coverage provenance"
            )
        else:
            try:
                coverage = project.coverage_descriptor(coverage_id)
            except EvidenceValidationError as exc:
                errors.append(str(exc))
            else:
                if coverage.scope_snapshot_id != scope_id:
                    errors.append(
                        f"corpus statistic evidence {evidence_id} has coverage outside its scope"
                    )
    artifact_keys = {"artifact_ids", "artifact_type", "field"}
    if artifact_keys.intersection(provenance):
        _validate_artifact_statistic_provenance(
            project, evidence_id, scope_id, payload, provenance, errors
        )
    if provenance.get("calculation_version") in {
        "surface-frequency-v1",
        "surface-frequency-v2",
    }:
        _validate_surface_frequency_statistic(
            project, evidence_id, scope_id, payload, provenance, errors
        )
    if isinstance(provenance.get("calculation_version"), str) and provenance[
        "calculation_version"
    ].startswith(_BUNDLE_CALCULATION_PREFIX):
        _validate_lexical_bundle_statistic(
            project, evidence_id, scope_id, payload, provenance, errors
        )
    if provenance.get("calculation_version") == "surface-document-dispersion-v1":
        _validate_document_dispersion_statistic(
            project, evidence_id, scope_id, payload, provenance, errors, version="v1"
        )
    if provenance.get("calculation_version") == "surface-document-dispersion-v2":
        _validate_document_dispersion_statistic(
            project, evidence_id, scope_id, payload, provenance, errors, version="v2"
        )
    if provenance.get("calculation_version") == "surface-window-association-v2":
        _validate_window_association_statistic(
            project, evidence_id, scope_id, payload, provenance, errors
        )
    if _is_comparison_statistic(provenance):
        _validate_comparison_statistic_provenance(
            project, evidence_id, scope_id, analysis_run_id, payload, provenance, errors
        )


def _validate_surface_frequency_statistic(
    project,
    evidence_id: str,
    scope_id: str,
    payload: dict[str, object],
    provenance: dict[str, object],
    errors: list[str],
) -> None:
    """Recompute the versioned frequency result against its immutable scope."""
    from .corpus import (
        SURFACE_NORMALIZATION_V2,
        SURFACE_TOKENIZER_V2,
        TOKEN,
        _frequency_v1,
        _surface_form_v2,
        _tokenize_v2,
        frequency,
    )

    term = provenance.get("term")
    version = provenance.get("calculation_version")
    if not isinstance(term, str):
        errors.append(
            f"surface frequency evidence {evidence_id} has invalid calculation provenance"
        )
        return
    if version == "surface-frequency-v1":
        if (
            provenance.get("tokenizer") != TOKEN.pattern
            or TOKEN.findall(term) != [term]
            or term.casefold() != term
        ):
            errors.append(
                f"surface frequency evidence {evidence_id} has invalid calculation provenance"
            )
            return
        count_frequency = _frequency_v1
    elif version == "surface-frequency-v2":
        tokens = list(_tokenize_v2(term))
        if (
            provenance.get("tokenizer") != SURFACE_TOKENIZER_V2
            or provenance.get("normalization") != SURFACE_NORMALIZATION_V2
            or provenance.get("unicode_data_version") != unicodedata.unidata_version
            or len(tokens) != 1
            or tokens[0].start_offset != 0
            or tokens[0].end_offset != len(term)
            or _surface_form_v2(term) != term
        ):
            errors.append(
                f"surface frequency evidence {evidence_id} has unsupported or invalid tokenizer provenance"
            )
            return
        count_frequency = frequency
    else:
        errors.append(
            f"surface frequency evidence {evidence_id} has invalid calculation provenance"
        )
        return
    try:
        counts = count_frequency(project, scope_id)
        coverage = project.coverage_descriptor(str(provenance.get("coverage_id", "")))
    except (DocumentError, EvidenceValidationError, ScopeViolationError) as exc:
        errors.append(f"surface frequency evidence {evidence_id} cannot be verified: {exc}")
        return

    population_count = sum(counts.values())
    population_definition = "all surface-token positions in the immutable ScopeSnapshot"
    if (
        not _is_exact_count(payload.get("value"))
        or payload.get("value") != counts.get(term, 0)
        or not _is_exact_count(provenance.get("population_count"))
        or provenance.get("population_count") != population_count
        or not _is_exact_count(provenance.get("population_tokens"))
        or provenance.get("population_tokens") != population_count
        or payload.get("scope_snapshot_id") != scope_id
        or payload.get("population_definition") != population_definition
        or coverage.scope_snapshot_id != scope_id
        or coverage.coverage_mode.value != "EXHAUSTIVE"
        or coverage.population_definition != population_definition
        or coverage.population_count != population_count
        or coverage.inspection_count != population_count
    ):
        errors.append(
            f"surface frequency evidence {evidence_id} does not match its ScopeSnapshot coverage"
        )


def _validate_lexical_bundle_statistic(
    project,
    evidence_id: str,
    scope_id: str,
    payload: dict[str, object],
    provenance: dict[str, object],
    errors: list[str],
) -> None:
    """Recompute a versioned lexical-bundle count against its immutable scope."""
    from .corpus import TOKEN, _texts

    bundle = provenance.get("bundle")
    size = provenance.get("size")
    if (
        not isinstance(bundle, str)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or not 2 <= size <= 10
        or provenance.get("tokenizer") != TOKEN.pattern
    ):
        errors.append(f"lexical bundle evidence {evidence_id} has invalid calculation provenance")
        return
    terms = [token.casefold() for token in TOKEN.findall(bundle)]
    if len(terms) != size or " ".join(terms) != bundle:
        errors.append(f"lexical bundle evidence {evidence_id} has invalid calculation provenance")
        return
    try:
        counts: Counter[tuple[str, ...]] = Counter()
        population_count = 0
        for _, text in _texts(project, scope_id):
            document_terms = [token.casefold() for token in TOKEN.findall(text)]
            positions = len(document_terms) - size + 1
            if positions > 0:
                population_count += positions
                counts.update(
                    tuple(document_terms[index : index + size])
                    for index in range(positions)
                )
        coverage = project.coverage_descriptor(str(provenance.get("coverage_id", "")))
    except (DocumentError, EvidenceValidationError, ScopeViolationError, ValueError) as exc:
        errors.append(f"lexical bundle evidence {evidence_id} cannot be verified: {exc}")
        return

    expected_count = counts[tuple(terms)]
    population_definition = (
        f"all contiguous {size}-token surface n-gram positions in the immutable ScopeSnapshot"
    )
    expected_negative_status = (
        "ABSENCE_IN_EXHAUSTIVE_POPULATION" if expected_count == 0 else "NOT_APPLICABLE"
    )
    if (
        not _is_exact_count(payload.get("value"))
        or payload.get("value") != expected_count
        or not _is_exact_count(provenance.get("population_count"))
        or provenance.get("population_count") != population_count
        or payload.get("scope_snapshot_id") != scope_id
        or payload.get("population_definition") != population_definition
        or coverage.scope_snapshot_id != scope_id
        or coverage.coverage_mode.value != "EXHAUSTIVE"
        or coverage.population_definition != population_definition
        or coverage.population_count != population_count
        or coverage.inspection_count != population_count
        or coverage.negative_result_status.value != expected_negative_status
    ):
        errors.append(
            f"lexical bundle evidence {evidence_id} does not match its ScopeSnapshot coverage"
        )


def _validate_document_dispersion_statistic(
    project,
    evidence_id: str,
    scope_id: str,
    payload: dict[str, object],
    provenance: dict[str, object],
    errors: list[str],
    *,
    version: str,
) -> None:
    """Recompute current or historical dispersion evidence under its own tokenizer."""
    from .corpus import (
        SURFACE_NORMALIZATION_V2,
        SURFACE_TOKENIZER_V2,
        TOKEN,
        _document_dispersion_result_v1,
        _surface_form_v2,
        _tokenize_v2,
        document_dispersion,
    )

    label = payload.get("label")
    prefix = "surface occurrences of "
    suffix = " across the scoped document population"
    if not isinstance(label, str) or not label.startswith(prefix) or not label.endswith(suffix):
        errors.append(f"document dispersion evidence {evidence_id} has invalid calculation label")
        return
    try:
        term = ast.literal_eval(label[len(prefix) : -len(suffix)])
    except (SyntaxError, ValueError):
        errors.append(f"document dispersion evidence {evidence_id} has invalid calculation label")
        return
    if version == "v1":
        valid_term = (
            isinstance(term, str)
            and len(TOKEN.findall(term)) == 1
            and TOKEN.findall(term)[0].casefold() == term
        )
    else:
        tokens = list(_tokenize_v2(term)) if isinstance(term, str) else []
        valid_term = len(tokens) == 1 and _surface_form_v2(tokens[0].surface) == term
    if not valid_term:
        errors.append(f"document dispersion evidence {evidence_id} has invalid calculation label")
        return

    try:
        result = (
            document_dispersion(project, scope_id, term)
            if version == "v2"
            else _document_dispersion_result_v1(project, scope_id, term)
        )
        coverage = project.coverage_descriptor(str(provenance.get("coverage_id", "")))
    except (DocumentError, EvidenceValidationError, ScopeViolationError, ValueError) as exc:
        errors.append(f"document dispersion evidence {evidence_id} cannot be verified: {exc}")
        return

    population_count = result["population_tokens"]
    population_definition = (
        "all surface-token positions in every revision of the immutable ScopeSnapshot"
    )
    completion_criterion = (
        "every scoped canonical text was tokenized with corpus.TOKEN and counted by revision"
        if version == "v1"
        else f"every scoped canonical text was tokenized with {SURFACE_TOKENIZER_V2} and counted by revision"
    )
    selection_policy = (
        f"exact, {'NFC-normalized ' if version == 'v2' else ''}case-folded occurrence enumeration for {term!r} over every scoped revision"
    )
    expected_negative_status = (
        "ABSENCE_IN_EXHAUSTIVE_POPULATION" if result["total_occurrences"] == 0 else "NOT_APPLICABLE"
    )
    documents = [dict(row) for row in result["documents"]]
    if (
        payload.get("label")
        != f"surface occurrences of {term!r} across the scoped document population"
        or not _is_exact_count(payload.get("value"))
        or payload.get("value") != result["total_occurrences"]
        or payload.get("scope_snapshot_id") != scope_id
        or payload.get("population_definition") != population_definition
        or provenance.get("population_count") != population_count
        or provenance.get("population_tokens") != population_count
        or provenance.get("document_count") != result["document_count"]
        or provenance.get("document_frequency") != result["document_frequency"]
        or provenance.get("document_range") != result["document_range"]
        or provenance.get("documents") != documents
        or coverage.scope_snapshot_id != scope_id
        or coverage.coverage_mode.value != "EXHAUSTIVE"
        or coverage.population_definition != population_definition
        or coverage.population_count != population_count
        or coverage.inspection_count != population_count
        or coverage.selection_policy != selection_policy
        or coverage.completion_criterion != completion_criterion
        or coverage.termination_reason != "completed deterministic enumeration"
        or coverage.negative_result_status.value != expected_negative_status
        or (
            version == "v2"
            and (
                provenance.get("tokenizer") != SURFACE_TOKENIZER_V2
                or provenance.get("normalization") != SURFACE_NORMALIZATION_V2
                or provenance.get("unicode_data_version") != unicodedata.unidata_version
            )
        )
    ):
        errors.append(
            f"document dispersion evidence {evidence_id} does not match its ScopeSnapshot coverage"
        )


def _validate_window_association_statistic(
    project,
    evidence_id: str,
    scope_id: str,
    payload: dict[str, object],
    provenance: dict[str, object],
    errors: list[str],
) -> None:
    """Recompute all inputs to the versioned deterministic window-association score."""
    from .corpus import (
        SURFACE_NORMALIZATION_V2,
        SURFACE_TOKENIZER_V2,
        TOKEN,
        _association_population,
        _association_population_v1,
        _surface_form_v2,
        _tokenize_v2,
    )

    query = provenance.get("query")
    collocate = provenance.get("collocate")
    window = provenance.get("window")
    min_frequency = provenance.get("min_frequency")
    if not isinstance(query, str) or not isinstance(collocate, str):
        errors.append(
            f"window association evidence {evidence_id} has invalid calculation provenance"
        )
        return
    if (
        type(window) is not int
        or window < 1
        or window > 100
        or type(min_frequency) is not int
        or min_frequency < 1
    ):
        errors.append(
            f"window association evidence {evidence_id} has invalid calculation provenance"
        )
        return

    tokenizer = provenance.get("tokenizer")
    if tokenizer == SURFACE_TOKENIZER_V2:
        query_tokens = list(_tokenize_v2(query))
        collocate_tokens = list(_tokenize_v2(collocate))
        if (
            provenance.get("normalization") != SURFACE_NORMALIZATION_V2
            or provenance.get("unicode_data_version") != unicodedata.unidata_version
            or len(query_tokens) != 1
            or len(collocate_tokens) != 1
            or _surface_form_v2(query_tokens[0].surface) != query
            or _surface_form_v2(collocate_tokens[0].surface) != collocate
            or query == collocate
        ):
            errors.append(
                f"window association evidence {evidence_id} has invalid calculation provenance"
            )
            return
        association_population = _association_population
        completion_criterion = (
            f"every scope revision tokenized by {SURFACE_TOKENIZER_V2}; all focus positions "
            "and in-window pairs enumerated without crossing revision boundaries"
        )
    elif tokenizer == TOKEN.pattern:
        query_tokens = TOKEN.findall(query)
        collocate_tokens = TOKEN.findall(collocate)
        if (
            len(query_tokens) != 1
            or len(collocate_tokens) != 1
            or query_tokens[0].casefold() != query
            or collocate_tokens[0].casefold() != collocate
            or query == collocate
        ):
            errors.append(
                f"window association evidence {evidence_id} has invalid calculation provenance"
            )
            return
        association_population = _association_population_v1
        completion_criterion = (
            "every scope revision tokenized by corpus.TOKEN; all focus positions and "
            "in-window pairs enumerated without crossing revision boundaries"
        )
    else:
        errors.append(
            f"window association evidence {evidence_id} has invalid calculation provenance"
        )
        return
    try:
        token_counts, pair_counts, focus_frequency, population_tokens, pair_opportunities = (
            association_population(project, scope_id, query, window)
        )
        coverage = project.coverage_descriptor(str(provenance.get("coverage_id", "")))
    except (DocumentError, EvidenceValidationError, ScopeViolationError, ValueError) as exc:
        errors.append(f"window association evidence {evidence_id} cannot be verified: {exc}")
        return

    pair_frequency = pair_counts[collocate]
    token_frequency = token_counts[collocate]
    non_focus_tokens = population_tokens - focus_frequency
    if (
        pair_frequency < min_frequency
        or non_focus_tokens <= 0
        or pair_opportunities <= 0
        or token_frequency <= 0
    ):
        errors.append(f"window association evidence {evidence_id} has no reproducible score")
        return
    expected_frequency = pair_opportunities * token_frequency / non_focus_tokens
    mutual_information = math.log2(pair_frequency / expected_frequency)
    t_score = (pair_frequency - expected_frequency) / pair_frequency**0.5
    population_definition = (
        "all non-focus in-window co-occurrence opportunities for the query token "
        "in every revision of the immutable ScopeSnapshot"
    )
    if (
        payload.get("scope_snapshot_id") != scope_id
        or payload.get("population_definition") != population_definition
        or payload.get("value") != mutual_information
        or provenance.get("measure") != "mutual_information_log2"
        or provenance.get("focus_frequency") != focus_frequency
        or provenance.get("pair_frequency") != pair_frequency
        or provenance.get("token_frequency") != token_frequency
        or provenance.get("population_tokens") != population_tokens
        or provenance.get("pair_opportunities") != pair_opportunities
        or provenance.get("population_count") != pair_opportunities
        or provenance.get("expected_frequency") != expected_frequency
        or provenance.get("mutual_information") != mutual_information
        or provenance.get("t_score") != t_score
        or coverage.scope_snapshot_id != scope_id
        or coverage.coverage_mode.value != "EXHAUSTIVE"
        or coverage.population_definition != population_definition
        or coverage.population_count != pair_opportunities
        or coverage.completion_criterion != completion_criterion
        or coverage.termination_reason != "completed deterministic window-pair enumeration"
    ):
        errors.append(
            f"window association evidence {evidence_id} does not match its ScopeSnapshot coverage"
        )


def _validate_comparison_statistic_provenance(
    project,
    evidence_id: str,
    scope_id: str,
    analysis_run_id: str,
    payload: dict[str, object],
    provenance: dict[str, object],
    errors: list[str],
) -> None:
    """Tie a comparison statistic to its immutable two scopes and exact result row."""
    comparison_id = provenance.get("comparison_run_id")
    coverage_id = provenance.get("coverage_id")
    left_scope_id = provenance.get("left_scope_snapshot_id")
    right_scope_id = provenance.get("right_scope_snapshot_id")
    basis = provenance.get("comparability_basis")
    selected_row = provenance.get("selected_row")
    if (
        not isinstance(comparison_id, str)
        or not comparison_id
        or provenance.get("analysis_run_id") != analysis_run_id
        or not isinstance(coverage_id, str)
        or not isinstance(left_scope_id, str)
        or not isinstance(right_scope_id, str)
        or not isinstance(basis, str)
        or not basis.strip()
        or not isinstance(selected_row, dict)
    ):
        errors.append(f"comparison statistic evidence {evidence_id} has incomplete provenance")
        return
    try:
        comparison = project.comparison_run(comparison_id)
        if comparison["storage_kind"] != "rows-v1":
            raise EvidenceValidationError("comparison does not use normalized row storage")
        populations = project.comparison_rows_page(comparison_id, offset=0, limit=1)
        if (
            comparison["left_scope_id"] != left_scope_id
            or comparison["right_scope_id"] != right_scope_id
            or comparison["metric"] != provenance.get("comparison_metric")
            or comparison["calculation_version"] != provenance.get("calculation_version")
            or comparison["result_hash"] != provenance.get("comparison_result_hash")
        ):
            raise EvidenceValidationError("comparison run provenance does not match stored record")
        left_revisions = project.scope_revisions(left_scope_id)
        right_revisions = project.scope_revisions(right_scope_id)
        if (
            provenance.get("left_revision_ids") != left_revisions
            or provenance.get("right_revision_ids") != right_revisions
            or set(project.scope_revisions(scope_id)) != set(left_revisions) | set(right_revisions)
        ):
            raise EvidenceValidationError("comparison populations do not match the analysis scope")
        left_population = int(populations["left_population"])
        right_population = int(populations["right_population"])
        population_count = left_population + right_population
        coverage = project.coverage_descriptor(coverage_id)
        expected_definition = (
            "surface-token positions in two separate exhaustive populations: "
            f"left ScopeSnapshot {left_scope_id}; right ScopeSnapshot {right_scope_id}"
        )
        if (
            coverage.scope_snapshot_id != scope_id
            or coverage.coverage_mode.value != "EXHAUSTIVE"
            or coverage.population_definition != expected_definition
            or coverage.population_count != population_count
            or coverage.inspection_count != population_count
            or provenance.get("population_count") != population_count
            or provenance.get("left_population") != left_population
            or provenance.get("right_population") != right_population
            or payload.get("scope_snapshot_id") != scope_id
            or payload.get("population_definition") != expected_definition
        ):
            raise EvidenceValidationError(
                "comparison statistic coverage does not match both populations"
            )
        if not isinstance(selected_row.get("token"), str):
            raise EvidenceValidationError("comparison statistic selected row has no token")
        stored = project.db.execute(
            "SELECT token,rank_value,row_json FROM comparison_result_rows "
            "WHERE comparison_id=? AND token=?",
            (comparison_id, selected_row["token"]),
        ).fetchone()
        if not stored:
            raise EvidenceValidationError("comparison statistic selected row is no longer present")
        verified_row = project._validate_stored_comparison_row(stored, str(comparison["metric"]))
        if verified_row != selected_row:
            raise EvidenceValidationError(
                "comparison statistic selected row does not match storage"
            )
        value_key = "rate_difference" if comparison["metric"] == "frequency" else "log_likelihood"
        expected_value = verified_row.get(value_key)
        value = payload.get("value")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isinstance(expected_value, (int, float))
            or float(value) != float(expected_value)
        ):
            raise EvidenceValidationError("comparison statistic value does not match selected row")
    except (
        DocumentError,
        EvidenceValidationError,
        ScopeViolationError,
        ValueError,
        TypeError,
    ) as exc:
        errors.append(f"comparison statistic evidence {evidence_id} is invalid: {exc}")


def _validate_artifact_statistic_provenance(
    project,
    evidence_id: str,
    scope_id: str,
    payload: dict[str, object],
    provenance: object,
    errors: list[str],
) -> None:
    """Recompute derived-token frequencies from the complete, hash-verified artifact set."""
    if not isinstance(provenance, dict):
        errors.append(f"corpus statistic evidence {evidence_id} has invalid artifact provenance")
        return
    artifact_ids = provenance.get("artifact_ids")
    artifact_type = provenance.get("artifact_type")
    field = provenance.get("field")
    calculation_version = provenance.get("calculation_version")
    if calculation_version == "derived-token-artifact-frequency-v1":
        field_specs = {
            "normalized": ("surface_tokens", "surface-token-v1"),
            "lemma": ("parsed_tokens", "parsed-token-v1"),
            "pos": ("parsed_tokens", "parsed-token-v1"),
            "dependency_relation": ("parsed_tokens", "parsed-token-v1"),
        }
    elif calculation_version == "derived-token-artifact-frequency-v2":
        from .nlp_artifacts import PARSED_TOKEN_SCHEMA_VERSION

        field_specs = {
            "lemma": ("parsed_tokens", PARSED_TOKEN_SCHEMA_VERSION),
            "pos": ("parsed_tokens", PARSED_TOKEN_SCHEMA_VERSION),
            "dependency_relation": ("parsed_tokens", PARSED_TOKEN_SCHEMA_VERSION),
        }
    elif calculation_version == "derived-surface-token-frequency-v2":
        from .nlp_artifacts import SURFACE_TOKEN_SCHEMA_VERSION

        field_specs = {"normalized": ("surface_tokens", SURFACE_TOKEN_SCHEMA_VERSION)}
    else:
        errors.append(f"corpus statistic evidence {evidence_id} has invalid artifact provenance")
        return
    if (
        not isinstance(artifact_ids, list)
        or not artifact_ids
        or not all(isinstance(item, str) and item for item in artifact_ids)
        or len(artifact_ids) != len(set(artifact_ids))
        or not isinstance(field, str)
        or field not in field_specs
    ):
        errors.append(f"corpus statistic evidence {evidence_id} has invalid artifact provenance")
        return
    expected_artifact_type, expected_schema_version = field_specs[field]
    if (
        artifact_type != expected_artifact_type
        or provenance.get("artifact_schema_version") != expected_schema_version
        or provenance.get("query_engine") != "duckdb"
        or not _nonblank_string(provenance.get("query_value"))
    ):
        errors.append(f"corpus statistic evidence {evidence_id} has invalid artifact provenance")
        return
    try:
        import duckdb
    except ImportError:
        errors.append(
            f"corpus statistic evidence {evidence_id} cannot be verified: DuckDB is unavailable"
        )
        return
    placeholders = ",".join("?" * len(artifact_ids))
    try:
        rows = project.db.execute(
            f"SELECT id,revision_id,artifact_type,schema_version FROM artifact_registry "
            f"WHERE id IN ({placeholders})",
            artifact_ids,
        ).fetchall()
        scope_revisions = set(project.scope_revisions(scope_id))
        if (
            len(rows) != len(artifact_ids)
            or any(
                row["artifact_type"] != artifact_type
                or row["schema_version"] != expected_schema_version
                for row in rows
            )
            or {row["revision_id"] for row in rows} != scope_revisions
            or len(rows) != len(scope_revisions)
        ):
            raise EvidenceValidationError(
                "derived statistic no longer has complete in-scope artifact provenance"
            )

        from .nlp_artifacts import _column_frequency, _query_counts, _scoped_artifacts

        query_value = str(provenance["query_value"])
        if field == "normalized":
            if calculation_version == "derived-token-artifact-frequency-v1":
                if query_value != query_value.casefold():
                    raise EvidenceValidationError("surface artifact query is not normalized")
            else:
                from .corpus import _surface_form_v2

                if query_value != _surface_form_v2(query_value):
                    raise EvidenceValidationError("surface artifact query is not normalized")
        if field == "normalized":
            artifacts = _scoped_artifacts(project, scope_id, artifact_ids, expected_artifact_type)
            counts = _query_counts(
                project,
                artifacts,
                "SELECT normalized,count(*) FROM read_parquet(?) "
                "WHERE normalized IS NOT NULL AND normalized <> '' "
                "GROUP BY normalized ORDER BY normalized",
            )
        else:
            counts = _column_frequency(project, scope_id, artifact_ids, field)
        population_count = sum(counts.values())
        expected_count = counts.get(query_value, 0)
        population_definition = (
            f"all non-null, non-empty {field} values in one {artifact_type} artifact per revision "
            "of the immutable ScopeSnapshot"
        )
        is_complete_field_population = True
        if (
            calculation_version == "derived-token-artifact-frequency-v1"
            and expected_artifact_type == "parsed_tokens"
        ):
            # Preserve readable/recomputable legacy parser statistics, but do
            # not infer an exhaustive token population from v1 artifacts.
            is_complete_field_population = False
            population_definition += (
                "; parser annotation coverage is LIMITED and counts are over available annotations only"
            )
        if calculation_version == "derived-token-artifact-frequency-v2":
            from .nlp_artifacts import _scope_parsed_coverage

            parsed_artifacts = _scoped_artifacts(
                project, scope_id, artifact_ids, expected_artifact_type
            )
            coverage_status = _scope_parsed_coverage(project, parsed_artifacts)
            field_coverage = coverage_status[
                "dependency_parse" if field == "dependency_relation" else field
            ]
            tokenization_coverage = coverage_status["tokenization"]
            if (
                provenance.get("field_coverage") != field_coverage
                or provenance.get("tokenization_coverage") != tokenization_coverage
            ):
                raise EvidenceValidationError(
                    "derived statistic annotation coverage does not match its artifacts"
                )
            is_complete_field_population = (
                field_coverage["status"] == "SUPPORTED"
            )
            if not is_complete_field_population:
                population_definition += (
                    "; parser annotation coverage is LIMITED and counts are over available annotations only"
                )
        coverage = project.coverage_descriptor(str(provenance.get("coverage_id", "")))
        expected_negative_status = (
            "ABSENCE_IN_EXHAUSTIVE_POPULATION"
            if expected_count == 0 and is_complete_field_population
            else "NOT_APPLICABLE"
        )
        expected_coverage_mode = (
            "EXHAUSTIVE" if is_complete_field_population else "BROAD"
        )
        if (
            payload.get("scope_snapshot_id") != scope_id
            or payload.get("population_definition") != population_definition
            or payload.get("value") != expected_count
            or provenance.get("population_count") != population_count
            or coverage.scope_snapshot_id != scope_id
            or coverage.coverage_mode.value != expected_coverage_mode
            or coverage.population_definition != population_definition
            or coverage.population_count != population_count
            or coverage.inspection_count != population_count
            or coverage.selection_policy
            != f"exact deterministic {field} enumeration for {query_value!r} in every selected scope artifact"
            or coverage.completion_criterion
            != "all selected, complete scope artifacts read through DuckDB and enumerated"
            or coverage.termination_reason != "completed deterministic artifact enumeration"
            or coverage.negative_result_status.value != expected_negative_status
        ):
            raise EvidenceValidationError(
                "derived statistic value or coverage does not match its complete artifacts"
            )
    except (ProjectError, ValueError, TypeError, duckdb.Error) as exc:
        errors.append(f"corpus statistic evidence {evidence_id} cannot be verified: {exc}")


def _validate_claim_contract(
    project, finding, scope_id: str, analysis_run_id: str, links, errors: list[str]
) -> None:
    if (
        finding["evidence_sufficiency"] not in {"SUFFICIENT", "LIMITED", "INSUFFICIENT"}
        or finding["interpretation_confidence"] not in {"HIGH", "MEDIUM", "LOW"}
    ):
        errors.append("finding has invalid epistemic metadata")
    try:
        raw_claims = json.loads(finding["claim_types_json"])
        if not isinstance(raw_claims, list) or any(
            not isinstance(item, str) for item in raw_claims
        ):
            raise ValueError("claim types must be a JSON array of strings")
        claims = [ClaimType(item) for item in raw_claims]
    except (TypeError, ValueError, json.JSONDecodeError):
        errors.append("finding has invalid claim type metadata")
        return
    try:
        limitations = json.loads(finding["limitations_json"])
    except (TypeError, json.JSONDecodeError):
        limitations = []
        errors.append("finding has invalid limitations metadata")
    if not isinstance(limitations, list) or any(
        not isinstance(item, str) for item in limitations
    ):
        limitations = []
        errors.append("finding has invalid limitations metadata")
    has_limitations = any(item.strip() for item in limitations)
    # Import locally to keep validation usable without making the project/skill
    # modules depend on one another during initialization.
    from .skills import BUILTIN_SKILLS

    try:
        allowed_claims = BUILTIN_SKILLS.allowed_claims_for(
            finding["skill_id"], finding["skill_version"]
        )
    except SkillManifestError as exc:
        errors.append(str(exc))
        allowed_claims = None
    if allowed_claims is not None:
        unsupported_claims = set(claims) - allowed_claims
        if unsupported_claims:
            names = ", ".join(sorted(claim.value for claim in unsupported_claims))
            errors.append(
                f"built-in skill {finding['skill_id']!r} does not allow claim type(s): {names}"
            )
    text_supported_claims = {ClaimType.OBSERVATIONAL, ClaimType.EXAMPLE_BASED}
    if text_supported_claims.intersection(claims) and not _has_linked_evidence(
        links,
        kinds={"TEXTUAL"},
        roles={"SUPPORT", "EXAMPLE"},
    ):
        errors.append("observational and example-based claims require supporting textual evidence")
    if ClaimType.INTERPRETIVE in claims and not _has_linked_evidence(links, kinds={"TEXTUAL"}):
        errors.append("interpretive claims require linked textual evidence")
    if ClaimType.CONTEXTUAL in claims and not _has_linked_evidence(
        links,
        kinds={"EXTERNAL_SOURCE", "USER_CONTEXT"},
        roles={"CONTEXT", "SUPPORT"},
    ):
        errors.append(
            "contextual claims require external-source or explicitly labeled user-context evidence"
        )
    coverage_id = finding["coverage_id"]
    coverage = None
    if coverage_id:
        try:
            coverage = project.coverage_descriptor(coverage_id)
        except EvidenceValidationError as exc:
            errors.append(str(exc))
        else:
            if coverage.scope_snapshot_id != scope_id:
                errors.append("coverage descriptor is outside the analysis scope")
    requires_coverage = {
        ClaimType.ABSENCE,
        ClaimType.UNIVERSAL,
        ClaimType.QUANTIFIED,
        ClaimType.INTERPRETIVE,
        ClaimType.COMPARATIVE,
    }
    if any(claim in requires_coverage for claim in claims) and coverage is None:
        errors.append("claim type requires a coverage descriptor")
        return
    if coverage is None:
        return
    if ClaimType.ABSENCE in claims and not _has_zero_statistic_for_coverage(
        project, links, scope_id, coverage_id, coverage
    ):
        errors.append(
            "absence claim requires a zero-valued corpus statistic linked to the same scope and coverage"
        )
    if ClaimType.UNIVERSAL in claims and (
        coverage.coverage_mode is not CoverageMode.EXHAUSTIVE
        or coverage.population_count is None
        or coverage.population_count != len(project.scope_revisions(scope_id))
        or coverage.inspection_count != coverage.population_count
        or not is_scope_revision_population_definition(coverage.population_definition)
    ):
        errors.append(
            "universal claim requires one population item per scoped revision and complete inspection"
        )
    if ClaimType.UNIVERSAL in claims and not _has_supporting_text_in_each_revision(
        project, links, scope_id
    ):
        errors.append("universal claim requires supporting textual evidence in every scoped revision")
    if ClaimType.UNIVERSAL in claims and not (
        _has_linked_evidence(links, kinds={"TEXTUAL"}, roles={"SUPPORT"})
        or _has_quantified_statistic_for_coverage(project, links, scope_id, coverage_id, coverage)
    ):
        errors.append("universal claim requires supporting evidence for its declared population")
    has_interpretation = isinstance(finding["interpretation"], str) and bool(
        finding["interpretation"].strip()
    )
    has_alternative_reading = isinstance(finding["alternative_interpretation"], str) and bool(
        finding["alternative_interpretation"].strip()
    )
    population_defined = bool(
        coverage.population_definition and coverage.population_count is not None
    )
    quantified_statistic = (
        _has_quantified_statistic_for_coverage(project, links, scope_id, coverage_id, coverage)
        if coverage_id and coverage
        else False
    )
    if ClaimType.QUANTIFIED in claims and not quantified_statistic:
        errors.append(
            "exact quantified claim requires deterministic statistic evidence tied to the same coverage and population"
        )
    comparative_statistic = (
        _has_comparison_statistic_for_coverage(
            project, links, analysis_run_id, scope_id, coverage_id
        )
        if coverage_id
        else False
    )
    if ClaimType.COMPARATIVE in claims and not comparative_statistic:
        errors.append(
            "comparative claim requires linked comparison evidence for two explicit populations"
        )
    for claim in claims:
        if not claim_allowed(
            claim,
            coverage.coverage_mode,
            statistical_evidence=quantified_statistic,
            population_defined=population_defined,
            limitations=has_limitations,
            interpretation=has_interpretation,
            alternative_reading=has_alternative_reading,
            comparable_populations=comparative_statistic,
            negative_result_status=coverage.negative_result_status,
        ):
            errors.append(f"claim type {claim} is not supported by persisted coverage and evidence")


def require_valid_finding(project, finding_id: str) -> None:
    result = validate_finding(project, finding_id)
    if not result.valid:
        raise EvidenceValidationError("; ".join(result.errors))
