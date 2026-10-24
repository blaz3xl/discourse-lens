"""Deterministic, scope-bound corpus operations without NLP dependencies."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from heapq import nsmallest
from itertools import chain, islice
from math import log, log2
from sys import maxsize
from typing import TypedDict, TypeGuard, cast

from .errors import DocumentError, EvidenceValidationError, ScopeViolationError
from .types import (
    ClaimType,
    CoverageDescriptor,
    CoverageMode,
    CoverageTerminationState,
    EvidenceRole,
    NegativeResultStatus,
)

COMPARISON_CALCULATION_VERSION = "surface-comparison-v2"

TOKEN = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
SURFACE_TOKENIZER_V2 = "unicode-surface-tokenizer-v2"
SURFACE_NORMALIZATION_V2 = "NFC-after-casefold-v1"
_V2_TOKEN_CONTINUATION = frozenset({"’", "'", "-"})
_MAX_KWIC_WINDOW = maxsize - 1


@dataclass(frozen=True, slots=True)
class _SurfaceToken:
    """A token span in the original Python string, with half-open codepoint offsets."""

    surface: str
    start_offset: int
    end_offset: int


def _is_word_character(character: str) -> bool:
    r"""Mirror Unicode ``\w``'s alphanumeric-plus-underscore core for tokenizer v2."""
    return character == "_" or character.isalnum()


def _tokenize_v2(text: str) -> Iterator[_SurfaceToken]:
    """Tokenize Unicode words plus combining marks without normalizing source offsets.

    Unicode alphanumeric and mark categories are interpreted using the active Python
    Unicode database; persisted v2 provenance records its version for revalidation.
    Connector punctuation remains internal, matching the v1 surface-token convention.
    """
    yield from _tokenize_v2_chunks((text,))


def _tokenize_v2_chunks(chunks: Iterable[str]) -> Iterator[_SurfaceToken]:
    """Tokenize a stream of text chunks with the exact v2 surface contract.

    Keeping this state machine shared with the Parquet builder prevents artifact
    token boundaries and normalized forms from drifting from in-memory corpus
    operations. Offsets count Python Unicode code points in the original text.
    """
    offset = 0
    token_start: int | None = None
    token_characters: list[str] = []
    pending_connectors: list[str] = []
    token_end = 0

    for chunk in chunks:
        for character in chunk:
            if token_start is None:
                if _is_word_character(character):
                    token_start = offset
                    token_characters = [character]
                    pending_connectors = []
                    token_end = offset + 1
            elif _is_word_character(character) or unicodedata.category(character).startswith("M"):
                token_characters.extend(pending_connectors)
                pending_connectors = []
                token_characters.append(character)
                token_end = offset + 1
            elif character in _V2_TOKEN_CONTINUATION:
                pending_connectors.append(character)
            else:
                yield _SurfaceToken("".join(token_characters), token_start, token_end)
                token_start = None
                token_characters = []
                pending_connectors = []
            offset += 1

    if token_start is not None:
        yield _SurfaceToken("".join(token_characters), token_start, token_end)


def _surface_form_v2(value: str) -> str:
    """Return a canonically normalized, case-folded v2 surface form."""
    return unicodedata.normalize("NFC", value.casefold())


def _is_plain_int(value: object) -> TypeGuard[int]:
    """Reject bool and non-integer numeric values at integer API boundaries."""
    return type(value) is int


def _valid_comparison_pagination(offset: object, limit: object) -> bool:
    return _is_plain_int(offset) and _is_plain_int(limit) and offset >= 0 and 1 <= limit <= 1_000


def _validate_comparison_metric(metric: str) -> None:
    if metric not in {"frequency", "keyness"}:
        raise ScopeViolationError("corpus comparison metric must be frequency or keyness")


class DocumentDispersionRow(TypedDict):
    revision_id: str
    occurrences: int
    token_count: int
    per_1000_tokens: float


class FrequencyPageRow(TypedDict):
    token: str
    count: int


class FrequencyPage(TypedDict):
    scope_id: str
    offset: int
    limit: int
    total_types: int
    total_tokens: int
    has_more: bool
    rows: list[FrequencyPageRow]


class NgramPageRow(TypedDict):
    ngram: str
    count: int


class NgramPage(TypedDict):
    scope_id: str
    size: int
    offset: int
    limit: int
    rows: list[NgramPageRow]
    total_values: int
    has_more: bool


class CollocationPageRow(TypedDict):
    term: str
    count: int


class CollocationsPage(TypedDict):
    scope_id: str
    query: str
    window: int
    offset: int
    limit: int
    rows: list[CollocationPageRow]
    total_values: int
    has_more: bool


class DocumentDispersionResult(TypedDict):
    term: str
    scope_id: str
    calculation_version: str
    document_count: int
    document_frequency: int
    document_range: float
    total_occurrences: int
    population_tokens: int
    documents: list[DocumentDispersionRow]


class DocumentDispersionPage(TypedDict):
    term: str
    scope_id: str
    calculation_version: str
    document_count: int
    document_frequency: int
    document_range: float
    total_occurrences: int
    population_tokens: int
    offset: int
    limit: int
    total_rows: int
    has_more: bool
    documents: list[DocumentDispersionRow]


class DocumentDispersionEvidenceResult(DocumentDispersionResult):
    coverage_id: str
    evidence_id: str


class LexicalBundleRow(TypedDict):
    bundle: str
    frequency: int
    document_frequency: int


class LexicalBundlesResult(TypedDict):
    scope_id: str
    calculation_version: str
    size: int
    min_frequency: int
    min_document_frequency: int
    offset: int
    limit: int
    candidate_count: int
    rows: list[LexicalBundleRow]
    has_more: bool
    truncated: bool


class LexicalBundleEvidenceResult(TypedDict):
    bundle: str
    size: int
    count: int
    population_count: int
    scope_id: str
    coverage_id: str
    evidence_id: str


class KwicRow(TypedDict):
    """One scope-bound KWIC occurrence addressable in canonical revision text."""

    revision_id: str
    start_offset: int
    end_offset: int
    left: str
    keyword: str
    right: str


class KwicPage(TypedDict):
    scope_id: str
    query: str
    window: int
    offset: int
    limit: int
    total_count: int
    has_more: bool
    rows: list[KwicRow]


class KwicEvidenceResult(TypedDict):
    scope_id: str
    revision_id: str
    query: str
    start_offset: int
    end_offset: int
    exact_text: str
    coverage_id: str
    evidence_id: str
    citation: str


class ComparisonPage(TypedDict):
    metric: str
    offset: int
    limit: int
    total_rows: int
    has_more: bool
    rows: list[dict[str, float | int | str]]


class CollocationAssociationRow(TypedDict):
    term: str
    pair_frequency: int
    token_frequency: int
    expected_frequency: float
    mutual_information: float
    t_score: float


class CollocationAnalysisResult(TypedDict):
    scope_id: str
    calculation_version: str
    tokenizer: str
    normalization: str
    unicode_data_version: str
    query: str
    window: int
    focus_frequency: int
    population_tokens: int
    pair_opportunities: int
    min_frequency: int
    offset: int
    limit: int
    total: int
    has_more: bool
    rows: list[CollocationAssociationRow]
    truncated: bool
    interpretation_note: str


class CollocationEvidenceResult(TypedDict):
    scope_id: str
    query: str
    collocate: str
    window: int
    focus_frequency: int
    pair_frequency: int
    token_frequency: int
    population_tokens: int
    pair_opportunities: int
    expected_frequency: float
    mutual_information: float
    t_score: float
    coverage_id: str
    evidence_id: str


def _texts(project, scope_id: str) -> Iterator[tuple[str, str]]:
    row = project.db.execute(
        "SELECT revision_ids_json FROM scope_snapshots WHERE id=?", (scope_id,)
    ).fetchone()
    if not row:
        raise ScopeViolationError("scope is required")
    import json

    for revision_id in json.loads(row[0]):
        yield revision_id, project.revision_text(revision_id)


def frequency(project, scope_id: str) -> Counter[str]:
    """Count NFC-normalized, case-folded Unicode surface tokens in one scope."""
    counts: Counter[str] = Counter()
    for _, text in _texts(project, scope_id):
        counts.update(_surface_form_v2(token.surface) for token in _tokenize_v2(text))
    return counts


def _frequency_v1(project, scope_id: str) -> Counter[str]:
    """Recompute the historical v1 surface frequency without changing its tokenizer."""
    counts: Counter[str] = Counter()
    for _, text in _texts(project, scope_id):
        counts.update(match.group(0).casefold() for match in TOKEN.finditer(text))
    return counts


def frequency_page(project, scope_id: str, *, offset: int = 0, limit: int = 100) -> FrequencyPage:
    """Return a stable bounded frequency ranking and metadata for explicit paging."""
    if (
        not _is_plain_int(offset)
        or not _is_plain_int(limit)
        or offset < 0
        or limit < 1
        or limit > 1_000
    ):
        raise ScopeViolationError("frequency offset must be non-negative and limit must be 1..1000")
    counts = frequency(project, scope_id)
    rows: list[FrequencyPageRow] = []
    if offset < len(counts):
        selected = nsmallest(
            offset + limit,
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        rows = [{"token": token, "count": count} for token, count in selected[offset:]]
    return {
        "scope_id": scope_id,
        "offset": offset,
        "limit": limit,
        "total_types": len(counts),
        "total_tokens": sum(counts.values()),
        "has_more": offset + len(rows) < len(counts),
        "rows": rows,
    }


def frequency_statistic_evidence(project, scope_id: str, term: str) -> dict[str, object]:
    """Persist one exact surface-token frequency with exhaustive provenance.

    This is deliberately a deterministic corpus result, not an interpretation.
    It preserves the immutable scope, population and completion criterion needed
    if a later Finding makes a quantified or bounded-absence claim from it.
    """
    tokens = list(_tokenize_v2(term))
    if len(tokens) != 1:
        raise ScopeViolationError("frequency statistic requires exactly one surface token")
    normalized = _surface_form_v2(tokens[0].surface)
    counts = frequency(project, scope_id)
    population_count = sum(counts.values())
    if population_count == 0:
        raise ScopeViolationError("frequency statistic requires a non-empty token population")
    population_definition = "all surface-token positions in the immutable ScopeSnapshot"
    exact_count = counts[normalized]
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope_id,
            population_definition=population_definition,
            population_count=population_count,
            selection_policy=(
                f"case-folded exact surface-token count for {normalized!r} over every scoped revision"
            ),
            inspection_count=population_count,
            completion_criterion=(
                f"all scope texts tokenized by {SURFACE_TOKENIZER_V2} and enumerated"
            ),
            termination_reason="completed deterministic enumeration",
            negative_result_status=(
                NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION
                if exact_count == 0
                else NegativeResultStatus.NOT_APPLICABLE
            ),
        )
    )
    evidence_id = project.corpus_statistic_evidence(
        f"surface frequency for {normalized!r}",
        exact_count,
        scope_snapshot_id=scope_id,
        population_definition=population_definition,
        provenance={
            "calculation_version": "surface-frequency-v2",
            "coverage_id": coverage_id,
            "population_count": population_count,
            "term": normalized,
            "population_tokens": population_count,
            "tokenizer": SURFACE_TOKENIZER_V2,
            "normalization": SURFACE_NORMALIZATION_V2,
            "unicode_data_version": unicodedata.unidata_version,
        },
    )
    return {
        "term": normalized,
        "count": counts[normalized],
        "population_count": population_count,
        "scope_id": scope_id,
        "coverage_id": coverage_id,
        "evidence_id": evidence_id,
    }


def _document_dispersion_result(
    project,
    scope_id: str,
    term: str,
    *,
    row_offset: int = 0,
    row_limit: int | None = None,
) -> DocumentDispersionResult:
    term_tokens = list(_tokenize_v2(term))
    if len(term_tokens) != 1:
        raise ScopeViolationError("document dispersion requires exactly one surface token")
    normalized = _surface_form_v2(term_tokens[0].surface)
    documents: list[DocumentDispersionRow] = []
    total_occurrences = 0
    population_tokens = 0
    document_frequency = 0
    document_count = 0
    for revision_id, text in _texts(project, scope_id):
        tokens_in_document = [_surface_form_v2(item.surface) for item in _tokenize_v2(text)]
        occurrence_count = sum(item == normalized for item in tokens_in_document)
        token_count = len(tokens_in_document)
        total_occurrences += occurrence_count
        population_tokens += token_count
        document_frequency += occurrence_count > 0
        if document_count >= row_offset and (row_limit is None or len(documents) < row_limit):
            documents.append(
                DocumentDispersionRow(
                    revision_id=revision_id,
                    occurrences=occurrence_count,
                    token_count=token_count,
                    per_1000_tokens=occurrence_count / token_count * 1000 if token_count else 0.0,
                )
            )
        document_count += 1
    if not document_count:
        raise ScopeViolationError("document dispersion requires a non-empty explicit scope")
    if population_tokens == 0:
        raise ScopeViolationError("document dispersion requires a non-empty token population")
    return {
        "term": normalized,
        "scope_id": scope_id,
        "calculation_version": "surface-document-dispersion-v2",
        "document_count": document_count,
        "document_frequency": document_frequency,
        "document_range": document_frequency / document_count,
        "total_occurrences": total_occurrences,
        "population_tokens": population_tokens,
        "documents": documents,
    }


def _document_dispersion_result_v1(project, scope_id: str, term: str) -> DocumentDispersionResult:
    """Recompute the historical v1 statistic solely for evidence validation."""
    tokens = TOKEN.findall(term)
    if len(tokens) != 1:
        raise ScopeViolationError("document dispersion requires exactly one surface token")
    normalized = tokens[0].casefold()
    documents: list[DocumentDispersionRow] = []
    total_occurrences = population_tokens = document_frequency = document_count = 0
    for revision_id, text in _texts(project, scope_id):
        tokens_in_document = [item.casefold() for item in TOKEN.findall(text)]
        occurrence_count = sum(item == normalized for item in tokens_in_document)
        token_count = len(tokens_in_document)
        total_occurrences += occurrence_count
        population_tokens += token_count
        document_frequency += occurrence_count > 0
        documents.append(
            DocumentDispersionRow(
                revision_id=revision_id,
                occurrences=occurrence_count,
                token_count=token_count,
                per_1000_tokens=occurrence_count / token_count * 1000 if token_count else 0.0,
            )
        )
        document_count += 1
    if not document_count:
        raise ScopeViolationError("document dispersion requires a non-empty explicit scope")
    if not population_tokens:
        raise ScopeViolationError("document dispersion requires a non-empty token population")
    return {
        "term": normalized,
        "scope_id": scope_id,
        "calculation_version": "surface-document-dispersion-v1",
        "document_count": document_count,
        "document_frequency": document_frequency,
        "document_range": document_frequency / document_count,
        "total_occurrences": total_occurrences,
        "population_tokens": population_tokens,
        "documents": documents,
    }


def document_dispersion(project, scope_id: str, term: str) -> DocumentDispersionResult:
    """Describe a surface term's distribution across every revision in a scope.

    This reports exact document counts/rates and the document range (DF/N).
    It intentionally does not collapse dispersion into a single inferential or
    normalized index whose assumptions would be easy to miss.
    """
    return _document_dispersion_result(project, scope_id, term)


def document_dispersion_page(
    project, scope_id: str, term: str, *, offset: int = 0, limit: int = 100
) -> DocumentDispersionPage:
    """Return exhaustive dispersion totals with a bounded, ordered document page."""
    if (
        not _is_plain_int(offset)
        or not _is_plain_int(limit)
        or offset < 0
        or limit < 1
        or limit > 1_000
    ):
        raise ScopeViolationError(
            "dispersion offset must be non-negative and limit must be 1..1000"
        )
    result = _document_dispersion_result(
        project, scope_id, term, row_offset=offset, row_limit=limit
    )
    total_rows = result["document_count"]
    return {
        "term": result["term"],
        "scope_id": result["scope_id"],
        "calculation_version": result["calculation_version"],
        "document_count": result["document_count"],
        "document_frequency": result["document_frequency"],
        "document_range": result["document_range"],
        "total_occurrences": result["total_occurrences"],
        "population_tokens": result["population_tokens"],
        "offset": offset,
        "limit": limit,
        "total_rows": total_rows,
        "has_more": offset + len(result["documents"]) < total_rows,
        "documents": result["documents"],
    }


def document_dispersion_evidence(
    project, scope_id: str, term: str
) -> DocumentDispersionEvidenceResult:
    """Persist an exhaustive citable statistic with the complete per-document result."""
    result = document_dispersion(project, scope_id, term)
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope_id,
            population_definition=(
                "all surface-token positions in every revision of the immutable ScopeSnapshot"
            ),
            population_count=result["population_tokens"],
            selection_policy=(
                f"exact, NFC-normalized case-folded occurrence enumeration for {result['term']!r} over every scoped revision"
            ),
            inspection_count=result["population_tokens"],
            completion_criterion=(
                f"every scoped canonical text was tokenized with {SURFACE_TOKENIZER_V2} and counted by revision"
            ),
            termination_reason="completed deterministic enumeration",
            negative_result_status=(
                NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION
                if result["total_occurrences"] == 0
                else NegativeResultStatus.NOT_APPLICABLE
            ),
        )
    )
    evidence_id = project.corpus_statistic_evidence(
        f"surface occurrences of {result['term']!r} across the scoped document population",
        result["total_occurrences"],
        scope_snapshot_id=scope_id,
        population_definition=(
            "all surface-token positions in every revision of the immutable ScopeSnapshot"
        ),
        provenance={
            "calculation_version": result["calculation_version"],
            "coverage_id": coverage_id,
            "document_count": result["document_count"],
            "document_frequency": result["document_frequency"],
            "document_range": result["document_range"],
            "population_count": result["population_tokens"],
            "population_tokens": result["population_tokens"],
            "documents": result["documents"],
            "tokenizer": SURFACE_TOKENIZER_V2,
            "normalization": SURFACE_NORMALIZATION_V2,
            "unicode_data_version": unicodedata.unidata_version,
        },
    )
    return {**result, "coverage_id": coverage_id, "evidence_id": evidence_id}


def ngrams(project, scope_id: str, size: int) -> Counter[tuple[str, ...]]:
    """Count n-grams with the versioned Unicode surface tokenizer v2."""
    if not _is_plain_int(size) or size < 1:
        raise ValueError("n-gram size must be positive")
    result: Counter[tuple[str, ...]] = Counter()
    for _, text in _texts(project, scope_id):
        terms = [_surface_form_v2(item.surface) for item in _tokenize_v2(text)]
        result.update(tuple(terms[i : i + size]) for i in range(len(terms) - size + 1))
    return result


def ngram_page(
    project, scope_id: str, size: int, *, offset: int = 0, limit: int = 100
) -> NgramPage:
    """Return one stable, bounded frequency page of surface n-grams."""
    if (
        not _is_plain_int(offset)
        or not _is_plain_int(limit)
        or offset < 0
        or limit < 1
        or limit > 1_000
    ):
        raise ScopeViolationError("n-gram offset must be non-negative and limit must be 1..1000")
    counts = ngrams(project, scope_id, size)
    selected = (
        nsmallest(
            offset + limit,
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if offset < len(counts)
        else []
    )
    rows: list[NgramPageRow] = [
        {"ngram": " ".join(terms), "count": count} for terms, count in selected[offset:]
    ]
    return {
        "scope_id": scope_id,
        "size": size,
        "offset": offset,
        "limit": limit,
        "rows": rows,
        "total_values": len(counts),
        "has_more": offset + len(rows) < len(counts),
    }


def lexical_bundles(
    project,
    scope_id: str,
    *,
    size: int = 3,
    min_frequency: int = 2,
    min_document_frequency: int = 1,
    offset: int = 0,
    limit: int = 100,
) -> LexicalBundlesResult:
    """List frequent surface n-grams with occurrence and revision dispersion.

    This deterministic descriptive operation enumerates every canonical text in
    the required immutable scope. It does not imply a discourse function.
    """
    if not _is_plain_int(size) or not 2 <= size <= 10:
        raise ScopeViolationError("lexical bundle size must be between 2 and 10")
    if not _is_plain_int(min_frequency) or min_frequency < 1:
        raise ScopeViolationError("minimum bundle frequency must be positive")
    if not _is_plain_int(min_document_frequency) or min_document_frequency < 1:
        raise ScopeViolationError("minimum document frequency must be positive")
    if not _is_plain_int(offset) or offset < 0:
        raise ScopeViolationError("lexical bundle offset must be non-negative")
    if not _is_plain_int(limit) or not 1 <= limit <= 1_000:
        raise ScopeViolationError("lexical bundle limit must be between 1 and 1000")

    frequencies: Counter[tuple[str, ...]] = Counter()
    document_frequencies: Counter[tuple[str, ...]] = Counter()
    for _, text in _texts(project, scope_id):
        terms = [item.casefold() for item in TOKEN.findall(text)]
        document_bundles = {
            tuple(terms[index : index + size]) for index in range(len(terms) - size + 1)
        }
        frequencies.update(_document_bundle_occurrences(terms, size))
        document_frequencies.update(document_bundles)

    candidate_count = sum(
        1
        for bundle, count in frequencies.items()
        if count >= min_frequency and document_frequencies[bundle] >= min_document_frequency
    )
    selected = (
        nsmallest(
            offset + limit,
            (
                (bundle, count, document_frequencies[bundle])
                for bundle, count in frequencies.items()
                if count >= min_frequency
                and document_frequencies[bundle] >= min_document_frequency
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if offset < candidate_count
        else []
    )
    rows: list[LexicalBundleRow] = [
        {
            "bundle": " ".join(bundle),
            "frequency": count,
            "document_frequency": document_frequency,
        }
        for bundle, count, document_frequency in selected[offset:]
    ]
    has_more = offset + len(rows) < candidate_count
    return {
        "scope_id": scope_id,
        "calculation_version": "surface-lexical-bundles-v1",
        "size": size,
        "min_frequency": min_frequency,
        "min_document_frequency": min_document_frequency,
        "offset": offset,
        "limit": limit,
        "candidate_count": candidate_count,
        "rows": rows,
        "has_more": has_more,
        "truncated": has_more,
    }


def lexical_bundle_statistic_evidence(
    project, scope_id: str, bundle: str, *, size: int
) -> LexicalBundleEvidenceResult:
    """Persist an exact exhaustive surface-bundle count for later citable Findings."""
    if not _is_plain_int(size) or not 2 <= size <= 10:
        raise ScopeViolationError("lexical bundle size must be between 2 and 10")
    terms = [item.casefold() for item in TOKEN.findall(bundle)]
    if len(terms) != size:
        raise ScopeViolationError(f"bundle must contain exactly {size} surface tokens")
    normalized_bundle = " ".join(terms)
    counts: Counter[tuple[str, ...]] = Counter()
    population_count = 0
    for _, text in _texts(project, scope_id):
        document_terms = [item.casefold() for item in TOKEN.findall(text)]
        positions = len(document_terms) - size + 1
        if positions > 0:
            population_count += positions
            counts.update(tuple(document_terms[index : index + size]) for index in range(positions))
    if population_count == 0:
        raise ScopeViolationError("bundle evidence requires a non-empty n-gram population")
    coverage = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope_id,
            population_definition=(
                f"all contiguous {size}-token surface n-gram positions in the immutable ScopeSnapshot"
            ),
            population_count=population_count,
            selection_policy=(
                f"exact exhaustive surface n-gram count for {normalized_bundle!r} at size {size}"
            ),
            inspection_count=population_count,
            completion_criterion=(
                "all scope texts tokenized by corpus.TOKEN and every n-gram position enumerated"
            ),
            termination_reason="completed deterministic enumeration",
            negative_result_status=(
                NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION
                if counts[tuple(terms)] == 0
                else NegativeResultStatus.NOT_APPLICABLE
            ),
        )
    )
    evidence = project.corpus_statistic_evidence(
        f"surface lexical bundle {normalized_bundle!r} frequency",
        counts[tuple(terms)],
        scope_snapshot_id=scope_id,
        population_definition=(
            f"all contiguous {size}-token surface n-gram positions in the immutable ScopeSnapshot"
        ),
        provenance={
            "calculation_version": f"surface-lexical-bundles-v1-size-{size}",
            "coverage_id": coverage,
            "bundle": normalized_bundle,
            "population_count": population_count,
            "size": size,
            "tokenizer": TOKEN.pattern,
        },
    )
    return {
        "bundle": normalized_bundle,
        "size": size,
        "count": counts[tuple(terms)],
        "population_count": population_count,
        "scope_id": scope_id,
        "coverage_id": coverage,
        "evidence_id": evidence,
    }


def _document_bundle_occurrences(terms: list[str], size: int) -> Counter[tuple[str, ...]]:
    """Count overlapping surface bundles within one revision."""
    return Counter(tuple(terms[index : index + size]) for index in range(len(terms) - size + 1))


def _single_surface_query_v2(query: str, operation: str) -> str:
    tokens = list(_tokenize_v2(query))
    if len(tokens) != 1:
        raise ScopeViolationError(f"{operation} requires exactly one surface token")
    return _surface_form_v2(tokens[0].surface)


def _iter_kwic_rows(project, scope_id: str, query: str, window: int) -> Iterator[KwicRow]:
    if not _is_plain_int(window) or not 0 <= window <= _MAX_KWIC_WINDOW:
        raise ValueError("window must be a non-negative integer within platform bounds")
    needle = _single_surface_query_v2(query, "KWIC")
    for revision_id, text in _texts(project, scope_id):
        matches = iter(_tokenize_v2(text))
        pending = deque(islice(matches, window + 1))
        left: deque[_SurfaceToken] = deque(maxlen=window)
        while pending:
            match = pending[0]
            term = match.surface
            if _surface_form_v2(term) == needle:
                yield {
                    "revision_id": revision_id,
                    "start_offset": match.start_offset,
                    "end_offset": match.end_offset,
                    "left": " ".join(item.surface for item in left),
                    "keyword": term,
                    "right": " ".join(item.surface for item in islice(pending, 1, window + 1)),
                }
            left.append(match)
            pending.popleft()
            following = next(matches, None)
            if following is not None:
                pending.append(following)


def kwic(project, scope_id: str, query: str, window: int = 5) -> list[KwicRow]:
    """Return all KWIC rows for explicit exhaustive programmatic enumeration."""
    return list(_iter_kwic_rows(project, scope_id, query, window))


def kwic_page(
    project,
    scope_id: str,
    query: str,
    *,
    window: int = 5,
    offset: int = 0,
    limit: int = 100,
) -> KwicPage:
    """Return a bounded deterministic page while counting the full scoped result set."""
    if (
        not _is_plain_int(offset)
        or not _is_plain_int(limit)
        or offset < 0
        or limit < 1
        or limit > 1_000
    ):
        raise ScopeViolationError("KWIC offset must be non-negative and limit must be 1..1000")
    rows: list[KwicRow] = []
    total_count = 0
    page_end = offset + limit
    for row in _iter_kwic_rows(project, scope_id, query, window):
        if offset <= total_count < page_end:
            rows.append(row)
        total_count += 1
    return {
        "scope_id": scope_id,
        "query": query,
        "window": window,
        "offset": offset,
        "limit": limit,
        "total_count": total_count,
        "has_more": page_end < total_count,
        "rows": rows,
    }


def kwic_evidence(
    project,
    scope_id: str,
    revision_id: str,
    query: str,
    start_offset: int,
    end_offset: int,
) -> KwicEvidenceResult:
    """Persist textual evidence only for a matching KWIC hit inside the scope."""
    query_tokens = list(_tokenize_v2(query))
    if len(query_tokens) != 1 or revision_id not in project.scope_revisions(scope_id):
        raise EvidenceValidationError("selected offsets are not a KWIC hit in this scope")
    text = project.revision_text(revision_id)
    if (
        not _is_plain_int(start_offset)
        or not _is_plain_int(end_offset)
        or start_offset < 0
        or end_offset <= start_offset
        or end_offset > len(text)
    ):
        raise EvidenceValidationError("selected offsets are not a KWIC hit in this scope")
    exact_text = text[start_offset:end_offset]
    exact_tokens = list(_tokenize_v2(exact_text))
    normalized_query = _surface_form_v2(query_tokens[0].surface)
    if (
        len(exact_tokens) != 1
        or exact_tokens[0].start_offset != 0
        or exact_tokens[0].end_offset != len(exact_text)
        or _surface_form_v2(exact_tokens[0].surface) != normalized_query
    ):
        raise EvidenceValidationError("selected offsets are not a KWIC hit in this scope")
    token_population = 0
    candidate_count = 0
    selected_token_hit = False
    for scoped_revision_id, scoped_text in _texts(project, scope_id):
        for match in _tokenize_v2(scoped_text):
            if (
                scoped_revision_id == revision_id
                and match.start_offset == start_offset
                and match.end_offset == end_offset
            ):
                selected_token_hit = True
            token_population += 1
            candidate_count += _surface_form_v2(match.surface) == normalized_query
    if not selected_token_hit:
        raise EvidenceValidationError("selected offsets are not a KWIC hit in this scope")
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope_id,
            population_definition="all surface-token positions in the immutable ScopeSnapshot",
            population_count=token_population,
            inspection_count=token_population,
            completion_criterion=(
                f"all scope texts tokenized by {SURFACE_TOKENIZER_V2} and enumerated"
            ),
            termination_reason="completed deterministic exact-token enumeration",
        )
    )
    span_id = project.create_span(revision_id, start_offset, end_offset)
    evidence_id = project.textual_evidence(
        span_id,
        provenance={
            "acquisition_method": "corpus_kwic",
            "query": query,
            "scope_snapshot_id": scope_id,
            "coverage_descriptor_id": coverage_id,
            "coverage_mode": CoverageMode.EXHAUSTIVE.value,
            "candidate_count": candidate_count,
            "inspected_count": token_population,
            "tokenizer": SURFACE_TOKENIZER_V2,
            "normalization": SURFACE_NORMALIZATION_V2,
            "unicode_data_version": unicodedata.unidata_version,
            "calculation_version": "surface-kwic-v2",
        },
    )
    return {
        "scope_id": scope_id,
        "revision_id": revision_id,
        "query": query,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "exact_text": project.span_text(span_id),
        "coverage_id": coverage_id,
        "evidence_id": evidence_id,
        "citation": project.render_citation(evidence_id),
    }


def collocations(project, scope_id: str, query: str, window: int = 5) -> Counter[str]:
    """Count v2 surface-token neighbors around one query token in a scope."""
    if not _is_plain_int(window) or not 1 <= window <= 100:
        raise ScopeViolationError("collocation window must be between 1 and 100")
    result: Counter[str] = Counter()
    needle = _single_surface_query_v2(query, "collocations")
    for _, text in _texts(project, scope_id):
        terms = [_surface_form_v2(item.surface) for item in _tokenize_v2(text)]
        for index, term in enumerate(terms):
            if term == needle:
                result.update(terms[max(0, index - window) : index])
                result.update(terms[index + 1 : index + 1 + window])
    result.pop(needle, None)
    return result


def collocations_page(
    project,
    scope_id: str,
    query: str,
    *,
    window: int = 5,
    offset: int = 0,
    limit: int = 100,
) -> CollocationsPage:
    """Return a stable, bounded page of scoped window collocate counts."""
    if (
        not _is_plain_int(offset)
        or not _is_plain_int(limit)
        or offset < 0
        or limit < 1
        or limit > 1_000
    ):
        raise ScopeViolationError(
            "collocation offset must be non-negative and limit must be 1..1000"
        )
    counts = collocations(project, scope_id, query, window)
    selected = (
        nsmallest(offset + limit, counts.items(), key=lambda item: (-item[1], item[0]))
        if offset < len(counts)
        else []
    )
    rows: list[CollocationPageRow] = [
        {"term": term, "count": count} for term, count in selected[offset:]
    ]
    return {
        "scope_id": scope_id,
        "query": query,
        "window": window,
        "offset": offset,
        "limit": limit,
        "rows": rows,
        "total_values": len(counts),
        "has_more": offset + len(rows) < len(counts),
    }


def _association_population_v1(project, scope_id: str, needle: str, window: int):
    """Recompute association inputs with the historical regex tokenizer."""
    token_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    focus_frequency = 0
    population_tokens = 0
    pair_opportunities = 0
    for _, text in _texts(project, scope_id):
        terms = [token.casefold() for token in TOKEN.findall(text)]
        population_tokens += len(terms)
        token_counts.update(terms)
        focus_positions = [index for index, term in enumerate(terms) if term == needle]
        focus_frequency += len(focus_positions)
        for index in focus_positions:
            start = max(0, index - window)
            end = min(len(terms), index + window + 1)
            left = [term for term in terms[start:index] if term != needle]
            right = [term for term in terms[index + 1 : end] if term != needle]
            pair_opportunities += len(left) + len(right)
            pair_counts.update(left)
            pair_counts.update(right)
    return token_counts, pair_counts, focus_frequency, population_tokens, pair_opportunities


def _association_population(project, scope_id: str, needle: str, window: int):
    """Count v2-tokenized surface and in-window association populations."""
    token_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    focus_frequency = 0
    population_tokens = 0
    pair_opportunities = 0
    for _, text in _texts(project, scope_id):
        terms = [_surface_form_v2(token.surface) for token in _tokenize_v2(text)]
        population_tokens += len(terms)
        token_counts.update(terms)
        focus_positions = [index for index, term in enumerate(terms) if term == needle]
        focus_frequency += len(focus_positions)
        for index in focus_positions:
            start = max(0, index - window)
            end = min(len(terms), index + window + 1)
            left = [term for term in terms[start:index] if term != needle]
            right = [term for term in terms[index + 1 : end] if term != needle]
            pair_opportunities += len(left) + len(right)
            pair_counts.update(left)
            pair_counts.update(right)
    return token_counts, pair_counts, focus_frequency, population_tokens, pair_opportunities


def collocation_analysis(
    project,
    scope_id: str,
    query: str,
    *,
    window: int = 5,
    min_frequency: int = 2,
    offset: int = 0,
    limit: int = 100,
) -> CollocationAnalysisResult:
    """Return descriptive window-association measures over an explicit scope.

    Observed pair frequency counts every non-self token position within ``window``
    positions of each focus occurrence, without crossing revision boundaries.
    Let ``N`` be all surface-token positions in the scope, ``F`` the number of
    focus occurrences, and ``M`` the sum of valid non-focus neighbor slots over
    those occurrences (overlapping windows count a slot once per focus). For a
    non-focus collocate with corpus frequency ``f`` and observed pair count
    ``O``, expected frequency is ``E = M * f / (N - F)``. The reported measures
    are ``MI = log2(O / E)`` and ``t = (O - E) / sqrt(O)``. Edge positions
    contribute only their available in-document neighbors to ``M``; windows
    never cross revision boundaries. These are descriptive rankings, not
    significance tests.
    """
    needle = _single_surface_query_v2(query, "collocation analysis")
    if not _is_plain_int(window) or window < 1 or window > 100:
        raise ScopeViolationError("collocation window must be between 1 and 100")
    if (
        not _is_plain_int(min_frequency)
        or not _is_plain_int(offset)
        or not _is_plain_int(limit)
        or min_frequency < 1
        or offset < 0
        or limit < 1
        or limit > 10_000
    ):
        raise ScopeViolationError(
            "min_frequency must be positive, offset must be non-negative, and limit must be 1..10000"
        )

    (
        token_counts,
        pair_counts,
        focus_frequency,
        population_tokens,
        pair_opportunities,
    ) = _association_population(project, scope_id, needle, window)

    ranked: list[CollocationAssociationRow] = []
    if population_tokens > 0 and focus_frequency > 0:
        for term, pair_frequency in pair_counts.items():
            if pair_frequency < min_frequency:
                continue
            term_frequency = token_counts[term]
            non_focus_tokens = population_tokens - focus_frequency
            if non_focus_tokens <= 0:
                continue
            expected = pair_opportunities * term_frequency / non_focus_tokens
            if expected <= 0:
                continue
            ranked.append(
                {
                    "term": term,
                    "pair_frequency": pair_frequency,
                    "token_frequency": term_frequency,
                    "expected_frequency": expected,
                    "mutual_information": log2(pair_frequency / expected),
                    "t_score": (pair_frequency - expected) / pair_frequency**0.5,
                }
            )
        ranked.sort(key=lambda row: (-row["t_score"], -row["pair_frequency"], row["term"]))

    total = len(ranked)
    page_end = offset + limit
    rows = ranked[offset:page_end]
    has_more = page_end < total
    return {
        "scope_id": scope_id,
        "calculation_version": "surface-window-association-v2",
        "tokenizer": SURFACE_TOKENIZER_V2,
        "normalization": SURFACE_NORMALIZATION_V2,
        "unicode_data_version": unicodedata.unidata_version,
        "query": needle,
        "window": window,
        "focus_frequency": focus_frequency,
        "population_tokens": population_tokens,
        "pair_opportunities": pair_opportunities,
        "min_frequency": min_frequency,
        "offset": offset,
        "limit": limit,
        "total": total,
        "has_more": has_more,
        "rows": rows,
        "truncated": has_more,
        "interpretation_note": (
            "MI and t-score are descriptive window-association rankings, not "
            "significance tests; expected counts condition on non-focus tokens and "
            "the actual in-window positions."
        ),
    }


def collocation_association_evidence(
    project,
    scope_id: str,
    query: str,
    collocate: str,
    *,
    window: int = 5,
    min_frequency: int = 2,
) -> CollocationEvidenceResult:
    """Persist one exact association score with exhaustive population provenance."""
    query_tokens = list(_tokenize_v2(query))
    collocate_tokens = list(_tokenize_v2(collocate))
    if len(query_tokens) != 1 or len(collocate_tokens) != 1:
        raise ScopeViolationError("query and collocate must each be exactly one surface token")
    if (
        not _is_plain_int(window)
        or not _is_plain_int(min_frequency)
        or window < 1
        or window > 100
        or min_frequency < 1
    ):
        raise ScopeViolationError("window and minimum frequency are outside supported bounds")
    needle = _surface_form_v2(query_tokens[0].surface)
    term = _surface_form_v2(collocate_tokens[0].surface)
    if term == needle:
        raise ScopeViolationError("self-collocation evidence is not supported")
    (
        token_counts,
        pair_counts,
        focus_frequency,
        population_tokens,
        pair_opportunities,
    ) = _association_population(project, scope_id, needle, window)
    pair_frequency = pair_counts[term]
    token_frequency = token_counts[term]
    if pair_frequency < min_frequency:
        raise ScopeViolationError("collocate does not meet the requested pair-frequency threshold")
    if not population_tokens or not focus_frequency or not token_frequency:
        raise ScopeViolationError("association evidence requires a non-empty token population")
    non_focus_tokens = population_tokens - focus_frequency
    if non_focus_tokens <= 0 or pair_opportunities <= 0:
        raise ScopeViolationError("association evidence has no comparable window positions")
    expected = pair_opportunities * token_frequency / non_focus_tokens
    mutual_information = log2(pair_frequency / expected)
    t_score = (pair_frequency - expected) / pair_frequency**0.5
    population_definition = (
        "all non-focus in-window co-occurrence opportunities for the query token "
        "in every revision of the immutable ScopeSnapshot"
    )
    coverage_id, evidence_id = project.corpus_statistic_evidence_with_coverage(
        f"window mutual information for {needle!r} and {term!r}",
        mutual_information,
        coverage=CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=scope_id,
            population_definition=population_definition,
            population_count=pair_opportunities,
            completion_criterion=(
                f"every scope revision tokenized by {SURFACE_TOKENIZER_V2}; all focus "
                "positions and in-window pairs enumerated without crossing revision boundaries"
            ),
            termination_reason="completed deterministic window-pair enumeration",
        ),
        population_definition=population_definition,
        provenance={
            "calculation_version": "surface-window-association-v2",
            "population_count": pair_opportunities,
            "measure": "mutual_information_log2",
            "query": needle,
            "collocate": term,
            "window": window,
            "min_frequency": min_frequency,
            "focus_frequency": focus_frequency,
            "pair_frequency": pair_frequency,
            "token_frequency": token_frequency,
            "population_tokens": population_tokens,
            "pair_opportunities": pair_opportunities,
            "expected_frequency": expected,
            "mutual_information": mutual_information,
            "t_score": t_score,
            "tokenizer": SURFACE_TOKENIZER_V2,
            "normalization": SURFACE_NORMALIZATION_V2,
            "unicode_data_version": unicodedata.unidata_version,
            "interpretation_note": (
                "MI and t-score are descriptive window-association rankings, not "
                "significance tests; expected counts condition on non-focus tokens "
                "and actual in-window positions."
            ),
        },
    )
    return {
        "scope_id": scope_id,
        "query": needle,
        "collocate": term,
        "window": window,
        "focus_frequency": focus_frequency,
        "pair_frequency": pair_frequency,
        "token_frequency": token_frequency,
        "population_tokens": population_tokens,
        "pair_opportunities": pair_opportunities,
        "expected_frequency": expected,
        "mutual_information": mutual_information,
        "t_score": t_score,
        "coverage_id": coverage_id,
        "evidence_id": evidence_id,
    }


def _comparison_counts(project, left_scope_id: str, right_scope_id: str):
    # Comparison-v2 corrects keyness scoring while retaining the historical
    # surface-comparison-v1 tokenizer and therefore its population counts.
    left = _frequency_v1(project, left_scope_id)
    right = _frequency_v1(project, right_scope_id)
    left_total, right_total = sum(left.values()), sum(right.values())
    if not left_total or not right_total:
        raise ScopeViolationError("comparison requires two non-empty explicit scopes")
    return left, right, left_total, right_total


def _comparison_metric_rows(
    left: Counter[str],
    right: Counter[str],
    left_total: int,
    right_total: int,
    metric: str,
) -> Iterator[dict[str, float | int | str]]:
    tokens = chain(left.keys(), (token for token in right if token not in left))
    combined_total = left_total + right_total
    for token in tokens:
        observed_left, observed_right = left[token], right[token]
        if metric == "frequency":
            left_rate = observed_left / left_total
            right_rate = observed_right / right_total
            yield {
                "token": token,
                "left_count": observed_left,
                "right_count": observed_right,
                "left_rate": left_rate,
                "right_rate": right_rate,
                "rate_difference": left_rate - right_rate,
            }
            continue
        observed_nonfocus_left = left_total - observed_left
        observed_nonfocus_right = right_total - observed_right
        term_total = observed_left + observed_right
        nonfocus_total = combined_total - term_total
        expected_left = left_total * term_total / combined_total
        expected_right = right_total * term_total / combined_total
        expected_nonfocus_left = left_total * nonfocus_total / combined_total
        expected_nonfocus_right = right_total * nonfocus_total / combined_total
        score = 0.0
        if observed_left:
            score += observed_left * log(observed_left / expected_left)
        if observed_right:
            score += observed_right * log(observed_right / expected_right)
        # G² for token keyness is the 2×2 likelihood ratio: focal token vs.
        # all other tokens, across the left and right populations. Omitting
        # these complement cells underestimates the statistic and changes the
        # relative ranking, especially for terms frequent in small populations.
        if observed_nonfocus_left:
            score += observed_nonfocus_left * log(observed_nonfocus_left / expected_nonfocus_left)
        if observed_nonfocus_right:
            score += observed_nonfocus_right * log(
                observed_nonfocus_right / expected_nonfocus_right
            )
        signed = 2.0 * score
        if observed_left / left_total < observed_right / right_total:
            signed *= -1
        yield {
            "token": token,
            "left_count": observed_left,
            "right_count": observed_right,
            "log_likelihood": signed,
        }


def _comparison_order_key(row: dict[str, float | int | str], metric: str) -> tuple[float, str]:
    score_key = "rate_difference" if metric == "frequency" else "log_likelihood"
    return -abs(float(row[score_key])), str(row["token"])


def _comparison_page(
    project, left_scope_id: str, right_scope_id: str, *, metric: str, offset: int, limit: int
) -> ComparisonPage:
    if not _valid_comparison_pagination(offset, limit):
        raise EvidenceValidationError(
            "comparison page requires non-negative offset and limit 1..1000"
        )
    left, right, left_total, right_total = _comparison_counts(
        project, left_scope_id, right_scope_id
    )
    total_rows = len(left) + sum(1 for token in right if token not in left)
    rows: list[dict[str, float | int | str]] = []
    if offset < total_rows:
        top = nsmallest(
            offset + limit,
            _comparison_metric_rows(left, right, left_total, right_total, metric),
            key=lambda row: _comparison_order_key(row, metric),
        )
        rows = top[offset:]
    return {
        "metric": metric,
        "offset": offset,
        "limit": limit,
        "total_rows": total_rows,
        "has_more": offset + len(rows) < total_rows,
        "rows": rows,
    }


def compare_frequency_page(
    project,
    left_scope_id: str,
    right_scope_id: str,
    *,
    offset: int = 0,
    limit: int = 100,
) -> ComparisonPage:
    """Return a bounded normalized-frequency page; memory is O(offset + limit)."""
    return _comparison_page(
        project, left_scope_id, right_scope_id, metric="frequency", offset=offset, limit=limit
    )


def keyness_page(
    project,
    left_scope_id: str,
    right_scope_id: str,
    *,
    offset: int = 0,
    limit: int = 100,
) -> ComparisonPage:
    """Return a bounded log-likelihood page; this is descriptive, not interpretive."""
    return _comparison_page(
        project, left_scope_id, right_scope_id, metric="keyness", offset=offset, limit=limit
    )


def compare_frequency(
    project, left_scope_id: str, right_scope_id: str
) -> list[dict[str, float | int | str]]:
    """Return every normalized surface-frequency row (use the page API for large vocabularies)."""
    left, right, left_total, right_total = _comparison_counts(
        project, left_scope_id, right_scope_id
    )
    return sorted(
        _comparison_metric_rows(left, right, left_total, right_total, "frequency"),
        key=lambda row: _comparison_order_key(row, "frequency"),
    )


def keyness_log_likelihood(
    project, left_scope_id: str, right_scope_id: str
) -> list[dict[str, float | int | str]]:
    """Return every keyness row; use `keyness_page` for large vocabularies.

    This deterministic descriptive statistic is not an interpretation of ideology
    or intent. Scope populations remain explicit in the caller's analysis.
    """
    left, right, left_total, right_total = _comparison_counts(
        project, left_scope_id, right_scope_id
    )
    return sorted(
        _comparison_metric_rows(left, right, left_total, right_total, "keyness"),
        key=lambda row: _comparison_order_key(row, "keyness"),
    )


def _persist_comparison(
    project,
    left_scope: str,
    right_scope: str,
    metric: str,
    *,
    offset: int = 0,
    limit: int = 100,
) -> dict[str, object]:
    """Calculate and persist one bilateral comparison without materializing ranked rows."""
    _validate_comparison_metric(metric)
    if not _valid_comparison_pagination(offset, limit):
        raise ScopeViolationError("comparison page requires non-negative offset and limit 1..1000")
    left_counts, right_counts, left_total, right_total = _comparison_counts(
        project, left_scope, right_scope
    )

    comparison_id = project.create_comparison_run_stream(
        left_scope,
        right_scope,
        metric=metric,
        calculation_version=COMPARISON_CALCULATION_VERSION,
        config={"tokenizer": "corpus.TOKEN", "metric": metric},
        rows=_comparison_metric_rows(left_counts, right_counts, left_total, right_total, metric),
    )
    page = project.comparison_rows_page(comparison_id, offset=offset, limit=limit)
    return {
        "comparison_id": comparison_id,
        "metric": metric,
        "calculation_version": COMPARISON_CALCULATION_VERSION,
        "left_scope_id": left_scope,
        "right_scope_id": right_scope,
        "rows": page["rows"],
        "total_rows": page["total_rows"],
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(cast(list[object], page["rows"])) < int(page["total_rows"]),
    }


def compare_named_corpora(
    project,
    left_corpus_id: str,
    right_corpus_id: str,
    *,
    metric: str = "frequency",
    offset: int = 0,
    limit: int = 100,
) -> dict[str, object]:
    """Compare two named corpora through fresh immutable snapshots of memberships."""
    if left_corpus_id == right_corpus_id:
        raise ScopeViolationError("comparison requires two distinct corpora")
    if not _valid_comparison_pagination(offset, limit):
        raise ScopeViolationError("comparison page requires non-negative offset and limit 1..1000")
    _validate_comparison_metric(metric)
    result = _persist_comparison(
        project,
        project.corpus_snapshot(left_corpus_id),
        project.corpus_snapshot(right_corpus_id),
        metric,
        offset=offset,
        limit=limit,
    )
    return {**result, "left_corpus_id": left_corpus_id, "right_corpus_id": right_corpus_id}


def compare_documents(
    project,
    left_revision_id: str,
    right_revision_id: str,
    *,
    metric: str = "frequency",
    offset: int = 0,
    limit: int = 100,
) -> dict[str, object]:
    """Persist a bilateral comparison of two explicit immutable document revisions."""
    if left_revision_id == right_revision_id:
        raise ScopeViolationError("comparison requires two distinct document revisions")
    if not _valid_comparison_pagination(offset, limit):
        raise ScopeViolationError("comparison page requires non-negative offset and limit 1..1000")
    _validate_comparison_metric(metric)
    result = _persist_comparison(
        project,
        project.snapshot([left_revision_id]),
        project.snapshot([right_revision_id]),
        metric,
        offset=offset,
        limit=limit,
    )
    return {
        **result,
        "left_revision_id": left_revision_id,
        "right_revision_id": right_revision_id,
    }


def comparison_statistic_evidence(
    project,
    analysis_run_id: str,
    comparison_id: str,
    token: str,
    *,
    comparability_basis: str,
) -> dict[str, object]:
    """Persist one verified row from a completed, two-population comparison.

    A researcher supplies the comparability rationale; DiscourseLens records it
    but does not independently assert that the populations are comparable.
    The analysis run must explicitly cover exactly the union of both frozen
    comparison scopes. The comparison result is exhaustively hash-verified
    before its selected token row is linked as evidence.
    """
    if not isinstance(comparability_basis, str) or not comparability_basis.strip():
        raise ScopeViolationError("comparative evidence requires a stated comparability basis")
    if len(comparability_basis) > 2_000:
        raise ScopeViolationError("comparability basis exceeds the 2000-character limit")
    normalized_terms = TOKEN.findall(token) if isinstance(token, str) else []
    if len(normalized_terms) != 1 or normalized_terms[0].casefold() != token.casefold():
        raise ScopeViolationError("comparison evidence requires one normalized surface token")
    normalized_token = normalized_terms[0].casefold()

    run = project.db.execute(
        "SELECT scope_id FROM analysis_runs WHERE id=?", (analysis_run_id,)
    ).fetchone()
    if not run:
        raise ScopeViolationError("comparison evidence references an unknown analysis run")
    comparison = project.comparison_run(comparison_id)
    if comparison["storage_kind"] != "rows-v1":
        raise ScopeViolationError(
            "comparative evidence currently requires normalized rows-v1 comparison storage"
        )
    left_scope_id = str(comparison["left_scope_id"])
    right_scope_id = str(comparison["right_scope_id"])
    left_revisions = project.scope_revisions(left_scope_id)
    right_revisions = project.scope_revisions(right_scope_id)
    analysis_revisions = project.scope_revisions(str(run["scope_id"]))
    if set(analysis_revisions) != set(left_revisions) | set(right_revisions):
        raise ScopeViolationError(
            "analysis run scope must equal the union of both comparison populations"
        )

    try:
        population_result = project.comparison_rows_page(comparison_id, offset=0, limit=1)
    except (DocumentError, EvidenceValidationError) as exc:
        raise ScopeViolationError(f"comparison result cannot support evidence: {exc}") from exc
    if population_result.get("result_hash_verified") is not True:
        raise EvidenceValidationError("comparison result hash was not verified")
    stored = project.db.execute(
        "SELECT token,rank_value,row_json FROM comparison_result_rows "
        "WHERE comparison_id=? AND token=?",
        (comparison_id, normalized_token),
    ).fetchone()
    if not stored:
        raise ScopeViolationError("surface token is not present in the comparison result")
    selected_row = project._validate_stored_comparison_row(stored, str(comparison["metric"]))
    metric = str(comparison["metric"])
    value_key = "rate_difference" if metric == "frequency" else "log_likelihood"
    value = selected_row.get(value_key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise EvidenceValidationError("comparison statistic has no numeric result")
    left_population = int(population_result["left_population"])
    right_population = int(population_result["right_population"])
    population_count = left_population + right_population
    if left_population <= 0 or right_population <= 0 or population_count <= 0:
        raise EvidenceValidationError("comparison populations are empty or corrupt")

    population_definition = (
        "surface-token positions in two separate exhaustive populations: "
        f"left ScopeSnapshot {left_scope_id}; right ScopeSnapshot {right_scope_id}"
    )
    coverage_id = project.coverage(
        CoverageDescriptor(
            coverage_mode=CoverageMode.EXHAUSTIVE,
            scope_snapshot_id=str(run["scope_id"]),
            population_definition=population_definition,
            population_count=population_count,
            selection_policy=(
                f"verified comparison {comparison_id}; selected exact surface token "
                f"{normalized_token!r} from the complete result"
            ),
            inspection_count=population_count,
            completion_criterion=(
                "all surface-token positions in each frozen comparison scope were enumerated; "
                "all normalized comparison rows and the result hash were verified"
            ),
            termination_reason="completed deterministic comparison",
            termination_state=CoverageTerminationState.COMPLETED,
        )
    )
    evidence_id = project.corpus_statistic_evidence(
        f"{metric} comparison for {normalized_token!r}",
        float(value),
        scope_snapshot_id=str(run["scope_id"]),
        population_definition=population_definition,
        provenance={
            "comparison_run_id": comparison_id,
            "comparison_metric": metric,
            "comparison_result_hash": str(comparison["result_hash"]),
            "analysis_run_id": analysis_run_id,
            "calculation_version": str(comparison["calculation_version"]),
            "coverage_id": coverage_id,
            "population_count": population_count,
            "left_scope_snapshot_id": left_scope_id,
            "right_scope_snapshot_id": right_scope_id,
            "left_revision_ids": left_revisions,
            "right_revision_ids": right_revisions,
            "left_population": left_population,
            "right_population": right_population,
            "selected_row": selected_row,
            "comparability_basis": comparability_basis.strip(),
        },
    )
    return {
        "comparison_id": comparison_id,
        "metric": metric,
        "token": normalized_token,
        "value": float(value),
        "left_population": left_population,
        "right_population": right_population,
        "population_count": population_count,
        "scope_id": str(run["scope_id"]),
        "coverage_id": coverage_id,
        "evidence_id": evidence_id,
    }


def create_comparative_finding(
    project,
    comparison_id: str,
    token: str,
    *,
    comparability_basis: str,
) -> dict[str, object]:
    """Create one auditable Finding from an explicitly selected comparison row.

    This deterministic workflow makes no discourse interpretation. It records
    the bilateral result as a `COMPARATIVE` claim and gives the researcher a
    validated AnalysisRun/Finding lineage through the same Core path used by
    other clients.
    """
    if not isinstance(comparability_basis, str) or not comparability_basis.strip():
        raise ScopeViolationError("comparative evidence requires a stated comparability basis")
    if len(comparability_basis) > 2_000:
        raise ScopeViolationError("comparability basis exceeds the 2000-character limit")
    comparison = project.comparison_run(comparison_id)
    if comparison["storage_kind"] != "rows-v1":
        raise ScopeViolationError(
            "comparative findings currently require normalized rows-v1 comparison storage"
        )
    left_revisions = project.scope_revisions(str(comparison["left_scope_id"]))
    right_revisions = project.scope_revisions(str(comparison["right_scope_id"]))
    union_revisions = sorted(set(left_revisions) | set(right_revisions))
    if not union_revisions:
        raise ScopeViolationError("comparison populations cannot be empty")
    # Preflight before creating a historical AnalysisRun so a misspelled token
    # or a corrupt stored result does not leave a failed run behind.
    project.comparison_rows_page(comparison_id, offset=0, limit=1)
    normalized_terms = TOKEN.findall(token) if isinstance(token, str) else []
    if len(normalized_terms) != 1 or normalized_terms[0].casefold() != token.casefold():
        raise ScopeViolationError("comparison finding requires one normalized surface token")
    normalized_token = normalized_terms[0].casefold()
    selected = project.db.execute(
        "SELECT token,rank_value,row_json FROM comparison_result_rows "
        "WHERE comparison_id=? AND token=?",
        (comparison_id, normalized_token),
    ).fetchone()
    if not selected:
        raise ScopeViolationError("surface token is not present in the comparison result")
    selected_row = project._validate_stored_comparison_row(selected, str(comparison["metric"]))
    scope_id = project.snapshot(union_revisions)
    analysis_run_id = project.create_run(
        scope_id,
        {
            "workflow": "deterministic-comparison-finding-v1",
            "comparison_run_id": comparison_id,
            "comparison_metric": str(comparison["metric"]),
            "selected_term": normalized_token,
        },
    )
    try:
        project.start_run(analysis_run_id)
        evidence_result = comparison_statistic_evidence(
            project,
            analysis_run_id,
            comparison_id,
            normalized_token,
            comparability_basis=comparability_basis,
        )
        metric = str(comparison["metric"])
        left_count = int(selected_row["left_count"])
        right_count = int(selected_row["right_count"])
        left_population = cast(int, evidence_result["left_population"])
        right_population = cast(int, evidence_result["right_population"])
        left_rate = left_count / left_population
        right_rate = right_count / right_population
        if metric == "frequency":
            difference = float(selected_row["rate_difference"])
            observation = (
                f"Surface token {normalized_token!r}: left count {left_count}/"
                f"{left_population} (rate {left_rate!r}); right count {right_count}/"
                f"{right_population} (rate {right_rate!r}); left-minus-right rate difference "
                f"{difference!r}."
            )
            phenomenon = "relative_surface_frequency_difference"
        else:
            score = float(selected_row["log_likelihood"])
            observation = (
                f"Surface token {normalized_token!r}: left count {left_count}/"
                f"{left_population} (rate {left_rate!r}); right count {right_count}/"
                f"{right_population} (rate {right_rate!r}); signed log-likelihood "
                f"{score!r} (positive favors the left population under this calculation)."
            )
            phenomenon = "surface_keyness_difference"
        finding_id = project.create_finding(
            analysis_run_id,
            "core.corpus-comparison",
            "1.0.0",
            phenomenon,
            observation,
            claim_types=[ClaimType.COMPARATIVE],
            evidence_sufficiency="SUFFICIENT",
            interpretation_confidence="LOW",
            limitations=[
                "This deterministic comparison is descriptive and does not establish cause, intent, or ideological meaning.",
                "The researcher supplied the population-comparability basis; DiscourseLens did not independently assess it.",
            ],
            coverage_id=str(evidence_result["coverage_id"]),
        )
        evidence_id = str(evidence_result["evidence_id"])
        project.attach_evidence(
            finding_id,
            evidence_id,
            EvidenceRole.SUPPORT,
            "Hash-verified selected row from the two frozen comparison populations.",
        )
        from .validation import validate_finding

        validation = validate_finding(project, finding_id)
        if not validation.valid:
            raise EvidenceValidationError("; ".join(validation.errors))
        citation = project.render_citation(evidence_id)
        project.finalize_run(analysis_run_id)
    except Exception as exc:
        project.fail_run(analysis_run_id, exc)
        raise
    return {
        "analysis_run_id": analysis_run_id,
        "finding_id": finding_id,
        "comparison_id": comparison_id,
        "evidence_id": evidence_id,
        "coverage_id": evidence_result["coverage_id"],
        "token": normalized_token,
        "metric": metric,
        "observation": observation,
        "citation": citation,
        "validation": {
            "valid": validation.valid,
            "errors": list(validation.errors),
            "semantic_entailment_assessed": validation.semantic_entailment_assessed,
        },
    }
