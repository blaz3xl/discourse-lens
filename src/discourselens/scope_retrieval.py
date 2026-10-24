"""Small, explicit retrieval plans used by trusted built-in skills."""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass

from .errors import ModelResponseError, ProjectError, ScopeViolationError
from .models import ModelGateway, RerankResult
from .network_policy import is_loopback_endpoint
from .project import Project
from .types import CoverageMode

FRAMEWORK_CONTEXT_PLAN = "retrieval/framework-context-lexical-v1"
MAX_FRAMEWORK_CANDIDATES = 1_000
MAX_FRAMEWORK_HITS = 12
_WORD = re.compile(r"[\w’'-]+", re.UNICODE)
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "them",
        "this",
        "to",
        "was",
        "were",
        "with",
        "y",
        "de",
        "la",
        "el",
        "los",
        "las",
        "un",
        "una",
        "que",
        "en",
        "por",
        "para",
        "con",
        "del",
        "es",
        "son",
        "se",
        "su",
        "sus",
        "como",
        "al",
        "lo",
    ]
)


@dataclass(frozen=True)
class TargetedContextCandidate:
    revision_id: str
    paragraph_id: str
    start: int
    end: int
    text: str
    score: float


@dataclass(frozen=True)
class TargetedContextResult:
    scope_id: str
    query_terms: tuple[str, ...]
    candidates: tuple[TargetedContextCandidate, ...]
    inspected_count: int
    candidate_limit_reached: bool
    output_limit_reached: bool = False


@dataclass(frozen=True)
class RerankedContextCandidate:
    """A scoped lexical candidate with a separate provider relevance score."""

    candidate: TargetedContextCandidate
    rank: int
    reranker_score: float


@dataclass(frozen=True)
class RerankedTargetedContextResult:
    """Reranking provenance; candidate coverage remains targeted and unchanged."""

    scope_id: str
    query: str
    provider_id: str
    credential_alias: str | None
    model: str
    requested_top_n: int | None
    effective_top_n: int | None
    coverage_mode: CoverageMode
    retrieval: TargetedContextResult
    ranked_candidates: tuple[RerankedContextCandidate, ...]


def rerank_framework_context(
    project: Project,
    scope_id: str,
    retrieval: TargetedContextResult,
    gateway: ModelGateway,
    *,
    provider_id: str,
    model: str,
    credential_alias: str | None = None,
    query: str,
    top_n: int | None = None,
) -> RerankedTargetedContextResult:
    """Reorder only validated lexical candidates from an explicit ScopeSnapshot.

    This opt-in step sends candidate text to the configured reranking provider.
    It cannot retrieve new passages, widen scope, or change the lexical retrieval's
    coverage descriptor.  Provider scores are kept separate from lexical scores.
    """
    if not isinstance(retrieval, TargetedContextResult):
        raise ScopeViolationError("reranking requires a validated targeted retrieval result")
    if retrieval.scope_id != scope_id:
        raise ScopeViolationError(
            "reranking scope differs from the original retrieval ScopeSnapshot"
        )
    if not isinstance(query, str) or not query.strip():
        raise ScopeViolationError("reranking query must not be empty")
    if top_n is not None and (type(top_n) is not int or top_n < 1):
        raise ScopeViolationError("reranking top_n must be a positive integer")
    if (
        not isinstance(retrieval.candidates, tuple)
        or len(retrieval.candidates) > MAX_FRAMEWORK_HITS
        or type(retrieval.inspected_count) is not int
        or not 0 <= retrieval.inspected_count <= MAX_FRAMEWORK_CANDIDATES
        or type(retrieval.candidate_limit_reached) is not bool
        or type(retrieval.output_limit_reached) is not bool
        or len(retrieval.candidates) > retrieval.inspected_count
        or (
            retrieval.candidate_limit_reached
            and retrieval.inspected_count != MAX_FRAMEWORK_CANDIDATES
        )
    ):
        raise ScopeViolationError("reranking retrieval metadata exceeds validated bounds")
    query_terms = retrieval.query_terms
    if (
        not isinstance(query_terms, tuple)
        or len(query_terms) > 8
        or any(type(term) is not str or not term or term.casefold() != term for term in query_terms)
        or len(set(query_terms)) != len(query_terms)
    ):
        raise ScopeViolationError("reranking retrieval query terms are malformed")

    revisions = set(project.scope_revisions(scope_id))
    candidates = retrieval.candidates
    if not candidates:
        return RerankedTargetedContextResult(
            scope_id,
            query,
            provider_id,
            credential_alias,
            model,
            top_n,
            None,
            CoverageMode.TARGETED,
            retrieval,
            (),
        )
    if project.offline_mode:
        reranker = gateway.rerankers.get(provider_id)
        endpoint = getattr(reranker, "endpoint", None)
        if not endpoint or not is_loopback_endpoint(endpoint):
            raise ProjectError("project offline mode blocks remote reranking")
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if (
            not isinstance(candidate, TargetedContextCandidate)
            or type(candidate.start) is not int
            or type(candidate.end) is not int
            or candidate.start < 0
            or candidate.end <= candidate.start
            or type(candidate.score) not in {int, float}
            or type(candidate.revision_id) is not str
            or not candidate.revision_id
            or type(candidate.paragraph_id) is not str
            or not candidate.paragraph_id
            or type(candidate.text) is not str
        ):
            raise ScopeViolationError("reranking candidate has invalid offsets or lexical score")
        try:
            lexical_score = float(candidate.score)
        except (OverflowError, TypeError, ValueError):
            raise ScopeViolationError("reranking candidate has invalid offsets or lexical score") from None
        if not math.isfinite(lexical_score) or lexical_score <= 0:
            raise ScopeViolationError("reranking candidate has invalid offsets or lexical score")
        words = set(_WORD.findall(candidate.text.casefold()))
        expected_score = float(sum(term in words for term in query_terms))
        if lexical_score != expected_score:
            raise ScopeViolationError("reranking candidate lexical score does not match retrieval terms")
        identity = (candidate.revision_id, candidate.paragraph_id)
        if candidate.revision_id not in revisions or identity in seen:
            raise ScopeViolationError(
                "reranking candidates are duplicated or outside explicit scope"
            )
        seen.add(identity)
        paragraph = project.db.execute(
            "SELECT revision_id,start_offset,end_offset,text FROM paragraphs WHERE id=?",
            (candidate.paragraph_id,),
        ).fetchone()
        if (
            paragraph is None
            or paragraph["revision_id"] != candidate.revision_id
            or paragraph["start_offset"] != candidate.start
            or paragraph["end_offset"] != candidate.end
            or paragraph["text"] != candidate.text
            or project.revision_text(candidate.revision_id)[candidate.start : candidate.end]
            != candidate.text
        ):
            raise ScopeViolationError(
                "reranking candidate does not resolve to canonical paragraph text"
            )

    results = gateway.rerank(
        provider_id,
        model,
        query,
        [candidate.text for candidate in candidates],
        top_n=min(top_n, len(candidates)) if top_n is not None else len(candidates),
    )
    if not isinstance(results, list):
        raise ModelResponseError("reranker returned malformed results")
    indexes: set[int] = set()
    ranked: list[RerankedContextCandidate] = []
    for rank, result in enumerate(results, start=1):
        if not isinstance(result, RerankResult):
            raise ModelResponseError("reranker returned a malformed result")
        index = result.index
        score = result.relevance_score
        if type(index) is not int or type(score) not in {int, float}:
            raise ModelResponseError("reranker returned an invalid candidate index or score")
        try:
            numeric_score = float(score)
        except (OverflowError, TypeError, ValueError):
            raise ModelResponseError("reranker returned an invalid candidate index or score") from None
        if (
            index < 0
            or index >= len(candidates)
            or index in indexes
            or not math.isfinite(numeric_score)
        ):
            raise ModelResponseError("reranker returned an invalid candidate index or score")
        indexes.add(index)
        ranked.append(
            RerankedContextCandidate(
                candidate=candidates[index],
                rank=rank,
                reranker_score=numeric_score,
            )
        )
    if top_n is not None and len(ranked) > top_n:
        raise ModelResponseError("reranker returned more candidates than requested")
    if not ranked:
        raise ModelResponseError("reranker returned no results for a non-empty candidate set")
    return RerankedTargetedContextResult(
        scope_id,
        query,
        provider_id,
        credential_alias,
        model,
        top_n,
        min(top_n, len(candidates)) if top_n is not None else len(candidates),
        CoverageMode.TARGETED,
        retrieval,
        tuple(ranked),
    )


def retrieve_framework_context(
    project: Project,
    scope_id: str,
    seed_evidence: list[sqlite3.Row],
    *,
    phenomenon: str,
    observation: str,
    query_texts: tuple[str, ...] | None = None,
    limit: int = MAX_FRAMEWORK_HITS,
) -> TargetedContextResult:
    """Find additional lexical context only within the caller's immutable scope.

    This is intentionally TARGETED, bounded, and deterministic. It cannot support
    corpus-wide absence or prevalence claims. Seed passages remain available, but
    they do not constrain retrieval to their own chunks or paragraphs.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_FRAMEWORK_HITS:
        raise ValueError(f"framework context limit must be between 1 and {MAX_FRAMEWORK_HITS}")
    scoped_revisions = set(project.scope_revisions(scope_id))
    source_texts = list(query_texts) if query_texts is not None else [phenomenon, observation]
    seed_paragraphs: set[str] = set()
    for item in seed_evidence:
        span_id = item["span_id"]
        if not span_id:
            continue
        span = project.db.execute(
            "SELECT revision_id,start_offset,end_offset FROM source_spans WHERE id=?", (span_id,)
        ).fetchone()
        if not span:
            raise ScopeViolationError("seed evidence references an unknown source span")
        if str(span["revision_id"]) not in scoped_revisions:
            raise ScopeViolationError("seed evidence is outside the explicit retrieval scope")
        if query_texts is None:
            source_texts.append(project.span_text(span_id))
        paragraphs = project.db.execute(
            "SELECT id FROM paragraphs WHERE revision_id=? AND start_offset<? AND end_offset>? "
            "ORDER BY start_offset,id",
            (span["revision_id"], span["end_offset"], span["start_offset"]),
        ).fetchall()
        seed_paragraphs.update(str(paragraph["id"]) for paragraph in paragraphs)

    counts: Counter[str] = Counter()
    for text in source_texts:
        for word in _WORD.findall(text.casefold()):
            if len(word) >= 4 and word not in _STOPWORDS and not word.isdecimal():
                counts[word] += 1
    terms = tuple(
        word for word, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:8]
    )
    if not terms:
        return TargetedContextResult(scope_id, (), (), 0, False)

    query = " OR ".join(f'"{term}"' for term in terms)
    rows = project.search_paragraphs(query, scope_id, limit=MAX_FRAMEWORK_CANDIDATES)
    if not isinstance(rows, list) or len(rows) > MAX_FRAMEWORK_CANDIDATES:
        raise ProjectError("paragraph search returned malformed or unbounded results")
    candidates: list[TargetedContextCandidate] = []
    seen_paragraphs: set[str] = set()
    canonical_texts: dict[str, str] = {}
    for row in rows:
        try:
            revision_value = row["revision_id"]
            paragraph_value = row["paragraph_id"]
            text_value = row["text"]
            start = row["start_offset"]
            end = row["end_offset"]
        except (IndexError, KeyError, TypeError) as exc:
            raise ProjectError("paragraph search returned a malformed candidate") from exc
        if (
            type(revision_value) is not str
            or type(paragraph_value) is not str
            or type(text_value) is not str
            or type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
        ):
            raise ProjectError("paragraph search returned a malformed candidate")
        revision_id = revision_value
        paragraph_id = paragraph_value
        text = text_value
        if revision_id not in scoped_revisions:
            raise ScopeViolationError("paragraph search returned a revision outside the explicit scope")
        if paragraph_id in seen_paragraphs:
            raise ProjectError("paragraph search returned a duplicate candidate")
        seen_paragraphs.add(paragraph_id)
        if revision_id not in canonical_texts:
            canonical_texts[revision_id] = project.revision_text(revision_id)
        paragraph = project.db.execute(
            "SELECT revision_id,start_offset,end_offset,text FROM paragraphs WHERE id=?",
            (paragraph_id,),
        ).fetchone()
        if (
            paragraph is None
            or paragraph["revision_id"] != revision_id
            or paragraph["start_offset"] != start
            or paragraph["end_offset"] != end
            or paragraph["text"] != text
            or canonical_texts[revision_id][start:end] != text
        ):
            raise ScopeViolationError("paragraph search returned non-canonical candidate text or offsets")
        if paragraph_id in seed_paragraphs:
            continue
        words = set(_WORD.findall(text.casefold()))
        score = float(sum(term in words for term in terms))
        if score:
            candidates.append(
                TargetedContextCandidate(
                    revision_id=revision_id,
                    paragraph_id=paragraph_id,
                    start=start,
                    end=end,
                    text=text,
                    score=score,
                )
            )
    revision_order = {rid: index for index, rid in enumerate(project.scope_revisions(scope_id))}
    candidates.sort(
        key=lambda item: (
            -item.score,
            revision_order[item.revision_id],
            item.start,
            item.paragraph_id,
        )
    )
    return TargetedContextResult(
        scope_id,
        terms,
        tuple(candidates[:limit]),
        len(rows),
        len(rows) == MAX_FRAMEWORK_CANDIDATES,
        len(candidates) > limit,
    )
