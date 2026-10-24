"""Capture offline lexical rankings from an explicit frozen project scope."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from discourselens import Project
from discourselens.errors import ScopeViolationError
from discourselens.vectors import _scoped_vector_artifacts

from .retrieval_eval import RESULTS_VERSION, parse_rankings

MAX_CAPTURE_RESULTS = 1_000


@dataclass(frozen=True)
class CaptureQuery:
    """One query and the immutable scope from which its candidates are retrieved."""

    query_id: str
    query_text: str
    scope_id: str


@dataclass(frozen=True)
class VectorCaptureQuery:
    """One explicit raw-vector query for offline vector-ranking capture.

    Raw vectors let the evaluation harness run without model downloads or paid
    embedding APIs. They must come from the embedding pipeline under evaluation
    for the resulting ranks to say anything about semantic retrieval quality.
    """

    query_id: str
    scope_id: str
    artifact_ids: tuple[str, ...]
    query_vector: tuple[float, ...]


def capture_lexical_rankings(
    project: Project,
    queries: Sequence[CaptureQuery],
    *,
    retriever_id: str,
    limit: int = 100,
) -> dict[str, object]:
    """Capture BM25 exact-phrase results as ranked document revisions.

    ``Project.search_exact_ranked`` ranks matching paragraphs. This capture maps each
    paragraph to its revision and keeps the revision's first occurrence, so the output
    unit is ``DOCUMENT_REVISION`` and the paragraph with its strongest returned rank
    determines that revision's position. The limit counts unique revisions. This is a
    mechanical export of API order, not a semantic-quality measurement.
    """
    if not isinstance(retriever_id, str) or not retriever_id.strip():
        raise ValueError("retriever_id must be a non-empty string")
    if type(limit) is not int or not 1 <= limit <= MAX_CAPTURE_RESULTS:
        raise ValueError(f"limit must be an integer from 1 to {MAX_CAPTURE_RESULTS}")
    if not queries:
        raise ValueError("at least one capture query is required")

    seen_query_ids: set[str] = set()
    query_results: list[dict[str, object]] = []
    for query in queries:
        if not isinstance(query, CaptureQuery):
            raise TypeError("queries must contain CaptureQuery values")
        if not isinstance(query.query_id, str) or not query.query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        if query.query_id in seen_query_ids:
            raise ValueError(f"duplicate query_id {query.query_id!r}")
        seen_query_ids.add(query.query_id)
        if not isinstance(query.query_text, str) or not query.query_text.strip():
            raise ValueError(f"query {query.query_id!r} must have non-empty query_text")
        if not isinstance(query.scope_id, str) or not query.scope_id.strip():
            raise ValueError(f"query {query.query_id!r} must have a non-empty scope_id")

        allowed_revisions = set(project.scope_revisions(query.scope_id))
        revision_ids: list[str] = []
        seen_revisions: set[str] = set()
        for row in project.search_exact_ranked(query.query_text, query.scope_id):
            revision_id = row["revision_id"]
            if not isinstance(revision_id, str) or not revision_id:
                raise ScopeViolationError("lexical retrieval returned a malformed revision ID")
            if revision_id not in allowed_revisions:
                raise ScopeViolationError(
                    "lexical retrieval returned a revision outside the frozen scope"
                )
            if revision_id not in seen_revisions:
                seen_revisions.add(revision_id)
                if len(revision_ids) < limit:
                    revision_ids.append(revision_id)

        query_results.append(
            {
                "query_id": query.query_id,
                "results": [
                    {"unit_id": revision_id, "rank": rank}
                    for rank, revision_id in enumerate(revision_ids, start=1)
                ],
            }
        )

    payload: dict[str, object] = {
        "schema_version": RESULTS_VERSION,
        "unit_kind": "DOCUMENT_REVISION",
        "retrievers": [
            {"retriever_id": retriever_id, "query_results": query_results}
        ],
    }
    # Keep the exported payload inside the same contract used by the evaluator.
    parse_rankings(payload, seen_query_ids)
    return payload


def capture_vector_rankings(
    project: Project,
    queries: Sequence[VectorCaptureQuery],
    *,
    retriever_id: str,
    limit: int = 100,
) -> dict[str, object]:
    """Capture explicit-scope vector rankings as unique document revisions.

    The Core validates each artifact against the immutable scope before search.
    Multiple chunk hits for one revision collapse to its first (best) returned
    rank. This captures mechanics and order only; random or hand-authored vectors
    are not semantic-quality evidence.
    """
    if not isinstance(retriever_id, str) or not retriever_id.strip():
        raise ValueError("retriever_id must be a non-empty string")
    if type(limit) is not int or not 1 <= limit <= MAX_CAPTURE_RESULTS:
        raise ValueError(f"limit must be an integer from 1 to {MAX_CAPTURE_RESULTS}")
    if not queries:
        raise ValueError("at least one capture query is required")

    seen_query_ids: set[str] = set()
    query_results: list[dict[str, object]] = []
    for query in queries:
        if not isinstance(query, VectorCaptureQuery):
            raise TypeError("queries must contain VectorCaptureQuery values")
        if not isinstance(query.query_id, str) or not query.query_id.strip():
            raise ValueError("query_id must be a non-empty string")
        if query.query_id in seen_query_ids:
            raise ValueError(f"duplicate query_id {query.query_id!r}")
        seen_query_ids.add(query.query_id)
        if not isinstance(query.scope_id, str) or not query.scope_id.strip():
            raise ValueError(f"query {query.query_id!r} must have a non-empty scope_id")
        if not query.artifact_ids or any(
            not isinstance(artifact_id, str) or not artifact_id.strip()
            for artifact_id in query.artifact_ids
        ):
            raise ValueError(f"query {query.query_id!r} must select vector artifacts")
        if len(set(query.artifact_ids)) != len(query.artifact_ids):
            raise ValueError(f"query {query.query_id!r} has duplicate artifact IDs")

        allowed_revisions = set(project.scope_revisions(query.scope_id))
        # Validate the full selection together first: this rejects duplicate
        # vector indexes for one revision before per-index capture begins.
        artifacts = _scoped_vector_artifacts(
            project,
            query.scope_id,
            list(query.artifact_ids),
            operation="vector benchmark capture",
        )

        # Document-level rank is each revision's best matching chunk. Asking the
        # Core once for `limit` global chunks and then deduplicating can let one
        # long document crowd every other revision out of the ranking. Query the
        # top chunk in each already-validated per-revision index, then sort those
        # document scores globally. Each search retains normal Core provenance.
        best_by_revision: list[tuple[float, str, str]] = []
        for artifact in artifacts:
            artifact_id = artifact["id"]
            revision_id = artifact["revision_id"]
            if not isinstance(artifact_id, str) or not isinstance(revision_id, str):
                raise ScopeViolationError("vector artifact metadata is malformed")
            if revision_id not in allowed_revisions:
                raise ScopeViolationError(
                    "vector artifact is outside the frozen benchmark scope"
                )
            hits = project.search_vectors_with_provenance(
                query.scope_id,
                [artifact_id],
                query.query_vector,
                limit=1,
            )
            if len(hits) > 1:
                raise ScopeViolationError(
                    "per-revision vector retrieval returned multiple top hits"
                )
            for hit in hits:
                if hit.revision_id != revision_id or hit.artifact_id != artifact_id:
                    raise ScopeViolationError(
                        "vector retrieval returned a hit outside the selected revision index"
                    )
                best_by_revision.append(
                    (float(hit.distance), hit.revision_id, hit.chunk_id)
                )

        best_by_revision.sort(key=lambda candidate: candidate)
        revision_ids = [
            revision_id for _, revision_id, _ in best_by_revision[:limit]
        ]

        query_results.append(
            {
                "query_id": query.query_id,
                "results": [
                    {"unit_id": revision_id, "rank": rank}
                    for rank, revision_id in enumerate(revision_ids, start=1)
                ],
            }
        )

    payload: dict[str, object] = {
        "schema_version": RESULTS_VERSION,
        "unit_kind": "DOCUMENT_REVISION",
        "retrievers": [
            {"retriever_id": retriever_id, "query_results": query_results}
        ],
    }
    parse_rankings(payload, seen_query_ids)
    return payload
