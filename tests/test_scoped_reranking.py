from __future__ import annotations

from typing import cast

import pytest

from discourselens import Project
from discourselens.errors import ModelResponseError, ScopeViolationError
from discourselens.models import ModelGateway, RerankResult
from discourselens.scope_retrieval import (
    TargetedContextCandidate,
    TargetedContextResult,
    rerank_framework_context,
    retrieve_framework_context,
)
from discourselens.types import CoverageMode


class _FixtureReranker:
    provider_id = "fixture-reranker"

    def __init__(self, response: list[RerankResult] | None = None):
        self.response = response
        self.documents: list[str] = []

    def rerank(self, model, query, documents, *, top_n=None):
        self.documents = list(documents)
        if self.response is not None:
            return self.response
        return [RerankResult(index, float(index)) for index in reversed(range(len(documents)))]


def test_reranker_only_reorders_canonical_candidates_within_scope(tmp_path):
    project = Project.init(tmp_path / "project")
    _, in_scope = project.import_text(
        "selected.txt",
        "Migration discourse describes border policies.\n\n"
        "Migration discourse centers visa policy.",
    )
    project.import_text("outside.txt", "Migration discourse describes the same border policies.")
    scope_id = project.snapshot([in_scope])
    retrieval = retrieve_framework_context(
        project,
        scope_id,
        [],
        phenomenon="border policies",
        observation="migration discourse",
        query_texts=("migration discourse",),
    )
    assert len(retrieval.candidates) == 2
    assert all(item.revision_id == in_scope for item in retrieval.candidates)

    reranker = _FixtureReranker()
    result = rerank_framework_context(
        project,
        scope_id,
        retrieval,
        ModelGateway({}, rerankers={reranker.provider_id: reranker}),
        provider_id=reranker.provider_id,
        model="fixture-v1",
        query="migration and policy framing",
        top_n=20,
    )

    assert reranker.documents == [item.text for item in retrieval.candidates]
    assert len(result.ranked_candidates) == 2
    ranked = result.ranked_candidates[0]
    assert ranked.candidate is retrieval.candidates[-1]
    assert ranked.rank == 1
    assert ranked.reranker_score == 1.0
    assert ranked.candidate.score == retrieval.candidates[-1].score
    assert {item.candidate.paragraph_id for item in result.ranked_candidates} == {
        item.paragraph_id for item in retrieval.candidates
    }
    assert result.coverage_mode is CoverageMode.TARGETED
    assert result.retrieval is retrieval
    assert result.scope_id == scope_id
    assert result.requested_top_n == 20
    assert result.effective_top_n == 2


def test_reranker_rejects_candidate_outside_explicit_scope(tmp_path):
    project = Project.init(tmp_path / "project")
    _, selected = project.import_text("selected.txt", "Migration discourse in scope.")
    _, outside = project.import_text("outside.txt", "Migration discourse outside scope.")
    scope_id = project.snapshot([selected])
    retrieval = TargetedContextResult(
        scope_id=scope_id,
        query_terms=("migration",),
        candidates=(
            TargetedContextCandidate(
                revision_id=outside,
                paragraph_id=str(
                    project.db.execute(
                        "SELECT id FROM paragraphs WHERE revision_id=?", (outside,)
                    ).fetchone()[0]
                ),
                start=0,
                end=len("Migration discourse outside scope."),
                text="Migration discourse outside scope.",
                score=1.0,
            ),
        ),
        inspected_count=1,
        candidate_limit_reached=False,
    )
    reranker = _FixtureReranker()

    try:
        rerank_framework_context(
            project,
            scope_id,
            retrieval,
            ModelGateway({}, rerankers={reranker.provider_id: reranker}),
            provider_id=reranker.provider_id,
            model="fixture-v1",
            query="migration",
        )
    except ScopeViolationError:
        pass
    else:
        raise AssertionError("out-of-scope candidate was accepted")
    assert reranker.documents == []


def test_reranker_rejects_fabricated_text_and_invalid_provider_indices(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("selected.txt", "Migration discourse in scope.")
    scope_id = project.snapshot([revision])
    retrieval = retrieve_framework_context(
        project,
        scope_id,
        [],
        phenomenon="migration discourse",
        observation="policy framing",
        query_texts=("migration discourse",),
    )
    candidate = retrieval.candidates[0]
    fabricated = TargetedContextResult(
        scope_id,
        retrieval.query_terms,
        (
            TargetedContextCandidate(
                candidate.revision_id,
                candidate.paragraph_id,
                candidate.start,
                candidate.end,
                "Model-injected replacement text.",
                candidate.score,
            ),
        ),
        retrieval.inspected_count,
        retrieval.candidate_limit_reached,
    )
    reranker = _FixtureReranker()
    gateway = ModelGateway({}, rerankers={reranker.provider_id: reranker})

    try:
        rerank_framework_context(
            project,
            scope_id,
            fabricated,
            gateway,
            provider_id=reranker.provider_id,
            model="fixture-v1",
            query="migration",
        )
    except ScopeViolationError:
        pass
    else:
        raise AssertionError("fabricated candidate text was accepted")
    assert reranker.documents == []

    reranker.response = [RerankResult(99, 0.9)]
    try:
        rerank_framework_context(
            project,
            scope_id,
            retrieval,
            gateway,
            provider_id=reranker.provider_id,
            model="fixture-v1",
            query="migration",
        )
    except ModelResponseError:
        pass
    else:
        raise AssertionError("invalid provider index was accepted")


@pytest.mark.parametrize(
    "corruption", ["boolean_start", "non_finite_score", "fabricated_score"]
)
def test_reranker_rejects_corrupt_candidate_offsets_and_scores(tmp_path, corruption):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("selected.txt", "Migration discourse in scope.")
    scope_id = project.snapshot([revision])
    retrieval = retrieve_framework_context(
        project,
        scope_id,
        [],
        phenomenon="migration discourse",
        observation="policy framing",
        query_texts=("migration discourse",),
    )
    candidate = retrieval.candidates[0]
    corrupted = TargetedContextCandidate(
        candidate.revision_id,
        candidate.paragraph_id,
        cast(int, False) if corruption == "boolean_start" else candidate.start,
        candidate.end,
        candidate.text,
        (
            float("nan")
            if corruption == "non_finite_score"
            else candidate.score + 100.0
            if corruption == "fabricated_score"
            else candidate.score
        ),
    )
    malformed_retrieval = TargetedContextResult(
        retrieval.scope_id,
        retrieval.query_terms,
        (corrupted,),
        retrieval.inspected_count,
        retrieval.candidate_limit_reached,
    )
    reranker = _FixtureReranker()

    with pytest.raises(ScopeViolationError, match="invalid|canonical|lexical score"):
        rerank_framework_context(
            project,
            scope_id,
            malformed_retrieval,
            ModelGateway({}, rerankers={reranker.provider_id: reranker}),
            provider_id=reranker.provider_id,
            model="fixture-v1",
            query="migration discourse",
        )

    assert reranker.documents == []


def test_reranker_validates_top_n_even_when_lexical_retrieval_is_empty(tmp_path):
    project = Project.init(tmp_path / "project")
    _, revision = project.import_text("selected.txt", "A short document.")
    scope_id = project.snapshot([revision])
    retrieval = TargetedContextResult(scope_id, (), (), 0, False)
    reranker = _FixtureReranker()

    try:
        rerank_framework_context(
            project,
            scope_id,
            retrieval,
            ModelGateway({}, rerankers={reranker.provider_id: reranker}),
            provider_id=reranker.provider_id,
            model="fixture-v1",
            query="migration",
            top_n=True,
        )
    except ScopeViolationError:
        pass
    else:
        raise AssertionError("boolean top_n was accepted for an empty retrieval")
