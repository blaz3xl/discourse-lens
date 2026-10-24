from __future__ import annotations

import pytest

import discourselens.vectors as vector_module
from discourselens import Project
from discourselens.errors import ScopeViolationError
from discourselens.vectors import (
    MAX_VECTOR_SEARCH_RESULTS,
    VectorChunk,
    _search_vectors_with_candidate_count,
    build_vector_index,
    hybrid_search,
    hybrid_search_with_preset,
    search_vectors,
    semantic_search_with_preset,
)


class _EmbeddingFixture:
    provider_id = "cohere-embeddings"
    endpoint = "https://api.cohere.com"

    def __init__(self) -> None:
        self.query_calls = 0

    def embed(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        return [(1.0, 0.0) for _ in texts]

    def embed_query(self, model: str, texts: list[str]) -> list[tuple[float, ...]]:
        self.query_calls += 1
        return [(1.0, 0.0) for _ in texts]


@pytest.mark.parametrize("search_kind", ["vector", "hybrid"])
@pytest.mark.parametrize("limit", [0, MAX_VECTOR_SEARCH_RESULTS + 1, True])
def test_raw_vector_search_rejects_unbounded_result_limits(
    tmp_path, search_kind: str, limit: object
) -> None:
    project = Project.init(tmp_path / f"{search_kind}-{limit}")
    with pytest.raises(
        ScopeViolationError,
        match=f"limit must be between 1 and {MAX_VECTOR_SEARCH_RESULTS}",
    ):
        if search_kind == "vector":
            search_vectors(project, "missing-scope", ["index"], (1.0, 0.0), limit=limit)  # type: ignore[arg-type]
        else:
            hybrid_search(
                project,
                "missing-scope",
                ["index"],
                "query",
                (1.0, 0.0),
                limit=limit,  # type: ignore[arg-type]
            )
    project.close()


@pytest.mark.parametrize("search_kind", ["semantic", "hybrid"])
def test_preset_search_rejects_limit_before_preset_lookup_or_query_embedding(
    tmp_path, monkeypatch, search_kind: str
) -> None:
    project = Project.init(tmp_path / f"preset-{search_kind}")
    provider = _EmbeddingFixture()

    def forbidden_preset_lookup(*args, **kwargs):
        raise AssertionError("invalid result limit must be rejected before preset lookup")

    monkeypatch.setattr(project, "model_preset", forbidden_preset_lookup)
    function = (
        semantic_search_with_preset
        if search_kind == "semantic"
        else hybrid_search_with_preset
    )

    with pytest.raises(
        ScopeViolationError,
        match=f"limit must be between 1 and {MAX_VECTOR_SEARCH_RESULTS}",
    ):
        function(
            project,
            "missing-preset",
            "missing-scope",
            ["index"],
            "query",
            provider,
            limit=MAX_VECTOR_SEARCH_RESULTS + 1,
        )

    assert provider.query_calls == 0
    project.close()


def test_vector_search_accepts_the_documented_maximum(tmp_path) -> None:
    project = Project.init(tmp_path / "maximum-boundary")
    _, revision_id = project.import_text("source.txt", "A bounded vector result.")
    artifact_id = build_vector_index(
        project,
        revision_id,
        [VectorChunk("chunk-1", 0, len("A bounded vector result."), (1.0, 0.0))],
    )

    hits = search_vectors(
        project,
        project.snapshot([revision_id]),
        [artifact_id],
        (1.0, 0.0),
        limit=MAX_VECTOR_SEARCH_RESULTS,
    )

    assert len(hits) == 1
    assert hits[0].revision_id == revision_id
    project.close()


def test_vector_search_merges_revision_candidates_with_bounded_global_heap(
    tmp_path, monkeypatch
) -> None:
    project = Project.init(tmp_path / "bounded-global-merge")
    vector_sets = [
        ("a.txt", ((0.0, 1.0), (0.6, 0.8))),
        ("b.txt", ((0.8, 0.6), (0.9, 0.435889894))),
        ("c.txt", ((1.0, 0.0), (0.99, 0.14106736))),
    ]
    revisions: list[str] = []
    artifact_ids: list[str] = []
    for filename, vectors in vector_sets:
        _, revision_id = project.import_text(filename, "abcdef")
        revisions.append(revision_id)
        artifact_ids.append(
            build_vector_index(
                project,
                revision_id,
                [
                    VectorChunk(f"{filename}-{index}", index, index + 1, vector)
                    for index, vector in enumerate(vectors)
                ],
            )
        )

    observed_heap_sizes: list[int] = []
    original_heappush = vector_module.heapq.heappush

    def track_heap_size(heap, item) -> None:
        original_heappush(heap, item)
        observed_heap_sizes.append(len(heap))

    monkeypatch.setattr(vector_module.heapq, "heappush", track_heap_size)
    hits, candidate_count, metric = _search_vectors_with_candidate_count(
        project,
        project.snapshot(revisions),
        artifact_ids,
        (1.0, 0.0),
        limit=2,
    )

    assert candidate_count == 6  # Two local candidates inspected per revision.
    assert metric == "cosine"
    assert [hit.chunk_id for hit in hits] == ["c.txt-0", "c.txt-1"]
    assert observed_heap_sizes
    assert max(observed_heap_sizes) <= 2
    project.close()
