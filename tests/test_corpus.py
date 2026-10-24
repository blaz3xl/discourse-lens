from typing import cast

import pytest

from discourselens import Project
from discourselens.corpus import (
    _texts,
    collocation_analysis,
    collocation_association_evidence,
    collocations,
    compare_documents,
    compare_frequency,
    compare_frequency_page,
    compare_named_corpora,
    comparison_statistic_evidence,
    document_dispersion,
    document_dispersion_evidence,
    frequency,
    frequency_page,
    frequency_statistic_evidence,
    keyness_log_likelihood,
    keyness_page,
    kwic,
    kwic_evidence,
    kwic_page,
    lexical_bundle_statistic_evidence,
    lexical_bundles,
    ngrams,
)
from discourselens.errors import DocumentError, EvidenceValidationError, ScopeViolationError
from discourselens.types import CoverageMode


def test_scope_bound_deterministic_corpus_operations(tmp_path):
    project = Project.init(tmp_path / "p")
    _, a = project.import_text("a", "Apples are green. Apples grow.")
    project.import_text("b", "Elephants live in Africa.")
    scope = project.snapshot([a])
    assert frequency(project, scope)["apples"] == 2
    assert ("apples", "are") in ngrams(project, scope, 2)
    assert len(kwic(project, scope, "apples")) == 2
    assert not kwic(project, scope, "elephants")
    assert collocations(project, scope, "apples")["green"] == 2


def test_frequency_pages_have_stable_tie_order_and_exact_population_metadata(tmp_path):
    project = Project.init(tmp_path / "frequency-pagination")
    _, revision = project.import_text("one.txt", "zulu alpha beta alpha")
    scope = project.snapshot([revision])

    first = frequency_page(project, scope, limit=2)
    second = project.corpus_frequency_page(scope, offset=2, limit=2)

    assert first["rows"] == [{"token": "alpha", "count": 2}, {"token": "beta", "count": 1}]
    assert second["rows"] == [{"token": "zulu", "count": 1}]
    assert first["total_types"] == second["total_types"] == 3
    assert first["total_tokens"] == second["total_tokens"] == 4
    assert first["has_more"] and not second["has_more"]


def test_frequency_v2_coalesces_nfc_and_nfd_surface_forms(tmp_path):
    project = Project.init(tmp_path / "frequency-unicode-normalization")
    _, revision = project.import_text("unicode.txt", "café cafe\u0301 CAFÉ")
    scope = project.snapshot([revision])

    counts = frequency(project, scope)
    page = frequency_page(project, scope)
    evidence = frequency_statistic_evidence(project, scope, "cafe\u0301")

    assert counts == {"café": 3}
    assert page["rows"] == [{"token": "café", "count": 3}]
    assert page["total_tokens"] == 3
    assert evidence["term"] == "café"
    assert evidence["count"] == 3
    metadata = project.evidence_content(str(evidence["evidence_id"]))
    provenance = cast(dict[str, object], metadata["provenance"])
    assert provenance["calculation_version"] == "surface-frequency-v2"
    assert provenance["normalization"] == "NFC-after-casefold-v1"


def test_v2_frequency_ngrams_and_collocations_share_unicode_token_counts(tmp_path):
    project = Project.init(tmp_path / "corpus-v2-tokenizer-parity")
    text = "Café red cafe\u0301 blue can't can’t state-of-art co-op"
    _, revision = project.import_text("unicode.txt", text)
    scope = project.snapshot([revision])

    frequencies = frequency(project, scope)
    onegrams = ngrams(project, scope, 1)
    collocates = collocations(project, scope, "CAFE\u0301", window=1)
    association = collocation_analysis(
        project, scope, "cafe\u0301", window=1, min_frequency=1
    )

    assert frequencies == {
        "café": 2,
        "red": 1,
        "blue": 1,
        "can't": 1,
        "can’t": 1,
        "state-of-art": 1,
        "co-op": 1,
    }
    assert onegrams == {(term,): count for term, count in frequencies.items()}
    assert sum(onegrams.values()) == sum(frequencies.values()) == 8
    assert collocates == {"red": 2, "blue": 1}
    assert association["population_tokens"] == sum(frequencies.values())
    assert association["focus_frequency"] == frequencies["café"] == 2
    assert association["tokenizer"] == "unicode-surface-tokenizer-v2"
    assert association["normalization"] == "NFC-after-casefold-v1"
    assert {
        row["term"]: row["pair_frequency"] for row in association["rows"]
    } == collocates


def test_kwic_v2_preserves_nfd_offsets_and_exact_canonical_excerpt(tmp_path):
    project = Project.init(tmp_path / "kwic-unicode-offsets")
    source_text = "🐦 cafe\u0301—migration"
    _, revision = project.import_text("unicode.txt", source_text)
    scope = project.snapshot([revision])

    rows = kwic(project, scope, "café")

    assert len(rows) == 1
    assert rows[0]["keyword"] == "cafe\u0301"
    assert rows[0]["start_offset"] == 2
    assert rows[0]["end_offset"] == 7
    assert source_text[rows[0]["start_offset"] : rows[0]["end_offset"]] == "cafe\u0301"
    evidence = kwic_evidence(
        project, scope, revision, "café", rows[0]["start_offset"], rows[0]["end_offset"]
    )
    assert evidence["exact_text"] == "cafe\u0301"
    assert project.render_citation(str(evidence["evidence_id"])) == "[unicode.txt · ¶1]"


def test_comparison_v1_keeps_its_historical_tokenizer_with_nfd_text(tmp_path):
    project = Project.init(tmp_path / "comparison-tokenizer-v1")
    _, left = project.import_text("left.txt", "cafe\u0301")
    _, right = project.import_text("right.txt", "cafe")

    rows = compare_frequency(project, project.snapshot([left]), project.snapshot([right]))

    assert rows == [
        {
            "token": "cafe",
            "left_count": 1,
            "right_count": 1,
            "left_rate": 1.0,
            "right_rate": 1.0,
            "rate_difference": 0.0,
        }
    ]


def test_out_of_range_pages_do_not_materialize_ranked_rows(tmp_path, monkeypatch):
    import discourselens.corpus as corpus_module

    project = Project.init(tmp_path / "out-of-range-pages")
    _, left_revision = project.import_text("left.txt", "alpha beta alpha")
    _, right_revision = project.import_text("right.txt", "gamma delta beta")
    left_scope = project.snapshot([left_revision])
    right_scope = project.snapshot([right_revision])

    def unexpected_ranking(*args, **kwargs):
        raise AssertionError("an out-of-range page should not rank the full result set")

    monkeypatch.setattr(corpus_module, "nsmallest", unexpected_ranking)
    offset = 10**100

    frequencies = frequency_page(project, left_scope, offset=offset, limit=2)
    frequency_comparison = compare_frequency_page(
        project, left_scope, right_scope, offset=offset, limit=2
    )
    keyness_comparison = keyness_page(project, left_scope, right_scope, offset=offset, limit=2)

    assert frequencies["total_types"] == 2
    assert frequencies["total_tokens"] == 3
    assert frequencies["rows"] == []
    assert frequencies["has_more"] is False
    for page in (frequency_comparison, keyness_comparison):
        assert page["total_rows"] == 4
        assert page["rows"] == []
        assert page["has_more"] is False


def test_scoped_text_reader_loads_one_revision_at_a_time():
    class FakeDatabase:
        def execute(self, _query, _parameters):
            return self

        def fetchone(self):
            return ('["revision-a","revision-b"]',)

    class FakeProject:
        db = FakeDatabase()

        def __init__(self):
            self.read_revisions: list[str] = []

        def revision_text(self, revision_id: str) -> str:
            self.read_revisions.append(revision_id)
            return f"text for {revision_id}"

    project = FakeProject()
    text_stream = _texts(project, "scope")
    assert project.read_revisions == []
    assert next(text_stream) == ("revision-a", "text for revision-a")
    assert project.read_revisions == ["revision-a"]
    assert next(text_stream) == ("revision-b", "text for revision-b")
    assert project.read_revisions == ["revision-a", "revision-b"]
    with pytest.raises(StopIteration):
        next(text_stream)


def test_kwic_offsets_resolve_exact_unicode_text_into_citable_evidence(tmp_path):
    project = Project.init(tmp_path / "kwic-evidence")
    _, revision = project.import_text("speech.txt", "“Café migración”—migración.")
    scope = project.snapshot([revision])

    rows = project.corpus_kwic(scope, "migración", window=1)

    assert len(rows) == 2
    assert rows[0]["revision_id"] == revision
    assert rows[0]["keyword"] == "migración"
    assert rows[0]["start_offset"] == 6
    assert rows[0]["end_offset"] == 15
    span_id = project.create_span(revision, rows[0]["start_offset"], rows[0]["end_offset"])
    assert project.span_text(span_id) == rows[0]["keyword"]
    evidence_id = project.textual_evidence(span_id)
    assert project.render_citation(evidence_id) == "[speech.txt · ¶1]"


def test_kwic_normalizes_case_and_query_punctuation_to_the_surface_token(tmp_path):
    project = Project.init(tmp_path / "kwic-normalized-query")
    _, revision = project.import_text("speech.txt", "“Café migración”—migración.")
    scope = project.snapshot([revision])

    rows = kwic(project, scope, "MIGRACIÓN,")

    assert [row["keyword"] for row in rows] == ["migración", "migración"]
    assert [row["start_offset"] for row in rows] == [6, 17]
    page = kwic_page(project, scope, "(Migración)")
    assert page["total_count"] == 2


@pytest.mark.parametrize("query", ["", "two words", "!!!"])
def test_kwic_rejects_queries_without_exactly_one_surface_token(tmp_path, query):
    project = Project.init(tmp_path / "kwic-invalid-query")
    _, revision = project.import_text("speech.txt", "Migration matters.")
    scope = project.snapshot([revision])

    with pytest.raises(ScopeViolationError, match="KWIC requires exactly one surface token"):
        kwic(project, scope, query)

    with pytest.raises(ScopeViolationError, match="KWIC requires exactly one surface token"):
        kwic_page(project, scope, query)


def test_kwic_pages_are_bounded_ordered_and_report_full_result_count(tmp_path):
    project = Project.init(tmp_path / "kwic-pagination")
    _, first = project.import_text("first.txt", "Migration one. Migration two.")
    _, second = project.import_text("second.txt", "Migration three.")
    scope = project.snapshot([first, second])

    page_one = kwic_page(project, scope, "migration", window=1, offset=0, limit=2)
    page_two = project.corpus_kwic_page(scope, "migration", window=1, offset=2, limit=2)

    assert page_one["total_count"] == 3
    assert page_one["has_more"] is True
    assert len(page_one["rows"]) == 2
    assert page_two["total_count"] == 3
    assert page_two["has_more"] is False
    assert len(page_two["rows"]) == 1
    assert [row["revision_id"] for row in page_one["rows"] + page_two["rows"]] == [
        first,
        first,
        second,
    ]
    assert [row["start_offset"] for row in page_one["rows"] + page_two["rows"]] == [
        0,
        15,
        0,
    ]


def test_kwic_iterator_reads_only_a_sliding_token_window_before_first_result(tmp_path, monkeypatch):
    import discourselens.corpus as corpus_module

    project = Project.init(tmp_path / "kwic-streaming")
    _, revision = project.import_text("one.txt", "target alpha beta gamma delta")
    scope = project.snapshot([revision])
    original_tokenizer = corpus_module._tokenize_v2
    consumed: list[str] = []

    def tracking_tokenizer(text):
        for token in original_tokenizer(text):
            consumed.append(token.surface)
            yield token

    monkeypatch.setattr(corpus_module, "_tokenize_v2", tracking_tokenizer)
    rows = corpus_module._iter_kwic_rows(project, scope, "target", 2)

    first = next(rows)

    assert first["keyword"] == "target"
    assert first["right"] == "alpha beta"
    # The query token is normalized first; the document iterator then reads only
    # the focus plus the requested right context before yielding the first row.
    assert consumed == ["target", "target", "alpha", "beta"]


@pytest.mark.parametrize(("offset", "limit"), [(-1, 1), (0, 0), (0, 1001)])
def test_kwic_page_rejects_invalid_pagination(tmp_path, offset, limit):
    project = Project.init(tmp_path / f"kwic-invalid-{offset}-{limit}")
    _, revision = project.import_text("one.txt", "Migration.")

    with pytest.raises(ScopeViolationError, match="KWIC offset"):
        kwic_page(project, project.snapshot([revision]), "migration", offset=offset, limit=limit)


def test_kwic_evidence_accepts_only_a_hit_inside_the_snapshot(tmp_path):
    project = Project.init(tmp_path / "kwic-evidence-scope")
    _, included = project.import_text("included.txt", "Café migración.")
    _, excluded = project.import_text("excluded.txt", "migración elsewhere")
    scope = project.snapshot([included])

    result = kwic_evidence(project, scope, included, "migración", 5, 14)

    assert result["exact_text"] == "migración"
    assert result["citation"] == "[included.txt · ¶1]"
    evidence = project.db.execute(
        "SELECT span_id FROM evidence_artifacts WHERE id=?", (result["evidence_id"],)
    ).fetchone()
    assert evidence is not None
    assert project.span_text(evidence["span_id"]) == "migración"
    evidence_metadata = project.evidence_content(result["evidence_id"])
    assert evidence_metadata["acquisition_method"] == "corpus_kwic"
    assert evidence_metadata["calculation_version"] == "surface-kwic-v2"
    assert evidence_metadata["coverage_mode"] == "EXHAUSTIVE"
    assert evidence_metadata["query"] == "migración"
    assert evidence_metadata["scope_snapshot_id"] == scope
    assert evidence_metadata["tokenizer"] == "unicode-surface-tokenizer-v2"
    coverage = project.coverage_descriptor(result["coverage_id"])
    assert coverage.scope_snapshot_id == scope
    assert coverage.population_count == 2
    with pytest.raises(EvidenceValidationError, match="not a KWIC hit"):
        kwic_evidence(project, scope, excluded, "migración", 0, 9)
    with pytest.raises(EvidenceValidationError, match="not a KWIC hit"):
        kwic_evidence(project, scope, included, "elephants", 5, 14)


@pytest.mark.parametrize("query", ["migration", "policy"])
def test_kwic_evidence_rejects_substrings_inside_longer_surface_tokens(tmp_path, query):
    project = Project.init(tmp_path / f"kwic-substring-{query}")
    _, revision = project.import_text("included.txt", "premigration policyholder")
    scope = project.snapshot([revision])
    start_offset = 3 if query == "migration" else 13
    end_offset = start_offset + len(query)

    with pytest.raises(EvidenceValidationError, match="not a KWIC hit"):
        kwic_evidence(project, scope, revision, query, start_offset, end_offset)


def test_kwic_evidence_validates_selected_hit_without_materializing_all_kwic_rows(
    tmp_path, monkeypatch
):
    import discourselens.corpus as corpus_module

    project = Project.init(tmp_path / "bounded-kwic-evidence")
    _, revision = project.import_text("repeat.txt", "Migration migration policy.")
    scope = project.snapshot([revision])

    def unexpected_full_kwic(*args, **kwargs):
        raise AssertionError("evidence validation must not materialize every KWIC row")

    monkeypatch.setattr(corpus_module, "kwic", unexpected_full_kwic)

    result = kwic_evidence(project, scope, revision, "migration", 0, 9)

    evidence = project.evidence_content(result["evidence_id"])
    assert result["exact_text"] == "Migration"
    assert evidence["candidate_count"] == 2
    assert evidence["inspected_count"] == 3


def test_collocation_association_scores_are_reproducible_and_scope_bound(tmp_path):
    project = Project.init(tmp_path / "association")
    _, selected = project.import_text("selected.txt", "apple red apple red apple blue")
    project.import_text("outside.txt", "target outside")
    scope = project.snapshot([selected])

    result = collocation_analysis(project, scope, "apple", window=1, min_frequency=1)

    assert result["calculation_version"] == "surface-window-association-v2"
    assert result["scope_id"] == scope
    assert result["focus_frequency"] == 3
    assert result["population_tokens"] == 6
    assert result["pair_opportunities"] == 5
    assert result["interpretation_note"].endswith(
        "not significance tests; expected counts condition on non-focus tokens and "
        "the actual in-window positions."
    )
    rows = {row["term"]: row for row in result["rows"]}
    assert rows["red"]["pair_frequency"] == 4
    assert rows["red"]["token_frequency"] == 2
    assert rows["red"]["expected_frequency"] == pytest.approx(10 / 3)
    assert rows["red"]["mutual_information"] == pytest.approx(0.2630344)
    assert rows["red"]["t_score"] == pytest.approx(1 / 3)
    assert "outside" not in rows
    assert collocation_analysis(project, scope, "apple", window=1, min_frequency=1) == result


def test_collocation_association_does_not_pair_across_revision_boundaries(tmp_path):
    project = Project.init(tmp_path / "association-boundary")
    _, left = project.import_text("left.txt", "target")
    _, right = project.import_text("right.txt", "outside")
    result = collocation_analysis(
        project, project.snapshot([left, right]), "target", window=1, min_frequency=1
    )
    assert result["focus_frequency"] == 1
    assert result["rows"] == []


def test_collocation_expectation_counts_only_available_edge_slots_per_revision(tmp_path):
    project = Project.init(tmp_path / "association-edge-population")
    _, left = project.import_text("left.txt", "a x")
    _, right = project.import_text("right.txt", "b c")

    result = collocation_analysis(
        project, project.snapshot([left, right]), "x", window=1, min_frequency=1
    )

    # N=4 tokens, F=1 focus, M=1 available neighbor slot, f(a)=1, O(a)=1.
    # Therefore E=1*(1/3), MI=log2(3), and t=(1-1/3)/sqrt(1).
    assert result["population_tokens"] == 4
    assert result["focus_frequency"] == 1
    assert result["pair_opportunities"] == 1
    assert [row["term"] for row in result["rows"]] == ["a"]
    row = result["rows"][0]
    assert row["pair_frequency"] == 1
    assert row["token_frequency"] == 1
    assert row["expected_frequency"] == pytest.approx(1 / 3)
    assert row["mutual_information"] == pytest.approx(1.5849625)
    assert row["t_score"] == pytest.approx(2 / 3)


def test_collocation_association_evidence_persists_exact_statistic_and_coverage(tmp_path):
    project = Project.init(tmp_path / "association-evidence")
    _, revision = project.import_text("sample.txt", "apple red apple red apple blue")
    scope = project.snapshot([revision])

    result = collocation_association_evidence(
        project, scope, "apple", "red", window=1, min_frequency=1
    )

    assert result["scope_id"] == scope
    assert result["pair_frequency"] == 4
    assert result["pair_opportunities"] == 5
    assert result["mutual_information"] == pytest.approx(0.2630344)
    coverage = project.coverage_descriptor(result["coverage_id"])
    assert coverage.coverage_mode is CoverageMode.EXHAUSTIVE
    assert coverage.scope_snapshot_id == scope
    assert coverage.population_count == result["pair_opportunities"]
    payload = project.evidence_content(result["evidence_id"])
    assert payload["value"] == pytest.approx(0.2630344)
    assert payload["scope_snapshot_id"] == scope
    provenance = cast(dict[str, object], payload["provenance"])
    assert provenance["pair_frequency"] == 4
    assert provenance["pair_opportunities"] == 5
    assert provenance["t_score"] == pytest.approx(1 / 3)
    assert provenance["tokenizer"] == "unicode-surface-tokenizer-v2"
    assert provenance["normalization"] == "NFC-after-casefold-v1"

    with pytest.raises(ScopeViolationError, match="pair-frequency threshold"):
        collocation_association_evidence(
            project, scope, "apple", "absent", window=1, min_frequency=1
        )


@pytest.mark.parametrize("query", ["", "two tokens"])
def test_collocation_association_requires_one_surface_token(tmp_path, query):
    project = Project.init(tmp_path / "association-query")
    _, revision = project.import_text("text.txt", "one two")
    with pytest.raises(ScopeViolationError, match="exactly one surface token"):
        collocation_analysis(project, project.snapshot([revision]), query)


def test_basic_collocations_match_association_query_normalization(tmp_path):
    project = Project.init(tmp_path / "collocation-normalized-query")
    _, revision = project.import_text("sample.txt", "Apple, red apple BLUE")
    scope = project.snapshot([revision])

    counts = collocations(project, scope, "APPLE,", window=1)
    analysis = collocation_analysis(project, scope, "apple", window=1, min_frequency=1)

    assert counts == {row["term"]: row["pair_frequency"] for row in analysis["rows"]}
    assert counts == {"red": 2, "blue": 1}


@pytest.mark.parametrize("query", ["", "two tokens", "!!!"])
def test_basic_collocations_reject_queries_without_exactly_one_surface_token(tmp_path, query):
    project = Project.init(tmp_path / "collocation-invalid-query")
    _, revision = project.import_text("sample.txt", "apple red")

    with pytest.raises(
        ScopeViolationError, match="collocations requires exactly one surface token"
    ):
        collocations(project, project.snapshot([revision]), query)


def test_lexical_bundles_are_deterministic_scoped_and_report_dispersion(tmp_path):
    project = Project.init(tmp_path / "p")
    _, first = project.import_text("a", "the public must act; the public must respond")
    _, second = project.import_text("b", "the public must listen")
    project.import_text("outside", "the public must disappear")
    scope = project.snapshot([first, second])

    result = lexical_bundles(project, scope, size=3, min_frequency=2)

    assert result["scope_id"] == scope
    assert result["calculation_version"] == "surface-lexical-bundles-v1"
    assert result["candidate_count"] == len(result["rows"])
    assert result["truncated"] is False
    row = next(item for item in result["rows"] if item["bundle"] == "the public must")
    assert row["frequency"] == 3
    assert row["document_frequency"] == 2
    assert all("disappear" not in item["bundle"] for item in result["rows"])
    assert result == lexical_bundles(project, scope, size=3, min_frequency=2)


def test_lexical_bundles_filters_and_bounds_results(tmp_path):
    project = Project.init(tmp_path / "p")
    _, one = project.import_text("a", "we should act we should respond")
    _, two = project.import_text("b", "we should act")
    scope = project.snapshot([one, two])

    result = lexical_bundles(
        project,
        scope,
        size=2,
        min_frequency=2,
        min_document_frequency=2,
        limit=1,
    )

    assert result["candidate_count"] >= 1
    assert len(result["rows"]) == 1
    assert result["truncated"] is (result["candidate_count"] > 1)
    assert result["rows"][0]["document_frequency"] == 2


def test_lexical_bundle_evidence_is_exhaustive_and_reopenable(tmp_path):
    root = tmp_path / "p"
    project = Project.init(root)
    _, first = project.import_text("a", "the public must act; the public must respond")
    _, second = project.import_text("b", "the public should listen")
    scope = project.snapshot([first, second])

    result = lexical_bundle_statistic_evidence(project, scope, "The public must", size=3)

    project.close()
    project = Project.open(root)
    evidence = project.evidence_content(result["evidence_id"])
    coverage = project.coverage_descriptor(result["coverage_id"])
    assert result["bundle"] == "the public must"
    assert result["count"] == 2
    assert result["population_count"] == 8
    assert evidence["value"] == 2
    assert evidence["scope_snapshot_id"] == scope
    assert coverage.coverage_mode == CoverageMode.EXHAUSTIVE
    assert coverage.population_count == result["population_count"]
    assert coverage.scope_snapshot_id == scope
    project.close()


def test_surface_frequency_uses_lazy_token_matches(tmp_path, monkeypatch):
    from discourselens import corpus

    project = Project.init(tmp_path / "lazy-frequency")
    _, revision = project.import_text("sample.txt", "Café café migration")
    scope = project.snapshot([revision])

    original_tokenizer = corpus._tokenize_v2

    def finditer_only(text: str):
        yield from original_tokenizer(text)

    monkeypatch.setattr(corpus, "_tokenize_v2", finditer_only)

    counts = corpus.frequency(project, scope)

    assert counts == {"café": 2, "migration": 1}


@pytest.mark.parametrize(
    "options",
    [
        {"size": 1},
        {"size": 11},
        {"min_frequency": 0},
        {"min_document_frequency": 0},
        {"limit": 0},
        {"limit": 1001},
    ],
)
def test_lexical_bundles_rejects_invalid_parameters(tmp_path, options):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a", "the public must act")

    with pytest.raises(ScopeViolationError):
        lexical_bundles(project, project.snapshot([revision]), **options)


def test_frequency_statistic_evidence_has_exhaustive_scope_provenance(tmp_path):
    project = Project.init(tmp_path / "p")
    _, revision = project.import_text("a.txt", "Apples are green. Apples grow.")
    scope = project.snapshot([revision])

    result = frequency_statistic_evidence(project, scope, "Apples")

    assert result["count"] == 2
    assert result["population_count"] == 5
    coverage = project.coverage_descriptor(str(result["coverage_id"]))
    evidence = project.evidence_content(str(result["evidence_id"]))
    assert coverage.coverage_mode.value == "EXHAUSTIVE"
    assert coverage.scope_snapshot_id == scope
    assert evidence["value"] == 2 and evidence["scope_snapshot_id"] == scope
    provenance = cast(dict[str, object], evidence["provenance"])
    assert provenance["coverage_id"] == result["coverage_id"]
    assert provenance["population_tokens"] == 5
    assert provenance["calculation_version"] == "surface-frequency-v2"
    assert provenance["tokenizer"] == "unicode-surface-tokenizer-v2"


def test_document_dispersion_reports_counts_rates_and_zero_occurrence_documents(tmp_path):
    project = Project.init(tmp_path / "p")
    _, first = project.import_text("first.txt", "Migration migration policy.")
    _, second = project.import_text("second.txt", "Migration economy.")
    _, third = project.import_text("third.txt", "Global economy.")
    scope = project.snapshot([first, second, third])

    result = document_dispersion(project, scope, "Migration")

    assert result["term"] == "migration"
    assert result["document_count"] == 3
    assert result["document_frequency"] == 2
    assert result["document_range"] == 2 / 3
    assert result["total_occurrences"] == 3
    assert result["population_tokens"] == 7
    rows = result["documents"]
    assert [row["occurrences"] for row in rows] == [2, 1, 0]
    assert [row["per_1000_tokens"] for row in rows] == [pytest.approx(2000 / 3), 500, 0]


def test_document_dispersion_uses_nfc_surface_tokens_like_frequency(tmp_path):
    project = Project.init(tmp_path / "dispersion-nfc")
    text = "cafe\u0301 café"
    _, revision = project.import_text("nfd.txt", text)
    scope = project.snapshot([revision])

    dispersion = document_dispersion(project, scope, "cafe\u0301")
    frequencies = project.corpus_frequency(scope)

    assert dispersion["term"] == "café"
    assert dispersion["calculation_version"] == "surface-document-dispersion-v2"
    assert dispersion["population_tokens"] == sum(frequencies.values()) == 2
    assert dispersion["total_occurrences"] == frequencies["café"] == 2


def test_document_dispersion_evidence_preserves_exhaustive_scope_distribution(tmp_path):
    project = Project.init(tmp_path / "p")
    _, first = project.import_text("first.txt", "Migration policy.")
    _, second = project.import_text("second.txt", "Migration economy.")
    scope = project.snapshot([first, second])

    result = document_dispersion_evidence(project, scope, "migration")

    evidence = project.evidence_content(str(result["evidence_id"]))
    coverage = project.coverage_descriptor(str(result["coverage_id"]))
    provenance = cast(dict[str, object], evidence["provenance"])
    assert evidence["value"] == 2
    assert evidence["scope_snapshot_id"] == scope
    assert provenance["coverage_id"] == result["coverage_id"]
    assert provenance["document_frequency"] == 2
    assert len(cast(list[object], provenance["documents"])) == 2
    assert coverage.coverage_mode.value == "EXHAUSTIVE"
    assert coverage.scope_snapshot_id == scope


def test_named_corpus_creates_an_immutable_explicit_scope(tmp_path):
    project = Project.init(tmp_path / "p")
    _, one = project.import_text("one.txt", "Apples are green.")
    _, two = project.import_text("two.txt", "Elephants live in Africa.")
    corpus = project.create_corpus("comparison", "Two selected documents")
    project.add_revision_to_corpus(corpus, one)
    scope = project.corpus_snapshot(corpus)
    project.add_revision_to_corpus(corpus, two)

    assert project.scope_revisions(scope) == [one]
    assert set(project.scope_revisions(project.corpus_snapshot(corpus))) == {one, two}
    assert project.list_corpora()[0]["revision_count"] == 2


def test_subcorpus_is_an_explicit_frozen_subset_of_its_parent_corpus(tmp_path):
    project = Project.init(tmp_path / "p")
    _, apples = project.import_text("apples.txt", "Apples are green.")
    _, elephants = project.import_text("elephants.txt", "Elephants live in Africa.")
    corpus = project.create_corpus("all")
    project.add_revision_to_corpus(corpus, apples)
    project.add_revision_to_corpus(corpus, elephants)

    subcorpus = project.create_subcorpus(corpus, "fruit", [apples])
    project.add_revision_to_corpus(corpus, apples)
    scope = project.subcorpus_snapshot(subcorpus)

    assert project.scope_revisions(scope) == [apples]
    assert project.list_subcorpora(corpus)[0]["revision_count"] == 1


def test_frequency_comparison_is_normalized_and_descriptive(tmp_path):
    project = Project.init(tmp_path / "p")
    _, left = project.import_text("left.txt", "Apples apples green.")
    _, right = project.import_text("right.txt", "Elephants green.")
    rows = compare_frequency(project, project.snapshot([left]), project.snapshot([right]))
    apple = next(row for row in rows if row["token"] == "apples")
    assert apple["left_count"] == 2 and apple["right_count"] == 0


def test_comparison_ties_keep_lexical_deterministic_order(tmp_path):
    project = Project.init(tmp_path / "comparison-ties")
    _, left = project.import_text("left.txt", "beta alpha")
    _, right = project.import_text("right.txt", "gamma delta")
    left_scope = project.snapshot([left])
    right_scope = project.snapshot([right])
    rows = compare_frequency(project, left_scope, right_scope)
    keyness_rows = keyness_log_likelihood(project, left_scope, right_scope)

    assert [row["token"] for row in rows] == ["alpha", "beta", "delta", "gamma"]
    assert [row["token"] for row in keyness_rows] == ["alpha", "beta", "delta", "gamma"]


def test_surface_comparison_pages_match_full_results_and_have_stable_ties(tmp_path):
    project = Project.init(tmp_path / "comparison-pages")
    _, left = project.import_text("left.txt", "beta alpha alpha")
    _, right = project.import_text("right.txt", "gamma delta alpha")
    left_scope = project.snapshot([left])
    right_scope = project.snapshot([right])

    for metric, full, page_reader in (
        ("frequency", compare_frequency(project, left_scope, right_scope), compare_frequency_page),
        ("keyness", keyness_log_likelihood(project, left_scope, right_scope), keyness_page),
    ):
        first = page_reader(project, left_scope, right_scope, offset=0, limit=2)
        second = page_reader(project, left_scope, right_scope, offset=2, limit=2)
        assert first["metric"] == second["metric"] == metric
        assert first["total_rows"] == second["total_rows"] == len(full)
        assert first["has_more"] is True
        assert second["has_more"] is False
        assert first["rows"] + second["rows"] == full

    assert project.corpus_compare_frequency_page(left_scope, right_scope, limit=1)["rows"] == [
        compare_frequency(project, left_scope, right_scope)[0]
    ]


def test_keyness_is_deterministic_and_requires_two_explicit_scopes(tmp_path):
    project = Project.init(tmp_path / "p")
    _, left = project.import_text("left.txt", "apples apples apples green")
    _, right = project.import_text("right.txt", "elephants elephants green")
    rows = keyness_log_likelihood(project, project.snapshot([left]), project.snapshot([right]))
    apples = next(row for row in rows if row["token"] == "apples")
    elephants = next(row for row in rows if row["token"] == "elephants")
    assert float(apples["log_likelihood"]) > 0 > float(elephants["log_likelihood"])


def test_named_corpus_comparison_records_immutable_scopes(tmp_path):
    project = Project.init(tmp_path / "p")
    _, apples = project.import_text("apples.txt", "apples apples green")
    _, elephants = project.import_text("elephants.txt", "elephants elephants green")
    left = project.create_corpus("left")
    right = project.create_corpus("right")
    project.add_revision_to_corpus(left, apples)
    project.add_revision_to_corpus(right, elephants)

    result = compare_named_corpora(project, left, right, metric="keyness")

    assert result["metric"] == "keyness"
    persisted = project.comparison_run(str(result["comparison_id"]))
    assert persisted["calculation_version"] == "surface-comparison-v2"
    assert persisted["left_scope_id"] == result["left_scope_id"]
    assert persisted["right_scope_id"] == result["right_scope_id"]
    assert persisted["result_hash"]
    assert project.scope_revisions(str(result["left_scope_id"])) == [apples]
    assert project.scope_revisions(str(result["right_scope_id"])) == [elephants]
    assert isinstance(result["rows"], list)
    apples_row = next(row for row in result["rows"] if row["token"] == "apples")
    assert float(apples_row["log_likelihood"]) > 0


def test_named_corpus_comparison_can_create_scope_bound_finding(tmp_path):
    project = Project.init(tmp_path / "named-comparison-finding")
    _, left_revision = project.import_text("left.txt", "migration migration policy")
    _, right_revision = project.import_text("right.txt", "migration public discourse")
    left_corpus = project.create_corpus("left")
    right_corpus = project.create_corpus("right")
    project.add_revision_to_corpus(left_corpus, left_revision)
    project.add_revision_to_corpus(right_corpus, right_revision)

    comparison = compare_named_corpora(project, left_corpus, right_corpus, metric="frequency")
    result = project.create_comparative_finding(
        str(comparison["comparison_id"]),
        "MIGRATION",
        comparability_basis="Both corpora contain generated examples of comparable length and genre.",
    )

    assert result["validation"] == {
        "valid": True,
        "errors": [],
        "semantic_entailment_assessed": False,
    }
    assert project.scope_revisions(str(comparison["left_scope_id"])) == [left_revision]
    assert project.scope_revisions(str(comparison["right_scope_id"])) == [right_revision]
    run = project.analysis_run_metadata(str(result["analysis_run_id"]))
    assert run["revision_ids"] == sorted([left_revision, right_revision])
    assert project.validate_finding(str(result["finding_id"])).valid


def test_comparison_statistic_evidence_normalizes_casefolded_token_for_lookup(tmp_path):
    project = Project.init(tmp_path / "comparison-evidence-casefold")
    _, left = project.import_text("left.txt", "CAFÉ café policy")
    _, right = project.import_text("right.txt", "café public")
    comparison = compare_documents(project, left, right, metric="frequency")
    union_scope = project.snapshot([left, right])
    run_id = project.create_run(union_scope, {})

    result = comparison_statistic_evidence(
        project,
        run_id,
        str(comparison["comparison_id"]),
        "CAFÉ",
        comparability_basis="Both selected documents are examples from the same declared genre.",
    )

    assert result["token"] == "café"
    assert result["scope_id"] == union_scope
    evidence = project.evidence_content(str(result["evidence_id"]))
    provenance = cast(dict[str, object], evidence["provenance"])
    selected_row = cast(dict[str, object], provenance["selected_row"])
    assert selected_row["token"] == "café"


def test_persisted_comparison_streams_ranked_rows_and_returns_a_bounded_page(tmp_path, monkeypatch):
    from discourselens import corpus

    project = Project.init(tmp_path / "streamed-comparison")
    left_text = " ".join([*(f"alpha{index}" for index in range(600)), "shared"])
    right_text = " ".join([*(f"beta{index}" for index in range(600)), "shared"])
    _, left = project.import_text("left.txt", left_text)
    _, right = project.import_text("right.txt", right_text)

    def reject_legacy_materialized_comparison(*args, **kwargs):
        pytest.fail("persisted comparisons must not build the legacy full row list")

    monkeypatch.setattr(corpus, "compare_frequency", reject_legacy_materialized_comparison)
    result = corpus.compare_documents(project, left, right, metric="frequency", limit=7)

    assert len(cast(list[object], result["rows"])) == 7
    assert result["total_rows"] == 1_201
    assert result["has_more"] is True
    comparison_id = str(result["comparison_id"])
    persisted = project.comparison_run(comparison_id)
    assert persisted["storage_kind"] == "rows-v1"
    assert persisted["result_json"] == "[]"
    assert (
        project.db.execute(
            "SELECT COUNT(*) FROM comparison_result_rows WHERE comparison_id=?", (comparison_id,)
        ).fetchone()[0]
        == 1_201
    )

    from discourselens.reporting import comparison_payload_page

    last_page = comparison_payload_page(project, comparison_id, offset=1_198, limit=10)
    assert len(last_page["rows"]) == 3
    assert last_page["pagination"]["rows"]["total_count"] == 1_201
    assert last_page["pagination"]["rows"]["has_more"] is False
    project.close()
    with Project.open(tmp_path / "streamed-comparison") as reopened:
        reopened_page = reopened.comparison_rows_page(comparison_id, offset=1_198, limit=10)
        assert reopened_page["rows"] == last_page["rows"]
        assert reopened_page["result_hash_verified"] is True


def test_streamed_comparison_hash_matches_legacy_canonical_json(tmp_path):
    import hashlib
    import json

    project = Project.init(tmp_path / "comparison-hash-compatibility")
    _, left = project.import_text("left.txt", "beta Café alpha Café")
    _, right = project.import_text("right.txt", "gamma café alpha")
    left_scope = project.snapshot([left])
    right_scope = project.snapshot([right])
    expected = compare_frequency(project, left_scope, right_scope)

    result = compare_documents(project, left, right, metric="frequency", limit=1)
    persisted = project.comparison_run(str(result["comparison_id"]))
    expected_hash = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()

    assert persisted["result_hash"] == expected_hash


def test_invalid_streamed_comparison_rolls_back_run_and_rows(tmp_path):
    project = Project.init(tmp_path / "comparison-rollback")
    _, left = project.import_text("left.txt", "alpha beta")
    _, right = project.import_text("right.txt", "gamma delta")
    left_scope = project.snapshot([left])
    right_scope = project.snapshot([right])

    def invalid_rows():
        yield {
            "token": "alpha",
            "left_count": 1,
            "right_count": 0,
            "left_rate": 0.5,
            "right_rate": 0.0,
            "rate_difference": 0.5,
        }
        yield {"token": "beta", "left_count": 1, "right_count": 0}

    with pytest.raises(DocumentError, match="invalid schema"):
        project.create_comparison_run_stream(
            left_scope,
            right_scope,
            metric="frequency",
            calculation_version="fixture-v1",
            config={},
            rows=invalid_rows(),
        )
    assert project.db.execute("SELECT COUNT(*) FROM comparison_runs").fetchone()[0] == 0
    assert project.db.execute("SELECT COUNT(*) FROM comparison_result_rows").fetchone()[0] == 0


def test_document_comparison_persists_two_explicit_document_scopes(tmp_path):
    project = Project.init(tmp_path / "p")
    _, left = project.import_text("left.txt", "apples apples green")
    _, right = project.import_text("right.txt", "elephants elephants green")

    result = compare_documents(project, left, right, metric="keyness")

    assert project.scope_revisions(str(result["left_scope_id"])) == [left]
    assert project.scope_revisions(str(result["right_scope_id"])) == [right]
    assert project.comparison_run(str(result["comparison_id"]))["metric"] == "keyness"


def test_invalid_comparison_metric_does_not_create_scopes(tmp_path):
    project = Project.init(tmp_path / "invalid-comparison-metric")
    _, left_revision = project.import_text("left.txt", "alpha beta")
    _, right_revision = project.import_text("right.txt", "gamma delta")
    left_corpus = project.create_corpus("left")
    right_corpus = project.create_corpus("right")
    project.add_revision_to_corpus(left_corpus, left_revision)
    project.add_revision_to_corpus(right_corpus, right_revision)

    with pytest.raises(ScopeViolationError, match="metric must be frequency or keyness"):
        compare_documents(project, left_revision, right_revision, metric="unsupported")
    assert project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == 0

    with pytest.raises(ScopeViolationError, match="metric must be frequency or keyness"):
        compare_named_corpora(project, left_corpus, right_corpus, metric="unsupported")
    assert project.db.execute("SELECT COUNT(*) FROM scope_snapshots").fetchone()[0] == 0
