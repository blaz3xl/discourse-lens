# Benchmarks

This directory starts with a small, synthetic contract smoke dataset. It is not a
methodological benchmark and does not measure model or researcher performance.
The separate, future methodological benchmark described below does not exist yet.

## Contract smoke fixtures

`datasets/epistemic-contract-smoke/` is labeled
`SYNTHETIC_CONTRACT_FIXTURES_NOT_VALIDATION`. Its six short English, Spanish, and
Arabic examples are wholly synthetic, including the placeholder names. They are
intended to exercise the shape of a case and the distinction between supporting
evidence, counterevidence, bounded claims, and limitations. They do not establish
scientific, linguistic, or methodological validity, and they provide no accuracy
or performance estimate. The case count is an inventory count only.

`schemas/case-v1.json` describes one JSONL record. Span offsets are half-open
Unicode code-point offsets into that record's exact `text` value: `start` is
included and `end` is excluded. They are not UTF-8 byte offsets or grapheme
cluster offsets. Do not normalize or otherwise alter the text before resolving a
span. The schema checks record shape; consumers must also check that each span is
nonempty, has `start < end`, and satisfies `end <=` the text's Unicode
code-point length. Strings with escaped unpaired UTF-16 surrogates are rejected
by the evaluator so offsets and JSON output use valid Unicode scalar values.
Limitation values are exact `snake_case` tags, not prose to be evaluated
semantically.

The case-v1 schema also permits an additive evidence-identity contract. Cases
that use it declare `scope_revision_ids`, an `evidence_catalog` mapping each
known `evidence_id` to its `revision_id`, and gold `support_evidence_ids` and
`counterevidence_evidence_ids`. Predictions may include those same two fields
on each finding as arrays of `{ "evidence_id", "revision_id" }` references.
Older case-v1 records and predictions without these optional fields remain
accepted; their evidence-ID metric denominators are empty.

The fixtures are kept in this repository for internal development. No license is
assigned to this dataset while the project's license decision is pending. Do not
publish or redistribute it until that decision is closed.

## Offline evaluator

Run the mechanical evaluator with:

```sh
python -m benchmarks.evaluate DATASET.jsonl PREDICTIONS.jsonl
```

The `coverage_label_comparison` output uses the smoke-protocol ordering
`EXAMPLE_BASED < TARGETED < BROAD < EXHAUSTIVE` only to compare a prediction's
declared label with a case's declared minimum. This ordering is a benchmark
convention; comparing labels does not inspect retrieval runs, searched text, or
population coverage, and must not be described as measuring actual search or
retrieval completeness. A missing prediction has no coverage declaration and is
reported separately from a declared label below the minimum.

All output metrics are mechanical: supporting and counterevidence spans use
exact half-open offset matches; evidence IDs use exact ID-and-revision matches;
claim labels, alternative interpretation presence, and limitation tags are
compared as declared. Span metrics count distinct exact spans per case:
duplicate entries inside one span array are rejected, while reusing a span across
findings counts it once. Evidence-ID precision, recall, and F1 are reported
separately for support and counterevidence, so an ID match never substitutes for
a span match. Repeated IDs are reported as duplicates; repeating the same
ID-and-revision pair does not inflate the set-based metric. Unknown IDs,
cataloged IDs outside the declared revision scope, and known in-scope IDs paired
with the wrong revision are listed in separate diagnostics; each distinct
invalid ID/revision pair counts as a false
positive, while a gold identity not matched at its declared revision counts as a
false negative. Claim and limitation tags are also set-valued per case. The
evaluator does not measure semantic quality or validate CDA, and these identity
scores do not establish that evidence semantically supports a claim.

## Future adjudicated methodological benchmark

A real-data benchmark must be a separate, versioned release with its own protocol,
dataset manifest, and documented intended uses. Before publishing one, complete
and retain all of the following:

- Record source provenance for every item: source identity, collection date and
  method, population and sampling process, transformations, and any excerpting or
  redaction.
- Resolve rights and terms for source material, excerpts, annotations, and
  derived artifacts. Document the applicable license and any access or reuse
  limits. Obtain consent where required, and complete relevant privacy and ethics
  review before release.
- Publish annotation instructions and version them with the dataset. Have at
  least two trained annotators label each item independently before adjudication.
- Record disagreements and adjudication decisions, including who adjudicated and
  which guideline version was used.
- Report inter-annotator agreement with the metric and unit suited to each task,
  label, and language; include denominators and uncertainty where appropriate.
  Do not present agreement as proof of construct validity.
- Describe known limitations before release, including sampling and coverage,
  language and dialect representation, annotation ambiguity, domain and time
  bounds, and any remaining source or consent restrictions. Validate every
  published span against its frozen source text.

Until those requirements are met, no real-data methodological benchmark or
benchmark validity claim is represented here.

## Retrieval ranking evaluation

Run the offline ranking evaluator with versioned judgments and result files:

```sh
python -m benchmarks.retrieval_eval \
  benchmarks/datasets/retrieval-ranking-smoke/judgments-v1.json \
  benchmarks/datasets/retrieval-ranking-smoke/results-v1.json \
  --ks 1,3,5
```

The input contracts are described in `schemas/retrieval-judgments-v1.json` and
`schemas/retrieval-ranking-results-v1.json`. Both files declare the same `unit_kind`:
`DOCUMENT_REVISION`, `SOURCE_SPAN`, or `CHUNK`. Query judgments assign integer
grades from 0 to 3 to `unit_id` values; 0 means nonrelevant, and grades 1 through
3 mean relevant. A ranking result row gives each returned unit an explicit
positive integer rank. Ranks must be unique within a query, but may have gaps.
Duplicate query, retriever, and unit IDs are rejected. Every result query ID must
exist in the judgments file. A missing result row is treated as an empty ranking.

For `CHUNK` comparisons, both files must include identical `chunker_provenance`
with the chunker name, version, and configuration. The evaluator checks that this
provenance matches. `SOURCE_SPAN` IDs follow the `SourceSpan` convention: they are
revision-bound stable coordinates, identify the source revision, and must not be
reused across revisions.

For each retriever, the evaluator reports macro-averaged metrics across queries:

- `Recall@k` is the fraction of positively judged unit IDs appearing at ranks
  at most `k`, divided by the count of positively judged IDs for that query.
- Reciprocal rank is `1 / rank` for the first positively judged result, or zero if
  none appears; `mean_reciprocal_rank` averages these query scores.
- `nDCG@k` uses gain `2^grade - 1` and discount `log2(rank + 1)`, normalized by
  the ideal ordering of that query's positive grades. Grade-0 and unjudged results
  have zero gain.

Queries with no positive-grade judgments are excluded from all metric means by
default. Pass `--empty-relevance zero` to include them as zero-scoring queries.
Counts in each retriever's output make the selected denominator explicit. If a
metric has no queries in its denominator, it is `null`.

The `retrieval-ranking-smoke` fixture is synthetic and exists only to check the
metric calculations and input validation. Its synthetic smoke tests do not
validate semantic relevance or CDA, and the harness does not call retrieval
providers or network services. These scores are not evidence of retrieval quality.

### Capturing local lexical results

`retrieval_capture.py` provides a small API-only helper for exporting a public
retrieval API's results. Open a project and pass an existing frozen scope for each
query; the helper calls `Project.search_exact_ranked()` and returns a ranking payload
that can be saved as `results.json` and passed to the evaluator:

```sh
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path

from benchmarks.retrieval_capture import CaptureQuery, capture_lexical_rankings
from discourselens import Project

with Project.open("PATH/TO/PROJECT") as project:
    results = capture_lexical_rankings(
        project,
        [CaptureQuery("q1", "migration policy", "FROZEN_SCOPE_ID")],
        retriever_id="lexical-bm25",
        limit=100,
    )

Path("results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
PY

python -m benchmarks.retrieval_eval judgments.json results.json --ks 1,3,5
```

The output uses `DOCUMENT_REVISION` units: matching paragraphs are ordered by the
public BM25 exact-phrase API, then collapsed to one entry per revision at its first
matching paragraph rank. The helper checks returned revision IDs against each
query's frozen scope, so later imports or revisions outside that snapshot are not
emitted. This is an offline retrieval-mechanics fixture; it makes no semantic
relevance or retrieval-quality claim.

### Capturing local vector results

`capture_vector_rankings()` accepts explicit query vectors and vector artifact
IDs for each immutable scope. It calls the Core's scope-enforcing vector search,
captures the best chunk for each validated document revision, then globally
orders those document-level scores. This avoids a document with many chunks
crowding the revision ranking. This keeps
benchmark capture independent of embedding APIs and model downloads:

```python
from benchmarks.retrieval_capture import VectorCaptureQuery, capture_vector_rankings
from discourselens import Project

with Project.open("PATH/TO/PROJECT") as project:
    results = capture_vector_rankings(
        project,
        [
            VectorCaptureQuery(
                query_id="q1",
                scope_id="FROZEN_SCOPE_ID",
                artifact_ids=("VECTOR_ARTIFACT_ID",),
                query_vector=(0.2, 0.7, 0.1),  # supplied by the evaluated embedder
            )
        ],
        retriever_id="local-vector-v1",
        limit=100,
    )
```

Use query vectors generated by the same embedding configuration under
evaluation; the helper does not create embeddings. Random or hand-authored
vectors are useful only for deterministic mechanics/scope tests, not relevance
quality. Captured vector ranks use the same versioned result schema and can be
evaluated against independently prepared graded judgments with
`benchmarks.retrieval_eval`.

## Synthetic corpus workload smoke benchmark

`corpus_workload.py` creates seeded synthetic `.txt` documents in a temporary
workspace, imports them through the public `Project` API, creates an explicit
scope, and times frequency counts, KWIC, and optionally scoped FTS search. The
workspace and imported project are removed after the JSON result is collected.
The result records the seed and synthetic text hash, Python/platform/SQLite
environment, corpus dimensions, elapsed wall times, and project/database/source/
artifact byte sizes.

Run the safe default workload (3 documents, 120 tokens each):

```sh
PYTHONPATH=src python -m benchmarks.corpus_workload
```

Include scoped FTS timing and write JSON to a new file with:

```sh
PYTHONPATH=src python -m benchmarks.corpus_workload --fts-search --output run.json
```

Larger workloads require the explicit `--allow-large` flag. For example, this
generates 500,000 tokens across 100 local text files:

```sh
PYTHONPATH=src python -m benchmarks.corpus_workload \
  --documents 100 --tokens-per-document 5000 --seed 7 --allow-large --fts-search
```

The hard bounds are 1,000 documents, 100,000 tokens per document, and 1,000,000
total tokens. Above-default runs use temporary storage and remain local; no model,
remote API, real corpus, or Docker service is involved. Text generation is
deterministic for a given seed and configuration. Timings are environment-specific
synthetic smoke measurements; they do not support scientific or real-corpus
performance claims.
