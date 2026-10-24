# Python SDK

The SDK exposes Core operations directly; it does not shell out to the CLI or
require an MCP host. Install the features needed by the workflow, for example:

```bash
uv tool install 'discourselens[corpus]'
# or, inside an application environment:
uv add 'discourselens[corpus]'
```

The base package supports project/document/evidence operations. The optional
`corpus` extra enables Parquet/DuckDB token artifacts; `nlp` installs local
backend libraries, but NLP models remain separately selected local assets.

## Project lifecycle and explicit scopes

Use a context manager to close the local SQLite connection reliably. Imports
return stable document/revision IDs; scopes freeze the selected revisions and
are required by retrieval and corpus operations.

```python
from pathlib import Path

from discourselens import Project

with Project.init(Path('research-project')) as project:
    imported = project.import_document(Path('speech.pdf'))
    if imported.revision_id is None:
        raise RuntimeError(f'import did not produce a revision: {imported.status}')

    scope_id = project.snapshot([imported.revision_id])
    counts = project.corpus_frequency(scope_id)
    page = project.corpus_kwic_page(scope_id, 'migration', window=5, limit=50)
    print(counts.most_common(20))
    print(page['total_count'], page['has_more'])

with Project.open(Path('research-project'), read_only=True) as project:
    assert project.scope_revisions(scope_id) == [imported.revision_id]
```

For a corpus intake, `Project.import_documents(paths)` applies the same document
dispatcher to each input independently and preserves input order. It never
concatenates files into one revision, and a failure for one path does not erase
successful imports from other paths. The bounded result contains per-item
status/IDs and aggregate counts; it deliberately omits absolute paths and raw
exception text.

```python
with Project.open(Path('research-project')) as project:
    batch = project.import_documents([
        Path('corpus/speech-a.pdf'),
        Path('corpus/speech-b.pdf'),
    ])
    print(batch.success_count, batch.duplicate_count, batch.failed_count)
    for item in batch.items:
        print(item.index, item.status, item.revision_id, item.failure_code)
```

`SUCCESS`, `DUPLICATE`, and `FAILED` are distinct item outcomes. A duplicate
points to the existing document and revision rather than creating duplicate
canonical state. Per-item failure codes are stable categories; inspect the
source locally when more detail is needed.

Use `Project.reprocess_document(document_id, pdf_path)` to extract a PDF again
as a new revision of an existing document. It returns the same `ImportResult`
shape as `import_document`, including status, document ID, revision ID, and
warnings. The operation is writable only; opening the project with
`read_only=True` rejects it. An unknown document ID raises `DocumentError`.

Reprocessing is revision creation, not duplicate import detection: a successful
call creates a fresh revision even when the PDF bytes and extracted text match
an existing revision. The default `.reprocessed` extractor suffix records the
distinct extraction event. Existing revision IDs, frozen scopes, and evidence
remain attached to their original revisions. A PDF that needs OCR can be
reprocessed with explicit local OCR settings:

```python
with Project.open(Path('research-project')) as project:
    refreshed = project.reprocess_document(
        document_id,
        Path('updated-scan.pdf'),
        ocr_language='eng',
        ocr_dpi=300,
    )
    print(refreshed.status, refreshed.revision_id)
```

An existing project can be opened with `Project.open(path)`. `read_only=True`
is available for inspection and verification. A scope snapshot is historical:
later imports or metadata changes do not expand it. Corpus and subcorpus helpers
also produce explicit snapshots rather than implicit live scopes.

Use `document_revision_summary(revision_id)` to retrieve path-free metadata for
one immutable revision without reaching into the project's SQLite connection:

```python
with Project.open(Path('research-project'), read_only=True) as project:
    revision = project.document_revision_summary(revision_id)
    print(revision['display_name'], revision['page_count'])
```

The result contains revision and document IDs, display name, source and
canonical hashes, extractor and MIME type, source size, creation time, OCR
settings, and page/paragraph counts for that revision. It does not contain
source paths or document text. Unknown revision IDs raise `DocumentError`.
Core stores only the basename of a path-like import name as the document's
display name, recognizing both slash styles. The project format does not store
a separate per-revision original filename.

Use `corpora_page(*, offset=0, limit=100)` and
`subcorpora_page(corpus_id=None, *, offset=0, limit=100)` to browse the corpus
catalog without loading it all at once. Both return an `items` list with
`offset`, `limit`, `total`, and `has_more`. Items are ordered by `name`, then
`id` for deterministic paging. `subcorpora_page` filters to the exact parent
when `corpus_id` is supplied; `None` lists subcorpora across all parents. The
offset must be an integer from 0 through SQLite's signed 64-bit maximum, and
the integer limit must be between 1 and 1,000. Existing `list_corpora()` and
`list_subcorpora(corpus_id=None)` remain available for workflows that
deliberately need an unbounded list.

## Analysis-run listing

Use `analysis_runs_page(*, offset=0, limit=100)` for bounded listing across the
project's immutable runs. It returns an `items` list plus `offset`, `limit`,
`total`, and `has_more`; items retain the safe metadata and ordering of the
legacy list method (`started_at` descending, then ID descending). The offset
must be an integer in SQLite's signed 64-bit range, and the integer limit must
be between 1 and 1,000. Existing scripts can continue using
`list_analysis_runs(limit=100)`, which returns a list directly.

```python
with Project.open(Path('research-project'), read_only=True) as project:
    page = project.analysis_runs_page(offset=0, limit=100)
    print(page['total'], page['has_more'])
    for run in page['items']:
        print(run['id'], run['status'], run['scope_id'])
```

## Qualitative research memos

Research memos are attached to an existing frozen scope. Use the bounded page
API when reading memos from adapters or other workflows that should not load an
unbounded collection:

```python
with Project.open(Path('research-project'), read_only=True) as project:
    page = project.research_memos_page(scope_id, offset=0, limit=100)
    print(page['total_count'], page['has_more'])
    for memo in page['rows']:
        print(memo['title'], memo['content'])
```

`research_memos_page(scope_id, *, offset=0, limit=100)` validates the frozen
scope and returns `scope_id`, `offset`, `limit`, `total_count`, `has_more`, and
`rows`, ordered by creation time and ID. Each row contains `id`, `scope_id`,
`title`, `content`, and `created_at`, plus `title_characters`,
`content_characters`, `title_truncated`, and `content_truncated`. To keep page
payloads bounded, titles are capped at 512 characters and content at 4,000;
the character counts report the original stored lengths. The page limit is 100.
The legacy `research_memos(scope_id)` method remains available when a local
workflow deliberately needs the full, unbounded list.

## Qualitative codebook browsing

`Project.codebook_pages(*, offset=0, limit=100)` lists project codebooks, and
`Project.codes_for_codebook_page(codebook_id, *, offset=0, limit=100)` lists
every code in one codebook, including unused codes. Both return `items`,
`offset`, `limit`, `total`, and `has_more`; code items include
`parent_code_id` for reconstructing the hierarchy. Pages are deterministically
ordered and limited to 100 rows. Unknown codebook IDs and invalid bounds fail
explicitly.

```python
with Project.open(Path("research-project"), read_only=True) as project:
    books = project.codebook_pages(offset=0, limit=100)
    for book in books["items"]:
        codes = project.codes_for_codebook_page(book["id"], offset=0, limit=100)
        print(book["name"], [code["label"] for code in codes["items"]])
```

## Evidence-backed deterministic analysis

Core offers surface frequency, KWIC, n-grams, collocations, lexical bundles,
keyness/comparison, and citable statistic/excerpt operations without an LLM.
Persisted findings link to Core-issued evidence IDs; citations are rendered from
those records.

Use `Project.compare_named_corpora(left_corpus_id, right_corpus_id)` to persist
a frequency or keyness comparison from two named corpora. The operation freezes
each corpus's current membership into its own immutable scope and returns a
bounded page of ranked rows, along with the comparison ID and both scope IDs.
Pass `metric="keyness"` for log-likelihood; `offset` and `limit` select the
result page. Use the comparison ID with `create_comparative_finding` to turn a
selected row into a validated comparative finding.

```python
with Project.open(Path("research-project")) as project:
    comparison = project.compare_named_corpora(
        left_corpus_id,
        right_corpus_id,
        metric="keyness",
        offset=0,
        limit=25,
    )
    print(comparison["comparison_id"], comparison["total_rows"])
    if comparison["rows"]:
        finding = project.create_comparative_finding(
            comparison["comparison_id"],
            comparison["rows"][0]["token"],
            comparability_basis="Both corpora cover the same genre and period.",
        )
```

For large result sets, use the bounded page methods rather than materializing
the full counter in an adapter: `corpus_frequency_page`, `corpus_kwic_page`,
`corpus_ngram_page`, `corpus_collocations_page`, `corpus_dispersion_page`,
`corpus_parsed_frequency_page`, `corpus_surface_frequency_page`, and the
comparison page methods all require an existing explicit scope and return
stable ordering plus `offset`, `limit`, and continuation metadata.
`corpus_collocation_analysis` also accepts `offset`/`limit` and returns the
complete eligible-row `total` with `has_more`; its `truncated` field remains a
compatibility alias for `has_more`. Association ranking and ties are stable
across pages.
`corpus_lexical_bundles` is also paginated with `offset`/`limit`; its
`candidate_count` describes the complete filtered population while `rows`
contains only the requested page. Dispersion pages retain exhaustive scope
totals while bounding only the per-revision rows.
The unbounded `corpus_ngrams`, `corpus_collocations`, and `corpus_dispersion`
methods remain available for local analysis scripts.
Standalone n-grams use `unicode-surface-tokenizer-v2` with NFC-after-casefold
normalization, matching surface frequency. Lexical bundle calculations retain
their separately versioned v1 tokenization.

```python
with Project.open(Path('research-project')) as project:
    result = project.extract_evidence(
        [imported.revision_id], languages=('en',), max_examples=24
    )
    for finding_id in result['finding_ids']:
        finding = project.get_finding(finding_id)
        validation = project.validate_finding(finding_id)
        # `valid` checks structural evidence lineage and scope only. It does not
        # decide whether the cited text semantically supports the interpretation.
        print(
            finding['phenomenon'],
            validation.valid,
            validation.semantic_entailment_assessed,
        )

    analysis = project.analyze_textual_descriptors(
        [imported.revision_id], languages=('en',)
    )
    report_path = project.export_analysis_report(
        analysis['analysis_run_id'], Path('analysis.json'), format='json'
    )
    bundle_path = project.export_research_bundle(
        analysis['analysis_run_id'], Path('research-bundle')
    )
```

`Project.export_research_bundle(..., include_canonical_text=False)` omits
canonical text for revisions referenced by cited textual evidence by default.
The bundle still contains cited evidence excerpts; they can be long or even cover
an entire revision if that was the saved evidence span. Set
`include_canonical_text=True` only when the bundle should carry full canonical
text; it can be sensitive, so review cited excerpts, destination, and recipients
first.
Included text is bound to its revision's canonical hash. The entire manifest,
including optional text, is subject to the 64 MiB manifest limit.

Evidence extraction and textual descriptors are deterministic candidate or
observation workflows; they do not establish author intent or automatically
validate a theoretical interpretation. Read the versioned skill methodology
and limitations before using a finding analytically. A successful
`validate_finding` result means provenance, scope and evidence references pass
structural checks; `semantic_entailment_assessed` is currently always false.
Researchers (or a separately configured review workflow) must judge whether
the cited passage actually supports the interpretation.

## Researcher-controlled quote and group-comparison workflows

### Evidence-bound research question

`Project.ask_question(revision_ids, question, framework_id, preset_name,
limit=8)` composes scoped candidate retrieval with the registered framework
review workflow. It requires an explicit, non-empty revision scope, a
registered interpretive framework, and a saved answer-model preset. Lexical
retrieval is the default. Optional hybrid retrieval requires
`retrieval_mode="hybrid"`, a saved embedding preset, and explicit vector
artifact IDs covering exactly the selected revisions; incomplete or outside-
scope index sets fail closed before query embedding. In hybrid mode, the
question may be sent to the embedding provider and retrieved passages may be
sent to the answer provider, so inspect both destinations and offline settings.

```python
with Project.open(Path("research-project")) as project:
    result = project.ask_question(
        ["rev_statement"],
        "How is responsibility represented?",
        "fairclough-text",
        preset_name="analysis",
        limit=8,
    )
    if result["status"] == "answered":
        print(result["answer_interpretation"])
        for evidence in result["evidence"]:
            print(evidence["citation"], evidence["excerpt"])
    else:
        print(result["limitations"][0])
```

To opt into hybrid candidate retrieval, pass `retrieval_mode="hybrid"`,
`embedding_preset_name="embeddings"`, and the indexed artifact IDs for every
revision in the frozen scope. The answer model `preset_name` and embedding
preset are independent selections.

The result carries candidate and answer run IDs, the frozen scope and targeted
coverage, answer/alternative interpretation, limitations, and Core-rendered
citations. If retrieval returns no candidates, the model is not called and the
result says only that no examples were found under that retrieval plan; this
does not prove absence from the scope. This command is not unrestricted chat.

## Researcher-controlled quote and group-comparison workflows

`Project.run_scope_question_candidates(revision_ids, framework_id, question,
limit=12)` runs the built-in lexical candidate search for one non-empty
research question in an explicit frozen revision scope. `framework_id` must be
a registered interpretive framework; `critical-reviewer` is not an interpretive
target. The limit must be from 1 through 12. The result exposes the analysis,
skill, retrieval, scope, and coverage IDs, candidate evidence IDs, an optional
finding ID, and a structured `CoverageDescriptor` whose coverage mode is
`CoverageMode.TARGETED`. A question with no lexical matches completes normally
with `candidate_count == 0`, `finding_id is None`, and `evidence_ids == []`;
targeted retrieval does not establish absence.

```python
with Project.open(Path("research-project")) as project:
    result = project.run_scope_question_candidates(
        ["rev_statement"],
        "fairclough-text",
        "How is public responsibility represented?",
        limit=8,
    )
    print(result["scope_snapshot_id"], result["candidate_count"])
    print(result["coverage"].coverage_mode)
```

To run the same workflow end to end without a configured model provider, use
the synthetic offline example:

```bash
python examples/offline_scope_question_workflow.py /tmp/discourselens-scope-question
```

It imports two selected passages and one lexical match outside the scope, then
prints the analysis, skill, retrieval, scope and finding IDs alongside the
candidate evidence text. The output reports `TARGETED` coverage: retrieved
passages are review candidates, and missing candidates would not establish
absence or prevalence.

`Project.analyze_quote_attribution_candidates(revision_ids, revision_languages=...)`
inventories quotation and nearby attribution-cue candidates in the explicitly
selected revisions. The SDK snapshots that revision list into a frozen scope
for the analysis run; later imports do not expand it. `revision_languages` is
optional and, when provided, must map every selected revision ID to `"en"` or
`"es"`. Without an assignment, the workflow still scans quote marks but does
not assume a language for attribution-cue matching. Results include the run
and `scope_snapshot_id`, candidate counts, and evidence-linked candidate
records. A quote or nearby reporting verb is only a candidate: it does not
establish that a quotation is authentic, identify its speaker, or show that an
author endorses it or intends a particular effect.

`Project.prepare_van_dijk_group_comparison(revision_ids, *, groups,
shared_topic_event, comparability_basis, codebook_id)` creates a descriptive,
evidence-linked comparison from two researcher-labeled groups. Each group is a
mapping with a `label` and `annotation_ids` list. The SDK freezes the supplied
revision IDs in a scope snapshot, then validates every selected annotation
against that scope and the declared codebook. It counts only the annotation
IDs supplied by the researcher; it does not retrieve or add annotations. Both
groups must have distinct labels, and at least one annotation must be selected
across the two groups. One group may have an empty `annotation_ids` list; its
`coding_status` and code distribution remain `UNKNOWN`, not evidence of absence.
The comparison does not infer ideology, asymmetry, or prevalence. Its counts
describe only the selected annotation records, and the topic, comparability
basis, and group labels are researcher-provided.

```python
with Project.open(Path("research-project")) as project:
    revision_ids = ["rev_resident_statement", "rev_official_statement"]

    quote_result = project.analyze_quote_attribution_candidates(
        revision_ids,
        revision_languages={
            "rev_resident_statement": "en",
            "rev_official_statement": "en",
        },
    )
    print(quote_result["scope_snapshot_id"], quote_result["candidate_count"])

    comparison = project.prepare_van_dijk_group_comparison(
        revision_ids,
        groups=[
            {"label": "Residents", "annotation_ids": ["ann_resident_road_safety"]},
            {"label": "Officials", "annotation_ids": []},
        ],
        shared_topic_event="the road safety announcement",
        comparability_basis="both revisions discuss the same announced safety review",
        codebook_id="codebook_road_safety",
    )
    print(comparison["groups"][1]["coding_status"])  # UNKNOWN
```

Replace the example IDs with revision IDs and annotation IDs already present
in the project. The annotation IDs must belong to the declared codebook and
their spans must fall inside the selected revision scope.

## Local social-actor and agency candidates

`Project.analyze_social_actor_candidates(revision_ids, artifact_ids)` and
`Project.analyze_agency_candidates(revision_ids, artifact_ids)` run deterministic
candidate extraction over explicitly selected local `parsed_tokens` artifacts.
`revision_ids` define the immutable analysis scope; `artifact_ids` select the
artifacts to read, and every artifact must belong to that scope. Use artifact
IDs returned by `build_parsed_token_artifact()`. The API does not choose or
build artifacts implicitly. An artifact set can include at most one parsed-token
artifact for each revision.

The social-actor operation records noun, proper-noun, and pronoun expressions
tagged by the parser. The agency operation records selected dependency relations
between a dependent and its predicate. Agency extraction is eligible only when
the `dependency_parse` capability check for the selected parser artifact reports
`SUPPORTED`; otherwise the operation fails closed. Both operations produce
evidence-linked observations for researcher review. They do not classify social
or discursive representation, interpret agency or responsibility, or infer
author intention. They are deterministic local processing and do not call an
LLM.

```python
with Project.open(Path("research-project")) as project:
    revision_ids = ["rev_statement"]
    artifact_ids = ["artifact_parsed_tokens_statement"]

    social_actors = project.analyze_social_actor_candidates(
        revision_ids, artifact_ids
    )
    agency = project.analyze_agency_candidates(revision_ids, artifact_ids)
    print(social_actors["analysis_run_id"], social_actors["finding_ids"])
    print(agency["analysis_run_id"], agency["finding_ids"])
```

Replace the example IDs with existing revision and parsed-token artifact IDs.
Read the versioned candidate methodology and inspect its parser-linked evidence
before making any interpretive claim.

## Local NLP artifacts

Derived token/parse artifacts are versioned and rebuildable; they do not mutate
the immutable document revision. Select and load a local backend explicitly.
The SDK does not download NLP models implicitly.

```python
from discourselens.nlp import load_spacy

with Project.open(Path('research-project')) as project:
    backend = load_spacy('en_core_web_sm', language='en')
    revision_id = imported.revision_id
    if revision_id is None:
        raise RuntimeError(f'import did not produce a revision: {imported.status}')

    scope_id = project.snapshot([revision_id])
    artifact = project.build_parsed_token_artifact(revision_id, backend)
    print(artifact.artifact_id, artifact.model, artifact.model_version)
    page = project.corpus_parsed_frequency_page(
        scope_id,
        [artifact.artifact_id],
        field='lemma',
        offset=0,
        limit=100,
    )
    print(page['rows'])
    print(page['total_values'], page['has_more'])
```

Parsed-frequency pages accept `lemma`, `pos`, or `dependency_relation`. Supply
exactly one parsed-token artifact for every revision in the immutable scope;
Core rejects missing, duplicate, extra, or out-of-scope artifacts. Counts cover
that complete artifact population before pagination. Each result includes the
`scope_id`, `revision_ids`, `artifact_ids`, and `field`, plus `offset`, `limit`,
`total_values` (the full distinct-value count), `has_more`, and the selected
`rows` as `{value, count}` records.

Run `discourselens nlp capabilities --help` to inspect an explicitly selected
backend/model; operational and assurance statuses are reported separately.
Do not infer CDA validation from successful parsing.

Surface-token frequency pages use the same derived artifact registry and frozen
scope. `corpus_surface_frequency_page` accepts `field="normalized"` for token
occurrence counts or `field="document_frequency"` for the number of revisions
containing each token. It requires exactly one compatible surface-token
artifact for every revision in the scope; incomplete or out-of-scope artifact
sets are rejected. Results are deterministically ordered by descending count
then token, and return bounded `token`/`count` rows plus scope/artifact
provenance and continuation metadata. New artifacts use a
`surface-token-v2-ucd-<version>` schema, the same Unicode tokenizer and
NFC-after-casefold normalization as inline corpus frequency and n-grams. Pages
expose the calculation version, schema, tokenizer, normalization rule, and
Python Unicode database version. Existing
`surface-token-v1` files are not reinterpreted and must be rebuilt before these
current frequency operations can use them:

```python
with Project.open(Path("research-project")) as project:
    scope_id = project.snapshot([revision_id])
    artifact = project.build_surface_token_artifact(revision_id)
    page = project.corpus_surface_frequency_page(
        scope_id,
        [artifact.artifact_id],
        field="normalized",
        offset=0,
        limit=100,
    )
    print(page["rows"], page["total_values"], page["has_more"])
```

The CLI `corpus normalized-frequency` and `corpus document-frequency` commands
and the corresponding MCP tools use this same Core page contract.

## Embeddings and scoped vector search

Vector indexes are derived artifacts. Build them from a selected embedding
preset and immutable revision, then pass a frozen scope and explicit artifact
IDs to semantic or hybrid search:

```python
with Project.open(Path('research-project')) as project:
    artifact_id = project.build_paragraph_embedding_index('embed', revision_id)
    scope_id = project.snapshot([revision_id])
    semantic_hits = project.semantic_search_with_preset(
        'embed', scope_id, [artifact_id], 'How is responsibility described?'
    )
    hybrid_hits = project.hybrid_search_with_preset(
        'embed', scope_id, [artifact_id], 'responsibility', limit=20
    )
```

Without an explicitly injected local embedding provider, indexing can send
paragraph text and semantic/hybrid search can send the query text to the
provider selected by the preset. Check the endpoint and offline policy first.
Index scope and embedding provenance are validated before query embedding;
incompatible model/chunker indexes fail rather than mixing vector spaces. The
request preview and index builder use the same sequential batches of at most 64
paragraphs (or a stricter declared provider limit), without concurrent sends.

Semantic and hybrid calls return bounded retrieval candidates, not a coverage
certificate. An empty result means only that no candidate was returned by that
query/index configuration; it does not establish corpus-wide absence. Do not use
top-k output for universal or exact-count claims. Analysis findings must retain
their separate `CoverageDescriptor`, and corpus-wide counts/absence require an
appropriate exhaustive deterministic operation.

For a caller-supplied local vector, the Core also exposes provenance-recording
raw search operations without creating an embedding-provider dependency:

```python
with Project.open(Path('research-project')) as project:
    scope_id = project.snapshot([revision_id])
    result = project.search_vectors_with_provenance(
        scope_id, [artifact_id], (0.12, -0.04, 0.31), limit=10
    )
    print(result.retrieval_run_id, result.scope_id, result.hits)

    hybrid = project.hybrid_search_with_provenance(
        scope_id,
        [artifact_id],
        'responsibility',
        (0.12, -0.04, 0.31),
        limit=10,
    )
    print(hybrid.retrieval_run_id, hybrid.hits)
```

These calls do not infer which model produced a supplied vector. The persisted
run records the vector's dimension and digest, never the vector itself; hybrid
search records a digest and character count for the lexical query, not its raw
text. Scope, selected artifacts, ranking policy, candidate counts, hit spans,
and component ranks remain inspectable through the retrieval-run records.

## Model-assisted workflows

Remote/local model selection, credential references, endpoint choice and
capability declarations are configured explicitly in the project. For direct
generation from one saved preset, use `Project.generate_with_preset`. It returns
the execution ID and response, while Core records secret-free execution
metadata; prompt and response text are not persisted. The project offline
policy and the selected model's capability declarations still apply. Use
`preview_model_preset_request` to inspect the prompt destination before sending.
For image input, pass a validated local `GenerationImage`; image support requires
an exact-model vision declaration and is checked by Core:

```python
from discourselens.models import GenerationImage

with Project.open(Path("research-project")) as project:
    image = GenerationImage.from_file(Path("evidence.png"))
    result = project.generate_with_preset(
        "analysis",
        "Describe this image in relation to the research question.",
        image=image,
    )
    print(result.execution_id, result.response.text)
```

For a controlled CDA review, save a model preset first, then call:

```python
with Project.open(Path('research-project')) as project:
    result = project.analyze_controlled_cda(
        [imported.revision_id],
        preset_name='analysis',
        frameworks=('fairclough-text', 'van-leeuwen-social-actors'),
    )
    print(result['analysis_run_id'])
```

This operation may send selected evidence to the configured provider. Use
`project.preview_model_preset_request(...)` when you need to inspect a request
before sending it, and use persistent offline mode to block non-loopback remote
operations. Never put secret values in project configuration or source code.
See [Provider Setup](PROVIDERS.md) for credential and routing details.

## Optional model cost estimates

The SDK can save an explicit, versioned rate for a provider/model pair. Rates
are supplied by you per million provider-reported tokens; saving or reading a
rate makes no network request and does not query provider price tables:

```python
from pathlib import Path

from discourselens import Project
from discourselens.models import ModelPrice

price = ModelPrice(
    provider_id='openai',
    model='MODEL_ID',
    input_per_million_tokens='0.25',
    output_per_million_tokens='1.25',
    currency='USD',
    source_reference='https://provider.example/pricing',
    rate_version='docs-2026-01',
    rate_date='2026-01-15',
)

with Project.open(Path('research-project')) as project:
    project.save_model_price(price)
    saved_price = project.model_price(
        'openai', 'MODEL_ID', rate_version='docs-2026-01'
    )
    assert saved_price == price
```

The example rates are illustrative. Use a source and values you have verified;
do not place credentials in `source_reference`. Add a new rate version when
prices change. Model executions retain their selected rate snapshot and
estimate, so updating rates does not rewrite the past. The CLI command
`discourselens models price list research-project --provider openai --model MODEL_ID`
lists stored rate versions. `discourselens models costs research-project`
lists persisted snapshots and estimates; it does not re-estimate them with the
current rate. A missing rate or required usage count produces a `null`
estimate, not zero. Reasoning usage is separated only when you explicitly
supply a reasoning rate. Estimates depend on provider-reported usage and the
rates you entered; they do not guarantee the amount on a provider invoice.

Read the persisted cost ledger through a bounded page:

```python
with Project.open(Path('research-project'), read_only=True) as project:
    page = project.model_costs_page(offset=0, limit=100)
    print(page['total_count'], page['has_more'])
    for execution in page['rows']:
        print(execution['execution_id'], execution['estimated_cost'])
```

Core accepts offsets from zero through SQLite's signed 64-bit maximum and
limits from 1 to 1000. Each page includes `offset`, `limit`, `total_count`,
`has_more`, and `rows`. Rows contain execution/provider/model IDs, token usage,
the persisted estimated cost and currency, timestamps, any error type, and the
validated `model_price_snapshot` used for that execution. The page omits
credential aliases and request/response hashes; it validates and normalizes the
stored snapshot rather than looking up current rates or calling a provider.

## Interface and provenance notes

The SDK returns bounded pages for potentially large result sets. Prefer
`*_page` operations for frequency, KWIC and comparison outputs; inspect their
`offset`, `limit`, total and continuation fields. Analysis runs and scope
snapshots are immutable provenance records; a changed configuration creates a
new run rather than rewriting the past. `Project.export_research_bundle` and
`discourselens.verify_research_bundle` support portable integrity checking.

For command equivalents see the [CLI reference](CLI_REFERENCE.md); for MCP
stdio launch and host configuration see [MCP Setup](MCP_SETUP.md).
