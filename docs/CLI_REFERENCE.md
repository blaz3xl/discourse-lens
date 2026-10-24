# Command-line reference

The `discourselens` command is a local adapter over the same Core used by the
Python SDK and MCP server. Use `discourselens --help` and
`discourselens <group> --help` for the installed version's full argument and
option schemas; this guide describes the command groups and the main research
workflows.

## Global options

```text
discourselens --version
discourselens --help
discourselens --verbose <command>   # lifecycle events on stderr
discourselens --debug <command>     # diagnostics on stderr
```

Machine-readable output is available on commands that produce tabular data.
Select `--format json`, `jsonl`, `csv`, or `table` where that command exposes
`--format`; use `--help` to confirm supported formats and pagination options.
Paged results include continuation/count metadata where applicable. CSV output
escapes formula-like text; JSON is the choice when values must remain unchanged.

All document analysis and retrieval operates on an explicit immutable revision
or saved scope. No command silently broadens a document selection to the whole
project.

## Main command groups

| Group | Purpose |
| --- | --- |
| `project` | Initialize a project, set offline mode and inspect import configuration. |
| `document` | Import one or many PDF, DOCX, HTML, Markdown or UTF-8 text files; explicitly reprocess PDFs; inspect revisions/metadata and read bounded pages, paragraphs, and source-span context. |
| `search` | Exact phrase or bounded FTS lexical search inside an explicitly supplied revision or scope. |
| `corpus` | Frequency, dispersion, KWIC, n-grams, collocations, keyness, corpus membership, and derived token/NLP operations. |
| `analyze` | Deterministic textual descriptors and evidence extraction, candidate skills, controlled CDA review, counterevidence and resume/batch workflows. |
| `profiles` | Save, inspect, list and run declarative analysis profiles. |
| `providers`, `models` | Configure credential references, exact model presets/capabilities, routing, and optional model prices/cost estimates. |
| `nlp` | Inspect explicitly selected local NLP backend/model capabilities without downloading models. |
| `vectors` | Preview embedding requests, build local derived indexes, and run scoped semantic/hybrid search. |
| `finding`, `evidence`, `coding` | Validate/review findings, inspect synthesized-finding lineage, add researcher-supplied context, and maintain qualitative codebooks/annotations. |
| `run` | List and inspect immutable analysis-run provenance. |
| `report`, `export` | Render findings/analyses/comparisons and export/verify research bundles. |
| `skills` | Discover trusted built-in skills, run scoped candidate retrieval, and inspect methodology documentation. |

## Evidence-grounded ask

`ask` runs a framework-bounded review over only the explicitly selected immutable
revisions, using a saved model preset. Repeat `--revision` for a multi-document
scope. It is not an unrestricted chat command. Human output shows Core-rendered
citations and states the targeted-coverage limitation; it does not treat missing
retrieval hits as proof that something is absent. Use `--format json` to preserve
the complete structured result and run/evidence provenance.

Candidate retrieval is lexical by default. To use hybrid retrieval, provide a
saved embedding preset and the comma-separated IDs of the indexed artifacts to
search. The artifact IDs must be non-empty and unique. Embedding options are only
valid with `--retrieval hybrid`.

```bash
discourselens ask research "How is responsibility represented?" \
  --revision REVISION_A --revision REVISION_B \
  --framework fairclough-text --preset university --limit 8
discourselens ask research "How are social actors represented?" \
  --revision REVISION_A --framework van-leeuwen-social-actors \
  --preset university --format json
discourselens ask research "How are social actors represented?" \
  --revision REVISION_A --framework van-leeuwen-social-actors \
  --preset university --retrieval hybrid --embedding-preset embeddings \
  --embedding-artifacts ARTIFACT_A,ARTIFACT_B
```

The framework must be a registered interpretive framework; `critical-reviewer`
is intentionally excluded because it reviews an existing interpretation rather
than answering a research question. `--limit` accepts 1–12 evidence candidates.
If retrieval finds none, text output says that no examples were found under the
retrieval plan and explicitly warns that this does not establish corpus-wide
absence; it will not display an answer for that result.

Legacy top-level `init` and `import` aliases remain available; prefer
`project init` and `document import` in new scripts.

## Research memo listing

`coding memos PROJECT SCOPE` returns the full memo list when paging options are
omitted. Add `--offset` or `--limit` to request a bounded Core page; an omitted
offset defaults to 0 and an omitted limit defaults to 100. The limit accepts
1–100. Paged rows report original title/content character counts and whether
either field was truncated (to 512 and 4,000 characters respectively).

```bash
discourselens coding memos research SCOPE_ID --offset 0 --limit 25
```

`coding annotations PROJECT REVISIONS` likewise returns its legacy JSON array
when both paging options are omitted. Supplying `--offset` or `--limit` opts
into a bounded page; an omitted offset defaults to 0 and an omitted limit to
100 (maximum 100). Paged JSON includes `items`, `revision_count`, `offset`,
`limit`, `total`, and `has_more`.

```bash
discourselens coding annotations research REVISION_ID --offset 0 --limit 50
```

`coding list-codebooks PROJECT` and `coding list-codes PROJECT CODEBOOK_ID`
read the codebook catalog and a codebook's complete code hierarchy, including
codes not yet applied to annotations. Both return one bounded JSON page by
default (`offset=0`, `limit=100`; maximum 100); code rows include
`parent_code_id` so clients can reconstruct nested codes. These commands open
the project read-only.

```bash
discourselens coding list-codebooks research --offset 0 --limit 50
discourselens coding list-codes research CODEBOOK_ID --offset 0 --limit 100
```

`corpus list` and `corpus list-subcorpora` preserve their legacy JSON arrays
when both paging options are omitted. Supplying either `--offset` or `--limit`
opts into a bounded page; the omitted offset defaults to 0 and the omitted
limit to 100 (maximum 1,000). Paged JSON contains `items`, `offset`, `limit`,
`total`, and `has_more`. The existing `--corpus` filter on
`list-subcorpora` applies before pagination.

```bash
discourselens corpus list research --offset 100 --limit 100
discourselens corpus list-subcorpora research --corpus CORPUS_ID --offset 0 --limit 50
```

## Bounded document readers

Use these readers to retrieve persisted source text by canonical ID.
`document list PROJECT` returns bounded document metadata with each document's
latest revision; it does not return source text or absolute paths. The JSON page
defaults to offset 0 and limit 100, with a maximum limit of 1,000, and includes
`items`, `offset`, `limit`, `total`, and `has_more`.

```bash
discourselens document list research --offset 0 --limit 100
```

For PDFs, `document page` takes a zero-based page index within an immutable
revision. Its result uses the stored citation label, which can differ from that
index. `document paragraph` takes a canonical paragraph ID, such as one from a
scoped search result. Both cap returned text at 8,000 characters by default;
`--max-chars` accepts 1–10,000 and the result reports `text_truncated` when
capped.

`document context` takes a canonical source-span ID and returns bounded text
before and after that exact span. `--before` and `--after` default to 200
characters each and accept 0–10,000; `--max-chars` separately caps the
`exact_text` field at 8,000 characters by default (maximum 10,000) and the
result reports `exact_text_truncated` when capped.

`document show PROJECT REVISION` returns revision provenance metadata, including
the document's display name, source and canonical hashes, extraction details,
OCR settings, and page/paragraph counts. It contains no absolute paths or
source text.

```bash
discourselens document show research REVISION_ID
discourselens document page research REVISION_ID 0 --max-chars 4000
discourselens document paragraph research PARAGRAPH_ID --max-chars 4000
discourselens document context research SPAN_ID --before 300 --after 300
```

`document import-many PROJECT FILE...` imports the supplied paths in argument
order and returns one JSON outcome per input, with aggregate success, duplicate,
and failure counts. Each item includes its zero-based input index and basename so
same-named files remain distinguishable without returning full local paths. A
failed item does not discard successful imports; the command emits every outcome
and exits with status 1 if any item failed. Directory arguments are treated as
individual inputs; the command does not recurse. `--max-bytes` overrides the
configured size limit for each input. PDF OCR remains opt-in and local through
`--ocr-language` and `--ocr-dpi`.

```bash
discourselens document import-many research first.txt second.pdf third.docx \
  --max-bytes 50000000 --ocr-language eng+spa
```

`document reprocess PROJECT DOCUMENT_ID FILE` extracts a PDF again and stores
the result as a new revision of the existing document. This is explicit and
local; it never creates a new document or changes prior revisions. The JSON
result contains the document/revision IDs, status and warnings, but no source
path. `--extractor-suffix` defaults to `.reprocessed` and records a distinct
extractor version. OCR stays opt-in through `--ocr-language` and `--ocr-dpi`;
`--max-bytes` overrides the configured file-size limit. Invalid documents or
files exit nonzero. If extraction finds no text, the command emits its JSON
`OCR_REQUIRED` result, explains that local OCR can be enabled, and exits with
status 1.

```bash
discourselens document reprocess research DOCUMENT_ID revised-scan.pdf \
  --ocr-language eng --ocr-dpi 300
```

## Offline corpus workflow

```bash
discourselens project init research
discourselens document import research speech.pdf
discourselens document show research REVISION_ID
discourselens corpus frequency research REVISION_ID --format json
discourselens corpus kwic research REVISION_ID migration --window 5 --format json
discourselens corpus frequency-evidence research REVISION_ID migration
discourselens corpus frequency-evidence-scope research SCOPE_ID migration
discourselens run list research
```

`corpus frequency-evidence PROJECT REVISION TERM` remains the revision-only
form. To persist an exhaustive deterministic token count for an existing
`ScopeSnapshot`, use `corpus frequency-evidence-scope PROJECT SCOPE_ID TERM`.
`TERM` is one surface token. The scope form reads that snapshot directly
without creating or changing its membership, then records
`CorpusStatisticEvidence` with an exhaustive `CoverageDescriptor`.

`run list` preserves its legacy JSON array output when `--offset` is omitted.
Supply `--offset` to opt into a paged response; `--limit` defaults to 100 and
accepts 1–1,000, while the zero-based offset must fit a signed 64-bit integer.
Paged JSON contains `items`, `offset`, `limit`, `total`, and `has_more`.

To count tokens from derived surface-token artifacts, first build the artifacts
for the complete selected revision population with `corpus build-tokens`.
New artifacts use the versioned `surface-token-v2-ucd-<version>` schema and
shared tokenizer, with NFC normalization after case folding. Their counts match
inline corpus frequency and n-grams. Existing `surface-token-v1` artifacts are
preserved but must be rebuilt before use by current normalized-frequency or
document-frequency commands.
`normalized-frequency` returns occurrence counts; `document-frequency` counts
how many revisions contain each token. Both commands are bounded pages with
deterministic tie ordering and explicit artifact provenance:

```bash
discourselens corpus build-tokens research REVISION_ID
discourselens corpus normalized-frequency research REVISION_ID ARTIFACT_ID \
  --offset 0 --limit 100 --format json
discourselens corpus document-frequency research REV_A,REV_B ART_A,ART_B \
  --offset 0 --limit 100 --format json
```

For parser-derived counts across a saved scope, use the read-only
`parsed-frequency-scope PROJECT SCOPE_ID --artifacts ID1,ID2 --field
{lemma,pos,dependency_relation} [--offset N] [--limit N]` command. It consumes
the existing `ScopeSnapshot` directly and does not create or expand membership.
Supply exactly one compatible parsed-token artifact per scoped revision. The
artifacts must agree on parser/model/schema identity. `lemma` counts stored
lemma annotations, `pos` counts POS tags, and `dependency_relation` counts
dependency labels (reported with dependency-parse field coverage). These are
counts from the selected parser artifacts, not new NLP runs or discourse
interpretations. Pages default to 100 rows and accept `--limit` up to 1,000;
JSON retains scope, artifact, coverage, and page metadata. The output formats
are `json`, `jsonl`, `csv`, and `table`.

```bash
discourselens corpus parsed-frequency-scope research SCOPE_ID \
  --artifacts PARSED_ARTIFACT_A,PARSED_ARTIFACT_B \
  --field lemma --offset 0 --limit 100 --format json
```

The revision-list form remains separate:
`parsed-frequency PROJECT --revisions REV_A,REV_B --artifacts ID_A,ID_B
--field lemma`. It creates a scope from those revisions rather than consuming a
saved scope.

## Saved-scope dependency patterns

`dependency-patterns-scope` reads parser-derived dependency candidates from an
existing `ScopeSnapshot` without creating a snapshot or changing project data.
Its syntax is:

```text
discourselens corpus dependency-patterns-scope PROJECT SCOPE_ID \
  --artifacts ID1,ID2 [--relation RELATION] [--head-lemma LEMMA] \
  [--dependent-lemma LEMMA] [--offset N] [--limit N]
```

Supply exactly one compatible `parsed_tokens` artifact per revision in the
saved scope; all selected artifacts must agree on parser, model, and schema
identity. The artifact list must cover the full frozen scope, not a subset.
Optional filters select an exact dependency relation and case-insensitive head
or dependent lemma. Each filter must be nonblank and at most 256 characters.
The CLI accepts 1..10,000 unique artifact IDs, each 1..128 characters.
`offset` is zero-based; `limit` defaults to 100 and is bounded at 1,000.

Each row reports `head_lemma`, `relation`, `dependent_lemma`, occurrence
`frequency`, and `document_frequency` (distinct scoped revisions). The page
also reports the frozen `scope_id`, selected artifacts, filters, pagination,
and parser coverage. `coverage.patterns` can be `LIMITED` when lemma or
dependency-parse annotations are incomplete even if the artifact set
covers every revision. These are syntactic candidates only, not discourse
interpretations or proof of agency, responsibility, intent, ideology, or
corpus-wide absence. Do not treat a filtered page as an exhaustive search for
all linguistic realizations.

The revision-list form is separate and creates a snapshot from explicit
revision IDs:

```text
discourselens corpus dependency-patterns PROJECT \
  --revisions REV_A,REV_B --artifacts ID_A,ID_B \
  [--relation RELATION] [--head-lemma LEMMA] [--dependent-lemma LEMMA]
```

```bash
discourselens run list research --offset 100 --limit 100
```

## Literal exact search

Use `search exact` when the selection is one document revision. It takes a
revision ID and searches only that revision; the command creates a one-revision
snapshot for the result. For a saved multi-revision selection, use
`search exact-scope` with its existing `ScopeSnapshot` ID:

```bash
discourselens search exact research REVISION_ID 'public interest' \
  --offset 0 --limit 100
discourselens search exact-scope research SCOPE_ID 'public interest' \
  --offset 0 --limit 100
```

`search exact-scope` reads the frozen membership as saved. It does not create or
expand a snapshot, include revisions imported later, or write project data.
Both commands return matching paragraphs in pages with `offset`, `limit`,
`total_count` for the complete selection, and `has_more` for continuation.
`--offset` is zero-based and non-negative; `--limit` defaults to 100 and accepts
1..1,000. The JSON response includes row data and scope metadata. The exact
search remains literal phrase matching; use `search lexical` for FTS syntax
over a saved scope.

For FTS syntax over an existing immutable scope, use `search lexical` with an
explicit scope ID. It returns at most 100 paragraph rows per page and examines
no more than 5,000 candidates. `has_more` refers only to the retrieved window;
`candidate_window_full` is true at the cap, where later matches may remain
unseen. Offsets through 4,999 are valid, including the final candidate:

```bash
discourselens search lexical research SCOPE_ID 'migration OR asylum' --limit 50
```

Frequency/KWIC and other corpus operations are deterministic and do not require
a configured LLM. A scanned PDF is not sent to a remote OCR service; local OCR
is a separate, explicit import option. See [Document Engine](architecture/DOCUMENT_ENGINE.md)
for extraction status and limitations.

`corpus ngrams` and `corpus collocations` keep their complete legacy output by
default. Add `--offset` and/or `--limit` to opt into a deterministic page (the
default page size is 100; maximum is 1,000). Paginated JSON wraps rows with
`offset`, `limit`, `total`, and `has_more`; table/CSV/JSONL formats emit the
selected rows and report page status on stderr. Paged n-grams and collocations
sort by descending count, then lexical value, matching the Python SDK and MCP
tie order:

```bash
discourselens corpus ngrams research REVISION_ID --size 2 --offset 0 --limit 100
discourselens corpus collocations research REVISION_ID migration --offset 100 --limit 100
```

For n-grams across an existing frozen `ScopeSnapshot`, use the read-only
`corpus ngrams-scope PROJECT SCOPE_ID --size N` command. It reads the saved
membership directly without creating or expanding a scope.

```bash
discourselens corpus ngrams-scope research SCOPE_ID --size 2 \
  --offset 0 --limit 100 --format json
```

`--offset` is zero-based and defaults to 0; `--limit` defaults to 100 and
accepts 1..1,000. `--format` accepts `json`, `jsonl`, `csv`, or `table`.
N-grams sort by descending count and then lexical value. JSON preserves the
Core page metadata: `scope_id`, `size`, `offset`, `limit`, `rows`,
`total_values`, and `has_more`. The existing
`corpus ngrams PROJECT REVISION_ID --size N` command remains revision-only and
keeps its complete legacy output by default.

`corpus collocation-analysis` pages its complete eligible association ranking.
The default page starts at offset 0 with limit 100; `--offset` and `--limit`
select later pages. JSON includes `offset`, `limit`, the exact eligible-row
`total`, and `has_more`; the legacy `truncated` field mirrors `has_more`.
Every page uses the same deterministic t-score, pair-frequency, and lexical
tie order within the explicit revision scope:

```bash
discourselens corpus collocation-analysis research REV_A,REV_B migration \
  --window 5 --min-frequency 2 --offset 100 --limit 100
```

To persist the exhaustive association evidence for one selected pair, the
revision-list form remains `corpus collocation-evidence PROJECT REVISIONS
QUERY COLLOCATE`; it constructs a scope from those revisions. If a frozen scope
already exists, use `corpus collocation-evidence-scope PROJECT SCOPE_ID QUERY
COLLOCATE [--window N] [--min-frequency N]`. It uses the existing
`ScopeSnapshot` directly—no new scope is created—and records exhaustive,
deterministic corpus-statistic evidence and coverage provenance for that
population. `--window` defaults to 5 and accepts 1..100; `--min-frequency`
defaults to 2 and must be positive. Mutual information (MI) and t-score are
descriptive association measures, not significance tests or p-values. `QUERY`
and `COLLOCATE` must each be one surface token of at most 256 characters; a
self-collocation is not supported.

```bash
discourselens corpus collocation-evidence-scope research SCOPE_ID migration policy \
  --window 5 --min-frequency 2
```

`corpus dispersion` likewise keeps its full legacy document listing unless
`--offset` or `--limit` is supplied. A paged response includes exact totals and
document frequency over the entire frozen revision selection, while
`documents` contains only the selected page:

```bash
discourselens corpus dispersion research REV_A,REV_B migration --offset 100 --limit 100
```

For an already-frozen `ScopeSnapshot`, use the read-only
`corpus dispersion-scope PROJECT SCOPE_ID TERM` command. It reads that saved
membership directly; it does not create a snapshot, add later revisions, or
write project data. Pages default to 100 documents and accept `--offset` and
`--limit` up to 1,000. JSON retains the full-scope totals and `scope_id`, with
only the selected document rows in `documents`:

```bash
discourselens corpus dispersion-scope research SCOPE_ID migration \
  --offset 0 --limit 100 --format json
```

`corpus lexical-bundles` returns a stable page ordered by descending bundle
frequency, then lexical bundle text. `candidate_count` is the full filtered
population; `offset`, `limit`, `has_more`, and `truncated` describe the selected
page:

```bash
discourselens corpus lexical-bundles research REV_A,REV_B --size 3 \
  --min-frequency 2 --offset 100 --limit 100 --format json
```

To analyze that bundle distribution over an already-frozen `ScopeSnapshot`, use
the read-only `corpus lexical-bundles-scope PROJECT SCOPE_ID` command. It reads
the existing membership directly and never creates or expands a scope. The
page is bounded (100 rows by default; `--limit` accepts 1..1,000), preserves
each complete bundle string, and reports the full filtered `candidate_count`:

```bash
discourselens corpus lexical-bundles-scope research SCOPE_ID --size 3 \
  --min-frequency 2 --min-documents 1 --offset 0 --limit 100 --format json
```

`--min-documents` sets the minimum number of scoped revisions containing a
bundle. `--offset` and `--limit` page the deterministic ranking by descending
frequency and then bundle text. This differs from
`corpus lexical-bundles PROJECT REV_A,REV_B`, which takes an explicit revision
selection and creates a scope for that selection. MCP exposes the same
read-only page as `corpus_lexical_bundles_for_scope(scope_id, size=3,
min_frequency=2, min_document_frequency=1, offset=0, limit=100)`.

To persist one exact bundle count over the existing frozen membership, use
`corpus lexical-bundle-evidence-scope PROJECT SCOPE_ID BUNDLE --size N`:

```bash
discourselens corpus lexical-bundle-evidence-scope research SCOPE_ID \
  "the public must" --size 3
```

This reuses the saved `ScopeSnapshot` without creating another one and persists
the count as `CorpusStatisticEvidence` with exhaustive `CoverageDescriptor`
provenance for all contiguous n-gram positions in the scope. It uses the
existing `surface-lexical-bundles-v1` tokenization. `BUNDLE` must contain
exactly `N` surface tokens, with `--size` between 2 and 10. This records a
descriptive count, not a CDA interpretation. The legacy
`corpus lexical-bundle-evidence PROJECT REVISION_IDS BUNDLE --size N` command
continues to take a revision list and creates a scope for that selection.
MCP exposes the same existing-scope write as
`corpus_lexical_bundle_evidence_for_scope(scope_id, bundle, size)`; it requires
a writable session.

For KWIC over parser lemmas, provide both an explicit immutable revision
selection and one compatible parsed-token artifact ID for each selected
revision. Lemma matching is exact and case-sensitive; the structured rows
include the canonical surface form, lemma, character offsets, and bounded token
context. Pages contain at most 1,000 rows. JSON preserves page metadata; JSONL,
CSV, and table render the selected rows:

```bash
discourselens corpus lemma-kwic research migrant \
  --revisions REV_A,REV_B \
  --artifacts PARSED_ARTIFACT_A,PARSED_ARTIFACT_B \
  --window 5 --offset 0 --limit 100 --format json
```

To persist one exact match as textual evidence, copy that row's
`revision_id`, `start_offset`, and `end_offset` into `lemma-kwic-evidence`.
The command verifies the offsets against the selected parsed artifacts and
canonical revision text, then returns the evidence ID, exact text, and citation:

```bash
discourselens corpus lemma-kwic-evidence research migrant \
  --revisions REV_A,REV_B \
  --artifacts PARSED_ARTIFACT_A,PARSED_ARTIFACT_B \
  --revision-id REV_A --start-offset 24 --end-offset 31
```

The command above is the revision-list form: it creates a scope from
`--revisions`. To preserve an already-frozen scope, use
`corpus lemma-kwic-evidence-scope PROJECT SCOPE_ID LEMMA --artifacts ID1,ID2
--revision-id REVISION_ID --start-offset N --end-offset N`. It consumes the
existing immutable `ScopeSnapshot` directly and does not create a new one. The
selected revision must belong to that scope, and the supplied parsed-token
artifacts must cover its revisions with compatible parser/model/schema
identity. Copy `start_offset` and `end_offset` from the exact lemma-KWIC row;
the Core verifies the occurrence against the selected artifact and canonical
revision text before persisting textual evidence.

```bash
discourselens corpus lemma-kwic-evidence-scope research SCOPE_ID migrant \
  --artifacts PARSED_ARTIFACT_A,PARSED_ARTIFACT_B \
  --revision-id REV_A --start-offset 24 --end-offset 31
```

To work with several revisions, create an explicit corpus and frozen subcorpus
using `corpus create`, `corpus add`, and `corpus create-subcorpus`. Comparison
commands take the selected revisions/corpora as explicit inputs. A later import
does not expand an already-created scope snapshot.

`corpus compare-frequency` and `corpus keyness` preserve their structured page
metadata in JSON. With `jsonl`, `csv`, or `table`, they emit only the selected
rows on stdout and report row range, total, and `has_more` on stderr.

For a comparison between two saved selections, use
`corpus keyness-scopes PROJECT LEFT_SCOPE_ID RIGHT_SCOPE_ID`. The command reads
the two existing frozen `ScopeSnapshot`s directly; it is read-only and does not
create or expand either scope.

```bash
discourselens corpus keyness-scopes research LEFT_SCOPE_ID RIGHT_SCOPE_ID \
  --offset 0 --limit 100 --format json
```

`--offset` defaults to 0 and is zero-based; `--limit` defaults to 100 and
accepts 1..1,000. `--format` accepts `json`, `jsonl`, `csv`, or `table`. JSON
preserves `left_scope_id` and `right_scope_id` with the page's `total_rows` and
`has_more` metadata. The other formats emit selected rows and report page
status on stderr, including both scope IDs for reproducibility. The existing
`corpus keyness PROJECT LEFT_REVISION RIGHT_REVISION` remains revision-only.
The MCP `corpus_keyness` tool already accepts left and right `ScopeSnapshot`
IDs directly.

`corpus snapshot PROJECT CORPUS_ID` freezes the corpus's current membership and
returns a reusable `scope_id` together with its revision IDs. Each invocation
creates a new snapshot; later corpus additions do not change that snapshot. Use
`--format text` for a concise human-readable listing instead of the default JSON.

```bash
discourselens corpus snapshot research CORPUS_ID
discourselens corpus snapshot research CORPUS_ID --format text
```

Read the complete membership of an existing immutable scope in bounded pages
with `corpus scope-revisions`. This command is read-only and returns the same
`revision_count`, `revision_ids`, and `has_more` page fields as the MCP reader;
`--limit` accepts 1..100 and an offset beyond the end returns an empty page.

```bash
discourselens corpus scope-revisions research SCOPE_ID --offset 0 --limit 100
```

Read frequency counts and KWIC results across an existing frozen scope with
`corpus frequency-scope` and `corpus kwic-scope`. Both commands require the
saved `ScopeSnapshot` ID; they are read-only and do not create a new snapshot,
add later revisions, or write project data.

```bash
discourselens corpus frequency-scope research SCOPE_ID \
  --offset 0 --limit 100 --format json
discourselens corpus kwic-scope research SCOPE_ID migration \
  --window 5 --offset 0 --limit 100 --format json
```

Each returns one bounded page. `--offset` is zero-based and defaults to 0;
`--limit` defaults to 100 and accepts 1..1,000. Frequency JSON reports the
page rows, `total_types`, full-scope `total_tokens`, and `has_more`. KWIC JSON
reports the matching `total_count`, `has_more`, and canonical hit offsets along
with the selected rows. `--window` sets the number of context tokens on either
side of a KWIC hit; it defaults to 5 and accepts 0..100. Use the returned
offsets when you need to retrieve or cite a particular source span.

To persist a selected hit from this page as textual evidence, pass its
`revision_id`, `start_offset`, and `end_offset`, the same query, and the same
frozen scope ID to `corpus kwic-evidence-scope`:

```bash
discourselens corpus kwic-evidence-scope research SCOPE_ID REVISION_ID \
  migration START_OFFSET END_OFFSET
```

The command rejects revisions outside the supplied scope and offsets that do
not identify a matching KWIC hit. On success, it writes the evidence record but
does not create a snapshot or expand the scope. Run it against a write-enabled
project. The existing
`corpus kwic-evidence PROJECT REVISION_ID QUERY START_OFFSET END_OFFSET`
command remains the one-revision form and creates its one-revision snapshot.

For window collocate counts over the same saved scope, use
`corpus collocations-scope PROJECT SCOPE_ID QUERY`. The query is one surface
token; the command counts the other surface tokens within the selected window
on either side of each occurrence. It reads the existing `ScopeSnapshot`
without creating or expanding it, including with later imports. It is read-only
and does not write project data.

```bash
discourselens corpus collocations-scope research SCOPE_ID migration \
  --window 5 --offset 0 --limit 100 --format json
```

`--window` defaults to 5 and accepts 1..100. `--offset` is zero-based and
defaults to 0; `--limit` defaults to 100 and accepts 1..1,000. Select JSON,
JSONL, CSV, or table output with `--format`. JSON includes `offset`, `limit`,
the full collocate total, `has_more`, and the selected rows; rows sort by
descending count and then token text. The existing
`corpus collocations PROJECT REVISION_ID QUERY` command remains the
single-revision form.

Saved profiles distinguish a joint scope from independent batch processing:
`profiles run PROJECT NAME REV_A,REV_B` executes one analysis over the explicit
multi-revision scope. `profiles batch PROJECT NAME REV_A,REV_B` creates one
independent `AnalysisRun` per revision; the singular `profile` command alias is
also available. The batch selection is validated in full before any run is
created. Add `--continue-on-error` to return a status for every selected
revision and preserve successful results when another revision fails. Failure
items include a safe error type/message and the failed `analysis_run_id` when
one was persisted. Retry selected failures with `profiles batch` and only those
revision IDs; resume a failed run by ID with
`analyze resume-controlled-cda-review PROJECT ANALYSIS_RUN_ID`.

## Provider and model setup

Credential references save metadata, not secret values. To enter a provider key,
use `providers store-key`; it prompts with hidden input and stores the value in
the OS keyring. The service and account are deterministic from the provider ID
and alias, so the resulting reference is safe to record in a project. The command
checks for an existing entry before prompting and again before writing; it does
not accept the secret as an argument or environment override. OS keyring APIs do
not provide a portable atomic create-if-absent operation, so concurrent processes
must not store or remove the same alias.

```bash
discourselens providers store-key openai university
```

It prints `keyring:discourselens:openai/university`. Add that reference to the
project separately; project YAML, reports, and source control should contain the
reference only:

```bash
discourselens providers add-credential research openai university \
  --secret-reference keyring:discourselens:openai/university
discourselens models save-preset research analysis openai university MODEL_ID
discourselens models list-presets research
```

For environment-managed credentials, `add-credential --secret-env NAME` remains
available and stores only the environment-variable name.

To remove a keyring value, identify it by the same provider and alias. The
command asks for confirmation and deletes only the derived
`keyring:discourselens:<provider>/<alias>` entry:

```bash
discourselens providers remove-key openai university
```

`--yes` skips that prompt for automation. Use it only when the provider and
alias are known: it permanently removes that exact OS-keyring entry. It does not
delete the credential reference or model presets saved in a project, so they
will be unable to resolve the key until you store it again with
`providers store-key`. The keyring API has no portable atomic compare-and-delete;
avoid changing the same alias from another process while this command runs.

Compatible local endpoints must be configured explicitly. Use
`models preview-request` before generation to inspect the selected destination
and request metadata without resolving secret values or making a network call.
Remote generation, embedding, reranking and OCR can send selected text outside
the project; inspect the destination and selected evidence/scope before running.
`project offline PROJECT --enabled` blocks non-loopback remote operations while
leaving deterministic local corpus tools available.

`models generate PROJECT PRESET --prompt-file FILE` preserves its buffered JSON
result by default. To display completion text rather than JSON, add
`--format text`. Opt-in streaming is limited to that text format:

```bash
discourselens models generate research analysis --prompt-file question.txt \
  --stream --format text
```

Streaming requires an exact-model capability profile declaring
`streaming: true`; it fails closed when that declaration or the adapter's
streaming implementation is unavailable. Streamed deltas go only to stdout;
diagnostics go to stderr. `--stream` with JSON is rejected rather than changing
the existing machine-readable response contract. A completed stream records
request/response hashes, not prompt or completion text; interrupted streams are
recorded as incomplete and do not receive a response hash.

Buffered Gemini, OpenAI Chat Completions, and Anthropic Messages generation can
include one explicitly selected local PNG, JPEG, or WebP image:

```bash
discourselens models generate research analysis --prompt-file question.txt \
  --image figure.png --format text
```

Image input requires a capability profile for the exact `(provider_id, MODEL_ID)`
with `vision: true`; it is not enabled by default and is not available to other
providers or streaming. The CLI validates the local file's extension, signature,
and size before execution. One image is limited to 8 MiB and the fully encoded
provider request to 18 MiB. A remote endpoint receives the image bytes as part
of that explicit request. The execution ledger stores only the image SHA-256,
MIME type, and byte size, never its path or bytes. Gemini's `generateContent`
image part uses inline `inline_data`; see Google's [image understanding](https://ai.google.dev/gemini-api/docs/generate-content/image-understanding)
and [text generation](https://ai.google.dev/gemini-api/docs/generate-content/text-generation)
references. OpenAI Chat Completions uses a data URL in an `image_url` content
part; see the [OpenAI image input guide](https://developers.openai.com/api/docs/guides/images-vision).
Anthropic Messages uses a base64 `image` content block with a `source` object;
see Anthropic's [vision guide](https://docs.anthropic.com/en/docs/build-with-claude/vision).

Model capabilities are exact-model declarations and may narrow, but never
invent, what the chosen adapter can transmit. Reasoning options are accepted
only when supported by that selected model/adapter. See
[Provider Setup](PROVIDERS.md) and [Model Routing](architecture/MODEL_ROUTING.md).

Optional cost estimates use rates you supply explicitly; the CLI does not look
up provider prices or contact a pricing service. Rates are per million
provider-reported tokens. For example (the rates are illustrative):

```bash
discourselens models price set research openai MODEL_ID \
  --input-per-million-tokens 0.25 \
  --output-per-million-tokens 1.25 \
  --currency USD \
  --source-reference 'https://provider.example/pricing' \
  --rate-version docs-2026-01 \
  --rate-date 2026-01-15
discourselens models price list research --provider openai --model MODEL_ID
discourselens models costs research --format json --limit 100 --offset 0
```

Add a new rate version when prices change; the same version cannot be changed
to different rates. Each execution stores the selected rate snapshot and its
estimate, so `models costs` reports historical stored values and never
re-estimates them from the current price record. It includes executions with
`null` estimates when no rate or required usage count was available. An
optional `--reasoning-per-million-tokens` is used only when you explicitly
provide a separate reasoning rate. These values are estimates based on
provider-reported usage and your supplied rates, not guaranteed invoice
amounts. Do not put credentials in a source reference.

`models costs` is paginated: `--limit` defaults to 100 and accepts at most
1,000 rows; `--offset` is a zero-based offset. JSON returns an envelope with
`offset`, `limit`, `total_count`, `has_more`, and `rows`. JSONL begins with a
`{"_page": ...}` metadata line followed by the execution rows with their usual
field names. Table and CSV output contain only the selected page's rows.

## Vector search

Build vector artifacts from an explicit revision, then search within that same
revision and the selected artifacts:

```bash
discourselens vectors preview-paragraphs research embed REVISION_ID
discourselens vectors build-paragraphs research embed REVISION_ID
discourselens vectors search research embed REVISION_ID ARTIFACT_ID \
  --query-file query.txt
discourselens vectors hybrid-search-preset research embed REVISION_ID ARTIFACT_ID \
  --query-file query.txt
```

Indexing may send the revision's paragraph text to the configured embedding
provider; semantic and preset-backed hybrid search may send the query text.
Review the preview and provider endpoint first. Query text is read from a UTF-8
file so it is not placed in shell history. The preset-backed hybrid command
checks selected-index provenance before embedding and fuses lexical and vector
candidates only inside the supplied revision snapshot. The lower-level
`vectors hybrid-search` command accepts a caller-supplied numeric vector; it
cannot infer that vector's model provenance and is intended for expert/local
integrations.

Semantic and hybrid search return a JSON envelope containing `scope_id`,
`retrieval_run_id`, and `hits`, including when no hits are found. If a `scope`
argument is a revision ID, the command creates and returns the corresponding
immutable scope snapshot ID; passing an existing scope ID reuses it. Hybrid hits
preserve their lexical and vector ranks and identify the vector artifact when a
vector candidate contributed.

## Analysis and provenance

```bash
discourselens analyze evidence-extraction research REVISION_ID --languages en
discourselens analyze controlled-cda-review research \
  --revisions REVISION_ID --preset analysis
discourselens run inspect research ANALYSIS_RUN_ID
discourselens finding validate research FINDING_ID
discourselens finding sources research FINDING_ID
discourselens report analysis research ANALYSIS_RUN_ID
discourselens export research-bundle research ANALYSIS_RUN_ID ./bundle
discourselens export verify-bundle ./bundle/manifest.json
```

`export research-bundle` excludes full canonical document text by default, but
still includes cited evidence excerpts. An excerpt can be long or even cover an
entire revision if that was the saved evidence span. Pass
`--include-canonical-text` only when the bundle should also contain full
canonical text for revisions referenced by cited textual evidence. Source text
may be sensitive; review the excerpts, destination, and recipients. The full
manifest is still limited to 64 MiB, so an export that exceeds that bound fails
rather than producing an oversized bundle. Use
`discourselens export research-bundle --help` for the option's privacy notice.

`analyze textual-descriptors` accepts one revision ID, as before, or a
comma-separated ordered selection. The exact selection is recorded in the
immutable analysis scope; empty entries and duplicate IDs are rejected before
the project is opened.

```bash
discourselens analyze textual-descriptors research REV_A,REV_B --languages en,es
```

`run inspect` returns the complete provenance summary when both paging options
are omitted. Supplying either `--offset` or `--limit` pages every provenance
collection; the omitted option defaults to offset `0` or limit `100`. The maximum
page size is 100, and JSON includes per-collection totals and `has_more` flags.
Advance the zero-based offset to read the next page:

```bash
discourselens run inspect research ANALYSIS_RUN_ID --offset 100 --limit 100
```

Controlled review is evidence-bound and records its scope, skill/prompt versions,
model routing and execution provenance. Findings distinguish observations from
interpretations and retain alternative readings, counterevidence and limitations
when required by their contract. Model output is not itself a source of citation
locations: the Core renders citations from persisted evidence records. The
`finding validate` command checks structural provenance, scope and evidence
references; it does not establish semantic entailment. Its output explicitly
reports `semantic_entailment_assessed: false`, so a valid result still requires
researcher review of whether the cited text supports the interpretation.

`finding sources` shows the persisted lineage of a synthesized finding as a
JSON object containing its ID and source finding IDs with relation types. It
returns an empty `sources` array for a finding without lineage and does not
include source observations or evidence text.

For batch analysis, the batch commands create independent runs per revision;
they do not concatenate documents. A failed resumable controlled review can be
continued with `analyze resume-controlled-cda-review` without replaying completed
stages.

### Deterministic parser candidate analyzers

`analyze social-actor-candidates` and `analyze agency-candidates` take an
explicit immutable revision scope through one or more repeated `--revision`
options, plus one or more explicit parsed-token artifact IDs through repeated
`--artifact` options. Each artifact must be a `parsed_tokens` artifact belonging
to a selected revision; at most one parsed-token artifact may be supplied per
revision. The commands use only those artifact IDs and do not find or build
artifacts implicitly.

```bash
discourselens analyze social-actor-candidates research \
  --revision REVISION_A --revision REVISION_B \
  --artifact PARSED_TOKENS_A --artifact PARSED_TOKENS_B

discourselens analyze agency-candidates research \
  --revision REVISION_A --revision REVISION_B \
  --artifact PARSED_TOKENS_A --artifact PARSED_TOKENS_B
```

The social-actor command records parser-tagged noun, proper-noun, and pronoun
expressions as candidates. The agency command records dependency-relation
candidates linking a dependent expression to its predicate, and requires
`dependency_parse` to be `SUPPORTED` for each selected parser artifact; it
rejects missing, mismatched, or other capability statuses. These deterministic
outputs are evidence-linked observations for review. They do not classify
social or discursive representation, interpret agency or responsibility, or
infer author intention, and these commands do not call an LLM.

## Individual built-in skills

`skills scope-question-candidates` runs bounded lexical retrieval inside only
the immutable revisions explicitly supplied with repeated `--revision`
options. Select an interpretive `--framework` and supply a researcher-authored
`--question`; `--limit` accepts 1–12 and defaults to 12. JSON includes run,
scope, evidence, and coverage provenance. `jsonl`, `csv`, and `table` return a
provenance summary. Retrieval coverage is always `TARGETED`: zero results mean
only that the lexical plan found no examples, not that the phenomenon is absent.
The passage candidates are inputs for separate framework interpretation.

```bash
discourselens skills scope-question-candidates research \
  --revision REVISION_A --revision REVISION_B \
  --framework fairclough-text \
  --question "How are workers represented?" --limit 12 --format json
```

`skills quote-attributions` runs the deterministic quote-mark inventory only
over the immutable revisions explicitly supplied with repeated `--revision`
options. Without `--language`, it reports paired quotation candidates without
searching for reporting cues. `--language en` or `--language es` applies that
explicit language assignment to every selected revision. JSON returns the full
run/evidence provenance; row formats return candidate rows and write bounded
run metadata to stderr.

```bash
discourselens skills quote-attributions research \
  --revision REVISION_ID --language en --format json
```

`skills van-dijk-group-comparison` prepares a descriptive comparison from two
researcher-labeled groups of existing qualitative annotation IDs. Revisions
must be explicitly selected, and the input JSON contains exactly `groups`,
`shared_topic_event`, `comparability_basis`, and `codebook_id`. Each group has a
label and its selected `annotation_ids`. Counts cover those selected records
only; an empty group's coding is `UNKNOWN`, not evidence of absence. The
operation does not infer ideology, political identity, or an asymmetry score.

```json
{
  "groups": [
    {"label": "Resident accounts", "annotation_ids": ["ANNOTATION_A"]},
    {"label": "Official statements", "annotation_ids": ["ANNOTATION_B"]}
  ],
  "shared_topic_event": "the road safety announcement",
  "comparability_basis": "both sets address the same announced review",
  "codebook_id": "CODEBOOK_ID"
}
```

```bash
discourselens skills van-dijk-group-comparison research \
  --revision REVISION_A --revision REVISION_B \
  --input comparison.json --format json
```

## Python and MCP

Use `Project` from the Python SDK for scripted workflows; it exposes the same
Core operations without shelling out to this CLI. The optional MCP adapter is
launched with stdio and does not require a port. See [MCP Setup](MCP_SETUP.md)
for local and Docker launch examples, and [README](../README.md) for installation
and SDK quick start.
