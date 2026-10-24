# Architecture validation report

Evidence review: 2026-10-05. This report reconciles
[SPIKE-001](spikes/SPIKE-001-nlp-storage.md)
through [SPIKE-005](spikes/SPIKE-005-language-capabilities.md) with the current
implementation, `.github/workflows/ci.yml`, and the local offline package probe
recorded below. It describes implementation and observed evidence; it is not a
formal architecture freeze or a product-wide readiness approval.

Status terms used here:

- **Implemented** means the current source contains the described behavior.
- **Measured** means a bounded local experiment or smoke produced the recorded
  result.
- **Partial** means evidence covers only a stated environment, workload, or
  technical capability.
- **Unmeasured** means this evidence set does not establish the claim.
- **Pending decision** means an explicit product, legal, or architecture
  decision is still required.

## Evidence lineage and claim constraints — IMPLEMENTED; method validation PARTIAL

The project implements immutable document revisions, canonical-text source spans,
persisted evidence lineage, immutable scope snapshots, and structural checks for
coverage and claim types. The current contracts and persistence paths are in
`src/discourselens/types.py`, `src/discourselens/project.py`,
`src/discourselens/migrations.py`, and `src/discourselens/policy.py`; regression
coverage is present in `tests/test_foundation.py`, `tests/test_invariants.py`, and
`tests/test_validation.py`.

These checks constrain provenance, scope, and which claim forms are structurally
eligible. They do not establish that evidence semantically entails an
interpretation, nor validate a CDA method or its conclusions.

## Skills and execution boundary — IMPLEMENTED; methodology validation PARTIAL

Skill manifests, a built-in registry, declared execution kinds, capability
eligibility, and scope-bound execution are implemented in `src/discourselens/`.
The counterevidence flow requires the finding's original `ScopeSnapshot` and
records bounded retrieval coverage. The registry does not import project-supplied
Python or shell code. Tests cover manifest, scope, and execution contracts, but
their presence is not evidence that every methodology is validated. No
third-party executor is enabled.

## NLP storage — MEASURED on a matched synthetic workload; real-corpus behavior UNMEASURED

SQLite/FTS5 remains the canonical project and lexical-search store. Parquet/Zstd
token artifacts are derived and rebuildable; DuckDB is the analytical query
engine in the optional `corpus` extra. The corrected 10M comparison uses the same
deterministic rows and real character offsets for SQLite and Arrow/Parquet.
Separate runs did not control page-cache state and do not establish production
throughput or linguistic quality.

At 10M tokens, Parquet/Zstd wrote in 21.10 s, occupied 62,897,477 bytes (60.0
MiB), peaked at 182.5 MiB RSS, and DuckDB verified all 10,000,000 rows. SQLite
token rows occupied 958,226,432 bytes (913.7 MiB); its bounded write plus six
query measurements took 155.17 s, with 24.5 MiB peak RSS. Across three query
executions, Parquet/DuckDB was faster for lemma frequency (52.6 vs 186.14 ms),
document frequency (86.6 vs 1,822.7 ms), POS filtering (52.3 vs 3,437.01 ms),
dependency filtering (52.5 vs 4,193.13 ms), and per-document aggregation (94.4
vs 2,175.04 ms). Indexed SQLite was faster for the measured KWIC candidate query
(0.13 vs 6.8 ms). A batch-vectorized SQLite sentence-blob artifact occupied
228,134,912 bytes (217.6 MiB) after a 137.85 s write and passed `quick_check`,
but its one-pass 10M lemma scan did not finish within the remaining 102 s of a
240 s bound. No complete blob query time is claimed. The earlier 36.1 MiB
Parquet result used synthetic offsets and is explicitly non-comparable.

Both matched 10M data and earlier 1M data are deterministic synthetic workloads.
Larger real corpora, model quality, and multilingual NLP quality remain
unmeasured.

## Corpus and document processing — IMPLEMENTED; scale evidence PARTIAL

The code implements canonical document storage, SQLite FTS5 search, document
extraction, and optional derived token artifacts. See
[`DOCUMENT_ENGINE.md`](DOCUMENT_ENGINE.md) for the extraction and source-span
projection limits. SPIKE-001 measures a synthetic 10M-token artifact and selected
queries; it does not validate the behavior or performance of arbitrary user
corpora. Parquet does not replace FTS5.

## Vector retrieval — IMPLEMENTED in code; formal backend approval PENDING

The current implementation in `src/discourselens/vectors.py` writes
rebuildable per-revision sqlite-vec artifacts. New v4 artifacts explicitly use
cosine distance. Historical v1-v3 artifacts retain their original default
squared-L2 score and are never rewritten; preset-backed semantic queries require
rebuilding those derived indexes. Raw searches may still use an old index, but
its score definition is reported truthfully, and mixed-metric searches are
rejected before ranking. Scoped search validates artifact membership before
ranking; preset queries check provider, model, dimension, chunker, input type,
and endpoint provenance before making the embedding request. Regression tests
for metric/version compatibility, endpoint binding, index scope, and provenance
are present in `tests/test_vectors.py` and `tests/test_scope_retrieval_isolation.py`.
Published vector artifacts are also checked against their actual sqlite-vec
table DDL at validation and search time: a v4 artifact must declare cosine,
while historical v1-v3 artifacts retain their legacy L2 interpretation. This
prevents a rewritten or mislabeled artifact from silently changing score
semantics.

SPIKE-002's final 2026-09-30 comparison supersedes earlier mechanics figures. It
used deterministic random 768-dimensional vectors and single top-10 queries, not
representative semantic embeddings or warmed repeated-query medians. At 10,000
vectors, sqlite-vec built in 2.397 s, queried in 41.933 ms, and persisted
32,284,672 bytes; FAISS `IndexFlatL2` took 0.033 s / 11.577 ms / 30,720,045
bytes; LanceDB took 0.154 s / 73.428 ms / 30,794,184 bytes.

At 100,000 vectors, sqlite-vec took 52.854 s to build, 436.870 ms for the first
top-10 query, and persisted 316,776,448 bytes. FAISS took 1.411 s / 72.722 ms /
307,200,045 bytes. LanceDB took 3.169 s / 1.290 s / 307,992,218 bytes. Each
candidate returned ten hits after build and after copying/reopening its artifact;
the ordered top-10 IDs matched. The individual copy and reopen/query timings are
recorded in SPIKE-002. Thus FAISS leads these bounded mechanics measurements,
while LanceDB's 100k query and reopen-query were slower than the other candidates.

The strict-scope fixtures passed for all three candidates: sqlite-vec and FAISS
searched only the selected per-revision index; LanceDB used a `revision_id`
metadata prefilter. The strengthened fixture first proved that an unfiltered
search returns closer results from B, then proved an A-scoped search returns
only A. All three candidates also passed append, delete, fresh rebuild, copy,
reopen, and top-10 identity checks at 10k and 100k. FAISS was the fastest
measured 100k rebuild (0.490 s), while sqlite-vec took 38.459 s; detailed
single-run lifecycle measurements are in SPIKE-002. These candidate fixtures
check scope-selection mechanics,
not their integration into the application APIs. Separately, regression tests
exercise the current sqlite-vec search APIs with local fixture indexes.
`search_vectors` covers document, corpus, subcorpus, and metadata scopes; the
`test_hybrid_search_with_preset_embeds_once_and_keeps_scope_strict` test in
`tests/test_vectors.py` uses an injected embedding fixture, returns an in-scope
result, and rejects an out-of-scope index before another query-embedding call.
This establishes scope enforcement on the tested current API path under
fixtures; it does not establish behavior with live remote providers or
production corpora.
The global merge retains at most the requested top-k `VectorHit` objects across
revision-local indexes, so retained hit memory is O(k), not O(k × revision
count). It still opens/searches each explicitly selected revision index, and
hybrid lexical inspection still scales with the exact-match population; these
are work/fan-out costs rather than result-memory growth.
In particular, FAISS has no metadata predicate, so its correctness depends on
selecting only the approved index. Random vectors do not measure semantic
relevance, and injected providers do not measure remote-provider quality. The
results support FAISS as a leading mechanics candidate where separate per-
revision index lifecycle is acceptable; they do not establish final backend
selection, relevance quality on real embeddings, many-revision behavior, or
operational lifecycle tradeoffs. A final multi-revision fan-out run measured
100 revisions × 1,000 vectors each using the legacy L2 behavior: sqlite-vec
build took 221.444 s and global top-10 search took 785.892 ms; FAISS took
0.755 s and 389.827 ms, with identical top-10 IDs and selected-scope prefilter
behavior. This harness did not include project-registry validation, atomic
publication, or a durable FAISS chunk map. A smaller 10-revision explicit-cosine
run also matched top-10 IDs: sqlite-vec took 26.623 s to build and 73.206 ms to
search all revisions; FAISS took 0.095 s and 34.681 ms. Those are single-run
synthetic mechanics observations, not semantic-quality evidence or a 100k
cosine result. The implemented sqlite-vec backend remains the current code path;
formal backend approval and the product decision remain pending.

## Packaging and supported platforms — Linux locally TESTED; hosted platform jobs CONFIGURED

The declared Python minimum is 3.12. The checked-in CI workflow configures
native smoke jobs on Ubuntu, macOS, and Windows for Python 3.12, 3.13, and 3.14.
It also configures a Linux quality job on Python 3.12 that builds and inspects a
wheel, exercises project and MCP workflows, and runs the full test suite and
static checks. These are workflow definitions, not evidence that any hosted
job has passed. The native smoke step uses Bash on each matrix runner; hosted
Linux, macOS, and Windows results remain unverified in this evidence set.

On 2026-10-04, the native smoke commands were also run in Linux containers
with `uv` 0.12.19 and the locked dependencies on CPython 3.13.16 and 3.14.8.
Both reported `DiscourseLens 0.1.0` and passed all 59 tests across
`test_foundation.py`, `test_pdf_documents.py`, `test_textual_documents.py`,
and `test_packaging_smoke.py`. The full locked Python 3.12.14 suite and quality
checks are recorded in the local verification section below. These runs add
Linux coverage; they do not establish macOS, Windows, or hosted CI results.

The local offline package probe ran on Linux x86_64 with CPython 3.12.8 and
3.14.4. On both interpreters, isolated offline installation and import passed,
as did CLI help, project initialization, document import, duplicate detection,
and an SDK reopen check. This establishes only the listed local packaging and
workflow paths on Linux for those interpreter versions. It does not establish
macOS or Windows compatibility, hosted CI results, or full test-suite support
on either interpreter. SPIKE-003's separate CPython 3.12.8 `uv` co-installation
measurement likewise did not test other operating systems.

On 2026-10-02, a freshly built wheel was installed with `uv pip --target` and
imported from that installed path (not the checkout) under the existing Linux
CPython 3.12.8 experiment environment. The packaged CLI version, project
creation/reopen, and an MCP SDK stdio `initialize`/`tools/list` exchange passed;
the MCP server exposed 98 tools and served the bundled evidence-extraction
methodology resource. This checks wheel contents and installed entry points
using already-present runtime dependencies; it is not a fresh dependency
resolution, named-host compatibility test, or hosted CI result.

CI also defines constrained Docker MCP and local-OCR workflows. Those checks
are configured; no hosted result is asserted here. Current optional groups are
listed in `pyproject.toml`; NLP model packages and model files remain separately
selected local assets.

## Docker and MCP — IMPLEMENTED; production stdio MEASURED locally; host compatibility UNMEASURED

On 2026-09-30, the production default image built locally at 299,581,780 bytes
(about 285.7 MiB); it contains the CLI, corpus, vectors, and MCP extras, while
OCR and local NLP packages remain opt-in. It has no declared/exposed ports and
runs as UID 10001 by default. A real MCP SDK `initialize` and `tools/list`
exchange against an explicit temporary project passed over Docker stdio with
`--network none`, a non-root UID, read-only root filesystem, and zero published
ports (17 tools in the tested read-only profile). The default CLI `--version`
and `--help` smoke also passed. The rendered Compose configuration had no
`ports` mapping and required an explicit project mount.

The opt-in OCR image built locally at 342,698,886 bytes (about 326.8 MiB). A
generated image-only PDF was imported with Docker networking disabled and the
root filesystem read-only; explicitly requested local Tesseract OCR returned
`SUCCESS_WITH_WARNINGS`, persisted its language/DPI/extractor provenance, and
made no cloud request. This is a narrow English fixture test, not a general OCR
accuracy claim. An optional NLP build resolved spaCy, Stanza, and PyTorch in the
build stage, but its image export was cancelled after 479 seconds; no final NLP
image or size is claimed. That Linux resolution included multiple CUDA runtime
wheels without a configured GPU, so the profile remains materially larger and
non-default. CI also defines Docker and local-OCR checks, but no hosted workflow
result is asserted here. No named desktop or IDE host compatibility result is
available.

## Language capabilities — IMPLEMENTED; local technical fixtures TESTED; CDA validation UNMEASURED

The expanded SPIKE-005 harness ran eleven curated fixtures per language using
spaCy 3.8.16 with `en_core_web_sm` and `es_core_news_sm` 3.8.0, and Stanza
1.14.0's local Arabic `default` package. Sentence segmentation, tokens, lemmas,
POS, morphology, dependencies, and NER were `SUPPORTED/TESTED` in those fixture
runs. In the production capability probe, token output must cover every
non-whitespace source character exactly once with matching offsets; empty or
partial token coverage leaves sentence segmentation and NER `LIMITED`, even if
the selected route advertises those components. The listed local fixture routes
passed that gate. Each input yielded one sentence. These are local technical
observations; they do not validate any CDA method or establish general language
accuracy.

| Language | Tokens / morphology-bearing | NER actor spans | Agency cues (fixture hits) | Passive cues (fixture hits) | Modality cues (fixture hits) | Social-actor cues (fixture hits) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English | 78 / 70 | 3 | 12 (11) | 6 (3) | 3 (3) | 22 (11) |
| Spanish | 78 / 65 | 4 | 12 (11) | 0 (0) | 2 (2) | 24 (11) |
| Arabic | 64 / 46 | 4 | 11 (10) | 0 (0) | 2 (2) | 27 (11) |

For candidate columns, parentheses indicate fixtures with one or more cues.
Totals may overlap across syntactic and NER signals and are not unique,
gold-labeled actor mentions. English agency/passive/modality/social-actor
candidates were observed. Spanish passive and Arabic passive candidates were
`LIMITED` because these fixtures produced no such cue; the other listed
candidate groups produced at least one cue. These preprocessors nominate
material for review and do not infer agency, responsibility, stance, or
modality.

The production capability probe and operation-level eligibility gates are
implemented in `src/discourselens/languages.py`. A production CLI smoke in the
isolated experiment environment selected the installed spaCy routes for English
and Spanish and the local Stanza `ar-padt` route for Arabic. All three exited
successfully; after complete token coverage, the selected routes reported their
tested token, lemma, POS, morphology, dependency and segmentation operations as
`SUPPORTED/TESTED`, while Arabic NER was `UNSUPPORTED/TESTED`. Exact search, surface frequency, and KWIC
were `SUPPORTED/TESTED` for all three. In the main project environment,
language-dependent capabilities remain `UNASSESSED` when optional packages or
models are absent; Arabic remains unassessed in the default registry until a
Stanza route is explicitly selected. The separate Stanza `default` package's
NER result must not be conflated with `ar-padt`. No CDA skill is claimed
`VALIDATED` by these results.

## Licensing — PUBLIC AGPL LICENSE APPLIED; PAID COMMERCIAL PATH PENDING

The current production package declares PyMuPDF as a required base dependency
(`pyproject.toml`), and the default Docker image includes it. Upstream documents
PyMuPDF/MuPDF as available under GNU AGPL or a commercial license from Artifex
([licensing FAQ](https://pymupdf.readthedocs.io/en/latest/faq/index.html),
[licensing information](https://pymupdf.io/licensing)). The repository now
applies SPDX `AGPL-3.0-only` to DiscourseLens-authored software in the public
release path. This is consistent with the public AGPL route for PyMuPDF, subject
to review of the actual integration and release artifacts. The AGPL allows
commercial use when its terms are followed, so it does not require every
commercial user to pay. A separate paid DiscourseLens license for proprietary
terms remains a possible path only for rights its signers control; it does not
replace Artifex's separate license. Artifex describes its commercial agreements
as case-specific and offers SaaS/OEM models ([licensing information](https://artifex.com/licensing)).
See [`docs/LICENSING.md`](../LICENSING.md). SPIKE-006 found
`pdfplumber` (MIT) returns character geometry for selected text but is not
drop-in and changes canonical text/offsets; `pypdf` (BSD-3-Clause) did not return
selected-glyph boxes in the fixture. A backend change would require a versioned
adapter and revision/re-extraction policy.

The isolated SPIKE-005 environment's package metadata declares MIT for spaCy and
`en_core_web_sm`, GNU GPL 3.0 for `es_core_news_sm`, and Apache License 2.0 for
the Stanza library. The license for Stanza's Arabic model assets and their
training data was not established by that package metadata. These declarations
are evidence about the installed packages, not a complete asset or data
provenance review. The default Docker dependency closure also contains
permissively licensed DuckDB, PyArrow, sqlite-vec and MCP packages, but this
inventory is not a substitute for legal review. The public AGPL license is
applied, while confirmation of authority over all project contributions,
third-party distribution terms, the paid commercial contract/contact path, and
GPL/model/data obligations remain release tasks. No separate proprietary
commercial agreement has been created.

## Local verification — 2026-10-04

The complete test suite passed in a Linux container with CPython 3.12.14,
`uv` 0.12.19, and the exact `uv.lock` resolution: 3,792 passed and 3 skipped
in 363.16 s. `ruff check src tests`, `pyright` (0 errors, warnings, or
informational diagnostics), and `compileall` also passed in that environment.
The CI native-smoke command set also passed on Linux containers with CPython
3.13.16 and 3.14.8: the CLI reported version 0.1.0 and all 59 selected
Foundation, PDF, textual-document, and packaging tests passed on each version.
An earlier host run under CPython 3.14.4 with pip-resolved dependencies passed
3,790 tests with 5 skipped; it is supplemental evidence, not a substitute for
the locked Linux run.

A wheel built from the current checkout was installed into a temporary target
and exercised through the MCP SDK. Its stdio server initialized, listed 115
tools, and returned the bundled evidence-extraction methodology resource. The
default Docker image built from the locked dependency set, ran the CLI help
smoke, and reported image user `10001:10001` with no exposed ports. A Docker
stdio MCP initialize/tools-list exchange also passed with networking disabled,
a read-only root filesystem, dropped capabilities, and no published ports.
The read-only Docker profile also completed a real MCP stdio handshake and
listed 45 tools from a read-only project mount. The opt-in OCR image imported
an image-only PDF with no network and a read-only root filesystem; it returned
`SUCCESS_WITH_WARNINGS` with `LOCAL_OCR_ENABLED`, and the persisted canonical
text matched the fixture. That image contained Tesseract 5.5.0-1+b1 and English
and Spanish data 1:4.1.0-2; these Debian versions remain build-time dependent.
Rendered Compose configuration likewise contained no host-port mappings.

These local checks do not establish hosted CI outcomes on Linux, macOS, or
Windows. This checkout has no configured Git remote and no `gh` executable, so
hosted workflow results could not be queried. The checks also do not resolve the
legal, methodology, representative retrieval-quality, or formal architecture
decisions below.

## Follow-up release verification — 2026-10-05

After adding the browser-based qualitative coding and Van Dijk group-comparison
workflow, the full local suite passed: **3,800 passed, 5 skipped** in
147.72 s. `ruff check src tests`, `pyright src tests` (0 errors, warnings, or
informational diagnostics), and `py_compile` over all Python files passed.
Compose configuration validation passed for the web deployment and the
OAuth-only writable MCP overlay. No Snyk scan was run.

The rebuilt `discourselens:release-check` image was exercised as a non-root
container with a read-only root filesystem, dropped capabilities, and a bound
project directory. A real Chrome browser authenticated to the web workspace,
imported text containing a non-BMP Unicode character, selected and coded
passages in the rendered paragraphs, and prepared the two-group Van Dijk
comparison. The result retained the selected text and canonical codepoint
offsets and was labeled `EXAMPLE_BASED`. A separate container stop/reopen flow
confirmed that annotations and rendered citations persisted in the mounted
project. This verifies the local application path; it does not validate an
external user's OAuth login, a public domain/reverse proxy, or the linguistic
adequacy of the analyst's coding and comparison.

## Overall status

Foundation Phase 1 and later implementation authorization are historical context,
not a blanket sign-off for the current architecture. Current implementation,
bounded measurements, and pending decisions are listed separately above. In
particular, formal vector-backend approval and legal licensing approval remain
pending; semantic quality, remote-provider quality, method validation, named-host compatibility,
and cross-platform CI outcomes are not established by the evidence summarized
here.

## Recommended readiness — CONDITIONAL; formal freeze not yet supported

The current evidence supports a provisional implementation baseline: local
100k-vector mechanics and strict-scope fixtures completed, and local multilingual
technical fixtures exercised the listed NLP operations. It does not support
final vector-backend selection, production semantic-retrieval claims, or CDA
validation. Keep sqlite-vec as the implemented code path behind the backend
boundary while the decision remains pending; FAISS is the mechanics leader, not
an approved replacement.

Before a formal architecture freeze or multilingual CDA readiness claim:

- Evaluate representative project embeddings with relevance judgments and
  end-to-end retrieval workloads; measure multi-revision lifecycle and validate
  production-path scope integration for replacement backend strategies. The
  current sqlite-vec API path has fixture-based scope regression coverage, but
  the benchmark candidate fixtures do not establish application integration.
- Resolve model and resource licensing, including the Spanish model's declared
  GNU GPL 3.0 and the unverified license lineage for Stanza's Arabic weights and
  training data.
- Validate each intended CDA method with language-specific annotations and
  expert review. Spanish and Arabic passive-candidate fixtures currently report
  `LIMITED`; keep those operation gates explicit unless further evidence changes
  them.
- Review hosted Linux/macOS/Windows CI outcomes; the checked-in workflow is
  configured, but no hosted run result is established here.
