# SPIKE-002 — Vector backend

## Result (2026-09-30 and 2026-10-01 revalidation)

The 10,000- and 100,000-vector harnesses compared deterministic random
768-dimensional `float32` vectors generated with seed `20260925`. The query was
the first stored vector. Both scales returned ten results, with ordered top-10
IDs matching after copy and reopen. These random-vector and adversarial-scope
checks measure index mechanics, not semantic retrieval quality.

The run used uv 0.5.9, CPython 3.12.8, NumPy 2.5.3, sqlite-vec 0.1.9, LanceDB
0.39.0, PyArrow 25.0.1, and FAISS CPU 1.15.1. Candidate packages were installed
from the local uv cache into a temporary isolated environment with offline mode.
The repository environment did not have NumPy or the candidate packages
installed; no project dependency or production code was changed. uv copied
cached package files because the cache and temporary environment were on
different filesystems.

Build time includes candidate-specific index creation and persistence. Query
time is one query immediately after the build, with no warm-up or repeated-sample
median. Copy and reopen operations used a second artifact in the same temporary
directory. Reopen time covers opening the copied index; reopen-query time covers
one additional top-10 query.

### Strict-scope correctness

The adversarial fixture put ten vectors in revision A and ten closer vectors in
revision B. Each engine first ran an unfiltered top-10 query and had to return
only B, then ran the A-scoped retrieval path and had to return only A:

| Candidate | A-scoped strategy | Unfiltered result | Scoped result |
| --- | --- | --- |
| sqlite-vec | Select A's per-revision index before searching | B only | A only; passed |
| FAISS `IndexFlatL2` | Select A's per-revision index before searching | B only | A only; passed |
| LanceDB | Mixed A/B table, `revision_id = 'A'`, `prefilter=True` | B only | A only; passed |

These checks validate physical-index selection and LanceDB's metadata prefilter,
not the production scope-validation path. FAISS and sqlite-vec used one physical
index per revision; neither run tested a metadata predicate in a mixed index.
FAISS has no metadata predicate, so strict scope depends on selecting only the
approved per-revision index before the query.

### 10k mechanics and copied persistence

| Candidate | Build | Single top-10 query | Persisted size | Copy | Reopen | Reopen query | Top-10 IDs match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sqlite-vec | 2.397 s | 41.933 ms | 32,284,672 B | 30.031 ms | 0.742 ms | 42.753 ms | Yes |
| FAISS `IndexFlatL2` | 0.033 s | 11.577 ms | 30,720,045 B | 16.003 ms | 28.753 ms | 9.709 ms | Yes |
| LanceDB | 0.154 s | 73.428 ms | 30,794,184 B | 17.300 ms | 3.167 ms | 58.285 ms | Yes |

All candidates returned ten rows before and after reopening the copied index, and
the ordered top-10 IDs matched. The updated benchmark records copy size, copy
time, reopen time, reopen-query time, and round-trip equality for each candidate;
it exits unsuccessfully if a candidate errors or the IDs change.

### 100k mechanics and copied persistence

Each candidate was run sequentially in its own process. Every run returned ten
hits before and after reopening the copied index, and the ordered top-10 IDs
matched. The JSON measurements are saved under
`experiments/spike_002_vector_backend/results_100k_2026-09-30/`.
That directory contains `benchmark_sqlite_vec.json`, `benchmark_lancedb.json`,
and `benchmark_faiss.json`. The stronger 2026-10-01 fixtures, which also verify
that each engine returns B before the scope restriction and only A afterward,
are in `experiments/spike_002_vector_backend/results_lifecycle_2026-10-01/`.

| Candidate | Build | Single top-10 query | Persisted / copied size | Copy | Reopen | Reopen query | IDs match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sqlite-vec | 52.854 s | 436.870 ms | 316,776,448 B | 720.707 ms | 1.315 ms | 774.064 ms | Yes |
| FAISS `IndexFlatL2` | 1.411 s | 72.722 ms | 307,200,045 B | 1.049 s | 1.599 s | 81.124 ms | Yes |
| LanceDB | 3.169 s | 1.290 s | 307,992,218 B | 278.678 ms | 8.968 ms | 2.919 s | Yes |

For each candidate, the copied artifact size equaled the source size.

### Append, delete, and rebuild (2026-10-01)

Each run appended 10% of the base row count, queried the new vector, deleted the
appended rows, and checked that the row count returned to the base count and no
deleted IDs appeared in search. It then rebuilt a fresh artifact from the live
base rows, copied and reopened that artifact, and confirmed the ordered top-10
IDs and row count. All checks passed. Times are single runs without warm-up or
repeated-sample medians. The 10k candidates ran sequentially; the 100k candidates
ran one at a time in separate processes.

| Candidate | Base rows | Append | Delete | Fresh rebuild | Rebuilt / copied bytes | Copy | Reopen | Reopen query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sqlite-vec | 10k (+1k) | 0.619 s | 1.109 s | 3.600 s | 32,129,024 | 14.389 ms | 0.341 ms | 27.066 ms |
| FAISS `IndexIDMap2(IndexFlatL2)` | 10k (+1k) | 45.678 ms | 2.116 ms | 40.299 ms | 30,800,090 | 24.116 ms | 45.177 ms | 22.259 ms |
| LanceDB | 10k (+1k) | 26.296 ms | 8.641 ms | 0.236 s | 30,803,658 | 51.820 ms | 5.320 ms | 75.430 ms |
| sqlite-vec | 100k (+10k) | 3.157 s | 6.067 s | 38.459 s | 314,757,120 | 207.441 ms | 0.560 ms | 417.422 ms |
| FAISS `IndexIDMap2(IndexFlatL2)` | 100k (+10k) | 0.853 s | 22.897 ms | 0.490 s | 308,000,090 | 878.050 ms | 1.538 s | 76.861 ms |
| LanceDB | 100k (+10k) | 141.275 ms | 7.874 ms | 3.160 s | 308,001,503 | 274.344 ms | 4.797 ms | 1.189 s |

The FAISS lifecycle check used `IndexIDMap2(IndexFlatL2)` so appended IDs could
be removed without relying on shifting positional labels. Its copy/reopen path
uses `write_index`/`read_index`. LanceDB rebuild here means creating a fresh
table from live rows; native fragment compaction was not exercised. The base
LanceDB benchmark also created no ANN vector index, so its query timings measure
the table-search path used by this harness.

Raw lifecycle and scope JSON is under
`experiments/spike_002_vector_backend/results_lifecycle_2026-10-01/`.

### Per-revision sqlite-vec insertion batch comparison (2026-10-01)

An apples-to-apples local timing compared the existing row-at-a-time inserts with
bounded `executemany` batches for 10,000 chunks of 768 dimensions. The synthetic
fixture reused one deterministic 768-dimensional vector across rows. Both paths
ran twice in alternating order against the same temporary project; timing covered
the complete index build, including staged validation, hashing, publication, and
registry insertion.

| Insert path | Run 1 | Run 2 | Median |
| --- | ---: | ---: | ---: |
| Row at a time | 39.721 s | 49.959 s | 44.840 s |
| Bounded `executemany` | 53.149 s | 42.619 s | 47.884 s |

The batch path was about 6.8% slower by median, with high run-to-run variation.
Two runs per path are a small sample, so this is limited evidence; it does not
justify changing the current insertion loop. These measurements do not establish
performance at 100k or 1M rows and should not be extrapolated to those sizes.

The temporary environment used for the 2026-10-01 work left 3.7 GiB free on
`/tmp` and 4.4 GiB MemAvailable after the runs; during the 100k sqlite-vec run,
the lowest sampled values were 3.4 GiB free on `/tmp` and 3.2 GiB MemAvailable.
No 1M run was attempted: its raw 768-dimensional `float32` matrix alone is about
2.86 GiB, before candidate structures, copied artifacts, and the isolated
environment are counted. The 1M persistence/copy workload does not fit safely
within the available tmpfs budget.

### Reproduction

Run from the repository root. The commands use cached packages offline and
write result JSON without changing project dependencies:

```sh
SPIKE_TMP="$(mktemp -d /tmp/discourselens-spike002-100k.XXXXXX)"
SPIKE_OUTPUT="$(mktemp -d experiments/spike_002_vector_backend/results_100k_XXXXXX)"
uv venv --offline --python 3.12.8 --no-python-downloads "$SPIKE_TMP/venv"
uv pip install --offline --python "$SPIKE_TMP/venv/bin/python" \
  'numpy==2.5.3' 'sqlite-vec==0.1.9' 'lancedb==0.39.0' \
  'pyarrow==25.0.1' 'faiss-cpu==1.15.1'

"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/scope_test.py \
  --candidate sqlite-vec \
  --output "$SPIKE_OUTPUT/scope_sqlite_vec.json"
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/benchmark.py \
  --chunks 100000 --candidate sqlite-vec \
  --output "$SPIKE_OUTPUT/benchmark_sqlite_vec.json"
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/lifecycle.py \
  --chunks 10000 --candidate sqlite-vec \
  --output "$SPIKE_OUTPUT/lifecycle_10000_sqlite_vec.json"
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/lifecycle.py \
  --chunks 100000 --candidate sqlite-vec \
  --output "$SPIKE_OUTPUT/lifecycle_100000_sqlite_vec.json"

"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/scope_test.py \
  --candidate lancedb \
  --output "$SPIKE_OUTPUT/scope_lancedb.json"
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/benchmark.py \
  --chunks 100000 --candidate lancedb \
  --output "$SPIKE_OUTPUT/benchmark_lancedb.json"
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/lifecycle.py \
  --chunks 10000 --candidate lancedb \
  --output "$SPIKE_OUTPUT/lifecycle_10000_lancedb.json"
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/lifecycle.py \
  --chunks 100000 --candidate lancedb \
  --output "$SPIKE_OUTPUT/lifecycle_100000_lancedb.json"

"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/scope_test.py \
  --candidate faiss \
  --output "$SPIKE_OUTPUT/scope_faiss.json"
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/benchmark.py \
  --chunks 100000 --candidate faiss \
  --output "$SPIKE_OUTPUT/benchmark_faiss.json"
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/lifecycle.py \
  --chunks 10000 --candidate faiss \
  --output "$SPIKE_OUTPUT/lifecycle_10000_faiss.json"
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/lifecycle.py \
  --chunks 100000 --candidate faiss \
  --output "$SPIKE_OUTPUT/lifecycle_100000_faiss.json"
```

### Multi-revision fan-out (2026-10-01)

`experiments/spike_002_vector_backend/multi_revision.py` models the retrieval
shape of `search_vectors`: the caller preselects revision indexes, each selected
index is opened and searched for its own top 10, and the candidate rows are
globally sorted and truncated. The deterministic query is independent of the
stored random vectors; revision `r0001` contains an exact copy at chunk 0. A
one-revision scope containing only `r0000` must return only `r0000`, while the
all-revision scope must return `r0001-0000` first. All runs passed these checks.
The two engines returned identical ordered top-10 IDs in each measured dataset.

The 100-revision run contains 1,000 768-dimensional vectors per revision. It
records the legacy/current-at-spike-start L2 behavior: the production `vec0`
declaration omitted `distance_metric=cosine`, so sqlite-vec 0.1.9 used its L2
default. FAISS used `IndexFlatL2`; its squared distances were square-rooted for
score parity. The `cosine` distances described by production comments and
provenance were therefore not the metric returned by legacy v1-v3 indexes. The
production decision is to request cosine explicitly for new indexes while
leaving legacy v1-v3 indexes on L2; the 100k L2 run does not measure that new
behavior.

| Candidate | Build | Persisted bytes | Selected scope: open / vector query / aggregation → total | 100-revision scope: open / vector query / aggregation → total |
| --- | ---: | ---: | ---: | ---: |
| sqlite-vec, L2 | 221.444 s | 328,908,800 | 0.964 / 7.188 / 0.022 → 8.560 ms | 82.681 / 665.017 / 0.547 → 785.892 ms |
| FAISS `IndexFlatL2` | 0.755 s | 308,009,000 | 2.236 / 0.889 / 0.018 → 3.374 ms | 275.558 / 98.042 / 0.511 → 389.827 ms |

The all-revision search opened 100 indexes, gathered 1,000 candidates, and
returned the same global top 10 from both engines. The selected-scope search
opened only `r0000`; it did not search the exact match in excluded revision
`r0001`. Scope-list construction was a trivial in-memory lookup and is excluded
from the timing comparison. Index build includes the per-revision vector data
and, for sqlite-vec, the chunks table join used by `search_vectors`; it excludes
project registry validation, atomic artifact publication, and embedding work.
Artifact hashing was measured separately and is not included in the search
times. The single-run hash timings were 2.620 s for sqlite-vec and 3.054 s for
FAISS, so they are reported as observations rather than a stable comparison.
The FAISS files store per-revision vectors and integer local IDs, but omit a
durable local-ID-to-chunk-ID/offset map; this fixed fixture reconstructs chunk
IDs from row order. Consequently FAISS build and size omit application metadata
that a production adapter would need, while sqlite-vec's files include the
`chunks` table. These are backend mechanics totals, not equivalent complete
application artifacts.

A smaller cosine parity run used 10 revisions × 1,000 × 768 vectors. sqlite-vec
declared `distance_metric=cosine`; FAISS used normalized vectors with
`IndexFlatIP` and returned `1 - similarity`. It passed the same strict-scope
checks, and ordered top-10 IDs matched for both scopes.

| Candidate, cosine | Build | Persisted bytes | Selected scope: open / vector query / aggregation → total | 10-revision scope: open / vector query / aggregation → total |
| --- | ---: | ---: | ---: | ---: |
| sqlite-vec | 26.623 s | 32,890,880 | 0.920 / 6.164 / 0.014 → 7.398 ms | 7.050 / 63.036 / 0.040 → 73.206 ms |
| FAISS `IndexFlatIP` | 0.095 s | 30,800,900 | 2.018 / 0.576 / 0.017 → 2.810 ms | 27.202 / 6.266 / 0.037 → 34.681 ms |

Reproduce the default legacy L2 workload or the bounded cosine check with the
isolated environment from the earlier reproduction steps:

```sh
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/multi_revision.py \
  --candidate sqlite-vec \
  --output experiments/spike_002_vector_backend/results_multirevision_2026-10-01/multi_revision_100x1000_sqlite_vec.json
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/multi_revision.py \
  --candidate faiss \
  --output experiments/spike_002_vector_backend/results_multirevision_2026-10-01/multi_revision_100x1000_faiss.json

"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/multi_revision.py \
  --candidate sqlite-vec --metric cosine --revisions 10 \
  --chunks-per-revision 1000 --dimensions 768 \
  --output experiments/spike_002_vector_backend/results_multirevision_2026-10-01/multi_revision_10x1000_sqlite_vec_cosine.json
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/multi_revision.py \
  --candidate faiss --metric cosine --revisions 10 \
  --chunks-per-revision 1000 --dimensions 768 \
  --output experiments/spike_002_vector_backend/results_multirevision_2026-10-01/multi_revision_10x1000_faiss_cosine.json
```

These timings are single runs without warm-up or repeated-sample medians. The
cosine workload is 10k vectors; its build, size, and query measurements must not
be extrapolated to 100k or 1M. The result files are
`experiments/spike_002_vector_backend/results_multirevision_2026-10-01/multi_revision_100x1000_sqlite_vec.json`,
`experiments/spike_002_vector_backend/results_multirevision_2026-10-01/multi_revision_100x1000_faiss.json`,
`experiments/spike_002_vector_backend/results_multirevision_2026-10-01/multi_revision_10x1000_sqlite_vec_cosine.json`,
and `experiments/spike_002_vector_backend/results_multirevision_2026-10-01/multi_revision_10x1000_faiss_cosine.json`.
The harness's earlier Python
`ru_maxrss` values were identical even for tiny fixtures in this WSL environment
and are not treated as valid RSS measurements. The harness now reads Linux
`VmHWM` from `/proc/self/status`; quick smoke runs validate that capture, but the
100k and 10k measurement processes were not repeated. One `ps` sample during the
sqlite-vec build read 344,704 KiB; there is no comparable per-candidate peak-RSS
measurement, so RSS is unavailable for this comparison.

### 100-revision explicit-cosine fan-out (2026-10-02)

Both candidates ran the same deterministic 100-revision × 1,000-vector fixture
(100,000 vectors total, 768 dimensions, `float32`, seed `20261001`, top-k 10).
sqlite-vec declared `distance_metric=cosine`; FAISS normalized stored and query
vectors and used `IndexFlatIP`. Each process used Python 3.12.8, NumPy 2.5.3,
sqlite-vec 0.1.9 or FAISS CPU 1.15.1, and one FAISS OpenMP thread. The source
matrix occupied 307,203,072 bytes. The query was independently seeded, and
`r0001-0000` was set to an exact copy of it as an excluded-revision adversary.

| Candidate | Build | Persisted bytes | Selected `r0000`: open / query / aggregate → total | All 100: open / query / aggregate → total |
| --- | ---: | ---: | ---: | ---: |
| sqlite-vec, cosine | 114.634 s | 328,908,800 | 0.341 / 2.731 / 0.008 → 3.207 ms | 27.897 / 286.654 / 0.163 → 327.217 ms |
| FAISS `IndexFlatIP`, cosine | 0.639 s | 308,009,000 | 1.600 / 0.554 / 0.017 → 2.420 ms | 141.393 / 40.933 / 0.506 → 190.465 ms |

Both deterministic scope checks passed for both candidates: the selected
one-revision request opened only `r0000` and returned ten `r0000` hits; the
all-revision request opened all 100 indexes, gathered 1,000 candidates, and
returned ten hits with `r0001-0000` first. Ordered top-10 IDs matched between
the two candidates for each scope. Exact-match cosine distances differ slightly
from floating-point rounding (`1.11e-16` for sqlite-vec and `5.96e-8` for
FAISS). All 100 artifact hashes were computed separately from build and query
timings: 0.951 s for sqlite-vec and 1.803 s for FAISS. Reported peak process RSS
was 359,022,592 bytes and 393,011,200 bytes respectively; this includes the
source matrix and candidate libraries, so it is not index-only memory.

The harness is still a backend-mechanics fixture, not a production lifecycle
test:

| Concern | What this harness covers |
| --- | --- |
| Durable ID/offset mapping | sqlite-vec stores chunk IDs and offsets in its `chunks` table. FAISS stores integer local labels, but has no durable local-ID-to-chunk-ID/offset map; this fixture reconstructs chunk IDs from row order. |
| Project registry validation | Not exercised. The harness passes an already selected list of revision indexes directly to the search functions. |
| Atomic artifact publication | Not exercised. Index files are built in a temporary directory and queried there; no staged publish, atomic swap, or publication failure/recovery path runs. |

Reproduce both explicit-cosine measurements with the offline environment from
the earlier reproduction steps:

```sh
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/multi_revision.py \
  --candidate sqlite-vec --metric cosine --revisions 100 \
  --chunks-per-revision 1000 --dimensions 768 \
  --output experiments/spike_002_vector_backend/results_explicit_cosine_2026-10-02/multi_revision_100x1000_sqlite_vec_cosine.json
"$SPIKE_TMP/venv/bin/python" experiments/spike_002_vector_backend/multi_revision.py \
  --candidate faiss --metric cosine --revisions 100 \
  --chunks-per-revision 1000 --dimensions 768 \
  --output experiments/spike_002_vector_backend/results_explicit_cosine_2026-10-02/multi_revision_100x1000_faiss_cosine.json
```

These are one run per candidate, without warm-up or repeated-sample medians.
FAISS built about 179× faster, used about 6% less persisted space, and had
about 1.7× faster all-scope search; opening all 100 FAISS indexes took longer
than opening the sqlite-vec files. Selected-scope totals differed by only
0.787 ms. The result supports FAISS as the mechanics leader on this fixture,
but the ID-map, registry, and publication gaps mean it does not compare complete
application artifacts or establish production behavior. Neither candidate was
evaluated for semantic retrieval quality; these ratios are not stable capacity
estimates.

## Assessment

For mechanics-only evaluation at 10k and 100k, FAISS led the measured build,
query, delete, and rebuild timings and had the smallest persisted file. It is the
mechanics leader for a next-stage prototype if separate per-revision indexes and
an external metadata lifecycle are acceptable. sqlite-vec is the SQLite
integrated alternative; it passed scope, mutation, and copy/reopen checks but
was slower in the measured builds and lifecycle operations. LanceDB passed its
metadata prefilter, mutation, and copied-persistence checks; its 100k query and
reopen-query times were slower in this no-ANN-index, single-query harness. The
multi-revision measurements strengthen FAISS's mechanics case: at 100k L2 its
build was about 294× faster and all-scope search about 2× faster, with about 6%
less persisted space. At 10k explicit cosine, FAISS all-scope search was about
2.1× faster and persisted space about 6% smaller; at 100k explicit cosine, its
build was about 179× faster and all-scope search about 1.7× faster, again with
about 6% less persisted space. These are single-run observations, not a
production capacity forecast.

The evidence supports FAISS as the mechanics leader, but it does not by itself
establish a production default. The multi-revision harness selects physical
indexes from an already approved list; it does not exercise project registry
validation, immutable-scope validation, a durable FAISS chunk-ID/offset map,
atomic artifact publication, or failure and recovery behavior through the actual
`Project` API. The explicit-cosine runs validate backend metric mechanics, but do
not exercise the new production path or legacy-index migration behavior. A
default decision still needs end-to-end scope and integrity checks through the
project API, operational lifecycle coverage across many per-revision artifacts,
and retrieval-quality evaluation with representative real embeddings. Native
LanceDB compaction also remains unmeasured if LanceDB remains under consideration.
Vectors remain derived and rebuildable; they never hold canonical text,
SourceSpan, EvidenceArtifact, citations, or scope identity.
