# SPIKE-001 — NLP storage

## Result

The 10M-token scale check is complete for the selected Parquet/Zstd + DuckDB
path. A corrected rerun now uses the same deterministic token rows and real
character offsets as the SQLite fixture. The earlier 37.8 MB Parquet result
used synthetic offsets and is retained below as historical, non-comparable data.

At 1,000 tokens SQLite token rows occupied 98,304 bytes; sentence blobs used
32,768 bytes, but a lemma scan took 50.29 ms versus 0.10 ms for indexed rows.
At 1M tokens the partial SQLite-row artifact occupied 92,127,232 bytes and the
blob artifact 22 MiB. The blob scan became disproportionate before the full 1M
benchmark completed because each sentence must be decompressed and decoded.

The vectorized PyArrow/Zstd writer completed 1M tokens in 4.1 s and produced a
3.6 MiB Parquet artifact. DuckDB warm medians were 6–17 ms for the measured
frequency, filter, KWIC-candidate and aggregation queries; peak RSS was about
237 MiB.

## Historical 10M Parquet result (synthetic offsets)

An earlier version of the Arrow fixture used `start = i * 8` and a `stop`
column. The SQLite fixture used cumulative character offsets. That run produced
a 37,810,151-byte Parquet file in 16.5 s, with peak RSS 258,016 KiB and 10M rows
verified by DuckDB. Its query medians were 15.4 ms for lemma frequency, 27.4 ms
for document frequency, 3.7 ms for KWIC candidates, 15.2 ms for POS filtering,
15.0 ms for dependency filtering, and 29.1 ms for per-document aggregation.
These remain historical measurements; the different offset values make the
storage size and timings unsuitable for comparison with the SQLite fixtures.

An earlier 10M SQLite comparison measured 958,226,432 bytes for token rows and
228,134,912 bytes for sentence blobs. Its three-run SQLite-row medians were
276.0 ms for lemma frequency, 3,139.8 ms for document frequency, 0.27 ms for
KWIC, 3,493.6 ms for POS filtering, 4,114.5 ms for dependency filtering, and
2,677.0 ms for per-document aggregation. That combined run reached its 360 s
limit during three repeated blob scans. The current matched rerun below
supersedes those SQLite query timings. Its historical Parquet comparisons are
not comparable because of the synthetic offsets described above.

## Matched 10M fixture rerun

The corrected Arrow fixture uses cumulative character offsets and was checked
against the SQLite fixture: the first 1,000 complete rows, including `start`
and `end`, matched exactly. The measured environment was Python 3.12.8,
DuckDB 1.5.5, PyArrow 25.0.1, Linux x86_64 on an Intel Core i7-8550U with 8
reported CPUs. At the start, the host reported 5.4 GiB available RAM and `/tmp`
had 5.2 GiB free on a 5.9 GiB tmpfs. The runs used fresh spike-specific
directories under `/tmp`; no new swap use was observed.

Measured commands and retained artifacts:

```bash
experiments/.venv/bin/python experiments/spike_001_nlp_storage/vectorized_parquet.py \
  --tokens 10000000 --out /tmp/discourselens-spike-001-10m-vectorized-70Jaau/tokens.parquet

/usr/bin/time -v timeout --signal=TERM --kill-after=10s 240s \
  experiments/.venv/bin/python experiments/spike_001_nlp_storage/benchmark.py \
  --tokens 10000000 --architectures sqlite_rows \
  --out /tmp/discourselens-spike-001-10m-comparison-rerun-bNLBTv

/usr/bin/time -v timeout --signal=TERM --kill-after=10s 240s \
  experiments/.venv/bin/python experiments/spike_001_nlp_storage/benchmark.py \
  --tokens 10000000 --architectures sqlite_blobs_vectorized \
  --out /tmp/discourselens-spike-001-10m-blob-vectorized-pIhvLr
```

The Arrow path completed with 10,000,000 rows and a matching DuckDB read-back:

| Measurement | Result |
| --- | ---: |
| Vectorized write time | 21,098.3 ms |
| Parquet/Zstd artifact | 62,897,477 bytes (60.0 MiB) |
| Peak RSS | 186,848 KiB (182.5 MiB) |
| Total process wall time | 23.37 s |

The first execution after writing and the median of the next three executions
were:

| Query | First execution | Warm median |
| --- | ---: | ---: |
| Lemma frequency | 64.0 ms | 52.6 ms |
| Document frequency | 108.3 ms | 86.6 ms |
| POS filter | 58.8 ms | 52.3 ms |
| KWIC candidates | 7.2 ms | 6.8 ms |
| Per-document aggregation | 94.0 ms | 94.4 ms |
| Dependency relation filter | 61.8 ms | 52.5 ms |
| Lemma + POS filter | 104.1 ms | 90.4 ms |
| Document + lemma filter | 7.9 ms | 7.5 ms |

The SQLite token-row run completed under a 240 s cap. Its artifact was opened
read-only afterward; the row count was 10,000,000 and `PRAGMA quick_check`
returned `ok`.

| Measurement | Result |
| --- | ---: |
| Write plus all six three-repeat query measurements | 155,027.85 ms (155.17 s wall) |
| SQLite artifact | 958,226,432 bytes (913.7 MiB) |
| Peak RSS | 25,116 KiB (24.5 MiB) |

The same six query definitions were measured with three executions each for
SQLite and three warm executions for Parquet/DuckDB. The runs were separate and
page-cache state was not controlled.

| Query | SQLite token rows | Parquet/Zstd + DuckDB |
| --- | ---: | ---: |
| Lemma frequency | 186.14 ms | 52.6 ms |
| Document frequency | 1,822.7 ms | 86.6 ms |
| KWIC candidates | 0.13 ms | 6.8 ms |
| POS filter | 3,437.01 ms | 52.3 ms |
| Dependency relation filter | 4,193.13 ms | 52.5 ms |
| Per-document aggregation | 2,175.04 ms | 94.4 ms |

For the blob rerun, the owned harness now builds sentence payloads from bounded
100,000-token Arrow batches and inserts each batch with `executemany`. The 10M
write completed in 137,853.92 ms and produced 500,000 sentence rows occupying
228,134,912 bytes (217.6 MiB). `PRAGMA quick_check` returned `ok`; first,
middle, and last payload samples each decoded to 20 tokens. Peak RSS was
236,520 KiB (231.0 MiB). A single full lemma-frequency scan did not finish
before the command's 240 s limit, which left about 102 s after the measured
write and validation. No full 10M blob-query time or extrapolation is claimed.

The fresh matched artifacts show Parquet/Zstd at about 15.2 times smaller than
SQLite token rows and 3.6 times smaller than sentence blobs. Parquet/DuckDB was
faster on the measured broad scans and aggregations; indexed SQLite remained
faster for the limited KWIC query. These are synthetic workload measurements,
not semantic-retrieval or linguistic-quality evidence. The separate runs are
not cache-normalized, and they do not establish production throughput. No 100M
run was attempted.

## Recommendation

Keep SQLite/FTS5 canonical and lexical. Reject compressed sentence blobs as the
general NLP query store: their storage footprint is smaller than token rows,
but the 10M full lemma scan did not finish within the bounded run. Select
Parquet/Zstd as derived NLP storage and DuckDB as the corpus analytics engine.
Keep DuckDB in the `corpus` extra, not the minimal Core install.
