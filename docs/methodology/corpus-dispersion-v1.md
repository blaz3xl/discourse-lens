# Corpus document dispersion v1

`corpus dispersion` computes a deterministic surface-token distribution over an
explicit immutable set of document revisions. For the normalized query token
it reports:

- occurrences in each revision;
- token population in each revision;
- occurrences per 1,000 surface tokens in each revision;
- document frequency (number of scoped revisions with at least one occurrence);
- document range, defined here as document frequency divided by scoped revision
  count;
- total occurrences and total token population.

The implementation uses the versioned `corpus.TOKEN` surface tokenizer and
case-folded matching. It does not perform lemmatization, significance testing,
or infer a discourse function. Zero-occurrence documents remain in the
population. `dispersion-evidence` persists the exhaustive calculation, coverage
descriptor, scope, calculation version, and per-document values as
`CORPUS_STATISTIC` evidence. The reported range is a descriptive proportion,
not a universal statistical dispersion score.

## Derived artifact integrity

New Parquet token artifacts and sqlite-vec indexes store a SHA-256 hash of the
written file bytes in the project artifact registry. Reads verify the resolved
path remains inside the project and that the bytes still match before handing
the artifact to DuckDB or sqlite-vec. A mismatch or path escape fails closed
with an instruction to rebuild the derived artifact. Projects created before
the file-hash migration retain their historical registry rows and remain
readable, but those legacy artifacts have no trustworthy byte-integrity value;
rebuild them to receive the current verification guarantee. The prior
`content_hash` field continues to carry its historical derivation fingerprint
and is not treated as a byte hash.
