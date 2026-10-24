# Deterministic corpus statistics v1

This document records the historical `surface-frequency-v1` evidence
contract. Those artifacts calculate a surface-token count over an explicit
immutable `ScopeSnapshot` and store:

- the normalized requested token and its count;
- the total population of surface-token positions;
- an `EXHAUSTIVE` `CoverageDescriptor` with the tokenization and completion
  criterion; and
- a `CORPUS_STATISTIC` evidence artifact tied to that exact scope.

The result is descriptive. It can support a later exact quantified finding only
when the finding links the persisted statistic and its coverage; it does not by
itself interpret a pattern, establish authorial intention, or make a claim
about documents outside the frozen scope.

The v1 calculation uses `corpus.TOKEN`, a Unicode-aware surface-token regular
expression. Its pattern and behavior remain unchanged so historical evidence
can still be recalculated. New surface-frequency and KWIC evidence use the
versioned Unicode-aware v2 contract described in
[`corpus-tokenization-v2.md`](corpus-tokenization-v2.md); v2 does not silently
reinterpret this historical evidence or other calculations still on v1.

## Derived NLP artifact statistics

`corpus artifact-frequency-evidence` (and its local MCP counterpart) records
the frequency of one exact `normalized`, `lemma`, `pos`, or
`dependency_relation` value from Parquet/Zstd artifacts. Its provenance records
the artifact IDs, their declared type/schema, the queried field/value and
DuckDB as the query engine. Normalized surface artifacts use a
`surface-token-v2-ucd-<version>` schema; their evidence records the shared
Unicode tokenizer, NFC-after-casefold normalization, and Python Unicode
database version. Parsed fields retain the selected local parser provenance.

It accepts an `EXHAUSTIVE` label only when the caller supplies exactly one
compatible artifact for every revision in the immutable scope. A partial build,
mixed artifact type, duplicate artifact, or out-of-scope artifact is rejected.
The population is therefore explicitly the non-null values of the requested
field in that complete versioned artifact set—not an unsupported claim about
all possible NLP parsers or revisions.

Earlier `surface-token-v1` artifacts and evidence keep their original contract.
Current artifact-backed surface-frequency operations require a rebuilt v2
artifact rather than treating v1 case-folded values as NFC-normalized data.

## Corpus comparison

Named-corpus frequency and keyness comparisons create a persistent
`comparison_run`. It names two distinct immutable scope snapshots, the
calculation version, tokenizer/configuration, ordered deterministic rows and a
SHA-256 result hash. This is deliberately not stored as an ordinary
single-scope Finding: a comparative statement has two populations, and both
must remain explicit. Subsequent changes to either named corpus create new
snapshots and a new comparison run rather than changing a historical result.
`report comparison` and the local MCP `get_comparison` operation verify that
result hash before rendering it, expose both token populations and preserve
methodological limitations in the output.

New comparison results use `surface-comparison-v2`. Frequency-rate difference
rows retain their prior calculation; keyness now computes signed G² from the
full 2×2 table (the selected token and its complement in each population).
Historical `surface-comparison-v1` rows remain stored and validated as recorded;
they are never silently recalculated with the v2 formula.
