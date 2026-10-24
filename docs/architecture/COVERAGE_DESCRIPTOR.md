# CoverageDescriptor v0.1

Fields: `coverage_mode`, `scope_snapshot_id`, `population_definition`, optional
`population_count`, `selection_policy`, `inspection_count`,
`completion_criterion`, `termination_reason`, `termination_state`, and
`negative_result_status`.

`EXAMPLE_BASED`, `TARGETED`, and `BROAD` may state that retrieved examples were
found. Only `EXHAUSTIVE` deterministic enumeration over the declared population
may support exact counts, universal claims, or corpus-bounded absence when its
population and completion criterion are defined and its `termination_state` is
`COMPLETED`. `NO_MATCHES_UNDER_PLAN` records a retrieval result without making a
corpus-wide absence claim. `ABSENCE_IN_EXHAUSTIVE_POPULATION` is only valid for a
completed exhaustive descriptor and is required for an `ABSENCE` finding. A
zero-hit targeted search may say “No examples were found under this retrieval
plan,” not “No such examples exist in the corpus.”

Markdown and HTML reports include the descriptor's `scope_snapshot_id` so the
reported population can be traced to the exact immutable scope used by the
finding or retrieval.

## Persisted finding contract

A finding that declares an `INTERPRETIVE`, `ABSENCE`, `UNIVERSAL`, or
`QUANTIFIED` claim stores a `coverage_id`. The referenced descriptor must point
to the same immutable `ScopeSnapshot` as the finding's `AnalysisRun`.

- `INTERPRETIVE` requires limitations and may use non-exhaustive coverage.
- `ABSENCE` requires `EXHAUSTIVE` coverage with an explicit exhaustive
  zero-result status and a zero-valued corpus-statistic artifact referencing the
  same scope and coverage; `UNIVERSAL` requires `EXHAUSTIVE` coverage.
- `QUANTIFIED` additionally requires a defined population count and a persisted
  `CORPUS_STATISTIC` evidence artifact from that same scope.

This keeps “these passages support a reading” distinct from a claim about all
items in a corpus. The validator rechecks the relationship before reports or
MCP validation responses are produced.
