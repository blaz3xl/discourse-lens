# Counterevidence lexical candidate retrieval v2

This deterministic skill scans the complete canonical text of every revision
in the Finding's original immutable `ScopeSnapshot` for a researcher-provided
literal query. It records exact matches as `RetrievalRun` / `RetrievalHit`
candidate locations. It does **not** attach them to the Finding as
`COUNTEREVIDENCE`, create evidence artifacts, or claim that a match refutes the
interpretation. A researcher or later review step must assess a candidate before
it can be linked to a Finding with that evidential role.

The skill scans canonical text directly rather than treating FTS as a
completeness oracle. Its `EXHAUSTIVE` coverage is limited to occurrences of the
exact case-insensitive literal query over the selected population; it is not an
exhaustive search for the broader concept of counterevidence. A zero-hit result
means only that the literal query was not found in that bounded population. It
never means no counterevidence exists.

Each hit points to a canonical `SourceSpan` inside the original scope. Chunks,
seed evidence, and prior supporting passages do not constrain this search.

To bound project-local retrieval storage, the executor stops as soon as a query
matches more than 10,000 literal occurrences. It creates no partial coverage
descriptor or retrieval-hit set. A direct retrieval request fails with a typed
limit error; the controlled CDA pipeline records the stage as `SKIPPED`, exposes
`LIMIT_REACHED` and a warning, then continues other configured stages. Narrow
the query and retry. Below the ceiling, all matches are persisted in bounded
batches while the API returns offset/limit pages. The ceiling is an operational
limit, not a sampling rule.
