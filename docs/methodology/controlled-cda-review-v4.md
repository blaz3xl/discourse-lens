# Controlled CDA review v4

Version 4 preserves the immutable-scope, staged review and evidence-bound
synthesis described in v3. When the critical reviewer is selected, it now
also runs a deterministic lexical candidate search derived from the prior
framework finding's alternative interpretation. This does not depend on a
researcher-supplied contrast phrase. It is logged independently as a targeted
`RetrievalRun` over the original `ScopeSnapshot`, excluding seed paragraphs.

Candidates are not automatically evidence against a finding. The model sees
their candidate status and may classify a cited candidate as
`COUNTEREVIDENCE` only after contextual assessment; Core validates the
candidate identity and persists the role only on the critical review finding.
An empty or capped search remains non-exhaustive and cannot support an absence
claim. The separate literal exact-phrase search remains optional and
researcher-directed.

If a researcher-supplied literal counterevidence query exceeds the local
10,000-match persistence ceiling, the pipeline records that stage as `SKIPPED`
with `LIMIT_REACHED`, includes a warning, and continues. It creates no partial
retrieval evidence or claim of completion; the researcher can refine the query
and rerun the analysis.

All previous guarantees remain: document/corpus scope is never expanded,
findings and citations remain Core-owned, and synthesis can only use validated
findings and their existing evidence identifiers.
