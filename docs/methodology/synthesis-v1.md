# Evidence-constrained synthesis v1

Synthesis is a final, optional-to-call but pipeline-controlled LLM-assisted
stage over persisted, validated findings from one `AnalysisRun`. It does not
retrieve new corpus passages, create new `EvidenceArtifact` records, or accept
model-generated citations. Its allowed citation set is the union of evidence
IDs already attached to its source findings.

The executor verifies each source finding before constructing the prompt and
requires every source to belong to the current run. The synthesized finding
records explicit `SYNTHESIZES` lineage to each source finding, references only
Core-owned evidence selected by the model, and retains an alternative reading,
low interpretation confidence, limited evidence sufficiency, and
example-based coverage.

Synthesis must not introduce a new phenomenon, causal explanation, author
intention, psychological state, political affiliation, universal statement,
or corpus-wide absence claim. It is a concise cross-finding summary, not a new
analysis of the whole corpus. If the validated findings do not support a shared
interpretation, the result should say so and preserve their differences.
