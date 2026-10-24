# Controlled CDA review v2

`run_controlled_cda_review` executes a persisted, ordered Core pipeline over
one immutable `ScopeSnapshot`:

1. deterministic textual descriptors;
2. bounded descriptive evidence extraction;
3. the historical v2 researcher-supplied literal counterevidence matcher;
4. independently versioned framework interpretation stages, each with its own
   bounded targeted retrieval against the original scope;
5. critical review of the preceding interpretations when configured;
6. one evidence-constrained synthesis over the validated critical-review
   findings (or validated framework findings when no critical reviewer is
   selected).

The synthesis executor validates every source finding, requires all sources to
belong to the current run, and is given only their persisted findings and
Core-owned evidence. Its output can cite only evidence IDs already attached to
those validated findings. It cannot create evidence, locations, pages, or
citations. The output finding records `SYNTHESIZES` lineage to each input
finding. Synthesis coverage is example-based, with low interpretation
confidence and explicit limitations; it is not a corpus-wide conclusion.

Framework retrieval is targeted lexical retrieval, not exhaustive semantic
search. It is scoped and audited separately from canonical evidence. No
absence, universal, or prevalence claim follows from empty or capped targeted
results. The bounded evidence-extraction inventory likewise remains a
candidate-selection heuristic, not an exhaustive detector.

The pipeline snapshots the preset and capabilities for every interpretive
stage and for synthesis before execution. Failed runs retain completed skill
runs and can resume against those snapshots; completed synthesis is reused
rather than repeated. Batch execution creates separate runs per document and
never concatenates their text or expands their scope.

This document is retained as the v2 method record. In v3, literal matches are
retrieval candidates, not `COUNTEREVIDENCE` links; see
`controlled-cda-review-v3.md`.
