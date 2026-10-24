# Controlled CDA review v3

Version 3 preserves the ordered, immutable-scope workflow and evidence-bound
synthesis described in v2, with one methodological correction: literal
counterevidence searches produce scoped retrieval candidates, not
`FindingEvidence` links. A textual match cannot be assumed to refute a claim.
Candidates remain inspectable through the retrieval run and canonical spans;
they receive a `COUNTEREVIDENCE` role only after a later review explicitly
validates that relation.

The `EXHAUSTIVE` label applies only to enumeration of the exact, case-insensitive
literal query over all canonical text in the original `ScopeSnapshot`. It does
not mean that the broader phenomenon has been exhaustively searched. With no
researcher-supplied contrast query, the stage remains `NOT_RUN`; the pipeline
does not invent a contrary query or infer absence.

When candidates exist, the critical reviewer receives them marked as
unassessed. It may return explicit `counterevidence_ids` only for cited
candidates it concludes challenge the reviewed interpretation. The Core checks
that each ID is one of the recorded retrieval hits; a candidate cited only as
context is not assigned the `COUNTEREVIDENCE` role.

Each framework still performs its own bounded targeted retrieval against the
original scope. Synthesis uses only validated findings and their existing
evidence identifiers, preserves explicit lineage, and cannot mint citations.
