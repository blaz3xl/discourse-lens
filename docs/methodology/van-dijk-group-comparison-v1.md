# Van Dijk group comparison preparation v1

## Purpose

`builtin.van-dijk-group-comparison` prepares a descriptive, evidence-linked
comparison from existing local qualitative annotations. It is deterministic:
the researcher supplies exactly two distinct group labels, one shared
topic/event, a comparability basis, one codebook ID, and the annotation IDs to
include in each group. The executor does not retrieve annotations, choose
examples, or interpret their labels.

## Scope and evidence

Every selected annotation must already belong to the declared codebook and
refer to a `SourceSpan` in the analysis run's original immutable
`ScopeSnapshot`. An annotation cannot be selected in both groups. Each
annotation remains linked to its canonical text span, and its evidence record
stores the annotation ID, codebook/code IDs, researcher-supplied group label,
topic/event, comparability basis, and original scope ID.

The persisted finding and returned artifact describe the codebook labels,
selected annotation records, per-code counts, exact source citations, and the
researcher's stated comparison basis. Counts are counts of only the explicitly
selected annotation records. They are not corpus token frequencies, corpus
prevalence estimates, or a comparison of all documents in a corpus.

## Missing coding and claim limits

If a group has no selected annotations, its coding status is `UNKNOWN` and its
code distribution is left empty. This does not mean the group lacks a code or
that a feature is absent from the corpus. A zero per-code count in a non-empty
selected list describes only that list; it does not establish corpus absence.
The finding uses `EXAMPLE_BASED` claim semantics only when at least one selected
text span is attached. If neither group contains a selected annotation, the
executor rejects the request before creating a skill run or finding; it cannot
prepare an evidence-linked comparison from two empty selections.

This preparation does not infer political identity, ideology, motivation,
author intention, universal prevalence, or absence. It produces no ideology,
asymmetry, or other interpretive score. A researcher may use the cited
annotation examples in later analysis, while keeping interpretation and
counterexamples separate from this descriptive record.

## Relation to qualitative coding

Codebook names, labels, definitions, and annotations are human-controlled local
records. This skill reuses those records as supplied and does not treat a code
label as an automatic discourse or ideology classification. See also
[Qualitative coding v1](qualitative-coding-v1.md) and
[Van Dijk-oriented ideology-pattern review v1](van-dijk-ideology-review-v1.md).
