# Controlled CDA review v1

`run_controlled_cda_review` is a controlled orchestration profile, not an
agent-selected chain of tools. It creates one immutable `AnalysisRun` over an
explicit `ScopeSnapshot` and runs these ordered stages:

1. deterministic textual descriptors record configured modality observations;
2. `builtin.evidence-extraction` scans the complete original sentence scope
   with its versioned candidate rules and creates one bounded,
   evidence-linked inventory finding when any candidate sentences are selected;
3. researcher-supplied lexical counterevidence queries, when supplied, are
   exhaustively matched against the original immutable scope for each matching
   phenomenon; otherwise the run result reports `NOT_RUN` with the reason;
4. selected versioned framework review profiles create evidence-bound,
   alternative-reading findings;
5. when selected, `critical-reviewer` runs after those profiles and critiques
   each preceding interpretation using its inherited Core-owned evidence.

The pipeline does not fabricate counterevidence queries. If no defensible query
is supplied, the omission is visible and no absence or counterevidence claim is
made. A completed literal search exhaustively enumerates only occurrences of
that literal query; it does not exhaustively establish semantic counterevidence.
Framework reviews
can cite only Core-owned textual evidence IDs from their seed finding; model
output cannot introduce pages, locations, sources, or evidence IDs.
`critical-reviewer` cannot run alone: it requires an existing framework
interpretation to review.

## Seed-extraction boundary

The v1 seed inventory is a bounded deterministic candidate pass, not a
comprehensive evidence extractor. It scans every sentence in the scope but
uses a deliberately incomplete English/Spanish cue inventory and attaches at
most 48 selected sentences. A cue hit is not a finding about the discourse
phenomenon; downstream framework review must inspect the exact passages and
may reject the candidate interpretation. An empty inventory means only that
the configured rules selected no sentences. In that case every configured
framework stage is persisted as `SKIPPED` with
`NO_DESCRIPTIVE_SEED_EVIDENCE`; no model call or absence claim is implied.

## Execution records

Every stage has a persisted `SkillRun`. A skill is `RUNNING` while its executor
is active and becomes `COMPLETED` only after its findings and evidence are
stored. If a controlled run fails, completed prior stages remain historical,
the active skill is marked `FAILED` with its error type, and the parent
`AnalysisRun` is `FAILED`. The run configuration and frozen scope are never
rewritten during this transition.

When no counterevidence query is supplied for a seed finding, a `SKIPPED`
counterevidence `SkillRun` is persisted with reason
`NO_RESEARCHER_PROVIDED_CONTRAST_QUERY`. It is not reported as a completed
search and remains visible after resuming or reopening the run.

A failed controlled run may be resumed through Core, CLI, or MCP. Re-entry is
allowed only for that same failed run and reads its saved profile configuration,
model preset name, framework versions, and immutable scope. The run also stores
prompt references, versions and SHA-256 digests for the static framework prompt
template (including shared output/safety instructions) and synthesis prompt; it
does not store prompt bodies or research text in this prompt snapshot. Resume
fails closed if a built-in prompt snapshot differs from the currently registered
prompt. Completed stages are identified by their persisted input-finding lineage
and configuration hash, then reused; failed stages receive a new `SkillRun`
attempt. The failed attempt remains in the provenance record rather than being
overwritten.

## Batch and routing

Batch controlled CDA creates one independent `AnalysisRun` per selected
DocumentRevision. It is deliberately sequential and never concatenates
documents, shares a scope, or reuses evidence across those runs. A saved Model
Routing Profile may assign different explicit presets to different framework
stages. The resolved framework-to-preset map is stored in each run
configuration before execution and is used again on resume.
