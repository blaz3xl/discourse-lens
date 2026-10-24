# Framework reviews v4

Version 4 adds a deterministic, targeted lexical retrieval pass for the
critical reviewer. Its search terms are derived from the alternative
interpretation recorded by the preceding framework review and are searched
only inside that review's original immutable `ScopeSnapshot`. Seed evidence
paragraphs are excluded so the retrieval does not merely rediscover the
passage already supporting the reviewed finding.

The pass is recorded as
`retrieval/counterevidence-alternative-lexical-v1`, with `TARGETED` coverage,
query terms, inspected/returned counts, and ordinary `RetrievalHit` provenance.
It runs even when the researcher did not provide a literal contrast query.
It is a candidate-generation heuristic: lexical overlap is neither a
contradiction nor confirmation, and a zero-hit result is not evidence of
absence. Candidate passages are exposed as unassessed. The critical reviewer's
explicit classification may attach one as `COUNTEREVIDENCE` to its own review
finding after Core validates the candidate identity and declared role; those
structural checks do not validate the classification's semantic correctness.

The existing researcher-provided exact-phrase search remains a separate
optional retrieval workflow. These retrieval methods never widen the
`ScopeSnapshot`, never alter the original finding, and cannot support
corpus-wide absence, universal, or prevalence claims.

Framework model outputs also separate excerpts from descriptions. Core stores
`Finding.observation` using only the validated verbatim excerpts and their
evidence IDs. A model-supplied description is persisted separately as an
`UNVERIFIED` observation candidate linked to that cited textual evidence. Review
prompts and reports preserve that status; the candidate description is not
promoted into the observation field.
