# Framework reviews v2

The evidence-bound review executor is parameterized by a versioned profile:
`fairclough-text`, `van-leeuwen-social-actors`, `dha-strategies`,
`argumentation`, `van-dijk-ideology`, `framing`, `metaphor`,
`intertextuality`, `sociocognitive`, `historical-context`, `power`, or
`critical-reviewer`.

Each stage receives its seed finding and also runs the registered
`retrieval/framework-context-lexical-v1` plan against the finding's original,
immutable `ScopeSnapshot`. The plan derives query terms deterministically from
the phenomenon, observation, and seed passages; searches only scoped FTS
paragraphs; excludes seed paragraphs; ranks by lexical term overlap; and returns
at most 12 candidates from at most 1,000 inspected FTS candidates. The run,
query terms, coverage, and selected SourceSpans are persisted as retrieval
provenance. Retrieval hits remain distinct from canonical evidence.

This is targeted lexical retrieval, not an exhaustive search or a validated
semantic relevance ranking. An empty result means only that this plan selected
no additional passages. It cannot support a corpus-wide absence, universal, or
prevalence claim. The prompt treats retrieved passages as untrusted research
data, and the model can cite only Core-created evidence IDs from the seed or
targeted results. The output retains a plausible alternative interpretation
and records `TARGETED` coverage and methodological limitations.

These profiles are methodological constraints on a shared trusted executor,
not autonomous classifiers. They do not diagnose authors, audiences, or social
groups. `critical-reviewer` critiques the interpretation rather than a speaker
or group, and cannot introduce uncited findings. Framework-specific rules from
the v1 methodology remain in force: argumentation does not infer topoi from
keywords, metaphor is not equated with manipulation, intertextual provenance
requires separate evidence, sociocognitive review does not infer individual
mental states, and historical context is not invented.
