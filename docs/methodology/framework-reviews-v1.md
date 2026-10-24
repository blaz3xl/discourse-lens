# Framework reviews v1

The evidence-bound review executor may be parameterized with one of these
versioned profiles: `fairclough-text`, `van-leeuwen-social-actors`,
`dha-strategies`, `argumentation`, `van-dijk-ideology`, `framing`, `metaphor`,
`intertextuality`, `sociocognitive`, `historical-context`, `power`, and
`critical-reviewer`.

Each profile is a methodological constraint on a shared trusted executor, not
an autonomous classifier. It can only cite persisted textual evidence supplied
by Core; it must return an alternative interpretation, and it must not infer
authorial intent, psychological state, political affiliation, or fabricate
locations. The generated finding records the selected profile ID and version.

`critical-reviewer` is adversarial toward the proposed reading, not toward a
speaker, author or social group. It may identify insufficient support or an
overclaim, but cannot introduce a new uncited textual claim.

`argumentation` does not label topoi from lexical triggers. `metaphor` does not
equate figurative language with manipulation. `intertextuality` describes only
the voices and references visible in the record; provenance requires distinct
external evidence. `sociocognitive` never infers individual mental states.
`historical-context` must distinguish text, researcher-supplied context and
verified external material, and is required to state when that material is
insufficient.
