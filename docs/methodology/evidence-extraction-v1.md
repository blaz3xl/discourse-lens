# Descriptive evidence extraction v1

`builtin.evidence-extraction` is a deterministic candidate-selection skill, not
a CDA interpreter. It scans every persisted canonical sentence in the supplied
immutable `ScopeSnapshot`, using a small versioned set of English/Spanish cues:
modality, selected actor-reference terms, reported-speech/quotation markers,
negation, and a narrow surface passive pattern.

A match selects the complete sentence as a candidate passage. Multiple cue
matches in one sentence still create one passage. The skill creates one
`descriptive_candidate_inventory` finding for a run, with exact Core-owned
textual evidence. It does not label social actors, infer agency or ideology,
attribute author intention, or decide whether a construction is significant.

The output coverage is `BROAD`: all persisted sentences in the original scope
are scanned, but the candidate rules are intentionally incomplete and the
passage list is capped (48 by default). Selection is deterministic and
round-robin across revisions. Consequently, an empty inventory means only
that these particular rules selected no sentences; it is not evidence of
absence. A capped set cannot support prevalence or universal claims.

Unsupported languages are rejected instead of silently receiving English or
Spanish rules. Extending the cue inventory requires a versioned methodological
change, tests, and documentation update. This v1 skill uses only canonical
sentences and text; it does not call an NLP model or a remote service.
