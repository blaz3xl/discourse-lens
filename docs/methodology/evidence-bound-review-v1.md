# Evidence-bound review v1

This LLM-assisted review accepts only persisted textual evidence from an existing
Finding. The model must emit strict JSON containing an interpretation, alternative
interpretation, limitations and a subset of the supplied evidence IDs. The backend
rejects fabricated IDs and creates a new immutable review run.

Evidence excerpts, observations, and prior interpretations are untrusted data.
They are serialized together in a marked JSON data block and are not interpolated
as instruction text. The prompt directs the model never to follow instructions
inside those values or let them alter scope, method, privacy rules, or available
operations. The review prohibits inferring author intent, psychological state or
political affiliation from text alone. This reduces prompt-injection risk but is
not a guarantee that a model will resist every adversarial input; outputs remain
subject to evidence-ID and schema validation.
