# Local language capabilities

`discourselens nlp capabilities LANGUAGE` and the MCP
`language_capabilities` tool inspect configured local capabilities. They do not
download models or call remote services. The CLI/Python SDK can probe an
explicitly selected already-installed model, including Stanza, for example:

```bash
discourselens nlp capabilities ar --backend stanza --model ar-padt --model-dir /path/to/stanza_resources
```

This runs a small language-matched local fixture for English, Spanish, or
Arabic and records which token fields and parser outputs were actually exposed.
For a language without a probe fixture, language-dependent capabilities remain
`UNASSESSED`; DiscourseLens does not reuse an English sample as a proxy.
`SUPPORTED/TESTED` describes this technical probe only, not general linguistic
quality. For recognized spaCy/Stanza backends, `UNSUPPORTED` means the selected
pipeline does not contain a required component; `LIMITED` means the component is
present but the probe did not expose a required annotation. The probe first
requires complete token coverage: every non-whitespace source character must be
covered exactly once, in source order, with matching half-open offsets. Empty or
partial token output is `LIMITED` and cannot make dependent
`sentence_segmentation` or `named_entities` `SUPPORTED`, even when the selected
pipeline advertises those components. A missing Stanza lemma is preserved as
missing rather than replaced with the surface form, so it cannot make
lemmatization appear available. Both `UNSUPPORTED` and `LIMITED` block a
required capability unless a future operation explicitly defines a degraded
mode. Eligibility accepts exactly the operational status `SUPPORTED`;
assurance is recorded separately, and no assurance value upgrades a different
or unrecognized operational status into an eligible capability. A required
`SUPPORTED` capability with `UNASSESSED` assurance remains runnable and records
an assurance limitation. `TESTED` means the local capability probe ran, not
that linguistic quality or a CDA method was validated.
Stanza model identifiers map to local resource package names (for example,
`ar-padt` selects Stanza package `padt`; `default` selects the configured
per-processor defaults). The loader explicitly disables resource downloads and
fails if the selected package is not already available.

Surface frequency, exact search and KWIC are language-independent local text
operations. Lemmas, POS, morphology, dependencies and NER are reported as
`SUPPORTED/TESTED` only when token coverage is complete and the configured
local model exposes the required annotation/component. For sentence segmentation
and NER, this operational probe is a backend/component gate; it does not claim
that every input contains multiple sentence boundaries or named-entity spans.
Sentence segmentation is not inferred from a generic tokenizer; Stanza's
sentence-aware tokenizer is distinguished from spaCy's tokenizer.
Languages without a configured/installed default model, including
Arabic in the current baseline, report those NLP operations as `UNASSESSED`
until an explicit backend/model is probed. This says nothing about
methodological validation of CDA skills: parser availability is preprocessing
capability, not a guarantee of theory-level accuracy.

Saved controlled profiles may declare concrete local preprocessing requirements
(for example `pos` or `dependency_parse`). Before a run is created,
DiscourseLens records the installed capability status for every declared
language and rejects the profile when a required capability is anything other
than `SUPPORTED`. The stored AnalysisRun retains this preflight record. Profiles
with no local NLP requirement remain runnable for their explicitly supported
deterministic operations; this does not promote their interpretive methodology
to `VALIDATED`.

Saved controlled profiles may select an exact local NLP backend/model for each
declared language. For Stanza, the profile stores an environment-variable name
for the local resources directory, never its resolved machine-specific path:

```json
{
  "languages": ["en", "ar"],
  "required_capabilities": ["dependency_parse"],
  "nlp_routes": {
    "en": {"backend": "spacy", "model": "en_core_web_sm"},
    "ar": {
      "backend": "stanza",
      "model": "ar-padt",
      "model_dir_env": "DL_STANZA_RESOURCES"
    }
  }
}
```

Before a run, the selected installed backend is probed against that language's
fixture. The AnalysisRun snapshot records the backend, model, environment
variable name, observed statuses, and limitations; the environment variable's
value is not persisted. A missing/unloadable explicit route is ineligible and
never falls back to another model. Profiles without `nlp_routes` continue to use
the default local capability registry. This is technical preprocessing
eligibility only; it does not validate a CDA method or imply that the controlled
LLM pipeline consumes parser output unless a skill explicitly requests it.

The optional real-model regressions in `tests/test_nlp_backends.py` exercise the
installed spaCy defaults when those packages are present. Its Stanza integration
case is opt-in through `DISCOURSELENS_TEST_STANZA_RESOURCES`, which must point to
an existing local resource directory; the loader uses `DownloadMethod.NONE` and
does not fetch models. These tests verify only the production probe against a
small language-matched sample, not CDA methodology or broad linguistic quality.

## Deterministic cue-language gate

The current built-in textual-descriptor and evidence-extraction cue inventories
are English/Spanish only. In a controlled profile with explicitly labeled
documents and a mixed-language declaration, these stages run only on revisions
whose `language` metadata maps to an available cue inventory. Other revisions
receive no cue findings; the run freezes the selected revision/language map,
records `SKIPPED` stages when no safe assignment is possible, and exposes a
warning. If an unassessed language is in the profile and document language
metadata is missing, both cue stages are skipped rather than applying English
rules to potentially different-language text. Lexical tools and configured
research-question retrieval may still run over the immutable whole
`ScopeSnapshot`. A skipped or zero-result cue stage never indicates that a
phenomenon is absent. LLM interpretation remains provider-dependent and is not
language-quality or methodological validation.

When a profile declares exactly one supported cue language, that declaration may
be used for unlabeled revisions only if every existing language assignment is
consistent with it. The AnalysisRun freezes this per-revision assumption and
emits a warning that the language was declared, not detected. A multilingual
profile with missing assignments does not guess: it runs only on revisions that
have an explicit eligible language and records the unassigned revisions as
excluded. If no revision can be assigned safely, both cue stages are skipped.
