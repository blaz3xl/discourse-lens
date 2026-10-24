# SPIKE-005 — Language capabilities

## Result (local rerun 2026-10-01)

The expanded local harness ran eleven curated technical fixtures per language.
All eleven input strings were segmented as one sentence each. This is a bounded
technical feasibility check, **not** a validation dataset for any CDA
methodology.

| Language | Backend / model | Sentence, token, lemma, POS, morphology, dependencies, NER | Candidate preprocessing | Assurance | Limitation |
| --- | --- | --- | --- | --- | --- |
| English | spaCy `en_core_web_sm` 3.8.0 | SUPPORTED | Agency, passive, modality and social actors: SUPPORTED | TESTED | Small model; fixture check only. |
| Spanish | spaCy `es_core_news_sm` 3.8.0 | SUPPORTED | Agency, modality and social actors: SUPPORTED; passive: LIMITED | TESTED | No passive cue found in these fixtures; fixture check only. |
| Arabic | Stanza `default` | SUPPORTED | Agency, modality and social actors: SUPPORTED; passive: LIMITED | TESTED | No passive cue found in these fixtures; fixture check only. |

Exact fixture evidence:

| Language | Tokens | Morphology-bearing tokens | NER spans (actor labels) | Agency cues (fixtures) | Passive cues (fixtures) | Modal cues (fixtures) | Social-actor cues (fixtures) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English | 78 | 70 | 3 | 12 (11) | 6 (3) | 3 (3) | 22 (11) |
| Spanish | 78 | 65 | 4 | 12 (11) | 0 (0) | 2 (2) | 24 (11) |
| Arabic | 64 | 46 | 4 | 11 (10) | 0 (0) | 2 (2) | 27 (11) |

Parentheses give the number of fixtures containing at least one cue. Cue totals
can include overlapping syntactic and NER signals; they are not counts of
unique, gold-labeled actor mentions. A `SUPPORTED` candidate status means the
configured heuristic produced at least one cue in this fixture set. `LIMITED`
means no cue was observed here, not that the language can never express it.
NER is `SUPPORTED` only when the configured component emits at least one entity
span in these fixtures.

The models are local experimental assets and are excluded from version control.
Production must expose these statuses through `LanguageCapability` and refuse a
required operation that is LIMITED, UNASSESSED or UNSUPPORTED unless that
operation explicitly declares an allowed degraded mode.

## Harness

`experiments/spike_005_language_capabilities/check.py` runs only local
libraries/models. It sets Stanza `DownloadMethod.NONE`, so a missing local
resource fails instead of causing a resource-manifest network request. It does
not call an LLM or send fixture text to a remote API.

Fixtures exist for Spanish, English and Arabic, including basic agency/modality
and social-actor preprocessing examples. A baseline run on 2026-09-29 used ten
fixtures per language in the isolated Python 3.12 experiment environment with
spaCy 3.8.16, `en_core_web_sm`/`es_core_news_sm` 3.8.0 and Stanza 1.14.0. In the
main project `.venv`, the two spaCy model packages are not
installed; neither optional spaCy nor Stanza is installed there, so the default
capability command correctly reports English/Spanish parser operations as
`UNASSESSED`. Arabic is also `UNASSESSED` by default until a user explicitly
selects an installed local Stanza model and resource directory.

The 2026-10-01 rerun used the isolated Python 3.12.8 experiment environment
with spaCy 3.8.16, `en_core_web_sm`/`es_core_news_sm` 3.8.0, Stanza 1.14.0 and
Torch 2.14.0+cu130. The exact command was
`experiments/.venv/bin/python experiments/spike_005_language_capabilities/check.py`;
it exited successfully and reproduced the evidence table above. All assets
were already local, and the Stanza pipeline uses `DownloadMethod.NONE`; no
downloads or paid APIs were used. The main project `.venv` was also checked and
still has neither optional backend nor either spaCy model installed, so its
optional backend probes remain unassessed unless users select local models.

An earlier production backend integration check, separate from this expanded
fixture run, passed 14/14 cases in `tests/test_nlp_backends.py`, including the
real English/Spanish spaCy probes and explicit Arabic `ar-padt` Stanza probe.

The candidate preprocessors collect syntactic subject/passive/modal cues and
broad nominal participants plus actor-like NER spans for review. They do not
classify actors or infer agency, responsibility, stance, or modality. Each
language's named-actor fixture helps exercise NER output, but this small set does
not measure precision or recall.

The production CLI capability path was smoke-checked on 2026-10-01 from the
isolated environment with these local model routes:

```bash
PYTHONPATH=src experiments/.venv/bin/python -c 'from discourselens.cli import app; app()' nlp capabilities en --backend spacy --model en_core_web_sm
PYTHONPATH=src experiments/.venv/bin/python -c 'from discourselens.cli import app; app()' nlp capabilities es --backend spacy --model es_core_news_sm
PYTHONPATH=src experiments/.venv/bin/python -c 'from discourselens.cli import app; app()' nlp capabilities ar --backend stanza --model ar-padt --model-dir experiments/spike_005_language_capabilities/models
```

All three commands exited successfully. For all routes, sentence segmentation,
tokenization, lemma, POS, morphology and dependency parsing reported
`SUPPORTED/TESTED`. English and Spanish named entities reported
`SUPPORTED/TESTED`; Arabic `ar-padt` reported named entities
`UNSUPPORTED/TESTED` because that package has no NER component. The CLI also
reported exact search, surface frequency and KWIC as `SUPPORTED/TESTED` for each
language. The Stanza loader uses `DownloadMethod.NONE`, and the selected spaCy
models and Stanza resources were already local; no model download or paid API
call occurred. This explicit Arabic route does not change Arabic's default
registry status of `UNASSESSED`.

The English/Spanish result depends on mapping spaCy's self-headed `ROOT` token
to the backend's canonical root representation (`dependency_head=None`); before
that fix the real parser trees were incorrectly reported `LIMITED`.

The spike harness's Stanza `default` package includes a separate NER model;
these are different explicit configurations and must not be conflated. No CDA
skill is claimed validated.

The isolated environment's package metadata declares MIT for spaCy and
`en_core_web_sm`, GNU GPL 3.0 for `es_core_news_sm`, and Apache License 2.0 for
the Stanza library. Those declarations do not establish the licenses of model
training data or Stanza's Arabic model assets; that asset-level license remains
unverified and needs review before redistribution or production use. The Arabic
tokenize, MWT, lemma, POS, dependency, NER and supporting model resources are
already present locally. spaCy loads the installed model packages, while Stanza
uses `DownloadMethod.NONE` with the local resource directory.

The capability model remains sound: frequency/KWIC can be eligible independently
of dependency-agency; missing dependency capability must reject that operation
and produce an explicit partial-profile warning.
