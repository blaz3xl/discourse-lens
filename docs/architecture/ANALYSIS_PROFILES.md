# Declarative analysis profiles

Profiles are stored as versioned JSON in the local project and currently target
only the trusted `controlled-cda` pipeline. They may declare framework IDs,
language marker sets, framework-specific researcher questions, researcher-supplied
counterevidence queries, an optional model-routing profile and a default preset.
For a saved-profile run, `preset_name` (the CLI `--preset` option) overrides only
the profile's `default_preset`. It supplies all framework and synthesis stages
when no model-routing profile is selected. When one is selected, that routing
profile's explicit stage routes and its own fallback determine the stages;
`preset_name` does not replace either. A stage with neither route nor fallback
fails before execution. The run's resolved `framework_presets` map and immutable
`framework_preset_snapshots` record the presets that actually executed; the
profile values and `preset_name` are inputs to resolution, not historical
execution provenance.

Counter-evidence literal scans inspect the complete original scope and persist all
retrieval hits up to a 10,000-occurrence per-run safety ceiling. Larger result sets
are rejected before partial retrieval records are created and require a narrower
query. CLI/MCP/Python result payloads are paginated with `offset` and `limit`.
`research_questions` is an optional mapping from a selected interpretive
framework ID to one question. Each question starts an independent deterministic
lexical retrieval over the run's original immutable `ScopeSnapshot`; it is not
limited to cue-derived seed evidence. Returned passages are persisted as
`RetrievalHit` records and linked to an example-candidate finding with
`TARGETED` coverage. That finding is then reviewed only by its selected
framework. An empty targeted result is not treated as corpus-wide absence.
`question_candidate_limit` controls how many ranked passage candidates are
returned per question (default 12, maximum 12). The limit is included in the
canonical profile payload/hash and immutable AnalysisRun configuration, so resume
uses the same query and bound. Textual evidence extraction remains a separate
descriptive pass and does not gate this retrieval stage.
`counterevidence_queries` keys must exactly identify a phenomenon emitted by
the current evidence-extraction skill. Unsupported/misspelled targets and blank
queries are rejected before an AnalysisRun is created; they are never silently
ignored. The current extractor emits `descriptive_candidate_inventory`.
The built-in descriptive cue passes currently cover English and Spanish only.
For mixed-language profiles, they use per-document `language` metadata to run
only where a matching cue inventory exists. If an unassessed language is in the
profile and scoped documents cannot be assigned safely, those stages are
persisted as `SKIPPED` with a warning; lexical research-question retrieval may
still proceed over the original scope. The AnalysisRun freezes this language
execution plan. A profile declaring exactly one supported cue language may apply
that declared language to unlabeled revisions only when all existing labels are
consistent; this is recorded as an explicit assumption and warning, never as
language detection. A multilingual profile does not assign its cue inventories
to unlabeled revisions by guesswork.
Profiles may also pin a local NLP backend/model per declared language through
`nlp_routes`. spaCy routes identify an installed model; Stanza routes reference
the local resources directory via an environment-variable name. The environment
value/path is not stored. Preflight probes the exact selected route, snapshots
its identity and capability statuses into the AnalysisRun. A route whose
capabilities are unassessed or unsupported is not used to parse scoped documents
for that operation; its skip is recorded instead. This is capability preflight,
with no fallback. By default, routes are not run. Profiles may explicitly set
`local_preprocessors` to `social_actor_candidates` and/or `agency_candidates`;
the former requires POS and the latter dependency parsing. For these stages,
each scoped document must have an explicit `language` document-metadata value
and a matching profile route. The exact local route builds parsed-token Parquet
artifacts, then the trusted candidate-only skill executors consume those
artifacts within the same AnalysisRun. Capability eligibility is evaluated per
operation and revision: a revision with `SUPPORTED` status for that operation is
processed, while `LIMITED`, `EXPERIMENTAL`, `UNSUPPORTED`, or `UNASSESSED` status
is recorded as an operation-specific skip. A mixed result is `PARTIAL`; if no scoped revision
is eligible, the operation still receives a `SKIPPED` SkillRun. The immutable
AnalysisRun configuration and result warnings record each operation's processed
and skipped revision IDs, language, capability status, and limitations. Parsed
artifacts are built only for revisions eligible for at least one operation.
All local SkillRuns retain the original AnalysisRun ScopeSnapshot; the operation
plan and artifact IDs disclose the narrower local-processing population.
CounterEvidence and framework retrieval continue to use the complete original
scope. Missing language metadata, an explicit language route, or a configured
Stanza resource environment value remains a hard configuration error. Explicit
profile-level `required_capabilities` also remain run-level requirements. There
is no language or model fallback. Candidate skills record syntactic observations
only and do not claim validated CDA interpretation or absence in skipped
revisions.
They may also opt in to one scoped reranking route by declaring
`reranking_provider_id`, `reranking_credential_alias`, `reranking_model`, and
an optional positive `reranking_top_n`. All three route identifiers are
required together; the credential value itself is never stored. Profile runs,
CLI batch runs, and MCP controlled/batch tools preserve the original retrieval
hits and persist reranker ordering/scores separately. No reranking request is
made unless this route is explicitly configured.

Profiles cannot import Python, launch a shell, access the network or register
new skills. Unknown frameworks and invalid contracts fail before an AnalysisRun
is created. The resolved profile choices are then copied into the immutable run
configuration. A saved-profile run additionally records the stored name,
declared Semantic Versioning 2.0.0 version, a canonical SHA-256 hash and the
validated canonical JSON payload itself. The immutable run configuration remains
explainable in a research bundle even if the local profile record is later
unavailable; later profile edits cannot rewrite historical provenance.

When an AnalysisRun completes, its finding content, FindingEvidence links and
finding lineage are frozen in SQLite and each finding receives the run's
`completed_at` as `finalized_at`. Failed runs may be resumed under their
original snapshot; completed findings cannot be extended or rewritten. Human
review remains a separate append-only layer and can still record
`ACCEPTED`/`MODIFIED`/`REJECTED` decisions without changing the generated
finding or its evidence links.

Profiles are discoverable through the Core project API, `discourselens profiles
list` / `info`, and the local MCP profile discovery tools. Discovery and reading
are data-only operations; they never load third-party executors.
