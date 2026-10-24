# Model routing

A Model Routing Profile maps named controlled-analysis framework stages and
the final `synthesis` stage to saved model presets. Synthesis may have its own
route or use the profile's explicit fallback preset; without either, the
controlled run is rejected before execution. The profile stores preset names,
never secret values or provider credentials. A missing interpretation stage
without a fallback is an error, not a silent model fallback.

For a saved analysis profile, `preset_name` (CLI `--preset`) overrides that
analysis profile's `default_preset`; it is a default for runs without a Model
Routing Profile and does not replace explicit stage routes. When a Model
Routing Profile is selected, its stage routes and its own fallback determine
the resolved stage map; `preset_name` does not replace either. A required stage
without a route or fallback fails before execution.

Each preset may also store JSON generation parameters. They are passed through
the provider-neutral request contract. An adapter declares the parameter names
it can map; undeclared names fail before a request is sent rather than being
silently dropped. Provider adapters map declared names to their own wire format.
These declarations describe adapter-level support, not a guarantee that every
model under a provider accepts every parameter; provider errors remain visible.
HTTP throttling is normalized to `RateLimitError` (with `retry_after_seconds`
when the provider sends a numeric `Retry-After` header); recognized account
quota/billing exhaustion becomes `QuotaError`. Neither path retries
automatically. Provider error bodies are not copied into exceptions or the
execution ledger, while the ledger preserves the normalized error class.

OpenAI-compatible generation and embedding may omit authentication only when an
explicit `no-auth` selector is bound to an HTTP(S) loopback endpoint
(`localhost`, `127.0.0.1`, or `::1`). The CLI exposes this as `--no-auth`.
Remote and private-LAN endpoints still require an environment/keyring reference,
and the endpoint/transport restriction is checked again at execution time.
Unauthenticated requests omit `Authorization` rather than sending a dummy token.

Projects may store an exact `(provider_id, model)` capability profile with
`models set-capabilities` and inspect it with `models list-capabilities`.
Execution intersects the project declaration with the adapter's transport
capabilities: a profile can restrict an adapter but cannot invent an unsupported
wire mapping. A documented built-in profile is used when the project has no
profile for an exact model ID covered by the application's model metadata
registry; the project profile takes precedence. The current OpenAI entry is
`gpt-6-luna`, based on [OpenAI's model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).
Unknown IDs still require an explicit profile.
Profiles default to `UNASSESSED`; `TESTED` or `VALIDATED` is
researcher-supplied assurance, not an automatic DiscourseLens certification.
The effective intersection and its provenance are snapshotted per execution.
The `structured_output` flag is an exact-model opt-in: it enables a native JSON
Schema response format only for adapters with an implemented mapping. When the
profile is absent or false, review/synthesis retain prompt-guided JSON plus local
Pydantic/evidence validation; this fallback is visible in the recorded effective
capabilities. When enabled, the schema name and canonical schema are included in
the request fingerprint, but the schema contents are not copied to the execution
ledger. Constrained decoding concerns output shape only; local schema, evidence-ID,
scope and citation validation remains mandatory.

The adapters map this to OpenAI Chat Completions `response_format.json_schema`,
Anthropic Messages `output_config.format`, and Gemini GenerateContent
`generationConfig.responseFormat.text`. Each provider/model must be explicitly
declared because model support and accepted JSON Schema subsets vary. See the
[OpenAI structured outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs),
[Anthropic structured outputs guide](https://platform.claude.com/docs/en/build-with-claude/structured-outputs),
and [Gemini structured output guide](https://ai.google.dev/gemini-api/docs/structured-output).
Native OpenAI Chat Completions maps a declared effort to `reasoning_effort`,
and Gemini GenerateContent maps a declared effort to `generationConfig.thinkingConfig.thinkingLevel`.
Neither adapter enables reasoning for an exact model without a project capability
profile; generic OpenAI-compatible endpoints still have no inferred reasoning
mapping. Anthropic requires an exact model profile declaring both effort and
`reasoning_modes`. A preset chooses `--reasoning-mode adaptive` or `manual`;
adaptive maps to `thinking.type=adaptive` plus `output_config.effort`, while
manual requires an explicit `--reasoning-budget-tokens` value (at least 1024 and
below `max_tokens`) and maps to `thinking.type=enabled`. When thinking is enabled,
temperature must be omitted or 1. The selected mode, effort, budget and effective
capability intersection are persisted in the execution ledger. This follows
Anthropic's model-specific API contract; manual budgets are not inferred from an
effort label. See [Anthropic's extended-thinking documentation](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
and [effort documentation](https://platform.claude.com/docs/en/build-with-claude/effort).

Cohere Chat v2 profiles can declare `enabled` and `disabled` reasoning modes
for an exact model. Those modes map directly to the `thinking.type` toggle in
the `/v2/chat` request. Cohere's local `manual` mode accepts a positive token
budget (including budgets below 1,024), requires explicit `max_tokens`, and
reserves at least 1,000 tokens for the final response. It maps to
`thinking.type=enabled` with `thinking.token_budget`. Each selected mode must
appear in that model's capability profile before execution; undeclared modes
are rejected before provider transport. See [Cohere's reasoning guide](https://docs.cohere.com/docs/reasoning)
and [Chat v2 reference](https://docs.cohere.com/v2/reference/chat).

For example, an adaptive-capable exact model can be declared and selected with:

```bash
discourselens models set-capabilities PROJECT anthropic MODEL \
  --capabilities-json '{"reasoning_efforts":["low","high"],"reasoning_modes":["adaptive"]}' \
  --source-reference 'https://platform.claude.com/docs/en/build-with-claude/extended-thinking'
discourselens models save-preset PROJECT review anthropic lab MODEL \
  --reasoning high --reasoning-mode adaptive
```

Manual thinking additionally requires a budget of at least 1,024 tokens and a
`max_tokens` generation parameter greater than that budget. The preview command
rejects a missing or inconsistent model declaration before execution.

Every model-execution ledger entry stores a secret-screened snapshot of the
resolved parameter object. Editing a preset later therefore does not rewrite the
settings used by historical executions. Prompts, completions, and credentials
remain outside the ledger. The `models save-preset` command accepts a JSON
object through `--parameters-json`.

Reranking is a separate capability and route from generation presets. A
controlled profile can explicitly select a reranking provider, credential
alias, model, and optional top-N. The API never accepts key material. Retrieval
first enforces its immutable scope; the reranker receives only those validated
canonical candidates. Original lexical rank/score and reranker rank/score are
stored independently and exported with the run. Offline mode blocks a selected
remote reranker before corpus text is sent.
Reranking provenance distinguishes the full submitted candidate count from the
provider's returned top-N count; an oversized requested top-N is capped to the
available candidate population and both requested and effective limits are
recorded.

The reviewed reranking-only adapters are `jina-reranker` and
`cohere-reranker`; neither is a generation preset. Cohere uses its v2 Rerank
endpoint, caps the request at 1,000 supplied candidates, and pins the documented
4,096-token per-document truncation setting. The Core still validates the
original scope before either remote adapter receives candidate text.

Embedding transport is separately selected by an embedding preset. The Jina
adapter targets the v1 embeddings endpoint and uses `retrieval.passage` for
indexed text plus `retrieval.query` for queries only for documented v3–v5
embedding model families; other explicit model IDs use symmetric input. The
index records the document-side mode, and semantic retrieval checks that
provenance together with provider, exact model, dimension, revision scope and
chunking version before embedding a query. The adapter batches at most 128 texts
per request and does not enable provider-side truncation. See [Jina's Embedding
API](https://jina.ai/embeddings/) for the task parameter and model-specific
behavior.

Gemini embeddings are registered separately as `gemini-embeddings`, distinct
from the generation provider `gemini`. `gemini-embedding-001` sends explicit
`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types with automatic truncation
disabled. `gemini-embedding-2` does not accept task types, so the adapter uses
Google's documented `title: none | text: ...` document and
`task: search result | query: ...` query formats. Separate inputs are sent as
individual items in `batchEmbedContents`, preserving one vector per paragraph.
The document-side mode is recorded in index provenance; unknown Gemini
embedding model IDs are rejected until their retrieval format is reviewed. See
Google's [embedding guide](https://ai.google.dev/gemini-api/docs/embeddings)
and [REST reference](https://ai.google.dev/api/embeddings).

`vectors preview-paragraphs` inspects a paginated set of canonical paragraphs
against the saved embedding preset before index construction. It reports the
effective endpoint, provider input role, canonical offsets, hashes, and request
batch shape without reading secret values or performing network I/O. Exact
provider-prepared text and request bodies are opt-in with `--include-text`.
The preview and provider adapters share the same provider-specific wire-payload
builder, including Gemini's model-specific prefixes/configuration, so the
inspectable body cannot drift independently from execution.

When a controlled CDA run starts, DiscourseLens resolves every selected
framework and synthesis stage to a preset and persists the resolved
`framework_presets` map plus per-stage `framework_preset_snapshots` in the
immutable `AnalysisRun` configuration. These resolved values are the
authoritative provenance for what actually executed; `preset_name` and the
named routing profile are resolution inputs. Resume uses the historical map
and snapshots, so later edits to a named routing profile cannot rewrite the
model choices of an older run. Each stage snapshot records its exact model,
credential alias, effective endpoint, reasoning request, generation parameters
and capability profile. The secret value and raw credential reference are never
copied into run configuration.
A one-way SHA-256 fingerprint of the configured reference is stored so execution
and resume can reject an alias that has since been remapped. This pins the
reference selector, not the secret value behind it; rotating a key behind the
same environment/keyring reference remains under the provider's credential
management. A retry uses the snapshot even if the current preset, endpoint or
capability profile has changed; legacy failed runs without the current snapshot
contract are rejected for resume rather than silently resolved from today's
settings.
