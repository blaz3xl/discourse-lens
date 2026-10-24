# Providers and model configuration

DiscourseLens stores provider selection and non-secret credential references in
the project. It does not infer that a model supports a feature just because its
provider transport is available. A successful local preview is a configuration
check, not proof that a remote endpoint, credential, model ID, or model feature
will work. Provider IDs must be present in the reviewed catalog below; an
explicit endpoint does not register an unknown provider ID.

## Discover transports and configure credentials

`discourselens providers list` displays the reviewed transports and their
operation flags without reading credentials or making a network request. Use
`--format json` for the same catalog as JSON. Credential references can be
listed with `discourselens providers list-credentials PROJECT`; the command
does not resolve them, but the output includes the reference selector and
alias. `credential_alias` is a human label such as `university` that identifies
the saved reference. Never put an API key, token, password, or other key
material in an alias. Credential and preset listings show this label and the
reference selector, never the resolved environment-variable or keyring value.

`providers add-credential` takes a project, provider ID, and alias, plus exactly
one authentication option. Give it an environment-variable name or a keyring
selector, never a key value:

```bash
discourselens providers add-credential research openai university \
  --secret-reference DL_OPENAI_UNIVERSITY
```

`--secret-env` is an alias for `--secret-reference`. The referenced environment
variable must be available to the process that executes the request. To use an
OS keyring entry, install the optional extra with `pip install
"discourselens[secrets]"`, provision the entry separately, and save only its
`keyring:SERVICE:ACCOUNT` selector:

```bash
discourselens providers add-credential research anthropic university \
  --secret-reference keyring:discourselens:anthropic/university
```

Native OpenAI, Anthropic, Gemini, and Cohere endpoints have reviewed defaults;
an explicit `--endpoint` can override them. Both compatible transports require
an explicit endpoint. They require a credential reference except that a reviewed
OpenAI-compatible ID can use `--no-auth` with an explicit loopback endpoint.
For `anthropic-compatible`, the endpoint is the base URL of a service that
implements Anthropic Messages; the adapter sends requests to its Messages
route. For example:

```bash
discourselens providers add-credential research anthropic-compatible university \
  --secret-reference ANTHROPIC_GATEWAY_KEY \
  --endpoint https://gateway.example/v1
```

For a local OpenAI-compatible server that needs no auth, the only supported
no-auth configuration is an explicit HTTP(S) loopback URL:

```bash
discourselens providers add-credential research openai-compatible local \
  --no-auth --endpoint http://localhost:11434/v1
```

The no-auth exception is limited to reviewed OpenAI-compatible provider IDs and
loopback hosts (`localhost`, `127.0.0.1`, or `::1`). It does not apply to
`anthropic-compatible`, which always requires a credential reference. Private-
LAN addresses are not treated as local. Proxy settings must also permit a
direct loopback connection. Remote endpoints require an environment or keyring
reference.

The current catalog separates operations rather than claiming one universal
provider API. These are the registered provider IDs and implemented adapter
operations:

| Operation | Provider IDs |
| --- | --- |
| Native generation | `openai`, `anthropic`, `gemini`, `cohere` |
| Anthropic Messages-compatible generation (explicit endpoint and credential) | `anthropic-compatible` |
| OpenAI-compatible generation (explicit endpoint) | `openai-compatible`, `deepseek`, `zai`, `xai`, `mistral`, `openrouter`, `together`, `groq`, `fireworks`, `ollama`, `lm-studio`, `vllm` |
| Embeddings | `openai`, OpenAI-compatible IDs above, `gemini-embeddings`, `cohere-embeddings`, `voyage-embeddings`, `jina-embeddings` |
| Reranking only | `jina-reranker`, `cohere-reranker` |

These flags mean a reviewed transport/factory exists, not that every model,
account, endpoint, parameter, or feature combination is supported. For example,
embedding-only and reranking-only IDs are not generation providers.
OpenAI-compatible means the endpoint speaks the corresponding OpenAI-compatible
wire protocol; generation uses `/chat/completions` and embeddings use
`/embeddings`. A catalog entry provides the corresponding adapter mapping, but
the configured endpoint must implement that operation. This does not imply
feature parity with native OpenAI, nor that an arbitrary model ID exists at that
endpoint. Unknown provider IDs are rejected until a transport is registered.
`anthropic-compatible` is a separate Messages-protocol transport, not an
OpenAI-compatible provider ID. Its catalog entry enables text generation only;
it does not provide embeddings or reranking. Model reasoning and optional
generation capabilities are not inferred. An exact-model capability profile
can declare only features the adapter can map for that endpoint and model; a
profile cannot create a missing wire mapping.
The `zai` provider ID covers Z.AI/GLM models through the OpenAI-compatible
transport only; it is not a native GLM adapter, and the selected endpoint must
implement the compatible routes used by the requested operation.

## Presets, capabilities, and request preview

Save a named provider/credential/model selection with
`models save-preset PROJECT NAME PROVIDER CREDENTIAL_ALIAS MODEL`. Presets store
selection metadata and JSON generation parameters, not resolved credentials.
`CREDENTIAL_ALIAS` is the human label for a registered credential reference;
it is never a place for key material:

```bash
discourselens models save-preset research analysis openai university MODEL_ID
discourselens models list-presets research
```

Replace `MODEL_ID` with an ID supported by the selected endpoint. The CLI does
not discover remote model catalogs or certify model availability. Optional
generation settings use `--parameters-json` with a JSON object; an adapter
rejects parameter names it cannot map instead of silently dropping them, while
the remote model may still reject a parameter the adapter can transmit.

Capability declarations are keyed to the exact `(provider_id, model)` pair and
can only narrow the selected adapter's support. Allowed fields are `text`,
`system_prompt`, `vision`, `structured_output`, `streaming`, `embeddings`,
`reasoning_efforts`, `reasoning_modes`, and `generation_parameters`. Declare
only features confirmed for that model/endpoint. For example, after checking an
exact model's documentation, a researcher can record reasoning information:

```bash
discourselens models set-capabilities research anthropic MODEL_ID \
  --capabilities-json '{"reasoning_efforts":["low","high"],"reasoning_modes":["adaptive"]}' \
  --source-reference 'https://platform.claude.com/docs/en/build-with-claude/extended-thinking'
```

The default assurance is `UNASSESSED`; `TESTED` and `VALIDATED` are researcher-
supplied labels, not DiscourseLens certifications. The source reference is
provenance supplied by the researcher. A declaration cannot add a wire mapping
that the adapter does not implement. Native OpenAI and Gemini reasoning
parameters, Anthropic thinking modes, and Cohere thinking toggles are
model-specific; generic OpenAI-compatible transports do not infer native
reasoning support. For Anthropic, the exact-model profile must declare both
an effort and a supported mode. Manual thinking additionally requires a
budget of at least 1,024 tokens, less than `max_tokens`; while thinking is enabled, temperature must be omitted
or `1`. These rules are not portable assumptions for other providers. See
[Model Routing](architecture/MODEL_ROUTING.md) for adapter mappings and limits.

Vision input supports one explicitly supplied local PNG, JPEG, or WebP image
for buffered Gemini, OpenAI Chat Completions, and Anthropic Messages generation.
The exact `(provider_id, MODEL_ID)` profile must declare `vision: true`; support
is never assumed for every model. OpenAI-compatible and Anthropic-compatible
endpoints must implement their corresponding image wire format, and each
endpoint/model pair must be confirmed before opting in. The file is
signature-checked and limited to 8 MiB; the fully encoded provider request is
limited to 18 MiB. The request ledger stores only image digest, MIME type, and
size. Images are not automatically extracted from imported documents, image
URLs are not fetched, and image streaming or vision through other adapters is
unsupported. Gemini uses the `generateContent` `inline_data` part from its [image understanding guide](https://ai.google.dev/gemini-api/docs/generate-content/image-understanding).
OpenAI Chat Completions uses a text and `image_url` content-part list with a
local image encoded as a data URL, as described in the [OpenAI image input guide](https://developers.openai.com/api/docs/guides/images-vision) and [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat).
Anthropic Messages uses an `image` content block with a base64 `source` and `media_type`, followed by the prompt text, as described in Anthropic's [vision guide](https://docs.anthropic.com/en/docs/build-with-claude/vision).

Text streaming is an explicit `ModelGateway.stream(provider_id, request)` call
for OpenAI Chat Completions, Anthropic Messages, and native Gemini
`streamGenerateContent`. It yields text deltas and requires
`streaming: true` in the exact model capability declaration used for a saved
preset; compatible endpoints must be confirmed individually. Gemini uses
`alt=sse` and the `x-goog-api-key` header, emits only candidate text parts,
ignores thought and non-text parts, and accepts only `finishReason: STOP` as a
complete response. `MAX_TOKENS` raises a sanitized incomplete-response error;
text deltas already emitted before that terminal event are partial and must not
be treated as a completed answer. Safety blocks and malformed/incomplete events
fail with sanitized errors. OpenAI Chat Completions accepts only a `stop`
finish reason; length limits, content filtering, and tool-call endings raise a
sanitized incomplete-response error after any already-yielded deltas. Anthropic
requires `message_start`, a successful `message_delta.stop_reason` (`end_turn`
or `stop_sequence`), then `message_stop`; `max_tokens` and other stop reasons
raise a sanitized incomplete-response error after any already-yielded deltas.
Streaming structured-output requests are
rejected, malformed/incomplete SSE responses become sanitized domain errors,
and closing the returned iterator closes the HTTP response. Anthropic streaming
requires `message_start` followed by `message_stop` and emits only
`text_delta`; pings, unknown events, tool/input JSON, and thinking/signature
deltas are validated and discarded, so private reasoning never reaches the
text-only result. Exact-profiled Anthropic adaptive/manual reasoning settings
are mapped in buffered and streaming calls; the stream does not expose thinking
or signature deltas. This text-only stream contract does not report final usage
metadata and is not used by analysis pipelines. Buffered `generate()` calls
also require an explicit normal terminal reason: OpenAI `stop`, Anthropic
`end_turn` or `stop_sequence`, and Gemini `STOP`. Truncated, filtered, refused,
tool-handoff, missing, or unknown terminal states fail as sanitized response
errors instead of being returned as complete model output. See the [Anthropic Messages
streaming protocol](https://platform.claude.com/docs/en/build-with-claude/streaming)
and [Gemini GenerateContent API](https://ai.google.dev/api/generate-content).

Cohere Chat v2 sends system/user messages to `/v2/chat`, uses bearer
authentication, and returns assistant text plus `usage.tokens` counts when
reported. JSON Schema output is available only when an exact-model profile opts
in; Cohere documents the feature for Command R/R+ and newer models, and JSON
mode requires the prompt to explicitly request JSON. The adapter does not infer
support from the provider name. Documented generation parameters are mapped
explicitly (`top_p` to `p`, `top_k` to `k`); unsupported parameters fail before
transport. Exact-model profiles may opt into the documented `enabled` and
`disabled` reasoning toggles. A saved toggle preset is accepted for execution
only when the exact `(cohere, MODEL_ID)` capability profile declares its mode;
for example:

```bash
discourselens models set-capabilities research cohere MODEL_ID \
  --capabilities-json '{"reasoning_modes":["enabled","disabled"]}' \
  --source-reference 'https://docs.cohere.com/docs/reasoning'
discourselens models save-preset research analysis cohere lab MODEL_ID \
  --reasoning-mode enabled
```

Budgeted reasoning uses the adapter's `manual` mode, which maps to Cohere's
documented enabled-thinking `token_budget` field and is accepted only when that
exact profile declares `manual`. The budget may be any positive integer, but
`max_tokens` must be explicit and leave at least 1,000 tokens beyond the
thinking budget for the final response. No reasoning-effort mapping is declared.
Cohere streaming accepts only completed text responses,
omits thinking blocks, and rejects tool/unknown events and incomplete finishes.
Text deltas emitted before a later stream error are partial and must not be
treated as a completed answer. See the [Cohere Chat v2
reference](https://docs.cohere.com/v2/reference/chat), [Chat streaming
reference](https://docs.cohere.com/v2/reference/chat-stream), and [reasoning
guide](https://docs.cohere.com/docs/reasoning).

Before a generation request, preview its destination and local capability
checks with:

```bash
discourselens models preview-request research analysis prompt.txt
```

Preview does not resolve secret values or contact the endpoint. It can report
configuration blockers such as a missing compatible endpoint or unsupported
declared settings; it cannot establish that credentials are valid, the remote
model exists, or the service will accept the request. Add `--include-prompt`
only if you want the prompt printed in local output; that text is not persisted.
When ready to make an actual request, `models generate` requires a UTF-8 prompt
file:

```bash
discourselens models generate research analysis --prompt-file prompt.txt
```

This sends the prompt to the configured provider. A prompt file avoids placing
research text directly in shell history; it does not make remote execution
private.

## Optional model cost estimates

Cost tracking is opt-in and project-scoped. Enter rates that you have verified
yourself; these commands store and list only the supplied values and do not
query provider price tables or make network requests. Input and output rates
are expressed per million provider-reported tokens. A reasoning rate is
optional and is used only when you explicitly supply one:

```bash
discourselens models price set research openai MODEL_ID \
  --input-per-million-tokens 0.25 \
  --output-per-million-tokens 1.25 \
  --reasoning-per-million-tokens 1.25 \
  --currency USD \
  --source-reference 'https://provider.example/pricing' \
  --rate-version docs-2026-01 \
  --rate-date 2026-01-15
discourselens models price list research --provider openai --model MODEL_ID --format json
discourselens models costs research --format json
```

The numeric rates above are illustrative. Replace them with rates and a source
reference you reviewed. Do not put API keys, tokens, or other credentials in a
source reference. Add a new `--rate-version` when rates change; a version cannot
be repurposed for different rates. Each model execution keeps the selected rate
snapshot and its estimate, so later rates do not rewrite earlier records.
`models costs` reads those persisted values; it does not calculate historical
costs from the current rate. If a rate or required usage count is unavailable,
the estimate is `null`, not zero. An optional reasoning rate separates
reasoning usage only when that rate is explicitly recorded.

These are estimates from provider-reported usage and user-supplied rates, not a
guaranteed invoice amount. Provider billing definitions and adjustments may
not match the usage fields or rates entered here.

## Routing profiles

`models save-routing PROJECT NAME --routes-json JSON` maps framework IDs to
saved preset names. An optional `--fallback-preset` is used for selected stages
not explicitly mapped. The map accepts only stage-to-preset metadata; it never
contains credentials. A controlled CDA run still needs a preset argument and
an explicit revision scope:

```bash
discourselens models save-routing research staged \
  --routes-json '{"fairclough-text":"analysis","synthesis":"analysis"}' \
  --fallback-preset analysis
discourselens analyze controlled-cda-review research \
  --revisions REVISION_ID --preset analysis --routing-profile staged
```

Only known framework IDs and `synthesis` are valid routes. Every selected
framework stage and synthesis must resolve to a preset, either through its
route or the explicit fallback; missing routes fail before execution. A routing
fallback is not automatic retry, load balancing, or failover when a selected
provider request fails. Inspect saved metadata with `models list-routing`.

## Embeddings and reranking

Embedding selection uses a saved model preset too, but with an embedding-capable
provider ID. For example, select an explicit provider/model and inspect the
request window before building the local vector index:

```bash
discourselens models save-preset research embed openai university MODEL_ID
discourselens vectors preview-paragraphs research embed REVISION_ID \
  --offset 0 --limit 20
discourselens vectors build-paragraphs research embed REVISION_ID
```

The preview is local and secret-free; by default it reports offsets, hashes,
prepared input metadata, and request shape, not paragraph text. `--include-text`
prints provider-prepared text and payload locally. Building an index sends the
selected revision's paragraph text to the configured embedding endpoint. Query
embedding for semantic search also sends the explicit query. Vector indexes
require the optional `vectors` extra and preserve provider, model, dimension,
and text-preparation provenance; a different provider/model requires a
compatible rebuilt index.

The native embedding adapters are separate from generation where their IDs end
in `-embeddings`. Gemini embedding currently accepts only the reviewed model
IDs `gemini-embedding-001` and `gemini-embedding-2`; other IDs fail closed.
Jina uses distinct retrieval-side inputs only for documented v3–v5 families;
other IDs use symmetric inputs. Cohere and Voyage use their own document/query
input modes. These adapter details still do not guarantee that a selected
account or endpoint accepts a particular model. See [Models and privacy in the
README](../README.md#models-and-privacy) for the current provider-specific
embedding behavior.

Reranking is a separate opt-in operation, not a generation preset. Store a
credential under `jina-reranker` or `cohere-reranker`, then provide the provider,
alias, and model explicitly on an evidence review. The reranker sees only the
query and candidate passages supplied after local scope validation; it does
not retrieve or expand corpus scope. Returned relevance scores are not
probabilities.

```bash
discourselens providers add-credential research jina-reranker university \
  --secret-reference JINA_API_KEY
discourselens analyze evidence-review research FINDING_ID review \
  --reranking-provider jina-reranker \
  --reranking-credential university \
  --reranking-model jina-reranker-v3.5 \
  --context-evidence-id CONTEXT_EVIDENCE_ID
```

The optional `--reranking-top-n` limits passages passed to the review. Reranking
does not convert a targeted candidate set into exhaustive corpus coverage.
Cohere reranking is capped at 1,000 supplied candidates and may truncate long
documents at its configured per-document limit.

## Offline mode and privacy

Enable persistent offline mode with the CLI's documented boolean option:

```bash
discourselens project offline research --enabled
```

This blocks non-loopback remote model generation, embeddings, and reranking;
deterministic local corpus operations remain available. An explicit loopback
endpoint can be used locally subject to the endpoint/proxy checks above. Offline
mode is an execution guard, not a substitute for reviewing the destination.

Any remote generation, embedding, or reranking can send selected prompts,
paragraphs, queries, or candidate passages outside the project. Preview the
destination and content window first, and use only the material appropriate for
that service. Model execution audit records keep hashes and metadata rather than
prompt/completion text or secret values. Generation records also retain the
effective endpoint used for that request; userinfo, query/fragment data, and
credential-like endpoint material are rejected. Historical rows from before
this field was added retain a null endpoint. Cost-only projections omit it. This
storage behavior does not
prevent the remote provider from receiving the request. See [CLI Reference:
provider and model setup](CLI_REFERENCE.md#provider-and-model-setup), [CLI
Reference: Python and MCP](CLI_REFERENCE.md#python-and-mcp), the
[README's MCP section](../README.md#mcp), and [MCP Setup](MCP_SETUP.md).
