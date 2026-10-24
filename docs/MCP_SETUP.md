# MCP setup

DiscourseLens supports MCP over standard input/output (stdio), the default
transport for local clients, and optional read-only Streamable HTTP at `/mcp`.
For stdio, keep stdout reserved for protocol messages; `--verbose` and
`--debug` diagnostics are written to stderr. Do not allocate a pseudo-terminal
for a host-managed stdio session.

The browser workspace and MCP adapter are separate services. Starting the web
app on `127.0.0.1` does not start a public MCP endpoint or expose the web login.
The browser's local accounts are not used for MCP authorization; point each MCP
server process at an explicit DiscourseLens project directory.

## Choose a connection

- **Claude Code on the same computer:** use stdio. The client starts the MCP
  process locally; no HTTP listener or tunnel is needed. Follow [Local MCP
  client](#local-mcp-client).
- **ChatGPT with a private local MCP server:** use [OpenAI Secure MCP Tunnel
  with Docker](#chatgpt-secure-mcp-tunnel-with-docker). The tunnel client makes
  an outbound connection, so neither the MCP HTTP service nor the browser app
  needs a public URL.
- **A client that requires a public HTTPS MCP URL:** use the optional
  [Cloudflare Quick Tunnel](#temporary-cloudflare-quick-tunnel). It exposes
  only the read-only MCP service, uses a bearer token, and gets a temporary
  hostname. For a stable OAuth connector with per-user authorization, see the
  advanced [Cloudflare OAuth bridge](#cloudflare-worker-oauth-bridge-advanced).

ChatGPT custom MCP connections are currently managed from the web Plugins
experience, and availability and permissions depend on the account and
workspace. Check [OpenAI's current setup and availability
guide](https://help.openai.com/en/articles/12584461-connecting-custom-mcp-servers-in-chatgpt).

## Local MCP client

From the repository checkout root, install the MCP extra and initialize a
project. Replace both absolute paths with directories on your computer:

```sh
uv sync --extra mcp
uv run discourselens project init /absolute/path/to/research-project
```

Register the stdio server with Claude Code. `--scope user` stores this setup in
your local user configuration; omit it to use the default scope. See the
[Claude Code MCP guide](https://code.claude.com/docs/en/mcp) for other scopes
and configuration options:

```sh
claude mcp add --scope user --transport stdio discourselens -- \
  uv run --project /absolute/path/to/discourselens discourselens-mcp \
  --project /absolute/path/to/research-project
claude mcp list
```

The client starts and stops this process with its session. Add `--read-only`
after the project path to restrict the session to the read-only tool set. To
connect another local MCP client, use the same `command` and `args` shape in
its documented stdio configuration. The MCP process reads and writes the
project path supplied above; it does not inherit a browser account session.

## Remote HTTP deployment

Configure `DISCOURSELENS_MCP_OAUTH_ISSUER_URL`,
`DISCOURSELENS_MCP_OAUTH_RESOURCE_URL`, `DISCOURSELENS_MCP_OAUTH_AUDIENCE`, and
`DISCOURSELENS_MCP_OAUTH_JWKS_URL` in the MCP container. The issuer must mint
RS256 access tokens containing `iss`, `sub`, `client_id` (or `azp`), `aud`,
`iat`, `exp`, and `scope`; the audience must match the configured value and the
JWKS URL must publish the signing key. The MCP server verifies those tokens but
does not host user accounts, login, consent, token issuance, refresh, or an
authorization server. The external authorization server must support the MCP
client's OAuth flow, PKCE, and authorization metadata discovery; it must
support dynamic client registration or let the operator configure client
credentials accepted by the client. The access token must contain the claims
and scope listed above. Put TLS and a public HTTPS hostname in front of the
container, and set the resource URL to the public `/mcp` endpoint. Test the
full browser authorization flow with the intended client before sharing the
connector; token verifier unit tests do not test an identity provider's login
or registration flow.

The default deployment profile is read-only. To grant the intended evaluator
write access to an isolated project, issue that user's token the
`discourselens:write` scope and start the overlay:

```bash
export DISCOURSELENS_PROJECT_DIR="/absolute/path/to/research-project"
docker compose -f compose.mcp-http.yaml -f compose.mcp-http-write.yaml up --build -d
```

The overlay makes the project mount writable and exposes the full MCP tool
registry. Treat the write scope as permission to modify the mounted project,
including importing documents, creating evidence and annotations, and running
analysis. Do not issue it to an untrusted or shared audience. The default
profile remains a read-only, read-only-mounted container and uses the
`discourselens:read` scope.

These examples use generic `command` and `args` fields. MCP hosts differ in
their configuration file and environment-variable handling; put the fields
into the host's documented MCP server configuration rather than assuming a
specific host format.

### ChatGPT Secure MCP Tunnel with Docker

For a private read-only evaluation, the Compose overlay starts the MCP server
and OpenAI's official `tunnel-client` sidecar without a public hostname. Create
the project and two secret files. The runtime key comes from OpenAI Platform;
the bearer token protects only the internal MCP HTTP hop between the sidecar
and DiscourseLens:

```bash
export DISCOURSELENS_PROJECT_DIR="/absolute/path/to/research-project"
export DISCOURSELENS_UID="$(id -u)"
export DISCOURSELENS_GID="$(id -g)"
export DISCOURSELENS_OPENAI_TUNNEL_ID="tunnel_0123456789abcdef0123456789abcdef" # replace with your tunnel ID
mkdir -p ./secrets
umask 077
openssl rand -hex 32 > ./secrets/mcp-http-token
install -m 600 /dev/null ./secrets/openai-runtime-api-key
# Open this file in a protected editor and save the OpenAI runtime key there.
export DISCOURSELENS_MCP_HTTP_TOKEN_FILE="$PWD/secrets/mcp-http-token"
export DISCOURSELENS_OPENAI_RUNTIME_KEY_FILE="$PWD/secrets/openai-runtime-api-key"
docker compose \
  -f compose.mcp-http.yaml \
  -f compose.chatgpt-tunnel.yaml \
  --profile chatgpt-tunnel up --build -d
curl --fail "http://127.0.0.1:${DISCOURSELENS_TUNNEL_HEALTH_PORT:-8081}/readyz"
```

Create the tunnel in OpenAI Platform and associate it with the target ChatGPT
workspace before starting Compose. Associate both the Platform organization
that owns the tunnel and the ChatGPT workspace that will use it. The sidecar
reads the runtime key and MCP bearer token from mounted secret files; ensure
both files are readable by the UID/GID configured above. Running the tunnel
client and selecting a tunnel requires the Platform **Tunnels Read + Use**
permission; creating or editing the tunnel requires **Tunnels Read + Manage**.
ChatGPT custom MCP access is controlled separately by workspace permissions.
See the [OpenAI Secure MCP Tunnel guide](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
for the current tunnel setup and permission requirements.

In ChatGPT, open **Plugins**, select **+ → Add custom MCP server**, enter a name,
choose **Tunnel** for Connection, then select or enter the tunnel ID. Choose no
additional server authentication when prompted: the bearer token is sent only
inside Docker and should not be pasted into ChatGPT. Review the warning and
create the plugin. The tunnel client must stay running for tool discovery and
calls. Current workspace permissions and plan availability are listed in the
[OpenAI setup guide](https://help.openai.com/en/articles/12584461-connecting-custom-mcp-servers-in-chatgpt).

The MCP service mounts the project read-only and exposes its read-only tool
set. The web app is not part of this Compose command and remains local. The
[official OpenAI tunnel-client image](https://github.com/openai/tunnel-client/pkgs/container/tunnel-client)
is pinned by version and digest in Compose; review its release notes and update
both values deliberately when changing versions.

## Temporary Cloudflare Quick Tunnel

Use this for a short-lived remote test when a client needs a normal HTTPS MCP
URL. It exposes the read-only MCP service on a temporary `trycloudflare.com`
hostname; it does not expose the browser workspace. The MCP bearer token is
required by the server and should be entered only in the remote client's
authentication settings. A Quick Tunnel hostname changes when the tunnel is
restarted, so it is not a stable deployment URL.

```sh
export DISCOURSELENS_PROJECT_DIR="/absolute/path/to/research-project"
export DISCOURSELENS_MCP_HTTP_TOKEN="$(openssl rand -hex 32)"
export DISCOURSELENS_UID="$(id -u)"
export DISCOURSELENS_GID="$(id -g)"
docker compose \
  -f compose.mcp-http.yaml \
  -f compose.mcp-cloudflare-preview.yaml \
  --profile mcp-cloudflare-preview up --build -d
docker compose \
  -f compose.mcp-http.yaml \
  -f compose.mcp-cloudflare-preview.yaml \
  logs -f cloudflare-mcp-origin
```

Copy the generated HTTPS hostname and append `/mcp` for the connector URL. In
the connector, choose bearer-token authentication and enter the value of
`DISCOURSELENS_MCP_HTTP_TOKEN`. Do not use `http://127.0.0.1:8000` or route the
web port through `cloudflared`. Stop the MCP tunnel after the test:

```sh
docker compose \
  -f compose.mcp-http.yaml \
  -f compose.mcp-cloudflare-preview.yaml \
  --profile mcp-cloudflare-preview down
```

For a stable public hostname, configure a named Cloudflare Tunnel whose private
origin is the MCP container on port 8001. Keep the bearer requirement enabled
and keep the origin port unpublished. The advanced OAuth bridge below adds
per-user connector authorization; a Cloudflare Tunnel by itself does not add
user accounts or access control.

## Prepare a project

Install the MCP extra in a native environment, for example from a checkout
with `uv`:

```sh
uv sync --extra mcp
```

Create or choose a project directory. Use an absolute path and keep it
available to the MCP process:

```sh
export DISCOURSELENS_PROJECT_DIR="/absolute/path/to/research-project"
discourselens project init "$DISCOURSELENS_PROJECT_DIR"
```

The directory must be an existing DiscourseLens project when the server starts.
`--project` is required. An optional `--import-root /absolute/path` limits MCP
document imports to that directory; if omitted, imports are limited to the
project root.

Writable MCP file import and PDF reprocessing require operating-system support
for descriptor-relative, no-follow path traversal. The server pins the opened
file before opening the project and reads from that handle under the project's
size limit; it never validates a pathname and then reopens it. If those secure
primitives are unavailable (including on Windows in the current MCP adapter),
file import/reprocessing fails closed with an explicit unsupported-platform
error. This restriction applies only to MCP path-based file access: native
Python SDK and CLI imports remain available through their own local file APIs.
Managed research-bundle export, chunk reads, and verification use the same secure
descriptor-relative primitives and likewise fail closed where they are absent.

## Build local parsed-token artifacts

The writable `build_parsed_token_artifact` tool requires one explicit
`revision_id`, `backend`, and portable `model` ID. The revision's document must
have researcher-declared `language` metadata. A spaCy model must be an installed
spaCy package; arbitrary directories are not loaded. Stanza uses the fixed
server setting `DISCOURSELENS_STANZA_MODEL_DIR` for local resources; the MCP
caller cannot select an environment variable or provide its path/value.
spaCy and Stanza packages must resolve under this interpreter's approved
`site-packages`; `PYTHONPATH` shadows and editable installs resolving outside
those roots are rejected.
Loading a selected spaCy model imports its locally installed Python package.
The `spacy_models` entry-point check validates model kind and packaging; it
does not establish code safety or publisher authenticity. Install and run only
trusted model packages. Use operating-system or container isolation when
working with untrusted model environments.
Install the local extras with `uv sync --extra mcp --extra nlp --extra corpus`.
Parsing is local and makes no network calls or downloads, including in project
offline mode. This write tool is unavailable in `--read-only` sessions.
The resulting artifact stores each token's surface text as well as annotations;
treat it as sensitive corpus data when copying, backing up, or exporting it.

## Read parsed-token frequency in a frozen scope

The read-only `corpus_parsed_frequency_for_scope(scope_id, artifact_ids, field,
offset=0, limit=100)` tool reads an existing immutable `ScopeSnapshot` directly;
it does not create a snapshot or mutate project data. Provide exactly one
compatible parsed-token artifact for every revision in the scope. The selected
artifacts must share the required parser/model/schema identity and may not
include revisions outside the scope.

`field` is one of `lemma`, `pos`, or `dependency_relation`: these count stored
parser lemmas, part-of-speech tags, or dependency labels, respectively. This is
a deterministic summary of local annotations, not a discourse interpretation
or a new parsing operation. The page defaults to 100 values and accepts up to
1,000; it includes scope/artifact provenance, coverage metadata, `total_values`,
and `has_more`. The complete serialized JSON result is capped at 64 KiB.

The revision-list tool `corpus_parsed_frequency(revision_ids, artifact_ids,
field, offset=0, limit=100)` remains distinct: it creates a scope from the
supplied revision selection and is not exposed in `--read-only` mode.

## Read dependency-pattern candidates in a frozen scope

The read-only tool
`corpus_dependency_patterns_for_scope(scope_id, artifact_ids, relation=None,
head_lemma=None, dependent_lemma=None, offset=0, limit=100)` reads an existing
`ScopeSnapshot` directly. It does not create a snapshot, change membership, or
include documents imported later. Supply exactly one compatible `parsed_tokens`
artifact for every revision in the scope; the artifacts must agree on
parser/model/schema identity. The tool rejects unknown, out-of-scope,
incomplete, or incompatible artifact selections. Provide 1..10,000 unique
artifact IDs, each at most 128 characters.

Optional `relation`, `head_lemma`, and `dependent_lemma` filters restrict the
dependency relation and head/dependent parser lemmas. The relation comparison
is exact; lemma comparisons are case-insensitive. `offset` is zero-based and
the bounded page accepts `limit` 1..1,000 (default 100). Results include rows
with `head_lemma`, `relation`, `dependent_lemma`, occurrence `frequency`, and
distinct-revision `document_frequency`, along with scope/artifact provenance,
filters, and parser coverage. The complete serialized response may not exceed
64 KiB; reduce `limit` if the tool rejects an oversized page. Each optional
filter is nonblank and at most 256 characters; `offset` is bounded to the
non-negative signed 64-bit range.

These are syntactic parser candidates, not discourse interpretations or
conclusions about agency, responsibility, intention, or ideology. Annotation
coverage may mark `coverage.patterns` as `LIMITED` when lemma or dependency-parse
annotations are incomplete; a restricted filter/page or limited parser output
cannot establish corpus-wide absence. The existing
`corpus_dependency_patterns(revision_ids, artifact_ids, relation=None,
head_lemma=None, dependent_lemma=None, limit=100, offset=0)` tool is separate:
it creates a scope from the supplied revision list and is unavailable in
`--read-only` sessions.

## Deterministic textual descriptors

The writable adapter exposes `run_textual_descriptors`. It requires
`revision_ids`, an explicit list of 1–100 unique revision IDs. One call creates
one aggregate analysis run over that selected scope; it does not create
independent per-revision runs. Optional `languages` is a list containing `en`,
`es`, or both. If omitted, the operation uses both declared marker inventories.

This is a deterministic scan for the declared textual marker inventory; it
does not interpret a marker's function, infer author intention, or call an LLM.
A result with no detections means only that no marker matched the selected
inventory and language settings in the supplied scope. It does not establish
that modality or the phenomenon is absent.

## Evidence-grounded research questions

The writable `ask_question` tool answers one question using a registered
interpretive framework and an explicitly supplied set of document revisions.
It requires a saved model preset, creates candidate and answer run records, and
is not available in `--read-only` mode. It is not unrestricted chat: only
retrieved evidence from the requested scope is sent in the model request, and
the answer is stored with validated evidence IDs and Core-rendered citations.

Retrieval is targeted. If it finds no candidates, the tool returns that state
without calling the model; this means only that no examples were found under
the retrieval plan, not that the phenomenon is absent. Review the configured
provider/preset and the selected document scope before invoking the tool.
`retrieval_mode` defaults to `lexical`. To use `hybrid`, also supply one saved
`embedding_preset_name` and 1–100 unique, explicit `embedding_artifact_ids`;
these identify the embedding index artifacts used for retrieval. Embedding
options are rejected in lexical mode. In hybrid mode, the question may be sent
to the provider configured by the embedding preset, and retrieved passages may
be sent to the answer provider configured by `preset_name`. Check both provider
destinations and the project's offline setting first. Credential secrets are
never tool arguments. `ask_question` is unavailable in `--read-only` mode for
both retrieval modes.

## Import local documents

The writable server provides `import_document` for one file and
`import_documents` for an ordered batch of 1..100 paths. Each batch path is
resolved and checked against the configured import root independently before
Core reads it. Out-of-root paths produce an item-level `FAILED` result with
`OUTSIDE_IMPORT_ROOT`; other files can still succeed, be reported as duplicates,
or fail independently. Results include each input's zero-based index, status,
document and revision IDs, failure code, and warnings, plus aggregate counts.
They omit source paths and raw exception messages. Import tools are unavailable
in `--read-only` mode.

The writable server also provides `reprocess_document(document_id, path, ...)`
for explicitly creating a new PDF revision for an existing document. The
required `document_id` selects the existing document; `path` supplies the PDF
to extract. Optional arguments are `extractor_suffix` (default
`.reprocessed`), `max_bytes`, `ocr_language`, and `ocr_dpi` (default `300`). The
path follows the same resolution and import-root checks as `import_document`.
The result contains the document ID, new revision ID, status, and warnings, but
never the source path. This tool is unavailable in `--read-only` mode.

## Evidence-bound built-in skills

The writable server exposes two additional, explicitly scoped deterministic
operations. Both create ordinary immutable analysis/skill runs; neither loads
project-supplied code or calls a model.

`run_quote_attribution_candidates` requires 1–100 unique `revision_ids`.
Optional `language` is limited to `en` or `es` and explicitly assigns that cue
language to every selected revision. Without it, the operation inventories
paired quote marks without looking for language-specific attribution cues. Its
candidate records do not establish quotation authenticity, speaker identity,
endorsement, influence, or author intent.

`prepare_van_dijk_group_comparison` requires the same bounded explicit
`revision_ids`, exactly two researcher-labeled groups, `shared_topic_event`,
`comparability_basis`, and one `codebook_id`. Each group supplies its
researcher-selected `annotation_ids`. The Core verifies every annotation,
codebook, and span against the original scope. Counts describe only those
selected annotation records; an unannotated group remains `UNKNOWN`. The tool
does not infer ideology, political identity, corpus prevalence, absence, or an
asymmetry score.

Both tools return persisted IDs and evidence provenance. They do not broaden the
scope to revisions imported later. They are omitted in `--read-only` mode.

The writable adapter also exposes the deterministic local tools
`run_social_actor_candidates` and `run_agency_candidates`. Each accepts explicit
`revision_ids` and `artifact_ids`; the selected artifacts must be `parsed_tokens`
artifacts belonging to the frozen revision scope, with at most one artifact per
revision. Supply artifact IDs created by the local NLP workflow. The tools do not
build or select artifacts implicitly and do not call an LLM.

`run_social_actor_candidates` records parser-tagged noun, proper-noun, and
pronoun expressions as social-actor candidates. `run_agency_candidates` records
dependency-relation candidates linking a dependent expression to its predicate;
it runs only when `dependency_parse` is `SUPPORTED` for the selected parser
artifact. If that capability check is missing, mismatched, or has another
status, the tool rejects the operation. These outputs are evidence-linked
observations only: they do not classify social or discursive representation,
interpret agency or responsibility, or infer author intention.

## Native stdio

For a native installation, configure the MCP host to start the executable and
pass the project path as an argument:

```json
{
  "command": "/absolute/path/to/venv/bin/discourselens-mcp",
  "args": ["--project", "/absolute/path/to/research-project"]
}
```

The host should own the process lifetime and communicate over its stdin/stdout.
For a read-only session, add `"--read-only"` to `args`:

```json
["--project", "/absolute/path/to/research-project", "--read-only"]
```

`--read-only` opens the project without migrations or writes, restricts the MCP
tool list to read-only tools, and removes prompts. It is distinct from the
container's filesystem `--read-only` option described below.

The read-only `list_analysis_runs(offset=0, limit=100)` tool returns a bounded
page of secret-free AnalysisRun metadata. `offset` is non-negative and `limit`
is 1..1000. The response has an `items` array and `offset`, `limit`, `total`,
and `has_more` page metadata.

The read-only `list_annotations(revision_ids, offset=0, limit=100)` tool reads
only the explicitly supplied, existing, unique revision IDs; it does not create
a new scope snapshot. It returns an `items` page plus `revision_count`, `offset`,
`limit`, `total`, and `has_more` metadata (`limit` is 1..100). Annotation text
is capped at 4,000 characters per string; `truncated_fields` identifies any
values that were shortened.

The read-only `list_codebooks(offset=0, limit=100)` and
`list_codes(codebook_id, offset=0, limit=100)` tools page through codebooks and
all codes in one codebook, including unused codes. Code records retain
`parent_code_id`; result strings are bounded and truncation is reported in
`truncated_fields`. Both tools are discoverable in restricted `--read-only`
sessions; limits are 1..100.

Scope creation and corpus/subcorpus snapshot tools return a bounded preview of
revision IDs, together with the full `revision_count` and a truncation flag.
Use the read-only `get_scope_revisions(scope_id, offset=0, limit=100)` tool to
page through the complete immutable membership when needed. It returns
`revision_ids`, `revision_count`, `offset`, `limit`, and `has_more`; pages are
bounded to 1..100 IDs and this reader is available in `--read-only` sessions.

The read-only `corpus_frequency_for_scope(scope_id, offset=0,
limit=100)` and `corpus_kwic_for_scope(scope_id, query, window=5, offset=0,
limit=100)` tools read an existing immutable `ScopeSnapshot`. They do not create
a snapshot or write project data, and later imports or membership changes do
not broaden the frozen scope. Both return bounded pages with `offset`, `limit`,
and `has_more`; `limit` is 1..1000. Frequency pages include selected token
counts and full-scope type and token totals. KWIC pages include the full hit
count. Combined KWIC `left`, `keyword`, and `right` text is capped at 64 KiB
UTF-8 per response. Shortened rows set `text_truncated: true`, and the page sets
that flag if any row was shortened; canonical hit offsets and pagination
metadata remain intact.

For derived surface-token rankings, the read-only
`corpus_normalized_frequency_for_scope(scope_id, artifact_ids, offset=0,
limit=100)` and `corpus_document_frequency_for_scope(scope_id, artifact_ids,
offset=0, limit=100)` tools read the
existing `ScopeSnapshot` without creating a new snapshot or writing project
data. Supply one compatible `surface_tokens` artifact ID for every revision
in the scope; the IDs must be unique and the complete set must use the
supported surface-token schema. Both tools return a bounded page of `{token,
count}` rows plus `scope_id`, `revision_ids`, artifact and tokenizer provenance,
`field`, `offset`, `limit`, `total_values`, and `has_more`. Page size is 1..1000
and the serialized JSON response is capped at 64 KiB. Normalized frequency
counts token occurrences; document frequency counts distinct scoped revisions
containing each normalized token. The revision-list tools
`corpus_normalized_frequency(revision_ids, artifact_ids, offset=0, limit=100)`
and `corpus_document_frequency(revision_ids, artifact_ids, offset=0, limit=100)`
remain available for a fresh explicit revision selection; they create a
snapshot for that selection and therefore are not the existing-scope read-only
forms.

The read-only `corpus_collocations_for_scope(scope_id, query, window=5,
offset=0, limit=100)` tool counts window collocates for one surface-token query
inside an existing `ScopeSnapshot`. It does not create or expand a snapshot;
later imports do not enter the selected scope, and it writes no project data.
It is available in `--read-only` sessions. `window` is 1..100, `offset` is
non-negative, and `limit` is 1..1000. The page includes `rows`, `total_values`,
`offset`, `limit`, and `has_more`. Rows sort by descending count and then token
text. The existing `corpus_collocations(revision_id, query, window=5,
offset=0, limit=100)` tool remains the one-revision form.

To persist one of those scoped KWIC hits as textual evidence, use the
write-enabled `corpus_kwic_evidence_for_scope(scope_id, revision_id, query,
start_offset, end_offset)` tool in a normal MCP session. Copy the `revision_id`
and canonical offsets from the selected `corpus_kwic_for_scope` row and supply
the same query and scope ID. It rejects revisions outside that frozen scope and
offsets that do not identify a matching hit. The operation does not create or
expand a snapshot; it is unavailable in `--read-only` sessions.
The existing `corpus_kwic_evidence(revision_id, query, start_offset,
end_offset)` tool remains the single-revision form and creates a one-revision
snapshot for validation.

To persist an exhaustive deterministic surface-token count over an existing
scope, use the write-enabled `corpus_frequency_evidence_for_scope(scope_id,
term)` tool in a normal MCP session. It reads the existing immutable
`ScopeSnapshot` directly and does not create or change its membership. The
`term` must be one surface token. The operation records
`CorpusStatisticEvidence` with exhaustive coverage described by a
`CoverageDescriptor`; it is unavailable in `--read-only` sessions. The
revision-only `corpus_frequency_evidence(revision_id, term)` tool remains
available in writable sessions and creates a one-revision scope.

The read-only `search_exact_for_scope(scope_id, phrase, offset=0, limit=100)`
tool searches for a literal phrase across the revisions already in an existing
`ScopeSnapshot`. It does not create a snapshot, expand its membership, or write
project data. The result includes matching paragraph rows, `total_count` across
the frozen scope, `offset`, `limit`, and `has_more`; `limit` is 1..1000. This is
separate from `search_exact(revision_id, phrase, offset=0, limit=100)`, which
searches one explicitly named revision and creates a one-revision snapshot for
its result.

The read-only `list_corpora(offset=0, limit=100)` and
`list_subcorpora(corpus_id=None, offset=0, limit=100)` tools return bounded
pages of collection metadata and membership counts. Both use an `items` array
with `offset`, `limit`, `total`, and `has_more` page metadata (`limit` is
1..100). `list_subcorpora` retains its optional parent-corpus filter. These
tools read existing collections without creating snapshots.

The read-only `list_documents(offset=0, limit=100)` tool returns a page of
document metadata with `items`, `offset`, `limit`, `total`, and `has_more`
fields (`limit` is 1..100). Items include document ID/display name/creation
time, revision count, and latest-revision ID, creation time, extraction status,
and MIME type. Canonical text and absolute paths are never returned.

The read-only `get_document_revision(revision_id)` tool returns metadata for
one existing immutable revision: its IDs and document display name, source and
canonical hashes, extractor and MIME type, source size, creation time, OCR
settings, and page/paragraph counts. It never returns canonical document text
or absolute paths.

## Semantic retrieval from an existing local index

The read-only `list_semantic_indexes(scope_id, offset=0, limit=100)` tool lists
verified local vector indexes for the revisions in an existing frozen scope.
The scope must contain 1..100 revisions. It returns artifact IDs, revision IDs,
document names, provider/model identity, and bounded pagination; it does not
read secrets or send text to a provider.

The read-only `search_semantic_passages(scope_id, artifact_ids, preset_name,
query_text, limit=5)` tool embeds a bounded query using the explicitly named
saved preset, searches only the selected compatible indexes in that frozen
scope, and returns at most five cited passages. Provide exactly one compatible
index for each revision in the scope. The query is sent to the
configured embedding endpoint. Retrieved passages and vectors remain in the
project, but this read-only operation does not persist a search run or create
spans. It reports `TOP_K_RETRIEVAL_NOT_EXHAUSTIVE` and
`NOT_PERSISTED_READ_ONLY`; use the normal writable CLI/web search when a
persistent retrieval audit record is required. Project offline mode blocks
remote endpoints.

The project owner must build indexes before connecting a read-only evaluator.
In the web workspace, build the selected revision's index, copy its full index
ID from the status message, then run one semantic search and copy the full
scope ID shown in its result status. In the connected ChatGPT or Claude
conversation, call `list_semantic_indexes` with that scope ID and then
`search_semantic_passages` with the matching artifact IDs and the same saved
preset name. The frozen scope keeps later imports outside that retrieval. No
index is built automatically on import, and the read-only MCP profile cannot
create one.

The read-only `get_finding_reviews(finding_id, offset=0, limit=100)` tool reads
the ordered human review history for one finding. It returns a `reviews` page
and `finding_id`, `offset`, `limit`, `total_count`, and `has_more` metadata
(`limit` is 1..100). Review notes are bounded in MCP responses; `truncated_fields`
identifies any shortened strings. This tool does not create or modify reviews.

In writable sessions, `export_research_bundle(analysis_run_id,
include_canonical_text=false)` exports a secret-screened portable bundle into
project-managed storage and returns an opaque `bundle_id`, byte size, and
SHA-256 digest. Canonical text for revisions referenced by cited textual
evidence is excluded by default. Set `include_canonical_text=true` only when the
bundle should carry full canonical text; it may be sensitive. Cited evidence
excerpts remain in the default bundle and can be long or even cover an entire
revision if that was the saved evidence span. Review them as well. The full
manifest, including optional text, is bounded by the 64 MiB manifest limit. It
never accepts or returns a filesystem path.
`get_research_bundle_chunk(bundle_id, offset=0, max_bytes=24576)`
transfers the manifest as base64 in bounded chunks of at most 32 KiB;
`verify_research_bundle(bundle_id)` verifies the stored artifact using the
existing bundle verifier. Bundle IDs are random, scoped to the configured
project, and cannot select arbitrary files. Export is unavailable in
`--read-only` sessions; chunk retrieval and verification are available there.
The existing 64 MiB manifest ceiling applies. Verification detects corruption
but does not prove authorship. Managed bundle export, chunk reads, and
verification, like MCP path-based imports, fail closed when descriptor-relative
no-follow access is unavailable (including on Windows with the current adapter).

The `search_lexical` tool is available in both normal and `--read-only` MCP
sessions. Supply an existing `scope_id` and an FTS5 query; the response is a
structured page of at most 100 paragraphs with revision/document IDs and
paragraph offsets. Each request examines at most 5,000 candidates. `has_more`
reports only additional rows inside that window; `candidate_window_full` warns
when later matches may remain unseen. Offsets through 4,999 are valid, including
the last candidate. Paragraph text is capped to 64 KiB of UTF-8 per response;
rows shortened to fit carry `text_truncated: true`, while canonical offsets and
provenance remain unchanged. Use the bounded page/paragraph readers for a
selected source unit rather than treating a search result's text field as
complete when that flag is true.

The `corpus_kwic` tool returns a paged set of hits for one explicit revision.
The combined `left`, `keyword`, and `right` text is capped to 64 KiB of UTF-8
per response. A shortened row is marked `text_truncated: true`, and the page
sets the same flag if any row was shortened; canonical hit offsets and
pagination metadata are preserved.

The read-only `corpus_keyness` tool compares two existing `ScopeSnapshot` IDs
without persisting a comparison run. It accepts `left_scope_id`,
`right_scope_id`, and bounded `offset`/`limit` arguments, and returns the page
along with each scope's revisions and surface-token population, exhaustive
coverage details, and the keyness calculation version.

The read-only `corpus_compare_frequency` tool accepts the same two frozen scope
IDs and pagination arguments. It returns normalized surface-token rates for the
requested page, each scope's revision/token population, and exhaustive coverage
metadata; it does not persist a comparison run. Use it for rate differences;
use `corpus_keyness` when the requested measure is signed log-likelihood.
`compare_documents` is a separate persistent workflow for comparing two
individual revisions.

In normal (writable) MCP mode, `corpus_normalized_frequency` and
`corpus_document_frequency` take explicit `revision_ids`, one surface-token
artifact ID per revision, and bounded `offset`/`limit` arguments. They create a
frozen scope and use the same Core page API as the Python SDK and CLI. These
tools are intentionally not exposed in read-only mode because snapshot creation
persists scope metadata. Pages are rejected rather than truncated if token
values would exceed the MCP response text budget. New artifacts use a
`surface-token-v2-ucd-<version>` schema; pages expose that schema, the tokenizer,
normalization rule, Unicode database version, and calculation version. Legacy
`surface-token-v1` artifacts must be rebuilt before these current frequency
tools can use them.

The read-only `list_model_costs` tool is available in `--read-only` sessions.
It returns a paged view of persisted model executions and their validated price
snapshots; it does not look up current rates or contact a provider. Pass
`offset` and `limit` to continue through the ledger (`offset` defaults to 0,
`limit` to 100, and MCP caps `limit` at 100). The response includes `offset`,
`limit`, `total_count`, `has_more`, and rows with execution/provider/model IDs,
usage, estimated cost/currency, timestamps, error type, and the stored price
snapshot. It omits credential aliases and request/response hashes.

The read-only `list_research_memos(scope_id, offset=0, limit=100)` tool reads
memos attached to one existing frozen scope and is included in the read-only
allowlist. Its page envelope contains `scope_id`, `offset`, `limit`,
`total_count`, `has_more`, and `rows` (`limit` is 1..100). Memo titles and
content are capped at 512 and 4,000 characters; each row includes the original
character counts and `title_truncated`/`content_truncated` flags.

`corpus_ngrams` and `corpus_collocations` also return deterministic bounded
pages. Each requires one explicit revision and accepts `offset`/`limit`; page
metadata reports the distinct-value total and whether further rows remain.
N-grams and collocations use the `unicode-surface-tokenizer-v2` tokenizer and
NFC-after-casefold normalization, matching surface frequency. Lexical bundle
calculations retain their separately versioned v1 tokenizer. Rows are deterministic counts, but these descriptive pages do not create a
`CorpusStatisticEvidence` record; use a suitable citable evidence operation
when the count will support a quantified finding.

The read-only `corpus_ngrams_for_scope(scope_id, size, offset=0, limit=100)`
tool reads an existing immutable `ScopeSnapshot`; it does not create or expand
the scope or write project data. `size` selects the number of surface tokens in
each n-gram, and pages preserve the Core fields `scope_id`, `size`, `offset`,
`limit`, `rows`, `total_values`, and `has_more` (`limit` is 1..1000). The
tool is available in `--read-only` sessions, and the serialized MCP response is
bounded to 64 KiB of UTF-8 JSON. The existing
`corpus_ngrams(revision_id, size, offset=0, limit=100)` tool remains the
single-revision form.

`corpus_collocation_analysis` accepts explicit `revision_ids` plus `offset` and
`limit` to page the full eligible association ranking. Its response includes
the exact `total`, `has_more`, and compatibility `truncated` metadata; every
page preserves the same deterministic t-score, pair-frequency, then lexical
tie order used by Core and the CLI.

To persist one selected pair's exhaustive association evidence over an
already-frozen scope, use the write-enabled
`corpus_collocation_evidence_for_scope(scope_id, query, collocate, window=5,
min_frequency=2)` tool. It consumes the supplied `ScopeSnapshot` unchanged,
does not create a new snapshot, and records deterministic corpus-statistic
evidence with exhaustive coverage/provenance for that population. `query` and
`collocate` must each be one surface token of at most 256 characters; a
self-collocation is not supported. `window` is 1..100 and `min_frequency` is
positive. Mutual information (MI) and t-score are
descriptive association measures, not significance tests or p-values. This
tool is unavailable in `--read-only` sessions. The existing
`corpus_collocation_evidence(revision_ids, query, collocate, window=5,
min_frequency=2)` tool remains the revision-list form and creates a scope from
the supplied revisions.

`corpus_lexical_bundles` also accepts `offset`/`limit` and returns a bounded
page ordered by descending occurrence frequency, then lexical bundle. Its
`candidate_count` remains the full filtered population, while `has_more` and
`truncated` describe whether candidates remain after the current page.

For a saved immutable scope, use the read-only
`corpus_lexical_bundles_for_scope(scope_id, size=3, min_frequency=2,
min_document_frequency=1, offset=0, limit=100)` tool. It reads the existing
`ScopeSnapshot` directly, so it does not create or expand scope membership.
`size` must be 2..10, the frequency filters must be positive, and pages contain
at most 1,000 rows. The response retains the complete bundle strings and full
`candidate_count`; it does not truncate bundle identities. MCP bounds the
serialized page to 64 KiB of UTF-8 JSON. The existing
`corpus_lexical_bundles(revision_ids, size=3, min_frequency=2,
min_document_frequency=1, limit=100, offset=0)` tool takes an explicit revision
selection and creates its scope from those revisions. The CLI counterpart for
the existing-scope read is `corpus lexical-bundles-scope PROJECT SCOPE_ID`;
`corpus lexical-bundles PROJECT REVISION_IDS` remains the revision-selection
command.

To persist an exact bundle count over the existing scope, use the write-enabled
`corpus_lexical_bundle_evidence_for_scope(scope_id, bundle, size)` tool. It
reuses the immutable `ScopeSnapshot` and records `CorpusStatisticEvidence` with
exhaustive `CoverageDescriptor` provenance for every contiguous n-gram position
in that scope; it does not create a fresh snapshot. The supplied bundle must
contain exactly `size` surface tokens (`size` is 2..10). This uses the existing
`surface-lexical-bundles-v1` tokenization and records a descriptive count, not a
CDA interpretation. It is unavailable in `--read-only` sessions. The existing
`corpus_lexical_bundle_evidence(revision_ids, bundle, size)` tool remains the
revision-list form and creates a scope from those revisions. The CLI counterpart
for the existing-scope write is
`corpus lexical-bundle-evidence-scope PROJECT SCOPE_ID BUNDLE --size N`; the
legacy `corpus lexical-bundle-evidence PROJECT REVISION_IDS BUNDLE --size N`
continues to take a revision list.

`corpus_dispersion` returns exhaustive token/document totals with a bounded
page of per-revision rows (`offset`/`limit`, default 100). In the CLI, full
legacy output remains the default and paging is opt-in. The dedicated
`corpus_dispersion_evidence` operation remains exhaustive and citable.

For an existing frozen `ScopeSnapshot`, the read-only
`corpus_dispersion_for_scope(scope_id, term, offset=0, limit=100)` tool reads
that saved membership directly. It does not create or expand a snapshot, and
later imports do not enter the scope. The tool is available in `--read-only`
sessions; `term` must be one surface token, `offset` is non-negative, and
`limit` is 1..1,000. Each response keeps the full-scope occurrence/document
totals and `scope_id`, while `documents` contains only the requested page. The
serialized response is bounded to 64 KiB of UTF-8 JSON. The existing
`corpus_dispersion` tool remains the explicit-revision-list form.

## Docker stdio

Build the image separately from the host-managed MCP session, then pass the
project as an explicit bind mount. The project mount is writable in this
example, while the container root filesystem is read-only. No port is
published; do not add `-p`/`--publish` for stdio.

```sh
# Choose a unique tag per checkout when sharing a Docker engine.
docker build -t discourselens:local-checkout-a .

export DISCOURSELENS_PROJECT_DIR="/absolute/path/to/research-project"
export DISCOURSELENS_UID="$(id -u)"
export DISCOURSELENS_GID="$(id -g)"

docker run --rm -i \
  --read-only --tmpfs /tmp \
  --user "$DISCOURSELENS_UID:$DISCOURSELENS_GID" \
  --mount "type=bind,src=$DISCOURSELENS_PROJECT_DIR,dst=/workspace/project" \
  --network none \
  discourselens:local-checkout-a discourselens-mcp --project /workspace/project
```

`-i` keeps stdin open; omit `-t` so Docker does not allocate a pseudo-terminal.
`--network none` disables container networking entirely. This is appropriate
when no network-backed model or embedding service is needed. If you intentionally
use a remote provider, remove only `--network none`; the process still has no
published inbound port, but it can make outbound requests. A host-local service
at `127.0.0.1` is not automatically reachable as `127.0.0.1` from inside a
container.

The image defaults to UID/GID `10001:10001`; the command above instead runs with
the host user's numeric IDs so the bind-mounted project can be accessed. If
your Docker environment uses different ownership semantics, adjust the user or
project permissions deliberately. The MCP process still needs write access to
the project unless its `--read-only` mode is selected.

The same Docker invocation can be represented as generic host `command` and
`args` fields (replace the project path and image tag):

```json
{
  "command": "docker",
  "args": [
    "run", "--rm", "-i", "--read-only", "--tmpfs", "/tmp",
    "--user", "1000:1000",
    "--mount", "type=bind,src=/absolute/path/to/research-project,dst=/workspace/project",
    "--network", "none",
    "discourselens:local-checkout-a", "discourselens-mcp",
    "--project", "/workspace/project"
  ]
}
```

This is only the command/argument shape: it does not assume a host-specific
configuration schema or environment expansion. The numeric user must be able
to access the mounted project.

### Compose option

The repository's `compose.yaml` has no `ports:` mapping. Its default image tag
is `discourselens:<project-name>`, so separate checkout directories get
separate local tags by default. When multiple checkouts share one Docker engine,
give each a distinct Compose project name with `COMPOSE_PROJECT_NAME` or
`docker compose --project-name dl-checkout-a`. Set `DISCOURSELENS_IMAGE` to
override the tag when needed. The default `mcp` service has a read-only
container filesystem and a writable project bind mount. It has no added
capabilities and uses the default
Compose network, which allows configured remote model or embedding providers.
No inbound port is published. The `mcp-readonly` service is under the `read-only`
profile; it uses `network_mode: none`, a read-only project mount, and the MCP
`--read-only` flag. Choose it when the process must have no network access; the
tradeoff is that remote providers are unreachable (local corpus tools remain
available).

Set the required host project path and match the project owner where needed.
Review the resolved configuration before starting a service:

```sh
export DISCOURSELENS_PROJECT_DIR="/absolute/path/to/research-project"
export DISCOURSELENS_UID="$(id -u)"
export DISCOURSELENS_GID="$(id -g)"
# Optional when the checkout directory name is not unique:
export COMPOSE_PROJECT_NAME="dl-checkout-a"
docker compose config
docker compose run --rm --no-deps -T mcp
```

For the isolated read-only profile:

```sh
docker compose --profile read-only config
docker compose --profile read-only run --rm --no-deps -T mcp-readonly
```

`-T` disables pseudo-TTY allocation; Compose run keeps stdin interactive by
default. These are one-off stdio processes; `docker compose up` is not needed.

## Provider credentials and network safety

Project configuration stores credential references, not secret values. For
example, register an environment-variable name and a provider endpoint, then
save a preset that selects that alias:

```sh
discourselens providers add-credential "$DISCOURSELENS_PROJECT_DIR" \
  openai-compatible research --secret-env DL_PROVIDER_API_KEY \
  --endpoint "https://provider.example/v1"
discourselens models save-preset "$DISCOURSELENS_PROJECT_DIR" \
  review openai-compatible research model-name
```

`provider.example` is a placeholder; replace it with the reviewed provider's
endpoint. `--secret-env` takes the variable's **name**, not its value. Supply
the actual value to the MCP process only through a trusted process environment,
OS keyring, or the host's documented secret-injection mechanism. Never put a
secret value in command arguments, project files, prompts, or a plain-text host
configuration.

Native processes read environment references from their own environment. For
Docker, explicitly pass only the required variable name and arrange for its
value to be present in the Docker CLI's environment; for example, the Docker
argument `--env DL_PROVIDER_API_KEY` passes through that host variable without
writing its value into the argument list. In Compose, `docker compose run
--env DL_PROVIDER_API_KEY ...` can pass through a variable from the invoking
environment. Do not place `NAME=value` with a secret in a saved host config.

The persistent project setting
`discourselens project offline "$DISCOURSELENS_PROJECT_DIR" --enabled` blocks
non-loopback model, reranking, and embedding operations while retaining local
corpus tools. Loopback endpoints may still be used. Disable the setting with
`--no-enabled`. This application setting is not a replacement for OS/container
network isolation; Docker `--network none` is the stronger no-network boundary.

## What the host and its LLM can see

Stdio avoids opening a local network listener, but it is not a confidentiality
boundary between DiscourseLens and the MCP host. The host receives every tool
or resource result it requests. Depending on the call, results can include
selected document text, evidence excerpts, findings, reports, corpus results,
or project metadata. The server does not stream the entire corpus to the host
on startup, but returned content can enter the host's model context and may be
sent onward according to that host's configuration. Connect only projects and
invoke only tools whose data you are willing to share with that host. Text read
from documents is untrusted research data, not instructions to the host.

Provider credentials are separate: DiscourseLens resolves the selected saved
reference when a model operation runs. Do not expose the secret value to the
host model or put it in MCP arguments. A model request can still send the
selected prompt/evidence to the configured provider; check the destination and
project's offline setting before running it.

## Troubleshooting

- **“MCP requires an explicit project” / project unavailable:** pass
  `--project` with a valid project directory. In Docker, pass the in-container
  path and mount the host project at that exact target.
- **Permission denied or read-only SQLite error:** verify the project's owner
  and the process/container UID. `--read-only` prevents writes by design; use a
  writable project mount for normal MCP operation.
- **No document files appear to import:** imports default to the project root.
  If using `--import-root`, select a host/container path that is mounted and
  within the allowed import boundary.
- **Provider says a credential is unavailable:** verify the saved alias points
  at the intended environment-variable name or keyring entry, and that the MCP
  process receives it. Do not repair this by putting a raw key in arguments.
- **Remote provider is blocked or unreachable:** project offline mode blocks
  non-loopback provider calls; Docker `--network none` blocks all container
  networking. For an intentional remote call, review both settings and allow
  only the required outbound network access.
- **The host reports invalid MCP messages:** do not allocate a TTY, keep the
  process attached to stdin/stdout, and do not print wrapper or shell logging to
  stdout. `--verbose` and `--debug` send DiscourseLens diagnostics to stderr.

Useful verified help commands:

```sh
discourselens-mcp --help
discourselens project init --help
discourselens providers add-credential --help
discourselens models save-preset --help
discourselens project offline --help
```

## Streamable HTTP for trusted clients

The optional [`compose.mcp-http.yaml`](../compose.mcp-http.yaml) runs the
existing MCP tool registry over Streamable HTTP in read-only mode, with a
read-only project mount. It binds the published port to `127.0.0.1:8001` and
requires a 32-character-or-longer bearer token:

```sh
export DISCOURSELENS_PROJECT_DIR="/absolute/path/to/research-project"
export DISCOURSELENS_MCP_HTTP_TOKEN="$(openssl rand -hex 32)"
docker compose -f compose.mcp-http.yaml up --build -d
```

The MCP URL on the host is `http://127.0.0.1:8001/mcp`. Clients must send
`Authorization: Bearer $DISCOURSELENS_MCP_HTTP_TOKEN`. To expose it beyond
loopback, place a TLS proxy in front and keep the MCP bearer secret protected.
This static-token transport is useful for compatible MCP clients and local
testing. It does not implement OAuth 2.1, user login, or per-user project
authorization. Use it directly only with clients that support bearer-token
authentication. For private ChatGPT access, use the Secure MCP Tunnel above;
for public OAuth connectors with separate user accounts, use the Cloudflare
bridge below.

## Cloudflare Worker OAuth bridge (advanced)

Use the [Cloudflare MCP OAuth bridge](../deploy/cloudflare-mcp-oauth/README.md)
when you need a conventional public HTTPS connector with OAuth, dynamic client
registration, PKCE, consent, and per-user read-scope enforcement. It uses a
separately configured identity provider. This identity provider is only for
remote MCP connector authorization; it does not replace or control the browser
workspace's local email/password accounts. The Python MCP service stays in
Docker, read-only, behind a private bearer token and Cloudflare Tunnel.

The bridge authorizes an email allowlist, then routes each grant by its verified
OIDC issuer and subject to that person's private MCP project. Its signed
internal assertion binds the authenticated MCP identity to the Docker request;
a missing or invalid assertion is rejected. Browser password accounts do not
share projects with this bridge automatically. The direct external OIDC
resource-server mode below remains a separate deployment option for an operator
whose authorization server issues MCP audience tokens.

## OAuth with an external OIDC identity provider

The MCP adapter can validate RS256 access tokens from an existing OAuth/OIDC
provider. It does not host the identity provider or create user accounts. The
provider must expose a JWKS endpoint, issue tokens with the configured issuer,
the MCP resource in `aud`, and the required `discourselens:read` scope. Configure
an OAuth authorization-code flow with PKCE in that identity provider and use
its supported client-registration method for the ChatGPT/Claude connector.

Set these values in the MCP container environment. The `resource` must be the
public HTTPS MCP URL, including `/mcp`; the issuer must exactly match the JWT
`iss` claim; the audience must appear in the token's `aud` claim; and the JWKS
URL must serve the issuer's signing keys:

```bash
unset DISCOURSELENS_MCP_HTTP_TOKEN
export DISCOURSELENS_MCP_OAUTH_ISSUER_URL="https://login.example.org/realms/discourselens"
export DISCOURSELENS_MCP_OAUTH_RESOURCE_URL="https://research.example.org/mcp"
export DISCOURSELENS_MCP_OAUTH_AUDIENCE="https://research.example.org/mcp"
export DISCOURSELENS_MCP_OAUTH_JWKS_URL="https://login.example.org/realms/discourselens/protocol/openid-connect/certs"
export DISCOURSELENS_MCP_OAUTH_REQUIRED_SCOPES="discourselens:read"
docker compose -f compose.mcp-http.yaml up --build -d
```

Do not set the static MCP bearer token with OAuth. OAuth mode requires
`--read-only` and Streamable HTTP; it exposes only the explicitly allowlisted
read tools. The MCP SDK publishes protected-resource metadata for the URL.
Keep the host port on loopback and route public HTTPS through a TLS proxy. The
identity provider itself must be reachable by the user's browser and the MCP
client. The read-only policy is shared by every authenticated account. In direct
OIDC resource-server mode, verified issuer/subject identities resolve to
isolated tenant project directories when `DISCOURSELENS_TENANT_ID_KEY` is
configured. The Cloudflare bridge uses the same mapping through its signed
tenant assertion. This adapter does not implement per-user write scopes. Confirm the
identity provider's resource metadata and client registration with the exact
ChatGPT or Claude connector before exposing the endpoint; DiscourseLens
validates external tokens but does not itself operate the authorization server.
