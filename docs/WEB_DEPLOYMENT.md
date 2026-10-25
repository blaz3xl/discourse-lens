# DiscourseLens local web app

The browser workspace runs the shared Python Core in Docker and keeps one
isolated project per local account. It supports document import, literal
search, corpus statistics, cited analysis, and model-assisted conversations.
Provider credentials are encrypted in each project database. Passwords are
stored as salted scrypt hashes in a private account database.

## Start locally

Run the launcher. It creates a private `.env.local`, generates the session and
encryption keys if needed, builds the web image, and creates the project
folder:

```bash
./scripts/start-local.sh
```

Open `http://127.0.0.1:8000` and register with a username (email address) and
password. Registration is enabled for this local setup, and Docker publishes
the app only on localhost. Account workspaces and credentials persist under
`DISCOURSELENS_PROJECT_DIR`; keep `.env.local` private and stable across
restarts.

For local Ollama embeddings, run `./scripts/start-local.sh --local-embeddings`.
This starts the CPU-limited service and downloads
`embeddinggemma:300m-qat-q4_0` into a Docker volume. Ollama does not configure a
preset inside a new account. In **Proveedores de modelos**, add provider
`ollama`, credential alias `local`, and endpoint `http://ollama:11434/v1`; then
add the embedding preset in **Modelos y presets**.

The upload limit defaults to 100 MiB. Uploaded files are streamed to a
temporary file and copied into the project's managed source storage by the
Core.

## Persistence and retrieval

The bind-mounted `DISCOURSELENS_PROJECT_DIR` is the persistent store. It holds
the project database, canonical document text, imported source files, and
derived artifacts. Docker container replacement does not erase that directory.
Back it up as a whole and keep backups protected like the original corpus.

Import does not create an embedding index automatically. The web workspace can
explicitly build a paragraph index and run semantic retrieval after an operator
configures an embedding preset. Vector files are stored under
`.discourselens/artifacts/vectors` inside the mounted project. Building them
sends paragraph text to the selected embedding provider; later vector search
can run against the local index. The index stays in the project volume, not in
ChatGPT or Claude. Text and excerpts returned by MCP tools can still enter the
connected model's context.

Broad conversational analysis can read the complete canonical document when it
fits the 100,000-character tool limit, retaining page references and explicit
coverage. Longer documents require continuation. Contextual research uses the
selected OpenAI preset's native web search, followed by bounded public-page
fetches; other providers can use direct URL fetching but automatic source
discovery currently requires OpenAI. Search receives public metadata queries.
Fetched excerpts and URLs enter the conversation model's context and are saved
as external evidence with retrieval time and content hash. They remain distinct
from the uploaded document. Project offline mode blocks search and fetching.
Public fetching rejects private addresses, credential-bearing URLs and private
redirect destinations, and pins connections to the validated DNS addresses.

Provider keys can be entered in the browser's **Proveedores de modelos** panel
and are stored encrypted in the project database. Create model presets there
under **Modelos y presets**, or through the CLI. The CLI can still use
environment or OS-keyring references.
For that operator-managed route, make the selected secret environment variable
available to the container through the optional provider env file:

```bash
discourselens providers add-credential "$DISCOURSELENS_PROJECT_DIR" \
  openai university --secret-env DL_OPENAI_KEY
discourselens models save-preset "$DISCOURSELENS_PROJECT_DIR" \
  embed openai university text-embedding-3-small
umask 077
printf '%s\n' 'DL_OPENAI_KEY=replace-with-provider-key' > provider.env
export DISCOURSELENS_PROVIDER_ENV_FILE="$PWD/provider.env"
docker compose -f compose.web.yaml up -d
```

In the browser, select a document revision and the saved preset name, inspect
the provider/endpoint preview, acknowledge paragraph egress, and build the
index. The UI then lists local indexes for that revision. A semantic search
requires a separate acknowledgement because the query is sent to the selected
embedding provider. Retrieval returns cited passages and a run ID; it does not
generate a discourse interpretation or an LLM answer. Lexical search and corpus
statistics remain local.

## Debugging a chat request

Chat tracing is off by default. Set `DISCOURSELENS_DEBUG=on` for one diagnostic
run to log the exact prompt, model response, JSON parse position, selected model
settings, and tool inputs/results. These logs can contain conversation text and
document excerpts, so keep them private and turn tracing off when the diagnosis
is complete. Provider API keys and authorization headers are never logged.

```bash
DISCOURSELENS_DEBUG=on docker compose -f compose.web.yaml up -d --build web
docker compose -f compose.web.yaml logs -f web
```

Afterward, restart with `DISCOURSELENS_DEBUG=off` (or unset the variable) so
normal chat content is not written to diagnostic logs.

### Local embeddings with Ollama

For an evaluation without a paid embedding API, start the CPU-limited Ollama
service and download the compact multilingual embedding model:

```bash
docker network create discourselens-local-embeddings
docker compose -p discourselens-ollama -f compose.ollama.yaml up -d
docker compose -p discourselens-ollama -f compose.ollama.yaml exec ollama \
  ollama pull embeddinggemma:300m-qat-q4_0
```

The quantized model file is about 238 MB and is multilingual. Its 2,048-token
context makes it suitable for normal paragraph-sized PDF text; longer inputs
are rejected by Ollama instead of being silently truncated. The service is
limited to two CPU cores, 2 GiB RAM, a 2,048-token runtime context, and one
loaded model; container swap is disabled. It publishes no host port and keeps
model files in the `discourselens-ollama-models` Docker volume. The Ollama API
is reachable only from the web and MCP containers on a dedicated Docker network.
DiscourseLens uses Ollama's native embedding endpoint with truncation disabled.

The evaluation project uses the `embed-local` preset and `ollama/local`
credential reference. To configure a new project, first create the protected
provider env file from the non-secret template (`DL_OLLAMA_API_KEY=ollama` is a
compatibility value that Ollama ignores), then run:

```bash
cp provider.env.example provider.env
chmod 600 provider.env
discourselens providers add-credential "$DISCOURSELENS_PROJECT_DIR" \
  ollama local --secret-env DL_OLLAMA_API_KEY \
  --endpoint http://ollama:11434/v1
discourselens models save-preset "$DISCOURSELENS_PROJECT_DIR" \
  embed-local ollama local embeddinggemma:300m-qat-q4_0
```

Keep `DISCOURSELENS_PROVIDER_ENV_FILE` pointed to that file and use the
`compose.local-embeddings.yaml` web overlay and
`compose.mcp-local-embeddings.yaml` MCP overlay when starting the Compose stacks
so indexing and query embedding use the same local model. Build
a representative pilot index before importing all PDFs:

```bash
DISCOURSELENS_PROVIDER_ENV_FILE="$PWD/provider.env" \
  docker compose -f compose.web.yaml -f compose.local-embeddings.yaml up -d
DISCOURSELENS_PROVIDER_ENV_FILE="$PWD/provider.env" \
docker compose -p discourselens-evaluation-mcp \
  -f compose.mcp-http.yaml -f compose.mcp-local-embeddings.yaml up -d
```

Before giving an evaluator access, run the isolated end-to-end smoke check from
the web container. It generates a two-page PDF in a temporary project, then
checks authenticated upload, paragraph extraction, real local embedding,
semantic results with citations, and persistence after reopen. It then launches
a loopback-only read-only MCP server against that same temporary index and
checks token rejection, skills discovery, and semantic retrieval with citations
over Streamable HTTP. It also fetches the Van Dijk LLM-assisted skill manifest
and its bundled methodology resource so a connected model can retrieve the
reviewed instructions. It does not modify the evaluation project:

```bash
docker exec -e DISCOURSELENS_RUN_LOCAL_OLLAMA_INTEGRATION=1 \
  discourselens-evaluation-web-1 \
  python /opt/discourselens/scripts/smoke_local_ollama.py
```

The web and HTTP MCP Docker images include local Tesseract OCR data for English,
Spanish, and Dutch by default. OCR is opt-in for each import: when a PDF is
scanned, choose its language and DPI in the upload panel. The native/stdio
Docker image can include Tesseract by setting `DISCOURSELENS_INSTALL_OCR=1`.

These resource limits prevent Ollama from consuming all CPU/RAM, but they do
not guarantee how long a 2,000-page corpus will take. Run a representative pilot
first and measure indexing time. A single PDF import is limited to 1,000 pages,
so a 2,000-page corpus must be split across files. Extracted text remains local;
OCR runs locally on scanned PDFs when explicitly selected and takes longer. Paragraphs
above the model's 2,048-token context fail explicitly and must be split before
indexing. Vectors are stored in the project's local artifact directory, not in
Ollama or a remote database.

A local load check on 2026-10-05 generated and imported two digital PDFs with
1,000 pages each and one short paragraph per page. Import took 12.3 seconds;
indexing 2,000 paragraphs with the bounded Ollama service took 2,057.6 seconds
(34 minutes 18 seconds, 0.97 paragraphs/second). Ollama used about 531 MiB of
its 2 GiB memory limit during the run. This is a measured baseline for that
host and short paragraphs, not a promise for other machines or denser PDFs:
documents with several paragraphs per page require proportionally more
embeddings, and scanned PDFs add OCR time. The benchmark used a temporary
project and did not modify the evaluation project's data.

A later repeat of the same sparse, synthetic workload completed in 507.39
seconds: PDF generation 4.07 seconds, import 1.66 seconds, indexing 501.38
seconds, and semantic query 0.28 seconds. The two indexes produced five cited
hits; citation-render time was not measured separately. Ollama peaked at about
334 MiB and at its two-CPU limit. The large timing difference between local
runs is a reason to measure on the deployment host with representative source
PDFs before promising an indexing time.

## ChatGPT and Claude connectors

The HTTP endpoint in `compose.mcp-http.yaml` is a trusted-client transport,
not an account login service. Startup requires either a high-entropy static
bearer token or external OAuth configuration. The static token is not a
substitute for OAuth in a ChatGPT or Claude account connector flow. Do not
publish the static-token endpoint to the internet as a way to give one person
private access. See the MCP setup guide for the read-only and OAuth-only
write-enabled profiles and the concrete Docker plus Secure MCP Tunnel setup
for a private ChatGPT evaluation without a public hostname.

For ChatGPT, use OpenAI Secure MCP Tunnel with the private HTTP MCP endpoint.
An OpenAI Platform operator must create the tunnel, run `tunnel-client` next to
DiscourseLens with its runtime API key, associate the tunnel with the target
ChatGPT workspace, and grant the user tunnel use and developer-mode access. The
user then creates a ChatGPT app, chooses **Tunnel**, and selects or enters the
tunnel ID. The Docker profile uses a read-only project mount. It supports
semantic retrieval from indexes prepared in advance; the query is sent to the
configured embedding provider and the top passages are returned with citations.
The exact preparation and tool-call flow is in the [MCP setup guide](MCP_SETUP.md#semantic-retrieval-from-an-existing-local-index).

To run the browser workspace alongside a private ChatGPT connector, start the
web service first. Then configure the MCP service and tunnel as described in
the MCP guide:

```bash
docker compose -f compose.web.yaml up --build -d
docker compose \
  -f compose.mcp-http.yaml \
  -f compose.chatgpt-tunnel.yaml \
  --profile chatgpt-tunnel up --build -d
```

The local web account and the MCP connector use separate authentication. Review
the MCP project mount and access policy before enabling a connector so it does
not expose private account workspaces.

For Claude, a custom connector needs a reachable MCP URL and its own
authorization configuration. The local web launcher does not create a public
tunnel. Configure any remote MCP service separately using the [MCP setup
guide](MCP_SETUP.md); its account and project access are independent from the
browser's local password account.

Before inviting an external evaluator, decide who operates the hostname, TLS,
authorization, backups, and embedding-provider credentials. Those hosting
services require separate deployment-specific configuration.

## Remote access

The published host port is bound to `127.0.0.1`. For remote access, route a TLS
reverse proxy to the local service. Configure the proxy to preserve the original
`Host` header, impose its own request-size limit at or below 100 MiB, and set
trusted forwarding headers only from the proxy. Do not publish port 8000
directly to an untrusted network. Use a separate project directory per
deployment; local accounts are isolated at the project boundary.

## Existing interfaces

The native CLI and stdio MCP adapter remain available as before. They can open
the same project directory when their filesystem access and UID/GID permit it.
Model-preset configuration is available in web and CLI. Exact-model capability
review remains explicit; the web does not fetch provider model catalogs.
Conversations use the selected saved preset and its configured provider key.

The chat's balance control uses an account-local, user-entered budget and saved
model prices; it has no payment or provider-wallet integration. Authenticated
`/api/spending` polling is read-only. Budget and price changes use the same
account routing and origin checks as other settings. No billing administration
key is required. See [the spending guide](SPENDING.es.md) for coverage, price
snapshots, streaming usage and unknown-cost behavior.

```bash
uv tool install "discourselens[corpus,vectors,mcp]"
discourselens --help
discourselens-mcp --help
```
