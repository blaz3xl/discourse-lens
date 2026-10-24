# DiscourseLens remote MCP OAuth bridge

This Cloudflare Worker supplies the OAuth 2.1 authorization-server endpoints
that ChatGPT and Claude expect when they connect to a remote MCP service. It
authenticates people through a separately configured OpenID Connect provider,
shows consent in the provider's browser flow, and proxies authorized read-only
MCP requests to the Docker MCP service.

The DiscourseLens web app is not the OAuth authorization or consent page for
MCP. A connector user chooses **Connect** in Claude or ChatGPT and signs in
through the configured identity provider. The browser workspace now uses its
own local password accounts. Those accounts are independent from MCP OAuth and
do not share a project with this bridge automatically.

Each authenticated OIDC identity is routed to its own private DiscourseLens
project directory. The bridge signs the verified OIDC issuer and subject for
the internal Docker hop; MCP checks that signature and opens only that user's
MCP project directory. Email is used only for the invitation allowlist; it
does not determine ownership.

## Components

```text
Claude / ChatGPT
      │ OAuth 2.1 + MCP, HTTPS
      ▼
Cloudflare Worker (OAuth AS, DCR, PKCE, consent, scope enforcement)
      │ internal bearer + signed owner identity, HTTPS
      ▼
Cloudflare Tunnel ──► Docker MCP service (read-only, per-user project)
```

The Worker signs users in through the OIDC provider configured below. It keeps
the provider's access token out of MCP and stores the verified issuer, subject,
and email in the OAuth grant. The Docker MCP token and a separate tenant-header
signing key are held as Worker secrets. The Worker forwards the internal token
and a short-lived signed identity assertion only to the configured MCP origin.
The MCP image remains read-only and exposes only its existing allowlisted read
tools.

## Configure an OIDC login provider

Create a confidential OIDC client in an identity provider you control. Register
this exact callback after you know the Worker hostname:

```text
https://<worker-host>/callback
```

Enable Authorization Code with PKCE. The provider must issue an ID token signed
with RS256, ES256, or PS256 and include a verified email claim. The bridge checks
the signature, issuer, audience, nonce, expiry, and `email_verified` claim. It
rejects every email not in `AUTHORIZED_EMAILS`.

For a linguist evaluation, prefer invitation or administrator-managed accounts
with verified email. Add only the intended addresses to `AUTHORIZED_EMAILS`.
The bridge's identity provider manages MCP connector login; browser account
creation and passwords are managed separately by the web app.

## Deploy the Worker

Use Node.js 24 or newer. Install the pinned dependencies and create the KV
namespace used for OAuth client registrations, grants, and one-time state:

```bash
cd deploy/cloudflare-mcp-oauth
npm ci
npx wrangler kv namespace create OAUTH_KV
```

Copy the namespace ID into `wrangler.jsonc`. Set these non-secret values in
`wrangler.jsonc`:

- `MCP_AUTH_BASE_URL`: the HTTPS Worker origin, with no path;
- `MCP_RESOURCE_URL`: that same origin followed by `/mcp`;
- `OIDC_ISSUER_URL` and `OIDC_CLIENT_ID`;
- `MCP_UPSTREAM_URL`: the HTTPS `/mcp` URL for the Docker MCP origin.

Use the Cloudflare Worker hostname assigned to your account, or a custom
hostname already routed to this Worker. Add the callback URL to the OIDC client
and deploy the Worker:

```bash
npx wrangler secret put OIDC_CLIENT_SECRET
npx wrangler secret put AUTHORIZED_EMAILS
npx wrangler secret put MCP_BACKEND_TOKEN
npx wrangler secret put MCP_TENANT_SIGNING_KEY
npx wrangler deploy
```

Enter the identity provider's client secret, the comma-separated exact email
allowlist for the intended connector users, the same random token used by the
Docker MCP service, and the tenant signing key generated below. Do not put
these values in `wrangler.jsonc` or commit them. Keep KV and Worker secrets in
the same Cloudflare account as the deployed Worker.

## Start the Docker MCP origin

Set a persistent project storage root, a random MCP token, and the stable
MCP tenant identity and signing keys. Set `DISCOURSELENS_MCP_TENANT_SIGNING_KEY` to the same value you
will save as the Worker secret `MCP_TENANT_SIGNING_KEY`. These keys are separate
from the login provider's client secret. The MCP container is read-only and
publishes its port only on host loopback:

```bash
export DISCOURSELENS_PROJECT_DIR="/absolute/path/to/research-project"
export DISCOURSELENS_MCP_HTTP_TOKEN="$(openssl rand -hex 32)"
export DISCOURSELENS_TENANT_ID_KEY="$(openssl rand -hex 32)"
export DISCOURSELENS_MCP_TENANT_SIGNING_KEY="$(openssl rand -hex 32)"
docker compose -f compose.mcp-http.yaml up --build -d
```

The MCP container mounts the storage root read-only and rejects requests
without a valid Worker-signed owner identity. MCP identities resolve to their
own MCP project directories and are separate from browser password accounts.

Enter the same tenant assertion key when Wrangler prompts for
`MCP_TENANT_SIGNING_KEY`. The backend proxy token is stored as
`MCP_BACKEND_TOKEN`. Keep both out of `wrangler.jsonc`, source control, and
messages.

Use the same value as the Worker secret `MCP_BACKEND_TOKEN`. The public-facing
MCP URL that users add to Claude or ChatGPT is the Worker URL, not the origin
tunnel URL:

```text
https://<worker-host>/mcp
```

### Temporary tunnel for an evaluator visit

For a short test only, add the included Quick Tunnel overlay:

```bash
docker compose \
  -f compose.mcp-http.yaml \
  -f compose.mcp-cloudflare-preview.yaml \
  --profile mcp-cloudflare-preview up --build -d
docker compose \
  -f compose.mcp-http.yaml \
  -f compose.mcp-cloudflare-preview.yaml \
  logs -f cloudflare-mcp-origin
```

Copy the generated `trycloudflare.com` hostname into `MCP_UPSTREAM_URL` in
`wrangler.jsonc`, ending in `/mcp`, then deploy the Worker again. Quick Tunnel
hostnames change after a restart. The origin tunnel has no email gate; the
Docker MCP still rejects requests without its private bearer. Do not give that
token to connector clients or browsers.

For production, replace the Quick Tunnel with a named Cloudflare Tunnel and a
stable hostname. Configure its private service target as `http://mcp-http:8001`
on the Docker Compose network. Do not publish port 8001 to the Internet. Keep
the backend bearer requirement enabled and rotate the matching Worker secret
and Docker token together.

## Connect Claude or ChatGPT

Set `DISCOURSELENS_MCP_PUBLIC_URL` in the web deployment to the Worker resource
URL, then open **Conectar Claude o ChatGPT** and copy it. The web panel's API
token only unlocks the creator's workspace; do not give it to invited MCP
users.

- **Claude:** Settings → Connectors → Add custom connector → paste the Worker
  URL → Connect. Claude opens the consent and OIDC login flow.
- **ChatGPT:** open Plugins → **+ → Add custom MCP server**, enter the Worker
  URL, select OAuth, review the discovered tools, and finish the consent/login
  flow. Availability and permissions depend on the ChatGPT plan and workspace
  administrator; see [OpenAI's current setup guide](https://help.openai.com/en/articles/12584461-connecting-custom-mcp-servers-in-chatgpt).

The first connection dynamically registers the MCP client, validates its PKCE
authorization request, presents client/redirect/scope consent, then signs the
person in through OIDC. Only `discourselens:read` is granted. Refresh tokens
are enabled so a valid connection can survive access-token expiry. A user can
remove/revoke the connector from the provider UI; the OAuth grant is retained
in the Worker's KV until expiry unless revoked through its token endpoint.
Removing a person from `AUTHORIZED_EMAILS` immediately blocks their already
issued access tokens at the Worker before any request reaches the Docker MCP;
also remove/revoke their connector grant in the provider UI.

## Local checks

```bash
npm run check
npm test
npm audit --omit=dev
```

These checks cover TypeScript types, OAuth consent escaping, backend-token
isolation, MCP protocol-header forwarding, response-header filtering, and
scope rejection. Finish with a real Claude and/or ChatGPT connection against
the deployed Worker before treating provider compatibility as verified.
