import OAuthProvider, {
  AuthorizationError,
  CimdFetchError,
  authorizationErrorRedirect,
  insufficientScope,
  type AuthRequest,
  type OAuthHelpers,
  type OAuthResourceContext,
} from "@cloudflare/workers-oauth-provider";
import { createRemoteJWKSet, jwtVerify } from "jose";
import { consentHtml, escapeHtml } from "./consent";
import { proxyMcp as forwardMcp, requireHttpsUrl } from "./mcp_proxy";
import { fetchJson, UPSTREAM_TIMEOUT_MS } from "./oidc_http";

const READ_SCOPE = "discourselens:read";
const OFFLINE_SCOPE = "offline_access";

interface IdentityClaims {
  issuer: string;
  sub: string;
  email: string;
  name?: string;
}

interface UpstreamState {
  verifier: string;
  nonce: string;
}

interface Env {
  OAUTH_KV: KVNamespace;
  OAUTH_PROVIDER: OAuthHelpers;
  MCP_AUTH_BASE_URL: string;
  MCP_RESOURCE_URL: string;
  MCP_UPSTREAM_URL: string;
  MCP_BACKEND_TOKEN: string;
  MCP_TENANT_SIGNING_KEY: string;
  OIDC_ISSUER_URL: string;
  OIDC_CLIENT_ID: string;
  OIDC_CLIENT_SECRET: string;
  OIDC_TOKEN_AUTH_METHOD?: string;
  AUTHORIZED_EMAILS: string;
}

const secureHeaders = {
  "Cache-Control": "no-store",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
};

function jsonResponse(payload: unknown, status = 200): Response {
  return Response.json(payload, { status, headers: secureHeaders });
}

function issuerUrl(env: Env): URL {
  const parsed = new URL(env.MCP_AUTH_BASE_URL);
  if (parsed.protocol !== "https:" || parsed.pathname !== "/" || parsed.search || parsed.hash) {
    throw new Error("MCP_AUTH_BASE_URL must be an HTTPS origin without a path");
  }
  return parsed;
}

function resourceUrl(env: Env): URL {
  const base = issuerUrl(env);
  const resource = requireHttpsUrl(env.MCP_RESOURCE_URL, "MCP_RESOURCE_URL");
  if (resource.origin !== base.origin || resource.pathname !== "/mcp" || resource.search || resource.hash) {
    throw new Error("MCP_RESOURCE_URL must be MCP_AUTH_BASE_URL followed by /mcp");
  }
  return resource;
}

function authorizedEmails(env: Env): Set<string> {
  return new Set(
    env.AUTHORIZED_EMAILS.split(",")
      .map((email) => email.trim().toLocaleLowerCase("en-US"))
      .filter(Boolean),
  );
}

function errorResponse(error: unknown): Response {
  if (error instanceof AuthorizationError && error.redirectTo) {
    return Response.redirect(error.redirectTo, 302);
  }
  if (error instanceof AuthorizationError) {
    return new Response(escapeHtml(error.description), {
      status: 400,
      headers: { ...secureHeaders, "Content-Type": "text/plain; charset=utf-8" },
    });
  }
  if (error instanceof CimdFetchError) {
    return new Response("No se pudo verificar los metadatos del cliente MCP.", {
      status: 400,
      headers: { ...secureHeaders, "Content-Type": "text/plain; charset=utf-8" },
    });
  }
  throw error;
}

function randomVerifier(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return base64Url(bytes);
}

function base64Url(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

async function pkceChallenge(verifier: string): Promise<string> {
  return base64Url(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier))));
}

interface OidcMetadata {
  issuer: string;
  authorization_endpoint: URL;
  token_endpoint: URL;
  jwks_uri: URL;
}

async function oidcMetadata(env: Env): Promise<OidcMetadata> {
  const issuer = requireHttpsUrl(env.OIDC_ISSUER_URL, "OIDC_ISSUER_URL");
  const issuerUrl = issuer.href;
  const discoveryBase = issuerUrl.endsWith("/") ? issuerUrl : `${issuerUrl}/`;
  const wellKnown = new URL(".well-known/openid-configuration", discoveryBase);
  const metadata = await fetchJson(wellKnown);
  if (metadata.issuer !== issuerUrl) {
    throw new Error("OIDC discovery issuer did not match configuration");
  }
  const authorization = requireHttpsUrl(String(metadata.authorization_endpoint ?? ""), "OIDC authorization endpoint");
  const token = requireHttpsUrl(String(metadata.token_endpoint ?? ""), "OIDC token endpoint");
  const jwks = requireHttpsUrl(String(metadata.jwks_uri ?? ""), "OIDC JWKS endpoint");
  return { issuer: String(metadata.issuer), authorization_endpoint: authorization, token_endpoint: token, jwks_uri: jwks };
}

async function beginLogin(
  request: AuthRequest,
  env: Env,
  oauth: OAuthHelpers,
  priorHeaders: Headers,
): Promise<Response> {
  const metadata = await oidcMetadata(env);
  const verifier = randomVerifier();
  const nonce = randomVerifier();
  const transaction = await oauth.beginUpstream(request, {
    data: { verifier, nonce } satisfies UpstreamState,
    headers: priorHeaders,
  });
  const redirectUri = new URL("/callback", issuerUrl(env)).href;
  const location = new URL(metadata.authorization_endpoint);
  location.searchParams.set("response_type", "code");
  location.searchParams.set("client_id", env.OIDC_CLIENT_ID);
  location.searchParams.set("redirect_uri", redirectUri);
  location.searchParams.set("scope", "openid email profile");
  location.searchParams.set("state", transaction.state);
  location.searchParams.set("nonce", nonce);
  location.searchParams.set("code_challenge", await pkceChallenge(verifier));
  location.searchParams.set("code_challenge_method", "S256");
  transaction.headers.set("Location", location.href);
  return new Response(null, { status: 302, headers: transaction.headers });
}

function base64Basic(value: string): string {
  let binary = "";
  for (const byte of new TextEncoder().encode(value)) binary += String.fromCharCode(byte);
  return btoa(binary);
}

async function exchangeOidcCode(
  env: Env,
  metadata: OidcMetadata,
  code: string,
  verifier: string,
): Promise<string> {
  const redirectUri = new URL("/callback", issuerUrl(env)).href;
  const form = new URLSearchParams({
    grant_type: "authorization_code",
    code,
    redirect_uri: redirectUri,
    client_id: env.OIDC_CLIENT_ID,
    code_verifier: verifier,
  });
  const headers: Record<string, string> = {
    "Content-Type": "application/x-www-form-urlencoded",
  };
  if ((env.OIDC_TOKEN_AUTH_METHOD ?? "client_secret_basic") === "client_secret_basic") {
    headers.Authorization = `Basic ${base64Basic(`${env.OIDC_CLIENT_ID}:${env.OIDC_CLIENT_SECRET}`)}`;
    form.delete("client_id");
  } else if (env.OIDC_TOKEN_AUTH_METHOD === "client_secret_post") {
    form.set("client_secret", env.OIDC_CLIENT_SECRET);
  } else {
    throw new Error("Unsupported OIDC token endpoint authentication method");
  }
  const body = await fetchJson(metadata.token_endpoint, { method: "POST", headers, body: form });
  const idToken = body.id_token;
  if (typeof idToken !== "string" || !idToken) throw new Error("OIDC response did not contain an ID token");
  return idToken;
}

async function verifyIdentity(
  env: Env,
  metadata: OidcMetadata,
  idToken: string,
  expectedNonce: string,
): Promise<IdentityClaims> {
  const jwks = createRemoteJWKSet(metadata.jwks_uri, { timeoutDuration: UPSTREAM_TIMEOUT_MS });
  const verified = await jwtVerify(idToken, jwks, {
    issuer: metadata.issuer,
    audience: env.OIDC_CLIENT_ID,
    algorithms: ["RS256", "ES256", "PS256"],
    requiredClaims: ["sub", "exp", "iat", "iss", "aud", "nonce"],
  });
  const claims = verified.payload;
  if (claims.nonce !== expectedNonce) throw new Error("OIDC nonce did not match");
  if (
    typeof claims.sub !== "string" ||
    typeof claims.email !== "string" ||
    claims.email_verified !== true
  ) {
    throw new Error("OIDC identity lacks a verified email or subject");
  }
  const email = claims.email.trim().toLocaleLowerCase("en-US");
  if (!authorizedEmails(env).has(email)) throw new Error("This identity is not authorized for the project");
  const identity: IdentityClaims = { issuer: metadata.issuer, sub: claims.sub, email };
  if (typeof claims.name === "string") identity.name = claims.name.slice(0, 200);
  return identity;
}

async function authorize(request: Request, env: Env): Promise<Response> {
  const oauth = env.OAUTH_PROVIDER;
  try {
    if (request.method === "GET") {
      const authRequest = await oauth.parseAuthRequest(request);
      if (!authRequest.scope.includes(READ_SCOPE)) {
        return Response.redirect(
          authorizationErrorRedirect(authRequest, "invalid_scope", "Se requiere acceso de solo lectura."),
          302,
        );
      }
      const grantedScopes = authRequest.scope.filter(
        (scope) => scope === READ_SCOPE || scope === OFFLINE_SCOPE,
      );
      const details = await oauth.describeConsent(authRequest);
      const consent = await oauth.beginConsent(authRequest);
      consent.headers.set("Content-Type", "text/html; charset=utf-8");
      consent.headers.set("Cache-Control", "no-store");
      return new Response(consentHtml(details, consent.handle, grantedScopes), {
        headers: consent.headers,
      });
    }
    if (request.method === "POST") {
      const form = await request.formData();
      const handle = form.get("handle");
      if (typeof handle !== "string" || !handle) {
        return new Response("Solicitud de consentimiento no válida.", { status: 400, headers: secureHeaders });
      }
      if (form.get("decision") !== "approve") {
        const denied = await oauth.denyConsent(request, handle);
        return new Response(null, { status: 302, headers: denied.headers });
      }
      const selected = form.getAll("scope").filter((scope): scope is string => typeof scope === "string");
      if (!selected.includes(READ_SCOPE)) {
        return new Response("Debes autorizar el acceso de solo lectura para conectar.", {
          status: 400,
          headers: secureHeaders,
        });
      }
      const approved = await oauth.approveConsent(request, handle, { scope: selected });
      return await beginLogin(approved.request, env, oauth, approved.headers);
    }
    return new Response("Método no permitido.", { status: 405, headers: { ...secureHeaders, Allow: "GET, POST" } });
  } catch (error) {
    return errorResponse(error);
  }
}

async function callback(request: Request, env: Env): Promise<Response> {
  const oauth = env.OAUTH_PROVIDER;
  try {
    const { request: authRequest, data, headers } = await oauth.finishUpstream<UpstreamState>(request);
    const url = new URL(request.url);
    if (url.searchParams.has("error")) {
      headers.set("Location", authorizationErrorRedirect(authRequest, "access_denied"));
      return new Response(null, { status: 302, headers });
    }
    const code = url.searchParams.get("code");
    if (!code) throw new Error("OIDC callback did not include an authorization code");
    const metadata = await oidcMetadata(env);
    const idToken = await exchangeOidcCode(env, metadata, code, data.verifier);
    const identity = await verifyIdentity(env, metadata, idToken, data.nonce);
    const { redirectTo } = await oauth.completeAuthorization({
      request: authRequest,
      userId: base64Url(new TextEncoder().encode(`${metadata.issuer}|${identity.sub}`)),
      metadata: { email: identity.email },
      scope: authRequest.scope,
      props: identity,
    });
    headers.set("Location", redirectTo);
    headers.set("Cache-Control", "no-store");
    return new Response(null, { status: 302, headers });
  } catch (error) {
    if (error instanceof AuthorizationError && error.redirectTo) {
      return Response.redirect(error.redirectTo, 302);
    }
    if (error instanceof AuthorizationError) {
      return new Response(escapeHtml(error.description), {
        status: 400,
        headers: { ...secureHeaders, "Content-Type": "text/plain; charset=utf-8" },
      });
    }
    return new Response("No se pudo autenticar la cuenta para el conector MCP.", {
      status: 403,
      headers: { ...secureHeaders, "Content-Type": "text/plain; charset=utf-8" },
    });
  }
}

function selectRequestHeaders(incoming: Headers): Headers {
  const headers = new Headers();
  for (const name of [
    "accept",
    "content-type",
    "mcp-protocol-version",
    "mcp-session-id",
    "last-event-id",
  ]) {
    const value = incoming.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

function selectResponseHeaders(incoming: Headers): Headers {
  const headers = new Headers(secureHeaders);
  for (const name of [
    "content-type",
    "mcp-protocol-version",
    "mcp-session-id",
    "last-event-id",
    "cache-control",
    "retry-after",
  ]) {
    const value = incoming.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

async function proxyMcp(
  request: Request,
  env: Env,
  context: OAuthResourceContext<unknown>,
): Promise<Response> {
  const principal = context.props as IdentityClaims | undefined;
  const principalAllowed = Boolean(
    principal?.email && authorizedEmails(env).has(principal.email.toLocaleLowerCase("en-US")),
  );
  return forwardMcp(
    request,
    env,
    context.auth,
    principalAllowed,
    () => insufficientScope(context.auth, [READ_SCOPE]),
    principal ? { issuer: principal.issuer, sub: principal.sub } : undefined,
  );
}

async function defaultHandler(request: Request, env: Env): Promise<Response> {
  const path = new URL(request.url).pathname;
  if (path === "/authorize") return authorize(request, env);
  if (path === "/callback" && request.method === "GET") return callback(request, env);
  if (path === "/healthz" && request.method === "GET") return jsonResponse({ status: "ok" });
  return new Response("Not found.", { status: 404, headers: secureHeaders });
}

const oauthProviders = new Map<string, OAuthProvider<Env>>();

function providerFor(env: Env): OAuthProvider<Env> {
  const base = issuerUrl(env);
  const resource = resourceUrl(env);
  let provider = oauthProviders.get(base.href);
  if (!provider) {
    provider = new OAuthProvider<Env>({
      apiRoute: "/mcp",
      apiHandler: {
        fetch: (request, environment, context) =>
          proxyMcp(request, environment, context as OAuthResourceContext<unknown>),
      },
      defaultHandler: { fetch: defaultHandler },
      authorizeEndpoint: "/authorize",
      tokenEndpoint: "/token",
      clientRegistrationEndpoint: "/register",
      clientIdMetadataDocumentEnabled: true,
      scopesSupported: [READ_SCOPE, OFFLINE_SCOPE],
      requiredScopes: [READ_SCOPE],
      accessTokenTTL: 3600,
      refreshTokenTTL: 2_592_000,
      resourceMetadata: {
        resource: resource.href,
        authorization_servers: [base.origin],
      },
    });
    oauthProviders.set(base.href, provider);
  }
  return provider;
}

export default {
  fetch(request: Request, env: Env, context: ExecutionContext): Promise<Response> {
    return providerFor(env).fetch(request, env, context);
  },
};

export { authorize, callback, proxyMcp };
