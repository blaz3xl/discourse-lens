export interface ProxyEnvironment {
  MCP_AUTH_BASE_URL: string;
  MCP_RESOURCE_URL: string;
  MCP_UPSTREAM_URL: string;
  MCP_BACKEND_TOKEN: string;
  MCP_TENANT_SIGNING_KEY: string;
}

export interface ProxyAuth {
  scope: string[];
}

export interface TenantIdentity {
  issuer: string;
  sub: string;
}

const secureHeaders = {
  "Cache-Control": "no-store",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
};

export function requireHttpsUrl(value: string, label: string): URL {
  const parsed = new URL(value);
  if (
    parsed.protocol !== "https:" ||
    !parsed.hostname ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error(`${label} must be a public HTTPS URL`);
  }
  return parsed;
}

export function selectRequestHeaders(incoming: Headers): Headers {
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

export function selectResponseHeaders(incoming: Headers): Headers {
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

function base64Url(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

async function tenantAssertion(identity: TenantIdentity, keyText: string): Promise<[string, string]> {
  if (
    keyText.length < 32 ||
    !identity.issuer.startsWith("https://") ||
    !identity.sub ||
    identity.sub.length > 1024 ||
    /[\r\n]/.test(identity.sub)
  ) {
    throw new Error("Tenant identity signing is not configured correctly");
  }
  const payload = base64Url(
    new TextEncoder().encode(`${identity.issuer}\n${identity.sub}\n${Math.floor(Date.now() / 1000) + 60}`),
  );
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(keyText),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payload));
  return [payload, Array.from(new Uint8Array(signature), (byte) => byte.toString(16).padStart(2, "0")).join("")];
}

export async function proxyMcp(
  request: Request,
  env: ProxyEnvironment,
  auth: ProxyAuth,
  principalAllowed: boolean,
  insufficientScope: () => Response,
  tenantIdentity?: TenantIdentity,
): Promise<Response> {
  if (!principalAllowed) {
    return new Response("La cuenta ya no está autorizada para este proyecto.", {
      status: 403,
      headers: secureHeaders,
    });
  }
  if (!auth.scope.includes("discourselens:read")) return insufficientScope();
  if (!env.MCP_BACKEND_TOKEN || env.MCP_BACKEND_TOKEN.length < 32) {
    return new Response("El acceso al MCP interno no está configurado.", {
      status: 503,
      headers: secureHeaders,
    });
  }
  if (!tenantIdentity || !env.MCP_TENANT_SIGNING_KEY || env.MCP_TENANT_SIGNING_KEY.length < 32) {
    return new Response("El aislamiento de espacios privados no está configurado.", {
      status: 503,
      headers: secureHeaders,
    });
  }
  const upstream = requireHttpsUrl(env.MCP_UPSTREAM_URL, "MCP_UPSTREAM_URL");
  const resource = requireHttpsUrl(env.MCP_RESOURCE_URL, "MCP_RESOURCE_URL");
  const base = requireHttpsUrl(env.MCP_AUTH_BASE_URL, "MCP_AUTH_BASE_URL");
  if (
    resource.origin !== base.origin ||
    resource.pathname !== "/mcp" ||
    resource.search ||
    resource.hash
  ) {
    return new Response("MCP_RESOURCE_URL must be MCP_AUTH_BASE_URL followed by /mcp.", {
      status: 500,
      headers: secureHeaders,
    });
  }
  if (upstream.pathname !== "/mcp") {
    return new Response("MCP_UPSTREAM_URL debe terminar en /mcp.", {
      status: 500,
      headers: secureHeaders,
    });
  }
  const init: RequestInit & { duplex: "half" } = {
    method: request.method,
    headers: selectRequestHeaders(request.headers),
    body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    signal: AbortSignal.timeout(120_000),
    duplex: "half",
  };
  const upstreamRequest = new Request(upstream, init);
  upstreamRequest.headers.set("Authorization", `Bearer ${env.MCP_BACKEND_TOKEN}`);
  const [assertion, signature] = await tenantAssertion(tenantIdentity, env.MCP_TENANT_SIGNING_KEY);
  upstreamRequest.headers.set("X-DiscourseLens-Tenant", assertion);
  upstreamRequest.headers.set("X-DiscourseLens-Tenant-Signature", signature);
  const response = await fetch(upstreamRequest, { redirect: "error" });
  return new Response(response.body, {
    status: response.status,
    headers: selectResponseHeaders(response.headers),
  });
}
