import { afterEach, describe, expect, it, vi } from "vitest";
import { consentHtml } from "../src/consent";
import { fetchJson } from "../src/oidc_http";
import { proxyMcp, selectRequestHeaders, selectResponseHeaders } from "../src/mcp_proxy";
import type { ConsentDescription } from "@cloudflare/workers-oauth-provider";

const env = {
  MCP_AUTH_BASE_URL: "https://discourselens-auth.example.workers.dev",
  MCP_RESOURCE_URL: "https://discourselens-auth.example.workers.dev/mcp",
  MCP_UPSTREAM_URL: "https://discourselens-origin.example.net/mcp",
  MCP_BACKEND_TOKEN: "b".repeat(64),
  MCP_TENANT_SIGNING_KEY: "t".repeat(64),
  OIDC_ISSUER_URL: "https://identity.example.net/realms/research",
  OIDC_CLIENT_ID: "connector-login",
  OIDC_CLIENT_SECRET: globalThis.crypto.randomUUID(),
  AUTHORIZED_EMAILS: "creator@example.org,linguist@example.org",
} as Parameters<typeof proxyMcp>[1];

afterEach(() => vi.unstubAllGlobals());

describe("OIDC metadata fetching", () => {
  it("rejects redirects using the Cloudflare Workers supported manual mode", async () => {
    let redirectMode: RequestRedirect | undefined;
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      redirectMode = init?.redirect;
      return new Response(null, {
        status: 302,
        headers: { location: "https://unexpected.example/" },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchJson(new URL("https://identity.example/.well-known/openid-configuration")))
      .rejects.toThrow("OIDC provider returned an unexpected redirect");
    expect(redirectMode).toBe("manual");
  });
});

describe("OAuth protected MCP proxy", () => {
  it("injects its internal bearer and forwards only MCP protocol headers", async () => {
    let sent: Request | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (request: Request) => {
        sent = request;
        return new Response("mcp-result", {
          headers: {
            "content-type": "application/json",
            "mcp-session-id": "session-1",
            "set-cookie": "private=secret",
            "x-internal-host": "hidden",
          },
        });
      }),
    );
    const request = new Request(env.MCP_RESOURCE_URL, {
      method: "POST",
      headers: {
        accept: "application/json, text/event-stream",
        authorization: "Bearer client-token-must-not-pass-through",
        "content-type": "application/json",
        "mcp-protocol-version": "2025-11-25",
        cookie: "session=private",
        "x-forwarded-for": "198.51.100.1",
        "x-discourselens-tenant": "attacker-selected-tenant",
      },
      body: JSON.stringify({ jsonrpc: "2.0", method: "tools/list", id: 1 }),
    });

    const response = await proxyMcp(
      request,
      env,
      { scope: ["discourselens:read"] },
      true,
      () => new Response("insufficient scope", { status: 403 }),
      { issuer: "https://identity.example.net/realms/research", sub: "linguist-7" },
    );

    expect(response.status).toBe(200);
    expect(await response.text()).toBe("mcp-result");
    expect(sent?.url).toBe(env.MCP_UPSTREAM_URL);
    expect(sent?.headers.get("authorization")).toBe(`Bearer ${env.MCP_BACKEND_TOKEN}`);
    expect(sent?.headers.get("cookie")).toBeNull();
    expect(sent?.headers.get("x-forwarded-for")).toBeNull();
    expect(sent?.headers.get("x-discourselens-tenant")).not.toBe("attacker-selected-tenant");
    expect(sent?.headers.get("mcp-protocol-version")).toBe("2025-11-25");
    expect(sent?.headers.get("x-discourselens-tenant")).toBeTruthy();
    expect(sent?.headers.get("x-discourselens-tenant-signature")).toMatch(/^[a-f0-9]{64}$/);
    expect(response.headers.get("mcp-session-id")).toBe("session-1");
    expect(response.headers.get("set-cookie")).toBeNull();
    expect(response.headers.get("x-internal-host")).toBeNull();
  });

  it("rejects tokens without the read scope before reaching the project", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const request = new Request(env.MCP_RESOURCE_URL, {
      method: "POST",
      body: "{}",
    });

    const response = await proxyMcp(
      request,
      env,
      { scope: [] },
      true,
      () => new Response("insufficient scope", { status: 403 }),
    );

    expect(response.status).toBe(403);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("stops an already-issued grant from reaching the project after allowlist removal", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const request = new Request(env.MCP_RESOURCE_URL, { method: "POST", body: "{}" });

    const response = await proxyMcp(
      request,
      env,
      { scope: ["discourselens:read"] },
      false,
      () => new Response("insufficient scope", { status: 403 }),
    );

    expect(response.status).toBe(403);
    expect(await response.text()).toContain("ya no está autorizada");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("fails closed when the verified grant identity is missing", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const response = await proxyMcp(
      new Request(env.MCP_RESOURCE_URL, { method: "POST", body: "{}" }),
      env,
      { scope: ["discourselens:read"] },
      true,
      () => new Response("insufficient scope", { status: 403 }),
    );
    expect(response.status).toBe(503);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("OAuth consent page", () => {
  it("escapes untrusted client metadata and names the exact token redirect host", () => {
    const details: ConsentDescription = {
      clientId: "client-1",
      clientName: "<img src=x onerror=alert(1)>",
      clientDomain: "assistant.example",
      redirectUri: "https://assistant.example/callback",
      redirectHost: "assistant.example",
      redirectIsLoopback: false,
      scope: ["discourselens:read"],
    };

    const html = consentHtml(details, "opaque-handle", ["discourselens:read"]);

    expect(html).toContain("&#60;img src=x onerror=alert(1)&#62;");
    expect(html).not.toContain("<img src=x");
    expect(html).toContain("assistant.example");
    expect(html).toContain('value="discourselens:read"');
    expect(html).toContain('value="opaque-handle"');
  });
});

describe("proxy header allowlists", () => {
  it("does not forward authorization, cookies, client forwarding headers or response cookies", () => {
    const incoming = new Headers({
      authorization: "Bearer client-secret",
      cookie: "auth=secret",
      "x-forwarded-host": "attacker.invalid",
      "mcp-session-id": "s1",
    });
    const requestHeaders = selectRequestHeaders(incoming);
    expect(requestHeaders.get("authorization")).toBeNull();
    expect(requestHeaders.get("cookie")).toBeNull();
    expect(requestHeaders.get("x-forwarded-host")).toBeNull();
    expect(requestHeaders.get("mcp-session-id")).toBe("s1");

    const responseHeaders = selectResponseHeaders(
      new Headers({ "set-cookie": "private=secret", "mcp-session-id": "s1" }),
    );
    expect(responseHeaders.get("set-cookie")).toBeNull();
    expect(responseHeaders.get("mcp-session-id")).toBe("s1");
  });
});
