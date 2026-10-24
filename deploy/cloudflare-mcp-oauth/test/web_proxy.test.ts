import { afterEach, describe, expect, it, vi } from "vitest";
import worker from "../src/web_proxy";

afterEach(() => vi.unstubAllGlobals());

describe("web workspace proxy", () => {
  it("forwards paths, query, cookies, and public proxy headers to the configured tunnel", async () => {
    let forwarded: Request | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (request: Request) => {
        forwarded = request;
        return new Response("workspace", { headers: { "content-type": "text/plain" } });
      }),
    );
    const request = new Request("https://web.example.workers.dev/api/documents?limit=3", {
      headers: {
        cookie: "discourselens_session=opaque",
        origin: "https://web.example.workers.dev",
        "user-agent": "Python-urllib/3.14",
      },
    });

    const response = await worker.fetch(request, {
      WEB_UPSTREAM_URL: "https://research-preview.trycloudflare.com",
    });

    expect(response.status).toBe(200);
    expect(await response.text()).toBe("workspace");
    expect(forwarded?.url).toBe("https://research-preview.trycloudflare.com/api/documents?limit=3");
    expect(forwarded?.headers.get("cookie")).toBe("discourselens_session=opaque");
    expect(forwarded?.headers.get("x-forwarded-host")).toBe("web.example.workers.dev");
    expect(forwarded?.headers.get("x-forwarded-proto")).toBe("https");
    expect(forwarded?.headers.get("user-agent")).toBe("DiscourseLens-Web-Proxy/0.1");
    expect(response.headers.get("x-content-type-options")).toBe("nosniff");
  });

  it("streams a PDF upload and preserves the upstream status", async () => {
    let forwarded: Request | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (request: Request) => {
        forwarded = request;
        return Response.json({ status: "SUCCESS" }, { status: 201 });
      }),
    );
    const request = new Request("https://web.example.workers.dev/api/documents", {
      method: "POST",
      headers: { "content-type": "application/pdf", "x-document-filename": "sample.pdf" },
      body: "%PDF-1.7 test document",
    });

    const response = await worker.fetch(request, {
      WEB_UPSTREAM_URL: "https://research-preview.trycloudflare.com",
    });

    expect(response.status).toBe(201);
    expect(await response.json()).toEqual({ status: "SUCCESS" });
    expect(await forwarded?.text()).toBe("%PDF-1.7 test document");
  });

  it("rejects untrusted upstream URLs", async () => {
    const request = new Request("https://web.example.workers.dev/");
    await expect(
      worker.fetch(request, { WEB_UPSTREAM_URL: "https://attacker.example" }),
    ).rejects.toThrow("WEB_UPSTREAM_URL must be an HTTPS Cloudflare Quick Tunnel origin");
  });
});
