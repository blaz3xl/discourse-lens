interface Env {
  WEB_UPSTREAM_URL: string;
}

const secureHeaders = {
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "strict-origin-when-cross-origin",
};

function upstreamOrigin(value: string): URL {
  const url = new URL(value);
  if (
    url.protocol !== "https:" ||
    !url.hostname.endsWith(".trycloudflare.com") ||
    url.username ||
    url.password ||
    url.pathname !== "/" ||
    url.search ||
    url.hash
  ) {
    throw new Error("WEB_UPSTREAM_URL must be an HTTPS Cloudflare Quick Tunnel origin");
  }
  return url;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "TRACE" || request.method === "CONNECT") {
      return new Response("Method not allowed", { status: 405, headers: secureHeaders });
    }

    const publicUrl = new URL(request.url);
    const origin = upstreamOrigin(env.WEB_UPSTREAM_URL);
    const upstreamUrl = new URL(publicUrl.pathname + publicUrl.search, origin);
    const headers = new Headers(request.headers);
    headers.delete("host");
    headers.set("x-forwarded-host", publicUrl.host);
    headers.set("x-forwarded-proto", "https");
    headers.set("user-agent", "DiscourseLens-Web-Proxy/0.1");

    const requestInit: RequestInit & { duplex: "half" } = {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
      redirect: "manual",
      duplex: "half",
    };
    const upstreamRequest = new Request(upstreamUrl, requestInit);
    const response = await fetch(upstreamRequest);
    const responseHeaders = new Headers(response.headers);
    for (const [name, value] of Object.entries(secureHeaders)) {
      if (!responseHeaders.has(name)) responseHeaders.set(name, value);
    }
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  },
};
