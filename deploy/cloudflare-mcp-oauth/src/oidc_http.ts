const MAX_JSON_BYTES = 65_536;
export const UPSTREAM_TIMEOUT_MS = 12_000;

export async function fetchJson(url: URL, init?: RequestInit): Promise<Record<string, unknown>> {
  const response = await fetch(url, {
    ...init,
    redirect: "manual",
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    headers: { Accept: "application/json", ...init?.headers },
  });
  if (response.status >= 300 && response.status < 400) {
    throw new Error("OIDC provider returned an unexpected redirect");
  }
  const length = response.headers.get("content-length");
  if (length && /^[0-9]+$/.test(length) && Number(length) > MAX_JSON_BYTES) {
    throw new Error("OIDC response exceeded limit");
  }
  if (!response.ok) throw new Error("OIDC provider request failed");
  if (!response.body) throw new Error("OIDC provider response was empty");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    totalBytes += value.byteLength;
    if (totalBytes > MAX_JSON_BYTES) {
      await reader.cancel();
      throw new Error("OIDC response exceeded limit");
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const payload: unknown = JSON.parse(new TextDecoder().decode(bytes));
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    throw new Error("OIDC provider response was malformed");
  }
  return payload as Record<string, unknown>;
}
