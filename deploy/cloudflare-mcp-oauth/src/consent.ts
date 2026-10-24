import type { ConsentDescription } from "@cloudflare/workers-oauth-provider";

export function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => `&#${char.charCodeAt(0)};`);
}

export function consentHtml(details: ConsentDescription, handle: string, scopes: string[]): string {
  const requestedScopes = scopes
    .map(
      (scope) =>
        `<label><input type="checkbox" name="scope" value="${escapeHtml(scope)}" checked> ${escapeHtml(scope)}</label>`,
    )
    .join("<br>");
  const client = escapeHtml(details.clientName);
  const host = escapeHtml(details.redirectHost);
  const clientOrigin = details.clientDomain
    ? `Cliente publicado por <strong>${escapeHtml(details.clientDomain)}</strong>.`
    : "El nombre del cliente lo ha proporcionado el propio cliente MCP.";
  const localWarning = details.redirectIsLoopback
    ? "<p><strong>La respuesta volverá a una aplicación local de este equipo. Continúa solo si acabas de iniciar la conexión.</strong></p>"
    : "";
  return `<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Autorizar DiscourseLens</title><style>body{font:16px system-ui;max-width:620px;margin:8vh auto;padding:24px;color:#17221e;background:#f6f4ee}main{background:#fffefa;border:1px solid #dce2d8;border-radius:14px;padding:28px}h1{font:500 32px Georgia}button{padding:12px 18px;margin:18px 8px 0 0;border:0;border-radius:7px;background:#176b4b;color:white;font-weight:700}button[name=decision][value=deny]{background:#e8ece3;color:#17221e}</style><main><h1>Autorizar ${client} para DiscourseLens</h1><p>${clientOrigin}</p><p>El código de autorización volverá a <strong>${host}</strong>.</p>${localWarning}<p>Se solicita acceso de solo lectura al proyecto. El proveedor de identidad te pedirá iniciar sesión después de aprobar.</p><form method="post" action="/authorize"><input type="hidden" name="handle" value="${escapeHtml(handle)}">${requestedScopes}<br><button name="decision" value="approve">Continuar y autorizar</button><button name="decision" value="deny">Denegar</button></form></main></html>`;
}
