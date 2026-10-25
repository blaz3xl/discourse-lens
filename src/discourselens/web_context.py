"""Web discovery and immutable contextual excerpts, distinct from the corpus."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import socket
import ssl
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urljoin, urlsplit, urlunsplit

from .documents import _CanonicalHtmlParser
from .errors import ModelProviderError, ProjectError
from .evidence_limits import MAX_CONTEXT_EVIDENCE_CHARS

if TYPE_CHECKING:
    from .project import Project

MAX_WEB_BYTES = 2 * 1024 * 1024
MAX_WEB_EXTRACTED_CHARS = 300_000


def _is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    if not address.is_global or address.is_multicast or address.is_reserved:
        return False
    if isinstance(address, ipaddress.IPv4Address):
        # Python 3.12 predates several special-purpose range corrections.
        return address not in ipaddress.ip_network("192.0.0.0/24")
    return not (
        address.is_site_local or address.sixtofour or address.teredo
        or address in ipaddress.ip_network("64:ff9b::/96")
        or address in ipaddress.ip_network("64:ff9b:1::/48")
    )


def _public_destination(url: str) -> tuple[str, int, list[tuple[Any, ...]]]:
    """Resolve only public HTTP(S) addresses; the connection uses these same IPs."""
    if not isinstance(url, str) or len(url) > 2_048 or any(ord(c) <= 32 for c in url):
        raise ProjectError("web context requires a bounded public HTTP(S) URL")
    try:
        parsed = urlsplit(url)
        hostname, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        raise ProjectError("web context URL is invalid") from None
    if parsed.scheme not in {"https", "http"} or not hostname or parsed.username is not None:
        raise ProjectError("web context requires a public HTTP(S) URL without credentials")
    if parsed.password is not None or port not in {80, 443}:
        raise ProjectError("web context supports only public HTTP(S) ports")
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError:
        raise ProjectError("web context host could not be resolved") from None
    if not addresses or any(not _is_public_address(str(row[4][0])) for row in addresses):
        raise ProjectError("web context cannot access private or special network addresses")
    return hostname, port, addresses


def _request_public_url(url: str) -> tuple[int, dict[str, str], bytes]:
    hostname, port, addresses = _public_destination(url)
    parsed = urlsplit(url)

    def pinned_connection(_address, timeout=15, source_address=None):
        last_error: OSError | None = None
        for family, socktype, proto, _canonical, destination in addresses:
            connection = socket.socket(family, socktype, proto)
            try:
                connection.settimeout(timeout)
                connection.connect(destination)
                return connection
            except OSError as error:
                connection.close()
                last_error = error
        raise last_error or OSError("public source connection failed")

    # http.client ignores environment proxies. Pin TCP to the addresses checked
    # above while HTTPS retains the real hostname for SNI and certificate checks.
    connection: http.client.HTTPConnection
    if parsed.scheme == "https":
        connection = http.client.HTTPSConnection(hostname, port, timeout=15, context=ssl.create_default_context())
    else:
        connection = http.client.HTTPConnection(hostname, port, timeout=15)
    cast(Any, connection)._create_connection = pinned_connection
    try:
        request_path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection.request("GET", request_path, headers={
            "User-Agent": "DiscourseLens/0.1 (context research)",
            "Accept": "text/html,application/pdf,text/plain", "Accept-Encoding": "identity",
        })
        response = connection.getresponse()
        headers = {key.casefold(): value for key, value in response.getheaders()}
        body = response.read(MAX_WEB_BYTES + 1)
        if len(body) > MAX_WEB_BYTES:
            raise ProjectError("web source exceeds the 2 MiB retrieval limit")
        return response.status, headers, body
    except (OSError, http.client.HTTPException):
        raise ProjectError("web source could not be retrieved") from None
    finally:
        connection.close()


class _PageMetadata(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.in_title = False
        self.published_at: str | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        if tag == "title":
            self.in_title = True
        if tag == "meta" and str(values.get("property") or values.get("name") or "").casefold() in {
            "article:published_time", "date", "pubdate", "citation_publication_date", "dc.date"
        }:
            content = values.get("content")
            if isinstance(content, str):
                self.published_at = content[:128]

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)


def fetch_web_source(
    project: Project, url: str, *, start_offset: int = 0, max_chars: int = 16_000
) -> dict[str, object]:
    """Read a public source and preserve its excerpt and acquisition provenance."""
    if project.offline_mode:
        raise ProjectError("project offline mode blocks web context research")
    if type(start_offset) is not int or start_offset < 0:
        raise ProjectError("web source start_offset must be a nonnegative integer")
    if type(max_chars) is not int or not 1 <= max_chars <= MAX_CONTEXT_EVIDENCE_CHARS:
        raise ProjectError("web source max_chars must be between 1 and 20000")
    requested_url, current_url = url, url
    for _ in range(5):
        if project.offline_mode:
            raise ProjectError("project offline mode blocks web context research")
        status, headers, body = _request_public_url(current_url)
        if status in {301, 302, 303, 307, 308}:
            if not headers.get("location"):
                raise ProjectError("web source redirect has no destination")
            current_url = urljoin(current_url, headers["location"])
            continue
        if status != 200:
            raise ProjectError(f"web source returned HTTP {status}")
        break
    else:
        raise ProjectError("web source exceeded the redirect limit")
    if headers.get("content-encoding", "identity").casefold() not in {"identity", ""}:
        raise ProjectError("web source returned an unsupported compressed representation")
    content_type = headers.get("content-type", "").split(";", 1)[0].strip().casefold()
    metadata = _PageMetadata()
    extraction_truncated = False
    if content_type == "application/pdf":
        import pymupdf

        try:
            with pymupdf.open(stream=body, filetype="pdf") as pdf:
                if pdf.needs_pass:
                    raise ProjectError("web PDF requires a password")
                texts = []
                length = 0
                for page in pdf:
                    value = page.get_text()
                    texts.append(value[:MAX_WEB_EXTRACTED_CHARS - length])
                    length += len(texts[-1])
                    if length >= MAX_WEB_EXTRACTED_CHARS or len(texts) >= 200:
                        extraction_truncated = True
                        break
                text = "\n".join(texts)
                title = (pdf.metadata or {}).get("title") or current_url
        except (RuntimeError, ValueError):
            raise ProjectError("web PDF could not be read") from None
    elif content_type in {"text/html", "application/xhtml+xml", "text/plain"}:
        decoded = body.decode("utf-8", errors="replace")
        if content_type == "text/plain":
            text, title = decoded, current_url
        else:
            parser = _CanonicalHtmlParser()
            parser.feed(decoded)
            parser.close()
            text = parser.text()
            metadata.feed(decoded)
            title = " ".join("".join(metadata.title_parts).split()) or current_url
    else:
        raise ProjectError("web context supports HTML, plain text and PDF sources")
    if not text.strip():
        raise ProjectError("web source has no readable text; no context evidence was created")
    if start_offset >= len(text):
        raise ProjectError("web source start_offset exceeds the extracted text")
    end_offset = min(len(text), start_offset + max_chars)
    excerpt = text[start_offset:end_offset]
    retrieved_at = datetime.now(UTC).isoformat()
    evidence_id = project.external_source_evidence(
        urlsplit(current_url).netloc, excerpt, retrieved_at=retrieved_at, url=current_url, title=str(title)[:500],
        retrieval_metadata={
            "acquisition": "public_web_fetch", "requested_url": requested_url,
            "content_sha256": hashlib.sha256(body).hexdigest(), "content_type": content_type,
            "start_offset": start_offset, "end_offset": end_offset,
            "published_at_as_reported": metadata.published_at,
            "text_truncated": end_offset < len(text) or extraction_truncated,
        },
    )
    return {
        "kind": "EXTERNAL_SOURCE", "evidence_id": evidence_id,
        "citation": project.render_citation(evidence_id),
        "title": str(title)[:500], "url": current_url, "requested_url": requested_url,
        "retrieved_at": retrieved_at, "published_at_as_reported": metadata.published_at,
        "content_sha256": hashlib.sha256(body).hexdigest(), "content_type": content_type,
        "start_offset": start_offset, "end_offset": end_offset,
        "text_truncated": end_offset < len(text) or extraction_truncated,
        "next_start_offset": end_offset if end_offset < len(text) else None,
        "excerpt": excerpt,
        "source_role": "external_context_not_primary_document",
    }


def research_web_context(
    project: Project, query: str, preset_name: str, *, as_of_date: str | None = None
) -> dict[str, object]:
    """Discover and inspect contextual sources with the configured OpenAI preset."""
    from .execution import execute_model_preset
    from .providers import OpenAIWebSearchProvider, WebResearchResponse
    from .secrets import ReferenceSecretStore

    if project.offline_mode:
        raise ProjectError("project offline mode blocks web context research")
    if not isinstance(query, str) or not query.strip() or len(query) > 1_000:
        raise ProjectError("web context query must contain 1..1000 characters of public metadata")
    if as_of_date is not None:
        try:
            date.fromisoformat(as_of_date)
        except (TypeError, ValueError):
            raise ProjectError("web context as_of_date must use YYYY-MM-DD") from None
    if project.model_preset(preset_name)["provider_id"] != "openai":
        raise ModelProviderError(
            "web discovery currently requires a saved OpenAI model with native web search; "
            "direct fetch_web_source remains available with other providers"
        )

    def provider_factory(endpoint, reference, provider_id):
        if provider_id != "openai" or not reference:
            raise ModelProviderError("web discovery requires the selected authenticated OpenAI preset")
        return OpenAIWebSearchProvider(
            endpoint=endpoint, credential_reference=reference,
            secret_store=ReferenceSecretStore(project), timeout_seconds=90,
        )

    result = execute_model_preset(
        project, preset_name,
        f"Public research query: {query}\nHistorical reference date: {as_of_date or 'not established'}. "
        "Build the relevant public background, using this query as starting metadata rather "
        "than limiting research to finding copies of the document. For a document analysis, "
        "research and report in this order: (1) the author/institution's identity, public "
        "trajectory, affiliations and relevant publicly documented positions; (2) the "
        "production and circulation event, genre and audiences; (3) the pertinent "
        "sociopolitical setting and unresolved gaps. Put links to author/institution "
        "background first, followed by production-event sources. Document copies and "
        "launch announcements alone cannot establish an actor's background. Prefer dated official, "
        "institutional and author-published sources. For an identifiable public author or "
        "institution, investigate relevant public trajectory, affiliations and documented "
        "positions as well as the production event; launch metadata alone is not a biography "
        "or sociopolitical context. Prioritize complementary sources for those roles and "
        "identify important unresolved context. Distinguish historical facts from later "
        "developments. Return a concise discovery summary with source links. Do not classify "
        "the attached document: its text is not part of this search.",
        system_prompt="Use web search to discover verifiable contextual sources. Never invent "
        "a source, quotation, publication date or author identity. Search results are leads "
        "to inspect, not substitute evidence for a primary-document interpretation.",
        provider_factory=provider_factory,
    )
    response = result.response
    if not isinstance(response, WebResearchResponse):
        raise ModelProviderError("web discovery response lacks source provenance")
    inspected, failures = [], []
    for source in response.web_sources[:6]:
        try:
            inspected.append(fetch_web_source(project, source["url"]))
        except ProjectError as error:
            failures.append({"url": source["url"], "error": str(error)[:240]})
        if len(inspected) >= 2:
            break
    return {
        "query": query, "as_of_date": as_of_date, "discovery_only": not bool(inspected),
        "summary": response.text[:10_000], "sources": list(response.web_sources),
        "inspected_sources": inspected, "inspection_failures": failures,
        "model_execution_id": result.execution_id,
        "source_discipline": "The summary is a discovery lead. Ground contextual facts in the inspected excerpts, preserving dates and source perspective. Use fetch_web_source for additional sources or intervals.",
    }


def get_external_context(project: Project, evidence_id: str) -> dict[str, object]:
    """Reuse a saved contextual source without treating supplied text as fetched."""
    row = project.db.execute("SELECT kind FROM evidence_artifacts WHERE id=?", (evidence_id,)).fetchone()
    if not row or row["kind"] != "EXTERNAL_SOURCE":
        raise ProjectError("external context evidence was not found")
    content = project.evidence_content(evidence_id)
    provenance = content.get("retrieval_metadata")
    provenance = provenance if isinstance(provenance, dict) else {}
    return content | {
        **{key: provenance.get(key) for key in (
            "content_sha256", "content_type", "start_offset", "end_offset", "published_at_as_reported"
        )},
        "kind": "EXTERNAL_SOURCE", "evidence_id": evidence_id,
        "citation": project.render_citation(evidence_id),
        "source_role": "external_context_not_primary_document",
        "acquisition": "public_web_fetch" if provenance.get("acquisition") == "public_web_fetch" else "user_supplied",
    }
