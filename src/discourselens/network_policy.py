"""Small, deterministic network-policy helpers for local-first execution."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse
from urllib.request import Request, getproxies, proxy_bypass


def _explicit_loopback_request(endpoint: str | None) -> tuple[str, str] | None:
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or "?" in endpoint
        or "#" in endpoint
    ):
        return None
    # urllib.parse removes a few ASCII control characters while splitting URLs.
    # Reject those before parsing so a malformed authority cannot be normalized
    # into ``localhost`` here while another consumer interprets it differently.
    if endpoint != endpoint.strip() or any(
        ord(character) < 0x20 or ord(character) == 0x7F for character in endpoint
    ):
        return None
    try:
        parsed = urlparse(endpoint)
    except ValueError:
        # Malformed bracketed authorities can fail inside urlparse itself.
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    try:
        hostname = parsed.hostname
        # ProxyHandler checks ``Request.host`` (authority, including IPv6
        # brackets and an explicit port), not the normalized hostname. Keep
        # that exact representation for the bypass check below so the policy
        # and the eventual urllib request agree about whether the proxy is used.
        request_host = Request(endpoint).host
        # Accessing ``port`` validates that an explicit port is numeric and in range.
        _ = parsed.port
    except ValueError:
        return None
    if hostname is None or request_host is None:
        return None
    # urlparse accepts junk after a bracketed IPv6 host (for example,
    # ``http://[::1]external``) and still reports ``::1`` as the hostname.
    # That is not a valid authority and may be interpreted differently by the
    # eventual HTTP client.
    if parsed.netloc.startswith("["):
        closing_bracket = parsed.netloc.find("]")
        if closing_bracket < 0:
            return None
        authority_suffix = parsed.netloc[closing_bracket + 1 :]
        if authority_suffix and not authority_suffix.startswith(":"):
            return None
    if hostname.casefold() == "localhost":
        return parsed.scheme, request_host
    try:
        is_loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        is_loopback = False
    return (parsed.scheme, request_host) if is_loopback else None


def is_explicit_loopback_endpoint(endpoint: str | None) -> bool:
    """Return true for a syntactically valid, literal localhost/loopback URL.

    This does not consult DNS or proxy configuration. Private-LAN addresses are
    intentionally not treated as local because they can still disclose project
    material.
    """
    return _explicit_loopback_request(endpoint) is not None


def is_loopback_endpoint(endpoint: str | None) -> bool:
    """Return true only when an explicit loopback URL will avoid environment proxies.

    DNS is never consulted: offline mode should not turn an ostensibly local model
    request into a network lookup. An active environment proxy must explicitly
    bypass the local hostname, otherwise urllib could send the request off-host.
    """
    loopback_request = _explicit_loopback_request(endpoint)
    if loopback_request is None:
        return False
    scheme, request_host = loopback_request
    try:
        proxies = getproxies()
        proxy_configured = bool(proxies.get(scheme) or proxies.get("all"))
        return not proxy_configured or proxy_bypass(request_host)
    except (OSError, ValueError):
        # If the environment proxy configuration cannot be interpreted, do not
        # grant the local/offline exception based on the URL alone.
        return False
