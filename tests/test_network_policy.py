import socket
import urllib.request

import pytest

from discourselens import network_policy


@pytest.mark.parametrize(
    ("endpoint", "proxy_scheme", "request_host"),
    [
        ("http://127.0.0.1:11434/v1", "http", "127.0.0.1:11434"),
        ("https://localhost:11434/v1", "https", "localhost:11434"),
        ("http://[::1]:11434/v1", "all", "[::1]:11434"),
    ],
)
def test_loopback_exception_fails_closed_when_urllib_would_use_proxy(
    monkeypatch, endpoint, proxy_scheme, request_host
):
    proxy_checks: list[str] = []
    monkeypatch.setattr(
        network_policy,
        "getproxies",
        lambda: {proxy_scheme: "http://proxy.invalid:3128"},
    )
    monkeypatch.setattr(
        network_policy,
        "proxy_bypass",
        lambda host: proxy_checks.append(host) or False,
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: pytest.fail("network policy must not resolve DNS"),
    )

    assert not network_policy.is_loopback_endpoint(endpoint)
    assert proxy_checks == [request_host]


def test_explicit_no_proxy_bypass_preserves_user_selected_local_model_exception(monkeypatch):
    monkeypatch.setattr(
        network_policy,
        "getproxies",
        lambda: {"http": "http://proxy.invalid:3128"},
    )
    monkeypatch.setattr(
        network_policy,
        "proxy_bypass",
        lambda host: host == "localhost:11434",
    )

    assert network_policy.is_loopback_endpoint("http://localhost:11434/v1")


@pytest.mark.parametrize(
    ("no_proxy", "allowed"),
    [("::1", False), ("[::1]", True)],
)
def test_ipv6_proxy_bypass_matches_urllib_request_authority(
    monkeypatch, no_proxy, allowed
):
    proxies = {"http": "http://proxy.invalid:3128", "no": no_proxy}
    proxy_checks: list[str] = []
    monkeypatch.setattr(network_policy, "getproxies", lambda: proxies)
    proxy_bypass_name = "proxy_bypass_environment"
    environment_proxy_bypass = getattr(urllib.request, proxy_bypass_name)
    monkeypatch.setattr(
        network_policy,
        "proxy_bypass",
        lambda host: proxy_checks.append(host)
        or environment_proxy_bypass(host, proxies),
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: pytest.fail("network policy must not resolve DNS"),
    )

    assert network_policy.is_loopback_endpoint("http://[::1]:11434/v1") is allowed
    assert proxy_checks == ["[::1]:11434"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:1234/v1",
        "http://[::1]:8000/v1",
    ],
)
def test_explicit_loopback_model_endpoints_remain_allowed_without_proxy(
    monkeypatch, endpoint
):
    monkeypatch.setattr(network_policy, "getproxies", dict)
    monkeypatch.setattr(
        network_policy,
        "proxy_bypass",
        lambda host: pytest.fail("proxy bypass must not be checked without a proxy"),
    )
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: pytest.fail("network policy must not resolve DNS"),
    )

    assert network_policy.is_loopback_endpoint(endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://[not-ipv6]/v1",
        "http://[::1]external/v1",
        "http://@localhost:11434/v1",
        "http://:@localhost:11434/v1",
        "http://user:password@localhost:11434/v1",
        "http://localhost:invalid/v1",
        "http://localhost:65536/v1",
        "http://%6cocalhost:11434/v1",
        "http://localhost%2eattacker.test:11434/v1",
        "http://127.1:11434/v1",
        "http://2130706433:11434/v1",
        "http://local\nhost:11434/v1",
        " http://localhost:11434/v1",
        "http://localhost:11434/v1?",
        "http://localhost:11434/v1#",
    ],
)
def test_malformed_or_credentialed_endpoint_is_not_treated_as_loopback(
    monkeypatch, endpoint
):
    monkeypatch.setattr(network_policy, "getproxies", dict)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: pytest.fail("network policy must not resolve DNS"),
    )

    assert not network_policy.is_loopback_endpoint(endpoint)
