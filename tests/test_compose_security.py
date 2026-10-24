import re
from pathlib import Path

import yaml


def test_default_compose_mcp_is_stdio_only_and_confined():
    compose_path = Path(__file__).parents[1] / "compose.yaml"
    config = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    service = config["services"]["mcp"]

    assert "ports" not in service
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["stdin_open"] is True
    assert service["tty"] is False
    assert (
        service["build"]["args"]["DISCOURSELENS_INSTALL_OCR"] == "${DISCOURSELENS_INSTALL_OCR:-0}"
    )
    assert (
        service["build"]["args"]["DISCOURSELENS_INSTALL_NLP"] == "${DISCOURSELENS_INSTALL_NLP:-0}"
    )
    expected_image = "${DISCOURSELENS_IMAGE:-discourselens:${COMPOSE_PROJECT_NAME}}"
    assert all(
        configured["image"] == expected_image
        for configured in config["services"].values()
    )
    assert service["volumes"] == [
        "${DISCOURSELENS_PROJECT_DIR:?Set an explicit project path}:/workspace/project"
    ]
    assert service["command"] == ["discourselens-mcp", "--project", "/workspace/project"]


def test_read_only_compose_mcp_has_no_ports_or_project_write_access():
    compose_path = Path(__file__).parents[1] / "compose.yaml"
    config = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    service = config["services"]["mcp-readonly"]

    assert service["profiles"] == ["read-only"]
    assert "ports" not in service
    assert service["network_mode"] == "none"
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["stdin_open"] is True
    assert service["tty"] is False
    assert service["volumes"] == [
        {
            "type": "bind",
            "source": "${DISCOURSELENS_PROJECT_DIR:?Set an explicit project path}",
            "target": "/workspace/project",
            "read_only": True,
        }
    ]
    assert service["command"] == [
        "discourselens-mcp",
        "--project",
        "/workspace/project",
        "--read-only",
    ]


def test_chatgpt_tunnel_overlay_uses_pinned_sidecar_and_secret_files():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "compose.chatgpt-tunnel.yaml").read_text(encoding="utf-8"))
    tunnel = config["services"]["tunnel-client"]
    mcp = config["services"]["mcp-http"]

    assert tunnel["profiles"] == ["chatgpt-tunnel"]
    assert tunnel["image"].startswith("ghcr.io/openai/tunnel-client:v0.0.17@sha256:")
    assert re.fullmatch(r".+@sha256:[0-9a-f]{64}", tunnel["image"])
    assert tunnel["read_only"] is True
    assert tunnel["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in tunnel["security_opt"]
    assert tunnel["user"] == "${DISCOURSELENS_UID:-10001}:${DISCOURSELENS_GID:-10001}"
    assert "docker.sock" not in str(tunnel)
    assert any("http://mcp-http:8001/mcp" in item for item in tunnel["command"])
    assert "file:/run/secrets/openai-runtime-api-key" in str(tunnel["command"])
    assert "file:/run/secrets/mcp-http-token" in str(tunnel["command"])
    assert mcp["environment"]["DISCOURSELENS_MCP_HTTP_TOKEN"] == ""
    assert mcp["environment"]["DISCOURSELENS_MCP_HTTP_TOKEN_FILE"] == (
        "/run/secrets/mcp-http-token"
    )
    assert set(tunnel["secrets"]) == {
        "openai-runtime-api-key",
        "mcp-http-token",
    }
    assert "--mcp.discovery-extra-headers=Authorization: file:/run/secrets/mcp-http-token" in (
        tunnel["command"]
    )
    assert tunnel["depends_on"]["mcp-http"]["condition"] == "service_healthy"


def test_mcp_quick_tunnel_targets_only_the_token_protected_origin():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "compose.mcp-cloudflare-preview.yaml").read_text())
    service = config["services"]["cloudflare-mcp-origin"]

    assert service["profiles"] == ["mcp-cloudflare-preview"]
    assert re.fullmatch(r".+@sha256:[0-9a-f]{64}", service["image"])
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "http://mcp-http:8001" in service["command"]
    assert service["depends_on"]["mcp-http"]["condition"] == "service_healthy"
    assert "ports" not in service


def test_http_mcp_compose_waits_for_server_listener_health():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "compose.mcp-http.yaml").read_text(encoding="utf-8"))
    service = config["services"]["mcp-http"]

    assert service["healthcheck"]["interval"] == "15s"
    assert service["healthcheck"]["start_period"] == "15s"
    assert "socket.create_connection" in service["healthcheck"]["test"][-1]


def test_compose_services_default_to_non_root_without_docker_socket():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "compose.yaml").read_text(encoding="utf-8"))

    for service in config["services"].values():
        assert service["user"] == "${DISCOURSELENS_UID:-1000}:${DISCOURSELENS_GID:-1000}"
        assert "docker.sock" not in str(service)

    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "USER 10001:10001" in dockerfile


def test_compose_does_not_claim_shared_host_resources_or_external_networks():
    compose_path = Path(__file__).parents[1] / "compose.yaml"
    config = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    for service in config["services"].values():
        assert "ports" not in service
        assert "container_name" not in service
        assert service.get("network_mode") != "host"

    for network in config.get("networks", {}).values():
        assert not network.get("external", False)


def test_docker_image_does_not_declare_a_network_listener():
    root = Path(__file__).parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")

    assert not any(
        line.lstrip().upper().startswith("EXPOSE ") for line in dockerfile.splitlines()
    )


def test_docker_base_images_are_pinned_by_digest():
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text(encoding="utf-8")
    base_images = [
        next(part for part in line.split()[1:] if not part.startswith("--"))
        for line in dockerfile.splitlines()
        if line.lstrip().upper().startswith("FROM ")
    ]

    assert base_images
    assert all(
        re.fullmatch(r".+@sha256:[0-9a-f]{64}", image) is not None for image in base_images
    )


def test_docker_build_context_excludes_research_data_and_credentials():
    dockerignore = Path(__file__).parents[1] / ".dockerignore"
    excluded = set(dockerignore.read_text(encoding="utf-8").splitlines())

    assert {".env", ".env.*", ".discourselens", "secrets/"} <= excluded
    assert {
        ".aws/",
        ".ssh/",
        ".config/gcloud/",
        "**/credentials.json",
        "**/*credential*.json",
        "**/service-account*.json",
        "**/*-service-account.json",
        "**/application_default_credentials.json",
        "**/id_rsa",
        "**/id_dsa",
        "**/id_ecdsa",
        "**/id_ed25519",
        "**/id_ecdsa_sk",
        "**/id_ed25519_sk",
        "**/*.ppk",
    } <= excluded
    assert not {
        "*.json",
        "**/*.json",
        "src/",
        "docs/",
        "schemas/",
        "pyproject.toml",
        "uv.lock",
    } & excluded
    assert {
        "*.pdf",
        "*.docx",
        "*.txt",
        "*.parquet",
        "*.sqlite",
        "*.key",
        "*.pem",
        "corpus/",
        "research/",
        "projects/",
    } <= excluded


def test_direct_docker_example_uses_host_uid_for_project_bind_mount():
    readme = Path(__file__).parents[1] / "README.md"

    assert 'docker run --rm --user "$(id -u):$(id -g)" \\' in readme.read_text(
        encoding="utf-8"
    )
