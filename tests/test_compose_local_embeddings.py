from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def test_local_embedding_service_is_private_persistent_and_resource_limited():
    config = yaml.safe_load((ROOT / "compose.ollama.yaml").read_text(encoding="utf-8"))
    service = config["services"]["ollama"]

    assert service["image"] == "ollama/ollama:0.34.0"
    assert "ports" not in service
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert service["deploy"]["resources"]["limits"] == {
        "cpus": "2.0",
        "memory": "2G",
    }
    assert service["environment"]["OLLAMA_NUM_PARALLEL"] == "1"
    assert service["environment"]["OLLAMA_CONTEXT_LENGTH"] == "2048"
    assert service["mem_limit"] == "2g"
    assert service["memswap_limit"] == "2g"
    assert "ollama-models:/root/.ollama" in service["volumes"]
    assert service["networks"]["local-embeddings"]["aliases"] == ["ollama"]


def test_web_and_http_mcp_share_only_the_dedicated_embedding_network():
    expected_network = {
        "external": True,
        "name": "discourselens-local-embeddings",
    }
    overlay = yaml.safe_load(
        (ROOT / "compose.local-embeddings.yaml").read_text(encoding="utf-8")
    )
    mcp_overlay = yaml.safe_load(
        (ROOT / "compose.mcp-local-embeddings.yaml").read_text(encoding="utf-8")
    )
    for config, service_name in (
        (overlay, "web"),
        (mcp_overlay, "mcp-http"),
    ):
        assert "local-embeddings" in config["services"][service_name]["networks"]
        assert config["networks"]["local-embeddings"] == expected_network
