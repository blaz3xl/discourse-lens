import json
from email.message import Message
from typing import cast
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from discourselens import Project
from discourselens.errors import AuthenticationError, ModelProviderError, ModelResponseError
from discourselens.execution import (
    build_reranking_gateway,
    credential_reference_fingerprint,
)
from discourselens.models import (
    CredentialReference,
    ModelGateway,
    RerankingProvider,
    RerankResult,
)
from discourselens.profiles import ControlledCdaProfile
from discourselens.provider_catalog import supported_providers
from discourselens.providers import CohereRerankingProvider, JinaRerankingProvider


class _SecretStore:
    def get(self, reference: str) -> str:
        assert reference == "JINA_TOKEN"
        return "fixture-secret"


class _CohereSecretStore:
    def get(self, reference: str) -> str:
        assert reference == "COHERE_TOKEN"
        return "cohere-fixture-secret"


def test_jina_reranker_sends_only_explicit_candidates_and_returns_provider_order():
    response = MagicMock()
    response.read.return_value = json.dumps(
        {
            "results": [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.42},
            ]
        }
    ).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = JinaRerankingProvider("JINA_TOKEN", _SecretStore())

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        results = provider.rerank(
            "jina-reranker-v3.5",
            "agency construction",
            ["First scoped passage.", "Second scoped passage."],
            top_n=2,
        )

    assert isinstance(provider, RerankingProvider)
    assert provider.provider_id == "jina-reranker"
    assert results[0].index == 1 and results[0].relevance_score == 0.91
    assert results[1].index == 0 and results[1].relevance_score == 0.42
    request = opened.call_args.args[0]
    assert isinstance(request, Request)
    assert request.full_url == "https://api.jina.ai/v1/rerank"
    assert request.get_header("Authorization") == "Bearer fixture-secret"
    assert request.data is not None
    assert json.loads(cast(bytes, request.data).decode("utf-8")) == {
        "model": "jina-reranker-v3.5",
        "query": "agency construction",
        "documents": ["First scoped passage.", "Second scoped passage."],
        "top_n": 2,
        "return_documents": False,
    }


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 4, "relevance_score": 0.8}],
        [{"index": True, "relevance_score": 0.8}],
        [{"index": 0.0, "relevance_score": 0.8}],
        [{"index": "0", "relevance_score": 0.8}],
        [{"index": 0, "relevance_score": True}],
        [{"index": 0, "relevance_score": "0.8"}],
        [{"index": 0, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.7}],
        [{"index": 0, "relevance_score": float("nan")}],
        [{"index": 0, "relevance_score": float("inf")}],
        [{"index": 0, "relevance_score": 10**400}],
        [{"index": 0, "relevance_score": 0.8}, {"index": 1, "relevance_score": 0.7}],
    ],
)
def test_jina_reranker_rejects_invalid_indices_scores_or_excess_results(results):
    response = MagicMock()
    response.read.return_value = json.dumps({"results": results}).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = JinaRerankingProvider("JINA_TOKEN", _SecretStore())

    with (
        patch("discourselens.providers._open_provider_request", return_value=manager),
        pytest.raises(ModelResponseError),
    ):
        provider.rerank("model", "query", ["only candidate"], top_n=1)


def test_jina_reranker_validates_candidates_before_network_call():
    provider = JinaRerankingProvider("JINA_TOKEN", _SecretStore())
    with (
        patch("discourselens.providers._open_provider_request") as opened,
        pytest.raises(ModelProviderError, match="top_n"),
    ):
        provider.rerank("model", "query", ["one"], top_n=2)
    opened.assert_not_called()


def test_jina_reranker_maps_authentication_failure_without_secret_leak():
    provider = JinaRerankingProvider("JINA_TOKEN", _SecretStore())
    failure = HTTPError("https://api.jina.ai/v1/rerank", 401, "unauthorized", Message(), None)
    with (
        patch("discourselens.providers._open_provider_request", side_effect=failure),
        pytest.raises(AuthenticationError, match="Jina rejected") as error,
    ):
        provider.rerank("model", "query", ["candidate"])
    assert "fixture-secret" not in str(error.value)


def test_provider_catalog_declares_jina_reranking_only():
    entry = {item.provider_id: item for item in supported_providers()}["jina-reranker"]
    assert entry.reranking
    assert not entry.generation
    assert not entry.embeddings


def test_cohere_reranker_uses_v2_contract_and_returns_only_candidate_ranks():
    response = MagicMock()
    response.read.return_value = json.dumps(
        {
            "results": [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.42},
            ]
        }
    ).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = CohereRerankingProvider("COHERE_TOKEN", _CohereSecretStore())

    with patch("discourselens.providers._open_provider_request", return_value=manager) as opened:
        results = provider.rerank(
            "rerank-v4.0-pro", "agency construction", ["Passage A", "Passage B"], top_n=2
        )

    assert isinstance(provider, RerankingProvider)
    assert provider.provider_id == "cohere-reranker"
    assert [item.index for item in results] == [1, 0]
    assert [item.relevance_score for item in results] == [0.91, 0.42]
    request = opened.call_args.args[0]
    assert isinstance(request, Request)
    assert request.full_url == "https://api.cohere.com/v2/rerank"
    assert request.get_header("Authorization") == "Bearer cohere-fixture-secret"
    assert request.data is not None
    assert json.loads(cast(bytes, request.data).decode("utf-8")) == {
        "model": "rerank-v4.0-pro",
        "query": "agency construction",
        "documents": ["Passage A", "Passage B"],
        "top_n": 2,
        "max_tokens_per_doc": 4096,
    }


def test_cohere_reranker_rejects_oversized_scope_before_network_call():
    provider = CohereRerankingProvider("COHERE_TOKEN", _CohereSecretStore())
    with (
        patch("discourselens.providers._open_provider_request") as opened,
        pytest.raises(ModelProviderError, match="at most 1000 candidates"),
    ):
        provider.rerank("rerank-v3.5", "query", ["candidate"] * 1_001)
    opened.assert_not_called()


@pytest.mark.parametrize(
    "results",
    [
        [{"index": 2, "relevance_score": 0.8}],
        [{"index": True, "relevance_score": 0.8}],
        [{"index": 0.0, "relevance_score": 0.8}],
        [{"index": "0", "relevance_score": 0.8}],
        [{"index": 0, "relevance_score": True}],
        [{"index": 0, "relevance_score": "0.8"}],
        [{"index": 0, "relevance_score": float("nan")}],
        [{"index": 0, "relevance_score": float("inf")}],
        [{"index": 0, "relevance_score": 10**400}],
        [{"index": 0, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.7}],
    ],
)
def test_cohere_reranker_rejects_invalid_rank_response(results):
    response = MagicMock()
    response.read.return_value = json.dumps({"results": results}).encode()
    manager = MagicMock()
    manager.__enter__.return_value = response
    provider = CohereRerankingProvider("COHERE_TOKEN", _CohereSecretStore())

    with (
        patch("discourselens.providers._open_provider_request", return_value=manager),
        pytest.raises(ModelResponseError),
    ):
        provider.rerank("rerank-v4.0-pro", "query", ["only candidate"], top_n=1)


def test_cohere_reranker_maps_invalid_token_without_secret_leak():
    provider = CohereRerankingProvider("COHERE_TOKEN", _CohereSecretStore())
    failure = HTTPError("https://api.cohere.com/v2/rerank", 498, "invalid token", Message(), None)
    with (
        patch("discourselens.providers._open_provider_request", side_effect=failure),
        pytest.raises(AuthenticationError, match="Cohere rejected") as error,
    ):
        provider.rerank("rerank-v3.5", "query", ["candidate"])
    assert "cohere-fixture-secret" not in str(error.value)


def test_provider_catalog_declares_cohere_reranking_only():
    entry = {item.provider_id: item for item in supported_providers()}["cohere-reranker"]
    assert entry.reranking
    assert not entry.generation
    assert not entry.embeddings


def test_cohere_reranker_is_available_to_validated_profiles():
    profile = ControlledCdaProfile(
        frameworks=["fairclough-text"],
        reranking_provider_id="cohere-reranker",
        reranking_credential_alias="lab",
        reranking_model="rerank-v4.0-pro",
    )
    profile.validate_method_contract()


def test_cohere_reranking_gateway_pins_named_credential_and_default_endpoint(tmp_path):
    project = Project.init(tmp_path / "project")
    project.save_credential_reference(CredentialReference("cohere-reranker", "lab", "COHERE_TOKEN"))

    gateway = build_reranking_gateway(project, "cohere-reranker", "lab")

    provider = gateway.rerankers["cohere-reranker"]
    assert isinstance(provider, CohereRerankingProvider)
    assert provider.endpoint == "https://api.cohere.com"
    assert gateway.credential_reference_fingerprints["cohere-reranker"] == (
        credential_reference_fingerprint(project, "cohere-reranker", "lab")
    )


def test_model_gateway_routes_reranking_only_to_explicit_registered_provider():
    class _FixtureReranker:
        provider_id = "fixture-reranker"

        def rerank(self, model, query, documents, *, top_n=None):
            assert (model, query, documents, top_n) == ("r", "q", ["candidate"], 1)
            return [RerankResult(0, 0.7)]

    gateway = ModelGateway({}, rerankers={"fixture-reranker": _FixtureReranker()})
    assert gateway.rerank("fixture-reranker", "r", "q", ["candidate"], top_n=1) == [
        RerankResult(0, 0.7)
    ]
    with pytest.raises(ModelProviderError, match="no reranking provider registered"):
        gateway.rerank("unregistered", "r", "q", ["candidate"])


def test_project_reranking_gateway_resolves_named_credential_without_reading_secret(tmp_path):
    project = Project.init(tmp_path / "project")
    project.save_credential_reference(
        CredentialReference(
            "jina-reranker", "university", "JINA_UNIVERSITY", "https://rerank.example/v1"
        )
    )

    gateway = build_reranking_gateway(project, "jina-reranker", "university")

    provider = gateway.rerankers["jina-reranker"]
    assert isinstance(provider, JinaRerankingProvider)
    assert provider.credential_reference == "JINA_UNIVERSITY"
    assert provider.endpoint == "https://rerank.example/v1"
    assert gateway.credential_reference_fingerprints["jina-reranker"] == (
        credential_reference_fingerprint(project, "jina-reranker", "university")
    )
    project.close()


def test_reranking_gateway_rejects_remapped_credential_reference(tmp_path):
    project = Project.init(tmp_path / "project")
    project.save_credential_reference(
        CredentialReference("jina-reranker", "university", "JINA_UNIVERSITY")
    )
    snapshot_fingerprint = credential_reference_fingerprint(project, "jina-reranker", "university")
    project.save_credential_reference(
        CredentialReference("jina-reranker", "university", "JINA_REPLACEMENT")
    )

    with pytest.raises(ModelProviderError, match="refusing implicit credential substitution"):
        build_reranking_gateway(
            project,
            "jina-reranker",
            "university",
            expected_credential_reference_hash=snapshot_fingerprint,
        )


@pytest.mark.parametrize(
    "index,score",
    [
        (True, 0.9),
        (0.0, 0.9),
        ("0", 0.9),
        (0, True),
        (0, float("nan")),
        (0, float("inf")),
        (0, 10**400),
    ],
)
def test_scoped_reranker_rejects_coercive_index_or_invalid_score(index, score, tmp_path):
    from discourselens.scope_retrieval import rerank_framework_context, retrieve_framework_context

    project = Project.init(tmp_path / "project")
    _, revision_id = project.import_text(
        "selected.txt",
        "Migration discourse constructs institutional agency.\n\n"
        "Migration discourse represents collective actors.",
    )
    scope_id = project.snapshot([revision_id])
    retrieval = retrieve_framework_context(
        project,
        scope_id,
        [],
        phenomenon="migration discourse",
        observation="agency actors",
        query_texts=("migration discourse agency actors",),
    )
    assert len(retrieval.candidates) == 2

    class _UntrustedReranker:
        provider_id = "fixture-untrusted-reranker"

        def rerank(self, model, query, documents, *, top_n=None):
            assert documents == [candidate.text for candidate in retrieval.candidates]
            return [RerankResult(cast(int, index), cast(float, score))]

    reranker = _UntrustedReranker()
    with pytest.raises(ModelResponseError):
        rerank_framework_context(
            project,
            scope_id,
            retrieval,
            ModelGateway({}, rerankers={reranker.provider_id: reranker}),
            provider_id=reranker.provider_id,
            model="fixture-v1",
            query="migration discourse",
        )


def test_scoped_reranker_rejects_malformed_provider_result(tmp_path):
    from discourselens.scope_retrieval import rerank_framework_context, retrieve_framework_context

    project = Project.init(tmp_path / "project")
    _, revision_id = project.import_text("selected.txt", "Migration discourse constructs agency.")
    scope_id = project.snapshot([revision_id])
    retrieval = retrieve_framework_context(
        project,
        scope_id,
        [],
        phenomenon="migration discourse",
        observation="agency",
        query_texts=("migration discourse agency",),
    )
    assert retrieval.candidates

    class _MalformedReranker:
        provider_id = "fixture-malformed-reranker"

        def rerank(self, model, query, documents, *, top_n=None):
            return [None]

    reranker = _MalformedReranker()
    malformed_gateway = ModelGateway(
        {}, rerankers={reranker.provider_id: cast(RerankingProvider, reranker)}
    )
    with pytest.raises(ModelResponseError, match="malformed result"):
        rerank_framework_context(
            project,
            scope_id,
            retrieval,
            malformed_gateway,
            provider_id=reranker.provider_id,
            model="fixture-v1",
            query="migration discourse",
        )
