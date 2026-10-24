import re

import pytest

from discourselens.errors import ProjectError
from discourselens.languages import (
    LanguageCapability,
    capability_statuses,
    evaluate_capabilities,
    inspect_local_backend,
    local_execution_preflight,
    local_language_capabilities,
    probe_backend_capabilities,
)
from discourselens.nlp import ParsedToken


class _ProbeBackend:
    def __init__(
        self,
        language: str = "ar",
        backend_id: str = "stanza",
        model_id: str = "fixture-ar",
        processor_names: tuple[str, ...] = ("tokenize", "lemma", "pos", "depparse", "ner"),
        empty_lemma: bool = False,
        invalid_dependency_tree: bool = False,
    ):
        self.language = language
        self.backend_id = backend_id
        self.model_id = model_id
        self.empty_lemma = empty_lemma
        self.invalid_dependency_tree = invalid_dependency_tree
        self.last_input: str | None = None
        self.pipeline = type(
            "Pipeline", (), {"processors": {name: object() for name in processor_names}}
        )()

    def parse(self, text: str) -> list[ParsedToken]:
        self.last_input = text
        return [
            ParsedToken(
                match.group(),
                "" if self.empty_lemma else match.group().casefold(),
                "NOUN",
                "Case=Nom",
                None if index == 0 else index if self.invalid_dependency_tree and index == 1 else 0,
                "root" if index == 0 else "dep",
                match.start(),
                match.end(),
            )
            for index, match in enumerate(re.finditer(r"\w+|[^\w\s]", text))
        ]


def test_unsupported_or_unassessed_language_capabilities_never_silently_run():
    for status in (
        "SUPPORTED",
        "LIMITED",
        "UNASSESSED",
        "UNSUPPORTED",
        "EXPERIMENTAL",
        "UNKNOWN",
    ):
        result = evaluate_capabilities({"dependency_parse": status}, {"dependency_parse"})
        assert result.runnable is (status == "SUPPORTED")
        expected_limitations = (
            () if status == "SUPPORTED" else (f"dependency_parse: {status}",)
        )
        assert result.limitations == expected_limitations
    assert not evaluate_capabilities({}, {"dependency_parse"}).runnable


def test_assurance_does_not_upgrade_experimental_operational_status():
    capability = LanguageCapability(
        language="ar",
        capability="dependency_parse",
        backend="stanza",
        model="fixture-ar",
        operational_status="EXPERIMENTAL",
        assurance_status="TESTED",
    )

    statuses = capability_statuses((capability,))

    assert statuses == {"dependency_parse": "EXPERIMENTAL"}
    decision = evaluate_capabilities(statuses, {"dependency_parse"})
    assert not decision.runnable
    assert decision.limitations == ("dependency_parse: EXPERIMENTAL",)


def test_supported_capability_with_unassessed_assurance_remains_runnable_with_limitation(
    monkeypatch,
):
    capability = LanguageCapability(
        language="en",
        capability="dependency_parse",
        backend="spacy",
        model="fixture-en",
        operational_status="SUPPORTED",
        assurance_status="UNASSESSED",
    )
    monkeypatch.setattr(
        "discourselens.languages.inspect_local_backend",
        lambda *_args, **_kwargs: (capability,),
    )

    preflight = local_execution_preflight(
        ("en",),
        {"dependency_parse"},
        routes={"en": {"backend": "spacy", "model": "fixture-en"}},
    )["en"]

    assert preflight["statuses"]["dependency_parse"] == "SUPPORTED"
    assert preflight["runnable"]
    assert preflight["limitations"] == ["dependency_parse assurance: UNASSESSED"]


def test_unconfigured_arabic_nlp_is_explicitly_unassessed_but_raw_text_tools_remain_available():
    capabilities = local_language_capabilities("ar")
    statuses = capability_statuses(capabilities)
    by_name = {item.capability: item for item in capabilities}

    assert statuses["exact_search"] == "SUPPORTED"
    assert by_name["exact_search"].assurance_status == "TESTED"
    assert statuses["kwic"] == "SUPPORTED"
    assert statuses["dependency_parse"] == "UNASSESSED"
    assert by_name["dependency_parse"].assurance_status == "UNASSESSED"
    assert not evaluate_capabilities(statuses, {"dependency_parse"}).runnable


def test_local_preflight_never_treats_unassessed_nlp_as_runnable():
    preflight = local_execution_preflight(("ar",), {"dependency_parse"})
    assert not preflight["ar"]["runnable"]
    assert preflight["ar"]["limitations"] == ["dependency_parse: UNASSESSED"]


def test_explicit_backend_probe_reports_observed_arabic_backend_capabilities():
    backend = _ProbeBackend()
    capabilities = probe_backend_capabilities(backend)
    statuses = capability_statuses(capabilities)
    by_name = {item.capability: item for item in capabilities}

    assert by_name["tokenization"].backend == "stanza"
    assert by_name["tokenization"].model == "fixture-ar"
    assert by_name["tokenization"].assurance_status == "TESTED"
    assert statuses["lemma"] == "SUPPORTED"
    assert statuses["pos"] == "SUPPORTED"
    assert statuses["morphology"] == "SUPPORTED"
    assert statuses["dependency_parse"] == "SUPPORTED"
    assert statuses["sentence_segmentation"] == "SUPPORTED"
    assert statuses["named_entities"] == "SUPPORTED"
    assert backend.last_input is not None and any(
        "\u0600" <= char <= "\u06ff" for char in backend.last_input
    )


@pytest.mark.parametrize(
    ("language", "expected_marker"),
    [("en", "Officials"), ("es", "autoridades"), ("ar", "السلطات")],
)
def test_local_nlp_probe_uses_a_fixture_in_the_backend_language(language, expected_marker):
    backend = _ProbeBackend(language=language)

    probe_backend_capabilities(backend)

    assert backend.last_input is not None
    assert expected_marker in backend.last_input


def test_unknown_language_probe_does_not_reuse_english_sample():
    backend = _ProbeBackend(language="fr")

    capabilities = probe_backend_capabilities(backend)

    statuses = capability_statuses(capabilities)
    assert statuses["exact_search"] == "SUPPORTED"
    assert statuses["dependency_parse"] == "UNASSESSED"
    assert backend.last_input is None


def test_spacy_tokenizer_alone_does_not_claim_sentence_segmentation():
    backend = _ProbeBackend(language="es", backend_id="spacy", processor_names=("tokenize",))

    capabilities = probe_backend_capabilities(backend)

    status = {item.capability: item.operational_status for item in capabilities}
    assert status["tokenization"] == "SUPPORTED"
    assert status["sentence_segmentation"] == "UNSUPPORTED"


def test_backend_component_without_probe_annotations_is_limited_not_supported():
    backend = _ProbeBackend(empty_lemma=True)

    capabilities = probe_backend_capabilities(backend)

    lemma = next(item for item in capabilities if item.capability == "lemma")
    assert lemma.operational_status == "LIMITED"
    assert lemma.limitation is not None and "did not expose" in lemma.limitation


def test_partial_token_coverage_cannot_enable_token_dependent_operations(monkeypatch):
    class PartialBackend(_ProbeBackend):
        def parse(self, text: str) -> list[ParsedToken]:
            self.last_input = text
            match = re.search(r"\w+", text)
            assert match is not None
            return [
                ParsedToken(
                    match.group(),
                    match.group().casefold(),
                    "NOUN",
                    "Case=Nom",
                    None,
                    "root",
                    match.start(),
                    match.end(),
                )
            ]

    capabilities = probe_backend_capabilities(PartialBackend())
    statuses = capability_statuses(capabilities)
    by_name = {item.capability: item for item in capabilities}

    assert statuses["tokenization"] == "LIMITED"
    assert statuses["pos"] == "LIMITED"
    assert statuses["dependency_parse"] == "LIMITED"
    assert "complete token coverage" in (by_name["pos"].limitation or "")

    monkeypatch.setenv("DL_FIXTURE_HOME", "unused-fixture-dir")
    monkeypatch.setattr(
        "discourselens.languages.inspect_local_backend", lambda *args, **kwargs: capabilities
    )
    preflight = local_execution_preflight(
        ("ar",),
        {"pos"},
        routes={
            "ar": {
                "backend": "stanza",
                "model": "fixture-ar",
                "model_dir_env": "DL_FIXTURE_HOME",
            }
        },
    )["ar"]

    assert not preflight["runnable"]
    assert preflight["statuses"]["pos"] == "LIMITED"


def test_empty_probe_output_cannot_enable_sentence_or_named_entity_skills():
    class EmptyOutputBackend(_ProbeBackend):
        def parse(self, text: str) -> list[ParsedToken]:
            self.last_input = text
            return []

    statuses = capability_statuses(probe_backend_capabilities(EmptyOutputBackend()))

    assert statuses["tokenization"] == "LIMITED"
    assert statuses["sentence_segmentation"] == "LIMITED"
    assert statuses["named_entities"] == "LIMITED"
    assert not evaluate_capabilities(statuses, {"named_entities"}).runnable


def test_dependency_probe_rejects_cyclic_heads():
    backend = _ProbeBackend(invalid_dependency_tree=True)

    capabilities = probe_backend_capabilities(backend)

    dependency = next(item for item in capabilities if item.capability == "dependency_parse")
    assert dependency.operational_status == "LIMITED"


def test_dependency_probe_rejects_boolean_head_indices():
    class BooleanHeadBackend(_ProbeBackend):
        def parse(self, text: str) -> list[ParsedToken]:
            tokens = super().parse(text)
            token = tokens[2]
            tokens[2] = ParsedToken(
                token.surface,
                token.lemma,
                token.pos,
                token.morphology,
                True,
                token.dependency_relation,
                token.char_start,
                token.char_end,
            )
            return tokens

    statuses = capability_statuses(probe_backend_capabilities(BooleanHeadBackend()))

    assert statuses["tokenization"] == "SUPPORTED"
    assert statuses["dependency_parse"] == "LIMITED"


def test_tokenization_probe_rejects_boolean_character_offsets():
    class BooleanOffsetBackend(_ProbeBackend):
        def parse(self, text: str) -> list[ParsedToken]:
            tokens = super().parse(text)
            token = tokens[0]
            tokens[0] = ParsedToken(
                token.surface[:1],
                token.lemma[:1],
                token.pos,
                token.morphology,
                None,
                "root",
                0,
                1,
            )
            tokens.insert(
                1,
                ParsedToken(
                    token.surface[1:],
                    token.lemma[1:],
                    token.pos,
                    token.morphology,
                    0,
                    "dep",
                    True,
                    2,
                ),
            )
            return tokens

    statuses = capability_statuses(probe_backend_capabilities(BooleanOffsetBackend()))

    assert statuses["tokenization"] == "LIMITED"


def test_absent_backend_component_is_reported_unsupported():
    backend = _ProbeBackend(processor_names=("tokenize", "pos", "depparse"))

    capabilities = probe_backend_capabilities(backend)

    by_name = {item.capability: item for item in capabilities}
    assert by_name["lemma"].operational_status == "UNSUPPORTED"
    assert "does not include" in (by_name["lemma"].limitation or "")
    assert by_name["sentence_segmentation"].operational_status == "SUPPORTED"


def test_observed_annotations_remain_eligible_when_component_inventory_is_unavailable():
    backend = _ProbeBackend()
    backend.__dict__.pop("pipeline")

    statuses = capability_statuses(probe_backend_capabilities(backend))

    assert statuses["tokenization"] == "SUPPORTED"
    assert statuses["lemma"] == "SUPPORTED"
    assert statuses["pos"] == "SUPPORTED"
    assert statuses["morphology"] == "SUPPORTED"
    assert statuses["dependency_parse"] == "SUPPORTED"


def test_explicitly_empty_backend_component_inventory_reports_unsupported():
    backend = _ProbeBackend(processor_names=())

    statuses = capability_statuses(probe_backend_capabilities(backend))

    assert statuses["tokenization"] == "UNSUPPORTED"
    assert statuses["lemma"] == "UNSUPPORTED"
    assert statuses["dependency_parse"] == "UNSUPPORTED"


def test_explicit_stanza_inspection_loads_only_the_selected_local_model(monkeypatch, tmp_path):
    import discourselens.nlp as nlp_module

    requested: list[tuple[str, str, str]] = []

    def load_stanza(language: str, model: str, model_dir: str):
        requested.append((language, model, model_dir))
        return _ProbeBackend(language, "stanza", model)

    monkeypatch.setattr(nlp_module, "load_stanza", load_stanza)
    result = inspect_local_backend("AR", backend="stanza", model="ar-ud", model_dir=str(tmp_path))

    assert requested == [("ar", "ar-ud", str(tmp_path))]
    assert {item.model for item in result if item.backend == "stanza"} == {"ar-ud"}


def test_explicit_language_probe_rejects_backend_identity_mismatch(monkeypatch, tmp_path):
    import discourselens.nlp as nlp_module

    monkeypatch.setattr(
        nlp_module,
        "load_stanza",
        lambda language, model, model_dir: _ProbeBackend(language, "stanza", "other-model"),
    )

    with pytest.raises(ValueError, match="does not match the explicit selection"):
        inspect_local_backend(
            "ar", backend="stanza", model="selected-model", model_dir=str(tmp_path)
        )


def test_local_preflight_uses_exact_profile_route_without_persisting_resource_path(
    monkeypatch, tmp_path
):
    selected: list[tuple[str, str, str, str | None]] = []

    def inspect(language, *, backend, model, model_dir=None):
        selected.append((language, backend, model, model_dir))
        return probe_backend_capabilities(_ProbeBackend(language, backend, model))

    monkeypatch.setattr("discourselens.languages.inspect_local_backend", inspect)
    monkeypatch.setenv("DL_TEST_STANZA_HOME", str(tmp_path))
    result = local_execution_preflight(
        ("AR",),
        {"dependency_parse"},
        routes={
            "ar": {
                "backend": "stanza",
                "model": "ar-padt",
                "model_dir_env": "DL_TEST_STANZA_HOME",
            }
        },
    )["ar"]

    assert result["runnable"]
    assert result["statuses"]["dependency_parse"] == "SUPPORTED"
    assert result["backend"] == "stanza"
    assert result["model"] == "ar-padt"
    assert result["model_dir_env"] == "DL_TEST_STANZA_HOME"
    assert result["route_source"] == "profile"
    assert selected == [("ar", "stanza", "ar-padt", str(tmp_path))]
    assert str(tmp_path) not in repr(result)


def test_profile_route_never_falls_back_when_stanza_resource_env_is_missing(monkeypatch):
    monkeypatch.delenv("DL_MISSING_STANZA_HOME", raising=False)
    result = local_execution_preflight(
        ("ar",),
        {"dependency_parse"},
        routes={
            "ar": {
                "backend": "stanza",
                "model": "ar-padt",
                "model_dir_env": "DL_MISSING_STANZA_HOME",
            }
        },
    )["ar"]

    assert not result["runnable"]
    assert result["statuses"]["dependency_parse"] == "UNASSESSED"
    assert any("DL_MISSING_STANZA_HOME" in item for item in result["limitations"])


def test_unneeded_broken_local_route_does_not_block_lexical_only_preflight(monkeypatch):
    monkeypatch.delenv("DL_OPTIONAL_STANZA_HOME", raising=False)

    result = local_execution_preflight(
        ("ar",),
        set(),
        routes={
            "ar": {
                "backend": "stanza",
                "model": "ar-padt",
                "model_dir_env": "DL_OPTIONAL_STANZA_HOME",
            }
        },
    )["ar"]

    # The route failure remains inspectable, but no NLP capability is required.
    # This preserves lexical-only/offline operation when an optional model is absent.
    assert result["runnable"]
    assert result["statuses"]["dependency_parse"] == "UNASSESSED"
    assert any("DL_OPTIONAL_STANZA_HOME" in item for item in result["limitations"])


def test_unavailable_explicit_profile_route_stays_unassessed_and_redacts_machine_path(
    monkeypatch, tmp_path
):
    def unavailable(*args, **kwargs):
        raise ProjectError(f"model directory unavailable at {tmp_path / 'private-models'}")

    monkeypatch.setattr("discourselens.languages.inspect_local_backend", unavailable)
    result = local_execution_preflight(
        ("en",),
        {"dependency_parse"},
        routes={"en": {"backend": "spacy", "model": "custom-en-model"}},
    )["en"]

    assert not result["runnable"]
    assert result["statuses"]["dependency_parse"] == "UNASSESSED"
    assert result["limitations"] == [
        "dependency_parse: UNASSESSED",
        (
            "local NLP route: selected local NLP route could not be loaded; verify the optional "
            "dependency, installed model, and local resource configuration"
        )
    ]
    assert str(tmp_path) not in repr(result)


def test_spacy_language_mismatch_is_reported_unassessed_not_as_usable_capability(monkeypatch):
    import discourselens.nlp as nlp_module

    def mismatched_model(_language: str, _model: str):
        raise ProjectError("spaCy model language does not match the selected document language")

    monkeypatch.setattr(nlp_module, "load_spacy", mismatched_model)

    capabilities = local_language_capabilities("en")
    by_name = {item.capability: item for item in capabilities}

    assert by_name["dependency_parse"].operational_status == "UNASSESSED"
    assert by_name["dependency_parse"].assurance_status == "UNASSESSED"
    assert by_name["dependency_parse"].limitation == (
        "spaCy model language does not match the selected document language"
    )
