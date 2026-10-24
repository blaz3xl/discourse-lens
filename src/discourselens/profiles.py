"""Safe declarative analysis profiles backed by trusted Core orchestration."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ProjectError
from .frameworks import FRAMEWORK_REVIEWS
from .languages import KNOWN_CAPABILITIES, LocalPreflight, local_execution_preflight
from .provider_catalog import supported_providers
from .scope_retrieval import MAX_FRAMEWORK_HITS
from .secrets import validate_credential_alias
from .skill_contracts import EVIDENCE_EXTRACTION_FINDING_PHENOMENA
from .versioning import is_semantic_version

MAX_LOCAL_NLP_MODEL_ID_CHARS = 128
MAX_LOCAL_NLP_MODEL_DIR_ENV_CHARS = 128
LOCAL_NLP_MODEL_ID_PATTERN = (
    rf"^[A-Za-z0-9][A-Za-z0-9_-]{{0,{MAX_LOCAL_NLP_MODEL_ID_CHARS - 1}}}$"
)


class LocalNlpRoute(BaseModel):
    """Portable reference to one explicitly selected, already-installed local model."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["spacy", "stanza"]
    model: str = Field(min_length=1, max_length=MAX_LOCAL_NLP_MODEL_ID_CHARS)
    model_dir_env: str | None = Field(default=None, max_length=MAX_LOCAL_NLP_MODEL_DIR_ENV_CHARS)

    @field_validator("model")
    @classmethod
    def nonblank_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("local NLP model identifier must not be blank")
        if not re.fullmatch(LOCAL_NLP_MODEL_ID_PATTERN, normalized):
            raise ValueError(
                "local NLP model must be a bounded portable identifier beginning with a letter or digit"
            )
        return normalized

    @field_validator("model_dir_env")
    @classmethod
    def environment_variable_name(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("model_dir_env must be an environment-variable name")
        return value

    @model_validator(mode="after")
    def backend_resources(self) -> LocalNlpRoute:
        if self.backend == "stanza" and self.model_dir_env is None:
            raise ValueError("Stanza routes require model_dir_env; paths are resolved at runtime")
        if self.backend == "spacy" and self.model_dir_env is not None:
            raise ValueError("model_dir_env is only supported for Stanza routes")
        return self


class ControlledCdaProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1.0.0"
    kind: str = "controlled-cda"
    frameworks: list[str] = Field(min_length=1)
    languages: list[str] = Field(default_factory=lambda: ["en", "es"], min_length=1)
    required_capabilities: list[str] = Field(default_factory=list)
    nlp_routes: dict[str, LocalNlpRoute] = Field(default_factory=dict)
    local_preprocessors: list[Literal["social_actor_candidates", "agency_candidates"]] = Field(
        default_factory=list
    )
    research_questions: dict[str, str] = Field(default_factory=dict)
    counterevidence_queries: dict[str, str] = Field(default_factory=dict)
    question_candidate_limit: int = Field(
        default=MAX_FRAMEWORK_HITS, ge=1, le=MAX_FRAMEWORK_HITS, strict=True
    )
    routing_profile: str | None = None
    default_preset: str | None = None
    reranking_provider_id: str | None = None
    reranking_credential_alias: str | None = None
    reranking_model: str | None = None
    reranking_top_n: int | None = Field(default=None, gt=0, strict=True)

    @field_validator("routing_profile", "default_preset")
    @classmethod
    def nonblank_model_route_references(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("profile model-route references must not be blank")
        return normalized

    @field_validator("version")
    @classmethod
    def semantic_version(cls, value: str) -> str:
        if not is_semantic_version(value):
            raise ValueError("profile version must follow Semantic Versioning 2.0.0")
        return value

    @field_validator("local_preprocessors")
    @classmethod
    def unique_local_preprocessors(
        cls, value: list[Literal["social_actor_candidates", "agency_candidates"]]
    ) -> list[Literal["social_actor_candidates", "agency_candidates"]]:
        if len(value) != len(set(value)):
            raise ValueError("local_preprocessors must not contain duplicates")
        return value

    @field_validator("research_questions")
    @classmethod
    def nonblank_research_questions(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for framework, question in value.items():
            key = framework.strip()
            phrase = question.strip()
            if not key or not phrase:
                raise ValueError("research_questions requires non-empty framework IDs and questions")
            if key in normalized:
                raise ValueError("research_questions contains duplicate normalized frameworks")
            normalized[key] = phrase
        return normalized

    @field_validator("counterevidence_queries")
    @classmethod
    def normalized_counterevidence_queries(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for phenomenon, query in value.items():
            key = phenomenon.strip()
            phrase = query.strip()
            if not key or not phrase:
                raise ValueError("counterevidence_queries requires non-empty phenomena and queries")
            if key in normalized:
                raise ValueError("counterevidence_queries contains duplicate normalized phenomena")
            normalized[key] = phrase
        return normalized

    def validate_method_contract(self) -> None:
        if self.kind != "controlled-cda":
            raise ProjectError("only controlled-cda declarative profiles are supported")
        if len(self.frameworks) != len(set(self.frameworks)):
            raise ProjectError("profile frameworks must not contain duplicates")
        unknown = sorted(set(self.frameworks) - set(FRAMEWORK_REVIEWS))
        if unknown:
            raise ProjectError("profile references unknown framework(s)")
        invalid_question_frameworks = sorted(
            set(self.research_questions) - (set(self.frameworks) - {"critical-reviewer"})
        )
        if invalid_question_frameworks:
            raise ProjectError(
                "profile research questions must target selected interpretive frameworks"
            )
        if any(not language.strip() for language in self.languages):
            raise ProjectError("profile languages must be non-empty")
        normalized_languages = {language.casefold().strip() for language in self.languages}
        normalized_routes = [language.casefold().strip() for language in self.nlp_routes]
        if len(normalized_routes) != len(set(normalized_routes)):
            raise ProjectError("profile NLP routes contain duplicate language identifiers")
        unknown_route_languages = sorted(set(normalized_routes) - normalized_languages)
        if unknown_route_languages:
            raise ProjectError(
                "profile NLP routes must target declared languages"
            )
        unknown_capabilities = sorted(set(self.required_capabilities) - KNOWN_CAPABILITIES)
        if unknown_capabilities:
            raise ProjectError(
                "profile references unknown language capability/capabilities"
            )
        if any(
            not key.strip() or not value.strip()
            for key, value in self.counterevidence_queries.items()
        ):
            raise ProjectError("profile counterevidence queries must be non-empty strings")
        unknown_counterevidence_targets = sorted(
            set(self.counterevidence_queries) - EVIDENCE_EXTRACTION_FINDING_PHENOMENA
        )
        if unknown_counterevidence_targets:
            raise ProjectError(
                "profile counterevidence queries must target a supported evidence-extraction finding phenomenon"
            )
        reranking_values = (
            self.reranking_provider_id,
            self.reranking_credential_alias,
            self.reranking_model,
        )
        if self.reranking_credential_alias is not None:
            try:
                validate_credential_alias(self.reranking_credential_alias)
            except ProjectError:
                raise ProjectError("profile credential alias is unsafe metadata") from None
        if any(value is not None for value in reranking_values) and not all(
            value and value.strip() for value in reranking_values
        ):
            raise ProjectError(
                "reranking profile requires provider ID, credential alias, and model together"
            )
        if self.reranking_top_n is not None and self.reranking_provider_id is None:
            raise ProjectError("reranking_top_n requires a configured reranking provider")
        if self.reranking_provider_id and not any(
            item.provider_id == self.reranking_provider_id and item.reranking
            for item in supported_providers()
        ):
            raise ProjectError("profile references an unsupported reranking provider")

    def local_preflight(
        self, languages: tuple[str, ...] | None = None
    ) -> dict[str, LocalPreflight]:
        """Return explicit eligibility evidence for the profile's declared NLP needs."""
        routes = {
            language.casefold().strip(): route.model_dump()
            for language, route in self.nlp_routes.items()
        }
        required = set(self.required_capabilities)
        if "social_actor_candidates" in self.local_preprocessors:
            required.add("pos")
        if "agency_candidates" in self.local_preprocessors:
            required.add("dependency_parse")
        return local_execution_preflight(
            tuple(languages or self.languages), required, routes=routes
        )
