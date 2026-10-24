from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .versioning import is_semantic_version


class CoverageMode(StrEnum):
    EXAMPLE_BASED = "EXAMPLE_BASED"
    TARGETED = "TARGETED"
    BROAD = "BROAD"
    EXHAUSTIVE = "EXHAUSTIVE"


class CoverageTerminationState(StrEnum):
    COMPLETED = "COMPLETED"
    LIMIT_REACHED = "LIMIT_REACHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class NegativeResultStatus(StrEnum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NO_SEARCHABLE_TERMS = "NO_SEARCHABLE_TERMS"
    NO_MATCHES_UNDER_PLAN = "NO_MATCHES_UNDER_PLAN"
    ABSENCE_IN_EXHAUSTIVE_POPULATION = "ABSENCE_IN_EXHAUSTIVE_POPULATION"


class ClaimType(StrEnum):
    OBSERVATIONAL = "OBSERVATIONAL"
    EXAMPLE_BASED = "EXAMPLE_BASED"
    INTERPRETIVE = "INTERPRETIVE"
    COMPARATIVE = "COMPARATIVE"
    QUANTIFIED = "QUANTIFIED"
    ABSENCE = "ABSENCE"
    UNIVERSAL = "UNIVERSAL"
    CONTEXTUAL = "CONTEXTUAL"


class EvidenceKind(StrEnum):
    TEXTUAL = "TEXTUAL"
    CORPUS_STATISTIC = "CORPUS_STATISTIC"
    EXTERNAL_SOURCE = "EXTERNAL_SOURCE"
    USER_CONTEXT = "USER_CONTEXT"


class EvidenceRole(StrEnum):
    SUPPORT = "SUPPORT"
    COUNTEREVIDENCE = "COUNTEREVIDENCE"
    EXCEPTION = "EXCEPTION"
    EXAMPLE = "EXAMPLE"
    CONTEXT = "CONTEXT"
    ALTERNATIVE = "ALTERNATIVE"


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ExecutionKind(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    HYBRID = "HYBRID"
    LLM_ASSISTED = "LLM_ASSISTED"


class ScopeSnapshot(BaseModel):
    scope_id: str
    revision_ids: list[str] = Field(min_length=1)


class CoverageDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True)

    coverage_mode: CoverageMode
    scope_snapshot_id: str
    population_definition: str | None = None
    population_count: int | None = Field(default=None, ge=0)
    selection_policy: str | None = None
    inspection_count: int | None = Field(default=None, ge=0)
    completion_criterion: str | None = None
    termination_reason: str | None = None
    termination_state: CoverageTerminationState | None = None
    negative_result_status: NegativeResultStatus = NegativeResultStatus.NOT_APPLICABLE

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_termination_state(cls, data: Any) -> Any:
        """Read old completed exhaustive descriptors without weakening new state checks."""
        if not isinstance(data, dict):
            return data
        mode = data.get("coverage_mode")
        reason = data.get("termination_reason")
        normalized_reason = reason.strip().casefold() if isinstance(reason, str) else ""
        if (
            data.get("termination_state") is None
            and mode in {CoverageMode.EXHAUSTIVE, CoverageMode.EXHAUSTIVE.value}
            and (normalized_reason == "completed" or normalized_reason.startswith("completed "))
        ):
            return {**data, "termination_state": CoverageTerminationState.COMPLETED}
        return data

    @model_validator(mode="after")
    def exhaustive_is_defined(self):
        if not self.scope_snapshot_id.strip():
            raise ValueError("coverage requires a non-empty scope_snapshot_id")
        if self.selection_policy is not None and not self.selection_policy.strip():
            raise ValueError("selection_policy must be non-empty when supplied")
        if (
            self.population_count is not None
            and self.inspection_count is not None
            and self.inspection_count > self.population_count
        ):
            raise ValueError("inspection_count cannot exceed population_count")
        if (
            self.coverage_mode is CoverageMode.EXHAUSTIVE
            and self.population_count is not None
            and self.inspection_count is not None
            and self.inspection_count != self.population_count
        ):
            raise ValueError("EXHAUSTIVE coverage inspection_count must match population_count")
        if self.coverage_mode is CoverageMode.EXHAUSTIVE:
            if not self.population_definition or not self.population_definition.strip():
                raise ValueError("EXHAUSTIVE coverage requires population_definition")
            if not self.completion_criterion or not self.completion_criterion.strip():
                raise ValueError("EXHAUSTIVE coverage requires completion_criterion")
            if not self.termination_reason or not self.termination_reason.strip():
                raise ValueError("EXHAUSTIVE coverage requires termination_reason")
            if self.termination_state is not CoverageTerminationState.COMPLETED:
                raise ValueError("EXHAUSTIVE coverage requires a COMPLETED termination_state")
        if self.negative_result_status is NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION:
            if self.coverage_mode is not CoverageMode.EXHAUSTIVE:
                raise ValueError("exhaustive absence status requires EXHAUSTIVE coverage")
            if self.termination_state is not CoverageTerminationState.COMPLETED:
                raise ValueError("exhaustive absence status requires completed enumeration")
            if not self.population_definition or not self.completion_criterion:
                raise ValueError("exhaustive absence status requires a defined population")
        return self


class SkillManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    version: str
    framework: str
    execution_kind: ExecutionKind
    input_schema_ref: str
    output_schema_ref: str
    executor: str
    methodology_refs: list[str] = []
    required_capabilities: list[str] = []
    allowed_claims: list[ClaimType] = []
    prohibited_claims: list[str] = []
    retrieval_plan_ref: str
    prompt_refs: list[str] = []
    dependencies: list[str] = []
    documentation_ref: str
    model_input_schema_ref: str | None = None
    model_output_schema_ref: str | None = None

    @field_validator(
        "id",
        "framework",
        "input_schema_ref",
        "output_schema_ref",
        "executor",
        "retrieval_plan_ref",
        "documentation_ref",
    )
    @classmethod
    def required_text_is_nonblank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("required manifest text fields must not be blank")
        return normalized

    @field_validator("model_input_schema_ref", "model_output_schema_ref")
    @classmethod
    def optional_schema_ref_is_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("optional model schema references must not be blank")
        return normalized

    @field_validator("version")
    @classmethod
    def semantic_version(cls, value: str) -> str:
        if not is_semantic_version(value):
            raise ValueError("version must follow Semantic Versioning 2.0.0")
        return value

    @model_validator(mode="after")
    def methodology_is_present(self):
        if not self.methodology_refs or any(not item.strip() for item in self.methodology_refs):
            raise ValueError("methodology_refs must contain non-empty references")
        for field_name, references in (
            ("required_capabilities", self.required_capabilities),
            ("prohibited_claims", self.prohibited_claims),
            ("prompt_refs", self.prompt_refs),
            ("dependencies", self.dependencies),
        ):
            if any(not item.strip() for item in references):
                raise ValueError(f"{field_name} must not contain blank values")
        if (
            not self.retrieval_plan_ref.startswith("retrieval/")
            or not self.retrieval_plan_ref[len("retrieval/") :].strip()
        ):
            raise ValueError("invalid retrieval plan reference")
        return self


class LanguageCapability(BaseModel):
    language: str
    capability: str
    operational_status: str
    assurance_status: str
    backend: str | None = None


class ExecutionEligibility(BaseModel):
    runnable: bool
    warnings: list[str] = []
    missing_capabilities: list[str] = []
