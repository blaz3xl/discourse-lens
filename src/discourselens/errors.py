class ProjectError(Exception):
    pass


class MigrationError(ProjectError):
    pass


class DocumentError(ProjectError):
    pass


class RevisionError(DocumentError):
    pass


class SourceSpanError(ProjectError):
    pass


class EvidenceValidationError(ProjectError):
    pass


class ScopeViolationError(ProjectError):
    pass


class RetrievalLimitError(ProjectError):
    """A retrieval operation stopped before returning a misleading partial result."""

    def __init__(self, message: str, *, limit: int) -> None:
        super().__init__(message)
        self.limit = limit


class SkillManifestError(ProjectError):
    pass


class ModelProviderError(ProjectError):
    pass


class AuthenticationError(ModelProviderError):
    pass


class RateLimitError(ModelProviderError):
    """Provider throttling, optionally with a numeric retry delay in seconds."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class QuotaError(ModelProviderError):
    """Provider account quota, credit, or billing limit exhaustion."""


class ModelResponseError(ModelProviderError):
    pass


class OptionalDependencyError(ProjectError):
    pass


class ArtifactIntegrityError(ProjectError):
    pass
