from .types import ClaimType, CoverageMode, NegativeResultStatus

_UNIVERSAL_SCOPE_POPULATION_DEFINITIONS = frozenset(
    {
        "one population item per scoped revision in the immutable scopesnapshot",
        # Preserve the earlier document-level label already stored in projects.
        "all documents in the immutable scopesnapshot",
    }
)


def is_scope_revision_population_definition(value: object) -> bool:
    """Whether a descriptor names the exact per-revision universal population."""
    return (
        isinstance(value, str)
        and value.strip().casefold() in _UNIVERSAL_SCOPE_POPULATION_DEFINITIONS
    )


def claim_allowed(
    claim: ClaimType,
    coverage: CoverageMode,
    *,
    statistical_evidence: bool = False,
    population_defined: bool = False,
    limitations: bool = False,
    interpretation: bool = False,
    alternative_reading: bool = False,
    comparable_populations: bool = False,
    negative_result_status: NegativeResultStatus = NegativeResultStatus.NOT_APPLICABLE,
) -> bool:
    """Check structural eligibility, never whether evidence semantically entails a claim.

    The caller must independently validate persisted evidence, scope, and provenance.
    Unknown enum values fail closed instead of falling through as unrestricted claims.
    """
    try:
        claim = ClaimType(claim)
        coverage = CoverageMode(coverage)
        negative_result_status = NegativeResultStatus(negative_result_status)
    except (TypeError, ValueError):
        return False

    if claim is ClaimType.ABSENCE:
        return (
            coverage is CoverageMode.EXHAUSTIVE
            and negative_result_status is NegativeResultStatus.ABSENCE_IN_EXHAUSTIVE_POPULATION
        )
    if claim is ClaimType.UNIVERSAL:
        return coverage is CoverageMode.EXHAUSTIVE
    if claim is ClaimType.QUANTIFIED:
        return coverage is CoverageMode.EXHAUSTIVE and statistical_evidence and population_defined
    if claim is ClaimType.INTERPRETIVE:
        return limitations and interpretation and alternative_reading
    if claim is ClaimType.COMPARATIVE:
        return coverage is CoverageMode.EXHAUSTIVE and population_defined and comparable_populations
    return True
