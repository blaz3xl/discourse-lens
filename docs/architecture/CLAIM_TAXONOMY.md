# Claim taxonomy v0.1

`OBSERVATIONAL` and `EXAMPLE_BASED` require linked textual evidence in a
supporting or example role. `INTERPRETIVE` requires linked textual evidence; its
evidence roles retain their relation to the source hypothesis (so a counterexample
may still ground a different reading). An interpretive claim also requires a
non-empty interpretation, alternative reading, and limitations. `QUANTIFIED` requires
versioned deterministic corpus-statistic evidence whose scope, population
definition, population count, and `CoverageDescriptor` exactly match the finding's
exhaustive coverage. Merely linking a numeric statistic from the same scope is
insufficient. `ABSENCE` and `UNIVERSAL` require exhaustive coverage; absence
additionally requires a zero statistic linked to that same coverage descriptor.
`CONTEXTUAL` requires external-source evidence with recorded provenance or
explicitly labeled user context; DiscourseLens records this provenance but does
not independently verify the truth of an external source. `COMPARATIVE` requires
exhaustive coverage and a linked, hash-verified row from a `ComparisonRun`. The
analysis run's scope must equal the union of the two distinct frozen population
scopes; evidence preserves each side separately, including its revisions,
population size, selected statistic, comparison hash, and calculation version.
The researcher must provide a comparability basis. DiscourseLens records that
basis but does not independently assess it. Comparison statistics cannot satisfy
`QUANTIFIED` or `ABSENCE` evidence requirements.
