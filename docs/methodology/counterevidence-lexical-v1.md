# Counterevidence lexical candidate retrieval v1

This deterministic review skill searches an explicit literal phrase inside the
immutable `ScopeSnapshot` of the Finding's AnalysisRun. Every hit is attached as
`COUNTEREVIDENCE` evidence with exact canonical offsets.

It does not decide whether a hit actually refutes an interpretation. A zero-hit
result means only that no example was found under the declared targeted retrieval
plan; it is never a corpus-wide absence claim.
