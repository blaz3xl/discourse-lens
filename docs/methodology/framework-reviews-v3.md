# Framework reviews v3

Version 3 keeps the scoped, bounded targeted-retrieval contract in
`framework-reviews-v2.md` and adds a structured distinction for counterevidence
candidate assessment. A literal query match remains an unclassified retrieval
candidate. The critical reviewer may return `counterevidence_ids` only for such
Core-supplied candidate IDs that it explicitly concludes challenge the reviewed
interpretation, and those IDs must also appear in its cited `evidence_ids`.
Other reviewers cannot assign this role. Candidates that are cited but not
classified as challenging remain contextual evidence.

The Core validates both the candidate identity and the role declaration before
persisting `FindingEvidence`. No model can create a SourceSpan, page location,
or evidence identifier. A lexical hit is never counterevidence solely because
it matched a researcher-supplied phrase.

Framework observation descriptions are model-supplied candidate descriptions.
The Core checks that each one names supplied textual evidence and includes an
exact excerpt, but it does not validate the description's semantic fidelity or
prevent an interpretation or author-intent claim from appearing in it. Findings
and reports label these descriptions as unvalidated candidates and state this
limitation; a researcher should assess them before relying on the description.
