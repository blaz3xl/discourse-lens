# Textual Descriptors v1

`builtin.textual-descriptors` is a deterministic descriptive preprocessing skill.
It enumerates a small, declared inventory of English and Spanish modal markers in
an explicit immutable scope and attaches each observation to exact textual spans.

It may state that a declared form occurs in the scoped text. It does not infer
authorial intention, psychological state, political affiliation, manipulation, or
ideology. The output is candidate evidence for later CDS interpretation and
review; it is not an interpretation by itself. Any future claim that a CDS skill
is methodologically validated requires separate, language- and task-specific
evaluation.

The inventory and executor are versioned together at `1.0.0`. Adding or changing
forms is a methodological change and requires a new skill version.

The skill rejects a requested language without a declared marker inventory rather
than silently applying English or Spanish assumptions. Raw-text frequency, KWIC
and exact search remain language-independent; this constraint applies only to
this linguistic preprocessor.

Each run records an exhaustive `CoverageDescriptor` over the canonical Unicode
code points in the language-eligible revisions selected from the original
`ScopeSnapshot`. This documents how much text was scanned; it does not turn the
finite marker inventory into an exhaustive linguistic account of modality. A
finding therefore remains limited to the declared surface form and its attached
textual examples.
