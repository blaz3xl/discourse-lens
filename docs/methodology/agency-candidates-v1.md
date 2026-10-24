# Dependency agency candidates v1

This deterministic preprocessor joins parser dependency relations to their parsed
heads inside an explicit immutable scope. Relations such as `nsubj`, `nsubj:pass`,
`obj` and `obl` are saved as evidence-bound candidates for textual/transitivity
review.

Each candidate cites its persisted sentence when available, so the quotation
contains both the dependent expression and its predicate in context. If no
sentence boundary is available, the evidence span covers the exact dependent
and head token range. The parser relation itself remains an observed feature,
not an interpretation.

It does not claim that a passive construction deliberately hides agency, nor does
it infer author intention. Parser error and the limits of the language model remain
visible in each resulting finding.
