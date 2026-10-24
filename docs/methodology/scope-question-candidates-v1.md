# Scope-question candidate retrieval v1

The scope-question candidate stage performs bounded retrieval for one researcher-authored question and one selected interpretive framework. The default lexical executor is deterministic; the optional hybrid executor combines lexical retrieval with a selected semantic index. Both execute against the analysis run's original immutable `ScopeSnapshot`; neither requires or restricts itself to the descriptive cue inventory.

The lexical plan extracts a small set of non-stopword terms, searches indexed paragraphs within that scope, ranks candidate passages by term overlap, and returns at most 12 passages. The optional hybrid plan uses reciprocal-rank fusion over scoped exact lexical results and the selected local vector indexes, also returning at most 12 passages. Hybrid mode requires exactly one compatible vector index for every revision in the immutable scope; a missing or partial index fails closed before query embedding. Every returned passage is recorded as a scope-linked `RetrievalHit` and canonical textual evidence. Coverage is always `TARGETED`, including when the result set is empty.

The stage creates an example-candidate finding only when passages were returned. It does not answer the research question, interpret the passages, estimate prevalence, or assert absence. Zero results mean only that no examples were found under the selected targeted retrieval plan. Framework interpretation is a separate subsequent skill run.

Questions are stored in versioned analysis-profile configuration and copied into the immutable `AnalysisRun` configuration. They may target selected interpretive frameworks only; the critical reviewer is not an initial research-question target.

## Limitations

Lexical term extraction may miss paraphrases, inflections, and passages whose wording differs from the question. Semantic retrieval can still miss relevant material, and its quality depends on the chosen embedding model and index. The returned top candidates are not an exhaustive corpus inspection in either mode. Hybrid mode sends the question to the configured embedding provider; review that destination and offline policy before opting in. Researchers should refine the question or use corpus tools when broader coverage is required.
