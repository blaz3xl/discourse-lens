# Unicode surface tokenization v2

Current surface frequency, KWIC, document dispersion, n-grams, collocations,
and window-association calculations use `unicode-surface-tokenizer-v2`. It
recognizes alphanumeric word characters, combining marks attached to a word,
and the existing internal apostrophe and hyphen connectors. Count/query
matching applies case-folding followed by NFC normalization, so canonically
equivalent NFC and NFD spellings share one surface form.

Derived surface-token Parquet artifacts use schema
`surface-token-v2-ucd-<version>` and store the same v2 token boundaries and
normalized forms. Binding the schema version to Python's Unicode database
version keeps a later runtime from treating values produced under different
Unicode rules as equivalent. Artifact-backed normalized frequency and
document-frequency pages report their calculation version, schema, tokenizer,
normalization rule, and Unicode database version. Existing
`surface-token-v1` artifacts retain their original bytes and historical
evidence remains validated under its recorded contract; current artifact-backed
frequency operations reject those files until they are rebuilt as v2 artifacts.

Tokenization runs over the original canonical text. KWIC start/end offsets are
half-open Python Unicode-codepoint offsets into that unnormalized string, and
persisted evidence cites the exact source slice. In particular, v2 never
normalizes the text before computing offsets or storing an excerpt.

Frequency, KWIC, document-dispersion, and window-association v2 evidence record
the tokenizer identifier, normalization rule, and Python Unicode database
version. Validation recalculates the result against the evidence's immutable
scope and rejects an unsupported Unicode database version. Historical
frequency, KWIC, document-dispersion v1, and window-association evidence retain
their recorded `corpus.TOKEN` behavior.

Corpus comparison retains its historical tokenizer and population counts under
its explicitly versioned comparison contract. Lexical bundles likewise remain
on `surface-lexical-bundles-v1`; their results are not interchangeable with
v2-tokenized n-grams when Unicode normalization or combining marks affect token
boundaries.
