# Counterevidence lexical candidate retrieval v1.1

Historical method record for runs created with skill version 1.1.0. Current
version 2.0.0 stores literal matches as retrieval candidates and does not attach
them directly to Findings; see `counterevidence-lexical-v2.md`.

This deterministic review skill scans the complete canonical text of every
revision in the Finding's original immutable `ScopeSnapshot` for a researcher-
provided literal phrase. It does not limit retrieval to passages that originally
supported the Finding and does not use FTS candidate selection as a completeness
shortcut. Every exact occurrence is attached as `COUNTEREVIDENCE` evidence with
canonical Unicode offsets. The `SkillRun` stores an `EXHAUSTIVE` descriptor whose
population is the number of canonical text code points scanned; this exhaustiveness
applies only to the literal query, not to the broader concept of counterevidence.

It does not decide whether a hit actually refutes an interpretation. A zero-hit
result means only that the exact literal query had no occurrence in the scoped
canonical texts. It never means that no counterevidence exists in the corpus.
