# Provenance-bound historical context review v1

## Identity and purpose

- Skill: `builtin.historical-context`
- Skill version: `1.3.0`
- Prompt reference: `historical-context-review-v1`
- Execution: LLM-assisted interpretation using only run-supplied textual and context evidence.

This skill relates text to context when relevant evidence is available. It is not a historical fact-retrieval service and does not use model memory as a source.

## Context provenance classes

Keep three classes distinct in both reasoning and output:

1. **TEXTUAL CONTEXT:** wording and co-text in the selected document(s), each with ordinary DiscourseLens evidence IDs.
2. **USER-PROVIDED CONTEXT:** statements/material supplied by the researcher. Attribute them as user-provided; do not present them as independently verified.
3. **VERIFIED EXTERNAL CONTEXT:** separately supplied records with a source/reference, retrieval or publication date where available, and provenance. This skill does not browse or independently verify sources.

Never merge classes silently. If no adequate context evidence is available, say **“insufficient contextual information.”**

## Historical relation check

For each proposed connection:

- identify the exact textual passage and any explicit temporal/actor reference;
- identify the context record and its provenance class;
- state the temporal or institutional relation supported by those records;
- explain what contextual interpretation is plausible and what remains uncertain;
- note conflicting or missing sources, dates, actors, chronology or co-text.

Context may include co-text, genre/institutional situation, or wider sociopolitical history, but a wider level requires corresponding evidence. A text's mention of an event does not verify that event's history; a date in user notes does not become a verified date; model familiarity is not provenance. Do not infer causation or a historical lineage from temporal coincidence.

## Required output discipline

- **Text observation:** what the cited document says.
- **Context record:** what the supplied source says and which provenance class it belongs to.
- **Interpretation:** the bounded relation between them.
- **Alternative reading:** another plausible contextual relation or an explanation that remains text-internal.
- **Limitations:** missing verification, chronology, historical comparison, context scope, or conflicting records.

Cite text and context separately. Never mint references, URLs, dates, historical events or institutional relationships. Do not turn uncertainty into a confident narrative.

## Methodological reference

- Reisigl, Martin, and Ruth Wodak. 2016. “The Discourse-Historical Approach (DHA).” In *Methods of Critical Discourse Studies*, 3rd ed., edited by Ruth Wodak and Michael Meyer, Chapter 2. SAGE. The DHA explicitly treats historical/contextual analysis as part of its research practice. [Publisher chapter listing](https://us.sagepub.com/en-us/nam/book/methods-critical-discourse-studies).

## Known limitations

- Context records may be partial, contested or non-independent; provenance should make these constraints visible.
- This skill cannot validate historical evidence on its own and cannot establish causality from textual/contextual association.
- Do not run when required context capabilities/provenance are absent; return insufficient-context status rather than silently degrading.
