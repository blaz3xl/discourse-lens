# Evidence-bound framing review v1

## Identity and purpose

- Skill: `builtin.framing`
- Skill version: `1.3.0`
- Prompt reference: `framing-review-v1`
- Execution: LLM-assisted interpretation over Core-owned textual evidence and scoped comparison context.

This review treats a frame as a possible organization of meaning in discourse, not as a deception detector. It records textual selections and their plausible interpretive functions without claiming to know the author's intent or the audience's response.

## Analysis questions

1. Which actors, events, features or relations are selected and made salient in the supplied passage? Identify the linguistic evidence (for example naming, predicates, metaphor, agency, argument, repetition or prominence) rather than naming a frame from topic keywords alone.
2. What issue or condition, if any, is defined as the problem?
3. What cause or responsibility, if any, is attributed, and is the relation explicit or only a possible interpretation?
4. What moral evaluation, if any, is expressed?
5. What treatment, response or recommendation, if any, is proposed or implied?

The latter four are possible frame functions, not mandatory fields that every passage must fill. Report “not evidenced in this passage” when appropriate. That wording does not establish corpus-wide absence.

## Comparison, context and effects

- A claim that a feature is selected, emphasized or omitted relative to alternatives needs a defined comparison set (such as parallel passages, actors or events) and coverage suitable for that claim. Without it, describe only what is salient in the cited passage.
- Keep observations distinct from interpretations; provide a credible alternative frame/reading and relevant counterexamples within the original immutable `ScopeSnapshot`.
- Distinguish a textual frame from **framing effects**. Claims about what audiences noticed, remembered, believed or did require separate reception evidence; textual analysis alone cannot establish them.
- Additional retrieval may broaden the examples only within the original `ScopeSnapshot`. Preserve `CoverageDescriptor` and do not turn a targeted sample into prevalence, universal or absence claims.

## Required output discipline

- **Observation:** exact Evidence IDs, wording, grammatical/lexical form and any explicit causal/evaluative/recommendatory relation.
- **Interpretation:** the possible frame function and why that reading fits the cited material.
- **Alternative interpretation:** a plausible different organization of the same issue.
- **Limitations:** missing comparison, context, causal support, language uncertainty or non-exhaustive retrieval.

Do not infer that a frame is deceptive, manipulative or intentionally constructed to mislead. Do not infer author intention, audience effects or causality unless separately evidenced. A frame may be partial or ambiguous.

## Methodological reference

- Entman, Robert M. 1993. “Framing: Toward Clarification of a Fractured Paradigm.” *Journal of Communication* 43(4): 51–58. [Publisher record](https://doi.org/10.1111/j.1460-2466.1993.tb01304.x).

## Known limitations

- Frame boundaries depend on the research question, text/context and the analyst's operational definition; this skill does not impose a universal frame taxonomy.
- Selected textual evidence can support a plausible frame reading but cannot, by itself, establish audience effects, author motive, systematic media bias or manipulation.
- This is not a validated automatic frame classifier; human review and the recorded alternative reading remain important.
