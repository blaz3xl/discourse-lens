# Fairclough-oriented textual review v1

## Identity and purpose

- Skill: `builtin.fairclough-text`
- Skill version: `1.3.0`
- Prompt reference: `fairclough-text-review-v1`
- Execution: LLM-assisted interpretation over Core-owned textual evidence and targeted retrieved context.

This skill examines how linguistic choices in cited passages may realize meanings and relations. It does not infer a writer's intention, diagnose ideology or establish social effects from wording alone. A finding describes a bounded interpretation, not an objective bias classification.

## Research questions

For each supplied passage, ask only questions relevant to its wording:

1. Which lexical choices, evaluations, metaphors or repetitions are directly observable?
2. What grammatical choices are present, such as transitivity/agency, nominalization, mood, modality, pronouns or participant relations?
3. What plausible textual or representational effect could those choices have in this passage?
4. What alternative reading fits the same evidence?
5. Is a link to discourse practice or wider social practice supported by separately supplied context, or is that context insufficient?

These are analytic prompts, not a checklist requiring every feature to appear in every finding.

## Required output discipline

Keep the following layers separate:

- **Observation:** describe a recoverable linguistic form in the cited excerpt. For example, “The clause has no explicit grammatical agent.”
- **Interpretation:** state a plausible effect or representational implication, with a rationale tied to the observation. For example, “This wording may background agency in this passage.”
- **Alternative interpretation:** state a credible non-critical or otherwise different reading.
- **Limitations:** disclose missing co-text, context, comparison, retrieval coverage or linguistic uncertainty.

The example above does not license “the author deliberately hides responsibility.” Do not infer intention, psychological state, political affiliation, causality, or audience response solely from textual form.

## Evidence and context policy

- Every finding must cite existing Core-issued evidence IDs; the model must not create IDs, locations or quotations.
- A targeted lexical retrieval contributes examples, not an exhaustive account of the document or corpus.
- Evidence extraction is a seed, not a ceiling: this review may use the additional passages retrieved in the same original `ScopeSnapshot`.
- Distinguish textual co-text from user-provided context and verified external context. Do not silently promote one category to another.
- Without relevant contextual evidence, stop at the textual/representational reading and state that broader social-practice claims remain underdetermined.
- Targeted examples cannot support universal, prevalence, or corpus-wide absence claims.

## Scope of analysis

The review may consider vocabulary/evaluation, transitivity and agency, nominalization, mood/modality, pronouns and participant relations, cohesion/repetition, clause relations, and exchange patterns where the cited text supports them. It must not force a feature label onto ambiguous wording or generalize from one isolated form.

## Methodological basis

This operational contract draws on Fairclough's distinction between text description and subsequent interpretation/explanation, and on his text-analysis treatments of social events, relations between clauses, exchange/mood, representation, modality and evaluation. DiscourseLens narrows those concepts to a reproducible evidence-bound workflow; the skill is not a substitute for the full theoretical method or researcher judgment.

References:

- Fairclough, Norman. 2015. *Language and Power*. 3rd ed. Routledge. The publisher contents identify separate chapters on discourse as social practice, description, and interpretation/explanation. [Routledge record](https://www.routledge.com/Language-andPower/Fairclough/p/book/9781138790971).
- Fairclough, Norman. 2003. *Analysing Discourse: Textual Analysis for Social Research*. Routledge. The publisher contents include texts/social events/practices, sentence and clause relations, exchange and grammatical mood, representations, modality, and evaluation. [Routledge record](https://www.routledge.com/AnalysingDiscourse-Textual-Analysis-for-SocialResearch/Fairclough/p/book/9780415258937).

## Known limitations

- The review is LLM-assisted and interpretive; prompt adherence is not methodological validation.
- Retrieval is targeted and can miss relevant passages. Inspect the recorded `CoverageDescriptor` and `RetrievalRun` before making document- or corpus-level claims.
- The skill does not itself provide verified social/historical context, audience studies, causal evidence, or a validated language-quality score.
- Findings require human review; plausible alternatives and limitations remain part of the result.
