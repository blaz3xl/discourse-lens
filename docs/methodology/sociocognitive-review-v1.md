# Evidence-bound sociocognitive review v1

## Identity and purpose

- Skill: `builtin.sociocognitive`
- Skill version: `1.3.0`
- Prompt reference: `sociocognitive-review-v1`
- Execution: LLM-assisted interpretation over Core-owned textual evidence and explicitly supplied contextual/empirical sources.

This review uses sociocognitive concepts as theoretical lenses for discourse/context relations. It does not infer a person's actual mental state from text.

## Levels of analysis

Keep three levels distinct:

1. **Textual construction:** observable wording, presupposition, attribution, categorization, modality, assumed common ground or representation. Phrase claims as “the passage treats X as given” or “the text represents group Y as …”.
2. **Theoretical sociocognitive interpretation:** a possible context model, shared-knowledge assumption or social representation that could help explain the textual construction. Label it as an interpretation and offer an alternative.
3. **Empirical claim about people/groups:** what a particular author, reader, population or community actually knows, believes or remembers. This requires independently supplied population-relevant evidence; a text or model response alone cannot establish it.

Context models are a theoretical construct in van Dijk's approach, not objects directly observed by this skill. Likewise, a passage's presupposition does not prove that its audience accepts the presupposed content.

## Evidence and contextual requirements

- Cite exact textual Evidence IDs for each described construction.
- Identify the source and provenance class for any contextual evidence about a community, institution, history or shared knowledge.
- Do not generalize a sample to an entire social group without a defined population, comparison and adequate coverage.
- Retrieve additional text only within the finding's original immutable `ScopeSnapshot`; a targeted set of passages cannot establish universal shared knowledge or audience effect.
- If the proposed cognitive/social relation has no contextual or empirical support, say so and retain only the text-level observation.

## Required output discipline

- **Observation:** exact wording and the discourse cue.
- **Interpretation:** a bounded theoretical reading of what the text treats as relevant/given and why.
- **Alternative interpretation:** another explanation, including genre or co-text where plausible.
- **Limitations:** missing group/context evidence, population boundaries, language uncertainty or non-exhaustive retrieval.

Never infer an author's private belief, intention, psychological state, political affiliation, a reader's comprehension or a group's shared mental state from wording alone. Avoid mind-reading even when a theoretical vocabulary makes such claims sound plausible.

## Methodological references

- van Dijk, Teun A. 2008. *Discourse and Context: A Sociocognitive Approach*. Cambridge University Press. [Book record](https://www.cambridge.org/core/books/discourse-and-context/CFF89959D40865995A8BE28F576D8CC9).
- van Dijk, Teun A. 2009. *Society and Discourse: How Social Contexts Influence Text and Talk*, especially “Context, situation and society.” Cambridge University Press. [Chapter record](https://www.cambridge.org/core/books/abs/society-and-discourse/context-situation-and-society/5E4A7079AAD8938FCB20399EFEB527B6).

## Known limitations

- Context-model theory does not license direct diagnosis of an individual's cognition from linguistic features.
- Socially shared knowledge and group representations require external empirical context and careful population definition.
- This is not a cognitive-state, belief, audience-effects or psychological classifier.
