# Van Dijk-oriented ideology-pattern review v1

## Identity and purpose

- Skill: `builtin.van-dijk-ideology`
- Skill version: `1.3.0`
- Prompt reference: `van-dijk-ideology-review-v1`
- Execution: LLM-assisted interpretation over Core-owned textual evidence and explicitly identified context.

This skill describes possible discourse patterns relevant to group representation and ideology. It does not detect an author's ideology, diagnose a person, or decide that a passage is extremist, racist or propagandistic.

## Scope and context

- Identify proposed in-group/out-group references from the text or explicitly supplied context; do not assume group membership from the analyst's perspective.
- Describe each passage before interpreting a pattern: exact wording, attributed action/quality, agency, evaluation, emphasis and relevant voice.
- Treat social knowledge, shared representations and context models as theoretical lenses, not directly observable mental states. Connect text to such context only through explicitly identified corpus evidence, user-provided context, or verified external sources.
- Additional evidence may be retrieved only within the finding's original immutable `ScopeSnapshot`. Record whether the evidence is example-based, targeted, broad or exhaustive.

## Ideological-square heuristic

The familiar four-part heuristic asks whether discourse emphasizes positive representations of an identified in-group and negative representations of an identified out-group, while de-emphasizing negative in-group and positive out-group representations. Use it as a comparative question, not as a sentence-level classifier.

Before claiming a systematic asymmetry, require a defined comparison set that makes the represented groups and relevant topic/events comparable. Inspect counterexamples and exceptions, including positive or agentive representations that complicate the proposed pattern. If retrieval is targeted or examples are sparse, report only the observed passages and say that corpus-wide distribution remains untested. Do not treat an unmentioned feature as de-emphasized without a defined opportunity/population and adequate coverage.

## Required output discipline

Keep distinct:

- **Observation:** exact Evidence IDs and the textual representation, without motive claims.
- **Interpretation:** a possible group-related pattern and the evidence/context that makes it plausible.
- **Alternative interpretation:** a credible account that does not depend on the proposed ideological reading.
- **Counterevidence and limitations:** contrary/neutral cases, incomplete comparison, ambiguous group reference, missing context or non-exhaustive retrieval.

Never infer author intention, private belief, psychological state, political affiliation, prejudice, audience effect or causality from the text alone. A pattern in discourse is not proof of an individual author's mental state or a group's actual properties. Do not assign an ideology label from one lexical item or passage.

## Methodological references

- van Dijk, Teun A. 2003. “Ideology and Discourse.” Author-hosted lecture text discussing the ideological-square heuristic. [Primary text](https://discourses.org/wp-content/uploads/2022/07/Teun-A.-van-Dijk-2003-Ideology-and-discourse.pdf).
- van Dijk, Teun A. 2008. *Discourse and Context: A Sociocognitive Approach*. Cambridge University Press. [Book record](https://www.cambridge.org/core/books/discourse-and-context/CFF89959D40865995A8BE28F576D8CC9).
- van Dijk, Teun A. 2009. *Society and Discourse: How Social Contexts Influence Text and Talk*. Cambridge University Press, especially “Context, situation and society.” [Chapter record](https://www.cambridge.org/core/books/abs/society-and-discourse/context-situation-and-society/5E4A7079AAD8938FCB20399EFEB527B6).

## Known limitations

- The ideological-square heuristic is neither a universal discourse template nor a measure of bias; its application depends on a justified comparison and context.
- A language model may mistake quotation, irony, genre convention or a single topic selection for endorsement or a stable pattern.
- This skill has not been methodologically validated as an automatic ideology classifier. Human review and transparent limitations remain necessary.
