# DHA discursive-strategy review v1

## Identity and purpose

- Skill: `builtin.dha-strategies`
- Skill version: `1.3.0`
- Prompt reference: `dha-strategies-review-v1`
- Execution: LLM-assisted interpretation over Core-owned textual evidence and explicitly identified context.

This is a bounded review informed by the Discourse-Historical Approach (DHA). It operationalizes five questions for examining salient discourse strategies; it is not a complete DHA study or an automatic ideology detector.

## Context and analytical sequence

Use the supplied ScopeSnapshot and distinguish:

1. **Text and co-text:** the cited passage and relevant surrounding material.
2. **User-provided context:** supplied as such, without presenting it as independently verified.
3. **Verified external context:** use only when a source and provenance are explicitly supplied.

Move between context/topic and close textual description where the supplied material permits. Do not manufacture historical context. Select strategies relevant to the research question, topic, genre and evidence; do not force every strategy into every passage. A targeted passage review is not exhaustive coverage of a discourse or corpus.

## Five strategy questions

- **Nomination:** How are actors, objects, phenomena, events, processes or actions named and referred to? Describe the actual wording and its referential function; do not infer stable identity or group membership beyond the text.
- **Predication:** What qualities, characteristics or evaluations are attributed to those referents? Identify the linguistic realization and its local target. Do not convert a lexical association into a claim about the speaker's belief or intent.
- **Argumentation:** What claims and reasons are related, and what warrant connects them, if that relation is evidenced? A topos is a contextual warrant or conclusion rule, not a keyword category. If a premise, warrant or conclusion is missing or ambiguous, say so rather than completing the argument speculatively.
- **Perspectivization:** From whose viewpoint are nominations, attributions or arguments presented? Track explicit quotation/reporting, attribution, deictic or modal cues where relevant. Do not equate reported speech with endorsement, or infer alignment from a single cue.
- **Intensification / mitigation:** Which linguistic forms strengthen or attenuate a proposition or its force? Describe the form and its scope; do not infer deliberate persuasion or psychological certainty from it alone.

Strategies may overlap, and their interpretation depends on context. Record only distinctions supported by the cited text and available context.

## Required output discipline

Keep these layers separate:

- **Observation:** exact Evidence IDs and the observable expression/structure.
- **Interpretation:** the selected strategy and why it is plausible in this context.
- **Alternative interpretation:** a credible reading that does not depend on the proposed critical interpretation.
- **Limitations:** missing co-text/context, ambiguous speaker attribution, language/model uncertainty, or retrieval coverage limits.

Do not infer author intention, private beliefs, political affiliation, audience effects or causality from a strategy label. Do not make a universal, prevalence or absence claim from selected examples. Additional evidence may be retrieved only inside the finding's original immutable ScopeSnapshot; distinguish examples from exhaustive enumeration.

## Methodological references

- Reisigl, Martin, and Ruth Wodak. 2016. “The Discourse-Historical Approach (DHA).” In *Methods of Critical Discourse Studies*, 3rd ed., edited by Ruth Wodak and Michael Meyer, Chapter 2. SAGE. [Publisher chapter listing](https://us.sagepub.com/en-us/nam/book/methods-critical-discourse-studies).
- Reisigl, Martin. 2017. “The Discourse-Historical Approach.” In *The Routledge Handbook of Critical Discourse Studies*, edited by John Flowerdew and John E. Richardson, 44–59. Routledge. [University of Vienna publication record](https://ucrisportal.univie.ac.at/en/publications/the-discourse-historical-approach-2/).
- Bhatia, Aditi, et al. 2020. “Language Testing in the ‘Hostile Environment’: The Discursive Construction of ‘Secure English Language Testing’ in the UK.” *Applied Linguistics*. [Oxford Academic](https://doi.org/10.1093/APPLIN%2FAMZ017). Its methods section illustrates the five strategy questions and explains why selected strategies vary with text type and research focus.

## Known limitations

- This is an LLM-assisted, evidence-bound review, not a validated DHA coding instrument.
- The five strategies are not an exhaustive substitute for DHA's broader research practice, contextual work or iterative analysis.
- A language model may miss genre conventions, historical intertext, irony, multilingual meaning or implicit argument structure; expose such uncertainty instead of filling gaps.
- A keyword, lexical polarity or isolated grammatical form is not sufficient by itself to establish a strategy's function.
