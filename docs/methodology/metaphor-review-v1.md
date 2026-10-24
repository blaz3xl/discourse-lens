# Evidence-bound metaphor review v1

## Identity and purpose

- Skill: `builtin.metaphor`
- Skill version: `1.3.0`
- Prompt reference: `metaphor-review-v1`
- Execution: LLM-assisted interpretation over Core-owned textual evidence.

This skill identifies and discusses candidate metaphor-related wording. It does not automatically establish conceptual metaphor systems, deception, author intent or audience effects.

## Candidate identification

For each candidate, record:

1. **Lexical expression:** exact word/phrase and Evidence ID.
2. **Contextual meaning:** what it means in this passage and local co-text.
3. **More basic meaning (if defensibly known):** a distinct, contemporary sense that is more concrete or bodily grounded, without treating etymology as proof.
4. **Contrast and comparison:** whether the contextual sense contrasts with, but can be understood in comparison to, the proposed basic sense.
5. **Status:** candidate metaphor-related expression, likely non-metaphorical, or uncertain—with rationale.

This is an explicit screening heuristic informed by MIP, not a full MIP/MIPVU annotation by default: the Core does not yet supply a language-specific lexical database or a calibrated annotation protocol. Do not state that an expression is metaphorical merely because it is vivid, figurative-sounding, evaluative or politically salient.

## Interpretation layers

Keep separate:

- a **metaphor-related linguistic expression**;
- a proposed **source-to-target conceptual mapping** (which requires additional contextual support and may remain uncertain);
- a proposed **discursive function** (which requires the surrounding argument, actors and co-text).

Give a plausible literal, conventionalized or otherwise non-critical alternative reading. A single occurrence supports only a local observation; claims of recurrence, conventional pattern or corpus distribution require a defined scope and suitable systematic/exhaustive retrieval. Additional examples must come from the original immutable `ScopeSnapshot`, with coverage recorded.

## Required output discipline

- **Observation:** exact expression, local context and evidence IDs.
- **Interpretation:** candidate contrast/mapping/function, with its warrant.
- **Alternative interpretation:** credible non-metaphorical or different metaphorical reading.
- **Limitations:** lexical-resource absence, language/model uncertainty, missing co-text or non-exhaustive retrieval.

Do not infer deception, manipulation, author intention, private beliefs, audience effects or causality from metaphor use alone. Do not generalize from one isolated example.

## Methodological references

- Pragglejaz Group. 2007. “MIP: A Method for Identifying Metaphorically Used Words in Discourse.” *Metaphor and Symbol* 22(1): 1–39. [Author-hosted article copy](https://www.lancaster.ac.uk/staff/eiaes/Pragglejaz_Group_2007).
- Steen, Gerard J., Aletta G. Dorst, J. Berenike Herrmann, Anna A. Kaal, Tina Krennmayr, and Tryntje Pasma. 2010. *A Method for Linguistic Metaphor Identification: From MIP to MIPVU*. John Benjamins. [Publisher record](https://www.benjamins.com/catalog/celcr.14).

## Known limitations

- The basic/contextual meaning distinction can be language- and context-sensitive and requires lexical knowledge not currently guaranteed by the runtime.
- MIP/MIPVU word-level identification does not by itself determine a discourse-level conceptual mapping or function.
- This review has not been validated as an automatic multilingual metaphor annotator; report unsupported/uncertain language capability rather than silently applying English assumptions.
