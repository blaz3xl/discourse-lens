# Evidence-bound intertextuality review v1

## Identity and purpose

- Skill: `builtin.intertextuality`
- Skill version: `1.3.0`
- Prompt reference: `intertextuality-review-v1`
- Execution: LLM-assisted interpretation over Core-owned textual evidence and explicitly supplied source/context material.

This skill describes relations among voices and texts that are signaled in the evidence. It does not infer a specific source or influence from resemblance alone.

## Candidate relation record

For each candidate relation, record:

- the current passage and exact Evidence ID;
- relation form: direct/indirect reported speech, quotation, citation/reference, attribution, repeated wording, or possible allusion;
- represented speaker/source and reporting frame, only as the current text identifies them;
- source/target text availability and provenance status;
- visible wording changes, omissions, additions or recontextualization, only if both texts can be compared;
- confidence/limitations in ordinary qualitative terms, not numeric probabilities.

An explicit attribution establishes what the current text attributes; it does not independently authenticate the original speaker or source. Mark resemblance as a candidate allusion or echo unless the source is separately supplied and verified. Do not invent bibliographic citations or source passages.

## Keep relations distinct

- **Intertextual relation:** a link between texts or utterances, such as a quotation, citation or re-used formulation.
- **Reported voice:** speech represented within another utterance. Distinguish the reporting speaker, the represented speaker and the reported content; quotation is not automatic endorsement.
- **Interdiscursive mixing:** possible combination of genres, styles or discourse types. Do not claim this merely because a text includes multiple speakers or sources.

Where source and target are both available, compare exact wording and context. Describe transformations before interpreting their possible effect; avoid assuming intentional distortion. If only the current text is available, analyze its attribution/reporting construction and leave source verification open.

## Required output discipline

- **Observation:** current-text wording, reporting/citation markers and Evidence IDs.
- **Interpretation:** the possible textual relation and its contextual function.
- **Alternative interpretation:** a plausible explanation such as common phrasing, conventional formula, independent formulation or incomplete source access.
- **Limitations:** unavailable source, uncertain attribution, missing context, language uncertainty or incomplete coverage.

Do not infer author intention, borrowing, plagiarism, endorsement, influence or audience effects without separate evidence. Additional retrieval is confined to the original immutable `ScopeSnapshot`; external material must remain separately identified and provenance-bearing.

## Methodological references

- Fairclough, Norman. 1992. “Intertextuality in Critical Discourse Analysis.” *Linguistics and Education* 4: 269–293. The chapter is discussed as an early exploration of intertextuality/interdiscursivity in the [Cambridge Handbook of Discourse Studies](https://www.cambridge.org/core/books/abs/cambridge-handbook-of-discourse-studies/historicity-interdiscursivity-and-intertextuality-in-discourse-studies/A95F56F2712F8AA211E1BA03EE5DAA50).
- Matoesian, Gregory M. 2001. “Production Media and Intertextual Authority in Reported Speech.” In *Law and the Language of Identity*. Oxford University Press. [Chapter record](https://academic.oup.com/book/48436/chapter-abstract/421391476).

## Known limitations

- The text may transform reported speech through selection, paraphrase, genre and context; without the source, the transformation itself cannot be established.
- Automatic similarity or quotation matching is not implemented by this review and would not by itself establish influence or provenance.
- This is not a validated source-attribution or plagiarism detector.
