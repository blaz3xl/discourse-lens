# Evidence-bound argumentation review v1

## Identity and purpose

- Skill: `builtin.argumentation`
- Skill version: `1.3.0`
- Prompt reference: `argumentation-review-v1`
- Execution: LLM-assisted interpretation over Core-owned textual evidence and scoped context.

This skill maps evidenced reasoning relations in discourse. It does not decide whether a claim is true, whether an author is deceptive, or whether an argument is formally valid unless a separately specified evaluation method and the required evidence are available.

## Argument map

For each candidate argument, record the roles that can be supported:

- **Claim / conclusion:** the position or result advanced.
- **Premise / data:** the reason, observation, rule or example offered in support.
- **Warrant:** the bridge that explains why the premise supports the claim, when that bridge is explicit or can be responsibly reconstructed from supplied context.
- **Justification / backing:** any further support given for the warrant or premise.
- **Qualification / counterargument:** wording that limits the claim, and any exception, rebuttal or competing reason.

These are analytic labels, not a demand that every passage contain every component. Mark a relation as explicit, context-supported inference, ambiguous, or not established. Do not turn an implied premise or missing warrant into a quoted fact. An enthymeme may be described as incomplete or reconstructed cautiously with an alternative reading.

## Topoi and contextual warrants

A topos is a content-related warrant or conclusion rule linking a reason to a conclusion. Name a candidate topos (for example threat, numbers, authority, burden, responsibility, justice or usefulness) only when the evidenced relation and relevant context support that classification. The presence of words such as “risk,” “many,” “expert,” “cost” or “responsibility” is not enough. If the bridge cannot be shown, report the wording and leave the topos unassigned.

## Required output discipline

Keep these layers distinct:

- **Observation:** exact Evidence IDs and the statements/connectives/attributions present.
- **Argument map:** roles and support relations, with explicit vs inferred links marked.
- **Interpretation:** a possible argument strategy or topos and its contextual basis.
- **Alternative interpretation:** a credible different account of the relation or function.
- **Limitations:** missing premise/warrant, context, counterargument, ambiguity, language uncertainty or retrieval coverage.

Describe structure before evaluation. Do not label an argument a fallacy merely because a premise is unstated, and do not infer truth, falsity, manipulation, author intention or audience effect from the map alone. Claims about empirical validity require independent evidence outside this skill's text-structural description.

Counterarguments and exceptions must be sought within the finding's original immutable `ScopeSnapshot`, not only among passages initially supporting the argument. A targeted search may yield “no counterargument found under this retrieval plan”; it cannot establish that none exists in the corpus.

## Methodological reference

- Reisigl, Martin, and Ruth Wodak. 2016. “The Discourse-Historical Approach (DHA).” In *Methods of Critical Discourse Studies*, 3rd ed., edited by Ruth Wodak and Michael Meyer, Chapter 2. SAGE. The DHA treats argumentation strategies/topoi as contextual parts of discourse analysis rather than keyword labels. [Publisher chapter listing](https://us.sagepub.com/en-us/nam/book/methods-critical-discourse-studies).

## Known limitations

- Argument structure is often distributed across sentences, speakers, genre conventions and external context; isolated excerpts may not support a complete map.
- Models may hallucinate unstated warrants or confuse a topic with an argument. Require visible evidence for each link and retain uncertainty.
- This is not a formal logic prover, truth evaluator or validated automatic argument-mining system.
