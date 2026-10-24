# Van Leeuwen-oriented social actor review v1

## Identity and purpose

- Skill: `builtin.van-leeuwen-social-actors`
- Skill version: `1.3.0`
- Prompt reference: `van-leeuwen-social-actors-review-v1`
- Execution: LLM-assisted interpretation over Core-owned textual evidence and targeted retrieved context.

This skill describes how participants in social practices are represented in supplied discourse. A category is an analytic reading of linguistic choices, not an automatic property of an actor, a moral judgment, or a diagnosis of the text's author.

## Research questions

For each evidenced actor expression:

1. Which participant is named or referred to, and what action, process or relation is explicitly represented?
2. Is the participant included in the cited representation? Is any claim of exclusion actually testable from this text/scope, or is the actor merely not mentioned in this passage?
3. How is the participant positioned in the clause/event (for example, activated, affected/passivated, or otherwise allocated a role)?
4. Is the participant represented individually or as a group; specifically or generically; personally or impersonally?
5. Is the expression a name, a functional role, an identity/category, a relation, an institution or an abstraction? Which exact wording supports that distinction?
6. What contextual interpretation and credible alternative reading follow, and what cannot be concluded from the available scope?

Do not run every category as a checklist. Use only categories for which the wording and context are sufficient.

## Category guide

- **Inclusion / exclusion:** identify an actor represented in the cited event. A passage that does not name an actor is not, by itself, proof of suppression or exclusion. Claims about absence require a defined event/actor population and sufficient comparison or exhaustive inspection; otherwise report only “not named in this passage” or “not found under this retrieval plan.”
- **Activation / passivation (role allocation):** describe whether an actor is represented as doing/initiating an action or as affected by it, and identify the participant role actually licensed by the clause. Grammatical subjecthood alone does not determine semantic agency. Parser dependencies are candidates, not a finished classification.
- **Individualization / assimilation:** distinguish a specific individual from representation as a group where the linguistic form warrants it. Do not infer uniformity of group members from collectivization.
- **Specification / genericization:** distinguish a contextually specific actor from a generic or generalized representation only when the wording/context establishes that contrast.
- **Personalization / impersonalization:** distinguish human-person representation from institutional, objectified or abstract representation. Do not infer concealment or motive from an impersonal expression alone.
- **Nomination / categorization:** describe naming versus representation by category/role/identity, such as a name, function, relational identity or social category. Do not infer identity beyond the text.
- **Institutionalization / instrumentalization:** where supported, describe an institution or means being used metonymically to represent participants or action. State that the wording may compress agency; do not claim the institution or instrument literally acted without contextual support.

These are descriptive distinctions for interpretation. One expression can instantiate multiple relevant choices; ambiguity should remain visible.

## Required output discipline

Keep four layers distinct:

- **Observation:** quote or precisely locate the expression using supplied Evidence IDs; describe its grammatical/lexical form and explicit role.
- **Interpretation:** state the relevant representation category and explain the contextual basis for applying it.
- **Alternative interpretation:** give a plausible reading that does not depend on the proposed critical reading.
- **Limitations:** disclose missing co-text, event comparison, language/model uncertainty, or non-exhaustive retrieval.

Example: “The sentence refers to the decision as an action but does not state an agent” is an observation. “The wording may background the decision-maker” is an interpretation. “The surrounding paragraph may already identify the decision-maker” is a relevant alternative/context check. “The author deliberately hides responsibility” is not licensed by the sentence alone.

## Evidence and context policy

- Cite only existing Core-issued evidence IDs. Never mint quotations, pages, actor names or locations.
- Seed evidence does not cap analysis: additional passages may be considered only from the finding's original immutable `ScopeSnapshot`.
- Keep a retrieved example distinct from an exhaustive count or population-level claim.
- Do not infer psychological state, author intention, political affiliation, prejudice, causality, or audience effects from actor representation alone.
- A targeted search cannot establish corpus-wide absence, universality or prevalence.

## Methodological basis

The skill operationalizes a bounded textual workflow informed by van Leeuwen's framework for representing social actors, while retaining contextual interpretation and researcher review. The framework includes exclusion, participant roles, group/individual representation, categorization and institutional/metonymic representation; this implementation does not claim to exhaust the source framework.

Reference:

- van Leeuwen, Theo. 2008. *Discourse and Practice: New Tools for Critical Analysis*. Oxford University Press, especially Chapter 2, “Representing Social Actors.” [Oxford Academic record](https://academic.oup.com/book/7877) and [Chapter 2 record](https://academic.oup.com/book/7877/chapter-abstract/153101051).

## Known limitations

- This is an LLM-assisted interpretation, not a validated automatic actor classifier.
- Parser candidates can help find noun/pronoun expressions but do not decide inclusion, activation, passivation or categorization.
- The shared retrieval plan is targeted and may miss relevant passages; inspect `CoverageDescriptor` and `RetrievalRun` provenance.
- The skill does not independently establish real-world group membership, institutional authority, author motive, or whether an omitted actor was expected to appear.
