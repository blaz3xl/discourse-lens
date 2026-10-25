"""Versioned framework instructions for evidence-bound LLM review skills.

These profiles constrain a shared, evidence-bound review executor.  They do
not turn a framework into an automatic diagnosis: every resulting finding
still has to cite Core-owned evidence records and retain an alternative
reading.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ProjectError


@dataclass(frozen=True)
class FrameworkReview:
    """Immutable methodology profile used by the trusted review executor."""

    id: str
    version: str
    instruction: str
    methodology_ref: str | None = None
    prompt_ref: str | None = None


FRAMEWORK_REVIEWS: dict[str, FrameworkReview] = {
    "fairclough-text": FrameworkReview(
        "fairclough-text",
        "1.3.0",
        """Use a two-layer Fairclough-oriented reading of only the supplied passages. First state DESCRIPTION: quote the exact Evidence IDs and identify observable lexical/grammatical/cohesive choices, selecting only relevant dimensions: vocabulary and evaluation; transitivity, agency and nominalization; mood, modality and commitment; pronouns/participant relations; cohesion, repetition and clause relations. Do not force every dimension into every passage. Then state INTERPRETATION: explain one plausible textual or representational effect and a credible alternative reading. Keep textual form distinct from interpretation. Connect the text to discourse practice or wider social practice only when the supplied corpus or explicitly verified/user-provided context supports that connection; identify which context source is used. An omitted agent is an observation; deliberate concealment is not established by that observation alone. Do not infer author intention, private beliefs, political affiliation, causality, or audience effects. Cite only supplied evidence IDs; do not make universal, prevalence, or absence claims from targeted examples.""",
        methodology_ref="docs/methodology/fairclough-text-review-v1.md",
        prompt_ref="fairclough-text-review-v1",
    ),
    "van-leeuwen-social-actors": FrameworkReview(
        "van-leeuwen-social-actors",
        "1.3.0",
        """Analyze only the supplied passages and Core-issued evidence. First describe the actor expression and the process/predicate or participant role explicitly present. Then assess only relevant representation choices: inclusion/exclusion; activation/passivation and role allocation; individualization/assimilation; genericization/specification; personalization/impersonalization; nomination/categorization, including functionalization or identification where supported; or institutionalization/instrumentalization. Do not treat grammatical subject as automatic proof of semantic agency, a noun as proof of a social category, or a single passage's silence as suppression/exclusion. Claims about omission require a defined event/actor population and appropriate comparison or exhaustive scope. Keep observable wording separate from contextual interpretation, cite exact Evidence IDs, give a plausible alternative reading, and state limitations. Never infer author intention, psychological state, political affiliation, prejudice, or causality from a representation pattern alone. Do not make universal or corpus-prevalence claims from targeted examples.""",
        methodology_ref="docs/methodology/van-leeuwen-social-actors-review-v1.md",
        prompt_ref="van-leeuwen-social-actors-review-v1",
    ),
    "dha-strategies": FrameworkReview(
        "dha-strategies",
        "1.3.0",
        "Use only supplied passages, Core-issued evidence and explicitly identified context. Select salient strategies rather than completing a checklist: nomination (how actors, events, processes or actions are named); predication (what qualities are attributed); argumentation (how claims, premises and warrants relate, naming a topos only when the contextual warrant is evidenced); perspectivization (whose viewpoint is voiced, reported or distanced); intensification/mitigation (what linguistic forms strengthen or attenuate a proposition). First describe the form and evidence, then interpret its contextual function and give a credible alternative reading. Keywords alone do not establish a topos, evaluation, speaker alignment or intent. Keep textual, user-provided and verified historical context distinct; do not invent historical links. Cite exact Evidence IDs, retain scope/coverage limitations, and avoid universal, prevalence or absence claims from targeted examples.",
        methodology_ref="docs/methodology/dha-strategies-review-v1.md",
        prompt_ref="dha-strategies-review-v1",
    ),
    "argumentation": FrameworkReview(
        "argumentation",
        "1.3.0",
        "Analyze only supplied passages and Core-issued evidence. Map a claim/conclusion, supporting premise or data, and any warrant/justification only when their relation is textually or contextually evidenced; distinguish an explicit link from an analyst-supplied inference. Record missing, challenged or qualified steps rather than completing an enthymeme as fact. Treat a topos as a context-dependent warrant connecting reason to conclusion, never as a keyword or topic match. Separate description of argument structure from evaluation of its support; do not claim formal validity, fallacy, truth or deception without an explicitly declared method and sufficient evidence. Give a plausible alternative reading, cite exact Evidence IDs, consider counterarguments in the original ScopeSnapshot, and retain scope/context limitations. Do not infer author intention or audience effect, and do not make corpus-wide absence or prevalence claims from targeted examples.",
        methodology_ref="docs/methodology/argumentation-review-v1.md",
        prompt_ref="argumentation-review-v1",
    ),
    "van-dijk-ideology": FrameworkReview(
        "van-dijk-ideology",
        "1.7.0",
        "Analyze the ideological orientations manifested by the supplied discourse, using Core-issued textual evidence and explicitly sourced context. Establish authorial/institutional voice, date, genre, audiences, purposes and represented actors; distinguish these from externally verified affiliations and private beliefs. Distinguish fundamental socially shared ideologies, issue-specific attitudes, personal situation models, context models and discourse. Reconstruct configurations of ideologies and explain their relationships in the national/historical situation, not just a topic or label inventory. Examine values, identities, goals, activities, norms, interests, resources, institutional proposals and power relations. Name and define the best-supported categories and argue how relevant formulations and concrete proposals support them; infer implicit commitments where text and context justify them without demanding an explicit doctrine. Under van Dijk's 2024 approach, authoritarianism is governance and populism a discursive strategy, not additional ideologies. The ideological square is one comparative heuristic, not the entire theory or a prerequisite to every inference. Identify groups from textual/contextual cues and use comparable representations before claiming a systematic asymmetry. Test counterexamples at the same analytical level; rights/democracy declarations alone do not refute exclusion, and silence alone proves neither ideological absence nor strategic concealment. Historical comparisons require separately inspected editions and dates; never attribute an earlier manifesto to the current document. Calibrate conclusions to actual coverage and seek missing evidence when available. Cite exact supplied Evidence IDs. Separate textual observations, sourced contextual facts and theoretical inference. Do not infer private beliefs, intentions, psychological states or an individual's affiliation from wording. Do not invent context, citations, prevalence, universal or absence claims.",
        methodology_ref="docs/methodology/van-dijk-ideology-review-v2.md",
        prompt_ref="van-dijk-ideology-review-v2",
    ),
    "framing": FrameworkReview(
        "framing",
        "1.3.0",
        "Analyze only supplied passages and Core-issued evidence. Describe which textual elements are selected or made salient, then identify any evidenced frame functions: problem definition, causal interpretation, moral evaluation, or treatment recommendation. These functions may be partial; do not force all four. Ground actors, predicates, metaphors, arguments and omissions in exact passages. An omission claim requires a defined comparison/population and suitable coverage. Distinguish a textual frame from framing effects on an audience, which require separate reception evidence. Give a plausible alternative reading, retain counterexamples from the original ScopeSnapshot, and state coverage/context limitations. A frame is not by itself deception, manipulation or proof of author intention. Do not infer causality, audience response, or universal prevalence from targeted examples.",
        methodology_ref="docs/methodology/framing-review-v1.md",
        prompt_ref="framing-review-v1",
    ),
    "metaphor": FrameworkReview(
        "metaphor",
        "1.3.0",
        "Analyze only supplied passages and Core-issued evidence. Identify a candidate lexical expression and compare its context-specific meaning with a more basic contemporary meaning; state the contrast and the evidence before calling the use metaphor-related. This is an evidence-bound screening heuristic, not a claim of full MIP/MIPVU annotation unless a language-appropriate lexical protocol is explicitly supplied. Keep metaphor-related wording distinct from a broader conceptual source-to-target mapping and from claims about discourse function. Require recurrence or comparison evidence for a repeated/systematic pattern; one example supports only a passage-level observation. Give an alternative literal/conventional reading, cite exact Evidence IDs, consider counterexamples in the original ScopeSnapshot, and state language/coverage limits. Do not infer deception, author intention, audience effects or universal prevalence from a metaphor candidate.",
        methodology_ref="docs/methodology/metaphor-review-v1.md",
        prompt_ref="metaphor-review-v1",
    ),
    "intertextuality": FrameworkReview(
        "intertextuality",
        "1.3.0",
        "Analyze only supplied passages and Core-issued evidence. Identify textual relations actually signaled: direct/indirect reported speech, quotation, citation/reference, explicit attribution, repeated formulation or a possible allusion. Record who is represented as speaking, the reporting/quoting form, boundaries and any visible transformation; distinguish the current text from the voice it represents. Do not infer that a quoted or reported voice is endorsed by the current speaker. A resemblance is not proof of borrowing, influence, a specific source or an allusion; verify provenance only against a separately supplied source with provenance, otherwise label it a candidate. Compare wording and context only when both source and target are available. Distinguish intertextual links between texts from broader interdiscursive mixing of genres/discourses. Cite each passage with its own Evidence IDs, give an alternative explanation, and retain scope/context limitations. Do not invent citations, sources, intentions or effects.",
        methodology_ref="docs/methodology/intertextuality-review-v1.md",
        prompt_ref="intertextuality-review-v1",
    ),
    "sociocognitive": FrameworkReview(
        "sociocognitive",
        "1.4.0",
        "Analyze supplied passages with Core-issued evidence and explicitly sourced context. Distinguish observable discourse structures, event/situation models, communicative context models, socially shared knowledge, issue-specific attitudes and fundamental ideologies. Explain the proposed relations between those levels with textual and contextual support instead of merely listing theoretical terms. Establish participants, roles, goals, common ground and historically relevant institutions; distinguish the external situation from its textual representation and hypothesized subjective relevance. Phrase observations as textual constructions, then make calibrated theoretical inferences and compare alternatives. A presupposition is not proof of audience acceptance; a policy, emotion or topic is not itself an ideology. Claims about an actual population's shared knowledge require population-relevant empirical evidence. Do not reduce contextual analysis to co-text or biography. Preserve provenance, original ScopeSnapshot and coverage limits; cite exact supplied Evidence IDs. Do not claim direct access to a person's cognition, infer private beliefs or intentions from wording, or invent audience effects. When using the 2024 radical-right study, distinguish ideological configurations from their constituent ideologies and never transfer its earlier corpus findings to a later uploaded edition.",
        methodology_ref="docs/methodology/sociocognitive-review-v2.md",
        prompt_ref="sociocognitive-review-v2",
    ),
    "historical-context": FrameworkReview(
        "historical-context",
        "1.3.0",
        "Use only context material explicitly supplied to this run and preserve its provenance. Keep TEXTUAL CONTEXT, USER-PROVIDED CONTEXT, and VERIFIED EXTERNAL CONTEXT separate; do not upgrade user-provided claims to verified facts or fill gaps from model memory. For every historical connection, identify the source/context evidence, date or period as supplied, and how it bears on the passage. Distinguish what the text itself says from contextual interpretation. If source, chronology, institutional relation or relevant context is missing, state 'insufficient contextual information' and do not speculate. External facts require separately supplied, provenance-bearing evidence; never invent events, history or citations. Cite text and context evidence separately, give an alternative contextual reading, and state coverage/verification limitations.",
        methodology_ref="docs/methodology/historical-context-review-v1.md",
        prompt_ref="historical-context-review-v1",
    ),
    "power": FrameworkReview(
        "power",
        "1.3.0",
        "Analyze only supplied passages and Core-issued evidence. Describe textual constructions relevant to power—who is quoted or attributed authority, who is represented as acting/affected, how responsibility is assigned or backgrounded, and what reasons/authority are used to legitimate or delegitimate a practice. Distinguish a textual representation or authority claim from actual institutional power, access, ownership or social effect; those require separately supplied contextual evidence. Treat naturalization, exclusion and systematic voice inequality as pattern claims requiring a defined comparison set and adequate coverage, not inference from one silence or passage. Give a plausible alternative reading, cite exact Evidence IDs, consider counterexamples within the original ScopeSnapshot, and state methodological/context limitations. Do not infer author intention, causal effects or actual power relations from wording alone; do not equate legitimation with truth or manipulation.",
        methodology_ref="docs/methodology/power-review-v1.md",
        prompt_ref="power-review-v1",
    ),
    "critical-reviewer": FrameworkReview(
        "critical-reviewer",
        "1.3.0",
        "Try to refute the proposed interpretation: assess whether the cited evidence is sufficient, whether it overgeneralizes, whether it infers intent or fabricated context, and state the strongest plausible alternative reading. Review targeted lexical candidates retrieved from the prior alternative reading, but never treat a lexical match alone as a refutation. Do not introduce uncited findings.",
    ),
}


def framework_review(identifier: str) -> FrameworkReview:
    """Return a known profile; unknown profiles are configuration errors."""
    try:
        return FRAMEWORK_REVIEWS[identifier]
    except KeyError as exc:
        raise ProjectError(f"unknown framework review {identifier!r}") from exc
