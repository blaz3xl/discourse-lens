# Skill Manifest v0.1

Required sections: identity/version/execution kind; methodology/framework and
bibliography references; input/output schemas; executor compatibility; language,
NLP and model capability requirements; allowed/prohibited claims; evidence and
counterevidence policies; RetrievalPlan references; prompt digests; dependencies;
documentation references.

`input_schema_ref` and `output_schema_ref` describe the JSON-serializable
SkillExecutor call input and result, respectively. The call schema covers the
executor's explicit data arguments; `SkillContext`, project handles, and
provider/reranking clients are injected runtime dependencies, not serialized
arguments. These references never describe the payload exchanged with an LLM.
For `LLM_ASSISTED` and `HYBRID` skills, the optional
`model_input_schema_ref` and `model_output_schema_ref` separately describe the
serialized model research-data input and the provider's JSON response shape.
Their absence is valid for deterministic skills and for existing v0.1 manifests.

`version` follows Semantic Versioning 2.0.0: the `major.minor.patch` core is
required; prerelease and build metadata are supported. Numeric core and
prerelease identifiers cannot contain leading zeroes.

The manifest is authoritative metadata. Prose, bibliography, schemas and prompts
are independent versioned artifacts referenced by it. The current registry
contains reviewed built-in skills with trusted Core executors. Declarative
profiles compose registered Core operations; an `executor` reference does not
load third-party code, and arbitrary Python, shell, filesystem and network
execution are not supported.

`SkillRegistry.plan_dependencies(requested_skill_ids)` validates the requested
registered skills' dependency closure and returns deterministic topological
layers. Independent skills appear in the same layer; this is a scheduling
boundary, not an execution engine, and does not imply concurrent execution. The
controlled CDA pipeline continues to use its separately recorded, explicit
methodological order.

At runtime, persisted Finding validation applies a built-in skill's
`allowed_claims` only when the Finding's skill ID and version match its registered
manifest. A mismatched version for a registered built-in is invalid. Claim-type
evidence and coverage requirements are validated separately under the claim
taxonomy. An unregistered skill ID does not gain executor support or a built-in
manifest claim contract.

`prohibited_claims` records epistemic and prompt policy. The Core does not scan
Finding prose for prohibited meanings or assess semantic entailment; a Finding
can pass structural validation while making an unsupported statement in free
text. The validation result reports `semantic_entailment_assessed: false`.
