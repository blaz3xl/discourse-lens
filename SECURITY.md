# Security policy

Document inputs are untrusted. DiscourseLens treats local files as data: it does
not execute PDF JavaScript, embedded actions, attachments, shell commands or URLs.

Report vulnerabilities privately to the project maintainers rather than publishing
exploit details in an issue. Do not include corpus content, API keys or credential
references that could expose an institution in a report.

Secrets are never stored in project SQLite, manifests or reports. Use environment
variable references or optional OS-keyring references only.

Application lifecycle logs are opt-in at the CLI/MCP boundary and are emitted as
JSONL to stderr. The event formatter accepts only allowlisted identifiers,
statuses, byte counts and exception type names; document paths/content and model
prompts/responses are not log fields. Debug mode adds only safe import metadata.

The local MCP server's `DISCOURSELENS_PROJECT` value is trusted host launch
configuration, not a tool argument. It must be an absolute path without `..`
segments and resolve to the selected project; callers cannot switch the active
project through MCP requests. Project-relative artifact paths are independently
validated to remain inside that project before access.
The `.discourselens` storage root, project database and managed artifact
directories must be real project-local paths, not symlinks. Project creation and
opening reject symlinked managed paths so an imported project cannot redirect
artifact access outside its root.
MCP managed-bundle chunk reads and verification also fail closed when the
platform lacks descriptor-relative no-follow filesystem support.

On POSIX systems, newly created project databases, canonical source text and
originals, derived artifacts, research-bundle manifests, and exported reports
are restricted to the creating user (`0600`). This does not retroactively
change permissions on files created by older versions, and Windows access
control continues to follow the project's inherited ACLs. Researchers sharing
a project or exporting reports should review the destination directory's
permissions explicitly.

For model-assisted review and synthesis, trusted methodology/policy instructions
are sent through the provider's native system/instruction channel when that
capability is declared. Corpus text, researcher questions and prior findings are
sent separately as user-supplied research data. DiscourseLens does not grant the
model tools or filesystem access; returned evidence IDs are still checked against
the persisted input set. This separation reduces instruction confusion but does
not make prompt injection impossible, and remote providers receive only the
request context selected by the workflow.
