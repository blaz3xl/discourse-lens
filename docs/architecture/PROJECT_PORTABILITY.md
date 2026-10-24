# Project portability

Project identity uses relative logical paths, content hashes, revision IDs and
artifact references only. Absolute host paths, Docker paths and Windows paths are
not inputs to document, span or run identity. Native and container execution open
the same mounted project directory; research bundles preserve hashes and relative
references, never `/workspace/project` or host-specific paths.

Research bundles omit canonical document text and original source files by
default. Their verifier checks the manifest digest and internal span metadata,
including that a textual excerpt's Unicode code-point length matches its span
range. The result therefore reports
`source_location_verification=NOT_VERIFIED_CANONICAL_TEXT_NOT_INCLUDED` when
textual evidence is present: without source text, same-length offset tampering
cannot be compared with the document. The SHA-256 digest detects accidental
corruption, not authorship or malicious edits, because it is not a signature and
can be recomputed. Bundles with no textual evidence report
`source_location_verification=NOT_APPLICABLE`.

Import-size configuration resolves in this order: a per-import CLI or SDK
override, an explicitly stored `import_max_bytes` in the project’s
`discourselens.yaml`, the user file at
`$XDG_CONFIG_HOME/discourselens/config.yaml` (or
`~/.config/discourselens/config.yaml`), `DISCOURSELENS_IMPORT_MAX_BYTES`, then
the 200 MiB default. User configuration currently accepts only
`import_max_bytes`. New projects leave `discourselens.yaml` absent until an
explicit project setting is persisted, so user and environment values can take
effect and the default project remains portable across hosts. On platforms
without safe directory-relative file access, a project with no project YAML
continues to use defaults; existing project YAML and explicit project-config
writes fail closed there. Open or update such configuration on a platform with
safe directory-relative I/O.

For compatibility, an existing project file that contains
`import_max_bytes: 209715200` is treated as an explicit project choice and takes
precedence over user and environment settings. Remove that key to inherit a
lower-precedence value.

Project and user YAML are parsed with `safe_load` from bounded files (64 KiB
maximum), reject secret-bearing keys, recursive aliases and unknown settings,
and refuse symlinks at their config paths. Project YAML reads and atomic updates
require safe directory-relative I/O; updates replace the directory entry, so a
pre-existing hard link is not truncated in place. Config files contain setting
values only; credentials and secret values remain external to configuration
and portable bundles.
