# Foundation Phase 1 — closeout record

**Phase status: COMPLETE.** This document records the scope and outcome of the
initial Foundation phase. It is a historical phase report, not a live inventory
of every capability added later.

## Delivered in Foundation Phase 1

- A portable, local project format backed by SQLite migrations.
- Immutable `DocumentRevision` records and project-relative canonical text.
- Revision-bound, half-open Unicode `SourceSpan` coordinates and deterministic
  textual evidence citations.
- Persisted `Finding` / `FindingEvidence` lineage with controlled evidence roles.
- Explicit immutable `ScopeSnapshot`, FTS5 scope enforcement, and coverage/claim
  validation contracts.
- Persisted analysis-run lifecycle, Skill Manifest v0.1 validation, language
  capability eligibility, and derived-artifact registry foundations.
- Project reopen/copy portability and invariant tests.

## Subsequent implementation

Items that were deferred at the Phase 1 closeout have since been developed in
separate slices. In particular, PDF extraction, pages and paragraphs,
`SourceSpanSegment` projection, local opt-in OCR, and reprocessing are documented
in [Document Engine](DOCUMENT_ENGINE.md). Current user-facing commands and the
Python/MCP interfaces are described in the [project README](../../README.md).

The original Phase 1 exclusions—such as vector retrieval, corpus analytics,
provider execution, full methodological skills, CLI product behavior, and MCP—
describe what was outside that phase at the time. They must not be read as a
claim that those areas are still absent or complete; assess each against its
current implementation and tests.

## Architectural boundary retained

The Foundation still owns canonical project metadata and historical identity.
Later document, corpus, retrieval, and analysis features build on those records;
they do not rewrite revisions or replace the evidence lineage. Derived NLP and
retrieval artifacts remain versioned and rebuildable rather than canonical
project state.
