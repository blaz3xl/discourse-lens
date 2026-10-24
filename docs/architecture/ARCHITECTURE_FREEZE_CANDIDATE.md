# Architecture Freeze Candidate

This candidate describes the current implementation and proposed boundaries. It
does not record a formal architecture freeze. In particular, sqlite-vec is the
backend currently implemented in code, but no later formal approval of the vector
backend is recorded. DiscourseLens-authored software is now licensed under SPDX
`AGPL-3.0-only`, matching the public AGPL route for the required PyMuPDF
dependency. This permits commercial use when AGPL terms are followed; it does
not require every commercial user to pay. A separate paid license for
proprietary use would cover only rights its signers control and would not replace
the separate Artifex license path for PyMuPDF. See [LICENSING.md](../LICENSING.md)
and [the validation report](ARCHITECTURE_VALIDATION.md) for the current limits.

- **Core:** Python, with CLI, Python SDK, and MCP as adapters over shared project
  behavior.
- **Canonical persistence:** SQLite project metadata and immutable revision/run
  records; canonical extracted text stored as hashed project artifacts.
- **Full-text retrieval:** SQLite FTS5. Parquet token artifacts are derived
  analytics data and do not replace FTS.
- **NLP artifacts:** Parquet/Zstd, versioned and rebuildable; DuckDB is in the
  optional `corpus` extra. A corrected matched 10M-token synthetic run wrote a
  60.0 MiB artifact in 21.1 s; the SQLite token-row counterpart occupied 913.7
  MiB. DuckDB scans/aggregations were faster in the measured fixture while
  indexed SQLite was faster for the KWIC candidate query. Compressed sentence
  blobs used 217.6 MiB, but the one-pass full lemma scan did not finish within
  the 240 s bound. These data do not establish real-corpus throughput or language
  quality.
- **Vectors:** Current code implements sqlite-vec as a derived, rebuildable
  backend behind the optional `vectors` extra, with a separate index artifact per
  immutable revision. New v4 artifacts explicitly use cosine distance;
  historical v1-v3 artifacts remain squared-L2 and are not mutated. Preset
  semantic queries require those derived indexes to be rebuilt as v4. Scope and
  embedding provenance are checked before ranking or query embedding. SPIKE-002
  measured random 768-dimensional vectors at 10k
  and 100k: all candidates passed strict A-vs-B scope, append, delete, rebuild,
  copy, and reopen checks. FAISS led bounded rebuild mechanics; sqlite-vec and
  LanceDB also passed lifecycle checks. These
  benchmark fixtures check candidate mechanics. Separate regression tests
  exercise the current sqlite-vec API path with local fixture indexes.
  `search_vectors` covers document, corpus, subcorpus, and metadata scopes; the
  `test_hybrid_search_with_preset_embeds_once_and_keeps_scope_strict` test uses
  an injected embedding fixture, verifies an in-scope result, and rejects an
  out-of-scope index before another query-embedding call. This scope evidence is
  fixture-based; semantic relevance on real embeddings, live remote-provider
  quality, and application integration of replacement backends remain
  unmeasured. A 100-revision synthetic fan-out run found FAISS substantially
  faster than sqlite-vec on legacy L2 indexes; a smaller cosine run also favored
  FAISS, but neither included the complete production lifecycle or real
  relevance judgments. FAISS remains the mechanics leader; formal backend
  selection and approval remain pending.
- **Evidence:** DocumentRevision → SourceSpan → EvidenceArtifact → FindingEvidence.
  Persisted citations derive from canonical text and remain scope-bound.
- **Skills:** Contract-validated manifests and a built-in trusted executor registry with
  explicit retrieval plans, execution kinds, and capability requirements. No
  project-supplied Python or shell executor is loaded.
- **Claims:** CoverageDescriptor and ClaimTaxonomy constrain structural
  eligibility for absence, universal, quantified, and comparative claims. They
  do not determine whether evidence semantically entails a claim.
- **Languages:** Operation-level capability gating with no silent fallback.
  SPIKE-005 exercised eleven local fixtures per language and the production CLI
  routes. The production gate requires complete token coverage with matching
  offsets; empty or partial token output leaves sentence segmentation and NER
  `LIMITED`, even if their components are advertised. The listed English and
  Spanish spaCy fixture routes passed that gate and reported tokenization, lemma,
  POS, morphology, dependencies, NER, and segmentation as `SUPPORTED/TESTED`.
  Arabic Stanza `ar-padt` reported those preprocessing operations except NER,
  which is `UNSUPPORTED/TESTED`; a separate Stanza `default` fixture run did
  return NER and must not be conflated with the production route. Agency,
  modality, and social-actor candidate rules are preprocessing only. CDA
  methodology validation remains unmeasured, and absent models remain
  `UNASSESSED`.
- **Interfaces:** CLI, SDK, and MCP expose Core-backed project behavior. MCP
  stdio and bundled resources have checks in the CI workflow; named desktop/IDE
  host compatibility is unmeasured.
- **Distribution:** Native Python packaging and Docker are implemented. CI
  declares Linux/macOS/Windows smoke jobs and Docker checks; the workflow file
  alone does not establish that those jobs passed. The local wheel smoke and a
  299.6 MB default Docker image/MCP stdio smoke (zero published ports) are
  recorded in the validation report; the optional NLP image remains unbuilt
  because the Stanza/PyTorch dependency resolution is unusually large.
- **Models:** Provider-neutral routing with explicit presets, capability
  profiles, and secret references; effective configuration is snapshotted for
  execution.
- **Licensing:** The public license is SPDX `AGPL-3.0-only`, recorded in the root
  `LICENSE` and `pyproject.toml`. AGPL permits commercial use when its terms are
  followed, so this public license cannot require payment from every commercial
  user. A separate paid DiscourseLens license can offer proprietary terms only
  for code whose rights are controlled by its signers. PyMuPDF remains a
  mandatory dependency in the base package and default Docker image; its
  separate commercial license comes from Artifex. See [LICENSING.md](../LICENSING.md)
  and [SPIKE-006](spikes/SPIKE-006-pdf-backend.md). Before accepting commercial
  payments or distributing a commercial installer/container, establish the
  signed DiscourseLens terms, confirm contributor relicensing rights, and
  confirm the Artifex license path for that delivery model. The inventory also
  identifies unresolved optional NLP model/data rights; see
  [DEPENDENCY_LICENSE_INVENTORY.md](DEPENDENCY_LICENSE_INVENTORY.md).

## Recommended readiness

Keep this architecture as a provisional implementation baseline. Retain
sqlite-vec as the current implementation behind the backend boundary; the
mechanics evidence makes FAISS the leading candidate, while final selection is
pending representative embeddings, relevance judgments, multi-revision
workloads, and production-path integration/scope verification for any
replacement backend. Resolve the Spanish model license, Arabic model/data terms,
and have counsel review the applied AGPL-3.0-only license, PyMuPDF distribution,
and project rights/contributor history before distribution. Keep Spanish and Arabic
passive operations explicitly limited pending stronger fixture or corpus
evidence, and validate each intended CDA method with language-specific
annotation and expert review before any `VALIDATED` claim. Formal freeze also
depends on reviewing hosted cross-platform CI results, which are configured but
not yet evidenced in this report.
