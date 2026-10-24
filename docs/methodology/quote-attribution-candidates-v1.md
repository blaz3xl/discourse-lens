# Quotation and attribution candidates v1

## Purpose and boundary

`builtin.quote-attribution-candidates` creates a deterministic, span-level inventory
of visibly quoted text in the canonical revisions named by the original
`ScopeSnapshot`. It is a preprocessing inventory for later intertextuality review.
Every returned quotation span includes its opening and closing marks, and the
evidence record links it to the exact source revision and canonical offsets. The
candidate also carries sentence context and, when a supported language has been
explicitly assigned, nearby reporting-word candidates.

The skill does not identify a speaker or source, verify that quoted wording is
authentic, or determine whether a speaker endorsed it. A nearby reporting word is
a lexical cue only. It does not establish that the cue semantically attributes the
quotation. The skill makes no claim about influence, borrowing, plagiarism,
interdiscursivity, or author intent.

## Deterministic inventory

The quote-mark inventory recognizes non-empty paired forms using straight double
quotes (`"..."`), curly double quotes (`“...”` and `„...“`), guillemets
(`«...»`), and single-angle guillemets (`‹...›`). Returned spans include both
marks. Straight double quotes are accepted only when local whitespace and
punctuation boundaries make opening and closing roles reasonably clear.

Any nested or structurally ambiguous quote group is excluded as a whole. A blank
line resets an unmatched open quote so it cannot consume later paragraphs.
Unmatched and ambiguous markers are counted and reported with the run result and,
when candidates exist, in the finding limitations. Single quotation marks,
curly single quotes, OCR substitutions, and other unlisted quote forms are not
inventoried. An empty paired form is not returned as a quotation candidate.

The language-specific cue inventory contains a reviewed English set (`according
to`, `said`, `says`, `stated`, `states`, `claimed`, `claims`, `argued`, `argues`,
`reported`, `reports`, `wrote`, `writes`, `told`, `announced`) and a reviewed
Spanish set (`según`, `dijo`, `dice`, `afirmó`, `afirma`, `sostuvo`, `sostiene`,
`informó`, `informa`, `declaró`, `declara`, `escribió`, `escribe`, `contó`,
`cuenta`). Cues are matched case-insensitively within 160 canonical code points
of the quotation and within its sentence context. The nearest matching cue is
retained as its own exact evidence span, with earlier text winning an equal-distance
tie.

The caller may supply an explicit language assignment for every revision in the
scope. Only `en` and `es` are supported; any other assigned language is rejected.
Without assignments, quote marks are still inventoried across the complete scope,
but no language-specific attribution cues are scanned. There is no automatic
language detection or fallback to English or Spanish rules.

## Evidence and coverage

Each quote candidate has a stable candidate key derived from its revision ID and
canonical offsets. Evidence provenance associates three kinds of spans with that
key: the quotation, the sentence context, and zero or more nearby cue spans.
All evidence is textual evidence from the same revision and original scope.

Coverage is exhaustive over canonical text in the original `ScopeSnapshot` for
the declared quote-mark inventory. This describes the scan, not the prevalence or
absence of quotation, reported speech, citations, or intertextual relations.
Unmatched and ambiguous marker counts describe what this scanner could not safely
pair; they do not characterize all possible quotation conventions in the source.

## Known limitations

- Quotation can be expressed without quotation marks, and the inventory misses
  unquoted reported speech, bibliographic references, paraphrase, and repeated
  wording.
- The nearby cue list is intentionally small and English/Spanish only. A lexical
  match can be unrelated to the quotation, and an actual attribution may use an
  unlisted form or another language.
- Nested quotation conventions vary. The skill excludes nested or ambiguous
  groups rather than attempting to assign speaker layers.
- Sentence segmentation and OCR quality can affect the retained context and
  marker boundaries.
- A source text is not independently supplied or checked by this inventory.
