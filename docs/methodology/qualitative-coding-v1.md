# Qualitative coding v1

DiscourseLens supports researcher-controlled local codebooks, hierarchical
codes and annotations. An annotation attaches one Code to one canonical
`SourceSpan`; it never copies text, invents coordinates or changes the
immutable DocumentRevision.

Annotations are read through an explicit immutable `ScopeSnapshot`. This means
that an annotation on document A is not silently returned while inspecting
document B. Codebook labels and definitions are human-controlled metadata, not
automatic CDS conclusions. They can support later review and reporting, but do
not by themselves establish an interpretation.

Research memos are also local, human-authored records. Each memo names one
immutable `ScopeSnapshot`, preserving the exact document selection that the
researcher considered when writing it. A memo is contextual research material;
it is never silently converted into textual corpus evidence or model input.
