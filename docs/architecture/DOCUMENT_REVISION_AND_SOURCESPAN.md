# DocumentRevision and SourceSpan v0.1

Canonical offsets are half-open Unicode code-point offsets. A SourceSpan belongs
to one immutable DocumentRevision and remains valid through NLP, chunker and
embedding changes. Cross-page spans use SourceSpanSegments; discontinuous evidence
uses multiple SourceSpans. Changed bytes, canonical text, structural/page anchors,
OCR output, or coordinate normalization create a revision. NLP, chunks, embeddings
and analytics create derived artifacts only. Citation renderers resolve artifact
IDs; models never mint page locations or quotations.
