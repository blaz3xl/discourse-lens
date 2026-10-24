# SPIKE-003 — Native packaging

Linux environment tested: uv 0.5.9 created an isolated CPython 3.12.8 virtual
environment and installed DuckDB 1.5.5 and PyArrow 25.0.1. SQLite FTS5 is
available. `pipx` was absent and was intentionally not installed globally.

PyMuPDF, Pydantic, spaCy, MCP, sqlite-vec, LanceDB, FAISS and Stanza ultimately
imported in that Linux environment. This proves only co-installation feasibility,
not a minimal dependency set nor language-model availability. macOS and Windows
are untested.

Provisional extras: `corpus` (document extraction/FTS), `nlp` (backends/models),
`vectors` (one selected backend), `mcp`, `ocr`, `local-models`, and `dev`.
The base must not include models, OCR engines or vector backends by default.
