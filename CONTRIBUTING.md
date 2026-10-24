# Contributing

Use an isolated environment and install the required extras before changing code:

```bash
uv venv experiments/.venv --python python3
uv pip install --python experiments/.venv/bin/python -e ".[corpus,vectors,dev]"
```

Run the local quality gate before proposing a change:

```bash
PYTHONPATH=src experiments/.venv/bin/python -m pytest -q
experiments/.venv/bin/ruff check src tests
experiments/.venv/bin/pyright
```

Do not add secrets, private research corpora, generated NLP models or derived
project artifacts to version control. Methodological changes require updated skill
documentation, manifest versioning and focused evidence/scope tests.

By submitting a contribution, you make it available under this repository's
AGPL-3.0-only license. Submit only work you have the right to license. Unless a
contributor separately grants those rights in writing, maintainers cannot include
that contribution in a proprietary commercial license for DiscourseLens.
