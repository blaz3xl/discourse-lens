from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
import sys

mods=["fitz","pydantic","spacy","duckdb","pyarrow","mcp","sqlite_vec","lancedb","faiss"]
print(json.dumps({"python":sys.version,"uv":shutil.which("uv"),"pipx":shutil.which("pipx"),"fts5":any("ENABLE_FTS5" in x[0] for x in sqlite3.connect(":memory:").execute("pragma compile_options")),"modules":{m:bool(importlib.util.find_spec(m)) for m in mods}},indent=2))
