"""Conceptual portability check: logical IDs exclude a project root."""
from __future__ import annotations

import hashlib

payload=b"Apples are green."
logical_id=hashlib.sha256(b"document-revision-v1\0"+payload).hexdigest()
roots=["/home/alice/research/project","/workspace/project",r"C:\\Users\\Bob\\Research\\project"]
assert len({logical_id for _ in roots}) == 1
print(logical_id)
