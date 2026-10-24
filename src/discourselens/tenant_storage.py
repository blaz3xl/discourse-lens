"""Private workspace paths for MCP identities authenticated by an external provider."""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

from .errors import ProjectError


def tenant_directory_name(issuer: str, subject: str, key: str) -> str:
    """Derive an opaque stable directory name without exposing identity values."""
    key_bytes = key.encode("utf-8")
    if len(key_bytes) < 32:
        raise ProjectError("DISCOURSELENS_TENANT_ID_KEY must contain at least 32 UTF-8 bytes")
    message = (issuer + "\0" + subject).encode("utf-8")
    return hmac.new(key_bytes, message, hashlib.sha256).hexdigest()


def tenant_project_path(
    root: str | os.PathLike[str], issuer: str, subject: str, key: str
) -> Path:
    """Resolve an identity only to its deterministic, contained tenant directory."""
    storage_root = Path(root).resolve()
    tenant_root = storage_root / "tenants"
    if tenant_root.is_symlink():
        raise ProjectError("tenant storage root must not be a symbolic link")
    digest = tenant_directory_name(issuer, subject, key)
    project_root = tenant_root / digest
    if project_root.is_symlink() or not project_root.resolve().is_relative_to(tenant_root.resolve()):
        raise ProjectError("tenant project directory is unsafe")
    return project_root
