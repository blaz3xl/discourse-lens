from pathlib import Path

import pytest

from discourselens.errors import ProjectError
from discourselens.tenant_storage import tenant_directory_name, tenant_project_path


def test_mcp_tenant_paths_are_stable_private_and_contained(tmp_path: Path):
    root = tmp_path / "shared-storage"
    secret_key = "tenant-identification-key-" * 2
    first = tenant_project_path(root, "https://identity.test/realm", "user-1", secret_key)
    assert first == tenant_project_path(root, "https://identity.test/realm", "user-1", secret_key)
    assert first != tenant_project_path(root, "https://identity.test/realm", "user-2", secret_key)
    assert first.parent == root.resolve() / "tenants"
    assert len(tenant_directory_name("issuer", "user", secret_key)) == 64
    with pytest.raises(ProjectError, match="at least 32"):
        tenant_directory_name("issuer", "user", "short")


def test_mcp_tenant_path_rejects_symlinked_tenant_root(tmp_path: Path):
    root = tmp_path / "storage"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    (root / "tenants").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProjectError, match="symbolic link"):
        tenant_project_path(root, "https://identity.test/realm", "user", "x" * 40)
