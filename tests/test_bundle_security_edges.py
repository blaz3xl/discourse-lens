"""Focused adversarial boundaries for portable research-bundle verification."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from discourselens import Project
from discourselens import bundle as bundle_module
from discourselens.bundle import (
    export_bundle,
    export_bundle_to_directory_fd,
    supports_descriptor_relative_bundle_export,
    verify_bundle,
)
from discourselens.errors import ProjectError


def _resign_bundle(path: Path, payload: dict[str, object]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "integrity"}
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "digest": hashlib.sha256(canonical).hexdigest(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def exported_manifest(tmp_path):
    project = Project.init(tmp_path / "project")
    try:
        _, revision_id = project.import_text("source.txt", "The selected source stays local.")
        run_id = project.create_run(
            project.snapshot([revision_id]), {"profile": "bundle-security-fixture"}
        )
        yield export_bundle(project, run_id, tmp_path / "bundle")
    finally:
        project.close()


def test_verifier_accepts_an_explicit_symlinked_manifest_read_target(exported_manifest, tmp_path):
    link = tmp_path / "manifest-link.json"
    try:
        link.symlink_to(exported_manifest)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    assert verify_bundle(link)["status"] == "VALID"
    assert link.is_symlink()


def test_exported_manifest_can_be_copied_and_reopened(exported_manifest, tmp_path):
    copied_manifest = tmp_path / "portable-copy" / "manifest.json"
    copied_manifest.parent.mkdir()
    copied_manifest.write_bytes(exported_manifest.read_bytes())

    result = verify_bundle(copied_manifest)

    assert result["status"] == "VALID"
    assert result["integrity"] == "VERIFIED_SHA256"


def test_bundle_export_writes_manifest_relative_to_pinned_directory_fd(tmp_path):
    if not supports_descriptor_relative_bundle_export():
        pytest.skip("descriptor-relative bundle writes are unavailable")

    project = Project.init(tmp_path / "fd-export-project")
    try:
        _, revision_id = project.import_text("source.txt", "A descriptor-relative export.")
        run_id = project.create_run(
            project.snapshot([revision_id]), {"profile": "fd-export-fixture"}
        )
        destination = tmp_path / "pinned-bundle"
        destination.mkdir()
        directory_fd = os.open(
            destination, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            serialized = export_bundle_to_directory_fd(project, run_id, directory_fd)
        finally:
            os.close(directory_fd)

        manifest = destination / "manifest.json"
        assert manifest.read_bytes() == serialized
        assert verify_bundle(serialized)["integrity"] == "VERIFIED_SHA256"
    finally:
        project.close()


def test_verifier_rejects_unrecognized_traversal_reference_without_opening_it(
    exported_manifest, tmp_path, monkeypatch
):
    outside = tmp_path / "outside.json"
    outside.write_text("keep this sentinel", encoding="utf-8")
    payload = json.loads(exported_manifest.read_text(encoding="utf-8"))
    payload["external_files"] = [
        {"path": os.path.relpath(outside, exported_manifest.parent)}
    ]
    _resign_bundle(exported_manifest, payload)

    original_open = os.open
    opened: list[Path] = []
    manifest_target = exported_manifest.resolve()

    def guarded_open(path, flags, *args, **kwargs):
        target = Path(path).resolve()
        opened.append(target)
        if target != manifest_target:
            raise AssertionError("bundle verification followed a manifest path reference")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(bundle_module.os, "open", guarded_open)
    with pytest.raises(ProjectError, match="unsupported top-level fields"):
        verify_bundle(exported_manifest)
    assert opened == [manifest_target]

    monkeypatch.setattr(bundle_module.os, "open", original_open)
    assert outside.read_text(encoding="utf-8") == "keep this sentinel"


def test_verifier_rejects_named_pipe_without_blocking(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes are unavailable")
    manifest = tmp_path / "manifest.fifo"
    os.mkfifo(manifest)

    with pytest.raises(ProjectError, match="regular file"):
        verify_bundle(manifest)


def test_verifier_bounds_deep_malformed_json_before_schema_walk(tmp_path):
    manifest = tmp_path / "deep.json"
    manifest.write_text("[" * 10_000 + "0" + "]" * 10_000, encoding="utf-8")

    with pytest.raises(ProjectError, match="unreadable or invalid JSON"):
        verify_bundle(manifest)


def test_verifier_accepts_exact_size_limit_and_rejects_one_byte_over(
    exported_manifest, monkeypatch
):
    size = exported_manifest.stat().st_size
    monkeypatch.setattr("discourselens.bundle.MAX_BUNDLE_MANIFEST_BYTES", size)
    assert verify_bundle(exported_manifest)["status"] == "VALID"

    monkeypatch.setattr("discourselens.bundle.MAX_BUNDLE_MANIFEST_BYTES", size - 1)
    with pytest.raises(ProjectError, match="configured size limit"):
        verify_bundle(exported_manifest)


def test_canonical_digest_detects_corruption_but_does_not_authenticate_authorship(
    exported_manifest,
):
    payload = json.loads(exported_manifest.read_text(encoding="utf-8"))
    payload["analysis_run"]["config"]["profile"] = "edited without a new digest"
    exported_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProjectError, match="integrity check failed"):
        verify_bundle(exported_manifest)

    _resign_bundle(exported_manifest, payload)
    result = verify_bundle(exported_manifest)
    assert result["status"] == "VALID"
    assert result["integrity"] == "VERIFIED_SHA256"
