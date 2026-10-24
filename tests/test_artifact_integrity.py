import os
from pathlib import Path

import pytest

from discourselens import artifact_integrity
from discourselens.artifact_integrity import verify_project_artifact
from discourselens.errors import ArtifactIntegrityError


def test_embedded_null_artifact_path_raises_integrity_error(tmp_path: Path) -> None:
    (tmp_path / ".discourselens").mkdir()

    with pytest.raises(ArtifactIntegrityError, match="path is invalid or unavailable"):
        verify_project_artifact(tmp_path, "artifact\x00.parquet", None)


@pytest.mark.parametrize("relative_path", [None, b"artifact.parquet", 17])
def test_non_text_artifact_path_raises_integrity_error(tmp_path: Path, relative_path) -> None:
    (tmp_path / ".discourselens").mkdir()

    with pytest.raises(ArtifactIntegrityError, match="path is invalid or unavailable"):
        verify_project_artifact(tmp_path, relative_path, None)  # type: ignore[arg-type]


def test_non_text_artifact_hash_raises_integrity_error(tmp_path: Path) -> None:
    hidden = tmp_path / ".discourselens"
    hidden.mkdir()
    artifact = hidden / "artifact.parquet"
    artifact.write_bytes(b"data")

    with pytest.raises(ArtifactIntegrityError, match="integrity metadata is invalid"):
        verify_project_artifact(tmp_path, "artifact.parquet", b"invalid hash")  # type: ignore[arg-type]


def test_hash_cache_distinguishes_filesystem_device(monkeypatch: pytest.MonkeyPatch) -> None:
    fingerprints: list[tuple[int, int, int, int, int]] = []

    def hash_matches(
        path: Path,
        expected_hash: str,
        expected_fingerprint: tuple[int, int, int, int, int],
    ) -> bool:
        del path, expected_hash
        fingerprints.append(expected_fingerprint)
        return True

    monkeypatch.setattr(artifact_integrity, "_file_hash_matches", hash_matches)
    artifact_integrity._cached_hash_matches.cache_clear()
    try:
        assert artifact_integrity._cached_hash_matches("artifact.parquet", "expected", 4, 1, 2, 10, 20)
        assert artifact_integrity._cached_hash_matches("artifact.parquet", "expected", 4, 1, 2, 11, 20)
        assert fingerprints == [(10, 20, 4, 1, 2), (11, 20, 4, 1, 2)]
    finally:
        artifact_integrity._cached_hash_matches.cache_clear()


def test_hash_verification_rejects_symlink_replacement_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hidden = tmp_path / ".discourselens"
    hidden.mkdir()
    artifact = hidden / "artifact.parquet"
    artifact.write_bytes(b"inside project")
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"outside project")
    expected_hash = artifact_integrity.sha256_file(outside)
    original_hash_file = artifact_integrity.sha256_file

    def replace_then_hash(path: Path) -> str:
        path.unlink()
        try:
            path.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"symlinks are unavailable: {exc}")
        return original_hash_file(path)

    monkeypatch.setattr(artifact_integrity, "sha256_file", replace_then_hash)
    artifact_integrity._cached_hash_matches.cache_clear()
    try:
        with pytest.raises(ArtifactIntegrityError, match="integrity check failed"):
            verify_project_artifact(tmp_path, "artifact.parquet", expected_hash)
    finally:
        artifact_integrity._cached_hash_matches.cache_clear()


def test_cached_hash_verification_rejects_file_changed_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hidden = tmp_path / ".discourselens"
    hidden.mkdir()
    artifact = hidden / "artifact.parquet"
    artifact.write_bytes(b"trusted!")
    expected_hash = artifact_integrity.sha256_file(artifact)

    def replace_during_cache_lookup(
        path: str,
        expected_hash_arg: str,
        size_bytes: int,
        modified_ns: int,
        changed_ns: int,
        device: int,
        inode: int,
    ) -> bool:
        del expected_hash_arg, size_bytes, modified_ns, changed_ns, device, inode
        Path(path).write_bytes(b"tampered")
        return True

    monkeypatch.setattr(artifact_integrity, "_cached_hash_matches", replace_during_cache_lookup)

    with pytest.raises(ArtifactIntegrityError, match="integrity check failed"):
        verify_project_artifact(tmp_path, "artifact.parquet", expected_hash)


def test_legacy_artifact_rejects_symlink_retargeted_during_path_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hidden = tmp_path / ".discourselens"
    hidden.mkdir()
    artifact = hidden / "artifact.parquet"
    artifact.write_bytes(b"inside project")
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"outside project")
    original_is_file = Path.is_file
    retargeted = False

    def retarget_before_file_check(path: Path) -> bool:
        nonlocal retargeted
        if path == artifact and not retargeted:
            path.unlink()
            try:
                path.symlink_to(outside)
            except OSError as exc:
                pytest.skip(f"symlinks are unavailable: {exc}")
            retargeted = True
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", retarget_before_file_check)

    with pytest.raises(ArtifactIntegrityError, match="path is invalid or unavailable"):
        verify_project_artifact(tmp_path, "artifact.parquet", None)

    assert retargeted


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are unavailable")
def test_sha256_file_rejects_named_pipe_without_blocking(tmp_path: Path) -> None:
    named_pipe = tmp_path / "artifact.fifo"
    os.mkfifo(named_pipe)

    with pytest.raises(OSError, match="regular file"):
        artifact_integrity.sha256_file(named_pipe)
