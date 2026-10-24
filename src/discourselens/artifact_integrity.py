"""Project-local integrity helpers for rebuildable derived artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

from .errors import ArtifactIntegrityError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as artifact:
        opened_before = os.fstat(artifact.fileno())
        if not stat.S_ISREG(opened_before.st_mode):
            raise OSError("artifact is not a regular file")
        path_before = path.stat()
        if _stat_fingerprint(opened_before) != _stat_fingerprint(path_before):
            raise OSError("artifact changed while it was being opened")
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
        opened_after = os.fstat(artifact.fileno())
        path_after = path.stat()
        fingerprint = _stat_fingerprint(opened_before)
        if (
            _stat_fingerprint(opened_after) != fingerprint
            or _stat_fingerprint(path_after) != fingerprint
        ):
            raise OSError("artifact changed while it was being hashed")
    return digest.hexdigest()


def _stat_fingerprint(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _file_hash_matches(
    path: Path,
    expected_hash: str,
    expected_fingerprint: tuple[int, int, int, int, int],
) -> bool:
    try:
        if _stat_fingerprint(path.stat()) != expected_fingerprint:
            return False
        matches = sha256_file(path) == expected_hash
        return matches and _stat_fingerprint(path.stat()) == expected_fingerprint
    except OSError:
        # A disappeared or replaced path during verification is an integrity
        # failure, including an O_NOFOLLOW rejection during the pinned open.
        return False


@lru_cache(maxsize=1024)
def _cached_hash_matches(
    path: str,
    expected_hash: str,
    size_bytes: int,
    modified_ns: int,
    changed_ns: int,
    device: int,
    inode: int,
) -> bool:
    # Include change time and the full device/inode identity so restoring mtime
    # after a same-size write, or replacing the mounted file, cannot reuse a
    # stale process-local integrity result. Recheck the fingerprint around the
    # hash so a replacement between stat and open cannot validate another file.
    expected_fingerprint = (device, inode, size_bytes, modified_ns, changed_ns)
    return _file_hash_matches(Path(path), expected_hash, expected_fingerprint)


def verify_project_artifact(
    project_root: Path, relative_path: str, expected_hash: str | None
) -> Path:
    """Resolve an artifact inside the project and verify its persisted byte hash.

    Legacy artifacts created before file hashes were stored remain readable, but
    new artifacts fail closed on missing files, path escapes, or byte changes.
    """
    # These values usually come from SQLite. Its dynamic typing means a damaged
    # or hand-edited registry can still return a BLOB/number despite the declared
    # TEXT column type; reject those values at this boundary with the project
    # error callers already know how to handle.
    if not isinstance(relative_path, str):
        raise ArtifactIntegrityError("derived artifact path is invalid or unavailable")
    if expected_hash is not None and not isinstance(expected_hash, str):
        raise ArtifactIntegrityError("derived artifact integrity metadata is invalid; rebuild it")

    try:
        relative = PurePosixPath(relative_path)
        windows_relative = PureWindowsPath(relative_path)
        if (
            relative.is_absolute()
            or windows_relative.is_absolute()
            or windows_relative.drive
            or ".." in relative.parts
            or ".." in windows_relative.parts
        ):
            raise ArtifactIntegrityError("derived artifact path is invalid or unavailable")
        project = project_root.resolve(strict=True)
        expected_root = project / ".discourselens"
        root = expected_root.resolve(strict=True)
        if root != expected_root or not root.is_dir():
            raise ArtifactIntegrityError("derived artifact path is invalid or unavailable")
        candidate = root.joinpath(*relative.parts).resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ArtifactIntegrityError("derived artifact path is invalid or unavailable")
        # `is_file()` follows symlinks and can race with an entry replacement.
        # Re-resolve even for legacy rows without a file hash, which otherwise
        # returned the checked path without detecting an external retarget.
        if candidate.resolve(strict=True) != candidate or not candidate.is_relative_to(root):
            raise ArtifactIntegrityError("derived artifact path is invalid or unavailable")
        if expected_hash is not None:
            stat = candidate.stat()
            # Windows exposes ctime as creation time, not reliable content-change
            # time. Hash on every verification there rather than cache on weak metadata.
            expected_fingerprint = _stat_fingerprint(stat)
            matches = (
                _file_hash_matches(candidate, expected_hash, expected_fingerprint)
                if os.name == "nt"
                else _cached_hash_matches(
                    str(candidate),
                    expected_hash,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                    stat.st_dev,
                    stat.st_ino,
                )
            )
            if not matches:
                raise ArtifactIntegrityError("derived artifact integrity check failed; rebuild it")
            # A positive cache entry is keyed by the fingerprint observed before
            # lookup. The file could change after that stat and before the cache
            # returns, so confirm the path still names that exact version before
            # handing it to a caller.
            if _stat_fingerprint(candidate.stat()) != expected_fingerprint:
                raise ArtifactIntegrityError("derived artifact integrity check failed; rebuild it")
            if candidate.resolve(strict=True) != candidate or not candidate.is_relative_to(root):
                raise ArtifactIntegrityError("derived artifact path is invalid or unavailable")
        return candidate
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactIntegrityError("derived artifact path is invalid or unavailable") from exc
