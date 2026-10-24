"""Project-local non-secret configuration with a deliberately small contract."""

from __future__ import annotations

import errno
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import ProjectError

DEFAULT_IMPORT_MAX_BYTES = 200 * 1024 * 1024
MAX_PROJECT_CONFIG_BYTES = 64 * 1024
_SECRET_MARKERS = {"api_key", "apikey", "credential", "password", "secret", "token"}
IMPORT_MAX_BYTES_ENV = "DISCOURSELENS_IMPORT_MAX_BYTES"


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_format: int = Field(default=1, ge=1, strict=True)
    import_max_bytes: int = Field(default=DEFAULT_IMPORT_MAX_BYTES, ge=1, strict=True)


def load_project_config(root: str | Path) -> ProjectConfig:
    project_root = Path(root)
    if not _supports_directory_relative_config_io():
        if _project_config_is_absent_without_opening(project_root):
            return ProjectConfig()
        raise ProjectError(
            "safe project configuration access is unavailable for an existing config on this platform"
        )
    raw = _load_project_config_from_directory(project_root)
    return _validate_config_mapping(raw, source="project")


def _project_config_is_absent_without_opening(root: Path) -> bool:
    """Check only path metadata when secure project-config I/O is unavailable."""
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise ProjectError("project configuration directory is unavailable or unsafe") from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ProjectError("project configuration directory is unavailable or unsafe")

    config_path = root / "discourselens.yaml"
    try:
        config_metadata = config_path.lstat()
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise ProjectError("project configuration cannot be inspected safely") from exc
    if stat.S_ISLNK(config_metadata.st_mode):
        raise ProjectError("project configuration must not be a symlink")
    if not stat.S_ISREG(config_metadata.st_mode):
        raise ProjectError("project configuration must be a regular file")
    return False


def resolve_project_config(
    root: str | Path,
    *,
    user_config_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    cli_import_max_bytes: int | None = None,
) -> ProjectConfig:
    """Resolve import settings in CLI, project, user, environment, default order."""
    selected_environment = os.environ if environ is None else environ
    selected_user_path = (
        Path(user_config_path)
        if user_config_path is not None
        else _default_user_config_path(selected_environment)
    )

    defaults = ProjectConfig()
    values = defaults.model_dump()
    environment_values = _environment_config_values(selected_environment)
    user_values = _load_config_mapping(selected_user_path, source="user")
    validated_user = _validate_config_mapping(user_values, source="user")
    project_config = load_project_config(root)

    values.update(environment_values)
    values.update({name: getattr(validated_user, name) for name in validated_user.model_fields_set})
    values.update({name: getattr(project_config, name) for name in project_config.model_fields_set})
    if cli_import_max_bytes is not None:
        values["import_max_bytes"] = cli_import_max_bytes

    try:
        validated = ProjectConfig.model_validate(values)
    except ValidationError:
        raise ProjectError("configuration is invalid") from None

    # Preserve only the fields explicitly stored in the project. This keeps a later
    # write of ``project.config`` from freezing a user or environment value into it.
    return ProjectConfig.model_construct(
        **validated.model_dump(),
        _fields_set=set(project_config.model_fields_set),
    )


def _default_user_config_path(environ: Mapping[str, str]) -> Path:
    config_home = environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "discourselens" / "config.yaml"
    home = environ.get("HOME")
    home_path = Path(home) if home else Path.home()
    return home_path / ".config" / "discourselens" / "config.yaml"


def _environment_config_values(environ: Mapping[str, str]) -> dict[str, int]:
    # Read only declared settings. Other ``DISCOURSELENS_*`` variables may be
    # provider credential references; treating them as config can make an
    # otherwise valid project fail merely because a selected credential exists.
    import_limit_values = [
        value
        for name, value in environ.items()
        if name.casefold() == IMPORT_MAX_BYTES_ENV.casefold()
    ]

    if not import_limit_values:
        return {}
    if len(set(import_limit_values)) > 1:
        raise ProjectError("environment configuration is ambiguous")
    raw_value = import_limit_values[0]
    # Environment variables are strings by definition. Parse only a decimal integer here,
    # then let the strict ProjectConfig field enforce the actual integer contract.
    if not raw_value.isascii() or not raw_value.isdecimal():
        raise ProjectError("environment configuration is invalid")
    try:
        parsed_value = int(raw_value)
    except ValueError as exc:
        # Python can reject very long decimal strings before numeric validation. Keep
        # malformed environment settings within the configuration error contract.
        raise ProjectError("environment configuration is invalid") from exc
    validated = _validate_config_mapping({"import_max_bytes": parsed_value}, source="environment")
    return {"import_max_bytes": validated.import_max_bytes}


def _load_config_mapping(path: Path, *, source: str) -> dict[str, Any]:
    path_label = f"{source} configuration"
    if path.is_symlink():
        raise ProjectError(f"{path_label} must not be a symlink")
    if not path.exists():
        return {}
    if not path.is_file():
        raise ProjectError(f"{path_label} must be a regular file")
    descriptor = -1
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_BINARY", 0)
        )
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ProjectError(f"{path_label} must be a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            encoded = stream.read(MAX_PROJECT_CONFIG_BYTES + 1)
        if len(encoded) > MAX_PROJECT_CONFIG_BYTES:
            raise ProjectError(f"{path_label} exceeds the {MAX_PROJECT_CONFIG_BYTES}-byte limit")
        raw = yaml.safe_load(encoded.decode("utf-8"))
    except ProjectError:
        raise
    except (OSError, UnicodeDecodeError, yaml.YAMLError, RecursionError):
        # Parser exceptions can retain snippets of the user's config contents.
        raise ProjectError(f"{path_label} cannot be read") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ProjectError(f"{path_label} must be a YAML mapping")
    try:
        contains_secret = _contains_secret_key(raw)
    except ValueError as exc:
        raise ProjectError(f"{path_label} must not contain recursive YAML aliases") from exc
    if contains_secret:
        raise ProjectError(f"{path_label} must not contain secrets")
    return raw


def _validate_config_mapping(raw: dict[str, Any], *, source: str) -> ProjectConfig:
    if source == "user" and any(key != "import_max_bytes" for key in raw):
        raise ProjectError("user configuration is invalid")
    try:
        return ProjectConfig.model_validate(raw)
    except ValidationError:
        # Pydantic errors include rejected input values, which may be credentials.
        raise ProjectError(f"{source} configuration is invalid") from None


def write_project_config(root: str | Path, config: ProjectConfig) -> None:
    """Write only validated non-secret project settings in a portable YAML form."""
    project_root = Path(root)
    path = project_root / "discourselens.yaml"
    try:
        # Pydantic's model_copy(update=...) and model_construct() deliberately skip
        # validation. Revalidate at the persistence boundary so invalid/secret-like
        # values cannot be serialized through an otherwise typed ProjectConfig.
        candidate = {
            field_name: getattr(config, field_name) for field_name in ProjectConfig.model_fields
        }
        validated = ProjectConfig.model_validate(candidate)
    except ValidationError:
        raise ProjectError("project configuration is invalid") from None
    portable_values: dict[str, int] = {"project_format": validated.project_format}
    if "import_max_bytes" in config.model_fields_set:
        portable_values["import_max_bytes"] = validated.import_max_bytes
    payload = yaml.safe_dump(portable_values, sort_keys=True)
    safe_directory_io = _supports_directory_relative_config_io()
    if not config.model_fields_set:
        if _project_config_is_absent_without_opening(project_root):
            return
        if not safe_directory_io:
            raise ProjectError(
                "safe project configuration access is unavailable for an existing config on this platform"
            )
    if not safe_directory_io:
        raise ProjectError("safe project configuration writes are unavailable on this platform")
    if path.is_symlink():
        raise ProjectError("project configuration must not be a symlink")
    _write_project_config_in_directory(project_root, payload)


def _supports_directory_relative_config_io() -> bool:
    return (
        os.name == "posix"
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.open in os.supports_dir_fd
        and os.rename in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
    )


def _open_pinned_project_directory(root: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(root, flags)
        opened = os.fstat(descriptor)
        current = os.stat(root, follow_symlinks=False)
    except OSError as exc:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise ProjectError("project configuration directory is unavailable or unsafe") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
    ):
        os.close(descriptor)
        raise ProjectError("project configuration directory changed during access")
    return descriptor


def _read_config_mapping_at_directory(directory_fd: int) -> dict[str, Any]:
    path_label = "project configuration"
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open("discourselens.yaml", flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ProjectError(f"{path_label} must not be a symlink") from None
        raise ProjectError(f"{path_label} cannot be read safely") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProjectError(f"{path_label} must be a regular file")
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        with stream:
            encoded = stream.read(MAX_PROJECT_CONFIG_BYTES + 1)
        if len(encoded) > MAX_PROJECT_CONFIG_BYTES:
            raise ProjectError(f"{path_label} exceeds the {MAX_PROJECT_CONFIG_BYTES}-byte limit")
        raw = yaml.safe_load(encoded.decode("utf-8"))
    except ProjectError:
        raise
    except (OSError, UnicodeDecodeError, yaml.YAMLError, RecursionError):
        # Parser exceptions can retain snippets of the user's config contents.
        raise ProjectError(f"{path_label} cannot be read") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ProjectError(f"{path_label} must be a YAML mapping")
    try:
        contains_secret = _contains_secret_key(raw)
    except ValueError as exc:
        raise ProjectError(f"{path_label} must not contain recursive YAML aliases") from exc
    if contains_secret:
        raise ProjectError(f"{path_label} must not contain secrets")
    return raw


def _load_project_config_from_directory(root: Path) -> dict[str, Any]:
    directory_fd = _open_pinned_project_directory(root)
    try:
        return _read_config_mapping_at_directory(directory_fd)
    finally:
        os.close(directory_fd)


def _write_project_config_in_directory(root: Path, payload: str) -> None:
    directory_fd = _open_pinned_project_directory(root)
    temporary_name: str | None = None
    descriptor = -1
    try:
        # A random exclusive basename plus dir_fd keeps both temporary creation
        # and atomic replacement anchored to the opened project directory.
        for _ in range(10):
            candidate = f".discourselens-config-{secrets.token_hex(12)}"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor < 0 or temporary_name is None:
            raise ProjectError("project configuration temporary file cannot be reserved")

        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            descriptor = -1
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())

        opened = os.fstat(directory_fd)
        current = os.stat(root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise ProjectError("project configuration directory changed during write")
        try:
            existing = os.stat("discourselens.yaml", dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and stat.S_ISLNK(existing.st_mode):
            raise ProjectError("project configuration must not be a symlink")

        os.rename(
            temporary_name,
            "discourselens.yaml",
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = None
        os.fsync(directory_fd)
    except ProjectError:
        raise
    except OSError as exc:
        raise ProjectError("project configuration cannot be written safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        os.close(directory_fd)


def _contains_secret_key(
    value: Any,
) -> bool:
    visited: set[int] = set()
    active: set[int] = set()
    stack: list[tuple[Any, bool]] = [(value, False)]
    while stack:
        current, leaving = stack.pop()
        if not isinstance(current, (dict, list)):
            continue
        container_id = id(current)
        if leaving:
            active.remove(container_id)
            visited.add(container_id)
            continue
        if container_id in active:
            raise ValueError("recursive YAML alias")
        if container_id in visited:
            continue
        active.add(container_id)
        stack.append((current, True))
        if isinstance(current, dict):
            for key, child in current.items():
                if _is_secret_key(str(key)):
                    return True
                stack.append((child, False))
        else:
            stack.extend((child, False) for child in current)
    return False


def _is_secret_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_").replace(" ", "_")
    if normalized in _SECRET_MARKERS:
        return True
    parts = set(normalized.split("_"))
    return bool(
        parts & {"credential", "credentials", "password", "secret", "token"}
        or ("api" in parts and "key" in parts)
        or "apikey" in parts
    )
