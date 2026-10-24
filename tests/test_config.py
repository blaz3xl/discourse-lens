import os
import traceback
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

import discourselens.config as config_module
from discourselens import Project
from discourselens.analysis import run_exact_phrase_analysis
from discourselens.bundle import export_bundle
from discourselens.cli import app
from discourselens.config import (
    DEFAULT_IMPORT_MAX_BYTES,
    IMPORT_MAX_BYTES_ENV,
    MAX_PROJECT_CONFIG_BYTES,
    ProjectConfig,
    load_project_config,
    resolve_project_config,
    write_project_config,
)
from discourselens.documents import import_document
from discourselens.errors import DocumentError, ProjectError
from discourselens.models import CredentialReference


def test_project_config_is_portable_non_secret_and_controls_import_limit(tmp_path):
    if not config_module._supports_directory_relative_config_io():
        pytest.skip("explicit project YAML requires safe directory-relative I/O")
    project = Project.init(tmp_path / "project")
    project.set_project_config(ProjectConfig(import_max_bytes=4))
    source = tmp_path / "source.txt"
    source.write_text("five!", encoding="utf-8")

    assert load_project_config(project.root).import_max_bytes == 4
    with pytest.raises(DocumentError, match="configured import size limit"):
        import_document(project, source)


def test_project_config_serialization_has_no_absolute_path_identity(tmp_path):
    if not config_module._supports_directory_relative_config_io():
        pytest.skip("explicit project YAML requires safe directory-relative I/O")
    roots = (tmp_path / "first-host" / "project", tmp_path / "second-host" / "project")
    config = ProjectConfig(import_max_bytes=4321)

    for root in roots:
        root.mkdir(parents=True)
        write_project_config(root, config)

    payloads = [(root / "discourselens.yaml").read_text(encoding="utf-8") for root in roots]
    assert payloads[0] == payloads[1]
    assert all(str(root) not in payload for root, payload in zip(roots, payloads, strict=True))
    assert [load_project_config(root) for root in roots] == [config, config]


def test_import_limit_precedence_cli_project_user_environment_default(tmp_path):
    if not config_module._supports_directory_relative_config_io():
        pytest.skip("explicit project YAML requires safe directory-relative I/O")
    root = tmp_path / "project"
    project = Project.init(root)
    project.close()
    project_config_path = root / "discourselens.yaml"
    user_config_path = tmp_path / "user-config.yaml"
    user_config_path.write_text("import_max_bytes: 300\n", encoding="utf-8")
    environ = {IMPORT_MAX_BYTES_ENV: "200", "OPENAI_API_KEY": "sk-env-secret-fixture"}

    assert not project_config_path.exists()
    assert (
        resolve_project_config(
            root,
            user_config_path=user_config_path,
            environ=environ,
        ).import_max_bytes
        == 300
    )
    resolved = resolve_project_config(
        root,
        user_config_path=user_config_path,
        environ=environ,
    )
    assert "sk-env-secret-fixture" not in str(resolved.model_dump())

    project_config_path.write_text("project_format: 1\nimport_max_bytes: 400\n", encoding="utf-8")
    assert (
        resolve_project_config(
            root,
            user_config_path=user_config_path,
            environ=environ,
        ).import_max_bytes
        == 400
    )
    assert (
        resolve_project_config(
            root,
            user_config_path=user_config_path,
            environ=environ,
            cli_import_max_bytes=500,
        ).import_max_bytes
        == 500
    )

    project_config_path.write_text("project_format: 1\n", encoding="utf-8")
    assert (
        resolve_project_config(
            root,
            user_config_path=tmp_path / "missing-user-config.yaml",
            environ=environ,
        ).import_max_bytes
        == 200
    )
    assert (
        resolve_project_config(
            root,
            user_config_path=tmp_path / "missing-user-config.yaml",
            environ={},
        ).import_max_bytes
        == DEFAULT_IMPORT_MAX_BYTES
    )


def test_project_config_uses_user_environment_and_default_sources_at_runtime(tmp_path, monkeypatch):
    if not config_module._supports_directory_relative_config_io():
        pytest.skip("explicit project YAML requires safe directory-relative I/O")
    root = tmp_path / "project"
    Project.init(root).close()
    config_home = tmp_path / "xdg"
    user_config_path = config_home / "discourselens" / "config.yaml"
    user_config_path.parent.mkdir(parents=True)
    user_config_path.write_text("import_max_bytes: 300\n", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv(IMPORT_MAX_BYTES_ENV, "200")

    project = Project.open(root)
    try:
        assert project.config.import_max_bytes == 300
        user_effective_config = project.config
        project.set_project_config(user_effective_config)
        assert project.config.import_max_bytes == 300

        project.set_project_config(ProjectConfig(import_max_bytes=400))
        assert project.config.import_max_bytes == 400

        # Rewriting only project metadata leaves the lower-precedence value live.
        project.set_project_config(ProjectConfig(project_format=1))
        assert project.config.import_max_bytes == 300

        user_config_path.unlink()
        assert project.config.import_max_bytes == 200

        monkeypatch.delenv(IMPORT_MAX_BYTES_ENV)
        assert project.config.import_max_bytes == DEFAULT_IMPORT_MAX_BYTES
    finally:
        project.close()


def test_legacy_generated_project_default_remains_an_explicit_override(tmp_path):
    if not config_module._supports_directory_relative_config_io():
        pytest.skip("existing project YAML requires safe directory-relative I/O")
    root = tmp_path / "legacy-project"
    root.mkdir()
    (root / "discourselens.yaml").write_text(
        f"project_format: 1\nimport_max_bytes: {DEFAULT_IMPORT_MAX_BYTES}\n", encoding="utf-8"
    )
    user_config_path = tmp_path / "user-config.yaml"
    user_config_path.write_text("import_max_bytes: 300\n", encoding="utf-8")

    assert (
        resolve_project_config(
            root,
            user_config_path=user_config_path,
            environ={IMPORT_MAX_BYTES_ENV: "200"},
        ).import_max_bytes
        == DEFAULT_IMPORT_MAX_BYTES
    )


@pytest.mark.parametrize("source", ["project", "user"])
def test_config_sources_reject_secret_settings_without_echoing_values(tmp_path, source):
    root = tmp_path / "project"
    Project.init(root).close()
    secret_value = "sk-never-echo-this-config-fixture"
    user_config_path = tmp_path / "user-config.yaml"
    environ: dict[str, str] = {}

    if source == "project":
        (root / "discourselens.yaml").write_text(
            f"provider_api_key: {secret_value}\n", encoding="utf-8"
        )
    elif source == "user":
        user_config_path.write_text(f"secret_value: {secret_value}\n", encoding="utf-8")
    with pytest.raises(ProjectError, match="must not contain secrets") as error:
        resolve_project_config(
            root,
            user_config_path=user_config_path,
            environ=environ,
        )
    assert secret_value not in str(error.value)


def test_provider_credential_with_config_prefix_does_not_break_or_pollute_project_config(
    tmp_path, monkeypatch
):
    if not config_module._supports_directory_relative_config_io():
        pytest.skip("explicit project YAML requires safe directory-relative I/O")
    root = tmp_path / "project"
    sentinel = "provider-fixture-value-71a9-never-serialized"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-user-config"))
    monkeypatch.setenv("DISCOURSELENS_PROVIDER_A_API_KEY", sentinel)
    monkeypatch.delenv(IMPORT_MAX_BYTES_ENV, raising=False)
    project = Project.init(root)
    try:
        project.save_credential_reference(
            CredentialReference(
                "openai", "alias-a", "DISCOURSELENS_PROVIDER_A_API_KEY"
            )
        )

        assert project.config.import_max_bytes == DEFAULT_IMPORT_MAX_BYTES
        project.set_project_config(ProjectConfig(import_max_bytes=1234))

        config_text = (root / "discourselens.yaml").read_text(encoding="utf-8")
        saved_reference = project.credential_reference("openai", "alias-a")
        assert saved_reference["secret_reference"] == "DISCOURSELENS_PROVIDER_A_API_KEY"
        assert sentinel not in config_text
        assert sentinel not in str(project.config.model_dump())
    finally:
        project.close()


@pytest.mark.parametrize(
    ("payload", "secret_value"),
    [
        (
            "import_max_bytes: sk-config-validation-fixture-never-secret\n",
            "sk-config-validation-fixture-never-secret",
        ),
        (
            "import_max_bytes: [sk-config-parser-fixture-never-secret\n",
            "sk-config-parser-fixture-never-secret",
        ),
    ],
)
def test_invalid_project_config_tracebacks_do_not_expose_embedded_secret_values(
    tmp_path, payload, secret_value
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "discourselens.yaml").write_text(payload, encoding="utf-8")

    with pytest.raises(ProjectError) as error:
        load_project_config(project_root)

    rendered_traceback = "".join(traceback.format_exception(error.value))
    assert secret_value not in rendered_traceback


def test_invalid_environment_import_limit_does_not_echo_secret_value(tmp_path):
    root = tmp_path / "project"
    Project.init(root).close()
    secret_value = "sk-environment-setting-value-never-echoed"

    with pytest.raises(ProjectError, match="environment configuration is invalid") as error:
        resolve_project_config(
            root,
            user_config_path=tmp_path / "missing-user-config.yaml",
            environ={IMPORT_MAX_BYTES_ENV: secret_value},
        )

    assert secret_value not in str(error.value)


def test_external_environment_secrets_never_enter_config_or_bundle_manifest(tmp_path, monkeypatch):
    root = tmp_path / "project"
    secret_value = "sk-provider-environment-secret-never-serialized"
    monkeypatch.setenv("DL_CONFIG_TEST_PROVIDER_KEY", secret_value)
    project = Project.init(root)
    try:
        project.save_credential_reference(
            CredentialReference("anthropic", "research", "DL_CONFIG_TEST_PROVIDER_KEY")
        )
        project.set_project_config(ProjectConfig(import_max_bytes=1234))
        _, revision = project.import_text("source.txt", "Evidence remains portable.")
        run_id = run_exact_phrase_analysis(project, [revision], "Evidence")["analysis_run_id"]
        manifest = export_bundle(project, run_id, tmp_path / "bundle")
        config_text = (root / "discourselens.yaml").read_text(encoding="utf-8")
        manifest_text = manifest.read_text(encoding="utf-8")

        assert secret_value not in config_text
        assert secret_value not in manifest_text
        assert str(root) not in config_text
        assert str(root) not in manifest_text
    finally:
        project.close()


def test_project_config_rejects_secret_keys_and_invalid_yaml_shape(tmp_path):
    root = tmp_path / "project"
    Project.init(root).close()
    secret_value = "sk-config-test-value-must-not-be-echoed"
    (root / "discourselens.yaml").write_text(f"api_key: {secret_value}\n", encoding="utf-8")
    with pytest.raises(ProjectError, match="must not contain secrets") as error:
        load_project_config(root)
    assert secret_value not in str(error.value)

    (root / "discourselens.yaml").write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ProjectError, match="YAML mapping"):
        load_project_config(root)


def test_project_config_is_bounded_before_parsing(tmp_path):
    root = tmp_path / "project"
    Project.init(root).close()
    (root / "discourselens.yaml").write_bytes(b" " * (MAX_PROJECT_CONFIG_BYTES + 1))

    with pytest.raises(ProjectError, match="byte limit"):
        load_project_config(root)


def test_project_config_rejects_recursive_yaml_aliases(tmp_path):
    root = tmp_path / "project"
    Project.init(root).close()
    (root / "discourselens.yaml").write_text("loop: &loop [*loop]\n", encoding="utf-8")

    with pytest.raises(ProjectError, match="recursive YAML aliases"):
        load_project_config(root)


@pytest.mark.parametrize("source", ["project", "user"])
def test_config_yaml_fifo_is_rejected_without_blocking(tmp_path, monkeypatch, source):
    if (
        not hasattr(os, "mkfifo")
        or not hasattr(os, "O_NONBLOCK")
        or not config_module._supports_directory_relative_config_io()
    ):
        pytest.skip("nonblocking FIFO/config file checks are unavailable")

    root = tmp_path / "project"
    Project.init(root).close()
    if source == "project":
        config_path = root / "discourselens.yaml"
        config_path.unlink(missing_ok=True)
        os.mkfifo(config_path)
    else:
        config_path = tmp_path / "user-config.yaml"
        config_path.write_text("import_max_bytes: 123\n", encoding="utf-8")

    original_open = config_module.os.open
    original_is_file = Path.is_file
    swapped = False

    def replace_user_config_after_type_check(path: Path) -> bool:
        nonlocal swapped
        result = original_is_file(path)
        if source == "user" and path == config_path and result and not swapped:
            path.unlink()
            os.mkfifo(path)
            swapped = True
        return result

    def require_nonblocking_config_open(path, flags, *args, **kwargs):
        logical_path = os.fspath(path)
        is_target = (
            logical_path == config_path.name
            if source == "project"
            else logical_path == os.fspath(config_path)
        )
        if is_target:
            # Fail promptly instead of allowing a regression to hang pytest on a
            # blocking FIFO open.
            assert flags & os.O_NONBLOCK
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(config_module, "_supports_directory_relative_config_io", lambda: True)
    monkeypatch.setattr(Path, "is_file", replace_user_config_after_type_check)
    monkeypatch.setattr(config_module.os, "open", require_nonblocking_config_open)

    with pytest.raises(ProjectError, match="must be a regular file"):
        if source == "project":
            load_project_config(root)
        else:
            resolve_project_config(root, user_config_path=config_path, environ={})
    assert source == "project" or swapped


def test_project_config_rejects_unknown_fields_instead_of_silently_ignoring_typos(tmp_path):
    root = tmp_path / "project"
    Project.init(root).close()
    (root / "discourselens.yaml").write_text("import_max_byte: 123\n", encoding="utf-8")

    with pytest.raises(ProjectError, match="configuration is invalid"):
        load_project_config(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [("project_format", "true"), ("import_max_bytes", "true")],
)
def test_project_config_rejects_boolean_values_for_integer_settings(tmp_path, field, value):
    root = tmp_path / "project"
    Project.init(root).close()
    (root / "discourselens.yaml").write_text(f"{field}: {value}\n", encoding="utf-8")

    with pytest.raises(ProjectError, match="configuration is invalid"):
        load_project_config(root)


def test_environment_import_limit_requires_decimal_integer(tmp_path):
    root = tmp_path / "project"
    Project.init(root).close()
    user_config_path = tmp_path / "missing-user-config.yaml"

    with pytest.raises(ProjectError, match="environment configuration is invalid"):
        resolve_project_config(
            root,
            user_config_path=user_config_path,
            environ={IMPORT_MAX_BYTES_ENV: "true"},
        )

    resolved = resolve_project_config(
        root,
        user_config_path=user_config_path,
        environ={IMPORT_MAX_BYTES_ENV: "123"},
    )
    assert resolved.import_max_bytes == 123


def test_environment_import_limit_rejects_conflicting_case_variants(tmp_path):
    root = tmp_path / "project"
    Project.init(root).close()

    with pytest.raises(ProjectError, match="environment configuration is ambiguous"):
        resolve_project_config(
            root,
            user_config_path=tmp_path / "missing-user-config.yaml",
            environ={
                IMPORT_MAX_BYTES_ENV: "123",
                "discourselens_import_max_bytes": "456",
            },
        )


def test_environment_import_limit_wraps_oversized_decimal_conversion(tmp_path):
    root = tmp_path / "project"
    Project.init(root).close()

    with pytest.raises(ProjectError, match="environment configuration is invalid") as error:
        resolve_project_config(
            root,
            user_config_path=tmp_path / "missing-user-config.yaml",
            environ={IMPORT_MAX_BYTES_ENV: "1" * 5_000},
        )

    assert "111111" not in str(error.value)


def test_project_config_cannot_read_or_overwrite_a_symlink_target(tmp_path):
    root = tmp_path / "project"
    project = Project.init(root)
    project.close()
    outside = tmp_path / "outside.yaml"
    outside.write_text("import_max_bytes: 7\n", encoding="utf-8")
    config = root / "discourselens.yaml"
    config.unlink(missing_ok=True)
    try:
        config.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")

    with pytest.raises(ProjectError, match="must not be a symlink"):
        load_project_config(root)
    project = Project.open(root)
    try:
        with pytest.raises(ProjectError, match="must not be a symlink"):
            project.set_project_config(ProjectConfig(import_max_bytes=3))
    finally:
        project.close()
    assert outside.read_text(encoding="utf-8") == "import_max_bytes: 7\n"


def test_project_config_update_does_not_truncate_a_hardlink_target(tmp_path):
    root = tmp_path / "project"
    project = Project.init(root)
    project.close()
    outside = tmp_path / "outside.yaml"
    outside.write_text("protected external file\n", encoding="utf-8")
    config = root / "discourselens.yaml"
    config.unlink(missing_ok=True)
    try:
        os.link(outside, config)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")

    project = Project.open(root)
    try:
        project.set_project_config(ProjectConfig(import_max_bytes=3))
        assert project.config.import_max_bytes == 3
    finally:
        project.close()
    assert outside.read_text(encoding="utf-8") == "protected external file\n"


def test_project_config_writer_pins_project_directory_across_root_symlink_swap(
    tmp_path, monkeypatch
):
    if not config_module._supports_directory_relative_config_io():
        pytest.skip("directory-relative config writes are unavailable")
    root = tmp_path / "project"
    root.mkdir()
    config_path = root / "discourselens.yaml"
    config_path.write_text("project_format: 1\nimport_max_bytes: 111\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_config = outside / "discourselens.yaml"
    sentinel = "outside config must not change\n"
    outside_config.write_text(sentinel, encoding="utf-8")
    saved_root = tmp_path / "project-original"
    original_open = config_module.os.open
    swapped = False

    def swap_root_before_temp_create(path, flags, *args, **kwargs):
        nonlocal swapped
        if (
            isinstance(path, str)
            and path.startswith(".discourselens-config-")
            and not swapped
        ):
            root.rename(saved_root)
            try:
                root.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                saved_root.rename(root)
                pytest.skip(f"directory symlinks unavailable: {exc}")
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(config_module, "_supports_directory_relative_config_io", lambda: True)
    monkeypatch.setattr(config_module.os, "open", swap_root_before_temp_create)
    with pytest.raises(ProjectError, match="directory changed during write"):
        write_project_config(root, ProjectConfig(import_max_bytes=222))

    assert swapped
    assert outside_config.read_text(encoding="utf-8") == sentinel
    root.unlink()
    saved_root.rename(root)
    assert config_path.read_text(encoding="utf-8") == "project_format: 1\nimport_max_bytes: 111\n"
    assert list(root.glob(".discourselens-config-*")) == []


def test_project_config_reader_fails_closed_without_pinned_directory_io(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    root.mkdir()
    config_path = root / "discourselens.yaml"
    original = "project_format: 1\nimport_max_bytes: 111\n"
    config_path.write_text(original, encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_config = outside / "discourselens.yaml"
    sentinel = "project_format: 1\nimport_max_bytes: 777\n"
    outside_config.write_text(sentinel, encoding="utf-8")
    saved_root = tmp_path / "project-original"
    original_open = config_module.os.open
    swapped = False

    def swap_root_before_config_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if os.fspath(path) == os.fspath(config_path) and not swapped:
            root.rename(saved_root)
            try:
                root.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                saved_root.rename(root)
                pytest.skip(f"directory symlinks unavailable: {exc}")
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(config_module, "_supports_directory_relative_config_io", lambda: False)
    monkeypatch.setattr(config_module.os, "open", swap_root_before_config_open)
    try:
        with pytest.raises(ProjectError, match="safe project configuration access is unavailable"):
            load_project_config(root)
    finally:
        if root.is_symlink():
            root.unlink()
        if saved_root.exists():
            saved_root.rename(root)

    assert not swapped
    assert outside_config.read_text(encoding="utf-8") == sentinel
    assert config_path.read_text(encoding="utf-8") == original


def test_project_config_writer_fails_closed_without_pinned_directory_io(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    root.mkdir()
    config_path = root / "discourselens.yaml"
    original = "project_format: 1\nimport_max_bytes: 111\n"
    config_path.write_text(original, encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_config = outside / "discourselens.yaml"
    sentinel = "outside config must not change\n"
    outside_config.write_text(sentinel, encoding="utf-8")
    saved_root = tmp_path / "project-original"
    original_open = config_module.os.open
    swapped = False

    def swap_root_before_temp_create(path, flags, *args, **kwargs):
        nonlocal swapped
        if (
            isinstance(path, str)
            and path.startswith(str(root / ".discourselens-config-"))
            and not swapped
        ):
            root.rename(saved_root)
            try:
                root.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                saved_root.rename(root)
                pytest.skip(f"directory symlinks unavailable: {exc}")
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(config_module, "_supports_directory_relative_config_io", lambda: False)
    monkeypatch.setattr(config_module.os, "open", swap_root_before_temp_create)
    try:
        with pytest.raises(ProjectError, match="safe project configuration writes are unavailable"):
            write_project_config(root, ProjectConfig(import_max_bytes=222))
    finally:
        if root.is_symlink():
            root.unlink()
        if saved_root.exists():
            saved_root.rename(root)

    assert not swapped
    assert outside_config.read_text(encoding="utf-8") == sentinel
    assert config_path.read_text(encoding="utf-8") == original
    assert list(root.glob(".discourselens-config-*")) == []


@pytest.mark.parametrize("force_unsupported_io", [False, True])
def test_project_init_and_default_config_work_without_project_yaml(
    tmp_path, monkeypatch, force_unsupported_io
):
    root = tmp_path / "project"
    if force_unsupported_io:
        monkeypatch.setattr(config_module, "_supports_directory_relative_config_io", lambda: False)
    elif not config_module._supports_directory_relative_config_io():
        pytest.skip("secure project YAML path is unavailable")
    monkeypatch.setattr(
        config_module,
        "_default_user_config_path",
        lambda environ: tmp_path / "missing-user-config.yaml",
    )
    monkeypatch.setattr(config_module, "_environment_config_values", lambda environ: {})

    project = Project.init(root)
    try:
        config_path = root / "discourselens.yaml"
        assert not config_path.exists()
        resolved = project.config
        assert resolved.project_format == 1
        assert resolved.import_max_bytes == DEFAULT_IMPORT_MAX_BYTES
        assert project.db.execute("SELECT 1").fetchone()[0] == 1
        _, revision_id = project.import_text("offline.txt", "Defaults remain usable.")
        assert project.revision_text(revision_id) == "Defaults remain usable."
    finally:
        project.close()

    assert not (root / "discourselens.yaml").exists()


def test_custom_project_config_is_rejected_without_pinned_directory_io(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    config_path = root / "discourselens.yaml"
    original = "project_format: 1\nimport_max_bytes: 123\n"
    config_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(config_module, "_supports_directory_relative_config_io", lambda: False)

    with pytest.raises(ProjectError, match="safe project configuration access is unavailable"):
        load_project_config(root)
    with pytest.raises(ProjectError, match="safe project configuration writes are unavailable"):
        write_project_config(root, ProjectConfig(import_max_bytes=456))

    assert config_path.read_text(encoding="utf-8") == original


def test_project_config_write_revalidates_bypassed_model_before_atomic_replace(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    config_path = root / "discourselens.yaml"
    config_path.write_text("project_format: 1\nimport_max_bytes: 123\n", encoding="utf-8")
    original = config_path.read_text(encoding="utf-8")
    secret_like = "sk-config-write-must-not-persist"
    bypassed = ProjectConfig().model_copy(update={"import_max_bytes": secret_like})

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        with pytest.raises(ProjectError, match="configuration is invalid") as error:
            write_project_config(root, bypassed)

    rendered_traceback = "".join(traceback.format_exception(error.value))
    assert config_path.read_text(encoding="utf-8") == original
    assert secret_like not in config_path.read_text(encoding="utf-8")
    assert secret_like not in rendered_traceback
    assert list(root.glob(".discourselens-config-*")) == []
    assert not any(secret_like in str(warning.message) for warning in captured_warnings)


def test_project_config_writer_drops_bypassed_secret_values_in_extra_model_state(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    secret_value = "sk-config-extra-must-not-be-serialized"
    bypassed = ProjectConfig().model_copy(update={"api_key": secret_value})

    write_project_config(root, bypassed)

    serialized = (root / "discourselens.yaml").read_text(encoding="utf-8")
    assert secret_value not in serialized
    assert "api_key" not in serialized
    assert load_project_config(root) == ProjectConfig()


def test_project_and_sdk_import_limits_can_be_overridden_per_import(tmp_path):
    root = tmp_path / "project"
    Project.init(root).close()
    source = tmp_path / "source.txt"
    source.write_text("five!", encoding="utf-8")
    runner = CliRunner()

    too_small = runner.invoke(
        app,
        ["document", "import", str(root), str(source), "--max-bytes", "4"],
    )
    assert too_small.exit_code != 0
    assert "configured import size limit" in too_small.output

    imported = runner.invoke(
        app,
        ["document", "import", str(root), str(source), "--max-bytes", "5"],
    )
    assert imported.exit_code == 0, imported.output

    with Project.open(root) as project:
        larger_source = tmp_path / "another-source.txt"
        larger_source.write_text("still five!", encoding="utf-8")
        with pytest.raises(DocumentError, match="configured import size limit"):
            project.import_document(larger_source, max_bytes=4)
        assert project.import_document(larger_source, max_bytes=11).status == "SUCCESS"
