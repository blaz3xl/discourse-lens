from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_source_distribution_rebuilds_wheel_with_bundled_resources(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("source-distribution smoke requires uv, the repository's offline build tool")

    sdist_dir = tmp_path / "sdist"
    build_sdist = subprocess.run(
        [uv, "build", "--offline", "--sdist", "--out-dir", str(sdist_dir)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build_sdist.returncode == 0, build_sdist.stdout + build_sdist.stderr

    source_archives = list(sdist_dir.glob("discourselens-*.tar.gz"))
    assert len(source_archives) == 1
    source_tree = tmp_path / "source-tree"
    source_tree.mkdir()
    with tarfile.open(source_archives[0], "r:gz") as archive:
        archive.extractall(source_tree, filter="data")
    unpacked_projects = list(source_tree.iterdir())
    assert len(unpacked_projects) == 1
    unpacked_project = unpacked_projects[0]

    rebuilt_wheel_dir = tmp_path / "rebuilt-wheel"
    build_wheel = subprocess.run(
        [uv, "build", "--offline", "--wheel", "--out-dir", str(rebuilt_wheel_dir)],
        cwd=unpacked_project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build_wheel.returncode == 0, build_wheel.stdout + build_wheel.stderr

    rebuilt_wheels = list(rebuilt_wheel_dir.glob("discourselens-*.whl"))
    assert len(rebuilt_wheels) == 1
    expected_package_data = {
        *(
            f"discourselens/_resources/schemas/{path.name}"
            for path in (PROJECT_ROOT / "schemas").glob("*.json")
        ),
        *(
            f"discourselens/_resources/methodology/{path.name}"
            for path in (PROJECT_ROOT / "docs" / "methodology").glob("*.md")
        ),
    }
    with zipfile.ZipFile(rebuilt_wheels[0]) as package:
        assert expected_package_data <= set(package.namelist())


def test_built_wheel_installs_commands_and_bundled_resources(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("wheel smoke requires uv, the repository's offline build tool")

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    build = subprocess.run(
        [uv, "build", "--offline", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stdout + build.stderr

    wheels = list(wheel_dir.glob("discourselens-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    expected_package_data = {
        *(
            f"discourselens/_resources/schemas/{path.name}"
            for path in (PROJECT_ROOT / "schemas").glob("*.json")
        ),
        *(
            f"discourselens/_resources/methodology/{path.name}"
            for path in (PROJECT_ROOT / "docs" / "methodology").glob("*.md")
        ),
    }
    with zipfile.ZipFile(wheel) as package:
        wheel_files = set(package.namelist())
        assert expected_package_data <= wheel_files

        entry_points = configparser.ConfigParser()
        entry_points_path = next(
            path for path in wheel_files if path.endswith(".dist-info/entry_points.txt")
        )
        entry_points.read_string(package.read(entry_points_path).decode("utf-8"))
        assert dict(entry_points.items("console_scripts")) == {
            "discourselens": "discourselens.cli:app",
            "discourselens-mcp": "discourselens.mcp_server:main",
            "discourselens-web": "discourselens.web:main",
            "discourselens-web-user": "discourselens.web_accounts:main",
        }

    environment_root = tmp_path / "clean-venv"
    environment = subprocess.run(
        [
            uv,
            "venv",
            "--offline",
            "--python",
            sys.executable,
            str(environment_root),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert environment.returncode == 0, environment.stdout + environment.stderr

    environment_python = environment_root / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    requirements = tmp_path / "locked-mcp-requirements.txt"
    export = subprocess.run(
        [
            uv,
            "export",
            "--locked",
            "--offline",
            "--extra",
            "mcp",
            "--no-dev",
            "--no-emit-project",
            "--output-file",
            str(requirements),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert export.returncode == 0, export.stdout + export.stderr
    assert requirements.is_file()

    dependencies = subprocess.run(
        [
            uv,
            "pip",
            "sync",
            "--python",
            str(environment_python),
            str(requirements),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dependencies.returncode == 0, dependencies.stdout + dependencies.stderr

    install = subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--offline",
            "--python",
            str(environment_python),
            "--no-deps",
            str(wheel),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    scripts = environment_root / ("Scripts" if os.name == "nt" else "bin")
    executable_suffix = ".exe" if os.name == "nt" else ""
    command_env = os.environ.copy()
    # Do not let the checkout, a developer venv, or an ambient PYTHONHOME satisfy
    # imports accidentally; all entrypoint imports must resolve in this venv.
    command_env.pop("PYTHONPATH", None)
    command_env.pop("PYTHONHOME", None)
    for command, version_prefix in (
        ("discourselens", "DiscourseLens "),
        ("discourselens-mcp", "DiscourseLens MCP "),
    ):
        result = subprocess.run(
            [str(scripts / f"{command}{executable_suffix}"), "--version"],
            cwd=tmp_path,
            env=command_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.startswith(version_prefix)
        assert result.stderr == ""

        resource_probe = subprocess.run(
            [
                str(environment_python),
                "-c",
                """
from pathlib import Path
import sys
import json

import discourselens
from discourselens.skills import BUILTIN_SKILLS, skill_resource_text

environment_root = Path(sys.argv[1]).resolve()
expected_schema_names = set(json.loads(sys.argv[2]))
assert Path(discourselens.__file__).resolve().is_relative_to(environment_root)
schema_root = Path(discourselens.__file__).resolve().parent / "_resources" / "schemas"
installed_schema_names = {path.name for path in schema_root.glob("*.json")}
assert expected_schema_names <= installed_schema_names
assert {
    "evidence-reference-v2.json",
    "finding-report-v2.json",
    "analysis-report-v2.json",
    "research-bundle-v14.json",
    "research-bundle-v15.json",
    "research-bundle-v16.json",
} <= installed_schema_names
manifests = BUILTIN_SKILLS.manifests()
references = {
    reference
    for manifest in manifests
    for reference in (
        manifest.documentation_ref,
        manifest.input_schema_ref,
        manifest.output_schema_ref,
        manifest.model_input_schema_ref,
        manifest.model_output_schema_ref,
    )
    if reference is not None
}
assert references
assert all(skill_resource_text(reference).strip() for reference in references)
""",
                str(environment_root),
                json.dumps(sorted(path.name for path in (PROJECT_ROOT / "schemas").glob("*.json"))),
            ],
            cwd=tmp_path,
            env=command_env,
            capture_output=True,
            text=True,
            check=False,
        )
    assert resource_probe.returncode == 0, resource_probe.stdout + resource_probe.stderr
