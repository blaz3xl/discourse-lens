"""Run a small, disposable workload against the local Project corpus API.

The generated text is synthetic and deterministic for a given seed. These runs
exercise project import and lexical corpus operations; their timings are not
scientific claims about real corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from discourselens import Project

DEFAULT_DOCUMENTS = 3
DEFAULT_TOKENS_PER_DOCUMENT = 120
DEFAULT_SEED = 20260930
SEARCH_TERM = "migration"
KWIC_WINDOW = 4

# Larger workloads must opt in. These ceilings bound generated text to a small
# local text benchmark even when both dimensions are near their configured max.
DEFAULT_DOCUMENTS_CEILING = 10
DEFAULT_TOKENS_CEILING = 1_000
MAX_DOCUMENTS = 1_000
MAX_TOKENS_PER_DOCUMENT = 100_000
MAX_TOTAL_TOKENS = 1_000_000
MAX_SEED = (1 << 63) - 1

_VOCABULARY = (
    "policy",
    "debate",
    "community",
    "public",
    "change",
    "local",
    "report",
    "voice",
    "institution",
    "service",
    "evidence",
    "meeting",
    "language",
    "decision",
    "resident",
    "document",
    "context",
    "question",
    "review",
    "practice",
)
_MASK_64 = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_TARGET_INTERVAL = 17


@dataclass(frozen=True, slots=True)
class WorkloadOptions:
    documents: int = DEFAULT_DOCUMENTS
    tokens_per_document: int = DEFAULT_TOKENS_PER_DOCUMENT
    seed: int = DEFAULT_SEED
    fts_search: bool = False
    allow_large: bool = False


def _validate_options(options: WorkloadOptions) -> None:
    if type(options.documents) is not int or options.documents < 1:
        raise ValueError("documents must be a positive integer")
    if type(options.tokens_per_document) is not int or options.tokens_per_document < 1:
        raise ValueError("tokens per document must be a positive integer")
    if type(options.seed) is not int or not 0 <= options.seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}")
    if options.documents > MAX_DOCUMENTS:
        raise ValueError(f"documents cannot exceed the hard limit of {MAX_DOCUMENTS}")
    if options.tokens_per_document > MAX_TOKENS_PER_DOCUMENT:
        raise ValueError(
            "tokens per document cannot exceed the hard limit of "
            f"{MAX_TOKENS_PER_DOCUMENT}"
        )
    total_tokens = options.documents * options.tokens_per_document
    if total_tokens > MAX_TOTAL_TOKENS:
        raise ValueError(f"total generated tokens cannot exceed {MAX_TOTAL_TOKENS}")
    exceeds_default = (
        options.documents > DEFAULT_DOCUMENTS_CEILING
        or options.tokens_per_document > DEFAULT_TOKENS_CEILING
    )
    if exceeds_default and not options.allow_large:
        raise ValueError(
            "workloads above 10 documents or 1,000 tokens per document require --allow-large"
        )


def _splitmix64(value: int) -> int:
    """Return one deterministic 64-bit mix, independent of Python's random module."""
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & _MASK_64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & _MASK_64
    return value ^ (value >> 31)


def _synthetic_document_text(document_index: int, token_count: int, seed: int) -> str:
    state = (seed ^ ((document_index + 1) * _SPLITMIX_INCREMENT)) & _MASK_64
    tokens: list[str] = []
    for token_index in range(token_count):
        if token_index % _TARGET_INTERVAL == 0:
            tokens.append(SEARCH_TERM)
            continue
        state = (state + _SPLITMIX_INCREMENT) & _MASK_64
        tokens.append(_VOCABULARY[_splitmix64(state) % len(_VOCABULARY)])
    return " ".join(tokens) + ".\n"


def _project_file_sizes(project_root: Path) -> dict[str, int]:
    hidden = project_root / ".discourselens"

    def size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(
            entry.stat(follow_symlinks=False).st_size
            for entry in path.rglob("*")
            if entry.is_file() and not entry.is_symlink()
        )

    files = [
        entry
        for entry in project_root.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    ]
    database = hidden / "project.sqlite"
    return {
        "project_total_bytes": size(project_root),
        "project_file_count": len(files),
        "database_bytes": database.stat().st_size if database.is_file() else 0,
        "managed_source_bytes": size(hidden / "source"),
        "artifact_bytes": size(hidden / "artifacts"),
    }


def _environment() -> dict[str, object]:
    try:
        package_version = version("discourselens")
    except PackageNotFoundError:
        package_version = "unknown"
    return {
        "discourselens_version": package_version,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "sqlite_version": sqlite3.sqlite_version,
    }


def run_benchmark(options: WorkloadOptions | None = None) -> dict[str, object]:
    """Run a disposable local import/frequency/KWIC workload and return JSON data."""
    if options is None:
        options = WorkloadOptions()
    _validate_options(options)
    total_tokens = options.documents * options.tokens_per_document

    with tempfile.TemporaryDirectory(prefix="discourselens-corpus-workload-") as temporary:
        workspace = Path(temporary)
        inputs = workspace / "inputs"
        inputs.mkdir()
        project_root = workspace / "project"

        generated_digest = hashlib.sha256()
        input_paths: list[Path] = []
        generation_started = time.perf_counter()
        for index in range(options.documents):
            name = f"synthetic-document-{index + 1:05d}.txt"
            text = _synthetic_document_text(index, options.tokens_per_document, options.seed)
            path = inputs / name
            path.write_text(text, encoding="utf-8", newline="\n")
            generated_digest.update(name.encode("utf-8"))
            generated_digest.update(b"\0")
            generated_digest.update(text.encode("utf-8"))
            generated_digest.update(b"\0")
            input_paths.append(path)
        generation_seconds = time.perf_counter() - generation_started

        project_started = time.perf_counter()
        project = Project.init(project_root)
        project_initialization_seconds = time.perf_counter() - project_started
        try:
            import_started = time.perf_counter()
            imported = project.import_documents(input_paths)
            import_seconds = time.perf_counter() - import_started
            if imported.failed_count or imported.duplicate_count or imported.success_count != options.documents:
                raise RuntimeError("synthetic text import did not succeed for every document")
            revision_ids = [item.revision_id for item in imported.items]
            if any(revision_id is None for revision_id in revision_ids):
                raise RuntimeError("successful import did not return every revision ID")
            selected_revision_ids = [str(revision_id) for revision_id in revision_ids]

            scope_started = time.perf_counter()
            scope_id = project.snapshot(selected_revision_ids)
            scope_seconds = time.perf_counter() - scope_started

            frequency_started = time.perf_counter()
            frequencies = project.corpus_frequency(scope_id)
            frequency_seconds = time.perf_counter() - frequency_started

            kwic_started = time.perf_counter()
            kwic_rows = project.corpus_kwic(scope_id, SEARCH_TERM, window=KWIC_WINDOW)
            kwic_seconds = time.perf_counter() - kwic_started

            fts_seconds: float | None = None
            fts_match_count: int | None = None
            if options.fts_search:
                fts_started = time.perf_counter()
                fts_rows = project.search(SEARCH_TERM, scope_id)
                fts_seconds = time.perf_counter() - fts_started
                fts_match_count = len(fts_rows)

            document_count = project.documents_page(limit=MAX_DOCUMENTS)["total"]
            if type(document_count) is not int:
                raise RuntimeError("Project API returned an invalid document count")
        finally:
            project.close()

        sizes = _project_file_sizes(project_root)
        observed_total_tokens = sum(frequencies.values())
        if observed_total_tokens != total_tokens:
            raise RuntimeError(
                f"generated {total_tokens} tokens but the corpus API counted "
                f"{observed_total_tokens}"
            )

        return {
            "schema_version": "1",
            "benchmark": "synthetic-local-corpus-workload",
            "notice": (
                "Synthetic local workload only. Timings are environment-specific smoke "
                "measurements, not scientific performance claims about real corpora."
            ),
            "configuration": {
                "documents": options.documents,
                "tokens_per_document": options.tokens_per_document,
                "seed": options.seed,
                "fts_search_enabled": options.fts_search,
                "search_term": SEARCH_TERM,
                "kwic_window": KWIC_WINDOW,
                "large_workload_opt_in": options.allow_large,
                "synthetic_text_sha256": generated_digest.hexdigest(),
            },
            "environment": _environment(),
            "corpus_dimensions": {
                "documents": document_count,
                "scope_revisions": len(selected_revision_ids),
                "requested_tokens": total_tokens,
                "observed_tokens": observed_total_tokens,
                "frequency_types": len(frequencies),
                "search_term_frequency": int(frequencies.get(SEARCH_TERM, 0)),
                "kwic_rows": len(kwic_rows),
                "fts_matches": fts_match_count,
            },
            "elapsed_seconds": {
                "text_generation": generation_seconds,
                "project_initialization": project_initialization_seconds,
                "document_import": import_seconds,
                "scope_creation": scope_seconds,
                "frequency": frequency_seconds,
                "kwic": kwic_seconds,
                "fts_search": fts_seconds,
            },
            "project_size_bytes": sizes,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--tokens-per-document", type=int, default=DEFAULT_TOKENS_PER_DOCUMENT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--fts-search",
        action="store_true",
        help="include the Project API's scoped FTS search measurement",
    )
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help=(
            "opt in to more than 10 documents or 1,000 tokens per document; hard limits "
            "still apply"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to a new file (default: print JSON to stdout)",
    )
    arguments = parser.parse_args(argv)
    options = WorkloadOptions(
        documents=arguments.documents,
        tokens_per_document=arguments.tokens_per_document,
        seed=arguments.seed,
        fts_search=arguments.fts_search,
        allow_large=arguments.allow_large,
    )
    try:
        result = run_benchmark(options)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if arguments.output is None:
        sys.stdout.write(serialized)
    else:
        try:
            with arguments.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
        except FileExistsError:
            parser.error(f"refusing to overwrite existing output file: {arguments.output}")
        print(f"Wrote benchmark JSON to {arguments.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
